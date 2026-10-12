#!/usr/bin/env python3
"""A medida do estrato "corpo" (DOC-13 §9.5): o assistente de hoje responde o que o resumo
não diz? E responde quando tem a passagem?

    P = .venv-treino\\Scripts\\python.exe scripts\\medir_corpo.py
    P --rodar [--limite N]       # os três braços nas 150 (GPU, ~4 h, retomável)
    P --exportar-i3c             # 40 respostas, braço oculto, para o Claude julgar às cegas
    P --selar-i3c <veredictos>   # o hash dos veredictos — vai ao git ANTES de o juiz rodar
    P --julgar                   # o juiz (GPU, ~1 h); recusa sem o selo
    P --resumir                  # as leituras e as guardas da regra

Respostas, julgamentos e a amostra ficam em `data/processed/assistente/`, fora do git
(DOC-11 §8.1). Vão para o git o selo e o agregado, em `data/processed/avaliacao/`.

⚠️ `--resumir` é o único passo que mostra acerto pelo juiz, e só roda depois do selo: o
julgamento às cegas não pode ter visto o que o juiz disse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402

utf8()

DIR = RAIZ / "data/processed/assistente"
AVALIACAO = RAIZ / "data/processed/avaliacao"
ITENS = DIR / "itens_corpo.json"
PASSAGENS = DIR / "corpo_passagens.jsonl"
I3C_AMOSTRA = DIR / "corpo_i3c_para_julgar.json"
I3C_CHAVE = DIR / "corpo_i3c_chave.json"
I3C_SELO = AVALIACAO / "assistente_corpo_i3c_selo.json"
AGREGADO = AVALIACAO / "assistente_corpo.json"


def _sha(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def carregar() -> tuple[list[dict], dict[str, str], str]:
    """Os 150 itens e as passagens, conferidos contra os hashes versionados."""
    itens = json.loads(ITENS.read_text(encoding="utf-8"))
    passagens = {}
    for linha in PASSAGENS.read_text(encoding="utf-8").splitlines():
        d = json.loads(linha)
        if not d.get("sem_passagem"):
            passagens[d["arxiv_id"]] = d["passagem"]
    esperado = json.loads((AVALIACAO / "assistente_itens_corpo.json").read_text(encoding="utf-8"))
    sha = _sha(json.dumps(itens, ensure_ascii=False))
    sha_p = _sha(json.dumps([passagens[i["arxiv_id"]] for i in itens], ensure_ascii=False))
    if (sha, sha_p) != (esperado["sha256_itens"], esperado["sha256_passagens"]):
        raise SystemExit("os itens ou as passagens do corpo não são os versionados: alguém "
                         "mexeu no conjunto depois de fechado.")
    return itens, passagens, sha


def caminhos(sha_itens: str):
    from phifm.eval import assistente_bracos as b
    from phifm.rag.llm import MODELO

    ab = b.assinatura_bracos(MODELO.name, sha_itens)
    aj = b.assinatura_juiz(MODELO.name)
    return DIR / f"bracos_corpo_{ab[:12]}.jsonl", DIR / f"juiz_{aj[:12]}.jsonl", ab, aj


def rodar(a, itens, passagens, cache_b: Path) -> None:
    from medir_assistente import _modelo_e_assistente
    from phifm.eval import assistente_bracos as b

    modelo, assistente = _modelo_e_assistente(a, com_busca=True, reordenar=True)
    feitos0 = len({r.arxiv_id for r in b.ler_respostas(cache_b)})
    alvo = len(itens) if a.limite is None else min(len(itens), feitos0 + a.limite)
    t0, n = time.perf_counter(), [0]

    def ao_terminar(item, regs):
        n[0] += 1
        feitos = feitos0 + n[0]
        s = (time.perf_counter() - t0) / n[0]
        seg = " · ".join(f"{r.braco} {r.segundos:.0f}s" for r in regs)
        print(f"[{feitos}/{len(itens)}] {item['arxiv_id']} · {seg} · "
              f"P {'em ' + str(regs[0].posicao_p) if regs[0].posicao_p else 'fora'} · "
              f"{s:.0f} s/item · faltam ~{(alvo - feitos) * s / 3600:.1f} h", flush=True)

    try:
        b.rodar(assistente, itens, cache_b, a.limite, ao_terminar,
                por_item=lambda ass, it: b.rodar_item_corpo(ass, it, passagens[it["arxiv_id"]]))
    finally:
        modelo.parar()


def _amostra(itens, cache_b: Path):
    from phifm.eval import assistente_bracos as b

    respostas = b.ler_respostas(cache_b)
    if len({r.arxiv_id for r in respostas}) < len(itens):
        raise SystemExit("os braços ainda não rodaram em todos os itens (--rodar).")
    return respostas, b.amostra_i3c(respostas)


def exportar_i3c(itens, cache_b: Path) -> None:
    """A amostra para o julgamento às cegas: pergunta, gabarito e resposta sem as marcas
    [n]. O braço fica num arquivo À PARTE, que quem julga não abre."""
    from phifm.rag.assistente import sem_citacoes

    por_id = {i["arxiv_id"]: i for i in itens}
    _, amostra = _amostra(itens, cache_b)
    dados = [{"id": n, "pergunta": por_id[r.arxiv_id]["pergunta"],
              "gabarito": por_id[r.arxiv_id]["gabarito"],
              "resposta": sem_citacoes(r.texto).strip()} for n, r in enumerate(amostra)]
    I3C_AMOSTRA.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    I3C_CHAVE.write_text(json.dumps([{"id": n, "arxiv_id": r.arxiv_id, "braco": r.braco}
                                     for n, r in enumerate(amostra)]), encoding="utf-8")
    print(f"{len(dados)} respostas → {I3C_AMOSTRA} (o braço está em {I3C_CHAVE.name}; não abrir)")


def selar_i3c(veredictos: Path) -> None:
    from phifm.eval import assistente_bracos as b

    bruto = veredictos.read_text(encoding="utf-8")
    lidos = json.loads(bruto)
    amostra = json.loads(I3C_AMOSTRA.read_text(encoding="utf-8"))
    if sorted(int(k) for k in lidos) != [d["id"] for d in amostra]:
        raise SystemExit("os veredictos não cobrem exatamente a amostra.")
    if any(v not in b.VEREDICTOS for v in lidos.values()):
        raise SystemExit(f"veredicto fora de {b.VEREDICTOS}.")
    selo = {"regra": "DOC-13 §9.5 · I3c", "julgador": "claude-opus-5-5, às cegas (braço oculto)",
            "n": len(lidos), "arquivo": veredictos.name, "sha256_veredictos": _sha(bruto),
            "sha256_amostra": _sha(I3C_AMOSTRA.read_text(encoding="utf-8")),
            "selado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "antes_do_juiz": "o juiz só roda (--julgar) com este selo versionado"}
    I3C_SELO.write_text(json.dumps(selo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selo, ensure_ascii=False, indent=2))


def _veredictos_selados() -> dict[int, str]:
    if not I3C_SELO.exists():
        raise SystemExit("falta o selo de I3c: julgue a amostra às cegas e rode --selar-i3c.")
    selo = json.loads(I3C_SELO.read_text(encoding="utf-8"))
    bruto = (DIR / selo["arquivo"]).read_text(encoding="utf-8")
    if _sha(bruto) != selo["sha256_veredictos"]:
        raise SystemExit("os veredictos às cegas mudaram depois de selados.")
    return {int(k): v for k, v in json.loads(bruto).items()}


def julgar(a, itens, cache_b: Path, cache_j: Path) -> None:
    from medir_assistente import _modelo_e_assistente
    from phifm.eval import assistente_bracos as b

    _veredictos_selados()
    respostas = b.ler_respostas(cache_b)
    modelo, _ = _modelo_e_assistente(a, com_busca=False)
    t0, n = time.perf_counter(), [0]

    def ao_julgar(r, d):
        n[0] += 1
        if n[0] % 25 == 0:
            print(f"   {n[0]} julgamentos · {(time.perf_counter() - t0) / n[0]:.1f} s cada",
                  flush=True)

    try:
        julg = b.julgar_todas(modelo, itens, respostas, cache_j, ao_julgar)
    finally:
        modelo.parar()
    print(f"{len(julg)} textos no cache do juiz")


def resumir(itens, cache_b: Path, cache_j: Path, sha_itens: str, ab: str, aj: str) -> None:
    import numpy as np

    from phifm.eval import assistente_bracos as b

    respostas = b.ler_respostas(cache_b)
    por_id = {i["arxiv_id"]: i for i in itens}
    julg = b.ler_julgamentos(cache_j)
    veredictos = b.veredictos_por_braco(itens, respostas, julg)
    r = b.resumo_corpo(veredictos, respostas)

    meus = _veredictos_selados()
    chave = json.loads(I3C_CHAVE.read_text(encoding="utf-8"))
    de = {(x.arxiv_id, x.braco): x for x in respostas}
    do_juiz = []
    for c in chave:
        it, resp = por_id[c["arxiv_id"]], de[(c["arxiv_id"], c["braco"])]
        do_juiz.append(julg[b.chave_do_julgamento(it["pergunta"], it["gabarito"], resp.texto)]
                       ["veredicto"])
    i3c = b.apurar_i3([meus[c["id"]] for c in chave], do_juiz)
    i3c["por_braco"] = {
        braco: b.apurar_i3([meus[c["id"]] for c in chave if c["braco"] == braco],
                           [v for c, v in zip(chave, do_juiz, strict=True) if c["braco"] == braco])
        for braco in (b.BRACO_REORDENADO, b.BRACO_PASSAGEM)}

    saida = {
        "regra": "DOC-13 §9.5, escrita em 2026-10-11 antes de rodar (c9b5d58)",
        "estrato": "corpo — a resposta está numa passagem do corpo do artigo e não no resumo",
        "ressalvas": ["perguntas e passagens escolhidas e escritas pelo Claude, sem revisão humana",
                      "112 dos 150 itens vêm da passagem de mais números: pende para medidas e "
                      "detalhes de método",
                      "juiz calibrado em perguntas do resumo; neste estrato vale I3c",
                      "Bp recebe a passagem de onde a pergunta foi escrita: é o melhor caso; "
                      "um sistema real teria de ACHAR essa passagem dentro do artigo",
                      "o juiz confunde abstenção com erro neste estrato (7 das 9 abstenções da "
                      "amostra às cegas saíram como `errado`): a divisão errado × absteve dos "
                      "veredictos não vale; `certo` contra o resto vale (κ de I3c)"],
        "ocorrencias": [
            "2026-10-11: o app reiniciou a sessão e a rodada foi relançada sem que a anterior "
            "tivesse parado; 56 itens foram gerados duas vezes. Vale o PRIMEIRO registro "
            "gravado de cada (item, braço), regra fixada antes de qualquer veredicto. Das 168 "
            "respostas repetidas, 126 saíram com texto diferente e 47 itens com fontes "
            "diferentes: com dois clientes no mesmo servidor a geração não se repete bit a "
            "bit. O arquivo com as duas cópias ficou guardado (.bruto), fora do git."],
        "sha256_itens": sha_itens, "assinatura_bracos": ab, "assinatura_juiz": aj,
        "bracos": {b.BRACO_REORDENADO: "o sistema de hoje (ΦRank nos 50, 6 fontes com o resumo)",
                   b.BRACO_PASSAGEM: "as fontes de A′ com a passagem de P garantida",
                   "C": "sem fontes"},
        "I3c": i3c,
        "juiz_validado_no_estrato": bool(i3c["passa"]),
        **r,
        "segundos_medianos": {
            braco: round(float(np.median([x.segundos for x in respostas if x.braco == braco])), 1)
            for braco in (b.BRACO_REORDENADO, b.BRACO_PASSAGEM, "C")},
        "mecanicas": {
            braco: {"respostas_sem_nenhuma_citacao": sum(
                        1 for x in respostas if x.braco == braco and not x.citadas),
                    "P_citado": sum(1 for x in respostas if x.braco == braco and x.p_citada),
                    "citacoes_removidas_pelo_portao": sum(
                        len(x.removidas) for x in respostas if x.braco == braco)}
            for braco in (b.BRACO_REORDENADO, b.BRACO_PASSAGEM)}}
    AGREGADO.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    if not i3c["passa"]:
        print(f"⚠️ I3c NÃO passou (κ {i3c['kappa_certo']}): juiz não validado neste estrato.")
    print(json.dumps({k: v for k, v in saida.items()
                      if k not in ("ressalvas", "bracos", "sha256_itens", "assinatura_bracos",
                                   "assinatura_juiz")}, ensure_ascii=False, indent=1))
    print(f"→ {AGREGADO.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rodar", action="store_true")
    p.add_argument("--limite", type=int)
    p.add_argument("--exportar-i3c", action="store_true")
    p.add_argument("--selar-i3c", type=Path)
    p.add_argument("--julgar", action="store_true")
    p.add_argument("--resumir", action="store_true")
    p.add_argument("--indice", type=Path, default=RAIZ / "data/processed/indice_busca")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    args = p.parse_args()
    itens, passagens, sha = carregar()
    cache_b, cache_j, ab, aj = caminhos(sha)
    a = SimpleNamespace(indice=args.indice, spine=args.spine, limite=args.limite)
    if args.rodar:
        rodar(a, itens, passagens, cache_b)
    elif args.exportar_i3c:
        exportar_i3c(itens, cache_b)
    elif args.selar_i3c:
        selar_i3c(args.selar_i3c)
    elif args.julgar:
        julgar(a, itens, cache_b, cache_j)
    elif args.resumir:
        resumir(itens, cache_b, cache_j, sha, ab, aj)
    else:
        p.error("--rodar, --exportar-i3c, --selar-i3c, --julgar ou --resumir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
