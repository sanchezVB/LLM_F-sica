#!/usr/bin/env python3
"""A medida das passagens sob demanda (DOC-13 §9.6): com os trechos do texto completo das
fontes, o assistente passa a responder o que o resumo não diz — sem piorar o que ele diz?

    P = .venv-treino\\Scripts\\python.exe scripts\\medir_passagens.py
    P --rodar [--limite N]       # A′, A″, Bp e C nos 100 do corpo 2 (GPU, ~2 h, retomável)
    P --exportar-i3              # 40 respostas (A″ e A′), braço oculto, para o Claude julgar
    P --selar-i3 <veredictos>    # o hash dos veredictos — vai ao git ANTES de o juiz rodar
    P --julgar                   # o juiz no corpo 2 (GPU, ~40 min); recusa sem o selo
    P --primario [--limite N]    # A″ nos 500 do TESTE primário e o juiz (GPU, ~2,5 h)
    P --resumir                  # os três eixos da regra, as guardas e a decisão

`--primario` toca o conjunto de teste, uma vez: recusa enquanto a regra for PROPOSTA
(`REGRA_9_6_ACEITA` em `phifm.eval.assistente_bracos`).

Respostas, julgamentos e a amostra ficam em `data/processed/assistente/`, fora do git
(DOC-11 §8.1). Vão para o git o selo e o agregado, em `data/processed/avaliacao/`.

⚠️ Uma rodada por vez: `--rodar` e `--primario` seguram uma trava, e um segundo processo
é recusado. Em 2026-10-11 a medida do corpo foi relançada com a anterior ainda viva, e 56
itens saíram em dobro.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402

utf8()

DIR = RAIZ / "data/processed/assistente"
AVALIACAO = RAIZ / "data/processed/avaliacao"
ITENS = DIR / "itens_corpo2.json"
PASSAGENS = DIR / "corpo_passagens.jsonl"
TEXTOS = RAIZ / "data/processed/textos_por_artigo"
I3_AMOSTRA = DIR / "passagens_i3_para_julgar.json"
I3_CHAVE = DIR / "passagens_i3_chave.json"
I3_SELO = AVALIACAO / "assistente_passagens_i3_selo.json"
AGREGADO = AVALIACAO / "assistente_passagens.json"
SALTO_I3 = 6


def _sha(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _ler_jsonl(arq: Path) -> list[dict]:
    return [json.loads(x) for x in arq.read_text(encoding="utf-8").splitlines() if x.strip()]


@contextmanager
def trava(nome: str):
    """Uma rodada por vez. A trava é do sistema operacional: some sozinha se o processo
    morre, e não depende de alguém lembrar de apagar um arquivo."""
    import msvcrt

    arq = DIR / f"{nome}.trava"
    f = arq.open("a+")
    try:
        msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        f.close()
        raise SystemExit(f"já existe uma rodada viva segurando {arq.name}: não lanço outra. "
                         "Confira os processos antes de insistir.") from None
    try:
        yield
    finally:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        f.close()


def carregar() -> tuple[list[dict], dict[str, dict], str]:
    """Os 100 itens do corpo 2 e as passagens, conferidos contra os hashes versionados."""
    itens = json.loads(ITENS.read_text(encoding="utf-8"))
    ouro = {d["arxiv_id"]: d for d in _ler_jsonl(PASSAGENS) if not d.get("sem_passagem")}
    esperado = json.loads((AVALIACAO / "assistente_itens_corpo2.json").read_text(encoding="utf-8"))
    sha = _sha(json.dumps(itens, ensure_ascii=False))
    sha_p = _sha(json.dumps([ouro[i["arxiv_id"]]["passagem"] for i in itens], ensure_ascii=False))
    if (sha, sha_p) != (esperado["sha256_itens"], esperado["sha256_passagens"]):
        raise SystemExit("os itens ou as passagens do corpo 2 não são os versionados: alguém "
                         "mexeu no conjunto depois de fechado.")
    return itens, ouro, sha


def assinatura(sha_itens: str) -> str:
    """O que muda as respostas: os braços de sempre e a regra das passagens."""
    from phifm.eval import assistente_bracos as b
    from phifm.rag import passagens as ps
    from phifm.rag.assistente import ROTULO_PASSAGENS
    from phifm.rag.llm import MODELO

    return _sha(f"{b.assinatura_bracos(MODELO.name, sha_itens)}\n{ps.SISTEMA_TRADUCAO}\n"
                f"{ps.K_PASSAGENS}\n{ps.LIMITE_TRECHO}\n{ROTULO_PASSAGENS}")


def _cache_do_juiz() -> Path:
    from phifm.eval import assistente_bracos as b
    from phifm.rag.llm import MODELO

    return DIR / f"juiz_{b.assinatura_juiz(MODELO.name)[:12]}.jsonl"


def _passo(modelo, tok):
    from phifm.eval.assistente import SEMENTE
    from phifm.rag import passagens as ps
    from phifm.retrieval.textos import Textos

    return ps.Passagens(modelo, ps.TrechosDoTexto(Textos(TEXTOS), ps.contador(tok)),
                        semente=SEMENTE)


# ── o corpo 2: os quatro braços ──────────────────────────────────────────────────────
def rodar(a, itens, ouro, cache: Path) -> None:
    from medir_assistente import _modelo_e_assistente
    from phifm.eval import assistente_bracos as b

    with trava("medir_passagens"):
        modelo, assistente = _modelo_e_assistente(a, com_busca=True, reordenar=True)
        passo = _passo(modelo, assistente.busca.tok)
        feitos0 = len({r.arxiv_id for r in b.ler_respostas(cache)})
        alvo = len(itens) if a.limite is None else min(len(itens), feitos0 + a.limite)
        t0, n = time.perf_counter(), [0]

        def ao_terminar(item, regs):
            n[0] += 1
            feitos = feitos0 + n[0]
            s = (time.perf_counter() - t0) / n[0]
            seg = " · ".join(f"{r.braco} {r.segundos:.0f}s" for r in regs)
            print(f"[{feitos}/{len(itens)}] {item['arxiv_id']} · {seg} · "
                  f"{len(regs[1].passagens)} trechos · "
                  f"P {'em ' + str(regs[0].posicao_p) if regs[0].posicao_p else 'fora'} · "
                  f"{s:.0f} s/item · faltam ~{(alvo - feitos) * s / 3600:.1f} h", flush=True)

        try:
            b.rodar(assistente, itens, cache, a.limite, ao_terminar,
                    por_item=lambda ass, it: b.rodar_item_passagens(
                        ass, passo, it, ouro[it["arxiv_id"]]["passagem"]))
        finally:
            modelo.parar()


# ── I3: o julgamento às cegas, selado antes do juiz ──────────────────────────────────
def _amostra(itens, cache: Path):
    from phifm.eval import assistente_bracos as b

    respostas = b.ler_respostas(cache)
    if len({r.arxiv_id for r in respostas}) < len(itens):
        raise SystemExit("os braços ainda não rodaram em todos os itens (--rodar).")
    return respostas, b.amostra_i3c(respostas, bracos=(b.BRACO_PASSAGENS, b.BRACO_REORDENADO),
                                    salto=SALTO_I3)


def exportar_i3(itens, cache: Path) -> None:
    from phifm.rag.assistente import sem_citacoes

    por_id = {i["arxiv_id"]: i for i in itens}
    _, amostra = _amostra(itens, cache)
    dados = [{"id": n, "pergunta": por_id[r.arxiv_id]["pergunta"],
              "gabarito": por_id[r.arxiv_id]["gabarito"],
              "resposta": sem_citacoes(r.texto).strip()} for n, r in enumerate(amostra)]
    I3_AMOSTRA.write_text(json.dumps(dados, ensure_ascii=False, indent=1), encoding="utf-8")
    I3_CHAVE.write_text(json.dumps([{"id": n, "arxiv_id": r.arxiv_id, "braco": r.braco}
                                    for n, r in enumerate(amostra)]), encoding="utf-8")
    print(f"{len(dados)} respostas → {I3_AMOSTRA} (o braço está em {I3_CHAVE.name}; não abrir)")


def selar_i3(veredictos: Path) -> None:
    from phifm.eval import assistente_bracos as b

    bruto = veredictos.read_text(encoding="utf-8")
    lidos = json.loads(bruto)
    amostra = json.loads(I3_AMOSTRA.read_text(encoding="utf-8"))
    if sorted(int(k) for k in lidos) != [d["id"] for d in amostra]:
        raise SystemExit("os veredictos não cobrem exatamente a amostra.")
    if any(v not in b.VEREDICTOS for v in lidos.values()):
        raise SystemExit(f"veredicto fora de {b.VEREDICTOS}.")
    selo = {"regra": "DOC-13 §9.6 · I3", "julgador": "claude-opus-5-5, às cegas (braço oculto)",
            "n": len(lidos), "arquivo": veredictos.name, "sha256_veredictos": _sha(bruto),
            "sha256_amostra": _sha(I3_AMOSTRA.read_text(encoding="utf-8")),
            "selado_em": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "antes_do_juiz": "o juiz só roda (--julgar) com este selo versionado"}
    I3_SELO.write_text(json.dumps(selo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selo, ensure_ascii=False, indent=2))


def _veredictos_selados() -> dict[int, str]:
    if not I3_SELO.exists():
        raise SystemExit("falta o selo de I3: julgue a amostra às cegas e rode --selar-i3.")
    selo = json.loads(I3_SELO.read_text(encoding="utf-8"))
    bruto = (DIR / selo["arquivo"]).read_text(encoding="utf-8")
    if _sha(bruto) != selo["sha256_veredictos"]:
        raise SystemExit("os veredictos às cegas mudaram depois de selados.")
    return {int(k): v for k, v in json.loads(bruto).items()}


def julgar(a, itens, cache: Path) -> None:
    from medir_assistente import _modelo_e_assistente
    from phifm.eval import assistente_bracos as b

    _veredictos_selados()
    respostas = b.ler_respostas(cache)
    modelo, _ = _modelo_e_assistente(a, com_busca=False)
    try:
        julg = b.julgar_todas(modelo, itens, respostas, _cache_do_juiz())
    finally:
        modelo.parar()
    print(f"{len(julg)} textos no cache do juiz")


# ── o teste primário: A″ sobre o cache de A′ ─────────────────────────────────────────
def _teste_primario():
    """`(itens, A′ por item, cache de A″)` do estrato primário do conjunto de teste."""
    import medir_assistente as ma
    from phifm.eval import assistente_bracos as b

    itens, sha = ma.carregar_itens()
    _, _, ab, _ = ma.caminhos(sha)
    a2 = {(r.estrato, r.arxiv_id): r
          for r in b.ler_respostas(DIR / f"bracos_A2_{ab[:12]}.jsonl")}
    itens = [i for i in itens if (i["estrato"], i["arxiv_id"]) in a2]
    return itens, a2, DIR / f"bracos_A3_{assinatura(sha)[:12]}.jsonl"


def _refazer_fontes(a2: dict, indice: Path, spine: Path):
    """As fontes de A′ como `Assistente.recuperar` as montou: título e ano do índice,
    resumo do spine. Uma leitura só para os ~3.000 artigos."""
    import polars as pl

    from phifm.rag.assistente import Fonte

    ids = sorted({x for r in a2.values() for x in r.fontes})
    docs = {r["arxiv_id"]: r for r in pl.scan_parquet(indice / "documentos.parquet")
            .filter(pl.col("arxiv_id").is_in(ids)).select("arxiv_id", "title", "year")
            .collect().iter_rows(named=True)}
    resumos = dict(pl.scan_parquet(spine).filter(pl.col("arxiv_id").is_in(ids))
                   .select("arxiv_id", "abstract").collect().iter_rows())

    def fontes_de(lista: list[str]) -> list[Fonte]:
        return [Fonte(n, x, docs[x]["title"], docs[x]["year"],
                      " ".join((resumos.get(x) or "").split()), float("nan"))
                for n, x in enumerate(lista, start=1)]

    return fontes_de


def primario(a) -> None:
    from transformers import AutoTokenizer

    from medir_assistente import _modelo_e_assistente
    from phifm.eval import assistente_bracos as b
    from phifm.eval.assistente import K_FONTES, SEMENTE
    from phifm.rag.assistente import Assistente

    if not b.REGRA_9_6_ACEITA:
        raise SystemExit("a §9.6 do DOC-13 ainda é PROPOSTA: o conjunto de teste não é tocado "
                         "antes do aceite do dono (REGRA_9_6_ACEITA em "
                         "phifm.eval.assistente_bracos).")
    itens, a2, cache = _teste_primario()
    with trava("medir_passagens"):
        fontes_de = _refazer_fontes(a2, a.indice, a.spine)
        modelo, _ = _modelo_e_assistente(a, com_busca=False)
        caminho = json.loads((a.indice / "manifesto.json").read_text(encoding="utf-8"))["modelo"]
        passo = _passo(modelo, AutoTokenizer.from_pretrained(caminho))
        assistente = Assistente(None, modelo, a.spine, k=K_FONTES, semente=SEMENTE)
        t0, n = time.perf_counter(), [0]

        def ao_terminar(item, reg):
            n[0] += 1
            if n[0] % 10 == 0:
                print(f"[A″ {n[0]}] {(time.perf_counter() - t0) / n[0]:.1f} s/item", flush=True)

        try:
            respostas = b.rodar_passagens_sobre_a2(assistente, passo, itens, a2, fontes_de,
                                                   cache, a.limite, ao_terminar)
            if len(respostas) == len(itens):
                b.julgar_todas(modelo, itens, respostas, _cache_do_juiz())
        finally:
            modelo.parar()
    print(f"A″ no teste primário: {len(respostas)} de {len(itens)} itens · "
          f"{sum(1 for r in respostas if r.igual_a_A)} iguais a A′")


# ── o resumo ─────────────────────────────────────────────────────────────────────────
def _alcance(itens, ouro, respostas) -> dict:
    """Onde a passagem de onde a pergunta saiu foi ENTREGUE — o que a seleção fez, sem juiz."""
    from perguntas_corpo import TRECHOS_POR_PASSAGEM
    from phifm.eval import assistente_bracos as b

    a3 = {r.arxiv_id: r for r in respostas if r.braco == b.BRACO_PASSAGENS}
    veio = [i for i in itens if a3[i["arxiv_id"]].posicao_p]
    entregue = []
    for i in veio:
        p, i0 = i["arxiv_id"], ouro[i["arxiv_id"]]["primeiro_trecho"]
        entregue.append(any(x == p and i0 <= pos < i0 + TRECHOS_POR_PASSAGEM
                            for x, pos in a3[p].passagens))
    com_passagem = sum(1 for i in itens if a3[i["arxiv_id"]].passagens)
    return {"itens_com_P_nas_fontes": len(veio),
            "…com_a_passagem_de_ouro_entregue": sum(entregue),
            "fracao": round(sum(entregue) / len(veio), 4) if veio else None,
            "itens_com_alguma_passagem_anexada": com_passagem,
            "trechos_anexados_por_item_media": round(
                sum(len(a3[i["arxiv_id"]].passagens) for i in itens) / len(itens), 2),
            "o_que_e_ouro": "pelo menos um dos 3 trechos da passagem de onde a pergunta saiu"}


def resumir(itens, ouro, cache: Path, sha: str) -> None:
    from phifm.eval import assistente_bracos as b

    respostas = b.ler_respostas(cache)
    por_id = {i["arxiv_id"]: i for i in itens}
    julg = b.ler_julgamentos(_cache_do_juiz())
    veredictos = b.veredictos_por_braco(itens, respostas, julg)
    corpo = b.resumo_passagens(veredictos, respostas)

    meus = _veredictos_selados()
    chave = json.loads(I3_CHAVE.read_text(encoding="utf-8"))
    de = {(x.arxiv_id, x.braco): x for x in respostas}
    do_juiz = [julg[b.chave_do_julgamento(por_id[c["arxiv_id"]]["pergunta"],
                                          por_id[c["arxiv_id"]]["gabarito"],
                                          de[(c["arxiv_id"], c["braco"])].texto)]["veredicto"]
               for c in chave]
    i3 = b.apurar_i3([meus[c["id"]] for c in chave], do_juiz)

    prim = None
    itens_t, a2, cache_t = _teste_primario()
    r3 = b.ler_respostas(cache_t)
    if len(r3) == len(itens_t):
        v_t = b.veredictos_por_braco(itens_t, [*a2.values(), *r3], julg)
        prim = b.nao_piora(v_t, r3)

    saida = {
        "regra": "DOC-13 §9.6, escrita antes de os itens do corpo 2 existirem",
        "regra_aceita_pelo_dono": b.REGRA_9_6_ACEITA,
        "ressalvas": ["perguntas e passagens escolhidas e escritas pelo Claude, sem revisão "
                      "humana; a passagem é a de mais números do artigo",
                      "o autor das perguntas é quem escolheu a regra de seleção, lexical",
                      "só entram artigos com texto completo como P: o ganho vale para os 52% "
                      "do índice que o têm"],
        "sha256_itens": sha, "assinatura": assinatura(sha),
        "bracos": {b.BRACO_REORDENADO: "o sistema de hoje (ΦRank nos 50, 6 fontes com o resumo)",
                   b.BRACO_PASSAGENS: "o mesmo, com os trechos do texto completo das fontes "
                                      "escolhidos para a pergunta",
                   b.BRACO_PASSAGEM: "as fontes de A′ com a passagem de P garantida — o teto",
                   "C": "sem fontes"},
        "I3": i3, "corpo_2": corpo, "alcance_da_selecao": _alcance(itens, ouro, respostas),
        "teste_primario": prim if prim else "ainda não rodado (--primario, depois do aceite)",
        "decisao": b.decidir_passagens(corpo, prim, i3)}
    AGREGADO.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in saida.items()
                      if k not in ("ressalvas", "bracos", "sha256_itens", "assinatura")},
                     ensure_ascii=False, indent=1))
    print(f"→ {AGREGADO.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rodar", action="store_true")
    p.add_argument("--limite", type=int)
    p.add_argument("--exportar-i3", action="store_true")
    p.add_argument("--selar-i3", type=Path)
    p.add_argument("--julgar", action="store_true")
    p.add_argument("--primario", action="store_true")
    p.add_argument("--resumir", action="store_true")
    p.add_argument("--indice", type=Path, default=RAIZ / "data/processed/indice_busca")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    args = p.parse_args()
    a = SimpleNamespace(indice=args.indice, spine=args.spine, limite=args.limite)
    if args.primario:
        primario(a)
        return 0
    itens, ouro, sha = carregar()
    cache = DIR / f"bracos_corpo2_{assinatura(sha)[:12]}.jsonl"
    if args.rodar:
        rodar(a, itens, ouro, cache)
    elif args.exportar_i3:
        exportar_i3(itens, cache)
    elif args.selar_i3:
        selar_i3(args.selar_i3)
    elif args.julgar:
        julgar(a, itens, cache)
    elif args.resumir:
        resumir(itens, ouro, cache, sha)
    else:
        p.error("--rodar, --exportar-i3, --selar-i3, --julgar, --primario ou --resumir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
