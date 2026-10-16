#!/usr/bin/env python3
"""O juiz contra físicos (DOC-13 §9.11): pessoas de fora do projeto julgam 100 respostas
do assistente às cegas, com a regra de correção do juiz, e o κ diz se ele concorda com elas.

    P = .venv-treino\\Scripts\\python.exe scripts\\julgamento_humano.py
    P --amostrar               # as 100 respostas e o selo — o selo vai ao git ANTES das folhas
    P --folha fisico1          # a página de um físico, na ordem dele; a assinatura entra no selo
    P --apurar A.json B.json   # os arquivos baixados das folhas: a guarda, o κ e a decisão

Use um pseudônimo em `--folha` (fisico1, fisico2): o nome vai ao git dentro do selo, e o de
verdade fica com o dono. Mande a página (`data/processed/assistente/humano/folha_*.html`)
só depois de o selo com a assinatura dela estar no git.

A amostra tem perguntas, gabaritos e respostas do conjunto de TESTE: fica fora do git
(DOC-11 §8.1) e vai aos físicos por decisão do dono, com o pedido de não repassar. Vão ao
git o selo e o agregado, em `data/processed/avaliacao/`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402

utf8()

import medir_passagens as mp  # noqa: E402
from phifm.eval import julgamento_humano as jh  # noqa: E402

DIR_H = mp.DIR / "humano"
AMOSTRA = DIR_H / "amostra_humano.json"
SELO = mp.AVALIACAO / "assistente_humano_selo.json"
AGREGADO = mp.AVALIACAO / "assistente_julgamento_humano.json"


def _recusa_sem_aceite() -> None:
    if not jh.REGRA_9_11_ACEITA:
        raise SystemExit("a §9.11 do DOC-13 ainda é PROPOSTA: a amostra não é sorteada antes "
                         "do aceite do dono (REGRA_9_11_ACEITA em "
                         "phifm.eval.julgamento_humano).")


def montar_amostra() -> list[dict]:
    """As 50 seladas (a rodada 2 do I3 refeita e conferida contra o selo dela) e as 50
    novas (A′ e A5 no primário, fora das rodadas do I3), com o veredicto do juiz v2."""
    import conferencia_do_juiz as cj
    import medir_assistente as ma
    import medir_busca_lexical as mbl
    from phifm.eval import assistente_bracos as b

    itens, sha = ma.carregar_itens()
    cache_b, _, _, _ = ma.caminhos(sha)
    por_id = {(i["estrato"], i["arxiv_id"]): i for i in itens}
    respostas = b.ler_respostas(cache_b)
    julg = b.ler_julgamentos(mp._cache_do_juiz())

    def texto(r) -> str:
        return b.sem_citacoes(r.texto).strip() or "(resposta vazia)"

    def entrada(r, origem: str, claude: str | None = None) -> dict:
        it = por_id[(r.estrato, r.arxiv_id)]
        chave = b.chave_do_julgamento(it["pergunta"], it["gabarito"], r.texto)
        if chave not in julg:
            raise SystemExit(f"{r.braco} de {r.arxiv_id} sem julgamento do juiz v2.")
        return {"id": jh.id_opaco(it["pergunta"], texto(r)), "origem": origem,
                "braco": r.braco, "estrato": r.estrato, "arxiv_id": r.arxiv_id,
                "pergunta": it["pergunta"], "gabarito": it["gabarito"], "resposta": texto(r),
                "juiz": julg[chave]["veredicto"], "claude": claude}

    r2 = b.amostra_i3(respostas, rodada=2)
    selados = cj.selados()["9.1_rodada_2"]
    if len(selados) != len(r2) or any(
            p != por_id[(r.estrato, r.arxiv_id)]["pergunta"] or t != texto(r)
            for (p, _, t, _), r in zip(selados, r2, strict=True)):
        raise SystemExit("a rodada 2 do I3 refeita não é a amostra selada em 2026-10-04.")
    entradas = [entrada(r2[k], "selado_r2", selados[k][3]) for k in jh.sortear_selados(len(r2))]

    excluir = {(r.estrato, r.arxiv_id) for r in (*b.amostra_i3(respostas, rodada=1), *r2)}
    _, a2, _, cache_a5 = mbl._carregar()
    a5 = {(r.estrato, r.arxiv_id): r for r in b.ler_respostas(cache_a5)
          if r.braco == b.BRACO_BUSCA_LEXICA}
    de = {b.BRACO_REORDENADO: a2, b.BRACO_BUSCA_LEXICA: a5}
    for braco, chaves in jh.sortear_novos([k for k in a2 if k in a5], excluir).items():
        entradas += [entrada(de[braco][k], braco) for k in chaves]
    jh.conferir_amostra(entradas)
    return entradas


def amostrar() -> None:
    _recusa_sem_aceite()
    entradas = montar_amostra()
    bruto = json.dumps(entradas, ensure_ascii=False, indent=1)
    sha = mp._sha(bruto)
    anterior = json.loads(SELO.read_text(encoding="utf-8")) if SELO.exists() else {}
    if anterior and anterior["sha256_amostra"] != sha:
        raise SystemExit("já existe uma amostra selada, e esta é outra: a amostra não muda "
                         "depois de selada.")
    DIR_H.mkdir(parents=True, exist_ok=True)
    AMOSTRA.write_text(bruto, encoding="utf-8")
    selo = {"regra": "DOC-13 §9.11", "regra_aceita_pelo_dono": jh.REGRA_9_11_ACEITA,
            "o_que_e": "o hash da amostra que vai aos físicos e a assinatura de cada folha; "
                       "o conteúdo (perguntas do teste) fica fora do git",
            "arquivo": AMOSTRA.name, "sha256_amostra": sha, "n": len(entradas),
            "composicao": {o: sum(e["origem"] == o for e in entradas) for o in jh.ORIGENS},
            "selado_em": anterior.get("selado_em") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "folhas": anterior.get("folhas", {})}
    SELO.write_text(json.dumps(selo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(selo, ensure_ascii=False, indent=1))
    print(f"→ {SELO.relative_to(RAIZ)}: vai ao git antes de qualquer folha sair.")


def _amostra() -> list[dict]:
    if not SELO.exists() or not AMOSTRA.exists():
        raise SystemExit("nenhuma amostra selada (--amostrar).")
    bruto = AMOSTRA.read_text(encoding="utf-8")
    if mp._sha(bruto) != json.loads(SELO.read_text(encoding="utf-8"))["sha256_amostra"]:
        raise SystemExit(f"{AMOSTRA.name} não é a amostra selada.")
    return json.loads(bruto)


def folha(julgador: str) -> None:
    import medir_assistente as ma

    _recusa_sem_aceite()
    if not re.fullmatch(r"[a-z0-9_-]{2,32}", julgador):
        raise SystemExit("o pseudônimo vai ao git: só minúsculas, dígitos, _ e - (fisico1).")
    entradas = _amostra()
    pagina, assinatura = ma.montar_folha(
        "Julgamento de respostas", jh.INTRO, ma.OPCOES_I3, jh.itens_da_folha(entradas, julgador),
        f"humano_{julgador}", f"veredictos_{julgador}.json")
    selo = json.loads(SELO.read_text(encoding="utf-8"))
    if selo["folhas"].get(julgador, assinatura) != assinatura:
        raise SystemExit(f"a folha de {julgador} já foi selada com outra assinatura.")
    selo["folhas"][julgador] = assinatura
    destino = DIR_H / f"folha_{julgador}.html"
    destino.write_text(pagina, encoding="utf-8")
    SELO.write_text(json.dumps(selo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(entradas)} respostas → {destino}\nassinatura {assinatura} → "
          f"{SELO.relative_to(RAIZ)} (commit antes de mandar a página)")


def apurar(arquivos: list[Path]) -> None:
    entradas = _amostra()
    selo = json.loads(SELO.read_text(encoding="utf-8"))
    de_quem = {a: n for n, a in selo["folhas"].items()}
    ids = {e["id"] for e in entradas}
    humanos, recebidos = {}, {}
    for arq in arquivos:
        d = json.loads(arq.read_text(encoding="utf-8"))
        nome = de_quem.get(d.get("assinatura_amostra"))
        if nome is None:
            raise SystemExit(f"{arq.name}: não é de nenhuma folha selada.")
        if nome in humanos:
            raise SystemExit(f"{arq.name}: {nome} já entrou por outro arquivo.")
        try:
            humanos[nome] = jh.ler_veredictos(d, selo["folhas"][nome], ids)
        except ValueError as e:
            raise SystemExit(f"{arq.name}: {e}") from None
        recebidos[nome] = mp._sha(arq.read_text(encoding="utf-8"))
    try:
        r = jh.apurar(entradas, humanos)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    saida = {"regra": "DOC-13 §9.11", "regra_aceita_pelo_dono": jh.REGRA_9_11_ACEITA,
             "sha256_amostra": selo["sha256_amostra"], "arquivos_sha256": recebidos, **r}
    AGREGADO.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in saida.items() if k != "por_origem (relatado)"},
                     ensure_ascii=False, indent=1))
    print(f"→ {AGREGADO.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--amostrar", action="store_true")
    g.add_argument("--folha", metavar="PSEUDONIMO")
    g.add_argument("--apurar", nargs="+", type=Path, metavar="VEREDICTOS")
    args = p.parse_args()
    if args.amostrar:
        amostrar()
    elif args.folha:
        folha(args.folha)
    else:
        apurar(args.apurar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
