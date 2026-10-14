#!/usr/bin/env python3
"""Onde o artigo certo se perde na busca — e se a busca lexical o acha (item 2 do dono,
2026-10-14). EXPLORATÓRIO, só no desenvolvimento; o teste não é tocado.

    P = .venv-treino\\Scripts\\python.exe scripts\\explorar_busca_lexical.py
    P --indexar      # BM25 sobre `título. resumo` dos 1,6 milhão (CPU, HD, ~3-4 GB de memória)
    P --posicoes     # a posição do artigo certo em cada busca, nos dois conjuntos
    P --resumir      # P nos 1/6/50/200/1000 por busca e conjunto, com IC pareado

Conjuntos, os dois já lidos (desenvolvimento):

- **corpo 1** — as 150 perguntas do corpo do artigo da §9.5 (`itens_corpo.json`); a
  consulta hipotética é a que o braço A′ gravou, a pergunta em inglês a da §9.6;
- **primário dev** — as 150 perguntas do resumo do desenvolvimento da §9.2; a consulta
  hipotética é a 1ª das três gravadas (a do produto), a pergunta em inglês a da §9.7.

O que se mede é a POSIÇÃO do artigo de onde a pergunta saiu (P) na lista inteira, sem
juiz: "P nos 50" é o que o ΦRank recebe hoje; "P nos 6", o que o modelo lê sem ΦRank.

⚠️ Uma rodada pesada por vez (16 GB): com a página do assistente aberta e o modelo no ar,
isto trava as duas.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

DIR = RAIZ / "data/processed/assistente"
DEV = DIR / "passagens_dev"
SAIDA = DIR / "busca_lexical"
INDICE = RAIZ / "data/processed/indice_busca"
LEXICO = RAIZ / "data/processed/indice_lexical"
SPINE = RAIZ / "data/processed/spine.parquet"
AGREGADO = RAIZ / "data/processed/avaliacao/assistente_busca_lexical_dev.json"
FAIXAS = (1, 6, 50, 200, 1000)
BUSCAS = {
    "denso_hyde": "o de hoje: o encoder com a consulta hipotética",
    "denso_ingles": "o encoder com a pergunta em inglês",
    "bm25_ingles": "BM25 em título + resumo com a pergunta em inglês",
    "bm25_hyde": "BM25 em título + resumo com a consulta hipotética",
    "rrf_hyde_bm25ingles": "fusão por posição: denso_hyde + bm25_ingles",
    "rrf_hyde_bm25hyde": "fusão por posição: denso_hyde + bm25_hyde",
    "rrf_tres": "fusão por posição: denso_hyde + bm25_ingles + bm25_hyde",
    "rrf_dois_densos_bm25": "fusão: denso_hyde + denso_ingles + bm25_ingles",
}


def _ler_jsonl(arq: Path) -> list[dict]:
    return [json.loads(x) for x in arq.read_text(encoding="utf-8").splitlines() if x.strip()]


def conjuntos() -> dict[str, list[dict]]:
    """`{conjunto: [{arxiv_id, hyde, ingles}]}`, só o que tem as duas consultas."""
    corpo = json.loads((DIR / "itens_corpo.json").read_text(encoding="utf-8"))
    hyde_c = {r["arxiv_id"]: r["consulta"] for r in _ler_jsonl(
        DIR / "bracos_corpo_c71c02f7d286.jsonl") if r["braco"] == "A2"}
    ing_c = json.loads((DEV / "ingles_corpo.json").read_text(encoding="utf-8"))
    dev = [i for i in json.loads((DIR / "itens_dev_busca.json").read_text(encoding="utf-8"))
           if i["estrato"] == "primario"]
    hyde_d = {r["arxiv_id"]: r["consultas"][0] for r in _ler_jsonl(
        DIR / "dev_busca_consultas.jsonl") if r["estrato"] == "primario"}
    ing_d = json.loads((DEV / "ingles_primario.json").read_text(encoding="utf-8"))
    saida = {}
    for nome, itens, hyde, ing in (("corpo_1", corpo, hyde_c, ing_c),
                                   ("primario_dev", dev, hyde_d, ing_d)):
        saida[nome] = [{"arxiv_id": i["arxiv_id"], "hyde": hyde[i["arxiv_id"]],
                        "ingles": ing[i["arxiv_id"]]}
                       for i in itens if i["arxiv_id"] in hyde and i["arxiv_id"] in ing]
        if len(saida[nome]) != len(itens):
            print(f"⚠️ {nome}: {len(itens) - len(saida[nome])} itens sem uma das consultas")
    return saida


def indexar() -> None:
    import polars as pl

    from phifm.retrieval import lexical as lx

    ids = pl.read_parquet(INDICE / "documentos.parquet", columns=["arxiv_id"])[
        "arxiv_id"].to_list()
    spine = (pl.scan_parquet(SPINE).select("arxiv_id", "title", "abstract")
             .unique("arxiv_id", keep="first"))
    # Uma leitura do spine só (~1,6 GB de texto na memória), na ordem do índice denso.
    tabela = (pl.DataFrame({"arxiv_id": ids, "_i": range(len(ids))}).lazy()
              .join(spine, on="arxiv_id", how="left").sort("_i").collect())
    lote = 100_000

    def lotes():
        for ini in range(0, len(ids), lote):
            d = tabela.slice(ini, lote)
            textos = [f"{t or ''}. {a or ''}" for t, a in zip(d["title"], d["abstract"],
                                                               strict=True)]
            yield d["arxiv_id"].to_list(), textos

    def ao_lote(feitos, termos, s):
        print(f"[{feitos:,}/{len(ids):,}] {termos:,} termos · {s / 60:.1f} min", flush=True)

    man = lx.construir(ids, lotes(), LEXICO, ao_lote)
    print(json.dumps(man, ensure_ascii=False, indent=2))


def posicoes() -> None:
    from phifm.retrieval import lexical as lx
    from phifm.retrieval.indice import Busca

    sets = conjuntos()
    SAIDA.mkdir(parents=True, exist_ok=True)
    arq = SAIDA / "posicoes.jsonl"
    feitos = {(d["conjunto"], d["arxiv_id"]) for d in _ler_jsonl(arq)} if arq.exists() else set()
    t0 = time.perf_counter()
    busca = Busca(INDICE)
    ids = busca.docs["arxiv_id"].to_list()
    lexica = lx.Lexica(LEXICO, ids)
    linha = {a: i for i, a in enumerate(ids)}
    print(f"índices carregados em {time.perf_counter() - t0:.0f} s", flush=True)
    n = 0
    with arq.open("a", encoding="utf-8") as f:
        for nome, itens in sets.items():
            for it in itens:
                if (nome, it["arxiv_id"]) in feitos:
                    continue
                lp = linha.get(it["arxiv_id"])
                if lp is None:
                    print(f"⚠️ {it['arxiv_id']} não está no índice")
                    continue
                s_bm25 = lexica.pontuar(it["ingles"])
                pos = {"denso_hyde": lx.posicoes_de(busca.pontuar(busca.embutir(it["hyde"]))),
                       "denso_ingles": lx.posicoes_de(busca.pontuar(busca.embutir(it["ingles"]))),
                       "bm25_ingles": lx.posicoes_de(s_bm25),
                       "bm25_hyde": lx.posicoes_de(lexica.pontuar(it["hyde"]))}
                fusoes = {"rrf_hyde_bm25ingles": ("denso_hyde", "bm25_ingles"),
                          "rrf_hyde_bm25hyde": ("denso_hyde", "bm25_hyde"),
                          "rrf_tres": ("denso_hyde", "bm25_ingles", "bm25_hyde"),
                          "rrf_dois_densos_bm25": ("denso_hyde", "denso_ingles", "bm25_ingles")}
                d = {"conjunto": nome, "arxiv_id": it["arxiv_id"],
                     **{b: int(p[lp]) for b, p in pos.items()}}
                for b, partes in fusoes.items():
                    d[b] = lx.posicao(lx.rrf(*(pos[x] for x in partes)), lp)
                # o BM25 achou alguma palavra de P? (escore zero = nenhuma em comum)
                d["bm25_ingles_zero"] = bool(s_bm25[lp] == 0)
                f.write(json.dumps(d) + "\n")
                f.flush()
                n += 1
                if n % 25 == 0:
                    print(f"[{n}] {(time.perf_counter() - t0) / n:.1f} s/item", flush=True)
    print(f"{n} itens novos → {arq}")


def resumir() -> None:
    import numpy as np

    from phifm.eval.assistente_bracos import diferenca_pareada

    ds = _ler_jsonl(SAIDA / "posicoes.jsonl")
    saida = {"o_que_e": "posição do artigo de onde a pergunta saiu (P), por busca; "
                        "exploratório, só desenvolvimento, sem juiz e sem ΦRank",
             "buscas": BUSCAS, "conjuntos": {}}
    for nome in ("corpo_1", "primario_dev"):
        sel = [d for d in ds if d["conjunto"] == nome]
        if not sel:
            continue
        t = {"n": len(sel), "bm25_ingles_sem_palavra_de_P": sum(d["bm25_ingles_zero"]
                                                                for d in sel)}
        for b in BUSCAS:
            t[b] = {f"P@{k}": round(float(np.mean([d[b] <= k for d in sel])), 4) for k in FAIXAS}
            if b != "denso_hyde":
                x = np.array([d[b] <= 50 for d in sel], dtype=float)
                y = np.array([d["denso_hyde"] <= 50 for d in sel], dtype=float)
                dif, ic = diferenca_pareada(x, y)
                t[b]["P@50_menos_hoje"] = {"valor": dif, "ic95": ic,
                                          "resgata": int(((x == 1) & (y == 0)).sum()),
                                          "perde": int(((x == 0) & (y == 1)).sum())}
        saida["conjuntos"][nome] = t
    AGREGADO.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    for nome, t in saida["conjuntos"].items():
        print(f"\n{nome} (n={t['n']}; BM25 sem palavra de P: {t['bm25_ingles_sem_palavra_de_P']})")
        print(f"  {'busca':<22}" + "".join(f"{'P@' + str(k):>8}" for k in FAIXAS)
              + "   Δ@50 [IC] · resgata/perde")
        for b in BUSCAS:
            linha = f"  {b:<22}" + "".join(f"{t[b][f'P@{k}']:>8.3f}" for k in FAIXAS)
            if "P@50_menos_hoje" in t[b]:
                c = t[b]["P@50_menos_hoje"]
                linha += (f"   {c['valor']:+.3f} [{c['ic95'][0]:+.3f}; {c['ic95'][1]:+.3f}]"
                          f" · {c['resgata']}/{c['perde']}")
            print(linha)
    print(f"\n→ {AGREGADO.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--indexar", action="store_true")
    p.add_argument("--posicoes", action="store_true")
    p.add_argument("--resumir", action="store_true")
    a = p.parse_args()
    if a.indexar:
        indexar()
    elif a.posicoes:
        posicoes()
    elif a.resumir:
        resumir()
    else:
        p.error("--indexar, --posicoes ou --resumir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
