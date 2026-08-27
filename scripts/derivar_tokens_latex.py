#!/usr/bin/env python3
"""DOC-05 §4.1 — as sequências de controle LaTeX do tokenizer do ΦEnc, DERIVADAS.

    .venv/Scripts/python.exe scripts/derivar_tokens_latex.py --max-docs 50000
    .venv/Scripts/python.exe scripts/derivar_tokens_latex.py

## Por que derivar em vez de escrever a lista à mão

DOC-05 §4.1 é explícito: *"Uma lista curada à mão de tokens importantes de Física
seria enviesada pelo autor da lista"*. O método usa o corpus que já existe.

## ⚠️ Frequência de DOCUMENTO, não frequência bruta

O piso é "presente em ≥ 500 documentos distintos". Contar ocorrências brutas deixaria
um único paper com 10.000 usos de uma macro pessoal (`\\eps`, `\\bra`) entrar no
vocabulário como se fosse notação universal. Documento distinto é o denominador que
mede difusão, e difusão é o que justifica gastar um token do orçamento.

## O orçamento que isto preenche (DOC-05 §7)

    bytes (fallback)              256   obrigatório
    especiais                     ~16
    dígitos                        10   §5.3, dígito único
    sequências de controle LaTeX ~2000  <- ESTE script
    estruturais matemáticos       ~64   §4.2, adicionados à mão de propósito
    unicode matemático           ~256
    aprendidos por BPE/Unigram  ~38400
    ────────────────────────────────────
    V = 40.960

A estimativa do documento é 1.200–2.500 passando o piso. Se sair muito fora disso, o
piso ou o corpus estão errados — e é melhor descobrir aqui que depois de congelar.

## Por que congelar isto cedo

DOC-05 §1: o tokenizer é *"a última decisão reversível barata do pipeline de dados"*.
Retokenizar depois significa refazer o upload de 20–80 GB numa conexão doméstica.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

import polars as pl

log = logging.getLogger("tokens-latex")

# ⚠️ `\\[a-zA-Z]+\\*?` — exatamente o padrão do DOC-05 §4.1. O `*` opcional captura
# as variantes estreladas (`\section*`, `\align*`), que são sequências distintas.
CONTROLE = re.compile(r"\\[a-zA-Z]+\*?")

# §4.2: raros como string, críticos como estrutura. Entram à mão porque o piso de
# frequência de documento não os pega — e o documento diz para adicioná-los.
ESTRUTURAIS = [
    "^{", "_{", "}", "{", "&", "\\\\",
    "\\,", "\\;", "\\!", "\\quad", "\\qquad",
]


def _fontes(raiz: Path) -> list[tuple[str, Path, str]]:
    """(rótulo, caminho, coluna de texto) das fontes que têm LaTeX.

    peS2o é texto completo de paper — a fonte mais rica em notação. Os pares de
    citação são título+resumo do arXiv, que também carregam LaTeX inline.
    """
    fontes = []
    pes2o = raiz / "data/processed/pes2o_fisica"
    if pes2o.is_dir() and any(pes2o.glob("part-*.parquet")):
        fontes.append(("peS2o", pes2o / "part-*.parquet", "texto"))
    pares = raiz / "data/processed/pares/pares_validacao.parquet"
    if pares.exists():
        fontes.append(("arXiv (pares)", pares, "positivo"))
    return fontes


def contar(caminho: Path, coluna: str, max_docs: int | None,
           rotulo: str) -> tuple[Counter, int]:
    """Frequência de DOCUMENTO de cada sequência de controle.

    Streaming e por lotes: o corpus tem 58 G caracteres e materializá-lo mataria a
    máquina. Cada documento contribui no MÁXIMO 1 para cada sequência — é isso que
    faz a contagem ser de documento e não de ocorrência.
    """
    df = Counter()
    n_docs = 0
    t0 = time.perf_counter()
    lazy = pl.scan_parquet(caminho).select(coluna)
    if max_docs:
        lazy = lazy.head(max_docs)
    # `collect(streaming)` devolve tudo; para não materializar, fatiamos com slice.
    total = lazy.select(pl.len()).collect(engine="streaming").item()
    passo = 100_000
    for inicio in range(0, total, passo):
        bloco = lazy.slice(inicio, passo).collect(engine="streaming")
        for texto in bloco[coluna]:
            if not texto:
                continue
            n_docs += 1
            # `set` ANTES de contar: sem isso a contagem volta a ser por ocorrência.
            for seq in set(CONTROLE.findall(texto)):
                df[seq] += 1
        feito = min(inicio + passo, total)
        taxa = feito / max(time.perf_counter() - t0, 1e-9)
        log.info("  %s: %s/%s documentos · %.0f/s · %s sequências distintas",
                 rotulo, f"{feito:,}", f"{total:,}", taxa, f"{len(df):,}")
    return df, n_docs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raiz", type=Path, default=Path("."))
    p.add_argument("--piso", type=int, default=500,
                   help="documentos distintos mínimos (DOC-05 §4.1: 500)")
    p.add_argument("--max-docs", type=int, default=None,
                   help="teto por fonte, para teste de fumaça")
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/tokenizer/tokens_latex.json"))
    a = p.parse_args()

    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    fontes = _fontes(a.raiz)
    if not fontes:
        raise SystemExit(
            "nenhuma fonte com texto encontrada. Esperava "
            "data/processed/pes2o_fisica/part-*.parquet ou "
            "data/processed/pares/pares_validacao.parquet")
    log.info("fontes: %s", ", ".join(r for r, _, _ in fontes))

    total_df: Counter = Counter()
    docs_por_fonte = {}
    for rotulo, caminho, coluna in fontes:
        df, n = contar(caminho, coluna, a.max_docs, rotulo)
        docs_por_fonte[rotulo] = n
        total_df.update(df)
        log.info("%s: %s documentos · %s sequências distintas", rotulo,
                 f"{n:,}", f"{len(df):,}")

    n_docs = sum(docs_por_fonte.values())
    passaram = {s: c for s, c in total_df.items() if c >= a.piso}
    ordenado = sorted(passaram.items(), key=lambda kv: (-kv[1], kv[0]))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "metodo": "DOC-05 §4.1 — frequência de DOCUMENTO, não de ocorrência",
        "piso_documentos": a.piso,
        "documentos_varridos": n_docs,
        "documentos_por_fonte": docs_por_fonte,
        "sequencias_distintas_vistas": len(total_df),
        "sequencias_que_passaram": len(ordenado),
        "estimativa_do_documento": "1.200-2.500",
        "estruturais_adicionados_a_mao": ESTRUTURAIS,
        "nota_estruturais": ("DOC-05 §4.2: raros como string, críticos como "
                             "estrutura. O piso de frequência não os pega."),
        "sequencias": [{"token": s, "documentos": c} for s, c in ordenado],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 70)
    print(f"  {n_docs:,} documentos varridos · {len(total_df):,} sequências distintas")
    print(f"  passaram o piso de {a.piso} documentos: {len(ordenado):,}")
    print("  estimativa do DOC-05 §4.1: 1.200-2.500")
    print("=" * 70)
    print("  as 25 mais difundidas:")
    for s, c in ordenado[:25]:
        print(f"    {s:<18} {c:>9,} documentos ({100*c/max(n_docs,1):.1f}%)")
    print("=" * 70)
    print(f"  -> {a.out}")
    if not 800 <= len(ordenado) <= 4000:
        print()
        print(f"  ⚠️ {len(ordenado)} está fora da faixa esperada de 1.200-2.500 do")
        print("     DOC-05 §4.1. Antes de congelar, verifique se o piso e o corpus")
        print("     são os certos — congelar errado custa reupload de 20-80 GB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
