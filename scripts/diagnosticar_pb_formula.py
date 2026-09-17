#!/usr/bin/env python3
"""O que os itens do PB-Formula exigem de fato: variação notacional e documentos quase iguais.

    PYTHONPATH=src python scripts/diagnosticar_pb_formula.py
    PYTHONPATH=src python scripts/diagnosticar_pb_formula.py --pb <ensaio> --out <json>

Roda DEPOIS de `montar_pb_formula.py`, sobre as ocorrências que ele deixou em disco, e
não muda nenhum item. Grava `itens_estratos.parquet` ao lado de `itens.parquet` — uma
linha por item, pela `forma` — e um resumo versionado.

## Por que existe

Um ensaio de 2 partes (26.292 documentos) e depois o corpus inteiro (2026-09-17,
374.739 itens) mostraram duas coisas que a montagem não mede:

1. **Grafia.** 53,1% dos itens têm todos os alvos com a grafia da consulta byte a byte,
   e mais 37,0% só diferem em espaço, rótulo, alinhamento ou pontuação final. Só **7,8%**
   exigem variação notacional em todos os alvos. O DOC-11 §6.3 promete "sob variação
   notacional"; em nove de cada dez itens um casador de string basta.
   `formula.estrato_de_grafia` classifica cada item, e a avaliação tem de reportar
   recall por estrato.
2. **Documentos quase iguais.** 36,3% dos itens ligam a consulta a um alvo com o qual
   ela divide 5 ou mais equações de conteúdo, e 1.490 pares de documentos dividem 50 ou
   mais — a mesma obra em dois registros. Nesses itens o que se recupera é o documento
   parecido, não a equação. `max_formas_em_comum` é, para cada item, o número de formas
   que o documento da consulta divide com o alvo MAIS parecido.

Notacional e sem alvo quase igual (< 5 formas em comum) sobram **24.508** itens.

A contagem de pares usa TODAS as formas em 2 a `max_documentos` documentos, inclusive
as que o teto por documento descartou: é a semelhança entre os documentos que se quer,
não entre os itens que sobraram.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

import polars as pl  # noqa: E402

from phifm.eval.benchmarks.formula import (  # noqa: E402
    ESTRATOS_DE_GRAFIA,
    MAX_DOCUMENTOS,
    estrato_de_grafia,
)

PB = Path("data/processed/pb_formula")
RESUMO = Path("data/processed/avaliacao/pb_formula_diagnostico.json")
LIMIARES_COMUNS = (2, 3, 5, 10, 20, 50)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb", type=Path, default=PB)
    ap.add_argument("--out", type=Path, default=RESUMO)
    ap.add_argument("--max-documentos", type=int, default=MAX_DOCUMENTOS)
    a = ap.parse_args()

    itens = pl.read_parquet(a.pb / "itens.parquet")
    lf = pl.scan_parquet(str(a.pb / "ocorrencias" / "*.parquet"))
    # A mesma chave de 64 bits da montagem: contar ~28 milhões de formas pelo texto do
    # hash não cabe em 8 GB.
    chave = pl.col("forma").str.slice(0, 16).str.to_integer(base=16, dtype=pl.UInt64)
    # ⚠️ A contagem é de LINHAS, sem teto superior. O corpus tem arxiv_id repetido
    # (6.778 ids em duas linhas), então uma forma em 21 linhas pode estar em 20
    # documentos e virar item. Filtrar por linhas até `max_documentos`, como a primeira
    # versão fazia, deixou 5 itens com alvo sem ocorrência — a guarda abaixo pegou.
    contagem = (lf.select(chave.alias("k")).group_by("k").agg(pl.len().alias("linhas"))
                .filter(pl.col("linhas") >= 2).collect(engine="streaming"))
    occ = (lf.with_columns(chave.alias("k"))
           .filter(pl.col("k").is_in(contagem["k"].implode()))
           .select("forma", "documento", "latex").collect(engine="streaming"))
    del contagem
    # Um documento por (forma, id), com a grafia mais curta — a mesma regra de
    # `ocorrencias_do_documento`. Sem isto as duas cópias de um id contariam os pares
    # em dobro.
    occ = (occ.sort(pl.col("latex").str.len_bytes(), "latex")
           .unique(subset=["forma", "documento"], keep="first", maintain_order=True))
    docs_por_forma = occ.group_by("forma").agg(pl.len().alias("docs"))

    # ── pares de documentos e quantas formas dividem ────────────────────────
    d = (occ.join(docs_por_forma.filter(pl.col("docs").is_between(2, a.max_documentos)),
                  on="forma", how="semi")
         .select("forma", "documento"))
    pares = (d.join(d, on="forma", suffix="_b")
             .filter(pl.col("documento") < pl.col("documento_b"))
             .group_by("documento", "documento_b").agg(pl.len().alias("comuns")))

    explodido = itens.select("forma", "consulta", "documento_da_consulta",
                             pl.col("alvos").alias("alvo")).explode("alvo")
    com_pares = (explodido
                 .with_columns(pl.min_horizontal("documento_da_consulta", "alvo")
                               .alias("documento"),
                               pl.max_horizontal("documento_da_consulta", "alvo")
                               .alias("documento_b"))
                 .join(pares, on=["documento", "documento_b"], how="left"))
    proximidade = com_pares.group_by("forma").agg(
        pl.col("comuns").max().alias("max_formas_em_comum"))

    # ── grafia dos alvos contra a da consulta ───────────────────────────────
    grafias = (explodido
               .join(occ.select("forma", pl.col("documento").alias("alvo"),
                                pl.col("latex").alias("grafia_alvo")),
                     on=["forma", "alvo"], how="left")
               .group_by("forma").agg(pl.col("consulta").first(),
                                      pl.col("grafia_alvo")))
    faltando = grafias.filter(pl.col("grafia_alvo").list.eval(pl.element().is_null()).list.any())
    if faltando.height:
        raise SystemExit(f"{faltando.height} itens com alvo sem ocorrência em disco: as "
                         "ocorrências não são as da montagem que gerou `itens.parquet`")
    estratos = grafias.select(
        "forma",
        pl.struct("consulta", "grafia_alvo").map_elements(
            lambda r: estrato_de_grafia(r["consulta"], r["grafia_alvo"]),
            return_dtype=pl.String).alias("grafia"))

    saida = estratos.join(proximidade, on="forma", how="left")
    if saida.height != itens.height:
        raise SystemExit(f"{saida.height} estratos para {itens.height} itens")
    saida.write_parquet(a.pb / "itens_estratos.parquet", compression="zstd")

    n = saida.height
    por_grafia = {e: int((saida["grafia"] == e).sum()) for e in ESTRATOS_DE_GRAFIA}
    m = saida["max_formas_em_comum"]
    cruzado = (saida.with_columns((pl.col("max_formas_em_comum") >= 5).alias("quase_igual"))
               .group_by("grafia", "quase_igual").agg(pl.len().alias("itens"))
               .sort("grafia", "quase_igual"))
    resumo = {
        "benchmark": "PB-Formula (DOC-11 §6.3) — diagnóstico, não muda os itens",
        "exploratorio": True,
        "itens": n,
        "grafia": {e: {"itens": c, "fracao": round(c / n, 4)} for e, c in por_grafia.items()},
        "documento_quase_igual": {
            "definicao": "max_formas_em_comum: formas de conteúdo que o documento da "
                         "consulta divide com o alvo mais parecido",
            "itens_com_ao_menos": {str(k): {"itens": int((m >= k).sum()),
                                            "fracao": round(float((m >= k).mean()), 4)}
                                   for k in LIMIARES_COMUNS},
            "pares_de_documentos": pares.height,
            "pares_com_ao_menos": {str(k): int((pares["comuns"] >= k).sum())
                                   for k in LIMIARES_COMUNS},
        },
        "grafia_x_quase_igual_5": cruzado.to_dicts(),
        "nota": ("A grafia de cada documento é a mais curta que ele usa para a forma: "
                 "`identica` e `so_espacos` são pisos. Recall no PB-Formula tem de sair "
                 "por estrato de grafia, e com e sem os itens de documento quase igual."),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 74)
    print(f"  PB-Formula · diagnóstico de {n:,} itens")
    print("=" * 74)
    for e, c in por_grafia.items():
        print(f"  grafia {e:<12} {c:>10,}  ({c / n:.1%})")
    for k in LIMIARES_COMUNS:
        print(f"  alvo divide >= {k:>2} formas  {int((m >= k).sum()):>10,}  ({(m >= k).mean():.1%})")
    print("=" * 74)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
