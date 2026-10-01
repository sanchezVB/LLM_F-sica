#!/usr/bin/env python3
"""Reaplica o registro de licenças à tabela mestra, sem reconstruí-la.

    .venv/bin/python scripts/reaplicar_licencas.py             # só mostra o plano
    .venv/bin/python scripts/reaplicar_licencas.py --gravar    # grava

## Por que isto existe, em vez de `build_spine.py`

O registro de licenças mudou em 2026-10-01: passou a ler a VERSÃO da licença
(`by/3.0` saía rotulado `CC-BY-4.0`) e a reconhecer `licenses/publicdomain/`, que
caía em `NOASSERTION`. As colunas `spdx_id` e `partition` da tabela mestra foram
gravadas pelo registro antigo e ficaram defasadas.

Reconstruir a tabela inteira conserta isso e arrisca todo o resto. Ela junta o
arXiv ao OpenAlex, e a entrada bruta do OpenAlex **não existe mais** na máquina do
corpus: uma reconstrução de 2026-09 gravou nulo sobre 14.052.319 referências e
terminou com sucesso. Para trocar duas colunas não se refaz uma junção que não se
consegue mais refazer.

Então este script faz só o que a mudança pede: lê `license`, recalcula `spdx_id` e
`partition`, e deixa as outras colunas e a ordem das linhas como estão.

## O que ele confere antes de trocar o arquivo

A troca é por arquivo temporário e `replace`, e só acontece se, lido de volta:

1. o número de linhas, o esquema e a sequência de `arxiv_id` são os mesmos;
2. a soma dos hashes de linha sobre as OUTRAS colunas é a mesma — nenhuma mudou;
3. `spdx_id` e `partition` batem com o registro, licença por licença.

E ele **recusa** qualquer plano em que um registro saia de `eval_only`. Tirar um
documento da partição que nunca treina é decisão de ADR (ADR-0001 §4), não efeito
colateral de uma correção de rótulo; se um dia for a intenção, que seja escrita.

## Depois de gravar

Os bytes da tabela mestra mudam, e com eles o manifesto da etapa e a raiz da
cadeia. Reconstruir a raiz (`scripts/manifesto_corpus.py`) e atualizar o hash onde
ele é citado é parte da mesma operação — uma raiz que atesta o arquivo antigo é
pior que raiz nenhuma.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phifm.core.console import utf8 as console_utf8  # noqa: E402
from phifm.core.licensing.registry import Partition, resolve  # noqa: E402

SPINE = Path("data/processed/spine.parquet")
RECALCULADAS = ("spdx_id", "partition")


def _expr(coluna: str, distintas: list[str | None]) -> pl.Expr:
    """`license` → `spdx_id` ou `partition`, pelo registro atual.

    Um `replace_strict` sobre os valores distintos, e não `map_elements`: são dez
    licenças em 1,6 milhão de linhas, e chamar Python por linha tornaria a
    gravação em streaming impossível.
    """
    def valor(raw: str | None) -> str:
        r = resolve(raw)
        return r.spdx_id if coluna == "spdx_id" else r.partition.value

    chaves = [r for r in distintas if r is not None]
    return (pl.col("license")
            .replace_strict(chaves, [valor(r) for r in chaves],
                            default=valor(None), return_dtype=pl.Utf8)
            .alias(coluna))


def plano(spine: Path) -> pl.DataFrame:
    """Uma linha por (licença, rótulo atual): o que há, o que passa a haver, quantos."""
    atual = (pl.scan_parquet(spine)
             .group_by("license", *RECALCULADAS).len()
             .collect())
    return atual.with_columns(
        pl.col("license").map_elements(lambda r: resolve(r).spdx_id,
                                       return_dtype=pl.Utf8, skip_nulls=False
                                       ).alias("spdx_id_novo"),
        pl.col("license").map_elements(lambda r: resolve(r).partition.value,
                                       return_dtype=pl.Utf8, skip_nulls=False
                                       ).alias("partition_novo"),
    ).sort("len", descending=True)


def mudancas(p: pl.DataFrame) -> pl.DataFrame:
    return p.filter((pl.col("spdx_id") != pl.col("spdx_id_novo"))
                    | (pl.col("partition") != pl.col("partition_novo"))
                    | pl.col("spdx_id").is_null() | pl.col("partition").is_null())


def recusar_saida_de_eval_only(p: pl.DataFrame) -> None:
    sai = p.filter((pl.col("partition") == Partition.EVAL_ONLY.value)
                   & (pl.col("partition_novo") != Partition.EVAL_ONLY.value))
    if sai.height:
        linhas = "\n".join(
            f"  {r['license']!r}: {r['len']:,} registros, eval_only → {r['partition_novo']}"
            for r in sai.iter_rows(named=True))
        raise SystemExit(
            "RECUSADO: o plano tira registros de `eval_only`, a partição que nunca "
            "treina.\n" + linhas + "\nIsso é decisão de ADR (ADR-0001 §4), não "
            "correção de rótulo. Nada foi gravado.")


def _digesto(caminho: Path) -> tuple[int, int]:
    """(linhas, soma dos hashes de linha sobre as colunas que NÃO são recalculadas)."""
    r = (pl.scan_parquet(caminho)
         .select(pl.len().alias("n"),
                 pl.struct(pl.all().exclude(*RECALCULADAS)).hash(seed=17).sum().alias("h"))
         .collect(engine="streaming"))
    return r["n"][0], r["h"][0]


def gravar(spine: Path) -> Path:
    """Recalcula as duas colunas e troca o arquivo — só se as três conferências passarem."""
    lf = pl.scan_parquet(spine)
    esquema = lf.collect_schema()
    distintas = lf.select("license").unique().collect()["license"].to_list()

    tmp = spine.with_suffix(".parquet.tmp")
    lf.with_columns(*(_expr(c, distintas) for c in RECALCULADAS)
                    ).sink_parquet(tmp, compression="zstd")
    try:
        novo = pl.scan_parquet(tmp)
        if novo.collect_schema() != esquema:
            raise SystemExit("ABORTADO: o esquema mudou. Nada foi trocado.")
        ids_antes = lf.select("arxiv_id").collect()["arxiv_id"]
        ids_depois = novo.select("arxiv_id").collect()["arxiv_id"]
        if not ids_antes.equals(ids_depois):
            raise SystemExit("ABORTADO: a sequência de `arxiv_id` mudou. Nada foi trocado.")
        if _digesto(spine) != _digesto(tmp):
            raise SystemExit("ABORTADO: alguma coluna fora de `spdx_id` e `partition` "
                             "mudou. Nada foi trocado.")
        resto = mudancas(plano(tmp))
        if resto.height:
            raise SystemExit(f"ABORTADO: {resto['len'].sum():,} registros ainda divergem "
                             "do registro depois de recalculados. Nada foi trocado.")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(spine)
    return spine


def _imprimir(p: pl.DataFrame, total: int) -> None:
    m = mudancas(p)
    if not m.height:
        print("Nada a fazer: `spdx_id` e `partition` já batem com o registro.")
        return
    print(f"{m['len'].sum():,} de {total:,} registros mudam:\n")
    for r in m.iter_rows(named=True):
        print(f"  {r['len']:>9,}  {r['license']}")
        if r["spdx_id"] != r["spdx_id_novo"]:
            print(f"             spdx_id    {r['spdx_id']} → {r['spdx_id_novo']}")
        if r["partition"] != r["partition_novo"]:
            print(f"             partition  {r['partition']} → {r['partition_novo']}")
    print("\nPartições:")
    antes = p.group_by("partition").agg(pl.col("len").sum()).sort("partition")
    depois = dict(p.group_by("partition_novo").agg(pl.col("len").sum()).iter_rows())
    for parte, n in antes.iter_rows():
        d = depois.get(parte, 0)
        print(f"  {parte:11} {n:>9,} ({100 * n / total:5.2f}%)  →  "
              f"{d:>9,} ({100 * d / total:5.2f}%)")


def main() -> int:
    console_utf8()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spine", type=Path, default=SPINE)
    ap.add_argument("--gravar", action="store_true",
                    help="grava; sem isto, só mostra o plano")
    a = ap.parse_args()

    p = plano(a.spine)
    total = int(p["len"].sum())
    _imprimir(p, total)
    recusar_saida_de_eval_only(p)
    if not mudancas(p).height:
        return 0
    if not a.gravar:
        print("\nNada foi gravado. Repita com --gravar.")
        return 0
    gravar(a.spine)
    print(f"\nGravado: {a.spine} ({a.spine.stat().st_size / 1e6:.1f} MB).")
    print("⚠️ Os bytes mudaram: reconstrua a raiz da cadeia "
          "(scripts/manifesto_corpus.py) e atualize o hash onde ele é citado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
