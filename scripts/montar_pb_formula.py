#!/usr/bin/env python3
"""Monta o PB-Formula sobre o corpus de Física inteiro. DOC-11 §6.3.

    PYTHONPATH=src python scripts/montar_pb_formula.py            # as duas fases
    PYTHONPATH=src python scripts/montar_pb_formula.py --limite-partes 1 --out <tmp>   # ensaio

A parte pura — filtro de conteúdo, agrupamento, guardas, teto do pool — está em
`phifm.eval.benchmarks.formula` e é testada na suíte rápida. Aqui só o que depende do
disco e do tamanho do corpus.

## Por que duas fases, e não "extrair e agrupar"

O corpus de Física tem 835.379 documentos. Medido em 20.000, são ~34 formas de
conteúdo DISTINTAS por documento: ~28 milhões de ocorrências no corpus inteiro, quase
todas únicas. Juntar isso em memória não cabe em 8 GB.

1. **Extração**, em paralelo, uma parte do corpus por processo: cada parte vira um
   parquet de `(forma, documento, latex)` em `ocorrencias/`. Escrito num arquivo
   temporário e renomeado no fim, então uma parte que existe está completa — e a fase
   RETOMA pulando as que existem. São horas de CPU; uma queda no meio não recomeça do
   zero.
2. **Montagem**: conta documentos por forma usando os 64 primeiros bits do hash (8
   bytes por chave em vez de 32 caracteres — a contagem de ~28 milhões de formas cabe),
   carrega SÓ as ocorrências das formas em 2+ documentos, e entrega essas ao
   `montar_itens`, que agrupa pelo hash COMPLETO. Uma colisão nos 64 bits (esperado:
   ~2×10⁻⁵ pares em 28 milhões) só traria uma ocorrência a mais para dentro, nunca
   agruparia errado.

## Escopo: só a fatia de Física

A fatia `math`+`cs` coletada em 2026-09-12 NÃO entra. O benchmark é de Física (PhysBench),
e misturar mudaria o que ele mede.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

import polars as pl  # noqa: E402

from phifm.core.schema.reprodutibilidade import entrada_de, gravar_manifesto_etapa  # noqa: E402
from phifm.eval.benchmarks.formula import (  # noqa: E402
    MAX_DOCUMENTOS,
    MAX_ITENS_POR_DOCUMENTO,
    MIN_CARACTERES,
    Ocorrencia,
    conferir_pool,
    montar_itens,
    ocorrencias_do_documento,
)

log = logging.getLogger("pb-formula")

CORPUS = Path("data/processed/redpajama_fisica")
SAIDA = Path("data/processed/pb_formula")
RESUMO_VERSIONADO = Path("data/processed/avaliacao/pb_formula_montagem.json")


def extrair_parte(args: tuple[str, str, int]) -> dict:
    """Uma parte do corpus -> um parquet de ocorrências. Roda num processo filho."""
    parte, destino, min_caracteres = args
    t0 = time.perf_counter()
    df = pl.read_parquet(parte, columns=["arxiv_id", "texto"])
    formas_col, docs_col, latex_col = [], [], []
    cont = {"documentos": df.height, "equacoes": 0, "falhas_canonizacao": 0, "de_conteudo": 0}
    for aid, texto in df.iter_rows():
        if not texto:
            continue
        formas, c = ocorrencias_do_documento(texto, min_caracteres)
        for k in ("equacoes", "falhas_canonizacao", "de_conteudo"):
            cont[k] += c[k]
        for forma, latex in formas.items():
            formas_col.append(forma)
            docs_col.append(aid)
            latex_col.append(latex)
    del df
    tmp = Path(destino).with_suffix(".tmp")
    pl.DataFrame({"forma": formas_col, "documento": docs_col, "latex": latex_col},
                 schema={"forma": pl.String, "documento": pl.String, "latex": pl.String}
                 ).write_parquet(tmp, compression="zstd")
    os.replace(tmp, destino)
    cont.update({"parte": Path(parte).name, "ocorrencias": len(formas_col),
                 "segundos": round(time.perf_counter() - t0, 1)})
    Path(destino).with_suffix(".json").write_text(json.dumps(cont), encoding="utf-8")
    return cont


def fase_extracao(partes: list[Path], dir_oc: Path, processos: int, min_car: int) -> None:
    dir_oc.mkdir(parents=True, exist_ok=True)
    pendentes = [p for p in partes if not (dir_oc / f"{p.stem}.parquet").exists()]
    log.info("extração: %d partes, %d já feitas, %d pendentes · %d processos",
             len(partes), len(partes) - len(pendentes), len(pendentes), processos)
    if not pendentes:
        return
    t0, feitas = time.perf_counter(), 0
    tarefas = [(str(p), str(dir_oc / f"{p.stem}.parquet"), min_car) for p in pendentes]
    with mp.Pool(processos) as pool:
        for r in pool.imap_unordered(extrair_parte, tarefas):
            feitas += 1
            dec = time.perf_counter() - t0
            log.info("  %s · %d docs · %d ocorrências · %.0f s · %d/%d · faltam ~%.0f min",
                     r["parte"], r["documentos"], r["ocorrencias"], r["segundos"],
                     feitas, len(pendentes), dec / feitas * (len(pendentes) - feitas) / 60)


def fase_montagem(partes: list[Path], dir_oc: Path, out: Path, a) -> dict:
    shards = [dir_oc / f"{p.stem}.parquet" for p in partes]
    faltando = [s.name for s in shards if not s.exists()]
    if faltando:
        raise SystemExit(f"a extração não terminou: faltam {len(faltando)} partes "
                         f"({faltando[:3]}…). Rode de novo sem --so-montar.")
    contagens = [json.loads(s.with_suffix(".json").read_text(encoding="utf-8")) for s in shards]
    extracao = {k: sum(c[k] for c in contagens)
                for k in ("documentos", "equacoes", "falhas_canonizacao", "de_conteudo",
                          "ocorrencias")}

    lf = pl.scan_parquet([str(s) for s in shards])
    chave = pl.col("forma").str.slice(0, 16).str.to_integer(base=16, dtype=pl.UInt64)
    t0 = time.perf_counter()
    por_forma = (lf.select(chave.alias("k")).group_by("k").agg(pl.len().alias("docs"))
                 .collect(engine="streaming"))
    formas_totais = por_forma.height
    um_doc = int((por_forma["docs"] == 1).sum())
    selecionadas = por_forma.filter(pl.col("docs") >= 2)["k"]
    log.info("contagem: %s formas · %s em 1 documento · %s em 2+ · %.0f s",
             f"{formas_totais:,}", f"{um_doc:,}", f"{selecionadas.len():,}",
             time.perf_counter() - t0)
    del por_forma

    occ = (lf.with_columns(chave.alias("k")).filter(pl.col("k").is_in(selecionadas.implode()))
           .select("forma", "documento", "latex").collect(engine="streaming"))
    ocorrencias = [Ocorrencia(documento=d, forma=f, latex=lx) for f, d, lx in occ.iter_rows()]
    del occ
    itens, est = montar_itens(ocorrencias, max_documentos=a.max_documentos,
                              max_por_documento=a.max_por_documento)
    # As formas de um documento só nunca chegaram ao `montar_itens`: entram aqui, para a
    # estatística descrever o corpus inteiro e não só o que passou pela contagem.
    est.ocorrencias = extracao["ocorrencias"]
    est.formas = formas_totais
    est.descartadas_um_documento_so += um_doc

    pool = {i.documento_da_consulta for i in itens}
    for i in itens:
        pool |= i.alvos
    conferir_pool(itens, pool)

    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "forma": [i.forma for i in itens],
        "consulta": [i.consulta for i in itens],
        "documento_da_consulta": [i.documento_da_consulta for i in itens],
        "alvos": [sorted(i.alvos) for i in itens],
    }).write_parquet(out / "itens.parquet", compression="zstd")

    resumo = {
        "benchmark": "PB-Formula (DOC-11 §6.3)",
        "corpus": str(a.corpus).replace("\\", "/"),
        "partes": len(partes),
        "parametros": {"min_caracteres": a.min_caracteres,
                       "max_documentos": a.max_documentos,
                       "max_por_documento": a.max_por_documento},
        "extracao": extracao,
        "estatisticas": est.como_dict(),
        "pool_documentos": len(pool),
        "teto_do_pool": 1.0,
        "alvos_por_item": {
            "media": round(sum(len(i.alvos) for i in itens) / max(len(itens), 1), 3),
            "max": max((len(i.alvos) for i in itens), default=0)},
    }
    (out / "pb_formula.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    return resumo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=SAIDA)
    ap.add_argument("--processos", type=int, default=4,
                    help="uma parte do corpus em memória por processo; a maior tem "
                         "1,05 GB de texto, então 4 cabem em 8 GB livres")
    ap.add_argument("--min-caracteres", type=int, default=MIN_CARACTERES)
    ap.add_argument("--max-documentos", type=int, default=MAX_DOCUMENTOS)
    ap.add_argument("--max-por-documento", type=int, default=MAX_ITENS_POR_DOCUMENTO)
    ap.add_argument("--limite-partes", type=int, default=None, help="ensaio")
    ap.add_argument("--so-montar", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    partes = sorted(a.corpus.glob("part-*.parquet"))
    if a.limite_partes:
        partes = partes[:a.limite_partes]
    if not partes:
        raise SystemExit(f"nenhum part-*.parquet em {a.corpus}")
    dir_oc = a.out / "ocorrencias"
    if not a.so_montar:
        fase_extracao(partes, dir_oc, a.processos, a.min_caracteres)
    resumo = fase_montagem(partes, dir_oc, a.out, a)

    if a.limite_partes is None:
        RESUMO_VERSIONADO.parent.mkdir(parents=True, exist_ok=True)
        RESUMO_VERSIONADO.write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
        gravar_manifesto_etapa(
            etapa="pb_formula",
            descricao="PB-Formula: itens de recuperação por equação sob variação notacional",
            raiz=a.out, entradas=[entrada_de(a.corpus)],
            parametros={"script": "scripts/montar_pb_formula.py", **resumo["parametros"],
                        "corpus": resumo["corpus"], "partes": resumo["partes"]},
            registros=resumo["estatisticas"]["itens"])

    e = resumo["estatisticas"]
    print()
    print("=" * 74)
    print(f"  PB-Formula · {resumo['partes']} partes · {resumo['extracao']['documentos']:,} documentos")
    print("=" * 74)
    print(f"  equações extraídas       {resumo['extracao']['equacoes']:>12,}")
    print(f"  falhas de canonização    {resumo['extracao']['falhas_canonizacao']:>12,}")
    print(f"  formas de conteúdo       {e['formas']:>12,}")
    print(f"  em 1 documento só        {e['descartadas_um_documento_so']:>12,}")
    print(f"  comuns demais (>{a.max_documentos})       {e['descartadas_comuns_demais']:>12,}")
    print(f"  teto por documento       {e['descartadas_teto_por_documento']:>12,}")
    print(f"  ITENS                    {e['itens']:>12,}  ·  {e['documentos_de_consulta']:,} documentos de consulta")
    print(f"  pool                     {resumo['pool_documentos']:>12,} documentos · teto 1,0")
    print("=" * 74)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
