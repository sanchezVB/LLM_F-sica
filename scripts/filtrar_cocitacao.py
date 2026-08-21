#!/usr/bin/env python3
"""Remove dos negativos minerados os que a literatura co-cita com o positivo.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/filtrar_cocitacao.py

## O defeito que isto conserta

`minerar_negativos.py` exclui o que a **âncora** cita. Medido em 2026-08-18, isso é
necessário e insuficiente: **15,2%** dos negativos minerados são **co-citados com o
positivo** — existe um paper que cita os dois juntos —, contra 0,1% num controle de
negativos sorteados. Razão de 212×.

Co-citação é o sinal clássico de relevância em recuperação: dois documentos citados
pelo mesmo artigo tratam da mesma coisa. Usá-los como negativo em contraste ensina o
modelo a afastar o que a literatura agrupa.

O treino com os negativos não filtrados ficou ABAIXO da linha de base, e a
assinatura nas métricas foi específica:

    passo    recall@1   recall@10
    base       0,265      0,665
    200        0,247      0,655
    400        0,238      0,680

Recall@1 caindo enquanto recall@10 sobe é o modelo aproximando a vizinhança inteira
e perdendo a capacidade de escolher qual dela é a citação certa.

## Filtrar em vez de re-minerar

Re-minerar custaria ~45 min (a busca; os vetores estão em cache). Filtrar o artefato
existente custa minutos e dá o mesmo resultado: a mineração ordenou por
similaridade, e remover alguns candidatos dessa ordem não muda a ordem dos que
sobram. O que se perde é volume — cada par fica com ~6,8 negativos em vez de 8.

## ⚠️ O que este filtro NÃO resolve

Co-citação é uma aproximação de relevância, não a relevância. Um paper relacionado
que ninguém citou junto com o positivo continua passando como negativo. O filtro
reduz o ruído de 15,2% para o que a co-citação alcança — e o número residual não é
medido, porque medi-lo exigiria julgamento humano sobre relevância, que é
justamente o que o benchmark próprio não tem (ressalva do G1 no DOC-00 §5).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    Entrada,
    gravar_manifesto_etapa,
)

log = logging.getLogger("cocitacao")


def citadores_por_documento(pares: Path) -> dict[str, set[str]]:
    """`documento → conjunto de papers que o citam`, da tabela INTEIRA de arestas.

    Da tabela inteira e não do subconjunto minerado: co-citação é uma propriedade da
    literatura, não do recorte que este treino usa. Um paper que cita o positivo e o
    negativo juntos estabelece a relação independentemente de estar nos 400 mil
    primeiros pares.
    """
    arestas = pl.read_parquet(pares / "pares_treino.parquet",
                              columns=["arxiv_id", "arxiv_citado"])
    return {k: set(v) for k, v in
            arestas.group_by("arxiv_citado")
            .agg(pl.col("arxiv_id").unique()).iter_rows()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--negativos", type=Path,
                   default=Path("data/processed/negativos_dificeis/pares_com_negativos.parquet"))
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/negativos_dificeis/pares_limpos.parquet"))
    p.add_argument("--min-negativos", type=int, default=1,
                   help="pares que sobrarem com menos que isto são reportados, não "
                        "descartados: descartar mudaria o conjunto de treino")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s",
                        stream=sys.stdout)

    log.info("montando os citadores da tabela inteira de arestas…")
    cit = citadores_por_documento(a.pares)
    log.info("documentos com citadores conhecidos: %s", f"{len(cit):,}")

    d = pl.read_parquet(a.negativos)
    antes_total = int(d["negativos_id"].list.len().sum())

    limpos_id: list[list[str]] = []
    limpos_tx: list[list[str]] = []
    removidos = 0
    for pos, nids, ntxs in d.select(["arxiv_citado", "negativos_id",
                                     "negativos"]).iter_rows():
        quem_cita_pos = cit.get(pos, set())
        ids, txs = [], []
        for nid, ntx in zip(nids, ntxs, strict=True):
            if quem_cita_pos & cit.get(nid, set()):
                removidos += 1
                continue
            ids.append(nid)
            txs.append(ntx)
        limpos_id.append(ids)
        limpos_tx.append(txs)

    d = d.with_columns([pl.Series("negativos_id", limpos_id),
                        pl.Series("negativos", limpos_tx)])
    depois_total = int(d["negativos_id"].list.len().sum())
    vazios = int(d["negativos_id"].list.len().eq(0).sum())
    poucos = int(d["negativos_id"].list.len().lt(a.min_negativos).sum())

    a.out.parent.mkdir(parents=True, exist_ok=True)
    d.write_parquet(a.out, compression="zstd")

    meta = {
        "origem": str(a.negativos),
        "negativos_antes": antes_total,
        "negativos_depois": depois_total,
        "removidos_por_cocitacao": removidos,
        "taxa_removida": round(removidos / max(antes_total, 1), 4),
        "media_por_par_antes": round(antes_total / max(len(d), 1), 2),
        "media_por_par_depois": round(depois_total / max(len(d), 1), 2),
        "pares_sem_negativo_depois": vazios,
        "pares_abaixo_do_minimo": poucos,
        "criterio": ("removido se existe um paper que cita o positivo E o negativo "
                     "(co-citação, sinal clássico de relevância)"),
        "ressalva": ("co-citação aproxima relevância, não a define. Um paper "
                     "relacionado que ninguém citou junto com o positivo continua "
                     "passando como negativo, e esse residual NÃO é medido."),
    }
    (a.out.parent / "_cocitacao.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    gravar_manifesto_etapa(
        etapa="negativos_limpos",
        descricao="Negativos difíceis sem os co-citados com o positivo",
        raiz=a.out,
        entradas=[Entrada(caminho=str(a.negativos)),
                  Entrada(caminho=str(a.pares / "pares_treino.parquet"))],
        parametros={"script": "scripts/filtrar_cocitacao.py", **meta},
        registros=len(d))

    print()
    print("=" * 70)
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print("=" * 70)
    if vazios:
        print(f"  ⚠️ {vazios:,} pares ficaram SEM negativo. Eles seguem no conjunto "
              "com sentinela mascarada — descartá-los mudaria o conjunto de treino "
              "e o experimento deixaria de isolar uma variável só.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
