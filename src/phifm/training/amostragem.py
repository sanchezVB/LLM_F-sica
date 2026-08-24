"""Amostragem de pares para treino e avaliação — polars puro, sem torch.

Mora fora de `rerank.py` de propósito: é a guarda contra o defeito que fez a
métrica do ΦRank medir 35 documentos em vez de 500, e uma guarda só serve se o
teste dela roda na suíte rápida. `rerank.py` importa torch, e o teste morria na
coleta da venv principal.
"""

from __future__ import annotations

import polars as pl


def amostrar_por_documento(d: pl.DataFrame, n: int,
                           semente: int = 17) -> tuple[pl.DataFrame, int]:
    """Sorteia `n` linhas e devolve `(amostra, documentos distintos)`.

    ⚠️ Existe para que `head` nunca volte. Medido em 2026-08-24: o parquet de pares
    vem AGRUPADO por documento citado, então as primeiras linhas são poucos papers
    repetidos ~14 a ~22 vezes:

        val.head(  200) ->   200 linhas ·  16 documentos
        val.head(  500) ->   500 linhas ·  35 documentos
        val.sample(500) ->   500 linhas · 259 documentos

    Linhas do mesmo documento citado não são observações independentes: o n efetivo
    de qualquer métrica é o número de DOCUMENTOS. Com 35, o acerto@1 de 0,364 tinha
    intervalo de 95% de ±0,159 e não se distinguia da base de 0,198 — e as divisões
    contaminada e honesta reportaram o mesmo número porque as duas mediam as mesmas
    três dezenas de papers.

    O segundo valor devolvido é o n efetivo, e serve para dimensionar o intervalo.
    """
    amostra = d.sample(n=n, seed=semente) if n and n < len(d) else d
    n_doc = (amostra["arxiv_citado"].n_unique()
             if "arxiv_citado" in amostra.columns else len(amostra))
    return amostra, int(n_doc)
