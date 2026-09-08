"""Amostragem de pares para treino e avaliação — polars puro, sem torch.

Mora fora de `rerank.py` de propósito: é a guarda contra o defeito que fez a
métrica do ΦRank medir 35 documentos em vez de 500, e uma guarda só serve se o
teste dela roda na suíte rápida. `rerank.py` importa torch, e o teste morria na
coleta da venv principal.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


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


def sortear_para_parquet(lf: pl.LazyFrame, n: int, destino: Path,
                         semente: int = 17) -> tuple[int, int, int]:
    """Sorteia `n` linhas e ESCREVE direto no parquet. `(linhas, total, n_doc)`.

    ⚠️ Não materializa. `amostrar_do_plano` coleta antes de devolver, e a 6 M de
    pares isso são ~2,4 GB em disco — 4 a 5 GB em memória, com 7,1 GB livres
    nesta máquina. O `sink_parquet` escreve em fluxo.

    A contagem de documentos vem DEPOIS, de uma varredura de uma coluna só do
    arquivo escrito: é o que foi gravado que interessa, não o que o plano
    pretendia gravar.
    """
    total = int(lf.select(pl.len()).collect().item())
    plano = lf
    if n and n < total:
        idx = np.sort(np.random.default_rng(semente).choice(total, size=n,
                                                            replace=False))
        plano = (lf.with_row_index("_i")
                   .filter(pl.col("_i").is_in(idx))
                   .drop("_i"))
    destino.parent.mkdir(parents=True, exist_ok=True)
    plano.sink_parquet(destino, compression="zstd")

    escrito = int(pl.scan_parquet(destino).select(pl.len()).collect().item())
    esperado = min(n, total) if n else total
    if escrito != esperado:
        raise RuntimeError(
            f"pedi {esperado} linhas e o parquet tem {escrito}. O treino rodaria "
            "sobre outro conjunto e a comparação com os runs anteriores deixaria "
            "de isolar a variável.")
    coluna = pl.scan_parquet(destino).select("arxiv_citado").collect(
        engine="streaming")["arxiv_citado"]
    return escrito, total, int(coluna.n_unique())


def amostrar_do_plano(lf: pl.LazyFrame, n: int,
                      semente: int = 17) -> tuple[pl.DataFrame, int, int]:
    """Sorteia `n` linhas SEM materializar o plano. `(amostra, total, n_doc)`.

    ⚠️ O corte tem de acontecer no plano, e não depois dele. `train_embedding.py`
    registra o preço: com 1,0 GB de RAM livre, coletar os 6,5 M pares antes de
    cortar matava o processo antes do primeiro passo, sem deixar traceback.

    E tem de ser sorteio, não `head`. Medido em `pares_treino.parquet`
    (6.564.111 linhas, 667.304 documentos citados distintos):

        n=   20.000   head ->    984 docs   sample ->  17.837 docs   18,1x
        n=  400.000   head -> 17.844 docs   sample -> 191.300 docs   10,7x
        n=1.500.000   head -> 67.232 docs   sample -> 390.966 docs    5,8x

    O run da T1a — 400 mil pares, que produziu o campeão atual do G1 — treinou
    com **17.844** documentos distintos onde o sorteio do MESMO tamanho daria
    **191.300**. Mesma GPU, mesmo tempo.

    Vale suspeitar do platô que interrompeu o run de 1,5 M em 38%: 256 mil pares
    a mais, "metade inéditos", mas tirados de 67 mil documentos — exaurir um
    conjunto pequeno de documentos se parece exatamente com um platô de dados.
    """
    total = int(lf.select(pl.len()).collect().item())
    if not n or n >= total:
        d = lf.collect(engine="streaming")
        return d, total, int(d["arxiv_citado"].n_unique()
                             if "arxiv_citado" in d.columns else d.height)
    # Índices sorteados e ORDENADOS: o filtro varre o arquivo uma vez, em ordem.
    idx = np.sort(np.random.default_rng(semente).choice(total, size=n,
                                                        replace=False))
    d = (lf.with_row_index("_i")
           .filter(pl.col("_i").is_in(idx))
           .drop("_i")
           .collect(engine="streaming"))
    if d.height != n:
        raise RuntimeError(
            f"pedi {n} linhas e o plano devolveu {d.height}. O treino rodaria "
            "sobre outro conjunto e a comparação com o campeão deixaria de "
            "isolar a variável.")
    return d, total, int(d["arxiv_citado"].n_unique()
                         if "arxiv_citado" in d.columns else d.height)


# ─────────────────────────────────────────────────────────────────────────────
# O pool de candidatos da recuperação dentro do lote
#
# ## ⚠️ O irmão do defeito acima, e ele custou o Portão G1
#
# O protocolo do G1 (e o `avaliar` do treino) é: `sim = ancoras @ positivos.T`,
# e a resposta certa da linha *i* é a **coluna** *i*. Isso só é uma tarefa bem
# posta se cada coluna for distinta. Duas coisas quebravam isso:
#
#  1. `val.head(n)`. A ordem de `pares_validacao.parquet` não é neutra — a
#     mediana do comprimento da âncora cai de ~1.180 para ~870 do início ao
#     fim, e o primeiro bloco de 2.000 fica no percentil 94. Pior: em
#     `head(2000)` havia só 1.147 textos positivos distintos, com 62% das
#     linhas num positivo repetido e um deles aparecendo 28 vezes. Textos
#     byte-idênticos dão cosseno idêntico, e o desempate do `argsort` é
#     arbitrário: a diagonal cai num posto qualquer entre as 28.
#
#  2. Âncoras repetidas. Linhas com a mesma consulta compartilham UM ranking,
#     então no máximo uma delas pode ter posto 1.
#
# Teto de um modelo PERFEITO, medido:
#
#     head(2000)          recall@1 0,5235   nDCG@10 0,7562   <- o que o G1 usou
#     sample(2000)        recall@1 0,9364   nDCG@10 0,9761
#     dedup + sample      recall@1 1,0000   nDCG@10 1,0000
#
# O G1 reportou recall@1 0,2620 contra um teto de 0,5235. O empate era JUSTO
# entre modelos — todos sofriam igual —, e é por isso que a tabela parecia
# válida. O que ele destruía era a margem: o G1.2 se decide em +0,003 de
# nDCG@10, e ruído arbitrário em 62% dos itens não deixa +0,003 sobreviver.
# Pelo mesmo motivo, parte dos discordantes do McNemar era moeda.
# ─────────────────────────────────────────────────────────────────────────────

# Semente do pool. O sorteio existe porque a ordem do parquet não é neutra; a
# semente existe porque uma amostra que muda a cada execução tornaria o cache —
# e a comparação entre modelos — sem sentido.
SEMENTE_POOL = 17
COLUNAS = ("ancora", "positivo")


def preparar_pool(val: pl.DataFrame, n: int | None,
                  semente: int = SEMENTE_POOL,
                  exigir_n: bool = True) -> pl.DataFrame:
    """`n` linhas com âncora e alvo ÚNICOS. Ver a docstring do módulo.

    Sorteia com semente (não `head`, que pega o extremo longo de um arquivo
    ordenado), desduplica pelo **texto** — é o texto idêntico que produz cosseno
    idêntico, não o `arxiv_citado` — e corta em `n`. Do `pares_validacao.parquet`
    de 133.540 linhas sobram 20.078 depois da desduplicação, folga de 10× para o
    `n=2.000` do protocolo.
    """
    if n is not None and n < 0:
        raise ValueError(f"n={n} não faz sentido")
    faltando = [c for c in COLUNAS if c not in val.columns]
    if faltando:
        raise ValueError(f"o quadro de validação não tem {faltando}")

    pool = val.sample(fraction=1.0, shuffle=True, seed=semente)
    for coluna in COLUNAS:
        pool = pool.unique(subset=[coluna], keep="first", maintain_order=True)
    if n and pool.height < n:
        # ⚠️ `exigir_n=True` no portão, e não por gosto: comparar dois modelos
        # com n diferente compara dificuldades diferentes, que é a primeira
        # linha da tabela de invalidação em `eval/encoders.py`. O monitor de
        # treino passa False porque abortar um treino de horas por causa do
        # tamanho do pool de acompanhamento seria pior — e lá a comparação é
        # entre checkpoints do MESMO pool.
        if exigir_n:
            raise ValueError(
                f"depois de desduplicar sobraram {pool.height} linhas e o "
                f"protocolo pede {n}. Abaixar o n muda o protocolo, e os "
                "números deixam de ser comparáveis com os anteriores — a "
                "decisão é sua, explicitamente.")
        log.warning("pool com %d linhas depois de desduplicar, abaixo do n=%d "
                    "pedido; medindo com %d", pool.height, n, pool.height)
    pool = pool.head(n) if n else pool

    # ⚠️ A guarda, e não só a intenção: um pool com alvo repetido tem teto abaixo
    # de 1,0 e a tabela sai com a cara certa.
    for coluna in COLUNAS:
        repetidas = pool.height - pool[coluna].n_unique()
        if repetidas:
            raise AssertionError(
                f"{repetidas} linha(s) com {coluna!r} repetido no pool. Textos "
                "idênticos empatam no cosseno, o desempate é arbitrário e o teto "
                "da métrica cai abaixo de 1,0.")
    return pool


def teto_do_pool(d: pl.DataFrame) -> dict[str, float]:
    """O que um modelo PERFEITO conseguiria neste pool.

    Existe para a afirmação "o teto é 1,0" ser medida e não suposta — foi este
    número que revelou que o protocolo do G1 jogava fora 43% do recall@1
    alcançável antes de qualquer modelo ser carregado.

    Num grupo de `g` colunas idênticas a diagonal cai num posto uniforme em
    1..`g`; com `a` linhas na mesma consulta, as `a` dividem um só ranking. O
    número efetivo de concorrentes indistinguíveis é `g * a`.
    """
    cp = Counter(d["positivo"].to_list())
    ca = Counter(d["ancora"].to_list())
    r1 = r10 = nd = mrr = 0.0
    for anc, pos in zip(d["ancora"].to_list(), d["positivo"].to_list(),
                        strict=True):
        eff = cp[pos] * ca[anc]
        r1 += 1 / eff
        r10 += min(eff, 10) / eff
        nd += sum(1 / math.log2(r + 1) for r in range(1, min(eff, 10) + 1)) / eff
        mrr += sum(1 / r for r in range(1, eff + 1)) / eff
    n = d.height
    return {"recall_1": r1 / n, "recall_10": r10 / n, "ndcg_10": nd / n,
            "mrr": mrr / n}
