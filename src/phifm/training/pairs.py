"""Pares de citação para o treino contrastivo do ΦEmb (DOC-07 §3.1).

A tese do documento, e o motivo de o grafo do OpenAlex ter sido coletado: **o
par de citação é supervisão gratuita**. Se o artigo A cita o B, o autor de A
precisou de B para escrever — então os dois tratam de assunto próximo, e isso
foi rotulado sem ninguém anotar nada. É a intuição do SPECTER.

## O que sai daqui

``(âncora, positivo)`` em texto: título + resumo de cada lado. Os negativos
**não** são amostrados aqui — vêm de dentro do lote no treino (in-batch
negatives), que é mais eficiente e menos enviesado que sortear negativos fáceis.

## Duas armadilhas que este módulo evita

**1. Vazamento de tema entre treino e validação.** Separar pares ao acaso põe
o mesmo artigo nos dois lados: A→B no treino e A→C na validação. O modelo
decora A e a validação mente. A divisão é feita **por âncora**, não por par.

**2. Grau desbalanceado.** Um artigo de revisão com 300 referências geraria 300
pares e dominaria o gradiente, ensinando o modelo sobre revisões em vez de
Física. `MAX_POR_ANCORA` limita quantos pares cada artigo contribui.

## Por que a colheita é parcial, e por que isso está certo

`referenced_works` traz IDs do OpenAlex de **todo** o mundo acadêmico, mas só
resolvemos os que também estão na nossa coleta — ou seja, os que têm origem no
arXiv e caíram numa partição já processada. Medido: ~8% das referências de um
artigo. Com 22,7 M de arestas isso ainda dá ~1,8 M de pares, ordem de grandeza
folgada para um fine-tune de encoder.

Resolver os outros 92% exigiria coletar títulos e resumos de obras sem origem
no arXiv, o que é outro projeto de coleta e não melhora o sinal: o par
arXiv→arXiv é justamente o de Física.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

# Teto de pares por âncora. Ver §"grau desbalanceado" na docstring.
MAX_POR_ANCORA = 8
# Resumo curto quase sempre é registro incompleto, não artigo curto.
MIN_CARACTERES = 120
FRACAO_VALIDACAO = 0.02


def carregar_grafo(dir_snapshot: Path) -> pl.DataFrame:
    """Arestas `(arxiv_id da âncora, openalex_id citado)`."""
    shards = sorted(dir_snapshot.glob("part-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"nenhum shard em {dir_snapshot}")
    df = pl.read_parquet(shards, columns=["openalex_id", "arxiv_id", "referenced_works"])
    df = df.filter(pl.col("arxiv_id").is_not_null())
    log.info("grafo: %s obras com arxiv_id", f"{df.height:,}")
    return df


def montar_pares(grafo: pl.DataFrame, espinha: pl.DataFrame, semente: int = 17) -> pl.DataFrame:
    """Constrói os pares e anexa os textos das duas pontas."""
    # openalex_id → arxiv_id, para traduzir `referenced_works`.
    mapa = grafo.select("openalex_id", pl.col("arxiv_id").alias("arxiv_citado"))

    arestas = (
        grafo.select("arxiv_id", "referenced_works")
        .explode("referenced_works")
        .rename({"referenced_works": "openalex_id"})
        .filter(pl.col("openalex_id").is_not_null())
        .join(mapa, on="openalex_id", how="inner")      # só as que sabemos resolver
        .filter(pl.col("arxiv_id") != pl.col("arxiv_citado"))   # autocitação de versão
        .select("arxiv_id", "arxiv_citado")
        .unique()
    )
    log.info("arestas resolvidas: %s", f"{arestas.height:,}")

    # Teto por âncora, com ordem embaralhada para não pegar sempre as primeiras
    # referências — que num paper tendem a ser as introdutórias e genéricas.
    arestas = (
        arestas.sample(fraction=1.0, shuffle=True, seed=semente)
        .with_columns(pl.int_range(pl.len()).over("arxiv_id").alias("_i"))
        .filter(pl.col("_i") < MAX_POR_ANCORA)
        .drop("_i")
    )
    log.info("após teto de %d por âncora: %s", MAX_POR_ANCORA, f"{arestas.height:,}")

    textos = (
        espinha.select("arxiv_id", "title", "abstract")
        .with_columns(
            (pl.col("title").fill_null("") + ". " + pl.col("abstract").fill_null(""))
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("texto")
        )
        .filter(pl.col("texto").str.len_chars() >= MIN_CARACTERES)
        .select("arxiv_id", "texto")
    )

    pares = (
        arestas
        .join(textos, on="arxiv_id", how="inner")
        .rename({"texto": "ancora"})
        .join(textos.rename({"arxiv_id": "arxiv_citado", "texto": "positivo"}),
              on="arxiv_citado", how="inner")
        .select("arxiv_id", "arxiv_citado", "ancora", "positivo")
    )
    log.info("pares com texto nas duas pontas: %s", f"{pares.height:,}")
    return pares


def dividir(pares: pl.DataFrame, semente: int = 17) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Divide POR ÂNCORA, não por par.

    Dividir por par põe o mesmo artigo nos dois lados — A→B no treino, A→C na
    validação — e o modelo decora a âncora em vez de aprender a relação. A
    validação então mede memorização e reporta um número bom e falso.
    """
    ancoras = pares.select("arxiv_id").unique().sample(fraction=1.0, shuffle=True, seed=semente)
    n_val = max(int(ancoras.height * FRACAO_VALIDACAO), 1)
    val_ids = ancoras.head(n_val)

    val = pares.join(val_ids, on="arxiv_id", how="semi")
    tr = pares.join(val_ids, on="arxiv_id", how="anti")

    # Garantia explícita: nenhuma âncora aparece nos dois lados.
    vazamento = tr.join(val.select("arxiv_id").unique(), on="arxiv_id", how="semi").height
    if vazamento:
        raise RuntimeError(f"{vazamento} pares de treino compartilham âncora com a validação")

    log.info("treino: %s pares | validação: %s pares (%s âncoras)",
             f"{tr.height:,}", f"{val.height:,}", f"{val_ids.height:,}")
    return tr, val


def construir(dir_snapshot: Path, espinha: Path, saida: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    grafo = carregar_grafo(dir_snapshot)
    esp = pl.read_parquet(espinha, columns=["arxiv_id", "title", "abstract"])
    tr, val = dividir(montar_pares(grafo, esp))

    saida.mkdir(parents=True, exist_ok=True)
    for nome, df in (("treino", tr), ("validacao", val)):
        destino = saida / f"pares_{nome}.parquet"
        tmp = destino.with_suffix(".parquet.tmp")
        df.write_parquet(tmp, compression="zstd")
        tmp.replace(destino)
        log.info("→ %s (%s pares, %.1f MB)", destino.name, f"{df.height:,}",
                 destino.stat().st_size / 1e6)
    return tr, val
