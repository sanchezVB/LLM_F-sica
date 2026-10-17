"""Passo 2 do acervo maior: o id do Semantic Scholar de cada resumo novo do peS2o.

O filtro de Física (`hf_filtrado`) gravou o texto CRU de cada registro, mas não o id.
Para citar um artigo de fora do arXiv, o assistente precisa do id, que dá o link.

O conserto não roda o classificador de novo: baixa o peS2o original, da MESMA revisão que
o filtro usou, arquivo por arquivo, e procura cada registro pelo sha256 do texto. É o
hash que o passo 1 (`pes2o_novos.sha_do_texto`) gravou para cada novo. Para quando todos
foram achados, e retoma de onde parou.

A busca no conjunto de alvos usa os 8 primeiros bytes do hash. Com ~3 milhões de alvos, a
chance de dois textos diferentes colidirem é da ordem de 10⁻⁷. O hash inteiro vai junto
na saída e é conferido por quem usa.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable

import polars as pl

COLUNAS_TEXTO = ("text", "texto", "content", "abstract")
COLUNAS_ID = ("id", "corpusid", "corpus_id", "paper_id")
COLUNAS_DATA = ("created", "year", "added")


def _chave(sha_hex: str) -> int:
    return int(sha_hex[:16], 16)


def alvos(shas: Iterable[str]) -> set[int]:
    return {_chave(s) for s in shas}


def _coluna(df: pl.DataFrame, nomes: tuple[str, ...], obrigatoria: bool) -> str | None:
    achada = next((c for c in nomes if c in df.columns), None)
    if achada is None and obrigatoria:
        raise ValueError(f"nenhuma coluna entre {nomes} (tem {df.columns})")
    return achada


def procurar(df: pl.DataFrame, procurados: set[int]) -> list[dict]:
    """Os registros de `df` cujo texto está entre os procurados, com id, data e origem."""
    c_texto = _coluna(df, COLUNAS_TEXTO, True)
    c_id = _coluna(df, COLUNAS_ID, True)
    c_data = _coluna(df, COLUNAS_DATA, False)
    c_origem = "source" if "source" in df.columns else None
    saida = []
    for i, texto in enumerate(df[c_texto].fill_null("").to_list()):
        if not texto:
            continue
        sha = hashlib.sha256(texto.encode("utf-8")).hexdigest()
        if _chave(sha) not in procurados:
            continue
        saida.append({"sha256": sha, "s2_id": str(df[c_id][i]),
                      "data": None if c_data is None else str(df[c_data][i]),
                      "origem": None if c_origem is None else df[c_origem][i]})
    return saida


def recuperar(procurados: set[int], arquivos: list[str], baixar: Callable[[str], object],
              ler: Callable[[object], Iterable[pl.DataFrame]], feitos: set[str],
              gravar: Callable[[str, list[dict]], None],
              ao_terminar_arquivo: Callable[[str, int, int], None] | None = None) -> dict:
    """Percorre `arquivos` na ordem, pulando os `feitos`, até achar todos os procurados.

    `baixar(nome)` devolve o arquivo local; `ler(local)` itera DataFrames; `gravar(nome,
    achados)` guarda o que achou no arquivo — o chamador marca `nome` como feito depois.
    Os já achados em execuções anteriores saem de `procurados` antes (o chamador sabe
    quais são)."""
    restantes = set(procurados)
    lidos, achados_total = 0, 0
    for nome in arquivos:
        if not restantes:
            break
        if nome in feitos:
            continue
        local = baixar(nome)
        achados: list[dict] = []
        for df in ler(local):
            for r in procurar(df, restantes):
                k = _chave(r["sha256"])
                if k in restantes:
                    restantes.discard(k)
                    achados.append(r)
        gravar(nome, achados)
        lidos += 1
        achados_total += len(achados)
        if ao_terminar_arquivo:
            ao_terminar_arquivo(nome, len(achados), len(restantes))
    return {"arquivos_lidos_agora": lidos, "achados_agora": achados_total,
            "faltam": len(restantes), "parou_cedo": not restantes}
