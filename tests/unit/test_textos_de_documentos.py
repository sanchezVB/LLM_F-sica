"""O texto de cada documento é UM só no projeto: o do treino e o do índice de busca.

`phifm.training.pairs.textos_de_documentos` monta `título. resumo`, e os pares de treino e
o índice passam os dois por ela. Um índice com outro texto não daria erro — daria
vetores fora da distribuição do treino e uma busca pior sem causa visível.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.training.pairs import MIN_CARACTERES, textos_de_documentos  # noqa: E402

LONGO = "x " * 80


def _spine(linhas):
    return pl.LazyFrame(linhas, schema={"arxiv_id": pl.String, "title": pl.String,
                                        "abstract": pl.String}, orient="row")


def test_o_formato_e_titulo_ponto_resumo_com_espacos_normalizados():
    t = textos_de_documentos(_spine([("1", "  Um   título\n", f"Resumo\t{LONGO}")])).collect()
    assert t["texto"][0].startswith("Um título . Resumo x")
    assert "  " not in t["texto"][0] and "\n" not in t["texto"][0]


def test_titulo_ou_resumo_nulos_nao_derrubam():
    t = textos_de_documentos(_spine([("1", None, LONGO), ("2", LONGO, None)])).collect()
    assert t.height == 2
    assert t["texto"][0].startswith(". x") and t["texto"][1].endswith(".")


def test_curtos_ficam_de_fora_pela_MESMA_regra_do_treino():
    t = textos_de_documentos(_spine([("curto", "a", "b"), ("ok", "T", LONGO)])).collect()
    assert t["arxiv_id"].to_list() == ["ok"]
    assert MIN_CARACTERES == 120


def test_bate_com_os_PARES_REAIS_do_treino():
    """A prova forte: o texto que a função monta é o `positivo` dos pares de validação.

    Medido em 2026-09-24: 88.807 de 88.807 idênticos. Aqui, uma amostra.
    """
    pares = RAIZ / "data/processed/pares/pares_validacao.parquet"
    spine = RAIZ / "data/processed/spine.parquet"
    if not (pares.exists() and spine.exists()):
        pytest.skip("os dados do projeto não estão nesta máquina")
    v = pl.read_parquet(pares).unique("arxiv_citado").head(500)
    ids = v["arxiv_citado"].to_list()
    t = textos_de_documentos(
        pl.scan_parquet(spine).filter(pl.col("arxiv_id").is_in(ids))).collect()
    j = v.join(t.rename({"arxiv_id": "arxiv_citado"}), on="arxiv_citado", how="left")
    assert j["texto"].null_count() == 0
    assert (j["texto"] == j["positivo"]).all()
