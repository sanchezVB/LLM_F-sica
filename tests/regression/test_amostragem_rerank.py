"""A amostragem que decide se a métrica do ΦRank mede algo.

Este arquivo existe por causa de um erro que custou uma semana de conclusões
erradas. `avaliar` usava `val.head(500)`, e o parquet de pares vem **agrupado por
documento citado** — então as 500 linhas eram 35 papers repetidos ~14 vezes.

Linhas do mesmo documento não são observações independentes. O n efetivo era 35, o
intervalo de 95% do acerto@1 era ±0,159, e o "ganho" de 0,198 para 0,364 caía
inteiro dentro do ruído. Pior: a divisão contaminada e a divisão honesta
reportaram o MESMO número, porque as duas mediam os mesmos 35 papers — o que fez
parecer que o vazamento não importava.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.training.amostragem import amostrar_por_documento  # noqa: E402


def _agrupado(n_docs: int = 50, por_doc: int = 20) -> pl.DataFrame:
    """Imita a forma real do parquet: agrupado, cada documento em linhas seguidas."""
    return pl.DataFrame({
        "arxiv_citado": [f"doc{d:03d}" for d in range(n_docs) for _ in range(por_doc)],
        "ancora": [f"consulta {d}-{r}" for d in range(n_docs) for r in range(por_doc)],
        "positivo": [f"texto do doc {d}" for d in range(n_docs) for _ in range(por_doc)],
    })


def test_head_cobriria_poucos_documentos_e_sample_cobre_muitos():
    """O teste que impede o `head` de voltar.

    Com 50 documentos de 20 linhas, as primeiras 100 linhas são 5 papers. Um sorteio
    das mesmas 100 linhas cobre uma ordem de magnitude mais.
    """
    d = _agrupado()
    quantos_o_head_cobriria = d.head(100)["arxiv_citado"].n_unique()
    _, n_doc = amostrar_por_documento(d, 100, semente=17)

    assert quantos_o_head_cobriria == 5, "a fixture deixou de imitar o agrupamento"
    assert n_doc > 4 * quantos_o_head_cobriria, (
        f"amostragem cobriu {n_doc} documentos, quase o mesmo que o head "
        f"({quantos_o_head_cobriria}) — voltou a ser `head`?")


def test_devolve_documentos_distintos_e_nao_linhas():
    """O segundo valor dimensiona o intervalo de confiança; se contar linhas, mente."""
    d = _agrupado(n_docs=10, por_doc=30)
    amostra, n_doc = amostrar_por_documento(d, 120, semente=17)
    assert len(amostra) == 120
    assert n_doc <= 10, f"n_doc={n_doc} passou dos 10 documentos existentes — contou linhas"
    assert n_doc == amostra["arxiv_citado"].n_unique()


def test_n_maior_que_o_disponivel_devolve_tudo():
    d = _agrupado(n_docs=3, por_doc=4)
    amostra, n_doc = amostrar_por_documento(d, 9_999, semente=17)
    assert len(amostra) == 12 and n_doc == 3


def test_n_zero_devolve_tudo_para_contar_o_universo():
    """`treinar` usa n=0 só para saber quantos documentos existiam ANTES do corte."""
    d = _agrupado(n_docs=7, por_doc=5)
    amostra, n_doc = amostrar_por_documento(d, 0, semente=17)
    assert len(amostra) == 35 and n_doc == 7


def test_mesma_semente_mesma_amostra():
    """Sem isto, retomar um treino avaliaria noutro conjunto e a curva seria ruído."""
    d = _agrupado()
    a, na = amostrar_por_documento(d, 100, semente=17)
    b, nb = amostrar_por_documento(d, 100, semente=17)
    assert a.equals(b) and na == nb


def test_sementes_diferentes_amostras_diferentes():
    d = _agrupado()
    a, _ = amostrar_por_documento(d, 100, semente=17)
    b, _ = amostrar_por_documento(d, 100, semente=18)
    assert not a.equals(b)


def test_sem_a_coluna_de_documento_cai_para_linhas():
    """Não deve explodir num DataFrame sem `arxiv_citado` — só perde o n efetivo."""
    d = pl.DataFrame({"ancora": ["a", "b", "c"], "positivo": ["x", "y", "z"]})
    amostra, n = amostrar_por_documento(d, 2, semente=17)
    assert len(amostra) == 2 and n == 2
