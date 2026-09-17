"""O protocolo do PB-Formula: estrato primário, escore por documento e métricas.

As três decisões que estas funções carregam foram tomadas ANTES de qualquer número
(ver a docstring de `scripts/avaliar_pb_formula.py`), e cada uma tem um modo de falha
que nenhum erro denunciaria:

- medir o benchmark inteiro premiaria um casador de string, porque 90% dos itens se
  resolvem assim;
- pontuar o documento pela MÉDIA das equações faria um artigo de 200 equações perder
  para um de 3 pela quantidade;
- contar acerto por equação, e não por item, deixaria um item com 3 alvos valer o
  triplo de um com 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval.benchmarks.pb_formula_protocolo import (  # noqa: E402
    escore_por_documento,
    itens_primarios,
    metricas_por_item,
)


def test_o_primario_exige_notacional_E_sem_documento_quase_igual():
    grafia = np.array(["notacional", "notacional", "identica", "superficial"])
    comuns = np.array([1, 9, 1, 2])
    m = itens_primarios(grafia, comuns)
    assert list(m) == [True, False, False, False]


def test_o_escore_do_documento_e_o_MAXIMO_das_equacoes_dele():
    # 4 equações: as duas primeiras do documento 0, as outras dos documentos 1 e 2.
    escores = np.array([[0.1, 0.9, 0.5, 0.2]], dtype=np.float32)
    doc = np.array([0, 0, 1, 2])
    d = escore_por_documento(escores, doc, 3)
    assert d.tolist() == [[pytest.approx(0.9), pytest.approx(0.5), pytest.approx(0.2)]]


def test_documento_sem_equacao_no_indice_fica_em_menos_infinito():
    d = escore_por_documento(np.array([[0.4]]), np.array([0]), 2)
    assert d[0][1] == -np.inf, "um documento sem equação não pode empatar em 0,0"


def test_recall_por_ITEM_acerta_com_qualquer_alvo():
    # item 0: alvos {3, 7}; a ordem põe 7 em terceiro -> acerta em k=10, não em k=1.
    ordem = np.array([[5, 2, 7, 3, 1]])
    m = metricas_por_item(ordem, [{3, 7}], ks=(1, 10))
    assert m["recall_1"].tolist() == [0.0]
    assert m["recall_10"].tolist() == [1.0]
    assert m["posto"].tolist() == [3.0]
    assert m["rr"].tolist() == [pytest.approx(1 / 3)]


def test_item_sem_alvo_no_pool_LEVANTA():
    """É o teto do pool < 1,0 — a armadilha que o G1 pagou."""
    with pytest.raises(ValueError, match="sem alvo"):
        metricas_por_item(np.array([[0, 1]]), [set()])
