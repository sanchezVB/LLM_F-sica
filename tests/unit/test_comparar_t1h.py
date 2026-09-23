"""A leitura da regra do T1h e o IC de Bonferroni, sem GPU.

A conta do nDCG vem do G1 e já tem teste. Aqui ficam a REGRA — qual intervalo e qual
custo levam a "subir de volume" — e o parâmetro `alfa` do bootstrap, que tem de alargar
o intervalo sem mudar nada para quem usa o padrão.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

_spec = importlib.util.spec_from_file_location("_t1h", RAIZ / "scripts" / "comparar_t1h.py")
t1h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1h)


def _nd(lo, hi):
    return {"diferenca": (lo + hi) / 2, "ic95": [lo, hi]}


def test_os_tres_desfechos():
    assert t1h.ler_candidata(_nd(0.01, 0.03), 2.0)[0] == "ACIMA"
    assert t1h.ler_candidata(_nd(-0.03, -0.01), 2.0)[0] == "ABAIXO"
    assert t1h.ler_candidata(_nd(-0.01, 0.01), 2.0)[0] == "EMPATE"


def test_so_SOBE_quem_esta_ACIMA_e_dentro_do_teto_de_custo():
    assert t1h.ler_candidata(_nd(0.01, 0.03), 2.0)[2] is True
    assert t1h.ler_candidata(_nd(0.01, 0.03), 2.5)[2] is True
    assert t1h.ler_candidata(_nd(0.01, 0.03), 2.6)[2] is False, "custo acima do teto"
    assert t1h.ler_candidata(_nd(-0.01, 0.03), 1.0)[2] is False, "empate não sobe"


def test_o_alfa_de_Bonferroni_e_o_da_regra():
    assert t1h.ALFA_BONFERRONI == 0.05 / 4
    assert t1h.TETO_DE_CUSTO == 2.5


def test_o_bootstrap_com_alfa_ALARGA_e_o_padrao_NAO_MUDA():
    # O módulo do bootstrap importa torch na carga; na venv rápida o teste pula.
    import pytest

    pytest.importorskip("torch")
    from avaliar_ablacao_secundarias import bootstrap_pareado_itens

    rng = np.random.default_rng(0)
    a = rng.random(500)
    b = a + rng.normal(0.02, 0.1, 500)
    p95 = bootstrap_pareado_itens(a, b)
    p95_explicito = bootstrap_pareado_itens(a, b, alfa=0.05)
    assert p95 == p95_explicito, "o padrão tem de continuar sendo 95%"
    bonf = bootstrap_pareado_itens(a, b, alfa=0.05 / 4)
    assert bonf["diferenca"] == p95["diferenca"]
    assert bonf["ic95"][0] < p95["ic95"][0] and bonf["ic95"][1] > p95["ic95"][1]
