"""As duas leituras da secundária do caminho B, sem GPU.

A conta do nDCG e do bootstrap vem do G1 e já tem teste; o que este arquivo prende é
a REGRA — qual intervalo leva a qual desfecho, e com que sinal. Uma leitura invertida
trocaria "o CPT paga o próprio custo" por "não paga" sem nada acusar.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

_spec = importlib.util.spec_from_file_location(
    "_cmp", RAIZ / "scripts" / "comparar_t2eq_cpt_emb.py")
cmp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp)


def _nd(lo: float, hi: float) -> dict:
    return {"diferenca": (lo + hi) / 2, "ic95": [lo, hi]}


def test_entre_bracos_tem_os_tres_desfechos():
    assert cmp.ler_entre_bracos(_nd(0.01, 0.05))[0] == "TRATADO À FRENTE"
    assert cmp.ler_entre_bracos(_nd(-0.05, -0.01))[0] == "CONTROLE À FRENTE"
    assert cmp.ler_entre_bracos(_nd(-0.01, 0.02))[0] == "EMPATE"


def test_contra_a_barra_tem_os_tres_desfechos():
    assert cmp.ler_contra_a_barra("tratado", _nd(0.01, 0.05))[0] == "CPT ACIMA DA BARRA"
    assert cmp.ler_contra_a_barra("tratado", _nd(-0.05, -0.01))[0] == "CPT ABAIXO DA BARRA"
    assert cmp.ler_contra_a_barra("tratado", _nd(-0.01, 0.02))[0] == "EMPATE COM A BARRA"


def test_o_ZERO_na_borda_NAO_decide():
    """`lo == 0` não é "acima": a regra pede que o IC EXCLUA zero."""
    assert cmp.ler_entre_bracos(_nd(0.0, 0.05))[0] == "EMPATE"
    assert cmp.ler_contra_a_barra("x", _nd(-0.05, 0.0))[0] == "EMPATE COM A BARRA"


def test_a_leitura_NOMEIA_o_braco_medido():
    _, texto = cmp.ler_contra_a_barra("controle", _nd(0.01, 0.05))
    assert "controle" in texto


def test_a_barra_registrada_e_a_do_passo_1():
    assert cmp.BARRA_DO_PASSO_1 == 0.5270


def test_a_leitura_da_BASE_tem_os_tres_desfechos():
    """`nd` é gte − modernbert: acima de zero, o CPT partiu da base errada."""
    assert cmp.ler_a_base(_nd(0.01, 0.05))[0] == "GTE À FRENTE"
    assert cmp.ler_a_base(_nd(-0.05, -0.01))[0] == "MODERNBERT À FRENTE"
    assert cmp.ler_a_base(_nd(-0.01, 0.02))[0] == "EMPATE"
    assert cmp.ler_a_base(_nd(0.0, 0.05))[0] == "EMPATE"
