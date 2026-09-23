"""A leitura da regra do T1i, sem GPU: cada pequena a 1 M contra o sistema.

Prende o SINAL — `candidata − sistema` acima de zero é a candidata à frente —, os três
desfechos, e que a vitória carrega o custo no texto, porque uma vitória lida sem ele
vira troca de encoder por engano.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

_spec = importlib.util.spec_from_file_location("_t1i", RAIZ / "scripts" / "comparar_t1i.py")
t1i = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1i)


def _nd(lo, hi):
    return {"diferenca": (lo + hi) / 2, "ic95": [lo, hi]}


def test_os_tres_desfechos_e_o_sinal():
    assert t1i.ler_candidata(_nd(0.005, 0.02), 1.75)[0] == "ACIMA"
    assert t1i.ler_candidata(_nd(-0.02, -0.005), 1.75)[0] == "ABAIXO"
    assert t1i.ler_candidata(_nd(-0.01, 0.01), 1.75)[0] == "EMPATE"
    assert t1i.ler_candidata(_nd(0.0, 0.02), 1.75)[0] == "EMPATE"


def test_a_vitoria_carrega_o_CUSTO_e_a_decisao_do_dono():
    _, texto = t1i.ler_candidata(_nd(0.005, 0.02), 1.75)
    assert "1.75×" in texto and "dono" in texto


def test_o_IC_e_o_de_Bonferroni_sobre_duas_e_os_registros_batem():
    assert t1i.ALFA_BONFERRONI == 0.05 / 2
    assert t1i.SISTEMA_REGISTRADO == 0.6223 and t1i.GTE_BASE_1M_REGISTRADO == 0.6211
