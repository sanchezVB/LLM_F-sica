"""A leitura da regra do T1g, sem GPU: qual intervalo leva a qual desfecho.

A conta vem do G1 e já tem teste. O que este arquivo prende é o SINAL — "gte −
sistema" acima de zero é o GTE à frente — e que o custo de serviço vai junto no texto
da vitória, porque uma vitória lida sem ele vira troca de encoder por engano.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

_spec = importlib.util.spec_from_file_location("_t1g", RAIZ / "scripts" / "comparar_t1g.py")
t1g = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t1g)


def _nd(lo: float, hi: float) -> dict:
    return {"diferenca": (lo + hi) / 2, "ic95": [lo, hi]}


def test_os_tres_desfechos_e_o_sinal():
    assert t1g.ler_t1g(_nd(0.01, 0.03))[0] == "GTE À FRENTE"
    assert t1g.ler_t1g(_nd(-0.03, -0.01))[0] == "SISTEMA À FRENTE"
    assert t1g.ler_t1g(_nd(-0.01, 0.01))[0] == "EMPATE"


def test_o_zero_na_borda_NAO_decide():
    assert t1g.ler_t1g(_nd(0.0, 0.03))[0] == "EMPATE"


def test_a_vitoria_do_GTE_carrega_o_CUSTO_no_texto():
    _, texto = t1g.ler_t1g(_nd(0.01, 0.03))
    assert "4,4×" in texto and "NÃO é troca automática" in texto


def test_os_numeros_registrados_sao_os_do_ESTADO():
    assert t1g.SISTEMA_REGISTRADO == 0.6223
    assert t1g.MINILM_15M_REGISTRADO == 0.5780
    assert t1g.CUSTO_DE_SERVICO["razao"] == 4.4
