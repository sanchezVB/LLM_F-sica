"""Integridade dos agregados G2 com itens sintéticos, sem ler benchmarks."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from scripts import medir_g2_robusto as medida

from phifm.eval.g2_corretor import Veredito
from phifm.eval.g2_itens import Item


def _item(nome):
    return Item("ugphysics", nome, "sintético", ("1",), ("NV",))


def _amostra(nome, j=0, parada="eos", texto="1"):
    return {"bench": "ugphysics", "item_id": nome, "j": j, "parada": parada,
            "texto": texto, "fichas": 0 if parada == "erro" else 1}


def test_retry_nao_transforma_cinquenta_por_cento_em_sessenta_e_seis():
    amostras = [_amostra("x", parada="erro", texto=""), _amostra("x"), _amostra("y")]
    vereditos = {("ugphysics", "x", 0): Veredito.PASS,
                 ("ugphysics", "y", 0): Veredito.FAIL}
    agregado = medida._agregar([_item("x"), _item("y")], amostras, vereditos,
                               {"ugphysics": 1})["ugphysics"]
    assert agregado["itens"] == 2 and agregado["amostras"] == 2
    assert agregado["acerto"] == 0.5


def test_erro_operacional_persistente_conta_uma_vez_como_zero():
    amostras = [_amostra("x", parada="erro", texto="")] * 3 + [_amostra("y")]
    vereditos = {("ugphysics", "x", 0): Veredito.ERROR,
                 ("ugphysics", "y", 0): Veredito.PASS}
    agregado = medida._agregar([_item("x"), _item("y")], amostras, vereditos,
                               {"ugphysics": 1})["ugphysics"]
    assert agregado["amostras"] == 2 and agregado["acerto"] == 0.5
    assert agregado["fracoes"]["error"] == 0.5


@pytest.mark.parametrize("amostras", [[], [_amostra("x")],
    [_amostra("x"), _amostra("y"), _amostra("z")],
    [_amostra("x"), _amostra("y", j=1)]])
def test_nao_publica_media_com_amostras_ausentes_ou_extras(amostras):
    vereditos = {(a["bench"], a["item_id"], a["j"]): Veredito.PASS for a in amostras}
    with pytest.raises(ValueError, match="incomplet|inesperad"):
        medida._agregar([_item("x"), _item("y")], amostras, vereditos, {"ugphysics": 1})


def test_nao_escolhe_entre_duas_respostas_validas_distintas_da_mesma_amostra():
    linhas = [_amostra("x", texto="1"), _amostra("x", texto="2")]
    with pytest.raises(ValueError, match="conflit"):
        medida._agregar([_item("x")], linhas,
                        {("ugphysics", "x", 0): Veredito.PASS}, {"ugphysics": 1})


def test_duplicacao_identica_nao_da_peso_extra_ao_item():
    linhas = [_amostra("x"), _amostra("x"), _amostra("y")]
    v = {("ugphysics", "x", 0): Veredito.PASS, ("ugphysics", "y", 0): Veredito.FAIL}
    assert medida._agregar([_item("x"), _item("y")], linhas, v,
                           {"ugphysics": 1})["ugphysics"]["acerto"] == 0.5


def test_exige_todas_as_k_amostras_de_cada_item():
    linhas = [_amostra("x", j=0), _amostra("x", j=1), _amostra("y", j=0)]
    v = {(a["bench"], a["item_id"], a["j"]): Veredito.PASS for a in linhas}
    with pytest.raises(ValueError, match="incomplet"):
        medida._agregar([_item("x"), _item("y")], linhas, v, {"ugphysics": 2})


def test_conjunto_vazio_e_recusado():
    with pytest.raises(ValueError, match="vazi"):
        medida._agregar([], [], {}, {"ugphysics": 1})
