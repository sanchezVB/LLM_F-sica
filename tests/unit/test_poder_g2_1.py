"""O recálculo do poder do G2.1 com notas SINTÉTICAS: as notas por item, o ΦGen simulado,
e o poder que vai de 0 (sem ganho) a 1 (ganho enorme) sem mudar a regra."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

import poder_g2_1 as pg  # noqa: E402
from phifm.eval.g2_corretor import Veredito  # noqa: E402


def test_notas_por_item_e_a_fracao_de_pass_com_k():
    itens = [SimpleNamespace(bench="ugphysics", id="a"), SimpleNamespace(bench="ugphysics", id="b"),
             SimpleNamespace(bench="phybench", id="p")]
    amostras = [{"bench": "ugphysics", "item_id": "a", "j": j} for j in range(3)] + \
               [{"bench": "ugphysics", "item_id": "b", "j": 0}] + \
               [{"bench": "phybench", "item_id": "p", "j": j} for j in range(2)]
    v = {("ugphysics", "a", 0): Veredito.PASS, ("ugphysics", "a", 1): Veredito.FAIL,
         ("ugphysics", "a", 2): Veredito.PASS, ("ugphysics", "b", 0): Veredito.TRUNCADO,
         ("phybench", "p", 0): Veredito.PASS, ("phybench", "p", 1): Veredito.INCONCLUSIVE}
    n = pg.notas_por_item(itens, amostras, v, {"ugphysics": 3, "phybench": 2})
    assert n == {"ugphysics": [2 / 3, 0.0], "phybench": [0.5]}


def test_simular_phigen_sobe_a_nota_e_respeita_o_teto():
    rng = np.random.default_rng(0)
    base = {"x": [0.0] * 2000, "y": [0.98] * 2000}
    s = pg.simular_phigen(rng, base, {"x": 4, "y": 4}, 10.0)
    assert abs(s["x"].mean() - 0.10) < 0.01 and abs(s["y"].mean() - 1.0) < 0.01
    assert set(np.unique(s["x"])) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    # concentrado em 30%: a média é a mesma, e só ~30% dos itens saem do zero
    c = pg.simular_phigen(rng, {"x": [0.0] * 4000}, {"x": 16}, 9.0, fracao=0.3)
    assert abs(c["x"].mean() - 0.09) < 0.01 and 0.25 < (c["x"] > 0).mean() < 0.35


def test_pontos_de_80_no_ganho_nominal_e_no_efetivo():
    por = {"10.0": {"poder": 0.0, "delta_macro_medio": 9.5},
           "12.0": {"poder": 0.6, "delta_macro_medio": 11.0},
           "13.0": {"poder": 1.0, "delta_macro_medio": 12.0}}
    p = pg.pontos_de_80(por)
    assert p == {"ganho_para_poder_80": 12.5, "ganho_efetivo_para_poder_80": 11.5}
    assert pg.pontos_de_80({"1.0": {"poder": 0.1}, "2.0": {"poder": 0.2}}) == \
        {"ganho_para_poder_80": None, "ganho_efetivo_para_poder_80": None}


def test_poder_vai_de_zero_a_um_sem_mudar_a_regra():
    rng = np.random.default_rng(1)
    base = {b: list(rng.random(300) * 0.3) for b in ("ugphysics", "olympiadbench", "phybench", "scibench")}
    grupos = {"ugphysics": list(np.arange(300) // 2), "olympiadbench": list(range(300)),
              "phybench": None, "scibench": None}
    k = {"ugphysics": 3, "olympiadbench": 8, "phybench": 16, "scibench": 8}
    r = pg.poder(base, grupos, k, ganhos=(0.0, 40.0), replicas=6, n_bootstrap=300)
    assert r["por_ganho"]["0.0"]["poder"] == 0.0 and r["por_ganho"]["40.0"]["poder"] == 1.0
    assert r["ganho_para_poder_80"] is not None and 0 < r["ganho_para_poder_80"] < 40
    assert r["por_ganho"]["40.0"]["condicao_3"] == 1.0
    # marcas do placebo em TODOS os itens: a condição 3 fica sem itens... com taxa 1 tudo sai
    r2 = pg.poder(base, grupos, k, ganhos=(40.0,), replicas=2, n_bootstrap=200,
                  taxas_de_marcas=dict.fromkeys(k, 0.5))
    assert r2["por_ganho"]["40.0"]["condicao_1"] == 1.0
