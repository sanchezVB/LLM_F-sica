"""A correção num processo que pode ser morto: corrige, mata no teto, refaz uma vez num
processo novo e sobe outro; e o autoteste pelo mesmo caminho."""
from __future__ import annotations

import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_corretor as c  # noqa: E402
from phifm.eval.g2_correcao import CorretorIsolado  # noqa: E402
from phifm.eval.g2_corretor import Veredito as V  # noqa: E402
from phifm.eval.g2_itens import Item  # noqa: E402


def test_corrige_mata_no_teto_refaz_e_sobe_outro():
    item = Item("ugphysics", "X/1", "Q", ("3.5",), ("NV",), (None,))
    with CorretorIsolado(teto_s=120) as cor:
        assert cor.corrigir(item, r"The final answer is $\boxed{3.5}$.") is V.PASS
        assert cor.corrigir(item, r"$\boxed{4}$") is V.FAIL
        cor.teto_s = 0.0001          # força o estouro: morto, refeito num novo, morto de novo
        assert cor.corrigir(item, r"$\boxed{3.5}$") is V.INCONCLUSIVE
        assert cor.mortes == 2
        cor.teto_s = 120             # e um novo trabalhador sobe sozinho
        assert cor.corrigir(item, r"$\boxed{3.5}$") is V.PASS
        r = cor.autoteste(item)
        assert r["ok"] and r["id"] == "X/1" and r["perturbacoes"] == 2


def test_thread_que_estoura_o_teto_fica_viva_e_e_contada():
    antes = c.threads_vazadas()
    assert c._com_teto(lambda: time.sleep(0.6) or V.PASS, 0.05) == (V.INCONCLUSIVE, True)
    assert c.threads_vazadas() == antes + 1
    time.sleep(1.0)
    assert c.threads_vazadas() == antes


def test_isolar_a_grandeza_sem_solve():
    E, V_, B, u = (c.sp.Symbol(x) for x in ("E", "V", "B", "u"))
    alvo = c.sp.pi * B * u ** 2 / V_ ** 2
    assert c._equivalentes(c._isolar(E * V_ ** 2 - c.sp.pi * B * u ** 2, E), alvo) is V.PASS
    assert c._equivalentes(c._isolar(1 / E - V_ ** 2 / (c.sp.pi * B * u ** 2), E), alvo) is V.PASS
    assert c._equivalentes(c._isolar(c.sp.sqrt(E) - c.sp.sqrt(alvo), E), alvo) is V.PASS
    assert c._isolar(E ** 3 - E - 1, E) is None           # não linear: não isola
    assert c.comparar_definicao(r"E V^{2} = \pi B u^{2}", r"E = \frac{\pi B u^{2}}{V^{2}}") is V.PASS
    assert c.comparar_definicao(r"E V^{2} = 2\pi B u^{2}", r"E = \frac{\pi B u^{2}}{V^{2}}") is V.FAIL
