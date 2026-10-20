"""A correção num processo que pode ser morto: corrige, mata no teto e sobe outro."""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval.g2_correcao import CorretorIsolado  # noqa: E402
from phifm.eval.g2_corretor import Veredito as V  # noqa: E402
from phifm.eval.g2_itens import Item  # noqa: E402


def test_corrige_mata_no_teto_e_sobe_outro():
    item = Item("ugphysics", "X/1", "Q", ("3.5",), ("NV",), (None,))
    with CorretorIsolado(teto_s=120) as cor:
        assert cor.corrigir(item, r"The final answer is $\boxed{3.5}$.") is V.PASS
        assert cor.corrigir(item, r"$\boxed{4}$") is V.FAIL
        cor.teto_s = 0.0001          # força o estouro: o processo é morto
        assert cor.corrigir(item, r"$\boxed{3.5}$") is V.INCONCLUSIVE
        assert cor.mortes == 1
        cor.teto_s = 120             # e um novo trabalhador sobe sozinho
        assert cor.corrigir(item, r"$\boxed{3.5}$") is V.PASS
