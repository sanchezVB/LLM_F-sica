"""O MMLU do G2.3 com itens SINTÉTICOS: o formato 5-shot, a letra pelo logprob e os
agregados (sem física, 57, física)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_mmlu as m  # noqa: E402


def _item(i, materia="anatomy", resposta=2):
    return {"id": f"t/{i}", "subject": materia, "question": f"Q{i}?",
            "choices": ["w", "x", "y", "z"], "answer": resposta}


def test_formato_classico_5_shot():
    ex = [_item(k, resposta=k % 4) for k in range(5)]
    p = m.montar_prompt(_item(9), ex)
    assert p.startswith("The following are multiple choice questions (with answers) about "
                        "anatomy.\n\nQ0?\nA. w\nB. x\nC. y\nD. z\nAnswer: A\n\n")
    assert p.endswith("Q9?\nA. w\nB. x\nC. y\nD. z\nAnswer:")
    assert p.count("Answer:") == 6


def test_letra_pelo_logprob_entre_as_quatro():
    top = [{"id": 99, "logprob": -0.1}, {"id": 356, "logprob": -0.5},
           {"id": 362, "logprob": -2.0}]
    letra, lp = m.letra_prevista(top)
    assert letra == "C" and lp["B"] == -math.inf and lp["A"] == -2.0
    assert m.letra_prevista([{"id": 7, "logprob": -0.1}])[0] is None    # nenhuma das quatro


def test_agregados_separam_fisica():
    itens = [_item(0), _item(1, "college_physics"), _item(2, "astronomy", 0), _item(3)]
    prev = {"t/0": "C", "t/1": "C", "t/2": "B", "t/3": None}
    r = m.agregar(itens, prev)
    assert r["sem_fisica"] == {"itens": 2, "acerto": 0.5}
    assert r["fisica"] == {"itens": 2, "acerto": 0.5}
    assert r["mmlu_57"]["acerto"] == 0.5 and r["sem_previsao"] == 1


def test_fisica_tem_640_itens_no_teste():
    teste = m.carregar("test")
    assert len(teste) == 14042
    assert sum(x["subject"] in m.FISICA for x in teste) == 640
