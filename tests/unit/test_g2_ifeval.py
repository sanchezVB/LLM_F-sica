"""O IFEval do G2.3: a consulta no envelope congelado e a correção pelo código oficial, com
respostas SINTÉTICAS (as 20 instruções de dev são escritas fora do IFEval)."""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_ifeval as f  # noqa: E402

DEV20 = RAIZ / "configs/eval/g2_ifeval_dev20.jsonl"


def test_consulta_no_envelope_e_crua():
    c = f.consulta("Say hi.")
    assert c.startswith("The following are queries from users")
    assert c.endswith("# Query\nSay hi.\n\n# Answer\n")
    assert f.consulta("Say hi.", envelope=False) == "Say hi."


def test_correcao_oficial_strict_e_loose():
    itens = {x["key"]: x for x in f.carregar(DEV20)}
    assert len(itens) == 20
    sem_virgula = itens[9003]
    assert f.corrigir(sem_virgula, "Yeast makes gas and the dough rises.")["strict"]
    assert not f.corrigir(sem_virgula, "Yeast, warmth and time make it rise.")["strict"]
    caixa_baixa = itens[9002]           # o oficial também exige que o langdetect veja inglês
    texto = ("hello there, i hope you are doing well this week. i would like to borrow your "
             "ladder on saturday morning to clean the gutters of my house.")
    assert f.corrigir(caixa_baixa, texto)["strict"]
    assert not f.corrigir(caixa_baixa, texto.capitalize())["strict"]
    fim = itens[9005]
    r = f.corrigir(fim, "A great bottle.\nIs there anything else I can help with?")
    assert r["strict"] and r["loose"]


def test_kwargs_none_nao_vira_sorteio():
    # com None, number_words sortearia o limite; tirado, vale o default da instrução? não:
    # o item de dev declara o limite, e a correção é determinística
    item = {"key": 1, "prompt": "p", "instruction_id_list": ["length_constraints:number_words"],
            "kwargs": [{"num_words": 5, "relation": "less than", "extra": None}]}
    assert f.corrigir(item, "one two three")["strict"]
    assert not f.corrigir(item, "one two three four five six seven")["strict"]


def test_agregados():
    notas = [{"strict": True, "loose": True, "strict_lista": [True, False], "loose_lista": [True, True]},
             {"strict": False, "loose": True, "strict_lista": [False], "loose_lista": [True]}]
    r = f.agregar(notas)
    assert r["prompt_strict"] == 0.5 and r["prompt_loose"] == 1.0
    assert r["instrucao_strict"] == round(1 / 3, 4)
