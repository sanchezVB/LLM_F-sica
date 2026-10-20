"""A montagem dos prompts do G2 (DOC-12 §3.7.4) com itens SINTÉTICOS: o formato, a linha
de resposta pelo tipo efetivo e o filtro de irmãos dos exemplos."""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_prompt as gp  # noqa: E402
from phifm.eval.g2_itens import Item  # noqa: E402


def _ug(tipo, gab, unidade=None, **extra):
    return Item("ugphysics", "M/1", "A ball falls.", (gab,), (tipo,), (None,), unidade,
                "It falls fast.", {"materia": "M", "nivel": "Math Derivation", **extra})


def test_formato_do_prompt():
    ex = _ug("NV", "9.8", "m/s^2")
    item = _ug("EX", r"\frac{1}{2} g t^2")
    p = gp.montar_prompt(item, [ex])
    assert p.startswith(gp.CABECALHO)
    assert ("Problem: A ball falls.\nThe answer is a number in m/s^2; put only the number in "
            "\\boxed{}.\nSolution: It falls fast. The final answer is $\\boxed{9.8}$.\n\n") in p
    assert p.endswith("Problem: A ball falls.\nThe answer is an expression.\nSolution:")


def test_linha_de_resposta_segue_o_tipo_efetivo():
    # NV com gabarito simbólico é corrigido como EX, e a linha diz "expressão"
    assert gp.linha_de_resposta(_ug("NV", r"\frac{B \pi a^{2}}{R}")) == "The answer is an expression."
    assert gp.linha_de_resposta(_ug("NV", "3.2")).startswith("The answer is a number;")
    assert gp.linha_de_resposta(_ug("EQ", r"\tau = 4.5 \times 10^{9}")).startswith(
        "The answer is a number;")
    assert gp.linha_de_resposta(_ug("MC", "C")) == "Answer with the option letter(s)."


def test_linhas_de_olympiad_phybench_e_scibench():
    oly = Item("olympiadbench", "1", "Q", ("3", "4"), ("Numerical", "Numerical"), (0.1, 0.1),
               "m")
    linha = gp.linha_de_resposta(oly)
    assert "a numerical value" in linha and "2 answers" in linha and "unit outside" in linha
    phy = Item("phybench", "1", "Q", ("x",), ("EX",))
    assert "single symbolic expression" in gp.linha_de_resposta(phy)
    sci = Item("scibench", "class/1", "Q", ("8.8e10",), ("NV",), (None,), "m", None,
               {"livro": "class", "resposta_exibida": "8.8 \\times 10^{10}"})
    assert gp.linha_de_resposta(sci) == "The unit of the answer is m. Give only the number in \\boxed{}."
    assert "\\boxed{8.8 \\times 10^{10}}" in gp.formatar_exemplo(sci, gp.linha_de_resposta(sci))


def test_irmao_de_teste_por_13grama():
    teste = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    from phifm.eval import g2
    t13 = g2.ngramas(teste, 13)
    assert gp.irmao_de_teste("x " + " ".join(teste.split()[:13]) + " y", t13)
    assert not gp.irmao_de_teste(" ".join(teste.split()[:12]), t13)
