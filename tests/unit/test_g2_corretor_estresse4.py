"""Regressões da QUARTA rodada do estresse adversarial do corretor do G2 (2026-10-23). A
maior parte vinha das regras de extração e de texto das rodadas 2 e 3; a extração passou
a seguir os juízes oficiais (vale a última caixa; a frase só sem caixa). Formas sintéticas,
sem conteúdo de benchmark."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_corretor as c  # noqa: E402
from phifm.eval.g2_corretor import Veredito as V  # noqa: E402

PRE = ("We start from the relevant conservation law and write the governing relation. "
       "Substituting the given quantities, we obtain the result.\n")

# ── extração: a caixa vence; a frase só sem caixa ────────────────────────────


@pytest.mark.parametrize("texto, esperado", [
    # a frase hesitante ou condicionada não vence a caixa
    (r"So $\boxed{180}$ J. The final answer is $90$ J (or $180$ J with drag).", "180"),
    (PRE + "So $E = \\boxed{13.06}$ eV.\n\nThe final answer is $12.75$ eV if we neglect the recoil.",
     "13.06"),
    (PRE + "So $E = \\boxed{13.06}$ eV.\n\nThe final answer is $12.75$ eV.\nBut that ignores the "
     "recoil, so it is $13.06$ eV.", "13.06"),
    # caixa vazia, placeholder e envoltório
    (PRE + "The final answer is $\\boxed{130}$ N.\nThe answer is a numerical value. Give the unit "
     "outside the \\boxed{}.", "130"),
    (PRE + "The final answer is $\\boxed{D}$ .\n\nEach solution ends with: The final answer is "
     "$\\boxed{...}$.", "D"),
    (PRE + "The final answer is $\\boxed{{D}}$ .", "D"),
    (PRE + "The final answer is $\\boxed{\\textbf{M_{2} \\ddot{x}_{2} = M_{2} g - T}}$ .",
     r"M_{2} \ddot{x}_{2} = M_{2} g - T"),
    (PRE + "The final answer is $\\boxed{\\mathbf{\\frac{B^{2}}{2 M c^{2}}}}$ .",
     r"\frac{B^{2}}{2 M c^{2}}"),
    # problema novo inventado com outro cabeçalho, seguido de "Solution:"
    (PRE + "The final answer is $\\boxed{13.06}$ eV.\n\nQuestion: A ball of mass $2$ kg is dropped."
     "\nSolution: Using $v = gt$. The final answer is $\\boxed{12.75}$.", "13.06"),
    (PRE + "The final answer is $\\boxed{D}$ .\n\nExample 3: A ball is dropped.\nSolution: The "
     "final answer is $\\boxed{B}$.", "D"),
    (PRE + "The final answer is $\\boxed{202}$ km.\n\nExercise 1.2 A ball is dropped.\nSolution: "
     "The final answer is $\\boxed{263}$.", "202"),
    # eco da linha de resposta do OlympiadBench
    (PRE + "The final answer is $\\boxed{130}$ N.\n\nA charge moves in a field.\nThe answer is a "
     "numerical value. Give the unit outside the \\boxed{}.\nSolution: The final answer is "
     "$\\boxed{170}$ N.", "130"),
    # marcador no meio do raciocínio não apaga a resposta que vem depois
    (" Before computing, note that the final answer is independent of the mass.\nProblem 1. We "
     "first find the reduced mass.\nProblem 2. Then the energy.\nSo $E = \\boxed{13.06}$ eV.",
     "13.06"),
    (" Part one gives $\\mu = \\boxed{0.5}$.\nProblem 2. Now the energy follows.\nThe final "
     "answer is $\\boxed{7}$ .", "7"),
    (" First, $v_0 = \\boxed{3}$.\nThe answer is an expression.\nTherefore $E = \\boxed{13.06}$ eV.",
     "13.06"),
])
def test_extracao_caixa(texto, esperado):
    assert c.extrair_resposta(texto) == esperado


@pytest.mark.parametrize("texto, esperado", [
    # sem caixa: frase legítima
    (PRE + "Note that the final answer is $13.06$ eV.", "13.06"),
    (PRE + "Since the result does not depend on the mass, the final answer is $130$ N.", "130"),
    (PRE + "Without loss of generality, we take the field along $z$, and the final answer is "
     "$13.06$ eV.", "13.06"),
    (PRE + "The final answer is $13.06$ eV.\nCorrection factors are negligible here.", "13.06"),
    (PRE + "The final answer is $130$ N.\nActually, this agrees with the tabulated value.", "130"),
    (PRE + "The final answer is E = 13.06 eV.", "E = 13.06 eV"),
    (PRE + "The final answer is B^2/(2 M c^2).", "B^2/(2 M c^2)"),
    (PRE + "The final answer is 13.06 eV, i.e. about $2.1 \\times 10^{-18}$ J.", "13.06 eV"),
    (PRE + "The final answer is ≈ $13.06$ eV.", "13.06"),
    (PRE + "The final answer is E = $130$ N.", "130"),
    (PRE + "The final answer is $0.3\\,\\mu$m.", r"0.3\,\mu m"),
    (PRE + "The final answer is $5$ m/s (or 18 km/h).", "5"),
    # sem caixa: frase que não responde
    (PRE + "The final answer is $13.06$ eV, or $12.75$ eV if the recoil is included.", None),
    (PRE + "The final answer is $130$ N (or $170$ N with the correction).", None),
    (PRE + "The final answer is $60$ GeV (or $120$ GeV if drag is included).", None),
    ("Had we neglected recoil, the final answer is $13.06$ eV.", None),
    (PRE + "The final answer is $13.06$ eV.\nBut that ignores the recoil.", None),
    (PRE + "Before computing, note that the final answer is independent of the mass.", None),
])
def test_extracao_frase_sem_caixa(texto, esperado):
    assert c.extrair_resposta(texto) == esperado


@pytest.mark.parametrize("texto, n, esperado", [
    (PRE + "The final answers are (1) $a = 5$ m/s$^2$; (2) $t = 2$ s.", 2, 2),
    (PRE + "The final answers are $5$ m/s$^2$ ($a$) and $2$ s ($t$).", 2, 2),
    (PRE + "The final answers are $86000$ $\\mu$m and $6727$ $\\mu$m.", 2, 2),
    (PRE + "The final answers are $5$ $\\rm m/s^2$ and $2$ $\\rm s$.", 2, 2),
])
def test_varias_respostas_na_frase(texto, n, esperado):
    lida = c.extrair_respostas(texto, n)
    assert lida is not None and len(c.respostas_da_caixa(lida)) == esperado
    assert c.comparar_varias(lida, ["5", "2"] if "5" in lida else ["86000", "6727"],
                             [lambda r, g: c.comparar_numero(r, g, rel=1e-2)] * 2) is V.PASS


def test_corte_e_linear_com_caixas_abertas():
    texto = " We compute.\n" + "".join(f"\\boxed{{a_{{{k}}}\nProblem {k}: x\n" for k in range(400))
    texto += "The final answer is $\\boxed{13.06}$ eV."
    t0 = time.perf_counter()
    c.extrair_respostas(texto, 1)
    assert time.perf_counter() - t0 < 2


# ── número ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("resposta, gabarito, unidade", [
    ("136 – 204 MeV/c^{2}", "136", r"\text{MeV}/c^{2}"),    # faixa
    (r"130 \cdot 2 N", "130", "N"),                          # conta inacabada
    ("90 – 135 J", "90.0", "J"),
    (r"80.4^{2} m/s", "80.4", "m/s"),
    (r"80.4 (1 - 2) J", "80.4", "J"),
    (r"\text{至少 } 60", "60", None),                         # qualificador em chinês
    (r"\text{不超过 } 130 \text{ N}", "130", "N"),
    (r"90 \text{ thousand}", "90.0", None),                   # texto que muda o valor
    (r"60 \text{ negative}", "60", None),
    (r"60 \mathrm{N_A}", "60", None),                         # símbolo em texto
    (r"60 / \text{n}", "60", None),
    (r"2.861 \cdot 10^{1} \pm 0.1 \cdot 10^{1}", "286.1", None),   # escala com \cdot
    (r"2.02 \times 10^{1} \pm 0.1 \cdot 10^{1}", "202.0", None),
    (r"3 \cdot 10^{3} \pm 2 \times 10^{2}", "3e5", None),
    (r"\arcsin\theta = 0.132", "0.132", None),               # função não é grandeza
    (r"\arctan(x) = 0.132", "0.132", None),
])
def test_numero_reprova(resposta, gabarito, unidade):
    assert c.comparar_numero(resposta, gabarito, rel=1e-2, unidade=unidade) is V.FAIL


@pytest.mark.parametrize("resposta, gabarito, unidade", [
    (r"2.861 \cdot 10^{1} \pm 0.1 \cdot 10^{1}", "28.61", None),
    (r"3 \cdot 10^{3} \pm 2 \times 10^{2}", "3000", None),
    (r"10^{3} \pm 2 \times 10^{2}", "1000", None),
    (r"1.50 \pm 0.05 \times 10^{-3}", "1.5e-3", None),       # rodada 3, segue
    (r"2.8 \text{ meters}", "2.8", None),
    (r"5 \text{ meters per second}", "5", None),
    (r"3 \text{ electron volts}", "3", None),
    (r"0.09\,\text{°C}", "0.09", None),
    (r"3.4 \text{\AA}", "3.4", None),
    (r"7 \text{ km/s/Mpc}", "7", None),
    (r"2 \text{ (approx.)}", "2", None),
    (r"2 \text{ exactly}", "2", None),
    (r"0.3\,\mu", "0.3", r"\mu \mathrm{m}"),                # μ separado da letra
    (r"8.7\,\mu", "8.7", r"\mu\mathrm{b}"),
    (r"3 W/(m K)", "3", None),
    (r"6.626 \times 10^{-34} J s", r"6.626 \times 10^{-34}", r"\mathrm{J \cdot s}"),
    (r"\theta = 0.132", "0.132", None),
])
def test_numero_passa(resposta, gabarito, unidade):
    assert c.comparar_numero(resposta, gabarito, rel=1e-2, unidade=unidade) is V.PASS


# ── expressão ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("resposta, gabarito", [
    (r"\frac{1}{v}, \text{ whereas in the other case it is } 0", r"\frac{1}{v}"),
    (r"\frac{1}{v}, \text{ i.e. at most this value}", r"\frac{1}{v}"),
    (r"\frac{a}{b} \text{ where } t < T, \text{ and } 0 \text{ elsewhere}", r"\frac{a}{b}"),
    (r"\arcsin\theta = \frac{1}{v}", r"\frac{1}{v}"),
    (r"\sum_{n=1}^{2000000} \frac{v}{n}", r"\frac{1}{v}"),
])
def test_expressao_reprova(resposta, gabarito):
    assert c.comparar_expressao(resposta, gabarito) is V.FAIL


def test_torre_de_potencias_nao_e_calculada():
    t0 = time.perf_counter()
    assert c.comparar_expressao(r"e^{e^{e^{e^{v}}}}", r"\frac{1}{v}") is V.INCONCLUSIVE
    assert c.comparar_expressao(r"x^{x^{x^{x^{x^{x}}}}}", "x") is V.INCONCLUSIVE
    assert time.perf_counter() - t0 < 2
    assert c.comparar_expressao(r"e^{-e^{x}}", r"\exp(-\exp(x))") is V.PASS


def test_where_que_define_simbolo_segue_saindo():
    assert c.comparar_expressao(r"\frac{3}{8} g, \text{where } g \text{ is the gravity}",
                                r"\frac{3}{8} g") is V.PASS


# ── V/F e letras ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("resposta, esperado", [
    (r"\text{Correct answer is no}", None),
    (r"\text{No doubt, it is conserved}", None),
    (r"\text{No question, it holds}", None),
    (r"\text{No matter what, it holds}", None),
    (r"\text{False alarm, it is true}", None),
    (r"\text{True only if the field is uniform}", None),
    (r"\text{Yes, the answer is no}", None),
    (r"\text{True: the answer is no}", None),
    (r"\text{Yes, it is correct}", True),
    (r"\text{No, it is not conserved}", False),
])
def test_verdade(resposta, esperado):
    assert c.verdade(resposta) is esperado


@pytest.mark.parametrize("resposta, esperado", [
    (r"\text{A. B is also correct}", frozenset()),
    (r"\text{C. B}", frozenset()),
    (r"C. B too", frozenset()),
    (r"\text{C. Probably D}", frozenset()),
    (r"\text{(D) The spin is not conserved}", frozenset("D")),   # rodada 3, segue
    (r"C. 4.2 J", frozenset("C")),
])
def test_letras(resposta, esperado):
    assert c.letras(resposta) == esperado


# ── segunda leva da rodada 4 (atacante de falsos positivos, refeito) ─────────


@pytest.mark.parametrize("resposta, gabarito", [
    (r"\text{upper bound} = 60", "60"),
    (r"\text{at most} = 130", "130"),
    (r"\text{not} = 90", "90"),
    (r"\text{error} = 90", "90"),
    (r"\not= 60", "60"),
])
def test_nome_em_texto_qualificador_reprova(resposta, gabarito):
    assert c.comparar_numero(resposta, gabarito, rel=1e-2) is V.FAIL


def test_nome_em_texto_de_grandeza_passa():
    assert c.comparar_numero(r"\text{speed} = 80.4", "80.4", rel=1e-2) is V.PASS


@pytest.mark.parametrize("resposta, esperado", [
    (r"\text{Yes/No}", None), (r"\text{No/Yes}", None), (r"\text{True/False}", None),
    (r"\text{Yes \& no}", None), (r"\text{Yes -- or rather no}", None),
])
def test_verdade_com_barra(resposta, esperado):
    assert c.verdade(resposta) is esperado


@pytest.mark.parametrize("resposta, esperado", [
    (r"\text{C. Option D is also valid}", frozenset()),
    (r"\text{(C). Answer D could be also correct}", frozenset()),
    (r"\text{C. Option C is the right one}", frozenset("C")),
])
def test_letras_option(resposta, esperado):
    assert c.letras(resposta) == esperado


@pytest.mark.parametrize("resposta, gabarito, esperado", [
    (r"\sqrt{E} = -\frac{m v^{2}}{2}", r"E = \frac{m^{2} v^{4}}{4}", V.FAIL),
    (r"\sqrt{E} = \frac{m v^{2}}{2}", r"E = \frac{m^{2} v^{4}}{4}", V.PASS),
    (r"-\sqrt{\lambda} = \sqrt{\frac{a}{b}}", r"\lambda = \frac{a}{b}", V.FAIL),
])
def test_isolar_pela_raiz_confere_a_equacao(resposta, gabarito, esperado):
    assert c.comparar_definicao(resposta, gabarito) is esperado


@pytest.mark.parametrize("resposta, gabarito", [
    (r"\frac{d(\sigma Q)}{d\varepsilon}", r"\frac{\sigma Q}{\varepsilon}"),
    (r"\frac{Q\, d\sigma}{d\varepsilon}", r"\frac{\sigma Q}{\varepsilon}"),
    (r"\frac{d\sigma}{d\Omega}", r"\frac{\sigma}{\Omega}"),
])
def test_derivada_nao_calculada_reprova(resposta, gabarito):
    assert c.comparar_expressao(resposta, gabarito) is V.FAIL


def test_intervalo_com_not_reprova():
    assert c.comparar_intervalo(r"r \not\in (2 a_{0}, \infty)", r"(2 a_{0}, \infty)") is V.FAIL
    assert c.comparar_intervalo(r"\arcsin(r) > 2 a_{0}", r"(2 a_{0}, \infty)") is V.FAIL


def test_error_zero_do_olympiad_e_tolerancia_nao_declarada():
    """Achado do autoteste do OlympiadBench-teste (só por flags estruturais): `error` 0 fazia
    a perturbação igual ao gabarito, e 16 Numerical eram excluídos."""
    from phifm.eval import g2_itens
    from phifm.eval.g2_itens import Item
    assert g2_itens.tolerancias_olympiad("0", 1) == [None]
    assert g2_itens.tolerancias_olympiad("0.1,0", 2) == [0.1, None]
    assert g2_itens.tolerancias_olympiad(None, 2) == [None, None]
    assert g2_itens.tolerancias_olympiad("0.5", 3) == [0.5, 0.5, 0.5]
    item = Item("olympiadbench", "x", "Q", ("3.0",), ("Numerical",),
                tuple(g2_itens.tolerancias_olympiad("0", 1)))
    r = g2_itens.autoteste(item)
    assert r["ok"] and r["perturbacoes"] == 2


@pytest.mark.parametrize("texto, esperado", [
    # iteração 1 do dev: a resposta em texto, sem caixa e sem "final"
    ("Therefore, the decay is possible. The answer is True.", "True"),
    (r"so it violates isospin. The answer is False.", "False"),
    ("The answer is $5$ m/s.", "5"),
    # a frase do prompt e a prosa não são resposta
    ("The answer is a number; put only the number in \\boxed{}.", None),
    ("The answer is independent of the mass.", None),
])
def test_frase_the_answer_is_sem_caixa(texto, esperado):
    assert c.extrair_resposta(texto) == esperado


def test_frase_com_alternativa_entre_as_partes():
    texto = PRE + "The final answers are $3e-05$ and $1.8e-05$ or $5.4e-05$."
    assert c.extrair_respostas(texto, 2) is None
