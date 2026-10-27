"""Regressões da SEGUNDA rodada do estresse adversarial do corretor do G2 (2026-10-20),
sobre a versão reescrita. Cada caso é um achado reproduzido, escrito sem conteúdo de
benchmark."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_corretor as c  # noqa: E402
from phifm.eval import g2_itens  # noqa: E402
from phifm.eval.g2_corretor import Veredito as V  # noqa: E402
from phifm.eval.g2_itens import Item  # noqa: E402

# ── extração ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("texto, esperado", [
    (r"Hence $E \approx \boxed{8.2}$. Note that the final answer is larger than the naive "
     r"estimate $7.617$.", "8.2"),
    (r"So $\boxed{8.2}$. The final answer is not $7.617$; it is $8.2$.", "8.2"),
    (r"So $\boxed{8.2}$. Thus the final answer is either $7.617$ or $8.2$.", "8.2"),
    (r"$\boxed{\text{No}}$." + "\n\nWe can verify that the final answer is correct.", r"\text{No}"),
    (r"The final answer is $\boxed{13.06}$. This final answer is reasonable since it is close.",
     "13.06"),
    (r"First, $T = \boxed{300}$ K. ... The final answer is 352 m/s.", "300"),   # rodada 4
    ("The **final answer** is $13.06$ eV.", "13.06"),
])
def test_frase_final_so_vence_quando_responde(texto, esperado):
    assert c.extrair_resposta(texto) == esperado


@pytest.mark.parametrize("texto, esperado", [
    ("Refer to Problem 2.1 and set coordinates. The final answer is $\\boxed{B = 0}$.", "B = 0"),
    ("As shown in Problem 2.5, the range is R. So $\\boxed{202}$.", "202"),
    ("Problem analysis: the block slides. $\\boxed{5}$.", "5"),
    ("So $\\boxed{7.617}$ eV.\n\nQuestion: is this consistent? Redoing it, $\\boxed{8.2}$.", "8.2"),
    ("So $\\boxed{13.06}$.\n\n## Problem 2\nA ball. $\\boxed{5.1}$", "13.06"),
    ("So $\\boxed{13.06}$.\n\nThe following are physics problems with step-by-step solutions.",
     "13.06"),
    ("So $\\boxed{13.06}$.\n\nA new mass.\nThe answer is a number; put only the number in "
     "\\boxed{}.\nSolution: $\\boxed{9}$", "13.06"),
])
def test_corte_de_continuacao(texto, esperado):
    assert c.extrair_resposta(texto) == esperado


def test_corte_e_linear_em_texto_longo():
    t0 = time.perf_counter()
    assert c.extrair_resposta("x" + "\n" * 30000 + r" The final answer is $\boxed{13.06}$.") == "13.06"
    assert time.perf_counter() - t0 < 2


def test_varias_caixas_contam_com_unidade_e_tupla():
    assert c.extrair_respostas(r"(1) $a = \boxed{2\,\mathrm{m/s^2}}$. (2) $v = \boxed{8\,\mathrm{m/s}}$.",
                               2) == r"2\,\mathrm{m/s^2}, 8\,\mathrm{m/s}"
    assert c.extrair_respostas(r"First $F = \boxed{4}$ N. The final answer is $\boxed{(2, 8)}$.",
                               2) == "(2, 8)"
    num = lambda r, g: c.comparar_numero(r, g, rel=1e-2)  # noqa: E731
    for resp in ("2；8", "2、8", r"2 \text{ 和 } 8",
                 r"\begin{aligned} a &= 2 \\ v &= 8 \end{aligned}"):
        assert c.comparar_varias(resp, ["2", "8"], [num, num]) is V.PASS, resp


# ── números ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("resposta, gabarito", [
    (r"2I + 1 = \frac{7}{2}", r"I = \frac{7}{2}"),
    (r"v^{2} = 1.41", "1.41"),
    (r"8.2 \approx 7.617", "7.617"),
    (r"\text{less than } 7.617", "7.617"),
    (r"7.617 \text{ or zero}", "7.617"),
    (r"\text{not } 7.617", "7.617"),
    ("30 T", "30"), ("7.617 kT", "7.617"), ("5.2 / g", "5.2"), ("-2 g", "-2"),
    ("5.2 m g h", "5.2"),
])
def test_numero_falsos_positivos(resposta, gabarito):
    assert c.comparar_numero(resposta, gabarito, rel=1e-2) is V.FAIL


@pytest.mark.parametrize("resposta, gabarito, unidade", [
    (r"0.3\,\mu\mathrm{m}", "0.3", r"\mu \mathrm{m}"),
    (r"136 \text{MeV}/c^{2}", "136", r"\text{MeV}/c^{2}"),
    (r"0.22\, k", "0.22", "k"),
    ("13.06eV", "13.06", r"\mathrm{eV}"),
    ("352m/s", "352", None),
    ("300 K", "300", "K"),
    (r"10^{4}\ \mathrm{J}/\mathrm{K}", r"10^{4}", r"\mathrm{J}/\mathrm{K}"),
    (r"13.06 \pm 0.01", "13.06", None),
    (r"\text{13.06 eV}", "13.06", None),
    (r"\textbf{13.06}", "13.06", None),
    (r"v = 4.2", "4.2", None),
    (r"E \approx -13.6", "-13.6", None),
    (r"I = 3.5", r"I = \frac{7}{2}", None),
])
def test_numero_com_unidade_e_formas(resposta, gabarito, unidade):
    assert c.comparar_numero(resposta, gabarito, rel=1e-2, unidade=unidade) is V.PASS


# ── expressões ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("resposta, gabarito", [
    (r"C^{-1} = 4 \pi \epsilon_{0} a", r"4 \pi \varepsilon_{0} a"),
    (r"v^{2} = \frac{4}{3}\sqrt{gH}", r"\frac{4}{3}\sqrt{gH}"),
    (r"\ln(C) = 4 \pi \epsilon_{0} a", r"4 \pi \varepsilon_{0} a"),
    (r"\frac{4}{3}\sqrt{gH} \text{ or zero}", r"\frac{4}{3}\sqrt{gH}"),
    (r"\text{not } \frac{4}{3}\sqrt{gH}", r"\frac{4}{3}\sqrt{gH}"),
])
def test_expressao_falsos_positivos(resposta, gabarito):
    assert c.comparar_expressao(resposta, gabarito) is not V.PASS


@pytest.mark.parametrize("resposta, gabarito", [
    (r"\frac{1}{8\pi}\left[(\sin\phi+\cos\phi)\sin\theta+2\cos\theta\right]^{2}",
     r"\frac{1}{8 \pi}[\sin \theta(\sin \phi+\cos \phi)+2 \cos \theta]^{2}"),
    (r"\cos\theta \cdot \frac{ma}{m+M}", r"\frac{m a}{m + M} \cos \theta"),
    (r"m_\mathrm{p} + 2M", r"m_{\mathrm{p}} + 2M"),
    (r"\bigl(\frac{1}{2}\bigr) v", r"\frac{1}{2} v"),
    (r"\left. \frac{v}{2} \right.", r"\frac{1}{2} v"),
    (r"{\textstyle\frac{1}{2}} v", r"\frac{1}{2} v"),
    (r"\frac v2", r"\frac{1}{2} v"),
    (r"\frac{1}{2} m l^2 (\dot{\theta_1}^2 + \dot{\theta_2}^2)",
     r"\frac{1}{2} m l^{2} \dot{\theta}_{1}^{2}+\frac{1}{2} m l^{2} \dot{\theta}_{2}^{2}"),
    (r"\frac{q}{4\pi\varepsilon_0 r^2}\hat{\mathbf{r}}", r"\frac{q}{4 \pi \varepsilon_{0} r^{2}} \boldsymbol{e}_{r}"),
    (r"\frac{1}{2}\vec v", r"\frac{1}{2} v"),
    ("R ln 2", r"R \ln 2"),
    ("1/2 v", r"\frac{1}{2} v"),
    ("mu0*n*I*r/(4*R)", r"\frac{\mu_{0} n I r}{4 R}"),
    (r"R\log_e 2", r"R \ln 2"),
    (r"\frac{g^2 e^2}{2 ω V (M_Σ + M_Λ)^2}", r"\frac{g^2 e^2}{2 \omega V (M_{\Sigma} + M_{\Lambda})^2}"),
    (r"\frac{aT^4}{3}\ \text{J/m}^3", r"\frac{a T^{4}}{3}"),
    (r"u = \frac{a T^{4}}{3}, \text{ where } a \text{ is the radiation constant}", r"\frac{a T^{4}}{3}"),
])
def test_expressao_escritas_equivalentes(resposta, gabarito):
    assert c.comparar_expressao(resposta, gabarito) is V.PASS


@pytest.mark.parametrize("resposta, gabarito", [
    (r"v(t) = v_0\sqrt{\frac{m_0}{m_0+2 A t}}", r"v = \sqrt{\frac{m_{0}}{m_{0} + 2 A t}} v_{0}"),
    (r"T = T_0 - \frac{b}{r}", r"T(r) =T_{0}-\frac{b}{r}"),
])
def test_lado_esquerdo_com_ou_sem_argumentos(resposta, gabarito):
    assert c.comparar_definicao(resposta, gabarito) is V.PASS


def test_expoente_de_um_caractere_nao_cola_digitos():
    g = r"[\text{Xe}] 4f^{14} 5d^2 6s^2"
    assert c.comparar_expressao(r"[\text{Xe}] 4f^{14} 5d^{26} s^{2}", g) is not V.PASS


def test_letras_and_nao_vira_alternativa():
    assert c.letras("A and C") == frozenset({"A", "C"})
    assert c.letras("B and C") == frozenset({"B", "C"})


def test_funcao_nomeada_nao_e_linear():
    g = r"\operatorname{erf}\left(\frac{x}{2a}\right)"
    assert c.comparar_expressao(r"\operatorname{erf}\left(\frac{x}{2a}\right)", g) is V.PASS
    assert c.comparar_expressao(r"2\operatorname{erf}\left(\frac{x}{4a}\right)", g) is V.FAIL


def test_ch_russo_e_cosh():
    g = r"a \operatorname{ch} \left(\frac{x}{a}\right) - x"
    assert c.comparar_expressao(r"a \cosh\left(\frac{x}{a}\right) - x", g) is V.PASS
    assert c.comparar_expressao(r"2a \operatorname{ch}\left(\frac{x}{2a}\right) - x", g) is V.FAIL


def test_torre_de_potencias_nao_trava():
    t0 = time.perf_counter()
    v = c.com_teto(lambda: c.comparar_expressao(r"10^{10^{7}} M c^2",
                                                r"\frac{(E_{2}-E_{1})^{2}}{2 M c^{2}}"))
    assert v is not V.PASS
    assert time.perf_counter() - t0 < 15


# ── intervalo, letras e V/F ──────────────────────────────────────────────────


def test_intervalo_confere_a_variavel():
    g = r"N_1 < \frac{N}{2}"
    for r in (r"N_2 < \frac{N}{2}", r"N < \frac{N}{2}", r"N_2 \in (-\infty, \frac{N}{2})"):
        assert c.comparar_intervalo(r, g) is V.FAIL, r
    assert c.comparar_intervalo(r"N_{1} \in (-\infty, \frac{N}{2})", g) is V.PASS


@pytest.mark.parametrize("resposta, esperado", [
    ("(A), (C)", {"A", "C"}), ("(B)(D)", {"B", "D"}), ("(A) or (C)", set()),
    (r"\text{(A) is wrong; (C)}", set()), ("B: no; D: yes", set()), ("A. Also C is correct.", set()),
    (r"\text{Option C}", {"C"}), ("[C]", {"C"}), (r"\text{Answer: C}", {"C"}),
    (r"\text{\textbf{C}}", {"C"}),
])
def test_letras_listas_e_rotulos(resposta, esperado):
    assert c.letras(resposta) == frozenset(esperado)


@pytest.mark.parametrize("resposta, esperado", [
    (r"\text{No, it is not true}", False), (r"\text{True, there is no contradiction}", True),
    (r"\text{\textbf{False}}", False), ("No idea", None), ("Yes and no", None),
    ("The statement is false", False), ("Not true", False),
])
def test_verdade(resposta, esperado):
    assert c.verdade(resposta) is esperado


# ── carregador ───────────────────────────────────────────────────────────────


def test_olympiad_sem_dolar_solto_e_autoteste_pega_dolar():
    item = Item("olympiadbench", "x", "Q", ("a$", "$b"), ("Expression", "Expression"), (None, None))
    assert not g2_itens.autoteste(item)["ok"]
    # Os benchmarks ficam fora do git (/data/*): no CI e numa worktree nova não existem.
    if not (g2_itens.BENCH / "olympiadbench").exists():
        pytest.skip("os benchmarks do G2 não estão nesta cópia (data/raw/benchmarks)")
    for i in g2_itens.carregar("olympiadbench", "dev"):
        assert not any("$" in g for g in i.gabaritos), i.id
