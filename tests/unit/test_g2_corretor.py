"""O corretor do G2 (DOC-12 §3.7.4) com respostas SINTÉTICAS, escritas aqui e não tiradas
de benchmark. O que se tranca:

1. a extração: o último `\\boxed{}` balanceado, o "final answer is" e o sem resposta;
2. os números: unidade, `\\times 10^{n}`, notação e, frações, π e as tolerâncias;
3. as expressões: π, exp, o "μ (…)" lido como função e o `x =` na frente;
4. equação, intervalo, letras, V/F e várias respostas em qualquer ordem;
5. o teto de tempo.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_corretor as c  # noqa: E402
from phifm.eval.g2_corretor import Veredito as V  # noqa: E402

# ── extração ─────────────────────────────────────────────────────────────────


def test_extrai_o_ultimo_boxed_balanceado():
    t = r"First \boxed{1}. Then $\boxed{\frac{a}{b^{2}}}$ is the final."
    assert c.extrair_resposta(t) == r"\frac{a}{b^{2}}"


def test_boxed_cortado_cai_no_anterior_e_final_answer_e_sem_resposta():
    assert c.extrair_resposta(r"\boxed{2} and then \boxed{\frac{1}{") == "2"
    assert c.extrair_resposta("So the final answer is $3.5$.\nProblem: next") == "3.5"
    assert c.extrair_resposta("I could not finish.") is None
    assert c.extrair_resposta(r"\boxed{}") is None


# ── números ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("texto, valor", [
    (r"3.1 \times 10^{3}", 3100.0),
    (r"3.1\times10^3", 3100.0),
    ("3.1e3", 3100.0),
    (r"2.0 \cdot 10^{-5}", 2e-5),
    (r"\frac{1}{4}", 0.25),
    (r"\frac{\sqrt{2} \pi}{6}", 2 ** 0.5 * 3.141592653589793 / 6),
    (r"364.7 \mathrm{nm}", 364.7),
    (r"12\,\text{m/s}", 12.0),
    ("30^\\circ", 30.0),
    (r"15\%", 15.0),
    ("1,250", 1250.0),
    (r"v = 4.2", 4.2),
    (r"E \approx -13.6", -13.6),
    (r"-0.5", -0.5),
    ("7.5 m/s", 7.5),
])
def test_numero(texto, valor):
    assert c.numero(texto) == pytest.approx(valor, rel=1e-12)


def test_numero_recusa_expressao_com_variavel():
    assert c.numero(r"\frac{h c}{\lambda}") is None


def test_comparar_numero_relativo_absoluto_e_simetrico():
    assert c.comparar_numero("100.9", "100", rel=1e-2) is V.PASS
    assert c.comparar_numero("101.1", "100", rel=1e-2) is V.FAIL
    assert c.comparar_numero("1.05", "1.0", abs_=0.05) is V.PASS
    assert c.comparar_numero("1.06", "1.0", abs_=0.05) is V.FAIL
    # o math.isclose do SciBench: relativo ao MAIOR dos dois
    assert c.comparar_numero("105", "100", rel=0.05, simetrico=True) is V.PASS
    assert c.comparar_numero("106", "100", rel=0.05, simetrico=True) is V.FAIL
    assert c.comparar_numero("x", "100", rel=1e-2) is V.FAIL
    assert c.comparar_numero("100", "T", rel=1e-2) is V.INCONCLUSIVE


# ── expressões ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("resposta, gabarito", [
    (r"\frac{1}{2} m v^{2}", r"0.5 m v^2"),
    (r"\sqrt{g R}", r"(g R)^{1/2}"),
    (r"\pi r^{2}", r"r^{2} \pi"),
    (r"e^{-\beta E}", r"\exp(-\beta E)"),
    (r"\mathrm{e}^{-x/a}", r"\exp\left(-\frac{x}{a}\right)"),
    (r"\mu g - \mu \alpha v", r"\mu (g - \alpha v)"),
    (r"u_{1} = \frac{f d}{d - f}", r"\frac{f d}{d-f}"),
    (r"\frac{\sigma T^{4}}{2}", r"0.5 \sigma T^{4}"),
    (r"\vec{F} = q E", r"E q"),
    (r"\frac{2}{3} p", r"\dfrac{2p}{3}"),
])
def test_expressoes_equivalentes(resposta, gabarito):
    assert c.comparar_expressao(resposta, gabarito) is V.PASS


@pytest.mark.parametrize("resposta, gabarito", [
    (r"\frac{1}{3} m v^{2}", r"\frac{1}{2} m v^{2}"),
    (r"\sqrt{g R} + 1", r"\sqrt{g R}"),
    (r"\pi r", r"\pi r^{2}"),
    (r"1.41 v", r"\sqrt{2} v"),
    ("0", r"m g"),
])
def test_expressoes_diferentes(resposta, gabarito):
    assert c.comparar_expressao(resposta, gabarito) is V.FAIL


# ── equação, intervalo, letras, V/F e várias respostas ──────────────────────


def test_equacao_por_razao_constante():
    assert c.comparar_equacao(r"2 v = 2 v_{0} - 2 a t", r"v = v_{0} - a t") is V.PASS
    assert c.comparar_equacao(r"v - v_{0} + a t = 0", r"v = v_{0} - a t") is V.PASS
    assert c.comparar_equacao(r"v = v_{0} + a t", r"v = v_{0} - a t") is V.FAIL
    assert c.comparar_equacao(r"v_{0} - a t", r"v = v_{0} - a t") is V.FAIL


def test_intervalo():
    assert c.comparar_intervalo(r"(2 a, \infty)", r"\left(2a, \infty\right)") is V.PASS
    assert c.comparar_intervalo(r"[0.995, 3)", r"[1, 3)") is V.PASS
    assert c.comparar_intervalo(r"(1, 3)", r"[1, 3)") is V.FAIL
    assert c.comparar_intervalo(r"[1.1, 3)", r"[1, 3)") is V.FAIL
    assert c.comparar_intervalo(r"x > 1", r"(1, \infty)") is V.FAIL


def test_desigualdade():
    assert c.comparar_intervalo(r"N_{1} < \frac{N}{2}", r"N_1 < \frac{N}{2}") is V.PASS
    assert c.comparar_intervalo(r"\frac{N}{2} > N_{1}", r"N_1 < \frac{N}{2}") is V.PASS
    assert c.comparar_intervalo(r"2 N_{1} < N", r"N_1 < \frac{N}{2}") is V.PASS
    assert c.comparar_intervalo(r"N_{1} \leq \frac{N}{2}", r"N_1 < \frac{N}{2}") is V.FAIL
    assert c.comparar_intervalo(r"N_{1} > \frac{N}{2}", r"N_1 < \frac{N}{2}") is V.FAIL
    assert c.comparar_intervalo(r"\Delta m \geq \frac{\hbar}{2 \tau}",
                                r"\Delta m \geq \frac{\hbar}{2\tau}") is V.PASS


def test_chaves_escapadas_sao_parenteses():
    assert c.comparar_expressao(r"N H \{ \coth x - 1 \}", r"N H (\coth x - 1)") is V.PASS


def test_letras_e_vf():
    assert c.comparar_letras("(C)", "C") is V.PASS
    assert c.comparar_letras(r"\text{B}", "b") is V.PASS
    assert c.comparar_letras("C, D", "D, C") is V.PASS
    assert c.comparar_letras("C", "C, D") is V.FAIL
    assert c.comparar_vf(r"\text{Yes}", "Yes") is V.PASS
    assert c.comparar_vf("True", "Yes") is V.PASS
    assert c.comparar_vf("No", "Yes") is V.FAIL
    assert c.comparar_vf("maybe", "No") is V.FAIL


def test_varias_respostas_em_qualquer_ordem():
    num = lambda r, g: c.comparar_numero(r, g, rel=1e-2)  # noqa: E731
    assert c.comparar_varias("36, 18", ["18", "36"], [num, num]) is V.PASS
    assert c.comparar_varias("18, 37", ["18", "36"], [num, num]) is V.FAIL
    assert c.comparar_varias("18", ["18", "36"], [num, num]) is V.FAIL
    # uma vírgula dentro de parênteses não separa respostas
    assert c._partes("(1, 2), 3") == ["(1, 2)", "3"]


@pytest.mark.parametrize("resposta, gabarito", [
    (r"\frac{1}{2} m \dot{x}^{2}", r"\frac{m}{2} \dot{x}^{2}"),
    (r"m l^{2} \dot{\theta}_{1}", r"l^{2} m \dot{\theta}_{1}"),
    (r"N \mu H \coth x", r"N \mu H \operatorname{coth} x"),
    (r"\frac{1}{2}\langle E \rangle", r"\frac{\langle E\rangle}{2}"),
    (r"\frac{d \sin\theta}{\lambda}", r"\frac{d}{\lambda} \sin\theta"),
    (r"q |E|", r"|E| q"),
    (r"q \left|E\right|", r"|E| q"),
    (r"|a - b| m", r"m \left| b - a \right|"),
])
def test_notacoes_de_fisica(resposta, gabarito):
    assert c.comparar_expressao(resposta, gabarito) is V.PASS


def test_modulo_nao_some():
    # o parse do verify descartava `\left|E\right|`: q|E| e q ficavam iguais
    assert c.comparar_expressao(r"q", r"q \left|E\right|") is V.FAIL
    assert c.comparar_expressao(r"(a - b) m", r"m |a - b|") is V.FAIL


def test_ponto_distingue_coordenada_de_velocidade():
    assert c.comparar_expressao(r"m \dot{x}", r"m x") is V.FAIL


def test_graus_e_minutos():
    assert c.numero("64^\\circ 10'") == pytest.approx(64 + 10 / 60)


@pytest.mark.parametrize("gabarito", [
    r"\int_{0}^{\infty} f(r) \, \mathrm{d} r",
    r"\sum_{n=1}^{\infty} \frac{1}{n^{2}}",
    r"T \left(\frac{\partial S}{\partial T}\right)_{p}",
    r"\frac{d v}{d t}",
    r"\left.\frac{\mathrm{d} \psi}{\mathrm{d} r}\right|_{r=a}",
])
def test_calculo_nao_e_verificavel_por_regra(gabarito):
    assert not c.verificavel(gabarito)
    assert c.comparar_expressao(gabarito, gabarito) is V.INCONCLUSIVE


def test_teto_de_tempo():
    def devagar():
        time.sleep(2)
        return V.PASS

    assert c.com_teto(devagar, segundos=0.2) is V.INCONCLUSIVE
    assert c.com_teto(lambda: V.FAIL, segundos=1) is V.FAIL
    assert c.com_teto(lambda: 1 / 0, segundos=1) is V.ERROR
