"""Regressões do estresse adversarial do corretor do G2 (2026-10-19). Três agentes
atacaram o corretor com respostas sintéticas sobre itens de DEV, e cada caso abaixo é um
achado reproduzido, escrito aqui sem conteúdo de benchmark. Os primeiros são os FALSOS
POSITIVOS, que inflariam a nota; depois vêm os falsos negativos e a extração."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_corretor as c  # noqa: E402
from phifm.eval import g2_itens  # noqa: E402
from phifm.eval.g2_corretor import Veredito as V  # noqa: E402
from phifm.eval.g2_itens import Item  # noqa: E402

# ── falsos positivos ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("resposta, gabarito", [
    (r"\frac{g^{2} t}{u - g t}, \frac{g^{2} t}{u + g t}", r"\frac{g^{2} t}{u - g t}"),
    (r"\frac{4 \pi^{2} r^{3}}{G t^{2}} \to \frac{4 \pi^{2} r^{2}}{G t^{2}}",
     r"\frac{4 \pi^{2} r^{3}}{G t^{2}}"),
    (r"2M = \frac{4 \pi^{2} r^{3}}{G t^{2}}", r"\frac{4 \pi^{2} r^{3}}{G t^{2}}"),
    (r"\omega = \omega \frac{1-\beta}{2}", r"\omega^{\prime} = \omega \frac{1-\beta}{2}"),
    (r"\frac{1}{c^{3/2}} \delta(x - c y)", r"\frac{1}{\sqrt{c}} \delta\left(\frac{x}{c} - y\right)"),
])
def test_alternativas_lado_esquerdo_e_funcao_nao_passam(resposta, gabarito):
    """O parser frouxo lia só o prefixo (`A, B` → A); `2M =` e `\\omega =` contra
    `\\omega' =` eram descartados; δ(u) virava δ·u, linear."""
    assert c.comparar_expressao(resposta, gabarito) is not V.PASS


def test_numero_nao_le_so_o_coeficiente():
    assert c.numero(r"2\sqrt{g h}") is None
    assert c.numero("2 or 3") is None
    assert c.numero(r"2 \text{ or } 3") is None
    assert c.numero(r"6\sqrt{2}a") is None
    assert c.numero("6a") is None
    assert c.comparar_numero(r"352 - 400\ \mathrm{m/s}", "352", rel=1e-2) is V.FAIL
    assert c.numero(r"1 - \frac{3}{4Z^{2}}", estrito=True) is None
    assert c.comparar_numero(r"2\sqrt{\frac{m}{k}}", r"2 \pi \sqrt{\frac{m}{5k}}", rel=1e-2) \
        is V.INCONCLUSIVE   # gabarito simbólico não é número


def test_unidade_com_expoente_nao_eleva_o_numero():
    assert c.numero(r"58.2 \mathrm{~J}\mathrm{~K}^{-1}") == pytest.approx(58.2)
    assert c.numero(r"18.76 \mathrm{m}^{2}") == pytest.approx(18.76)
    assert c.numero(r"1.16\,\mathrm{m/s}^{2}") == pytest.approx(1.16)
    assert c.numero(r"5\ \mathrm{cm} \cdot \mathrm{s}^{-1}") == pytest.approx(5)
    assert c.numero(r"468 \, \mathrm{m \cdot s^{-1}}") == pytest.approx(468)


def test_infinito_so_casa_com_o_mesmo_infinito():
    assert c.comparar_intervalo("(0, 3)", r"(-\infty, 3)") is V.FAIL
    assert c.comparar_intervalo(r"[-\infty, \infty]", "[20, 50]") is V.FAIL
    assert c.comparar_expressao(r"-\infty", r"\frac{1.98}{30}") is V.FAIL
    assert c.comparar_intervalo(r"(-\infty, 3)", r"(-\infty, 3)") is V.PASS


def test_valor_num_ponto_nao_vira_zero():
    item = Item("ugphysics", "X/1", "Q", (r"\theta(t) = 4 \arctan\left[\tan\frac{\theta(0)}{4} "
                                          r"\mathrm{e}^{\alpha t}\right]",), ("EQ",), (None,))
    assert g2_itens.tipo_efetivo(item) != "NV"
    assert g2_itens.corrigir_resposta(item, r"\theta(t) = 0") is not V.PASS
    assert g2_itens.corrigir_resposta(item, r"\theta(t) = \theta(0) e^{\alpha t}") is not V.PASS


@pytest.mark.parametrize("resposta, gabarito", [
    (r"\text{not } A", "A"), (r"\text{All but } A", "A"), (r"a = 5 \text{ m/s}", "a"),
])
def test_letra_so_vale_como_alternativa(resposta, gabarito):
    assert c.comparar_letras(resposta, gabarito) is V.FAIL


@pytest.mark.parametrize("resposta, gabarito", [
    (r"5 \text{ N}", "No"), ("F = 0", "No"), ("T = 0", "Yes"), ("Yes and no", "Yes"),
])
def test_vf_nao_le_letra_solta(resposta, gabarito):
    assert c.comparar_vf(resposta, gabarito) is V.FAIL


def test_graus_e_minutos_com_prime():
    assert c.numero(r"11^\circ 30^\prime", estrito=True) == pytest.approx(11.5)
    assert c.comparar_numero(r"11^\circ", r"11^\circ 30^\prime", rel=1e-2) is V.FAIL


def test_texto_normalizado_mantem_as_chaves():
    assert c.texto_normalizado(r"e^{-i\omega t}") != c.texto_normalizado(r"e^{-i}\omega t")


def test_milhar_com_espaco_fino_nao_vira_produto():
    assert c.numero(r"1\,418") == pytest.approx(1418)
    assert c.comparar_numero(r"1\,418", "418.0", rel=0.05, simetrico=True) is V.FAIL


# ── falsos negativos ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("resposta, gabarito", [
    (r"\frac{mv_0}{m+M}", r"\frac{m v_{0}}{m+M}"),
    (r"\frac{\sqrt{2}}{\pi}\sqrt{\frac{lk_BT}{\tau}}",
     r"\frac{\sqrt{2}}{\pi} \sqrt{\frac{l k_{\mathrm{B}} T}{\tau}}"),
    (r"\frac{p_{0} R+\beta}{8 p_{0} R+4 \beta}", r"k=\frac{p_0R+\beta }{8p_0R+4\beta }"),
    (r"\hbar\omega_1(n_1+\frac{1}{2})", r"\left(n_{1}+\frac{1}{2}\right)\hbar \omega_{1}"),
    (r"2m_{e} + \frac{2m_{e}^{2}}{M}", r"\frac{2m_{\mathrm{e}}(m_{\mathrm{e}}+M)}{M}"),
    (r"\frac{\lambda}{2 \pi \epsilon_{0} x}", r"\frac{\lambda}{2 \pi \varepsilon_{0} x}"),
    (r"-\varphi_0", r"-\phi_{0}"),
    (r"\left(\frac{g R^{2} t^{2}}{\theta^{2}}\right)^{1/3}-R",
     r"\sqrt[3]{\frac{gR^2t^2}{\theta ^2}} - R"),
    (r"h\sqrt[3]{\frac{3n}{8\pi}}", r"\left(\frac{3 n}{8 \pi}\right)^{1/3} h"),
    (r"\frac{M}{M'}-1", r"\frac{M}{M^{\prime}}-1"),
    (r"M a^2 \dot\varphi^2", r"M a^{2} \dot{\varphi}^{2}"),
    (r"\text{the result is } \frac{1}{5} m(b^{2} + c^{2})", r"\frac{1}{5} m(b^{2} + c^{2})"),
    (r"\sqrt{5 R g}\ \mathrm{m/s}", r"\sqrt{5 R g}"),
    (r"v = \sqrt{5gR} \approx 7", r"\sqrt{5 R g}"),
    (r"\mu = \tan \varepsilon \approx 0.27", r"\mu = \tan \varepsilon"),
    ("sqrt(5gR)", r"\sqrt{5 R g}"),
    ("aT^4/3", r"\frac{a T^{4}}{3}"),
    (r"{a T^4 \over 3}", r"\frac{a T^{4}}{3}"),
    (r"\cfrac{a T^{4}}{3}", r"\frac{a T^{4}}{3}"),
    (r"\frac{1}{4π}", r"\frac{1}{4 \pi}"),
    (r"\frac{3}{2} N·k", r"\frac{3}{2} Nk"),
])
def test_escritas_equivalentes_passam(resposta, gabarito):
    assert c.comparar_expressao(resposta, gabarito) is V.PASS


@pytest.mark.parametrize("resposta, gabarito", [
    (r"\tan\varepsilon = \mu", r"\mu = \tan \varepsilon"),
    (r"\lambda (t_2 - t_1) = \ln\frac{C_1}{C_2}", r"\lambda = \frac{\ln(C_{1}/C_{2})}{t_{2} - t_{1}}"),
    (r"x + t = \frac{\sin t}{\sin 1}", r"x=\frac{\sin t}{\sin 1}-t"),
])
def test_definicao_rearranjada_passa(resposta, gabarito):
    assert c.comparar_definicao(resposta, gabarito) is V.PASS


def test_funcao_seguida_de_fracao_tem_so_a_fracao_como_argumento():
    assert c.comparar_expressao(r"\tan\left(\frac{x}{4}\right) e^{y}", r"\tan\frac{x}{4} e^{y}") is V.PASS
    assert c.comparar_expressao(r"\tan\left(\frac{x e^{y}}{4}\right)", r"\tan\frac{x}{4} e^{y}") is V.FAIL
    assert c.comparar_expressao(r"\sin(\omega t)", r"\sin \omega t") is V.PASS


def test_simbolo_antes_de_parentese_e_produto_salvo_token_unico():
    assert c.comparar_expressao(r"m(g+a)^{2}", r"m (a + g)^{2}") is V.PASS
    assert c.comparar_expressao(r"m(g+a)^{2}", r"(m (a + g))^{2}") is V.FAIL
    assert c.comparar_expressao(r"\theta(0) e^{\alpha t}", r"e^{\alpha t}\theta(0)") is V.PASS
    assert c.comparar_expressao(r"\theta e^{\alpha t}", r"e^{\alpha t}\theta(0)") is V.FAIL
    assert c.comparar_expressao(r"x(t) + 1", r"1 + x(t)") is V.PASS


def test_equacao_segue_a_regra_da_razao_constante():
    """A regra aceita (§3.7.4) diz razão CONSTANTE. Multiplicar por variável não passa, e
    isso vale igual nos dois braços (o verificador do estresse confirmou que não é
    defeito)."""
    gab = r"\frac{n_{1}}{n_{2}} = \exp\left(-\frac{E_{1}-E_{2}}{k T}\right)"
    assert c.comparar_equacao(r"n_1 = n_2 e^{-(E_1-E_2)/(kT)}", gab) is V.FAIL
    assert c.comparar_equacao(r"2\frac{n_1}{n_2} = 2 e^{-(E_1-E_2)/(kT)}", gab) is V.PASS


def test_definicao_isolando_a_grandeza():
    assert c.comparar_definicao(r"E V^2 = \pi B u_0^2", r"E = \frac{\pi B u_0^2}{V^2}") is V.PASS
    assert c.comparar_definicao(r"E V = \pi B u_0^2", r"E = \frac{\pi B u_0^2}{V^2}") is V.FAIL


def test_letras_coladas_nao_sao_grandeza():
    assert c.comparar_expressao(r"kM = \frac{4 \pi^{2} r^{3}}{G t^{2}}",
                                r"\frac{4 \pi^{2} r^{3}}{G t^{2}}") is V.FAIL
    assert c.comparar_expressao(r"M = \frac{4 \pi^{2} r^{3}}{G t^{2}}",
                                r"\frac{4 \pi^{2} r^{3}}{G t^{2}}") is V.PASS


@pytest.mark.parametrize("texto, valor", [
    (r"6\,000\,000\,000\ \mathrm{K}", 6e9), ("6{,}000{,}000{,}000", 6e9),
    (r"4{,}943", 4943), (r"\sim 1.3\times 10^{3}\ \mathrm{s}", 1300), (r"4\sqrt2", 4 * 2 ** 0.5),
    ("144.7°", 144.7), ("47 ℃", 47), (r"146 J/(mol·K)", 146), (r"\SI{352}{m/s}", 352),
    (r"\num{6e9}\ \mathrm{K}", 6e9), ("e^{-2.45}", 2.718281828459045 ** -2.45),
    (r"\frac{1}{5 \times 10^{1}}", 0.02), ("3.1e3", 3100), ("352 m/s", 352),
])
def test_numeros_em_escritas_variadas(texto, valor):
    assert c.numero(texto) == pytest.approx(valor, rel=1e-9)


def test_letras_e_vf_com_texto():
    assert c.comparar_letras(r"C.\ 4.2\ \mathrm{J}", "C") is V.PASS
    assert c.comparar_letras(r"\text{(C) } E = h\nu", "C") is V.PASS
    assert c.comparar_vf(r"\text{No, it is not conserved}", "No") is V.PASS
    assert c.comparar_vf(r"\text{False (the statement is incorrect)}", "No") is V.PASS
    assert c.comparar_vf(r"\text{Yes, it is}", "Yes") is V.PASS


def test_varias_respostas_em_varias_formas():
    num = lambda r, g: c.comparar_numero(r, g, rel=1e-2)  # noqa: E731
    for resp in (r"2 \text{ and } 8", "2; 8", "(2, 8)", r"2 \quad 8"):
        assert c.comparar_varias(resp, ["2", "8"], [num, num]) is V.PASS, resp
    assert c.comparar_varias(r"86,000\ \mathrm{Pa}, 6,727\ \mathrm{m}", ["86000", "6727"],
                             [num, num]) is V.PASS


def test_desigualdade_geqslant():
    assert c.comparar_intervalo(r"\Delta m \geqslant \frac{\hbar}{2\tau}",
                                r"\Delta m \geq \frac{\hbar}{2\tau}") is V.PASS


# ── extração ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("texto, esperado", [
    (r"... so $\boxed{352}$." + "\n\nProblem 2: A block. Solution: $a = \\boxed{4.9}$", "352"),
    (r"... so $\boxed{352}$." + "\n\n**Problem:** x. $v = \\boxed{7}$", "352"),
    (r"First, $T = \boxed{300}$ K. Then ... The final answer is 352 m/s.", "352 m/s"),
    ("The final answer is $352$ m/s.", "352"),
    ("The final answer is approximately 352 m/s.", "352 m/s"),
    (r"The final answer is $\frac{3}{2} Nk$, where $k$ is Boltzmann's constant.", r"\frac{3}{2} Nk"),
    (r"The final answer is $\boxed{352}$.", "352"),
])
def test_extracao(texto, esperado):
    assert c.extrair_resposta(texto) == esperado


def test_varias_caixas_num_item_de_varias_respostas():
    texto = r"The final answer is $\boxed{2}$ m/s$^2$, $\boxed{8}$ m/s."
    assert c.extrair_respostas(texto, 2) == "2, 8"
    assert c.extrair_respostas(r"$\boxed{2, 8}$", 2) == "2, 8"


def test_excecoes_viram_veredito():
    assert c.comparar_expressao("1/0", r"\frac{3}{2} N k") in (V.FAIL, V.INCONCLUSIVE)
    assert c.comparar_equacao(r"\mu = (x > 0)", r"\mu = \tan \varepsilon") is V.FAIL
