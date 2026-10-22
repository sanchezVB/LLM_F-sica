"""Regressões da TERCEIRA rodada do estresse adversarial do corretor do G2 (2026-10-21),
quase todas criadas pelas correções da segunda. Cada caso é um achado reproduzido, escrito
sem conteúdo de benchmark (formas sintéticas)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_corretor as c  # noqa: E402
from phifm.eval.g2_corretor import Veredito as V  # noqa: E402

# ── extração ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("texto, esperado", [
    # condição ou contrafactual ANTES da frase, na mesma sentença: vale a caixa
    (r"So $\boxed{12.75}$ eV. Had we neglected recoil, the final answer is $13.06$ eV.",
     "12.75"),
    (r"So $E = \boxed{12.75}$ eV. The final answer is $13.06$ eV? No, it is lower.", "12.75"),
    # "Solution:", o eco "The answer is a number" e "**Problem**" sem número não cortam
    ("So $E = \\boxed{13.06}$ eV.\nSolution: redo with the reduced mass. $E = \\boxed{12.75}$ eV.",
     "12.75"),
    ("So $\\boxed{13.06}$ eV.\nThe answer is a number, but the recoil changes it: $\\boxed{12.75}$",
     "12.75"),
    ("So $\\boxed{13.06}$ eV.\n**Problem**\nI ignored the recoil. Correcting: $\\boxed{12.75}$ eV.",
     "12.75"),
    # comentário DEPOIS do valor não anula a frase
    ("Energy gives v^2 = 2gh. The final answer is $80.4$, correct to three significant figures.",
     "80.4"),
    ("The final answer is $80.4$ m/s, which is consistent with free fall.", "80.4"),
    ("Thus the final answer is $80.4$ m/s.\n\nCheck: the final answer is reasonable since v < c.",
     "80.4"),
    # enchimento entre a frase e o valor
    ("The final answer is therefore $80.4$ m/s.", "80.4"),
    ("The final answer is, therefore, $80.4$ m/s.", "80.4"),
    # marcador no meio do raciocínio, antes de qualquer resposta, não apaga o resto
    ("Let us restate the method.\nSolution: from energy conservation, $v = \\boxed{80.4}$ m/s.",
     "80.4"),
    ("We compute.\nThe answer is a number close to 80: $\\boxed{80.4}$", "80.4"),
])
def test_extracao_rodada3(texto, esperado):
    assert c.extrair_resposta(texto) == esperado


def test_dois_valores_na_frase_sem_caixa_nao_decidem():
    assert c.extrair_resposta("The final answer is $13.06$ eV, $12.75$ eV.") != "13.06"


@pytest.mark.parametrize("texto", [
    r"(1) $a = \boxed{2}$ m/s$^2$. (2) $v = \boxed{8}$ m/s. The final answers are $2$ m/s$^2$ "
    r"and $8$ m/s.",
    r"(1) $a = \boxed{2}$. (2) $v = \boxed{8}$. The final answer is $a = 2$ m/s$^2$, $v = 8$ m/s.",
])
def test_frase_final_com_varias_respostas_traz_todas(texto):
    lida = c.extrair_respostas(texto, 2)
    assert len(c.respostas_da_caixa(lida)) == 2
    assert c.comparar_varias(lida, ["2", "8"], [lambda r, g: c.comparar_numero(r, g, rel=1e-2)] * 2) \
        is V.PASS


# ── número ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("resposta, gabarito, unidade", [
    (r"6/a", "6", "a"),                      # a unidade só multiplicava o último termo
    (r"8 - 2a", "6", "a"),
    (r"1 - 0.47c", "0.53", "c"),
    (r"2/s", "2", "s"),
    (r"5.4 \times 10^{-4} h/g", r"5.4 \times 10^{-4}", None),   # letra com barra não é unidade
    (r"5.4 \times 10^{-4} g^{2}", r"5.4 \times 10^{-4}", None),
    (r"5.4 \times 10^{-4} ML^{2}", r"5.4 \times 10^{-4}", None),
    (r"8.2 \approx 7.617", "7.617", None),    # rodada 2, segue reprovando
    (r"v^{2} = 1.41", "1.41", None),
    (r"2I + 1 = 3.5", "3.5", None),
    (r"2.8 \text{ or less}", "2.8", None),
    (r"2.8 \text{ at least}", "2.8", None),
])
def test_numero_reprova(resposta, gabarito, unidade):
    assert c.comparar_numero(resposta, gabarito, rel=1e-2, unidade=unidade) is V.FAIL


@pytest.mark.parametrize("resposta, gabarito, unidade", [
    (r"6 a", "6", "a"),
    (r"3 m/s", "3", None),
    (r"\approx 13.06\ \text{eV}", "13.06", None),     # lado vazio da cadeia não conta
    ("≈ 13.06 eV", "13.06", None),
    (r"\sim 1.3\times 10^{3}\ \mathrm{s}", "1300", None),
    (r"\sqrt{6464} \approx 80.4", "80.4", None),       # lado do meio na tolerância do item
    (r"v = \sqrt{6464} \approx 80.4", "80.4", None),
    (r"\frac{1}{3} \approx 0.333", "0.333", None),
    (r"|v| = 80.4", "80.4", None),                     # módulo e nome em texto
    (r"v = \sqrt{2 g h} \approx 80.4", "80.4", None),  # expressão simbólica no meio
    (r"\text{speed} = 80.4", "80.4", None),
    (r"2.8 \text{ meters}", "2.8", None),              # unidade por extenso
    (r"6.3 \times 10^{-3} \text{ radians}", r"6.3 \times 10^{-3}", None),
    (r"0.09\,\text{°C}", "0.09", None),
    (r"6.626 \times 10^{-34} J s", r"6.626 \times 10^{-34}", r"\mathrm{J \cdot s}"),
    ("3.4 Å", "3.4", r"\mathrm{\AA}"),
    (r"12 N m", "12", r"\mathrm{N \cdot m}"),
    (r"1.50 \pm 0.05 \times 10^{-3}", "1.5e-3", None),  # a escala é do valor central
    (r"(1.50 \pm 0.05) \times 10^{-3}", "1.5e-3", None),
    (r"1.50 \times 10^{-3} \pm 0.05 \times 10^{-3}", "1.5e-3", None),
    (r"8.33 \times 10^14", r"8.33 \times 10^{14}", None),   # expoente de dois dígitos
    (r"6.02 \times 10^23", r"6.02 \times 10^{23}", None),
    (r"48.189\,685", "48.189685", None),                    # grupos SI depois da vírgula
    (r"0.000\,123", "0.000123", None),
    (r"6.626\,070 \times 10^{-34}", r"6.626070 \times 10^{-34}", None),
])
def test_numero_passa(resposta, gabarito, unidade):
    assert c.comparar_numero(resposta, gabarito, rel=1e-6, unidade=unidade) is V.PASS \
        or c.comparar_numero(resposta, gabarito, rel=1e-2, unidade=unidade) is V.PASS


def test_numero_tolerancia_justa_no_meio_da_cadeia():
    # 80,399 ≈ 80,4 a 1e-2, mas não a 1e-6
    assert c.comparar_numero(r"\sqrt{6464} \approx 80.4", "80.4", rel=1e-2) is V.PASS
    assert c.comparar_numero(r"\sqrt{6464} \approx 80.4", "80.4", rel=1e-6) is V.FAIL


def test_gabarito_com_expoente_sem_chaves():
    assert c.numero(r"\frac{25}{3}\times 10^14") == pytest.approx(25 / 3 * 1e14)
    assert c.numero(r"6\,000\,000") == 6e6


# ── expressão ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("resposta, gabarito", [
    (r"\frac{F_{2}(t)}{m}", r"\frac{F_{1}(t)}{m}"),         # força não é função nomeada
    (r"\mu m g", r"\mu F_{N}(0) m g"),
    (r"2\mathrm{erf}\left(\frac{x}{4a}\right)", r"\mathrm{erf}\left(\frac{x}{2a}\right)"),
    (r"2\operatorname{erf}\left[\frac{x}{4a}\right]", r"\operatorname{erf}\left[\frac{x}{2a}\right]"),
    (r"2\operatorname{erf}\frac{x}{4a}", r"\operatorname{erf}\frac{x}{2a}"),
    (r"\operatorname{cn}(u)", r"\operatorname{dn}(u)"),     # pseudofunções não colidem
    (r"\operatorname{Ci}(x)", r"\operatorname{sgn}(x)"),
    (r"\operatorname{erfc}(x)", r"\operatorname{Bi}(x)"),
    (r"a + 1", r"a + \operatorname{erf}(0)"),
    (r"\ln r^{2}", r"(\ln r)^{2}"),                          # o expoente fica no argumento
    (r"\sin\theta^{2}", r"(\sin\theta)^{2}"),
    (r"A\cos \omega t^{2}", r"A\cos^{2} (\omega t)"),
    (r"\max\left(a, b\right)", r"\min\left(a, b\right)"),
    (r"\frac{3}{8} g \text{ for the lighter block}, \frac{5}{8} g \text{ for the heavier}",
     r"\frac{3}{8} g"),                                      # "for" não corta mais
    (r"\frac{3}{8} g \text{ since the string is taut, or zero}", r"\frac{3}{8} g"),
    (r"\frac{3}{8} g \quad \text{for } t < T, \quad 0 \quad \text{otherwise}", r"\frac{3}{8} g"),
])
def test_expressao_reprova(resposta, gabarito):
    assert c.comparar_expressao(resposta, gabarito) is V.FAIL


@pytest.mark.parametrize("resposta, gabarito", [
    (r"F_0\left(\frac{1}{a b} + \frac{e^{-a t}}{a}\right)",
     r"F_{0}\left[\frac{1}{a b} + \frac{e^{-a t}}{a}\right]"),
    (r"\frac{F_0}{a b} + \frac{F_0 e^{-a t}}{a}", r"F_{0}\left[\frac{1}{a b} + \frac{e^{-a t}}{a}\right]"),
    (r"F_N(1 + \mu)", r"F_{N} + \mu F_{N}"),
    (r"\mathrm{erf}\left(\frac{x}{2a}\right)", r"\operatorname{erf}\left(\frac{x}{2a}\right)"),
    (r"erf(x/(2a))", r"\operatorname{erf}\left(\frac{x}{2a}\right)"),
    (r"\operatorname{erf}\frac{x}{2a}", r"\operatorname{erf}\left(\frac{x}{2a}\right)"),
    (r"\ln(r^{2})", r"\ln r^{2}"),
    (r"2\ln r", r"\ln(r^{2})"),
    (r"\frac{G_F^2 S}{4\pi^2}\cos^2\theta_c", r"\frac{G_{\mathrm{F}}^{2} S}{4 \pi^{2}} \cos^{2} \theta_{\mathrm{c}}"),
    (r"m g l (1-\cos\theta_{max})", r"m g l (1 - \cos \theta_{\text{max}})"),
    (r"\sin(x^{2})", r"\sin x^{2}"),
    (r"v^{2}\sin(\varphi)", r"\sin \varphi v^{2}"),
    (r"a\sin\alpha(\cos\alpha\,\hat{i} + \sin\alpha\,\hat{j})",
     r"a \sin \alpha (\cos \alpha \mathbf{i} + \sin \alpha \mathbf{j})"),   # versores
    (r"a\sin\alpha(\cos\alpha\,\hat{x} + \sin\alpha\,\hat{y})",
     r"a \sin \alpha (\cos \alpha \mathbf{i} + \sin \alpha \mathbf{j})"),
    (r"-k\arctan\left(\frac{a}{z}\right)\hat{k}", r"-k \arctan \frac{a}{z} \boldsymbol{k}"),
    (r"\frac{2a\cos\theta}{r^{3}}\,\mathbf{e}_r + \frac{a\sin\theta}{r^{3}}\,\mathbf{e}_\theta",
     r"\frac{a}{r^{3}} \left(2 \cos \theta \mathbf{e}_{r} + \sin \theta \mathbf{e}_{\theta}\right)"),
    (r"\frac{2a\cos\theta}{r^{3}}\hat{r} + \frac{a\sin\theta}{r^{3}}\hat{\theta}",
     r"\frac{a}{r^{3}} \left(2 \cos \theta \mathbf{e}_{r} + \sin \theta \mathbf{e}_{\theta}\right)"),
    (r"\mathbf{\hat{r}} \frac{k}{r^{2}}", r"\frac{k}{r^{2}} \hat{r}"),
    (r"n", r"\hat{n}"),                                      # n̂ (normal) só perde o chapéu
    (r"\min\left(\frac{v^{2}}{2g}, a\right)", r"\min\left(a, \frac{v^{2}}{2g}\right)"),
    (r"\max\left(\frac{v^{2}}{2g}, a\right)", r"\max \left(a, \frac{v^{2}}{2g}\right)"),
    ("q/(4πε0 r^2)", r"\frac{q}{4\pi\varepsilon_0 r^2}"),   # grega Unicode com índice
    (r"\frac{3}{8} g, \text{where } g \text{ is the gravity}", r"\frac{3}{8} g"),
])
def test_expressao_passa(resposta, gabarito):
    assert c.comparar_expressao(resposta, gabarito) is V.PASS


@pytest.mark.parametrize("resposta, gabarito, esperado", [
    (r"v(0) = \sqrt{\frac{a}{b}} u", r"v = \sqrt{\frac{a}{b}} u", V.FAIL),   # outro ponto
    (r"B(a) = \frac{k}{r}", r"B(r) = \frac{k}{r}", V.FAIL),
    (r"v(t) = \sqrt{\frac{a}{b}} u", r"v = \sqrt{\frac{a}{b}} u", V.PASS),
    (r"v = \sqrt{\frac{a}{b}} u", r"v(t) = \sqrt{\frac{a}{b}} u", V.PASS),
])
def test_lado_esquerdo_com_argumentos(resposta, gabarito, esperado):
    assert c.comparar_definicao(resposta, gabarito) is esperado


def test_lado_esquerdo_com_argumento_numerico_no_numero():
    assert c.comparar_numero("E(1) = 7.617", "E(2) = 7.617", rel=1e-2) is V.FAIL
    assert c.comparar_numero("E(2) = 7.617", "E(2) = 7.617", rel=1e-2) is V.PASS


# ── intervalo ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("resposta, gabarito", [
    (r"\tan\alpha \leq \mu", r"[\tan \alpha, \infty)"),
    (r"\ln 2 < x", r"(\ln 2, \infty)"),
    (r"\mu \geq \tan\alpha", r"[\tan \alpha, \infty)"),
])
def test_intervalo_com_extremo_funcao(resposta, gabarito):
    assert c.comparar_intervalo(resposta, gabarito) is V.PASS


# ── V/F e letras ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("resposta, esperado", [
    (r"\text{Yes; no}", None),
    (r"\text{No, yes}", None),
    (r"\text{No (yes for a heavy nucleus)}", None),
    (r"\text{Yes or no}", None),
    (r"\text{Yes it is}", True),
    (r"\text{No - the selection rule forbids it}", False),
    (r"\text{Yes} \quad (\text{since } \Delta l = 1)", True),
    (r"\text{No, it is not true}", False),
    (r"\text{True, there is no contradiction}", True),
    (r"\text{Yes. No energy is lost.}", True),
    (r"\text{No idea}", None),
])
def test_verdade(resposta, esperado):
    assert c.verdade(resposta) is esperado


@pytest.mark.parametrize("resposta, esperado", [
    (r"\text{C. Maybe D}", frozenset()),
    (r"\text{(C), perhaps D}", frozenset()),
    (r"\text{C. Alternatively, D}", frozenset()),
    (r"\text{A. Also C is correct}", frozenset()),
    (r"\text{(A) or (C)}", frozenset()),
    (r"\text{(D) The spin is not conserved}", frozenset("D")),
    (r"(D) No transition occurs", frozenset("D")),
    (r"(a) The field is zero or negligible", frozenset("A")),
    (r"C. 4.2 J", frozenset("C")),
])
def test_letras(resposta, esperado):
    assert c.letras(resposta) == esperado
