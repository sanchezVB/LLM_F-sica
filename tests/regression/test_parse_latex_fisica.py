"""Regressão do `parse` em LaTeX de Física (DOC-10 §3.1).

Defeitos do backend ANTLR do `sympy.parsing.latex`, achados em 2026-10-19 ao
construir o corretor do portão G2. O G2 os contornava por conta própria
(`eval/g2_corretor.py`); a correção foi para `verify/symbolic.parse` porque ele
é o barramento único dos cinco consumidores, inclusive a recompensa de RLVR.

  1. `q \\left|E\\right|` virava `q`: o módulo era descartado em silêncio.
  2. `\\pi` virava `Symbol('pi')`, e o numérico o sorteava como variável.
  3. A primeira chamada carregava o ANTLR sob o teto de 5 s do `_com_limite`.
  4. `d \\sin\\theta` virava o símbolo `dsin` (token DIFFERENTIAL).
  5. `I` declarado como corrente só valia no dimensional.
  6. Toda macro desconhecida virava símbolo livre (`\\displaystyle` →
     `displaystyle`, `\\text{m}` → `text·m`, `\\vec{F}` → `vec·F`).

Cada teste afirma a leitura certa E a consequência no barramento, porque o
defeito só custa caro quando vira veredito.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import sympy as sp
from sympy.core.function import AppliedUndef

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

from phifm.verify.bus import Claim, Verdict  # noqa: E402
from phifm.verify.numeric import NumericVerifier  # noqa: E402
from phifm.verify.symbolic import SymbolicVerifier, parse  # noqa: E402

q, E, d, theta, n, lam, corrente, R = sp.symbols("q E d theta n lambda I R")


def _iguais(e: sp.Basic | None, esperado: sp.Basic) -> bool:
    return e is not None and sp.simplify(e - esperado) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 1. Módulo — nenhuma grafia some, e nenhuma sobra é descartada
# ═══════════════════════════════════════════════════════════════════════════

MODULOS = [
    r"q \left|E\right|",   # virava `q`
    r"\left|E\right| q",   # virava None
    r"q |E|",              # sem `\`, ia para o `parse_expr` e virava None
    r"|E| q",
    r"q \lvert E \rvert",  # virava q·lvert·E·rvert
    r"q \left\lvert E \right\rvert",
    r"q \left\vert E \right\vert",
]


@pytest.mark.parametrize("latex", MODULOS)
def test_modulo_sobrevive_em_toda_grafia(latex):
    assert _iguais(parse(latex), q * sp.Abs(E)), parse(latex)


def test_modulo_nao_passa_por_igual_ao_que_ele_multiplica():
    r = SymbolicVerifier().verify(Claim(lhs=r"q \left|E\right|", rhs="q"))
    assert r.verdict is Verdict.FAIL, r.evidence


def test_bra_ket_continua_lendo_a_barra():
    """`\\left|` vira barra crua, e a regra do ket aceita as duas."""
    from sympy.physics.quantum import Ket

    assert isinstance(parse(r"\left|\psi\right\rangle"), Ket)


@pytest.mark.parametrize("latex", [r"q \cdot E ]", r"\frac{a}{b} )", r"x \to \infty"])
def test_sobra_que_o_parser_nao_entende_vira_none_e_nao_leitura_parcial(latex):
    """A gramática não exige fim de texto: `x \\to \\infty` saía `x`, e
    `q \\cdot E ]` saía `q*E`."""
    assert parse(latex) is None


@pytest.mark.parametrize(
    "latex,esperado",
    [
        (r"\left(\frac{a}{b}\right)^2", sp.Symbol("a") ** 2 / sp.Symbol("b") ** 2),
        (r"x\,", sp.Symbol("x")),
        (r"\frac{1}{2} m v^{2}\;", sp.Symbol("m") * sp.Symbol("v") ** 2 / 2),
    ],
)
def test_token_pulado_na_ponta_nao_reprova_latex_valido(latex, esperado):
    """O `strict` do SymPy confere as pontas do texto, e o `\\left` inicial ou
    o `\\;` final são tokens que a gramática pula."""
    assert _iguais(parse(latex), esperado)


# ═══════════════════════════════════════════════════════════════════════════
# 2. π — constante, nunca variável
# ═══════════════════════════════════════════════════════════════════════════


def test_pi_em_latex_e_a_constante():
    e = parse(r"\frac{\sqrt{2}\pi}{6}")
    assert e == sp.sqrt(2) * sp.pi / 6
    assert not e.free_symbols


def test_pi_aplicado_a_parenteses_e_produto():
    """`\\pi (R^2 - r^2)`, a área da coroa, saía como uma função `pi`."""
    e = parse(r"\pi (R^2 - r^2)")
    assert not e.atoms(AppliedUndef)
    assert _iguais(e, sp.pi * (R**2 - sp.Symbol("r") ** 2))


@pytest.mark.parametrize(
    "latex,esperado",
    [
        # O exemplo canônico do DOC-09 §5.1. A primeira correção levava o π
        # para dentro da potência: (π·L/g)^{1/2}.
        (r"2\pi\left(\frac{L}{g}\right)^{1/2}", 2 * sp.pi * sp.sqrt(sp.Symbol("L") / sp.Symbol("g"))),
        (r"\pi (x + 1)^2", sp.pi * (sp.Symbol("x") + 1) ** 2),
        (r"\pi\big(R^2 - r^2\big)", sp.pi * (R**2 - sp.Symbol("r") ** 2)),
    ],
)
def test_expoente_depois_de_pi_fica_com_o_parentese(latex, esperado):
    assert _iguais(parse(latex), esperado), parse(latex)


def test_forma_canonica_do_pendulo_passa_no_barramento():
    r = SymbolicVerifier().verify(
        Claim(lhs=r"2\pi\left(\frac{L}{g}\right)^{1/2}", rhs=r"2\pi\sqrt{\frac{L}{g}}")
    )
    assert r.verdict is Verdict.PASS, r.evidence


def test_pi_com_subscrito_continua_simbolo():
    assert parse(r"\pi_{0}") == sp.Symbol("pi_0")


def test_pi_em_latex_e_em_sympy_sao_a_mesma_coisa():
    """Antes, `2\\pi r` contra `2*pi*r` reprovava: π símbolo contra π número."""
    r = SymbolicVerifier().verify(Claim(lhs=r"2\pi r", rhs="2*pi*r"))
    assert r.verdict is Verdict.PASS, r.evidence


def test_numerico_nao_sorteia_pi():
    decimal = "0.74048048969306104116931349834344894976910361489595"  # √2·π/6
    r = NumericVerifier().verify(Claim(lhs=r"\frac{\sqrt{2}\pi}{6}", rhs=decimal))
    assert r.verdict is Verdict.PASS, r.evidence


# ═══════════════════════════════════════════════════════════════════════════
# 3. Carga a frio — o ANTLR carrega na importação, sem teto
# ═══════════════════════════════════════════════════════════════════════════


def test_parser_latex_ja_esta_carregado_depois_de_importar():
    """Num processo novo: importar o módulo já carrega o ANTLR e a cadeia do
    bra-ket (que traz o numpy), na thread que importa e sem deixar thread do
    `_com_limite` para trás. Com o teto baixado para 0,5 s, a primeira
    verificação ainda conclui — a frio, só a carga do ANTLR já leva mais."""
    codigo = textwrap.dedent(
        r"""
        import importlib.util, sys, threading
        import phifm.verify.symbolic as S

        assert "antlr4" in sys.modules
        assert "sympy.parsing.latex._parse_latex_antlr" in sys.modules
        assert importlib.util.find_spec("numpy") is None or "numpy" in sys.modules
        assert threading.active_count() == 1, threading.enumerate()

        S.TIMEOUT_S = 0.5
        assert S.parse(r"\frac{\sqrt{2}\pi}{6}") is not None
        print("ok")
        """
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    r = subprocess.run(
        [sys.executable, "-c", codigo], env=env, capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"


# ═══════════════════════════════════════════════════════════════════════════
# 4. `d` antes de macro — a grandeza d, não um diferencial
# ═══════════════════════════════════════════════════════════════════════════


def test_lei_de_bragg():
    e = parse(r"2 d \sin\theta = n \lambda")
    assert isinstance(e, sp.Equality)
    assert _iguais(e.lhs, 2 * d * sp.sin(theta))
    assert _iguais(e.rhs, n * lam)
    assert sp.Symbol("dsin") not in e.free_symbols


@pytest.mark.parametrize(
    "latex,esperado",
    [
        (r"d\sin\theta", d * sp.sin(theta)),
        (r"x d \cos\theta", sp.Symbol("x") * d * sp.cos(theta)),
        (r"d \cdot F", d * sp.Symbol("F")),            # virava dcdot·F
        (r"d\frac{1}{2}", d / 2),                      # virava dfrac·1·2
        (r"d\sqrt{2}", sp.sqrt(2) * d),
        (r"d\pi", sp.pi * d),                          # constante não tem diferencial
        (r"\delta \sin\theta", sp.Symbol("delta") * sp.sin(theta)),  # o `d` de `\delta` fica
    ],
)
def test_d_antes_de_funcao_operador_ou_constante_e_a_grandeza(latex, esperado):
    assert _iguais(parse(latex), esperado), parse(latex)


def test_diferencial_de_verdade_continua_diferencial():
    x = sp.Symbol("x")
    assert parse(r"\int_0^1 x\,dx") == sp.Integral(x, (x, 0, 1))
    assert parse(r"\frac{d\sin\theta}{dt}") == sp.Derivative(sp.sin(theta), sp.Symbol("t"))
    # Letra grega fica com a leitura da gramática, que é a de uma integral.
    assert parse(r"d\theta") == sp.Symbol("dtheta")


# ═══════════════════════════════════════════════════════════════════════════
# 5. `I` — corrente quando declarado, em todo verificador do barramento
# ═══════════════════════════════════════════════════════════════════════════


def test_I_em_latex_e_simbolo():
    """O ANTLR não consulta namespace, e o próprio SymPy escreve a unidade
    imaginária como `i` em LaTeX."""
    e = parse(r"I^{2} R")
    assert not e.has(sp.I)
    assert _iguais(e, corrente**2 * R)


@pytest.mark.parametrize("verificador", [SymbolicVerifier, NumericVerifier])
def test_I_declarado_como_corrente_vale_no_simbolico_e_no_numerico(verificador):
    """Antes só o dimensional lia `context["dimensions"]`; o simbólico e o
    numérico liam `I^2*R` como −R e reprovavam I²R escrito em LaTeX."""
    ctx = {"dimensions": {"I": "corrente"}}
    r = verificador().verify(Claim(lhs="I^2*R", rhs=r"I^{2} R", context=ctx))
    assert r.verdict is Verdict.PASS, r.evidence


# ═══════════════════════════════════════════════════════════════════════════
# 6. Macro que não nomeia grandeza — some se é tipografia, recusa se não é
# ═══════════════════════════════════════════════════════════════════════════

a, b, x, y = sp.symbols("a b x y")


@pytest.mark.parametrize(
    "latex,esperado",
    [
        (r"\displaystyle\frac{a}{b}", a / b),      # virava displaystyle·a/b
        (r"\textstyle x", x),
        (r"\big( a \big)", a),                     # virava a função `big`
        (r"\bigl(x+y\bigr)", x + y),
        (r"\big|x\big|", sp.Abs(x)),               # tamanho de barra ainda é módulo
    ],
)
def test_tipografia_some(latex, esperado):
    assert _iguais(parse(latex), esperado), parse(latex)


def test_limits_nao_atrapalha_o_somatorio():
    assert parse(r"\sum\limits_{i=1}^{n} i").doit() == (n**2 + n) / 2


@pytest.mark.parametrize(
    "latex,esperado",
    [
        (r"\mathrm{e}^{-\beta E}", sp.exp(-sp.Symbol("beta") * E)),  # virava mathrm/e^{βE}
        (r"\mathrm{e}", sp.E),
        (r"\mathrm{i}\omega t", sp.I * sp.Symbol("omega") * sp.Symbol("t")),
    ],
)
def test_constante_em_tipo_reto_e_a_da_iso(latex, esperado):
    """ISO 80000-2: e, i e d retos são a constante e o operador; em itálico,
    grandezas."""
    assert _iguais(parse(latex), esperado), parse(latex)


def test_d_reto_e_o_diferencial():
    t = sp.Symbol("t")
    assert parse(r"\int_0^1 x\,\mathrm{d}x") == sp.Integral(x, (x, 0, 1))
    assert parse(r"\frac{\mathrm{d}x}{\mathrm{d}t}") == sp.Derivative(x, t)


def test_e_reto_em_subscrito_e_o_eletron_e_nao_euler():
    e = parse(r"m_\mathrm{e} c^2")
    assert not e.has(sp.E)
    assert _iguais(e, sp.Symbol("m_e") * sp.Symbol("c") ** 2)


def test_sentinela_das_constantes_nao_vaza_para_um_nome():
    assert parse(r"x_{\mathrm{e} + 1}") is None


# Unidade é recusada por decisão do dono (2026-10-19): descartá-la aprovaria
# 5 km contra 5 m. Quem compara números tira a unidade por conta própria.
MACROS_SEM_GRANDEZA = [
    r"5\,\text{m}",
    r"12\,\text{m/s}",
    r"364.7 \mathrm{nm}",
    r"\vec{F}",
    r"\dot{x}",
    r"A^\dagger",
    r"a \approx b",
    r"30^\circ",
    r"a \pm b",
    r"\mathcal{E}",
    r"\operatorname{sgn}(x)",
    r"\mathbb{R}",
    r"x_{\vec{v}}",            # a macro dentro do nome do símbolo também conta
]


@pytest.mark.parametrize("latex", MACROS_SEM_GRANDEZA)
def test_macro_que_nao_nomeia_grandeza_leva_a_none(latex):
    assert parse(latex) is None, parse(latex)


def test_unidade_diferente_nao_passa_por_igual():
    """A razão da recusa, no barramento: 5 km contra 5 m não pode dar PASS."""
    r = SymbolicVerifier().verify(Claim(lhs=r"5\,\text{km}", rhs=r"5\,\text{m}"))
    assert r.verdict is not Verdict.PASS, r.evidence


def test_segunda_lei_vetorial_nao_reprova():
    """Antes, `vec·F` contra `m·vec·a` dava FAIL por `equals()` — a 2ª lei de
    Newton reprovada. Sem leitura confiável do vetor, o certo é se abster."""
    r = SymbolicVerifier().verify(Claim(lhs=r"\vec{F}", rhs=r"m\vec{a}"))
    assert r.verdict is Verdict.INCONCLUSIVE, r.evidence


@pytest.mark.parametrize(
    "latex,esperado",
    [
        (r"\hbar\omega", sp.Symbol("hbar") * sp.Symbol("omega")),
        (r"\ell", sp.Symbol("ell")),
        (r"\varepsilon_{0} E^2", sp.Symbol("varepsilon_0") * E**2),
        (r"\frac{\partial f}{\partial x}", sp.Derivative(sp.Symbol("f"), x)),
    ],
)
def test_macro_que_nomeia_grandeza_continua_lida(latex, esperado):
    assert _iguais(parse(latex), esperado), parse(latex)
