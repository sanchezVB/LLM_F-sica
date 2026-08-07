"""Subscrito nomeado sobrevive ao parser (DOC-03 §3, DOC-10 §3).

Subscrito nomeado é a forma normal de distinguir grandezas homônimas em
Física: `E_{cin}` e `E_{pot}` são duas energias diferentes, e `E` sozinho é
ambíguo entre energia, campo elétrico e módulo de Young. Se o subscrito não
sobrevive ao parser, os dois viram a mesma coisa — e o verificador compara
símbolos errados sem sinalizar nada.

O defeito concreto é do backend ANTLR do `sympy.parsing.latex`, que lê o
conteúdo de `_{...}` como expressão em vez de nome:

    E_{cin}    →  E_{c*(i*n)}
    \\rho_{xy}  →  rho_{x*y}

A segunda linha é a grave. Produto comuta, então `\\rho_{xy}` e `\\rho_{yx}`
colapsam **no mesmo símbolo** — e resistividade Hall é antissimétrica
(ρ_xy = −ρ_yx). O mesmo vale para `g_{\\mu\\nu}` contra `g_{\\nu\\mu}`.

Isto é exatamente a transformação que o DOC-03 §3.2 lista como **rejeitada**
("Normalizar posição de índices — `T^{\\mu}_{\\nu}` vs `T_{\\nu}^{\\mu}` diferem
em Relatividade Geral"). O canonicalizador não a faz; o parser fazia por baixo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.verify.bus import Claim, Verdict  # noqa: E402
from phifm.verify.dimensional import DimensionalVerifier  # noqa: E402
from phifm.verify.symbolic import SymbolicVerifier, parse  # noqa: E402

SI = {"unit_system": "SI"}


# ═══════════════════════════════════════════════════════════════════════════
# NÍVEL PARSE — o subscrito tem de virar UM símbolo
# ═══════════════════════════════════════════════════════════════════════════

NOMEADOS = [
    (r"E_{cin}", "E_cin"),
    (r"E_{pot}", "E_pot"),
    (r"v_{max}", "v_max"),
    (r"T_{eff}", "T_eff"),
    (r"\omega_{max}", "omega_max"),
    (r"\Phi_{tot}", "Phi_tot"),
    # Fonte reta é a forma tipográfica usual do subscrito nomeado.
    (r"E_{\rm cin}", "E_cin"),
    (r"E_{\mathrm{cin}}", "E_cin"),
    (r"E_{\text{cin}}", "E_cin"),
]


@pytest.mark.parametrize("latex,esperado", NOMEADOS, ids=[c[0] for c in NOMEADOS])
def test_subscrito_nomeado_vira_um_simbolo(latex, esperado):
    e = parse(latex)
    assert e is not None, f"{latex} não parseou"
    assert isinstance(e, sp.Symbol), (
        f"{latex} virou {type(e).__name__} `{e}` — o subscrito foi lido como "
        "expressão em vez de nome"
    )
    assert e.name == esperado, f"{latex} → `{e.name}`, esperado `{esperado}`"


@pytest.mark.parametrize("latex,esperado", NOMEADOS, ids=[c[0] for c in NOMEADOS])
def test_subscrito_nomeado_nao_vaza_simbolo_livre(latex, esperado):
    """`E_{cin}` virando `E_{c}·i·n` injeta `i` e `n` como grandezas livres.

    No dimensional isso é veneno: `n` está em `AMBIGUOS` (índice de refração,
    número de mols), então um símbolo inventado pelo parser passa a exigir
    declaração que ninguém tem como dar.
    """
    e = parse(latex)
    assert {s.name for s in e.free_symbols} == {esperado}


# ── o colapso, que é o defeito de verdade ────────────────────────────────

TROCADOS = [
    ("resistividade Hall (ρ_xy = −ρ_yx)", r"\rho_{xy}", r"\rho_{yx}"),
    ("métrica em RG",                     r"g_{\mu\nu}", r"g_{\nu\mu}"),
    ("condutividade",                     r"\sigma_{xy}", r"\sigma_{yx}"),
]


@pytest.mark.parametrize("nome,a,b", TROCADOS, ids=[c[0] for c in TROCADOS])
def test_indices_trocados_nao_colapsam(nome, a, b):
    """Produto comuta; nome não. Se o subscrito vira produto, a ordem do
    índice se perde e dois tensores distintos viram o mesmo símbolo."""
    pa, pb = parse(a), parse(b)
    assert pa is not None and pb is not None
    assert pa != pb, f"{nome}: `{a}` e `{b}` colapsaram em `{pa}`"


# ═══════════════════════════════════════════════════════════════════════════
# NÍVEL VERIFICADOR — o efeito no barramento
# ═══════════════════════════════════════════════════════════════════════════

def test_simbolico_nao_iguala_grandezas_homonimas():
    """Energia cinética e potencial são grandezas diferentes. Com o subscrito
    destruído o simbólico compara `E_{c·i·n}` com `E_{p·o·t}` — chega ao
    veredito certo pelo motivo errado, e o motivo errado é o que quebra
    quando os nomes por acaso anagramam."""
    r = SymbolicVerifier().verify(Claim(lhs=r"E_{cin}", rhs=r"E_{pot}"))
    assert r.verdict is not Verdict.PASS


def test_dimensional_usa_subscrito_declarado():
    """O ponto prático: `E_{cin}` é declarável em `context['dimensions']`
    justamente porque `E` sozinho é ambíguo. Isso exige que o identificador
    que chega ao verificador seja o mesmo que o autor declarou."""
    ctx = {**SI, "dimensions": {"E_cin": "energia"}}
    r = DimensionalVerifier().verify(Claim(lhs=r"E_{cin}", rhs=r"\frac{1}{2} m v^2", context=ctx))
    assert r.verdict is Verdict.PASS, f"{r.verdict.value} — {r.evidence}"


def test_dimensional_reprova_subscrito_declarado_errado():
    """Declaração vale, mas não salva Física errada: energia cinética não é
    `m·v`. Sem este caso o teste acima passaria com um verificador que
    aprova tudo que foi declarado."""
    ctx = {**SI, "dimensions": {"E_cin": "energia"}}
    r = DimensionalVerifier().verify(Claim(lhs=r"E_{cin}", rhs=r"m v", context=ctx))
    assert r.verdict is Verdict.FAIL, f"{r.verdict.value} — {r.evidence}"


def test_dois_subscritos_distintos_permanecem_distintos_no_dimensional():
    """`E_{cin} + E_{pot}` é soma coerente; `E_{cin} + v_{max}` não é. Se os
    identificadores colapsassem, o segundo passaria."""
    ctx = {**SI, "dimensions": {"E_cin": "energia", "E_pot": "energia", "v_max": "velocidade"}}
    dim = DimensionalVerifier()
    ok = dim.verify(Claim(lhs=r"E_{cin} + E_{pot}", context=ctx))
    assert ok.verdict is Verdict.PASS, f"{ok.verdict.value} — {ok.evidence}"
    ruim = dim.verify(Claim(lhs=r"E_{cin} + v_{max}", context=ctx))
    assert ruim.verdict is Verdict.FAIL, f"{ruim.verdict.value} — {ruim.evidence}"


# ── o que NÃO pode mudar ─────────────────────────────────────────────────

INTOCADOS = [
    (r"v_0", "v_0"),
    (r"m_0", "m_0"),
    (r"E_{0}", "E_0"),
    (r"k_B", "k_B"),
    (r"x_{i}", "x_i"),
    (r"T_{\mu}", "T_mu"),
]


@pytest.mark.parametrize("latex,esperado", INTOCADOS, ids=[c[0] for c in INTOCADOS])
def test_subscrito_de_um_caractere_continua_intacto(latex, esperado):
    """Subscrito de um caractere já parseava certo. A correção não pode
    renomeá-lo: `v_0` é a chave que `INEQUIVOCOS` usa."""
    e = parse(latex)
    assert isinstance(e, sp.Symbol) and e.name == esperado, f"{latex} → {e}"


def test_expoente_nao_e_confundido_com_subscrito():
    """`x^{2}` é potência, não nome. Blindar subscrito não pode capturar
    sobrescrito — e `T^{\\mu}_{\\nu}` depende dessa distinção (DOC-03 §3.2)."""
    e = parse(r"x^{2}")
    assert e == sp.Symbol("x") ** 2


def test_equacao_so_de_subscritos_ainda_parseia():
    """`E_{tot} = E_{cin} + E_{pot}` não tem macro nenhuma além dos subscritos.

    Normalizar apaga os `_{` que decidem o roteamento, então a expressão
    passaria a cair no `parse_expr`, que não aceita `=` — a equação inteira
    viraria `None`. O roteamento é decidido antes da normalização por causa
    deste caso.
    """
    e = parse(r"E_{tot} = E_{cin} + E_{pot}")
    assert e is not None, "equação com subscritos e sem macro deixou de parsear"
    assert {s.name for s in e.free_symbols} == {"E_tot", "E_cin", "E_pot"}


def test_macro_colada_em_macro_nao_funde():
    """`\\hbar\\omega_{max}` é LaTeX corriqueiro. Reescrever o subscrito sem
    separador produziria `\\hbaromega_max` — macro inexistente — e o parser
    devolveria outro símbolo sem levantar erro."""
    e = parse(r"\hbar\omega_{max}")
    assert e is not None
    assert {s.name for s in e.free_symbols} == {"hbar", "omega_max"}


def test_equacao_completa_com_subscrito_e_macro():
    """O caso que a heurística de roteamento torna difícil: há `\\frac`, então
    a expressão vai obrigatoriamente para o parser LaTeX, que é justamente
    quem estilhaça o identificador."""
    e = parse(r"E_{cin} = \frac{1}{2} m v_{max}^2")
    assert e is not None
    assert {s.name for s in e.free_symbols} == {"E_cin", "m", "v_max"}
