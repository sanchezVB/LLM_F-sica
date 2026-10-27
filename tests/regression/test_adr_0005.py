"""Regressão das peças do ADR-0005: contrato, leituras, tipos, motivos.

A tabela da recompensa está congelada em `tests/golden/test_recompensa_rlvr.py`.
Aqui ficam os defeitos que cada peça fechou, um teste por defeito, achados na
revisão adversarial do desenho em 2026-10-23.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.verify import (  # noqa: E402
    Claim,
    Motivo,
    NumericVerifier,
    Papel,
    SymbolicVerifier,
    Verdict,
    barramento_padrao,
    ler_gabarito,
    ler_resposta,
)
from phifm.verify.dimensional import Indeterminada, dimensao  # noqa: E402
from phifm.verify.symbolic import parse, traduzir_unicode  # noqa: E402

# ═══ contrato de resposta ═════════════════════════════════════════════════
CONTRATO = [
    (r"bla \boxed{1} e depois \boxed{x^2}", "x^2"),           # a última caixa
    (r"\boxed{\boxed{x}}", "x"),                              # caixa aninhada lê a de dentro
    (r"\boxed{$x$.}", "x"),                                   # envoltório e ponto final
    (r"\boxed{\(x\),}", "x"),
    (r"\boxed{E = m c^2}", "m c^2"),                          # grandeza solta =
    (r"\boxed{E_{k} = m c^2}", "m c^2"),
    (r"\boxed{v_\text{max} = a t}", "a t"),
    (r"\boxed{\omega' = \sqrt{k/m}}", r"\sqrt{k/m}"),
    (r"\boxed{\approx 3.0}", "3.0"),                          # aproximado no início
    (r"\boxed{K \approx \frac{1}{2} m v^2}", r"\frac{1}{2} m v^2"),
    (r"\boxed{K = A = B}", "K = A = B"),                      # cadeia: fica relação
    (r"\boxed{K(v) = v^2}", "K(v) = v^2"),                    # função não é grandeza solta
    (r"\boxed{x + 1 = 2}", "x + 1 = 2"),                      # lado esquerdo não é grandeza
    (r"\boxed{\sum_{n=1}^{N} n}", r"\sum_{n=1}^{N} n"),       # o = do subscrito não divide
]


@pytest.mark.parametrize("texto,esperado", CONTRATO, ids=[c[0] for c in CONTRATO])
def test_contrato_de_resposta(texto, esperado):
    assert ler_resposta(texto) == esperado


def test_sem_caixa_nao_ha_resposta_mas_ha_gabarito():
    assert ler_resposta("x^2") is None
    assert ler_gabarito("E = x^2") == "x^2"


# ═══ Unicode ══════════════════════════════════════════════════════════════
@pytest.mark.parametrize("texto,esperado", [
    ("½mv²", sp.Symbol("m") * sp.Symbol("v") ** 2 / 2),        # `mv` separa: rota LaTeX
    ("2½", sp.Rational(5, 2)),                                  # misto, não 2·½
    ("ω₀ t", sp.Symbol("omega_0") * sp.Symbol("t")),
    ("ħω", sp.Symbol("hbar") * sp.Symbol("omega")),
])
def test_unicode_vira_latex(texto, esperado):
    assert sp.simplify(parse(texto) - esperado) == 0, parse(texto)


@pytest.mark.parametrize("texto", ["é", "x√2", "v·é"])
def test_o_que_sobra_fora_do_ascii_e_recusado(texto):
    assert traduzir_unicode(texto) is None
    assert parse(texto) is None


# ═══ números absurdos, variantes, marcadores ══════════════════════════════
@pytest.mark.parametrize("texto", [r"10^{10^{8}}", "2**3**4", r"x^{100000}", "1" * 60])
def test_numero_absurdo_e_recusado_antes_de_avaliar(texto):
    assert parse(texto) is None


def test_numero_de_fisica_comum_continua_lido():
    assert parse(r"6.02\times10^{23}") is not None


def test_variante_grega_casa_com_a_declaracao():
    """`\\varepsilon_0` declarado no contexto casa com o `epsilon_0` lido."""
    ctx = {"unit_system": "SI", "dimensions": {"varepsilon_0": {"M": -1, "L": -3, "T": 4, "I": 2}}}
    r = barramento_padrao().check(
        Claim(lhs=r"\varepsilon_{0}", rhs=r"\epsilon_0", context=ctx))
    assert r.verdict is Verdict.PASS, r.evidence


def test_subscrito_de_um_token_depois_de_macro_e_o_mesmo_simbolo():
    """`\\epsilon_0` saía `epsilon_{0}` do ANTLR, e `\\epsilon_{0}`, `epsilon_0`."""
    assert parse(r"\epsilon_0") == parse(r"\epsilon_{0}") == sp.Symbol("epsilon_0")
    assert parse(r"\omega_c t") == sp.Symbol("omega_c") * sp.Symbol("t")


def test_marcador_de_subscrito_e_desfeito_em_nome_de_funcao():
    assert parse(r"E_{cin}(x)").atoms(sp.Function) == {sp.Function("E_cin")(sp.Symbol("x"))}


# ═══ tipos: equação, tupla ════════════════════════════════════════════════
@pytest.mark.parametrize("verificador", [SymbolicVerifier, NumericVerifier])
def test_equacao_contra_equacao_nao_passa(verificador):
    """O numérico comparava o booleano de dois `Eq`: qualquer par de equações passava."""
    # LaTeX: na sintaxe do SymPy o `=` nem parseia.
    r = verificador().verify(Claim(lhs=r"x^{2} + y^{2} = 2 R^{2}", rhs=r"x^{2} + y^{2} = R^{2}"))
    assert r.verdict is Verdict.INCONCLUSIVE and r.motivo is Motivo.GABARITO_INCOMPARAVEL


@pytest.mark.parametrize("verificador", [SymbolicVerifier, NumericVerifier])
def test_equacao_contra_expressao_e_incomparavel(verificador):
    r = verificador().verify(Claim(lhs=r"E = \frac{1}{2} m v^{2}", rhs=r"\frac{1}{2} m v^{2}"))
    assert r.verdict is Verdict.INCONCLUSIVE and r.motivo is Motivo.RESPOSTA_INCOMPARAVEL


# ═══ operadores ═══════════════════════════════════════════════════════════
def test_operador_so_na_resposta_nao_desliga_o_numerico():
    """O numérico sumia do vetor quando a resposta tinha `\\sigma`."""
    rs = barramento_padrao().run(Claim(lhs=r"m v^{2} \sigma^{0}", rhs=r"\frac{1}{2} m v^{2}"))
    numerico = [r for r in rs if r.verifier_id.startswith("numeric")]
    assert numerico and numerico[0].motivo is Motivo.OPERADORES_NA_RESPOSTA


def test_operadores_no_item_nao_passam_pela_comparacao_estrutural():
    """Na rota do `parse_expr`, `A*B` e `B*A` saem na mesma ordem canônica."""
    r = SymbolicVerifier().verify(Claim(lhs="A*B", rhs="B*A", context={"noncommutative": True}))
    assert r.verdict is not Verdict.PASS


# ═══ numérico e dimensional ═══════════════════════════════════════════════
def test_gabarito_zero_ainda_aceita_ruido_de_cancelamento():
    r = NumericVerifier().verify(Claim(lhs="sin(x)**2 + cos(x)**2 - 1", rhs="0"))
    assert r.verdict is Verdict.PASS, r.evidence


def test_resposta_que_nao_avalia_e_da_resposta():
    r = NumericVerifier().verify(Claim(lhs=r"\frac{m v^{2}}{0}", rhs=r"\frac{1}{2} m v^{2}"))
    assert r.verdict is not Verdict.PASS
    if r.verdict is Verdict.INCONCLUSIVE:
        assert r.motivo is Motivo.RESPOSTA_INDEFINIDA


def test_modulo_tem_a_dimensao_do_argumento():
    from phifm.verify.dimensional import VELOCIDADE

    assert dimensao(sp.Abs(sp.Symbol("v")), {"v": VELOCIDADE}) == VELOCIDADE


def test_funcao_indefinida_nao_tem_dimensao():
    with pytest.raises(Indeterminada):
        dimensao(sp.Function("f")(1), {})


# ═══ barramento ═══════════════════════════════════════════════════════════
def test_run_preserva_motivo_e_papel():
    rs = barramento_padrao().run(Claim(lhs=r"m v^{2}\,\text{J}", rhs=r"\frac{1}{2} m v^{2}"))
    simbolico = next(r for r in rs if r.verifier_id.startswith("symbolic"))
    assert simbolico.papel is Papel.EQUIVALENCIA
    assert simbolico.motivo is Motivo.RESPOSTA_ILEGIVEL


def test_tipo_desconhecido_e_erro_e_nao_equivalencia():
    """`"relação"` com cedilha não pode virar equivalência em silêncio."""
    rs = barramento_padrao().run(Claim(lhs="F", rhs="m*a", context={"tipo": "relação"}))
    assert [r.verdict for r in rs] == [Verdict.ERROR]


def test_contradicao_e_erro_com_motivo_proprio():
    from phifm.verify import VerificationResult

    c = barramento_padrao().combine([
        VerificationResult(Verdict.PASS, "eq", 1.0, ""),
        VerificationResult(Verdict.FAIL, "cond", 0.98, "", papel=Papel.CONDICAO),
    ])
    assert c.verdict is Verdict.ERROR and c.motivo is Motivo.CONTRADICAO
    assert "CONTRADIÇÃO" in c.evidence


def test_simbolico_e_deterministico():
    """`equals` sorteia com o RNG global do SymPy: sem semente, o mesmo Claim
    dava INCONCLUSIVE numa chamada e FAIL na seguinte."""
    c = Claim(lhs=r"-\sqrt{(q E)^{2}}", rhs="-q E")
    assert len({SymbolicVerifier().verify(c).verdict for _ in range(4)}) == 1


ABSTENCOES = [
    Claim(lhs=r"m v^{2}\,\text{J}", rhs=r"\frac{1}{2} m v^{2}"),
    Claim(lhs=r"\frac{m v^{2}}{0}", rhs=r"\frac{1}{2} m v^{2}"),
    Claim(lhs=r"m v^{2} f(1)", rhs=r"\frac{1}{2} m v^{2}", context={"unit_system": "SI"}),
    Claim(lhs=r"x^{2} + y^{2} = 2", rhs=r"x^{2} + y^{2} = 1"),
    Claim(lhs=r"\hat{A}\hat{B}", rhs=r"\hat{B}\hat{A}", context={"noncommutative": True}),
    Claim(lhs="F", rhs="m*a", context={"unit_system": "gaussian"}),
    Claim(lhs="m*v**2", rhs="m*v**2/2", context={"limit": {"var": "c", "to": "oo"}}),
]


@pytest.mark.parametrize("claim", ABSTENCOES, ids=[c.lhs for c in ABSTENCOES])
def test_toda_abstencao_tem_motivo(claim):
    """Uma abstenção sem motivo volta a ser neutra, e o atalho reabre em silêncio."""
    for r in barramento_padrao().run(claim):
        if not r.verdict.is_decisive:
            assert r.motivo not in (None, Motivo.SEM_MOTIVO), f"{r.verifier_id}: {r.evidence}"
