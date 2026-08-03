"""Suíte golden do barramento de verificação (DOC-10 §5).

Casos de Física **congelados**. Qualquer mudança de comportamento sobre eles
quebra o CI, mesmo que pareça melhoria — porque um bug aqui é global: o mesmo
código filtra dados, calcula recompensa de RLVR, corrige benchmark e checa
saída em serving (DOC-10 §1).

Três categorias, e a terceira é a que costuma faltar:

  1. sabidamente CORRETOS    → o verificador não pode gerar falso negativo
  2. sabidamente INCORRETOS  → não pode gerar falso positivo
  3. sabidamente INDECIDÍVEIS → tem de dizer INCONCLUSIVE em vez de CHUTAR

A terceira existe porque, em RL, o chute vira gradiente — e gradiente errado
com confiança alta é o pior insumo possível.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.verify.bus import REWARD, Claim, Verdict, VerificationResult, VerifierBus  # noqa: E402
from phifm.verify.numeric import NumericVerifier  # noqa: E402
from phifm.verify.symbolic import SymbolicVerifier  # noqa: E402


@pytest.fixture(scope="module")
def bus() -> VerifierBus:
    return VerifierBus([SymbolicVerifier(), NumericVerifier()])


# ═══ 1. CORRETOS — mesma Física, notação diferente ════════════════════════
CORRETOS = [
    ("pêndulo simples",        "2*pi*sqrt(L/g)",       "2*pi*(L/g)**(1/2)"),
    ("energia cinética",       "m*v**2/2",             "0.5*m*v**2"),
    ("identidade trig",        "sin(x)**2+cos(x)**2",  "1"),
    ("energia relativística",  "sqrt((p*c)**2+(m*c**2)**2)", "m*c**2*sqrt(1+(p/(m*c))**2)"),
    ("fator de Lorentz",       "1/sqrt(1-v**2/c**2)",  "c/sqrt(c**2-v**2)"),
    ("freq. do oscilador",     "sqrt(k/m)",            "sqrt(k)/sqrt(m)"),
    ("Coulomb reagrupado",     "q1*q2/(4*pi*eps*r**2)", "q1*q2*r**(-2)/(4*pi*eps)"),
    ("De Broglie",             "h/(m*v)",              "h/m/v"),
    ("dilatação do tempo",     "t0/sqrt(1-b**2)",      "t0*(1-b**2)**(-1/2)"),
    ("Stefan-Boltzmann",       "sig*A*T**4",           "A*sig*T*T*T*T"),
]


@pytest.mark.parametrize("nome,lhs,rhs", CORRETOS, ids=[c[0] for c in CORRETOS])
def test_correto_nao_e_reprovado(bus, nome, lhs, rhs):
    """Falso negativo em RLVR pune resposta certa — o pior erro possível.

    Foi assim que dois bugs apareceram em 2026-08-03: amostragem no plano
    complexo (corte de ramo) e entradas em float64 com tolerância de 1e-30.
    Ambos reprovavam Física correta.
    """
    r = bus.check(Claim(lhs=lhs, rhs=rhs))
    assert r.verdict is Verdict.PASS, f"{nome}: {r.verdict.value} — {r.evidence}"


# ═══ 2. INCORRETOS — erros que um físico reconhece de imediato ════════════
INCORRETOS = [
    ("fator 2 perdido",     "2*pi*sqrt(L/g)",   "pi*sqrt(L/g)"),
    ("sinal invertido",     "-G*M*m/r",         "G*M*m/r"),
    ("expoente errado",     "G*M*m/r**2",       "G*M*m/r**3"),
    ("massa no lugar errado", "m*v**2/2",       "v**2/(2*m)"),
    ("raiz esquecida",      "sqrt(k/m)",        "k/m"),
    ("c em vez de c²",      "m*c**2",           "m*c"),
]


@pytest.mark.parametrize("nome,lhs,rhs", INCORRETOS, ids=[c[0] for c in INCORRETOS])
def test_incorreto_e_reprovado(bus, nome, lhs, rhs):
    r = bus.check(Claim(lhs=lhs, rhs=rhs))
    assert r.verdict is Verdict.FAIL, f"{nome}: {r.verdict.value} — {r.evidence}"


# ═══ 3. INDECIDÍVEIS — tem de se abster, não chutar ═══════════════════════
class TestIndecidiveis:
    def test_operadores_nao_comutativos(self, bus):
        """O SymPy assume comutatividade e simplifica ``AB−BA`` para 0.

        Isso é FALSO para operadores quânticos, e toda a Mecânica Quântica
        está nessa distinção. Sem anotação de tipo confiável, o veredito
        correto é INCONCLUSIVE — nunca PASS.
        """
        r = bus.check(Claim(
            lhs=r"\hat{A}\hat{B} - \hat{B}\hat{A}", rhs="0",
            context={"noncommutative": True},
        ))
        assert r.verdict is not Verdict.PASS, (
            "aprovou comutação de operadores — erro de Física, não de software"
        )

    def test_expressao_ininteligivel_nao_vira_falha(self, bus):
        """Não parsear é INCONCLUSIVE. Tratar como FAIL puniria o modelo por
        uma limitação do nosso parser."""
        r = bus.check(Claim(lhs="}{ nao eh expressao ][", rhs="m*v**2/2"))
        assert r.verdict in (Verdict.INCONCLUSIVE, Verdict.ERROR)


# ═══ Reward hacking (DOC-09 §5.4) ════════════════════════════════════════
class TestRewardHacking:
    @pytest.mark.parametrize("trivial", ["0", "1"])
    def test_resposta_constante_trivial_e_reprovada(self, bus, trivial):
        r = bus.check(Claim(lhs=trivial, rhs="m*v**2/2"))
        assert r.verdict is Verdict.FAIL

    def test_constante_legitima_nao_e_confundida_com_degenerescencia(self, bus):
        """O contraponto: ``sin²+cos² = 1`` tem gabarito constante e ESTÁ certo.

        A checagem de degenerescência só pode rodar depois de estabelecida a
        não-equivalência. Aplicá-la antes reprovava esse caso — bug real
        encontrado em 2026-08-03.
        """
        r = bus.check(Claim(lhs="sin(x)**2+cos(x)**2", rhs="1"))
        assert r.verdict is Verdict.PASS


# ═══ Álgebra de resultados (DOC-10 §4) ═══════════════════════════════════
class TestAlgebraDeResultados:
    def test_inconclusive_mapeia_para_recompensa_neutra(self):
        """DOC-10 §2.1 — o ponto que decide se o RLVR funciona.

        Se INCONCLUSIVE virasse recompensa negativa, o modelo aprenderia a
        EVITAR problemas difíceis, que é o oposto do objetivo.
        """
        assert REWARD[Verdict.INCONCLUSIVE] == 0.0
        assert REWARD[Verdict.ERROR] == 0.0
        assert REWARD[Verdict.PASS] > 0 > REWARD[Verdict.FAIL]

    def test_discordancia_nunca_e_resolvida_por_voto(self, bus):
        """Se um verificador aprova e outro reprova, há bug em um dos dois.

        Voto majoritário esconderia exatamente o defeito que precisa ser
        investigado — e um bug no barramento é global.
        """
        combinado = bus.combine([
            VerificationResult(Verdict.PASS, "a", 1.0, ""),
            VerificationResult(Verdict.FAIL, "b", 1.0, ""),
            VerificationResult(Verdict.PASS, "c", 1.0, ""),
        ])
        assert combinado.verdict is Verdict.ERROR
        assert "DISCORDÂNCIA" in combinado.evidence

    def test_inconclusive_nao_envenena_a_conjuncao(self, bus):
        """Mistura de PASS e INCONCLUSIVE continua PASS, com confiança menor."""
        c = bus.combine([
            VerificationResult(Verdict.PASS, "a", 1.0, ""),
            VerificationResult(Verdict.INCONCLUSIVE, "b", 0.0, ""),
        ])
        assert c.verdict is Verdict.PASS
        assert c.confidence < 1.0

    def test_sem_verificador_aplicavel_e_inconclusive(self, bus):
        assert bus.combine([]).verdict is Verdict.INCONCLUSIVE

    def test_verificador_quebrado_nao_derruba_o_laco(self, bus):
        class Quebrado:
            id = "quebrado"
            def applicable(self, claim): return True
            def verify(self, claim): raise RuntimeError("boom")

        b = VerifierBus([Quebrado(), SymbolicVerifier(), NumericVerifier()])
        r = b.check(Claim(lhs="m*v**2/2", rhs="0.5*m*v**2"))
        assert r.verdict is Verdict.PASS, "um verificador quebrado invalidou o resultado"


def test_determinismo(bus):
    """Mesma entrada → mesma saída, sempre (DOC-10 §10).

    Sem isso, um resultado de benchmark não é reproduzível.
    """
    c = Claim(lhs="sqrt((p*c)**2+(m*c**2)**2)", rhs="m*c**2*sqrt(1+(p/(m*c))**2)")
    assert len({bus.check(c).verdict for _ in range(3)}) == 1
