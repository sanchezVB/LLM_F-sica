"""Suíte golden dos verificadores dimensional, de limites e de invariantes.

Mesma disciplina do `test_verifier_bus.py`: casos de Física congelados, nas
três categorias — CORRETO, INCORRETO e INDECIDÍVEL. A terceira é a que dá
trabalho e a que protege o RLVR: sem ela, um verificador que chuta passa no CI.

Cada caso aqui é uma afirmação sobre Física, não sobre a implementação. Se um
deles quebrar, a pergunta é "a Física mudou?" — e a resposta sendo não, o
código regrediu.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.verify.bus import Claim, Verdict, VerifierBus  # noqa: E402
from phifm.verify.conservation import (  # noqa: E402
    PARTICULAS,
    ConservationVerifier,
    Numeros,
    _somar,
)
from phifm.verify.dimensional import (  # noqa: E402
    ENERGIA,
    Dim,
    DimensionalVerifier,
    dimensao,
)
from phifm.verify.limits import REDUCOES, LimitsVerifier  # noqa: E402
from phifm.verify.numeric import NumericVerifier  # noqa: E402
from phifm.verify.symbolic import SymbolicVerifier  # noqa: E402

SI = {"unit_system": "SI"}


@pytest.fixture(scope="module")
def dim() -> DimensionalVerifier:
    return DimensionalVerifier()


@pytest.fixture(scope="module")
def lim() -> LimitsVerifier:
    return LimitsVerifier()


@pytest.fixture(scope="module")
def cons() -> ConservationVerifier:
    return ConservationVerifier()


@pytest.fixture(scope="module")
def barramento_completo() -> VerifierBus:
    """Os seis verificadores juntos — é assim que o RLVR vai usar."""
    return VerifierBus([
        SymbolicVerifier(), NumericVerifier(), DimensionalVerifier(),
        LimitsVerifier(), ConservationVerifier(),
    ])


# ═══════════════════════════════════════════════════════════════════════════
# DIMENSIONAL
# ═══════════════════════════════════════════════════════════════════════════

DIM_CORRETOS = [
    ("2ª lei de Newton",     "F",              "m*a"),
    ("energia cinética",     "m*v**2/2",       "F*d"),
    ("queda livre",          "v**2",           "2*g*d"),
    ("período do pêndulo",   "t",              "2*pi*sqrt(d/g)"),
    ("Planck-Einstein",      "h*f",            "m*c**2"),
    ("momento",              "m*v",            "F*t"),
    ("potência",             "F*v",            "m*v**2/t"),
    ("comprimento de Planck", "sqrt(hbar*G/c**3)", "d"),
]


@pytest.mark.parametrize("nome,lhs,rhs", DIM_CORRETOS, ids=[c[0] for c in DIM_CORRETOS])
def test_dimensional_aprova_fisica_correta(dim, nome, lhs, rhs):
    r = dim.verify(Claim(lhs=lhs, rhs=rhs, context=SI))
    assert r.verdict is Verdict.PASS, f"{nome}: {r.verdict.value} — {r.evidence}"


DIM_INCORRETOS = [
    # O caso literal do DOC-10 §3.2.
    ("F = m a² (DOC-10)",  "F",         "m*a**2"),
    ("energia sem quadrado", "m*v**2",  "m*v"),
    ("força como energia",   "F",       "m*v**2"),
    ("tempo como comprimento", "t",     "d"),
    ("g com expoente errado", "v**2",   "2*g*d**2"),
    ("Planck sem c²",        "h*f",     "m*c"),
]


@pytest.mark.parametrize("nome,lhs,rhs", DIM_INCORRETOS, ids=[c[0] for c in DIM_INCORRETOS])
def test_dimensional_reprova_incoerencia(dim, nome, lhs, rhs):
    r = dim.verify(Claim(lhs=lhs, rhs=rhs, context=SI))
    assert r.verdict is Verdict.FAIL, f"{nome}: {r.verdict.value} — {r.evidence}"
    assert r.counterexample, "FAIL dimensional sem contraexemplo não ensina nada"


def test_soma_incoerente_e_reprovada_sem_gabarito(dim):
    """`½mv² + m·a` não precisa de gabarito para estar errado."""
    r = dim.verify(Claim(lhs="m*v**2/2 + m*a", context=SI))
    assert r.verdict is Verdict.FAIL
    assert "incoerente" in r.evidence


def test_soma_coerente_e_aprovada_sem_gabarito(dim):
    r = dim.verify(Claim(lhs="m*v**2/2 + m*g*d", context=SI))
    assert r.verdict is Verdict.PASS


def test_argumento_de_funcao_precisa_ser_adimensional(dim):
    """`sin(t)` com t em segundos é erro de Física, não de notação."""
    r = dim.verify(Claim(lhs="sin(t)", rhs="1", context=SI))
    assert r.verdict is Verdict.FAIL
    r_ok = dim.verify(Claim(lhs="sin(omega*t)", rhs="1", context=SI))
    assert r_ok.verdict is Verdict.PASS


# ── os INDECIDÍVEIS, que é onde um verificador ingênuo mente ──────────────

def test_sem_sistema_declarado_nao_chuta(dim):
    """DOC-10 §3.2: `physics.conventions` nulo → INCONCLUSIVE.

    Assumir SI num documento em unidades naturais reprovaria `E = m`, que
    está certo lá.
    """
    r = dim.verify(Claim(lhs="F", rhs="m*a"))
    assert r.verdict is Verdict.INCONCLUSIVE
    assert "não declarado" in r.evidence


@pytest.mark.parametrize("simbolo", ["E", "k", "T", "L", "p", "P", "Q"])
def test_simbolo_ambiguo_nunca_e_resolvido_sozinho(dim, simbolo):
    """`E` pode ser energia, campo elétrico ou módulo de Young. Escolher a
    leitura mais comum é adivinhar, e adivinhar em RLVR vira gradiente."""
    r = dim.verify(Claim(lhs=simbolo, rhs="m*a", context=SI))
    assert r.verdict is Verdict.INCONCLUSIVE, f"{simbolo} foi resolvido por conta própria"
    assert "ambíguo" in r.evidence or "consistente se" in r.evidence


def test_I_e_unidade_imaginaria_ate_ser_declarado(dim):
    """Decisão consciente, documentada em `symbolic._FISICA`: `I` continua a
    unidade imaginária porque onda plana `exp(I*k*x)` é comum. Corrente
    elétrica se obtém declarando o símbolo, o que tem precedência."""
    livre = dim.verify(Claim(lhs="I", rhs="m*a", context=SI))
    assert livre.verdict is not Verdict.INCONCLUSIVE  # sp.I é número, é adimensional

    ctx = {**SI, "dimensions": {"I": "corrente"}}
    declarado = dim.verify(Claim(lhs="q", rhs="I*t", context=ctx))
    assert declarado.verdict is Verdict.PASS, declarado.evidence


def test_declaracao_resolve_o_ambiguo(dim):
    """Nível 1 da cascata: declarado, deixa de ser ambíguo."""
    ctx = {**SI, "dimensions": {"E": "energia"}}
    assert dim.verify(Claim(lhs="E", rhs="m*c**2", context=ctx)).verdict is Verdict.PASS
    assert dim.verify(Claim(lhs="E", rhs="m*c", context=ctx)).verdict is Verdict.FAIL


def test_declaracao_por_expoentes_tambem_vale(dim):
    ctx = {**SI, "dimensions": {"E_campo": {"M": 1, "L": 1, "T": -3, "I": -1}}}
    r = dim.verify(Claim(lhs="q*E_campo", rhs="F", context=ctx))
    assert r.verdict is Verdict.PASS


def test_gaussiano_admite_que_nao_sabe(dim):
    r = dim.verify(Claim(lhs="F", rhs="q*q/r**2", context={"unit_system": "gaussian"}))
    assert r.verdict is Verdict.INCONCLUSIVE
    assert "gaussiano" in r.evidence


# ── unidades naturais: o caso degenerado do DOC-10 §3.2 ──────────────────

def test_unidades_naturais_usam_dimensao_de_massa(dim):
    """Com ℏ = c = 1, `E = m` está CERTO — e em SI estaria errado.

    É o teste que prova que a projeção existe: o mesmo par de expressões
    recebe vereditos opostos nos dois sistemas, e os dois estão certos.
    """
    claim_nat = Claim(lhs="m", rhs="m*v**2", context={"unit_system": "natural"})
    assert dim.verify(claim_nat).verdict is Verdict.PASS

    claim_si = Claim(lhs="m", rhs="m*v**2", context=SI)
    assert dim.verify(claim_si).verdict is Verdict.FAIL


def test_unidades_naturais_ainda_reprovam_potencia_de_energia_errada(dim):
    """A projeção afrouxa a checagem, não a desliga — era o risco do §3.2."""
    r = dim.verify(Claim(lhs="m", rhs="m**2", context={"unit_system": "natural"}))
    assert r.verdict is Verdict.FAIL
    assert "E^1" in r.evidence and "E^2" in r.evidence


def test_unidades_naturais_tem_confianca_menor_que_si(dim):
    nat = dim.verify(Claim(lhs="m*v", rhs="m", context={"unit_system": "natural"}))
    si = dim.verify(Claim(lhs="F", rhs="m*a", context=SI))
    assert nat.verdict is si.verdict is Verdict.PASS
    assert nat.confidence < si.confidence, "checagem escalar não pode ter a mesma confiança"


def test_algebra_de_dimensoes():
    assert (ENERGIA / ENERGIA).adimensional
    assert ENERGIA.massa == 1, "energia é E¹ em unidades naturais"
    assert Dim((0, 1, -1, 0, 0, 0, 0)).massa == 0, "velocidade é E⁰"
    assert dimensao(sp.sympify("m*c**2"), {"m": Dim((1, 0, 0, 0, 0, 0, 0)),
                                           "c": Dim((0, 1, -1, 0, 0, 0, 0))}).exp == ENERGIA.exp


# ═══════════════════════════════════════════════════════════════════════════
# LIMITES
# ═══════════════════════════════════════════════════════════════════════════

def test_energia_cinetica_relativistica_reduz_a_newtoniana(lim):
    """O caso central do DOC-10 §3.4, e o que exige o modo `dominante`:
    o limite ESTRITO quando v → 0 é zero, não ½mv²."""
    r = lim.verify(Claim(
        lhs="m*c**2/sqrt(1-v**2/c**2) - m*c**2",
        context={"limit": {"name": "v/c->0", "expected": "m*v**2/2"}},
    ))
    assert r.verdict is Verdict.PASS, r.evidence
    assert "termo dominante" in r.evidence


def test_energia_relativistica_errada_e_reprovada(lim):
    """Fórmula plausível e errada: fator 2 no lugar errado."""
    r = lim.verify(Claim(
        lhs="m*c**2/sqrt(1-v**2/(2*c**2)) - m*c**2",
        context={"limit": {"name": "v/c->0", "expected": "m*v**2/2"}},
    ))
    assert r.verdict is Verdict.FAIL, r.evidence
    assert r.counterexample


def test_momento_relativistico_reduz_a_mv(lim):
    r = lim.verify(Claim(
        lhs="m*v/sqrt(1-v**2/c**2)",
        context={"limit": {"name": "v/c->0", "expected": "m*v"}},
    ))
    assert r.verdict is Verdict.PASS, r.evidence


def test_schwarzschild_vira_espaco_plano(lim):
    r = lim.verify(Claim(
        lhs="1 - 2*G*M/(r*c**2)",
        context={"limit": {"name": "r->inf", "expected": "1"}},
    ))
    assert r.verdict is Verdict.PASS, r.evidence


def test_bose_einstein_vira_boltzmann_em_alta_temperatura(lim):
    """n = 1/(exp(E/kT) − 1) → kT/E quando T → ∞."""
    r = lim.verify(Claim(
        lhs="1/(exp(eps/(k*T))-1)",
        context={"limit": {"var": "T", "to": "inf", "mode": "dominante",
                           "expected": "k*T/eps"}},
    ))
    assert r.verdict is Verdict.PASS, r.evidence


def test_limite_que_nao_reduz_e_reprovado(lim):
    r = lim.verify(Claim(
        lhs="1 - 2*G*M/(r*c**2)",
        context={"limit": {"name": "r->inf", "expected": "0"}},
    ))
    assert r.verdict is Verdict.FAIL, r.evidence


def test_expressao_sem_a_variavel_nao_e_aprovada(lim):
    """Aprovar aqui seria dar PASS a uma redução que nunca foi testada —
    caminho de reward hacking por omissão."""
    r = lim.verify(Claim(
        lhs="m*a", context={"limit": {"var": "v", "to": 0, "expected": "m*a"}}
    ))
    assert r.verdict is Verdict.INCONCLUSIVE
    assert "não contém" in r.evidence


def test_reducao_desconhecida_nao_e_inventada(lim):
    r = lim.verify(Claim(lhs="x", context={"limit": {"name": "q->42", "expected": "x"}}))
    assert r.verdict is Verdict.INCONCLUSIVE
    assert "desconhecida" in r.evidence


def test_biblioteca_cobre_a_tabela_do_doc10():
    """As oito reduções canônicas do DOC-10 §3.4 estão todas presentes."""
    assert len(REDUCOES) == 8
    for nome, r in REDUCOES.items():
        assert r["mode"] in ("limite", "dominante", "auto"), nome
        assert r["de"] and r["para"], nome


def test_nao_aplicavel_sem_contexto_de_limite(lim):
    assert not lim.applicable(Claim(lhs="m*v**2/2", rhs="0.5*m*v**2"))


# ═══════════════════════════════════════════════════════════════════════════
# INVARIANTES
# ═══════════════════════════════════════════════════════════════════════════

def test_decaimento_beta_conserva(cons):
    """n → p + e⁻ + ν̄ₑ. Carga 0→0, bariônico 1→1, leptônico 0→0."""
    r = cons.verify(Claim(lhs="", context={
        "reaction": {"before": ["n"], "after": ["p", "e-", "nu_e_bar"]}
    }))
    assert r.verdict is Verdict.PASS, r.evidence


def test_decaimento_beta_sem_antineutrino_viola_leptonico(cons):
    """O caso barato e decisivo: `n → p + e⁻` conserva carga e número
    bariônico, e viola número leptônico. Nenhum modelo nuclear necessário."""
    r = cons.verify(Claim(lhs="", context={
        "reaction": {"before": ["n"], "after": ["p", "e-"]}
    }))
    assert r.verdict is Verdict.FAIL
    assert "leptônico" in r.evidence
    assert "carga" not in r.evidence, "carga se conserva aqui; não pode ser acusada"


def test_aniquilacao_de_par(cons):
    r = cons.verify(Claim(lhs="", context={
        "reaction": {"before": ["e-", "e+"], "after": ["2gamma"]}
    }))
    assert r.verdict is Verdict.PASS, r.evidence


def test_proton_decaindo_em_positron_viola_barionico(cons):
    r = cons.verify(Claim(lhs="", context={
        "reaction": {"before": ["p"], "after": ["e+", "gamma"]}
    }))
    assert r.verdict is Verdict.FAIL
    assert "bariônico" in r.evidence


def test_decaimento_do_muon(cons):
    r = cons.verify(Claim(lhs="", context={
        "reaction": {"before": ["mu-"], "after": ["e-", "nu_mu", "nu_e_bar"]}
    }))
    assert r.verdict is Verdict.PASS, r.evidence


def test_decaimento_nuclear_por_notacao_Z_A(cons):
    """Carbono-14 → Nitrogênio-14 + e⁻ + ν̄ₑ, escrito como `Z:A`."""
    r = cons.verify(Claim(lhs="", context={
        "reaction": {"before": ["6:14"], "after": ["7:14", "e-", "nu_e_bar"]}
    }))
    assert r.verdict is Verdict.PASS, r.evidence


def test_particula_fora_da_tabela_nao_e_chutada(cons):
    """Tabela incompleta por escolha: chutar números quânticos de hádron
    exótico daria FAIL espúrio em Física correta."""
    r = cons.verify(Claim(lhs="", context={
        "reaction": {"before": ["Theta+"], "after": ["p", "K0"]}
    }))
    assert r.verdict is Verdict.INCONCLUSIVE
    assert "fora da tabela" in r.evidence


def test_numeros_quanticos_somam_como_grupo():
    n, faltando = _somar(["p", "e-", "nu_e_bar"])
    assert not faltando
    assert n.Q == 0 and n.B == 1 and n.Le == 0
    assert (PARTICULAS["e-"] + (-PARTICULAS["e-"])) == Numeros()


# ── conservação contínua ─────────────────────────────────────────────────

def test_energia_do_oscilador_harmonico_se_conserva(cons):
    """E = ½mv² + ½kx² com x = A·cos(ωt), v = −Aω·sen(ωt) e ω² = k/m."""
    r = cons.verify(Claim(lhs="", context={"conserved": {
        "quantity": "energia", "var": "t",
        "expr": "m*(A*sqrt(k/m)*sin(sqrt(k/m)*t))**2/2 + k*(A*cos(sqrt(k/m)*t))**2/2",
    }}))
    assert r.verdict is Verdict.PASS, r.evidence


def test_solucao_com_amortecimento_nao_conserva_energia(cons):
    """Com atrito a energia cai — e o verificador tem de dizer isso."""
    r = cons.verify(Claim(lhs="", context={"conserved": {
        "quantity": "energia", "var": "t",
        "expr": "exp(-t)*(m*(A*sin(t))**2/2 + k*(A*cos(t))**2/2)",
    }}))
    assert r.verdict is Verdict.FAIL, r.evidence
    assert r.counterexample


def test_substituicao_da_solucao_alegada(cons):
    """Modo `solution`: o invariante é genérico e a solução é substituída."""
    r = cons.verify(Claim(lhs="", context={"conserved": {
        "quantity": "momento angular", "var": "t", "expr": "m*r_t*v_t",
        "solution": {"r_t": "R", "v_t": "L0/(m*R)"},
    }}))
    assert r.verdict is Verdict.PASS, r.evidence


def test_unitaridade_de_gaussiana_normalizada(cons):
    r = cons.verify(Claim(lhs="", context={"conserved": {
        "quantity": "unitaridade", "var": "x",
        "expr": "(1/pi)**(1/4)*exp(-x**2/2)",
    }}))
    assert r.verdict is Verdict.PASS, r.evidence


def test_unitaridade_falha_em_estado_nao_normalizado(cons):
    r = cons.verify(Claim(lhs="", context={"conserved": {
        "quantity": "unitaridade", "var": "x", "expr": "exp(-x**2/2)",
    }}))
    assert r.verdict is Verdict.FAIL, r.evidence


def test_nao_aplicavel_sem_contexto(cons):
    assert not cons.applicable(Claim(lhs="m*v**2/2", rhs="0.5*m*v**2"))


# ═══════════════════════════════════════════════════════════════════════════
# BARRAMENTO COM OS CINCO JUNTOS
# ═══════════════════════════════════════════════════════════════════════════

def test_dimensional_nao_atrapalha_o_simbolico(barramento_completo):
    """Verificador novo no barramento não pode mudar veredito de caso já
    coberto — é a garantia que o DOC-10 §5 exige ao compor."""
    r = barramento_completo.check(Claim(lhs="m*v**2/2", rhs="0.5*m*v**2", context=SI))
    assert r.verdict is Verdict.PASS, r.evidence


def test_dimensional_pega_o_que_o_simbolico_deixa_passar(barramento_completo):
    """`F = m·a²` é algebricamente uma equação qualquer; só a dimensão a
    reprova. É o argumento do §5: cada verificador cobre um flanco."""
    r = barramento_completo.check(Claim(lhs="F", rhs="m*a**2", context=SI))
    assert r.verdict is Verdict.FAIL, r.evidence


def test_discordancia_entre_verificadores_vira_erro_nao_voto(barramento_completo):
    """Se o dimensional aprova e o simbólico reprova, o barramento não vota.

    `m·v²` e `F·d` têm a mesma dimensão e NÃO são a mesma expressão — o par
    perfeito para exercitar a regra de discordância do DOC-10 §4.
    """
    r = barramento_completo.check(Claim(lhs="m*v**2", rhs="F*d", context=SI))
    assert r.verdict is Verdict.ERROR
    assert "DISCORDÂNCIA" in r.evidence
    assert r.reward == 0.0, "discordância não pode virar gradiente"
