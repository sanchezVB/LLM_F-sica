"""A recompensa do RLVR (ADR-0005 §4.6), congelada.

Golden, como o resto do barramento (DOC-10 §5): mudar o que está aqui muda o
gradiente do RLVR, e só se muda por decisão do dono.

A regra que tudo isto protege, em uma linha: **no GRPO só o sinal de
(atalho − errada) conta.** Num grupo de sete FAIL, qualquer amostra que valha
mais que um FAIL recebe a vantagem de uma resposta certa. Por isso toda
abstenção que a RESPOSTA consegue provocar vale exatamente −1, e o neutro do
DOC-10 §2.1 é tirar a amostra do grupo (`None`), não dar 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.verify import (  # noqa: E402
    Motivo,
    Papel,
    Verdict,
    VerificationResult,
    admitir,
    avaliar,
    recompensa,
)
from phifm.verify.bus import MOTIVOS_DA_RESPOSTA, MOTIVOS_DO_ITEM  # noqa: E402

EQ, COND = Papel.EQUIVALENCIA, Papel.CONDICAO


def _r(verdict, papel=EQ, conf=1.0, motivo=None) -> VerificationResult:
    return VerificationResult(verdict, "v", conf, "", motivo=motivo, papel=papel)


PASS, FAIL, INC, ERR = Verdict.PASS, Verdict.FAIL, Verdict.INCONCLUSIVE, Verdict.ERROR


# ═══ 1. A tabela, em ordem ════════════════════════════════════════════════
TABELA = [
    ("vazio", [], None),
    ("discordância entre equivalências", [_r(PASS), _r(FAIL)], None),
    ("contradição: equivalência PASS e condição FAIL",
     [_r(PASS), _r(FAIL, COND, 0.98)], None),
    ("FAIL de equivalência vale −1 cheio", [_r(FAIL, conf=0.95)], -1.0),
    ("FAIL de condição decide, com a equivalência em tempo",
     [_r(INC, motivo=Motivo.TEMPO), _r(FAIL, COND, 0.98)], -1.0),
    ("FAIL de condição de baixa confiança não decide",
     [_r(INC, motivo=Motivo.INDECIDIVEL), _r(FAIL, COND, 0.85)], None),
    ("PASS de equivalência vence abstenção de condição",
     [_r(PASS, conf=0.9), _r(INC, COND, motivo=Motivo.SEM_CONVENCAO)], 1.0),
    ("PASS de equivalência vence verificador quebrado",
     [_r(PASS), _r(ERR, COND, 0.0, Motivo.EXCECAO)], 1.0),
    ("condição sozinha não prova equivalência",
     [_r(INC, motivo=Motivo.INDECIDIVEL), _r(PASS, COND, 0.95)], None),
    ("resposta ilegível vence o sem_convencao do dimensional",
     [_r(INC, motivo=Motivo.RESPOSTA_ILEGIVEL), _r(INC, COND, motivo=Motivo.SEM_CONVENCAO)], -1.0),
    ("gabarito ilegível tira do grupo",
     [_r(INC, motivo=Motivo.GABARITO_ILEGIVEL), _r(INC, motivo=Motivo.RESPOSTA_ILEGIVEL)], None),
    ("tempo do problema tira do grupo", [_r(INC, motivo=Motivo.TEMPO)], None),
    ("abstenção sem motivo tira do grupo", [_r(INC, motivo=Motivo.SEM_MOTIVO)], None),
]


@pytest.mark.parametrize("nome,resultados,esperado", TABELA, ids=[t[0] for t in TABELA])
def test_tabela_da_recompensa(nome, resultados, esperado):
    assert recompensa(resultados).valor == esperado


@pytest.mark.parametrize("motivo", sorted(MOTIVOS_DA_RESPOSTA))
def test_toda_abstencao_causada_pela_resposta_vale_um_fail(motivo):
    """De qualquer papel: o dimensional que não lê a resposta também conta."""
    for papel in (EQ, COND):
        assert recompensa([_r(INC, papel, 0.0, motivo)]).valor == -1.0


@pytest.mark.parametrize("motivo", sorted(MOTIVOS_DO_ITEM))
def test_abstencao_do_item_na_equivalencia_tira_do_grupo_com_alerta(motivo):
    r = recompensa([_r(INC, EQ, 0.0, motivo)])
    assert r.valor is None and r.alerta, "o item devia ter ficado fora na admissão"


def test_fail_nao_multiplica_pela_confianca():
    """−0,95 contra −1 faria sete erradas a −0,95 ganharem vantagem cheia
    contra uma a −1 no GRPO: ruído de 0,05 vira gradiente."""
    assert recompensa([_r(FAIL, conf=0.95)]).valor == recompensa([_r(FAIL, conf=1.0)]).valor


# ═══ 2. De ponta a ponta: os atalhos do ADR-0005 §1 ═══════════════════════
GABARITO = r"\frac{1}{2} m v^{2}"
SI = {"unit_system": "SI", "dimensions": {"m": "massa", "v": "velocidade"}}

ATALHOS = [
    ("errada", r"\boxed{m v^{2}}"),
    ("com unidade", r"\boxed{m v^{2}\,\text{J}}"),
    ("com sobra no fim", r"\boxed{m v^{2} ]}"),
    ("com \\vec", r"\boxed{m \vec{v}^{2}}"),
    ("abstenção em \\text", r"\boxed{\text{nao sei}}"),
    ("sem caixa", "m v^2"),
    ("função com nome do gabarito", r"\boxed{m v^{2} m(1)}"),
    ("função indefinida", r"\boxed{m v^{2} f(1)}"),
    ("divisão por zero", r"\boxed{\frac{m v^{2}}{0}}"),
    ("zero com expoente negativo", r"\boxed{m v^{2} \cdot 0^{-m}}"),
    ("torre de expoentes", r"\boxed{m v^{2} \frac{10^{10^{4}}}{10^{10^{4}}}}"),
    ("operador só na resposta", r"\boxed{m v^{2} \sigma^{0}}"),
    ("tupla", r"\boxed{(m v^2, m v^2/2)}"),
    ("fator com a dimensão certa", r"\boxed{2 m v^{2}}"),
]


@pytest.mark.parametrize("contexto", [{}, SI], ids=["sem contexto", "SI"])
@pytest.mark.parametrize("nome,texto", ATALHOS, ids=[a[0] for a in ATALHOS])
def test_atalho_vale_exatamente_um_fail(nome, texto, contexto):
    assert avaliar(texto, GABARITO, contexto).valor == -1.0


CERTAS = [
    ("expressão", r"\boxed{\frac{m v^2}{2}}"),
    ("grandeza = expressão", r"\boxed{E = \frac{1}{2} m v^{2}}"),
    ("grandeza ≈ expressão", r"\boxed{K \approx \frac{1}{2} m v^{2}}"),
    ("entre cifrões, com ponto final", r"\boxed{$\frac{1}{2} m v^{2}$.}"),
    ("Unicode", r"\boxed{½mv²}"),
]


@pytest.mark.parametrize("contexto", [{}, SI], ids=["sem contexto", "SI"])
@pytest.mark.parametrize("nome,texto", CERTAS, ids=[c[0] for c in CERTAS])
def test_resposta_certa_vale_um(nome, texto, contexto):
    assert avaliar(texto, GABARITO, contexto).valor == 1.0


def test_zero_contra_gabarito_minusculo_e_reprovado():
    """Com o piso absoluto de 1 no numérico, `0` passava contra 1,87·10⁻⁶⁴ e a
    resposta errada saía do grupo por discordância."""
    assert avaliar(r"\boxed{0}", r"1.87 \times 10^{-64}").valor == -1.0


# ═══ 3. A admissão do item ════════════════════════════════════════════════
def test_item_comum_e_admitido():
    assert admitir(GABARITO, SI)[0]


@pytest.mark.parametrize("gabarito", [
    r"2 d \sin\theta = n \lambda",   # equação: o barramento não compara Eq com Eq
    r"\sigma T^{4}",                 # o regex de operadores pega \sigma
], ids=["equação", "operador"])
def test_item_que_o_numerico_nao_aprova_fica_fora(gabarito):
    assert not admitir(gabarito)[0]
