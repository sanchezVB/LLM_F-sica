"""Barramento de verificação (DOC-10).

Uma única implementação servindo cinco consumidores — filtro de dados,
admissão de sintéticos, recompensa de RLVR, correção de benchmark e
auto-checagem em inferência. É o que torna estruturalmente impossível a
recompensa de treino divergir do corretor de avaliação (DOC-01 §1, P3).

`barramento_padrao()` é o ponto de entrada: quem precisa verificar Física
chama isso, não instancia verificadores à mão. Assim a composição fica num
lugar só, e um verificador novo entra para os cinco consumidores de uma vez.
"""

from __future__ import annotations

from phifm.verify.bus import (
    REWARD,
    Claim,
    Verdict,
    Verifier,
    VerificationResult,
    VerifierBus,
)
from phifm.verify.conservation import ConservationVerifier
from phifm.verify.dimensional import DimensionalVerifier
from phifm.verify.limits import LimitsVerifier
from phifm.verify.numeric import NumericVerifier
from phifm.verify.symbolic import SymbolicVerifier

__all__ = [
    "REWARD", "Claim", "Verdict", "Verifier", "VerificationResult", "VerifierBus",
    "ConservationVerifier", "DimensionalVerifier", "LimitsVerifier",
    "NumericVerifier", "SymbolicVerifier",
    "barramento_padrao",
]


def barramento_padrao() -> VerifierBus:
    """Os cinco verificadores implementados, na ordem do DOC-10 §3.

    Falta o sexto, `verify/sandbox` (§3.6): ele exige gVisor ou Firecracker, e
    `exec()` com builtins restritos está explicitamente descartado no documento
    como trivialmente evadível. Rodar código gerado por modelo sem esse
    isolamento seria pior que não rodar — então o barramento sai sem ele em vez
    de sair com um substituto inseguro.
    """
    return VerifierBus([
        SymbolicVerifier(),
        DimensionalVerifier(),
        NumericVerifier(),
        LimitsVerifier(),
        ConservationVerifier(),
    ])
