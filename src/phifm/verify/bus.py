"""Barramento de verificação — o protocolo (DOC-10 §2).

Uma única implementação servindo cinco consumidores: filtro de dados,
admissão de sintéticos, recompensa de RLVR, correção de benchmark e
auto-checagem em inferência. É o que torna **estruturalmente impossível** a
recompensa de treino divergir do corretor de avaliação (DOC-01 §1, P3).

O preço dessa garantia: um bug aqui é um bug global. Daí o requisito de
cobertura mais estrito do repositório e a suíte golden congelada.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"

    @property
    def is_decisive(self) -> bool:
        return self in (Verdict.PASS, Verdict.FAIL)


# ── A conversão que decide se o RLVR funciona ─────────────────────────────
#
# DOC-10 §2.1: se `INCONCLUSIVE` for tratado como `FAIL`, o modelo recebe
# recompensa negativa por resolver problemas difíceis — justamente aqueles
# cuja verificação simbólica não fecha (o teorema de Richardson garante que
# essa classe é não-vazia e indecidível). O gradiente resultante ensina a
# EVITAR problemas difíceis, que é o oposto do objetivo.
#
# Por isso `INCONCLUSIVE` e `ERROR` mapeiam para recompensa NEUTRA.
REWARD = {
    Verdict.PASS: 1.0,
    Verdict.FAIL: -1.0,
    Verdict.INCONCLUSIVE: 0.0,
    Verdict.ERROR: 0.0,
}


@dataclass(frozen=True)
class Claim:
    """O que se quer verificar."""

    lhs: str                       # expressão ou resposta candidata
    rhs: str | None = None         # gabarito, quando houver
    context: dict = field(default_factory=dict)
    # `context` pode trazer: unit_system, metric_signature, symbols declarados,
    # limite a tomar, invariante a checar. Ver DOC-03 §4.


@dataclass(frozen=True)
class VerificationResult:
    verdict: Verdict
    verifier_id: str
    confidence: float = 1.0
    evidence: str = ""
    cost_ms: int = 0
    counterexample: str | None = None

    @property
    def reward(self) -> float:
        return REWARD[self.verdict] * self.confidence

    def __str__(self) -> str:
        c = f" · contraexemplo: {self.counterexample}" if self.counterexample else ""
        return f"[{self.verdict.value.upper()}] {self.verifier_id}: {self.evidence}{c}"


@runtime_checkable
class Verifier(Protocol):
    id: str

    def applicable(self, claim: Claim) -> bool: ...
    def verify(self, claim: Claim) -> VerificationResult: ...


class VerifierBus:
    """Compõe verificadores e aplica a álgebra de resultados (DOC-10 §4)."""

    def __init__(self, verifiers: list[Verifier], high_confidence: float = 0.9):
        self.verifiers = verifiers
        self.high_confidence = high_confidence

    def run(self, claim: Claim) -> list[VerificationResult]:
        out: list[VerificationResult] = []
        for v in self.verifiers:
            try:
                if not v.applicable(claim):
                    continue
                t0 = time.perf_counter()
                r = v.verify(claim)
                out.append(
                    VerificationResult(
                        r.verdict, r.verifier_id, r.confidence, r.evidence,
                        int((time.perf_counter() - t0) * 1000), r.counterexample,
                    )
                )
            except Exception as exc:  # verificador quebrado nunca derruba o laço
                out.append(
                    VerificationResult(
                        Verdict.ERROR, getattr(v, "id", "?"), 0.0,
                        f"{type(exc).__name__}: {exc}"[:300],
                    )
                )
        return out

    def combine(self, results: list[VerificationResult]) -> VerificationResult:
        """Álgebra de resultados. **Não é um AND** — ver DOC-10 §4.

        Um `AND` ingênuo faria qualquer `INCONCLUSIVE` envenenar a conjunção,
        reintroduzindo o problema do §2.1 por outro caminho.
        """
        if not results:
            return VerificationResult(Verdict.INCONCLUSIVE, "bus", 0.0, "nenhum verificador aplicável")

        errors = [r for r in results if r.verdict is Verdict.ERROR]
        fails = [r for r in results if r.verdict is Verdict.FAIL]
        passes = [r for r in results if r.verdict is Verdict.PASS]
        incs = [r for r in results if r.verdict is Verdict.INCONCLUSIVE]

        # Discordância é evento de primeira classe (DOC-10 §4): nunca resolvida
        # por voto majoritário, porque voto esconderia um bug em vez de expô-lo.
        if fails and passes:
            return VerificationResult(
                Verdict.ERROR, "bus", 0.0,
                "DISCORDÂNCIA entre verificadores — investigar, não votar: "
                + " | ".join(str(r) for r in fails + passes),
            )

        if errors and not (fails or passes):
            return VerificationResult(Verdict.ERROR, "bus", 0.0,
                                      "; ".join(r.evidence for r in errors)[:400])

        if fails:
            decisivo = [r for r in fails if r.confidence >= self.high_confidence]
            escolhido = decisivo[0] if decisivo else fails[0]
            return VerificationResult(
                Verdict.FAIL, "bus", escolhido.confidence,
                "; ".join(str(r) for r in fails)[:400], counterexample=escolhido.counterexample,
            )

        if passes:
            # Confiança do conjunto = a do elo mais fraco. `INCONCLUSIVE`
            # presentes reduzem a confiança, não derrubam o veredito.
            conf = min(r.confidence for r in passes)
            if incs:
                conf *= 1.0 - 0.15 * min(len(incs), 3)
            return VerificationResult(
                Verdict.PASS, "bus", round(conf, 3),
                "; ".join(str(r) for r in passes)[:400],
            )

        return VerificationResult(
            Verdict.INCONCLUSIVE, "bus", 0.0,
            "nenhum verificador decidiu: " + "; ".join(r.evidence for r in incs)[:300],
        )

    def check(self, claim: Claim) -> VerificationResult:
        return self.combine(self.run(claim))
