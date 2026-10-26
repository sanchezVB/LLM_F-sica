"""Barramento de verificação — o protocolo (DOC-10 §2).

Uma única implementação servindo cinco consumidores: filtro de dados,
admissão de sintéticos, recompensa de RLVR, correção de benchmark e
auto-checagem em inferência. É o que torna **estruturalmente impossível** a
recompensa de treino divergir do corretor de avaliação (DOC-01 §1, P3).

O preço dessa garantia: um bug aqui é um bug global. Daí o requisito de
cobertura mais estrito do repositório e a suíte golden congelada.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"

    @property
    def is_decisive(self) -> bool:
        return self in (Verdict.PASS, Verdict.FAIL)


class Papel(StrEnum):
    """O que um PASS do verificador prova (ADR-0005 §4.2).

    O simbólico e o numérico provam que a resposta É o gabarito. O
    dimensional, o de limites e o de conservação só provam que ela não viola
    uma condição — `F·d` tem a dimensão de `m·v²` e não é `m·v²`.
    """

    EQUIVALENCIA = "equivalencia"
    CONDICAO = "condicao"


class Motivo(StrEnum):
    """Por que um verificador não decidiu (ADR-0005 §4.4).

    O veredito continua o mesmo para os consumidores que só leem o veredito.
    A recompensa do RLVR lê o motivo, porque ali importa QUEM causou a
    abstenção: a resposta (vale um FAIL) ou o item e o problema (sai do grupo).
    """

    # causados pela resposta
    RESPOSTA_ILEGIVEL = "resposta_ilegivel"
    RESPOSTA_INCOMPARAVEL = "resposta_incomparavel"
    RESPOSTA_INDEFINIDA = "resposta_indefinida"
    VOCABULARIO = "vocabulario"
    OPERADORES_NA_RESPOSTA = "operadores_na_resposta"
    EXCECAO_NA_RESPOSTA = "excecao_na_resposta"
    TEMPO_NA_RESPOSTA = "tempo_na_resposta"
    # causados pelo item
    GABARITO_ILEGIVEL = "gabarito_ilegivel"
    GABARITO_INCOMPARAVEL = "gabarito_incomparavel"
    OPERADORES = "operadores"
    SEM_CONVENCAO = "sem_convencao"
    # causados pelo problema
    TEMPO = "tempo"
    INDECIDIVEL = "indecidivel"
    # do barramento, ou defeito de verificador
    DISCORDANCIA = "discordancia"
    CONTRADICAO = "contradicao"
    EXCECAO = "excecao"
    SEM_MOTIVO = "sem_motivo"


MOTIVOS_DA_RESPOSTA = frozenset({
    Motivo.RESPOSTA_ILEGIVEL, Motivo.RESPOSTA_INCOMPARAVEL, Motivo.RESPOSTA_INDEFINIDA,
    Motivo.VOCABULARIO, Motivo.OPERADORES_NA_RESPOSTA, Motivo.EXCECAO_NA_RESPOSTA,
    Motivo.TEMPO_NA_RESPOSTA,
})
MOTIVOS_DO_ITEM = frozenset({
    Motivo.GABARITO_ILEGIVEL, Motivo.GABARITO_INCOMPARAVEL, Motivo.OPERADORES,
    Motivo.SEM_CONVENCAO,
})

TIPOS_DE_AFIRMACAO = ("equivalencia", "relacao")


def tipo_da_afirmacao(claim: Claim) -> str:
    """`equivalencia` (resposta contra gabarito, o padrão) ou `relacao` (ADR-0005 §4.1).

    Uma relação é uma equação física, como `E = m c²` no filtro de dados:
    comparar `E` com `m c²` como símbolos independentes não tem sentido, e ali
    só as condições necessárias decidem. Valor fora dos dois levanta, para um
    `"relação"` com cedilha não virar equivalência em silêncio.
    """
    tipo = claim.context.get("tipo") or "equivalencia"
    if tipo not in TIPOS_DE_AFIRMACAO:
        raise ValueError(f"context['tipo'] desconhecido: {tipo!r} — use {TIPOS_DE_AFIRMACAO}")
    return tipo


# ── A conversão que decide se o RLVR funciona ─────────────────────────────
#
# DOC-10 §2.1: se `INCONCLUSIVE` for tratado como `FAIL`, o modelo recebe
# recompensa negativa por resolver problemas difíceis — justamente aqueles
# cuja verificação simbólica não fecha (o teorema de Richardson garante que
# essa classe é não-vazia e indecidível). O gradiente resultante ensina a
# EVITAR problemas difíceis, que é o oposto do objetivo.
#
# Por isso `INCONCLUSIVE` e `ERROR` mapeiam para recompensa NEUTRA.
#
# ⚠️ Esta tabela serve a quem lê o veredito combinado. A recompensa do RLVR é
# `verify.recompensa.recompensa`, que lê o vetor e o motivo (ADR-0005): no GRPO
# o 0 não é neutro — num grupo de FAIL ele é o melhor da turma — e uma
# resposta que se torna ilegível de propósito sairia ganhando.
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
    # No fim e só por nome: as construções posicionais de antes continuam
    # valendo. O papel padrão é o mais estrito (equivalência), e é o barramento
    # quem o preenche a partir do verificador, em `run`.
    motivo: Motivo | None = field(default=None, kw_only=True)
    papel: Papel = field(default=Papel.EQUIVALENCIA, kw_only=True)

    @property
    def reward(self) -> float:
        """O mapeamento antigo, multiplicado pela confiança. Não é a recompensa
        do RLVR: ver `verify.recompensa.recompensa`."""
        return REWARD[self.verdict] * self.confidence

    def __str__(self) -> str:
        c = f" · contraexemplo: {self.counterexample}" if self.counterexample else ""
        return f"[{self.verdict.value.upper()}] {self.verifier_id}: {self.evidence}{c}"


@runtime_checkable
class Verifier(Protocol):
    """Um verificador. Pode declarar `papel` (`Papel`) como atributo de classe;
    sem ele, o barramento o trata como condição necessária — um verificador que
    não disse o que prova nunca vale como prova de equivalência."""

    id: str

    def applicable(self, claim: Claim) -> bool: ...
    def verify(self, claim: Claim) -> VerificationResult: ...


class Regra(StrEnum):
    """Qual linha da álgebra do ADR-0005 §4.3 decidiu. A mesma função serve ao
    `combine` e à recompensa do RLVR, para as duas nunca divergirem."""

    VAZIO = "vazio"
    DISCORDANCIA = "discordancia"
    CONTRADICAO = "contradicao"
    FAIL = "fail"
    PASS = "pass"
    CONDICAO_SEM_EQUIVALENCIA = "condicao_sem_equivalencia"
    PASS_DE_RELACAO = "pass_de_relacao"
    ERROR = "error"
    INCONCLUSIVE = "inconclusive"


def falha_decisiva(r: VerificationResult, high_confidence: float = 0.9) -> bool:
    """FAIL que conta como FAIL. O DOC-10 §4 diz "FAIL de verificador de ALTA
    confiança": abaixo disso — a conservação que só não conseguiu zerar a
    derivada, por exemplo — o FAIL não prova nada e conta como abstenção."""
    return r.verdict is Verdict.FAIL and r.confidence >= high_confidence


def classificar(results: list[VerificationResult], high_confidence: float = 0.9) -> Regra:
    """A álgebra de resultados (ADR-0005 §4.3). A primeira linha que casa decide."""
    if not results:
        return Regra.VAZIO
    eq = [r for r in results if r.papel is Papel.EQUIVALENCIA]
    cond = [r for r in results if r.papel is Papel.CONDICAO]
    eq_pass = any(r.verdict is Verdict.PASS for r in eq)
    eq_fail = any(falha_decisiva(r, high_confidence) for r in eq)
    cond_pass = any(r.verdict is Verdict.PASS for r in cond)
    cond_fail = any(falha_decisiva(r, high_confidence) for r in cond)

    # Discordância é evento de primeira classe (DOC-10 §4): nunca resolvida
    # por voto, porque voto esconderia um bug em vez de expô-lo. Mas só entre
    # dois verificadores que afirmam a MESMA coisa — a equivalência.
    if eq_pass and eq_fail:
        return Regra.DISCORDANCIA
    if eq_pass and cond_fail:
        return Regra.CONTRADICAO
    # Condição necessária aprovada não contradiz FAIL de equivalência:
    # `F·d` tem a dimensão de `m·v²` e não é `m·v²`. Até 2026-10-23 isso virava
    # DISCORDÂNCIA, e o erro mais comum (fator ou sinal trocado) valia 0.
    if eq_fail or cond_fail:
        return Regra.FAIL
    if eq_pass:
        return Regra.PASS
    if cond_pass:
        # Com um verificador de equivalência aplicável, condição aprovada não
        # prova equivalência: `m v² f(1)` passava só pelo dimensional.
        return Regra.CONDICAO_SEM_EQUIVALENCIA if eq else Regra.PASS_DE_RELACAO
    if any(r.verdict is Verdict.ERROR for r in results):
        return Regra.ERROR
    return Regra.INCONCLUSIVE


def _motivo_de(results: list[VerificationResult]) -> Motivo | None:
    """O motivo que representa uma abstenção do conjunto: o da equivalência primeiro."""
    for papel in (Papel.EQUIVALENCIA, Papel.CONDICAO):
        for r in results:
            if r.papel is papel and r.motivo is not None and not r.verdict.is_decisive:
                return r.motivo
    return None


class VerifierBus:
    """Compõe verificadores e aplica a álgebra de resultados (DOC-10 §4, ADR-0005)."""

    def __init__(self, verifiers: list[Verifier], high_confidence: float = 0.9):
        self.verifiers = verifiers
        self.high_confidence = high_confidence

    def run(self, claim: Claim) -> list[VerificationResult]:
        try:
            tipo_da_afirmacao(claim)
        except ValueError as exc:
            return [VerificationResult(Verdict.ERROR, "bus", 0.0, str(exc),
                                       motivo=Motivo.EXCECAO, papel=Papel.CONDICAO)]
        out: list[VerificationResult] = []
        for v in self.verifiers:
            papel = getattr(v, "papel", Papel.CONDICAO)
            try:
                if not v.applicable(claim):
                    continue
                t0 = time.perf_counter()
                r = v.verify(claim)
                # ⚠️ `replace`, e não um construtor campo a campo: um campo novo
                # (o motivo, o papel) sumia aqui, e a recompensa não via nada.
                motivo = r.motivo
                if motivo is None and not r.verdict.is_decisive:
                    motivo = Motivo.SEM_MOTIVO
                out.append(dataclasses.replace(
                    r, cost_ms=int((time.perf_counter() - t0) * 1000), papel=papel, motivo=motivo,
                ))
            except Exception as exc:  # verificador quebrado nunca derruba o laço
                out.append(
                    VerificationResult(
                        Verdict.ERROR, getattr(v, "id", "?"), 0.0,
                        f"{type(exc).__name__}: {exc}"[:300],
                        motivo=Motivo.EXCECAO, papel=papel,
                    )
                )
        return out

    def combine(self, results: list[VerificationResult]) -> VerificationResult:
        """Álgebra de resultados. **Não é um AND** — ver DOC-10 §4 e `classificar`.

        Um `AND` ingênuo faria qualquer `INCONCLUSIVE` envenenar a conjunção,
        reintroduzindo o problema do §2.1 por outro caminho.
        """
        regra = classificar(results, self.high_confidence)
        if regra is Regra.VAZIO:
            return VerificationResult(Verdict.INCONCLUSIVE, "bus", 0.0, "nenhum verificador aplicável")

        fails = [r for r in results if r.verdict is Verdict.FAIL]
        passes = [r for r in results if r.verdict is Verdict.PASS]
        decisivos = [r for r in fails if falha_decisiva(r, self.high_confidence)]
        # FAIL de baixa confiança conta como abstenção (ver `falha_decisiva`).
        abstencoes = [r for r in results
                      if r.verdict is Verdict.INCONCLUSIVE or r in fails and r not in decisivos]

        if regra is Regra.DISCORDANCIA:
            eq = [r for r in fails + passes if r.papel is Papel.EQUIVALENCIA]
            return VerificationResult(
                Verdict.ERROR, "bus", 0.0,
                "DISCORDÂNCIA entre verificadores de equivalência — investigar, não votar: "
                + " | ".join(str(r) for r in eq),
                motivo=Motivo.DISCORDANCIA,
            )
        if regra is Regra.CONTRADICAO:
            return VerificationResult(
                Verdict.ERROR, "bus", 0.0,
                "CONTRADIÇÃO: a equivalência aprovou e uma condição necessária reprovou — "
                "o gabarito, a tabela ou a forma da resposta é incoerente: "
                + " | ".join(str(r) for r in decisivos + passes),
                motivo=Motivo.CONTRADICAO,
            )
        if regra is Regra.FAIL:
            escolhido = ([r for r in decisivos if r.papel is Papel.EQUIVALENCIA] or decisivos)[0]
            return VerificationResult(
                Verdict.FAIL, "bus", escolhido.confidence,
                "; ".join(str(r) for r in decisivos)[:400],
                counterexample=escolhido.counterexample,
            )
        if regra in (Regra.PASS, Regra.PASS_DE_RELACAO):
            # Confiança do conjunto = a do elo mais fraco. Abstenções presentes
            # reduzem a confiança, não derrubam o veredito.
            conf = min(r.confidence for r in passes)
            if abstencoes:
                conf *= 1.0 - 0.15 * min(len(abstencoes), 3)
            return VerificationResult(
                Verdict.PASS, "bus", round(conf, 3),
                "; ".join(str(r) for r in passes)[:400],
            )
        if regra is Regra.ERROR:
            errors = [r for r in results if r.verdict is Verdict.ERROR]
            return VerificationResult(Verdict.ERROR, "bus", 0.0,
                                      "; ".join(r.evidence for r in errors)[:400],
                                      motivo=Motivo.EXCECAO)
        prefixo = ("condição necessária aprovou, mas nenhum verificador de equivalência "
                   "decidiu: " if regra is Regra.CONDICAO_SEM_EQUIVALENCIA
                   else "nenhum verificador decidiu: ")
        return VerificationResult(
            Verdict.INCONCLUSIVE, "bus", 0.0,
            prefixo + "; ".join(r.evidence for r in abstencoes)[:300],
            motivo=_motivo_de(results),
        )

    def check(self, claim: Claim) -> VerificationResult:
        return self.combine(self.run(claim))
