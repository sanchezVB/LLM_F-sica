"""Verificador de equivalência simbólica (DOC-10 §3.1).

Implementa a cascata: comparação estrutural → `simplify` → `equals` → delega
ao numérico. Timeout rígido em todos os caminhos.

**Há um limite matemático, e ele é declarado, não contornado.** Pelo teorema
de Richardson (1968), decidir se uma expressão elementar envolvendo racionais,
π, `log`, `exp`, `sin` e valor absoluto é identicamente zero é **indecidível**.
Não existe verificador simbólico completo. Qualquer sistema que alegue decidir
equivalência sempre ou está errado em algum caso, ou não termina.

É por isso que `INCONCLUSIVE` existe, que o timeout é obrigatório, e que o
verificador numérico é complemento indispensável — não plano B.

**Armadilha específica de Física.** O SymPy assume comutatividade por padrão:
``A*B - B*A`` simplifica para ``0``, o que é **falso** para operadores
quânticos e matrizes — e toda a Mecânica Quântica está nessa distinção. Sem
anotação de tipo confiável, expressões com operadores retornam
`INCONCLUSIVE`, **nunca** `PASS`.
"""

from __future__ import annotations

import re
import signal
from contextlib import contextmanager

import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from phifm.verify.bus import Claim, VerificationResult, Verdict

TIMEOUT_S = 5

# Marcadores que indicam objeto possivelmente não-comutativo. Presença de
# qualquer um desliga a conclusão positiva (ver docstring do módulo).
_NONCOMMUTATIVE = re.compile(
    r"\\hat|\\mathbf|\\vec|\\dagger|\\otimes|\\langle|\\rangle|\\bra|\\ket|"
    r"\\sigma|\\gamma\^|\\mathbb\{1\}|\\operatorname",
    re.I,
)

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


class _Timeout(Exception):
    pass


@contextmanager
def _limite(segundos: int):
    """Timeout por sinal. Sem isto, `simplify` pode não terminar — e num laço
    de RLVR com 1.000 rollouts isso trava a execução inteira."""

    def _alarme(signum, frame):
        raise _Timeout()

    anterior = signal.signal(signal.SIGALRM, _alarme)
    signal.setitimer(signal.ITIMER_REAL, segundos)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, anterior)


def parse(expr: str) -> sp.Expr | None:
    """Aceita LaTeX ou sintaxe SymPy. Devolve ``None`` se não parsear."""
    s = expr.strip().strip("$").strip()
    if not s:
        return None
    if "\\" in s or "^" in s or "_" in s:
        try:
            with _limite(TIMEOUT_S):
                return parse_latex(s)
        except Exception:
            pass
    try:
        with _limite(TIMEOUT_S):
            return parse_expr(s, transformations=_TRANSFORMS, evaluate=True)
    except Exception:
        return None


def _tem_operadores(claim: Claim) -> bool:
    if claim.context.get("noncommutative"):
        return True
    texto = f"{claim.lhs} {claim.rhs or ''}"
    return bool(_NONCOMMUTATIVE.search(texto))


def _degenerada(resposta: sp.Expr, gabarito: sp.Expr) -> bool:
    """Resposta trivial que tenta passar sem resolver o problema.

    DOC-09 §5.4 lista "degenerescência algébrica" como vetor de reward
    hacking: responder ``0`` a tudo satisfaz um verificador ingênuo.

    ⚠️ Esta checagem só pode rodar **depois** de estabelecida a
    não-equivalência. Aplicá-la antes reprova respostas constantes
    legítimas — ``sin²x + cos²x = 1`` tem gabarito constante e está certo.
    Foi o bug encontrado no primeiro teste em Física real (2026-08-03).
    """
    return (
        resposta.is_number
        and bool(gabarito.free_symbols)
        and resposta in (sp.Integer(0), sp.Integer(1))
    )


class SymbolicVerifier:
    id = "symbolic/sympy@0.1.0"

    def applicable(self, claim: Claim) -> bool:
        return claim.rhs is not None

    def verify(self, claim: Claim) -> VerificationResult:
        a, b = parse(claim.lhs), parse(claim.rhs or "")
        if a is None or b is None:
            qual = "lhs" if a is None else "rhs"
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0, f"não foi possível parsear {qual}"
            )

        # ── comparação estrutural (barata, decide a maioria dos casos) ────
        try:
            if a == b or sp.srepr(a) == sp.srepr(b):
                return VerificationResult(Verdict.PASS, self.id, 1.0, "idênticas estruturalmente")
        except Exception:
            pass

        operadores = _tem_operadores(claim)

        try:
            with _limite(TIMEOUT_S):
                d = sp.simplify(a - b)
        except (_Timeout, Exception):
            return VerificationResult(
                Verdict.INCONCLUSIVE, self.id, 0.0,
                f"simplify não concluiu em {TIMEOUT_S}s — delegar ao numérico",
            )

        if d == 0:
            if operadores:
                # Ver docstring: o SymPy comutou o que talvez não comute.
                return VerificationResult(
                    Verdict.INCONCLUSIVE, self.id, 0.0,
                    "expressão contém operadores/matrizes e o SymPy assume "
                    "comutatividade — diferença nula NÃO prova equivalência",
                )
            return VerificationResult(Verdict.PASS, self.id, 1.0, "simplify(a−b) = 0")

        try:
            with _limite(TIMEOUT_S):
                eq = a.equals(b)
        except (_Timeout, Exception):
            eq = None

        if eq is True and not operadores:
            return VerificationResult(Verdict.PASS, self.id, 0.95, "equals() confirmou")
        if eq is False or d != 0:
            # Só AGORA a degenerescência é diagnosticável: sabemos que as
            # expressões diferem, então uma resposta constante trivial é
            # tentativa de burlar, não resposta legítima.
            if _degenerada(a, b):
                return VerificationResult(
                    Verdict.FAIL, self.id, 1.0,
                    "resposta degenerada: constante trivial onde se esperava "
                    "expressão simbólica",
                    counterexample=f"{a} vs {b}",
                )
            if eq is False:
                return VerificationResult(
                    Verdict.FAIL, self.id, 0.95, "equals() refutou",
                    counterexample=f"a−b = {sp.simplify(d)}",
                )

        return VerificationResult(
            Verdict.INCONCLUSIVE, self.id, 0.0,
            "cascata simbólica não decidiu — delegar ao numérico "
            "(indecidibilidade de Richardson não é falha do modelo)",
        )
