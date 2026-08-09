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
import threading

import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (
    convert_xor,
    function_exponentiation,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

from phifm.core.latex.subscritos import blindar_subscritos, normalizar_subscritos
from phifm.verify.bus import Claim, Verdict, VerificationResult

TIMEOUT_S = 5

# Marcadores que indicam objeto possivelmente não-comutativo. Presença de
# qualquer um desliga a conclusão positiva (ver docstring do módulo).
_NONCOMMUTATIVE = re.compile(
    r"\\hat|\\mathbf|\\vec|\\dagger|\\otimes|\\langle|\\rangle|\\bra|\\ket|"
    r"\\sigma|\\gamma\^|\\mathbb\{1\}|\\operatorname",
    re.I,
)

# ⚠️ `implicit_multiplication_application` NÃO serve aqui, e o motivo é grave.
#
# Ele embute `split_symbols`, que parte todo identificador de mais de uma letra
# num produto de letras:
#
#     hbar  →  a*b*h*r        eps  →  e*p*s        kB  →  B*k
#
# Nomes com sublinhado ou de letra grega escapam (`v_0`, `omega`), mas os
# demais não. Isso passou desapercebido porque a comparação é entre dois lados
# que sofrem a MESMA mutilação: `eps` virava `e*p*s` nos dois e o teste
# passava. O defeito só aparece quando um lado tem `hbar` e o outro tem a
# constante escrita de outra forma — ou quando o dimensional pede a dimensão
# de `a*b*h*r` e recebe `Indeterminada`.
#
# A composição abaixo mantém tudo o que se quer de `implicit_*` (`2x`, `sin x`,
# `sin^2 x`) sem o `split_symbols`.
_TRANSFORMS = standard_transformations + (
    implicit_multiplication,
    implicit_application,
    function_exponentiation,
    convert_xor,
)

# ⚠️ O namespace do SymPy colide com a notação de Física. Sem intervenção:
#
#     E  →  número de Euler (2,718…)      quando quase sempre é energia
#     N  →  a função sp.N                 quando é número de partículas
#     Q  →  objeto de suposições          quando é carga ou calor
#     S  →  registro de singletons        quando é entropia ou ação
#     O  →  notação de Landau             quando é um símbolo qualquer
#     gamma, beta, zeta → funções especiais, quando são fator de Lorentz e v/c
#
# `E = m·c²` parseado com o E do SymPy vale 2,718 = m·c², e o verificador
# dimensional reprovaria Física correta. Em texto de Física o número de Euler
# se escreve `exp(1)`, nunca `E`, então a troca não perde nada.
#
# **`I` fica de fora de propósito.** Onda plana `exp(I*k*x)` é comum e precisa
# da unidade imaginária; corrente elétrica se resolve declarando o símbolo em
# `context["dimensions"]`, que tem precedência (ver `parse(locais=...)`).
_FISICA = {n: sp.Symbol(n) for n in ("E", "N", "Q", "S", "O", "gamma", "beta", "zeta")}


class _Timeout(Exception):
    pass


def _pode_usar_sinal() -> bool:
    """`SIGALRM` só existe em POSIX, e `signal.signal` só aceita a thread
    principal. Um `DataLoader` com `num_workers>0` cai no segundo caso mesmo
    no Linux."""
    return hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()


def _com_limite(fn, *args, **kwargs):
    """Executa `fn` com teto de TIMEOUT_S. Levanta `_Timeout` se estourar.

    Sem isto, `simplify` pode não terminar — e num laço de RLVR com 1.000
    rollouts isso trava a execução inteira.

    **Duas implementações, porque a garantia difere por plataforma e declarar
    isso é melhor que fingir que não difere:**

    *POSIX, thread principal* — `SIGALRM` interrompe o interpretador de
    verdade. O `simplify` para onde estiver e a CPU volta na hora.

    *Windows, ou thread secundária* — `signal.SIGALRM` não existe no Windows
    (`setitimer` também não), e fora da thread principal `signal.signal`
    levanta `ValueError`. Aqui a espera é limitada numa thread daemon: o
    chamador retorna em TIMEOUT_S, mas a thread segue até o `simplify`
    terminar por conta própria. Isso preserva o que o laço de RLVR precisa
    — **latência limitada** — mas NÃO recupera a CPU. Um caso patológico
    degrada a vazão em vez de congelar tudo, e o daemon não impede o processo
    de encerrar.

    A versão anterior usava `SIGALRM` sem checagem. No Windows o
    `AttributeError` resultante era engolido pelo `except Exception` de
    `parse()`, então **nenhuma expressão parseava** e todo o barramento
    devolvia `INCONCLUSIVE` — 20 dos 55 testes falhavam por isso.
    """
    if _pode_usar_sinal():
        def _alarme(signum, frame):
            raise _Timeout()

        anterior = signal.signal(signal.SIGALRM, _alarme)
        signal.setitimer(signal.ITIMER_REAL, TIMEOUT_S)
        try:
            return fn(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, anterior)

    caixa: dict = {}

    def _corpo():
        try:
            caixa["valor"] = fn(*args, **kwargs)
        except BaseException as exc:  # repassada ao chamador abaixo
            caixa["erro"] = exc

    t = threading.Thread(target=_corpo, daemon=True, name="phifm-verify")
    t.start()
    t.join(TIMEOUT_S)
    if t.is_alive():
        raise _Timeout()
    if "erro" in caixa:
        raise caixa["erro"]
    return caixa["valor"]


def parse(expr: str, locais: dict | None = None) -> sp.Expr | None:
    """Aceita LaTeX ou sintaxe SymPy. Devolve ``None`` se não parsear.

    `locais` são símbolos declarados pelo chamador e têm precedência sobre
    `_FISICA` e sobre o namespace do SymPy — é assim que quem quer `I` como
    corrente elétrica, e não como unidade imaginária, consegue dizer isso.
    """
    s = expr.strip().strip("$").strip()
    if not s:
        return None
    ld = {**_FISICA, **(locais or {})}

    # ⚠️ Sublinhado ou `^` soltos NÃO são evidência de LaTeX. A heurística
    # anterior (`"_" in s`) mandava `q*E_campo` para o `parse_latex`, que lê
    # `E_campo` como subscrito de uma letra e devolve `E_{c}` — símbolo
    # diferente, silenciosamente. Passava despercebido em `v_0` porque ali as
    # duas leituras coincidem.
    #
    # `^` sozinho não precisa do LaTeX: `convert_xor` já o traduz.
    #
    # A decisão sai do texto ORIGINAL porque a normalização logo abaixo apaga
    # justamente as marcas em que ela se baseia: `E_{tot} = E_{cin} + E_{pot}`
    # perde os `_{`, e sem eles a expressão iria para o `parse_expr`, que não
    # aceita `=` — equação inteira virando `None`.
    usar_latex = "\\" in s or "_{" in s or "^{" in s

    # Subscrito nomeado vira identificador ANTES de qualquer parser ver a
    # expressão: `E_{cin}` → `E_cin`, `\rho_{xy}` → `rho_xy`. Sem isto o ANTLR
    # lê o conteúdo de `_{...}` como produto e dois símbolos distintos colapsam
    # — ver `core.latex.subscritos`, que é onde a regra mora e por quê.
    #
    # Roda aqui, e não só na ingestão, porque quatro dos cinco consumidores do
    # barramento (rollout de RLVR, gabarito de benchmark, admissão de sintético
    # e auto-checagem em inferência) entregam LaTeX que nunca passou pelo
    # pipeline do DOC-03. Normalizar só na ingestão deixaria o RLVR — o
    # consumidor em que um símbolo trocado vira gradiente — sem proteção.
    s, _ = normalizar_subscritos(s)

    if usar_latex:
        # O ANTLR ainda estilhaça identificador composto mesmo já achatado
        # (`E_cin` → `E_{c}*(i*n)`, colapsando com `E_cal`). Marcadores de
        # subscrito numérico são a forma que ele atravessa intacta; a troca é
        # desfeita nos símbolos logo abaixo e não vaza para o chamador.
        blindado, marcadores = blindar_subscritos(s)
        try:
            e = _com_limite(parse_latex, blindado)
            if marcadores:
                e = e.xreplace(
                    {sp.Symbol(m): sp.Symbol(ident) for m, ident in marcadores.items()}
                )
            return e
        except Exception:
            pass
    try:
        return _com_limite(
            parse_expr, s, local_dict=ld, transformations=_TRANSFORMS, evaluate=True
        )
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
            d = _com_limite(sp.simplify, a - b)
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
            eq = _com_limite(a.equals, b)
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
