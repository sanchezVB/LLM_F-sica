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

import contextlib
import re
import signal
import threading

import sympy as sp
from sympy.core.function import AppliedUndef
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
# `context["dimensions"]`, que tem precedência (ver `parse(locais=...)` e
# `_declarados`).
#
# ⚠️ Isto vale para a sintaxe do SymPy. Em LaTeX o `I` já sai símbolo, porque o
# ANTLR não consulta namespace nenhum, e é a leitura certa: o próprio
# `sp.latex(sp.I)` escreve `i`. A consequência é que a mesma resposta muda de
# sentido conforme a rota — `I^{2} R` é I²R, e `I^2 R`, sem `\` nem `^{`, vai
# para o `parse_expr` e vale −R. Declarar o símbolo resolve as duas rotas.
#
# O dono reconfirmou a decisão em 2026-10-19, sabendo dessa consequência: na
# sintaxe do SymPy, `I` sem declaração segue a unidade imaginária.
_FISICA = {n: sp.Symbol(n) for n in ("E", "N", "Q", "S", "O", "gamma", "beta", "zeta")}


# ── O LaTeX que o ANTLR lê errado ──────────────────────────────────────────
#
# Defeitos do backend ANTLR do `sympy.parsing.latex` (SymPy 1.14), achados em
# 2026-10-19 ao construir o corretor do portão G2, que os contornava por conta
# própria em `eval/g2_corretor.py`. A correção mora aqui porque `parse` é o
# barramento único dos cinco consumidores (DOC-10, DOC-01 P3): contornar só no
# G2 deixaria o RLVR dando recompensa pela leitura errada.
#
# A ordem em `_ler_latex` importa: a tipografia sai antes de tudo (`\big|` é
# módulo), e as pontas são aparadas por último (as trocas anteriores podem
# deixar um `\,` na ponta).

# Letras gregas, que a gramática lê como `SYMBOL` e que de fato nomeiam grandeza.
_GREGAS = frozenset({
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "vartheta", "iota", "kappa", "varkappa", "lambda", "mu", "nu", "xi",
    "omicron", "pi", "varpi", "rho", "varrho", "sigma", "varsigma", "tau",
    "upsilon", "phi", "varphi", "chi", "psi", "omega", "Gamma", "Delta", "Theta",
    "Lambda", "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega",
})

# ⚠️ Tipografia. A regra `SYMBOL` da gramática aceita qualquer macro, então o
# que é só tamanho ou estilo virava grandeza:
#
#     \displaystyle\frac{a}{b}  →  displaystyle·a/b
#     \big( a \big)             →  a função `big` aplicada a a·big
#
# e a resposta certa reprovava contra o gabarito escrito sem elas. Saem
# trocadas por `\,`, que a gramática pula: apagar sem deixar nada colaria os
# vizinhos, e um espaço comum pode formar diferencial (`d\displaystyle x` →
# `d x` → dx).
_TIPOGRAFIA = re.compile(
    r"\\(?:displaystyle|textstyle|scriptstyle|scriptscriptstyle|[Bb]igg?[lrm]?|"
    r"limits|nolimits)(?![A-Za-z])"
)

# Constantes em tipo reto, pela ISO 80000-2: `\mathrm{e}` é o número de Euler,
# `\mathrm{i}` a unidade imaginária e `\mathrm{d}` o operador diferencial — o
# e, o i e o d em itálico são grandezas (carga, índice, distância). Sem isto,
# `\mathrm{e}^{-\beta E}` virava mathrm/e^{βE}.
#
# O `d` vira a letra, e o lexer faz dele o diferencial como faria com `dx`. O e
# e o i viram uma macro-sentinela, trocada pela constante depois do parse:
# não há como escrever o e de Euler em LaTeX que o ANTLR leia como constante.
#
# Não vale logo depois de `_`: `m_\mathrm{e}` é a massa do elétron, e quem
# trata subscrito é `normalizar_subscritos`.
_RETAS = re.compile(r"(?<!_)\\(?:mathrm|textrm|operatorname)\s*\{\s*([dei])\s*\}")
_SENTINELAS = {"e": "phifmeuler", "i": "phifmimaginaria"}
_CONSTANTES_RETAS = {
    sp.Symbol(_SENTINELAS["e"]): sp.E,
    sp.Symbol(_SENTINELAS["i"]): sp.I,
}


def _reta(m: re.Match) -> str:
    letra = m.group(1)
    return "d" if letra == "d" else f"{{\\{_SENTINELAS[letra]}}}"


# ⚠️ Macro que não nomeia grandeza. Mesmo com a tipografia fora, a regra
# `SYMBOL` segue transformando em símbolo livre toda macro que a gramática não
# conhece: `\text{m}` → m·text, `\vec{F}` → F·vec, `\dot{x}` → dot·x,
# `a \approx b` → a·approx·b, `30^\circ` → 30^circ, `A^\dagger` → A^dagger.
# Tratar uma a uma é correr atrás da próxima; o que se fecha aqui é a classe:
# se o resultado tem um símbolo cujo nome vem de uma macro do texto, e essa
# macro não é grega nem `\hbar`/`\ell`, a leitura é recusada e `parse` devolve
# None — INCONCLUSIVE, nunca uma leitura com um `text` ou um `vec` dentro.
#
# Macros que a gramática conhece (`\frac`, `\sin`, `\cdot`, `\int`) não geram
# símbolo, então não precisam de lista.
#
# **Unidade é recusada, por decisão do dono (2026-10-19).** `5\,\text{m}` dá
# None aqui. Descartar a unidade aprovaria 5 km contra 5 m, e levá-la ao
# dimensional exigiria uma gramática de unidades que não existe. Quem compara
# NÚMEROS tira a unidade por conta própria, sabendo em que unidade está o
# gabarito — é o que faz o `numero()` do corretor do G2.
_MACROS_DE_GRANDEZA = _GREGAS | {"hbar", "hslash", "ell"}


class _MacroLivre(ValueError):
    """O ANTLR leu como grandeza uma macro que não é nome de grandeza."""


def _recusar_macros_livres(e: sp.Basic, texto: str) -> None:
    suspeitas = set(re.findall(r"\\([A-Za-z]+)", texto)) - _MACROS_DE_GRANDEZA
    if not suspeitas:
        return
    nomes = [s.name for s in e.atoms(sp.Symbol)]
    nomes += [f.func.__name__ for f in e.atoms(AppliedUndef)]
    # O nome inteiro, e não só a base: `x_{\vec{v}}` sai o símbolo `x_{v*vec}`.
    achadas = {p for n in nomes for p in re.findall(r"[A-Za-z]+", n)} & suspeitas
    if achadas:
        raise _MacroLivre(", ".join(sorted(achadas)))


# ⚠️ Módulo descartado em silêncio. A gramática reserva `\left|` e `\right|`
# como tokens próprios (L_BAR, R_BAR), feitos para o bra-ket, e a regra do
# módulo só aceita a barra crua. `\left|E\right|` não casa com regra nenhuma, e
# a regra de entrada (`math: relation`) não exige o fim do texto — então o
# parser para no último ponto que entendeu e joga o resto fora, sem erro:
#
#     q \left|E\right|   →  q       q|E| e q passavam por iguais
#     \left|E\right| q   →  erro    e `parse` devolvia None
#
# `\lvert`, `\rvert` e `\vert` caem noutro canto da gramática (`SYMBOL`, que
# aceita qualquer macro) e viravam os símbolos livres `lvert` e `rvert`. Toda
# grafia do módulo vira a barra crua, que o bra-ket também aceita
# (`ket: (L_BAR | BAR) expr R_ANGLE`). A barra crua não muda: casa e volta igual.
_MODULO = re.compile(r"(?:\\(?:left|right)\s*)?(?:\||\\[lr]?vert(?![A-Za-z]))")

# ⚠️ O resto do texto, descartado pelo mesmo motivo. Sem fim de texto na
# gramática, QUALQUER sobra que o parser não entenda some (`q E ]` → `q*E`). O
# modo `strict` do SymPy recusa a sobra, mas confere o primeiro e o último
# token contra as pontas do texto, e por isso reprova LaTeX válido que comece
# ou termine num token que a gramática pula — o `\left` de
# `\left(\frac{a}{b}\right)^2`. Daí aparar as pontas antes.
#
# A lista é a dos tokens `-> skip` do LaTeX.g4 do SymPy 1.14. Se faltar algum,
# o `strict` recusa, e a expressão sai `None`: INCONCLUSIVE, nunca uma leitura
# errada.
_PULADO = (
    r"\s|\\[,:;!*\-./\"(=]|\\(?:quad|qquad|thinspace|medspace|thickspace|"
    r"negthinspace|negmedspace|negthickspace|left|right|vrule|vcenter|vbox|"
    r"vskip|vspace|hfil)(?![A-Za-z])"
)
_PONTAS = re.compile(rf"^(?:{_PULADO})+|(?:{_PULADO})+$")

# ⚠️ `d` antes de macro. O lexer tem um token DIFFERENTIAL (`d`, espaço
# opcional, e uma letra ou uma macro) para o `dx` das integrais. Como o lexer
# fica com o casamento mais longo, `d` seguido de QUALQUER macro vira o
# diferencial dela:
#
#     2d \sin\theta  →  2·dsin·θ     a lei de Bragg
#     d \cdot F      →  dcdot·F      e `d\frac{1}{2}` → dfrac·1·2
#
# Função, operador e constante não são variável de integração, então ali o `d`
# é a grandeza d. Um `\,` (espaço fino, que a gramática pula) entre os dois
# quebra o token sem mudar o sentido. Letra grega fica com a leitura de
# diferencial (`d\theta`, `d\phi`), que é a dela numa integral; `\pi` não,
# porque constante não tem diferencial. Fora de macro nada muda: `dx` segue
# ambíguo entre dx e d·x, e a leitura continua a da gramática.
_GREGAS_DE_INTEGRACAO = (_GREGAS - {"pi"}) | {"ell"}
# A primeira alternativa consome a macro inteira, para o `d` de `\delta` ou de
# `\dfrac` nunca ser lido como letra solta.
_D_ANTES_DE_MACRO = re.compile(r"\\[A-Za-z]+|d(?=[ \t\r\n]*\\([A-Za-z]+))")


def _separar_d(m: re.Match) -> str:
    macro = m.group(1)
    if macro is None or macro in _GREGAS_DE_INTEGRACAO:
        return m.group(0)
    return "d\\,"


# ⚠️ `\pi` antes de parêntese. A gramática lê `\pi(…)` como a função `pi`
# aplicada ao parêntese, e um expoente depois fica com a CHAMADA inteira:
#
#     2\pi\left(\frac{L}{g}\right)^{1/2}  →  2·pi(L/g)^{1/2}
#
# Trocar a função pelo produto depois do parse (a primeira correção, de
# 2026-10-19) levava o π para dentro da potência — (π·L/g)^{1/2} — e reprovava
# o exemplo canônico do DOC-09 §5.1 contra `2\pi\sqrt{L/g}`. Um `\cdot`
# explícito antes do parse impede a leitura como função, e o expoente fica com
# o parêntese. π como campo (o momento conjugado da TQC) fica de fora: em
# resposta de Física, `\pi` é a constante. O que pula a gramática (`\,`, `\;`)
# pode vir entre os dois, e é onde a tipografia removida deixa o seu `\,`.
_PI_ANTES_DE_PARENTESE = re.compile(
    r"\\pi(?![A-Za-z])(?=(?:\s|\\[,;:!])*(?:\(|\\left\s*\())"
)


def _pi(e: sp.Basic) -> sp.Basic:
    """`\\pi` → π.

    ⚠️ A gramática aceita qualquer macro como `SYMBOL` e só trata `\\infty` à
    parte, então `\\pi` sai `Symbol('pi')` — do próprio ANTLR, não do `xreplace`
    dos marcadores de subscrito: `\\frac{\\sqrt{2}\\pi}{6}` não tem marcador
    nenhum e já sai assim. Com π livre, o `NumericVerifier` o sorteia como
    variável, e uma resposta com 3.14159 nunca bate com o gabarito em π.
    """
    return e.xreplace({sp.Symbol("pi"): sp.pi})


def _ler_latex(texto: str) -> sp.Basic:
    texto = _TIPOGRAFIA.sub(lambda _: "\\,", texto)
    texto = _RETAS.sub(_reta, texto)
    texto = _MODULO.sub("|", texto)
    texto = _D_ANTES_DE_MACRO.sub(_separar_d, texto)
    texto = _PI_ANTES_DE_PARENTESE.sub(lambda _: "\\pi\\cdot ", texto)
    texto = _PONTAS.sub("", texto)
    e = _pi(parse_latex(texto, strict=True)).xreplace(_CONSTANTES_RETAS)
    _recusar_macros_livres(e, texto)
    return e


def _declarados(claim: Claim) -> dict[str, sp.Symbol]:
    """Os símbolos que `context["dimensions"]` declara, para `parse(locais=...)`.

    Símbolo declarado é símbolo, não constante do SymPy: quem declarou `I` como
    corrente não quer a unidade imaginária de volta. ⚠️ Até 2026-10-19 só o
    dimensional passava a declaração ao `parse`; o simbólico e o numérico — os
    que decidem PASS e FAIL — liam `I**2*R` como −R mesmo com `I` declarado.
    """
    return {n: sp.Symbol(n) for n in (claim.context.get("dimensions") or {})}


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


def _aquecer() -> None:
    """Carrega o parser LaTeX na importação, na thread que importa, sem teto.

    ⚠️ A primeira chamada ao `parse_latex` carrega o ANTLR e a cadeia de
    importação do bra-ket (`sympy.physics.quantum`, que traz numpy e
    scipy.sparse). No Windows isso pode passar de `TIMEOUT_S`, e aí a primeira
    verificação do processo saía INCONCLUSIVE por tempo — e a thread abandonada
    pelo `_com_limite` seguia importando o numpy em paralelo com a principal (já
    deu "partially initialized module numpy" num teste). Achado em 2026-10-19
    no corretor do G2, que tinha o seu próprio aquecimento.

    Sem o ANTLR instalado a chamada falha aqui e de novo em cada `parse`, que
    cai no `parse_expr` como sempre caiu; importar o módulo não pode quebrar.
    """
    with contextlib.suppress(Exception):
        _ler_latex(r"\frac{1}{2}")


_aquecer()


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
    # `|` precisa: no `parse_expr` ele é o `Or` lógico, e `q |E|` saía `None`.
    #
    # A decisão sai do texto ORIGINAL porque a normalização logo abaixo apaga
    # justamente as marcas em que ela se baseia: `E_{tot} = E_{cin} + E_{pot}`
    # perde os `_{`, e sem eles a expressão iria para o `parse_expr`, que não
    # aceita `=` — equação inteira virando `None`.
    usar_latex = "\\" in s or "_{" in s or "^{" in s or "|" in s

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
            e = _com_limite(_ler_latex, blindado)
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
        locais = _declarados(claim)
        a, b = parse(claim.lhs, locais), parse(claim.rhs or "", locais)
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
