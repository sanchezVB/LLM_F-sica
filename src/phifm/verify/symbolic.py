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
from phifm.verify.bus import (
    Claim,
    Motivo,
    Papel,
    Verdict,
    VerificationResult,
    tipo_da_afirmacao,
)

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
    return {nome_canonico(n): sp.Symbol(nome_canonico(n))
            for n in (claim.context.get("dimensions") or {})}


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


# ── Unicode, variantes gregas e números absurdos ──────────────────────────
#
# Texto puro com Unicode (`½mv²`) ia para o `parse_expr`, que lia `mv²` como um
# símbolo só: a resposta certa reprovava (ADR-0005 §4.7). A tradução emite
# LaTeX, e não nomes nus, para a expressão ir à rota do ANTLR, que separa `m v`.
# O que sobrar fora do ASCII dá None — ler `é` como símbolo é pior que recusar.
_UNICODE_SIMPLES = {
    "×": r"\times ", "·": r"\cdot ", "⋅": r"\cdot ", "∙": r"\cdot ", "−": "-", "–": "-",
    "π": r"\pi ", "ħ": r"\hbar ", "ℏ": r"\hbar ", "∞": r"\infty ", "µ": r"\mu ", "≈": r"\approx ",
    "′": "'", "ϵ": r"\epsilon ", "ϑ": r"\theta ", "ϕ": r"\phi ", "ϱ": r"\rho ", "ϰ": r"\kappa ",
    "α": r"\alpha ", "β": r"\beta ", "γ": r"\gamma ", "δ": r"\delta ", "ε": r"\epsilon ",
    "ζ": r"\zeta ", "η": r"\eta ", "θ": r"\theta ", "ι": r"\iota ", "κ": r"\kappa ",
    "λ": r"\lambda ", "μ": r"\mu ", "ν": r"\nu ", "ξ": r"\xi ", "ρ": r"\rho ", "σ": r"\sigma ",
    "τ": r"\tau ", "υ": r"\upsilon ", "φ": r"\phi ", "χ": r"\chi ", "ψ": r"\psi ", "ω": r"\omega ",
    "Γ": r"\Gamma ", "Δ": r"\Delta ", "Θ": r"\Theta ", "Λ": r"\Lambda ", "Ξ": r"\Xi ",
    "Π": r"\Pi ", "Σ": r"\Sigma ", "Υ": r"\Upsilon ", "Φ": r"\Phi ", "Ψ": r"\Psi ", "Ω": r"\Omega ",
}
_FRACOES = {"½": (1, 2), "⅓": (1, 3), "⅔": (2, 3), "¼": (1, 4), "¾": (3, 4)}
_SOBRESCRITOS = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+", strict=True))
_SUBSCRITOS = dict(zip("₀₁₂₃₄₅₆₇₈₉", "0123456789", strict=True))
_ESPACOS = re.compile("[  -   　]")


def traduzir_unicode(s: str) -> str | None:
    """Unicode de Física para LaTeX, ou None se sobrar algo fora do ASCII."""
    if s.isascii():
        return s
    s = _ESPACOS.sub(" ", s).replace("​", "")
    # `2½` é dois e meio, não 2·½.
    s = re.sub("(\\d)\\s*([½⅓⅔¼¾])",
               lambda m: "({}+\\frac{{{}}}{{{}}})".format(m.group(1), *_FRACOES[m.group(2)]), s)
    s = re.sub("[½⅓⅔¼¾]", lambda m: "\\frac{{{}}}{{{}}}".format(*_FRACOES[m.group(0)]), s)
    s = re.sub("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+",
               lambda m: "^{" + "".join(_SOBRESCRITOS[c] for c in m.group(0)) + "}", s)
    s = re.sub("[₀₁₂₃₄₅₆₇₈₉]+",
               lambda m: "_{" + "".join(_SUBSCRITOS[c] for c in m.group(0)) + "}", s)
    s = "".join(_UNICODE_SIMPLES.get(c, c) for c in s)
    return s if s.isascii() else None


# `\varepsilon` e `\epsilon` são a mesma letra; `\varepsilon_0` não casava com
# o `epsilon_0` da tabela de dimensões. `\varpi` fica de fora de propósito: virar
# `\pi` o transformaria na constante π. `\varsigma` também, por não ser comum.
_VARIANTE = re.compile(r"(?<![A-Za-z])(\\?)var(epsilon|theta|phi|rho|kappa)(?![A-Za-z])")


def nome_canonico(nome: str) -> str:
    """A forma canônica de um nome com variante grega: `varepsilon_0` → `epsilon_0`.

    Vale para o texto e para as chaves do contexto (`dimensions`, a variável
    de um limite), que precisam casar com o símbolo lido.
    """
    return _VARIANTE.sub(r"\1\2", nome)


# Número que nenhuma resposta de Física escreve e que trava quem o avalia: uma
# torre de expoentes inteiros (`10^{10^{8}}`) ou um expoente de cinco dígitos.
# O `parse` não os avalia, mas o `simplify` e o numérico sim — sem teto no
# numérico, e com memória de gigabytes (ADR-0005).
_NUMERO_ABSURDO = re.compile(
    r"\d\s*(?:\^|\*\*)\s*[{(]?\s*\d+\s*[})]?\s*(?:\^|\*\*)\s*[{(]?\s*\d"
    r"|(?:\^|\*\*)\s*[{(]?\s*[+-]?\d{5,}"
    r"|\d{60,}"
)


# ⚠️ Subscrito de um token, sem chaves, depois de uma macro: `\epsilon_0`. A
# normalização de subscritos só pega o achatado de dois caracteres ou mais, e a
# blindagem não pega o que vem depois de `\`, então o ANTLR o lê como
# `epsilon_{0}` — enquanto `\epsilon_{0}` vira `epsilon_0`. Dois símbolos para a
# mesma grandeza, e a resposta certa reprovava (achado em 2026-10-24, nos testes
# do ADR-0005). O lugar natural seria `core.latex.subscritos`, mas ele está no
# congelamento do G2; aqui só se achata um nome inteiro de um token só.
_SUBSCRITO_DO_ANTLR = re.compile(r"([A-Za-z]+)_\{([A-Za-z0-9]+)\}")


def _achatar_subscrito_do_antlr(e: sp.Basic) -> sp.Basic:
    trocas = {}
    for s in e.atoms(sp.Symbol):
        m = _SUBSCRITO_DO_ANTLR.fullmatch(s.name)
        if m:
            trocas[s] = sp.Symbol(f"{m.group(1)}_{m.group(2)}")
    return e.xreplace(trocas) if trocas else e


def parse(expr: str, locais: dict | None = None) -> sp.Expr | None:
    """Aceita LaTeX ou sintaxe SymPy. Devolve ``None`` se não parsear.

    `locais` são símbolos declarados pelo chamador e têm precedência sobre
    `_FISICA` e sobre o namespace do SymPy — é assim que quem quer `I` como
    corrente elétrica, e não como unidade imaginária, consegue dizer isso.
    """
    s = expr.strip().strip("$").strip()
    if not s or _NUMERO_ABSURDO.search(s):
        return None
    s = traduzir_unicode(s)
    if s is None:
        return None
    s = nome_canonico(s)
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
                # O marcador também pode ter virado nome de FUNÇÃO (`E_0(x)`), e
                # a letra-base dele muda conforme o texto.
                cabecas = {f: sp.Function(marcadores[f.func.__name__])(*f.args)
                           for f in e.atoms(AppliedUndef) if f.func.__name__ in marcadores}
                if cabecas:
                    e = e.xreplace(cabecas)
            return _achatar_subscrito_do_antlr(e)
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


def _operadores(claim: Claim) -> tuple[bool, bool]:
    """(no item, só na resposta).

    ⚠️ Separar os dois é o que impede a resposta de desligar verificadores:
    um `\\sigma` só na resposta fazia o numérico se declarar inaplicável e
    sumir do vetor, e a resposta errada saía sem nenhum FAIL (ADR-0005). O
    item é o gabarito e o contexto; a resposta não decide se há operadores.
    """
    no_item = bool(claim.context.get("noncommutative")) or bool(
        _NONCOMMUTATIVE.search(claim.rhs or "")
    )
    na_resposta = not no_item and bool(_NONCOMMUTATIVE.search(claim.lhs))
    return no_item, na_resposta


def _cabecas(e: sp.Basic) -> set[str]:
    """Nomes das funções indefinidas (`f(x)`, `m(1)`)."""
    return {f.func.__name__ for f in e.atoms(AppliedUndef)}


def _reinterpretar_produtos(resposta: sp.Basic, gabarito: sp.Basic) -> sp.Basic:
    """`m(g + a)` → m·(g + a), quando `m` é um SÍMBOLO do gabarito e não uma função dele.

    A gramática lê letra seguida de parêntese como aplicação de função. Quando
    o nome é uma grandeza do gabarito, a leitura de produto é a da Física — e
    a de função abria um atalho: `m v^{2} m(1)` saía com o simbólico e o
    numérico indecisos e o dimensional aprovando (ADR-0005). Fica função o que
    o gabarito também usa como função, e o que ele não usa (`f`, `V(r)`), que
    cai na regra de vocabulário.
    """
    simbolos = {s.name for s in gabarito.free_symbols}
    funcoes = _cabecas(gabarito)

    def _eh_produto(x: sp.Basic) -> bool:
        return (isinstance(x, AppliedUndef) and len(x.args) == 1
                and x.func.__name__ in simbolos and x.func.__name__ not in funcoes)

    return resposta.replace(_eh_produto, lambda x: sp.Symbol(x.func.__name__) * x.args[0])


def _vocabulario_extra(resposta: sp.Basic, gabarito: sp.Basic, claim: Claim) -> list[str]:
    """Símbolos e funções da resposta que o gabarito e o contexto não têm.

    Funções e símbolos em conjuntos separados: `m(1)` com `m` só como símbolo
    no gabarito é vocabulário novo. O contexto conta pelas chaves de
    `dimensions` e por `simbolos`, nunca pelo texto do enunciado — palavras de
    uma letra do português liberariam símbolos.
    """
    declarados = {nome_canonico(n) for n in (
        *(claim.context.get("dimensions") or {}), *(claim.context.get("simbolos") or ()))}
    extras = ({s.name for s in resposta.free_symbols} - {s.name for s in gabarito.free_symbols}
              - declarados)
    extras |= {f"{c}()" for c in _cabecas(resposta) - _cabecas(gabarito) - declarados}
    return sorted(extras)


def _ler_os_dois_lados(claim: Claim, locais: dict, no_item: bool, na_resposta: bool):
    """Lê gabarito e resposta, nessa ordem, e devolve ``(a, b, abstencao)``.

    O gabarito primeiro: se ele não lê, a culpa é do item, e a resposta nem
    importa. Depois os tipos: o `parse` devolve `Eq`, tupla e booleano apesar
    da anotação, e o numérico comparava o booleano de dois `Eq` — qualquer par
    de equações passava (ADR-0005).
    """
    b = parse(claim.rhs or "", locais)
    if b is None:
        return None, None, (Motivo.GABARITO_ILEGIVEL, "não foi possível parsear o gabarito (rhs)")
    if not isinstance(b, sp.Expr):
        return None, None, (Motivo.GABARITO_INCOMPARAVEL,
                            f"o gabarito não é uma expressão: {type(b).__name__}")
    a = parse(claim.lhs, locais)
    if a is None:
        # Com operadores no item, a recusa do `\hat` na resposta é do item.
        motivo = Motivo.OPERADORES if no_item else Motivo.RESPOSTA_ILEGIVEL
        return None, None, (motivo, "não foi possível parsear a resposta (lhs)")
    if not isinstance(a, sp.Expr):
        return None, None, (Motivo.RESPOSTA_INCOMPARAVEL,
                            f"a resposta não é uma expressão: {type(a).__name__} "
                            "contra uma expressão")
    if na_resposta:
        return None, None, (Motivo.OPERADORES_NA_RESPOSTA,
                            "marcador de operador só na resposta — o item não tem operadores")
    return _reinterpretar_produtos(a, b), b, None


def _equals_semeado(a: sp.Expr, b: sp.Expr):
    """`equals` sorteia pontos com o RNG global do SymPy: o mesmo Claim dava
    INCONCLUSIVE numa chamada e FAIL na seguinte. Semente fixa a cada chamada
    torna o verify determinístico (DOC-10 §10)."""
    from sympy.core.random import seed

    seed(17)
    return a.equals(b)


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
    papel = Papel.EQUIVALENCIA

    def applicable(self, claim: Claim) -> bool:
        # Numa relação (`E = m c²`), comparar os lados como símbolos
        # independentes não tem sentido (ADR-0005 §4.1).
        return claim.rhs is not None and tipo_da_afirmacao(claim) == "equivalencia"

    def _abster(self, motivo: Motivo, evidencia: str) -> VerificationResult:
        return VerificationResult(Verdict.INCONCLUSIVE, self.id, 0.0, evidencia, motivo=motivo)

    def verify(self, claim: Claim) -> VerificationResult:
        locais = _declarados(claim)
        no_item, na_resposta = _operadores(claim)
        a, b, abstencao = _ler_os_dois_lados(claim, locais, no_item, na_resposta)
        if abstencao is not None:
            return self._abster(*abstencao)
        extras = _vocabulario_extra(a, b, claim)

        # ── comparação estrutural (barata, decide a maioria dos casos) ────
        # ⚠️ Não com operadores no item: na rota do `parse_expr`, `A*B` e
        # `B*A` saem na mesma ordem canônica e passavam por idênticos.
        if not no_item:
            try:
                if a == b or sp.srepr(a) == sp.srepr(b):
                    return VerificationResult(Verdict.PASS, self.id, 1.0, "idênticas estruturalmente")
            except Exception:
                pass

        motivo_final = Motivo.INDECIDIVEL
        try:
            d = _com_limite(sp.simplify, a - b)
        except _Timeout:
            d, motivo_final = None, Motivo.TEMPO
        except Exception:
            d, motivo_final = None, Motivo.EXCECAO_NA_RESPOSTA

        if d is not None and d == 0:
            if no_item:
                # Ver docstring: o SymPy comutou o que talvez não comute.
                return self._abster(
                    Motivo.OPERADORES,
                    "expressão contém operadores/matrizes e o SymPy assume "
                    "comutatividade — diferença nula NÃO prova equivalência",
                )
            return VerificationResult(Verdict.PASS, self.id, 1.0, "simplify(a−b) = 0")

        eq = None
        if d is not None:
            try:
                eq = _com_limite(_equals_semeado, a, b)
            except _Timeout:
                motivo_final = Motivo.TEMPO
            except Exception:
                motivo_final = Motivo.EXCECAO_NA_RESPOSTA

        if eq is True:
            if no_item:
                return self._abster(
                    Motivo.OPERADORES, "equals() confirmou, mas o item tem operadores"
                )
            return VerificationResult(Verdict.PASS, self.id, 0.95, "equals() confirmou")
        if eq is False or (d is not None and d != 0):
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
                # `d` já vem do simplify com teto; um segundo simplify aqui,
                # sem teto, era um caminho para a resposta travar o verificador.
                return VerificationResult(
                    Verdict.FAIL, self.id, 0.95, "equals() refutou",
                    counterexample=f"a−b = {d}",
                )

        if extras:
            return self._abster(
                Motivo.VOCABULARIO,
                f"a resposta usa o que o gabarito e o contexto não têm: {', '.join(extras)}",
            )
        if no_item:
            return self._abster(Motivo.OPERADORES, "o item tem operadores e a cascata não decidiu")
        evidencia = {
            Motivo.TEMPO: f"simplify ou equals não concluiu em {TIMEOUT_S}s — delegar ao numérico",
            Motivo.EXCECAO_NA_RESPOSTA: "o simplify ou o equals levantou exceção com esta resposta",
        }.get(motivo_final, "cascata simbólica não decidiu — delegar ao numérico "
                            "(indecidibilidade de Richardson não é falha do modelo)")
        return self._abster(motivo_final, evidencia)
