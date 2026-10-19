"""O corretor do G2 (DOC-12 §3.7.4): extrai a resposta da geração e a compara com o
gabarito, por tipo. É um adaptador sobre o `parse` de `phifm.verify.symbolic`, com uma
camada de Física por cima:

- `\\pi` vira π: o parser LaTeX o devolve como símbolo livre.
- `e^{x}` com expoente simbólico vira exp(x): `e^{-\\beta E}` é o fator de Boltzmann. Com
  expoente numérico, `e` fica símbolo, porque `e^2` costuma ser carga ao quadrado.
- Uma "função" desconhecida de um argumento vira produto: o parser lê `\\mu (g - a)` como
  μ aplicado a (g − a).

Os vereditos são PASS, FAIL, SEM_RESPOSTA, INCONCLUSIVE e ERROR, e só PASS conta como
acerto (§3.7.4). Toda comparação tem um teto de `LIMITE_S`; estourar dá INCONCLUSIVE.
No Windows a thread que estourou segue até terminar sozinha (ver
`phifm.verify.symbolic._com_limite`). O harness corrige em processos separados para
recuperar a CPU.

Este é o código que o dev itera (até 5 vezes, §3.7.3) e que o congelamento fixa por hash.
"""
from __future__ import annotations

import math
import random
import re
import threading
from collections.abc import Callable
from enum import StrEnum

import mpmath as mp
import sympy as sp
from sympy.core.function import AppliedUndef

from phifm.verify.symbolic import parse

LIMITE_S = 10.0
N_AMOSTRAS = 20
TOL_EXPRESSAO = 1e-9   # relativa, entre as duas expressões avaliadas no mesmo ponto


def _aquecer() -> None:
    """Carrega o ANTLR do parser LaTeX na thread principal, sem teto. A primeira chamada
    leva segundos, e o `parse` do verify a roda numa thread com teto de 5 s no Windows. Sem
    isto, a PRIMEIRA resposta corrigida podia sair INCONCLUSIVE por tempo, e a thread
    abandonada seguia importando o numpy em paralelo com a principal."""
    import numpy  # noqa: F401
    from sympy.parsing.latex import parse_latex

    parse_latex(r"\frac{1}{2}")


_aquecer()


class Veredito(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SEM_RESPOSTA = "sem_resposta"
    TRUNCADO = "truncado"          # quem marca é o harness, pela parada por teto de fichas
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


# ── extração ─────────────────────────────────────────────────────────────────


def _chaves(texto: str, ini: int) -> str | None:
    """O conteúdo entre a chave aberta em `ini` e a que a fecha, ou None."""
    nivel = 0
    for j in range(ini, len(texto)):
        if texto[j] == "{":
            nivel += 1
        elif texto[j] == "}":
            nivel -= 1
            if nivel == 0:
                return texto[ini + 1:j]
    return None


def extrair_boxed(texto: str) -> str | None:
    """O conteúdo do ÚLTIMO `\\boxed{…}` (ou `\\fbox{…}`) de chaves balanceadas."""
    for m in reversed(list(re.finditer(r"\\(?:boxed|fbox)\s*", texto))):
        j = m.end()
        if j < len(texto) and texto[j] == "{":
            dentro = _chaves(texto, j)
            if dentro is not None:
                return dentro.strip()
        elif j < len(texto) and m.group().endswith(" "):
            return texto[j:].split()[0].strip("$.")   # `\boxed 5`
    return None


def extrair_resposta(texto: str) -> str | None:
    """O último `\\boxed{}`; senão, o que vem depois da última ocorrência de "final answer
    is" (até o fim da linha); senão, None (= sem resposta)."""
    if (b := extrair_boxed(texto)) is not None:
        return b or None
    achados = list(re.finditer(r"final answer is\s*:?\s*", texto, flags=re.I))
    if achados:
        resto = texto[achados[-1].end():].split("\n")[0].strip()
        resto = resto.strip().strip("$").strip().rstrip(".").strip().strip("$").strip()
        return resto or None
    return None


# ── normalização e leitura ───────────────────────────────────────────────────

_ENVOLTORIOS = (("$$", "$$"), ("$", "$"), ("\\[", "\\]"), ("\\(", "\\)"))
_SUMIR = (r"\left", r"\right", r"\displaystyle", r"\!", r"\,", r"\;", r"\:", r"\quad",
          r"\qquad", r"\big", r"\Big", r"\bigg", r"\Bigg")
_DECORACAO = re.compile(r"\\(?:vec|hat|bar|mathbf|boldsymbol|bm|mathit|tilde|overline|"
                        r"overrightarrow)\s*\{([^{}]*)\}")
_TEXTO = re.compile(r"\\(?:mathrm|text|textrm|rm|mbox|operatorname|mathsf|textit|"
                    r"textbf)\s*\{([^{}]*)\}")


def limpar(s: str) -> str:
    s = s.strip()
    for a, b in _ENVOLTORIOS:
        if s.startswith(a) and s.endswith(b) and len(s) >= len(a) + len(b):
            s = s[len(a):len(s) - len(b)].strip()
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for x in _SUMIR:
        s = s.replace(x, " ")
    s = s.replace("−", "-").replace("~", " ").replace("\\cdot", "*").replace("×", "\\times")
    s = s.replace("\\{", "(").replace("\\}", ")")       # chave escapada como parêntese
    s = _DECORACAO.sub(r"\1", s)
    s = re.sub(r"\\mathrm\s*\{\s*e\s*\}", "e", s)      # `\mathrm{e}^{x}` é o e de Euler
    s = re.sub(r"\\mathrm\s*\{\s*d\s*\}", "d", s)
    s = re.sub(r"\\mathrm\s*\{\s*i\s*\}", "i", s)
    # Ponto de derivada temporal vira subscrito nomeado: `\dot{\theta}_{1}` → `\theta_{dot1}`,
    # `\ddot{x}` → `x_{ddot}`. Sem isto o parser lê `dot*theta` e θ e θ̇ colapsam.
    s = re.sub(r"\\(ddot|dot)\s*\{\s*([^{}]*?)\s*\}\s*_\s*\{\s*(\w+)\s*\}", r"\2_{\1\3}", s)
    s = re.sub(r"\\(ddot|dot)\s*\{\s*([^{}]*?)\s*\}\s*_\s*(\w)", r"\2_{\1\3}", s)
    s = re.sub(r"\\(ddot|dot)\s*\{\s*([^{}]*?)\s*\}", r"\2_{\1}", s)
    # Média: `\langle E \rangle` → `E_{avg}`, quando dentro há um símbolo só.
    s = re.sub(r"\\langle\s*(\\?[A-Za-z]+(?:_\{?\w+\}?)?)\s*\\rangle", r"\1_{avg}", s)
    # `\operatorname{coth}` → `\coth`, que o parser conhece como função.
    s = re.sub(r"\\operatorname\s*\{\s*(sinh|cosh|tanh|coth|sech|csch|arcsin|arccos|"
               r"arctan|arsinh|arcosh|artanh|sin|cos|tan|cot|sec|csc|exp|ln|log)\s*\}",
               r"\\\1", s)
    # Linha e estrela viram subscrito nomeado, que o parser atravessa como identificador:
    # `\gamma'` → `\gamma_{prime}`, `p^{*}` → `p_{star}`.
    s = re.sub(r"(?<=[A-Za-z}])\s*''", "_{dprime}", s)
    s = re.sub(r"(?<=[A-Za-z}])\s*'", "_{prime}", s)
    s = re.sub(r"\^\s*\{\s*\*\s*\}|\^\s*\*", "_{star}", s)
    # Módulo: o `parse` do verify DESCARTA `\left|E\right|` em silêncio (`q|E|` → `q`).
    # √((X)²) é o módulo exato nos reais e o parser o lê.
    s = re.sub(r"\|([^|]+)\|(?!\s*_)", r"\\sqrt{(\1)^{2}}", s)
    # `d` solto antes de um comando (`d \sin\theta`, a rede de difração) vira o token de
    # diferencial do parser (`dsin`); o `\,` o separa.
    s = re.sub(r"(?<![A-Za-z\\_^])(?<!_\{)(?<!\^\{)d(\s*)(?=\\[A-Za-z])", r"d\\,\1", s)
    return s.strip().rstrip(".").strip()


def dividir_nivel_zero(s: str, padrao: str) -> list[str]:
    """Divide `s` em cada ocorrência de `padrao` FORA de {}, () e [], para que o `=` de
    `\\sum_{n=1}` ou a vírgula de `(a, b)` não contem como separador."""
    nivel, cortes, ultimo = 0, [], 0
    marcas = {m.start(): m.end() for m in re.finditer(padrao, s)}
    j = 0
    while j < len(s):
        if nivel == 0 and j in marcas:
            cortes.append(s[ultimo:j])
            ultimo = j = marcas[j]
            continue
        if s[j] in "{([":
            nivel += 1
        elif s[j] in "})]":
            nivel = max(0, nivel - 1)
        j += 1
    cortes.append(s[ultimo:])
    return cortes


def _fisica(e: sp.Expr) -> sp.Expr:
    """π, exp e o "μ (…)" que o parser lê como função (ver o docstring do módulo)."""
    e = e.xreplace({sp.Symbol("pi"): sp.pi})
    trocas = {}
    for f in e.atoms(AppliedUndef):
        if len(f.args) == 1:
            trocas[f] = sp.Symbol(f.func.__name__) * f.args[0]
    if trocas:
        e = e.xreplace(trocas)
    euler = sp.Symbol("e")

    def _exp(x):
        return isinstance(x, sp.Pow) and x.base == euler and bool(x.exp.free_symbols)

    return e.replace(_exp, lambda x: sp.exp(x.exp))


# Em resposta de Física, `I` é corrente, momento de inércia ou intensidade, não a unidade
# imaginária (que se escreve `i` ou `\mathrm{i}`). O `parse` do verify mantém o `I` do SymPy
# de propósito (onda plana); aqui ele é símbolo.
_LOCAIS = {"I": sp.Symbol("I")}


def expressao(s: str) -> sp.Expr | None:
    e = parse(limpar(s), locais=_LOCAIS)
    if e is None:
        return None
    try:
        return _fisica(e)
    except Exception:
        return None


_IGUAL = r"=|\\approx|\\simeq|\\equiv"


def _depois_do_igual(s: str) -> str:
    """O lado direito do último `=`, `\\approx` ou `\\simeq` de nível zero."""
    partes = dividir_nivel_zero(s, _IGUAL)
    return partes[-1].strip() if len(partes) > 1 else s


def numero(s: str) -> float | None:
    """O valor numérico de uma resposta, ou None. Tira unidade em `\\mathrm{}`/`\\text{}`,
    grau, por cento e separador de milhar; aceita `\\times 10^{n}`, notação e, frações e
    raízes. Com o que sobrar de letras depois do número (unidade solta), vale o número."""
    s = _depois_do_igual(limpar(s))
    # graus e minutos de arco: `64^\circ 10'` (a linha já virou `_{prime}` em `limpar`)
    gm = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*\^\s*\{?\s*\\circ\s*\}?\s*(\d+(?:\.\d+)?)\s*"
                      r"(?:_\{prime\}|')\s*", s)
    if gm:
        return float(gm.group(1)) + float(gm.group(2)) / 60
    s = _TEXTO.sub(" ", s)
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?|\\circ|\\%|%|\\degree", " ", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = re.sub(r"(\d)\s*(?:\\times|\*|x)\s*10\s*\^\s*\{?\s*([+-]?)\s*(\d+)\s*\}?", r"\1e\2\3", s)
    s = re.sub(r"^\s*10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?", r"1e\1", s).strip()
    e = expressao(s)
    if e is not None and not e.free_symbols:
        try:
            v = complex(sp.N(e))
        except Exception:
            v = None
        if v is not None and abs(v.imag) <= 1e-12 * max(1.0, abs(v.real)) and math.isfinite(v.real):
            return v.real
    m = re.match(r"^\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*[A-Za-z\\/^{}()\s\-\d.]*$", s)
    if m:
        return float(m.group(1))
    return None


# ── comparadores ─────────────────────────────────────────────────────────────


def _com_teto(fn: Callable[[], Veredito], segundos: float = LIMITE_S) -> Veredito:
    caixa: dict = {}

    def corpo():
        try:
            caixa["v"] = fn()
        except Exception:
            caixa["v"] = Veredito.ERROR

    t = threading.Thread(target=corpo, daemon=True, name="g2-corretor")
    t.start()
    t.join(segundos)
    return Veredito.INCONCLUSIVE if t.is_alive() else caixa["v"]


def comparar_numero(resposta: str, gabarito: str, rel: float | None = None,
                    abs_: float | None = None, simetrico: bool = False) -> Veredito:
    """|r − g| ≤ abs_ (absoluto) ou ≤ rel·|g| (relativo ao gabarito; com `simetrico`,
    rel·max(|r|, |g|), que é o `math.isclose` do SciBench)."""
    g = numero(gabarito)
    if g is None:
        return Veredito.INCONCLUSIVE
    r = numero(resposta)
    if r is None:
        return Veredito.FAIL
    if abs_ is not None:
        return Veredito.PASS if abs(r - g) <= abs_ * (1 + 1e-9) else Veredito.FAIL
    if rel is None:
        raise ValueError("comparar_numero precisa de rel ou abs_")
    escala = max(abs(r), abs(g)) if simetrico else abs(g)
    if escala == 0:
        return Veredito.PASS if r == g else Veredito.FAIL
    return Veredito.PASS if abs(r - g) <= rel * escala * (1 + 1e-12) else Veredito.FAIL


def _amostrar_iguais(a: sp.Expr, b: sp.Expr, razao: bool = False,
                     positiva: bool = False) -> Veredito:
    """Avalia a e b em N_AMOSTRAS pontos reais positivos a 50 dígitos. Sem `razao`: iguais
    em todos → PASS. Com `razao`: a/b constante e não nula em todos → PASS (e, com
    `positiva`, a constante tem de ser positiva)."""
    simbolos = sorted(a.free_symbols | b.free_symbols, key=str)
    rng = random.Random(17)
    mp.mp.dps = 50
    fa = sp.lambdify(simbolos, a, "mpmath")
    fb = sp.lambdify(simbolos, b, "mpmath")
    valores = []
    for _ in range(N_AMOSTRAS):
        pontos = []
        for _s in simbolos:
            x = rng.uniform(0.2, 5.0)
            while abs(x - 1) < 0.1:
                x = rng.uniform(0.2, 5.0)
            pontos.append(mp.mpf(repr(x)))
        try:
            va, vb = mp.mpc(fa(*pontos)), mp.mpc(fb(*pontos))
        except Exception:
            continue
        if not (mp.isfinite(va.real) and mp.isfinite(vb.real) and mp.isfinite(va.imag)
                and mp.isfinite(vb.imag)):
            continue
        if razao:
            if abs(vb) == 0:
                continue
            valores.append(va / vb)
        else:
            escala = max(abs(va), abs(vb), mp.mpf(1))
            valores.append(abs(va - vb) / escala)
    if len(valores) < N_AMOSTRAS // 2:
        return Veredito.INCONCLUSIVE
    if razao:
        r0 = valores[0]
        if abs(r0) < mp.mpf("1e-30") or (positiva and not r0.real > 0):
            return Veredito.FAIL
        ok = all(abs(v - r0) <= TOL_EXPRESSAO * abs(r0) for v in valores)
    else:
        ok = all(v <= TOL_EXPRESSAO for v in valores)
    return Veredito.PASS if ok else Veredito.FAIL


def _equivalentes(a: sp.Expr, b: sp.Expr) -> Veredito:
    try:
        if a == b or sp.srepr(a) == sp.srepr(b):
            return Veredito.PASS
    except Exception:
        pass
    if not (a.free_symbols | b.free_symbols):
        try:
            va, vb = complex(sp.N(a, 30)), complex(sp.N(b, 30))
            return (Veredito.PASS if abs(va - vb) <= TOL_EXPRESSAO * max(abs(va), abs(vb), 1)
                    else Veredito.FAIL)
        except Exception:
            return Veredito.INCONCLUSIVE
    try:
        if sp.simplify(a - b) == 0:
            return Veredito.PASS
    except Exception:
        pass
    return _amostrar_iguais(a, b)


def texto_normalizado(s: str) -> str:
    """Para o que o parser não lê (termo espectral, notação de estado): a forma sem
    espaços, sem `\\mathrm{}`/`\\text{}` e sem chaves soltas."""
    s = _TEXTO.sub(r"\1", limpar(s))
    return re.sub(r"[\s{}]", "", s)


# Integral, soma, produtório, derivada parcial, limite, nabla, bra-ket: o parser devolve
# uma expressão, mas sem o significado (∂S/∂p de um símbolo vira 0, `\sum` vira produto).
# Duas respostas erradas podiam "coincidir" nesse lixo. Gabarito com isso não é
# verificável por regra: INCONCLUSIVE, e o autoteste o exclui dos dois braços.
_CALCULO = re.compile(r"\\(?:int|iint|iiint|oint|sum|prod|partial|lim|nabla|langle|rangle)"
                      r"(?![A-Za-z])")
_BARRA_DE_AVALIACAO = re.compile(r"\|\s*_")     # `\right|_{r=a}`
# diferencial explícito: `\mathrm{d}`, `{\rm d}`, ou dX/dY (`\frac{d v}{d t}`, `\frac{d}{dx}`);
# `\frac{d \sin\theta}{\lambda}` (a rede de difração) não é derivada e passa
_DIFERENCIAL = re.compile(r"\\mathrm\s*\{\s*d\s*\}|\{\s*\\rm\s+d\s*\}|"
                          r"\\frac\s*\{\s*d\s*(?:\^\s*\{?\s*\d\s*\}?)?\s*[^{}]*\}\s*\{\s*d\s*"
                          r"[A-Za-z\\]")


def verificavel(gabarito: str) -> bool:
    """False para gabarito com cálculo, que a regra não sabe verificar."""
    return (_DIFERENCIAL.search(gabarito) is None
            and _BARRA_DE_AVALIACAO.search(gabarito) is None
            and _CALCULO.search(limpar(gabarito)) is None)


def comparar_expressao(resposta: str, gabarito: str) -> Veredito:
    """Equivalência simbólica com todas as variáveis positivas; um `x =` na frente, de um
    lado ou dos dois, é ignorado. Gabarito que o parser não lê é comparado como texto
    normalizado: igual passa, diferente falha. Gabarito com cálculo (`verificavel`) dá
    INCONCLUSIVE."""
    if not verificavel(gabarito):
        return Veredito.INCONCLUSIVE
    g = expressao(_depois_do_igual(limpar(gabarito)))
    if g is None:
        return (Veredito.PASS if texto_normalizado(resposta) == texto_normalizado(gabarito)
                else Veredito.FAIL)
    r = expressao(_depois_do_igual(limpar(resposta)))
    if r is None:
        return Veredito.FAIL
    return _equivalentes(r, g)


def _lados(s: str) -> tuple[sp.Expr, sp.Expr] | None:
    partes = dividir_nivel_zero(limpar(s), "=")
    if len(partes) != 2:
        return None
    lados = (expressao(partes[0]), expressao(partes[1]))
    return None if None in lados else lados  # type: ignore[return-value]


def comparar_equacao(resposta: str, gabarito: str) -> Veredito:
    """Equivalentes se (L − R) da resposta ÷ (L − R) do gabarito for uma constante não
    nula."""
    if not verificavel(gabarito):
        return Veredito.INCONCLUSIVE
    g = _lados(gabarito)
    if g is None:
        return Veredito.INCONCLUSIVE
    r = _lados(resposta)
    if r is None:
        return Veredito.FAIL
    fr, fg = r[0] - r[1], g[0] - g[1]
    try:
        q = sp.simplify(fr / fg)
        if q.is_number and q != 0 and q.is_finite:
            return Veredito.PASS
    except Exception:
        pass
    return _amostrar_iguais(fr, fg, razao=True)


_RELACAO = re.compile(r"\\geq?|\\leq?|\\geqslant|\\leqslant|\\gt|\\lt|≥|≤|>=|<=|>|<")


def _desigualdade(s: str) -> tuple[sp.Expr, bool] | None:
    """`L op R` como (f, estrita), com f > 0 (ou f ≥ 0) equivalente à desigualdade."""
    s = limpar(s)
    ops = list(_RELACAO.finditer(s))
    if len(ops) != 1:
        return None
    op = ops[0].group()
    lados = (expressao(s[:ops[0].start()]), expressao(s[ops[0].end():]))
    if None in lados:
        return None
    maior = op in (">", ">=", "\\gt", "\\ge", "\\geq", "\\geqslant", "≥")
    estrita = op in (">", "<", "\\gt", "\\lt")
    f = lados[0] - lados[1] if maior else lados[1] - lados[0]
    return f, estrita


def comparar_desigualdade(resposta: str, gabarito: str) -> Veredito:
    """Mesma desigualdade: mesma estrição, e f da resposta ÷ f do gabarito é uma constante
    POSITIVA (multiplicar por negativo inverte o sentido)."""
    g = _desigualdade(gabarito)
    if g is None:
        return Veredito.INCONCLUSIVE
    r = _desigualdade(resposta)
    if r is None or r[1] != g[1]:
        return Veredito.FAIL
    try:
        q = sp.simplify(r[0] / g[0])
        if q.is_number and q.is_finite and q.is_positive:
            return Veredito.PASS
    except Exception:
        pass
    return _amostrar_iguais(r[0], g[0], razao=True, positiva=True)


def _intervalo(s: str):
    s = limpar(s).replace("\\infty", "oo").replace("\\infin", "oo")
    m = re.fullmatch(r"([\[(])(.*)([\])])", s.strip())
    if not m:
        return None
    meio = _partes(m.group(2))
    if len(meio) != 2:
        return None
    return m.group(1), meio[0], meio[1], m.group(3)


def _extremo_igual(a: str, b: str, rel: float) -> Veredito:
    if a.strip().lstrip("+") == b.strip().lstrip("+"):
        return Veredito.PASS
    na, nb = numero(a), numero(b)
    if na is not None and nb is not None:
        if nb == 0:
            return Veredito.PASS if na == 0 else Veredito.FAIL
        return Veredito.PASS if abs(na - nb) <= rel * abs(nb) else Veredito.FAIL
    return comparar_expressao(a, b)


def comparar_intervalo(resposta: str, gabarito: str, rel: float = 1e-2) -> Veredito:
    """Mesmos colchetes e extremos iguais (numéricos a `rel` relativo, simbólicos por
    equivalência). União por `\\cup`, na mesma ordem. Gabarito escrito como desigualdade
    (`N_1 < N/2`) é comparado como desigualdade."""
    if _RELACAO.search(limpar(gabarito)):
        return comparar_desigualdade(resposta, gabarito)
    gs = [_intervalo(p) for p in re.split(r"\\cup|∪", gabarito)]
    if any(x is None for x in gs):
        return Veredito.INCONCLUSIVE
    rs = [_intervalo(p) for p in re.split(r"\\cup|∪", resposta)]
    if len(rs) != len(gs) or any(x is None for x in rs):
        return Veredito.FAIL
    for r, g in zip(rs, gs, strict=True):
        if r[0] != g[0] or r[3] != g[3]:
            return Veredito.FAIL
        for i in (1, 2):
            v = _extremo_igual(r[i], g[i], rel)
            if v is not Veredito.PASS:
                return v
    return Veredito.PASS


def letras(s: str) -> frozenset[str]:
    s = _TEXTO.sub(r" \1 ", limpar(s))
    return frozenset(x.upper() for x in re.findall(r"(?<![A-Za-z])([A-Ja-j])(?![A-Za-z])", s))


def comparar_letras(resposta: str, gabarito: str) -> Veredito:
    g = letras(gabarito)
    if not g:
        return Veredito.INCONCLUSIVE
    return Veredito.PASS if letras(resposta) == g else Veredito.FAIL


def verdade(s: str) -> bool | None:
    t = re.sub(r"[^a-z]", "", _TEXTO.sub(r"\1", limpar(s)).lower())
    if t in ("yes", "true", "correct", "t", "y"):
        return True
    if t in ("no", "false", "incorrect", "f", "n"):
        return False
    return None


def comparar_vf(resposta: str, gabarito: str) -> Veredito:
    g = verdade(gabarito)
    if g is None:
        return Veredito.INCONCLUSIVE
    return Veredito.PASS if verdade(resposta) is g else Veredito.FAIL


def _partes(s: str) -> list[str]:
    """Separa por vírgulas de nível zero (fora de {}, () e [])."""
    return [p.strip() for p in dividir_nivel_zero(s, ",") if p.strip()]


def comparar_varias(resposta: str, gabaritos: list[str],
                    comparadores: list[Callable[[str, str], Veredito]]) -> Veredito:
    """Várias respostas num `\\boxed{}`, separadas por vírgula: todas certas, em qualquer
    ordem (cada gabarito casa com uma resposta diferente)."""
    rs = _partes(limpar(resposta))
    if len(rs) != len(gabaritos):
        return Veredito.FAIL
    livres = list(range(len(rs)))
    pior = Veredito.PASS
    for g, comp in zip(gabaritos, comparadores, strict=True):
        achou = None
        for j in livres:
            v = comp(rs[j], g)
            if v is Veredito.PASS:
                achou = j
                break
            if v is Veredito.INCONCLUSIVE:
                pior = Veredito.INCONCLUSIVE
        if achou is None:
            return Veredito.FAIL if pior is Veredito.PASS else pior
        livres.remove(achou)
    return Veredito.PASS


def com_teto(fn: Callable[[], Veredito], segundos: float = LIMITE_S) -> Veredito:
    """A comparação inteira sob o teto de `LIMITE_S` (INCONCLUSIVE se estourar)."""
    return _com_teto(fn, segundos)
