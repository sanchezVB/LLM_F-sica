"""O corretor do G2 (DOC-12 §3.7.4): extrai a resposta da geração e a compara com o
gabarito, por tipo.

**O parser é estrito.** O `parse_latex` do SymPy, no modo padrão, devolve só o PREFIXO
que conseguiu ler: `A, B` vira `A`, e `X \\to Y` vira `X`. Uma resposta com duas
alternativas passava sempre que a primeira estava certa (28 de 31 casos no dev). Aqui o
LaTeX é lido com `strict=True`, e o que não é lido por inteiro não vale. A normalização de
subscritos é a do projeto (`core.latex.subscritos`), a mesma do `parse` do verify. O
adaptador tem o próprio parser porque o verify lê no modo frouxo.

Camada de Física sobre o parser:
- π, o número de Euler com expoente simbólico (fator de Boltzmann), `I` como símbolo;
- `\\mu (g - a)` como produto;
- `f(0)` como o valor de f num ponto (antes virava f·0 = 0);
- função de vários argumentos, e δ de Dirac, como uma pseudofunção não linear fixa;
- variantes gregas (`\\varepsilon` = `\\epsilon`), subscrito colado (`k_BT` = k_B·T), ponto,
  linha, módulo, raízes e Unicode.

Os vereditos são PASS, FAIL, SEM_RESPOSTA, TRUNCADO, INCONCLUSIVE e ERROR, e só PASS conta
(§3.7.4). A cascata é: estrutural; amostragem numérica em reais positivos a 50 dígitos; e
`simplify`, só quando a amostragem não decide. Toda comparação tem teto de `LIMITE_S`
segundos de relógio; quem estoura é refeito uma vez antes de virar INCONCLUSIVE.

Este é o código que o dev itera e que o congelamento fixa por hash. Toda mudança fica
registrada no DOC-12 §3.7.4.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
import threading
from collections.abc import Callable
from enum import StrEnum

import mpmath as mp
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.parsing.latex import parse_latex

from phifm.core.latex.subscritos import blindar_subscritos, normalizar_subscritos
from phifm.verify.symbolic import parse as parse_verify

LIMITE_S = 10.0
N_AMOSTRAS = 20
TOL_EXPRESSAO = 1e-9   # relativa, entre as duas expressões avaliadas no mesmo ponto


def _aquecer() -> None:
    """Carrega o ANTLR na thread principal, sem teto: a primeira chamada leva segundos, e
    sem isto a PRIMEIRA resposta corrigida podia estourar o teto."""
    import numpy  # noqa: F401

    parse_latex(r"\frac{1}{2}", strict=True)


_aquecer()


class Veredito(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SEM_RESPOSTA = "sem_resposta"
    TRUNCADO = "truncado"          # quem marca é o harness, pela parada por teto de fichas
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


# ── extração ─────────────────────────────────────────────────────────────────
#
# Política (rodada 4 do estresse, 2026-10-23): vale a ÚLTIMA caixa não vazia, como no juiz
# oficial do OlympiadBench e do UGPhysics. A frase "final answer is" só é lida quando NÃO
# há caixa. Nas rodadas 2 e 3, a frase depois da caixa podia vencê-la, e cada rodada achou
# furos nos dois sentidos: o comentário hesitante vencia a caixa certa, e a correção
# legítima perdia para a caixa intermediária. O few-shot termina sempre em
# `The final answer is $\boxed{…}$`.

# Problema novo inventado depois da resposta (o servidor só para em `\nProblem:`).
# FORTE, corta sozinho: `Problem:`, `Problem 2:`, `## Problem 2`, o eco do cabeçalho do
# documento e as linhas de resposta do prompt, cada uma INTEIRA como `g2_prompt` a gera (um
# teste confere as duas listas). FRACO: `Question:`, `Example 3:`, `Exercise 1.2`,
# `Problem 2.`, `Q:`. Esses só cortam se depois vier uma linha `Solution:` ou um eco; sem
# isso, é o raciocínio ("Question: is this consistent?"). `[ \t]` e não `\s`: `\n\s*` era
# quadrático.
_PROBLEMA_FORTE = re.compile(
    r"^[ \t]*(?:#+[ \t]*|\*\*)?(?:Problem|PROBLEM)"
    r"(?:[ \t]*\d+(?:\.\d+)*[ \t]*\**[ \t]*(?::|$)|[ \t]*\**[ \t]*:)", re.M)
_PROBLEMA_FRACO = re.compile(
    r"^[ \t]*(?:#+[ \t]*|\*\*)?(?:Problem|PROBLEM|Question|Exercise|Example|Q)"
    r"[ \t]*(?:#?\d+(?:\.\d+)*|[IVX]+)?[ \t]*(?:\([^)\n]*\))?[ \t]*\**[ \t]*[:.\-)]", re.M)
_ECO = re.compile(
    r"The following are physics problems with step-by-step solutions"
    r"|Each solution ends with: The final answer is"
    r"|^[ \t]*(?:"
    r"The answer is a number(?: in [^\n]*?)?; put only the number in \\boxed\{\}\."
    r"|The answer has \d+ equations; put all of them in one \\boxed\{\}, separated by commas\."
    r"|The answer is (?:a numerical value|an expression|an equation|an interval)\."
    r"(?:[ \t]+This problem has \d+ answers; put all of them in one \\boxed\{\}, separated by "
    r"commas\.)?(?:[ \t]+Give the unit outside the \\boxed\{\}\.)?"
    r"|Answer with the option letter\(s\)\.|Answer True or False\."
    r"|Give the final answer as a single symbolic expression in the given quantities, in one "
    r"\\boxed\{\}\."
    r"|(?:The unit of the answer is [^\n]*?\. )?Give only the number in \\boxed\{\}\."
    r")[ \t]*$", re.M)
_SOLUCAO = re.compile(r"^[ \t]*(?:\*\*)?Solution(?:\*\*)?[ \t]*:", re.M)
_LINHA_EM_BRANCO = re.compile(r"\n[ \t]*\n")
# "The final answer is", e também "The answer is" (iteração 1 do dev: "The answer is True."
# sem caixa virava sem resposta); só vale sem caixa e se trouxer valor (`_ler_frase`)
_FRASE_FINAL = re.compile(r"(?:final(?:\*\*|__)?\s+)?answers?(?:\*\*|__)?\s*(?:is|are)\b\s*:?\s*",
                          re.I)
_MATEMATICA = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|\$(.+?)\$"
                         r"|\\begin\{(?:equation|align)\*?\}(.+?)\\end\{(?:equation|align)\*?\}",
                         re.S)
# condição ou contrafactual que GOVERNA a frase: no começo da oração dela, ou de uma oração
# anterior ligada por vírgula ("Had we neglected recoil, the final answer is…"). "Note
# that", "does not depend", "Without loss of generality, …, and the final answer is" não
# governam (rodada 4).
_CONDICAO = re.compile(
    r"(?:[^,]*,\s*)*?(?:\*\*)?\s*(?:if|had|would|could|should|suppose|supposing|assuming|unless|"
    r"whether|without|naive|naively)\b[^,]*,?\s*(?:then\s*,?\s*)?(?:the\s+)?(?:\*\*)?\s*", re.I)
# entre a frase e a matemática, só enchimento (com o símbolo da grandeza e rótulos)
_ENCHIMENTO = re.compile(
    r"(?:\s|,|:|=|≈|~|∼|\\approx|\\sim|\*\*|therefore|thus|hence|then|simply|given by|equal to|"
    r"approximately|about|roughly|exactly|just|clearly|indeed|nearly|that|respectively|"
    r"\\?[A-Za-z]+(?:_\{?\w+\}?)?\s*(?:=|≈|\\approx)|\(\s*\w{1,3}\s*\)|\b\w\))*", re.I)
# logo depois do valor: dúvida ou alternativa
_DEPOIS_HESITANTE = re.compile(r"\s*(?:[^\s\w$]{0,2}\s*)?(?:\?|or\b|than\b|not\b|vs\.?|versus\b|"
                               r"instead\b|either\b|but\b|however\b|wait\b)", re.I)
# no resto da sentença: outro valor ligado ao primeiro, ou uma condição
_OUTRO_VALOR = re.compile(r"(?:,|;|\(|\bor\b|\band\b|\bthan\b|\bnot\b)\s*(?:[A-Za-z]+\s+){0,3}"
                          r"\$[^$]*\d", re.I)
_CONDICAO_DEPOIS = re.compile(r"\b(?:if|unless|otherwise|assuming|provided|instead)\b|\?", re.I)
# a sentença seguinte retrata a frase
_RETRATACAO = re.compile(
    r"\s*(?:\*\*)?\s*(?:no\b(?:\s*,|\s+wait)|wait\b|oops\b|hold on\b|but\b|however\b|"
    r"correction\s*[:!]|on second thought|let me (?:re|double)|i made|that'?s wrong|"
    r"this is wrong|actually,?\s+(?:it|the|this) (?:is|should))", re.I)
_PALAVRA_RESPOSTA = re.compile(r"(?:yes|no|true|false|correct|incorrect|option\s+[A-J]|"
                               r"\(?[A-J]\)?)\.?", re.I)
_CAIXA_VAZIA = re.compile(r"\s*(?:\.\.\.|…|\\[lcv]?dots|\\text\s*\{\s*(?:answer|\.\.\.)?\s*\}|"
                          r"answer|\?+)?\s*", re.I)


def _chaves(texto: str, ini: int) -> int | None:
    """O índice da chave que fecha a aberta em `ini`, ou None."""
    nivel = 0
    for j in range(ini, len(texto)):
        if texto[j] == "{":
            nivel += 1
        elif texto[j] == "}":
            nivel -= 1
            if nivel == 0:
                return j
    return None


def _pares_de_chaves(texto: str) -> dict[int, int]:
    """{abre: fecha} de todas as chaves balanceadas, numa passada só. Com uma `_chaves` por
    caixa, cada `\\boxed{` sem fechar varria o texto até o fim, e o corte da continuação
    ficou quadrático (rodada 4: 15 s fora do teto numa geração degenerada)."""
    pilha, pares = [], {}
    for j, ch in enumerate(texto):
        if ch == "{":
            pilha.append(j)
        elif ch == "}" and pilha:
            pares[pilha.pop()] = j
    return pares


def _caixas_com_fim(texto: str) -> list[tuple[int, int, str]]:
    """(início, fim, conteúdo) de cada `\\boxed{…}`/`\\fbox{…}` fechada, em ordem."""
    pares = _pares_de_chaves(texto)
    achadas = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*", texto):
        j = m.end()
        if j < len(texto) and texto[j] == "{":
            if j in pares:
                achadas.append((m.start(), pares[j] + 1, texto[j + 1:pares[j]].strip()))
        elif j < len(texto) and m.group().endswith(" ") and texto[j:].split():
            palavra = texto[j:].split()[0]                    # `\boxed 5`
            achadas.append((m.start(), j + len(palavra), palavra.strip("$.,")))
    return achadas


def caixas(texto: str) -> list[tuple[int, str]]:
    """(posição, conteúdo) de cada `\\boxed{…}`/`\\fbox{…}` de chaves balanceadas, em ordem.
    Caixa cortada (sem a chave de fechar) não entra."""
    return [(ini, conteudo) for ini, _, conteudo in _caixas_com_fim(texto)]


def extrair_boxed(texto: str) -> str | None:
    """O conteúdo da ÚLTIMA caixa (usado também para ler os gabaritos)."""
    cx = caixas(texto)
    return cx[-1][1] if cx else None


def _sem_envoltorio(s: str) -> str:
    """Tira o envoltório que cobre a caixa INTEIRA: `{D}`, `\\textbf{…}`, `\\mathbf{…}`,
    `\\boldsymbol{…}`, `\\bm{…}`, com chaves balanceadas. O `\\text{…}` fica: V/F e letras o
    leem."""
    for _ in range(5):
        s = s.strip()
        m = re.match(r"\\(?:textbf|mathbf|boldsymbol|bm)\s*\{|\{", s)
        if not m or _chaves(s, m.end() - 1) != len(s) - 1:
            break
        s = s[m.end():-1]
    return s.strip()


def _caixas_validas(texto: str) -> list[str]:
    """O conteúdo das caixas, sem envoltório, sem as vazias e sem as de placeholder (o eco
    de `\\boxed{}` e de `\\boxed{...}` do prompt)."""
    return [c for c in (_sem_envoltorio(c) for _, _, c in _caixas_com_fim(texto))
            if not _CAIXA_VAZIA.fullmatch(c)]


def _armado_em(texto: str) -> int | None:
    """A posição a partir da qual o texto já tem uma resposta, ou None. Arma: a linha
    canônica (frase final e caixa na mesma linha), uma caixa seguida de linha em branco, ou
    a frase final que se lê sem caixa. A caixa intermediária no meio do raciocínio e a frase
    recusada ("note that the final answer is independent of…") não armam (rodada 4)."""
    candidatos = []
    for ini, fim, conteudo in _caixas_com_fim(texto):
        if _CAIXA_VAZIA.fullmatch(conteudo):
            continue
        linha_ini = texto.rfind("\n", 0, ini) + 1
        if _FRASE_FINAL.search(texto, linha_ini, ini):
            candidatos.append(fim)
            break
        # a caixa fecha o parágrafo: a linha dela é seguida de uma linha em branco
        fim_da_linha = texto.find("\n", fim)
        branco = _LINHA_EM_BRANCO.match(texto, fim_da_linha) if fim_da_linha != -1 else None
        if branco:
            candidatos.append(branco.start())
            break
    for f in _FRASE_FINAL.finditer(texto):
        if _ler_frase(texto, f, 1) is not None:
            candidatos.append(f.end())
            break
    return min(candidatos) if candidatos else None


def cortar_continuacao(texto: str) -> str:
    """Corta no começo da LINHA do primeiro marcador de problema novo que venha DEPOIS de
    já haver uma resposta (`_armado_em`). Também corta antes de um `Solution:` que venha
    depois da resposta com o enunciado inventado entre os dois (uma linha não vazia): um
    `Solution:` logo em seguida é o modelo refazendo a própria conta."""
    arme = _armado_em(texto)
    if arme is None:
        return texto
    seguintes = [m.start() for m in _SOLUCAO.finditer(texto, arme)]
    seguintes += [m.start() for m in _ECO.finditer(texto, arme)]
    cortes = []
    for padrao in (_PROBLEMA_FORTE, _ECO):
        m = padrao.search(texto, arme)
        if m:
            cortes.append(m.start())
    for m in _PROBLEMA_FRACO.finditer(texto, arme):
        if any(p > m.start() for p in seguintes):
            cortes.append(m.start())
            break
    fim_da_linha = texto.find("\n", arme)
    if fim_da_linha != -1:
        for m in _SOLUCAO.finditer(texto, fim_da_linha + 1):
            if texto[fim_da_linha + 1:m.start()].strip():
                cortes.append(fim_da_linha + 1)
            break
    if not cortes:
        return texto
    corte = min(cortes)
    return texto[:texto.rfind("\n", 0, corte) + 1]


def _grupo(m: re.Match) -> str:
    return next(g for g in m.groups() if g is not None).strip()


def _pedaco_de_unidade(g: str) -> bool:
    """`m/s$^2$`, `$\\mathrm{m/s}$`, `$\\rm m/s^2$`, `$\\mu$m`, `$\\circ$`, `$\\,$`: pedaço de
    unidade ou espaço, não resposta."""
    g = re.sub(r"\\[,;:! ]|~", "", g).strip()
    if not g or re.match(r"[\^_]", g):
        return True
    if re.fullmatch(r"\\mu|μ|\\circ|\^?\{?\\circ\}?|\\%|%|\\Omega|Ω", g):
        return True
    g = re.sub(r"^\\(?:rm|mathrm)\b\s*", "", g)
    return _cauda_e_unidade(_UNIDADE_EM_TEXTO.sub(" ", g))


def _sentenca(depois: str) -> tuple[str, str]:
    partes = re.split(r"(?<=[.!?])\s+|\n", depois, maxsplit=1)
    return partes[0], (partes[1] if len(partes) > 1 else "")


def _ler_frase(texto: str, frase: re.Match, n: int) -> str | None:
    """A resposta que a frase "final answer is" traz, ou None se a frase não responde. Só
    vale sem caixa (a caixa vence sempre).

    - Na oração, ANTES da frase, nenhuma condição ou contrafactual que a governe.
    - Entre a frase e o valor, só enchimento ("therefore", "≈", "E =", "(1)").
    - Depois do valor, na sentença: nada de "?", "or", "if", "unless", e nenhum outro valor
      ligado ao primeiro ("$13.06$ eV, or $12.75$ eV"); a sentença seguinte não retrata
      ("But that ignores…"). O comentário ("correct to three figures") não importa.
    - Com n > 1 respostas, junta os n primeiros trechos da sentença, pulando pedaço de
      unidade e rótulo entre parênteses (`$5$ m/s$^2$ ($a$)`)."""
    ini = max(texto.rfind(c, 0, frase.start()) for c in ".!?\n;") + 1
    if _CONDICAO.fullmatch(texto[ini:frase.start()]):
        return None
    resto = texto[frase.end():frase.end() + 800]
    m = _MATEMATICA.search(resto)
    if m and _ENCHIMENTO.fullmatch(resto[:m.start()]):
        depois = resto[m.end():]
        sentenca, seguinte = _sentenca(depois)
        if (_DEPOIS_HESITANTE.match(depois) or _RETRATACAO.match(seguinte)
                or _CONDICAO_DEPOIS.search(sentenca)):
            return None
        primeiro = _grupo(m)
        # `$0.3\,\mu$m`: o μ só existe em modo matemático, e a letra da unidade fica fora
        mu = re.match(r"[A-Za-z]+", depois)
        if re.search(r"(?:\\mu|μ)\s*$", primeiro) and mu:
            primeiro = f"{primeiro} {mu.group()}"
        if n <= 1:
            return None if _OUTRO_VALOR.search(sentenca) else primeiro
        # alternativa entre as partes ("$2$ and $8$ or $9$", "$2$ (or $3$)")
        if re.search(r"\bor\b\s*(?:[A-Za-z]+\s+){0,3}\$", sentenca, re.I):
            return None
        partes = [primeiro]
        for m2 in _MATEMATICA.finditer(sentenca):
            if len(partes) >= n:
                break
            g = _grupo(m2)
            entre_parenteses = (sentenca[:m2.start()].rstrip().endswith("(")
                                and sentenca[m2.end():].lstrip().startswith(")"))
            if not entre_parenteses and not _pedaco_de_unidade(g):
                partes.append(g)
        return ", ".join(partes)
    # texto puro: até o primeiro `$`, ", i.e.", ", where", ", which", ";" ou fim de sentença
    linha = resto.split("\n")[0].replace("**", "").replace("`", "")
    linha = re.split(r"\$|,\s*(?:i\.e\.|where|which)\b|;", linha)[0]
    linha = re.sub(r"^(?:\s|,|:|=|≈|~|approximately|about|roughly|therefore|thus|simply)*", "",
                   linha, flags=re.I)
    linha = re.split(r",\s+[A-Za-z]|\.\s+[A-Za-z]|\.\s*$", linha)[0].strip()
    sentenca, seguinte = _sentenca(resto)
    if (not linha or _DEPOIS_HESITANTE.search(" " + linha) or _RETRATACAO.match(seguinte)
            or "?" in sentenca):
        return None
    if re.match(r"[-+(]?\s*[\d\\]", linha) or _PALAVRA_RESPOSTA.fullmatch(linha):
        return linha
    # começa por letra: `E = 13.06 eV`, `B^2/(2 M c^2)`; prosa ("independent of the mass") não
    if re.search(r"[=^/_*()\d]", linha) and not re.search(r"\b[A-Za-z]{3,}\s+[A-Za-z]{2,}\b", linha):
        return linha
    return None


def _da_frase(texto: str, n: int) -> str | None:
    """A última frase final que de fato responde (com n partes, quando n > 1); senão None."""
    for f in reversed(list(_FRASE_FINAL.finditer(texto))):
        lida = _ler_frase(texto, f, n)
        if lida is not None and (n <= 1 or len(respostas_da_caixa(lida)) >= n):
            return lida
    return None


def extrair_resposta(texto: str) -> str | None:
    """A última caixa não vazia; sem caixa, a frase final que responde (`_ler_frase`);
    senão None (= sem resposta). O texto é cortado antes de um problema novo inventado."""
    texto = cortar_continuacao(texto)
    cx = _caixas_validas(texto)
    if cx:
        return cx[-1]
    return _da_frase(texto, 1)


def extrair_respostas(texto: str, n: int) -> str | None:
    """Para item de n > 1 respostas: a última caixa, se tiver n partes; senão as últimas n
    caixas não vazias juntas (uma por resposta é hábito forte de modelo); sem caixa, a
    frase final com n partes, ou com uma."""
    if n <= 1:
        return extrair_resposta(texto)
    texto = cortar_continuacao(texto)
    cx = _caixas_validas(texto)
    if cx:
        if len(respostas_da_caixa(cx[-1])) >= n or len(cx) < n:
            return cx[-1]
        return ", ".join(cx[-n:])
    return _da_frase(texto, n) or _da_frase(texto, 1)


# ── normalização ─────────────────────────────────────────────────────────────

_UNICODE = (("−", "-"), ("–", "-"), ("×", "\\times "), ("·", "\\cdot "), ("π", "\\pi "),
            ("∞", "\\infty "), ("≈", "\\approx "), ("≤", "\\leq "), ("≥", "\\geq "),
            ("℃", "^{\\circ}C"), ("℉", "^{\\circ}F"), ("°", "^{\\circ}"), ("µ", "\\mu "),
            ("μ", "\\mu "), ("，", ","), ("；", ";"), ("：", ":"), ("（", "("), ("）", ")"),
            ("\u00a0", " "), ("\u2009", " "), ("\u202f", " "), ("\u2002", " "),
            ("\u2003", " "), ("\u200b", ""), ("√", "\\sqrt "))
_UNICODE_EXTRA = (("ħ", "\\hbar "), ("ℏ", "\\hbar "))
# alfabeto grego em Unicode (ϵ e ϕ caem nas variantes unificadas)
_GREGO_UNICODE = dict(zip(
    "αβγδεϵζηθϑικλνξρϱσςτυφϕχψωΓΔΘΛΞΠΣΥΦΨΩ",
    ["alpha", "beta", "gamma", "delta", "epsilon", "epsilon", "zeta", "eta", "theta",
     "theta", "iota", "kappa", "lambda", "nu", "xi", "rho", "rho", "sigma", "sigma", "tau",
     "upsilon", "phi", "phi", "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda",
     "Xi", "Pi", "Sigma", "Upsilon", "Phi", "Psi", "Omega"], strict=True))
_SOBRESCRITOS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUBSCRITOS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_ENVOLTORIOS = (("$$", "$$"), ("$", "$"), ("\\[", "\\]"), ("\\(", "\\)"))
# Com fim de palavra: `\right` dentro de `\rightarrow` virava "arrow" (achado no dev).
_SUMIR = re.compile(r"\\(?:left|right)\s*\.|\\(?:left|right|displaystyle|textstyle|"
                    r"scriptstyle|biggl|biggr|Biggl|Biggr|bigl|bigr|Bigl|Bigr|bigm|Bigm|bigg|"
                    r"Bigg|big|Big|qquad|quad)(?![A-Za-z])|\\[!,;:]|\\ ")
# Versor: `\hat{r}`, `\hat{\mathbf{z}}`, `\mathbf{\hat{r}}` e `\boldsymbol{e}_r` são o mesmo
# e_r; î, `\mathbf{i}`, `\boldsymbol{i}` e `\vec{i}` são o mesmo e_x (e j, k → y, z). O
# `\hat` de outra letra (n̂, operador x̂ p̂ de QM fica no x) só sai (rodada 3). O versor
# passa por um marcador que `_argumento_de_funcao` não engole: `\cos\alpha \mathbf{i}` é
# cos(α)·e_x, não cos(α·e_x).
_VERSOR = re.compile(r"\\hat\s*(?:\{\s*(x|y|z|r|i|j|k|\\theta|\\phi|\\rho)\s*\}|"
                     r"(x|y|z|r|i|j|k|\\theta|\\phi|\\rho)(?![A-Za-z]))")
_VERSOR_IJK = re.compile(r"\\(?:mathbf|boldsymbol|bm|vec)\s*(?:\{\s*([ijk])\s*\}|([ijk])(?![A-Za-z]))")
_VERSOR_E = re.compile(r"\\(?:mathbf|boldsymbol|bm|vec|hat)\s*(?:\{\s*e\s*\}|e(?![A-Za-z]))\s*_\s*"
                       r"(\{[^{}]*\}|\\[A-Za-z]+|\w)")
_CARTESIANO = {"i": "x", "j": "y", "k": "z"}
_MARCA_VERSOR = "\\VERSOR"
_HIPERBOLICAS_RUSSAS = {"ch": "cosh", "sh": "sinh", "th": "tanh", "cth": "coth"}
_FUNCOES_SEM_PARENTESE = ("sinh", "cosh", "tanh", "coth", "arcsin", "arccos", "arctan",
                          "sin", "cos", "tan", "cot", "sec", "csc", "ln", "log", "exp")
_TEXTO = re.compile(r"\\(?:mathrm|text|textrm|rm|mbox|operatorname|mathsf|textit|"
                    r"textbf)\s*\{([^{}]*)\}")
_VARIANTES = (("varepsilon", "epsilon"), ("varphi", "phi"), ("vartheta", "theta"),
              ("varrho", "rho"), ("varsigma", "sigma"), ("varkappa", "kappa"))
_GREGAS = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota",
           "kappa", "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "upsilon", "phi",
           "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma",
           "Upsilon", "Phi", "Psi", "Omega", "hbar", "pi")
_RECIPROCAS = {"coth": "tanh", "sech": "cosh", "csch": "sinh"}


def _com_par(s: str, comando: str, novo: Callable[[str], str]) -> str:
    """Troca `comando(…)` (parênteses balanceados) por `novo(conteúdo)`."""
    saida, i = "", 0
    while (k := s.find(comando, i)) != -1:
        j = k + len(comando)
        while j < len(s) and s[j] == " ":
            j += 1
        if j >= len(s) or s[j] != "(":
            saida += s[i:j]
            i = j
            continue
        nivel = 0
        for f in range(j, len(s)):
            nivel += {"(": 1, ")": -1}.get(s[f], 0)
            if nivel == 0:
                break
        else:
            saida += s[i:]
            return saida
        saida += s[i:k] + novo(s[j + 1:f])
        i = f + 1
    return saida + s[i:]


def _raiz_n(s: str) -> str:
    """`\\sqrt[n]{X}` → `\\left(X\\right)^{1/(n)}`, com contagem de chaves."""
    while (m := re.search(r"\\sqrt\s*\[([^\[\]]*)\]\s*\{", s)) is not None:
        fim = _chaves(s, m.end() - 1)
        if fim is None:
            break
        s = s[:m.start()] + f"({s[m.end():fim]})^{{1/({m.group(1)})}}" + s[fim + 1:]
    return s


def _over(s: str) -> str:
    """`{A \\over B}` → `\\frac{A}{B}` (grupo de chaves mais interno)."""
    for _ in range(10):
        m = re.search(r"\{([^{}]*)\\over([^{}]*)\}", s)
        if not m:
            break
        s = s[:m.start()] + f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}" + s[m.end():]
    return s


_FUNCOES_CONHECIDAS = {"sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh",
                       "arcsin", "arccos", "arctan", "arsinh", "arcosh", "artanh", "exp",
                       "ln", "log", "sqrt", "frac", "det", "max", "min", "arg", "Pr", "lg",
                       "delta"}   # δ(x): Dirac, função (vira pseudofunção em `_fisica`)
# operador antes do parêntese já é a multiplicação (e `limpar` roda mais de uma vez)
_OPERADORES = {"cdot", "times", "div", "pm", "mp", "ast", "star", "circ", "left", "right",
               "approx", "sim", "simeq", "equiv", "propto", "to", "in", "leq", "geq", "le",
               "ge", "lt", "gt", "neq", "cup", "cap"}
_UM_TOKEN = re.compile(r"\s*(?:[+-]?\d+(?:\.\d+)?|\\?[A-Za-z]+(?:\s*_\s*(?:\{[^{}]*\}|\w))?)\s*")


def _expoente_de_funcao(s: str, fecha: int) -> bool:
    """A chave que fecha em `fecha` é o expoente/subscrito de uma função conhecida?"""
    nivel = 0
    for j in range(fecha, -1, -1):
        nivel += {"}": 1, "{": -1}.get(s[j], 0)
        if nivel == 0:
            antes = s[:j].rstrip()
            if antes.endswith(("^", "_")):
                m = re.search(r"\\([A-Za-z]+)\s*$", antes[:-1])
                return bool(m and m.group(1) in _FUNCOES_CONHECIDAS)
            return False
    return False


def _potencia_de_funcao(s: str) -> str:
    """`f^{2}(z)` com argumento de um token é [f(z)]², notação comum de física."""
    return re.sub(r"(?<![A-Za-z\\])([A-Za-z])\s*\^\s*\{?\s*(\d)\s*\}?\s*\((\s*(?:[+-]?\d+(?:\.\d+)?"
                  r"|\\?[A-Za-z]+(?:\s*_\s*(?:\{[^{}]*\}|\w))?)\s*)\)", r"(\1(\3))^{\2}", s)


def _multiplicacao_explicita(s: str) -> str:
    """Símbolo seguido de parêntese é PRODUTO (`m(g+a)^2` = m·(g+a)², `\\mu (g - a)`), a
    não ser que o argumento seja um token só: aí é valor de função (`\\theta(0)`, `x(t)`).
    O parser lia todo `m(…)` como função, e `m(g+a)^2` saía (m·(g+a))²."""
    saida, i = "", 0
    for m in re.finditer(r"(\\[A-Za-z]+|(?<![A-Za-z\\])[A-Za-z]|\})\s*\(", s):
        base = m.group(1)
        if base.startswith("\\") and base[1:] in _FUNCOES_CONHECIDAS | _OPERADORES:
            continue
        if base == "}" and _expoente_de_funcao(s, m.start(1)):
            continue                     # `\tan^{-1}(…)`, `\sin^{2}(…)`, `\log_{2}(…)`
        if base == "}" and re.search(r"F_\{opfn[A-Za-z]+$", s[:m.start(1)]):
            continue                     # função nomeada (`_funcoes_nomeadas`)
        nivel, fim = 0, None
        for j in range(m.end() - 1, len(s)):
            nivel += {"(": 1, ")": -1}.get(s[j], 0)
            if nivel == 0:
                fim = j
                break
        dentro = s[m.end():fim] if fim is not None else ""
        # um token só (θ(0), x(t)) ou vários argumentos (κ(p, T)): é função, não produto
        if fim is None or _UM_TOKEN.fullmatch(dentro) or len(dividir_nivel_zero(dentro, ",")) > 1:
            continue
        saida += s[i:m.end() - 1] + " \\cdot "
        i = m.end() - 1
    return saida + s[i:]


def _funcao_de_fracao(s: str) -> str:
    """`\\tan\\frac{a}{4} e^{y}` → `\\tan(\\frac{a}{4}) e^{y}`. O parser leva para dentro da
    função tudo o que vem depois (tan(a·e^y/4)), e um gabarito assim coincidia com uma
    resposta errada. Com a fração logo depois da função, o argumento é só a fração."""
    padrao = re.compile(r"\\(sin|cos|tan|cot|sec|csc|sinh|cosh|tanh|arcsin|arccos|arctan|"
                        r"exp|ln|log)(\s*\^\s*\{[^{}]*\})?\s*(?=\\frac\s*\{)")
    saida, i = "", 0
    for m in padrao.finditer(s):
        ini = s.index("\\frac", m.end())
        a = _chaves(s, s.index("{", ini))
        if a is None or s.find("{", a) == -1 or s[a + 1:].lstrip()[:1] != "{":
            continue
        b = _chaves(s, s.index("{", a + 1))
        if b is None or ini < i:
            continue
        saida += s[i:ini] + "(" + s[ini:b + 1] + ")"
        i = b + 1
    return saida + s[i:]


def _milhar(s: str) -> str:
    """Grupos de milhar separados por `{,}`, `\\,`, `\\ `, `~` ou espaço: `6\\,000\\,000`. E os
    grupos de três depois da vírgula decimal, no estilo SI (`48.189\\,685`, `0.000\\,123`),
    só com espaço fino (`\\,`, `\\ `, `~`): antes viravam produto (rodada 3)."""
    s = re.sub(r"(?<![\w.^_{])(\d+\.(?:\d{3})+)((?:(?:\\,|\\ |~|\\;)\d{3})*(?:\\,|\\ |~|\\;)\d{1,3})"
               r"(?![\d^.])", lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), s)
    return re.sub(r"(?<![\w.^_{])(\d{1,3})((?:(?:\{,\}|\\,|\\ |~|\\;| )\d{3})+)(?!\d)",
                  lambda m: m.group(1) + re.sub(r"\D", "", m.group(2)), s)


def _marcar_versores(s: str) -> str:
    def marca(x: str) -> str:
        x = x.strip("{} ")
        return f"{_MARCA_VERSOR}{{{_CARTESIANO.get(x, x)}}}"

    s = _VERSOR_E.sub(lambda m: marca(m.group(1)), s)
    s = _VERSOR_IJK.sub(lambda m: marca(m.group(1) or m.group(2)), s)
    return _VERSOR.sub(lambda m: marca(m.group(1) or m.group(2)), s)


def _decoracoes(s: str) -> str:
    """Versores viram o marcador (`_VERSOR`); as outras decorações saem, com ou sem chaves
    (também aninhadas), até não sobrar nenhuma (`\\hat{\\mathbf{r}}`, `\\vec v`)."""
    for _ in range(4):
        antes = s
        s = _marcar_versores(s)
        s = re.sub(r"\\(?:mathbf|boldsymbol|bm|mathit|mathcal|mathscr|vec|bar|tilde|overline|"
                   r"overrightarrow|widetilde|widehat)\s*\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", s)
        s = _marcar_versores(s)
        s = re.sub(r"\\(?:vec|hat|bar|tilde|overline|overrightarrow|widetilde|widehat|mathbf|"
                   r"boldsymbol|bm|mathit|mathcal|mathscr)\s*\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", s)
        s = re.sub(r"\\(?:vec|hat|bar|tilde|mathbf|boldsymbol|bm)\s*(\\[A-Za-z]+|[A-Za-z])", r"\1",
                   s)
        if s == antes:
            return s
    return s


def _argumento_de_funcao(s: str) -> str:
    """`\\cos\\theta \\cdot \\frac{ma}{M}` → `\\cos{\\theta} \\cdot …`. O parser põe dentro de uma
    função sem parênteses TODA a cadeia multiplicativa seguinte (cos(θ·ma/M)). O argumento
    vai até o primeiro operador, parêntese, fração, raiz ou outra função; `\\sin \\omega t`
    continua sin(ωt), como se escreve em física."""
    # só letras gregas e símbolos entram no argumento; relação e operador (`\approx`,
    # `\cdot`, `\to`) e o marcador de versor fecham o argumento. O átomo leva o subscrito
    # (com um nível de chaves: `\theta_{\mathrm{c}}`) e o expoente: um átomo só com
    # expoente fica dentro (`\ln r^{2}` = ln(r²), `\sin x^{2}` = sin(x²), como o parser
    # lia); com vários átomos, o argumento para antes do que tem expoente
    # (`\sin\varphi v^{2}` = sin(φ)·v²). Rodada 3: o expoente ficava fora, ln(r)².
    gregas = "|".join(_GREGAS + ("ell", "infty", "epsilon", "phi", "theta", "rho"))
    atomo = (rf"(?:\\(?:{gregas})(?![A-Za-z])|[A-Za-z0-9.]+)\s*{_SUBSCRITO}"
             r"(?:\s*\^\s*(?:\{[^{}]*\}|\w))?")
    funcoes = "|".join(_FUNCOES_SEM_PARENTESE)
    funcao = rf"\\(?:{funcoes})(?![A-Za-z])"
    padrao = re.compile(rf"\\({funcoes})(?![A-Za-z])(\s*\^\s*(?:\{{[^{{}}]*\}}|\w))?\s*"
                        rf"((?:{atomo})(?:\s*(?:{atomo}))*)")

    def troca(m: re.Match) -> str:
        arg = m.group(3)
        if re.match(funcao, arg):
            return m.group(0)
        fim, n = 0, 0
        for p in re.finditer(atomo, arg):
            if re.match(funcao, p.group()):          # para antes de outra função
                break
            expoente = "^" in p.group()
            if expoente and n:
                break
            fim, n = p.end(), n + 1
            if expoente:
                break
        if not n:
            return m.group(0)
        usado = arg[:fim].rstrip()
        # parênteses, e não chaves: com `\sin{\theta} \cdot (…)` o parser ainda lê sin(θ·(…))
        return f"\\{m.group(1)}{m.group(2) or ''}({usado})" + arg[len(usado):]

    return padrao.sub(troca, s)


# Funções nomeadas que não são do parser (rodada 3). Viram uma cabeça reservada,
# `F_{opfn<nome>}`, que `_fisica` troca por uma pseudofunção não linear; nenhum símbolo
# de física se escreve assim (o `F_` puro pegava F_N, F_0[…] e F_1(t)). Vale para
# `\operatorname{X}` (qualquer nome), para `\mathrm{X}`/`\text{X}`/`\rm{X}` e o texto puro
# com X numa lista de funções especiais, seguidas de `(`, `[` ou `\frac`. Bessel com
# índice (`J_{0}(kr)`) fica de fora: J_0·(…) também é corrente, momento angular.
_CABECA_NOMEADA = "F_{opfn"
_ESPECIAIS = ("erfc", "erfi", "erf", "sinc", "sgn", "sign", "Si", "Ci", "Ei", "Ai", "Bi",
              "sn", "cn", "dn", "li", "Li", "Re", "Im", "Tr", "tr")
_ESPECIAIS_TEXTO = ("erfc", "erfi", "erf", "sinc", "sgn")


def _funcoes_nomeadas(s: str) -> str:
    especiais = "|".join(_ESPECIAIS)
    padrao = re.compile(
        rf"\\operatorname\s*\{{\s*([A-Za-z]{{2,}})\s*\}}"
        rf"|\\(?:mathrm|text|textrm|rm|mathit)\s*\{{\s*({especiais})\s*\}}"
        rf"|(?<![A-Za-z\\{{])({'|'.join(_ESPECIAIS_TEXTO)})(?![A-Za-z])")
    saida, i = "", 0
    for m in padrao.finditer(s):
        if m.start() < i:
            continue
        nome = m.group(1) or m.group(2) or m.group(3)
        j = m.end()
        while j < len(s) and s[j] == " ":
            j += 1
        cabeca = f"{_CABECA_NOMEADA}{nome}}}"
        if s.startswith("\\frac", j):
            a = _chaves(s, s.find("{", j)) if s.find("{", j) != -1 else None
            if a is None or s[a + 1:].lstrip()[:1] != "{":
                continue
            b = _chaves(s, s.index("{", a + 1))
            if b is None:
                continue
            saida += s[i:m.start()] + f"{cabeca}({s[j:b + 1]})"
            i = b + 1
        elif j < len(s) and s[j] in "([":
            saida += s[i:m.start()] + cabeca
            i = j
    return saida + s[i:]


def limpar(s: str) -> str:
    s = str(s).strip()
    # grega em Unicode com índice colado: `ε0`, `μ0` → `\epsilon_{0}` (era ε·0, rodada 3)
    s = re.sub("([μµ" + "".join(_GREGO_UNICODE) + r"])(\d+)",
               lambda m: f"\\{_GREGO_UNICODE.get(m.group(1), 'mu')}_{{{m.group(2)}}} ", s)
    for a, b in _UNICODE + _UNICODE_EXTRA:
        s = s.replace(a, b)
    s = "".join(f"\\{_GREGO_UNICODE[ch]} " if ch in _GREGO_UNICODE else ch for ch in s)
    s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", lambda m: "^{" + m.group().translate(_SOBRESCRITOS) + "}", s)
    s = re.sub(r"[₀₁₂₃₄₅₆₇₈₉]+", lambda m: "_{" + m.group().translate(_SUBSCRITOS) + "}", s)
    for a, b in _ENVOLTORIOS:
        if s.startswith(a) and s.endswith(b) and len(s) >= len(a) + len(b):
            s = s[len(a):len(s) - len(b)].strip()
    s = s.strip().strip("$").strip()             # `$` solto de um gabarito mal separado
    s = _milhar(s)
    s = re.sub(r"\\(?:SI|qty)\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1 \\mathrm{\2}", s)
    s = re.sub(r"\\num\s*\{([^{}]*)\}", r"\1", s)
    # notação e (`3.1e3`, `1.03e-05`): o parser LaTeX leria 3,1·e·3
    s = re.sub(r"(?<![A-Za-z\\_{])(\d+(?:\.\d*)?)[eE]([+-]?)(\d+)(?![A-Za-z])",
               lambda m: f"{m.group(1)} \\times 10^{{{m.group(2).replace('+', '')}{m.group(3)}}}",
               s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\cfrac", "\\frac")
    s = re.sub(r"\\frac\s*([A-Za-z0-9])\s*([A-Za-z0-9])", r"\\frac{\1}{\2}", s)   # `\frac v2`
    s = _over(s)
    s = _SUMIR.sub(" ", s)
    s = s.replace("~", " ").replace("\\{", "(").replace("\\}", ")")
    for a, b in _VARIANTES:
        s = re.sub(rf"\\{a}(?![A-Za-z])", rf"\\{b}", s)
    s = _decoracoes(s)
    s = re.sub(r"\\mathrm\s*\{\s*([edi])\s*\}", r"\1", s)     # e de Euler, d, i
    s = re.sub(r"\\(?:operatorname|mathrm)\s*\{\s*(ch|sh|th|cth)\s*\}",
               lambda m: "\\" + _HIPERBOLICAS_RUSSAS[m.group(1)], s)
    s = re.sub(r"\\log\s*_\s*\{?\s*e\s*\}?", r"\\ln", s)
    # ponto com o subscrito dentro: `\ddot{\theta_1}` → `\ddot{\theta}_{1}`
    s = re.sub(r"\\(ddot|dot)\s*\{\s*(\\?[A-Za-z]+)\s*_\s*(\{[^{}]*\}|\w)\s*\}", r"\\\1{\2}_\3", s)
    # ponto de derivada temporal: `\dot{\theta}_{1}` → `\theta_{dot1}`, `\dot x` → `x_{dot}`
    arg = r"(?:\{\s*([^{}]*?)\s*\}|(\\[A-Za-z]+|[A-Za-z]))"
    s = re.sub(rf"\\(ddot|dot)\s*{arg}\s*_\s*(?:\{{\s*(\w+)\s*\}}|(\w))",
               lambda m: f"{m.group(2) or m.group(3)}_{{{m.group(1)}{m.group(4) or m.group(5)}}}", s)
    s = re.sub(rf"\\(ddot|dot)\s*{arg}",
               lambda m: f"{m.group(2) or m.group(3)}_{{{m.group(1)}}}", s)
    s = re.sub(r"\\langle\s*(\\?[A-Za-z]+(?:_\{?\w+\}?)?)\s*\\rangle", r"\1_{avg}", s)
    s = re.sub(r"\\operatorname\s*\{\s*(sinh|cosh|tanh|coth|sech|csch|arcsin|arccos|"
               r"arctan|arsinh|arcosh|artanh|sin|cos|tan|cot|sec|csc|exp|ln|log)\s*\}",
               r"\\\1", s)
    # outra função nomeada (`\operatorname{erf}(x)`, `\mathrm{erf}\frac{x}{a}`, `erf(x)`)
    # vira a cabeça reservada; antes era o produto operatorname·e·r·f·x, linear em x
    s = _funcoes_nomeadas(s)
    s = re.sub(r"\\(coth|sech|csch)\s*(\{[^{}]*\}|\([^()]*\)|\[[^\[\]]*\]|"
               r"\\[A-Za-z]+(?:_\{?\w+\}?)?|[A-Za-z0-9](?:_\{?\w+\}?)?)",
               lambda m: "\\frac{1}{\\" + _RECIPROCAS[m.group(1)] + "{" + m.group(2) + "}}", s)
    # linha e estrela viram subscrito nomeado: `\gamma'`, `M^{\prime}` → `_{prime}`
    s = re.sub(r"\^\s*\{\s*\\prime\s*\\prime\s*\}|(?<=[A-Za-z}])\s*''", "_{dprime}", s)
    s = re.sub(r"\^\s*\{\s*\\prime\s*\}|\^\s*\\prime|(?<=[A-Za-z}])\s*'", "_{prime}", s)
    s = re.sub(r"\^\s*\{\s*\*\s*\}|\^\s*\*", "_{star}", s)
    # sobrescrito antes do subscrito: `v^2_2` → `v_{2}^{2}` (antes virava zero)
    s = re.sub(r"(\\?[A-Za-z]+)\s*\^\s*(\{[^{}]*\}|\w)\s*_\s*(\{[^{}]*\}|\w)",
               lambda m: f"{m.group(1)}_{{{m.group(3).strip('{}')}}}^{{{m.group(2).strip('{}')}}}",
               s)
    # subscrito com comando sem chaves: `k_\mathrm{B}`, `E_\text{kin}`, `E_\gamma`
    s = re.sub(r"_\s*(\\(?:mathrm|text|rm|textrm|mathit|operatorname)\s*\{[^{}]*\})", r"_{\1}", s)
    s = re.sub(r"_\s*(\\[A-Za-z]+)", r"_{\1}", s)
    # subscrito de um caractere sem chaves: `k_BT` é k_B·T, `p_0R` é p_0·R (semântica LaTeX)
    if "\\" in s or "{" in s:
        s = re.sub(r"_\s*([A-Za-z0-9])", r"_{\1}", s)
        # e o sobrescrito também: `x^2 3` era x^{23}, e `5d^2 6s^2` virava 5·d^{26}·s^2.
        # Dígitos COLADOS ficam juntos: `10^17` é 10¹⁷, não 10¹·7 (rodada 3)
        s = re.sub(r"\^\s*([+-]?\d+|[A-Za-z])", r"^{\1}", s)
    # dígitos separados por espaço que não são grupo de milhar não se colam (`1 30` ≠ 130)
    s = re.sub(r"(?<=\d)\s+(?=\d)", r" \\cdot ", s)
    s = _raiz_n(s)
    s = re.sub(r"\\sqrt\s*(?![\[{(])(\\[A-Za-z]+|[A-Za-z0-9])", r"\\sqrt{\1}", s)
    s = _com_par(s, "\\sqrt", lambda x: "\\sqrt{" + x + "}")
    s = _argumento_de_funcao(s)
    s = re.sub(re.escape(_MARCA_VERSOR) + r"\{([^{}]*)\}", r" e_{\1} ", s)
    s = _funcao_de_fracao(s)
    s = _potencia_de_funcao(s)
    s = _multiplicacao_explicita(s)
    # módulo: o parser frouxo descartava `\left|E\right|`; √((X)²) é o módulo nos reais
    s = re.sub(r"\|([^|]+)\|(?!\s*_)", r"\\sqrt{(\1)^{2}}", s)
    # `d` solto antes de comando (`d \sin\theta`) seria lido como diferencial
    s = re.sub(r"(?<![A-Za-z\\_^])(?<!_\{)(?<!\^\{)d(\s*)(?=\\[A-Za-z])", r"d\\,\1", s)
    return s.strip().rstrip(".,;").strip()


def _texto_para_latex(s: str) -> str:
    """Resposta sem LaTeX (`sqrt(5gR)`, `aT^4/3`) para o parser LaTeX, que separa letras
    coladas como produto."""
    s = s.replace("**", "^")
    s = _com_par(s, "sqrt", lambda x: "\\sqrt{" + x + "}")
    # `ln 2`, `sin(x)`: função de texto puro vira comando
    s = re.sub(r"(?<![A-Za-z\\])(sinh|cosh|tanh|arcsin|arccos|arctan|sin|cos|tan|exp|ln|log)"
               r"(?=\s*\(|\s+\w)", r"\\\1", s)
    # `1/2 v` é (1/2)·v em texto puro; o parser LaTeX leria 1/(2v)
    s = re.sub(r"(?<![\w.^_/])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![\w.^])", r"\\frac{\1}{\2}",
               s)
    # `mu0`, `epsilon0`: grega com índice colado
    s = re.sub(r"(?<![A-Za-z\\])(" + "|".join(_GREGAS) + r")(\d+)(?![A-Za-z0-9])", r"\\\1_{\2} ", s)
    s = re.sub(r"(?<![A-Za-z\\])(" + "|".join(_GREGAS) + r")(?![A-Za-z])", r"\\\1 ", s)
    s = re.sub(r"(?<![A-Za-z\\])(?:infinity|inf|oo)(?![A-Za-z])", r"\\infty ", s)
    return re.sub(r"_([A-Za-z0-9]+)", r"_{\1}", s)


def dividir_nivel_zero(s: str, padrao: str) -> list[str]:
    """Divide `s` em cada ocorrência de `padrao` FORA de {}, () e []."""
    return [p for p, _ in _dividir_com_sep(s, padrao)]


def _dividir_com_sep(s: str, padrao: str) -> list[tuple[str, str]]:
    """[(pedaço, separador que vinha ANTES dele)]."""
    marcas = {m.start(): (m.end(), m.group()) for m in re.finditer(padrao, s)}
    nivel, ultimo, sep, j, saida = 0, 0, "", 0, []
    while j < len(s):
        if nivel == 0 and j in marcas:
            saida.append((s[ultimo:j], sep))
            ultimo, sep = marcas[j]
            ultimo, j = marcas[j][0], marcas[j][0]
            continue
        if s[j] in "{([":
            nivel += 1
        elif s[j] in "})]":
            nivel = max(0, nivel - 1)
        j += 1
    saida.append((s[ultimo:], sep))
    return saida


# ── leitura ──────────────────────────────────────────────────────────────────

_LOCAIS = {"I": sp.Symbol("I")}
_FUNCOES_GENERICAS = {"delta"}       # δ(x) é Dirac, não δ·x


def _ler_latex(s: str) -> sp.Basic | None:
    s2, _ = normalizar_subscritos(s)
    blindado, marcadores = blindar_subscritos(s2)
    try:
        e = parse_latex(blindado, strict=True)
    except Exception:
        return None
    if marcadores:
        e = e.xreplace({sp.Symbol(m): sp.Symbol(i) for m, i in marcadores.items()})
        cabecas = {f: sp.Function(marcadores[f.func.__name__])(*f.args)
                   for f in e.atoms(AppliedUndef) if f.func.__name__ in marcadores}
        if cabecas:
            e = e.xreplace(cabecas)
    return e


def _nome(s: str) -> str:
    return re.sub(r"[{}\\\s]", "", s)


def _pseudofuncao(nome: str, args: tuple) -> sp.Expr:
    """Uma função não linear fixa por nome: argumentos ou escalas diferentes dão valores
    diferentes, e a mesma chamada, o mesmo valor."""
    h = int(hashlib.sha256(nome.encode()).hexdigest(), 16)
    # coeficientes distintos por posição (y(x, t) ≠ y(t, x)), num espaço grande, e um termo
    # constante do nome: com 13 formas, F_1 = F_2 e f(0) = 1 para todo nome (rodada 3)
    cs = [sp.Rational(3000 + 17 * i + (h >> (16 * i)) % 10007, 10000) for i in range(len(args))]
    c0 = sp.Rational(1 + (h >> 200) % 997, 100)
    soma = sum(c * a for c, a in zip(cs, args, strict=True))
    return sp.exp(soma / 4) + c0 + sum(c * a ** 2 for c, a in zip(cs, args, strict=True))


def _fisica(e: sp.Basic) -> sp.Expr:
    trocas = {}
    for f in e.atoms(AppliedUndef):
        nome = _nome(f.func.__name__)
        if nome.startswith("F_opfn"):                  # função nomeada: nunca linear
            trocas[f] = _pseudofuncao(nome, f.args)
            continue
        if nome in ("min", "max") and len(f.args) >= 2:   # simétricas: a ordem não importa
            trocas[f] = (sp.Min if nome == "min" else sp.Max)(*f.args)
            continue
        if all(a.is_number for a in f.args):           # θ(0): o valor num ponto
            trocas[f] = sp.Symbol(f"{nome}_at_" + "_".join(_nome(str(a)) for a in f.args))
        elif (len(f.args) == 1 and isinstance(f.args[0], sp.Symbol)
              and nome not in _FUNCOES_GENERICAS):
            trocas[f] = sp.Symbol(f"{nome}_of_{_nome(f.args[0].name)}")   # x(t), T(r)
        elif len(f.args) == 1 and nome not in _FUNCOES_GENERICAS:
            trocas[f] = sp.Symbol(nome) * f.args[0]    # o que escapou da multiplicação explícita
        else:
            trocas[f] = _pseudofuncao(nome, f.args)
    if trocas:
        e = e.xreplace(trocas)
    nomes = {s: sp.Symbol(_nome(s.name)) for s in e.free_symbols
             if isinstance(s, sp.Symbol) and re.search(r"[{}\\\s]", s.name)}
    if nomes:
        e = e.xreplace(nomes)
    e = e.xreplace({sp.Symbol("pi"): sp.pi})
    euler = sp.Symbol("e")
    return e.replace(lambda x: isinstance(x, sp.Pow) and x.base == euler
                     and bool(x.exp.free_symbols), lambda x: sp.exp(x.exp))


def expressao(s: str) -> sp.Expr | None:
    """A expressão lida em modo ESTRITO, ou None. Relação (`=`, `<`) não é expressão."""
    s = limpar(s)
    if not s:
        return None
    s = s.replace("[", "(").replace("]", ")")
    texto_puro = "\\" not in s and "{" not in s
    e = _ler_latex(_texto_para_latex(s) if texto_puro else s)
    if e is None and texto_puro:
        e = parse_verify(s, locais=_LOCAIS)
    if e is None or isinstance(e, Relational) or not isinstance(e, sp.Expr):
        return None
    try:
        return _fisica(e)
    except Exception:
        return None


_IGUAL = r"=|\\approx|\\simeq|\\equiv|\\propto|\\sim(?![A-Za-z])"
_APROXIMADO = ("\\approx", "\\simeq", "\\sim")


def _depois_do_igual(s: str) -> str:
    partes = dividir_nivel_zero(s, _IGUAL)
    return partes[-1].strip() if len(partes) > 1 else s


# Unidade solta depois de um número: tokens que são unidades (SI com prefixo, e algumas
# fora do SI), com expoentes inteiros, `/`, `*` e parênteses. Uma letra de variável (`6a`)
# não é unidade, e `\sqrt`, `\frac`, outro número ou `or` também não.
_UNIDADE = re.compile(
    r"(?:[YZEPTGMkhdcmunpfazyμ]|da)?(?:m|s|g|N|J|W|Pa|K|C|V|A|Ω|ohm|Ohm|Hz|eV|mol|L|l|atm|"
    r"bar|cal|rad|sr|T|G|F|H|S|Wb|Bq|Gy|Sv|lm|lx|cd|Å|b|B)|min|h|hr|d|day|days|yr|year|"
    r"years|au|AU|ly|pc|[kMG]pc|[kMG]yr|Da|u|amu|D|Torr|torr|mmHg|deg|degree|degrees|rev|rpm|"
    r"psi|lb|ft|mi|"
    r"mph|erg|dyn|gauss|Gauss|Ci|sec|cc|kcal|kJ|kWh|C|F")
# Unidade que é também símbolo usual de física: só vale se for a unidade do item
_UNIDADE_AMBIGUA = {"kT", "Mg", "mg", "ms", "cd", "Gy", "Da", "au", "pc", "rev", "sec", "cc",
                    "ML", "mL", "mT"}
# Letra única aceita como unidade numa unidade composta (`m/s`, `J/K`, `s^{-1}`); g, h, T,
# L, F, G… ficam de fora: `h/g`, `g^{2}`, `T^{2}` são produtos de símbolos.
_LETRA_UNIDADE = {"m", "s", "K", "N", "J", "W", "V", "A", "C", "Ω", "Å"}
# Qualificador em `\text{}`: a resposta não é o número puro (rodadas 2 e 3). O resto do
# texto sem algarismo sai (unidade por extenso, "exactly", "around").
_QUALIFICADOR = re.compile(
    r"\b(?:not|no|less|more|than|least|most|upper|lower|bound|or|zero|between|nearly|almost|"
    r"max|min|maximum|minimum|up to|over|under|below|above|exceeds?|except|only|unless|if)\b",
    re.I)
_NUM = (r"(?:[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?"
        r"(?:\s*(?:\\times|\\cdot|\*|x)\s*10\s*\^\s*\{?\s*[+-]?\s*\d+\s*\}?)?"
        r"|[+-]?10\s*\^\s*\{?\s*[+-]?\s*\d+\s*\}?)")
_UNIDADE_EM_TEXTO = re.compile(
    r"\\(?:mathrm|text|textrm|rm|mbox|operatorname|mathsf|textbf|textit)\s*"
    r"\{((?:[^{}]|\{[^{}]*\})*)\}(?:\s*\^\s*(?:\{[^{}]*\}|[-+]?\w))?")
_CONECTIVO = re.compile(r"\\text\s*\{\s*(?:or|and|to|ou|e|between|from)\s*\}|\b(?:or|and)\b")
# conteúdo de `\text{}` que pode sumir sem mudar o sentido; o resto (not, less than, at
# least, or zero, upper bound…) é qualificador e a leitura recusa
_NEUTRO = {"", "approximately", "approx", "approx.", "about", "roughly", "the answer is",
           "the result is", "answer", "result", "answer:", "result:", "is", "=", "≈",
           "exactly", "around", "in total", "total"}
# Unidade por extenso que pode sair de um `\text{}` (rodada 3: "2.8 \text{ meters}"). A
# rodada 4 mostrou que a lista NEGRA da rodada 3 deixava sair texto que muda o valor
# ("thousand", "half of", "negative"), qualificador em chinês e símbolo em texto (N_A,
# const): o `\text{}` volta a ser lista BRANCA, agora com as unidades por extenso.
_UNIDADE_POR_EXTENSO = re.compile(
    r"(?:(?:kilo|mega|giga|tera|milli|micro|nano|pico|femto|centi|square|cubic)[- ]?)?"
    r"(?:met(?:er|re)|second|gram|joule|newton|watt|volt|amp(?:ere)?|ohm|coulomb|farad|henry|"
    r"tesla|weber|pascal|kelvin|hertz|radian|steradian|lit(?:er|re)|mole?|electron[- ]?volt|"
    r"calorie|degree|minute|hour|day|year|light[- ]year|parsec|barn|lux|lumen|candela|"
    r"becquerel|gray|sievert|photon|atmosphere|torr|bar|gauss|erg|dyne|revolution|cycle|"
    r"turn|fringe|atom|nucleon|electron|particle|count|event)s?"
    r"|celsius|fahrenheit|per|squared|cubed|square|cubic", re.I)


def _valor_num(t: str) -> float:
    m = re.match(r"\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*(?:(?:\\times|\\cdot|\*|x)\s*10"
                 r"\s*\^\s*\{?\s*([+-]?)\s*(\d+)\s*\}?)?", t)
    if m is None or (m.group(1) == "10" and re.match(r"\s*[+-]?10\s*\^", t)):
        p = re.match(r"\s*([+-]?)10\s*\^\s*\{?\s*([+-]?)\s*(\d+)\s*\}?", t)
        return (-1 if p.group(1) == "-" else 1) * 10.0 ** int(p.group(2) + p.group(3))
    v = float(m.group(1))
    return v * 10.0 ** int(m.group(2) + m.group(3)) if m.group(3) else v


def _cauda_e_unidade(t: str) -> bool:
    """A cauda é feita só de unidades (com expoentes inteiros, `/`, `*`, parênteses)."""
    t = re.sub(r"(?:\\mu|μ)\s*", "μ", t).replace("\\Omega", "Ω").replace("\\cdot", "*")
    t = t.replace("{}", "")
    t = re.sub(r"\^\s*\{?\s*[+-]?\s*\d+(?:\.\d+)?\s*\}?", "", t)
    tokens = re.findall(r"[A-Za-zμΩÅ]+", t)
    resto = re.sub(r"[A-Za-zμΩÅ]+|[/*()\s.]", "", t)
    com_ev = any("eV" in x for x in tokens)
    return not resto and all(_UNIDADE.fullmatch(x) or (x == "c" and com_ev) for x in tokens)


def _tokens_de_unidade(t: str) -> list[str]:
    """`\\mathrm{J \\cdot s}`, `J s`, `J·s` → ['J', 's']; `\\AA`/`Å` → ['Å']; com expoentes."""
    t = str(t)
    for _ in range(3):
        t = re.sub(r"\\(?:mathrm|text|textrm|rm|mbox|operatorname)\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
                   r" \1 ", t)
    t = t.replace("\\AA", "Å").replace("\\mathring{A}", "Å").replace("\\Omega", "Ω")
    t = re.sub(r"(?:\\mu|μ|µ)\s*", "μ", t)
    t = re.sub(r"\\cdot|·|\*|\\,|\\;|~", " ", t)
    t = re.sub(r"\^\s*\{?\s*([+-]?)\s*(\d+)\s*\}?", r"^\1\2", t)
    return sorted(re.findall(r"[A-Za-zμΩÅ°]+(?:\^[+-]?\d+)?|/", t))


def _so_estrutura_de_unidade(t: str) -> bool:
    """A cauda tem só a FORMA de unidade: letras, `/`, `*`, `\\cdot`, parênteses no meio e
    expoente inteiro colado a letra. Número, `+`, `-` ou `=` fora de expoente, ou começo por
    `^` ou `(`, é conta, não unidade. Antes, a comparação por tokens com a unidade do item
    ignorava dígitos e operadores, e `130 \\cdot 2 N`, `60–90 GeV` e `80.4^{2} m/s` passavam
    pelo primeiro número (rodada 4)."""
    t = re.sub(r"\\(?:mathrm|text|textrm|rm|mbox|operatorname)\s*\{((?:[^{}]|\{[^{}]*\})*)\}",
               r" \1 ", t).strip()
    if re.match(r"[\^(]", t):
        return False
    t = re.sub(r"(?<=[A-Za-zμΩÅ°}])\s*\^\s*\{?\s*[+-]?\s*\d+(?:\.\d+)?\s*\}?", "", t)
    return not re.search(r"[\d+\-=]", t)


def _cauda_ok(t: str, unidade: str | None) -> bool:
    """A cauda depois do número pode ser descartada? Vale se for a unidade do PRÓPRIO
    item (comparada por tokens: `J \\cdot s` = `J s`), ou uma unidade sem ambiguidade.
    Letra solta (`30 T`, `-2 g`), várias letras soltas (`m g h`), `kT`, `Mg`, `ML^2`,
    divisão por uma unidade (`/ g`) e letra com expoente ou barra fora de
    `_LETRA_UNIDADE` (`h/g`, `g^{2}`) não valem, a não ser que sejam a unidade declarada:
    são também símbolos de física."""
    t = t.strip()
    if not t:
        return True
    if not _so_estrutura_de_unidade(t):
        return False
    if unidade:
        tu, tc = _tokens_de_unidade(unidade), _tokens_de_unidade(t)
        if tc == tu:
            return True
        # `$0.3\,\mu$m`, `\boxed{8.7\,\mu}b`: o prefixo μ separado da letra da unidade
        if tc == ["μ"] and any(x.startswith("μ") and len(x) > 1 for x in tu):
            return True
    if t.startswith("/") or not _cauda_e_unidade(t):
        return False
    tokens = re.findall(r"[A-Za-zμΩÅ]+", re.sub(r"(?:\\mu|μ)\s*", "μ", t))
    if any(x in _UNIDADE_AMBIGUA for x in tokens):
        return False
    soltas = [x for x in tokens if len(x) == 1]
    if not re.search(r"[/^]", t):
        return not soltas or len(tokens) > len(soltas) and all(x in _LETRA_UNIDADE for x in soltas)
    return all(x in _LETRA_UNIDADE for x in soltas)


def _tirar_unidade_do_item(s: str, unidade: str | None) -> str:
    """Tira do FIM da resposta a unidade declarada do item, como o prompt a mostra
    (`136 \\text{MeV}/c^{2}`, `0.22\\, k`, `13.06eV`), comparando sem espaços, e SÓ quando o
    que sobra é um número puro: com unidade "a", `6/a` e `8 - 2a` não viram 6 (rodada 3)."""
    if not unidade:
        return s
    u = re.sub(r"\s", "", limpar(unidade))
    t = re.sub(r"\s", "", s)
    if not u or not t.endswith(u) or len(t) == len(u):
        return s
    falta, j = len(u), len(s)
    while falta and j > 0:
        j -= 1
        if not s[j].isspace():
            falta -= 1
    if re.fullmatch(rf"\s*{_NUM}\s*(?:\\cdot|\\times|\*)?\s*", s[:j]):
        return s[:j]
    return s


def _texto_neutro(dentro: str, unidade: str | None) -> bool:
    """O `\\text{}` pode sair sem mudar o valor? Só unidade (símbolo, por extenso, a do
    item), grau e enchimento neutro. Letra fora do ASCII (o qualificador em chinês) não."""
    t = dentro.replace("\\AA", "Å").replace("\\mathring{A}", "Å").strip()
    t = re.sub(r"\^?\s*\{?\s*\\circ\s*\}?|°", " °", t)
    if re.search(r"[^\x00-\x7f°ÅμµΩ]", t):
        return False
    if t.lower().strip(" ()") in _NEUTRO or not t.strip(" ()"):
        return True
    if unidade and _tokens_de_unidade(t) == _tokens_de_unidade(unidade):
        return True
    sem_grau = re.sub(r"°\s*[CF]?", " ", t).strip()
    if not sem_grau or _cauda_e_unidade(sem_grau):
        return True
    palavras = re.findall(r"[A-Za-z]+(?:[- ](?:volts?|years?))?|[^\sA-Za-z]", sem_grau)
    return bool(palavras) and all(_UNIDADE_POR_EXTENSO.fullmatch(p) or p in "/()."
                                  for p in palavras)


def _tirar_texto(s: str, unidade: str | None) -> str | None:
    """Cada `\\text{}`/`\\mathrm{}`: com número dentro, é desembrulhado; se for unidade,
    grau ou enchimento neutro (`_texto_neutro`), sai com o expoente; qualquer outro texto
    é qualificador ou símbolo, e a leitura recusa (None). Separador `/`, `*`, `\\cdot` que
    sobra entre unidades tiradas sai também."""
    saida, i, tirou = "", 0, False
    for m in _UNIDADE_EM_TEXTO.finditer(s):
        dentro = re.sub(r"\\[,;!: ]", " ", m.group(1)).strip()
        saida += s[i:m.start()]
        if re.search(r"\d", dentro):
            saida += f" {dentro} "
        elif not _texto_neutro(dentro, unidade):
            return None
        else:
            saida += " "
            tirou = True
        i = m.end()
    saida += s[i:]
    if tirou:
        saida = re.sub(r"(?:\s*(?:/|\*|\\cdot)\s*)+$", "", saida.strip())
        saida = re.sub(r"(?<=\d)\s*(?:/|\*|\\cdot)\s*(?=\s*$)", "", saida)
    return saida


def numero(s: str, estrito: bool = False, unidade: str | None = None) -> float | None:
    """O valor numérico, ou None. Tira unidade em `\\mathrm{}`/`\\text{}` (com o expoente
    dela), grau, por cento, separador de milhar e a incerteza `\\pm` (vale o valor
    central). Com `estrito` (extremos de intervalo), só vale o que o parser lê inteiro e
    sem símbolo livre. Sem `estrito`, vale também `número + unidade` (`_cauda_ok`)."""
    s = limpar(s)
    s = re.sub(r"(?<![\d.])(\d{1,3})((?:,\d{3})+)(?![\d.])",
               lambda m: m.group(1) + m.group(2).replace(",", ""), s)
    if len(dividir_nivel_zero(s, r",|;")) > 1:
        return None
    s = _depois_do_igual(s).strip()
    gm = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*\^\s*\{?\s*\\circ\s*\}?\s*(\d+(?:\.\d+)?)\s*"
                      r"(?:_\{prime\}|')\s*", s)
    if gm:
        return float(gm.group(1)) + float(gm.group(2)) / 60
    if _CONECTIVO.search(s):
        return None
    # incerteza: vale o valor central. A escala depois da incerteza só sai junto se o valor
    # central já tiver a própria escala, com `\times`, `\cdot`, `*` ou o `10^` nu
    # (`3 \cdot 10^{3} \pm 2 \times 10^{2}` é 3e3); senão é do central
    # (`1.50 \pm 0.05 \times 10^{-3}` é 1,5e-3)
    if "\\pm" in s:
        escala_propria = re.search(r"10\s*\^", s[:s.index("\\pm")]) is not None
        s = re.sub(r"\\pm\s*\d+(?:\.\d*)?"
                   + (r"(?:\s*(?:\\times|\\cdot|\*)\s*10\s*\^\s*\{?\s*[+-]?\d+\s*\}?)?"
                      if escala_propria else ""), " ", s)
    s = _tirar_unidade_do_item(s, unidade)
    # o prefixo μ antes de uma unidade em `\mathrm{}` vai para dentro dela
    s = re.sub(r"\\mu\s*\\(mathrm|text|rm)\s*\{", r"\\\1{μ", s)
    s = _tirar_texto(s, unidade)
    if s is None:
        return None
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?\s*[CF]?|\\circ|\\%|%|\\degree|\{\}", " ", s)
    s = re.sub(r"(?:\s*(?:\*|\\cdot)\s*)+$", "", s).strip()
    e = expressao(s) if s else None
    if e is not None:
        if e.free_symbols == {sp.Symbol("e")}:
            e = e.subs(sp.Symbol("e"), sp.E)
        if not e.free_symbols:
            try:
                v = complex(sp.N(e))
            except Exception:
                v = None
            if (v is not None and math.isfinite(v.real)
                    and abs(v.imag) <= 1e-12 * max(1.0, abs(v.real))):
                return v.real
            return None
    if estrito:
        return None
    m = re.fullmatch(rf"\s*({_NUM})\s*(.*?)\s*", s)
    if m and _cauda_ok(m.group(2), unidade):
        return _valor_num(m.group(1))
    return None


_NOME_EM_TEXTO = re.compile(r"\s*\\(?:text|mathrm|textrm)\s*\{\s*([A-Za-z][A-Za-z ]*)\}\s*")


def _nome_em_texto(lado: str) -> bool:
    """`\\text{speed} = 80.4`: um nome de grandeza em texto, e não um qualificador
    (`\\text{upper bound} = 60`, `\\text{at most} = …`, `\\text{error} = …`, rodada 4)."""
    m = _NOME_EM_TEXTO.fullmatch(lado)
    return bool(m) and not (_QUALIFICADOR.search(m.group(1)) or re.search(
        r"\b(?:error|uncertainty|limit|bound|tolerance|deviation|range|estimate)\b",
        m.group(1), re.I))


def _numero_da_resposta(resposta: str, gabarito: str, unidade: str | None,
                        iguais: Callable[[float, float], bool]) -> float | None:
    """O número de uma resposta que pode vir em cadeia (`v = 4.2`, `E \\approx -13.6`,
    `v = \\sqrt{2gh} \\approx 80.4`, `\\approx 13.06`).

    Lados vazios (a resposta que começa por `\\approx`) não contam. O PRIMEIRO lado tem de
    ser o lado esquerdo do gabarito, ou, quando o gabarito não tem lado esquerdo, uma
    grandeza solta sem expoente (também `|v|` e `\\text{speed}`), ou um número igual ao
    final (na tolerância do item). No meio, uma expressão simbólica (com símbolos livres)
    não se confere; um número tem de ser igual ao final. `2I + 1 = 7/2`, `v^{2} = 1.41` e
    `8.2 \\approx 7.617` não passam."""
    # `|v|` e `\left|v\right|` de uma grandeza solta valem como a grandeza
    bruto = re.sub(r"\\left\s*\|\s*([^|]+?)\s*\\right\s*\||\|\s*([^|]+?)\s*\|",
                   lambda m: (m.group(1) or m.group(2))
                   if _GRANDEZA_SOLTA.fullmatch(m.group(1) or m.group(2)) else m.group(0),
                   resposta)
    lados = [(x, sep) for x, sep in _dividir_com_sep(limpar(bruto), _IGUAL) if x.strip()]
    if not lados:
        return None
    if len(lados) == 1:
        return numero(lados[0][0], unidade=unidade)
    valor = numero(lados[-1][0], unidade=unidade)
    if valor is None:
        return None
    g_lados = dividir_nivel_zero(limpar(gabarito), _IGUAL)
    esquerdo_g = g_lados[0] if len(g_lados) >= 2 else None
    for k, (lado, _sep) in enumerate(lados[:-1]):
        v = numero(lado, unidade=unidade)
        if v is not None:
            if not iguais(v, valor):
                return None
            continue
        if k == 0:
            if esquerdo_g is not None and _mesmo_lado(lado, esquerdo_g):
                continue
            if esquerdo_g is None and (_GRANDEZA_SOLTA.fullmatch(lado) or _nome_em_texto(lado)):
                continue
            return None
        e = expressao(lado)
        if e is None or not e.free_symbols:
            return None
    return valor


# ── comparadores ─────────────────────────────────────────────────────────────


def _com_teto(fn: Callable[[], Veredito], segundos: float = LIMITE_S) -> tuple[Veredito, bool]:
    caixa: dict = {}

    def corpo():
        try:
            caixa["v"] = fn()
        except Exception:
            caixa["v"] = Veredito.ERROR

    t = threading.Thread(target=corpo, daemon=True, name="g2-corretor")
    t.start()
    t.join(segundos)
    return (Veredito.INCONCLUSIVE, True) if t.is_alive() else (caixa["v"], False)


def threads_vazadas() -> int:
    """Comparações que estouraram o teto e seguem vivas, gastando CPU e disputando o GIL.
    Python não mata thread: quem as tem tem de reiniciar o processo (`g2_correcao`)."""
    return sum(t.name == "g2-corretor" and t.is_alive() for t in threading.enumerate())


def com_teto(fn: Callable[[], Veredito], segundos: float = LIMITE_S) -> Veredito:
    """A comparação inteira sob o teto. O teto é de relógio, e o relógio depende da carga:
    quem estoura é refeito UMA vez antes de virar INCONCLUSIVE. A correção da medida roda
    sem outra carga pesada na máquina."""
    v, estourou = _com_teto(fn, segundos)
    if estourou:
        v, _ = _com_teto(fn, segundos)
    return v


def comparar_numero(resposta: str, gabarito: str, rel: float | None = None,
                    abs_: float | None = None, simetrico: bool = False,
                    unidade: str | None = None) -> Veredito:
    """|r − g| ≤ abs_ (absoluto) ou ≤ rel·|g| (relativo ao gabarito; com `simetrico`,
    rel·max(|r|, |g|), o `math.isclose` do SciBench). `unidade` é a do item."""
    if abs_ is None and rel is None:
        raise ValueError("comparar_numero precisa de rel ou abs_")

    def iguais(r: float, g: float) -> bool:
        if abs_ is not None:
            return abs(r - g) <= abs_ * (1 + 1e-9)
        escala = max(abs(r), abs(g)) if simetrico else abs(g)
        if escala == 0:
            return r == g
        return abs(r - g) <= rel * escala * (1 + 1e-12)

    g = numero(gabarito)
    if g is None:
        return Veredito.INCONCLUSIVE
    r = _numero_da_resposta(resposta, gabarito, unidade, iguais)
    if r is None:
        return Veredito.FAIL
    return Veredito.PASS if iguais(r, g) else Veredito.FAIL


def _sem_inteiros_gigantes(e: sp.Expr) -> sp.Expr:
    """Inteiros grandes viram Float de 50 dígitos: `10^{10^{7}}` era calculado como inteiro
    Python a cada amostra, estourava o teto, e a thread que continuava viva segurava o GIL
    e contaminava os itens seguintes (estresse, rodada 2)."""
    def potencia_grande(x) -> bool:
        return (isinstance(x, sp.Pow) and x.is_number and x.exp.is_number
                and (isinstance(x.exp, sp.Pow) or abs(sp.N(x.exp, 15)) > 300))

    try:
        # `10^{10^{7}}`: o expoente é uma potência não avaliada, não um inteiro grande
        e = e.replace(potencia_grande,
                      lambda x: sp.Float(sp.N(x.base, 50), 50) ** sp.Float(sp.N(x.exp, 50), 50))
        grandes = {n: sp.Float(n, 50) for n in e.atoms(sp.Integer) if abs(n) > 10 ** 4}
    except Exception:
        return e
    return e.xreplace(grandes) if grandes else e


def _amostrar_iguais(a: sp.Expr, b: sp.Expr, razao: bool = False,
                     positiva: bool = False) -> Veredito:
    """a e b em N_AMOSTRAS pontos reais positivos a 50 dígitos. Sem `razao`: iguais em
    todos → PASS. Com `razao`: a/b constante não nula (positiva, com `positiva`) → PASS."""
    a, b = _sem_inteiros_gigantes(a), _sem_inteiros_gigantes(b)
    simbolos = sorted(a.free_symbols | b.free_symbols, key=str)
    rng = random.Random(17)
    mp.mp.dps = 50
    try:
        fa = sp.lambdify(simbolos, a, "mpmath")
        fb = sp.lambdify(simbolos, b, "mpmath")
    except Exception:
        return Veredito.INCONCLUSIVE
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
        if not all(mp.isfinite(x) for x in (va.real, va.imag, vb.real, vb.imag)):
            continue
        if razao:
            if abs(vb) == 0:
                continue
            valores.append(va / vb)
        else:
            valores.append(abs(va - vb) / max(abs(va), abs(vb), mp.mpf(1)))
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


def _torre(e: sp.Basic, limite: int = 3) -> bool:
    """Potência ou exponencial com expoente simbólico aninhada em `limite` níveis ou mais
    (`e^{e^{e^{v}}}`, `x^{x^{x}}`): a amostragem estourava o teto e deixava a thread viva
    (rodada 4). `e^{-x^2}` e `e^{-e^{x}}` passam."""
    def profundidade(x: sp.Basic) -> int:
        propria = (isinstance(x, sp.Pow) and bool(x.exp.free_symbols)) or isinstance(x, sp.exp)
        return int(propria) + max((profundidade(a) for a in x.args), default=0)

    try:
        return profundidade(e) >= limite
    except RecursionError:
        return True


def _equivalentes(a: sp.Expr, b: sp.Expr) -> Veredito:
    """Estrutural; amostragem numérica; `simplify` só se a amostragem não decidir."""
    try:
        if a == b or sp.srepr(a) == sp.srepr(b):
            return Veredito.PASS
    except Exception:
        pass
    if _torre(a) or _torre(b):
        return Veredito.INCONCLUSIVE
    if not (a.free_symbols | b.free_symbols):
        a, b = _sem_inteiros_gigantes(a), _sem_inteiros_gigantes(b)
        try:
            va, vb = complex(sp.N(a, 30)), complex(sp.N(b, 30))
        except Exception:
            return Veredito.INCONCLUSIVE
        if not (math.isfinite(abs(va)) and math.isfinite(abs(vb))):
            # ±∞ só casa com o mesmo infinito (antes `inf <= inf` passava qualquer número)
            return Veredito.PASS if va == vb else Veredito.FAIL
        # relativa pura: o piso de 1 aprovava qualquer valor abaixo de 1e-9
        escala = max(abs(va), abs(vb))
        if escala == 0:
            return Veredito.PASS
        return Veredito.PASS if abs(va - vb) <= TOL_EXPRESSAO * escala else Veredito.FAIL
    v = _amostrar_iguais(a, b)
    if v is not Veredito.INCONCLUSIVE:
        return v
    try:
        if sp.simplify(a - b) == 0:
            return Veredito.PASS
    except Exception:
        pass
    return Veredito.INCONCLUSIVE


# Integral, soma, produtório, ∂, limite, ∇, bra-ket, diferencial explícito, barra de
# avaliação: não verificável por regra (§3.7.4); o autoteste exclui.
_CALCULO = re.compile(r"\\(?:int|iint|iiint|oint|sum|prod|partial|lim|nabla|langle|rangle)"
                      r"(?![A-Za-z])|\\begin\s*\{")
_BARRA_DE_AVALIACAO = re.compile(r"\|\s*_")
_DIFERENCIAL = re.compile(r"\\mathrm\s*\{\s*d\s*\}|\{\s*\\rm\s+d\s*\}|"
                          r"\\frac\s*\{\s*d\s*(?:\^\s*\{?\s*\d\s*\}?)?\s*[^{}]*\}\s*\{\s*d\s*"
                          r"[A-Za-z\\]")


# Na RESPOSTA, também a derivada escrita com o `d` no meio do numerador
# (`\frac{Q\, d\sigma}{d\varepsilon}`): com grega, o `d` virava produto, e a derivada não
# calculada passava como o quociente σQ/ε (rodada 4).
_DIFERENCIAL_NA_RESPOSTA = re.compile(r"\\frac\s*\{[^{}]*(?<![A-Za-z\\])d\s*(?:\\[A-Za-z]+|[A-Za-z(])"
                                      r"[^{}]*\}\s*\{\s*d\s*(?:\\[A-Za-z]+|[A-Za-z])")


def verificavel(gabarito: str) -> bool:
    return (_DIFERENCIAL.search(gabarito) is None
            and _BARRA_DE_AVALIACAO.search(gabarito) is None
            and _CALCULO.search(limpar(gabarito)) is None)


def _resposta_verificavel(resposta: str) -> bool:
    """A resposta sem cálculo (∑ até 2·10⁶, ∫, derivada não calculada): com cálculo, ela
    reprova, porque não se confere (rodada 4)."""
    return verificavel(resposta) and _DIFERENCIAL_NA_RESPOSTA.search(resposta) is None


def texto_normalizado(s: str) -> str:
    """Para o gabarito que o parser não lê: sem espaços e sem `\\mathrm{}`/`\\text{}`, mas
    COM as chaves (sem elas, `e^{-i\\omega t}` e `e^{-i}\\omega t` colapsavam)."""
    s = _TEXTO.sub(r"\1", limpar(s))
    s = re.sub(r"\{(\w)\}", r"\1", s)
    return re.sub(r"\s", "", s)


# Uma grandeza só, com subscrito e argumentos opcionais: `H`, `c_{p}`, `\kappa(p, T)`.
# Letra solta é UMA letra: `kM` é k·M em LaTeX, não um nome (e `kM = …` não passava a ser
# "uma grandeza igual à resposta").
_SUBSCRITO = r"(?:_\s*(?:\{(?:[^{}]|\{[^{}]*\})*\}|\w))?"
_GRANDEZA = re.compile(rf"\s*(?:\\[Dd]elta\s*)?(?:\\[A-Za-z]+|[A-Za-z])\s*{_SUBSCRITO}"
                       r"\s*(?:\^\s*\{[^{}]*\})?\s*(?:\([^()]*\))?\s*")
# Do OUTRO lado de uma resposta, quando o gabarito não tem lado esquerdo: uma grandeza
# sem expoente e que não é função (`v^{2} = …`, `\ln(C) = …`, `C^{-1} = …` não valem). O
# comando tem de ser letra grega, `\ell` ou `\hbar`: com a lista negra de funções,
# `\arcsin\theta = X` e `\coth = X` valiam como `v = X` (rodada 4).
_GRANDEZA_SOLTA = re.compile(
    r"\s*(?:\\[Dd]elta\s*)?(?:\\(?:" + "|".join(_GREGAS + ("ell",)) + r")(?![A-Za-z])|[A-Za-z])"
    rf"\s*{_SUBSCRITO}\s*(?:\([^()]*\))?\s*")


def lado_esquerdo_e_grandeza(equacao: str) -> bool:
    lados = dividir_nivel_zero(limpar(equacao), "=")
    return len(lados) == 2 and _GRANDEZA.fullmatch(lados[0]) is not None


def _sem_argumentos(s: str) -> str:
    """`v(t)` → `v`, `T(r, t \\to \\infty)` → `T`: notação funcional de uma grandeza."""
    t = re.sub(r"\s*\([^()]*\)\s*$", "", s.strip())
    return t if _GRANDEZA.fullmatch(t) else s


def _argumentos(s: str) -> str | None:
    m = re.search(r"\(([^()]*)\)\s*$", s.strip())
    return m.group(1) if m and _sem_argumentos(s) != s else None


def _mesmo_lado(a: str, b: str) -> bool:
    """Dois lados esquerdos são o mesmo: equivalentes se o parser lê os dois, ou o mesmo
    texto normalizado. Sem a lista de argumentos só quando UM lado não a tem e a do outro
    é de variáveis (`v` e `v(t)`): `v(0)` não é `v`, nem `B(a)` é `B(r)` (rodada 3)."""
    pares = [(a, b)]
    xa, xb = _argumentos(a), _argumentos(b)
    if (xa is None) != (xb is None) and not re.search(r"\d", xa or xb):
        pares.append((_sem_argumentos(a), _sem_argumentos(b)))
    for x, y in pares:
        ex, ey = expressao(x), expressao(y)
        if ex is not None and ey is not None:
            if _equivalentes(ex, ey) is Veredito.PASS:
                return True
        elif texto_normalizado(x) == texto_normalizado(y):
            return True
    return False


def _sem_texto_em_volta(s: str) -> str:
    """Tira de uma resposta simbólica só o que não muda o sentido: enchimento neutro em
    `\\text{}` no começo ("the result is"), a cláusula explicativa (", where …"), `\\text{ }`
    vazio e unidades no fim (com expoente e chaves aninhadas). Um qualificador ("not",
    "less than", "or zero") FICA, e o parser estrito então recusa a resposta."""
    s = limpar(s)
    # só a definição de símbolos (", where …", "i.e.") sai, e só se não trouxer outra
    # resposta ou qualificador; "for", "with", "since" ficam e o parser recusa (rodada 3:
    # "X for case 1, Y for case 2" passava pela primeira)
    # (rodada 4: "whereas" casava com "where", e "i.e. at most …" saía; a cláusula com
    # desigualdade é condição, não definição)
    m = re.search(r",?\s*\\text\s*\{\s*(?:where\b|i\.e\.)", s)
    if m and not (_QUALIFICADOR.search(s[m.end():]) or re.search(
            r"\b(?:otherwise|else|also|alternatively|for|with|however|but|whereas|when|while|"
            r"elsewhere|after|before|exact|general|case)\b|<|>|\\[lg]eq?\b|\\[lg]t\b|\\neq\b",
            s[m.end():], re.I)):
        s = s[:m.start()]
    s = re.sub(r"\\text\s*\{\s*\}", " ", s)
    m = re.match(r"\s*\\(?:text|textrm|mathrm)\s*\{([^{}]*)\}", s)
    if m and m.group(1).strip().lower().rstrip(":") in _NEUTRO | {"the result is", "the answer is"}:
        s = s[m.end():]
    while True:
        m = re.search(_UNIDADE_EM_TEXTO.pattern + r"\s*$", s)
        if not m or not _cauda_e_unidade(re.sub(r"\\[,;!: ]", " ", m.group(1))):
            break
        s = s[:m.start()]
    return s.strip()


def comparar_expressao(resposta: str, gabarito: str) -> Veredito:
    """Equivalência com todas as variáveis positivas.

    Gabarito `L = expr`: compara-se expr. Resposta com relações (`=`, `\\approx`): PASS se
    um dos lados for equivalente a expr e cada outro lado for (a) o mesmo L do gabarito,
    (b) uma grandeza solta, quando o gabarito não tem L, ou (c) um número depois de
    `\\approx`. `2M = expr`, `\\omega = …` contra `\\omega' = …` e `x = 3, y = 2` não
    passam."""
    if not verificavel(gabarito):
        return Veredito.INCONCLUSIVE
    if not _resposta_verificavel(resposta):      # cálculo na RESPOSTA (\sum até 2·10⁶, ∫): não se confere
        return Veredito.FAIL
    g_lados = dividir_nivel_zero(limpar(gabarito), _IGUAL)
    esquerdo_g = g_lados[0] if len(g_lados) >= 2 else None
    g = expressao(g_lados[-1])
    if g is None:
        return (Veredito.PASS if texto_normalizado(resposta) == texto_normalizado(gabarito)
                else Veredito.FAIL)
    r_lados = _dividir_com_sep(_sem_texto_em_volta(resposta), _IGUAL)
    if len(r_lados) == 1:
        r = expressao(r_lados[0][0])
        return Veredito.FAIL if r is None else _equivalentes(r, g)
    vereditos = []
    for i, (lado, _sep) in enumerate(r_lados):
        e = expressao(lado)
        v = _equivalentes(e, g) if e is not None else Veredito.FAIL
        vereditos.append(v)
        if v is not Veredito.PASS:
            continue
        outros_ok = True
        for j, (outro, sep) in enumerate(r_lados):
            if j == i:
                continue
            if sep in _APROXIMADO and numero(outro) is not None:
                continue
            if esquerdo_g is not None and _mesmo_lado(outro, esquerdo_g):
                continue
            if esquerdo_g is None and _GRANDEZA_SOLTA.fullmatch(outro):
                continue
            outros_ok = False
            break
        if outros_ok:
            return Veredito.PASS
    return (Veredito.INCONCLUSIVE if Veredito.INCONCLUSIVE in vereditos
            and Veredito.PASS not in vereditos else Veredito.FAIL)


def _lados(s: str) -> tuple[sp.Expr, sp.Expr] | None:
    partes = dividir_nivel_zero(limpar(s), "=")
    if len(partes) != 2:
        return None
    lados = (expressao(partes[0]), expressao(partes[1]))
    return None if None in lados else lados  # type: ignore[return-value]


def _isolar(f: sp.Expr, x: sp.Symbol) -> sp.Expr | None:
    """A solução de f = 0 em x quando f é LINEAR em x, em 1/x ou em √x (x > 0), sem
    `sp.solve`: o `solve` entrava num `simplify`/`factor` sem fim, e a thread que estourava
    o teto seguia viva e travava o autoteste inteiro (2026-10-22)."""
    h = sp.Symbol("h_isolado", positive=True)
    for troca, volta in ((h, lambda s: s), (1 / h, lambda s: 1 / s), (h ** 2, lambda s: s ** 2)):
        fh = f.xreplace({x: troca})
        a = sp.diff(fh, h)
        if a == 0 or h in a.free_symbols:
            continue
        s = -fh.xreplace({h: 0}) / a
        if s.has(sp.zoo, sp.nan) or x in s.free_symbols:
            continue
        return volta(s)
    return None


def _isolando_a_grandeza(resposta: str, gabarito: str) -> Veredito:
    """Gabarito `G = expr` com G um símbolo: isola G na equação da resposta; PASS se ela
    for linear em G, 1/G ou √G e a solução for equivalente a expr (`E V^2 = \\pi B u_0^2`,
    `1/E = …`)."""
    g = _lados(gabarito)
    r = _lados(_sem_texto_em_volta(resposta))
    if g is None or r is None or not isinstance(g[0], sp.Symbol):
        return Veredito.FAIL
    try:
        sol = _isolar(r[0] - r[1], g[0])
    except Exception:
        return Veredito.INCONCLUSIVE
    if sol is None:
        return Veredito.FAIL
    # a solução tem de satisfazer a equação da resposta: pela raiz, `\sqrt{G} = -X` dava
    # G = X², que não a satisfaz (rodada 4)
    residuo = _equivalentes((r[0] - r[1]).xreplace({g[0]: sol}), sp.Integer(0))
    if residuo is not Veredito.PASS:
        return residuo
    return _equivalentes(sol, g[1])


def comparar_equacao(resposta: str, gabarito: str) -> Veredito:
    """A regra (§3.7.4): equivalentes se (L − R) da resposta ÷ (L − R) do gabarito for
    uma constante não nula. Só isso: `n_1 = n_2 e^{…}` contra `n_1/n_2 = e^{…}` não passa
    (a razão é n_2), igual nos dois braços."""
    if not verificavel(gabarito):
        return Veredito.INCONCLUSIVE
    g = _lados(gabarito)
    if g is None:
        return Veredito.INCONCLUSIVE
    if not _resposta_verificavel(resposta):
        return Veredito.FAIL
    r = _lados(_sem_texto_em_volta(resposta))
    if r is None:
        return Veredito.FAIL
    v = _amostrar_iguais(r[0] - r[1], g[0] - g[1], razao=True)
    if v is not Veredito.INCONCLUSIVE:
        return v
    try:
        q = sp.simplify((r[0] - r[1]) / (g[0] - g[1]))
        if q.is_number and q != 0 and q.is_finite:
            return Veredito.PASS
    except Exception:
        pass
    return Veredito.INCONCLUSIVE


def comparar_definicao(resposta: str, gabarito: str) -> Veredito:
    """Gabarito `G = expr` (G uma grandeza): o que se corrige é o VALOR de G. Passa pelo
    lado direito (com G conferido), pela regra da equação (lados trocados, termos
    movidos) ou isolando G na resposta. Desvio declarado no DOC-12 §3.7.4: a regra da EQ,
    sozinha, reprovaria `E V^2 = \\pi B u_0^2` contra `E = \\pi B u_0^2 / V^2`."""
    principal = comparar_expressao(resposta, gabarito)
    if principal is Veredito.PASS:
        return principal
    # os caminhos da equação só PROMOVEM a PASS: um gabarito que não se lê como equação
    # (`E_{gap} \approx …`) não torna inconclusivo o que o caminho principal decidiu
    for caminho in (comparar_equacao, _isolando_a_grandeza):
        if caminho(resposta, gabarito) is Veredito.PASS:
            return Veredito.PASS
    return principal


# ── desigualdade e intervalo ─────────────────────────────────────────────────

_RELACAO = re.compile(r"\\geqslant|\\leqslant|\\geq?|\\leq?|\\gt|\\lt|≥|≤|>=|<=|>|<")
_MAIOR = (">", ">=", "\\gt", "\\ge", "\\geq", "\\geqslant", "≥")
_ESTRITA = (">", "<", "\\gt", "\\lt")
_INVERSA = {">": "<", "<": ">", ">=": "<=", "<=": ">=", "\\gt": "\\lt", "\\lt": "\\gt",
            "\\ge": "\\le", "\\le": "\\ge", "\\geq": "\\leq", "\\leq": "\\geq",
            "\\geqslant": "\\leqslant", "\\leqslant": "\\geqslant", "≥": "≤", "≤": "≥"}


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
    f = lados[0] - lados[1] if op in _MAIOR else lados[1] - lados[0]
    return f, op in _ESTRITA


def comparar_desigualdade(resposta: str, gabarito: str) -> Veredito:
    """Mesma estrição, e f da resposta ÷ f do gabarito constante POSITIVA."""
    g = _desigualdade(gabarito)
    if g is None:
        return Veredito.INCONCLUSIVE
    if not _resposta_verificavel(resposta):
        return Veredito.FAIL
    r = _desigualdade(resposta)
    if r is None or r[1] != g[1]:
        return Veredito.FAIL
    v = _amostrar_iguais(r[0], g[0], razao=True, positiva=True)
    if v is not Veredito.INCONCLUSIVE:
        return v
    try:
        q = sp.simplify(r[0] / g[0])
        if q.is_number and q.is_finite and q.is_positive:
            return Veredito.PASS
    except Exception:
        pass
    return Veredito.INCONCLUSIVE


def _como_intervalo(s: str) -> tuple[str, str, str, str, str | None] | None:
    """Intervalo literal, `x \\in [a, b)`, `x ≥ a`, `a < x ≤ b` → (abre, a, b, fecha, x), com
    x a variável quando a forma a traz (senão None)."""
    s = limpar(s).replace("\\infty", "oo").replace("\\infin", "oo")
    if re.search(r"\\not(?![A-Za-z])|\\notin\b|\\neq\b", s):     # `\not\in (a, b)` (rodada 4)
        return None
    var = None
    m = re.match(rf"^\s*((?:\\?[A-Za-z]+){_SUBSCRITO})\s*\\in\s*", s)
    if m:
        var, s = m.group(1), s[m.end():]
    fim = max(s.rfind(")"), s.rfind("]"))
    if 0 <= fim < len(s) - 1:
        cauda = _UNIDADE_EM_TEXTO.sub(" ", s[fim + 1:])
        if _cauda_e_unidade(cauda):
            s = s[:fim + 1]
    m = re.fullmatch(r"\s*([\[(])(.*)([\])])\s*", s)
    if m:
        meio = _partes(m.group(2))
        return (m.group(1), meio[0], meio[1], m.group(3), var) if len(meio) == 2 else None
    ops = list(_RELACAO.finditer(s))
    if not ops:
        return None
    pedacos = ([s[:ops[0].start()]]
               + [s[a.end():b.start()] for a, b in zip(ops, ops[1:], strict=False)]
               + [s[ops[-1].end():]])
    if len(ops) == 1:
        x, a = pedacos
        op = ops[0].group()
        # a variável é uma grandeza SOLTA: `\tan(\alpha)` (o `\tan\alpha` depois de
        # `limpar`) é extremo, não variável (rodada 3)
        if not _GRANDEZA_SOLTA.fullmatch(x):
            x, a, op = a, x, _INVERSA[op]
        if not _GRANDEZA_SOLTA.fullmatch(x):
            return None
        aberto = op in _ESTRITA
        return (("(", a, "oo", ")", x) if op in _MAIOR and aberto else
                ("[", a, "oo", ")", x) if op in _MAIOR else
                ("(", "-oo", a, ")", x) if aberto else ("(", "-oo", a, "]", x))
    if (len(ops) == 2 and all(o.group() not in _MAIOR for o in ops)
            and _GRANDEZA_SOLTA.fullmatch(pedacos[1])):
        return ("(" if ops[0].group() in _ESTRITA else "[", pedacos[0], pedacos[2],
                ")" if ops[1].group() in _ESTRITA else "]", pedacos[1])
    return None


def _extremo_igual(a: str, b: str, rel: float) -> Veredito:
    a, b = a.strip().lstrip("+"), b.strip().lstrip("+")
    if "oo" in (a + b):
        return Veredito.PASS if a.replace(" ", "") == b.replace(" ", "") else Veredito.FAIL
    na, nb = numero(a, estrito=True), numero(b, estrito=True)
    if na is not None and nb is not None:
        if nb == 0:
            return Veredito.PASS if na == 0 else Veredito.FAIL
        return Veredito.PASS if abs(na - nb) <= rel * abs(nb) else Veredito.FAIL
    return comparar_expressao(a, b)


def comparar_intervalo(resposta: str, gabarito: str, rel: float = 1e-2) -> Veredito:
    """Mesmos colchetes e extremos iguais (numéricos a `rel`, simbólicos por
    equivalência; ∞ só com o mesmo ∞). A resposta pode vir como desigualdade, com
    `x \\in`, ou com a unidade depois do parêntese. União por `\\cup`. Gabarito que é uma
    desigualdade sem forma de intervalo é comparado como desigualdade."""
    desigualdade = _RELACAO.search(limpar(gabarito)) is not None
    if desigualdade and comparar_desigualdade(resposta, gabarito) is Veredito.PASS:
        return Veredito.PASS
    gs = [_como_intervalo(p) for p in re.split(r"\\cup|∪", gabarito)]
    if any(x is None for x in gs):
        return comparar_desigualdade(resposta, gabarito) if desigualdade else Veredito.INCONCLUSIVE
    rs = [_como_intervalo(p) for p in re.split(r"\\cup|∪", resposta)]
    if len(rs) != len(gs) or any(x is None for x in rs):
        return Veredito.FAIL
    for r, g in zip(rs, gs, strict=True):
        if r[0] != g[0] or r[3] != g[3]:
            return Veredito.FAIL
        # com variável dos dois lados, tem de ser a mesma (`N_2 < N/2` contra `N_1 < N/2`);
        # intervalo puro contra gabarito com variável vale (a variável é a do enunciado)
        if r[4] is not None and g[4] is not None and not _mesmo_lado(r[4], g[4]):
            return Veredito.FAIL
        for i in (1, 2):
            v = _extremo_igual(r[i], g[i], rel)
            if v is not Veredito.PASS:
                return v
    return Veredito.PASS


# ── letras, V/F e várias respostas ───────────────────────────────────────────


def letras(s: str) -> frozenset[str]:
    """As alternativas marcadas, só se a resposta for SÓ alternativa: `C`, `(C)`, `C, D`,
    `AC`; ou um rótulo explícito no começo (`(C) …`, `C. …`). Fora disso, vazio."""
    t = limpar(s)
    for _ in range(3):                                   # `\text{\textbf{C}}`
        t = _TEXTO.sub(r" \1 ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(?:option|choice|answer)\s*:?\s*", "", t, flags=re.I)
    t = re.sub(r"^\[\s*([A-Ja-j])\s*\]$", r"\1", t)
    # primeiro a LISTA inteira: `C`, `(C)`, `C, D`, `(A), (C)`, `(B)(D)`
    item = r"\(?\s*[A-Ja-j]\s*\)?"
    sep = r"\s*(?:,|;|&|\band\b|\s)?\s*"
    if re.fullmatch(rf"{item}(?:{sep}{item})*", t):
        # sem o "and" antes de ler as letras: o "a" e o "d" dele viravam alternativas
        return frozenset(x.upper() for x in re.findall(r"[A-Ja-j]",
                                                         re.sub(r"\band\b", " ", t)))
    if re.fullmatch(r"[A-J]{2,}", t):
        return frozenset(t)
    # um rótulo no começo, com o texto da opção depois (`(C) E = h\nu`, `C. 4.2 J`): só se
    # não houver OUTRO rótulo no resto, nem hesitação. O texto da opção copiado pode ter
    # not, no, or ("(D) The spin is not conserved"); o que recusa é a palavra que aponta
    # para outra letra ("Maybe D", "or (B)") ou a dúvida declarada (rodada 3).
    m = re.match(r"^(?:\(\s*([A-Ja-j])\s*\)|([A-Ja-j])\s*[.):])", t)
    if m:
        resto = t[m.end():]
        # a OUTRA letra é qualquer uma menos a do rótulo ("C. Option C is correct" segue C)
        letra = (m.group(1) or m.group(2)).upper()
        outras = "".join(x for x in "ABCDEFGHIJ" if x != letra)
        outra = rf"(?:\(\s*[{outras}{outras.lower()}]\s*\)|(?<![A-Za-z])[{outras}](?![A-Za-z]))"
        if (re.search(rf"\(\s*[{outras}{outras.lower()}]\s*\)|(?<![A-Za-z])(?<!\d)(?<!\d\s)"
                      rf"[{outras}{outras.lower()}]\s*[.):](?!\d)", resto)
                or re.search(r"\b(?i:or|not|maybe|perhaps|possibly|probably|alternatively|likely|"
                             rf"either|also|and|nor|vs\.?|versus|option|answer|choice)\s*[,:]?\s*{outra}",
                             resto)
                or re.search(r"\b(?:maybe|perhaps|not sure|unsure|uncertain)\b", resto, re.I)
                # a outra letra sozinha no resto ("C. B"), ou afirmada ("C. B is also
                # correct", "C. B too") (rodada 4)
                or re.fullmatch(rf"\s*[,;:]?\s*{outra}\s*\.?\s*", resto)
                or re.search(rf"{outra}\s*(?i:(?:is|are)\s+)?(?i:also|too|as well|equally|correct|"
                             r"valid|works|right)\b", resto)):
            return frozenset()
        return frozenset((m.group(1) or m.group(2)).upper())
    return frozenset()


def comparar_letras(resposta: str, gabarito: str) -> Veredito:
    g = letras(gabarito)
    if not g:
        return Veredito.INCONCLUSIVE
    return Veredito.PASS if letras(resposta) == g else Veredito.FAIL


_SIM = {"yes", "true", "correct"}
_NAO = {"no", "false", "incorrect"}


def verdade(s: str) -> bool | None:
    """A primeira palavra decide (yes/true/correct ou no/false/incorrect), desde que a
    oposta não apareça depois como resposta solta, no começo de uma oração (`Yes; no`,
    `No, yes`, `No (yes for …)`, `yes or no`). "No, it is not true" e "True, there is no
    contradiction" seguem decididos. Letra solta (T/F/Y/N) só se for a resposta inteira."""
    texto = limpar(s)
    for _ in range(3):                                   # `\text{\textbf{False}}`
        texto = _TEXTO.sub(r" \1 ", texto)
    texto = re.sub(r"\s+", " ", texto).strip().lower()
    palavras = re.findall(r"[a-z]+", texto)
    if not palavras:
        return None
    inteira = re.sub(r"[\s.()]", "", texto)
    if inteira in ("t", "y"):
        return True
    if inteira in ("f", "n"):
        return False
    if re.fullmatch(r"(?:the )?(?:statement|claim|answer) is (true|correct|false|incorrect)\.?",
                    texto):
        return re.search(r"(true|correct)$", texto.rstrip(".")) is not None
    if re.fullmatch(r"not (?:true|correct)\.?", texto):
        return False
    if re.match(r"(?:no idea|yes and no|not sure|yes or no|no or yes)\b", texto):
        return None
    # a primeira palavra que NÃO é a resposta: "No doubt, …", "No matter what", "Correct
    # answer is no", "False alarm", "True only if …" (rodada 4)
    if re.match(r"no\s+(?:doubt|question|matter|problem|wonder|way)\b|"
                r"(?:correct|true|false|incorrect)\s+(?:answer|statement|option|alarm|choice)\b|"
                r"(?:yes|no|true|false|correct|incorrect)\b\s*,?\s*(?:only\s+)?if\b", texto):
        return None
    m = re.match(r"(yes|no|true|false|correct|incorrect)\b", texto)
    if not m:
        return None
    valor = m.group(1) in _SIM
    opostas = _NAO if valor else _SIM
    # a oposta conta depois de "(" ("No (yes for a heavy nucleus)"), ou depois de pontuação,
    # "or" ou "is" quando é a palavra inteira da oração ("Yes; no.", "Yes, the answer is
    # no"); "Yes. No energy is lost" segue sim
    # Depois de "/" e "&" ("Yes/No", "Yes \& no") a oposta conta sempre (rodada 4).
    resto = texto[m.end():]
    for x in re.finditer(r"(\(|/|&|[,;.:!?\-–—]|\\cdot|\bor(?:\s+(?:rather|maybe|perhaps|possibly))?\b"
                         r"|\bis\b)\s*"
                         r"(yes|no|true|false|correct|incorrect)\b", resto):
        if x.group(2) in opostas and (x.group(1) in ("(", "/", "&")
                                      or re.match(r"\s*(?:[,;.:!?)\-–—]|$)", resto[x.end():])):
            return None
    return valor


def comparar_vf(resposta: str, gabarito: str) -> Veredito:
    g = verdade(gabarito)
    if g is None:
        return Veredito.INCONCLUSIVE
    return Veredito.PASS if verdade(resposta) is g else Veredito.FAIL


def _separadores(s: str) -> str:
    """`;`, `and`, `\\quad`, `\\\\` (e os de largura cheia, `、`, `和`, e o `&` de um ambiente
    `aligned`/`cases`) separam respostas como a vírgula."""
    for a, b in (("，", ","), ("；", ";"), ("、", ","), ("和", ","), ("及", ",")):
        s = s.replace(a, b)
    s = re.sub(r"\\begin\s*\{(?:aligned|align\*?|cases|array|gathered|split)\}(?:\{[^{}]*\})?"
               r"|\\end\s*\{(?:aligned|align\*?|cases|array|gathered|split)\}", " ", s)
    s = s.replace("&", " ")
    s = re.sub(r"\\text\s*\{\s*(?:and|,|;)\s*\}|\band\b|\\\\|;", ",", s)
    return re.sub(r"\\q?quad(?![A-Za-z])", ",", s)


def respostas_da_caixa(s: str) -> list[str]:
    """As respostas de uma caixa: separadores, `limpar` e a tupla `(a, b)` desembrulhada.
    A extração conta com ela e `comparar_varias` separa com ela; antes as duas contavam
    diferente (o `\\,` antes da unidade contava como vírgula)."""
    t = limpar(_separadores(s))
    partes = _partes(t)
    if len(partes) == 1:
        m = re.fullmatch(r"\s*\((.*)\)\s*", t)
        if m and len(_partes(m.group(1))) > 1:
            return _partes(m.group(1))
    return partes


def _partes(s: str) -> list[str]:
    """Separa por vírgulas de nível zero; vírgula de milhar (`86,000`) não separa."""
    s = re.sub(r"(?<![\d.])(\d{1,3})((?:,\d{3})+)(?![\d.])",
               lambda m: m.group(1) + m.group(2).replace(",", ""), s)
    return [p.strip() for p in dividir_nivel_zero(s, ",") if p.strip()]


def comparar_varias(resposta: str, gabaritos: list[str],
                    comparadores: list[Callable[[str, str], Veredito]]) -> Veredito:
    """Várias respostas, em qualquer ordem: cada gabarito casa com uma resposta
    diferente. Separadores: vírgula, `;`, `and`, `\\quad`; uma tupla `(a, b)` com
    exatamente n partes perde os parênteses."""
    rs = respostas_da_caixa(resposta)
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
