"""Expansão de macros definidas pelo autor (DOC-03 §2.2).

Sem isto, comparar equações entre duas versões do mesmo paper mede a notação do
autor, não o conteúdo. Medido em 2026-08-10 no arXiv 1607.04520:

    grupo                          n     usam macro
    casaram por forma canônica   571            20%
    NÃO casaram                  239            97%

O paper define 49 macros. A fonte escreve `\\Ecal_\\mu`, o RedPajama escreve
`\\mathcal{E}_\\mu`, e sem expansão isso conta como equação perdida. A auditoria
do S3b acusava 19,6% de degradação do RedPajama que era, em boa parte, este
defeito de medição.

## Escopo declarado, porque expansão completa exige um interpretador de TeX

O que é tratado:

    \\newcommand{\\nome}{corpo}              e a forma sem chaves
    \\newcommand{\\nome}[n]{corpo}           com #1..#9
    \\newcommand{\\nome}[n][padrão]{corpo}   argumento opcional
    \\renewcommand, \\providecommand, \\def simples
    \\DeclareMathOperator{\\nome}{texto}     → \\operatorname{texto}

O que NÃO é, e por que está certo não tentar:

    \\newenvironment          ambiente não é macro de equação
    \\def com padrão de       exige o parser de TeX de verdade; é raro em
      delimitadores            corpo de artigo
    expansão condicional      `\\ifmmode` e afins dependem de estado

**Um caso não tratado é deixado como está, nunca chutado.** Macro não expandida
faz a equação não casar — é falso negativo, que subestima a preservação. O erro
oposto, expandir errado, criaria falso POSITIVO e inflaria a preservação com
equações que não são as mesmas. Entre subestimar e mentir a favor, subestima.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Profundidade máxima. Macro que usa macro é comum (`\\Ecal` dentro de `\\Efull`);
# recursão infinita por definição circular, não. O teto para o laço sem
# depender de detectar o ciclo.
MAX_PROFUNDIDADE = 8

_NOME = r"\\([A-Za-z]+)"

# `\newcommand{\nome}[n][opt]{corpo}` — chaves no nome são opcionais.
_DEFINE = re.compile(
    r"\\(?:new|renew|provide)command\s*\*?\s*"
    r"(?:\{" + _NOME + r"\}|" + _NOME + r")"
    r"(?:\s*\[(\d)\])?"
    r"(?:\s*\[([^\]]*)\])?"
    r"\s*",
)
_DEF_SIMPLES = re.compile(r"\\def\s*" + _NOME + r"\s*(?=\{)")
_OPERADOR = re.compile(r"\\DeclareMathOperator\s*\*?\s*\{" + _NOME + r"\}\s*")


def _corpo_balanceado(s: str, i: int) -> tuple[str, int]:
    """Conteúdo do `{…}` que começa em `i`, respeitando aninhamento.

    Regex não serve aqui: o corpo de uma macro contém chaves, e `\\{[^}]*\\}`
    pararia na primeira interna. `\\{` escapado também não conta.
    """
    if i >= len(s) or s[i] != "{":
        return "", i
    nivel, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            nivel += 1
        elif c == "}":
            nivel -= 1
            if nivel == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)     # chave não fechada: pega o resto


def coletar_macros(tex: str) -> dict[str, tuple[int, str, str | None]]:
    """`nome` → (n_argumentos, corpo, padrão_opcional)."""
    macros: dict[str, tuple[int, str, str | None]] = {}

    for m in _DEFINE.finditer(tex):
        nome = m.group(1) or m.group(2)
        n = int(m.group(3)) if m.group(3) else 0
        corpo, _ = _corpo_balanceado(tex, m.end())
        if nome and corpo:
            macros[nome] = (n, corpo, m.group(4))

    for m in _DEF_SIMPLES.finditer(tex):
        corpo, _ = _corpo_balanceado(tex, m.end())
        if m.group(1) and corpo and m.group(1) not in macros:
            macros[m.group(1)] = (0, corpo, None)

    for m in _OPERADOR.finditer(tex):
        corpo, _ = _corpo_balanceado(tex, m.end())
        if m.group(1) and corpo:
            macros[m.group(1)] = (0, rf"\operatorname{{{corpo}}}", None)

    return macros


def _ler_argumentos(s: str, i: int, n: int, padrao: str | None) -> tuple[list[str], int]:
    """Lê `n` argumentos a partir de `i`. O opcional vem em `[…]`."""
    args: list[str] = []
    j = i
    if padrao is not None:
        if j < len(s) and s[j] == "[":
            fim = s.find("]", j)
            if fim < 0:
                return [], i
            args.append(s[j + 1:fim])
            j = fim + 1
        else:
            args.append(padrao)
    while len(args) < n:
        while j < len(s) and s[j] in " \t\n":
            j += 1
        if j < len(s) and s[j] == "{":
            corpo, j = _corpo_balanceado(s, j)
            args.append(corpo)
        elif j < len(s) and s[j] == "\\":
            # `\Ecal\mu` — um argumento pode ser uma macro sem chaves.
            m = re.match(r"\\[A-Za-z]+|\\.", s[j:])
            args.append(m.group(0))
            j += m.end()
        elif j < len(s):
            args.append(s[j])      # `\frac12`
            j += 1
        else:
            return [], i           # argumentos insuficientes: não expande
    return args, j


def preparar(tex: str, *, max_profundidade: int = MAX_PROFUNDIDADE) -> str:
    """**A entrada correta.** Coleta, remove as definições e expande, nesta ordem.

    ⚠️ As três operações têm de acontecer nesta sequência, e cada inversão tem um
    modo de falha próprio:

    `expandir` antes de `remover_definicoes` corrompe as definições — a macro é
    expandida DENTRO da própria definição:

        \\newcommand{\\dd}[2]{\\frac{#1}{#2}}
          vira  \\newcommand{\\frac{}}{[}2]{\\frac{#1}{#2}}

    `remover_definicoes` antes de `coletar_macros` apaga a informação de que a
    expansão precisa, e `expandir` vira silenciosamente um no-op.

    Chamar `expandir(tex)` direto em texto com definições cai no primeiro caso.
    Esta função existe para que a ordem não seja escolha de quem chama.
    """
    return expandir(remover_definicoes(tex), coletar_macros(tex),
                    max_profundidade=max_profundidade)


def expandir(tex: str, macros: dict[str, tuple[int, str, str | None]] | None = None,
             *, max_profundidade: int = MAX_PROFUNDIDADE) -> str:
    """Substitui as macros do autor pelos corpos, até o ponto fixo.

    Espera texto **sem** as linhas de definição — use `preparar`, que garante a
    ordem. Chamar aqui com definições presentes corrompe-as, e o aviso está na
    docstring de `preparar` com o exemplo.
    """
    if macros is None:
        macros = coletar_macros(tex)
    if not macros:
        return tex

    # Nomes longos primeiro: `\Ecalx` não pode ser lido como `\Ecal` + `x`.
    nomes = sorted(macros, key=len, reverse=True)
    padrao = re.compile(r"\\(" + "|".join(re.escape(n) for n in nomes) + r")(?![A-Za-z])")

    for _ in range(max_profundidade):
        saida, i, mudou = [], 0, False
        while True:
            m = padrao.search(tex, i)
            if not m:
                saida.append(tex[i:])
                break
            n, corpo, opc = macros[m.group(1)]
            args, fim = _ler_argumentos(tex, m.end(), n, opc)
            if n and not args:
                saida.append(tex[i:m.end()])   # sem argumentos: deixa como está
                i = m.end()
                continue
            for k, a in enumerate(args, 1):
                corpo = corpo.replace(f"#{k}", a)
            saida.append(tex[i:m.start()])
            saida.append(corpo)
            i, mudou = fim, True
        tex = "".join(saida)
        if not mudou:
            break
    return tex


def remover_definicoes(tex: str) -> str:
    """Tira as linhas de definição, para que não virem "equações".

    `\\newcommand{\\Ecal}{\\mathcal{E}}` no preâmbulo não é equação do artigo, e
    sem isto o extrator a colheria como uma.
    """
    for padrao in (_DEFINE, _DEF_SIMPLES, _OPERADOR):
        while True:
            m = padrao.search(tex)
            if not m:
                break
            _, fim = _corpo_balanceado(tex, m.end())
            tex = tex[:m.start()] + " " + tex[fim:]
    return tex
