"""Canonicalizador LaTeX — forma normal para COMPARAR equações (DOC-03 §3).

Destrava três frentes de uma vez:

  - dedup em nível de equação            (DOC-04 §5.5)
  - recuperação por fórmula              (DOC-13 §4.1) — a terceira perna da
                                          busca híbrida, que responde "que
                                          papers usam esta equação?"
  - descontaminação por equação          (DOC-04 §6.2) — o vetor que nenhum
                                          trabalho publicado trata

O terceiro é o que mais importa. Um item de benchmark cuja resposta é
``T = 2π√(L/g)`` está contaminado por qualquer documento que contenha essa
equação — mas nenhum casamento de string a encontra, porque o autor escreveu
``T = 2\\pi\\sqrt{\\frac{L}{g}}`` e o benchmark escreveu ``T=2\\pi(L/g)^{1/2}``.
A forma canônica encontra.

═══ A distinção que evita destruir o corpus ═══

    Canonicalização serve para COMPARAR, nunca para TREINAR.

O modelo treina no LaTeX **original** do autor, porque a diversidade
notacional é sinal, não ruído — um físico real encontra todas essas variantes.
A forma canônica existe apenas para responder "estas duas equações são a
mesma?". Confundir as duas produziria um modelo que só entende a nossa
normalização, inútil diante da literatura real.

Ambas são gravadas: ``equations[].latex`` (original, vai para o treino) e
``equations[].canonical_latex`` (derivado, vai para os índices).

═══ Princípio: na dúvida, não canonicalize ═══

Um falso negativo de dedup custa **um documento duplicado**. Um falso positivo
**destrói distinção física real em todo o corpus**. A assimetria de custo é
enorme, e por isso a §2 lista transformações deliberadamente REJEITADAS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from blake3 import blake3

# ══════════════════════════════════════════════════════════════════════════
# §1 — Transformações APLICADAS (seguras)
# ══════════════════════════════════════════════════════════════════════════

# Delimitadores de modo matemático → forma única. `$x$`, `\(x\)` e
# `\begin{math}x\end{math}` são o mesmo em conteúdo.
_MODO_MATEMATICO = [
    (re.compile(r"\\begin\{(?:math|displaymath)\}(.*?)\\end\{(?:math|displaymath)\}", re.S), r"\1"),
    (re.compile(r"\\\[(.*?)\\\]", re.S), r"\1"),
    (re.compile(r"\\\((.*?)\\\)", re.S), r"\1"),
    (re.compile(r"\$\$(.*?)\$\$", re.S), r"\1"),
    (re.compile(r"\$(.*?)\$", re.S), r"\1"),
]

# Ambientes de equação → conteúdo. `equation`, `equation*`, `align`, `gather`…
# diferem em numeração e alinhamento, não em Física.
_AMBIENTES = re.compile(
    r"\\begin\{(equation|eqnarray|align|alignat|gather|multline|displaymath|split)\*?\}"
    r"(.*?)"
    r"\\end\{\1\*?\}",
    re.S,
)

# Espaçamento tipográfico. `\,` `\;` `\!` `\quad` afetam a aparência, não o
# conteúdo — e variam livremente entre autores para a mesma equação.
_ESPACAMENTO = re.compile(r"\\[,;:!]|\\q?quad(?![A-Za-z])|\\hspace\s*\{[^}]*\}|\\ (?=[^\s])")

# Rótulos, numeração e anotações de apresentação.
_ANOTACOES = [
    re.compile(r"\\label\s*\{[^}]*\}"),
    re.compile(r"\\tag\s*\*?\{[^}]*\}"),
    re.compile(r"\\nonumber(?![A-Za-z])"),
    re.compile(r"\\notag(?![A-Za-z])"),
    re.compile(r"\\(?:left|right)\."),  # delimitador nulo
]

# Construtos equivalentes. `\dfrac` e `\tfrac` só mudam o tamanho da fonte.
_SINONIMOS = [
    (re.compile(r"\\[dt]frac(?![A-Za-z])"), r"\\frac"),
    (re.compile(r"\\[dt]binom(?![A-Za-z])"), r"\\binom"),
    # O lookahead precisa incluir os delimitadores de FECHAMENTO. Sem `)`,
    # `]` e `}`, o `\left(` era removido e o `\right)` sobrevivia — e as duas
    # formas nunca colapsavam.
    (re.compile(r"\\(?:big|Big|bigg|Bigg)[lrm]?(?=[\\(\)\[\]\{\}|.])"), ""),
    (re.compile(r"\\(?:left|right)(?=[\\(\)\[\]\{\}|.])"), ""),
    (re.compile(r"\\to(?![A-Za-z])"), r"\\rightarrow"),
    (re.compile(r"\\ne(?![A-Za-z])"), r"\\neq"),
    (re.compile(r"\\le(?![A-Za-z])"), r"\\leq"),
    (re.compile(r"\\ge(?![A-Za-z])"), r"\\geq"),
    (re.compile(r"\\ast(?![A-Za-z])"), r"\\*"),
]

# Chaves supérfluas em expoente/subscrito de um único átomo: `x^{2}` → `x^2`.
# Puramente sintático — `{2}` e `2` produzem o mesmo resultado tipográfico.
_CHAVE_UNITARIA = re.compile(r"([_^])\{([A-Za-z0-9]|\\[A-Za-z]+)\}")

# ══════════════════════════════════════════════════════════════════════════
# §2 — Transformações REJEITADAS, e por quê
# ══════════════════════════════════════════════════════════════════════════
#
# Cada linha abaixo é uma normalização que parece óbvia e destruiria Física.
# Estão aqui como documentação executável: o teste golden verifica que NÃO
# acontecem.
#
#   \epsilon ↔ \varepsilon      Em Física são grandezas DIFERENTES. É comum
#   \phi     ↔ \varphi          `\epsilon` ser permissividade e `\varepsilon`
#                               deformação, no mesmo paper.
#
#   ordenar operandos           Operadores não comutam: ÂB̂ ≠ B̂Â, e toda a
#                               Mecânica Quântica está nessa distinção.
#
#   simplificar via CAS         O SymPy não sabe o que é operador, matriz ou
#                               escalar sem anotação de tipo. Simplificação
#                               cega inventa igualdades falsas.
#
#   posição de índices          T^{μ}_{ν} e T_{ν}^{μ} diferem em Relatividade
#                               Geral. Ver `core/latex/subscritos.py`, que
#                               documenta o colapso ρ_xy = ρ_yx.
#
#   \mathbf{B} ↔ B              A distinção vetor/escalar é semântica.
#
#   \times ↔ \cdot ↔ justapor   Para escalares são o mesmo; para VETORES,
#                               `\times` é produto vetorial e `\cdot` é
#                               escalar — objetos de tipos diferentes. Só
#                               removemos `\cdot` quando não há marcador
#                               vetorial (ver `_pode_remover_cdot`).

_MARCADOR_VETORIAL = re.compile(
    r"\\(?:mathbf|vec|boldsymbol|bm|mathbb)\s*\{|\\hat\s*\{|\\times(?![A-Za-z])"
)


def _pode_remover_cdot(s: str) -> bool:
    """`a \\cdot b` = `ab` para escalares; para vetores é produto escalar.

    Remover `\\cdot` de `\\mathbf{A} \\cdot \\mathbf{B}` transformaria um
    escalar num produto de vetores — objetos de tipos diferentes. Só é seguro
    quando nenhum marcador vetorial aparece na expressão.
    """
    return not _MARCADOR_VETORIAL.search(s)


# ══════════════════════════════════════════════════════════════════════════
# §3 — Pipeline
# ══════════════════════════════════════════════════════════════════════════

_ESPACOS = re.compile(r"\s+")
_FIM_DE_MACRO = re.compile(r"\\[A-Za-z]+$")


def _remover_espacos(s: str) -> str:
    """Remove espaço, preservando o que delimita nome de macro.

    Em modo matemático o LaTeX ignora espaço na renderização: `m a` e `ma`
    produzem o mesmo resultado. Mas o espaço é o que termina o nome de uma
    macro — `\\pi x` sem ele vira `\\pix`, macro distinta e inexistente.

    Regra: o espaço só sobrevive entre `\\nome` e uma letra.
    """
    partes: list[str] = []
    i = 0
    for m in _ESPACOS.finditer(s):
        partes.append(s[i:m.start()])
        precisa = bool(_FIM_DE_MACRO.search(s[: m.start()])) and bool(
            s[m.end():m.end() + 1].isalpha()
        )
        if precisa:
            partes.append(" ")
        i = m.end()
    partes.append(s[i:])
    return "".join(partes)
_ALINHAMENTO = re.compile(r"(?<!\\)&")
_QUEBRA = re.compile(r"\\\\(?:\s*\[[^\]]*\])?")


@dataclass(frozen=True)
class FormaCanonica:
    canonica: str
    hash: str
    original: str

    def __eq__(self, outra: object) -> bool:
        return isinstance(outra, FormaCanonica) and self.hash == outra.hash

    def __hash__(self) -> int:
        return hash(self.hash)


def canonicalizar(latex: str) -> str:
    """LaTeX → forma canônica para comparação.

    Determinística e idempotente: ``canonicalizar(canonicalizar(x))`` é sempre
    ``canonicalizar(x)``. Sem isso, um índice construído em duas passadas
    divergiria de si mesmo.
    """
    if not latex or not latex.strip():
        return ""

    s = latex.strip()

    # 1. Ambientes → conteúdo (repetido: podem estar aninhados, ex. split em align)
    for _ in range(4):
        novo = _AMBIENTES.sub(lambda m: m.group(2), s)
        if novo == s:
            break
        s = novo

    # 2. Delimitadores de modo matemático
    for padrao, subst in _MODO_MATEMATICO:
        s = padrao.sub(subst, s)

    # 3. Anotações de apresentação
    for padrao in _ANOTACOES:
        s = padrao.sub("", s)

    # 4. Alinhamento e quebras. `&` é só diagramação e sai. Mas `\\` separa
    #    EQUAÇÕES DISTINTAS num `align`, e precisa sobreviver: sem ele,
    #    `E &= mc^2 \\ p &= mv` colapsaria em `E=mc^2p=mv`, colando o `2` no
    #    `p` e inventando um identificador que não existe.
    s = _QUEBRA.sub(" \\\\ ", s)
    s = _ALINHAMENTO.sub(" ", s)

    # 5. Sinônimos
    for padrao, subst in _SINONIMOS:
        s = padrao.sub(subst, s)

    # 6. Espaçamento tipográfico
    s = _ESPACAMENTO.sub(" ", s)

    # 7. `\cdot` — condicional, ver §2
    if _pode_remover_cdot(s):
        s = re.sub(r"\\cdot(?![A-Za-z])", " ", s)

    # 8. Chaves unitárias em índice
    for _ in range(3):  # `x^{{2}}` precisa de mais de uma passada
        novo = _CHAVE_UNITARIA.sub(r"\1\2", s)
        if novo == s:
            break
        s = novo

    # 9. Espaço em branco. Em modo matemático o espaço é ignorado na
    #    renderização, então `m a` e `ma` são a mesma coisa — EXCETO quando
    #    ele delimita o fim de um nome de macro. `\\pi x` sem o espaço vira
    #    `\\pix`, que é outra macro (inexistente). Ver `_remover_espacos`.
    s = _ESPACOS.sub(" ", s).strip()
    s = _remover_espacos(s)

    return s


def hash_canonico(latex: str) -> str:
    """Chave de índice. 16 bytes bastam: colisão em 10⁸ equações é ~10⁻¹⁴."""
    return blake3(canonicalizar(latex).encode("utf-8")).hexdigest(length=16)


def forma(latex: str) -> FormaCanonica:
    c = canonicalizar(latex)
    return FormaCanonica(canonica=c, hash=blake3(c.encode("utf-8")).hexdigest(length=16),
                         original=latex)


def equivalentes(a: str, b: str) -> bool:
    """Mesma equação sob variação notacional.

    **Não é equivalência matemática** — é identidade sintática após
    normalização. ``x+1`` e ``1+x`` são matematicamente iguais e canonicamente
    diferentes, de propósito: decidir a primeira exige um CAS, e o CAS é o
    `verify/symbolic`, não este módulo.
    """
    return canonicalizar(a) == canonicalizar(b)
