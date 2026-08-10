"""Extração de equações de um documento LaTeX (DOC-03 §3, insumo do S3b).

O canonicalizador responde "estas duas equações são a mesma?". Isto responde
"quais equações existem neste documento?" — e a auditoria do S3b precisa das
duas: extrai de cada lado, canonicaliza, compara conjuntos.

## Três cuidados que decidem se a medição mede o que promete

**1. Comentários primeiro.** O `.tex` do autor tem linhas comentadas com `%`, e
equações abandonadas ali são comuns. Um pipeline de terceiros que remove
comentários está CERTO — contá-las como "equação perdida" inventaria degradação
onde houve limpeza. O `%` escapado (`\\%`, que é o símbolo de porcentagem) não é
comentário e precisa sobreviver.

**2. Display antes de inline.** `$$…$$` contém `$…$` como subcadeia. Casar
inline primeiro parte a equação em duas metades sem sentido. A ordem aqui não é
estética.

**3. Fórmula trivial não conta.** `$n$`, `$x$`, `$i$` — um símbolo solto — são
milhares por paper e não carregam Física. Incluí-las inflaria o denominador e
mascararia a perda de equações reais, que é justamente o que se quer medir.
`MIN_SIMBOLOS` é o corte.
"""

from __future__ import annotations

import re

# Comentário LaTeX: `%` até o fim da linha, desde que não escapado. O olhar para
# trás exclui `\%` mas aceita `\\%` (barra escapada seguida de comentário).
_COMENTARIO = re.compile(r"(?<!\\)((?:\\\\)*)%[^\n]*")

# Ambientes de equação. Capturam o conteúdo, não o invólucro.
_AMBIENTE = re.compile(
    r"\\begin\{(equation|eqnarray|align|alignat|gather|multline|displaymath|split|"
    r"flalign|dmath)\*?\}(.*?)\\end\{\1\*?\}",
    re.S,
)

# Ordem OBRIGATÓRIA: display antes de inline. Ver §2 da docstring.
_DISPLAY = [
    re.compile(r"\\\[(.*?)\\\]", re.S),
    re.compile(r"\$\$(.*?)\$\$", re.S),
]
_INLINE = [
    re.compile(r"\\\((.*?)\\\)", re.S),
    # `$…$` sem `$` dentro, e sem casar `$$`.
    re.compile(r"(?<!\$)\$(?!\$)((?:[^$\\]|\\.)+?)\$(?!\$)", re.S),
]

# Abaixo disto é símbolo solto, não equação. Ver §3 da docstring.
MIN_SIMBOLOS = 4


def remover_comentarios(tex: str) -> str:
    """Tira comentários preservando `\\%`."""
    return _COMENTARIO.sub(r"\1", tex)


def _relevante(eq: str) -> bool:
    eq = eq.strip()
    if len(eq) < MIN_SIMBOLOS:
        return False
    # Precisa ter estrutura: um operador, uma relação ou uma macro. `abc` não é
    # equação; `a=b` e `\alpha` são.
    return bool(re.search(r"[=<>+\-/^_]|\\[A-Za-z]", eq))


def extrair_equacoes(tex: str, *, remover_comentario: bool = True) -> list[str]:
    """Equações do documento, na ordem em que aparecem, sem os delimitadores.

    Consome o texto conforme casa, para que nada seja extraído duas vezes — uma
    equação dentro de `align` não pode reaparecer como inline.
    """
    if remover_comentario:
        tex = remover_comentarios(tex)

    achadas: list[str] = []

    def colher(padrao, grupo: int, fonte: str) -> str:
        """Colhe as ocorrências e devolve o texto com elas removidas."""
        pedacos, fim = [], 0
        for m in padrao.finditer(fonte):
            eq = m.group(grupo)
            if _relevante(eq):
                achadas.append(eq.strip())
            pedacos.append(fonte[fim:m.start()])
            fim = m.end()
        pedacos.append(fonte[fim:])
        return " ".join(pedacos)

    tex = colher(_AMBIENTE, 2, tex)
    for p in _DISPLAY:
        tex = colher(p, 1, tex)
    for p in _INLINE:
        tex = colher(p, 1, tex)
    return achadas


def juntar_fontes(arquivos: dict[str, str]) -> str:
    """Concatena os `.tex` de um pacote de submissão.

    Uma submissão do arXiv costuma vir em vários arquivos (`main.tex`,
    `secoes/*.tex`). Extrair só do principal perderia as equações dos incluídos,
    e isso apareceria como degradação do RedPajama — que não teria acontecido.
    """
    return "\n".join(v for k, v in sorted(arquivos.items()) if k.endswith(".tex"))
