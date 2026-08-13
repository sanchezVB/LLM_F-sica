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
from dataclasses import dataclass, field

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


# ─── montagem do documento a partir do principal ────────────────────────────
#
# `\input{secoes/intro}`, `\input secoes/intro` (TeX aceita sem chaves) e o
# `\subfile` do pacote subfiles.
_INCLUSAO = re.compile(
    r"\\(?:input|include|subfile)\s*\{([^}]*)\}"
    r"|\\input\s+([^\s{}\\,]+)"
)
# LaTeX ignora tudo depois disto. Autor que deixa rascunho no fim do arquivo é
# comum, e contar essas equações infla a fonte.
_FIM = re.compile(r"\\end\s*\{document\}")
_CLASSE = re.compile(r"\\documentclass")
_CORPO = re.compile(r"\\begin\s*\{document\}")

# Nomes que o autor usa para o principal quando há mais de um candidato.
_NOMES_PROVAVEIS = ("main", "ms", "paper", "article", "manuscript", "root")
MAX_INCLUSAO = 12  # profundidade; ciclo já é barrado pelo conjunto de visitados


@dataclass
class Documento:
    """O documento montado, com o rastro de como foi montado.

    O rastro não é decoração: ele diz se a contagem de equações é confiável. Uma
    submissão em que o principal foi identificado e os `\\input` seguidos dá uma
    contagem fiel; uma em que caímos na concatenação dá uma contagem que pode
    estar INFLADA por arquivos que o documento nem inclui. Misturar as duas numa
    média e chamar o resultado de "degradação" foi o erro que isto corrige.
    """

    texto: str
    principal: str = ""
    incluidos: list[str] = field(default_factory=list)
    # `.tex` presentes no pacote e NÃO alcançados a partir do principal: são
    # exatamente os rascunhos e versões alternativas que inflavam a fonte.
    ignorados: list[str] = field(default_factory=list)
    faltantes: list[str] = field(default_factory=list)
    modo: str = "seguido"       # seguido | seguido-ambiguo | concatenado

    @property
    def confiavel(self) -> bool:
        return self.modo != "concatenado"


def _chave(nome: str) -> str:
    return nome.replace("\\", "/").lstrip("./").lower()


def _achar(alvo: str, arquivos: dict[str, str]) -> str | None:
    """Resolve o argumento de `\\input` num nome de arquivo do pacote.

    LaTeX acrescenta `.tex` quando não há extensão, e tarballs do arXiv às vezes
    divergem no caixa — daí a busca ser por chave normalizada.
    """
    indice = {_chave(k): k for k in arquivos}
    for cand in (alvo, alvo + ".tex"):
        k = indice.get(_chave(cand))
        if k is not None:
            return k
    # `secoes/intro` referenciado como `intro` (o autor mudou de diretório).
    base = _chave(alvo).rsplit("/", 1)[-1]
    for cand in (base, base + ".tex"):
        for k_norm, k in indice.items():
            if k_norm.rsplit("/", 1)[-1] == cand:
                return k
    return None


def _principal(arquivos: dict[str, str]) -> tuple[str | None, bool]:
    """(nome do principal, houve ambiguidade)."""
    tex = {k: v for k, v in arquivos.items() if k.lower().endswith(".tex")}
    cands = [k for k, v in tex.items() if _CLASSE.search(v)]
    if not cands:
        return None, False
    if len(cands) == 1:
        return cands[0], False
    # Vários `\documentclass`: versão alternativa, resposta a referee, etc.
    com_corpo = [k for k in cands if _CORPO.search(tex[k])] or cands
    if len(com_corpo) == 1:
        return com_corpo[0], True
    for nome in _NOMES_PROVAVEIS:
        for k in sorted(com_corpo):
            if _chave(k).rsplit("/", 1)[-1].startswith(nome):
                return k, True
    # Último critério: o maior. É palpite, e o `modo` registra isso.
    return max(com_corpo, key=lambda k: len(tex[k])), True


def montar_documento(arquivos: dict[str, str]) -> Documento:
    """Monta o documento como o LaTeX o veria, e diz o quanto confiar.

    ## Por que não basta concatenar

    A versão anterior juntava TODOS os `.tex` do pacote. Isso conta equações de
    arquivos que o documento não inclui — rascunho, versão anterior, seção
    cortada — e na auditoria do S3b elas apareciam como "equação que o RedPajama
    perdeu". O paper 1607.04847 declarava 1.427 equações na fonte contra 322 no
    RedPajama; boa parte da diferença era isto.

    O efeito é assimétrico e por isso perigoso: inflar a fonte **aumenta** a
    degradação medida, ou seja, empurra na direção de gastar US$ 100–180 que
    talvez não precisassem ser gastos.

    ## O que faz

    1. acha o `.tex` com `\\documentclass` — o principal;
    2. expande `\\input`/`\\include`/`\\subfile` no lugar, recursivamente;
    3. corta em `\\end{document}`, que é onde o LaTeX para de ler.

    ## Fora de escopo, declarado

    `\\input` dentro de `\\if…\\else` é seguido de qualquer forma (não avaliamos
    condicionais), e `\\includeonly` é ignorado. Ambos erram para o lado de
    incluir mais, o que ainda infla — mas são raros perto do caso de rascunho
    solto no pacote, e um `\\input` não resolvido fica registrado em `faltantes`
    em vez de ser silenciado.
    """
    limpos = {k: remover_comentarios(v) for k, v in arquivos.items()}
    principal, ambiguo = _principal(limpos)

    if principal is None:
        # Sem `\documentclass` não há como saber o que o documento inclui. Cai no
        # comportamento antigo e MARCA que caiu — quem consome decide.
        #
        # O corte em `\end{document}` é POR ARQUIVO. Cortar a concatenação inteira
        # no primeiro `\end{document}` descartaria todos os arquivos seguintes, e
        # quais são "os seguintes" dependeria da ordem alfabética — um resultado
        # que muda com o nome dos arquivos não é resultado.
        texto = "\n".join(_FIM.split(v)[0] for k, v in sorted(limpos.items())
                          if k.lower().endswith(".tex"))
        return Documento(texto=texto, modo="concatenado",
                         ignorados=sorted(k for k in limpos if k.lower().endswith(".tex")))

    d = Documento(texto="", principal=principal,
                  modo="seguido-ambiguo" if ambiguo else "seguido")
    visitados: set[str] = set()

    def expandir(nome: str, profundidade: int) -> str:
        if nome in visitados or profundidade > MAX_INCLUSAO:
            return ""
        visitados.add(nome)
        fonte = limpos[nome]
        saida, fim = [], 0
        for m in _INCLUSAO.finditer(fonte):
            alvo = (m.group(1) or m.group(2) or "").strip()
            saida.append(fonte[fim:m.start()])
            fim = m.end()
            achado = _achar(alvo, limpos)
            if achado is None:
                # `.sty`/`.cls` ausentes são normais; `.tex` ausente não é.
                if not alvo.lower().endswith((".sty", ".cls", ".bbl", ".bib")):
                    d.faltantes.append(alvo)
            elif achado not in visitados:
                d.incluidos.append(achado)
                saida.append(expandir(achado, profundidade + 1))
        saida.append(fonte[fim:])
        return "\n".join(saida)

    # Corta em `\end{document}` DEPOIS de expandir: uma seção incluída antes do
    # fim tem de entrar, e o que vem depois tem de sair.
    d.texto = _FIM.split(expandir(principal, 0))[0]
    d.ignorados = sorted(k for k in limpos
                         if k.lower().endswith(".tex") and k not in visitados)
    return d


def juntar_fontes(arquivos: dict[str, str]) -> str:
    """Texto do documento montado. Mantido para quem só quer o texto.

    Prefira `montar_documento` quando o rastro importar — e na auditoria ele
    importa, porque separa contagem fiel de contagem possivelmente inflada.
    """
    return montar_documento(arquivos).texto
