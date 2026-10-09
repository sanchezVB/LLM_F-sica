"""O cortador de trechos do texto completo (DOC-13 §3).

O índice de hoje tem um vetor por artigo, de `título. resumo`. Para indexar o corpo do
artigo é preciso cortá-lo — e o tamanho quem dita é o encoder, que lê `MAX_TOKENS` (192)
e não os 512 do alvo do DOC-13.

## As regras, na ordem em que mandam

1. **Nunca partir uma equação.** Um ambiente matemático (`equation`, `align`, `\\[…\\]`,
   `$$…$$`…) é um bloco indivisível. Se ele sozinho não cabe, o trecho sai MAIOR que o
   limite (`excede=True`) e o encoder trunca — a regra 1 ganha da regra de tamanho.
2. **A equação vai com o contexto.** Ela é empacotada junto do parágrafo que a antecede
   quando os dois cabem; quando não cabem, o trecho novo começa com a última frase do
   anterior.
3. **Seção é fronteira.** Nenhum trecho atravessa `\\section` / `\\subsection`.
4. Só então o tamanho: parágrafos inteiros enquanto couberem; parágrafo grande demais é
   cortado em frases; frase grande demais, em palavras — sem cortar dentro de `$…$`.

## O que sai do texto antes de cortar (`limpar`)

Comentários, bibliografia, `\\label`, `\\cite`, `\\ref`; tabelas e figuras, de que fica a
legenda; a casca de formatação (`\\textit{x}` → `x`); agradecimentos. Não carregam
assunto, e ocupam tokens de um orçamento de 192.

⚠️ O texto do RedPajama vem SEM preâmbulo: o `\\newcommand{\\be}{\\begin{equation}}` do
autor não está lá, só o `\\be … \\ee`. `expandir_atalhos` devolve o ambiente a esses
atalhos — só com prova no próprio documento, porque `\\ba` também é "a em negrito".

`contar` (textos → nº de tokens de cada um) vem de fora: o cortador não conhece
tokenizer, e os testes usam contagem de palavras.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_AMBIENTES = ("equation|align|alignat|flalign|eqnarray|gather|multline|displaymath|math"
              "|subequations|split")
# Acima disto não é uma equação: é um `\begin` cujo `\end` o autor escreveu com macro.
_MAX_EQ = "12000"
AMBIENTE = re.compile(
    r"\\begin\{(" + _AMBIENTES + r")\*?\}.{0," + _MAX_EQ + r"}?\\end\{\1\*?\}"
    r"|(?<!\\)\\\[.{0," + _MAX_EQ + r"}?(?<!\\)\\\]", re.DOTALL)
_ATALHOS = (("be", "ee"), ("beq", "eeq"), ("bea", "eea"), ("ba", "ea"), ("ben", "een"),
            ("beqa", "eeqa"), ("beqn", "eeqn"), ("bdm", "edm"), ("bal", "eal"))
# Tabelas e figuras: fica a legenda. As células de uma tabela são números soltos para um
# encoder de 192 tokens, e uma tabela média não cabe num trecho.
_FLUTUANTE = re.compile(r"\\begin\{((?:table|figure|sidewaystable|wrapfigure|longtable)\*?)\}"
                        r".*?\\end\{\1\}", re.DOTALL)
_TABULAR = re.compile(r"\\begin\{(tabular[x*]?|ruledtabular|tikzpicture)\}.*?\\end\{\1\}",
                      re.DOTALL)
_LEGENDA = re.compile(r"\\caption\s*(?:\[[^\]]*\]\s*)?\{")
_SECAO = re.compile(r"\\(section|subsection|subsubsection)\*?\s*\{((?:[^{}]|\{[^{}]*\})*)\}")
_FIM_DO_CORPO = re.compile(r"\\begin\{thebibliography\}|\\bibliography\s*\{|\\begin\{references\}"
                           r"|\\printbibliography")
_COMENTARIO = re.compile(r"(?<!\\)%.*")
# Definições de macro que sobraram no corpo. O RedPajama expandiu as macros DENTRO delas
# (`\newcommand{ \begin{equation}}{\begin{equation}}`): a linha inteira é entulho, e um
# `\begin` sem par para a auditoria.
_DEFINICAO = re.compile(r"(?m)^.*\\(?:(?:re)?newcommand|providecommand|DeclareMathOperator"
                        r"|def\s*\\|let\s*\\|catcode).*$")
_RUIDO = re.compile(
    r"\\label\s*\{[^{}]*\}"
    r"|~?\\(?:online)?cite[a-zA-Z]*\*?\s*(?:\[[^\]]*\]\s*)*\{[^{}]*\}"
    r"|~?\\(?:eq|auto|c|C)?ref\s*\{[^{}]*\}"
    r"|\\(?:begin|end)\{(?:itemize|enumerate|description|center|widetext|acknowledgments)\}"
    r"|\\(?:item|noindent|centering|maketitle|newpage|clearpage|medskip|bigskip|smallskip)"
    r"(?![a-zA-Z])"
    r"|\\[vh]space\*?\s*\{[^{}]*\}")
_CASCA = re.compile(r"\\(?:emph|textit|textbf|textrm|textsc|texttt|textsl|underline)\s*"
                    r"\{([^{}]*)\}"
                    r"|\{\\(?:em|it|bf|rm|sl|sc|tt)\b\s*([^{}]*)\}")
_INTRODUCAO = re.compile(r"(?i)^\W*(?:\d+\W*)?(introduction|intro\b|overview|motivation)")
_CONCLUSAO = re.compile(r"(?i)(conclusion|concluding|summary|discussion|outlook|final remarks)")
_FORA = re.compile(r"(?i)(acknowledg|funding|author contribution|data availability"
                   r"|conflicts? of interest|competing interest)")
# O dano de origem (ver `reparar`): `\begin{align` sem a chave, e `\[` reduzido a `\`.
_SEM_CHAVE = re.compile(r"(\\(?:begin|end)\{[a-zA-Z]+\*?)(?=\s)")
_BARRA_SO = re.compile(r"(?m)^[ \t]*\\[ \t]*$")
_COLCHETE = re.compile(r"(?m)(?<!\\)\\\[|(?<!\\)\\\]|^[ \t]*\\[ \t]*$")
_FRASE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\$(\[])")
_DOLAR = re.compile(r"(?<!\\)\$")


@dataclass
class Trecho:
    texto: str
    secao: str            # título da seção de primeiro nível
    tipo: str             # introducao · conclusao · corpo
    n_tokens: int         # soma das unidades: aproximado, a junção pode custar 1 ou 2
    tem_equacao: bool = False
    excede: bool = False  # uma unidade sozinha maior que o limite: o encoder trunca


def tipo_de_secao(titulo: str) -> str:
    if _FORA.search(titulo):
        return "fora"
    if _INTRODUCAO.search(titulo):
        return "introducao"
    return "conclusao" if _CONCLUSAO.search(titulo) else "corpo"


def _grupo(texto: str, inicio: int) -> str:
    """O conteúdo do `{…}` que começa logo antes de `inicio`, com as chaves casadas."""
    nivel, i = 1, inicio
    while i < len(texto) and nivel:
        c = texto[i]
        if c == "\\":
            i += 2
            continue
        nivel += (c == "{") - (c == "}")
        i += 1
    return texto[inicio:i - 1] if not nivel else texto[inicio:]


def _so_a_legenda(m: re.Match) -> str:
    legendas = [_grupo(m.group(0), c.end()) for c in _LEGENDA.finditer(m.group(0))]
    return "\n\n" + "\n\n".join(legendas) + "\n\n"


def _alternam(texto: str, abre: str, fecha: str) -> bool:
    """`abre` e `fecha` aparecem intercalados, começando por `abre`, e em mesmo número."""
    marcas = sorted([(m.start(), 0) for m in re.finditer(abre, texto)]
                    + [(m.start(), 1) for m in re.finditer(fecha, texto)])
    return bool(marcas) and all(lado == i % 2 for i, (_, lado) in enumerate(marcas)) \
        and len(marcas) % 2 == 0


def expandir_atalhos(texto: str) -> str:
    """`\\be … \\ee` → `\\begin{equation} … \\end{equation}`, só com prova no documento.

    A prova: somados aos `\\begin`/`\\end` de UM ambiente, os atalhos abrem e fecham
    intercalados do começo ao fim. Cobre o par puro (`\\be … \\ee`), o lado só
    (`\\ba … \\end{eqnarray}`) e a mistura dos dois; e recusa o `\\ba` que é "a em negrito",
    porque esse aparece dentro de equação e quebra a alternância."""
    def marca(nome: str) -> str:
        return r"\\" + nome + r"(?![a-zA-Z@])"

    for abre, fecha in _ATALHOS:
        if not (re.search(marca(abre), texto) or re.search(marca(fecha), texto)):
            continue
        for amb in ("equation", "eqnarray", "align", "gather", "multline"):
            nome = r"\{" + amb + r"\*?\}"
            if _alternam(texto, marca(abre) + r"|\\begin" + nome, marca(fecha) + r"|\\end" + nome):
                texto = re.sub(marca(abre), lambda _, a=amb: "\\begin{" + a + "}", texto)
                texto = re.sub(marca(fecha), lambda _, a=amb: "\\end{" + a + "}", texto)
                break
    return texto


def reparar(texto: str) -> str:
    """Desfaz o que dá para desfazer de um dano que vem do RedPajama: o removedor de
    comentários dele comeu o caractere ANTES de cada `%`. `\\begin{align}%` chegou como
    `\\begin{align`, e `\\[%` ou `\\]%` como uma `\\` sozinha na linha (medido em
    2026-10-09: 11 dos 36.803 trechos de 400 artigos tinham equação sem par só por isso).

    A barra sozinha vira `\\[` se a equação está fechada e `\\]` se está aberta — e só se,
    assim, o documento inteiro abrir e fechar intercalado. Sem essa prova, nada muda."""
    texto = re.sub(r"\\(begin|end)[ \t]+\{", lambda m: "\\" + m.group(1) + "{", texto)
    texto = _SEM_CHAVE.sub(lambda m: m.group(1) + "}", texto)
    if not _BARRA_SO.search(texto):
        return texto
    partes: list[str] = []
    aberto, fim = False, 0
    for m in _COLCHETE.finditer(texto):
        marca = m.group(0).strip()
        if marca == "\\":
            marca = r"\]" if aberto else r"\["
        elif (marca == r"\[") == aberto:
            return texto
        aberto = marca == r"\["
        partes += [texto[fim:m.start()], marca]
        fim = m.end()
    return texto if aberto else "".join(partes) + texto[fim:]


def limpar(texto: str) -> str:
    """O corpo sem bibliografia, comentários, rótulos, citações e referências cruzadas, com
    as tabelas e figuras reduzidas à legenda e os atalhos de equação expandidos."""
    texto = reparar(_DEFINICAO.sub("", _COMENTARIO.sub("", texto)))
    m = _FIM_DO_CORPO.search(texto)
    if m:
        texto = texto[:m.start()]
    texto = _TABULAR.sub(" ", _FLUTUANTE.sub(_so_a_legenda, texto))
    # ⚠️ Um espaço, e não nada: `$\ref{a}$` viraria `$$`, que abre equação destacada.
    texto = _RUIDO.sub(" ", texto)
    texto = _CASCA.sub(lambda m: m.group(1) if m.group(1) is not None else m.group(2), texto)
    return expandir_atalhos(texto)


def _espacos(s: str) -> str:
    return " ".join(s.replace("~", " ").split())


def exibidas(texto: str) -> tuple[list[tuple[int, int]], bool]:
    """Os vãos `$$…$$` de um parágrafo, e se algum ficou aberto no fim.

    Lido como o TeX lê: dentro de `$…$`, um `$$` é o fecho de uma fórmula e a abertura da
    seguinte (`$a$$b$`), não uma equação destacada."""
    vaos: list[tuple[int, int]] = []
    modo, inicio, pula = "", 0, -1
    for m in _DOLAR.finditer(texto):
        i = m.start()
        if i == pula:
            continue
        duplo = texto.startswith("$$", i)
        if not modo:
            modo, inicio = ("$$" if duplo else "$"), i
            pula = i + 1 if duplo else -1
        elif modo == "$":
            modo = ""
        elif duplo:
            vaos.append((inicio, i + 2))
            modo, pula = "", i + 1
    return vaos, modo == "$$"


def _dentro_de_matematica(pedaco: str) -> bool:
    """Um corte caiu no meio de um `$…$` se sobrar um `$` sem par."""
    return len(_DOLAR.findall(pedaco)) % 2 == 1


def _juntar_enquanto_aberto(pedacos: list[str]) -> list[str]:
    """Recola pedaços vizinhos enquanto o da esquerda terminar dentro de `$…$`."""
    if _dentro_de_matematica(" ".join(pedacos)):
        return pedacos        # um `$` sem par no todo: recolar engoliria o parágrafo
    saida: list[str] = []
    for p in pedacos:
        if saida and _dentro_de_matematica(saida[-1]):
            saida[-1] = f"{saida[-1]} {p}"
        else:
            saida.append(p)
    return saida


def frases(paragrafo: str) -> list[str]:
    return _juntar_enquanto_aberto([f for f in _FRASE.split(paragrafo) if f.strip()])


def _em_palavras(frase: str, n_tokens: int, limite: int) -> list[str]:
    """Uma frase maior que o limite, em pedaços de palavras — nunca dentro de `$…$`."""
    palavras = frase.split()
    n = -(-n_tokens // limite)
    passo = max(1, len(palavras) // n)
    return _juntar_enquanto_aberto(
        [" ".join(palavras[i:i + passo]) for i in range(0, len(palavras), passo)])


def _paragrafos(texto: str) -> list[tuple[str, bool]]:
    saida: list[tuple[str, bool]] = []
    for p in re.split(r"\n\s*\n", texto):
        fim = 0
        for a, b in exibidas(p)[0]:
            saida += [(p[fim:a], False), (p[a:b], True)]
            fim = b
        saida.append((p[fim:], False))
    return [(limpo, eq) for t, eq in saida if (limpo := _espacos(t))]


def blocos(corpo: str) -> list[tuple[str, bool]]:
    """`(texto, é_equação)` na ordem do documento: parágrafos e equações destacadas."""
    saida: list[tuple[str, bool]] = []
    fim = 0
    for m in AMBIENTE.finditer(corpo):
        saida += _paragrafos(corpo[fim:m.start()])
        saida.append((_espacos(m.group(0)), True))
        fim = m.end()
    return saida + _paragrafos(corpo[fim:])


def secoes(texto: str) -> list[tuple[str, str]]:
    """`(título de primeiro nível, corpo)` — cada subseção é um corpo à parte, com o
    título da seção-mãe: a fronteira vale, e o tipo (introdução, conclusão) é herdado."""
    saida: list[tuple[str, str]] = []
    titulo, fim = "", 0
    for m in _SECAO.finditer(texto):
        if texto[fim:m.start()].strip():
            saida.append((titulo, texto[fim:m.start()]))
        if m.group(1) == "section":
            titulo = _espacos(m.group(2))
        fim = m.end()
    if texto[fim:].strip():
        saida.append((titulo, texto[fim:]))
    return saida


Contar = Callable[[list[str]], list[int]]
_Unidade = tuple[str, bool, int]     # texto · é equação · tokens


def _unidades(corpo: str, contar: Contar, limite: int) -> list[_Unidade]:
    """Equações inteiras; parágrafos, ou as frases e palavras do que não couber."""
    saida: list[_Unidade] = []
    bs = blocos(corpo)
    for (t, eq), n in zip(bs, contar([t for t, _ in bs]), strict=True):
        if eq or n <= limite:
            saida.append((t, eq, n))
            continue
        fs = frases(t)
        for f, nf in zip(fs, contar(fs), strict=True):
            if nf <= limite:
                saida.append((f, False, nf))
            else:
                ps = _em_palavras(f, nf, limite)
                saida += [(p, False, k) for p, k in zip(ps, contar(ps), strict=True)]
    return saida


def _cauda(atual: list[_Unidade], contar: Contar, sobreposicao: int) -> _Unidade | None:
    """A última frase do trecho que fecha, para abrir o seguinte — se o trecho acabar em
    texto, se a frase for curta, e se ela não for o trecho inteiro."""
    texto, eq, _ = atual[-1]
    if eq or not sobreposicao:
        return None
    fs = frases(texto)
    if not fs or (len(atual) == 1 and len(fs) == 1):
        return None
    n = contar([fs[-1]])[0]
    return (fs[-1], False, n) if n <= sobreposicao else None


def cortar(texto: str, contar: Contar, limite: int = 190, sobreposicao: int = 40,
           minimo: int = 12) -> list[Trecho]:
    """Os trechos de um documento. `limite` é o orçamento de tokens do texto (o encoder
    ainda põe os dois especiais); `sobreposicao`, o tamanho máximo da frase repetida no
    começo de um trecho novo dentro da mesma subseção; abaixo de `minimo` o trecho é
    resto (um `\\appendix` solto, uma legenda de duas palavras) e não é devolvido."""
    trechos: list[Trecho] = []
    for titulo, corpo in secoes(limpar(texto)):
        tipo = tipo_de_secao(titulo)
        if tipo == "fora":
            continue
        grupos: list[list[_Unidade]] = []
        atual: list[_Unidade] = []
        for u in _unidades(corpo, contar, limite):
            if atual and sum(x[2] for x in atual) + u[2] > limite:
                cauda = _cauda(atual, contar, sobreposicao)
                grupos.append(atual)
                atual = [cauda] if cauda and cauda[2] + u[2] <= limite else []
            atual.append(u)
        grupos.append(atual)
        for g in grupos:
            n = sum(u[2] for u in g)
            if n >= minimo:
                trechos.append(Trecho(" ".join(u[0] for u in g), titulo, tipo, n,
                                      any(u[1] for u in g),
                                      excede=any(u[2] > limite for u in g)))
    return trechos


def cortes_em_equacao(trecho: str) -> int:
    """A auditoria M1: quantos ambientes matemáticos ficaram abertos ou fechados sem par
    neste trecho. Zero é a regra 1 cumprida."""
    nome = r"\{(?:" + _AMBIENTES + r")\*?\}"
    ambientes = len(re.findall(r"\\begin" + nome, trecho)) - len(re.findall(r"\\end" + nome, trecho))
    colchetes = (len(re.findall(r"(?<!\\)\\\[", trecho))
                 - len(re.findall(r"(?<!\\)\\\]", trecho)))
    return abs(ambientes) + abs(colchetes) + int(exibidas(trecho)[1])


def documento_integro(limpo: str) -> bool:
    """O texto (já passado por `limpar`) abre e fecha toda equação? Um `\\begin{equation}`
    sem `\\end`, ou um `$$` que atravessa parágrafo, já chega assim da origem — e o corte
    que a auditoria vê num trecho desses não é do cortador."""
    return cortes_em_equacao(AMBIENTE.sub(" ", limpo)) == 0 and not any(
        exibidas(p)[1] for p in re.split(r"\n\s*\n", limpo))
