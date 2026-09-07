#!/usr/bin/env python3
"""Renderiza um rascunho de artigo de `docs/papers/` em PDF.

    PYTHONPATH=src python scripts/artigo_pdf.py \
        --entrada docs/papers/rascunho-artigo-recuperacao-fisica.md \
        --saida build/rascunho-artigo-recuperacao-fisica.pdf

## Por que um renderizador e não o PDF escrito à mão

O `scripts/relatorio_pdf.py` monta um relatório LENDO os artefatos de medição —
nenhum número é digitado nele. Um artigo é outra coisa: o texto é a contribuição,
e ele precisa viver em Markdown versionado, revisável em diff, ao lado dos outros
rascunhos de `docs/papers/`.

Então a fonte é o `.md` e o PDF é ARTEFATO DE BUILD (`*.pdf` está no
`.gitignore`). Gerar o PDF e versioná-lo criaria duas cópias do mesmo texto, e
este repositório já pagou o preço disso uma vez: três cópias da tabela de estado,
no README, no índice de documentos e na carta de projeto, divergiram e as três
afirmavam coisas diferentes sobre o mesmo fato.

## O subconjunto de Markdown que ele entende

Títulos (`#`..`####`), parágrafos, `**negrito**`, `*itálico*`, `` `código` ``,
links (vira o texto), listas com `-` e `1.`, citação com `>`, blocos de código
com crase tripla, tabelas GFM e régua `---`.

Não entende: HTML embutido, listas aninhadas, imagens (são removidas), notas de
rodapé. Um bloco não reconhecido vira parágrafo — nunca desaparece em silêncio,
que é o modo de falha que importa num renderizador de documento.

## A guarda que existe por medida, e não por precaução

Linha de bloco de código não quebra: o `Preformatted` do reportlab escreve reto e
deixa o excesso sair da margem **sem erro e sem aviso**. As tabelas de largura
fixa deste projeto têm 60–75 caracteres, e uma delas passando a 90 sairia
cortada num PDF que alguém leria como completo.

`_fonte_que_cabe` calcula o corpo que faz a linha mais longa caber na margem e
`--estrito` levanta se nem no piso couber. É a mesma regra do resto do
repositório: alarme que corta em silêncio é pior que nenhum.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

log = logging.getLogger(__name__)

# Layout de artigo: margens de 25 mm, corpo em Times 12, entrelinha 1,3.
MARGEM = 25 * mm
LARGURA_UTIL = A4[0] - 2 * MARGEM
CORPO = 12
ENTRELINHA = 15.6

SERIFA = "Times-Roman"
SERIFA_N = "Times-Bold"
SERIFA_I = "Times-Italic"

TINTA = colors.black
SUAVE = colors.HexColor("#444444")
LINHA = colors.HexColor("#999999")
FUNDO = colors.HexColor("#f0f0f0")
REALCE = colors.HexColor("#666666")

# Courier tem largura fixa de 0,6 em por caractere. É isto que torna a conta de
# `_fonte_que_cabe` exata em vez de estimada.
RAZAO_COURIER = 0.6
CORPO_CODIGO_MAX = 10.0
CORPO_CODIGO_MIN = 6.0

# ── Cobertura de glifo, e por que existe uma guarda em vez de confiança ──────
#
# O documento fica nas fontes base-14 (Helvetica/Courier), que o reportlab tem
# embutidas: o PDF sai IDÊNTICO no Windows do autor e no runner do CI. Registrar
# uma DejaVu do sistema resolveria todo o Unicode e faria o resultado depender da
# máquina — a troca não vale num repositório cujo critério de aceitação declarado
# é reprodutibilidade byte a byte.
#
# O preço é que o base-14 não cobre tudo, e o reportlab **não avisa**: caractere
# ausente vira retângulo preto no PDF, sem exceção e sem log. Foi o que aconteceu
# com `10¹⁰` na primeira renderização — `¹` é Latin-1 e saiu, `⁰` é U+2070 e
# virou caixa.
#
# Então: o que dá para converter em marcação (expoente) é convertido, o que o
# reportlab substitui de Symbol/ZapfDingbats (grego, setas, relações) é
# permitido, e o resto **levanta**. Um glifo novo num rascunho futuro tem de ser
# uma decisão de quem escreve, não uma caixa preta que ninguém vê.
EXPOENTES = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
# Substituídos pelo reportlab a partir de Symbol e ZapfDingbats.
SIMBOLOS_OK = set("–—…‘’“”−≈≤≥≠∈∉⊂∞±×÷·§→←↔⇒⇐√∂∑∏∫✓✗")


def _conferir_glifos(md: str, estrito: bool) -> None:
    fora = sorted({
        c for c in md
        if ord(c) > 0x7F
        and c not in SIMBOLOS_OK
        and not (0xA0 <= ord(c) <= 0xFF)      # Latin-1: acentuação do português
        and not (0x370 <= ord(c) <= 0x3FF)    # grego: Φ, ρ, Σ, μ, Δ
        # ⚠️ `ord(c)`, não `c`: as chaves de um `str.maketrans` são ORDINAIS, e
        # `'⁰' in EXPOENTES` é sempre falso. A guarda reprovava exatamente os dois
        # caracteres que ela existe para isentar, porque `_inline` já os converte
        # em `<super>`.
        and ord(c) not in EXPOENTES
    })
    if not fora:
        return
    nomes = ", ".join(f"U+{ord(c):04X} {c!r}" for c in fora)
    aviso = f"caracteres sem glifo nas fontes base-14, sairiam como caixa preta: {nomes}"
    if estrito:
        raise ValueError(aviso)
    log.warning("%s", aviso)


# ─── estilos ────────────────────────────────────────────────────────────────

def _estilos() -> dict[str, ParagraphStyle]:
    s = getSampleStyleSheet()
    return {
        "titulo": ParagraphStyle("titulo", parent=s["Title"], fontName=SERIFA_N,
                                 fontSize=16, leading=20, textColor=TINTA,
                                 alignment=TA_CENTER, spaceAfter=12),
        "meta": ParagraphStyle("meta", parent=s["Normal"], fontName=SERIFA,
                               fontSize=11, leading=14, textColor=TINTA,
                               alignment=TA_CENTER, spaceAfter=3),
        "h1": ParagraphStyle("h1", parent=s["Heading1"], fontName=SERIFA_N,
                             fontSize=13, leading=16, textColor=TINTA,
                             spaceBefore=16, spaceAfter=6, keepWithNext=True),
        "h2": ParagraphStyle("h2", parent=s["Heading2"], fontName=SERIFA_N,
                             fontSize=12, leading=15, textColor=TINTA,
                             spaceBefore=12, spaceAfter=5, keepWithNext=True),
        "h3": ParagraphStyle("h3", parent=s["Heading3"], fontName=SERIFA_I,
                             fontSize=12, leading=15, textColor=TINTA,
                             spaceBefore=10, spaceAfter=4, keepWithNext=True),
        "p": ParagraphStyle("p", parent=s["Normal"], fontName=SERIFA,
                            fontSize=CORPO, leading=ENTRELINHA, textColor=TINTA,
                            alignment=TA_JUSTIFY, firstLineIndent=0, spaceAfter=8),
        "item": ParagraphStyle("item", parent=s["Normal"], fontName=SERIFA,
                               fontSize=CORPO, leading=ENTRELINHA, textColor=TINTA,
                               alignment=TA_JUSTIFY, leftIndent=10 * mm,
                               bulletIndent=4 * mm, spaceAfter=5),
        # Resumo e legenda: um corpo abaixo do texto, que é a convenção.
        "cita": ParagraphStyle("cita", parent=s["Normal"], fontName=SERIFA,
                               fontSize=11, leading=14, textColor=TINTA,
                               alignment=TA_JUSTIFY, spaceAfter=6),
        "legenda": ParagraphStyle("legenda", parent=s["Normal"], fontName=SERIFA,
                                  fontSize=10.5, leading=13, textColor=TINTA,
                                  alignment=TA_JUSTIFY, spaceBefore=4, spaceAfter=4,
                                  keepWithNext=True),
        "ref": ParagraphStyle("ref", parent=s["Normal"], fontName=SERIFA,
                              fontSize=11, leading=13.6, textColor=TINTA,
                              alignment=0, leftIndent=10 * mm, firstLineIndent=-10 * mm,
                              spaceAfter=4),
        "cel": ParagraphStyle("cel", parent=s["Normal"], fontName=SERIFA,
                              fontSize=9.5, leading=11.8, textColor=TINTA),
        "celh": ParagraphStyle("celh", parent=s["Normal"], fontName=SERIFA_N,
                               fontSize=9.5, leading=11.8, textColor=TINTA),
    }


# ─── marcação em linha ──────────────────────────────────────────────────────

def _inline(txt: str) -> str:
    """Markdown em linha para a marcação mínima que o `Paragraph` aceita.

    A ordem importa, e o trecho de código sai de cena antes da ênfase.
    `p^*` e `p *` — notação real da §5.4 — deixavam um `*` de cada lado DENTRO de
    duas crases diferentes, e o regex de itálico casava de um para o outro,
    atravessando o `</font>`. O reportlab levantava `saw </font> instead of
    expected </i>`, o que ao menos é barulhento; o mesmo casamento num par de
    negritos teria produzido um PDF errado em silêncio.
    """
    txt = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    txt = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", txt)
    txt = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", txt)
    txt = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+",
                 lambda m: f"<super>{m.group(0).translate(EXPOENTES)}</super>", txt)

    guardados: list[str] = []

    def _guardar(m: re.Match[str]) -> str:
        guardados.append(m.group(1))
        return f"\x00{len(guardados) - 1}\x00"

    txt = re.sub(r"`([^`]+)`", _guardar, txt)
    txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt)
    txt = re.sub(r"(?<![\*\w])\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", txt)
    return re.sub(
        r"\x00(\d+)\x00",
        lambda m: f'<font face="Courier" size="10.5">{guardados[int(m.group(1))]}</font>',
        txt,
    )


def _texto_puro(txt: str) -> str:
    """O que a célula mede para dimensionar coluna: sem marcação, com o conteúdo."""
    txt = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", txt)
    return re.sub(r"[`*]", "", txt).strip()


# ─── blocos ─────────────────────────────────────────────────────────────────

def _fonte_que_cabe(linhas: list[str], estrito: bool) -> float:
    """Maior corpo em que a linha mais longa ainda cabe na margem."""
    largura_max = LARGURA_UTIL - 10 * mm
    mais_longa = max((len(x) for x in linhas), default=1) or 1
    corpo = min(CORPO_CODIGO_MAX, largura_max / (mais_longa * RAZAO_COURIER))
    if corpo < CORPO_CODIGO_MIN:
        aviso = (f"bloco de código com {mais_longa} caracteres não cabe na margem "
                 f"nem a {CORPO_CODIGO_MIN} pt — o PDF sairia cortado")
        if estrito:
            raise ValueError(aviso)
        log.warning("%s (renderizando assim mesmo)", aviso)
        return CORPO_CODIGO_MIN
    return corpo


def bloco_codigo(linhas: list[str], estrito: bool = False) -> Table:
    corpo = _fonte_que_cabe(linhas, estrito)
    est = ParagraphStyle("cod", fontName="Courier", fontSize=corpo,
                         leading=corpo * 1.32, textColor=TINTA)
    t = Table([[Preformatted("\n".join(linhas), est)]],
              colWidths=[LARGURA_UTIL], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), FUNDO),
        ("BOX", (0, 0), (-1, -1), 0.25, LINHA),
        ("LEFTPADDING", (0, 0), (-1, -1), 5 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def bloco_citacao(paragrafos: list[str], st: dict) -> Table:
    """Bloco recuado — é o que carrega o Resumo e as citações em destaque.

    Recuo simétrico de 10 mm e um corpo abaixo do texto, sem fundo nem barra
    colorida: num artigo o Resumo se distingue por composição, não por adorno.
    """
    celulas = [[Paragraph(_inline(p), st["cita"])] for p in paragrafos]
    t = Table(celulas, colWidths=[LARGURA_UTIL - 20 * mm], hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


_NUMERICO = re.compile(r"^[−–—+±<>≈~]?[\d.,]+\s*(%|×|x|B|M|G|GB|MB|h|pt|s)?$|^—$|^-$|^$")


def _larguras(linhas: list[list[str]]) -> list[float]:
    """Reparte a margem entre colunas na proporção do conteúdo.

    O piso de cada coluna é a **maior palavra** que ela contém, medida na fonte
    em que será composta. Sem isso a repartição só olha o comprimento médio do
    texto, e uma coluna de números curtos com cabeçalho longo recebe largura
    menor que o próprio cabeçalho: `Caracteres/doc.` saía quebrado como
    `Caracteres/do` / `c.`, porque não há espaço onde quebrar e o reportlab parte
    dentro da palavra.

    Se a soma dos pisos não couber na margem — tabela com muitas colunas de
    rótulo longo —, todos são reduzidos na mesma proporção: aí a quebra dentro da
    palavra é inevitável, e o que se pode garantir é que ela seja distribuída em
    vez de concentrada numa coluna.
    """
    n = len(linhas[0])
    folga = 8.0  # os dois preenchimentos laterais da célula

    pisos, pesos = [], []
    for c in range(n):
        celulas = [_texto_puro(linha[c]) for linha in linhas]
        palavras = [p for celula in celulas for p in celula.split()]
        # O cabeçalho é composto em negrito, que é mais largo: mede-se por ele.
        pisos.append(folga + max(
            [stringWidth(p, SERIFA_N, 9.5) for p in palavras] or [0.0]))
        # A média entre o maior e o típico evita que uma célula longa isolada
        # engula a tabela inteira.
        tamanhos = sorted(len(celula) for celula in celulas)
        pesos.append(max(6.0, (tamanhos[-1] + tamanhos[len(tamanhos) // 2]) / 2))

    if sum(pisos) > LARGURA_UTIL:
        escala = LARGURA_UTIL / sum(pisos)
        return [p * escala for p in pisos]

    livre = LARGURA_UTIL - sum(pisos)
    total = sum(pesos)
    return [piso + livre * (peso / total)
            for piso, peso in zip(pisos, pesos, strict=True)]


def bloco_tabela(linhas: list[list[str]], st: dict) -> Table:
    corpo = linhas[1:]
    direita = [
        c for c in range(len(linhas[0]))
        if corpo and all(_NUMERICO.match(_texto_puro(linha[c])) for linha in corpo)
    ]
    dados = [[Paragraph(_inline(c), st["celh"]) for c in linhas[0]]]
    dados += [[Paragraph(_inline(c), st["cel"]) for c in linha] for linha in corpo]

    t = Table(dados, colWidths=_larguras(linhas), hAlign="LEFT", repeatRows=1)
    # Estilo de tabela científica: três filetes horizontais, nenhum vertical, sem
    # fundo. É a convenção tipográfica de periódico (booktabs), e não decoração:
    # régua vertical em tabela numérica compete com a leitura por coluna.
    estilo = [
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, TINTA),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, TINTA),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, TINTA),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    estilo += [("ALIGN", (c, 1), (c, -1), "RIGHT") for c in direita]
    t.setStyle(TableStyle(estilo))
    return t


# ─── analisador ─────────────────────────────────────────────────────────────

_CABECALHO = re.compile(r"^(#{1,4})\s+(.*)$")
_ITEM = re.compile(r"^(\s*)([-*]|\d+[.)])\s+(.*)$")
_REGUA = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
_QUEBRA = re.compile(r"^<!--\s*quebra\s*-->$")


def _celulas(linha: str) -> list[str]:
    return [c.strip() for c in linha.strip().strip("|").split("|")]


def _e_separador(linha: str) -> bool:
    if "|" not in linha:
        return False
    return all(re.fullmatch(r":?-{2,}:?", c) for c in _celulas(linha) if c)


_LEGENDA = re.compile(r"^\*\*(Tabela|Figura|Quadro)\s")
_REFERENCIA = re.compile(r"^\[\d+\]\s")


def _estilo_de(texto: str, no_cabecalho: bool) -> str:
    """Qual estilo um parágrafo recebe, pela função que ele exerce no artigo.

    Três papéis se distinguem do corpo por forma, e num artigo isso não é
    cosmético: a folha de rosto é centrada, a legenda de tabela é um corpo menor
    e fica colada à tabela, e a referência leva recuo pendente para o número
    saltar na varredura visual.
    """
    if no_cabecalho:
        return "meta"
    if _LEGENDA.match(texto):
        return "legenda"
    if _REFERENCIA.match(texto):
        return "ref"
    return "p"


def montar(md: str, st: dict, estrito: bool) -> list:
    linhas = md.splitlines()
    flow: list = []
    i, n = 0, len(linhas)
    primeiro_titulo = True
    # Tudo entre o título e a primeira seção é folha de rosto: autoria, filiação,
    # data. Delimitar por posição evita ter de marcar cada linha no Markdown.
    no_cabecalho = False

    while i < n:
        linha = linhas[i]

        if not linha.strip():
            i += 1
            continue

        if _QUEBRA.match(linha.strip()):
            flow.append(PageBreak())
            i += 1
            continue

        # ── bloco de código ────────────────────────────────────────────────
        if linha.lstrip().startswith("```"):
            i += 1
            corpo: list[str] = []
            while i < n and not linhas[i].lstrip().startswith("```"):
                corpo.append(linhas[i].rstrip())
                i += 1
            i += 1
            while corpo and not corpo[-1].strip():
                corpo.pop()
            flow += [bloco_codigo(corpo or [""], estrito), Spacer(1, 7)]
            continue

        # ── tabela ─────────────────────────────────────────────────────────
        if "|" in linha and i + 1 < n and _e_separador(linhas[i + 1]):
            cabecalho = _celulas(linha)
            i += 2
            corpo_tab = []
            while i < n and "|" in linhas[i] and linhas[i].strip():
                celulas = _celulas(linhas[i])
                celulas += [""] * (len(cabecalho) - len(celulas))
                corpo_tab.append(celulas[:len(cabecalho)])
                i += 1
            flow += [bloco_tabela([cabecalho] + corpo_tab, st), Spacer(1, 8)]
            continue

        # ── citação ────────────────────────────────────────────────────────
        if linha.lstrip().startswith(">"):
            paragrafos: list[str] = []
            atual: list[str] = []
            while i < n and linhas[i].lstrip().startswith(">"):
                conteudo = linhas[i].lstrip()[1:].strip()
                if conteudo:
                    atual.append(conteudo)
                elif atual:
                    paragrafos.append(" ".join(atual))
                    atual = []
                i += 1
            if atual:
                paragrafos.append(" ".join(atual))
            flow += [bloco_citacao(paragrafos, st), Spacer(1, 8)]
            continue

        # ── régua ──────────────────────────────────────────────────────────
        if _REGUA.match(linha):
            flow += [Spacer(1, 5),
                     HRFlowable(width="100%", thickness=0.5, color=LINHA,
                                spaceBefore=0, spaceAfter=0),
                     Spacer(1, 5)]
            i += 1
            continue

        # ── título ─────────────────────────────────────────────────────────
        m = _CABECALHO.match(linha)
        if m:
            nivel, texto = len(m.group(1)), m.group(2)
            if nivel == 1 and primeiro_titulo:
                flow.append(Paragraph(_inline(texto), st["titulo"]))
                primeiro_titulo = False
                no_cabecalho = True
            else:
                no_cabecalho = False
                flow.append(Paragraph(_inline(texto),
                                      st[{1: "h1", 2: "h1", 3: "h2"}.get(nivel, "h3")]))
            i += 1
            continue

        # ── lista ──────────────────────────────────────────────────────────
        m = _ITEM.match(linha)
        if m:
            itens: list[tuple[str, str]] = []
            while i < n:
                mi = _ITEM.match(linhas[i])
                if not mi:
                    # Continuação recuada pertence ao item anterior.
                    if itens and linhas[i].startswith(("  ", "\t")) and linhas[i].strip():
                        itens[-1] = (itens[-1][0], itens[-1][1] + " " + linhas[i].strip())
                        i += 1
                        continue
                    break
                marca = mi.group(2)
                marca = "•" if marca in {"-", "*"} else marca
                itens.append((marca, mi.group(3).strip()))
                i += 1
            for marca, texto in itens:
                flow.append(Paragraph(_inline(texto), st["item"],
                                      bulletText=_inline(marca)))
            flow.append(Spacer(1, 4))
            continue

        # ── parágrafo ──────────────────────────────────────────────────────
        partes = []
        while i < n and linhas[i].strip():
            atual_linha = linhas[i]
            if (_CABECALHO.match(atual_linha) or _ITEM.match(atual_linha)
                    or _REGUA.match(atual_linha) or atual_linha.lstrip().startswith((">", "```"))
                    or ("|" in atual_linha and i + 1 < n and _e_separador(linhas[i + 1]))):
                break
            partes.append(atual_linha.strip())
            i += 1
        texto = " ".join(partes)
        flow.append(Paragraph(_inline(texto), st[_estilo_de(texto, no_cabecalho)]))

    return _agrupar_titulos(flow)


LINHAS_QUE_CABEM_JUNTO = 8


def _agrupar_titulos(flow: list) -> list:
    """Impede título órfão no pé da página, sem abrir meia página em branco.

    `keepWithNext` cobre o caso de o próximo elemento ser um `Paragraph`; quando é
    tabela, não — daí o `KeepTogether` explícito.

    Mas só para tabela CURTA. A tabela de estado da §11 tem 15 linhas, e o
    `KeepTogether` a empurrava inteira para a página seguinte deixando dez
    centímetros vazios. Tabela longa parte melhor do que salta: o `repeatRows=1`
    repete o cabeçalho, então nenhuma linha fica órfã de rótulo.
    """
    saida: list = []
    pulo = False
    for j, elemento in enumerate(flow):
        if pulo:
            pulo = False
            continue
        seguinte = flow[j + 1] if j + 1 < len(flow) else None
        e_titulo = (isinstance(elemento, Paragraph)
                    and elemento.style.name in {"h1", "h2", "h3"})
        curta = (isinstance(seguinte, Table)
                 and len(seguinte._cellvalues) <= LINHAS_QUE_CABEM_JUNTO)
        if e_titulo and curta:
            saida.append(KeepTogether([elemento, seguinte]))
            pulo = True
        else:
            saida.append(elemento)
    return saida


# ─── documento ──────────────────────────────────────────────────────────────

def _rodape(rotulo: str):
    """Número de página centrado, e nada mais.

    A versão anterior punha um rótulo à esquerda e um filete acima. Num artigo o
    rodapé é folha corrida — cabeçalho e adorno são de relatório.
    """
    def desenhar(canvas, doc):
        canvas.saveState()
        canvas.setFont(SERIFA, 10)
        canvas.setFillColor(TINTA)
        canvas.drawCentredString(A4[0] / 2, 14 * mm, str(canvas.getPageNumber()))
        canvas.restoreState()
    return desenhar


def render(entrada: Path, saida: Path, estrito: bool = False) -> Path:
    md = entrada.read_text(encoding="utf-8")
    _conferir_glifos(md, estrito)
    st = _estilos()
    titulo = next((x[2:].strip() for x in md.splitlines() if x.startswith("# ")),
                  entrada.stem)
    saida.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(saida), pagesize=A4,
        leftMargin=MARGEM, rightMargin=MARGEM,
        topMargin=MARGEM, bottomMargin=MARGEM,
        title=titulo, author="Vinicius Sanchez", subject="rascunho de artigo",
    )
    rotulo = f"{date.today():%Y-%m-%d}"
    doc.build(montar(md, st, estrito), onFirstPage=_rodape(rotulo),
              onLaterPages=_rodape(rotulo))
    return saida


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--entrada", type=Path,
                   default=Path("docs/papers/rascunho-artigo-recuperacao-fisica.md"))
    p.add_argument("--saida", type=Path, default=None,
                   help="padrão: build/<nome-da-entrada>.pdf")
    p.add_argument("--estrito", action="store_true",
                   help="levanta em vez de avisar quando um bloco de código não cabe")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not a.entrada.exists():
        log.error("entrada não encontrada: %s", a.entrada)
        return 2
    saida = a.saida or Path("build") / f"{a.entrada.stem}.pdf"
    render(a.entrada, saida, estrito=a.estrito)
    log.info("%s · %.0f KB", saida, saida.stat().st_size / 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
