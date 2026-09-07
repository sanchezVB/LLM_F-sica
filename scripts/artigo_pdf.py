#!/usr/bin/env python3
"""Renderiza um rascunho de artigo de `docs/papers/` em PDF.

    PYTHONPATH=src python scripts/artigo_pdf.py \
        --entrada docs/papers/rascunho-artigo-programa-phifm.md \
        --saida build/rascunho-artigo-programa-phifm.pdf

## Por que um renderizador e não o PDF escrito à mão

O `scripts/relatorio_pdf.py` monta um relatório LENDO os artefatos de medição —
nenhum número é digitado nele. Um artigo é outra coisa: o texto é a contribuição,
e ele precisa viver em Markdown versionado, revisável em diff, ao lado dos outros
rascunhos de `docs/papers/`.

Então a fonte é o `.md` e o PDF é ARTEFATO DE BUILD (`*.pdf` está no
`.gitignore`). Gerar o PDF e versioná-lo criaria duas cópias do mesmo texto, e
este repositório já pagou o preço disso uma vez: três cópias da tabela de estado,
no README, no índice e no DOC-00 §11, divergiram e as três afirmavam coisas
diferentes sobre o mesmo fato.

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
fixa deste projeto (o bake-off do tokenizer, o T1c) têm 60–75 caracteres, e uma
delas passando a 90 sairia cortada num PDF que alguém leria como completo.

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
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
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

MARGEM = 20 * mm
LARGURA_UTIL = A4[0] - 2 * MARGEM

TINTA = colors.HexColor("#1a1a1a")
SUAVE = colors.HexColor("#5b6470")
LINHA = colors.HexColor("#d5dae0")
FUNDO = colors.HexColor("#f2f5f8")
REALCE = colors.HexColor("#3b6ea5")

# Courier tem largura fixa de 0,6 em por caractere. É isto que torna a conta de
# `_fonte_que_cabe` exata em vez de estimada.
RAZAO_COURIER = 0.6
CORPO_CODIGO_MAX = 8.5
CORPO_CODIGO_MIN = 5.5

# ── Cobertura de glifo, e por que existe uma guarda em vez de confiança ──────
#
# O documento fica nas fontes base-14 (Helvetica/Courier), que o reportlab tem
# embutidas: o PDF sai IDÊNTICO no Windows do autor e no runner do CI. Registrar
# uma DejaVu do sistema resolveria todo o Unicode e faria o resultado depender da
# máquina — a troca não vale num repositório cujo portão G1.5 é reprodutibilidade.
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
EXPOENTES = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
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
        "titulo": ParagraphStyle("titulo", parent=s["Title"], fontSize=19, leading=24,
                                 textColor=TINTA, alignment=0, spaceAfter=10),
        "meta": ParagraphStyle("meta", parent=s["Normal"], fontSize=9, leading=13,
                               textColor=SUAVE, spaceAfter=4),
        "h1": ParagraphStyle("h1", parent=s["Heading1"], fontSize=13.5, leading=17,
                             textColor=TINTA, spaceBefore=18, spaceAfter=7,
                             keepWithNext=True),
        "h2": ParagraphStyle("h2", parent=s["Heading2"], fontSize=11, leading=14,
                             textColor=TINTA, spaceBefore=12, spaceAfter=5,
                             keepWithNext=True),
        "h3": ParagraphStyle("h3", parent=s["Heading3"], fontSize=9.8, leading=13,
                             textColor=SUAVE, spaceBefore=10, spaceAfter=4,
                             keepWithNext=True),
        "p": ParagraphStyle("p", parent=s["Normal"], fontSize=9.4, leading=13.6,
                            textColor=TINTA, alignment=TA_JUSTIFY, spaceAfter=7),
        "item": ParagraphStyle("item", parent=s["Normal"], fontSize=9.4, leading=13.2,
                               textColor=TINTA, alignment=TA_JUSTIFY,
                               leftIndent=9 * mm, bulletIndent=3.5 * mm, spaceAfter=4),
        "cita": ParagraphStyle("cita", parent=s["Normal"], fontSize=9.2, leading=13.2,
                               textColor=TINTA, alignment=TA_JUSTIFY, spaceAfter=5),
        "cel": ParagraphStyle("cel", parent=s["Normal"], fontSize=7.9, leading=10.2,
                              textColor=TINTA),
        "celh": ParagraphStyle("celh", parent=s["Normal"], fontSize=7.9, leading=10.2,
                               textColor=TINTA, fontName="Helvetica-Bold"),
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
    txt = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+",
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
        lambda m: f'<font face="Courier" size="8.2">{guardados[int(m.group(1))]}</font>',
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
    celulas = [[Paragraph(_inline(p), st["cita"])] for p in paragrafos]
    t = Table(celulas, colWidths=[LARGURA_UTIL], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f9fb")),
        ("LINEBEFORE", (0, 0), (0, -1), 1.6, REALCE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


_NUMERICO = re.compile(r"^[−–—+±<>≈~]?[\d.,]+\s*(%|×|x|B|M|G|GB|MB|h|pt|s)?$|^—$|^-$|^$")


def _larguras(linhas: list[list[str]]) -> list[float]:
    """Reparte a margem entre colunas na proporção do conteúdo, com piso.

    Sem o piso, uma coluna de valores curtos ao lado de uma de prosa fica com 6 mm
    e quebra um número por linha.
    """
    n = len(linhas[0])
    pesos = []
    for c in range(n):
        # A média entre o maior e o típico evita que uma célula longa isolada
        # engula a tabela inteira.
        tamanhos = sorted(len(_texto_puro(linha[c])) for linha in linhas)
        maior = tamanhos[-1]
        tipico = tamanhos[len(tamanhos) // 2]
        pesos.append(max(6.0, (maior + tipico) / 2))
    piso = 13 * mm if n <= 6 else 10 * mm
    livre = LARGURA_UTIL - piso * n
    total = sum(pesos)
    return [piso + livre * (p / total) for p in pesos]


def bloco_tabela(linhas: list[list[str]], st: dict) -> Table:
    corpo = linhas[1:]
    direita = [
        c for c in range(len(linhas[0]))
        if corpo and all(_NUMERICO.match(_texto_puro(linha[c])) for linha in corpo)
    ]
    dados = [[Paragraph(_inline(c), st["celh"]) for c in linhas[0]]]
    dados += [[Paragraph(_inline(c), st["cel"]) for c in linha] for linha in corpo]

    t = Table(dados, colWidths=_larguras(linhas), hAlign="LEFT", repeatRows=1)
    estilo = [
        ("BACKGROUND", (0, 0), (-1, 0), FUNDO),
        ("GRID", (0, 0), (-1, -1), 0.25, LINHA),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINHA),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
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


def montar(md: str, st: dict, estrito: bool) -> list:
    linhas = md.splitlines()
    flow: list = []
    i, n = 0, len(linhas)
    primeiro_titulo = True

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
            else:
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
        estilo = "meta" if texto.startswith(("**Rascunho", "**Autoria", "**Estado do")) else "p"
        flow.append(Paragraph(_inline(texto), st[estilo]))

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
    def desenhar(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(SUAVE)
        canvas.drawString(MARGEM, 12 * mm, rotulo)
        canvas.drawRightString(A4[0] - MARGEM, 12 * mm, str(canvas.getPageNumber()))
        canvas.setStrokeColor(LINHA)
        canvas.setLineWidth(0.4)
        canvas.line(MARGEM, 15 * mm, A4[0] - MARGEM, 15 * mm)
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
        topMargin=18 * mm, bottomMargin=22 * mm,
        title=titulo, author="ΦFM", subject="rascunho de trabalho",
    )
    rotulo = f"ΦFM · rascunho de trabalho · {date.today():%Y-%m-%d}"
    doc.build(montar(md, st, estrito), onFirstPage=_rodape(rotulo),
              onLaterPages=_rodape(rotulo))
    return saida


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--entrada", type=Path,
                   default=Path("docs/papers/rascunho-artigo-programa-phifm.md"))
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
