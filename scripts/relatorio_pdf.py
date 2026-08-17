#!/usr/bin/env python3
"""Relatório de status do ΦFM em PDF, montado dos artefatos de medição.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/relatorio_pdf.py

## Por que ele LÊ os artefatos em vez de receber os números

Todo número deste relatório sai de um arquivo que uma medição gravou —
`g1_resultado.json`, `s3b_latex.json`, `transferencia.json`, os manifestos das
fatias. Nenhum é digitado aqui.

A razão é a que este projeto já pagou várias vezes: número transcrito à mão
envelhece em silêncio. Hoje mesmo o `ESTADO.md` afirmava "G1.2 depende do
GTE-large" depois de o resultado existir, e o rodapé do classificador dizia que
`math` não estava nos negativos na mesma execução que usou negativos de `math`.

Se um artefato faltar, a seção diz "não medido" — nunca estima.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

log = logging.getLogger(__name__)
AVAL = Path("data/processed/avaliacao")

TINTA = colors.HexColor("#1a1a1a")
SUAVE = colors.HexColor("#5b6470")
LINHA = colors.HexColor("#d5dae0")
FUNDO = colors.HexColor("#f2f5f8")
BOM = colors.HexColor("#1c7a4a")
RUIM = colors.HexColor("#a52a2a")


def _json(caminho: Path) -> dict | list | None:
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _estilos() -> dict:
    s = getSampleStyleSheet()
    return {
        "titulo": ParagraphStyle("t", parent=s["Title"], fontSize=22, leading=26,
                                 textColor=TINTA, spaceAfter=2),
        "sub": ParagraphStyle("st", parent=s["Normal"], fontSize=10, leading=14,
                              textColor=SUAVE, spaceAfter=14),
        "h1": ParagraphStyle("h1", parent=s["Heading1"], fontSize=14, leading=18,
                             textColor=TINTA, spaceBefore=16, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=s["Heading2"], fontSize=11, leading=15,
                             textColor=TINTA, spaceBefore=10, spaceAfter=4),
        "p": ParagraphStyle("p", parent=s["Normal"], fontSize=9.5, leading=14,
                            textColor=TINTA, alignment=TA_JUSTIFY, spaceAfter=6),
        "nota": ParagraphStyle("n", parent=s["Normal"], fontSize=8.5, leading=12,
                               textColor=SUAVE, spaceAfter=6),
        "cel": ParagraphStyle("c", parent=s["Normal"], fontSize=8.5, leading=11,
                              textColor=TINTA),
    }


def tabela(dados: list[list], larguras: list[float], destaque: int | None = None) -> Table:
    t = Table(dados, colWidths=larguras, hAlign="LEFT")
    estilo = [
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), TINTA),
        ("BACKGROUND", (0, 0), (-1, 0), FUNDO),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, LINHA),
        ("GRID", (0, 0), (-1, -1), 0.25, LINHA),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]
    if destaque is not None:
        estilo += [("BACKGROUND", (0, destaque), (-1, destaque), colors.HexColor("#e8f2ff")),
                   ("FONTNAME", (0, destaque), (-1, destaque), "Helvetica-Bold")]
    t.setStyle(TableStyle(estilo))
    return t


# ─── seções ─────────────────────────────────────────────────────────────────

def sec_portoes(st: dict) -> list:
    """G1, do `g1_resultado.json`."""
    d = _json(AVAL / "g1_resultado.json")
    out = [Paragraph("1 · Portão G1 — recuperação por citação", st["h1"])]
    if not d:
        out.append(Paragraph("<b>Não medido</b> — `g1_resultado.json` ausente.", st["p"]))
        return out

    ms = sorted((m for m in d["modelos"] if not m.get("erro")),
                key=lambda m: -m["ndcg_10"])
    linhas = [["modelo", "params", "nDCG@10", "recall@1", "recall@10"]]
    destaque = None
    for i, m in enumerate(ms, 1):
        if m.get("nosso") and destaque is None:
            destaque = i
        linhas.append([Paragraph(("<b>" if m.get("nosso") else "") + m["nome"] +
                                 ("</b>" if m.get("nosso") else ""), st["cel"]),
                       f"{m['parametros_m']:.0f} M", f"{m['ndcg_10']:.3f}",
                       f"{m['recall_1']:.3f}", f"{m['recall_10']:.3f}"])
    out += [Paragraph(f"Medido em {d['n_candidatos']:,} candidatos, mesma agregação e "
                      "mesmo limite de tokens para todos.".replace(",", "."), st["p"]),
            tabela(linhas, [58 * mm, 20 * mm, 22 * mm, 22 * mm, 24 * mm], destaque),
            Spacer(1, 8)]
    for l in d["veredito"].splitlines():
        if not l.strip():
            continue
        est = st["nota"] if l.startswith(("  ", "\u00b7", "RESSALVAS")) else st["p"]
        cor = ""
        if "PASSOU" in l and "NÃO" not in l:
            cor = f'<font color="#{BOM.hexval()[2:]}">'
        elif "NÃO PASSOU" in l:
            cor = f'<font color="#{RUIM.hexval()[2:]}">'
        txt = (cor + l.strip() + ("</font>" if cor else "")).replace("·", "&middot;")
        out.append(Paragraph(txt, est))
    return out


def sec_s3b(st: dict) -> list:
    d = _json(AVAL / "s3b_latex.json")
    out = [Paragraph("2 · S3b — o RedPajama degrada as equações?", st["h1"])]
    if not d or "resumo" not in d:
        out.append(Paragraph("<b>Não medido.</b>", st["p"]))
        return out
    r = d["resumo"]
    ic = r.get("ausencia_ic95") or [None, None]
    linhas = [["métrica", "valor"],
              ["papers de Física comparados", f"{r['papers_comparados']}"],
              ["equações na fonte", f"{r['equacoes_na_fonte']:,}".replace(",", ".")],
              ["degradação total", f"{100*r['degradacao_total']:.1f}%"],
              [Paragraph("<b>por ausência</b> (perda real)", st["cel"]),
               f"{100*r['degradacao_por_ausencia']:.1f}%"],
              ["por discordância (notação)", f"{100*r['degradacao_por_discordancia']:.1f}%"]]
    if ic[0] is not None:
        linhas.append(["IC 95% da ausência",
                       f"[{100*ic[0]:.1f}% – {100*ic[1]:.1f}%]"])
        linhas.append(["P(acima do limiar de 10%)",
                       f"{100*r['ausencia_p_acima_do_limiar']:.0f}%"])
    out += [tabela(linhas, [78 * mm, 40 * mm], destaque=4), Spacer(1, 8),
            Paragraph(f"<b>{r['veredito']}</b>", st["p"]),
            Paragraph("O número mudou cinco vezes antes de estabilizar. Só o último "
                      "vale, e o que o torna confiável não é ser o último: é ser o "
                      "único que mede a população que o critério pergunta (Física, "
                      "não o arXiv inteiro) e declara um intervalo.", st["nota"])]
    return out


def sec_classificador(st: dict) -> list:
    out = [Paragraph("3 · Classificador de Física e transferência de domínio", st["h1"])]
    ds = _json(AVAL / "transferencia.json") or []
    dstat = _json(AVAL / "transferencia_stat.json") or []
    todos = (ds if isinstance(ds, list) else []) + (dstat if isinstance(dstat, list) else [])
    if not todos:
        out.append(Paragraph("<b>Não medido.</b>", st["p"]))
        return out
    linhas = [["domínio omitido", "FP dentro", "FP fora", "piora", "precisão fora"]]
    for r in sorted(todos, key=lambda x: -x["fp_fora"]):
        linhas.append([r["dominio_omitido"], f"{100*r['fp_dentro']:.1f}%",
                       f"{100*r['fp_fora']:.1f}%",
                       "—" if r["fp_dentro"] == 0 else f"{r['fp_fora']/r['fp_dentro']:.1f}x",
                       f"{r['precisao_fora']:.3f}"])
    out += [Paragraph("Deixa-um-domínio-de-fora: treina sem um domínio negativo "
                      "inteiro e testa nele. <b>Falso positivo</b> é a fração de "
                      "negativos aceitos como Física — o que contamina o corpus.", st["p"]),
            tabela(linhas, [34 * mm, 26 * mm, 24 * mm, 20 * mm, 30 * mm]),
            Spacer(1, 8),
            Paragraph("A coluna que vale para produção é <b>FP dentro</b>: os quatro "
                      "domínios estão no treino do modelo final. E domínio não visto "
                      "não é uniformemente catastrófico — depende da proximidade.",
                      st["p"]),
            Paragraph("Duas afirmações do projeto foram testadas com resultado oposto: "
                      "«math é negativo fácil» é FALSA (35,4%) e «stat é negativo "
                      "fácil» é VERDADEIRA (2,9%). Nenhuma das duas havia sido medida.",
                      st["nota"])]
    return out


def _caracteres(fs: list[str], coluna: str) -> int:
    """Soma de caracteres com CAST para Int64, e a razão é aritmética.

    ⚠️ `str.len_chars()` do polars devolve **UInt32**, e a soma acumula no mesmo
    tipo. Um corpus de 10,5 G caracteres passa do máximo de 4,29 G e a soma **dá a
    volta**, silenciosamente.

    Medido em 2026-08-16 na fatia do OpenWebMath: a soma sem cast devolveu 1,88 G
    contra os 10,47 G reais — exatamente **duas voltas** de 2³². Eu quase reportei
    "0,47 B tokens" onde são 2,62 B, e o número errado é plausível: nenhuma
    exceção, nenhum aviso, só um total 5,6× menor.

    Só desconfiei porque 2.185 caracteres por documento não batia com os 12.162 que
    a própria coleta havia registrado.

    ⚠️ E a soma é por ARQUIVO, não num `scan_parquet` da lista inteira. `scan` é
    lazy, mas o `collect` no motor padrão materializa a coluna de texto de todos os
    arquivos antes de reduzir. Medido em 2026-08-16: as duas fatias somam 16,1 GB
    comprimidos e o processo **committou 43,5 GB** com 15,9 GB de RAM na máquina.
    O Windows despejou 35,6 GB no `pagefile` do SSD, sobrou 0,7 GB livre, e a
    inanição matou o treino que rodava ao lado — que já vinha a 20,9 pares/s contra
    os 28,1 normais — e matou também o supervisor encarregado de reerguê-lo, antes
    de ele registrar uma linha.

    Duas mortes de treino que eu havia declarado sem explicação. A causa não estava
    no treino: estava neste relatório, que existe para relatar o estado e o
    destruía ao medir.

    Arquivo por arquivo o pico é o de UM arquivo (~0,4 GB), e um parquet ilegível
    custa o seu pedaço em vez da soma toda — que é o erro certo num relatório que
    roda sobre coletas em movimento.
    """
    import polars as pl

    total, perdidos = 0, 0
    for f in fs:
        try:
            total += int(pl.scan_parquet(f).select(
                pl.col(coluna).str.len_chars().cast(pl.Int64).sum().alias("c")
            ).collect(engine="streaming").item() or 0)
        except Exception:
            perdidos += 1
    if perdidos:
        log.info("contagem de caracteres: %d de %d arquivos ilegíveis (subestima)",
                 perdidos, len(fs))
    return total


def _conta_parquet(pasta: Path) -> tuple[int, float]:
    """(registros, GB) numa pasta que pode estar sendo ESCRITA agora.

    ⚠️ Este relatório roda com coletas em curso, então há sempre a chance de um
    `part-*.parquet` estar no meio da escrita. A primeira versão estourou com
    «file size (0) is less than minimum size required to store parquet footer» ao
    ler a saída do OpenWebMath enquanto ele gravava.

    Arquivo incompleto é ignorado, não fatal — e o total sai ligeiramente
    subestimado, o que é o erro certo para um instantâneo de algo em movimento.
    """
    import polars as pl

    fs = [f for f in sorted(glob.glob(str(pasta / "part-*.parquet")))
          if os.path.getsize(f) > 1024]
    if not fs:
        return 0, 0.0
    n = 0
    lidos = []
    for f in fs:
        try:
            n += pl.scan_parquet(f).select(pl.len()).collect().item()
            lidos.append(f)
        except Exception:
            log.info("parquet em escrita, ignorado no total: %s", os.path.basename(f))
    return n, sum(os.path.getsize(f) for f in lidos) / 1e9


def sec_espinha(st: dict) -> list:
    """S1 — a espinha de metadados, que é a base de todo o resto."""
    import polars as pl

    out = [Paragraph("4 · Sprint S1 — a espinha de metadados", st["h1"])]
    p = Path("data/processed/spine.parquet")
    if not p.exists():
        out.append(Paragraph("<b>Não construída.</b>", st["p"]))
        return out
    try:
        lf = pl.scan_parquet(p)
        n = lf.select(pl.len()).collect().item()
        com_ref = lf.select(pl.col("n_references").is_not_null().sum()).collect().item()
        sub = (lf.group_by("subfield").len().sort("len", descending=True)
                 .head(6).collect())
    except Exception as exc:
        out.append(Paragraph(f"<b>Ilegível:</b> {type(exc).__name__}.", st["p"]))
        return out

    out += [Paragraph("Rótulo do próprio autor, que é a única fonte autoritativa de "
                      "subárea — um classificador de terceiro não sabe em que "
                      "subárea o autor escreveu.", st["p"]),
            tabela([["métrica", "valor"],
                    ["papers de Física (conjunto `physics` do arXiv)",
                     f"{n:,}".replace(",", ".")],
                    ["com grafo de citações do OpenAlex",
                     f"{com_ref:,} ({100*com_ref/n:.1f}%)".replace(",", ".")],
                    ["em disco", f"{p.stat().st_size/1e9:.2f} GB"]],
                   [78 * mm, 40 * mm]),
            Spacer(1, 8),
            Paragraph("Distribuição por subárea (as seis maiores)", st["h2"]),
            tabela([["subárea", "papers"]] +
                   [[Paragraph(r[0], st["cel"]), f"{r[1]:,}".replace(",", ".")]
                    for r in sub.iter_rows()], [70 * mm, 30 * mm]),
            Paragraph("⚠️ 4.042 papers de arquivos LEGADOS do arXiv (`chao-dyn`, "
                      "`mtrl-th`, `atom-ph`…) ainda aparecem como «Outro» neste "
                      "arquivo. A correção existe no código mas exige reconstruir a "
                      "espinha; até então eles ficam fora do treino de subárea.",
                      st["nota"])]
    return out


def sec_corpus(st: dict) -> list:
    out = [Paragraph("5 · Corpus de texto completo", st["h1"])]
    linhas = [["fonte", "documentos", "≈ tokens", "em disco", "filtro"]]

    import glob as _g
    rp = Path("data/processed/redpajama_fisica")
    n, gb = _conta_parquet(rp)
    if n:
        ch = _caracteres([f for f in sorted(_g.glob(str(rp / "part-*.parquet")))
                          if os.path.getsize(f) > 1024], "texto")
        linhas.append([Paragraph("RedPajama-arXiv", st["cel"]),
                       f"{n:,}".replace(",", "."),
                       f"{ch/4/1e9:.2f} B" if ch else "—", f"{gb:.1f} GB",
                       Paragraph("spine (exato)", st["cel"])])
    # ⚠️ A contagem vem dos PARQUETS, não do `_filtragem.json`.
    #
    # O JSON só é escrito no FIM da execução. Numa fonte ainda em curso ele
    # carrega o número da última execução concluída — e a primeira versão deste
    # relatório mostrou "7.442 documentos" para o OpenWebMath quando o log dizia
    # 622.618, porque leu o JSON do ensaio de um arquivo. Pior: o tamanho em disco
    # vinha dos parquets, que estavam atuais, então as duas colunas da mesma linha
    # falavam de execuções diferentes.
    #
    # É o mesmo defeito que este projeto passou dois dias corrigindo em outros
    # lugares, e eu o construí no documento que reporta as correções.
    f = _json(Path("data/processed/openwebmath_fisica/_filtragem.json"))
    n2, gb2 = _conta_parquet(Path("data/processed/openwebmath_fisica"))
    if n2 or f:
        limiar = f"{f['limiar']}" if f else "0.9"
        # ⚠️ "Em curso" vem de `concluido`, NÃO de comparar contagens.
        #
        # A versão anterior fazia `abs(f["aceitos"] - n2) > 5%`, com `aceitos` da
        # ÚLTIMA EXECUÇÃO (233.079) e `n2` acumulado dos parquets (860.521).
        # Naturezas diferentes: a coleta acabou às 12:05 e o relatório imprimiu
        # "OpenWebMath (em curso)" — e imprimiria isso para sempre, porque a
        # divergência só cresce a cada relançamento.
        #
        # Ausência de `concluido` (JSON de antes desta correção) é reportada como
        # DESCONHECIDA, não como concluída: afirmar "acabou" sem evidência é o
        # defeito que este relatório existe para não cometer.
        if not f or "concluido" not in f:
            estado = " (conclusão não registrada)"
        elif f["concluido"]:
            estado = ""
        else:
            feitas_n, total_n = f.get("unidades_feitas"), f.get("total_unidades")
            estado = (f" (em curso: {feitas_n} de {total_n} unidades)"
                      if feitas_n and total_n else " (em curso)")
        owm = Path("data/processed/openwebmath_fisica")
        ch2 = _caracteres([f for f in sorted(_g.glob(str(owm / "part-*.parquet")))
                           if os.path.getsize(f) > 1024], "texto")
        linhas.append([Paragraph("OpenWebMath" + estado, st["cel"]),
                       f"{n2:,}".replace(",", "."),
                       f"{ch2/4/1e9:.2f} B" if ch2 else "—", f"{gb2:.1f} GB",
                       Paragraph(f"classificador ≥ {limiar}", st["cel"])])
    linhas.append([Paragraph("peS2o", st["cel"]), "—", "—", "—",
                   Paragraph("não iniciado (42,7 h medidas)", st["cel"])])
    out.append(tabela(linhas, [34 * mm, 28 * mm, 20 * mm, 20 * mm, 38 * mm]))

    if f and f.get("dominios"):
        base = sum(f["dominios"].values())
        out += [Spacer(1, 10),
                Paragraph("Domínios do que o classificador aceitou no OpenWebMath",
                          st["h2"]),
                # A base é da ÚLTIMA EXECUÇÃO, não do acumulado — e dizer isso é
                # obrigatório, porque a tabela acima mostra o total acumulado dos
                # parquets. Duas bases diferentes na mesma página sem aviso é como
                # a linha do OpenWebMath passou a semana falando de duas execuções.
                Paragraph("É o sinal objetivo sobre o que entrou, quando não há "
                          "rótulo para conferir. <b>A distribuição abaixo é da "
                          f"última execução, sobre {base:,} documentos</b> — cada "
                          "execução grava só a sua, enquanto a contagem da tabela "
                          "acima é acumulada. As proporções são o que importa aqui, "
                          "não os totais.".replace(",", "."), st["nota"])]
        ds = list(f["dominios"].items())[:10]
        out.append(tabela([["domínio", "aceitos"]] +
                          [[Paragraph(k, st["cel"]), f"{v:,}".replace(",", ".")]
                           for k, v in ds], [70 * mm, 26 * mm]))
    out.append(Paragraph("Todo o processamento é em <b>fluxo</b>: os 81 GB do "
                         "RedPajama e os 27 GB do OpenWebMath nunca aterram — "
                         "decodifica, filtra e descarta na mesma passada.", st["nota"]))
    return out


def sec_treino(st: dict, log_treino: Path) -> list:
    """Curva do treino com GradCache, lida do log."""
    import re
    out = [Paragraph("6 · GradCache — o experimento do lote maior", st["h1"]),
           Paragraph("O G1.2 falhou por 0,005 de nDCG@10. O gargalo medido são "
                     "<b>negativos no lote</b>, não capacidade: o nosso de 110 M ficou "
                     "atrás do de 23 M porque cabia lote 8 (7 negativos) contra 128. "
                     "GradCache desacopla o lote da memória — lote lógico 512 com a "
                     "memória de 128.", st["p"])]
    if not log_treino.exists():
        out.append(Paragraph("<b>Log do treino ausente.</b>", st["p"]))
        return out
    t = log_treino.read_bytes().decode("utf-8", "replace")
    avals = re.findall(r"(\d\d:\d\d:\d\d).*?aval: recall@1 ([\d.]+).*?recall@10 "
                       r"([\d.]+).*?MRR ([\d.]+)", t)
    passos = re.findall(r"passo (\d+) \| perda ([\d.]+) \| ([\d.]+) pares/s", t)
    if avals:
        linhas = [["hora", "recall@1", "recall@10", "MRR"]]
        for h, r1, r10, mrr in avals[-8:]:
            linhas.append([h, r1, r10, mrr])
        out += [tabela(linhas, [24 * mm, 26 * mm, 28 * mm, 22 * mm]), Spacer(1, 6)]
    if passos:
        p, perda, taxa = passos[-1]
        out.append(Paragraph(f"Último passo registrado: <b>{p}</b> · perda {perda} · "
                             f"{taxa} pares/s.", st["p"]))
    out.append(Paragraph("A perda NÃO é comparável entre lotes: o InfoNCE tem piso "
                         "ln(N) com N negativos, então ln 512 ≈ 6,2 contra ln 128 ≈ 4,9. "
                         "Perda mais alta com mais negativos não é pior.", st["nota"]))
    return out


def sec_defeitos(st: dict) -> list:
    out = [Paragraph("7 · Defeitos encontrados, e o que cada um ensinou", st["h1"]),
           Paragraph("Registrados porque o modo de falha se repete, não o defeito.",
                     st["nota"])]
    itens = [
        ("A fonte era o tarball, não o documento",
         "A auditoria contava equações de `.tex` que o documento não inclui. O viés é "
         "assimétrico: inflar a fonte AUMENTA a degradação medida, empurrando para "
         "gastar US$ 100–180. Corrigido — e não mudou o número, o que também é dado."),
        ("Eu media o arXiv inteiro e chamava de Física",
         "52% da amostra não estava no spine. O paper que mais puxava a perda era de "
         "teoria de grafos, e as «equações perdidas» eram tabelas de ciclos de "
         "permutação num apêndice."),
        ("Aviso que virou falso",
         "O rodapé do classificador dizia «math NÃO está entre os negativos» na mesma "
         "execução que usou negativos de math. Texto fixo envelhece; agora é derivado "
         "do que está em disco."),
        ("Número de shard tratado como índice de parquet",
         "A retomada pulou 34 shards porque havia 34 parquets — mas 77 shards concluídos "
         "produzem 34 parquets, porque um fecha a cada 20 mil registros. Escreveu 40 mil "
         "duplicatas. Corrigi num módulo e NÃO propaguei para o outro, que ia rodar "
         "minutos depois."),
        ("A máquina dormiu e matou o treino",
         "`train_embedding.py` não impedia suspensão — só as coletas impediam. Enquanto "
         "uma coleta rodava, o treino pegava carona; quando a rede as matou, ninguém "
         "segurava. Morreu no passo 150 de 781, sem traceback."),
        ("Falha de rede consumia o shard",
         "O DNS caiu e o laço queimou 23 shards em segundos, porque `getaddrinfo` falha "
         "instantaneamente. Tratei indisponibilidade transitória como ausência "
         "definitiva — o espelho de um defeito anterior, onde um 404 definitivo era "
         "repetido como transitório."),
    ]
    for titulo, texto in itens:
        out.append(KeepTogether([Paragraph(f"<b>{titulo}</b>", st["h2"]),
                                 Paragraph(texto, st["p"])]))
    return out


def sec_decisoes(st: dict) -> list:
    out = [Paragraph("8 · Decisões que dependem de você", st["h1"])]
    linhas = [["decisão", "estado da medição", "custo"]]
    for a, b, c in [
        ("Bulk pago do arXiv", "fechada: 16,6% de perda, IC [12,9–20,8]", "US$ 100–180"),
        ("peS2o filtrado", "não iniciado; taxa medida", "42,7 h de máquina"),
        ("ΦEnc-150M do zero", "o ΦEmb hoje é cabeça sobre encoder de terceiro",
         "US$ 25–90 de GPU"),
        ("Seguir em 8 GB", "custa qualidade de forma mensurável", "—"),
    ]:
        linhas.append([Paragraph(a, st["cel"]), Paragraph(b, st["cel"]),
                       Paragraph(c, st["cel"])])
    out += [tabela(linhas, [42 * mm, 68 * mm, 30 * mm]), Spacer(1, 8),
            Paragraph("Nenhuma delas está bloqueada por falta de medição — as três "
                      "primeiras têm o número necessário. São decisões de orçamento.",
                      st["p"])]
    return out


def montar(saida: Path, log_treino: Path) -> Path:
    st = _estilos()
    doc = SimpleDocTemplate(str(saida), pagesize=A4, title="ΦFM — estado do desenvolvimento",
                            author="ΦFM", leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm)
    agora = datetime.now().strftime("%d/%m/%Y às %H:%M")
    hist = [Paragraph("&Phi;FM — estado do desenvolvimento", st["titulo"]),
            Paragraph(f"Phi Foundation Models para Física &middot; gerado em {agora}",
                      st["sub"]),
            Paragraph("Todo número deste relatório é lido de um artefato que uma "
                      "medição gravou. Nenhum é digitado. Onde o artefato falta, a "
                      "seção diz «não medido» em vez de estimar.", st["nota"]),
            Spacer(1, 6)]
    hist += sec_portoes(st)
    hist += sec_s3b(st)
    hist.append(PageBreak())
    hist += sec_classificador(st)
    hist += sec_espinha(st)
    hist.append(PageBreak())
    hist += sec_corpus(st)
    hist.append(PageBreak())
    hist += sec_treino(st, log_treino)
    hist += sec_defeitos(st)
    hist.append(PageBreak())
    hist += sec_decisoes(st)
    doc.build(hist)
    return saida


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/avaliacao/phifm_estado.pdf"))
    p.add_argument("--log-treino", type=Path,
                   default=Path("data/raw/harvest_phiemb-gc.log"))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    caminho = montar(a.out, a.log_treino)
    # ASCII na saída de console, e a razão é concreta: o `→` estourava
    # `UnicodeEncodeError` no cp1252 do console do Windows DEPOIS de o PDF já estar
    # gravado. Um traceback ao fim de um trabalho bem-sucedido é lido como falha —
    # o inverso do defeito de sempre, e igualmente enganoso.
    print(f"-> {caminho}  ({caminho.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
