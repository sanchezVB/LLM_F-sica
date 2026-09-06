#!/usr/bin/env python3
"""Sorteia a amostra de revisão do peS2o — uniforme dentro de cada estrato.

    .venv\\Scripts\\python.exe scripts\\amostrar_para_revisao.py

## ⚠️ Por que este script existe: a amostra anterior cobria 0,67% do corpus

O filtro montava a amostra de passagem, em `hf_filtrado.filtrar`:

    if len(f.amostra) < N_AMOSTRA and f.vistos % 97 == 0:

Ele **parava de amostrar** assim que juntava 400. Medido nos parquets prontos: os 400
documentos de `_amostra_para_revisao.json` estão todos em `part-00000` e
`part-00001`, e o último ocupa a posição **37.087 de 5.526.331 aceitos — 0,67%**.

E o corpus não é homogêneo ao longo dessa ordem. Ele tem dois regimes, e os resumos
vêm primeiro (`part-00000` a `part-00180`):

    s2ag  (título + resumo)   3.626.168 docs (65,6%)    1.280 B/doc
    s2orc (texto pleno)       1.900.163 docs (34,4%)   28.454 B/doc

Por documento os resumos são a maioria; **por token são 7,9% do corpus**. A amostra
antiga é 100% resumo: ela mede a precisão do filtro nos 7,9%, e nada nos 92,1%.

Isso inverte a justificativa escrita em `apurar_revisao.py`. Os 1,5–13,6% de
referência foram medidos em RESUMOS do arXiv, e o argumento para remedir era que o
corpus agora é texto pleno — outra distribuição. A amostra que ia testar isso era de
resumos.

É a **sexta** vez que este repositório tropeça em `head()` sobre dado ordenado:

    1. o peS2o amostrado no começo (esta)
    2. o `val.head(500)` que eram 35 documentos
    3. o `pares_treino` cortado por posição, 49,6% de vazamento
    4. as 8 primeiras de 44 partes do RedPajama, 57% menos equação em display
    5. o `head(200)` sobre uma LISTA de parquets, dentro do próprio conferidor

## O que este script faz em vez disso

Sorteio uniforme **sem reposição dentro de cada estrato**, sobre os parquets prontos
no disco — não sobre a ordem em que o filtro os viu. `--por-estrato 200` dá 200
resumos e 200 textos plenos, e `apurar_revisao.py` combina os dois com o peso em
TOKENS, que é o que degrada o pré-treino.

Os estratos entram **embaralhados** na folha: julgar 200 resumos e depois 200 papers
confundiria o efeito do estrato com o cansaço de quem julga.

## O que ele NÃO conserta

Nada sobre *recall* — quanta Física o filtro jogou fora. Isso exige uma amostra dos
REJEITADOS, que não foram gravados. É outra medição, e mais cara.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

CORPUS = Path("data/processed/pes2o_fisica")
# O rótulo do estrato vem da coluna `url`, que guarda o subconjunto de origem
# (`s2ag/train`, `s2orc/valid`, ...). Não é uma URL: o nome veio do esquema
# genérico de `_texto_e_url`, que aceita `url`, `metadata` ou `source`.
ESTRATOS = {"s2ag": "resumo", "s2orc": "texto pleno"}
CHARS_INICIO = 1000
CHARS_MEIO = 600
# Abaixo disto o documento cabe quase inteiro no início, e um segundo trecho não
# acrescenta nada ao julgamento.
MIN_PARA_MEIO = 3000


def estrato_de(url: str | None) -> str:
    """⚠️ Levanta em vez de cair num padrão.

    Um subconjunto novo do peS2o entrando calado no estrato errado desloca o peso
    em tokens, e o número final sairia com a cara certa.
    """
    for chave, rotulo in ESTRATOS.items():
        if url and url.startswith(chave):
            return rotulo
    raise ValueError(
        f"subconjunto não reconhecido: {url!r}. Os estratos conhecidos são "
        f"{sorted(ESTRATOS)}; um novo tem de entrar em ESTRATOS com o seu peso, "
        "não ser adivinhado.")


def bytes_de_texto(arq: Path) -> int:
    """Bytes UTF-8 da coluna `texto`, dos metadados do parquet.

    Serve de peso do estrato sem ler os 18 GB. Não é caractere nem token: a
    diferença para caractere é o texto não-ASCII, e o manifesto registra qual foi.
    """
    m = pq.ParquetFile(arq).metadata
    b = 0
    for rg in range(m.num_row_groups):
        for c in range(m.num_columns):
            cc = m.row_group(rg).column(c)
            if cc.path_in_schema == "texto":
                b += cc.total_uncompressed_size
    return b - 4 * m.num_rows  # o prefixo de comprimento de cada BYTE_ARRAY


def registro(estrato: str, arq: Path, linha: int, texto: str,
             score: float, url: str) -> dict:
    n = len(texto)
    reg = {
        "score": round(float(score), 4),
        "url": url,
        "estrato": estrato,
        "n_chars": n,
        "arquivo": arq.name,
        "linha": linha,
        "inicio": texto[:CHARS_INICIO],
    }
    # Num paper de 28 mil caracteres o início é a capa. Um trecho do miolo é onde
    # a contaminação aparece: um artigo de outra área tem uma introdução que soa
    # técnica e um corpo que não deixa dúvida.
    if n >= MIN_PARA_MEIO:
        i = int(n * 0.45)
        reg["meio"] = texto[i:i + CHARS_MEIO]
    return reg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=CORPUS)
    p.add_argument("--por-estrato", type=int, default=200)
    p.add_argument("--semente", type=int, default=17)
    p.add_argument("--out", type=Path, default=None,
                   help="por omissão, `_amostra_para_revisao.json` no corpus")
    p.add_argument("--manifesto", type=Path, default=Path(
        "data/processed/avaliacao/revisao_pes2o_amostragem.json"))
    a = p.parse_args()
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")

    arquivos = sorted(a.corpus.glob("part-*.parquet"))
    if not arquivos:
        raise SystemExit(f"nenhum part-*.parquet em {a.corpus}")

    # ⚠️ Os diretórios de saída ANTES da varredura, não depois. Este script lê os
    # 18 GB do corpus; morrer no `write_text` da última linha por causa de uma
    # pasta inexistente joga fora minutos de leitura de disco.
    saida = a.out or a.corpus / "_amostra_para_revisao.json"
    for destino in (saida, a.manifesto):
        destino.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. quantos por estrato, e o peso em bytes ────────────────────────────
    # Só a coluna `url`: ler os 18 GB de texto para sortear 400 linhas trocaria
    # minutos por nada.
    conta: list[Counter] = []
    bytes_arq: list[int] = []
    for arq in arquivos:
        urls = pl.read_parquet(arq, columns=["url"])["url"].to_list()
        conta.append(Counter(estrato_de(u) for u in urls))
        bytes_arq.append(bytes_de_texto(arq))

    total: Counter = Counter()
    for c in conta:
        total.update(c)
    n_docs = sum(total.values())
    # Peso: arquivo puro entra inteiro; arquivo misto — só a fronteira — entra
    # rateado pelo número de linhas, que é o melhor que os metadados dão.
    peso: Counter = Counter()
    for c, b in zip(conta, bytes_arq, strict=True):
        n = sum(c.values())
        for est, k in c.items():
            peso[est] += b * k // n
    total_bytes = sum(peso.values())

    # ── 2. o sorteio ─────────────────────────────────────────────────────────
    rng = random.Random(a.semente)
    escolhidos: dict[str, list[int]] = {}
    for est in sorted(total):
        k = min(a.por_estrato, total[est])
        if k < a.por_estrato:
            print(f"⚠️ estrato {est!r} só tem {total[est]} documentos; sorteando {k}")
        escolhidos[est] = sorted(rng.sample(range(total[est]), k))

    # ── 3. buscar as linhas sorteadas ────────────────────────────────────────
    ponteiro = dict.fromkeys(escolhidos, 0)
    base = dict.fromkeys(escolhidos, 0)
    coletado: list[dict] = []
    for arq, c in zip(arquivos, conta, strict=True):
        querer: dict[str, list[int]] = {}
        for est, idx in escolhidos.items():
            fim = base[est] + c.get(est, 0)
            k = ponteiro[est]
            locais = []
            while k < len(idx) and idx[k] < fim:
                locais.append(idx[k] - base[est])
                k += 1
            ponteiro[est] = k
            base[est] = fim
            if locais:
                querer[est] = locais
        if not querer:
            continue
        df = pl.read_parquet(arq)
        # posição de cada documento DENTRO do seu estrato, neste arquivo
        por_est: dict[str, list[int]] = defaultdict(list)
        for j, u in enumerate(df["url"].to_list()):
            por_est[estrato_de(u)].append(j)
        textos = df["texto"].to_list()
        escores = df["score"].to_list()
        urls = df["url"].to_list()
        for est, locais in querer.items():
            for local in locais:
                j = por_est[est][local]
                coletado.append(registro(est, arq, j, textos[j], escores[j], urls[j]))

    esperado = sum(len(v) for v in escolhidos.values())
    if len(coletado) != esperado:
        raise SystemExit(
            f"sorteou {esperado} e coletou {len(coletado)} — o mapa de índice "
            "global para (arquivo, linha) está errado, e a amostra não seria o "
            "que o manifesto diz que é")

    # ⚠️ Embaralha: 200 resumos seguidos de 200 papers confundiria o estrato com
    # o cansaço de quem julga.
    rng.shuffle(coletado)

    saida.write_text(json.dumps(coletado, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    a.manifesto.write_text(json.dumps({
        "corpus": str(a.corpus).replace("\\", "/"),
        "arquivos": len(arquivos),
        "documentos": n_docs,
        "estratos": {est: {
            "documentos": total[est],
            "fracao_de_documentos": round(total[est] / n_docs, 4),
            "bytes_de_texto": peso[est],
            "peso_em_tokens": round(peso[est] / total_bytes, 4),
            "sorteados": len(escolhidos[est]),
        } for est in sorted(total)},
        "semente": a.semente,
        "por_estrato": a.por_estrato,
        "chars_inicio": CHARS_INICIO,
        "chars_meio": CHARS_MEIO,
        "amostra_anterior": {
            "como_era": "f.vistos % 97 == 0, parando ao juntar 400",
            "cobertura": "part-00000 e part-00001; ultima posicao 37.087 de "
                         "5.526.331 aceitos = 0,67%",
            "estratos": "100% resumo, 0% texto pleno",
        },
        "nota": ("Sorteio uniforme sem reposição DENTRO de cada estrato. O peso em "
                 "tokens é aproximado por bytes UTF-8 da coluna `texto`, dos "
                 "metadados do parquet; a diferença para caracteres é o texto "
                 "não-ASCII. Isto mede PRECISÃO do filtro, não recall."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 74)
    print(f"  {n_docs:,} documentos em {len(arquivos)} parquets")
    for est in sorted(total):
        print(f"    {est:12s} {total[est]:>10,} docs "
              f"({100 * total[est] / n_docs:5.1f}% dos docs, "
              f"{100 * peso[est] / total_bytes:5.1f}% dos tokens) "
              f"-> {len(escolhidos[est])} sorteados")
    print(f"  semente {a.semente} · {len(coletado)} documentos embaralhados")
    print(f"  -> {saida}")
    print(f"  -> {a.manifesto}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
