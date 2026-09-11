#!/usr/bin/env python3
"""ONDE estão os 37% que não chegam ao top-100 — antes de gastar GPU atrás deles.

    .venv-treino\\Scripts\\python.exe scripts\\diagnosticar_teto.py

O `recall@100` do recuperador é **0,6325**, então 37% das consultas nunca recebem o
alvo no conjunto de candidatos, e nenhuma melhora de reordenação as alcança. O T1d
fechou dizendo que o trabalho seguinte é no recuperador. Esta é a medição que decide
*qual* trabalho.

## A pergunta, e por que ela vem ANTES de treinar qualquer coisa

37% de perda pode ser duas coisas muito diferentes:

- **lacuna de modelo** — o alvo está na posição 150, 300, 800: perto. Um recuperador
  melhor o traz para dentro, e o retorno de mais treino é grande;
- **teto da tarefa** — o alvo está na posição 40.000 de 88.807, indistinguível do
  acaso. Nada no texto da âncora aponta para ele, e nenhum recuperador de
  similaridade textual vai achá-lo. Treinar mais não move isso.

A distribuição do posto verdadeiro separa as duas em minutos, e a diferença entre
elas vale semanas de T4.

## ⚠️ E o controle que decide: o BM25 concorda?

Um alvo que o denso põe em 40.000 **e** o BM25 põe em 40.000 é um alvo sem sinal
textual — a citação existe por um motivo que o resumo não expressa. Dois métodos com
vieses completamente diferentes falhando juntos é a evidência mais forte disponível
sem julgamento humano.

Já um alvo que o denso perde e o BM25 acha é lacuna do denso, e é recuperável.

## O protocolo é o do T1b, de propósito

Mesmo universo (citados da validação), mesmas 2.000 consultas com a mesma semente.
Números medidos noutro pool não seriam comparáveis com o `recall@100` que motivou
esta investigação.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from phifm.core.console import utf8 as console_utf8  # noqa: E402
from phifm.core.modelos import RECUPERADOR  # noqa: E402
from phifm.eval.hibrido import BM25  # noqa: E402
from phifm.training.embedding import (  # noqa: E402
    escolher_dispositivo,
    media_mascarada,
)

# ⚠️ No IMPORT, e não dentro do `main()`: o argparse imprime `--help` antes
# de qualquer código nosso, e `Φ` não existe em cp1252. Ver `phifm.core.console`.
console_utf8()

log = logging.getLogger("teto")

# As faixas do relatório. A primeira é o que o T1b usa como profundidade, e as
# seguintes respondem "quão longe está o que ficou de fora".
FAIXAS = ((1, "1"), (10, "2–10"), (100, "11–100"), (1_000, "101–1.000"),
          (10_000, "1.001–10.000"), (10**9, "> 10.000"))


def _codificar(mod, tok, textos: list[str], dev, max_tokens: int, lote: int,
               rotulo: str) -> np.ndarray:
    saidas, t0 = [], time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(textos), lote):
            b = tok(textos[i:i + lote], padding="max_length", truncation=True,
                    max_length=max_tokens, return_tensors="pt")
            b = {k: v.to(dev) for k, v in b.items()}
            h = mod(**b).last_hidden_state
            v = F.normalize(media_mascarada(h, b["attention_mask"]).float(), dim=-1)
            saidas.append(v.cpu().numpy())
            if i and (i // lote) % 200 == 0:
                taxa = (i + lote) / (time.perf_counter() - t0)
                log.info("  %s: %s/%s · %.0f/s · faltam %.0f min", rotulo,
                         f"{i+lote:,}", f"{len(textos):,}", taxa,
                         (len(textos) - i) / taxa / 60)
    return np.vstack(saidas)


def _postos(escores: np.ndarray, alvo: int) -> int:
    """Posto do alvo, 0-based — a convenção do `avaliar_t1b.py`.

    ⚠️ Contando quantos o superam, e não por `argsort`: ordenar 88.807 escores
    2.000 vezes custa minutos, e a contagem custa uma comparação vetorizada. O
    resultado é o mesmo posto, com empate resolvido a favor do alvo (otimista) —
    e a nota do artefato diz isso.
    """
    return int((escores > escores[alvo]).sum())


def histograma(postos: list[int]) -> list[dict]:
    n, saida, anterior = len(postos), [], 0
    arr = np.array(postos)
    for limite, rotulo in FAIXAS:
        # `posto` é 0-based, então "top-k" é `posto < k`.
        quantos = int(((arr < limite) & (arr >= anterior)).sum())
        saida.append({"faixa": rotulo, "consultas": quantos,
                      "fracao": round(quantos / n, 4)})
        anterior = limite
    return saida


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    # ⚠️ `str` e não `Path`: aceita id do Hub além de caminho local.
    #
    # `Path("thenlper/gte-base")` vira `WindowsPath('thenlper/gte-base')` e o
    # `str()` dele sai com barra invertida — o `from_pretrained` não reconhece e
    # tenta abrir um diretório que não existe. Medir uma base de fora contra a
    # nossa é justamente o que este diagnóstico serve para fazer.
    p.add_argument("--emb", default=RECUPERADOR,
                   help="caminho local ou id do HuggingFace")
    p.add_argument("--n-consultas", type=int, default=2000)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--semente", type=int, default=17)
    p.add_argument("--dispositivo", default="auto")
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/avaliacao/teto_do_recuperador.json"))
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")

    # ⚠️ MESMO universo e MESMAS consultas do `avaliar_t1b.py`. Um pool diferente
    # daria números que não conversam com o recall@100 que motivou isto.
    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    pool = val.unique(subset=["arxiv_citado"], maintain_order=True)
    ids_pool = pool["arxiv_citado"].to_list()
    textos_pool = pool["positivo"].to_list()
    indice_de = {d: i for i, d in enumerate(ids_pool)}

    linhas = val.sample(n=min(a.n_consultas, len(val)), seed=a.semente)
    consultas = linhas["ancora"].to_list()
    alvos = [indice_de[d] for d in linhas["arxiv_citado"].to_list()]
    log.info("universo: %s · consultas: %s", f"{len(ids_pool):,}",
             f"{len(consultas):,}")

    t_inicio = time.perf_counter()
    dev = escolher_dispositivo(a.dispositivo)
    tok = AutoTokenizer.from_pretrained(a.emb)
    mod = AutoModel.from_pretrained(a.emb).to(dev).eval()
    V = _codificar(mod, tok, textos_pool, dev, a.max_tokens, a.lote, "universo")
    Vq = _codificar(mod, tok, consultas, dev, a.max_tokens, a.lote, "consultas")
    del mod
    Vt = V.T

    log.info("indexando BM25 sobre o mesmo universo…")
    bm = BM25().indexar(textos_pool)

    postos_denso, postos_bm = [], []
    t0 = time.perf_counter()
    for i, alvo in enumerate(alvos):
        postos_denso.append(_postos(Vq[i] @ Vt, alvo))
        postos_bm.append(_postos(bm.pontuar(consultas[i]), alvo))
        if i and i % 200 == 0:
            taxa = (i + 1) / (time.perf_counter() - t0)
            log.info("  %s/%s · %.1f/s", f"{i+1:,}", f"{len(alvos):,}", taxa)

    d = np.array(postos_denso)
    b = np.array(postos_bm)

    # ⚠️ A conta que decide. Entre as consultas que o DENSO perde no top-100:
    # quantas o BM25 acha? Essas são lacuna do denso, recuperáveis. Quantas os
    # DOIS perdem fundo? Essas são candidatas a teto da tarefa.
    perdidas = d >= 100
    n_perdidas = int(perdidas.sum())
    bm_acha_100 = int((perdidas & (b < 100)).sum())
    bm_acha_1000 = int((perdidas & (b < 1000)).sum())
    ambos_fundo = int((perdidas & (b >= 10_000) & (d >= 10_000)).sum())

    resultado = {
        "recuperador": str(a.emb).replace("\\", "/"),
        "max_tokens": a.max_tokens,
        "universo": len(ids_pool),
        "n_consultas": len(consultas),
        "semente": a.semente,
        "custo_segundos": round(time.perf_counter() - t_inicio, 1),
        "recall": {f"em_{k}": round(float((d < k).mean()), 4)
                   for k in (1, 10, 100, 1000, 10_000)},
        "posto_do_alvo": {
            "denso": histograma(postos_denso),
            "bm25": histograma(postos_bm),
        },
        "mediana_do_posto": {"denso": int(np.median(d)), "bm25": int(np.median(b))},
        "entre_as_perdidas_no_top100": {
            "consultas": n_perdidas,
            "fracao_do_total": round(n_perdidas / len(alvos), 4),
            "bm25_acha_no_top_100": bm_acha_100,
            "bm25_acha_no_top_1000": bm_acha_1000,
            "os_dois_alem_de_10000": ambos_fundo,
            "mediana_do_posto_denso": int(np.median(d[perdidas])),
            "mediana_do_posto_bm25": int(np.median(b[perdidas])),
        },
        "como_ler": (
            "Posto 0-based do alvo entre os documentos do universo. Empate é "
            "resolvido A FAVOR do alvo (conta-se quantos o SUPERAM), então os "
            "postos são otimistas — o recall daqui pode ficar acima do medido "
            "pelo `avaliar_t1b.py`, que usa argsort. O que a tabela responde: "
            "as consultas perdidas estão PERTO (lacuna de modelo, recuperável "
            "com mais treino) ou FUNDO (teto da tarefa, e nenhum recuperador de "
            "similaridade textual as alcança)? O BM25 é o controle: um alvo que "
            "os DOIS põem além de 10.000 é um alvo sem sinal textual, porque "
            "dois métodos de viés oposto falharam juntos."),
        # ⚠️ A comparação de ORÇAMENTO IGUAL, que é a única honesta.
        #
        # "denso@100 + bm25@100" custa ao reranqueador até 200 candidatos. Comparar
        # isso com "denso@100" faz a fusão parecer de graça. O controle é
        # `denso@200`: o mesmo orçamento, só com o denso.
        #
        # Medido em 2026-09-10: o denso sozinho VENCE em todas as profundidades, e
        # por margem larga. O que o BM25 acrescenta é sempre ~metade do que os
        # mesmos candidatos extras do denso dariam.
        "orcamento_igual": [
            {"composicao": f"denso@{k} + bm25@100",
             "candidatos": f"<={k + 100}",
             "recall": round(float(((d < k) | (b < 100)).mean()), 4),
             "controle_denso_sozinho": f"denso@{k + 100}",
             "recall_do_controle": round(float((d < k + 100).mean()), 4)}
            for k in (100, 200, 500, 1000)],
        "recall_por_profundidade_do_denso": {
            f"em_{k}": round(float((d < k).mean()), 4)
            for k in (10, 50, 100, 200, 500, 1000, 2000)},
        "postos_denso": postos_denso,
        "postos_bm25": postos_bm,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print()
    print("=" * 74)
    print(f"  ONDE ESTÁ O ALVO · {len(consultas):,} consultas · universo "
          f"{len(ids_pool):,}")
    print("=" * 74)
    print(f"  {'faixa do posto':<16} {'ΦEmb':>16} {'BM25':>16}")
    for hd, hb in zip(resultado["posto_do_alvo"]["denso"],
                      resultado["posto_do_alvo"]["bm25"], strict=True):
        print(f"  {hd['faixa']:<16} {hd['consultas']:>6} ({hd['fracao']:>6.1%})"
              f" {hb['consultas']:>8} ({hb['fracao']:>6.1%})")
    print(f"  {'mediana':<16} {resultado['mediana_do_posto']['denso']:>16,}"
          f" {resultado['mediana_do_posto']['bm25']:>16,}")
    print()
    e = resultado["entre_as_perdidas_no_top100"]
    print(f"  DAS {e['consultas']} PERDIDAS no top-100 ({e['fracao_do_total']:.1%} "
          "do total):")
    print(f"    o BM25 acha {e['bm25_acha_no_top_100']} no top-100 "
          f"({e['bm25_acha_no_top_100']/max(e['consultas'],1):.1%}) "
          "-> lacuna do denso, recuperável")
    print(f"    o BM25 acha {e['bm25_acha_no_top_1000']} no top-1.000 "
          f"({e['bm25_acha_no_top_1000']/max(e['consultas'],1):.1%})")
    print(f"    os DOIS além de 10.000: {e['os_dois_alem_de_10000']} "
          f"({e['os_dois_alem_de_10000']/max(e['consultas'],1):.1%}) "
          "-> candidatas a TETO DA TAREFA")
    print(f"    mediana do posto: ΦEmb {e['mediana_do_posto_denso']:,} · "
          f"BM25 {e['mediana_do_posto_bm25']:,}")
    print("=" * 74)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
