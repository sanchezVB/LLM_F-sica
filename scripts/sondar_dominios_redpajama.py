#!/usr/bin/env python3
"""Quanto LaTeX `math` e `cs` trariam ao corpus, medido numa amostra de shards.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/sondar_dominios_redpajama.py \\
        --shards 5 --out data/processed/avaliacao/sonda_dominios.json

## A pergunta

O corpus tem 27,75 B tokens, e o DOC-07 §2 pede 15–30 B — volume não é o gargalo.
**Integridade matemática é**: o peS2o (14,60 B dos 27,75 B) tem ambiente de equação
em **0,0%** dos documentos contra **84,9%** do RedPajama-arXiv. De LaTeX íntegro há
~10,5 B tokens.

A fatia de Física do RedPajama já está coletada (828.601 documentos distintos). Os
de `math` e `cs` passaram pelo mesmo download e foram **descartados pelo filtro** —
eles têm LaTeX íntegro pelo mesmo mecanismo. Esta sonda mede quantos são, para a
decisão de recoletar 81 GB sair de medição em vez de suposição.

## ⚠️ Os shards são SORTEADOS, e isso não é preciosismo

Medido neste projeto: as 8 primeiras partes do corpus têm **32,4% de matemática
contra 42,7% das demais**. A ordem dos shards carrega viés de composição, então
`urls[:n]` mediria a fatia errada — e para uma sonda que existe para extrapolar,
enviesar a amostra é enviesar a conclusão.

O sorteio usa `--semente` e a lista escolhida vai no artefato.

## ⚠️ Nada de texto é guardado

A sonda lê em fluxo, conta e descarta. Ela responde "quantos e quão grandes", não
"quais" — guardar o texto seria refazer a coleta, que é justamente o que ela existe
para decidir.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import requests

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8 as console_utf8  # noqa: E402
from phifm.core.env import contato_obrigatorio  # noqa: E402
from phifm.corpus.acquire.base import user_agent  # noqa: E402
from phifm.corpus.slices.redpajama import (  # noqa: E402
    REVISAO,
    _linhas_do_shard,
    _urls,
    ids_do_spine,
)

console_utf8()
log = logging.getLogger("sonda-dominios")

# Os domínios cujos metadados estão no disco, em `data/raw/arxiv_negativos/`.
DOMINIOS = ("math", "cs", "stat", "q_bio", "econ")


def ids_do_dominio(raiz: Path, dominio: str) -> set[str]:
    """Os `arxiv_id` de um domínio, dos metadados já coletados.

    ⚠️ Estes parquets foram coletados como NEGATIVOS para o classificador, então
    podem ser amostra e não o domínio inteiro. O artefato registra a contagem para
    que a extrapolação saiba o que tem debaixo dela: um id que não está aqui cai em
    `desconhecido`, e `desconhecido` é piso de erro, não ausência de sinal.
    """
    fs = sorted((raiz / "data/raw/arxiv_negativos" / dominio).rglob("*.parquet"))
    if not fs:
        log.warning("%s: nenhum parquet de metadados", dominio)
        return set()
    ids = set(pl.scan_parquet(fs).select("arxiv_id").collect()["arxiv_id"])
    log.info("%s: %s identificadores", dominio, f"{len(ids):,}")
    return ids


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=int, default=5,
                   help="quantos dos 100 sortear; cada um é ~0,81 GB de rede")
    p.add_argument("--spine", type=Path,
                   default=Path("data/processed/spine.parquet"))
    p.add_argument("--semente", type=int, default=17,
                   help="sorteio dos shards. Ver o § sobre viés de ordem")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    fisica = ids_do_spine(a.spine)
    por_dominio = {d: ids_do_dominio(RAIZ, d) for d in DOMINIOS}
    # ⚠️ Um id que está na Física E num negativo conta como Física: o filtro atual
    # já o guardou, então ele não é ganho. Sem esta subtração o ganho sairia
    # inflado pelo que já está no disco.
    for d, ids in por_dominio.items():
        antes = len(ids)
        por_dominio[d] = ids - fisica
        if antes != len(por_dominio[d]):
            log.info("%s: %s ids também estão na Física e saem da conta",
                     d, f"{antes - len(por_dominio[d]):,}")

    sessao = requests.Session()
    sessao.headers["User-Agent"] = user_agent(contato_obrigatorio())
    urls = _urls(sessao)
    rng = np.random.default_rng(a.semente)
    # ⚠️ `int(...)` e não o `np.int64` que o `permutation` devolve.
    #
    # Medido em 2026-09-12: a sonda leu os 5 shards, contou tudo, e morreu em
    # `TypeError: Object of type int64 is not JSON serializable` **na gravação do
    # artefato** — depois de 4 GB de rede e 2 min. Os números ficaram só no log.
    #
    # É a mesma família do "guarda depois da escrita não é guarda": um defeito no
    # ÚLTIMO passo espera o trabalho caro terminar para aparecer. O teste que
    # acompanha este script serializa o artefato com tipos de numpy dentro.
    escolhidos = [int(x) for x in sorted(rng.permutation(len(urls))[:a.shards] + 1)]
    log.info("%d shards no índice · sorteando %d: %s (semente %d)",
             len(urls), len(escolhidos), escolhidos, a.semente)

    n_docs = Counter()
    n_chars = Counter()
    t0 = time.perf_counter()
    from phifm.corpus.slices.redpajama import Progresso

    prog = Progresso()
    for i, shard in enumerate(escolhidos, 1):
        url = urls[shard - 1]
        log.info("shard %d (%d/%d)", shard, i, len(escolhidos))
        for linha in _linhas_do_shard(sessao, url, prog):
            # ⚠️ `_linhas_do_shard` devolve BYTES de linha, não dicts — ele existe
            # para não materializar o shard, e decodificar é do chamador. O
            # coletor faz o mesmo `json.loads` com o mesmo `continue` silencioso:
            # um registro corrompido não pode derrubar a passada.
            prog.registros_vistos += 1
            try:
                registro = json.loads(linha)
            except json.JSONDecodeError:
                n_docs["json_invalido"] += 1
                continue
            meta = registro.get("meta") or {}
            aid = meta.get("arxiv_id") or ""
            texto = registro.get("text") or ""
            if aid in fisica:
                classe = "fisica_ja_no_corpus"
            else:
                classe = "desconhecido"
                for d, ids in por_dominio.items():
                    if aid in ids:
                        classe = d
                        break
            n_docs[classe] += 1
            n_chars[classe] += len(texto)
        log.info("  acumulado: %s", dict(n_docs.most_common(4)))

    dt = time.perf_counter() - t0
    total_docs = sum(n_docs.values())
    fator = len(urls) / len(escolhidos)

    # ⚠️ A extrapolação é LINEAR nos shards e a amostra é sorteada, então ela vale
    # como estimativa central. O erro de amostragem NÃO está estimado aqui — com 5
    # de 100 shards ele não é desprezível, e o artefato diz isso em vez de deixar
    # o número parecer exato.
    ganho_docs = n_docs["math"] + n_docs["cs"]
    ganho_chars = n_chars["math"] + n_chars["cs"]
    artefato = {
        "protocolo": {
            "shards_no_indice": len(urls), "shards_sorteados": escolhidos,
            "semente": a.semente, "revisao_do_indice": REVISAO,
            "segundos": round(dt, 1),
            "gb_lidos": round(prog.bytes_lidos / 1e9, 2),
            "nota_do_sorteio": (
                "sorteados, não os primeiros: as 8 primeiras partes do corpus "
                "têm 32,4% de matemática contra 42,7% das demais, então a ordem "
                "carrega viés de composição"),
        },
        "metadados_disponiveis": {
            "fisica_na_tabela_mestra": len(fisica),
            **{d: len(ids) for d, ids in por_dominio.items()},
            "nota": ("os negativos foram coletados para o classificador e podem "
                     "ser amostra do domínio; id fora deles cai em "
                     "`desconhecido`, que é piso de erro"),
        },
        "na_amostra": {
            "documentos": dict(n_docs), "caracteres": dict(n_chars),
            "total_documentos": total_docs,
        },
        "extrapolado_para_100_shards": {
            "fator": round(fator, 2),
            "documentos": {k: int(v * fator) for k, v in n_docs.items()},
            "math_mais_cs_documentos": int(ganho_docs * fator),
            "math_mais_cs_caracteres": int(ganho_chars * fator),
            "math_mais_cs_tokens_a_4_chars": int(ganho_chars * fator / 4),
            "fisica_conferencia": int(n_docs["fisica_ja_no_corpus"] * fator),
        },
        "como_ler": (
            "⚠️ `fisica_conferencia` é a régua desta sonda: ela tem de bater com "
            "os 828.601 documentos distintos que já estão em disco. Se não bater, "
            "a extrapolação está errada e o ganho estimado também. E o erro de "
            "AMOSTRAGEM não está calculado: com poucos shards de 100 ele não é "
            "desprezível, então o número é ordem de grandeza, não medida."),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(artefato, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print()
    print("=" * 74)
    print(f"  SONDA DE DOMÍNIOS · {len(escolhidos)} de {len(urls)} shards · "
          f"{prog.bytes_lidos/1e9:.2f} GB · {dt/60:.1f} min")
    print("=" * 74)
    print(f"  {'classe':26} {'docs':>9} {'%':>7} {'G chars':>10}")
    for k, v in n_docs.most_common():
        print(f"  {k:26} {v:>9,} {100*v/max(total_docs,1):>6.1f}% "
              f"{n_chars[k]/1e9:>10.3f}")
    e = artefato["extrapolado_para_100_shards"]
    print()
    print(f"  extrapolando ×{fator:.0f} para os 100 shards:")
    print(f"    Física (conferência)   {e['fisica_conferencia']:>10,} docs  "
          "(em disco: 828.601)")
    print(f"    math + cs              {e['math_mais_cs_documentos']:>10,} docs")
    print(f"    math + cs              {e['math_mais_cs_tokens_a_4_chars']/1e9:>10.2f} B tokens "
          "(a ~4 chars/token)")
    print(f"    sobre os 10,54 B de LaTeX íntegro: "
          f"{100*e['math_mais_cs_tokens_a_4_chars']/10.54e9:+.0f}%")
    print("=" * 74)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
