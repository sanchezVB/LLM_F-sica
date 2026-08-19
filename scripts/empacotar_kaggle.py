#!/usr/bin/env python3
"""Empacota o mínimo para o T1a rodar no Kaggle.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/empacotar_kaggle.py

Produz `data/processed/kaggle_t1a/`, para subir como Kaggle Dataset.

## Por que 211 MB e não 2,68 GB

O `pares_treino.parquet` inteiro tem 6,56 M de arestas e 2,68 GB. O T1a usa
**400 mil pares** — o volume do campeão do G1.1, escolhido por medição: o treino de
1,5 M empatou estatisticamente com ele (p=0,950) e o de 511 negativos também
(p=0,636). Subir o conjunto inteiro seria pagar 13× de banda por dados que a
medição diz não comprar nada.

Medido: 168,2 MB de treino + 43,2 MB de validação = **211,5 MB**.

## O que vai, e por que cada coisa

| arquivo | para quê |
|---|---|
| `pares_treino.parquet` | as 400 mil arestas de citação |
| `pares_validacao.parquet` | as 133.540 de validação — a métrica sai daqui |
| `phifm_src.zip` | o pacote `phifm`, para o notebook não reimplementar o treino |
| `MANIFESTO.json` | hashes BLAKE3 e proveniência, para o que roda lá ser o que está aqui |

⚠️ O código vai como ZIP e não copiado no notebook. Um notebook que reimplementa o
laço de treino é um segundo laço para manter em sincronia, e a divergência entre os
dois seria invisível: os dois rodariam, com resultados diferentes, e nada apontaria
qual está certo. O manifesto existe para provar que o que rodou lá é este código.

## O que NÃO vai

Os modelos. O `all-MiniLM-L6-v2` é baixado do HuggingFace no próprio Kaggle, que
tem rede — subir 90 MB de pesos públicos seria desperdício. E o campeão não vai
porque o T1a treina de novo a partir da base; comparar contra o campeão é trabalho
do avaliador do G1, que roda aqui.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    git_sha_curto,
    hash_arquivo,
)

log = logging.getLogger("empacotar")

# Tudo do pacote, menos o que não roda lá nem faz sentido carregar.
EXCLUIR_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def _zipar_fonte(raiz: Path, destino: Path) -> int:
    n = 0
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((raiz / "src" / "phifm").rglob("*.py")):
            if any(p in EXCLUIR_DIRS for p in f.parts):
                continue
            z.write(f, f.relative_to(raiz / "src").as_posix())
            n += 1
        # O script de treino também, porque é o ponto de entrada de verdade —
        # e assim o notebook chama exatamente o que roda aqui.
        for nome in ("train_embedding.py",):
            z.write(raiz / "scripts" / nome, f"scripts/{nome}")
            n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    p.add_argument("--out", type=Path, default=Path("data/processed/kaggle_t1a"))
    p.add_argument("--max-pares", type=int, default=400_000,
                   help="volume do campeão do G1.1; mais que isso a medição diz "
                        "que não compra nada (p=0,950)")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s",
                        stream=sys.stdout)

    raiz = Path(__file__).resolve().parents[1]
    a.out.mkdir(parents=True, exist_ok=True)

    tr = pl.scan_parquet(a.pares / "pares_treino.parquet").head(a.max_pares).collect()
    tr.write_parquet(a.out / "pares_treino.parquet", compression="zstd")
    shutil.copy2(a.pares / "pares_validacao.parquet", a.out / "pares_validacao.parquet")
    n_py = _zipar_fonte(raiz, a.out / "phifm_src.zip")

    arquivos = sorted(f for f in a.out.iterdir() if f.is_file()
                      and f.name != "MANIFESTO.json")
    manifesto = {
        "git_sha": git_sha_curto(),
        "max_pares": a.max_pares,
        "linhas_treino": tr.height,
        "modulos_python": n_py,
        "hash_algo": "blake3",
        "arquivos": {f.name: {"blake3": hash_arquivo(f), "bytes": f.stat().st_size}
                     for f in arquivos},
        "nota": ("Subir como Kaggle Dataset. O notebook confere estes hashes antes "
                 "de treinar: rodar sobre dados que não são estes produziria um "
                 "número incomparável com os medidos aqui."),
    }
    (a.out / "MANIFESTO.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(v["bytes"] for v in manifesto["arquivos"].values())
    print()
    print("=" * 68)
    for nome, v in manifesto["arquivos"].items():
        print(f"  {nome:28s} {v['bytes']/1e6:8.1f} MB  {v['blake3'][:12]}…")
    print(f"  {'TOTAL':28s} {total/1e6:8.1f} MB")
    print("=" * 68)
    print(f"  -> {a.out}")
    print(f"  git {manifesto['git_sha']} · {n_py} módulos · {tr.height:,} pares")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
