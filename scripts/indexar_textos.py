#!/usr/bin/env python3
"""Regrava o texto completo em grupos pequenos, para o assistente ler um artigo por vez.

    .venv-treino\\Scripts\\python.exe scripts\\indexar_textos.py

Lê `data/processed/redpajama_fisica/` (835 mil artigos, 44 arquivos de um grupo só) e grava
`data/processed/textos_por_artigo/`: a mesma tabela em grupos de 32 linhas e o índice de
onde está cada artigo. ~12 GB no HD, ~20 min, retomável por arquivo, fora do git.
Ver `phifm.retrieval.textos`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()


def main() -> int:
    from phifm.retrieval.textos import POR_GRUPO, Textos, construir

    p = argparse.ArgumentParser()
    p.add_argument("--origem", type=Path, default=RAIZ / "data/processed/redpajama_fisica")
    p.add_argument("--destino", type=Path, default=RAIZ / "data/processed/textos_por_artigo")
    p.add_argument("--por-grupo", type=int, default=POR_GRUPO)
    a = p.parse_args()
    t0 = time.perf_counter()

    def ao_gravar(nome: str, n: int, total: int) -> None:
        print(f"{nome}: {n:,} artigos · {total:,} no total · {time.perf_counter() - t0:.0f} s",
              flush=True)

    man = construir(a.origem, a.destino, a.por_grupo, ao_gravar)
    textos = Textos(a.destino)
    primeiro = next(iter(textos._onde))
    t1 = time.perf_counter()
    n = len(textos.ler(primeiro) or "")
    print(f"{man['artigos']:,} artigos em {a.destino} · ler um ({n:,} caracteres): "
          f"{(time.perf_counter() - t1) * 1000:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
