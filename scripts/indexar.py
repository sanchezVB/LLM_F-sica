#!/usr/bin/env python3
"""Constrói o índice de busca: os artigos de Física do arXiv, embutidos pelo ΦEmb do sistema.

    .venv-treino\\Scripts\\python.exe scripts\\indexar.py
    .venv-treino\\Scripts\\python.exe scripts\\indexar.py --limite 20000 --saida data/processed/indice_ensaio

Cerca de 1 hora na RX 7600 para os ~1,6 milhão de artigos. Interrompido, rodar de novo
retoma de onde parou. Ver `phifm.retrieval.indice` para o que o índice garante.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    p.add_argument("--modelo", type=Path, default=RAIZ / "models/phiemb-do-sistema",
                   help="o ΦEmb do sistema; o índice grava a identidade dele")
    p.add_argument("--saida", type=Path, default=RAIZ / "data/processed/indice_busca")
    p.add_argument("--dispositivo", default="dml")
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--limite", type=int, default=None,
                   help="indexa só os primeiros N documentos (ensaio)")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    from phifm.retrieval.indice import construir

    t0 = time.perf_counter()
    r = construir(a.spine, a.modelo, a.saida, dispositivo=a.dispositivo, lote=a.lote,
                  limite=a.limite)
    print(f"\n✅ índice com {r['n']:,} documentos em {a.saida} · "
          f"{(time.perf_counter() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
