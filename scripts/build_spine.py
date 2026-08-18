#!/usr/bin/env python3
"""Sprint S1 · consolidação — espinha de metadados pronta para uso."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.corpus.normalize.spine import build, report  # noqa: E402
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    Entrada,
    gravar_manifesto_etapa,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arxiv", type=Path, default=Path("data/raw/arxiv_metadata"))
    p.add_argument("--openalex", type=Path, default=Path("data/raw/openalex_works"))
    p.add_argument("--out", type=Path, default=Path("data/processed/spine.parquet"))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    df = build(a.arxiv, a.openalex, a.out)
    print("\n" + "=" * 66)
    print(report(df))
    print("=" * 66)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
