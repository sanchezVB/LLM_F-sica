#!/usr/bin/env python3
"""Pares de citação para o ΦEmb (DOC-07 §3.1)."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.training.pairs import construir  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, default=Path("data/raw/openalex_snapshot"))
    p.add_argument("--spine", type=Path, default=Path("data/processed/spine.parquet"))
    p.add_argument("--out", type=Path, default=Path("data/processed/pares"))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s",
                        stream=sys.stdout)
    tr, val = construir(a.snapshot, a.spine, a.out)
    print(f"\ntreino {tr.height:,} · validação {val.height:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
