#!/usr/bin/env python3
"""Portão G1 — ΦEmb contra PhysBERT e embedders gerais (DOC-07 §3).

Roda na venv de TREINO (Python 3.12), e em CPU por padrão para não disputar VRAM
com um treino em curso.

    .venv-treino/Scripts/python.exe scripts/avaliar_encoders.py
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polars as pl  # noqa: E402

from phifm.eval.encoders import comparar, tabela, tabela_pareada, veredito  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    p.add_argument("--phiemb", type=Path, default=Path("models/phiemb"),
                   help="checkpoint do ΦEmb; omitido se não existir")
    p.add_argument("--n", type=int, default=2000,
                   help="candidatos na avaliação; 256 não separa margens de ~0,02")
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--lote", type=int, default=16)
    p.add_argument("--dispositivo", default="cpu", choices=["cpu", "dml"])
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    logging.info("validação: %s pares · usando os %d primeiros", f"{val.height:,}", a.n)

    extras = {}
    if (a.phiemb / "config.json").exists():
        extras["ΦEmb (nosso)"] = str(a.phiemb)
    else:
        logging.warning("%s não tem config.json — ΦEmb fora da comparação", a.phiemb)

    rs = comparar(val, extras, n=a.n, max_tokens=a.max_tokens,
                  lote=a.lote, dispositivo=a.dispositivo)

    print("\n" + "=" * 68)
    print(tabela(rs, a.n))
    print("=" * 68)
    print()
    print(tabela_pareada(rs))
    print()
    print("=" * 68)
    print(veredito(rs, a.n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
