#!/usr/bin/env python3
"""Tria JSONL train/dev contra referências explícitas; não baixa nem abre dados por padrão."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phifm.corpus.auditoria_cpt import auditar_corpus


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--treino", type=Path, required=True, help="JSONL id/text/partition/source")
    p.add_argument("--dev", type=Path, required=True)
    p.add_argument("--referencias", type=Path, required=True,
                   help="banco JSONL explicitamente autorizado: id/bench/text[/answer]")
    p.add_argument("--saida", type=Path, required=True, help="pasta NOVA no HD")
    a = p.parse_args()
    try:
        m = auditar_corpus(a.treino, a.dev, a.referencias, a.saida)
    except (OSError, ValueError) as exc:
        p.exit(1, f"Auditoria recusada: {exc}\n")
    print(json.dumps({"manifesto": str(a.saida / "manifesto.json"), "scope": m["scope"],
                      "outputs": m["outputs"], "g2_verificado": False},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
