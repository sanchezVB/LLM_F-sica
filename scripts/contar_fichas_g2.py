#!/usr/bin/env python3
"""Conta as fichas (tokenizador da Qwen3, o mesmo da base e do ΦGen) das soluções de DEV
que podem virar exemplo few-shot do G2: UG-dev e os `_sol` do SciBench. Roda na venv de
treino, a única com `tokenizers`:

    .venv-treino\\Scripts\\python.exe scripts\\contar_fichas_g2.py

Grava `data/processed/avaliacao/g2_fichas_dev.json` ({id: fichas da solução}), só ids e
contagens. `scripts/exemplos_g2.py` lê esse arquivo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tokenizers import Tokenizer

RAIZ = Path(__file__).resolve().parents[1]
BENCH = RAIZ / "data/raw/benchmarks"
SAIDA = RAIZ / "data/processed/avaliacao/g2_fichas_dev.json"
TOKENIZADOR = RAIZ / "models/base/Qwen3-1.7B-Base/tokenizer.json"


def main() -> int:
    tok = Tokenizer.from_file(str(TOKENIZADOR))
    dev = set(json.loads((RAIZ / "data/processed/avaliacao/g2_particao.json")
                         .read_text(encoding="utf-8"))["ugphysics"]["dev"])
    fichas: dict[str, int] = {}
    for arq in sorted((BENCH / "ugphysics").glob("*/en.jsonl")):
        for linha in arq.read_text(encoding="utf-8").splitlines():
            d = json.loads(linha)
            i = f"{arq.parent.name}/{d['index']}"
            if i in dev:
                fichas[i] = len(tok.encode(d.get("solution") or "").ids)
    for livro in ("class", "fund", "thermo"):
        for x in json.loads((BENCH / f"scibench/{livro}_sol.json").read_text(encoding="utf-8")):
            fichas[f"{livro}/{x['problemid']}"] = len(tok.encode(x.get("solution") or "").ids)
    SAIDA.write_text(json.dumps({"tokenizador": "Qwen3 (tokenizer.json c0382117…)",
                                 "fichas_da_solucao": dict(sorted(fichas.items()))},
                                ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(fichas)} soluções de dev contadas → {SAIDA.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
