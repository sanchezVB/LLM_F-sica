#!/usr/bin/env python3
"""O texto da conferência de quantização do G2 (DOC-12 §3.7.5). São 50 itens do UG-DEV,
com enunciado e solução, sorteados pela semente do G2. O arquivo sai em
`models/base/gguf/kl_dev.txt`, fora do git, porque é conteúdo de benchmark (DOC-11 §8.1).

    .venv\\Scripts\\python.exe scripts\\texto_kl_g2.py

Depois, o `llama-perplexity` com `--kl-divergence-base` no F16 e `--kl-divergence` no Q8_0
(ver ESTADO.md). A KL média tem de ficar ≤ 0,01 nats.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval.g2 import SEMENTE_G2  # noqa: E402

SAIDA = RAIZ / "models/base/gguf/kl_dev.txt"


def main() -> int:
    part = json.loads((RAIZ / "data/processed/avaliacao/g2_particao.json")
                      .read_text(encoding="utf-8"))
    dev = sorted(part["ugphysics"]["dev"])
    escolhidos = set(random.Random(SEMENTE_G2).sample(dev, 50))
    blocos = []
    for arq in sorted((RAIZ / "data/raw/benchmarks/ugphysics").glob("*/en.jsonl")):
        for linha in arq.read_text(encoding="utf-8").splitlines():
            d = json.loads(linha)
            if f"{arq.parent.name}/{d['index']}" in escolhidos:
                blocos.append(f"Problem: {d['problem']}\nSolution: {d['solution']}\n")
    texto = "\n".join(blocos)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(texto, encoding="utf-8", newline="\n")
    print(f"{len(blocos)} itens do dev, {len(texto):,} caracteres → {SAIDA.relative_to(RAIZ)} "
          f"· sha256 {hashlib.sha256(texto.encode()).hexdigest()[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
