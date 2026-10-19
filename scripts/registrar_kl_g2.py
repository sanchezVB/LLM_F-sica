#!/usr/bin/env python3
"""Registra a conferência de quantização do G2 (DOC-12 §3.7.5): a KL do Q8_0 contra o F16
tem de ficar ≤ 0,01 nats. A fonte é o log do `llama-perplexity --kl-divergence`.

    .venv\\Scripts\\python.exe scripts\\registrar_kl_g2.py --tamanho 1.7B \\
        --log models\\base\\gguf\\kl_1.7b.log \\
        --q8 models\\base\\gguf\\Qwen3-1.7B-Base-Q8_0.gguf \\
        --f16 models\\base\\gguf\\Qwen3-1.7B-Base-F16.gguf

Grava `data/processed/avaliacao/g2_kl_<tamanho>.json`, versionado. Ele traz os números, o
SHA-256 dos dois GGUF e as versões que converteram. Isso tudo entra no hash da configuração.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
KL_MAX = 0.01


def _sha256(arq: Path) -> str:
    h = hashlib.sha256()
    with arq.open("rb") as f:
        while bloco := f.read(1 << 24):
            h.update(bloco)
    return h.hexdigest()


def _valor(log: str, rotulo: str) -> tuple[float, float | None]:
    m = re.search(rf"{rotulo}\s*:\s*(-?[\d.]+)(?:\s*\S+\s*([\d.]+))?", log)
    if not m:
        raise SystemExit(f"não achei '{rotulo}' no log")
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else None)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tamanho", required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--q8", type=Path, required=True)
    p.add_argument("--f16", type=Path, required=True)
    p.add_argument("--conversao", default="convert_hf_to_gguf.py da tag b11159 (6b790a9c2); "
                   "torch 2.4.1+cpu, transformers 5.0.0, numpy 2.5.1, sentencepiece 0.2.2")
    a = p.parse_args()
    log = a.log.read_text(encoding="utf-8", errors="replace")
    kld, kld_ep = _valor(log, r"Mean\s+KLD")
    mesmo_topo, _ = _valor(log, r"Same top p")
    razao_ppl, _ = _valor(log, r"Mean PPL\(Q\)/PPL\(base\)")
    r = {
        "o_que_e": "conferência de quantização do G2 (DOC-12 §3.7.5): KL média do Q8_0 contra "
                   "o F16 em 50 itens do UG-dev (scripts/texto_kl_g2.py)",
        "tamanho": a.tamanho,
        "kl_media_nats": kld, "kl_erro_padrao": kld_ep, "kl_max_regra": KL_MAX,
        "passa": kld <= KL_MAX,
        "mesma_ficha_mais_provavel_pct": mesmo_topo, "razao_ppl_q8_f16": razao_ppl,
        "kl_quantis": {q: _valor(log, rf"{re.escape(q)}\s+KLD")[0]
                       for q in ("Median", "95.0%", "99.0%", "Maximum")},
        "gguf": {"q8_0": {"arquivo": a.q8.name, "bytes": a.q8.stat().st_size,
                          "sha256": _sha256(a.q8)},
                 "f16": {"arquivo": a.f16.name, "bytes": a.f16.stat().st_size,
                         "sha256": _sha256(a.f16)}},
        "conversao": a.conversao,
    }
    saida = RAIZ / f"data/processed/avaliacao/g2_kl_{a.tamanho}.json"
    saida.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"KL média {kld:.5f} nats (≤ {KL_MAX}: {'PASSA' if r['passa'] else 'NÃO PASSA'}); "
          f"mesma ficha no topo {mesmo_topo}%")
    print(f"Q8_0 sha256 {r['gguf']['q8_0']['sha256']}")
    print(f"→ {saida.relative_to(RAIZ)}")
    return 0 if r["passa"] else 1


if __name__ == "__main__":
    sys.exit(main())
