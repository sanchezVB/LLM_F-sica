#!/usr/bin/env python3
"""EXPLORATÓRIO: a recuperação do braço tratado é 8x a do controle. Artefato ou efeito?

    .venv-treino\\Scripts\\python.exe scripts\\diagnosticar_ablacao_recuperacao.py \\
        --modelo controle=models/phienc-t2a-E \\
        --modelo tratado=models/phienc-t2eq-E-tratado \\
        --modelo "A (p_equacao 0)=models/phienc-t2a-A" \\
        --out data/processed/avaliacao/t2eq_diagnostico_recuperacao.json

## ⚠️ Feito DEPOIS de ver o número

A secundária de recuperação deu nDCG@10 0,0171 no controle e 0,1391 no tratado —
dois modelos que diferem só em `p_equacao`. Um fator de 8 entre eles pede que a
explicação barata seja descartada antes da interessante.

A barata: embedding de MLM agregado por média costuma ser **anisotrópico** — os
vetores ocupam um cone estreito, o cosseno entre textos quaisquer fica perto de 1, e o
ranking vira ruído. Se o tratamento só mudou essa geometria, a recuperação sobe sem
que o modelo represente melhor o conteúdo. Este script mede:

- **cosseno médio entre textos DIFERENTES** do pool (1,0 = todos no mesmo lugar);
- **variância no primeiro componente principal** (perto de 1 = uma direção domina);
- **a recuperação depois de CENTRAR os vetores** (subtrair a média e renormalizar),
  que remove a direção comum. Se o controle centrado alcançar o tratado, a diferença
  era geometria; se o tratado continuar à frente, é conteúdo.

E um terceiro braço de referência: o A do T2a, também com `p_equacao` 0,0, outro
tokenizer. Se ele ficar perto do controle, a recuperação baixa é o que o MLM padrão
dá a este tamanho, e não um defeito do checkpoint do controle.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

from phifm import cache_hf_no_hd  # noqa: E402

cache_hf_no_hd()  # ⚠️ ANTES do transformers: o huggingface_hub lê o caminho do cache no import

import polars as pl  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from phifm.eval.encoders import _codificar  # noqa: E402
from phifm.training.amostragem import SEMENTE_POOL, preparar_pool  # noqa: E402
from phifm.training.embedding import escolher_dispositivo  # noqa: E402


def metricas(va: np.ndarray, vp: np.ndarray) -> dict:
    """recall@1, recall@10 e nDCG@10 com um relevante — a conta de `avaliar_carregado`."""
    sim = va @ vp.T
    ordem = np.argsort(-sim, axis=1)
    pos = np.argmax(ordem == np.arange(sim.shape[0])[:, None], axis=1) + 1
    ndcg = np.where(pos <= 10, 1.0 / np.log2(pos + 1.0), 0.0)
    return {"recall_1": round(float((pos == 1).mean()), 4),
            "recall_10": round(float((pos <= 10).mean()), 4),
            "ndcg_10": round(float(ndcg.mean()), 4)}


def geometria(v: np.ndarray, semente: int = 17) -> dict:
    rng = np.random.default_rng(semente)
    i = rng.integers(0, v.shape[0], 20_000)
    j = rng.integers(0, v.shape[0], 20_000)
    ok = i != j
    cos = (v[i[ok]] * v[j[ok]]).sum(1)
    centrado = v - v.mean(0)
    s = np.linalg.svd(centrado, compute_uv=False)
    return {"cosseno_medio_entre_textos_diferentes": round(float(cos.mean()), 4),
            "fracao_de_variancia_no_1o_componente": round(float(s[0] ** 2 / (s ** 2).sum()), 4)}


def centrar(v: np.ndarray, media: np.ndarray) -> np.ndarray:
    c = v - media
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", action="append", required=True, metavar="ROTULO=CAMINHO")
    ap.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--dispositivo", default="dml")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    amostra = preparar_pool(val, a.n, SEMENTE_POOL)
    dev = escolher_dispositivo(a.dispositivo)
    resultado = {"exploratorio": True,
                 "aviso": "feito DEPOIS de ver a secundária; não altera a regra",
                 "protocolo": {"n": a.n, "semente": SEMENTE_POOL, "max_tokens": 192},
                 "modelos": {}}
    for spec in a.modelo:
        rotulo, _, caminho = spec.partition("=")
        tok = AutoTokenizer.from_pretrained(caminho)
        mod = AutoModel.from_pretrained(caminho, attn_implementation="eager").to(dev).eval()
        with torch.no_grad():
            va = _codificar(mod, tok, amostra["ancora"].to_list(), dev, 192, 16).numpy()
            vp = _codificar(mod, tok, amostra["positivo"].to_list(), dev, 192, 16).numpy()
        # A média para centrar sai dos DOIS lados juntos: é a direção comum do espaço,
        # e não uma informação sobre qual positivo casa com qual âncora.
        media = np.vstack([va, vp]).mean(0)
        resultado["modelos"][rotulo] = {
            "caminho": caminho,
            "cru": metricas(va, vp),
            "centrado": metricas(centrar(va, media), centrar(vp, media)),
            "geometria": geometria(np.vstack([va, vp])),
        }
        del mod

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("=" * 86)
    print("  EXPLORATÓRIO · recuperação dos braços do §2.3: geometria ou conteúdo?")
    print("=" * 86)
    print(f"  {'modelo':<18} {'nDCG cru':>9} {'nDCG centrado':>14} {'cos médio':>10} "
          f"{'var 1º comp.':>13}")
    for rotulo, r in resultado["modelos"].items():
        print(f"  {rotulo:<18} {r['cru']['ndcg_10']:>9.4f} {r['centrado']['ndcg_10']:>14.4f} "
              f"{r['geometria']['cosseno_medio_entre_textos_diferentes']:>10.4f} "
              f"{r['geometria']['fracao_de_variancia_no_1o_componente']:>13.4f}")
    print("=" * 86)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
