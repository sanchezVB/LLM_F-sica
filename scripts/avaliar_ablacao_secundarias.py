#!/usr/bin/env python3
"""As secundárias da ablação do §2.3: recuperação de Física e sonda tensorial, PAREADAS.

    .venv-treino\\Scripts\\python.exe scripts\\avaliar_ablacao_secundarias.py \\
        --controle models/phienc-t2a-E --tratado models/phienc-t2eq-E-tratado \\
        --sonda data/processed/avaliacao/t2eq_sonda_tensorial.json \\
        --out data/processed/avaliacao/t2eq_secundarias.json

## O que a regra diz delas

`kaggle/t2eq_tratado.py`: *"Secundárias: recuperação de Física e sonda tensorial,
controle × tratado. Não derrubam a primária; se discordarem, a discordância é o
resultado."* A primária deu o negativo. Este script não mexe no desfecho.

## As medidas, e por que pareadas

- **Recuperação**: o protocolo do G1 — pool sorteado de 2.000 pares de citação,
  `preparar_pool` com `SEMENTE_POOL`, 192 tokens — pelas MESMAS funções de
  `eval/encoders.py` (`avaliar_um`, `comparar_pareado`). Não pelo
  `avaliar_encoders.py`, que acrescenta sempre os modelos do G1 e os concorrentes:
  uma passada longa para uma pergunta de dois braços. Recall@1 pelo McNemar exato;
  nDCG@10 com IC por bootstrap pareado por ITEM.

  ⚠️ MLM cru é recuperador fraco — o SciBERT dá nDCG@10 0,207 contra 0,370 do MiniLM
  treinado para embedding. O número absoluto não se compara com a tabela do G1; o
  que se lê é controle × tratado, onde a fraqueza é comum aos dois.

- **Sonda tensorial**: o artefato de `avaliar_sonda_tensorial.py` já traz o acerto
  por item. Os dois braços responderam os MESMOS 72 itens, então a comparação é a
  binomial exata sobre os itens discordantes (`statistics.proporcao`), e não duas
  taxas soltas. A escala da sonda vale para os dois: 0,5 é o acaso, e o piso de
  superfície é 0 de 72.
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

import polars as pl  # noqa: E402

from phifm.eval.encoders import avaliar_um, comparar_pareado  # noqa: E402
from phifm.eval.statistics.proporcao import binomial_exata_bicaudal  # noqa: E402
from phifm.training.amostragem import SEMENTE_POOL  # noqa: E402


def ndcg_por_item(posicoes: list[int]) -> np.ndarray:
    """nDCG@10 com um relevante, item a item — a mesma conta de `avaliar_carregado`."""
    p = np.asarray(posicoes, dtype=np.float64)
    return np.where(p <= 10, 1.0 / np.log2(p + 1.0), 0.0)


def bootstrap_pareado_itens(a: np.ndarray, b: np.ndarray, semente: int = 17,
                            n_boot: int = 10_000, alfa: float = 0.05) -> dict:
    """`b − a` por item, com IC de `1 − alfa`.

    `alfa` existe para a correção de Bonferroni (o T1h compara 4 bases contra o mesmo
    controle: 0,05/4). O padrão de 0,05 dá exatamente os percentis de antes — 2,5 e
    97,5 —, então nenhum resultado já publicado muda.
    """
    if a.shape != b.shape:
        raise ValueError("os braços não foram medidos nos mesmos itens")
    d = b - a
    rng = np.random.default_rng((semente, 0x1DC6))
    medias = np.array([d[rng.integers(0, d.size, d.size)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(medias, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    return {"diferenca": round(float(d.mean()), 5), "ic95": [round(float(lo), 5),
            round(float(hi), 5)], "cruza_zero": bool(lo <= 0 <= hi), "itens": int(d.size)}


def sonda_pareada(caminho: Path) -> dict:
    d = json.loads(caminho.read_text(encoding="utf-8"))
    m = d["modelos"]
    c, t = m["controle"], m["tratado"]
    ac, at = np.asarray(c["acertos_por_item"]), np.asarray(t["acertos_por_item"])
    if ac.shape != at.shape or c.get("familias_por_item") != t.get("familias_por_item"):
        raise SystemExit("a sonda dos dois braços não é sobre os mesmos itens")
    ganha_c = int(((ac == 1) & (at == 0)).sum())
    ganha_t = int(((at == 1) & (ac == 0)).sum())
    teste = binomial_exata_bicaudal(ganha_c, ganha_t)
    familias = {}
    for f in sorted(set(c["familias_por_item"])):
        sel = np.asarray(c["familias_por_item"]) == f
        gc = int(((ac == 1) & (at == 0) & sel).sum())
        gt = int(((at == 1) & (ac == 0) & sel).sum())
        familias[f] = {"itens": int(sel.sum()), "controle": round(float(ac[sel].mean()), 4),
                       "tratado": round(float(at[sel].mean()), 4),
                       "controle_melhor": gc, "tratado_melhor": gt,
                       "p": round(binomial_exata_bicaudal(gc, gt)["p"], 4)}
    return {"itens": int(ac.size), "acaso": 0.5,
            "piso_de_superficie": d["piso_de_superficie"],
            "taxa_controle": round(float(ac.mean()), 4),
            "taxa_tratado": round(float(at.mean()), 4),
            "controle_melhor": ganha_c, "tratado_melhor": ganha_t,
            "p": round(teste["p"], 4), "por_familia": familias,
            "nota": ("Os p por família são DESCRITIVOS: quatro testes sobre 18 itens "
                     "cada, sem correção, depois de ver as taxas.")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controle", required=True)
    ap.add_argument("--tratado", required=True)
    ap.add_argument("--sonda", type=Path, required=True)
    ap.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--dispositivo", default="dml")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    rc = avaliar_um(a.controle, "controle", val, n=a.n, dispositivo=a.dispositivo,
                    semente=SEMENTE_POOL)
    rt = avaliar_um(a.tratado, "tratado", val, n=a.n, dispositivo=a.dispositivo,
                    semente=SEMENTE_POOL)
    for r in (rc, rt):
        if r.erro:
            raise SystemExit(f"{r.nome}: {r.erro}")
    rec = {
        "protocolo": {"n": a.n, "semente": SEMENTE_POOL, "max_tokens": 192},
        "controle": {"recall_1": round(rc.recall_1, 4), "recall_10": round(rc.recall_10, 4),
                     "mrr": round(rc.mrr, 4), "ndcg_10": round(rc.ndcg_10, 4)},
        "tratado": {"recall_1": round(rt.recall_1, 4), "recall_10": round(rt.recall_10, 4),
                    "mrr": round(rt.mrr, 4), "ndcg_10": round(rt.ndcg_10, 4)},
        "recall_1_mcnemar": comparar_pareado(rc, rt),
        "ndcg_10_bootstrap_por_item": bootstrap_pareado_itens(
            ndcg_por_item(rc.posicoes), ndcg_por_item(rt.posicoes)),
        "nota": ("MLM cru, agregado por média: recuperador fraco nos dois braços. O "
                 "absoluto não se compara com o G1; o que se lê é controle × tratado."),
    }
    resultado = {"secundarias": True,
                 "regra": "não derrubam a primária; se discordarem, a discordância é o resultado",
                 "recuperacao": rec, "sonda_tensorial": sonda_pareada(a.sonda)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    s = resultado["sonda_tensorial"]
    nd = rec["ndcg_10_bootstrap_por_item"]
    mc = rec["recall_1_mcnemar"]
    print()
    print("=" * 78)
    print("  §2.3 · SECUNDÁRIAS · controle × tratado (não derrubam a primária)")
    print("=" * 78)
    print(f"  recuperação · pool {a.n}")
    for braco in ("controle", "tratado"):
        b = rec[braco]
        print(f"    {braco:<9} recall@1 {b['recall_1']:.4f} · recall@10 {b['recall_10']:.4f}"
              f" · MRR {b['mrr']:.4f} · nDCG@10 {b['ndcg_10']:.4f}")
    print(f"    recall@1 pareado: {mc.get('veredito')} (p={mc.get('p')})")
    print(f"    nDCG@10 tratado − controle: {nd['diferenca']:+.5f} {nd['ic95']}")
    print(f"  sonda tensorial · {s['itens']} itens · acaso 0,5 · piso de superfície "
          f"{s['piso_de_superficie'].get('taxa')}")
    print(f"    controle {s['taxa_controle']:.4f} · tratado {s['taxa_tratado']:.4f} · "
          f"discordantes {s['controle_melhor']} × {s['tratado_melhor']} · p={s['p']}")
    for f, v in s["por_familia"].items():
        print(f"      {f:<28} {v['controle']:.3f} → {v['tratado']:.3f}  "
              f"({v['controle_melhor']}×{v['tratado_melhor']}, p={v['p']}, descritivo)")
    print("=" * 78)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
