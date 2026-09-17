#!/usr/bin/env python3
"""As secundárias do T2a (tokenizer A × E): recuperação de Física e sonda tensorial, PAREADAS.

    .venv-treino\\Scripts\\python.exe scripts\\avaliar_t2a_secundarias.py \\
        --a models/phienc-t2a-A --e models/phienc-t2a-E \\
        --sonda data/processed/avaliacao/t2a_sonda_tensorial.json \\
        --out data/processed/avaliacao/t2a_secundarias.json

## O que a regra do T2a diz delas

`kaggle/t2a_tokenizer.py`, escrita antes: as duas são "agnósticas ao tokenizer por
construção" (cosseno entre textos) e "não podem derrubar a primária, e se contrariarem
isso é o resultado a reportar". A primária — bits por byte pela PLL-word-l2r — deu E à
frente por 0,047 bit/byte (DOC-05 §11.2-medido). Rodadas em 2026-09-17; nunca tinham
rodado.

As contas são as mesmas das secundárias do §2.3 (`avaliar_ablacao_secundarias.py`):
recuperação pelo protocolo do G1 com `avaliar_um` e `comparar_pareado`, nDCG@10 com IC
por bootstrap pareado por item, e a sonda pela binomial exata sobre itens discordantes.

⚠️ MLM cru é recuperador fraco nos dois braços, e o número absoluto não se compara com a
tabela do G1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402

utf8()

import polars as pl  # noqa: E402

from avaliar_ablacao_secundarias import bootstrap_pareado_itens, ndcg_por_item  # noqa: E402
from phifm.eval.encoders import avaliar_um, comparar_pareado  # noqa: E402
from phifm.eval.statistics.proporcao import binomial_exata_bicaudal  # noqa: E402
from phifm.training.amostragem import SEMENTE_POOL  # noqa: E402


def sonda_pareada(caminho: Path) -> dict:
    d = json.loads(caminho.read_text(encoding="utf-8"))
    ma, me = d["modelos"]["A"], d["modelos"]["E"]
    aa, ae = np.asarray(ma["acertos_por_item"]), np.asarray(me["acertos_por_item"])
    if aa.shape != ae.shape or ma.get("familias_por_item") != me.get("familias_por_item"):
        raise SystemExit("a sonda dos dois braços não é sobre os mesmos itens")
    ganha_a = int(((aa == 1) & (ae == 0)).sum())
    ganha_e = int(((ae == 1) & (aa == 0)).sum())
    return {"itens": int(aa.size), "acaso": 0.5,
            "piso_de_superficie": d["piso_de_superficie"],
            "taxa_A": round(float(aa.mean()), 4), "taxa_E": round(float(ae.mean()), 4),
            "A_melhor": ganha_a, "E_melhor": ganha_e,
            "p": round(binomial_exata_bicaudal(ganha_a, ganha_e)["p"], 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--e", required=True)
    ap.add_argument("--sonda", type=Path, required=True)
    ap.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--dispositivo", default="dml")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    ra = avaliar_um(a.a, "A", val, n=a.n, dispositivo=a.dispositivo, semente=SEMENTE_POOL)
    re_ = avaliar_um(a.e, "E", val, n=a.n, dispositivo=a.dispositivo, semente=SEMENTE_POOL)
    for r in (ra, re_):
        if r.erro:
            raise SystemExit(f"{r.nome}: {r.erro}")

    def resumo(r):
        return {"recall_1": round(r.recall_1, 4), "recall_10": round(r.recall_10, 4),
                "mrr": round(r.mrr, 4), "ndcg_10": round(r.ndcg_10, 4)}

    nd = bootstrap_pareado_itens(ndcg_por_item(ra.posicoes), ndcg_por_item(re_.posicoes))
    resultado = {
        "secundarias": True,
        "regra": ("kaggle/t2a_tokenizer.py: não derrubam a primária (E à frente em bits por "
                  "byte); se contrariarem, é o resultado a reportar"),
        "recuperacao": {"protocolo": {"n": a.n, "semente": SEMENTE_POOL, "max_tokens": 192},
                        "A": resumo(ra), "E": resumo(re_),
                        "ndcg_10_E_menos_A_bootstrap_por_item": nd,
                        "recall_1_mcnemar": comparar_pareado(ra, re_)},
        "sonda_tensorial": sonda_pareada(a.sonda),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    rec, s = resultado["recuperacao"], resultado["sonda_tensorial"]
    print()
    print("=" * 78)
    print("  T2a · SECUNDÁRIAS · A × E (não derrubam a primária: E à frente em bits/byte)")
    print("=" * 78)
    for braco in ("A", "E"):
        b = rec[braco]
        print(f"  recuperação {braco}: recall@1 {b['recall_1']:.4f} · recall@10 "
              f"{b['recall_10']:.4f} · nDCG@10 {b['ndcg_10']:.4f}")
    print(f"  nDCG@10 E − A: {nd['diferenca']:+.5f} {nd['ic95']} · recall@1: "
          f"{rec['recall_1_mcnemar'].get('veredito')}")
    print(f"  sonda: A {s['taxa_A']:.4f} · E {s['taxa_E']:.4f} · discordantes "
          f"{s['A_melhor']} × {s['E_melhor']} · p={s['p']}")
    print("=" * 78)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
