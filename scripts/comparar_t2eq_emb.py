#!/usr/bin/env python3
"""ADR-0003, opção C: aplica a regra do `kaggle/t2eq_emb.py` aos dois braços ajustados.

    .venv-treino\\Scripts\\python.exe scripts\\comparar_t2eq_emb.py \\
        --controle data/processed/t2eq_emb_controle/phiemb-t2eq-controle-200k-melhor \\
        --tratado  data/processed/t2eq_emb_tratado/phiemb-t2eq-tratado-200k-melhor \\
        --out data/processed/avaliacao/t2eq_emb_comparacao.json

A regra está na célula, escrita antes de os braços existirem. Primária: nDCG@10 no
protocolo do G1 (pool de 2.000 desduplicado, `SEMENTE_POOL`, 192 tokens), os DOIS
medidos nesta sessão, IC por bootstrap pareado por ITEM. Recall@1 pelo McNemar exato
vai junto, como diagnóstico.

Usa as mesmas funções de medida do G1 (`avaliar_um`, `comparar_pareado`) e o mesmo
bootstrap por item das secundárias do §2.3 (`avaliar_ablacao_secundarias.py`), para
não haver uma terceira conta de nDCG no repositório.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402

utf8()

import polars as pl  # noqa: E402

from avaliar_ablacao_secundarias import bootstrap_pareado_itens, ndcg_por_item  # noqa: E402
from phifm.eval.encoders import avaliar_um, comparar_pareado  # noqa: E402
from phifm.training.amostragem import SEMENTE_POOL  # noqa: E402


def ler_pela_regra(nd: dict) -> tuple[str, str]:
    lo, hi = nd["ic95"]
    if lo > 0:
        return "TRATADO À FRENTE", (
            "o ganho de recuperação do §2.3 sobrevive ao ajuste contrastivo. ADR-0003: "
            "seguir com o pré-treino continuado do ModernBERT-base com p_equacao 0,6.")
    if hi < 0:
        return "CONTROLE À FRENTE", (
            "o tratamento atrapalha a base de embedding. ADR-0003: pausar o ΦEnc.")
    return "EMPATE", (
        "o ajuste apaga a diferença a 200 mil pares. ADR-0003: o ganho do §2.3 não "
        "chega ao sistema nesta escala, e o ΦEnc pausa. ⚠️ A 400 mil ou 6 M a ordem "
        "poderia mudar.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controle", required=True)
    ap.add_argument("--tratado", required=True)
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
    nd = bootstrap_pareado_itens(ndcg_por_item(rc.posicoes), ndcg_por_item(rt.posicoes))
    desfecho, leitura = ler_pela_regra(nd)

    def resumo(r):
        return {"caminho": r.caminho, "recall_1": round(r.recall_1, 4),
                "recall_10": round(r.recall_10, 4), "mrr": round(r.mrr, 4),
                "ndcg_10": round(r.ndcg_10, 4)}

    resultado = {
        "experimento": "t2eq_emb — ADR-0003, opção C",
        "regra": "kaggle/t2eq_emb.py · REGRA",
        "protocolo": {"n": a.n, "semente": SEMENTE_POOL, "max_tokens": 192,
                      "pares_de_treino": 200_000},
        "controle": resumo(rc), "tratado": resumo(rt),
        "primaria_ndcg_10_bootstrap_por_item": nd,
        "diagnostico_recall_1_mcnemar": comparar_pareado(rc, rt),
        "desfecho": desfecho, "leitura": leitura,
        "antes_do_ajuste": {"ndcg_10_cru": {"controle": 0.0171, "tratado": 0.1391},
                            "fonte": "data/processed/avaliacao/t2eq_secundarias.json"},
        "ressalvas": [
            "Bases de 48 M a 0,6 B tokens, uma semente por braço.",
            "200 mil pares, metade da receita do T1a; o absoluto NÃO se compara com o "
            "MiniLM@400k nem com o GTE-base@400k do T1f.",
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    mc = resultado["diagnostico_recall_1_mcnemar"]
    print()
    print("=" * 78)
    print("  T2eq-emb · o ganho do §2.3 sobrevive ao ajuste? · pool do G1, n=2.000")
    print("=" * 78)
    print(f"  {'':<10} {'recall@1':>9} {'recall@10':>10} {'MRR':>7} {'nDCG@10':>9}   "
          f"(nDCG cru antes do ajuste)")
    for nome, r, cru in (("controle", resultado["controle"], 0.0171),
                         ("tratado", resultado["tratado"], 0.1391)):
        print(f"  {nome:<10} {r['recall_1']:>9.4f} {r['recall_10']:>10.4f} "
              f"{r['mrr']:>7.4f} {r['ndcg_10']:>9.4f}   ({cru:.4f})")
    print(f"  nDCG@10 tratado − controle: {nd['diferenca']:+.5f} {nd['ic95']}")
    print(f"  recall@1 pareado (diagnóstico): {mc.get('veredito')}")
    print()
    print(f"  DESFECHO: {desfecho}")
    print(f"  {leitura}")
    print("=" * 78)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
