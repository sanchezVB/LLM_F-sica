#!/usr/bin/env python3
"""T1g: aplica a regra do `kaggle/t1g_gte_1m.py` — o GTE@1M supera o ΦEmb do sistema?

    .venv-treino\\Scripts\\python.exe scripts\\comparar_t1g.py \\
        --gte data/processed/t1g_saida/phiemb-gte-base-1m-melhor \\
        --out data/processed/avaliacao/t1g_comparacao.json

Primária: nDCG@10 no protocolo do G1, `gte − sistema`, IC por bootstrap pareado por
ITEM, os dois medidos NESTA sessão. Diagnóstico: o GTE@1M contra o MiniLM@1,5M, as
duas bases perto do mesmo volume.

A CONTA vem do mesmo lugar dos outros comparadores (`avaliar_um`, `comparar_pareado`,
`bootstrap_pareado_itens`); o que é deste script é a leitura da regra do T1g.

## ⚠️ O custo de serviço vai no artefato, e não é enfeite

O GTE-base leva 4,4× o tempo do MiniLM para embutir o mesmo universo (954 s contra
217 s, T1f). Uma vitória no nDCG não é troca automática de encoder: a margem medida
vai para o dono do projeto junto com esse custo, e a decisão é dele. A regra diz isso
antes do número, e o JSON repete, para que ninguém leia só a primeira linha.
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

SISTEMA_REGISTRADO = 0.6223
MINILM_15M_REGISTRADO = 0.5780
CUSTO_DE_SERVICO = {"gte_base_s": 954, "minilm_s": 217, "razao": 4.4,
                    "fonte": "T1f — o mesmo universo embutido pelos dois"}


def ler_t1g(nd: dict) -> tuple[str, str]:
    """`nd` é `gte − sistema`, pelo bootstrap pareado por item."""
    lo, hi = nd["ic95"]
    if lo > 0:
        return "GTE À FRENTE", (
            "existe um recuperador melhor que o do sistema, a 1/6 do volume de pares. "
            "⚠️ NÃO é troca automática: o GTE custa 4,4× para embutir, para sempre. A "
            "margem vai ao dono do projeto junto com esse custo.")
    if hi < 0:
        return "SISTEMA À FRENTE", (
            "o volume ainda manda a 1 M: o MiniLM a 6 M segura o lugar. O GTE só "
            "voltaria à mesa com muito mais pares.")
    return "EMPATE", (
        "o GTE a 1 M alcança o sistema e não o passa. Com o custo de 4,4×, o sistema "
        "fica como está; o GTE@6M (~39 h de T4) segue aberto, sem sinal que o pague.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gte", required=True, help="o GTE-base ajustado em 1 M de pares")
    ap.add_argument("--sistema", default="models/phiemb-do-sistema",
                    help="o ΦEmb do sistema: MiniLM ajustado em 6 M de pares")
    ap.add_argument("--minilm15", default="models/phiemb-minilm-15m-sorteado-melhor",
                    help="diagnóstico: o MiniLM ajustado em 1,5 M (mesmo sorteio)")
    ap.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--dispositivo", default="dml")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import polars as pl

    from avaliar_ablacao_secundarias import (  # noqa: E402
        bootstrap_pareado_itens,
        ndcg_por_item,
    )
    from phifm.eval.encoders import avaliar_um, comparar_pareado  # noqa: E402
    from phifm.training.amostragem import SEMENTE_POOL  # noqa: E402

    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    medidos = {}
    for nome, caminho in (("sistema", a.sistema), ("gte", a.gte),
                          ("minilm15", a.minilm15)):
        r = avaliar_um(caminho, nome, val, n=a.n, dispositivo=a.dispositivo,
                       semente=SEMENTE_POOL)
        if r.erro:
            raise SystemExit(f"{nome}: {r.erro}")
        medidos[nome] = r

    def por_item(nome):
        return ndcg_por_item(medidos[nome].posicoes)

    def resumo(r):
        return {"caminho": r.caminho, "recall_1": round(r.recall_1, 4),
                "recall_10": round(r.recall_10, 4), "mrr": round(r.mrr, 4),
                "ndcg_10": round(r.ndcg_10, 4)}

    # ⚠️ `bootstrap_pareado_itens(a, b)` devolve `b − a`: aqui, gte − sistema.
    primaria = bootstrap_pareado_itens(por_item("sistema"), por_item("gte"))
    desfecho, leitura = ler_t1g(primaria)
    diag = bootstrap_pareado_itens(por_item("minilm15"), por_item("gte"))

    resultado = {
        "experimento": "t1g — GTE-base@1M contra o ΦEmb do sistema",
        "regra": "kaggle/t1g_gte_1m.py · REGRA",
        "protocolo": {"n": a.n, "semente": SEMENTE_POOL, "max_tokens": 192,
                      "todos_medidos_nesta_sessao": True},
        **{nome: resumo(r) for nome, r in medidos.items()},
        "registrados": {"sistema": SISTEMA_REGISTRADO,
                        "minilm15": MINILM_15M_REGISTRADO},
        "primaria_ndcg_10_gte_menos_sistema": primaria,
        "diagnostico_recall_1_mcnemar": comparar_pareado(medidos["sistema"],
                                                         medidos["gte"]),
        "desfecho": desfecho, "leitura": leitura,
        "diagnostico_gte_menos_minilm15": diag,
        "custo_de_servico": CUSTO_DE_SERVICO,
        "ressalvas": [
            "NÃO é ablação de uma variável: muda base E volume (1 M contra 6 M). É a "
            "pergunta do produto, declarada assim antes do número.",
            "Uma semente por modelo.",
            "O diagnóstico contra o MiniLM@1,5M dá ao MiniLM 1,5× mais pares.",
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print()
    print("=" * 78)
    print(f"  T1g · GTE-base@1M contra o ΦEmb do sistema · pool do G1, n={a.n}")
    print("=" * 78)
    print(f"  {'':<12} {'recall@1':>9} {'recall@10':>10} {'MRR':>7} {'nDCG@10':>9}")
    for nome in medidos:
        r = resultado[nome]
        print(f"  {nome:<12} {r['recall_1']:>9.4f} {r['recall_10']:>10.4f} "
              f"{r['mrr']:>7.4f} {r['ndcg_10']:>9.4f}")
    print(f"\n  PRIMÁRIA · nDCG@10 gte − sistema: "
          f"{primaria['diferenca']:+.5f} {primaria['ic95']}")
    print(f"  recall@1 pareado (diagnóstico): "
          f"{resultado['diagnostico_recall_1_mcnemar'].get('veredito')}")
    print(f"  DESFECHO: {desfecho}\n  {leitura}")
    print(f"\n  diagnóstico · gte − minilm@1,5M: {diag['diferenca']:+.5f} {diag['ic95']}")
    print(f"  custo de serviço: GTE {CUSTO_DE_SERVICO['razao']}× o MiniLM para embutir")
    print("=" * 78)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
