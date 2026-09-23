#!/usr/bin/env python3
"""T1i: aplica a regra do `colab/t1i_pequenas_1m.py` — as pequenas a 1 M contra o sistema.

    .venv-treino\\Scripts\\python.exe scripts\\comparar_t1i.py \\
        --dir data/processed/t1i_saida \\
        --out data/processed/avaliacao/t1i_comparacao.json

Primária: nDCG@10 no protocolo do G1, cada candidata (`gte-small`, `bge-small`, a 1 M de
pares) contra o ΦEmb do sistema (MiniLM-L6 a 6 M), bootstrap pareado por ITEM, IC de
97,5% (Bonferroni sobre 2). Diagnóstico: contra o GTE-base@1M do T1g, nos mesmos pares.
O custo de embutir, em razão ao sistema, vem pelo mesmo instrumento do T1h.

Todos medidos NESTA sessão. A conta é a do G1; a leitura é a da regra do T1i.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "colab"))

from phifm.core.console import utf8  # noqa: E402

utf8()

ALFA_BONFERRONI = 0.05 / 2
SISTEMA_REGISTRADO = 0.6223
GTE_BASE_1M_REGISTRADO = 0.6211


def ler_candidata(nd: dict, razao_custo: float) -> tuple[str, str]:
    """`nd` é `candidata − sistema`, com IC de 97,5%."""
    lo, hi = nd["ic95"]
    if lo > 0:
        return "ACIMA", (
            f"existe um encoder melhor que o do sistema, a 1/6 do volume de pares e "
            f"{razao_custo:.2f}× o custo de embutir. A troca vai ao dono do projeto com a "
            "margem e o custo medidos.")
    if hi < 0:
        return "ABAIXO", "o volume ainda manda a 1 M: o sistema segura o lugar."
    return "EMPATE", (
        f"alcança o sistema e não o passa; a {razao_custo:.2f}× o custo de embutir, o "
        "sistema fica.")


def main() -> int:
    import t1i_pequenas_1m as t1i
    from comparar_t1h import medir_custo

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True,
                    help="a pasta com as duas `phiemb-t1i-<base>-1m-melhor`")
    ap.add_argument("--sistema", default="models/phiemb-do-sistema")
    ap.add_argument("--gte-base", default="data/processed/t1g_saida/phiemb-gte-base-1m-melhor",
                    help="diagnóstico: o GTE-base@1M do T1g, nos mesmos pares")
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
    from phifm.training.amostragem import SEMENTE_POOL, preparar_pool  # noqa: E402

    caminhos = {"sistema": a.sistema,
                **{n: str(a.dir / f"phiemb-t1i-{n}-1m-melhor") for n in t1i.BASES},
                "gte-base": a.gte_base}
    faltam = [n for n, c in caminhos.items() if not (Path(c) / "config.json").exists()]
    if faltam:
        raise SystemExit(f"faltam checkpoints: {faltam}")

    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    textos_custo = preparar_pool(val, a.n, SEMENTE_POOL)["positivo"].to_list()
    medidos, custos = {}, {}
    for nome, caminho in caminhos.items():
        r = avaliar_um(caminho, nome, val, n=a.n, dispositivo=a.dispositivo,
                       semente=SEMENTE_POOL)
        if r.erro:
            raise SystemExit(f"{nome}: {r.erro}")
        medidos[nome] = r
        custos[nome] = medir_custo(caminho, textos_custo, a.dispositivo)
        print(f"  {nome}: nDCG@10 {r.ndcg_10:.4f} · embutir {len(textos_custo)} textos "
              f"em {custos[nome]:.1f} s", flush=True)

    rs = medidos["sistema"]
    resultado = {
        "experimento": "t1i — gte-small e bge-small a 1 M contra o ΦEmb do sistema",
        "regra": "colab/t1i_pequenas_1m.py · REGRA",
        "protocolo": {"n": a.n, "semente": SEMENTE_POOL, "max_tokens": 192,
                      "todos_medidos_nesta_sessao": True,
                      "ic": f"{100 * (1 - ALFA_BONFERRONI):.1f}% (Bonferroni, 2 comparações)"},
        "registrados": {"sistema": SISTEMA_REGISTRADO, "gte-base@1M": GTE_BASE_1M_REGISTRADO},
        "modelos": {},
    }
    for nome, r in medidos.items():
        e = {"caminho": caminhos[nome], "recall_1": round(r.recall_1, 4),
             "recall_10": round(r.recall_10, 4), "mrr": round(r.mrr, 4),
             "ndcg_10": round(r.ndcg_10, 4), "parametros_m": round(r.parametros_m, 1),
             "custo_s": round(custos[nome], 2),
             "razao_custo": round(custos[nome] / custos["sistema"], 2)}
        if nome in t1i.BASES:
            # ⚠️ `bootstrap_pareado_itens(a, b)` devolve `b − a`: candidata − sistema.
            nd = bootstrap_pareado_itens(ndcg_por_item(rs.posicoes),
                                         ndcg_por_item(r.posicoes), alfa=ALFA_BONFERRONI)
            desfecho, leitura = ler_candidata(nd, e["razao_custo"])
            diag = bootstrap_pareado_itens(ndcg_por_item(medidos["gte-base"].posicoes),
                                           ndcg_por_item(r.posicoes))
            e |= {"ndcg_10_menos_sistema": nd,
                  "diagnostico_recall_1_mcnemar": comparar_pareado(rs, r),
                  "desfecho": desfecho, "leitura": leitura,
                  "diagnostico_menos_gte_base_1m_ic95": diag}
        resultado["modelos"][nome] = e
    resultado["ressalvas"] = [
        "NÃO é ablação de uma variável contra o sistema: muda base E volume (1 M contra "
        "6 M). É a pergunta do produto, declarada assim antes do número.",
        "Uma semente por base.",
        "O IC da primária é de 97,5%; a chave `ic95` guarda esse intervalo pelo nome "
        "histórico do campo. O diagnóstico contra o GTE-base usa 95%.",
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 88)
    print(f"  T1i · as pequenas a 1 M contra o ΦEmb do sistema · pool do G1, n={a.n} · IC 97,5%")
    print("=" * 88)
    print(f"  {'':<10} {'params':>7} {'nDCG@10':>8} {'r@1':>7} {'custo':>7}   contra o sistema")
    for nome, e in resultado["modelos"].items():
        dif = ("(o sistema)" if nome == "sistema" else "(diagnóstico)" if nome == "gte-base"
               else f"{e['ndcg_10_menos_sistema']['diferenca']:+.4f} "
                    f"{e['ndcg_10_menos_sistema']['ic95']} · {e['desfecho']}")
        print(f"  {nome:<10} {e['parametros_m']:>6.1f}M {e['ndcg_10']:>8.4f} "
              f"{e['recall_1']:>7.4f} {e['razao_custo']:>6.2f}×   {dif}")
    for nome in t1i.BASES:
        e = resultado["modelos"][nome]
        print(f"\n  {nome}: {e['desfecho']} — {e['leitura']}")
        dg = e["diagnostico_menos_gte_base_1m_ic95"]
        print(f"    diagnóstico · {nome} − GTE-base@1M: {dg['diferenca']:+.4f} {dg['ic95']}")
    print("=" * 88)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
