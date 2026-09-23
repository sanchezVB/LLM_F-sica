#!/usr/bin/env python3
"""T1h: aplica a regra do `colab/t1h_bases_pequenas.py` — as cinco bases pequenas.

    .venv-treino\\Scripts\\python.exe scripts\\comparar_t1h.py \\
        --dir data/processed/t1h_saida \\
        --out data/processed/avaliacao/t1h_comparacao.json

Primária: nDCG@10 no protocolo do G1, cada candidata contra o MiniLM-L6@200k treinado na
mesma sessão do Colab, bootstrap pareado por ITEM, com IC de 98,75% (Bonferroni sobre 4
comparações). Os cinco medidos NESTA sessão.

Secundária, e é ela que decide o produto: o CUSTO de embutir, em razão ao MiniLM-L6.

## ⚠️ Por que o custo é medido aqui, e não tirado do avaliador

`avaliar_um` devolve `segundos`, e eles incluem carregar o modelo e o aquecimento da
GPU — numa base pequena isso é boa parte do total, e a razão sairia achatada. Aqui cada
modelo embute os MESMOS 2.000 textos DEPOIS de um aquecimento, no mesmo dispositivo, com
o mesmo lote e a mesma truncagem. O `.cpu()` de cada lote em `_codificar` sincroniza a
GPU, então o relógio mede o trabalho, e não o enfileiramento.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))
sys.path.insert(0, str(RAIZ / "colab"))

from phifm.core.console import utf8  # noqa: E402

utf8()

ALFA_BONFERRONI = 0.05 / 4
TETO_DE_CUSTO = 2.5


def ler_candidata(nd: dict, razao_custo: float) -> tuple[str, str, bool]:
    """`nd` é `candidata − minilm-l6`, com IC de 98,75%. Devolve (desfecho, leitura, sobe)."""
    lo, hi = nd["ic95"]
    if lo > 0:
        desfecho = "ACIMA"
        leitura = "a base pesa também no porte pequeno."
    elif hi < 0:
        desfecho = "ABAIXO"
        leitura = "o MiniLM-L6 é a melhor base pequena nesta receita."
    else:
        desfecho = "EMPATE"
        leitura = "no porte pequeno a base não se separa do MiniLM-L6."
    sobe = desfecho == "ACIMA" and razao_custo <= TETO_DE_CUSTO
    if desfecho == "ACIMA":
        leitura += (f" Custo {razao_custo:.2f}× o MiniLM-L6: "
                    + ("CANDIDATA a subir de volume (1 M de pares) contra o sistema."
                       if sobe else
                       f"acima do teto de {TETO_DE_CUSTO}×, não sobe."))
    return desfecho, leitura, sobe


def medir_custo(caminho: str, textos: list[str], dispositivo: str,
                max_tokens: int = 192, lote: int = 16) -> float:
    """Segundos para embutir `textos` com o modelo JÁ aquecido."""
    from transformers import AutoModel, AutoTokenizer

    from phifm.eval.encoders import _codificar
    from phifm.training.embedding import escolher_dispositivo

    dev = escolher_dispositivo(dispositivo)
    tok = AutoTokenizer.from_pretrained(caminho)
    mod = AutoModel.from_pretrained(caminho, attn_implementation="eager").to(dev).eval()
    _codificar(mod, tok, textos[:64], dev, max_tokens, lote)  # aquecimento
    t0 = time.perf_counter()
    _codificar(mod, tok, textos, dev, max_tokens, lote)
    return time.perf_counter() - t0


def main() -> int:
    import t1h_bases_pequenas as t1h

    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True,
                    help="a pasta com as cinco `phiemb-t1h-<base>-200k-melhor`")
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

    caminhos = {nome: a.dir / f"phiemb-t1h-{nome}-200k-melhor" for nome in t1h.BASES}
    faltam = [n for n, c in caminhos.items() if not (c / "config.json").exists()]
    if faltam:
        raise SystemExit(f"faltam checkpoints em {a.dir}: {faltam}")

    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    textos_custo = preparar_pool(val, a.n, SEMENTE_POOL)["positivo"].to_list()
    medidos, custos = {}, {}
    for nome, caminho in caminhos.items():
        r = avaliar_um(str(caminho), nome, val, n=a.n, dispositivo=a.dispositivo,
                       semente=SEMENTE_POOL)
        if r.erro:
            raise SystemExit(f"{nome}: {r.erro}")
        medidos[nome] = r
        custos[nome] = medir_custo(str(caminho), textos_custo, a.dispositivo)
        print(f"  {nome}: nDCG@10 {r.ndcg_10:.4f} · embutir {len(textos_custo)} textos "
              f"em {custos[nome]:.1f} s", flush=True)

    ref = t1h.REFERENCIA
    rr = medidos[ref]
    resultado = {
        "experimento": "t1h — bases pequenas no lugar do MiniLM-L6",
        "regra": "colab/t1h_bases_pequenas.py · REGRA",
        "protocolo": {"n": a.n, "semente": SEMENTE_POOL, "max_tokens": 192,
                      "pares_de_treino": 200_000, "todos_medidos_nesta_sessao": True,
                      "ic": f"{100 * (1 - ALFA_BONFERRONI):.2f}% (Bonferroni, 4 comparações)",
                      "custo": f"{len(textos_custo)} textos, lote 16, após aquecimento, "
                               f"dispositivo {a.dispositivo}"},
        "bases": {},
    }
    for nome, r in medidos.items():
        entrada = {"base": t1h.BASES[nome], "caminho": str(caminhos[nome]),
                   "recall_1": round(r.recall_1, 4), "recall_10": round(r.recall_10, 4),
                   "mrr": round(r.mrr, 4), "ndcg_10": round(r.ndcg_10, 4),
                   "parametros_m": round(r.parametros_m, 1),
                   "custo_s": round(custos[nome], 2),
                   "razao_custo": round(custos[nome] / custos[ref], 2)}
        if nome != ref:
            # ⚠️ `bootstrap_pareado_itens(a, b)` devolve `b − a`: candidata − referência.
            nd = bootstrap_pareado_itens(ndcg_por_item(rr.posicoes),
                                         ndcg_por_item(r.posicoes), alfa=ALFA_BONFERRONI)
            desfecho, leitura, sobe = ler_candidata(nd, entrada["razao_custo"])
            entrada |= {"ndcg_10_menos_referencia": nd,
                        "diagnostico_recall_1_mcnemar": comparar_pareado(rr, r),
                        "desfecho": desfecho, "leitura": leitura,
                        "candidata_a_subir_de_volume": sobe}
        resultado["bases"][nome] = entrada
    resultado["ressalvas"] = [
        "Uma semente por base; 200 mil pares, metade da receita do T1a.",
        "O e5-small e o bge-small foram feitos para usar prefixo, e aqui não usam.",
        "O IC da tabela é de 98,75%; a chave `ic95` do bootstrap guarda esse intervalo "
        "pelo nome histórico do campo.",
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 86)
    print(f"  T1h · bases pequenas contra o MiniLM-L6@200k · pool do G1, n={a.n} · IC 98,75%")
    print("=" * 86)
    print(f"  {'':<11} {'params':>7} {'nDCG@10':>8} {'r@1':>7} {'custo':>7}   diferença [IC]")
    for nome, e in resultado["bases"].items():
        dif = ("(referência)" if nome == ref else
               f"{e['ndcg_10_menos_referencia']['diferenca']:+.4f} "
               f"{e['ndcg_10_menos_referencia']['ic95']} · {e['desfecho']}")
        print(f"  {nome:<11} {e['parametros_m']:>6.1f}M {e['ndcg_10']:>8.4f} "
              f"{e['recall_1']:>7.4f} {e['razao_custo']:>6.2f}×   {dif}")
    sobem = [n for n, e in resultado["bases"].items() if e.get("candidata_a_subir_de_volume")]
    print(f"\n  candidatas a subir de volume (ACIMA e custo ≤ {TETO_DE_CUSTO}×): "
          f"{sobem or 'nenhuma'}")
    print("=" * 86)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
