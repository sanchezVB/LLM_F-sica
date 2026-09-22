#!/usr/bin/env python3
"""A SECUNDÁRIA do caminho B: os dois braços do CPT, ajustados, contra a barra.

    .venv-treino\\Scripts\\python.exe scripts\\comparar_t2eq_cpt_emb.py \\
        --controle models/phiemb-cpt-controle-200k-melhor \\
        --tratado  models/phiemb-cpt-tratado-200k-melhor \\
        --barra    models/phiemb-t2eq-modernbert-200k-melhor \\
        --out data/processed/avaliacao/t2eq_cpt_emb_comparacao.json

A regra está em `colab/t2eq_cpt_emb.py`, escrita antes de os dois ajustes existirem.

## Por que um script separado do `comparar_t2eq_emb.py`

Aquele aplica a regra do §2.3 aos braços de 48 M, e o artefato dele carrega fatos
daquele experimento: a escala (0,6 B), o nDCG cru antes do ajuste (0,0171 e 0,1391) e
as ressalvas de tamanho de base. Reaproveitá-lo aqui produziria um JSON com números
de OUTRO experimento dentro — que é exatamente o defeito corrigido em
`comparar_ablacao_phienc.py` em 2026-09-22, e ele chegava a inverter o lado de uma
assimetria.

A CONTA é a mesma, e continua vindo de um lugar só: `avaliar_um` e `comparar_pareado`
do G1, e `bootstrap_pareado_itens` das secundárias. O que muda aqui é a leitura.

## As duas perguntas, e por que a ordem importa

1. **Entre os braços** (`tratado − controle`): uma variável, `p_equacao`. É a
   repetição, a 150 M, do ganho que o §2.3 mediu a 48 M (+0,084).
2. **Contra a barra** (`melhor braço − ModernBERT-base@200k`): se o melhor dos dois
   não supera a base geral ajustada do mesmo jeito, o pré-treino continuado não paga
   o próprio custo. ⚠️ Não é ablação de uma variável — o CPT viu 0,4 B tokens de
   Física a mais.

Os TRÊS são medidos nesta sessão. Comparar com um número histórico mediria a deriva
do avaliador junto com a diferença de base.
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

BARRA_DO_PASSO_1 = 0.5270


def ler_entre_bracos(nd: dict) -> tuple[str, str]:
    """`nd` é `tratado − controle`, pelo bootstrap pareado por item."""
    lo, hi = nd["ic95"]
    if lo > 0:
        return "TRATADO À FRENTE", (
            "o ganho de recuperação do §2.3 se repete a 150 M. O mascaramento de "
            "equações entra no programa como objetivo de pré-treino continuado.")
    if hi < 0:
        return "CONTROLE À FRENTE", (
            "o tratamento ATRAPALHA a recuperação a 150 M. O objetivo do §2.3 sai "
            "do programa.")
    return "EMPATE", (
        "o ganho de 48 M não se repete na escala que importa. Com a primária de MLM "
        "também não decidida, o §2.3 fica sem evidência a favor fora do proxy.")


def ler_contra_a_barra(nome: str, nd: dict) -> tuple[str, str]:
    """`nd` é `melhor braço − barra`. A barra é o ModernBERT-base CRU ajustado igual."""
    lo, hi = nd["ic95"]
    if lo > 0:
        return "CPT ACIMA DA BARRA", (
            f"o braço {nome} supera a base geral ajustada do mesmo jeito. O "
            "pré-treino continuado paga os 0,4 B tokens que custou, e o encoder do "
            "sistema passa a ser candidato a trocar — decisão do dono, no ADR-0003.")
    if hi < 0:
        return "CPT ABAIXO DA BARRA", (
            f"nem o melhor braço ({nome}) alcança a base geral ajustada. O "
            "pré-treino continuado em 0,4 B de Física PIORA o que a base já fazia, e "
            "o caminho B fecha para o produto.")
    return "EMPATE COM A BARRA", (
        f"o melhor braço ({nome}) empata com a base geral ajustada. 0,4 B tokens de "
        "Física e ~15 h de T4 não mudaram o recuperador: o caminho B não paga o "
        "próprio custo, e o encoder do sistema continua sendo a base ajustada.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controle", required=True, help="o CPT p_equacao 0,0, ajustado")
    ap.add_argument("--tratado", required=True, help="o CPT p_equacao 0,6, ajustado")
    ap.add_argument("--barra", required=True,
                    help="o ModernBERT-base CRU ajustado nos mesmos pares (passo 1)")
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
    for nome, caminho in (("controle", a.controle), ("tratado", a.tratado),
                          ("barra", a.barra)):
        r = avaliar_um(caminho, nome, val, n=a.n, dispositivo=a.dispositivo,
                       semente=SEMENTE_POOL)
        if r.erro:
            raise SystemExit(f"{nome}: {r.erro}")
        medidos[nome] = r

    rc, rt, rb = medidos["controle"], medidos["tratado"], medidos["barra"]
    entre = bootstrap_pareado_itens(ndcg_por_item(rc.posicoes),
                                    ndcg_por_item(rt.posicoes))
    desfecho, leitura = ler_entre_bracos(entre)

    melhor_nome = max(("controle", "tratado"), key=lambda k: medidos[k].ndcg_10)
    # ⚠️ `bootstrap_pareado_itens(a, b)` devolve `b − a` — aqui, melhor − barra. O
    # sinal trocado inverteria "o CPT paga o próprio custo" sem nada acusar, e os
    # dois desfechos são frases plausíveis.
    contra = bootstrap_pareado_itens(ndcg_por_item(rb.posicoes),
                                     ndcg_por_item(medidos[melhor_nome].posicoes))
    d_barra, l_barra = ler_contra_a_barra(melhor_nome, contra)

    def resumo(r):
        return {"caminho": r.caminho, "recall_1": round(r.recall_1, 4),
                "recall_10": round(r.recall_10, 4), "mrr": round(r.mrr, 4),
                "ndcg_10": round(r.ndcg_10, 4)}

    resultado = {
        "experimento": "t2eq_cpt_emb — ADR-0003, caminho B, secundária",
        "regra": "colab/t2eq_cpt_emb.py · REGRA",
        "protocolo": {"n": a.n, "semente": SEMENTE_POOL, "max_tokens": 192,
                      "pares_de_treino": 200_000,
                      "todos_medidos_nesta_sessao": True},
        "controle": resumo(rc), "tratado": resumo(rt), "barra": resumo(rb),
        "barra_do_passo_1_registrada": BARRA_DO_PASSO_1,
        "entre_bracos_ndcg_10_tratado_menos_controle": entre,
        "diagnostico_recall_1_mcnemar": comparar_pareado(rc, rt),
        "desfecho": desfecho, "leitura": leitura,
        "contra_a_barra": {
            "melhor_braco": melhor_nome,
            "ndcg_10_melhor_menos_barra": contra,
            "desfecho": d_barra, "leitura": l_barra,
            "ressalva": ("não é ablação de uma variável: o CPT viu 0,4 B tokens de "
                         "Física a mais que a barra. Entre os DOIS BRAÇOS, sim."),
        },
        "primaria_de_mlm_ja_medida": {
            "diferenca_das_diferencas": -0.00039, "ic95": [-0.00161, 0.00087],
            "desfecho": "NÃO DECIDIDO",
            "fonte": "data/processed/avaliacao/t2eq_cpt_ablacao.json",
        },
        "ressalvas": [
            "CPT de 0,4 B tokens sobre o ModernBERT-base, uma semente por braço.",
            "200 mil pares, metade da receita do T1a; o absoluto NÃO se compara com "
            "o MiniLM@400k nem com o GTE-base@400k do T1f.",
            "O controle do CPT teve 2 spikes de norma com rollback de 182 lotes e o "
            "tratado nenhum — assimetria a favor do tratado, herdada do pré-treino.",
        ],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print()
    print("=" * 78)
    print("  Secundária do caminho B · o CPT ajustado · pool do G1, n=%d" % a.n)
    print("=" * 78)
    print(f"  {'':<12} {'recall@1':>9} {'recall@10':>10} {'MRR':>7} {'nDCG@10':>9}")
    for nome in ("controle", "tratado", "barra"):
        r = resultado[nome]
        print(f"  {nome:<12} {r['recall_1']:>9.4f} {r['recall_10']:>10.4f} "
              f"{r['mrr']:>7.4f} {r['ndcg_10']:>9.4f}")
    print(f"\n  ENTRE OS BRAÇOS · nDCG@10 tratado − controle: "
          f"{entre['diferenca']:+.5f} {entre['ic95']}")
    print(f"  recall@1 pareado (diagnóstico): "
          f"{resultado['diagnostico_recall_1_mcnemar'].get('veredito')}")
    print(f"  DESFECHO: {desfecho}\n  {leitura}")
    print(f"\n  CONTRA A BARRA · nDCG@10 {melhor_nome} − barra: "
          f"{contra['diferenca']:+.5f} {contra['ic95']}")
    print(f"  DESFECHO: {d_barra}\n  {l_barra}")
    print("=" * 78)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
