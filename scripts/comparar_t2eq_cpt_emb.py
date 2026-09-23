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


def ler_a_base(nd: dict) -> tuple[str, str]:
    """`nd` é `gte − barra`. A regra é a do braço gte em `kaggle/t2eq_emb.py`."""
    lo, hi = nd["ic95"]
    if lo > 0:
        return "GTE À FRENTE", (
            "os dois braços do CPT partiram da base errada. A resposta do ADR-0003 "
            "para o produto passa a ser trocar a base, não continuar o pré-treino; o "
            "que o CPT mediu sobre o §2.3 continua valendo.")
    if hi < 0:
        return "MODERNBERT À FRENTE", (
            "a escolha do caminho B se confirma: os braços do CPT partiram da melhor "
            "base disponível nesta receita.")
    return "EMPATE", (
        "as duas bases gerais chegam no mesmo lugar a 200 mil pares. A escolha do "
        "ModernBERT fica sem custo, e o argumento passa a ser o contexto de 8.192.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controle", default=None, help="o CPT p_equacao 0,0, ajustado")
    ap.add_argument("--tratado", default=None, help="o CPT p_equacao 0,6, ajustado")
    ap.add_argument("--gte", default=None,
                    help="o GTE-base ajustado nos mesmos pares (a base é a certa?)")
    ap.add_argument("--barra", required=True,
                    help="o ModernBERT-base CRU ajustado nos mesmos pares (passo 1)")
    ap.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--dispositivo", default="dml")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    # Os dois braços vêm juntos ou não vêm: um braço sozinho não tem a variável.
    if bool(a.controle) != bool(a.tratado):
        raise SystemExit("--controle e --tratado vêm juntos: um braço sozinho não "
                         "mede `p_equacao`.")
    if not a.controle and not a.gte:
        raise SystemExit("nada a comparar com a barra: dê os dois braços, --gte, ou "
                         "os três.")

    import polars as pl

    from avaliar_ablacao_secundarias import (  # noqa: E402
        bootstrap_pareado_itens,
        ndcg_por_item,
    )
    from phifm.eval.encoders import avaliar_um, comparar_pareado  # noqa: E402
    from phifm.training.amostragem import SEMENTE_POOL  # noqa: E402

    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    pedidos = [("barra", a.barra)]
    if a.controle:
        pedidos += [("controle", a.controle), ("tratado", a.tratado)]
    if a.gte:
        pedidos.append(("gte", a.gte))
    medidos = {}
    for nome, caminho in pedidos:
        r = avaliar_um(caminho, nome, val, n=a.n, dispositivo=a.dispositivo,
                       semente=SEMENTE_POOL)
        if r.erro:
            raise SystemExit(f"{nome}: {r.erro}")
        medidos[nome] = r

    def resumo(r):
        return {"caminho": r.caminho, "recall_1": round(r.recall_1, 4),
                "recall_10": round(r.recall_10, 4), "mrr": round(r.mrr, 4),
                "ndcg_10": round(r.ndcg_10, 4)}

    def por_item(nome):
        return ndcg_por_item(medidos[nome].posicoes)

    rb = medidos["barra"]
    resultado = {
        "experimento": "t2eq_cpt_emb — ADR-0003, caminho B, secundária",
        "regra": ("colab/t2eq_cpt_emb.py · REGRA (os braços do CPT) e "
                  "kaggle/t2eq_emb.py · REGRA_GTE (a base)"),
        "protocolo": {"n": a.n, "semente": SEMENTE_POOL, "max_tokens": 192,
                      "pares_de_treino": 200_000,
                      "todos_medidos_nesta_sessao": True},
        **{nome: resumo(r) for nome, r in medidos.items()},
        "barra_do_passo_1_registrada": BARRA_DO_PASSO_1,
    }

    # ⚠️ `bootstrap_pareado_itens(a, b)` devolve `b − a`. O sinal trocado inverteria
    # cada leitura abaixo sem nada acusar, e os desfechos opostos são frases
    # igualmente plausíveis.
    if a.controle:
        rc, rt = medidos["controle"], medidos["tratado"]
        entre = bootstrap_pareado_itens(por_item("controle"), por_item("tratado"))
        desfecho, leitura = ler_entre_bracos(entre)
        melhor_nome = max(("controle", "tratado"), key=lambda k: medidos[k].ndcg_10)
        contra = bootstrap_pareado_itens(por_item("barra"), por_item(melhor_nome))
        d_barra, l_barra = ler_contra_a_barra(melhor_nome, contra)
        resultado |= {
            "entre_bracos_ndcg_10_tratado_menos_controle": entre,
            "diagnostico_recall_1_mcnemar": comparar_pareado(rc, rt),
            "desfecho": desfecho, "leitura": leitura,
            "contra_a_barra": {
                "melhor_braco": melhor_nome,
                "ndcg_10_melhor_menos_barra": contra,
                "desfecho": d_barra, "leitura": l_barra,
                "ressalva": ("não é ablação de uma variável: o CPT viu 0,4 B tokens "
                             "de Física a mais que a barra. Entre os DOIS BRAÇOS, "
                             "sim."),
            },
            "primaria_de_mlm_ja_medida": {
                "diferenca_das_diferencas": -0.00039, "ic95": [-0.00161, 0.00087],
                "desfecho": "NÃO DECIDIDO",
                "fonte": "data/processed/avaliacao/t2eq_cpt_ablacao.json",
            },
        }
    if a.gte:
        base = bootstrap_pareado_itens(por_item("barra"), por_item("gte"))
        d_base, l_base = ler_a_base(base)
        resultado["a_base_e_a_certa"] = {
            "ndcg_10_gte_menos_barra": base,
            "diagnostico_recall_1_mcnemar": comparar_pareado(rb, medidos["gte"]),
            "desfecho": d_base, "leitura": l_base,
            "ressalva": ("uma variável, a BASE: mesmos pares, lote, 192 tokens e "
                         "semente. O GTE-base tem 110 M contra 150 M e contexto de "
                         "512 contra 8.192 — o contexto não entra nesta medida, e "
                         "entra no produto."),
        }
    resultado["ressalvas"] = [
        "Uma semente por braço; 200 mil pares, metade da receita do T1a — o absoluto "
        "NÃO se compara com o MiniLM@400k nem com o GTE-base@400k do T1f.",
    ] + ([
        "CPT de 0,4 B tokens sobre o ModernBERT-base. O controle do CPT teve 2 spikes "
        "de norma com rollback de 182 lotes e o tratado nenhum — assimetria a favor "
        "do tratado, herdada do pré-treino.",
    ] if a.controle else [])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print()
    print("=" * 78)
    print(f"  Secundária do caminho B · pool do G1, n={a.n} · todos nesta sessão")
    print("=" * 78)
    print(f"  {'':<12} {'recall@1':>9} {'recall@10':>10} {'MRR':>7} {'nDCG@10':>9}")
    for nome in medidos:
        r = resultado[nome]
        print(f"  {nome:<12} {r['recall_1']:>9.4f} {r['recall_10']:>10.4f} "
              f"{r['mrr']:>7.4f} {r['ndcg_10']:>9.4f}")
    if a.controle:
        e, c = (resultado["entre_bracos_ndcg_10_tratado_menos_controle"],
                resultado["contra_a_barra"])
        print(f"\n  ENTRE OS BRAÇOS · nDCG@10 tratado − controle: "
              f"{e['diferenca']:+.5f} {e['ic95']}")
        print(f"  DESFECHO: {resultado['desfecho']}\n  {resultado['leitura']}")
        m = c["ndcg_10_melhor_menos_barra"]
        print(f"\n  CONTRA A BARRA · nDCG@10 {c['melhor_braco']} − barra: "
              f"{m['diferenca']:+.5f} {m['ic95']}")
        print(f"  DESFECHO: {c['desfecho']}\n  {c['leitura']}")
    if a.gte:
        g = resultado["a_base_e_a_certa"]
        b = g["ndcg_10_gte_menos_barra"]
        print(f"\n  A BASE É A CERTA? · nDCG@10 gte − modernbert: "
              f"{b['diferenca']:+.5f} {b['ic95']}")
        print(f"  recall@1 pareado (diagnóstico): "
              f"{g['diagnostico_recall_1_mcnemar'].get('veredito')}")
        print(f"  DESFECHO: {g['desfecho']}\n  {g['leitura']}")
    print("=" * 78)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
