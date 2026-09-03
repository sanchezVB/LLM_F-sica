#!/usr/bin/env python3
"""T1b — mede a COMPOSIÇÃO: recuperar, fundir, reordenar.

    .venv-treino/Scripts/python.exe scripts/avaliar_t1b.py --n-consultas 2000

Cada peça já foi medida isolada. Isto mede o que o usuário recebe:

    consulta ─┬─► BM25   ─► top-100 léxico ─┐
              └─► ΦEmb   ─► top-100 denso  ─┴─► RRF ─► top-100 ─► ΦRank ─► top-10

## As quatro linhas que a tabela precisa ter, e por quê

| linha | o que isola |
|---|---|
| BM25 sozinho | a linha de base léxica, que não custa GPU nenhuma |
| ΦEmb sozinho | o campeão do G1.1, como ele é hoje |
| ΦEmb + BM25 (RRF) | o que a fusão acrescenta ao recuperador |
| **+ ΦRank** | o que o reranking acrescenta à fusão |

Sem as duas do meio, um ganho da composição não se atribui: seria impossível dizer
se veio da fusão ou do reranker. Este projeto já perdeu um experimento por mudar
base e lote ao mesmo tempo.

## ⚠️ O teto, que decide como ler tudo

`recall@100` do recuperador limita o resto. Um ΦRank perfeito sobre um recall@100
de 0,70 não passa de 0,70, e nenhuma melhora de reranking aparece nas consultas em
que o documento certo nunca chegou ao top-100.

Por isso o relatório imprime o recall@100 ANTES do nDCG: se o teto for baixo, o
trabalho seguinte é no recuperador, não no reranker — e olhar só o nDCG esconderia
isso.

## O universo é o mesmo do G1

Os documentos CITADOS da validação, e `--n-consultas` âncoras sorteadas dela. Mudar
o universo mudaria a dificuldade e o número deixaria de ser comparável ao veredito.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import (  # noqa: E402
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from phifm.eval.hibrido import (  # noqa: E402
    BM25,
    fundir_rrf,
    mcnemar_em,
    ndcg_em_10,
    recall_em,
    top_k,
)
from phifm.training.embedding import escolher_dispositivo, media_mascarada  # noqa: E402

log = logging.getLogger("t1b")


def _codificar(mod, tok, textos: list[str], dev, max_tokens: int,
               lote: int) -> np.ndarray:
    saidas = []
    with torch.no_grad():
        for i in range(0, len(textos), lote):
            b = tok(textos[i:i + lote], padding="max_length", truncation=True,
                    max_length=max_tokens, return_tensors="pt")
            b = {k: v.to(dev) for k, v in b.items()}
            h = mod(**b).last_hidden_state
            v = F.normalize(media_mascarada(h, b["attention_mask"]).float(), dim=-1)
            saidas.append(v.cpu().numpy())
    return np.vstack(saidas)


def _posicao(ordem: list[int], alvo: int) -> int | None:
    try:
        return ordem.index(alvo)
    except ValueError:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    p.add_argument("--emb", type=Path, default=Path("models/phiemb-minilm-melhor"))
    # O ΦRank do sistema é o de PhysBERT desde o T1c (2026-09-03): é a única das
    # três bases medidas que bate a fusão (p=0,0062). Ver `phifm.training.rerank`.
    p.add_argument("--rank", type=Path,
                   default=Path("models/phirank-physbert-melhor"))
    p.add_argument("--n-consultas", type=int, default=2000,
                   help="âncoras avaliadas; 2.000 é o protocolo do veredito do G1")
    p.add_argument("--profundidade", type=int, default=100,
                   help="quantos candidatos chegam ao ΦRank (DOC-07 §4 diz top-100)")
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--lote", type=int, default=64)
    # ⚠️ 8 e não 32, desde que o ΦRank do sistema passou a ser de 109 M (2026-09-03).
    #
    # Medido nesta máquina, na primeira avaliação depois da troca:
    #
    #     RuntimeError: Could not allocate tensor with 150994944 bytes.
    #                   There is not enough GPU video memory available!
    #
    # São os 151 MB de UM tensor intermediário: lote 32 × 384 tokens × 3.072 do
    # `intermediate` do BERT-base × 4 bytes. O MiniLM anterior tinha 1.536 e 6
    # camadas, então 32 caberia — o default foi calibrado para um modelo que deixou
    # de ser o default, e quem rodasse o avaliador local receberia OOM.
    #
    # 8 dá 37,7 MB por tensor e roda no cartão de 8 GB. Numa T4 de 16 GB um lote
    # maior é mais rápido e vale passar explicitamente; o default existe para
    # FUNCIONAR em todo lugar, não para ser ótimo em um.
    p.add_argument("--lote-rank", type=int, default=8)
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/avaliacao/t1b_resultado.json"))
    p.add_argument("--dispositivo", default="auto",
                   choices=["auto", "cuda", "dml", "cpu"])
    p.add_argument("--semente", type=int, default=17)
    p.add_argument("--depurar", type=int, default=0,
                   help="imprime a posição do alvo antes e depois do ΦRank nas N "
                        "primeiras consultas em que ele está no conjunto")
    a = p.parse_args()

    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    # Universo: os documentos citados da validação, iguais aos do veredito do G1.
    pool = val.unique(subset=["arxiv_citado"], maintain_order=True)
    ids_pool = pool["arxiv_citado"].to_list()
    textos_pool = pool["positivo"].to_list()
    indice_de = {d: i for i, d in enumerate(ids_pool)}

    linhas = val.sample(n=min(a.n_consultas, len(val)), seed=a.semente)
    consultas = linhas["ancora"].to_list()
    alvos = [indice_de[d] for d in linhas["arxiv_citado"].to_list()]
    log.info("universo: %s documentos · %s consultas", f"{len(ids_pool):,}",
             f"{len(consultas):,}")

    # ── BM25 ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    bm = BM25().indexar(textos_pool)
    log.info("BM25 indexado em %.0f s", time.perf_counter() - t0)

    # ── ΦEmb ────────────────────────────────────────────────────────────────
    dev = escolher_dispositivo(a.dispositivo)
    tok_e = AutoTokenizer.from_pretrained(a.emb)
    mod_e = AutoModel.from_pretrained(a.emb).to(dev).eval()
    t0 = time.perf_counter()
    V_pool = _codificar(mod_e, tok_e, textos_pool, dev, a.max_tokens, a.lote)
    V_cons = _codificar(mod_e, tok_e, consultas, dev, a.max_tokens, a.lote)
    log.info("ΦEmb codificou %s documentos em %.0f s",
             f"{len(ids_pool):,}", time.perf_counter() - t0)
    Vt = np.ascontiguousarray(V_pool.T)

    # ── ΦRank ───────────────────────────────────────────────────────────────
    tem_rank = a.rank.exists()
    if tem_rank:
        tok_r = AutoTokenizer.from_pretrained(a.rank)
        mod_r = AutoModelForSequenceClassification.from_pretrained(
            a.rank, num_labels=1).to(dev).eval()
    else:
        log.warning("ΦRank ausente em %s — as duas primeiras linhas saem mesmo "
                    "assim, e são o teto do que o reranker poderia melhorar", a.rank)

    pos_bm, pos_emb, pos_rrf, pos_rank = [], [], [], []
    depurados = [0]
    t0 = time.perf_counter()
    for i, (consulta, alvo) in enumerate(zip(consultas, alvos, strict=True)):
        e_bm = bm.pontuar(consulta)
        ord_bm = top_k(e_bm, a.profundidade)
        e_emb = V_cons[i] @ Vt
        ord_emb = top_k(e_emb, a.profundidade)
        ord_rrf = fundir_rrf(ord_emb, ord_bm)[:a.profundidade]

        pos_bm.append(_posicao(ord_bm, alvo))
        pos_emb.append(_posicao(ord_emb, alvo))
        pos_rrf.append(_posicao(ord_rrf, alvo))

        if tem_rank:
            with torch.no_grad():
                escores = []
                for j in range(0, len(ord_rrf), a.lote_rank):
                    pedaco = ord_rrf[j:j + a.lote_rank]
                    b = tok_r([consulta] * len(pedaco),
                              [textos_pool[d] for d in pedaco],
                              padding="max_length", truncation=True,
                              max_length=384, return_tensors="pt")
                    b = {k: v.to(dev) for k, v in b.items()}
                    escores.append(mod_r(**b).logits.float().view(-1).cpu().numpy())
            e = np.concatenate(escores)
            ord_rank = [ord_rrf[k] for k in np.argsort(-e)]
            pos_rank.append(_posicao(ord_rank, alvo))
            if a.depurar and pos_rrf[-1] is not None and depurados[0] < a.depurar:
                depurados[0] += 1
                k_ = ord_rrf.index(alvo)
                log.info("  [dep] alvo idx=%d · fusao pos %s -> ΦRank pos %s · "
                         "escore %.2f (max %.2f, min %.2f) · len(e)=%d len(rrf)=%d",
                         alvo, pos_rrf[-1], pos_rank[-1], e[k_], e.max(), e.min(),
                         len(e), len(ord_rrf))

        if i and i % 200 == 0:
            taxa = (i + 1) / (time.perf_counter() - t0)
            log.info("  %s/%s consultas · %.1f/s · faltam %.0f min",
                     f"{i+1:,}", f"{len(consultas):,}", taxa,
                     (len(consultas) - i) / taxa / 60)

    def bloco(nome: str, pos: list) -> dict:
        # ⚠️ A chave do recall do teto leva a PROFUNDIDADE no nome. Era
        # `recall_100` fixo, e com `--profundidade 50` o arquivo de resultado
        # afirmava recall@100 sobre um número que era recall@50 — o tipo de
        # rótulo errado que um relatório futuro copia sem verificar.
        return {"sistema": nome,
                "recall_1": round(recall_em(pos, 1), 4),
                "recall_10": round(recall_em(pos, 10), 4),
                f"recall_{a.profundidade}": round(recall_em(pos, a.profundidade), 4),
                "ndcg_10": round(ndcg_em_10(pos), 4)}

    sistemas = [bloco("BM25", pos_bm), bloco("ΦEmb", pos_emb),
                bloco("ΦEmb+BM25 (RRF)", pos_rrf)]
    if tem_rank:
        sistemas.append(bloco("ΦEmb+BM25+ΦRank", pos_rank))

    # ── comparação PAREADA contra a fusão ───────────────────────────────────
    # ⚠️ Sem isto a tabela convida a ler diferença onde há ruído: 300 consultas dão
    # erro padrão de ±0,024 em cada proporção, e duas linhas separadas por 0,01
    # pareceriam distintas. Os sistemas foram medidos nas MESMAS consultas, então o
    # que decide são as discordantes — ver `mcnemar_em`.
    #
    # A referência é a FUSÃO e não o melhor de todos, porque a pergunta do T1b é
    # exatamente "o reranker acrescenta algo à fusão?".
    pareados = []
    for nome, pos in (("BM25", pos_bm), ("ΦEmb", pos_emb),
                      *(( ("ΦEmb+BM25+ΦRank", pos_rank),) if tem_rank else ())):
        for k in (1, 10):
            pareados.append(mcnemar_em(pos_rrf, pos, k,
                                       "ΦEmb+BM25 (RRF)", nome))

    teto = recall_em(pos_rrf, a.profundidade)
    resultado = {
        "n_consultas": len(consultas), "universo": len(ids_pool),
        "profundidade": a.profundidade,
        "modelos": {"emb": str(a.emb), "rank": str(a.rank) if tem_rank else None},
        "teto_do_reranker": round(teto, 4),
        "nota_teto": (f"recall@{a.profundidade} da fusão. Um ΦRank perfeito não "
                      "passa disto, e nenhuma melhora de reranking aparece nas "
                      "consultas em que o documento certo não chegou."),
        "sistemas": sistemas,
        "pareado_contra_a_fusao": pareados,
        "nota_pareado": ("McNemar exato sobre 'o alvo chegou ao top-k'. A referência "
                         "é a fusão porque a pergunta do T1b é se o reranker "
                         "acrescenta algo a ela."),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print()
    print("=" * 74)
    print(f"  T1b · {len(consultas):,} consultas · universo de {len(ids_pool):,} "
          f"· top-{a.profundidade}")
    print("=" * 74)
    rk = f"recall_{a.profundidade}"
    print(f"  {'sistema':<24} {'r@1':>7} {'r@10':>7} {f'r@{a.profundidade}':>7} "
          f"{'nDCG@10':>9}")
    for s in sistemas:
        print(f"  {s['sistema']:<24} {s['recall_1']:>7.3f} {s['recall_10']:>7.3f} "
              f"{s[rk]:>7.3f} {s['ndcg_10']:>9.4f}")
    print("=" * 74)
    print(f"  TETO do reranker (recall@{a.profundidade} da fusão): {teto:.4f}")
    print()
    print("  PAREADO contra a fusão (McNemar exato, mesmas consultas):")
    for r in pareados:
        if "erro" in r:
            print(f"    {r['erro']}")
            continue
        print(f"    top-{r['k']:<3} {r['b']:<20} {r['ganha_a']:>3} a "
              f"{r['ganha_b']:<3} · {r['veredito']}")
    print("=" * 74)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
