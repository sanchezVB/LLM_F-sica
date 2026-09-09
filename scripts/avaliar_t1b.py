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

from phifm.core.modelos import RECUPERADOR, RERANQUEADOR  # noqa: E402
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
    p.add_argument("--emb", type=Path, default=Path(RECUPERADOR))
    # O ΦRank do sistema é o de PhysBERT desde o T1c (2026-09-03): é a única das
    # três bases medidas que bate a fusão (p=0,0062). Ver `phifm.training.rerank`.
    p.add_argument("--rank", type=Path,
                   default=Path(RERANQUEADOR))
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
    # ⚠️ A composicao decidida no T1b2, e o que ela poupa.
    #
    # A regra pre-registrada de 2026-09-08 tirou o BM25: a cadeia e
    # `PhiEmb -> PhiRank`. Sem a chave, o avaliador ainda indexa o BM25, funde por
    # RRF e reordena DUAS vezes -- e reordenar e 96% do custo, entao medir a
    # composicao que nao existe mais dobra o preco do braco.
    #
    # Com ela, dois bracos de RERANQUEADOR cabem na mesma sessao. Isso nao e
    # conveniencia: a pergunta do retreino e "quanto mudou", e essa exige o braco
    # de referencia medido na MESMA sessao. Foi o que salvou a conclusao do T1b2
    # de um erro de sete vezes.
    p.add_argument("--sem-fusao", action="store_true",
                   help="mede so a composicao decidida (PhiEmb -> PhiRank): nao "
                        "indexa BM25, nao funde, e reordena uma vez por consulta "
                        "em vez de duas")
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

    # ⚠️ O relógio da execução inteira, para o CUSTO entrar no artefato.
    #
    # Medido em 2026-09-08: eu estimei ~50 min para dois braços e o real foi
    # ~2h52 por braço — erro de 7×. E não havia como acertar: os artefatos de
    # avaliação gravam as MÉTRICAS e nunca o custo, então cada estimativa de
    # "quanto tempo leva a cadeia" tinha de ser remodelada do zero. Gravar o
    # tempo transforma a próxima estimativa numa consulta.
    t_inicio = time.perf_counter()
    custo: dict[str, float] = {}

    # ── BM25 ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    if a.sem_fusao:
        bm = None
        log.info("BM25 NAO indexado: --sem-fusao mede a composicao decidida no "
                 "T1b2 (PhiEmb -> PhiRank)")
    else:
        bm = BM25().indexar(textos_pool)
        custo["bm25_indexar_s"] = round(time.perf_counter() - t0, 1)
        log.info("BM25 indexado em %.0f s", custo["bm25_indexar_s"])

    # ── ΦEmb ────────────────────────────────────────────────────────────────
    dev = escolher_dispositivo(a.dispositivo)
    tok_e = AutoTokenizer.from_pretrained(a.emb)
    mod_e = AutoModel.from_pretrained(a.emb).to(dev).eval()
    t0 = time.perf_counter()
    V_pool = _codificar(mod_e, tok_e, textos_pool, dev, a.max_tokens, a.lote)
    V_cons = _codificar(mod_e, tok_e, consultas, dev, a.max_tokens, a.lote)
    custo["embutir_universo_s"] = round(time.perf_counter() - t0, 1)
    log.info("ΦEmb codificou %s documentos em %.0f s",
             f"{len(ids_pool):,}", custo["embutir_universo_s"])
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
    # ⚠️ O reranqueador sobre o DENSO SOZINHO, além de sobre a fusão.
    #
    # Medido no T1b2: com o recuperador novo, o recall@100 da fusão (0,6065) é
    # MENOR que o do ΦEmb sozinho (0,6325) — misturar o BM25 (0,4540) desloca
    # candidatos bons do top-100 e derruba o teto do reranker em 0,026. E o
    # pareado diz que a fusão virou empate com o denso sozinho (p=0,949 no
    # top-10, contra p=1,5e-08 com o recuperador antigo).
    #
    # Na MESMA execução, e não em dois runs: reordenar é 96% do custo e embutir o
    # universo é compartilhado, então duas passagens custam o mesmo que dois runs
    # — e o pareamento fica exato, com as mesmas consultas na mesma ordem.
    pos_rank_denso: list = []
    depurados = [0]
    t0 = time.perf_counter()
    for i, (consulta, alvo) in enumerate(zip(consultas, alvos, strict=True)):
        e_emb = V_cons[i] @ Vt
        ord_emb = top_k(e_emb, a.profundidade)
        pos_emb.append(_posicao(ord_emb, alvo))
        if a.sem_fusao:
            ord_rrf = ord_emb
        else:
            ord_bm = top_k(bm.pontuar(consulta), a.profundidade)
            ord_rrf = fundir_rrf(ord_emb, ord_bm)[:a.profundidade]
            pos_bm.append(_posicao(ord_bm, alvo))
            pos_rrf.append(_posicao(ord_rrf, alvo))

        if tem_rank:
            # `q` como PARAMETRO, nao capturado do laco: o ruff B023 pega
            # isso, e com razao. A funcao e chamada dentro da iteracao hoje,
            # mas uma chamada movida para fora veria a ULTIMA consulta em vez
            # desta, e o numero sairia com a cara certa.
            def _reordenar(q: str, candidatos: list) -> tuple[list, np.ndarray]:
                with torch.no_grad():
                    escores = []
                    for j in range(0, len(candidatos), a.lote_rank):
                        pedaco = candidatos[j:j + a.lote_rank]
                        b = tok_r([q] * len(pedaco),
                                  [textos_pool[d] for d in pedaco],
                                  padding="max_length", truncation=True,
                                  max_length=384, return_tensors="pt")
                        b = {k: v.to(dev) for k, v in b.items()}
                        escores.append(
                            mod_r(**b).logits.float().view(-1).cpu().numpy())
                e = np.concatenate(escores)
                return [candidatos[k] for k in np.argsort(-e)], e

            if a.sem_fusao:
                # Uma passagem so: a fusao nao existe nesta composicao, e o
                # `pos_rank_denso` E a cadeia.
                ord_rank_denso, e = _reordenar(consulta, ord_emb)
                pos_rank_denso.append(_posicao(ord_rank_denso, alvo))
                ord_rank = ord_rank_denso
            else:
                ord_rank, e = _reordenar(consulta, ord_rrf)
                pos_rank.append(_posicao(ord_rank, alvo))
                # A segunda passagem: o mesmo reranqueador sobre o top-100 DENSO.
                ord_rank_denso, _ = _reordenar(consulta, ord_emb)
                pos_rank_denso.append(_posicao(ord_rank_denso, alvo))
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

    if a.sem_fusao:
        sistemas = [bloco("ΦEmb", pos_emb)]
        if tem_rank:
            sistemas.append(bloco("ΦEmb+ΦRank", pos_rank_denso))
    else:
        sistemas = [bloco("BM25", pos_bm), bloco("ΦEmb", pos_emb),
                    bloco("ΦEmb+BM25 (RRF)", pos_rrf)]
        if tem_rank:
            sistemas.append(bloco("ΦEmb+BM25+ΦRank", pos_rank))
            sistemas.append(bloco("ΦEmb+ΦRank (sem fusão)", pos_rank_denso))

    # ── comparação PAREADA contra a fusão ───────────────────────────────────
    # ⚠️ Sem isto a tabela convida a ler diferença onde há ruído: 300 consultas dão
    # erro padrão de ±0,024 em cada proporção, e duas linhas separadas por 0,01
    # pareceriam distintas. Os sistemas foram medidos nas MESMAS consultas, então o
    # que decide são as discordantes — ver `mcnemar_em`.
    #
    # A referência é a FUSÃO e não o melhor de todos, porque a pergunta do T1b é
    # exatamente "o reranker acrescenta algo à fusão?".
    # ⚠️ A REFERENCIA do pareado muda com a composicao, e tem de mudar.
    #
    # Com fusao, a pergunta do T1b e "o reranker acrescenta algo a fusao?". Sem
    # fusao, a fusao nao existe: a pergunta passa a ser "o reranker acrescenta
    # algo ao RECUPERADOR?", e a referencia e o PhiEmb. Manter `pos_rrf` como
    # referencia num braco `--sem-fusao` compararia contra uma lista vazia.
    referencia, nome_ref = ((pos_emb, "ΦEmb") if a.sem_fusao
                            else (pos_rrf, "ΦEmb+BM25 (RRF)"))
    if a.sem_fusao:
        contra = [("ΦEmb+ΦRank", pos_rank_denso)] if tem_rank else []
    else:
        contra = [("BM25", pos_bm), ("ΦEmb", pos_emb),
                  *((("ΦEmb+BM25+ΦRank", pos_rank),
                     ("ΦEmb+ΦRank (sem fusão)", pos_rank_denso))
                    if tem_rank else ())]
    pareados = []
    for nome, pos in contra:
        for k in (1, 10):
            pareados.append(mcnemar_em(referencia, pos, k, nome_ref, nome))

    # ⚠️ O confronto DIRETO entre as duas cadeias — o teste que a regra
    # pré-registrada do T1b2 nomeia, e que a primeira versão desta função não
    # calculava.
    #
    # A regra era: "se ΦEmb+ΦRank VENCER ΦEmb+BM25+ΦRank no pareado, o BM25 sai
    # da composição; se EMPATAR, sai também". Mas todos os pareados acima têm a
    # FUSÃO como referência, porque a pergunta original do T1b era outra ("o
    # reranker acrescenta algo à fusão?"). Duas comparações contra um terceiro
    # sistema não são a comparação entre elas: em 2026-09-08 as duas cadeias
    # deram p=0,086 e p=0,149 contra a fusão, e nenhum desses números é o
    # veredito que a regra pedia.
    # Sem fusao nao HA duas cadeias para confrontar: a regra ja decidiu.
    confronto = ([mcnemar_em(pos_rank_denso, pos_rank, k,
                             "ΦEmb+ΦRank (sem fusão)", "ΦEmb+BM25+ΦRank")
                  for k in (1, 10)]
                 if tem_rank and not a.sem_fusao else [])

    teto = recall_em(pos_emb if a.sem_fusao else pos_rrf, a.profundidade)
    # ⚠️ DOIS tetos, porque agora há duas cadeias. O da fusão limita o
    # `ΦEmb+BM25+ΦRank`; o do denso limita o `ΦEmb+ΦRank`. Reportar um só faria
    # uma das duas cadeias parecer limitada pelo teto da outra.
    teto_denso = recall_em(pos_emb, a.profundidade)
    custo["total_s"] = round(time.perf_counter() - t_inicio, 1)
    custo["reordenar_s"] = round(
        custo["total_s"] - custo.get("embutir_universo_s", 0.0)
        - custo.get("bm25_indexar_s", 0.0), 1)
    resultado = {
        "n_consultas": len(consultas), "universo": len(ids_pool),
        "custo_segundos": custo,
        "nota_custo": (f"Medido nesta execução, em {a.dispositivo}. O reordenar "
                       "domina: são n_consultas × profundidade passagens de um "
                       "cross-encoder. Gravado porque os artefatos anteriores "
                       "guardavam as métricas e nunca o custo, e por isso uma "
                       "estimativa de 2026-09-08 errou por 7×."),
        "profundidade": a.profundidade,
        "modelos": {"emb": str(a.emb), "rank": str(a.rank) if tem_rank else None},
        "teto_do_reranker": round(teto, 4),
        "teto_do_reranker_sem_fusao": round(teto_denso, 4),
        "nota_teto": (f"recall@{a.profundidade} da fusão. Um ΦRank perfeito não "
                      "passa disto, e nenhuma melhora de reranking aparece nas "
                      "consultas em que o documento certo não chegou."),
        "sistemas": sistemas,
        "pareado_contra_a_fusao": pareados,
        # ⚠️ Separado de `pareado_contra_a_fusao` de propósito: aquela chave
        # promete que a referência é a fusão, e enfiar aqui um par sem ela
        # dentro faria o nome mentir para quem lê o artefato.
        "confronto_das_cadeias": confronto,
        "nota_confronto": ("O pareado que a regra pré-registrada do T1b2 nomeia: "
                           "a cadeia SEM o BM25 contra a cadeia COM. A regra é "
                           "que o BM25 só fica se a cadeia com ele VENCER — "
                           "empate já o tira, porque ele não paga o próprio "
                           "custo nem o teto que cobra."),
        "sem_fusao": bool(a.sem_fusao),
        "nota_pareado": (
            f"McNemar exato sobre 'o alvo chegou ao top-k'. A referência é "
            f"{nome_ref}: "
            + ("sem a fusão, a pergunta é se o reranker acrescenta algo ao "
               "RECUPERADOR — a composição decidida no T1b2 é ΦEmb → ΦRank."
               if a.sem_fusao else
               "a pergunta do T1b é se o reranker acrescenta algo à fusão.")),
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
    if confronto:
        print()
        print("  CONFRONTO das cadeias (a regra pré-registrada do BM25):")
        for c in confronto:
            if "erro" in c:
                print(f"    {c['erro']}")
                continue
            print(f"    top-{c['k']:<3} {c['ganha_a']:>3} a {c['ganha_b']:<3} · "
                  f"{c['veredito']}")
        print("    a regra: o BM25 só fica se a cadeia COM ele vencer.")
    print("=" * 74)
    print(f"  TETO do reranker (recall@{a.profundidade} "
          f"{'do ΦEmb' if a.sem_fusao else 'da fusão'}): {teto:.4f}")
    # ⚠️ Os DOIS tetos impressos, não só o da fusão: se a cadeia escolhida for a
    # sem fusão, imprimir só o da fusão mostraria o limite do braço descartado.
    # Com `--sem-fusao` os dois são o MESMO número, e imprimir duas vezes o mesmo
    # valor com dois rótulos convida a ler dois tetos onde há um.
    if tem_rank and not a.sem_fusao:
        print(f"  TETO sem a fusão (recall@{a.profundidade} do ΦEmb):    "
              f"{teto_denso:.4f}")
    print()
    print(f"  PAREADO contra {nome_ref} (McNemar exato, mesmas consultas):")
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
