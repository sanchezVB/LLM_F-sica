#!/usr/bin/env python3
"""Negativos com a DISTRIBUIÇÃO DA AVALIAÇÃO: o top-K do RRF, não o top-K do denso.

    .venv-treino/Scripts/python.exe scripts/minerar_do_recuperador.py --max-ancoras 30000

## O defeito que isto conserta, medido em 2026-08-24

`minerar_negativos.py` toma como negativo o **top-K do ΦEmb menos a citação
verdadeira**. Isso rotula NEGATIVO tudo que o recuperador coloca no topo, e o
cross-encoder aprende exatamente isso: `muito recuperado ⇒ não é a resposta`.

Medido no ΦRank treinado com aqueles negativos, escore médio por faixa de posição
na fusão (60 consultas, top-50):

    posições  0-4    -4,319
    posições  5-14   -4,053
    posições 15-29   -3,812
    posições 30-49   -3,723   <- o recuperador põe por último, o ΦRank prefere

Monotônico. Spearman(posição na fusão, escore) = +0,179, positivo em 83% das
consultas. E como no T1b os candidatos SÃO o top-50 do recuperador, o reranker
rebaixa justamente o que a fusão promoveu: nDCG@10 caiu de **0,1393** (fusão
sozinha) para **0,0179**, pior que ordem aleatória.

## A correção: treinar na distribuição do teste

Aqui o grupo é montado do **RRF top-K de verdade** — o mesmo BM25 + ΦEmb + fusão
por posto que a avaliação usa. O positivo é a citação verdadeira **que apareceu no
top-K**, e os negativos são os outros candidatos do mesmo top-K.

Com isso "estar no topo" deixa de prever o rótulo: o positivo também está no topo,
porque foi assim que ele entrou no grupo.

## ⚠️ O par só existe quando o recuperador acerta

Recall@50 é 0,443, então ~56% das âncoras não produzem grupo — o alvo nunca chegou
ao candidato. Isso é correto e não é perda: um reranker só age quando o documento
certo está no conjunto. Treinar nos casos em que ele não está seria treinar num
grupo sem resposta certa.

## O universo, e por que ele tem o tamanho que tem

O avaliador do T1b usa os citados da validação: **88.807** documentos. Aqui o pool
é amostrado no MESMO tamanho a partir dos citados do treino, porque a dificuldade da
recuperação depende do tamanho do universo — minerar num pool de 667 mil e avaliar
num de 89 mil produziria negativos mais difíceis do que os que o modelo vai ver.

As citações verdadeiras das âncoras sorteadas entram no pool obrigatoriamente; o
resto é sorteado. Sem isso o positivo poderia não existir no universo e a âncora
seria descartada por um motivo artificial.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    Entrada,
    gravar_manifesto_etapa,
)
from phifm.eval.hibrido import BM25, fundir_rrf, top_k  # noqa: E402
from phifm.training.amostragem import amostrar_por_documento  # noqa: E402
from phifm.training.embedding import (  # noqa: E402
    escolher_dispositivo,
    media_mascarada,
)

log = logging.getLogger("minerar-rrf")


def _codificar(mod, tok, textos: list[str], dev, max_tokens: int,
               lote: int, rotulo: str) -> np.ndarray:
    saidas, t0 = [], time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(textos), lote):
            b = tok(textos[i:i + lote], padding="max_length", truncation=True,
                    max_length=max_tokens, return_tensors="pt")
            b = {k: v.to(dev) for k, v in b.items()}
            h = mod(**b).last_hidden_state
            v = F.normalize(media_mascarada(h, b["attention_mask"]).float(), dim=-1)
            saidas.append(v.cpu().numpy())
            if i and (i // lote) % 200 == 0:
                feito = i + lote
                taxa = feito / (time.perf_counter() - t0)
                log.info("  %s: %s/%s · %.0f/s · faltam %.0f min", rotulo,
                         f"{feito:,}", f"{len(textos):,}", taxa,
                         (len(textos) - feito) / taxa / 60)
    return np.vstack(saidas)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    p.add_argument("--emb", type=Path, default=Path("models/phiemb-minilm-melhor"))
    p.add_argument("--out", type=Path, default=Path(
        "data/processed/negativos_dificeis/pares_do_recuperador.parquet"))
    p.add_argument("--max-ancoras", type=int, default=30000)
    p.add_argument("--universo", type=int, default=88807,
                   help="tamanho do pool; o padrão é o do avaliador do T1b")
    p.add_argument("--profundidade", type=int, default=50,
                   help="top-K do RRF de onde o grupo sai; igual ao do avaliador")
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--semente", type=int, default=17)
    p.add_argument("--dispositivo", default="auto",
                   choices=["auto", "cuda", "dml", "cpu"])
    a = p.parse_args()

    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    treino = a.pares / "pares_treino.parquet"

    # ── âncoras, sorteadas com diversidade de DOCUMENTO ─────────────────────
    # ⚠️ `sample`, nunca `head`: o parquet vem agrupado por documento citado e as
    # primeiras linhas são poucos papers repetidos. Ver `amostrar_por_documento`.
    linhas = pl.scan_parquet(treino).select(
        ["arxiv_id", "arxiv_citado", "ancora"]).collect(engine="streaming")
    log.info("pares de treino: %s", f"{len(linhas):,}")
    amostra, n_doc = amostrar_por_documento(linhas, a.max_ancoras, a.semente)
    log.info("âncoras sorteadas: %s · %s documentos citados distintos",
             f"{len(amostra):,}", f"{n_doc:,}")
    del linhas

    # ── exclusão: TODAS as citações verdadeiras de cada âncora ──────────────
    # Da tabela inteira de arestas, não do recorte. Um candidato que a âncora cita
    # não é negativo dela, mesmo que não seja o positivo desta linha.
    ancoras = set(amostra["arxiv_id"].to_list())
    arestas = pl.scan_parquet(treino).select(["arxiv_id", "arxiv_citado"]).filter(
        pl.col("arxiv_id").is_in(ancoras)).collect(engine="streaming")
    cita = {k: set(v) for k, v in arestas.group_by("arxiv_id")
            .agg(pl.col("arxiv_citado").unique()).iter_rows()}
    log.info("âncoras com lista de citações: %s (média %.1f citações)",
             f"{len(cita):,}", sum(len(v) for v in cita.values()) / max(len(cita), 1))
    del arestas

    # ── pool: as citações verdadeiras + sorteio até o tamanho do avaliador ──
    todos = pl.scan_parquet(treino).select("arxiv_citado").unique().collect(
        engine="streaming")["arxiv_citado"].to_list()
    obrigatorios = set(amostra["arxiv_citado"].to_list())
    resto = [d for d in todos if d not in obrigatorios]
    rng = random.Random(a.semente)
    rng.shuffle(resto)
    falta = max(a.universo - len(obrigatorios), 0)
    ids_pool = sorted(obrigatorios) + resto[:falta]
    log.info("pool: %s documentos (%s obrigatórios + %s sorteados de %s)",
             f"{len(ids_pool):,}", f"{len(obrigatorios):,}", f"{falta:,}",
             f"{len(todos):,}")
    if len(obrigatorios) > a.universo:
        log.warning("as citações verdadeiras já passam de --universo; o pool ficou "
                    "maior que o do avaliador e os negativos serão mais difíceis")
    del todos, resto

    alvo = set(ids_pool)
    docs = pl.scan_parquet(treino).select(["arxiv_citado", "positivo"]).filter(
        pl.col("arxiv_citado").is_in(alvo)).unique(
        subset=["arxiv_citado"]).collect(engine="streaming")
    ids_pool = docs["arxiv_citado"].to_list()
    textos_pool = docs["positivo"].to_list()
    onde = {d: i for i, d in enumerate(ids_pool)}
    log.info("pool materializado: %s documentos", f"{len(ids_pool):,}")
    del docs

    # ── índices ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    bm = BM25().indexar(textos_pool)
    log.info("BM25 indexado em %.0f s", time.perf_counter() - t0)

    dev = escolher_dispositivo(a.dispositivo)
    tok = AutoTokenizer.from_pretrained(a.emb)
    mod = AutoModel.from_pretrained(a.emb).to(dev).eval()
    V = _codificar(mod, tok, textos_pool, dev, a.max_tokens, a.lote, "pool")
    consultas = amostra["ancora"].to_list()
    Vq = _codificar(mod, tok, consultas, dev, a.max_tokens, a.lote, "consultas")
    Vt = np.ascontiguousarray(V.T)
    del mod, V

    # ── o grupo: RRF top-K, positivo dentro, negativos ao lado ─────────────
    # ⚠️ `posto_do_alvo` e `postos_negativos` sao gravados porque sem eles a
    # comparacao com o proprio recuperador vira conta de guardanapo. Em 2026-08-24
    # tive de ESTIMAR o acerto 8-way do RRF por
    # P(os n negativos sorteados caem abaixo do positivo) = C(K-1-r, n)/C(K-1, n),
    # quando o dado exato cabia em duas colunas. Com eles, a linha de base do
    # reranker sai do proprio arquivo, sem GPU e sem suposicao.
    saida = {"arxiv_id": [], "arxiv_citado": [], "ancora": [], "positivo": [],
             "negativos_id": [], "negativos": [], "posto_do_alvo": [],
             "postos_negativos": []}
    fora_do_topo = descartados_por_serem_citacao = 0
    posicoes = []
    ids_anc = amostra["arxiv_id"].to_list()
    ids_cit = amostra["arxiv_citado"].to_list()

    t0 = time.perf_counter()
    for i, (aid, cid, consulta) in enumerate(zip(ids_anc, ids_cit, consultas,
                                                 strict=True)):
        alvo_idx = onde.get(cid)
        if alvo_idx is None:
            continue
        ord_bm = top_k(bm.pontuar(consulta), a.profundidade)
        ord_emb = top_k(Vq[i] @ Vt, a.profundidade)
        rrf = fundir_rrf(ord_emb, ord_bm)[:a.profundidade]
        if alvo_idx not in rrf:
            fora_do_topo += 1
            continue
        posicoes.append(rrf.index(alvo_idx))

        proibidos = cita.get(aid, set())
        nids, ntxs, npostos = [], [], []
        for posto, d in enumerate(rrf):
            if d == alvo_idx:
                continue
            if ids_pool[d] in proibidos:
                descartados_por_serem_citacao += 1
                continue
            nids.append(ids_pool[d])
            ntxs.append(textos_pool[d])
            npostos.append(posto)

        saida["arxiv_id"].append(aid)
        saida["arxiv_citado"].append(cid)
        saida["ancora"].append(consulta)
        saida["positivo"].append(textos_pool[alvo_idx])
        saida["negativos_id"].append(nids)
        saida["negativos"].append(ntxs)
        saida["posto_do_alvo"].append(rrf.index(alvo_idx))
        saida["postos_negativos"].append(npostos)

        if i and i % 2000 == 0:
            taxa = (i + 1) / (time.perf_counter() - t0)
            log.info("  %s/%s âncoras · %.1f/s · %s grupos · faltam %.0f min",
                     f"{i+1:,}", f"{len(ids_anc):,}", taxa,
                     f"{len(saida['arxiv_id']):,}",
                     (len(ids_anc) - i) / taxa / 60)

    d = pl.DataFrame(saida)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    d.write_parquet(a.out, compression="zstd")

    n = max(len(d), 1)
    total_neg = int(d["negativos_id"].list.len().sum()) if len(d) else 0
    meta = {
        "ancoras_tentadas": len(ids_anc),
        "grupos_gerados": len(d),
        "alvo_fora_do_top_k": fora_do_topo,
        "recall_em_k": round(len(d) / max(len(ids_anc), 1), 4),
        "profundidade": a.profundidade,
        "universo": len(ids_pool),
        "negativos_total": total_neg,
        "negativos_por_grupo": round(total_neg / n, 2),
        "descartados_por_serem_citacao_verdadeira": descartados_por_serem_citacao,
        "posicao_do_alvo_no_rrf": {
            "media": round(float(np.mean(posicoes)), 2) if posicoes else None,
            "p50": int(np.percentile(posicoes, 50)) if posicoes else None,
            "p90": int(np.percentile(posicoes, 90)) if posicoes else None,
        },
        "por_que": ("negativos com a distribuição da AVALIAÇÃO (top-K do RRF). Os "
                    "de minerar_negativos.py vinham do top-K do denso menos o "
                    "positivo, o que rotula negativo tudo que o recuperador acha "
                    "bom — e o ΦRank aprendeu a inverter o recuperador"),
        "ainda_falta": ("passar por scripts/filtrar_cocitacao.py: co-citados com o "
                        "positivo continuam entrando como negativo"),
    }
    (a.out.parent / "_do_recuperador.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    gravar_manifesto_etapa(
        etapa="negativos_do_recuperador",
        descricao="Negativos com a distribuição da avaliação (RRF top-K)",
        # ⚠️ O ARQUIVO, nao o diretorio: `negativos_dificeis/` ja guarda a saida
        # de `minerar_negativos.py` e de `filtrar_cocitacao.py`, e gravar o
        # manifesto no diretorio apagaria o deles. Com o arquivo, o destino vira
        # `pares_do_recuperador.parquet_manifesto_etapa.json`.
        raiz=a.out,
        entradas=[Entrada(caminho=str(treino)), Entrada(caminho=str(a.emb))],
        parametros={"script": "scripts/minerar_do_recuperador.py",
                    "max_ancoras": a.max_ancoras, "universo": a.universo,
                    "profundidade": a.profundidade, "semente": a.semente},
        registros=len(d))

    print()
    print("=" * 70)
    print(f"  {len(d):,} grupos de {len(ids_anc):,} âncoras "
          f"(recall@{a.profundidade} = {len(d)/max(len(ids_anc),1):.3f})")
    print(f"  {total_neg/n:.2f} negativos por grupo · alvo em posição média "
          f"{meta['posicao_do_alvo_no_rrf']['media']}")
    print(f"  -> {a.out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
