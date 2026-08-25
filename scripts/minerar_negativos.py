#!/usr/bin/env python3
"""Negativos DIFÍCEIS minerados pelo próprio ΦEmb (DOC-07 §4).

    .venv-treino/Scripts/python.exe scripts/minerar_negativos.py --max-pares 400000

## Por que isto é a hipótese que sobrou

Duas alavancas de escala foram medidas e as duas são planas (ver ESTADO.md §"As
duas alavancas de escala são planas"): mais negativos no lote (511 contra 127) e
mais dados (1,5 M contra 400 mil pares) dão empate estatístico contra o campeão.

Nenhuma das duas mexeu na **dificuldade** dos negativos. Hoje eles são os outros
pares do mesmo lote — sorteados, portanto quase todos triviais: matéria condensada
contra astrofísica não ensina nada, o modelo já separa. Negativo difícil é o paper
que o próprio ΦEmb coloca no top-K **e não deveria**.

E o mesmo artefato é o insumo obrigatório do ΦRank (DOC-07 §4: "treinado com
negativos difíceis minerados pelo próprio ΦEmb"). Um trabalho, dois usos.

## ⚠️ O erro que arruinaria isto, e como é evitado

Os pares vêm de arestas de citação. Se A cita B **e C**, tomar C como "negativo
difícil" do par (A,B) ensina o modelo a **separar uma citação verdadeira** — é
ruído de rótulo, e ruído de rótulo com aparência de sinal difícil piora o modelo
enquanto parece estar melhorando o treino.

Por isso a exclusão usa a lista COMPLETA de citados de cada âncora, lida da tabela
de arestas inteira (6,56 M), não do subconjunto que está sendo minerado. Excluir só
o positivo da linha seria o defeito.

## ⚠️ MEDIDO EM 2026-08-18: a exclusão acima é NECESSÁRIA E INSUFICIENTE

O treino com estes negativos ficou ABAIXO da linha de base, e as métricas disseram
por quê antes de o agregado dizer:

    passo    recall@1   recall@10   nDCG@10
    base       0,265      0,665      0,454
    200        0,247      0,655      0,434
    400        0,238      0,680      0,440

Recall@1 caindo enquanto recall@10 sobe é a assinatura de negativos que são
**relevantes**: o modelo aproxima a vizinhança inteira da âncora e perde a
capacidade de escolher qual dela é a citação certa.

Medido em 32.000 negativos minerados, contra um controle de negativos sorteados:

    co-citados com o positivo, minerados : 15,2%
    co-citados com o positivo, aleatórios:  0,1%
    razão                                : 212x

Co-citação — existir um paper que cita o positivo E o negativo — é o sinal clássico
de relevância na literatura de recuperação. **15% dos "negativos difíceis" são
documentos que a literatura trata como relacionados ao positivo.**

A exclusão implementada aqui cobre o que a ÂNCORA cita. Não cobre o que é próximo
do POSITIVO, e é aí que o falso negativo entra. Consertar exige excluir também os
co-citados do positivo — o que é uma segunda passada sobre a tabela de arestas, não
uma linha.

Enquanto isso não for feito, este artefato NÃO serve para treinar o ΦEmb. Ele ainda
serve para o ΦRank: um cross-encoder que reordena o top-100 é treinado com pares
(consulta, documento) rotulados, e um co-citado rotulado como "menos relevante que
o citado" é rótulo defensável — o erro aqui foi tratá-lo como NEGATIVO em contraste,
que é uma afirmação muito mais forte.

## O universo de onde os negativos saem

Os **citados**, não todos os documentos. É esse o universo que `avaliar` usa: ela
ordena `n_candidatos` documentos positivos e pergunta em que posição está o certo.
Minerar negativos de um universo diferente do da avaliação produziria dificuldade
que a métrica não vê.

Medido nos primeiros 400 mil pares: 283.278 âncoras contra apenas 17.844 citados
distintos — cada positivo se repete ~22 vezes. O pool completo de citados
(667.304) é o que se usa, para o negativo ser difícil no universo real e não só
no pedaço que o treino viu.
"""
from __future__ import annotations

import argparse
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
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    Entrada,
    gravar_manifesto_etapa,
)
from phifm.training.embedding import escolher_dispositivo, media_mascarada  # noqa: E402

log = logging.getLogger("minerar")

# Âncoras por bloco na busca. 256 × 667 mil escores em fp32 são ~683 MB — o teto
# confortável. Com 1024 seriam 2,7 GB, e a busca é feita na CPU justamente para
# não depender do alocador do DirectML, que não devolve memória num `del`
# (medido: 1 GB alocado continua marcado depois de liberar a referência).
BLOCO_BUSCA = 256


def _codificar_tudo(textos: list[str], mod, tok, dev, lote: int,
                    max_tokens: int, rotulo: str,
                    cache: Path | None = None,
                    ids: list[str] | None = None) -> np.ndarray:
    """Codifica em fp32 e devolve na CPU. O que vai para a busca é numpy.

    ⚠️ Com `cache`, os vetores vão para disco e uma relançada não recodifica.
    Existe porque esta mineração morreu depois de 35 min de codificação e teve de
    começar do zero — a codificação é ~80% do tempo total e é perfeitamente
    determinística dado (modelo, textos, max_tokens), então recomputá-la é
    desperdício puro.
    """
    if cache and cache.exists():
        try:
            # ⚠️ `with`, e as matrizes COPIADAS para fora. `np.load` de um `.npz`
            # devolve um `NpzFile` de leitura tardia que mantém o arquivo ABERTO —
            # e no Windows um arquivo aberto não pode ser substituído. Sem isto, uma
            # rejeição de cache derrubava o minerador com "acesso negado" ao tentar
            # gravar o cache novo, no fim de 35 min de codificação. Pego por teste.
            with np.load(cache, allow_pickle=False) as z:
                v = np.array(z["vetores"])
                ids_cache = [str(x) for x in z["ids"]]
        except Exception as exc:
            log.warning("  %s: cache ilegível (%s) — recodificando", rotulo,
                        type(exc).__name__)
        else:
            # ⚠️ Confere os IDS, um por um, não a contagem.
            #
            # A primeira versão validava por `len(v) == len(textos)`. O pool vem de
            # `unique()` do polars, que não garante ordem, então duas execuções com
            # a MESMA contagem tinham ordens diferentes — e o cache pareava o vetor
            # i com o documento j. O resultado eram negativos aleatórios com
            # aparência plausível: nenhuma exceção, nenhum aviso, e um treino
            # aprendendo lixo.
            #
            # Foi pego porque `descartados_por_serem_citacao_verdadeira` caiu de 131
            # para 0 entre duas execuções idênticas. O guarda existia por precaução
            # e virou o único sinal.
            if ids is not None and ids_cache == list(ids):
                log.info("  %s: %s vetores lidos do cache %s (ids conferidos)",
                         rotulo, f"{len(v):,}", cache.name)
                return v
            log.warning("  %s: cache não corresponde a estes documentos — "
                        "recodificando (%s vetores em cache, %s textos agora)",
                        rotulo, f"{len(v):,}", f"{len(textos):,}")
    vetores = []
    t0 = time.perf_counter()
    # ⚠️ `no_grad`, não `inference_mode`: o segundo colide com o buffer
    # `position_ids` do BERT no DirectML — "Cannot set version_counter for
    # inference tensor". É o que o `avaliar` do módulo de treino já usa.
    with torch.no_grad():
        for i in range(0, len(textos), lote):
            pedaco = textos[i:i + lote]
            b = tok(pedaco, padding="max_length", truncation=True,
                    max_length=max_tokens, return_tensors="pt")
            b = {k: v.to(dev) for k, v in b.items()}
            saida = mod(**b).last_hidden_state
            v = F.normalize(media_mascarada(saida, b["attention_mask"]).float(), dim=-1)
            vetores.append(v.cpu().numpy())
            if (i // lote) % 200 == 0 and i:
                feito = i + len(pedaco)
                taxa = feito / (time.perf_counter() - t0)
                log.info("  %s %s/%s · %.0f textos/s · faltam %.0f min",
                         rotulo, f"{feito:,}", f"{len(textos):,}", taxa,
                         (len(textos) - feito) / taxa / 60)
    V = np.vstack(vetores)
    if cache and ids is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.parent / (cache.name + ".tmp")
        with open(tmp, "wb") as f:
            # `np.array(ids)` infere string de largura fixa (ex. <U10 para
            # "1811.01641") — gravável sem pickle. `dtype=object` não é, e
            # `allow_pickle` NÃO é parâmetro do `savez`: passá-lo fazia o numpy
            # tentar salvar o próprio booleano como um array chamado "allow_pickle".
            np.savez(f, vetores=V, ids=np.array(ids))
        tmp.replace(cache)      # atômico: nunca meio-arquivo, como o estado do treino
        log.info("  %s: %s vetores em cache -> %s", rotulo, f"{len(V):,}", cache.name)
    return V


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    p.add_argument("--modelo", type=Path, default=Path("models/phiemb-minilm-melhor"),
                   help="o campeão do G1.1 — é ele que define o que é difícil")
    p.add_argument("--out", type=Path, default=Path("data/processed/negativos_dificeis"))
    p.add_argument("--max-pares", type=int, default=400_000,
                   help="mesmo volume do campeão, para a comparação isolar só a "
                        "dificuldade dos negativos")
    p.add_argument("--n-negativos", type=int, default=8,
                   help="negativos guardados por par")
    p.add_argument("--top-k", type=int, default=32,
                   help="candidatos recuperados antes de excluir os positivos; "
                        "precisa ser > n_negativos + citações por âncora")
    # ⚠️ O pool NÃO encolhe com `--max-pares`: mesmo minerando para 2 mil pares,
    # os 667 mil citados são codificados, e é aí que está o tempo. `--max-pool`
    # existe para o teste de fumaça terminar em minutos — e reduzir o pool torna
    # os negativos MENOS difíceis, então não é opção para a corrida de verdade.
    p.add_argument("--max-pool", type=int, default=None,
                   help="teto do pool de candidatos; só para teste")
    p.add_argument("--lote", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--dispositivo", default="auto",
                   choices=["auto", "cuda", "dml", "cpu"])
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    from phifm.core.sistema import impedir_suspensao, liberar_suspensao

    impedir_suspensao()
    try:
        return _minerar(a)
    finally:
        liberar_suspensao()


def _minerar(a) -> int:
    todos = pl.scan_parquet(a.pares / "pares_treino.parquet")

    # ── as arestas que vão ser mineradas ─────────────────────────────────────
    alvo = todos.head(a.max_pares).collect()
    log.info("minerando para %s pares · %s âncoras distintas",
             f"{len(alvo):,}", f"{alvo['arxiv_id'].n_unique():,}")

    # ── a lista COMPLETA de citados por âncora, da tabela inteira ────────────
    #
    # ⚠️ Da tabela INTEIRA, não do subconjunto. Ver a docstring: se A cita B e C e
    # só B está no subconjunto, C ainda é citação verdadeira e não pode virar
    # negativo. Ler daqui custa uma passada em 6,56 M linhas e evita ruído de
    # rótulo que pareceria sinal difícil.
    t0 = time.perf_counter()
    citados_por_ancora = dict(
        todos.group_by("arxiv_id")
        .agg(pl.col("arxiv_citado").unique().alias("cit"))
        .collect().iter_rows())
    log.info("proibições carregadas: %s âncoras, %.0f s",
             f"{len(citados_por_ancora):,}", time.perf_counter() - t0)

    # ── o pool de candidatos: os CITADOS, que é o universo da avaliação ──────
    # ⚠️ `maintain_order=True`. `unique()` do polars NÃO garante ordem, e sem isso
    # o pool sai numa ordem diferente a cada execução. Ver o aviso em
    # `_codificar_tudo`: combinado com um cache chaveado só pela contagem, isso
    # pareava o vetor i com o documento j — negativos aleatórios com aparência
    # plausível, sem erro nenhum.
    pool = (todos.select(["arxiv_citado", "positivo"])
            .unique(subset=["arxiv_citado"], maintain_order=True).collect())
    if a.max_pool:
        pool = pool.head(a.max_pool)
        log.warning("pool limitado a %s — negativos MENOS difíceis, use só para teste",
                    f"{a.max_pool:,}")
    ids_pool = pool["arxiv_citado"].to_list()
    log.info("pool de candidatos: %s documentos citados", f"{len(ids_pool):,}")

    ancoras = alvo.unique(subset=["arxiv_id"], maintain_order=True)
    ids_anc = ancoras["arxiv_id"].to_list()

    # ── codificação ─────────────────────────────────────────────────────────
    dev = escolher_dispositivo(a.dispositivo)
    tok = AutoTokenizer.from_pretrained(a.modelo)
    mod = AutoModel.from_pretrained(a.modelo).to(dev).eval()
    log.info("campeão carregado de %s (%.1f M params)", a.modelo,
             sum(x.numel() for x in mod.parameters()) / 1e6)

    cache_dir = a.out / "_cache_vetores"
    V_pool = _codificar_tudo(pool["positivo"].to_list(), mod, tok, dev, a.lote,
                             a.max_tokens, "pool",
                             cache=cache_dir / f"pool_{len(ids_pool)}.npz",
                             ids=ids_pool)
    V_anc = _codificar_tudo(ancoras["ancora"].to_list(), mod, tok, dev, a.lote,
                            a.max_tokens, "âncoras",
                            cache=cache_dir / f"ancoras_{len(ids_anc)}.npz",
                            ids=ids_anc)
    log.info("codificado: pool %s · âncoras %s", V_pool.shape, V_anc.shape)

    # ── busca por blocos, na CPU ────────────────────────────────────────────
    Vt = np.ascontiguousarray(V_pool.T)
    duros: dict[str, list[str]] = {}
    descartados = 0
    t0 = time.perf_counter()
    for i in range(0, len(ids_anc), BLOCO_BUSCA):
        bloco = V_anc[i:i + BLOCO_BUSCA]
        escores = bloco @ Vt
        # `argpartition` em vez de ordenar 667 mil por linha: só o top-K importa,
        # e ordenar tudo custaria ~20× mais.
        idx = np.argpartition(-escores, a.top_k, axis=1)[:, :a.top_k]
        for j, linha in enumerate(idx):
            anc = ids_anc[i + j]
            proibidos = citados_por_ancora.get(anc, [])
            proibidos = set(proibidos) | {anc}
            escolhidos = []
            # ordena só o top-K, por escore decrescente: o mais difícil primeiro
            for c in linha[np.argsort(-escores[j, linha])]:
                d = ids_pool[c]
                if d in proibidos:
                    descartados += 1
                    continue
                escolhidos.append(d)
                if len(escolhidos) == a.n_negativos:
                    break
            duros[anc] = escolhidos
        if (i // BLOCO_BUSCA) % 100 == 0 and i:
            taxa = (i + len(bloco)) / (time.perf_counter() - t0)
            log.info("  busca %s/%s · %.0f âncoras/s · faltam %.0f min",
                     f"{i + len(bloco):,}", f"{len(ids_anc):,}", taxa,
                     (len(ids_anc) - i) / taxa / 60)

    # ── saída, com o texto dos negativos resolvido ───────────────────────────
    texto_de = dict(zip(ids_pool, pool["positivo"].to_list(), strict=True))
    saida = alvo.with_columns(
        pl.col("arxiv_id").map_elements(
            lambda d: duros.get(d, []), return_dtype=pl.List(pl.Utf8)
        ).alias("negativos_id"))
    saida = saida.with_columns(
        pl.col("negativos_id").map_elements(
            lambda ns: [texto_de[n] for n in ns], return_dtype=pl.List(pl.Utf8)
        ).alias("negativos"))

    vazios = int((saida["negativos_id"].list.len() == 0).sum())
    curtos = int((saida["negativos_id"].list.len() < a.n_negativos).sum())

    a.out.mkdir(parents=True, exist_ok=True)
    saida.write_parquet(a.out / "pares_com_negativos.parquet", compression="zstd")

    meta = {
        "modelo_minerador": str(a.modelo),
        "max_pares": a.max_pares, "n_negativos": a.n_negativos, "top_k": a.top_k,
        "pool_candidatos": len(ids_pool),
        "ancoras_mineradas": len(ids_anc),
        "descartados_por_serem_citacao_verdadeira": descartados,
        "pares_sem_negativo": vazios,
        "pares_com_menos_que_n": curtos,
        "universo": "documentos CITADOS, o mesmo que a avaliação ordena",
        "exclusao": "todos os citados da âncora na tabela inteira, mais a própria âncora",
    }
    (a.out / "_mineracao.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 72)
    print(f"  negativos difíceis -> {a.out/'pares_com_negativos.parquet'}")
    for k, v in meta.items():
        print(f"    {k}: {v}")
    print("=" * 72)
    # Descartados é o número que valida a exclusão: se fosse ZERO, ou o modelo não
    # recupera as citações verdadeiras (ruim), ou a exclusão não está funcionando.
    me = gravar_manifesto_etapa(
        etapa="negativos_dificeis",
        descricao=("Negativos difíceis minerados pelo próprio ΦEmb campeão "
                   "(DOC-07 §4)"),
        raiz=a.out,
        entradas=[Entrada(caminho=str(a.modelo)),
                  Entrada(caminho=str(a.pares / "pares_treino.parquet"))],
        parametros={"script": "scripts/minerar_negativos.py", **meta},
        registros=len(saida))
    print(f"  manifesto da etapa: {me.manifesto_id[:16]}…")
    print("=" * 72)
    if descartados == 0:
        print("  ⚠️ ZERO descartes. Ou o campeão não recupera nenhuma citação")
        print("     verdadeira no top-K, ou a exclusão não está funcionando.")
        print("     Nos dois casos, conferir antes de treinar com isto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
