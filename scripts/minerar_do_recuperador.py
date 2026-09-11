#!/usr/bin/env python3
"""Negativos com a DISTRIBUIÇÃO DA AVALIAÇÃO — seja ela qual for hoje.

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

O grupo é montado do **top-K que a avaliação realmente usa**. O positivo é a
citação verdadeira **que apareceu no top-K**, e os negativos são os outros
candidatos do mesmo top-K.

Com isso "estar no topo" deixa de prever o rótulo: o positivo também está no topo,
porque foi assim que ele entrou no grupo.

## ⚠️ E qual é essa distribuição mudou em 2026-09-08

Em 2026-08-24 a composição avaliada era a fusão RRF, então o grupo vinha do RRF.
No T1b2 a regra pré-registrada **tirou o BM25 da composição** — a cadeia passou a
ser `ΦEmb → ΦRank` — e o **default deste script mudou junto**: o grupo vem do
top-K do ΦEmb.

Não é uma reversão do conserto acima; é o mesmo princípio aplicado à composição
nova. Continuar minerando da fusão seria treinar o reordenador numa distribuição
que ele não vê mais — o defeito de 2026-08-24 com outra roupa. `--com-fusao`
reproduz os negativos antigos e existe só para isso.

## ⚠️ O par só existe quando o recuperador acerta

Recall@50 é **0,546** com o recuperador de 2026-09-08 (era 0,443 com a fusão do
recuperador antigo), então ~45% das âncoras não produzem grupo — o alvo nunca
chegou ao candidato. Isso é correto e não é perda: um reranker só age quando o
documento certo está no conjunto. Treinar nos casos em que ele não está seria
treinar num grupo sem resposta certa.

E a melhora do recuperador aparece **duas vezes**: **16.391** grupos em vez de
**14.289** para as mesmas 30.000 âncoras (0,546 contra 0,476), e na distribuição
que o modelo vai ver. O 0,443 acima é de outra medição, não da rodada de agosto —
aquela está no `registros` do manifesto por arquivo dela.

## ⚠️ Não é reprodutível bit a bit, e o motivo está na GPU

Duas execuções com a MESMA semente deram 16.362 e 16.391 grupos (2026-09-08,
RX 7600 por DirectML). A amostra de âncoras e o pool são idênticos — o que varia é
a **ordem do top-50**: a soma em float na GPU não é associativa, e documentos com
cosseno praticamente igual trocam de lugar. Perto da posição 50 isso decide se o
alvo entra no grupo ou não. Deu 0,18% de diferença.

Para negativos difíceis é inofensivo — eles são entrada de treino, não medição.
Mas duas coisas seguem dali: **o hash do parquet no manifesto não bate entre
execuções**, e **nenhum número deste script serve de medida**. O `recall@50` aqui
é diagnóstico; a medida é a do avaliador do T1b.

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
import contextlib
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

from phifm.core.console import utf8 as console_utf8  # noqa: E402
from phifm.core.modelos import RECUPERADOR  # noqa: E402
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

# ⚠️ No IMPORT, e não dentro do `main()`: o argparse imprime `--help` antes
# de qualquer código nosso, e `Φ` não existe em cp1252. Ver `phifm.core.console`.
console_utf8()

log = logging.getLogger("minerar-do-recuperador")


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
    p.add_argument("--emb", type=Path, default=Path(RECUPERADOR))
    # ⚠️ O nome do arquivo CARREGA a composição, e o default era fixo.
    #
    # Era `pares_do_recuperador.parquet` para as duas composições. Uma rodada com
    # `--com-fusao` sobrescreveria a sem fusão e vice-versa, e o parquet
    # resultante seria internamente consistente — não há hash que pegue isso,
    # porque o defeito não é corrupção, é o rótulo. É o mesmo problema do pacote
    # obsoleto que a `assinatura_do_manifesto` existe para pegar.
    #
    # Também protege os negativos de 2026-08-24, que são a evidência do ΦRank
    # instalado: eles ficam onde estão, no nome antigo.
    p.add_argument("--out", type=Path, default=None,
                   help="padrão: negativos_dificeis/pares_do_recuperador_"
                        "{denso,rrf}.parquet, conforme --com-fusao")
    p.add_argument("--max-ancoras", type=int, default=30000)
    p.add_argument("--universo", type=int, default=88807,
                   help="tamanho do pool; o padrão é o do avaliador do T1b")
    # ⚠️ O DEFAULT é minerar SEM o BM25, e isso mudou em 2026-09-08.
    #
    # Todo o argumento deste script é "treinar na distribuição do teste". Em
    # 2026-08-24 essa distribuição era a fusão RRF. No T1b2 a regra
    # pré-registrada tirou o BM25 da composição — a cadeia passou a ser
    # ΦEmb → ΦRank — e minerar da fusão voltaria a treinar o reordenador numa
    # distribuição que ele não vai ver. Seria o mesmo defeito de 2026-08-24 com
    # outra roupa, e é a causa mecânica provável de o ganho do ΦRank ter
    # encolhido de p=0,0081 para p=0,086.
    #
    # `--com-fusao` existe para reproduzir os negativos antigos, não para uso.
    p.add_argument("--com-fusao", action="store_true",
                   help="minera do RRF (ΦEmb+BM25) em vez do ΦEmb sozinho; era "
                        "o comportamento até 2026-09-08, quando a regra "
                        "pré-registrada do T1b2 tirou o BM25 da composição")
    p.add_argument("--profundidade", type=int, default=50,
                   help="top-K do RRF de onde o grupo sai; igual ao do avaliador")
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--lote", type=int, default=64)
    p.add_argument("--semente", type=int, default=17)
    p.add_argument("--dispositivo", default="auto",
                   choices=["auto", "cuda", "dml", "cpu"])
    a = p.parse_args()

    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    # ⚠️ O destino resolvido e criado ANTES do trabalho caro, não depois.
    # Embutir 119 mil sequências leva ~20 min; descobrir um destino inválido no
    # fim gastaria os 20 min para nada. Mesma lição de `amostrar_para_revisao.py`.
    if a.out is None:
        a.out = (Path("data/processed/negativos_dificeis")
                 / f"pares_do_recuperador_{'rrf' if a.com_fusao else 'denso'}"
                   ".parquet")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    log.info("composição: %s · destino %s",
             "RRF (ΦEmb+BM25)" if a.com_fusao else "ΦEmb sozinho", a.out)

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
    if a.com_fusao:
        t0 = time.perf_counter()
        bm = BM25().indexar(textos_pool)
        log.info("BM25 indexado em %.0f s", time.perf_counter() - t0)
    else:
        # Sem `--com-fusao` o índice não é consultado; construí-lo custaria
        # minutos sobre 89 mil documentos para nada.
        bm = None
        log.info("BM25 NÃO indexado: minerando do ΦEmb sozinho, que é a "
                 "composição decidida no T1b2 (2026-09-08)")

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
        ord_emb = top_k(Vq[i] @ Vt, a.profundidade)
        if a.com_fusao:
            ord_bm = top_k(bm.pontuar(consulta), a.profundidade)
            rrf = fundir_rrf(ord_emb, ord_bm)[:a.profundidade]
        else:
            # A composição decidida no T1b2: o candidato é o top-K DENSO.
            rrf = ord_emb[:a.profundidade]
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
        "composicao": "RRF (ΦEmb+BM25)" if a.com_fusao else "ΦEmb sozinho",
        "posicao_do_alvo_no_candidato": {
            "media": round(float(np.mean(posicoes)), 2) if posicoes else None,
            "p50": int(np.percentile(posicoes, 50)) if posicoes else None,
            "p90": int(np.percentile(posicoes, 90)) if posicoes else None,
        },
        "por_que": ("negativos com a distribuição da AVALIAÇÃO — que em "
                    "2026-08-24 era a fusão RRF e desde 2026-09-08 é o ΦEmb "
                    "sozinho, porque a regra pré-registrada do T1b2 tirou o BM25 "
                    "da composição. Os de minerar_negativos.py vinham do top-K do "
                    "denso MENOS o positivo, o que rotula negativo tudo que o "
                    "recuperador acha bom — e o ΦRank aprendeu a inverter o "
                    "recuperador"),
        "ainda_falta": ("passar por scripts/filtrar_cocitacao.py: co-citados com o "
                        "positivo continuam entrando como negativo"),
    }
    # ⚠️ O resumo leva o nome do PARQUET, e não um nome fixo.
    #
    # Era `_do_recuperador.json` para qualquer composição, e a rodada de
    # 2026-09-08 sobrescreveu em silêncio o resumo de 2026-08-24 — o diretório
    # não é versionado, então aquele número se perdeu (sobreviveu só porque
    # estava citado na docstring deste arquivo). Perder proveniência sem aviso é
    # o que o módulo de manifestos existe para impedir, e o resumo escrito à mão
    # ao lado dele estava fora dessa proteção.
    (a.out.parent / f"{a.out.stem}_resumo.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    gravar_manifesto_etapa(
        etapa="negativos_do_recuperador",
        descricao=("Negativos com a distribuição da avaliação: top-K do "
                   + ("RRF (ΦEmb+BM25)" if a.com_fusao else "ΦEmb sozinho")),
        # ⚠️ O ARQUIVO, nao o diretorio: `negativos_dificeis/` ja guarda a saida
        # de `minerar_negativos.py` e de `filtrar_cocitacao.py`, e gravar o
        # manifesto no diretorio apagaria o deles. Com o arquivo, o destino vira
        # `pares_do_recuperador.parquet_manifesto_etapa.json`.
        raiz=a.out,
        entradas=[Entrada(caminho=str(treino)), Entrada(caminho=str(a.emb))],
        parametros={"script": "scripts/minerar_do_recuperador.py",
                    "max_ancoras": a.max_ancoras, "universo": a.universo,
                    "profundidade": a.profundidade, "semente": a.semente,
                    # ⚠️ As ESTATISTICAS tambem, e nao so os argumentos.
                    #
                    # O manifesto por arquivo nao e compartilhado entre etapas,
                    # entao o que esta aqui sobrevive a um resumo solto
                    # sobrescrito. Ate 2026-09-08 este manifesto levava apenas
                    # os argumentos, e por isso a sobrescrita do resumo de
                    # agosto custou o recall@50 e a distribuicao de posicao
                    # daquela rodada de verdade -- restou so `registros`, que e
                    # a contagem de grupos. O `filtrar_cocitacao.py` ja fazia
                    # certo, e foi a comparacao entre os dois que mostrou isto.
                    **meta},
        registros=len(d))

    print()
    print("=" * 70)
    print(f"  {len(d):,} grupos de {len(ids_anc):,} âncoras "
          f"(recall@{a.profundidade} = {len(d)/max(len(ids_anc),1):.3f})")
    print(f"  {total_neg/n:.2f} negativos por grupo · alvo em posição média "
          f"{meta['posicao_do_alvo_no_candidato']['media']}")
    print(f"  composição: {meta['composicao']}")
    print(f"  -> {a.out}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
