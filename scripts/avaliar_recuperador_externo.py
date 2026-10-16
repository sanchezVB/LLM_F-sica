#!/usr/bin/env python3
"""O recuperador contra os de fora, cada um como os autores mandam (DOC-13 §9.12).

    P = .venv-externo\\Scripts\\python.exe scripts\\avaliar_recuperador_externo.py
    P --nossa-tarefa                      # parte 1, decide: o pool da Tabela 6 (~1 h de CPU)
    P --scidocs --harness ferramentas\\scirepeval   # parte 2, relatada: SciDocs pelo SciRepEval

## A venv à parte (uma vez)

O SPECTER2 precisa do pacote `adapters`, que fixa a versão do `transformers` — instalá-lo
na `.venv-treino` arriscaria quebrar o treino. E a parte 2 precisa do `datasets` e do
`pytrec_eval`. Então:

    py -3.12 -m venv .venv-externo
    .venv-externo\\Scripts\\pip install torch adapters transformers "datasets>=2.16,<3" ^
        pytrec-eval-terrier scikit-learn scipy polars numpy tqdm

⚠️ `datasets<3`: o harness chama `load_dataset(..., trust_remote_code=False)`, e a partir
do `datasets` 4 esse argumento não existe mais (conferido no 5.1.0 em 2026-10-16: o
`IREvaluator` quebraria na primeira tarefa).
    git clone https://github.com/allenai/scirepeval ferramentas\\scirepeval
    git -C ferramentas\\scirepeval checkout e04594e6401cb68fb4ddf6daa801fb219cccc0bb

As versões instaladas vão para o agregado. Os modelos de fora e os conjuntos do
SciRepEval vêm do Hugging Face no primeiro uso.

## Por que os vetores são gerados aqui e não pelo `scirepeval.py`

O `Model.__call__` do harness faz `input_ids.to('cuda')` sem condição: numa máquina sem
CUDA (a do dono é AMD), ele levanta. Os vetores saem de `phifm.eval.recuperador_externo`
— cada modelo na configuração do cartão dele —, e a AVALIAÇÃO é a do harness: os pares
consulta-candidato, a distância euclidiana e o `pytrec_eval` do `IREvaluator`, no commit
fixo. O que se compara com o publicado é o que o harness calcula.

Os vetores ficam em `data/processed/recuperador_externo/`, fora do git, e a rodada é
retomável. Vão ao git os agregados `avaliacao/recuperador_externo_*.json`.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from phifm.eval import recuperador_externo as rx  # noqa: E402

PASTA = RAIZ / "data/processed/recuperador_externo"
AVALIACAO = RAIZ / "data/processed/avaliacao"
AGREGADO_POOL = AVALIACAO / "recuperador_externo_tabela6.json"
AGREGADO_SCIDOCS = AVALIACAO / "recuperador_externo_scidocs.json"
PARTE = 4096                      # documentos por arquivo de vetores no SciDocs


def _recusa_sem_aceite() -> None:
    if not rx.REGRA_9_12_ACEITA:
        raise SystemExit("a §9.12 do DOC-13 ainda é PROPOSTA: nada roda antes do aceite do dono "
                         "(REGRA_9_12_ACEITA em phifm.eval.recuperador_externo).")


def _versoes() -> dict:
    saida = {"python": sys.version.split()[0]}
    for pacote in ("torch", "transformers", "adapters", "datasets", "pytrec-eval-terrier",
                   "pytrec_eval", "numpy", "polars"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            saida[pacote] = importlib.metadata.version(pacote)
    return saida


def _sha(textos) -> str:
    h = hashlib.sha256()
    for t in textos:
        h.update(str(t).encode("utf-8", "replace"))
        h.update(b"\n")
    return h.hexdigest()[:16]


# ── parte 1: a nossa tarefa ───────────────────────────────────────────────────
def _vetores_do_pool(c: rx.Config, pool: pl.DataFrame, digesto: str, dispositivo: str):
    """`(va, vp)` de `c` no pool, do cache se o pool for o mesmo."""
    arq = PASTA / "pool" / f"{c.chave}.npz"
    if arq.exists():
        d = np.load(arq)
        if str(d["digesto"]) == digesto:
            return d["va"], d["vp"]
    print(f"codificando {c.nome} ({c.caminho}{' + ' + c.adaptador if c.adaptador else ''}, "
          f"{c.pooling}, {c.formato}, {c.max_tokens} tokens)...", flush=True)
    t0 = time.perf_counter()
    carregado = rx.carregar(c, dispositivo)
    va = rx.codificar(c, pool["titulo_a"].to_list(), pool["resumo_a"].to_list(),
                      carregado=carregado)
    vp = rx.codificar(c, pool["titulo_p"].to_list(), pool["resumo_p"].to_list(),
                      carregado=carregado)
    arq.parent.mkdir(parents=True, exist_ok=True)
    np.savez(arq, va=va, vp=vp, digesto=np.array(digesto))
    print(f"  {time.perf_counter() - t0:.0f} s", flush=True)
    return va, vp


def nossa_tarefa(a) -> None:
    _recusa_sem_aceite()
    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    try:
        pool = rx.pool_com_campos(val, pl.scan_parquet(a.spine))
    except ValueError as e:
        raise SystemExit(str(e)) from None
    digesto = _sha([*pool["ancora"].to_list(), *pool["positivo"].to_list()])
    pos = {}
    for c in rx.NA_NOSSA_TAREFA:
        va, vp = _vetores_do_pool(c, pool, digesto, a.dispositivo)
        pos[c] = rx.posicoes(va, vp, c.similaridade)
    reproducao = rx.conferir_reproducao(
        {c.caminho: float(rx.ndcg10_por_item(pos[c]).mean()) for c in (rx.MINILM, rx.NOSSO)})
    contra = {c.nome: rx.comparar(pos[rx.NOSSO], pos[c]) for c in rx.CIENTIFICOS}
    saida = {
        "regra": "DOC-13 §9.12, parte 1", "regra_aceita_pelo_dono": rx.REGRA_9_12_ACEITA,
        "pool": {"n": pool.height, "digesto": digesto, "o_que_e": "o pool da Tabela 6"},
        "reproducao_da_tabela_6 (guarda)": reproducao,
        "phiemb_menos_cada_cientifico_recomendado (decide)": contra,
        "decisao": rx.decidir(reproducao, contra),
        "relatado": {
            "o_handicap_do_protocolo (recomendado − Tabela 6)": {
                c.nome: rx.comparar(pos[c], pos[rx.no_protocolo_da_tabela_6(c)])
                for c in rx.CIENTIFICOS},
            "phiemb_menos_gte_large_a_512": rx.comparar(pos[rx.NOSSO], pos[rx.GTE_512])},
        "tabela": {c.nome: {**rx.metricas(pos[c]), "config": asdict(c)}
                   for c in rx.NA_NOSSA_TAREFA},
        "versoes": _versoes(), "dispositivo": a.dispositivo}
    AGREGADO_POOL.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: saida[k] for k in ("reproducao_da_tabela_6 (guarda)", "decisao")},
                     ensure_ascii=False, indent=1))
    for c in rx.NA_NOSSA_TAREFA:
        m = rx.metricas(pos[c])
        print(f"  {c.nome:<55} nDCG@10 {m['ndcg_10']:.4f} · recall@1 {m['recall_1']:.4f}")
    print(f"→ {AGREGADO_POOL.relative_to(RAIZ)}")


# ── parte 2: o SciDocs pelo SciRepEval ───────────────────────────────────────
def _harness(caminho: Path):
    """O `IREvaluator` do commit fixo. Recusa outro commit: a métrica é a dele."""
    try:
        commit = subprocess.run(["git", "-C", str(caminho), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        raise SystemExit(f"{caminho} não é um clone do SciRepEval (ver a docstring).") from None
    if commit != rx.SCIREPEVAL_COMMIT:
        raise SystemExit(f"o SciRepEval está em {commit[:12]}; a regra fixa "
                         f"{rx.SCIREPEVAL_COMMIT[:12]} (git checkout).")
    sys.path.insert(0, str(caminho.resolve()))
    from evaluation.evaluator import IREvaluator

    return IREvaluator


def _documentos():
    """Os artigos do SciDocs (proximidade), um por `doc_id`, na ordem em que vêm."""
    import datasets

    meta = datasets.load_dataset(*rx.SCIDOCS_META, split="evaluation")
    ids, titulos, resumos, vistos = [], [], [], set()
    for d in meta:
        if d["doc_id"] in vistos:
            continue
        vistos.add(d["doc_id"])
        ids.append(d["doc_id"])
        titulos.append(d.get("title") or "")
        resumos.append(d.get("abstract") or "")
    return ids, titulos, resumos


def _vetores_scidocs(c: rx.Config, ids, titulos, resumos, dispositivo: str) -> dict:
    """`{doc_id: vetor}` de `c`, em partes de `PARTE` — retomável."""
    pasta = PASTA / "scidocs" / c.chave
    pasta.mkdir(parents=True, exist_ok=True)
    digesto = _sha(ids)
    carregado = None
    t0, partes = time.perf_counter(), range(0, len(ids), PARTE)
    for k, i in enumerate(partes):
        arq = pasta / f"{digesto}_{k:04d}.npy"
        if arq.exists():
            continue
        carregado = carregado or rx.carregar(c, dispositivo)
        np.save(arq, rx.codificar(c, titulos[i:i + PARTE], resumos[i:i + PARTE],
                                  carregado=carregado).astype(np.float32))
        feitas = k + 1
        s = (time.perf_counter() - t0) / feitas
        print(f"[{c.nome} {feitas}/{len(partes)}] faltam ~{(len(partes) - feitas) * s / 60:.0f}"
              " min", flush=True)
    v = np.concatenate([np.load(pasta / f"{digesto}_{k:04d}.npy") for k in range(len(partes))])
    return dict(zip(ids, v, strict=True))


def scidocs(a) -> None:
    import datasets
    import pytrec_eval

    _recusa_sem_aceite()
    IREvaluator = _harness(a.harness)
    ids, titulos, resumos = _documentos()
    print(f"SciDocs: {len(ids)} artigos", flush=True)
    oficial, por_consulta = {}, {}
    for c in rx.NA_TAREFA_DELES:
        emb = _vetores_scidocs(c, ids, titulos, resumos, a.dispositivo)
        oficial[c.nome], por_consulta[c.nome] = {}, {}
        for tarefa, config in rx.SCIDOCS_TAREFAS.items():
            ev = IREvaluator(tarefa, rx.SCIDOCS_META, ("allenai/scirepeval_test", config), None,
                             rx.SCIDOCS_METRICAS)
            oficial[c.nome][tarefa] = {k: float(v) for k, v in ev.evaluate(emb).items()}
            qrels = ev.get_qc_pairs(datasets.load_dataset("allenai/scirepeval_test",
                                                          config)["test"])
            run = ev.retrieval(emb, qrels)
            for q, m in pytrec_eval.RelevanceEvaluator(qrels, {"ndcg"}).evaluate(run).items():
                por_consulta[c.nome][(tarefa, q)] = m["ndcg"]
    nosso, base = rx.NOSSO.nome, rx.MINILM.nome
    saida = {
        "regra": "DOC-13 §9.12, parte 2 (relatada)",
        "regra_aceita_pelo_dono": rx.REGRA_9_12_ACEITA,
        "harness": {"url": rx.SCIREPEVAL_URL, "commit": rx.SCIREPEVAL_COMMIT,
                    "o_que_vem_dele": "os pares, a distância euclidiana e as métricas "
                                      "(IREvaluator); os vetores são gerados aqui"},
        "artigos": len(ids),
        "oficial (×100, como o harness)": oficial,
        "media_das_4 (ndcg)": {n: round(float(np.mean([t["ndcg"] for t in d.values()])), 2)
                               for n, d in oficial.items()},
        "phiemb_menos_a_base (a leitura)": rx.comparar_por_consulta(por_consulta[nosso],
                                                                    por_consulta[base]),
        "phiemb_menos_cada_cientifico (relatado)": {
            c.nome: rx.comparar_por_consulta(por_consulta[nosso], por_consulta[c.nome])
            for c in rx.CIENTIFICOS},
        "configs": {c.nome: asdict(c) for c in rx.NA_TAREFA_DELES},
        "versoes": _versoes(), "dispositivo": a.dispositivo}
    AGREGADO_SCIDOCS.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: saida[k] for k in ("media_das_4 (ndcg)", "phiemb_menos_a_base (a leitura)")},
                     ensure_ascii=False, indent=1))
    print(f"→ {AGREGADO_SCIDOCS.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--nossa-tarefa", action="store_true")
    g.add_argument("--scidocs", action="store_true")
    p.add_argument("--harness", type=Path, default=RAIZ / "ferramentas/scirepeval")
    p.add_argument("--pares", type=Path, default=RAIZ / "data/processed/pares")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    p.add_argument("--dispositivo", default="cpu", choices=["cpu", "dml", "cuda"])
    a = p.parse_args()
    if a.nossa_tarefa:
        nossa_tarefa(a)
    else:
        scidocs(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
