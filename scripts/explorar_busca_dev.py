#!/usr/bin/env python3
"""Candidatos para melhorar a busca do assistente, nas perguntas de DESENVOLVIMENTO.

Exploratório (DOC-13 §9.2, proposta): nada aqui decide, e nada toca no conjunto de
teste. O conjunto de desenvolvimento existe para isto — olhar à vontade, escolher um
candidato, e só então levá-lo UMA vez ao teste.

    # fase 1 — os resumos hipotéticos (Qwen3-8B na GPU, ~45 min, retomável)
    .venv-treino\\Scripts\\python.exe scripts\\explorar_busca_dev.py --gerar
    # fase 2 — os candidatos (sem o modelo de linguagem)
    .venv-treino\\Scripts\\python.exe scripts\\explorar_busca_dev.py --medir

A métrica é mecânica, sem juiz: a posição do artigo P de onde saiu a pergunta.

| candidato | o quê |
|---|---|
| base | o sistema de hoje: 1 resumo hipotético, os 6 primeiros |
| k10 | o mesmo, lendo 10 |
| hyde3 | 3 resumos hipotéticos (sementes diferentes), média dos vetores |
| phirank | os 50 primeiros da base reordenados pelo ΦRank (PhysBERT, cross-encoder) |
| gte | os 50 primeiros da base reordenados pelo ΦEmb GTE-base@400k |
| fusao | RRF das ordens da base e do GTE nos 50 primeiros |
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

DIR = RAIZ / "data/processed/assistente"
AVALIACAO = RAIZ / "data/processed/avaliacao"
ITENS = DIR / "itens_dev_busca.json"
CONSULTAS = DIR / "dev_busca_consultas.jsonl"
POSICOES = DIR / "dev_busca_posicoes.jsonl"
N_HYDE = 3
PROFUNDIDADE = 50
K_RRF = 60
PHIRANK = RAIZ / "models/phirank-physbert-melhor"
GTE = RAIZ / "models/phiemb-gte-base-400k-melhor"


def carregar_itens() -> list[dict]:
    itens = json.loads(ITENS.read_text(encoding="utf-8"))
    sha = hashlib.sha256(json.dumps(itens, ensure_ascii=False).encode()).hexdigest()
    esperado = json.loads((AVALIACAO / "assistente_itens_dev_busca.json")
                          .read_text(encoding="utf-8"))["sha256_itens"]
    if sha != esperado:
        raise SystemExit("os itens de desenvolvimento não são os versionados.")
    return itens


def _ler_jsonl(arq: Path) -> dict[tuple, dict]:
    if not arq.exists():
        return {}
    saida = {}
    for linha in arq.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            d = json.loads(linha)
            saida[(d["estrato"], d["arxiv_id"])] = d
    return saida


def gerar(itens: list[dict]) -> None:
    """Os `N_HYDE` resumos hipotéticos de cada pergunta. O primeiro usa a semente da
    regra — é o MESMO texto que o assistente geraria (ver `Assistente.consulta_hipotetica`)."""
    from phifm.eval.assistente import SEMENTE
    from phifm.rag.assistente import SISTEMA_HYDE
    from phifm.rag.llm import ModeloLocal

    feitos = _ler_jsonl(CONSULTAS)
    modelo = ModeloLocal()
    if not modelo.no_ar():
        print("subindo o llama-server (Qwen3-8B na GPU)...", flush=True)
        modelo.iniciar(log=RAIZ / "data/processed/llama_server.log")
    t0, n = time.perf_counter(), 0
    try:
        with CONSULTAS.open("a", encoding="utf-8") as f:
            for it in itens:
                chave = (it["estrato"], it["arxiv_id"])
                if chave in feitos:
                    continue
                consultas = []
                for i in range(N_HYDE):
                    texto = modelo.gerar(SISTEMA_HYDE, it["pergunta"], max_tokens=260,
                                         temperatura=0.3, semente=SEMENTE + i)
                    consultas.append(" ".join(texto.split()) or it["pergunta"])
                f.write(json.dumps({"estrato": chave[0], "arxiv_id": chave[1],
                                    "consultas": consultas}, ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                if n % 20 == 0:
                    s = (time.perf_counter() - t0) / n
                    print(f"{len(feitos) + n}/{len(itens)} · {s:.0f} s/item · faltam "
                          f"~{(len(itens) - len(feitos) - n) * s / 60:.0f} min", flush=True)
    finally:
        modelo.parar()


def _posicao(escores: np.ndarray, linha_p: int) -> int:
    return int((escores > escores[linha_p]).sum()) + 1


def _no_topo(escores: np.ndarray, i_p: int | None, pos_base: int) -> int:
    """Posição de P depois de reordenar os 50; fora dos 50, fica a da base."""
    if i_p is None:
        return pos_base
    return int((escores > escores[i_p]).sum()) + 1


def medir(itens: list[dict], spine: Path, indice: Path) -> None:
    import polars as pl
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    from phifm.eval.encoders import _codificar
    from phifm.retrieval.indice import MAX_TOKENS, Busca
    from phifm.training.pairs import textos_de_documentos

    consultas = _ler_jsonl(CONSULTAS)
    faltam = [i for i in itens if (i["estrato"], i["arxiv_id"]) not in consultas]
    if faltam:
        raise SystemExit(f"{len(faltam)} itens sem resumo hipotético: rode --gerar.")
    busca = Busca(indice, dispositivo="auto")
    linha = {a: i for i, a in enumerate(busca.docs["arxiv_id"].to_list())}
    ids_docs = busca.docs["arxiv_id"].to_list()

    tok_r = AutoTokenizer.from_pretrained(PHIRANK)
    mod_r = AutoModelForSequenceClassification.from_pretrained(PHIRANK).to(busca.dev).eval()
    tok_g = AutoTokenizer.from_pretrained(GTE)
    mod_g = AutoModel.from_pretrained(GTE, attn_implementation="eager").to(busca.dev).eval()
    max_r = json.loads((PHIRANK / "phirank.json").read_text(encoding="utf-8"))["config"]["max_tokens"]

    feitos = _ler_jsonl(POSICOES)
    t0, n = time.perf_counter(), 0
    with POSICOES.open("a", encoding="utf-8") as f:
        for it in itens:
            chave = (it["estrato"], it["arxiv_id"])
            if chave in feitos:
                continue
            cs = consultas[chave]["consultas"]
            lp = linha[it["arxiv_id"]]
            vs = [busca.embutir(c) for c in cs]
            s0 = busca.pontuar(vs[0])
            pos_base = _posicao(s0, lp)
            media = np.mean(vs, axis=0)
            pos_hyde3 = _posicao(busca.pontuar(media / np.linalg.norm(media)), lp)

            topo = np.argsort(-s0)[:PROFUNDIDADE]
            ids_topo = [ids_docs[j] for j in topo]
            textos = dict(textos_de_documentos(pl.scan_parquet(spine)
                                               .filter(pl.col("arxiv_id").is_in(ids_topo)))
                          .collect().iter_rows())
            docs = [textos.get(a, "") for a in ids_topo]
            dentro = it["arxiv_id"] in ids_topo
            with torch.no_grad():
                b = tok_r([cs[0]] * len(docs), docs, padding=True, truncation=True,
                          max_length=max_r, return_tensors="pt")
                logit = mod_r(**{k: v.to(busca.dev) for k, v in b.items()}).logits
                esc_r = logit.float().view(-1).cpu().numpy()
            q_g = _codificar(mod_g, tok_g, [cs[0]], busca.dev, MAX_TOKENS, 1)[0].numpy()
            d_g = _codificar(mod_g, tok_g, docs, busca.dev, MAX_TOKENS, 16).numpy()
            esc_g = d_g @ q_g

            i_p = ids_topo.index(it["arxiv_id"]) if dentro else None
            ordem_base = np.arange(len(docs))
            ordem_gte = np.argsort(np.argsort(-esc_g))
            esc_rrf = 1 / (K_RRF + ordem_base + 1) + 1 / (K_RRF + ordem_gte + 1)
            d = {"estrato": chave[0], "arxiv_id": chave[1], "base": pos_base,
                 "hyde3": pos_hyde3, "phirank": _no_topo(esc_r, i_p, pos_base),
                 "gte": _no_topo(esc_g, i_p, pos_base),
                 "fusao": _no_topo(esc_rrf, i_p, pos_base)}
            f.write(json.dumps(d) + "\n")
            f.flush()
            feitos[chave] = d
            n += 1
            if n % 25 == 0:
                print(f"{len(feitos)}/{len(itens)} · {(time.perf_counter() - t0) / n:.1f} s/item",
                      flush=True)

    resumir(itens, feitos)


# ── segunda leva (2026-10-09): depois de o ΦRank nos 50 ser adotado ───────────────────
POSICOES2 = DIR / "dev_busca_posicoes2.jsonl"
FUNDO = 200          # até onde o ΦRank é testado


def medir2(itens: list[dict], spine: Path, indice: Path) -> None:
    """Candidatos contra o sistema ADOTADO (ΦRank nos 50 primeiros, 6 fontes):

    | campo | o quê |
    |---|---|
    | rr50 | o sistema de hoje |
    | rr100, rr200 | reordenar mais fundo |
    | rr50_256 | os 50, com os pares cortados em 256 tokens (mais barato) |
    | rr50_q3 | os 50, com a média dos escores das 3 consultas hipotéticas |
    | h3_rr50, h3_rr100 | a busca com a média dos 3 vetores, e o ΦRank por cima |

    O cross-encoder pontua cada par (consulta, documento) sozinho: os escores dos 200
    primeiros servem para os 50, os 100 e os 200."""
    import polars as pl
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from phifm.retrieval.indice import Busca
    from phifm.training.pairs import textos_de_documentos

    consultas = _ler_jsonl(CONSULTAS)
    busca = Busca(indice, dispositivo="auto")
    ids_docs = busca.docs["arxiv_id"].to_list()
    linha = {a: i for i, a in enumerate(ids_docs)}
    tok = AutoTokenizer.from_pretrained(PHIRANK)
    mod = AutoModelForSequenceClassification.from_pretrained(PHIRANK).to(busca.dev).eval()
    max_r = json.loads((PHIRANK / "phirank.json").read_text(encoding="utf-8"))["config"]["max_tokens"]

    def pontuar(consulta: str, docs: list[str], max_len: int) -> np.ndarray:
        saida = []
        with torch.no_grad():
            for i in range(0, len(docs), 8):   # 25 estourou a VRAM com a placa dividida
                lote = docs[i:i + 8]
                b = tok([consulta] * len(lote), lote, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
                logit = mod(**{k: v.to(busca.dev) for k, v in b.items()}).logits
                saida.append(logit.float().view(-1).cpu().numpy())
        return np.concatenate(saida)

    def depois(ordem_ids: list[str], escores: np.ndarray, p_id: str, pos_fora: int) -> int:
        """Posição de P depois de reordenar `ordem_ids` por `escores`; fora deles, a de antes."""
        if p_id not in ordem_ids:
            return pos_fora
        i = ordem_ids.index(p_id)
        return int((escores > escores[i]).sum()) + 1

    feitos = _ler_jsonl(POSICOES2)
    t0, n = time.perf_counter(), 0
    with POSICOES2.open("a", encoding="utf-8") as f:
        for it in itens:
            chave = (it["estrato"], it["arxiv_id"])
            if chave in feitos:
                continue
            cs, p_id = consultas[chave]["consultas"], it["arxiv_id"]
            lp = linha[p_id]
            vs = [busca.embutir(c) for c in cs]
            s0 = busca.pontuar(vs[0])
            media = np.mean(vs, axis=0)
            s3 = busca.pontuar(media / np.linalg.norm(media))
            pos0, pos3 = _posicao(s0, lp), _posicao(s3, lp)
            topo0 = [ids_docs[j] for j in np.argsort(-s0)[:FUNDO]]
            topo3 = [ids_docs[j] for j in np.argsort(-s3)[:100]]
            todos = sorted(set(topo0) | set(topo3))
            textos = dict(textos_de_documentos(pl.scan_parquet(spine)
                                               .filter(pl.col("arxiv_id").is_in(todos)))
                          .collect().iter_rows())
            esc = dict(zip(todos, pontuar(cs[0], [textos.get(a, "") for a in todos], max_r),
                           strict=True))
            e0 = np.array([esc[a] for a in topo0])
            e3 = np.array([esc[a] for a in topo3])
            t50 = [textos.get(a, "") for a in topo0[:50]]
            e256 = pontuar(cs[0], t50, 256)
            eq3 = (e0[:50] + pontuar(cs[1], t50, max_r) + pontuar(cs[2], t50, max_r)) / 3
            d = {"estrato": chave[0], "arxiv_id": p_id, "base": pos0, "hyde3": pos3,
                 "rr50": depois(topo0[:50], e0[:50], p_id, pos0),
                 "rr100": depois(topo0[:100], e0[:100], p_id, pos0),
                 "rr200": depois(topo0, e0, p_id, pos0),
                 "rr50_256": depois(topo0[:50], e256, p_id, pos0),
                 "rr50_q3": depois(topo0[:50], eq3, p_id, pos0),
                 "h3_rr50": depois(topo3[:50], e3[:50], p_id, pos3),
                 "h3_rr100": depois(topo3, e3, p_id, pos3)}
            f.write(json.dumps(d) + "\n")
            f.flush()
            feitos[chave] = d
            n += 1
            if n % 20 == 0:
                print(f"{len(feitos)}/{len(itens)} · {(time.perf_counter() - t0) / n:.1f} s/item",
                      flush=True)
    resumir2(itens, feitos)


def resumir2(itens: list[dict], feitos: dict[tuple, dict]) -> None:
    candidatos = {"rr50 (hoje)": ("rr50", 6), "rr50 · 10 fontes": ("rr50", 10),
                  "rr100": ("rr100", 6), "rr200": ("rr200", 6),
                  "rr200 · 10 fontes": ("rr200", 10), "rr50 · 256 tokens": ("rr50_256", 6),
                  "rr50 · 3 consultas": ("rr50_q3", 6), "hyde3 + rr50": ("h3_rr50", 6),
                  "hyde3 + rr100": ("h3_rr100", 6)}
    rng = np.random.default_rng(20261009)
    saida = {"exploratorio": True, "conjunto": "desenvolvimento (DOC-13 §9.2/§9.3)",
             "referencia": "rr50, o sistema adotado em 2026-10-07", "por_estrato": {}}
    for estrato in ("primario", "pos_corte", "todos"):
        ds = [feitos[(i["estrato"], i["arxiv_id"])] for i in itens
              if estrato == "todos" or i["estrato"] == estrato]
        ref = np.array([d["rr50"] <= 6 for d in ds], dtype=float)
        tabela = {"n": len(ds), "teto_@50": round(float(np.mean([d["base"] <= 50 for d in ds])), 4),
                  "teto_@200": round(float(np.mean([d["base"] <= 200 for d in ds])), 4)}
        for nome, (campo, k) in candidatos.items():
            x = np.array([d[campo] <= k for d in ds], dtype=float)
            dif = x - ref
            reps = dif[rng.integers(0, len(dif), (5000, len(dif)))].mean(axis=1)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            tabela[nome] = {"recall": round(float(x.mean()), 4),
                            "menos_hoje": round(float(dif.mean()), 4),
                            "ic95": [round(float(lo), 4), round(float(hi), 4)],
                            "ganha": int((dif > 0).sum()), "perde": int((dif < 0).sum())}
        saida["por_estrato"][estrato] = tabela
    arq = AVALIACAO / "assistente_busca_dev2.json"
    arq.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    for estrato, tabela in saida["por_estrato"].items():
        print(f"\n{estrato} (n={tabela['n']}, teto@50 {tabela['teto_@50']}, "
              f"teto@200 {tabela['teto_@200']})")
        for nome in candidatos:
            c = tabela[nome]
            print(f"  {nome:20s} {c['recall']:.3f}  Δ {c['menos_hoje']:+.3f} "
                  f"[{c['ic95'][0]:+.3f}; {c['ic95'][1]:+.3f}]  +{c['ganha']}/−{c['perde']}")
    print(f"→ {arq.relative_to(RAIZ)}")


def resumir(itens: list[dict], feitos: dict[tuple, dict]) -> None:
    candidatos = {"base": ("base", 6), "k10": ("base", 10), "hyde3": ("hyde3", 6),
                  "phirank": ("phirank", 6), "gte": ("gte", 6), "fusao": ("fusao", 6)}
    rng = np.random.default_rng(20260929)
    saida = {"exploratorio": True, "conjunto": "desenvolvimento (DOC-13 §9.2)",
             "profundidade_reordenada": PROFUNDIDADE, "por_estrato": {}}
    for estrato in ("primario", "pos_corte", "todos"):
        ds = [feitos[(i["estrato"], i["arxiv_id"])] for i in itens
              if estrato == "todos" or i["estrato"] == estrato]
        base = np.array([d["base"] <= 6 for d in ds], dtype=float)
        linha = {"n": len(ds), "recall_base_@50": round(float(np.mean([d["base"] <= 50 for d in ds])), 4)}
        for nome, (campo, k) in candidatos.items():
            x = np.array([d[campo] <= k for d in ds], dtype=float)
            dif = x - base
            reps = dif[rng.integers(0, len(dif), (5000, len(dif)))].mean(axis=1)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            linha[nome] = {"recall": round(float(x.mean()), 4),
                           "menos_base": round(float(dif.mean()), 4),
                           "ic95": [round(float(lo), 4), round(float(hi), 4)],
                           "ganha": int((dif > 0).sum()), "perde": int((dif < 0).sum())}
        saida["por_estrato"][estrato] = linha
    arq = AVALIACAO / "assistente_busca_dev.json"
    arq.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    for estrato, linha in saida["por_estrato"].items():
        print(f"\n{estrato} (n={linha['n']}, base@50 {linha['recall_base_@50']})")
        for nome in candidatos:
            c = linha[nome]
            print(f"  {nome:8s} {c['recall']:.3f}  Δ {c['menos_base']:+.3f} "
                  f"[{c['ic95'][0]:+.3f}; {c['ic95'][1]:+.3f}]  +{c['ganha']}/−{c['perde']}")
    print(f"→ {arq.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gerar", action="store_true")
    p.add_argument("--medir", action="store_true")
    p.add_argument("--medir2", action="store_true",
                   help="segunda leva: candidatos contra o ΦRank já adotado")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    p.add_argument("--indice", type=Path, default=RAIZ / "data/processed/indice_busca")
    a = p.parse_args()
    itens = carregar_itens()
    if a.gerar:
        gerar(itens)
    if a.medir:
        medir(itens, a.spine, a.indice)
    if a.medir2:
        medir2(itens, a.spine, a.indice)
    if not (a.gerar or a.medir or a.medir2):
        p.error("--gerar e/ou --medir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
