#!/usr/bin/env python3
"""O piloto dos trechos do texto completo (PLANO-trechos-do-texto-completo §4, etapa 0).

A pergunta: **com trechos, o artigo certo sobe?** Sem indexar o corpus — só os concorrentes
de verdade de cada pergunta de DESENVOLVIMENTO. Exploratório: nada aqui toca o teste.

    P = .venv-treino\\Scripts\\python.exe scripts\\piloto_trechos.py
    P --auditar       # M1: 10.000 trechos sorteados, zero cortes em equação (CPU, ~1 min)
    P --candidatos    # os concorrentes de cada pergunta (a busca de hoje, ~1 min)
    P --cortar        # os trechos dos concorrentes (CPU, ~15 min, retomável)
    P --embutir       # os vetores dos trechos (GPU, ~1,5 h, retomável)
    P --pontuar       # o melhor trecho de cada (pergunta, artigo) (CPU, ~2 min)
    P --reordenar     # o ΦRank nos candidatos (GPU, ~40 min, retomável)
    P --resumir       # a tabela

## O que é comparado

Para cada pergunta, os artigos que a busca de hoje põe na frente do artigo certo P (os
`FUNDO` primeiros; mais fundo se P estiver mais longe, até `FUNDO_MAX`) e o próprio P.
Cada artigo com texto completo ganha, além do escore do resumo, o do seu melhor trecho:

| representação | o escore do artigo |
|---|---|
| resumo (hoje) | cosseno da consulta com `título. resumo` |
| melhor trecho — opção A | o maior cosseno entre TODOS os trechos |
| introdução e conclusão — opção B | o maior, só entre os trechos dessas seções |
| combinada | o resumo junto com o melhor trecho (máximo, soma, ou união das duas listas) |

O trecho é embutido como `título. trecho` — o formato em que o encoder foi treinado.
Por cima, o ΦRank reordena os 50 primeiros e ficam 6, como no produto.

⚠️ **Otimista num ponto, declarado.** Artigo sem texto completo, e artigo fora do conjunto
de concorrentes, entram só com o escore do resumo: um artigo que está além dos
concorrentes e que subiria com os seus trechos não é visto subindo.

Os trechos e os vetores ficam em `data/processed/assistente/piloto_trechos/` (fora do git,
~2 GB temporários). Vai para o git só o agregado.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from explorar_busca_dev import (  # noqa: E402
    CONSULTAS,
    PHIRANK,
    _ler_jsonl,
    _posicao,
    carregar_itens,
)

DIR = RAIZ / "data/processed/assistente"
PILOTO = DIR / "piloto_trechos"
AVALIACAO = RAIZ / "data/processed/avaliacao"
TEXTOS = RAIZ / "data/processed/redpajama_fisica"
INDICE = RAIZ / "data/processed/indice_busca"
SPINE = RAIZ / "data/processed/spine.parquet"

FUNDO = 300            # concorrentes por pergunta, pela busca de hoje
FUNDO_MAX = 1500       # …ou até P, se P estiver mais longe (e até aqui)
N_ESCORES = 3000       # escores do resumo guardados por pergunta, para contar quem fica à frente
MAX_TITULO = 28        # tokens do título na frente de cada trecho
ORCAMENTO = 188        # 192 do encoder − 2 especiais − o ". " entre título e trecho
N_PHIRANK = 100        # os primeiros de cada representação recebem escore do ΦRank
SEMENTE = 20261009


def _contador(modelo: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(modelo)

    def contar(textos: list[str]) -> list[int]:
        if not textos:
            return []
        return [len(x) for x in tok(textos, add_special_tokens=False, verbose=False)["input_ids"]]

    return contar


def _modelo_do_indice() -> str:
    return json.loads((INDICE / "manifesto.json").read_text(encoding="utf-8"))["modelo"]


# ── M1 ───────────────────────────────────────────────────────────────────────────────
def auditar(n: int = 10_000, n_partes: int = 8, docs_por_parte: int = 40) -> None:
    """M1 (DOC-13 §3): zero cortes dentro de ambiente matemático em 10.000 trechos.

    Um trecho com equação sem par é contado nos dois casos; o que a M1 cobra do cortador
    é o primeiro: o corte num documento ÍNTEGRO. No segundo, a equação já chega sem par
    no texto de origem — o autor a abre com uma macro cujo preâmbulo o RedPajama cortou,
    ou o RedPajama perdeu caracteres — e o cortador não tem como ver onde ela começa.
    É reportado à parte, não escondido."""
    import polars as pl

    from phifm.retrieval.trechos import cortar, cortes_em_equacao, documento_integro, limpar

    contar = _contador(_modelo_do_indice())
    rng = np.random.default_rng(SEMENTE)
    partes = sorted(TEXTOS.glob("part-*.parquet"))
    escolhidas = [partes[i] for i in sorted(rng.choice(len(partes), n_partes, replace=False))]
    trechos: list[tuple[str, bool, object]] = []        # (arxiv_id, íntegro, trecho)
    n_docs = n_integros = 0
    for p in escolhidas:
        df = pl.read_parquet(p)
        for j in sorted(rng.choice(df.height, docs_por_parte, replace=False)):
            a, texto = df.row(int(j))
            integro = documento_integro(limpar(texto))
            n_docs += 1
            n_integros += integro
            trechos += [(a, integro, t) for t in cortar(texto, contar, limite=160)]
    amostra = [trechos[i] for i in rng.permutation(len(trechos))[:n]]
    cortes = [(a, integro, t) for a, integro, t in amostra if cortes_em_equacao(t.texto)]
    tam = np.array([t.n_tokens for _, _, t in amostra])
    saida = {
        "regra": "DOC-13 §3 · M1: zero cortes dentro de ambiente matemático em 10.000 amostras",
        "semente": SEMENTE, "limite_tokens": 160,
        "partes": [p.name for p in escolhidas], "documentos": n_docs,
        "documentos_integros": n_integros, "trechos_sorteados": len(amostra),
        "trechos_com_equacao_sem_par": len(cortes),
        "…em_documento_integro (o que a M1 cobra)": sum(1 for _, i, _ in cortes if i),
        "…em_documento_que_ja_chega_com_equacao_sem_par": sum(1 for _, i, _ in cortes if not i),
        "documentos_com_corte": sorted({a for a, _, _ in cortes}),
        "trechos_com_equacao": round(float(np.mean([t.tem_equacao for _, _, t in amostra])), 4),
        "excedem_o_limite (equação maior que o trecho)": round(
            float(np.mean([t.excede for _, _, t in amostra])), 4),
        "tokens": {"media": round(float(tam.mean()), 1),
                   "p10": int(np.percentile(tam, 10)), "p50": int(np.percentile(tam, 50)),
                   "p90": int(np.percentile(tam, 90)), "max": int(tam.max())},
        "trechos_por_documento": round(len(trechos) / n_docs, 1),
        "tipos": {k: sum(1 for _, _, t in amostra if t.tipo == k)
                  for k in ("introducao", "corpo", "conclusao")},
    }
    arq = AVALIACAO / "trechos_auditoria_m1.json"
    arq.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(saida, ensure_ascii=False, indent=2))
    for a, integro, t in cortes[:6]:
        print(f"\n--- {a} · {'ÍNTEGRO' if integro else 'sem par na origem'}\n{t.texto[:500]}")
    print(f"→ {arq.relative_to(RAIZ)}")


# ── os concorrentes ──────────────────────────────────────────────────────────────────
def _itens() -> list[dict]:
    return [i for i in carregar_itens() if i["estrato"] == "primario"]


def candidatos() -> None:
    import polars as pl

    from phifm.retrieval.indice import Busca

    PILOTO.mkdir(parents=True, exist_ok=True)
    itens = _itens()
    consultas = _ler_jsonl(CONSULTAS)
    com_texto = set(pl.scan_parquet(TEXTOS / "part-*.parquet").select("arxiv_id")
                    .collect()["arxiv_id"].to_list())
    busca = Busca(INDICE, dispositivo="auto")
    ids_docs = busca.docs["arxiv_id"].to_list()
    linha = {a: i for i, a in enumerate(ids_docs)}
    registros, vetores, escores = [], [], []
    a_cortar: set[str] = set()
    for it in itens:
        p = it["arxiv_id"]
        v = busca.embutir(consultas[(it["estrato"], p)]["consultas"][0])
        s = busca.pontuar(v)
        pos = _posicao(s, linha[p])
        ordem = np.argsort(-s)[:N_ESCORES]
        topo = [ids_docs[j] for j in ordem[:min(max(FUNDO, pos), FUNDO_MAX)]]
        registros.append({"arxiv_id": p, "pos_base": pos, "s_p": float(s[linha[p]]),
                          "p_tem_texto": p in com_texto, "topo": topo})
        vetores.append(v)
        escores.append(s[ordem].astype(np.float32))
        a_cortar |= {a for a in topo if a in com_texto} | ({p} & com_texto)
    np.save(PILOTO / "consultas.npy", np.stack(vetores))
    np.save(PILOTO / "escores_topo.npy", np.stack(escores))
    (PILOTO / "candidatos.json").write_text(json.dumps(registros), encoding="utf-8")
    (PILOTO / "docs_a_cortar.json").write_text(json.dumps(sorted(a_cortar)), encoding="utf-8")
    pos = np.array([r["pos_base"] for r in registros])
    print(f"{len(itens)} perguntas · P com texto completo: "
          f"{sum(r['p_tem_texto'] for r in registros)} · P entre os 50: {(pos <= 50).sum()} · "
          f"além de {FUNDO}: {(pos > FUNDO).sum()} · além de {FUNDO_MAX}: {(pos > FUNDO_MAX).sum()}")
    print(f"concorrentes: {sum(len(r['topo']) for r in registros):,} · artigos distintos com "
          f"texto a cortar: {len(a_cortar):,}")


# ── os trechos ───────────────────────────────────────────────────────────────────────
def _titulo_curto(titulo: str, contar) -> tuple[str, int]:
    palavras = " ".join((titulo or "").split()).split(" ")
    while palavras and contar([" ".join(palavras)])[0] > MAX_TITULO:
        palavras.pop()
    t = " ".join(palavras)
    return t, contar([t])[0] if t else 0


def cortar_tudo() -> None:
    import polars as pl

    from phifm.retrieval.trechos import cortar

    saida = PILOTO / "trechos"
    saida.mkdir(parents=True, exist_ok=True)
    alvo = json.loads((PILOTO / "docs_a_cortar.json").read_text(encoding="utf-8"))
    titulos = dict(pl.read_parquet(INDICE / "documentos.parquet", columns=["arxiv_id", "title"])
                   .filter(pl.col("arxiv_id").is_in(alvo)).iter_rows())
    contar = _contador(_modelo_do_indice())
    partes = sorted(TEXTOS.glob("part-*.parquet"))
    # um artigo que aparece em duas partes é cortado na primeira
    vistos: set[str] = set()
    t0, n_docs, n_trechos = time.perf_counter(), 0, 0
    for p in partes:
        destino = saida / p.name
        if destino.exists():
            vistos |= set(pl.read_parquet(destino, columns=["arxiv_id"])["arxiv_id"].to_list())
            continue
        df = pl.scan_parquet(p).filter(pl.col("arxiv_id").is_in(alvo)).collect()
        linhas = []
        for a, texto in df.iter_rows():
            if a in vistos:
                continue
            vistos.add(a)
            titulo, n_t = _titulo_curto(titulos.get(a, ""), contar)
            for i, t in enumerate(cortar(texto, contar, limite=ORCAMENTO - n_t)):
                linhas.append((a, titulo, i, t.tipo, t.n_tokens, t.tem_equacao, t.excede, t.texto))
            n_docs += 1
        pl.DataFrame(linhas, orient="row", schema={
            "arxiv_id": pl.String, "titulo": pl.String, "i": pl.Int32, "tipo": pl.String,
            "n_tokens": pl.Int32, "tem_equacao": pl.Boolean, "excede": pl.Boolean,
            "texto": pl.String}).write_parquet(destino.with_suffix(".tmp"))
        destino.with_suffix(".tmp").rename(destino)
        n_trechos += len(linhas)
        print(f"{p.name}: {df.height} artigos · {n_docs:,} cortados, {n_trechos:,} trechos · "
              f"{time.perf_counter() - t0:.0f} s", flush=True)
    total = pl.scan_parquet(saida / "part-*.parquet").select(
        pl.len().alias("trechos"), pl.col("arxiv_id").n_unique().alias("artigos")).collect()
    print(f"total: {total['trechos'][0]:,} trechos de {total['artigos'][0]:,} artigos "
          f"(pedidos: {len(alvo):,})")


def _mapa():
    """Todos os trechos, na ordem dos arquivos: a linha é a posição do vetor."""
    import polars as pl

    return pl.concat([pl.read_parquet(p) for p in sorted((PILOTO / "trechos").glob("part-*.parquet"))])


def embutir(lote: int = 64, bloco: int = 20_000) -> None:
    """Os vetores de `título. trecho`, pela MESMA conta do índice (`_codificar`, 192 tokens)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from phifm.eval.encoders import _codificar
    from phifm.retrieval.indice import MAX_TOKENS, _silenciar_barras
    from phifm.training.embedding import escolher_dispositivo

    df = _mapa()
    n = df.height
    modelo = _modelo_do_indice()
    _silenciar_barras()
    dev = escolher_dispositivo("auto")
    tok = AutoTokenizer.from_pretrained(modelo)
    mod = AutoModel.from_pretrained(modelo, attn_implementation="eager").to(dev).eval()
    dim = int(mod.config.hidden_size)
    arq, prog_arq = PILOTO / "vetores.npy", PILOTO / "progresso.json"
    feitos = 0
    if prog_arq.exists():
        prog = json.loads(prog_arq.read_text(encoding="utf-8"))
        if (prog["n"], prog["dim"]) != (n, dim):
            raise SystemExit(f"{arq} é de outro conjunto de trechos (n={prog['n']:,}). Apague-o.")
        feitos = int(prog["feitos"])
        vetores = np.lib.format.open_memmap(arq, mode="r+")
    else:
        vetores = np.lib.format.open_memmap(arq, mode="w+", dtype=np.float16, shape=(n, dim))
    print(f"{n:,} trechos · {feitos:,} já embutidos · dispositivo {dev}", flush=True)
    t0, inicio = time.perf_counter(), feitos
    with torch.no_grad():
        while feitos < n:
            fim = min(feitos + bloco, n)
            parte = df.slice(feitos, fim - feitos)
            textos = [f"{t}. {x}" if t else x
                      for t, x in zip(parte["titulo"].to_list(), parte["texto"].to_list(), strict=True)]
            vetores[feitos:fim] = _codificar(mod, tok, textos, dev, MAX_TOKENS, lote).numpy().astype(
                np.float16)
            vetores.flush()
            feitos = fim
            prog_arq.write_text(json.dumps({"feitos": feitos, "n": n, "dim": dim}), encoding="utf-8")
            v = (feitos - inicio) / (time.perf_counter() - t0)
            print(f"  {feitos:,} / {n:,} · {v:.0f} trechos/s · faltam ~{(n - feitos) / v / 60:.0f} min",
                  flush=True)


# ── os escores ───────────────────────────────────────────────────────────────────────
def _conjunto(r: dict) -> list[str]:
    return r["topo"] if r["arxiv_id"] in r["topo"] else [*r["topo"], r["arxiv_id"]]


def pontuar() -> None:
    """Por (pergunta, artigo): o melhor trecho, o melhor de introdução/conclusão, e a
    média dos 3 melhores. `nan` para quem não tem texto completo."""
    prog = json.loads((PILOTO / "progresso.json").read_text(encoding="utf-8"))
    if prog["feitos"] != prog["n"]:
        raise SystemExit(f"faltam vetores: {prog['feitos']:,} de {prog['n']:,}. Rode --embutir.")
    mapa = _mapa().select("arxiv_id", "tipo")
    ids = mapa["arxiv_id"].to_numpy()
    borda = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True])
    faixa = {ids[a]: (int(a), int(b)) for a, b in zip(borda[:-1], borda[1:], strict=True)}
    if len(faixa) != len(set(ids.tolist())):
        raise SystemExit("os trechos de um artigo não estão juntos no mapa")
    beira = np.isin(mapa["tipo"].to_numpy(), ["introducao", "conclusao"])
    vetores = np.load(PILOTO / "vetores.npy", mmap_mode="r")
    consultas = np.load(PILOTO / "consultas.npy")
    topo = np.load(PILOTO / "escores_topo.npy")
    registros = json.loads((PILOTO / "candidatos.json").read_text(encoding="utf-8"))
    with (PILOTO / "escores.jsonl").open("w", encoding="utf-8") as f:
        for q, r in enumerate(registros):
            conj = _conjunto(r)
            s_abs = [float(x) for x in topo[q][:len(r["topo"])]]
            if len(conj) > len(r["topo"]):
                s_abs.append(r["s_p"])
            c_max, c_ic, c_m3, melhor = [], [], [], []
            for a in conj:
                if a not in faixa:
                    c_max.append(None), c_ic.append(None), c_m3.append(None), melhor.append(None)
                    continue
                i, j = faixa[a]
                sc = vetores[i:j].astype(np.float32) @ consultas[q]
                c_max.append(float(sc.max()))
                melhor.append(int(i + sc.argmax()))
                c_m3.append(float(np.sort(sc)[-3:].mean()))
                b = sc[beira[i:j]]
                c_ic.append(float(b.max()) if b.size else None)
            f.write(json.dumps({"arxiv_id": r["arxiv_id"], "ids": conj, "s_abs": s_abs,
                                "c_max": c_max, "c_ic": c_ic, "c_m3": c_m3,
                                "melhor": melhor}) + "\n")
    print(f"{len(registros)} perguntas → {(PILOTO / 'escores.jsonl').relative_to(RAIZ)}")


def _escores() -> list[dict]:
    saida = []
    for linha in (PILOTO / "escores.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(linha)
        for k in ("s_abs", "c_max", "c_ic", "c_m3"):
            d[k] = np.array([np.nan if x is None else x for x in d[k]], dtype=np.float32)
        saida.append(d)
    return saida


def _para_o_phirank(d: dict) -> list[str]:
    """Os `N_PHIRANK` primeiros por cada representação, e P: quem pode chegar aos 50."""
    escolhidos = {d["arxiv_id"]}
    for k in ("s_abs", "c_max", "c_ic", "c_m3"):
        x = np.where(np.isnan(d[k]), -np.inf, d[k])
        escolhidos |= {d["ids"][j] for j in np.argsort(-x)[:N_PHIRANK] if np.isfinite(x[j])}
    return [a for a in d["ids"] if a in escolhidos]


def reordenar() -> None:
    """O escore do ΦRank de cada candidato, com o `título. resumo` — como no produto. Ele
    pontua cada par sozinho: o mesmo escore serve a qualquer lista em que o artigo entre."""
    import polars as pl
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from phifm.training.embedding import escolher_dispositivo
    from phifm.training.pairs import textos_de_documentos

    itens = {i["arxiv_id"]: i for i in _itens()}
    consultas = _ler_jsonl(CONSULTAS)
    dev = escolher_dispositivo("auto")
    tok = AutoTokenizer.from_pretrained(PHIRANK)
    mod = AutoModelForSequenceClassification.from_pretrained(PHIRANK).to(dev).eval()
    max_r = json.loads((PHIRANK / "phirank.json").read_text(encoding="utf-8"))["config"]["max_tokens"]
    arq = PILOTO / "phirank.jsonl"
    feitos = {json.loads(x)["arxiv_id"] for x in arq.read_text(encoding="utf-8").splitlines()} \
        if arq.exists() else set()
    t0, n = time.perf_counter(), 0
    with arq.open("a", encoding="utf-8") as f:
        for d in _escores():
            p = d["arxiv_id"]
            if p in feitos:
                continue
            ids = _para_o_phirank(d)
            textos = dict(textos_de_documentos(pl.scan_parquet(SPINE)
                                               .filter(pl.col("arxiv_id").is_in(ids)))
                          .collect().iter_rows())
            consulta = consultas[(itens[p]["estrato"], p)]["consultas"][0]
            esc = []
            with torch.no_grad():
                for i in range(0, len(ids), 8):
                    lote = [textos.get(a, "") for a in ids[i:i + 8]]
                    b = tok([consulta] * len(lote), lote, padding=True, truncation=True,
                            max_length=max_r, return_tensors="pt")
                    logit = mod(**{k: v.to(dev) for k, v in b.items()}).logits
                    esc += logit.float().view(-1).cpu().tolist()
            f.write(json.dumps({"arxiv_id": p, "ids": ids, "escores": esc}) + "\n")
            f.flush()
            n += 1
            if n % 10 == 0:
                s = (time.perf_counter() - t0) / n
                print(f"{len(feitos) + n}/150 · {s:.0f} s/pergunta · faltam "
                      f"~{(150 - len(feitos) - n) * s / 60:.0f} min", flush=True)


# ── a tabela ─────────────────────────────────────────────────────────────────────────
K_FONTES, K_REORDENA = 6, 50


def _depois_do_phirank(d: dict, rr: dict[str, float], entram: np.ndarray) -> bool:
    """P fica entre as `K_FONTES` depois de o ΦRank reordenar os candidatos `entram`?"""
    ids = [d["ids"][j] for j in entram]
    p = d["arxiv_id"]
    if p not in ids:
        return False
    return sum(rr.get(a, -np.inf) > rr[p] for a in ids if a != p) < K_FONTES


def _por_escore(d: dict, f: np.ndarray, f_fora: np.ndarray,
                rr: dict[str, float]) -> tuple[int, bool]:
    """(posição de P na primeira fase, P nas fontes depois do ΦRank) para uma
    representação que dá um escore `f` a cada concorrente e `f_fora` a quem está fora."""
    ip = d["ids"].index(d["arxiv_id"])
    pos = 1 + int((np.delete(f, ip) > f[ip]).sum()) + int((f_fora > f[ip]).sum())
    if pos > K_REORDENA:
        return pos, False
    # os de fora que passam à frente de P ocupam vagas entre os 50, sem escore do ΦRank
    vagas = K_REORDENA - int((f_fora > np.sort(f)[-K_REORDENA]).sum())
    return pos, _depois_do_phirank(d, rr, np.argsort(-f)[:max(vagas, 1)])


def resumir() -> None:
    registros = {r["arxiv_id"]: r for r in
                 json.loads((PILOTO / "candidatos.json").read_text(encoding="utf-8"))}
    topo = np.load(PILOTO / "escores_topo.npy")
    ordem_q = {r: q for q, r in enumerate(registros)}
    phirank = {}
    for linha in (PILOTO / "phirank.jsonl").read_text(encoding="utf-8").splitlines():
        x = json.loads(linha)
        phirank[x["arxiv_id"]] = dict(zip(x["ids"], x["escores"], strict=True))
    dados = _escores()
    if len(phirank) < len(dados):
        raise SystemExit(f"o ΦRank só pontuou {len(phirank)} de {len(dados)} perguntas.")

    def candidatos_de(nome_c: str) -> dict:
        """As representações que usam o escore de trecho `nome_c`."""
        saida = {}
        for delta in (0.0, 0.03, 0.06, 0.1, 0.15):
            saida[f"máximo(resumo, trecho + {delta:.2f})"] = ("max", nome_c, delta)
        for alfa in (0.25, 0.5, 1.0, 2.0):
            saida[f"resumo + {alfa:g} × trecho"] = ("soma", nome_c, alfa)
        for metade in (25, 15):
            saida[f"união: {K_REORDENA - metade} do resumo + {metade} do trecho"] = (
                "uniao", nome_c, metade)
        return saida

    familias = {"hoje": {"resumo (hoje)": ("resumo", "s_abs", 0.0)},
                "A · todos os trechos": candidatos_de("c_max"),
                "A · média dos 3 melhores trechos": candidatos_de("c_m3"),
                "B · introdução e conclusão": candidatos_de("c_ic")}
    resultado: dict[str, dict[str, list]] = {}
    for familia, cands in familias.items():
        for nome, (tipo, nome_c, par) in cands.items():
            linhas = []
            for d in dados:
                r = registros[d["arxiv_id"]]
                q = ordem_q[d["arxiv_id"]]
                fora = topo[q][len(r["topo"]):]
                if FUNDO_MAX < r["pos_base"] <= N_ESCORES:      # P está entre os de fora
                    fora = np.delete(fora, r["pos_base"] - 1 - len(r["topo"]))
                s, c = d["s_abs"], d[nome_c]
                tem = ~np.isnan(c)
                rr = phirank[d["arxiv_id"]]
                if tipo == "resumo":
                    pos, ok = _por_escore(d, s, fora, rr)
                elif tipo == "max":
                    pos, ok = _por_escore(d, np.where(tem, np.maximum(s, c + par), s), fora, rr)
                elif tipo == "soma":
                    # quem não tem trecho entra com o trecho típico de quem tem: o resumo
                    # menos a distância mediana entre os dois, nesta pergunta
                    dist = float(np.median((s - c)[tem])) if tem.any() else 0.0
                    f = s + par * np.where(tem, c, s - dist)
                    pos, ok = _por_escore(d, f, fora + par * (fora - dist), rr)
                else:
                    ip = d["ids"].index(d["arxiv_id"])
                    do_trecho = [j for j in np.argsort(-np.where(tem, c, -np.inf)) if tem[j]][:par]
                    entram = list(dict.fromkeys(
                        [*np.argsort(-s)[:K_REORDENA - par].tolist(), *do_trecho]))
                    resto = [j for j in np.argsort(-s).tolist() if j not in entram]
                    entram = np.array(entram + resto[:K_REORDENA - len(entram)])
                    pos = 1 if ip in entram else K_REORDENA + 1
                    ok = _depois_do_phirank(d, rr, entram)
                ip = d["ids"].index(d["arxiv_id"])
                linhas.append((pos <= K_REORDENA, ok, not np.isnan(d["c_max"][ip])))
            resultado.setdefault(familia, {})[nome] = linhas

    rng = np.random.default_rng(SEMENTE)
    hoje = np.array([x[1] for x in resultado["hoje"]["resumo (hoje)"]], dtype=float)
    com_texto = np.array([x[2] for x in resultado["hoje"]["resumo (hoje)"]])
    saida = {"exploratorio": True,
             "conjunto": "desenvolvimento, primário (150) — DOC-13 §9.2; o teste não foi tocado",
             "barra": "ganho de 5 pontos em 'P nas 6 fontes' sobre o sistema de hoje",
             "n": len(dados), "p_com_trechos": int(com_texto.sum()),
             "fundo": FUNDO, "fundo_max": FUNDO_MAX, "familias": {}}
    for familia, cands in resultado.items():
        saida["familias"][familia] = {}
        print(f"\n{familia}")
        for nome, linhas in cands.items():
            nos50 = np.array([x[0] for x in linhas], dtype=float)
            x = np.array([x[1] for x in linhas], dtype=float)
            dif = x - hoje
            reps = dif[rng.integers(0, len(dif), (5000, len(dif)))].mean(axis=1)
            lo, hi = np.percentile(reps, [2.5, 97.5])
            saida["familias"][familia][nome] = {
                "p_nos_50": round(float(nos50.mean()), 4),
                "p_nas_6_fontes": round(float(x.mean()), 4),
                "menos_hoje": round(float(dif.mean()), 4),
                "ic95": [round(float(lo), 4), round(float(hi), 4)],
                "ganha": int((dif > 0).sum()), "perde": int((dif < 0).sum()),
                "menos_hoje_P_com_trechos": round(float(dif[com_texto].mean()), 4),
                "menos_hoje_P_sem_trechos": round(float(dif[~com_texto].mean()), 4)}
            c = saida["familias"][familia][nome]
            print(f"  {nome:44s} @50 {c['p_nos_50']:.3f} · fontes {c['p_nas_6_fontes']:.3f}  "
                  f"Δ {c['menos_hoje']:+.3f} [{c['ic95'][0]:+.3f}; {c['ic95'][1]:+.3f}]  "
                  f"+{c['ganha']}/−{c['perde']}  (com texto {c['menos_hoje_P_com_trechos']:+.3f} · "
                  f"sem {c['menos_hoje_P_sem_trechos']:+.3f})")

    # o diagnóstico do encoder: só entre artigos COM trechos, P sobe ou desce?
    diag = {"n": 0}
    postos = {k: [] for k in ("s_abs", "c_max", "c_m3", "c_ic")}
    for d in dados:
        ip = d["ids"].index(d["arxiv_id"])
        tem = ~np.isnan(d["c_max"])
        if not tem[ip]:
            continue
        diag["n"] += 1
        for k in postos:
            x = np.where(np.isnan(d[k]), -np.inf, d[k])[tem]
            postos[k].append(1 + int((x > d[k][ip]).sum()) if np.isfinite(d[k][ip]) else 10**6)
    for k, v in postos.items():
        v = np.array(v)
        diag[k] = {"mediana": float(np.median(v)), "em_1": round(float((v <= 1).mean()), 4),
                   "ate_6": round(float((v <= 6).mean()), 4),
                   "ate_50": round(float((v <= 50).mean()), 4)}
    diag["trecho_melhor_que_resumo"] = int((np.array(postos["c_max"]) < np.array(postos["s_abs"])).sum())
    diag["trecho_pior_que_resumo"] = int((np.array(postos["c_max"]) > np.array(postos["s_abs"])).sum())
    saida["diagnostico_so_entre_artigos_com_trechos"] = diag
    print(f"\ndiagnóstico — o posto de P só entre os concorrentes com trechos (n={diag['n']}):")
    for k in postos:
        print(f"  {k:6s} mediana {diag[k]['mediana']:.0f} · 1º {diag[k]['em_1']:.3f} · "
              f"até 6 {diag[k]['ate_6']:.3f} · até 50 {diag[k]['ate_50']:.3f}")
    print(f"  o melhor trecho põe P mais à frente que o resumo em {diag['trecho_melhor_que_resumo']}, "
          f"mais atrás em {diag['trecho_pior_que_resumo']}")
    arq = AVALIACAO / "assistente_piloto_trechos.json"
    arq.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {arq.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    for nome in ("auditar", "candidatos", "cortar", "embutir", "pontuar", "reordenar", "resumir"):
        p.add_argument(f"--{nome}", action="store_true")
    a = p.parse_args()
    etapas = [(a.auditar, auditar), (a.candidatos, candidatos), (a.cortar, cortar_tudo),
              (a.embutir, embutir), (a.pontuar, pontuar), (a.reordenar, reordenar),
              (a.resumir, resumir)]
    if not any(liga for liga, _ in etapas):
        p.error("escolha uma etapa")
    for liga, etapa in etapas:
        if liga:
            etapa()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
