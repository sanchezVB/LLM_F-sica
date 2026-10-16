#!/usr/bin/env python3
"""Passagens sob demanda, no desenvolvimento: dá para ACHAR a passagem dentro dos artigos
que a busca já trouxe? (PLANO-trechos-do-texto-completo §9)

A medida do corpo (DOC-13 §9.5) deu 0,033 para o assistente de hoje e 0,940 com a passagem
certa entre as fontes — mas a passagem veio de graça. Aqui ela tem de ser achada: os ~100
trechos de cada uma das 6 fontes, e uma regra que escolhe alguns para a PERGUNTA.

    P = .venv-treino\\Scripts\\python.exe scripts\\explorar_passagens.py
    P --preparar      # corta e embute o texto das fontes de A′ nas 150 do corpo (~10 min)
    P --consultas     # a pergunta em inglês e uma passagem hipotética (GPU, ~15 min)
    P --medir         # a grade: que consulta, quantos trechos, com que vizinhança (CPU)
    P --responder     # de ponta a ponta: A′ com as passagens anexadas (GPU, ~40 min)
    P --julgar        # o juiz da §9.1 nas respostas novas (GPU, ~20 min)
    P --resumir       # o acerto ao lado de A′ e de Bp, nos mesmos itens

⚠️ As 150 perguntas do corpo são, daqui em diante, DESENVOLVIMENTO: já foram olhadas, e
esta exploração escolhe a regra olhando para elas. O número que vale sai de perguntas
novas, com a regra escrita antes (DOC-13 §9.6).

⚠️ O modelo só escreve consultas; ninguém aqui RESPONDE. As medidas são de ALCANCE — a
passagem certa está no que seria entregue —, não de acerto: o acerto é do assistente, com
o juiz.

Tudo o que isto grava fica em `data/processed/assistente/passagens_dev/`, fora do git.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402
from phifm.rag.passagens import SISTEMA_TRADUCAO  # noqa: E402

utf8()

DIR = RAIZ / "data/processed/assistente"
SAIDA = DIR / "passagens_dev"
TEXTOS = RAIZ / "data/processed/redpajama_fisica"
INDICE = RAIZ / "data/processed/indice_busca"
ITENS = DIR / "itens_corpo.json"
PASSAGENS = DIR / "corpo_passagens.jsonl"
LIMITE = 160            # o corte das passagens do corpo e da auditoria M1
MAX_TITULO = 28         # tokens do título na frente do trecho: 160 + 28 cabem nos 192


def _ler_jsonl(arq: Path) -> list[dict]:
    return [json.loads(x) for x in arq.read_text(encoding="utf-8").splitlines() if x.strip()]


def _bracos() -> dict[str, dict]:
    """A resposta de A′ de cada item do corpo: as 6 fontes e a consulta hipotética."""
    arqs = sorted(DIR.glob("bracos_corpo_*.jsonl"))
    if len(arqs) != 1:
        raise SystemExit(f"esperava um cache dos braços do corpo, achei {len(arqs)}.")
    return {r["arxiv_id"]: r for r in _ler_jsonl(arqs[0]) if r["braco"] == "A2"}


def _contador(tok):
    def contar(textos: list[str]) -> list[int]:
        if not textos:
            return []
        return [len(x) for x in tok(textos, add_special_tokens=False, verbose=False)["input_ids"]]

    return contar


def _titulo_curto(titulo: str, contar) -> str:
    palavras = " ".join((titulo or "").split()).split(" ")
    while palavras and contar([" ".join(palavras)])[0] > MAX_TITULO:
        palavras.pop()
    return " ".join(palavras)


def preparar() -> None:
    import polars as pl
    import torch
    from transformers import AutoModel, AutoTokenizer

    from phifm.eval.encoders import _codificar
    from phifm.retrieval.indice import MAX_TOKENS, _silenciar_barras
    from phifm.retrieval.trechos import cortar
    from phifm.training.embedding import escolher_dispositivo

    SAIDA.mkdir(parents=True, exist_ok=True)
    itens = json.loads(ITENS.read_text(encoding="utf-8"))
    bracos = _bracos()
    ouro = {x["arxiv_id"]: x for x in _ler_jsonl(PASSAGENS) if not x.get("sem_passagem")}
    alvo = sorted({a for it in itens for a in bracos[it["arxiv_id"]]["fontes"]}
                  | {it["arxiv_id"] for it in itens})
    modelo = json.loads((INDICE / "manifesto.json").read_text(encoding="utf-8"))["modelo"]
    tok = AutoTokenizer.from_pretrained(modelo)
    contar = _contador(tok)
    titulos = dict(pl.read_parquet(INDICE / "documentos.parquet", columns=["arxiv_id", "title"])
                   .filter(pl.col("arxiv_id").is_in(alvo)).iter_rows())

    arq_trechos = SAIDA / "trechos.parquet"
    if not arq_trechos.exists():
        linhas, vistos, t0 = [], set(), time.perf_counter()
        for p in sorted(TEXTOS.glob("part-*.parquet")):
            df = pl.scan_parquet(p).filter(pl.col("arxiv_id").is_in(alvo)).collect()
            for a, texto in df.iter_rows():
                if a in vistos:
                    continue
                vistos.add(a)
                titulo = _titulo_curto(titulos.get(a, ""), contar)
                for i, t in enumerate(cortar(texto, contar, limite=LIMITE, sobreposicao=0)):
                    linhas.append((a, titulo, i, t.secao, t.tipo, t.n_tokens, t.excede, t.texto))
            print(f"{p.name}: {len(vistos)} artigos · {len(linhas):,} trechos · "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)
        pl.DataFrame(linhas, orient="row", schema={
            "arxiv_id": pl.String, "titulo": pl.String, "i": pl.Int32, "secao": pl.String,
            "tipo": pl.String, "n_tokens": pl.Int32, "excede": pl.Boolean,
            "texto": pl.String}).write_parquet(arq_trechos)
    df = pl.read_parquet(arq_trechos)
    com_texto = set(df["arxiv_id"].unique().to_list())
    print(f"{len(alvo)} artigos pedidos · {len(com_texto)} com texto completo · "
          f"{df.height:,} trechos")

    # A passagem de onde a pergunta saiu tem de reaparecer no mesmo lugar: é o que diz que
    # o corte daqui é o corte de lá, e que `primeiro_trecho` aponta o ouro.
    fora = []
    for it in itens:
        o = ouro[it["arxiv_id"]]
        ts = (df.filter(pl.col("arxiv_id") == it["arxiv_id"]).sort("i")["texto"].to_list())
        i = o["primeiro_trecho"]
        if " ".join(ts[i:i + 3]) != o["passagem"]:
            fora.append(it["arxiv_id"])
    if fora:
        raise SystemExit(f"{len(fora)} passagens não reaparecem no corte: {fora[:5]}")
    print("as 150 passagens de ouro reaparecem no corte, no mesmo lugar.")

    _silenciar_barras()
    dev = escolher_dispositivo("auto")
    mod = AutoModel.from_pretrained(modelo, attn_implementation="eager").to(dev).eval()

    def embutir(textos: list[str], nome: str) -> None:
        arq = SAIDA / nome
        if arq.exists():
            return
        t0 = time.perf_counter()
        with torch.no_grad():
            v = _codificar(mod, tok, textos, dev, MAX_TOKENS, 64).numpy().astype(np.float16)
        np.save(arq, v)
        print(f"{nome}: {len(textos):,} textos · {len(textos) / (time.perf_counter() - t0):.0f}/s",
              flush=True)

    trechos, tits = df["texto"].to_list(), df["titulo"].to_list()
    embutir(trechos, "v_trecho.npy")
    embutir([f"{t}. {x}" if t else x for t, x in zip(tits, trechos, strict=True)],
            "v_titulo_trecho.npy")
    embutir([it["pergunta"] for it in itens], "q_pergunta.npy")
    embutir([bracos[it["arxiv_id"]]["consulta"] for it in itens], "q_hyde.npy")
    print(f"→ {SAIDA.relative_to(RAIZ)}")


# A pergunta chega em português e o texto dos artigos está em inglês; a consulta hipotética
# da busca imita um RESUMO. Duas consultas a mais, escritas pelo modelo, para a passagem:
SISTEMAS_DE_CONSULTA = {
    "ingles": (SISTEMA_TRADUCAO, 200),
    "passagem": (
        "You write one paragraph from the BODY of a physics research paper — a methods or "
        "results section, not the abstract. Given a question, write a plausible paragraph of "
        "about 80 words, in English, that would contain the answer: the sentence that states "
        "the value, parameter or detail asked for, with the technical terms a paper would "
        "use around it. Output only the paragraph.", 220),
}


def consultas() -> None:
    """As consultas escritas pelo modelo (GPU: o llama-server sobe aqui). Retomável."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from phifm.eval.assistente import SEMENTE
    from phifm.eval.encoders import _codificar
    from phifm.rag.llm import ModeloLocal
    from phifm.retrieval.indice import MAX_TOKENS, _silenciar_barras

    itens = json.loads(ITENS.read_text(encoding="utf-8"))
    modelo = ModeloLocal()
    if not modelo.no_ar():
        print("subindo o llama-server (Qwen3-8B na GPU)...", flush=True)
        modelo.iniciar(log=RAIZ / "data/processed/llama_server.log")
    try:
        for nome, (sistema, max_tokens) in SISTEMAS_DE_CONSULTA.items():
            arq = SAIDA / f"consultas_{nome}.json"
            feitas = json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else []
            t0 = time.perf_counter()
            for it in itens[len(feitas):]:
                texto = modelo.gerar(sistema, it["pergunta"], max_tokens=max_tokens,
                                     temperatura=0.3, semente=SEMENTE)
                feitas.append(" ".join(texto.split()) or it["pergunta"])
                arq.write_text(json.dumps(feitas, ensure_ascii=False), encoding="utf-8")
            print(f"{nome}: {len(feitas)} consultas · {time.perf_counter() - t0:.0f} s",
                  flush=True)
    finally:
        modelo.parar()
    _silenciar_barras()
    caminho = json.loads((INDICE / "manifesto.json").read_text(encoding="utf-8"))["modelo"]
    tok = AutoTokenizer.from_pretrained(caminho)
    mod = AutoModel.from_pretrained(caminho, attn_implementation="eager").eval()
    for nome in SISTEMAS_DE_CONSULTA:
        textos = json.loads((SAIDA / f"consultas_{nome}.json").read_text(encoding="utf-8"))
        with torch.no_grad():
            v = _codificar(mod, tok, textos, torch.device("cpu"), MAX_TOKENS, 16)
        np.save(SAIDA / f"q_{nome}.npy", v.numpy().astype(np.float16))
    print(f"→ {SAIDA.relative_to(RAIZ)}")


_PALAVRA = re.compile(r"[a-z][a-z0-9]+|\d+(?:\.\d+)?")


def _palavras(texto: str) -> list[str]:
    return _PALAVRA.findall(texto.lower().replace(",", "."))


def bm25(consulta: str, docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """BM25 da consulta contra os trechos de UMA pergunta: a frequência de documento sai
    só deles — o que é raro entre os trechos das 6 fontes é o que distingue a passagem."""
    n = len(docs)
    df = Counter(p for d in docs for p in set(d))
    media = sum(len(d) for d in docs) / max(n, 1)
    termos = set(_palavras(consulta))
    s = np.zeros(n, dtype=np.float32)
    for j, d in enumerate(docs):
        tf = Counter(d)
        for t in termos & tf.keys():
            idf = np.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s[j] += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / media))
    return s


def _postos(s: np.ndarray) -> np.ndarray:
    """O posto de cada trecho (0 = o melhor)."""
    r = np.empty(s.size, dtype=np.int64)
    r[np.argsort(-s, kind="stable")] = np.arange(s.size)
    return r


def _escolher(postos: np.ndarray, donos: np.ndarray, pos: np.ndarray, k: int,
              vizinhos: int) -> set[int]:
    """Os `k` trechos entregues. Com `vizinhos`, cada escolhido traz os que o cercam no
    MESMO artigo, e o orçamento continua `k` — menos centros, mais contexto em volta."""
    ordem = np.argsort(postos, kind="stable")
    if not vizinhos:
        return set(ordem[:k].tolist())
    onde = {(int(donos[j]), int(pos[j])): j for j in range(postos.size)}
    escolhidos: set[int] = set()
    for j in ordem:
        grupo = [onde[c] for d in range(-vizinhos, vizinhos + 1)
                 if (c := (int(donos[j]), int(pos[j]) + d)) in onde]
        if len(escolhidos | set(grupo)) > k:
            break
        escolhidos |= set(grupo)
    return escolhidos


def medir() -> None:
    """A grade do alcance. Tudo na CPU, sobre o que `--preparar` gravou."""
    import polars as pl

    from perguntas_corpo import numeros

    itens = json.loads(ITENS.read_text(encoding="utf-8"))
    bracos = _bracos()
    ouro = {x["arxiv_id"]: x for x in _ler_jsonl(PASSAGENS) if not x.get("sem_passagem")}
    df = pl.read_parquet(SAIDA / "trechos.parquet").with_row_index("linha")
    v = {n: np.load(SAIDA / f"v_{n}.npy").astype(np.float32) for n in ("trecho", "titulo_trecho")}
    q = {n: np.load(SAIDA / f"q_{n}.npy").astype(np.float32) for n in ("pergunta", "hyde")}
    extras = {p.stem[2:]: np.load(p).astype(np.float32) for p in sorted(SAIDA.glob("q_*.npy"))
              if p.stem[2:] not in q}
    q |= extras
    textos_extra = {}
    for n in extras:
        arq = SAIDA / f"consultas_{n}.json"
        if arq.exists():
            textos_extra[n] = json.loads(arq.read_text(encoding="utf-8"))
    por_artigo = {a: g for (a,), g in df.group_by("arxiv_id")}
    ids_art = {a: n for n, a in enumerate(por_artigo)}

    casos = []
    n_fontes_com_texto = []
    for n, it in enumerate(itens):
        r = bracos[it["arxiv_id"]]
        com_texto = [a for a in r["fontes"] if a in por_artigo]
        n_fontes_com_texto.append(len(com_texto))
        if it["arxiv_id"] not in r["fontes"]:
            continue
        g = pl.concat([por_artigo[a] for a in com_texto]).sort("arxiv_id", "i")
        linhas = g["linha"].to_numpy()
        donos = np.array([ids_art[a] for a in g["arxiv_id"].to_list()])
        pos = g["i"].to_numpy()
        i0 = ouro[it["arxiv_id"]]["primeiro_trecho"]
        e_ouro = (donos == ids_art[it["arxiv_id"]]) & (pos >= i0) & (pos < i0 + 3)
        textos = g["texto"].to_list()
        consultas = {"pergunta": it["pergunta"], "hyde": r["consulta"]}
        consultas |= {k: t[n] for k, t in textos_extra.items()}
        casos.append({"n": n, "linhas": linhas, "donos": donos, "pos": pos, "ouro": e_ouro,
                      "textos": textos, "palavras": [_palavras(t) for t in textos],
                      "consultas": consultas, "tipos": g["tipo"].to_list(),
                      "nums": numeros(it["gabarito"]) if it["critica"] == "ok" else None})
    tam = np.array([c["linhas"].size for c in casos])
    print(f"{len(itens)} perguntas · P entre as fontes em {len(casos)} · fontes com texto por "
          f"pergunta: média {np.mean(n_fontes_com_texto):.1f} de 6 · trechos a escolher por "
          f"pergunta: mediana {int(np.median(tam))}, de {tam.min()} a {tam.max()}")
    com_nums = sum(1 for c in casos if c["nums"])

    regras: dict[str, list[np.ndarray]] = {}
    for c in casos:
        cs = c["consultas"]
        s = {f"{nq}·{nv}": v[nv][c["linhas"]] @ q[nq][c["n"]] for nq in q for nv in v}
        s |= {f"bm25·{nq}": bm25(t, c["palavras"]) for nq, t in cs.items()}
        for a, b_ in combinations(sorted(cs), 2):
            s[f"bm25·{a}+{b_}"] = bm25(f"{cs[a]} {cs[b_]}", c["palavras"])
        postos = {k: _postos(x) for k, x in s.items()}

        def rrf(*nomes: str, p=postos) -> np.ndarray:
            """Fusão por postos (k = 60): o denso e o lexical erram em lugares diferentes."""
            return _postos(sum(1 / (60 + p[n]) for n in nomes))

        for nq in cs:
            postos[f"rrf(bm25·{nq}, {nq}·trecho)"] = rrf(f"bm25·{nq}", f"{nq}·trecho")
        for a, b_ in combinations(sorted(cs), 2):
            postos[f"rrf(bm25·{a}, bm25·{b_})"] = rrf(f"bm25·{a}", f"bm25·{b_}")
        if {"ingles", "passagem"} <= cs.keys():
            postos["rrf(bm25: ingles, passagem, hyde)"] = rrf(
                "bm25·ingles", "bm25·passagem", "bm25·hyde")
            postos["rrf(bm25 e denso: ingles, passagem)"] = rrf(
                "bm25·ingles", "bm25·passagem", "ingles·trecho", "passagem·trecho")
            postos["rrf(bm25: ingles, passagem; denso: passagem)"] = rrf(
                "bm25·ingles", "bm25·passagem", "passagem·trecho")
        for k, x in postos.items():
            regras.setdefault(k, []).append(x)

    grade = []
    for nome, lista in regras.items():
        for k in (3, 6, 9, 12):
            for viz in (0, 1):
                if viz and k < 6:
                    continue
                ach = achou_num = todo = 0
                for c, postos in zip(casos, lista, strict=True):
                    esc = _escolher(postos, c["donos"], c["pos"], k, viz)
                    do_ouro = [j for j in esc if c["ouro"][j]]
                    ach += bool(do_ouro)
                    todo += len(do_ouro) == 3
                    if c["nums"]:
                        achou_num += c["nums"] <= numeros(" ".join(c["textos"][j] for j in esc))
                grade.append({"regra": nome, "k": k, "vizinhos": viz,
                              "ouro": round(ach / len(casos), 4),
                              "ouro_inteiro": round(todo / len(casos), 4),
                              "numeros_do_gabarito": round(achou_num / com_nums, 4)})
    grade.sort(key=lambda g: (-g["numeros_do_gabarito"], -g["ouro"]))
    melhor_posto = {nome: [int(p[c["ouro"]].min()) for c, p in zip(casos, lista, strict=True)]
                    for nome, lista in regras.items()}
    print(f"\nalcance em {len(casos)} perguntas com P entre as fontes ({com_nums} com número "
          "conferível no gabarito)")
    print(f"{'regra':50s} {'k':>3s} viz  ouro  inteiro  números")
    for g in grade[:45]:
        print(f"{g['regra']:50s} {g['k']:3d}  {g['vizinhos']}   {g['ouro']:.3f}  "
              f"{g['ouro_inteiro']:.3f}    {g['numeros_do_gabarito']:.3f}")
    print("\no posto do melhor trecho de ouro, por regra (mediana · dentro dos 3 · 6 · 12 · 30):")
    for nome, ps in sorted(melhor_posto.items(), key=lambda kv: np.mean(np.array(kv[1]) < 6),
                           reverse=True):
        a = np.array(ps)
        print(f"   {nome:50s} {int(np.median(a)) + 1:4d} · {np.mean(a < 3):.3f} · "
              f"{np.mean(a < 6):.3f} · {np.mean(a < 12):.3f} · {np.mean(a < 30):.3f}")
    (SAIDA / "grade.json").write_text(json.dumps(
        {"n_perguntas": len(casos), "n_com_numero": com_nums, "grade": grade,
         "posto_do_ouro": melhor_posto}, ensure_ascii=False), encoding="utf-8")


# ── o outro lado: as perguntas do RESUMO não podem piorar ────────────────────────────
ITENS_PRIMARIO = DIR / "itens_dev_busca.json"
FONTES_PRIMARIO = SAIDA / "primario_fontes.jsonl"


def _itens_primario() -> list[dict]:
    return [i for i in json.loads(ITENS_PRIMARIO.read_text(encoding="utf-8"))
            if i["estrato"] == "primario"]


def fontes_primario() -> None:
    """As 6 fontes de A′ para as 150 perguntas de desenvolvimento do primário: a consulta
    hipotética do cache (a da semente do produto), a busca e o ΦRank. CPU, retomável."""
    from phifm.eval.assistente import K_FONTES, SEMENTE
    from phifm.rag.assistente import Assistente
    from phifm.rag.reordenador import Reordenador
    from phifm.retrieval.indice import Busca

    itens = _itens_primario()
    consultas = {d["arxiv_id"]: d["consultas"][0]
                 for d in _ler_jsonl(DIR / "dev_busca_consultas.jsonl")
                 if d["estrato"] == "primario"}
    feitos = ({d["arxiv_id"] for d in _ler_jsonl(FONTES_PRIMARIO)}
              if FONTES_PRIMARIO.exists() else set())
    assistente = Assistente(Busca(INDICE, dispositivo="cpu"), None,
                            RAIZ / "data/processed/spine.parquet", k=K_FONTES, semente=SEMENTE,
                            reordenador=Reordenador())
    t0, n = time.perf_counter(), 0
    with FONTES_PRIMARIO.open("a", encoding="utf-8") as f:
        for it in itens:
            p = it["arxiv_id"]
            if p in feitos:
                continue
            ids = [x.arxiv_id for x in assistente.recuperar(consultas[p])]
            f.write(json.dumps({"arxiv_id": p, "consulta": consultas[p], "fontes": ids,
                                "posicao_p": ids.index(p) + 1 if p in ids else None}) + "\n")
            f.flush()
            n += 1
            if n % 10 == 0:
                print(f"{len(feitos) + n}/{len(itens)} · {(time.perf_counter() - t0) / n:.1f} "
                      "s por pergunta", flush=True)
    todos = _ler_jsonl(FONTES_PRIMARIO)
    print(f"{len(todos)} perguntas · P entre as 6 fontes em "
          f"{sum(1 for d in todos if d['posicao_p'])}")


# ── de ponta a ponta, no desenvolvimento: o assistente COM as passagens ──────────────
BRACO_DEV = "A3"        # as fontes e a consulta de A′, mais as passagens escolhidas


def _trechos_dev() -> dict[str, list]:
    import polars as pl

    from phifm.retrieval.trechos import Trecho

    saida: dict[str, list] = {}
    df = pl.read_parquet(SAIDA / "trechos.parquet").sort("arxiv_id", "i")
    for a, secao, tipo, n, excede, texto in df.select(
            "arxiv_id", "secao", "tipo", "n_tokens", "excede", "texto").iter_rows():
        saida.setdefault(a, []).append(Trecho(texto, secao, tipo, n, excede=excede))
    return saida


def _fontes_de_a2(bracos: dict[str, dict]) -> dict[str, list]:
    """As 6 fontes de A′ de cada item, refeitas do spine numa leitura só."""
    import polars as pl

    from phifm.rag.assistente import Fonte

    ids = sorted({a for r in bracos.values() for a in r["fontes"]})
    meta = {r["arxiv_id"]: r for r in pl.scan_parquet(RAIZ / "data/processed/spine.parquet")
            .filter(pl.col("arxiv_id").is_in(ids))
            .select("arxiv_id", "title", "abstract", "year").collect().iter_rows(named=True)}
    return {p: [Fonte(i, a, " ".join((meta[a]["title"] or "").split()), meta[a]["year"],
                      " ".join((meta[a]["abstract"] or "").split()), 0.0)
                for i, a in enumerate(r["fontes"], start=1)] for p, r in bracos.items()}


def _cache_dev(k: int) -> Path:
    return SAIDA / f"bracos_{BRACO_DEV}_k{k}.jsonl"


def _cache_do_juiz() -> Path:
    from phifm.eval import assistente_bracos as b
    from phifm.rag.llm import MODELO

    return DIR / f"juiz_{b.assinatura_juiz(MODELO.name)[:12]}.jsonl"


def responder(k: int, limite: int | None) -> None:
    """A resposta com as passagens, nas 150 do corpo. A consulta hipotética e as fontes são
    as de A′, do cache: o que muda entre A′ e este braço é só o que foi anexado. Retomável."""
    from dataclasses import asdict

    from phifm.eval import assistente_bracos as b
    from phifm.eval.assistente import K_FONTES, SEMENTE
    from phifm.rag import passagens as ps
    from phifm.rag.assistente import Assistente
    from phifm.rag.llm import ModeloLocal

    itens = json.loads(ITENS.read_text(encoding="utf-8"))
    bracos = _bracos()
    ingles = json.loads((SAIDA / "consultas_ingles.json").read_text(encoding="utf-8"))
    trechos = _trechos_dev()
    fontes = _fontes_de_a2(bracos)
    cache = _cache_dev(k)
    feitos = {r.arxiv_id for r in b.ler_respostas(cache)}
    modelo = ModeloLocal()
    if not modelo.no_ar():
        print("subindo o llama-server (Qwen3-8B na GPU)...", flush=True)
        modelo.iniciar(log=RAIZ / "data/processed/llama_server.log")
    assistente = Assistente(None, modelo, RAIZ / "data/processed/spine.parquet", k=K_FONTES,
                            semente=SEMENTE)
    passo = ps.Passagens(modelo, trechos.get, k=k, semente=SEMENTE)
    t_ini, n = time.perf_counter(), 0
    try:
        with cache.open("a", encoding="utf-8") as f:
            for i, it in enumerate(itens):
                if it["arxiv_id"] in feitos:
                    continue
                if limite is not None and n >= limite:
                    break
                r0 = bracos[it["arxiv_id"]]
                com = passo.anexar(it["pergunta"], fontes[it["arxiv_id"]], consulta=ingles[i])
                t0 = time.perf_counter()
                r = assistente.responder_com(it["pergunta"], com, r0["consulta"])
                reg = b._registro(it, BRACO_DEV, r, time.perf_counter() - t0)
                f.write(json.dumps(asdict(reg), ensure_ascii=False) + "\n")
                f.flush()
                n += 1
                s = (time.perf_counter() - t_ini) / n
                falta = len(itens) - len(feitos) - n
                print(f"[{len(feitos) + n}/{len(itens)}] {it['arxiv_id']} · {reg.segundos:.0f} s · "
                      f"{sum(len(x.passagens) for x in com)} blocos · P "
                      f"{'em ' + str(reg.posicao_p) if reg.posicao_p else 'fora'} · faltam "
                      f"~{falta * s / 60:.0f} min", flush=True)
    finally:
        modelo.parar()


def julgar_dev(k: int) -> None:
    from phifm.eval import assistente_bracos as b
    from phifm.rag.llm import ModeloLocal

    itens = json.loads(ITENS.read_text(encoding="utf-8"))
    respostas = b.ler_respostas(_cache_dev(k))
    modelo = ModeloLocal()
    if not modelo.no_ar():
        print("subindo o llama-server (Qwen3-8B na GPU)...", flush=True)
        modelo.iniciar(log=RAIZ / "data/processed/llama_server.log")
    try:
        julg = b.julgar_todas(modelo, itens, respostas, _cache_do_juiz())
    finally:
        modelo.parar()
    print(f"{len(julg)} textos no cache do juiz")


def resumir_dev(k: int) -> None:
    """O acerto com as passagens, ao lado de A′ e de Bp nos MESMOS itens. ⚠️ Desenvolvimento:
    não é a medida — as perguntas foram olhadas e a regra foi escolhida nelas."""
    from perguntas_corpo import numeros
    from phifm.eval import assistente_bracos as b
    from phifm.rag import passagens as ps
    from phifm.rag.assistente import bloco_de_fontes

    itens = json.loads(ITENS.read_text(encoding="utf-8"))
    bracos = _bracos()
    ouro = {x["arxiv_id"]: x for x in _ler_jsonl(PASSAGENS) if not x.get("sem_passagem")}
    ingles = json.loads((SAIDA / "consultas_ingles.json").read_text(encoding="utf-8"))
    trechos = _trechos_dev()
    fontes = _fontes_de_a2(bracos)
    antigas = b.ler_respostas(next(iter(sorted(DIR.glob("bracos_corpo_*.jsonl")))))
    novas = b.ler_respostas(_cache_dev(k))
    feitos = {r.arxiv_id for r in novas}
    itens_f = [it for it in itens if it["arxiv_id"] in feitos]
    v = b.veredictos_por_braco(itens, [r for r in antigas if r.arxiv_id in feitos] + novas,
                               b.ler_julgamentos(_cache_do_juiz()))
    passo = ps.Passagens(None, trechos.get, k=k)
    linhas = []
    for i, it in enumerate(itens):
        if it["arxiv_id"] not in feitos:
            continue
        p = it["arxiv_id"]
        esc = passo.escolher(ingles[i], bracos[p]["fontes"])
        i0 = ouro[p]["primeiro_trecho"]
        achou = any(e.arxiv_id == p and i0 <= e.posicao < i0 + 3 for e in esc)
        nums = numeros(it["gabarito"]) if it["critica"] == "ok" else None
        com = passo.anexar(it["pergunta"], fontes[p], consulta=ingles[i])
        linhas.append({"p_veio": p in bracos[p]["fontes"], "ouro": achou,
                       "nums": None if nums is None
                       else nums <= numeros(" ".join(e.texto for e in esc)),
                       "blocos": sum(len(x.passagens) for x in com),
                       "fontes_com_passagem": sum(1 for x in com if x.passagens),
                       "caracteres": len(bloco_de_fontes(com)),
                       **{br: v[br][("corpo", p)] == "certo" for br in ("A2", "Bp", BRACO_DEV)},
                       "veredicto": v[BRACO_DEV][("corpo", p)]})

    def fatia(nome: str, xs: list[dict]) -> None:
        if not xs:
            return
        acc = {br: np.array([x[br] for x in xs], dtype=float) for br in ("A2", "Bp", BRACO_DEV)}
        ponto, ic = b.proporcao(acc[BRACO_DEV])
        vs = Counter(x["veredicto"] for x in xs)
        print(f"   {nome:46s} n {len(xs):3d} · A′ {acc['A2'].mean():.3f} · com passagens "
              f"{ponto:.3f} [{ic[0]:.3f}; {ic[1]:.3f}] · Bp {acc['Bp'].mean():.3f} · "
              f"{dict(vs)}")

    print(f"k = {k} · {len(itens_f)} de {len(itens)} itens respondidos e julgados")
    fatia("todas", linhas)
    fatia("P entre as fontes", [x for x in linhas if x["p_veio"]])
    fatia("  …e a passagem de ouro foi entregue", [x for x in linhas if x["p_veio"] and x["ouro"]])
    fatia("  …e a passagem de ouro NÃO foi entregue",
          [x for x in linhas if x["p_veio"] and not x["ouro"]])
    fatia("  …e os números do gabarito foram entregues",
          [x for x in linhas if x["p_veio"] and x["nums"]])
    fatia("P fora das fontes", [x for x in linhas if not x["p_veio"]])
    car = np.array([x["caracteres"] for x in linhas])
    seg = np.array([r.segundos for r in novas])
    print(f"   fontes com passagem por pergunta: média "
          f"{np.mean([x['fontes_com_passagem'] for x in linhas]):.1f} · blocos: média "
          f"{np.mean([x['blocos'] for x in linhas]):.1f} · caracteres do bloco de fontes: mediana "
          f"{int(np.median(car)):,}, máximo {int(car.max()):,} · segundos por resposta: mediana "
          f"{np.median(seg):.1f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--preparar", action="store_true")
    p.add_argument("--consultas", action="store_true")
    p.add_argument("--medir", action="store_true")
    p.add_argument("--fontes-primario", action="store_true")
    p.add_argument("--responder", action="store_true")
    p.add_argument("--julgar", action="store_true")
    p.add_argument("--resumir", action="store_true")
    p.add_argument("-k", type=int, default=9, help="trechos entregues por pergunta")
    p.add_argument("--limite", type=int)
    a = p.parse_args()
    if a.preparar:
        preparar()
    elif a.consultas:
        consultas()
    elif a.medir:
        medir()
    elif a.fontes_primario:
        fontes_primario()
    elif a.responder:
        responder(a.k, a.limite)
    elif a.julgar:
        julgar_dev(a.k)
    elif a.resumir:
        resumir_dev(a.k)
    else:
        p.error("--preparar, --consultas, --medir, --responder, --julgar ou --resumir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
