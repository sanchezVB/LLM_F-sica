#!/usr/bin/env python3
"""A busca do artigo para perguntas do CORPO: reordenar os candidatos pelo texto completo
(pedido do dono em 2026-10-16). EXPLORATÓRIO, só no desenvolvimento; o teste não é tocado.

    P = .venv-treino\\Scripts\\python.exe scripts\\explorar_busca_corpo.py
    P --listas       # os 200 primeiros da busca de hoje (fundida), na GPU
    P --phirank      # o ΦRank deles, 50 perguntas por processo (repetir até acabar)
    P --texto        # o BM25 da pergunta em inglês no texto completo de cada candidato
    P --resumir      # P nas 6 fontes em cada variante, contra a busca de hoje

Por que: na §9.9 a busca fundida levou o artigo certo às fontes em +7 pontos nas perguntas do
resumo e só +4 nas do corpo. No corpo 1, ele está entre os 200 primeiros da fusão em 96%
das perguntas e entre os 6 em 73%: o que falta é ORDEM, e tanto a fusão quanto o ΦRank só
leem título e resumo. A pergunta do corpo cita o detalhe que está no corpo.

O texto completo de cada candidato é lido do HD (`phifm.retrieval.textos`) e pontuado pelo
BM25 da pergunta em inglês — o artigo inteiro, ou a melhor janela de 150 palavras (a
frequência de documento sai das janelas dos candidatos da pergunta, como nas passagens da
§9.6). Conjuntos: corpo 1 (150, o estrato que importa) e primário dev (150, para ver o efeito
nas perguntas do resumo).

⚠️ No corpo, P sempre tem texto completo (é assim que as perguntas foram escritas); no
índice, 52% dos artigos têm. Um candidato sem texto não ganha nem perde pela pontuação do
texto: fica com a posição que a busca de hoje lhe deu.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402

utf8()

import explorar_busca_lexical as ebl  # noqa: E402

SAIDA = ebl.DIR / "busca_corpo"
CANDIDATOS = SAIDA / "candidatos.jsonl"
LISTAS = SAIDA / "listas.jsonl"
TEXTO = SAIDA / "texto.jsonl"
TEXTOS = RAIZ / "data/processed/textos_por_artigo"
AGREGADO = RAIZ / "data/processed/avaliacao/assistente_busca_corpo_dev.json"
K = 200
JANELA, PASSO = 150, 75
K1, B = 1.2, 0.75


def _ler(arq: Path) -> dict[tuple, dict]:
    if not arq.exists():
        return {}
    return {(d["conjunto"], d["arxiv_id"]): d for d in ebl._ler_jsonl(arq)}


def listas() -> None:
    """Os K primeiros da busca fundida (§9.9) para cada pergunta. Só a busca na GPU: o
    ΦRank vem em outro processo (`--phirank`), porque os dois juntos esgotaram a memória
    da placa no meio das 300 perguntas (2026-10-16)."""
    import numpy as np

    from phifm.retrieval import lexical as lx
    from phifm.retrieval.indice import Busca

    sets = ebl.conjuntos()
    feitos = {**_ler(CANDIDATOS), **_ler(LISTAS)}
    busca = Busca(ebl.INDICE)
    ids = busca.docs["arxiv_id"].to_list()
    lexica = lx.Lexica(ebl.LEXICO, ids)
    SAIDA.mkdir(parents=True, exist_ok=True)
    with LISTAS.open("a", encoding="utf-8") as f:
        for nome, itens in sets.items():
            for it in itens:
                if (nome, it["arxiv_id"]) in feitos:
                    continue
                fundido = lx.rrf(lx.posicoes_de(busca.pontuar(busca.embutir(it["hyde"]))),
                                 lx.posicoes_de(lexica.pontuar(it["ingles"])))
                idx = np.argpartition(-fundido, K - 1)[:K]
                ordem = idx[np.argsort(-fundido[idx], kind="stable")]
                f.write(json.dumps({"conjunto": nome, "arxiv_id": it["arxiv_id"],
                                    "hyde": it["hyde"],
                                    "ids": [ids[int(j)] for j in ordem]}) + "\n")
    print(f"listas → {LISTAS}")


def phirank(lote: int) -> None:
    """O escore do ΦRank (consulta hipotética × `título. resumo`) dos K candidatos, em
    `lote` perguntas por processo: a memória que a DirectML segura cresce com o número de
    formatos de lote diferentes, e um processo novo a zera."""
    import polars as pl

    from phifm.rag.reordenador import Reordenador
    from phifm.training.embedding import escolher_dispositivo
    from phifm.training.pairs import textos_de_documentos

    feitos = _ler(CANDIDATOS)
    faltam = [v for k, v in _ler(LISTAS).items() if k not in feitos][:lote]
    if not faltam:
        print("ΦRank: nada a fazer")
        return
    todos = sorted({a for v in faltam for a in v["ids"]})
    textos = dict(textos_de_documentos(pl.scan_parquet(ebl.SPINE).filter(
        pl.col("arxiv_id").is_in(todos))).collect().iter_rows())
    rr = Reordenador()
    dev = escolher_dispositivo("auto")
    rr.dev, rr.mod = dev, rr.mod.to(dev)
    t0 = time.perf_counter()
    with CANDIDATOS.open("a", encoding="utf-8") as f:
        for n, v in enumerate(faltam, start=1):
            esc = []
            for i in range(0, K, 25):
                pedaco = v["ids"][i:i + 25]
                esc += [round(float(x), 4) for x in
                        rr.escores(v["hyde"], [textos.get(a, "") for a in pedaco])]
            f.write(json.dumps({"conjunto": v["conjunto"], "arxiv_id": v["arxiv_id"],
                                "ids": v["ids"], "phirank": esc}) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"[ΦRank {len(feitos) + n}] {(time.perf_counter() - t0) / n:.1f} s/item",
                      flush=True)
    print(f"ΦRank: {len(feitos) + len(faltam)} perguntas com candidatos")


def pontuar_texto(consulta: list[str], textos: list[str | None]) -> dict[str, list]:
    """O BM25 da consulta em cada texto: o artigo inteiro (`doc`) e a melhor janela de
    `JANELA` termos com passo `PASSO` (`janela`). A frequência de documento sai dos
    próprios textos (artigos para `doc`, janelas para `janela`). Sem texto: None."""
    import numpy as np

    from phifm.retrieval.lexical import termos

    q = sorted(set(consulta))
    pos = {t: i for i, t in enumerate(q)}
    contagens, comprimentos = [], []          # por texto: matriz janelas × termos, e o total
    for texto in textos:
        if texto is None:
            contagens.append(None)
            comprimentos.append(None)
            continue
        ts = termos(texto)
        marca = np.array([pos.get(t, -1) for t in ts], dtype=np.int32)
        acum = np.zeros((len(ts) + 1, len(q)), dtype=np.int32)
        for j in range(len(q)):
            acum[1:, j] = np.cumsum(marca == j)
        inicios = list(range(0, max(len(ts) - JANELA, 0) + 1, PASSO)) or [0]
        fins = [min(i + JANELA, len(ts)) for i in inicios]
        contagens.append((acum[fins] - acum[inicios], acum[-1]))
        comprimentos.append((np.array(fins) - np.array(inicios), len(ts)))
    com = [i for i, c in enumerate(contagens) if c is not None]
    saida = {"doc": [None] * len(textos), "janela": [None] * len(textos)}
    if not com or not q:
        return saida
    # artigo inteiro
    tf_doc = np.array([contagens[i][1] for i in com], dtype=np.float64)
    dl = np.array([comprimentos[i][1] for i in com], dtype=np.float64)
    idf = np.log1p((len(com) - (tf_doc > 0).sum(0) + 0.5) / ((tf_doc > 0).sum(0) + 0.5))
    norma = K1 * (1 - B + B * dl / max(dl.mean(), 1.0))
    doc = (idf * tf_doc * (K1 + 1) / (tf_doc + norma[:, None])).sum(1)
    # janelas
    jan = np.concatenate([contagens[i][0] for i in com]).astype(np.float64)
    jl = np.concatenate([comprimentos[i][0] for i in com]).astype(np.float64)
    dono = np.concatenate([[k] * len(contagens[i][0]) for k, i in enumerate(com)])
    idf_j = np.log1p((len(jan) - (jan > 0).sum(0) + 0.5) / ((jan > 0).sum(0) + 0.5))
    norma_j = K1 * (1 - B + B * jl / max(jl.mean(), 1.0))
    esc_j = (idf_j * jan * (K1 + 1) / (jan + norma_j[:, None])).sum(1)
    melhor = np.full(len(com), -np.inf)
    np.maximum.at(melhor, dono, esc_j)
    for k, i in enumerate(com):
        saida["doc"][i] = round(float(doc[k]), 4)
        saida["janela"][i] = round(float(melhor[k]), 4)
    return saida


def texto() -> None:
    from phifm.retrieval.lexical import termos
    from phifm.retrieval.textos import Textos

    sets = {(nome, it["arxiv_id"]): it for nome, itens in ebl.conjuntos().items() for it in itens}
    cands, feitos = _ler(CANDIDATOS), _ler(TEXTO)
    tx = Textos(TEXTOS)
    t0, n = time.perf_counter(), 0
    with TEXTO.open("a", encoding="utf-8") as f:
        for chave, c in cands.items():
            if chave in feitos:
                continue
            textos = [tx.ler(a) for a in c["ids"]]
            esc = pontuar_texto(termos(sets[chave]["ingles"]), textos)
            f.write(json.dumps({"conjunto": chave[0], "arxiv_id": chave[1],
                                "tem_texto": [t is not None for t in textos], **esc}) + "\n")
            f.flush()
            n += 1
            if n % 25 == 0:
                print(f"[texto {n}/{len(cands)}] {(time.perf_counter() - t0) / n:.1f} s/item",
                      flush=True)


def _posicoes(escores, desempate) -> list[int]:
    """Posição (1 = primeiro) de cada candidato pelo escore; None fica atrás de todos, na
    ordem de `desempate`."""
    chave = sorted(range(len(escores)),
                   key=lambda i: (escores[i] is None, -(escores[i] or 0), desempate[i]))
    pos = [0] * len(escores)
    for r, i in enumerate(chave, start=1):
        pos[i] = r
    return pos


def variantes(c: dict, t: dict) -> dict[str, list[str]]:
    """As 6 fontes de cada variante, a partir dos K candidatos da busca de hoje."""
    ids, k = c["ids"], len(c["ids"])
    base = list(range(1, k + 1))                     # a posição na busca fundida
    rr = _posicoes(c["phirank"], base)
    doc = _posicoes(t["doc"], base)
    jan = _posicoes(t["janela"], base)
    sem_texto = [x is None for x in t["janela"]]

    def topo(escore, n=6, dentre=None):
        cand = range(k) if dentre is None else dentre
        return [ids[i] for i in sorted(cand, key=lambda i: (-escore[i], i))[:n]]

    def rrf(*listas, c_=60):
        # Quem não tem texto não pode ganhar nem perder pela lista do texto: nela, fica com
        # a posição que a busca de hoje lhe deu. (A primeira versão o tirava da soma, e ele
        # perdia para todos os que têm texto — 2026-10-16, −17 a −24 pontos no resumo.)
        return [sum(1 / (c_ + (base[i] if (so_com and sem_texto[i]) else p[i]))
                    for p, so_com in listas) for i in range(k)]

    def phirank_nos(primeiros: list[int]) -> list[str]:
        return [ids[i] for i in sorted(primeiros, key=lambda i: (-c["phirank"][i], i))[:6]]

    hoje = phirank_nos(list(range(50)))
    saida = {"hoje · fusão, ΦRank nos 50": hoje,
             "fusão, sem ΦRank": ids[:6],
             "só o texto (melhor janela) nos 200": topo([-x for x in jan]),
             "só o texto (artigo inteiro) nos 200": topo([-x for x in doc])}
    for n_ in (50, 100, 200):
        dentre = list(range(n_))
        f2 = rrf((base, False), (jan, True))
        f3 = rrf((base, False), (jan, True), (rr, False))
        saida[f"fusão + janela, nos {n_}"] = topo(f2, dentre=dentre)
        saida[f"fusão + janela + ΦRank, nos {n_}"] = topo(f3, dentre=dentre)
        saida[f"fusão + artigo, nos {n_}"] = topo(rrf((base, False), (doc, True)), dentre=dentre)
    ordem_f2 = sorted(range(k), key=lambda i: (-rrf((base, False), (jan, True))[i], i))
    saida["fusão + janela nos 200, depois ΦRank nos 50"] = phirank_nos(ordem_f2[:50])
    return saida


def resumir() -> None:
    import numpy as np

    from phifm.eval.assistente_bracos import diferenca_pareada

    cands, txts = _ler(CANDIDATOS), _ler(TEXTO)
    rr_antigo = {(d["conjunto"], d["arxiv_id"]): d["rr50_rrf_hyde_bm25ingles"]
                 for d in ebl._ler_jsonl(ebl.SAIDA / "phirank.jsonl")}
    saida = {"o_que_e": "P entre as 6 fontes em cada variante, a partir dos 200 primeiros "
                        "da busca fundida da §9.9; exploratório, só desenvolvimento",
             "conjuntos": {}}
    for nome in ("corpo_1", "primario_dev"):
        chaves = [k for k in cands if k[0] == nome and k in txts]
        if not chaves:
            continue
        por_var: dict[str, list[bool]] = {}
        for k in chaves:
            for v, fontes in variantes(cands[k], txts[k]).items():
                por_var.setdefault(v, []).append(k[1] in fontes)
        hoje = np.array(por_var["hoje · fusão, ΦRank nos 50"], dtype=float)
        bate = sum((rr_antigo[k] <= 6) == bool(h) for k, h in zip(chaves, hoje, strict=True)
                   if k in rr_antigo)
        t = {"n": len(chaves), "P_nos_200": round(float(np.mean(
            [k[1] in cands[k]["ids"] for k in chaves])), 4),
             "P_tem_texto": round(float(np.mean([txts[k]["tem_texto"][cands[k]["ids"].index(k[1])]
                                                 if k[1] in cands[k]["ids"] else False
                                                 for k in chaves])), 4),
             "hoje_reproduz_a_9_8": f"{bate} de {len(chaves)}", "variantes": {}}
        p_texto = np.array([k[1] in cands[k]["ids"]
                            and txts[k]["tem_texto"][cands[k]["ids"].index(k[1])]
                            for k in chaves])
        for v, xs in por_var.items():
            x = np.array(xs, dtype=float)
            linha = {"P@6": round(float(x.mean()), 4),
                     "P@6_quando_P_tem_texto": round(float(x[p_texto].mean()), 4)
                     if p_texto.any() else None,
                     "P@6_quando_P_nao_tem": round(float(x[~p_texto].mean()), 4)
                     if (~p_texto).any() else None}
            if v != "hoje · fusão, ΦRank nos 50":
                dif, ic = diferenca_pareada(x, hoje)
                linha.update({"menos_hoje": dif, "ic95": ic,
                              "resgata": int(((x == 1) & (hoje == 0)).sum()),
                              "perde": int(((x == 0) & (hoje == 1)).sum())})
            t["variantes"][v] = linha
        saida["conjuntos"][nome] = t
    AGREGADO.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    for nome, t in saida["conjuntos"].items():
        print(f"\n{nome} (n={t['n']}; P nos 200: {t['P_nos_200']}; P com texto: "
              f"{t['P_tem_texto']}; hoje reproduz a §9.8 em {t['hoje_reproduz_a_9_8']})")
        for v, linha in t["variantes"].items():
            extra = (f"   {linha['menos_hoje']:+.3f} [{linha['ic95'][0]:+.3f}; "
                     f"{linha['ic95'][1]:+.3f}] · {linha['resgata']}/{linha['perde']}"
                     if "menos_hoje" in linha else "")
            com_t, sem_t = linha["P@6_quando_P_tem_texto"], linha["P@6_quando_P_nao_tem"]
            partes = (f" (P com texto {com_t:.3f}" if com_t is not None else " (") + (
                f" · sem {sem_t:.3f})" if sem_t is not None else ")")
            print(f"  {v:<44} P@6 {linha['P@6']:.3f}{partes}{extra}")
    print(f"\n→ {AGREGADO.relative_to(RAIZ)}")


def produto50() -> None:
    """A versão que vai ao produto: TUDO nos 50 primeiros da busca — o ΦRank, as janelas
    (com a frequência de documento das janelas desses 50) e a fusão das três ordens —, pelas
    funções do produto (`phifm.retrieval.lexical.melhor_janela` e `ordem_com_texto`)."""
    import numpy as np

    from phifm.eval.assistente_bracos import diferenca_pareada
    from phifm.retrieval.lexical import melhor_janela, ordem_com_texto
    from phifm.retrieval.textos import Textos

    sets = {(nome, it["arxiv_id"]): it for nome, itens in ebl.conjuntos().items() for it in itens}
    cands = _ler(CANDIDATOS)
    tx = Textos(TEXTOS)
    t0 = time.perf_counter()
    linhas = {}
    for chave, c in cands.items():
        ids, rr = c["ids"][:50], c["phirank"][:50]
        hoje = [ids[i] for i in sorted(range(50), key=lambda i: (-rr[i], i))[:6]]
        t1 = time.perf_counter()
        jan = melhor_janela(sets[chave]["ingles"], [tx.ler(a) for a in ids])
        novo = [ids[i] for i in ordem_com_texto(rr, jan)[:6]]
        linhas[chave] = (chave[1] in hoje, chave[1] in novo, time.perf_counter() - t1)
    agregado = json.loads(AGREGADO.read_text(encoding="utf-8"))
    agregado["produto_tudo_nos_50"] = {}
    for nome in ("corpo_1", "primario_dev"):
        xs = [v for k, v in linhas.items() if k[0] == nome]
        h = np.array([x[0] for x in xs], dtype=float)
        n = np.array([x[1] for x in xs], dtype=float)
        dif, ic = diferenca_pareada(n, h)
        agregado["produto_tudo_nos_50"][nome] = {
            "n": len(xs), "hoje": round(float(h.mean()), 4), "com_texto": round(float(n.mean()), 4),
            "menos_hoje": dif, "ic95": ic, "resgata": int(((n == 1) & (h == 0)).sum()),
            "perde": int(((n == 0) & (h == 1)).sum()),
            "segundos_texto_mediana": round(float(np.median([x[2] for x in xs])), 2)}
        print(nome, agregado["produto_tudo_nos_50"][nome])
    AGREGADO.write_text(json.dumps(agregado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(linhas)} perguntas em {time.perf_counter() - t0:.0f} s → {AGREGADO.name}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--listas", action="store_true")
    p.add_argument("--phirank", action="store_true")
    p.add_argument("--lote", type=int, default=50, help="perguntas por processo em --phirank")
    p.add_argument("--texto", action="store_true")
    p.add_argument("--resumir", action="store_true")
    p.add_argument("--produto50", action="store_true")
    a = p.parse_args()
    if a.produto50:
        produto50()
        return 0
    if a.listas:
        listas()
    elif a.phirank:
        phirank(a.lote)
    elif a.texto:
        texto()
    elif a.resumir:
        resumir()
    else:
        p.error("--listas, --phirank, --texto ou --resumir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
