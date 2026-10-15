#!/usr/bin/env python3
"""A medida da busca fundida com o BM25 (DOC-13 §9.9): o artigo certo chega mais às
fontes — e a resposta acerta mais?

    P = .venv-treino\\Scripts\\python.exe scripts\\medir_busca_lexical.py
    P --traduzir    # fase 1, só o modelo: as 650 perguntas em inglês
    P --buscar      # fase 2, sem o modelo: a busca fundida + ΦRank na CPU; e o corpo 2
    P --responder   # fase 3, só o modelo: as respostas onde as fontes mudam, e o juiz
    P --resumir     # a regra, no primário; o resto relatado

Em fases porque o modelo, a busca na CPU e o índice lexical juntos paginam na máquina de
16 GB (2026-10-14: 113 s por item). O sistema medido é o mesmo; ver o desvio na §9.9.

As fases tocam o conjunto de teste, uma vez: recusam enquanto a regra for PROPOSTA
(`REGRA_9_9_ACEITA` em `phifm.eval.assistente_bracos`). Respostas e julgamentos ficam em
`data/processed/assistente/`, fora do git; o agregado vai para
`data/processed/avaliacao/assistente_busca_lexical.json`.

⚠️ Uma rodada pesada por vez (16 GB): segura a trava de `medir_passagens.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402

utf8()

import medir_passagens as mp  # noqa: E402

LEXICO = RAIZ / "data/processed/indice_lexical"
AGREGADO = mp.AVALIACAO / "assistente_busca_lexical.json"
CORPO2 = mp.DIR / "busca_lexical" / "corpo2_A5.jsonl"
TRADUCOES = mp.DIR / "busca_lexical" / "teste_traducoes.jsonl"
FONTES = mp.DIR / "busca_lexical" / "teste_fontes.jsonl"


def _carregar():
    """`(itens, A′ por item, A por item, cache de A5)` dos 650 do teste."""
    import medir_assistente as ma
    from phifm.eval import assistente_bracos as b
    from phifm.rag.llm import MODELO
    from phifm.rag.passagens import SISTEMA_TRADUCAO

    itens, sha = ma.carregar_itens()
    cache_b, _, ab, _ = ma.caminhos(sha)
    a2 = {(r.estrato, r.arxiv_id): r for r in b.ler_respostas(mp.DIR / f"bracos_A2_{ab[:12]}.jsonl")}
    a = {(r.estrato, r.arxiv_id): r for r in b.ler_respostas(cache_b) if r.braco == "A"}
    itens = [i for i in itens if (i["estrato"], i["arxiv_id"]) in a2]
    lexico = json.loads((LEXICO / "manifesto.json").read_text(encoding="utf-8"))
    assin = mp._sha(f"{b.assinatura_bracos(MODELO.name, sha)}\n{SISTEMA_TRADUCAO}\n"
                    f"{lexico['hash_da_ordem']}\n{lexico['celulas']}\n{lexico['k1']}\n{lexico['b']}")
    return itens, a2, a, mp.DIR / f"bracos_{b.BRACO_BUSCA_LEXICA}_{assin[:12]}.jsonl"


def _recusa_sem_aceite() -> None:
    from phifm.eval import assistente_bracos as b

    if not b.REGRA_9_9_ACEITA:
        raise SystemExit("a §9.9 do DOC-13 ainda é PROPOSTA: o conjunto de teste não é tocado "
                         "antes do aceite do dono (REGRA_9_9_ACEITA em "
                         "phifm.eval.assistente_bracos).")


def _buscador(a):
    """O assistente SÓ com a busca: índice denso e ΦRank na CPU, como no produto, e o
    índice lexical. Sem modelo — a fase da busca não sobe o llama-server."""
    from phifm.eval.assistente import K_FONTES, SEMENTE
    from phifm.rag.assistente import Assistente
    from phifm.rag.reordenador import Reordenador
    from phifm.retrieval.indice import Busca
    from phifm.retrieval.lexical import Lexica

    busca = Busca(a.indice, dispositivo="cpu")
    lexica = Lexica(LEXICO, busca.docs["arxiv_id"].to_list())
    return Assistente(busca, None, a.spine, k=K_FONTES, semente=SEMENTE,
                      reordenador=Reordenador(), lexica=lexica)


def _modelo():
    from phifm.rag.llm import ModeloLocal

    modelo = ModeloLocal()
    if not modelo.no_ar():
        print("subindo o llama-server (Qwen3-8B na GPU)...", flush=True)
        modelo.iniciar(log=RAIZ / "data/processed/llama_server.log")
    return modelo


def _feitos(arq: Path) -> dict[tuple, dict]:
    if not arq.exists():
        return {}
    return {(d["estrato"], d["arxiv_id"]): d for d in
            (json.loads(x) for x in arq.read_text(encoding="utf-8").splitlines() if x.strip())}


def _progresso(rotulo: str, total: int, feitos0: int):
    t0, n = time.perf_counter(), [0]

    def passo():
        n[0] += 1
        if n[0] % 25 == 0:
            s = (time.perf_counter() - t0) / n[0]
            print(f"[{rotulo} {feitos0 + n[0]}/{total}] {s:.1f} s/item · faltam "
                  f"~{(total - feitos0 - n[0]) * s / 60:.0f} min", flush=True)
    return passo


# ── as três fases (desvio de 2026-10-14, DOC-13 §9.9): o mesmo sistema, sem disputar memória
def traduzir_fase(a) -> None:
    """Fase 1, só o modelo: a pergunta em inglês, a mesma chamada do produto."""
    from phifm.eval.assistente import SEMENTE
    from phifm.rag.passagens import traduzir

    _recusa_sem_aceite()
    itens, _, _, _ = _carregar()
    feitos = _feitos(TRADUCOES)
    TRADUCOES.parent.mkdir(parents=True, exist_ok=True)
    with mp.trava("medir_passagens"):
        modelo = _modelo()
        passo = _progresso("tradução", len(itens), len(feitos))
        try:
            with TRADUCOES.open("a", encoding="utf-8") as f:
                for it in itens:
                    if (it["estrato"], it["arxiv_id"]) in feitos:
                        continue
                    t0 = time.perf_counter()
                    ingles = traduzir(modelo, it["pergunta"], SEMENTE)
                    f.write(json.dumps({"estrato": it["estrato"], "arxiv_id": it["arxiv_id"],
                                        "ingles": ingles,
                                        "segundos": round(time.perf_counter() - t0, 2)},
                                       ensure_ascii=False) + "\n")
                    f.flush()
                    passo()
        finally:
            modelo.parar()
    print(f"traduções → {TRADUCOES}")


def buscar_fase(a) -> None:
    """Fase 2, sem o modelo: a busca fundida e o ΦRank, na CPU, como no produto."""
    _recusa_sem_aceite()
    itens, a2, _, _ = _carregar()
    traducoes = _feitos(TRADUCOES)
    if len(traducoes) < len(itens):
        raise SystemExit("faltam traduções (--traduzir).")
    feitos = _feitos(FONTES)
    with mp.trava("medir_passagens"):
        assistente = _buscador(a)
        passo = _progresso("busca", len(itens), len(feitos))
        with FONTES.open("a", encoding="utf-8") as f:
            for it in itens:
                chave = (it["estrato"], it["arxiv_id"])
                if chave in feitos:
                    continue
                t0 = time.perf_counter()
                fontes = assistente.recuperar(a2[chave].consulta, traducoes[chave]["ingles"])
                f.write(json.dumps({"estrato": chave[0], "arxiv_id": chave[1],
                                    "fontes": [x.arxiv_id for x in fontes],
                                    "segundos": round(time.perf_counter() - t0, 2)}) + "\n")
                f.flush()
                passo()
        corpo2(assistente)
    print(f"fontes → {FONTES}")


def responder_fase(a) -> None:
    """Fase 3, só o modelo: onde as 6 fontes de A5 são as de A′, na mesma ordem, a
    resposta de A′ vale (como na §9.2); onde mudam, o modelo responde com elas — as fontes
    refeitas como `Assistente.recuperar` as monta (título e ano do índice, resumo do
    spine). Depois, o juiz."""
    from dataclasses import asdict, replace

    from phifm.eval import assistente_bracos as b
    from phifm.eval.assistente import K_FONTES, SEMENTE
    from phifm.rag.assistente import Assistente

    _recusa_sem_aceite()
    itens, a2, _, cache = _carregar()
    traducoes, buscas = _feitos(TRADUCOES), _feitos(FONTES)
    if len(buscas) < len(itens):
        raise SystemExit("faltam buscas (--buscar).")
    feitos = {(r.estrato, r.arxiv_id) for r in b.ler_respostas(cache)}
    mudam = {k: SimpleNamespace(fontes=d["fontes"]) for k, d in buscas.items()
             if d["fontes"] != a2[k].fontes}
    fontes_de = mp._refazer_fontes(mudam, a.indice, a.spine) if mudam else None
    with mp.trava("medir_passagens"):
        modelo = _modelo()
        assistente = Assistente(None, modelo, a.spine, k=K_FONTES, semente=SEMENTE)
        passo = _progresso("A5", len(itens), len(feitos))
        try:
            with cache.open("a", encoding="utf-8") as f:
                for it in itens:
                    chave = (it["estrato"], it["arxiv_id"])
                    if chave in feitos:
                        continue
                    ra, t_rec = a2[chave], round(traducoes[chave]["segundos"]
                                                 + buscas[chave]["segundos"], 2)
                    if chave not in mudam:
                        reg = replace(ra, braco=b.BRACO_BUSCA_LEXICA, igual_a_A=True,
                                      segundos=t_rec)
                    else:
                        t0 = time.perf_counter()
                        r = assistente.responder_com(it["pergunta"],
                                                     fontes_de(buscas[chave]["fontes"]),
                                                     ra.consulta)
                        reg = b._registro(it, b.BRACO_BUSCA_LEXICA, r,
                                          t_rec + time.perf_counter() - t0)
                    reg = replace(reg, segundos_recuperacao=t_rec,
                                  consulta_passagens=traducoes[chave]["ingles"])
                    f.write(json.dumps(asdict(reg), ensure_ascii=False) + "\n")
                    f.flush()
                    passo()
            respostas = b.ler_respostas(cache)
            if len(respostas) == len(itens):
                b.julgar_todas(modelo, itens, [*a2.values(), *respostas], mp._cache_do_juiz())
        finally:
            modelo.parar()
    print(f"A5 no teste: {len(respostas)} de {len(itens)} itens · {len(mudam)} com fontes "
          "diferentes das de A′")


def corpo2(assistente) -> None:
    """Relatado, sem juiz e sem modelo: nos 100 do corpo 2, o artigo certo chega às 6
    fontes? Consulta hipotética e pergunta em inglês as que a §9.6 gravou."""
    from phifm.eval import assistente_bracos as b

    itens, _, sha = mp.carregar()
    rs = b.ler_respostas(mp.DIR / f"bracos_corpo2_{mp.assinatura(sha)[:12]}.jsonl")
    a2 = {r.arxiv_id: r for r in rs if r.braco == b.BRACO_REORDENADO}
    a3 = {r.arxiv_id: r for r in rs if r.braco == b.BRACO_PASSAGENS}
    feitos = {json.loads(x)["arxiv_id"] for x in CORPO2.read_text(encoding="utf-8").splitlines()
              if x.strip()} if CORPO2.exists() else set()
    CORPO2.parent.mkdir(parents=True, exist_ok=True)
    with CORPO2.open("a", encoding="utf-8") as f:
        for it in itens:
            p = it["arxiv_id"]
            if p in feitos:
                continue
            fontes = assistente.recuperar(a2[p].consulta, a3[p].consulta_passagens)
            ids = [x.arxiv_id for x in fontes]
            f.write(json.dumps({"arxiv_id": p, "A2": a2[p].posicao_p,
                                "A5": ids.index(p) + 1 if p in ids else None}) + "\n")
            f.flush()
    print(f"corpo 2 → {CORPO2}")


def resumir() -> None:
    import numpy as np

    from phifm.eval import assistente_bracos as b

    itens, a2, a, cache = _carregar()
    r5 = b.ler_respostas(cache)
    if len(r5) < len(itens):
        raise SystemExit(f"A5 rodou em {len(r5)} de {len(itens)} itens (--rodar).")
    julg = b.ler_julgamentos(mp._cache_do_juiz())
    v = b.veredictos_por_braco(itens, [*a2.values(), *r5], julg)
    tempo_a = float(np.median([r.segundos for r in a.values() if r.estrato == "primario"]))
    saida = {"regra": "DOC-13 §9.9", "regra_aceita_pelo_dono": b.REGRA_9_9_ACEITA,
             "o_que_e": "A5 = A′ com a busca densa fundida (RRF) ao BM25 da pergunta em inglês "
                        "sobre título + resumo; o ΦRank nos 50 e as 6 fontes como hoje",
             "primario": b.comparar_busca_lexica(v, [*a2.values(), *r5], tempo_a, "primario"),
             "pos_corte (relatado)": b.comparar_busca_lexica(v, [*a2.values(), *r5], tempo_a,
                                                             "pos_corte"),
             "traducao_igual_a_da_9_6": None}
    # A tradução de A5 é a mesma chamada da §9.6 (semente fixa): quantas saíram iguais?
    r4 = {(r.estrato, r.arxiv_id): r.consulta_passagens for r in b.ler_respostas(
        mp._teste_primario()[2])}
    pares = [(r.consulta_passagens, r4[(r.estrato, r.arxiv_id)]) for r in r5
             if (r.estrato, r.arxiv_id) in r4]
    saida["traducao_igual_a_da_9_6"] = {"itens": len(pares),
                                        "iguais": sum(1 for x, y in pares if x == y)}
    if CORPO2.exists():
        ds = [json.loads(x) for x in CORPO2.read_text(encoding="utf-8").splitlines() if x.strip()]
        x5 = np.array([d["A5"] is not None for d in ds], dtype=float)
        x2 = np.array([d["A2"] is not None for d in ds], dtype=float)
        dif, ic = b.diferenca_pareada(x5, x2)
        saida["corpo_2 (relatado, P nas 6 fontes)"] = {
            "n": len(ds), "A2": round(float(x2.mean()), 4), "A5": round(float(x5.mean()), 4),
            "A5_menos_A2": dif, "ic95": ic, "resgata": int(((x5 == 1) & (x2 == 0)).sum()),
            "perde": int(((x5 == 0) & (x2 == 1)).sum())}
    saida["decisao"] = saida["primario"]["decisao"]
    AGREGADO.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(saida, ensure_ascii=False, indent=1))
    print(f"→ {AGREGADO.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--traduzir", action="store_true")
    p.add_argument("--buscar", action="store_true")
    p.add_argument("--responder", action="store_true")
    p.add_argument("--resumir", action="store_true")
    p.add_argument("--indice", type=Path, default=RAIZ / "data/processed/indice_busca")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    args = p.parse_args()
    a = SimpleNamespace(indice=args.indice, spine=args.spine)
    if args.traduzir:
        traduzir_fase(a)
    elif args.buscar:
        buscar_fase(a)
    elif args.responder:
        responder_fase(a)
    elif args.resumir:
        resumir()
    else:
        p.error("--traduzir, --buscar, --responder ou --resumir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
