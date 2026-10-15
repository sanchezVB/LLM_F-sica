#!/usr/bin/env python3
"""A medida da busca fundida com o BM25 (DOC-13 §9.9): o artigo certo chega mais às
fontes — e a resposta acerta mais?

    P = .venv-treino\\Scripts\\python.exe scripts\\medir_busca_lexical.py
    P --rodar [--limite N]   # A5 nos 650 do teste sobre o cache de A′, o juiz, e o corpo 2
    P --resumir              # a regra, no primário; o resto relatado

`--rodar` toca o conjunto de teste, uma vez: recusa enquanto a regra for PROPOSTA
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


def _assistente(a, com_modelo: bool):
    from phifm.eval.assistente import K_FONTES, SEMENTE
    from phifm.rag.assistente import Assistente
    from phifm.rag.reordenador import Reordenador
    from phifm.retrieval.indice import Busca
    from phifm.retrieval.lexical import Lexica

    busca = Busca(a.indice, dispositivo="cpu")          # como no produto: a GPU é do modelo
    lexica = Lexica(LEXICO, busca.docs["arxiv_id"].to_list())
    modelo = None
    if com_modelo:
        from phifm.rag.llm import ModeloLocal

        modelo = ModeloLocal()
        if not modelo.no_ar():
            print("subindo o llama-server (Qwen3-8B na GPU)...", flush=True)
            modelo.iniciar(log=RAIZ / "data/processed/llama_server.log")
    return modelo, Assistente(busca, modelo, a.spine, k=K_FONTES, semente=SEMENTE,
                              reordenador=Reordenador(), lexica=lexica)


def rodar(a) -> None:
    from phifm.eval import assistente_bracos as b

    if not b.REGRA_9_9_ACEITA:
        raise SystemExit("a §9.9 do DOC-13 ainda é PROPOSTA: o conjunto de teste não é tocado "
                         "antes do aceite do dono (REGRA_9_9_ACEITA em "
                         "phifm.eval.assistente_bracos).")
    itens, a2, _, cache = _carregar()
    with mp.trava("medir_passagens"):
        modelo, assistente = _assistente(a, com_modelo=True)
        feitos0 = len(b.ler_respostas(cache))
        t0, n = time.perf_counter(), [0]

        def ao_terminar(item, reg):
            n[0] += 1
            if n[0] % 10 == 0:
                s = (time.perf_counter() - t0) / n[0]
                falta = len(itens) - feitos0 - n[0]
                print(f"[A5 {feitos0 + n[0]}/{len(itens)}] {s:.1f} s/item · "
                      f"faltam ~{falta * s / 60:.0f} min", flush=True)

        try:
            respostas = b.rodar_busca_lexica_sobre_a2(assistente, itens, a2, cache, a.limite,
                                                     ao_terminar)
            if len(respostas) == len(itens):
                b.julgar_todas(modelo, itens, [*a2.values(), *respostas], mp._cache_do_juiz())
        finally:
            modelo.parar()
    print(f"A5 no teste: {len(respostas)} de {len(itens)} itens · "
          f"{sum(1 for r in respostas if r.igual_a_A)} com as fontes de A′")
    if len(respostas) == len(itens):
        corpo2(assistente)


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
    p.add_argument("--rodar", action="store_true")
    p.add_argument("--limite", type=int)
    p.add_argument("--resumir", action="store_true")
    p.add_argument("--indice", type=Path, default=RAIZ / "data/processed/indice_busca")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    args = p.parse_args()
    a = SimpleNamespace(indice=args.indice, spine=args.spine, limite=args.limite)
    if args.rodar:
        rodar(a)
    elif args.resumir:
        resumir()
    else:
        p.error("--rodar ou --resumir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
