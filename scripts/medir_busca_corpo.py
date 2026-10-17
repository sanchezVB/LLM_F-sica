#!/usr/bin/env python3
"""A confirmação da §9.11 do DOC-13: no modo "texto completo", ordenar os candidatos também
pelo texto completo leva o artigo certo às fontes — sem tirar no resumo?

    P = .venv-treino\\Scripts\\python.exe scripts\\medir_busca_corpo.py
    P --rodar [--lote N]   # corpo 2 e teste primário, só a posição do artigo (sem o modelo)
    P --resumir            # os três eixos da regra

Sem resposta e sem juiz (escolha do dono): mede se o artigo de onde a pergunta saiu está
entre as 6 fontes, hoje e com a ordem pelo texto, pelo caminho do produto
(`Assistente.recuperar`). As consultas são as gravadas: nenhuma chamada ao modelo.

`--rodar` toca o teste primário (só a posição do artigo): recusa enquanto a regra for
PROPOSTA (`REGRA_9_11_ACEITA`). Retomável; `--lote` perguntas por processo, porque a
memória que a DirectML segura cresce ao longo da rodada (2026-10-16).
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

import medir_busca_lexical as mbl  # noqa: E402
import medir_passagens as mp  # noqa: E402

SAIDA = mp.DIR / "busca_corpo" / "confirmacao.jsonl"
TEXTOS = RAIZ / "data/processed/textos_por_artigo"
AGREGADO = mp.AVALIACAO / "assistente_busca_corpo.json"


def itens() -> list[dict]:
    """`[{conjunto, arxiv_id, hyde, ingles}]`: o corpo 2 e o teste primário."""
    from phifm.eval import assistente_bracos as b

    itens_c, _, sha = mp.carregar()
    rs = b.ler_respostas(mp.DIR / f"bracos_corpo2_{mp.assinatura(sha)[:12]}.jsonl")
    a2 = {r.arxiv_id: r for r in rs if r.braco == b.BRACO_REORDENADO}
    a3 = {r.arxiv_id: r for r in rs if r.braco == b.BRACO_PASSAGENS}
    saida = [{"conjunto": "corpo_2", "arxiv_id": i["arxiv_id"], "hyde": a2[i["arxiv_id"]].consulta,
              "ingles": a3[i["arxiv_id"]].consulta_passagens} for i in itens_c]
    itens_t, a2_t, _, _ = mbl._carregar()
    trad = mbl._feitos(mbl.TRADUCOES)
    saida += [{"conjunto": "teste_primario", "arxiv_id": i["arxiv_id"],
               "hyde": a2_t[(i["estrato"], i["arxiv_id"])].consulta,
               "ingles": trad[(i["estrato"], i["arxiv_id"])]["ingles"]}
              for i in itens_t if i["estrato"] == "primario"]
    return saida


def _assistente(indice: Path, spine: Path):
    """O produto no modo das passagens, sem modelo: busca densa na CPU, índice lexical,
    ΦRank na GPU (só ele: a busca densa na placa junto esgotou a memória) e o texto
    completo para ler. `reordenar_pelo_texto` ligado; `recuperar(pelo_texto=False)` dá a
    ordem de hoje."""
    from phifm.eval.assistente import K_FONTES, SEMENTE
    from phifm.rag import passagens as ps
    from phifm.rag.assistente import Assistente
    from phifm.rag.reordenador import Reordenador
    from phifm.retrieval.indice import Busca
    from phifm.retrieval.lexical import Lexica
    from phifm.retrieval.textos import Textos
    from phifm.training.embedding import escolher_dispositivo

    busca = Busca(indice, dispositivo="cpu")
    rr = Reordenador()
    dev = escolher_dispositivo("auto")
    rr.dev, rr.mod = dev, rr.mod.to(dev)
    passo = ps.Passagens(None, ps.TrechosDoTexto(Textos(TEXTOS), lambda xs: [len(x) for x in xs]))
    return Assistente(busca, None, spine, k=K_FONTES, semente=SEMENTE, reordenador=rr,
                      passagens=passo, lexica=Lexica(mbl.LEXICO, busca.docs["arxiv_id"].to_list()),
                      reordenar_pelo_texto=True)


def rodar(indice: Path, spine: Path, lote: int) -> None:
    from phifm.eval import assistente_bracos as b

    if not b.REGRA_9_11_ACEITA:
        raise SystemExit("a §9.11 do DOC-13 ainda é PROPOSTA: o teste não é tocado antes do "
                         "aceite do dono (REGRA_9_11_ACEITA em phifm.eval.assistente_bracos).")
    todos = itens()
    feitos = ({(d["conjunto"], d["arxiv_id"]) for d in mbl.ebl._ler_jsonl(SAIDA)}
              if SAIDA.exists() else set())
    faltam = [i for i in todos if (i["conjunto"], i["arxiv_id"]) not in feitos][:lote]
    if not faltam:
        print("nada a fazer")
        return
    assistente = _assistente(indice, spine)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with mp.trava("medir_passagens"), SAIDA.open("a", encoding="utf-8") as f:
        for n, it in enumerate(faltam, start=1):
            t = time.perf_counter()
            hoje = [x.arxiv_id for x in assistente.recuperar(it["hyde"], it["ingles"])]
            t_hoje = time.perf_counter() - t
            t = time.perf_counter()
            novo = [x.arxiv_id for x in assistente.recuperar(it["hyde"], it["ingles"],
                                                              pelo_texto=True)]
            t_novo = time.perf_counter() - t
            p = it["arxiv_id"]
            f.write(json.dumps({"conjunto": it["conjunto"], "arxiv_id": p,
                                "hoje": hoje.index(p) + 1 if p in hoje else None,
                                "novo": novo.index(p) + 1 if p in novo else None,
                                "segundos_hoje": round(t_hoje, 2),
                                "segundos_novo": round(t_novo, 2)}) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"[{len(feitos) + n}/{len(todos)}] {(time.perf_counter() - t0) / n:.1f} s/item",
                      flush=True)
    print(f"{len(feitos) + len(faltam)} de {len(todos)} feitos")


def resumir() -> None:
    import numpy as np

    from phifm.eval import assistente_bracos as b

    ds = mbl.ebl._ler_jsonl(SAIDA)
    saida = {"regra": "DOC-13 §9.11", "regra_aceita_pelo_dono": b.REGRA_9_11_ACEITA,
             "o_que_e": "o artigo certo entre as 6 fontes no modo 'texto completo': hoje (busca "
                        "fundida + ΦRank nos 50) contra a ordem que funde também a melhor "
                        "janela do texto completo; sem resposta e sem juiz", "conjuntos": {}}
    for nome in ("corpo_2", "teste_primario"):
        sel = [d for d in ds if d["conjunto"] == nome]
        h = np.array([d["hoje"] is not None for d in sel], dtype=float)
        n = np.array([d["novo"] is not None for d in sel], dtype=float)
        dif, ic = b.diferenca_pareada(n, h)
        saida["conjuntos"][nome] = {
            "n": len(sel), "hoje": round(float(h.mean()), 4), "novo": round(float(n.mean()), 4),
            "novo_menos_hoje": dif, "ic95": ic, "resgata": int(((n == 1) & (h == 0)).sum()),
            "perde": int(((n == 0) & (h == 1)).sum()),
            "P_em_1o": {"hoje": round(float(np.mean([d["hoje"] == 1 for d in sel])), 4),
                        "novo": round(float(np.mean([d["novo"] == 1 for d in sel])), 4)},
            "segundos_a_mais_mediana": round(float(np.median(
                [d["segundos_novo"] - d["segundos_hoje"] for d in sel])), 2)}
    c, t = saida["conjuntos"]["corpo_2"], saida["conjuntos"]["teste_primario"]
    tempo = float(np.median([d["segundos_novo"] - d["segundos_hoje"] for d in ds]))
    eixos = {"ganho_no_corpo_2": bool(c["ic95"][0] > 0),
             "nao_piora_no_teste": bool(t["ic95"][0] > -b.MARGEM_NAO_PIORA_TEXTO),
             "tempo": bool(tempo <= b.TEMPO_MAX_TEXTO_S)}
    saida["tempo_a_mais_mediana_s"] = round(tempo, 2)
    saida["eixos"] = eixos
    saida["decisao"] = ("ADOTA no modo texto completo" if all(eixos.values())
                        else "FICA COMO ESTÁ")
    AGREGADO.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(saida, ensure_ascii=False, indent=1))
    print(f"→ {AGREGADO.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rodar", action="store_true")
    p.add_argument("--lote", type=int, default=200)
    p.add_argument("--resumir", action="store_true")
    p.add_argument("--indice", type=Path, default=RAIZ / "data/processed/indice_busca")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    a = p.parse_args()
    if a.rodar:
        rodar(a.indice, a.spine, a.lote)
    elif a.resumir:
        resumir()
    else:
        p.error("--rodar ou --resumir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
