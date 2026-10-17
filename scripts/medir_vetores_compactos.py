#!/usr/bin/env python3
"""Os vetores do índice em int8 (DOC-13 §8.1): a busca sai a mesma, com metade da memória?

    P = .venv-treino\\Scripts\\python.exe scripts\\medir_vetores_compactos.py
    P --converter    # grava a versão int8 em data/processed/indice_busca_int8 (~5 min, CPU)
    P --medir        # a equivalência nas 300 perguntas do desenvolvimento (~15 min, sem o modelo)

O índice do produto não é tocado: a versão int8 vai para outra pasta, e a busca do
produto continua em float16. `--medir` recusa enquanto a regra for PROPOSTA
(`REGRA_8_1_ACEITA` em `phifm.retrieval.compacto`).

As perguntas são as dos dois conjuntos de desenvolvimento da §9.8 (corpo 1 e primário dev,
150 cada), com a consulta hipotética e a pergunta em inglês já gravadas — as mesmas de
`explorar_busca_lexical.conjuntos()`. Para cada uma: a busca densa em float16 e em int8,
cada uma fundida ao MESMO BM25 como no produto (§9.9), e os 50 que iriam ao ΦRank.

⚠️ Segura a trava de rodada pesada de `medir_passagens.py`: o índice denso, o lexical e a
versão int8 juntos ocupam ~4 GB.
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

import numpy as np  # noqa: E402

from phifm.retrieval import compacto as cp  # noqa: E402

INDICE = RAIZ / "data/processed/indice_busca"
DESTINO = RAIZ / "data/processed/indice_busca_int8"
LEXICO = RAIZ / "data/processed/indice_lexical"
SAIDA = RAIZ / "data/processed/assistente/vetores_compactos/dev.jsonl"
AGREGADO = RAIZ / "data/processed/avaliacao/vetores_compactos_dev.json"
CONTAGEM_PES2O = RAIZ / "data/processed/avaliacao/pes2o_fora_do_arxiv.json"


def _ler(arq: Path) -> list[dict]:
    if not arq.exists():
        return []
    return [json.loads(x) for x in arq.read_text(encoding="utf-8").splitlines() if x.strip()]


def medir_item(busca, lexica, m, e, linha: dict[str, int], conjunto: str, it: dict) -> dict:
    """Uma pergunta: as duas buscas densas, a mesma fusão com o BM25, os 50 de cada."""
    from phifm.retrieval.lexical import posicoes_de, rrf

    q = busca.embutir(it["hyde"])
    t0 = time.perf_counter()
    s16 = busca.pontuar(q)
    t1 = time.perf_counter()
    s8 = cp.pontuar(m, e, q)
    t2 = time.perf_counter()
    pb = posicoes_de(lexica.pontuar(it["ingles"]))
    f16, f8 = rrf(posicoes_de(s16), pb), rrf(posicoes_de(s8), pb)
    t16, t8 = cp.topo(f16, 50), cp.topo(f8, 50)
    lp = linha.get(it["arxiv_id"])
    return {"conjunto": conjunto, "arxiv_id": it["arxiv_id"],
            "sobreposicao_50_fundida": cp.sobreposicao(t16, t8),
            "sobreposicao_50_densa": cp.sobreposicao(cp.topo(s16, 50), cp.topo(s8, 50)),
            "p_exato": lp in t16, "p_int8": lp in t8,
            "ms_float16": round(1000 * (t1 - t0), 1), "ms_int8": round(1000 * (t2 - t1), 1)}


def medir() -> None:
    import explorar_busca_lexical as ebl
    import medir_passagens as mp
    from phifm.retrieval.indice import Busca
    from phifm.retrieval.lexical import Lexica

    if not cp.REGRA_8_1_ACEITA:
        raise SystemExit("a §8.1 do DOC-13 ainda é PROPOSTA: a medida não roda antes do aceite "
                         "do dono (REGRA_8_1_ACEITA em phifm.retrieval.compacto).")
    conjuntos = ebl.conjuntos()
    feitos = {(d["conjunto"], d["arxiv_id"]) for d in _ler(SAIDA)}
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    with mp.trava("medir_passagens"):
        busca = Busca(INDICE, dispositivo="cpu")
        ids = busca.docs["arxiv_id"].to_list()
        lexica = Lexica(LEXICO, ids)
        m, e = cp.carregar(DESTINO, INDICE)
        linha = {a: i for i, a in enumerate(ids)}
        with SAIDA.open("a", encoding="utf-8") as f:
            for nome, itens in conjuntos.items():
                for it in itens:
                    if (nome, it["arxiv_id"]) in feitos:
                        continue
                    f.write(json.dumps(medir_item(busca, lexica, m, e, linha, nome, it)) + "\n")
                    f.flush()
        forma, bytes16, bytes8 = busca.vetores.shape, int(busca.vetores.nbytes), int(m.nbytes)
    resumir(forma, bytes16, bytes8, sum(len(v) for v in conjuntos.values()))


def resumir(forma: tuple[int, int], bytes16: int, bytes8: int, esperados: int) -> None:
    itens = _ler(SAIDA)
    if len(itens) < esperados:
        raise SystemExit(f"{len(itens)} de {esperados} perguntas medidas: rode --medir de novo.")
    por = {c: cp.decidir([d for d in itens if d["conjunto"] == c])
           for c in sorted({d["conjunto"] for d in itens})}
    n_docs, dim = forma
    novos = (json.loads(CONTAGEM_PES2O.read_text(encoding="utf-8"))["s2ag"]["novos"]
             if CONTAGEM_PES2O.exists() else None)
    saida = {
        "regra": "DOC-13 §8.1", "regra_aceita_pelo_dono": cp.REGRA_8_1_ACEITA,
        "o_que_e": "a busca densa em int8 contra a de hoje (float16), cada uma fundida ao "
                   "mesmo BM25, nos 50 que iriam ao ΦRank — perguntas do desenvolvimento",
        "decide (os dois conjuntos juntos)": cp.decidir(itens), "por_conjunto": por,
        "sobreposicao_media_50_densa (relatado)": round(float(np.mean(
            [d["sobreposicao_50_densa"] for d in itens])), 4),
        "tempo_mediano_ms": {"float16": float(np.median([d["ms_float16"] for d in itens])),
                             "int8": float(np.median([d["ms_int8"] for d in itens]))},
        "memoria_dos_vetores_gb": {"float16": round(bytes16 / 1e9, 2),
                                   "int8": round(bytes8 / 1e9, 2)},
        "projecao_com_o_pes2o (relatado)": None if novos is None else {
            "documentos": n_docs + novos,
            "float16_gb": round((n_docs + novos) * dim * 2 / 1e9, 2),
            "int8_gb": round((n_docs + novos) * dim / 1e9, 2)},
        "decisao": "equivalente" if cp.decidir(itens)["equivalente"] else "nao_equivalente"}
    AGREGADO.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(saida, ensure_ascii=False, indent=1))
    print(f"→ {AGREGADO.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--converter", action="store_true")
    g.add_argument("--medir", action="store_true")
    a = p.parse_args()
    if a.converter:
        import medir_passagens as mp

        with mp.trava("medir_passagens"):
            man = cp.converter(INDICE, DESTINO)
        print(json.dumps(man, ensure_ascii=False, indent=1))
    else:
        medir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
