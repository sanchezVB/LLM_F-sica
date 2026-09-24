#!/usr/bin/env python3
"""Busca artigos de Física no índice local.

    .venv-treino\\Scripts\\python.exe scripts\\buscar.py "entanglement entropy of black holes"
    .venv-treino\\Scripts\\python.exe scripts\\buscar.py            # modo interativo

Sem consulta na linha de comando, abre um laço: o índice é carregado uma vez (~10 s) e
cada consulta responde em frações de segundo. Linha vazia sai.

⚠️ O encoder foi treinado com consultas que são TÍTULO + RESUMO de um artigo, e acha
melhor quando a consulta se parece com isso: colar o resumo de um artigo encontra os que
ele cita e os parecidos com ele. Uma pergunta curta também funciona, mas é outra
distribuição de entrada, e a qualidade dela não foi medida.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()


def mostrar(resultados) -> None:
    for r in resultados:
        autores = ", ".join(r.autores[:3]) + (" et al." if len(r.autores) >= 3 else "")
        print(f"\n{r.posicao:>2}. {r.titulo}")
        print(f"    {autores} · {r.ano or '—'} · {r.categoria or '—'} · "
              f"similaridade {r.escore:.3f}")
        print(f"    {r.link}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("consulta", nargs="*")
    p.add_argument("--indice", type=Path, default=RAIZ / "data/processed/indice_busca")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--dispositivo", default="auto")
    a = p.parse_args()

    from phifm.retrieval.indice import Busca

    t0 = time.perf_counter()
    busca = Busca(a.indice, dispositivo=a.dispositivo)
    print(f"índice: {busca.vetores.shape[0]:,} artigos · carregado em "
          f"{time.perf_counter() - t0:.1f} s")

    if a.consulta:
        mostrar(busca.buscar(" ".join(a.consulta), k=a.k))
        return 0
    while True:
        try:
            q = input("\nbusca> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        t0 = time.perf_counter()
        res = busca.buscar(q, k=a.k)
        mostrar(res)
        print(f"\n({time.perf_counter() - t0:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
