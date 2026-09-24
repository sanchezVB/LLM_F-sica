#!/usr/bin/env python3
"""Pergunta de Física, respondida com citações aos artigos do índice local.

    .venv-treino\\Scripts\\python.exe scripts\\perguntar.py "o que limita a coerência de qubits supercondutores?"
    .venv-treino\\Scripts\\python.exe scripts\\perguntar.py            # modo interativo

Sobe o `llama-server` (Qwen3-8B na GPU) se ele não estiver no ar, e o derruba ao sair se
foi este script que o subiu. A busca roda na CPU: com o modelo carregado sobram ~1 GB dos
8 GB da RX 7600, e a matriz do índice em float32 ocupa 2,4 GB. Custa ~1,4 s por pergunta,
contra ~10 s do modelo escrevendo — não é o gargalo.

⚠️ A qualidade das respostas NÃO foi medida. O portão confere que toda citação aponta
para uma fonte fornecida, não que a fonte diga o que a frase afirma. Ver
`phifm.rag.assistente`.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()


def mostrar(r, consulta: bool) -> None:
    if consulta:
        print("\n── consulta hipotética usada na busca ──")
        print(textwrap.fill(r.consulta, 100))
    print("\n── resposta ──")
    print(r.texto)
    print("\n── fontes ──")
    for f in r.fontes:
        marca = "●" if f.numero in r.citadas else "○"
        print(f"{marca} [{f.numero}] {f.titulo} ({f.ano or '—'}) · {f.link}")
    if r.removidas:
        print(f"\n⚠️ citações removidas (fontes que não existiam): "
              f"{', '.join(f'[{n}]' for n in r.removidas)}")
    if r.frases_sem_fonte:
        print("\n⚠️ frases sem citação — não confie nelas sem conferir:")
        for frase in r.frases_sem_fonte:
            print(f"   · {frase}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("pergunta", nargs="*")
    p.add_argument("--indice", type=Path, default=RAIZ / "data/processed/indice_busca")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    p.add_argument("-k", type=int, default=6, help="quantos artigos o modelo lê")
    p.add_argument("--dispositivo-busca", default="cpu")
    p.add_argument("--mostrar-consulta", action="store_true",
                   help="imprime o resumo hipotético que foi à busca")
    a = p.parse_args()

    from phifm.rag.assistente import Assistente
    from phifm.rag.llm import ModeloLocal
    from phifm.retrieval.indice import Busca

    modelo = ModeloLocal()
    t0 = time.perf_counter()
    if not modelo.no_ar():
        print("subindo o llama-server (Qwen3-8B na GPU)...")
        modelo.iniciar(log=RAIZ / "data/processed/llama_server.log")
    busca = Busca(a.indice, dispositivo=a.dispositivo_busca)
    assistente = Assistente(busca, modelo, a.spine, k=a.k)
    print(f"pronto em {time.perf_counter() - t0:.1f} s · "
          f"{busca.vetores.shape[0]:,} artigos no índice")
    try:
        if a.pergunta:
            mostrar(assistente.responder(" ".join(a.pergunta)), a.mostrar_consulta)
            return 0
        while True:
            try:
                q = input("\npergunta> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            t0 = time.perf_counter()
            mostrar(assistente.responder(q), a.mostrar_consulta)
            print(f"\n({time.perf_counter() - t0:.1f} s)")
    finally:
        modelo.parar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
