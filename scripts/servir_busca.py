#!/usr/bin/env python3
"""Abre a busca no navegador, só nesta máquina.

    .venv-treino\\Scripts\\python.exe scripts\\servir_busca.py

e acesse http://127.0.0.1:8765 . O índice carrega uma vez (~10 s); Ctrl+C encerra.
Ver `phifm.serving.web` para o que o servidor faz e o que ele NÃO faz (expor para a rede).
"""
from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--indice", type=Path, default=RAIZ / "data/processed/indice_busca")
    p.add_argument("--porta", type=int, default=8765)
    p.add_argument("--dispositivo", default="auto")
    p.add_argument("--sem-navegador", action="store_true",
                   help="não abre o navegador sozinho")
    a = p.parse_args()

    from phifm.retrieval.indice import Busca
    from phifm.serving.web import criar_servidor

    t0 = time.perf_counter()
    busca = Busca(a.indice, dispositivo=a.dispositivo)
    servidor = criar_servidor(busca, porta=a.porta)
    url = f"http://127.0.0.1:{servidor.server_address[1]}"
    print(f"índice: {busca.vetores.shape[0]:,} artigos · carregado em "
          f"{time.perf_counter() - t0:.1f} s")
    print(f"busca no ar em {url}  (Ctrl+C encerra)")
    if not a.sem_navegador:
        webbrowser.open(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servidor.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
