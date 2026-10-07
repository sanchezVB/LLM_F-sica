#!/usr/bin/env python3
"""Abre a busca no navegador, só nesta máquina — e, com `--assistente`, as perguntas.

    .venv-treino\\Scripts\\python.exe scripts\\servir_busca.py
    .venv-treino\\Scripts\\python.exe scripts\\servir_busca.py --assistente

e acesse http://127.0.0.1:8765 . O índice carrega uma vez (~10 s); Ctrl+C encerra.
Ver `phifm.serving.web` para o que o servidor faz e o que ele NÃO faz (expor para a rede).

Com `--assistente`, a página ganha a aba "Perguntar" (Qwen3-8B + a busca, com citações,
como o `perguntar.py`). O modelo só sobe na GPU na primeira pergunta e se desliga depois
de `--ocioso-min` sem uso. A busca vai para a CPU (~1,4 s em vez de ~30 ms): o modelo
ocupa ~6 GB dos 8 GB da RX 7600, e a matriz do índice em float32 outros 2,4 GB.
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
    p.add_argument("--dispositivo", default=None,
                   help="da busca; o padrão é auto, ou cpu com --assistente")
    p.add_argument("--assistente", action="store_true",
                   help="liga a aba de perguntas (Qwen3-8B na GPU, sob demanda)")
    p.add_argument("--ocioso-min", type=int, default=10,
                   help="minutos sem pergunta até o modelo sair da GPU")
    p.add_argument("--sem-reordenar", action="store_true",
                   help="desliga o ΦRank, que reordena os 50 primeiros da busca (adotado em 2026-10-07, DOC-13 §9.2: +10,6 pontos de acerto, ~+11 s na CPU)")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    p.add_argument("--sem-navegador", action="store_true",
                   help="não abre o navegador sozinho")
    a = p.parse_args()

    from phifm.retrieval.indice import Busca
    from phifm.serving.web import criar_servidor

    t0 = time.perf_counter()
    busca = Busca(a.indice, dispositivo=a.dispositivo or ("cpu" if a.assistente else "auto"))
    assistente = modelo = None
    if a.assistente:
        from phifm.rag.assistente import Assistente
        from phifm.rag.llm import ModeloLocal, ModeloSobDemanda

        modelo = ModeloSobDemanda(ModeloLocal(), log=RAIZ / "data/processed/llama_server.log",
                                  ocioso_s=a.ocioso_min * 60)
        from phifm.rag.reordenador import carregar_ou_avisar

        reordenador = None if a.sem_reordenar else carregar_ou_avisar()
        assistente = Assistente(busca, modelo, a.spine, reordenador=reordenador)
    servidor = criar_servidor(busca, porta=a.porta, assistente=assistente,
                              ocioso_min=a.ocioso_min)
    url = f"http://127.0.0.1:{servidor.server_address[1]}"
    print(f"índice: {busca.vetores.shape[0]:,} artigos · carregado em "
          f"{time.perf_counter() - t0:.1f} s")
    if a.assistente:
        print(f"assistente ligado: o modelo sobe na primeira pergunta e sai da GPU depois "
              f"de {a.ocioso_min} min sem uso")
    print(f"no ar em {url}  (Ctrl+C encerra)")
    if not a.sem_navegador:
        webbrowser.open(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servidor.server_close()
        if modelo is not None:
            modelo.parar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
