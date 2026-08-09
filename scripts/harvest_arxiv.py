#!/usr/bin/env python3
"""Sprint S1 · etapa 1 — espinha de metadados do arXiv (DOC-02 §9)."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.env import contato_obrigatorio  # noqa: E402
from phifm.core.sistema import impedir_suspensao, liberar_suspensao  # noqa: E402
from phifm.corpus.acquire.arxiv import harvest_physics  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw/arxiv_metadata"))
    p.add_argument("--set", dest="set_spec", default="physics")
    p.add_argument("--from", dest="from_date", default=None)
    p.add_argument("--until", dest="until_date", default=None)
    p.add_argument("--max-pages", type=int, default=None)
    a = p.parse_args()

    # Progresso vai para stdout, não stderr. O `logging` manda tudo para
    # stderr por padrão, e o `run_harvest.sh` mascarava isso juntando os dois
    # com `2>&1`. No Windows o `Start-Process` não junta streams, e o efeito
    # era um `harvest_arxiv.log` vazio com o conteúdo todo no `.log.err`.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    # Carrega o .env e recusa placeholder ANTES da primeira requisição.
    # Sem isto a coleta saía como `phifm-corpus@localhost` sem avisar.
    contact = contato_obrigatorio()
    logging.info("identificação de coleta: %s", contact)

    # Ver `core/sistema.py`: suspensão comeu 60% do tempo de relógio em
    # 2026-08-03. `try/finally` para não deixar a máquina impedida de dormir
    # se a coleta abortar.
    impedir_suspensao()
    try:
        m = harvest_physics(a.out, a.set_spec, a.from_date, a.until_date, a.max_pages, contact)
    finally:
        liberar_suspensao()

    print(f"\n{'concluído' if m.completed_at else 'parcial (retomável)'}: "
          f"{m.actual_count:,} registros"
          + (f" de {m.expected_count:,}" if m.expected_count else "")
          + f" · {m.requests_made} req · {m.bytes_downloaded/1e6:.1f} MB"
          f" · falhas: {len(m.failures)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
