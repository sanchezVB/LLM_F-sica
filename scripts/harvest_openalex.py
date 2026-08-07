#!/usr/bin/env python3
"""Sprint S1 · etapa 2 — grafo de citações via OpenAlex (DOC-02 §9)."""
import argparse, logging, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.env import contato_obrigatorio  # noqa: E402
from phifm.core.sistema import impedir_suspensao, liberar_suspensao  # noqa: E402
from phifm.corpus.acquire.openalex import harvest_arxiv_works  # noqa: E402

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw/openalex_works"))
    p.add_argument("--filter", dest="filter_expr", default=None)
    p.add_argument("--max-pages", type=int, default=None)
    a = p.parse_args()
    # stdout, não stderr — ver comentário equivalente em `harvest_arxiv.py`.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    # Aqui o contato não é só cortesia: é o `mailto` que dá acesso ao polite
    # pool do OpenAlex, com limite de taxa mais alto.
    contact = contato_obrigatorio()
    logging.info("identificação de coleta: %s", contact)
    impedir_suspensao()
    try:
        m = harvest_arxiv_works(a.out, a.filter_expr, a.max_pages, contact)
    finally:
        liberar_suspensao()
    print(f"\n{'concluído' if m.completed_at else 'parcial (retomável)'}: "
          f"{m.actual_count:,} obras"
          + (f" de {m.expected_count:,}" if m.expected_count else "")
          + f" · {m.requests_made} req · {m.bytes_downloaded/1e6:.0f} MB · falhas: {len(m.failures)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
