#!/usr/bin/env python3
"""Sprint S1 · etapa 2b — grafo de citações pelo snapshot do OpenAlex (DOC-02 §3.1).

Rota gratuita e sem cota, alternativa à API cotada. Ver a docstring de
`phifm.corpus.acquire.openalex_snapshot` para as medições que a justificam.
"""
import argparse, logging, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.env import contato_obrigatorio  # noqa: E402
from phifm.core.sistema import impedir_suspensao, liberar_suspensao  # noqa: E402
from phifm.corpus.acquire.openalex_snapshot import harvest_snapshot  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw/openalex_snapshot"))
    p.add_argument("--max-particoes", type=int, default=None,
                   help="teto de partições nesta execução (retomável)")
    a = p.parse_args()

    # stdout, não stderr — ver comentário em `harvest_arxiv.py`.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    contact = contato_obrigatorio()
    logging.info("identificação de coleta: %s", contact)
    impedir_suspensao()
    try:
        m = harvest_snapshot(a.out, a.max_particoes, contact)
    finally:
        liberar_suspensao()

    print(f"\n{'concluído' if m.completed_at else 'parcial (retomável)'}: "
          f"{m.actual_count:,} obras com origem no arXiv"
          + f" · {m.requests_made} req · {m.bytes_downloaded/1e9:.2f} GB"
          f" · falhas: {len(m.failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
