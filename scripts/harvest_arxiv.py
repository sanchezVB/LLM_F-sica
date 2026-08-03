#!/usr/bin/env python3
"""Sprint S1 · etapa 1 — espinha de metadados do arXiv (DOC-02 §9)."""
import argparse, logging, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.corpus.acquire.arxiv import harvest_physics  # noqa: E402

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw/arxiv_metadata"))
    p.add_argument("--set", dest="set_spec", default="physics")
    p.add_argument("--from", dest="from_date", default=None)
    p.add_argument("--until", dest="until_date", default=None)
    p.add_argument("--max-pages", type=int, default=None)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    contact = os.environ.get("PHIFM_CONTACT", "phifm-corpus@localhost")
    m = harvest_physics(a.out, a.set_spec, a.from_date, a.until_date, a.max_pages, contact)

    print(f"\n{'concluído' if m.completed_at else 'parcial (retomável)'}: "
          f"{m.actual_count:,} registros"
          + (f" de {m.expected_count:,}" if m.expected_count else "")
          + f" · {m.requests_made} req · {m.bytes_downloaded/1e6:.1f} MB"
          f" · falhas: {len(m.failures)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
