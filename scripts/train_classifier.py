#!/usr/bin/env python3
"""Sprint S2 — classificador de Física (DOC-02 §6)."""
import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polars as pl  # noqa: E402
from phifm.corpus.filter.classifier import train, save  # noqa: E402

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spine", type=Path, default=Path("data/processed/spine.parquet"))
    p.add_argument("--out", type=Path, default=Path("models/subfield-clf"))
    p.add_argument("--precision", type=float, default=0.95)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    df = pl.read_parquet(a.spine)
    clf, rep = train(df, task="subfield", label_col="subfield", target_precision=a.precision)
    save(clf, a.out)
    print("\n" + "="*72); print(rep); print("="*72)
    c = clf.calibration
    print(f"\nCALIBRAÇÃO (alvo de precisão = {c.target_precision:.2f}, DOC-02 §6)")
    print(f"cobertura: {100*c.coverage:.1f}% dos itens passam algum limiar\n")
    print(f"{'subárea':34} {'limiar':>7} {'precisão':>9} {'revocação':>10}")
    for cls in sorted(c.thresholds, key=lambda k: -c.achieved_recall[k]):
        t = c.thresholds[cls]
        tt = "  n/a" if t > 1 else f"{t:5.2f}"
        print(f"{cls[:32]:34} {tt:>7} {c.achieved_precision[cls]:8.3f} {c.achieved_recall[cls]:9.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
