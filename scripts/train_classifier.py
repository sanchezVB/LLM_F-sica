#!/usr/bin/env python3
"""Sprint S2 — classificador de Física (DOC-02 §6)."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polars as pl  # noqa: E402

from phifm.corpus.filter.classifier import montar_binario, save, train  # noqa: E402

ROTULO = {"subfield": "subfield", "is_physics": "is_physics"}
SAIDA = {"subfield": "models/subfield-clf", "is_physics": "models/isphysics-clf"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=sorted(ROTULO), default="subfield")
    p.add_argument("--spine", type=Path, default=Path("data/processed/spine.parquet"))
    p.add_argument("--negativos", type=Path, default=Path("data/raw/arxiv_negativos"),
                   help="apenas para --task is_physics")
    p.add_argument("--max-por-classe", type=int, default=400_000,
                   help="2,6 M de títulos+resumos não cabem em memória; um linear "
                        "sobre n-gramas satura muito antes disso")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--precision", type=float, default=0.95)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    if a.task == "is_physics":
        df = montar_binario(a.spine, a.negativos, a.max_por_classe)
    else:
        df = pl.read_parquet(a.spine)

    clf, rep = train(df, task=a.task, label_col=ROTULO[a.task],
                     target_precision=a.precision)
    save(clf, a.out or Path(SAIDA[a.task]))
    print("\n" + "=" * 72)
    print(rep)
    print("=" * 72)
    c = clf.calibration
    print(f"\nCALIBRAÇÃO (alvo de precisão = {c.target_precision:.2f}, DOC-02 §6)")
    print(f"cobertura: {100*c.coverage:.1f}% dos itens passam algum limiar\n")
    print(f"{'classe':34} {'limiar':>7} {'precisão':>9} {'revocação':>10}")
    for cls in sorted(c.thresholds, key=lambda k: -c.achieved_recall[k]):
        t = c.thresholds[cls]
        tt = "  n/a" if t > 1 else f"{t:5.2f}"
        print(f"{cls[:32]:34} {tt:>7} {c.achieved_precision[cls]:8.3f} {c.achieved_recall[cls]:9.3f}")

    if a.task == "is_physics":
        # ⚠️ Texto DERIVADO do que está em disco, não fixo. A versão anterior
        # dizia "`math` NÃO está entre eles" e continuou dizendo isso depois de
        # `math` entrar — um aviso que virou falso é pior que nenhum aviso,
        # porque quem lê confia nele.
        doms = sorted(d.name for d in a.negativos.iterdir() if d.is_dir())
        faltam = sorted({"math", "q_bio", "cs", "econ", "eess", "stat", "q_fin"} - set(doms))
        print()
        print("⚠️  LIMITE DE DOMÍNIO — estes números valem para o arXiv.")
        print(f"    Negativos presentes: {', '.join(doms)}.")
        if faltam:
            print(f"    AUSENTES: {', '.join(faltam)} — se algum deles for vizinho")
            print("    próximo da Física, o número acima é otimista.")
        print("    E nenhum domínio do arXiv representa TEXTO DE WEB, que é o que")
        print("    o OpenWebMath é. Para lá, isto não é previsão — é o que temos.")
        print("    Medir de verdade exige `scripts/avaliar_transferencia.py`.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
