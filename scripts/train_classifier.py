#!/usr/bin/env python3
"""Sprint S2 — classificador de Física (DOC-02 §6)."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polars as pl  # noqa: E402

from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    entrada_de,
    gravar_manifesto_etapa,
)
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
    destino = a.out or Path(SAIDA[a.task])
    save(clf, destino)
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

    # ⚠️ `max_por_classe` e `precision` são a IDENTIDADE deste classificador.
    #
    # O limiar de precisão 0,95 é o que define quanto entra no corpus: ele governa
    # a taxa de falso positivo das fatias filtradas por este modelo — o
    # OpenWebMath e o peS2o inteiros passam por ele. Reconstruir "0.95" do código
    # daria o default de HOJE, e um classificador treinado com outro alvo
    # produziria um corpus diferente sob o mesmo nome.
    #
    # E `--task` entra no nome da etapa: `subfield` e `is_physics` gravam em
    # diretórios diferentes e são modelos diferentes.
    me = gravar_manifesto_etapa(
        etapa=f"{a.task.replace('_', '')}_clf",
        descricao=("Classificador binário Física/não-Física, negativos "
                   "estratificados" if a.task == "is_physics"
                   else "Classificador de subárea da Física"),
        raiz=destino,
        entradas=[entrada_de(a.spine)] + (
            [entrada_de(a.negativos)] if a.task == "is_physics" else []),
        parametros={"script": "scripts/train_classifier.py", "task": a.task,
                    "max_por_classe": a.max_por_classe,
                    "precision": a.precision,
                    "spine": str(a.spine).replace("\\", "/"),
                    "negativos": (str(a.negativos).replace("\\", "/")
                                  if a.task == "is_physics" else None),
                    "linhas_de_treino": df.height,
                    "cobertura_da_calibracao": round(c.coverage, 5)},
        registros=df.height)
    print(f"\nmanifesto da etapa: {me.etapa} · {me.manifesto_id[:16]}…")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
