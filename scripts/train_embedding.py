#!/usr/bin/env python3
"""ΦEmb — fine-tune contrastivo sobre pares de citação (DOC-07 §3).

Roda na venv de TREINO, que é Python 3.12: o `torch-directml` não suporta 3.14.

    .venv-treino/Scripts/python.exe scripts/train_embedding.py --max-pares 20000
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polars as pl  # noqa: E402

from phifm.training.embedding import BASE_PADRAO, Config, TreinadorEmb  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    p.add_argument("--out", type=Path, default=Path("models/phiemb"))
    p.add_argument("--base", default=BASE_PADRAO)
    p.add_argument("--lote", type=int, default=16,
                   help="lote LÓGICO: é dele que saem os negativos do InfoNCE")
    p.add_argument("--sub-lote", type=int, default=None,
                   help="liga o GradCache. É o que vai à GPU de cada vez; o `--lote` "
                        "passa a ser só lógico. `--lote 512 --sub-lote 128` dá 511 "
                        "negativos com a memória de 128. Ver test_gradcache.py")
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-pares", type=int, default=None,
                   help="teto de pares nesta execução; 1,65 M leva ~90 h aqui")
    p.add_argument("--passos-aval", type=int, default=500)
    p.add_argument("--n-candidatos", type=int, default=256,
                   help="pool da avaliação; 256 dá erro padrão de ±0,031")
    p.add_argument("--dispositivo", default="auto", choices=["auto", "dml", "cpu"])
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    # ⚠️ Leitura PREGUIÇOSA com o corte antes do `collect`. `pl.read_parquet`
    # carregava os 6,56 M de pares inteiros — 2,46 GB comprimidos que expandem
    # para 8+ GB de texto — e só depois `--max-pares` cortava. Medido em
    # 2026-08-07: 1,0 GB de RAM livre, 31 GB de swap, e o processo morria antes
    # do primeiro passo sem deixar traceback. O corte tem de acontecer no plano,
    # não depois dele.
    plano = pl.scan_parquet(a.pares / "pares_treino.parquet")
    total = plano.select(pl.len()).collect().item()
    treino = (plano.head(a.max_pares) if a.max_pares else plano).collect()
    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    logging.info("pares: %s de %s disponíveis · %s validação",
                 f"{treino.height:,}", f"{total:,}", f"{val.height:,}")

    cfg = Config(base=a.base, lote=a.lote, sub_lote=a.sub_lote,
                 max_tokens=a.max_tokens, lr=a.lr,
                 max_pares=a.max_pares, passos_aval=a.passos_aval,
                 n_candidatos=a.n_candidatos, dispositivo=a.dispositivo)
    if a.sub_lote:
        logging.info("GradCache ligado: lote lógico %d (%d negativos) com a memória "
                     "de %d", a.lote, a.lote - 1, a.sub_lote)
    m = TreinadorEmb(cfg).treinar(treino, val, a.out)

    antes = m.historico[0]
    print(f"\n{'':22} {'recall@1':>10} {'recall@10':>11} {'MRR':>8}")
    print(f"{'antes (base)':22} {antes['recall_1']:10.3f} {antes['recall_10']:11.3f} {antes['mrr']:8.3f}")
    print(f"{'depois (ΦEmb)':22} {m.recall_1:10.3f} {m.recall_10:11.3f} {m.mrr:8.3f}")
    d1 = m.recall_1 - antes["recall_1"]
    print(f"{'ganho':22} {d1:+10.3f} {m.recall_10-antes['recall_10']:+11.3f} "
          f"{m.mrr-antes['mrr']:+8.3f}")
    print(f"\n{m.passo} passos · {m.pares_por_s:.1f} pares/s · → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
