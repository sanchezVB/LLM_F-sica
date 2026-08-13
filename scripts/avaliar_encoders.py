#!/usr/bin/env python3
"""Portão G1 — ΦEmb contra PhysBERT e embedders gerais (DOC-00 §5, DOC-07 §3).

Roda na venv de TREINO (Python 3.12), e em CPU por padrão para não disputar VRAM
com um treino em curso.

    .venv-treino/Scripts/python.exe scripts/avaliar_encoders.py

O cache (`--cache`) guarda as posições por item de cada modelo. Uma passada a
2.000 candidatos custa ~2,5 h de CPU e o PhysBERT sozinho leva 22 min; somar UM
modelo à comparação não deve custar a soma de todos. É invalidado inteiro se o
protocolo mudar — ver `comparar` em `phifm.eval.encoders`.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polars as pl  # noqa: E402

from phifm.eval.encoders import (  # noqa: E402
    comparar,
    salvar,
    tabela,
    tabela_pareada,
    veredito,
)

# Nossos candidatos. O `-melhor` é o checkpoint de pico, não o último passo — no
# treino sobre MiniLM o pico deu MRR 0,477 contra 0,469 do fim.
NOSSOS = {
    "ΦEmb/SciBERT (110M)": Path("models/phiemb"),
    "ΦEmb/MiniLM (23M)": Path("models/phiemb-minilm-melhor"),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    p.add_argument("--modelo", action="append", default=[], metavar="ROTULO=CAMINHO",
                   help="acrescenta um modelo nosso; repetível. Sem isto usa os padrões.")
    p.add_argument("--n", type=int, default=2000,
                   help="candidatos na avaliação; 256 não separa margens de ~0,02")
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--lote", type=int, default=16)
    p.add_argument("--dispositivo", default="cpu", choices=["cpu", "dml"])
    p.add_argument("--cache", type=Path,
                   default=Path("data/processed/avaliacao/g1_cache.json"),
                   help="posições por item, para não remedir; NÃO é o resultado")
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/avaliacao/g1_resultado.json"),
                   help="o resultado, que é versionado — o log do lançador não é")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    logging.info("validação: %s pares · usando os %d primeiros", f"{val.height:,}", a.n)

    pedidos = dict(NOSSOS)
    for spec in a.modelo:
        rotulo, _, caminho = spec.partition("=")
        if not caminho:
            logging.error("--modelo espera ROTULO=CAMINHO, recebi %r", spec)
            return 2
        pedidos[rotulo] = Path(caminho)

    extras = {}
    for rotulo, caminho in pedidos.items():
        if (caminho / "config.json").exists():
            extras[rotulo] = str(caminho)
        else:
            logging.warning("%s sem config.json — %s fora da comparação", caminho, rotulo)
    if not extras:
        logging.error("nenhum modelo nosso disponível; nada a comparar")
        return 1

    rs = comparar(val, extras, cache=a.cache, n=a.n, max_tokens=a.max_tokens,
                  lote=a.lote, dispositivo=a.dispositivo)

    print("\n" + "=" * 78)
    print(tabela(rs, a.n))
    print("=" * 78)
    print()
    print(tabela_pareada(rs))
    print()
    print("=" * 78)
    print(veredito(rs, a.n))
    salvar(rs, a.out, a.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
