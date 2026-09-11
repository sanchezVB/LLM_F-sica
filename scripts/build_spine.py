#!/usr/bin/env python3
"""Sprint S1 · consolidação — tabela mestra de metadados pronta para uso."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    entrada_de,
    gravar_manifesto_etapa,
)
from phifm.corpus.normalize.spine import build, report  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arxiv", type=Path, default=Path("data/raw/arxiv_metadata"))
    p.add_argument("--openalex", type=Path, default=Path("data/raw/openalex_works"))
    p.add_argument("--out", type=Path, default=Path("data/processed/spine.parquet"))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    df = build(a.arxiv, a.openalex, a.out)
    print("\n" + "=" * 66)
    print(report(df))
    print("=" * 66)

    # ⚠️ O manifesto sai DAQUI, com os caminhos que esta execução usou.
    #
    # Sem esta chamada, `manifesto_corpus.py` reconstrói os parâmetros lendo o
    # CÓDIGO e marca `parametros_reconstruidos=True` — e um parâmetro
    # reconstruído pode estar errado sem que nada acuse, porque não há com o que
    # confrontá-lo. Aqui `--arxiv` e `--openalex` são os de verdade, inclusive
    # quando alguém apontou para outro diretório.
    me = gravar_manifesto_etapa(
        etapa="spine",
        descricao=("Tabela mestra de metadados: arXiv juntado ao OpenAlex por "
                   "DOI e título"),
        raiz=a.out,
        entradas=[entrada_de(a.arxiv), entrada_de(a.openalex)],
        parametros={"script": "scripts/build_spine.py",
                    "arxiv": str(a.arxiv).replace("\\", "/"),
                    "openalex": str(a.openalex).replace("\\", "/")},
        registros=df.height)
    print(f"manifesto da etapa: {me.manifesto_id[:16]}… · "
          f"{me.registros:,} registros")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
