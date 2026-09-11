#!/usr/bin/env python3
"""Pares de citação para o ΦEmb (DOC-07 §3.1)."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    entrada_de,
    gravar_manifesto_etapa,
)
from phifm.training.pairs import (  # noqa: E402
    FRACAO_VALIDACAO,
    MAX_POR_ANCORA,
    construir,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, default=Path("data/raw/openalex_snapshot"))
    p.add_argument("--spine", type=Path, default=Path("data/processed/spine.parquet"))
    p.add_argument("--out", type=Path, default=Path("data/processed/pares"))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s",
                        stream=sys.stdout)
    tr, val = construir(a.snapshot, a.spine, a.out)
    print(f"\ntreino {tr.height:,} · validação {val.height:,}")

    # ⚠️ `MAX_POR_ANCORA` e `FRACAO_VALIDACAO` entram nos parâmetros porque são
    # DECISÕES, não detalhes de implementação: o teto por âncora existe para um
    # artigo de revisão com 300 referências não dominar o gradiente, e a fração
    # de validação define quanto do sinal sobra para medir. Reconstruí-los do
    # código daria o valor de HOJE, não o da execução que produziu o parquet —
    # e é exatamente aí que um parâmetro reconstruído mente sem ninguém ver.
    me = gravar_manifesto_etapa(
        etapa="pares_citacao",
        descricao="Pares âncora/positivo de citação para o treino contrastivo",
        raiz=a.out,
        entradas=[entrada_de(a.snapshot), entrada_de(a.spine)],
        parametros={"script": "scripts/build_pairs.py",
                    "snapshot": str(a.snapshot).replace("\\", "/"),
                    "spine": str(a.spine).replace("\\", "/"),
                    "max_por_ancora": MAX_POR_ANCORA,
                    "fracao_validacao": FRACAO_VALIDACAO,
                    "pares_treino": tr.height,
                    "pares_validacao": val.height},
        registros=tr.height + val.height)
    print(f"manifesto da etapa: {me.manifesto_id[:16]}… · {me.registros:,} pares")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
