#!/usr/bin/env python3
"""Sprint S3 · 4b — fatia de Física do RedPajama-arXiv, filtrada pelo spine.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/coletar_redpajama.py

Em fluxo: os 81 GB nunca aterram. Ver a docstring de
`phifm.corpus.slices.redpajama` para o desenho e as medições.
"""
import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.env import contato_obrigatorio  # noqa: E402
from phifm.core.sistema import impedir_suspensao, liberar_suspensao  # noqa: E402
from phifm.corpus.slices.redpajama import REVISAO, coletar  # noqa: E402
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    Entrada,
    gravar_manifesto_etapa,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/processed/redpajama_fisica"))
    p.add_argument("--spine", type=Path, default=Path("data/processed/spine.parquet"))
    p.add_argument("--max-shards", type=int, default=None,
                   help="teto nesta execução; retomável pelo que já está em disco")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    impedir_suspensao()
    try:
        pr = coletar(a.out, a.spine, a.max_shards, contato_obrigatorio())
    finally:
        liberar_suspensao()

    print("\n" + "=" * 70)
    print(f"RedPajama-arXiv · fatia de Física ({pr.shards_lidos} shards)")
    print()
    print(f"  registros vistos    : {pr.registros_vistos:,}")
    print(f"  guardados (Física)  : {pr.registros_guardados:,}  "
          f"({100*pr.taxa_fisica:.1f}%)")
    print(f"  lidos da rede       : {pr.bytes_lidos/1e9:.1f} GB")
    print(f"  texto guardado      : {pr.caracteres_guardados/1e9:.1f} G caracteres")
    # ~4 caracteres por token é a razão usual em inglês técnico. Estimativa, e
    # marcada como tal: o número exato depende do tokenizer, que é o DOC-05.
    print(f"  ≈ tokens            : {pr.caracteres_guardados/4/1e9:.1f} B (a ~4 chars/token)")
    print(f"  falhas              : {len(pr.falhas)}")
    for f in pr.falhas[:5]:
        print(f"    · {f}")
    print("=" * 70)
    print("O RedPajama perde 16,6% das equações (IC 12,9–20,8). Isto é a linha de")
    print("base gratuita; o bulk pago do arXiv se mede CONTRA ela, não no vácuo.")

    saida = a.out / "_progresso.json"
    saida.write_text(json.dumps(asdict(pr), indent=2, ensure_ascii=False), encoding="utf-8")
    me = gravar_manifesto_etapa(
        etapa="redpajama_fisica",
        descricao=("Fatia de Física do RedPajama-arXiv, filtrada por casamento "
                   "exato com a espinha"),
        raiz=a.out,
        entradas=[Entrada(caminho=str(a.spine))],
        parametros={"script": "scripts/coletar_redpajama.py",
                    "filtro": "spine (exato)", "max_shards": a.max_shards,
                    "revisao_indice": REVISAO,
                    "shards_lidos": pr.shards_lidos,
                    "registros_vistos": pr.registros_vistos},
        registros=pr.registros_guardados)
    print(f"manifesto da etapa: {me.manifesto_id[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
