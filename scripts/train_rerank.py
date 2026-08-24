#!/usr/bin/env python3
"""ΦRank — treina o cross-encoder de reranking (DOC-07 §4).

    .venv-treino/Scripts/python.exe scripts/train_rerank.py --max-grupos 12500

⚠️ Use `--negativos` apontando para os negativos **limpos**
(`scripts/filtrar_cocitacao.py`). Com os não filtrados, 15,4% dos negativos são
documentos co-citados com o positivo — treinar um reranker a rebaixá-los é ensinar a
rebaixar o que é relevante, e o efeito é invisível no treino porque a perda desce
normalmente.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    Entrada,
    gravar_manifesto_etapa,
)
from phifm.core.sistema import impedir_suspensao, liberar_suspensao  # noqa: E402
from phifm.training.rerank import BASE_PADRAO, ConfigRank, TreinadorRank  # noqa: E402

LIMPOS = Path("data/processed/negativos_dificeis/pares_limpos.parquet")


def _dividir_por_documento(d: pl.DataFrame, val_frac: float, minimo: int,
                           semente: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Divide por DOCUMENTO CITADO, não por linha.

    ⚠️ Isto conserta um defeito de desenho experimental que produziu um resultado
    bonito e falso em 2026-08-21.

    A primeira versão separava as últimas 8.000 LINHAS, e o comentário dizia que era
    para evitar vazamento. Medido depois:

        âncoras da "validação" já vistas no treino : 49,6%
        documentos citados já vistos no treino     : 39,2%

    A causa está num número que este projeto já tinha medido três dias antes e que
    eu não liguei: os 400 mil pares têm apenas **17.844 documentos citados
    distintos**, cada um repetindo ~22 vezes. Cortar por posição num conjunto assim
    não separa nada — os mesmos papers caem dos dois lados.

    O modelo então aprendeu "este paper específico é positivo" em vez de "este par é
    relevante". Reportou acerto@1 0,370, e sobre documentos inéditos derrubou o nDCG
    da composição de 0,139 para 0,020 — o `pares_validacao.parquet` real tem 88.807
    citados distintos, dos quais só 4,8% o ΦRank tinha visto.

    Dividir por documento garante que nenhum paper citado apareça dos dois lados. A
    métrica passa a medir generalização, e vai CAIR — o número honesto é menor que o
    inflado, e é o que serve para decidir.
    """
    import random

    citados = sorted(set(d["arxiv_citado"].to_list()))
    rng = random.Random(semente)
    rng.shuffle(citados)
    n = max(int(len(citados) * val_frac), 1)
    reservados = set(citados[:n])

    val = d.filter(pl.col("arxiv_citado").is_in(reservados))
    treino = d.filter(~pl.col("arxiv_citado").is_in(reservados))
    vaz_c = len(set(val["arxiv_citado"]) & set(treino["arxiv_citado"]))
    vaz_a = len(set(val["arxiv_id"]) & set(treino["arxiv_id"]))
    logging.info("divisão POR DOCUMENTO: %s citados reservados de %s",
                 f"{len(reservados):,}", f"{len(citados):,}")
    logging.info("grupos: %s treino · %s validação", f"{len(treino):,}", f"{len(val):,}")
    logging.info("vazamento: %d documentos citados, %d âncoras", vaz_c, vaz_a)
    if vaz_c:
        raise SystemExit(f"{vaz_c} documentos citados nos dois lados — a divisão "
                         "por documento falhou e a métrica seria inflada")
    if len(val) < minimo:
        logging.warning("validação com %s grupos, menos que os %s pedidos por "
                        "--grupos-aval; a métrica fica mais ruidosa",
                        f"{len(val):,}", f"{minimo:,}")
    # ⚠️ Âncoras nos dois lados são INEVITÁVEIS e não são o mesmo problema: um paper
    # que cita A (treino) e B (validação) aparece nos dois, mas as CONSULTAS são
    # diferentes e o documento a recuperar é inédito. O que envenena é o documento
    # citado repetido, porque é ele que o modelo memoriza.
    if vaz_a:
        logging.info("  (âncoras repetidas são esperadas: mesma consulta, documento "
                     "alvo inédito — não é o vazamento que importa)")
    return treino, val


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--negativos", type=Path, default=LIMPOS)
    p.add_argument("--out", type=Path, default=Path("models/phirank-minilm"))
    p.add_argument("--base", default=BASE_PADRAO)
    p.add_argument("--grupos", type=int, default=4,
                   help="grupos por passo; o lote em textos é grupos x (1+negativos)")
    p.add_argument("--n-negativos", type=int, default=7)
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-grupos", type=int, default=None,
                   help="teto de grupos; 12.500 grupos de 8 = 100 mil exemplos, "
                        "~51 min medidos nesta máquina")
    p.add_argument("--passos-aval", type=int, default=500)
    p.add_argument("--grupos-aval", type=int, default=500)
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="fração de DOCUMENTOS CITADOS reservada para validação")
    p.add_argument("--semente", type=int, default=17)
    p.add_argument("--dispositivo", default="auto",
                   choices=["auto", "cuda", "dml", "cpu"])
    p.add_argument("--sem-amp", action="store_true")
    a = p.parse_args()

    for fluxo in (sys.stdout, sys.stderr):
        try:
            fluxo.reconfigure(encoding="utf-8")
        except Exception:
            pass
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    if not a.negativos.exists():
        raise SystemExit(
            f"{a.negativos} não existe. Rode scripts/minerar_negativos.py e depois "
            "scripts/filtrar_cocitacao.py — sem o filtro, 15,4% dos negativos são "
            "co-citados com o positivo e o reranker aprende a rebaixar relevantes.")

    d = pl.read_parquet(a.negativos,
                        columns=["arxiv_id", "arxiv_citado", "ancora", "positivo",
                                 "negativos"])
    treino, val = _dividir_por_documento(d, a.val_frac, a.grupos_aval, a.semente)

    cfg = ConfigRank(base=a.base, grupos=a.grupos, n_negativos=a.n_negativos,
                     max_tokens=a.max_tokens, lr=a.lr, max_grupos=a.max_grupos,
                     passos_aval=a.passos_aval, grupos_aval=a.grupos_aval,
                     dispositivo=a.dispositivo, amp=not a.sem_amp)

    impedir_suspensao()
    try:
        m = TreinadorRank(cfg).treinar(treino, val, a.out)
    finally:
        liberar_suspensao()

    gravar_manifesto_etapa(
        etapa="phirank",
        descricao="Cross-encoder de reranking sobre MiniLM (DOC-07 §4)",
        raiz=a.out,
        entradas=[Entrada(caminho=str(a.negativos))],
        parametros={"script": "scripts/train_rerank.py", "base": a.base,
                    "grupos": a.grupos, "n_negativos": a.n_negativos,
                    "max_tokens": a.max_tokens, "max_grupos": a.max_grupos,
                    "acerto_ao_acaso": round(1 / (1 + a.n_negativos), 4),
                    "desvio_de_especificacao": (
                        "DOC-07 §4 pede ΦEnc; usado MiniLM porque o ΦEnc não existe")},
        registros=m.passo)

    print()
    print("=" * 66)
    print(f"  ΦRank · passo {m.passo:,} · acerto@1 {m.acerto_top1:.3f} "
          f"(acaso {1/(1+a.n_negativos):.3f}) · MRR {m.mrr_grupo:.3f}")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
