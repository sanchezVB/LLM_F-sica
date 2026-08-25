#!/usr/bin/env python3
"""S3b — auditoria de preservação de LaTeX do RedPajama (DOC-02 §3.2).

Decide se as fatias de terceiro servem, ANTES de construir corpus em cima delas
e antes de qualquer decisão de pagar egress do arXiv.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/auditar_latex.py
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.env import contato_obrigatorio  # noqa: E402
from phifm.eval.latex_audit import N_PAPERS, auditar, salvar  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=N_PAPERS,
                   help="papers a auditar; 200 já dá erro padrão de ~2%% por paper")
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/avaliacao/s3b_latex.json"))
    p.add_argument("--spine", type=Path, default=Path("data/processed/spine.parquet"),
                   # `%%` porque argparse formata a ajuda com o operador `%`.
                   help="restringe a amostra a Física; o shard do RedPajama é o "
                        "arXiv INTEIRO e 52%% dele não é Física")
    p.add_argument("--cache", type=Path, default=Path("data/raw/arxiv_fontes"),
                   help="tarballs e amostra do RedPajama; evita rebaixar a cada "
                        "iteração no comparador (cortesia, princípio A5)")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    contato_obrigatorio()          # cortesia com o arXiv, igual às coletas

    r = salvar(auditar(a.n, cache=a.cache, spine=a.spine), a.out)
    print("\n" + "=" * 68)
    if "erro" in r:
        print(r["erro"])
        return 1
    print(f"S3b — preservação de equações do RedPajama-arXiv ({r['papers_comparados']} papers)")
    print()
    print(f"  equações na fonte          : {r['equacoes_na_fonte']:,}")
    print(f"  preservadas no RedPajama   : {r['equacoes_preservadas']:,}")
    print(f"  preservação por equação    : {100*r['preservacao_por_equacao']:.1f}%")
    print(f"  mediana por paper          : {100*r['preservacao_mediana_por_paper']:.1f}%")
    print(f"  papers abaixo de 90%       : {r['papers_abaixo_de_90pc']} de {r['papers_comparados']}")
    print(f"  papers com erro de coleta  : {r['papers_com_erro']}")
    print("  excluídos (fonte montada por concatenação, contagem inflável)")
    print(f"                             : {r['papers_por_concatenacao_excluidos']}")
    print()
    print(f"  DEGRADAÇÃO TOTAL: {100*r['degradacao_total']:.1f}%   "
          f"(limiar do DOC-02: {100*r['limiar_do_doc02']:.0f}%)")
    print(f"    por AUSÊNCIA     : {100*r['degradacao_por_ausencia']:.1f}%  "
          "← perda real de conteúdo; só isto justifica pagar")
    print(f"    por DISCORDÂNCIA : {100*r['degradacao_por_discordancia']:.1f}%  "
          "← notação, ou resíduo do nosso comparador")
    ic = r.get("ausencia_ic95") or [None, None]
    if ic[0] is not None:
        print(f"    IC 95% da ausência: [{100*ic[0]:.1f}%, {100*ic[1]:.1f}%] "
              f"· P(>10%) = {100*r['ausencia_p_acima_do_limiar']:.0f}%")
    if r.get("degradacao_total_sem_filtro") is not None:
        print(f"    (sem filtrar por montagem daria {100*r['degradacao_total_sem_filtro']:.1f}% "
              "— a diferença É o viés da concatenação)")
    print("=" * 68)
    print(r["veredito"])
    print(f"\ndetalhe por paper em {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
