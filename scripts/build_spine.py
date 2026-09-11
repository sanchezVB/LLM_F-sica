#!/usr/bin/env python3
"""Sprint S1 · consolidação — tabela mestra de metadados pronta para uso.

## ⚠️ Uma reexecução deste script quase destruiu o grafo de citações

Medido em 2026-09-11. `data/raw/openalex_works` — a entrada que traz
`n_references` e `cited_by_count` — **não existe mais**: foi consumida ou
descartada depois que a tabela mestra foi construída, em 2026-08-07.

`attach_citations` trata "sem shards" como caso normal e devolve as duas colunas
nulas ("junção adiada"), o que é **certo na primeira construção** — a tabela mestra
podia ser montada antes de o OpenAlex chegar. Numa REEXECUÇÃO é destruição: o
script juntou com vazio, gravou `n_references` 100% nulo onde havia 14.052.319
referências, imprimiu o relatório inteiro e saiu com **código 0**.

O que sustenta os 6.697.651 pares de treino do ΦEmb é justamente esse grafo. Só
não se perdeu porque havia backup.

⚠️ E a lição mais desconfortável: a reexecução gravou um manifesto de etapa com
`parametros_reconstruidos=False`, atestando como "capturado na execução" um
artefato degradado. **Um manifesto capturado de uma execução ruim é pior que um
reconstruído**, porque tem mais credibilidade e nada o contradiz. Capturar
parâmetros é necessário e não é suficiente: o script tem de validar a própria
entrada, que é o que a guarda abaixo faz.
"""
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
from phifm.corpus.normalize.spine import build, report  # noqa: E402


def recusar_regressao(openalex: Path, out: Path) -> None:
    """Recusa trocar uma tabela mestra COM citações por uma sem. Ver a docstring.

    A regra é sobre a REGRESSÃO, não sobre a entrada faltar: construir sem o
    OpenAlex continua legítimo quando não há nada a perder. O que não pode é a
    junção adiada passar por cima de um grafo que já existe.
    """
    if list(openalex.glob("*.parquet")) if openalex.exists() else False:
        return                      # há shards: a junção vai acontecer
    if not out.exists():
        return                      # nada a perder: primeira construção
    try:
        nao_nulos = (pl.scan_parquet(out)
                     .select(pl.col("n_references").is_not_null().sum())
                     .collect().item())
    except Exception:
        return                      # sem a coluna, não há regressão a impedir
    if not nao_nulos:
        return                      # a que está lá também não tem citações
    raise SystemExit(
        f"{openalex} não tem shards do OpenAlex, e {out} já tem {nao_nulos:,} "
        "registros COM grafo de citações.\n\n"
        "Seguir gravaria `n_references` e `cited_by_count` nulos por cima deles, "
        "com código de saída 0 e um relatório completo — foi o que aconteceu em "
        "2026-09-11, e só não se perdeu o grafo porque havia backup. É ele que "
        "sustenta os 6,7 M de pares de treino do ΦEmb.\n\n"
        "Se a intenção é mesmo reconstruir sem citações, aponte `--out` para "
        "outro caminho."
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arxiv", type=Path, default=Path("data/raw/arxiv_metadata"))
    p.add_argument("--openalex", type=Path, default=Path("data/raw/openalex_works"))
    p.add_argument("--out", type=Path, default=Path("data/processed/spine.parquet"))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    # ⚠️ ANTES de `build`, que grava dentro. Uma guarda depois da escrita não é
    # guarda: ela relata um estrago já feito.
    recusar_regressao(a.openalex, a.out)
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
