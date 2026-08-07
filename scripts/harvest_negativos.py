#!/usr/bin/env python3
"""Sprint S2 — negativos para a classe `não-física` (DOC-02 §6).

O classificador do §6 é `fastText` sobre as 23 subáreas do §2 **mais uma classe
`não-física`**. Os positivos já existem: 1.595.422 registros do set `physics`,
rotulados de graça pelos próprios autores. Faltam os negativos, e é isso aqui.

## ⚠️ Pertencer ao set NÃO significa ser negativo

Medido em 2026-08-07 sobre 1.300 registros de cada set, olhando a categoria
**primária** de cada um:

| Set | Contaminado por Física | Primárias mais comuns |
|---|---|---|
| `cs` | 4,1% | cs.CL, cs.AI, cmp-lg |
| **`q-bio`** | **63,0%** | **physics.bio-ph, cond-mat.stat-mech, cond-mat.soft** |
| `econ` | 3,2% | econ.EM, econ.GN, econ.TH |

O set do OAI-PMH inclui os trabalhos **cruzados** para aquela área, não só os
que nasceram nela. Biofísica é a zona de sobreposição por excelência: quase
dois terços do set `q-bio` são artigos cuja categoria primária é Física, e o
autor apenas os listou também em q-bio.

Rotular isso como não-física seria ruído de rótulo **exatamente na fronteira
que o classificador precisa aprender** — o pior lugar possível para errar.

**A regra é a mesma que o `openalex.py` já aplica:** a categoria primária
atribuída pelo autor é autoritativa; a filiação a um set é só um seletor
grosseiro. Por isso a coleta guarda tudo (bruto é bruto, DOC-03) e o `resumo`
ao final reporta o número **utilizável**, não o número baixado.

`math` cai na mesma armadilha, e mais forte: o arXiv expõe `math:math:MP` e
`physics:math-ph` como a **mesma** Mathematical Physics. Fica de fora aqui, e
seria um bom negativo difícil se filtrado por primária.

`eess`, `stat` e `q-fin` são negativos limpos e poderiam entrar. Não entram
porque o plano não os pediu e mais negativo fácil não ensina fronteira nenhuma.

## Cortesia

Os três rodam **em sequência, nunca em paralelo**. O limite do arXiv é de uma
requisição a cada 3 segundos para o cliente inteiro, não por coleta — três
processos simultâneos triplicariam a taxa vista do outro lado e é exatamente o
que o princípio A5 proíbe.

Cada set tem manifesto próprio, então a retomada é independente: se `cs` cair
na metade, `q-bio` e `econ` não são refeitos.
"""
import argparse, logging, sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.env import contato_obrigatorio  # noqa: E402
from phifm.core.sistema import impedir_suspensao, liberar_suspensao  # noqa: E402
from phifm.corpus.acquire.arxiv import harvest_physics  # noqa: E402

# `harvest_physics` é genérico apesar do nome — recebe o set como parâmetro.
SETS = ("cs", "q-bio", "econ")

# Arquivos de Física do arXiv, como aparecem na categoria primária. `physics.`
# cobre as ~20 subáreas de `physics.*`; os demais são arquivos de primeiro nível.
ARQUIVOS_FISICA = (
    "astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
    "math-ph", "nlin", "nucl-ex", "nucl-th", "quant-ph", "physics.",
)


def e_fisica(primaria: str | None) -> bool:
    """A categoria primária é de Física?

    Prefixo, não igualdade: `cond-mat.stat-mech` e `physics.bio-ph` precisam
    casar. `nlin` entra porque Ciências Não-Lineares está sob `physics:nlin`
    na taxonomia OAI do arXiv.
    """
    if not primaria:
        return False
    return any(primaria.startswith(a) for a in ARQUIVOS_FISICA)


def resumir(destino: Path) -> tuple[int, int]:
    """(baixados, utilizáveis como negativo) num diretório de set."""
    shards = sorted(destino.glob("part-*.parquet"))
    if not shards:
        return 0, 0
    df = pl.read_parquet(shards, columns=["primary_category"])
    fisica = df["primary_category"].map_elements(e_fisica, return_dtype=pl.Boolean).sum()
    return len(df), len(df) - int(fisica)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/raw/arxiv_negativos"))
    p.add_argument("--sets", nargs="+", default=list(SETS))
    p.add_argument("--max-pages", type=int, default=None,
                   help="teto POR SET nesta execução (retomável)")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    contact = contato_obrigatorio()
    logging.info("identificação de coleta: %s", contact)
    logging.info("sets a coletar, em sequência: %s", ", ".join(a.sets))

    impedir_suspensao()
    falhas = 0
    incompletos = []
    destinos: list[tuple[str, Path]] = []
    try:
        for s in a.sets:
            destino = a.out / s.replace("-", "_")
            destinos.append((s, destino))
            logging.info("─── set %s → %s ───", s, destino)
            m = harvest_physics(destino, s, None, None, a.max_pages, contact)
            falhas += len(m.failures)
            if not m.completed_at:
                incompletos.append(s)
            logging.info("set %s: %s registros%s", s, f"{m.actual_count:,}",
                         "" if m.completed_at else " (parcial, retomável)")
    finally:
        liberar_suspensao()

    # O número que interessa é o UTILIZÁVEL, não o baixado — ver a docstring.
    estado = "concluído" if not incompletos else f"parcial — faltam: {', '.join(incompletos)}"
    print(f"\n{estado} · falhas: {falhas}\n")
    print(f"{'set':10} {'baixados':>12} {'utilizáveis':>12} {'contaminação':>14}")
    tb = tu = 0
    for s, destino in destinos:
        b, u = resumir(destino)
        tb += b
        tu += u
        pct = f"{100 * (b - u) / b:.1f}%" if b else "—"
        print(f"{s:10} {b:12,} {u:12,} {pct:>14}")
    print(f"{'TOTAL':10} {tb:12,} {tu:12,} {(f'{100*(tb-tu)/tb:.1f}%' if tb else '—'):>14}")
    print("\nUtilizável = categoria primária NÃO é de Física. Pertencer ao set não basta:")
    print("o set inclui trabalhos cruzados, e em q-bio isso é a maioria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
