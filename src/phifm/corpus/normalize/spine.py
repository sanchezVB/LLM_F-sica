"""Consolidação da espinha de metadados — o entregável do Sprint S1.

Transforma os shards brutos dos coletores em uma tabela única com:
  - deduplicação por `arxiv_id` (mantendo o registro mais recente)
  - licença resolvida em SPDX e partição decidida (`core.licensing`)
  - recorte de Física pela categoria **do autor**, não por classificador
  - grafo de citações anexado, quando o OpenAlex já tiver coletado

E emite o relatório que dimensiona o `PhysCorpus-Open` (ADR-0001 §6).
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from phifm.core.licensing.registry import resolve_partition, resolve_spdx

log = logging.getLogger(__name__)

# DOC-02 §2 — o seletor de Física. Prefixos de categoria do arXiv que
# constituem a família de Física. `math.*` entra como matemática de apoio,
# marcada separadamente para que o DOC-06 possa pesá-la à parte.
PHYSICS_PREFIXES = (
    "astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
    "math-ph", "nlin", "nucl-ex", "nucl-th", "physics", "quant-ph",
)
MATH_SUPPORT = ("math.AP", "math.CA", "math.DG", "math.OC", "math.PR",
                "math.ST", "math.NA", "math.RA", "math.FA", "math.DS",
                # Nomes LEGADOS das mesmas áreas. `is_math_support` casa a
                # categoria inteira, não o arquivo, então a normalização de
                # `_LEGADO` (que é arquivo→arquivo) não alcança estes: um paper
                # antigo traz a categoria `funct-an`, nunca `math.FA`. Sem isto,
                # 351 papers de análise funcional e geometria diferencial não
                # contavam como matemática de apoio.
                "funct-an", "dg-ga")

SUBFIELD_MAP = {
    "astro-ph": "Astrofísica e Cosmologia", "cond-mat": "Matéria Condensada",
    "gr-qc": "Relatividade e Gravitação", "hep-th": "Teoria de Campos e Cordas",
    "hep-ph": "Partículas (fenomenologia)", "hep-ex": "Partículas (experimental)",
    "hep-lat": "QCD na rede", "nucl-th": "Física Nuclear (teoria)",
    "nucl-ex": "Física Nuclear (experimental)", "quant-ph": "Mecânica Quântica",
    "math-ph": "Física Matemática", "nlin": "Dinâmica Não-Linear",
    "physics": "Física (diversos)",
}


# ─── arquivos legados do arXiv ───────────────────────────────────────────────
#
# Antes de 1998 o arXiv usava arquivos de primeiro nível que depois foram
# absorvidos. Eles CONTINUAM aparecendo como categoria primária dos papers
# antigos, e a lista acima não os conhecia. Medido em 2026-08-13 no spine:
#
#   4.042 papers com primária legada
#     · todos com subfield = "Outro"
#     · ZERO com is_physics = False
#
# O `is_physics` não sofria porque o arXiv retroagiu cross-lists modernas em todos
# eles — então a família era detectada por outra categoria. Mas `subfield` caía em
# "Outro", e `train()` DESCARTA "Outro": 4 mil papers de dinâmica não-linear,
# matéria condensada e física atômica ficavam fora do treino de subárea. São os
# mais antigos do acervo, justamente os de conteúdo clássico mais distintivo.
#
# Normalizar aqui, num lugar só, em vez de duplicar entradas em `PHYSICS_PREFIXES`
# e `SUBFIELD_MAP`: o legado passa a ser o arquivo moderno equivalente e tudo a
# jusante funciona sem saber que existiu legado.
# A lista cobre TODO legado, não só o de Física. Escrevi primeiro só os de Física
# — de memória — e perguntar ao dado achou cinco que faltavam: `q-alg`,
# `alg-geom`, `dg-ga`, `funct-an` e `cmp-lg`, legados de matemática e de CS.
#
# Não havia erro de rótulo: por não estarem em `PHYSICS_PREFIXES`, saíam como
# não-Física, que é o correto. Mas deixavam `q-alg` e `math` como buckets
# diferentes da mesma coisa, e o mapeamento é sobre a HISTÓRIA DO ARXIV — incluir
# só o lado da Física era arbitrário.
_LEGADO = {
    # ── Física ────────────────────────────────────────────────────────────
    # Ciências não-lineares, antes de `nlin` existir
    "chao-dyn": "nlin", "solv-int": "nlin", "patt-sol": "nlin",
    "comp-gas": "nlin", "adap-org": "nlin",
    # Absorvidos por `physics.*`
    "acc-phys": "physics", "ao-sci": "physics", "atom-ph": "physics",
    "bayes-an": "physics", "chem-ph": "physics", "plasm-ph": "physics",
    # Absorvidos por `cond-mat.*`
    "mtrl-th": "cond-mat", "supr-con": "cond-mat",
    # ── NÃO-Física, e continuam não sendo ─────────────────────────────────
    # Estão aqui para normalizar o bucket, não para virar Física: `math` e `cs`
    # não estão em `PHYSICS_PREFIXES`. Chegaram ao spine por cross-list, como
    # qualquer `math.AP` moderno.
    "q-alg": "math", "alg-geom": "math", "dg-ga": "math", "funct-an": "math",
    "cmp-lg": "cs",
}


def _archive(cat: str) -> str:
    """`cond-mat.str-el` → `cond-mat`; `hep-th` → `hep-th`; `chao-dyn` → `nlin`."""
    return _LEGADO.get(cat.split(".", 1)[0], cat.split(".", 1)[0])


def load_arxiv(raw_dir: Path) -> pl.DataFrame:
    """Carrega e deduplica os shards do arXiv.

    A deduplicação é necessária porque a retomada é "ao menos uma vez"
    (DOC-08 §7.2): um checkpoint interrompido refaz o lote pendente. Mantemos
    o registro de `datestamp` mais recente — que é a versão mais atual do
    metadado, não uma escolha arbitrária.
    """
    df = pl.read_parquet(str(raw_dir / "*.parquet"))
    before = df.height
    df = df.sort("datestamp", descending=True).unique(subset=["arxiv_id"], keep="first")
    log.info("arXiv: %s linhas → %s únicos (%.1f%% de duplicação por retomada)",
             f"{before:,}", f"{df.height:,}", 100 * (1 - df.height / before) if before else 0)
    return df


def annotate(df: pl.DataFrame) -> pl.DataFrame:
    """Anexa licença resolvida, partição e taxonomia."""
    return df.with_columns(
        pl.col("license").map_elements(resolve_spdx, return_dtype=pl.Utf8).alias("spdx_id"),
        pl.col("license").map_elements(resolve_partition, return_dtype=pl.Utf8).alias("partition"),
        pl.col("primary_category").map_elements(_archive, return_dtype=pl.Utf8).alias("archive"),
        # O mesmo `_LEGADO` aplicado à lista completa, não só à primária. Hoje é
        # redundante — todo paper legado tem cross-list moderna, medido — mas sem
        # isto um paper cujo ÚNICO arquivo fosse legado sairia com is_physics
        # falso, e a redundância é o que garante que isso nunca aconteça.
        pl.col("categories").list.eval(
            pl.element().str.split(".").list.first().replace(_LEGADO)
        ).list.unique().alias("archives"),
    ).with_columns(
        # Recorte de Física: qualquer categoria da família conta, não só a
        # primária. Um paper `math.AP` com cross-list em `gr-qc` é Física.
        pl.col("archives").list.eval(
            pl.element().is_in(list(PHYSICS_PREFIXES))
        ).list.any().alias("is_physics"),
        pl.col("categories").list.eval(
            pl.element().is_in(list(MATH_SUPPORT))
        ).list.any().alias("is_math_support"),
        pl.col("archive").replace_strict(SUBFIELD_MAP, default="Outro").alias("subfield"),
        pl.col("journal_ref").is_not_null().alias("peer_reviewed"),
        pl.col("created").str.slice(0, 4).cast(pl.Int32, strict=False).alias("year"),
    )


def attach_citations(df: pl.DataFrame, oa_dir: Path) -> pl.DataFrame:
    """Junta o grafo de citações do OpenAlex, se já houver dados."""
    shards = list(oa_dir.glob("*.parquet"))
    if not shards:
        log.info("OpenAlex ainda sem shards — junção adiada")
        return df.with_columns(
            pl.lit(None, dtype=pl.Int32).alias("n_references"),
            pl.lit(None, dtype=pl.Int32).alias("cited_by_count"),
        )
    # ⚠️ `referenced_works` NÃO entra, e a poda acontece no `select` ANTES do
    # filtro para que a coluna nunca seja lida do disco.
    #
    # Ela era produzida aqui e consumida em lugar nenhum: quem precisa do grafo
    # é `training/pairs.py`, que lê os shards do snapshot direto. Na espinha ela
    # só custava — 4,6 M de obras × ~45 identificadores é ~207 M de strings.
    #
    # Medido em 2026-08-07: com ela, `build_spine` sobre 692 shards consumiu a
    # RAM, foi para 14,2 GB de swap e o processo morreu depois de 2 h 22 sem
    # gravar nada. `scan_parquet` com motor de fluxo e a coluna fora resolve.
    oa = (
        pl.scan_parquet(str(oa_dir / "*.parquet"))
        .select("arxiv_id", "n_references", "cited_by_count")
        .filter(pl.col("arxiv_id").is_not_null())
        .unique(subset=["arxiv_id"], keep="first")
        .collect(engine="streaming")
    )
    out = df.join(oa, on="arxiv_id", how="left")
    matched = out["n_references"].is_not_null().sum()
    log.info("OpenAlex: %s obras, %s casadas com a espinha (%.1f%%)",
             f"{oa.height:,}", f"{matched:,}", 100 * matched / df.height if df.height else 0)
    return out


def report(df: pl.DataFrame) -> str:
    """Relatório que dimensiona o PhysCorpus-Open."""
    phys = df.filter(pl.col("is_physics"))
    L: list[str] = []
    a = L.append
    a(f"registros únicos          : {df.height:,}")
    a(f"família de Física         : {phys.height:,} ({100*phys.height/df.height:.1f}%)")
    a(f"matemática de apoio       : {df['is_math_support'].sum():,}")
    a(f"revisados por pares       : {phys['peer_reviewed'].sum():,} ({100*phys['peer_reviewed'].mean():.1f}%)")
    a("")
    a("--- partição (ADR-0001 §2) ---")
    for r in phys["partition"].value_counts(sort=True).iter_rows(named=True):
        a(f"  {r['partition']:12} {r['count']:>9,}  ({100*r['count']/phys.height:5.1f}%)")
    a("")
    a("--- licença SPDX ---")
    for r in phys["spdx_id"].value_counts(sort=True).head(8).iter_rows(named=True):
        a(f"  {r['spdx_id'][:44]:46} {r['count']:>9,}  ({100*r['count']/phys.height:5.1f}%)")
    a("")
    a("--- subárea ---")
    for r in phys["subfield"].value_counts(sort=True).iter_rows(named=True):
        a(f"  {r['subfield'][:34]:36} {r['count']:>9,}")
    if phys["year"].drop_nulls().len():
        a("")
        a("--- fração redistribuível por época (a opção CC é recente) ---")
        by = (phys.filter(pl.col("year").is_not_null())
                  .with_columns((pl.col("year") // 5 * 5).alias("lustro"))
                  .group_by("lustro")
                  .agg(pl.len().alias("n"),
                       (pl.col("partition") == "train_open").mean().alias("aberto"))
                  .sort("lustro"))
        for r in by.iter_rows(named=True):
            if r["n"] >= 100:
                a(f"  {r['lustro']}–{r['lustro']+4}  {r['n']:>8,}  aberto: {100*r['aberto']:5.1f}%")
    return "\n".join(L)


def build(raw_arxiv: Path, raw_openalex: Path, out: Path) -> pl.DataFrame:
    df = attach_citations(annotate(load_arxiv(raw_arxiv)), raw_openalex)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(out)
    log.info("→ %s (%s registros, %.1f MB)", out.name, f"{df.height:,}", out.stat().st_size / 1e6)
    return df
