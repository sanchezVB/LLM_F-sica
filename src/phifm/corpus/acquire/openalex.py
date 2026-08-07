"""Coletor do OpenAlex — grafo de citações e identificadores cruzados.

Segunda etapa do Sprint S1 (DOC-02 §9). O que o OpenAlex acrescenta sobre a
espinha do arXiv: **`referenced_works`**, ou seja, a lista de referências
resolvidas de cada trabalho. É a supervisão gratuita que o DOC-07 §3.1
identificou para treinar o ΦEmb — dezenas de milhões de pares positivos
(paper → paper citado) sem nenhuma anotação.

Duas armadilhas de filtro medidas em 2026-08-03, ambas com potencial de
arruinar o recorte em silêncio:

1. **`primary_location` vs `locations`.** Filtrar por
   `primary_location.source.id:<arXiv>` retorna 2,23 M trabalhos;
   `locations.source.id:<arXiv>` retorna 3,67 M. A diferença de 1,44 M são os
   papers **publicados em revista**: quando isso acontece, a revista vira a
   localização primária e o arXiv passa a secundária. Usar `primary_location`
   descartaria sistematicamente o material **revisado por pares** — viés
   exatamente na direção errada.

2. **A classificação de campo do OpenAlex não serve de filtro de Física.**
   Dos trabalhos com origem no arXiv, o OpenAlex atribui 1,28 M a
   "Physics and Astronomy", mas também 903 k a "Computer Science", 569 k a
   "Mathematics" e 378 k a "Engineering". Boa parte dessas é Física legítima
   (quant-ph com aprendizado de máquina, cond-mat com materiais). Filtrar por
   campo do OpenAlex perderia mais da metade do corpus.

**Consequência de projeto, e é o princípio A1 se pagando:** não filtramos por
campo aqui. Coletamos tudo com origem no arXiv e o recorte de Física vem da
**categoria atribuída pelo autor** na espinha do arXiv (`arxiv.py`), que é
autoritativa. O OpenAlex entra apenas como enriquecimento.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import polars as pl

from phifm.core.schema.manifest import (
    AcquisitionManifest,
    HarvestMethod,
    LicenseResolution,
    RateLimit,
    canonical_hash,
)
from phifm.corpus.acquire.base import CONTACT, ResumableHarvester

log = logging.getLogger(__name__)

ENDPOINT = "https://api.openalex.org/works"
ARXIV_SOURCE_ID = "S4306400194"  # arXiv (Cornell University), verificado 2026-08-03

# `select` reduz o payload de ~30 KB para ~9,5 KB por obra.
#
# `locations` é caro (1,7× do payload) e mesmo assim é obrigatório: sem ele a
# chave de junção com a espinha do arXiv cai de 98,5% para 1,5%. Medido em
# 2026-08-03 sobre 200 obras — ver §"chave de junção" no parser abaixo.
SELECT = ",".join(
    [
        "id", "doi", "title", "publication_year", "publication_date", "type",
        "ids", "referenced_works", "cited_by_count", "open_access",
        "primary_topic", "language", "locations",
    ]
)

PER_PAGE = 200  # máximo permitido pela API
FLUSH_EVERY = 50_000

SCHEMA: dict[str, pl.DataType] = {
    "openalex_id": pl.Utf8,
    "arxiv_id": pl.Utf8,
    "doi": pl.Utf8,
    "title": pl.Utf8,
    "publication_year": pl.Int32,
    "publication_date": pl.Utf8,
    "type": pl.Utf8,
    "language": pl.Utf8,
    "is_oa": pl.Boolean,
    "oa_status": pl.Utf8,
    "cited_by_count": pl.Int32,
    "n_references": pl.Int32,
    "referenced_works": pl.List(pl.Utf8),
    "openalex_field": pl.Utf8,
    "openalex_topic": pl.Utf8,
}

# Chave de junção. O arXiv ID NÃO está em `ids.arxiv` — esse campo não existe.
# Ele aparece em dois lugares, com coberturas radicalmente diferentes (medido
# sobre 200 obras em 2026-08-03):
#
#   DOI DataCite `10.48550/arXiv.<id>`   →   1,5% de cobertura
#   `locations[].landing_page_url`       →  98,5% de cobertura
#
# A razão da diferença: quando o paper é publicado em revista, o campo `doi`
# passa a ser o DOI da EDITORA, não o do arXiv. Como papers publicados são
# justamente os que mais nos interessam (revisados por pares), depender só do
# DOI perderia quase tudo. Por isso `locations` entra no SELECT apesar do custo.
# O arXiv tem DUAS formas de identificador e o padrão precisa aceitar as duas:
#   novo (2007+) : 2405.12345      → \d{4}\.\d{4,5}
#   antigo       : hep-th/9711200  → arquivo[.SUBCLASSE]/AAMMNNN
# Um padrão ingênuo como `[^/]+` trunca o antigo em "hep-th" e descarta em
# silêncio ~30% do acervo — tudo o que é anterior a 2007, inclusive a
# literatura fundacional de teoria de cordas e cosmologia.
_ID_CORE = r"(?:[a-z][a-z-]*(?:\.[A-Za-z]{2})?/\d{7}|\d{4}\.\d{4,5})"
_ARXIV_URL_RE = re.compile(rf"arxiv\.org/abs/({_ID_CORE})", re.I)
_ARXIV_DOI_RE = re.compile(rf"10\.48550/arxiv\.({_ID_CORE})", re.I)
_VERSION_RE = re.compile(r"v\d+$", re.I)


def extract_arxiv_id(w: dict) -> str | None:
    """Extrai o arXiv ID, normalizando o sufixo de versão (`1412.6980v5` → `1412.6980`).

    Sem a normalização, `2405.12345` e `2405.12345v2` seriam chaves distintas e
    a junção com a espinha do arXiv falharia silenciosamente numa fração dos
    registros — o pior tipo de defeito, porque a contagem parece plausível.
    """
    doi = w.get("doi") or ""
    if m := _ARXIV_DOI_RE.search(doi):
        return _VERSION_RE.sub("", m.group(1))
    for loc in w.get("locations") or []:
        url = loc.get("landing_page_url") or ""
        if m := _ARXIV_URL_RE.search(url):
            return _VERSION_RE.sub("", m.group(1))
    return None


def _short(oid: str | None) -> str | None:
    """https://openalex.org/W123 → W123. Guardar a URL inteira em cada uma das
    ~50 M arestas de citação desperdiçaria a maior parte do arquivo."""
    return oid.rsplit("/", 1)[-1] if oid else None


def parse_work(w: dict) -> dict:
    arxiv_id = extract_arxiv_id(w)
    topic = w.get("primary_topic") or {}
    oa = w.get("open_access") or {}
    refs = [_short(r) for r in (w.get("referenced_works") or [])]

    return {
        "openalex_id": _short(w.get("id")),
        "arxiv_id": arxiv_id,
        "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
        "title": w.get("title"),
        "publication_year": w.get("publication_year"),
        "publication_date": w.get("publication_date"),
        "type": w.get("type"),
        "language": w.get("language"),
        "is_oa": oa.get("is_oa"),
        "oa_status": oa.get("oa_status"),
        "cited_by_count": w.get("cited_by_count"),
        "n_references": len(refs),
        "referenced_works": [r for r in refs if r],
        "openalex_field": ((topic.get("field") or {}) or {}).get("display_name"),
        "openalex_topic": topic.get("display_name"),
    }


class OpenAlexHarvester(ResumableHarvester):
    """Coletor com paginação por cursor, retomável."""

    @staticmethod
    def make_manifest(filter_expr: str | None = None) -> AcquisitionManifest:
        # Ver docstring do módulo: `locations`, não `primary_location`.
        filter_expr = filter_expr or f"locations.source.id:{ARXIV_SOURCE_ID}"
        return AcquisitionManifest(
            source_name="openalex_works",
            harvest_method=HarvestMethod.REST_API,
            endpoint=ENDPOINT,
            query_spec={"filter": filter_expr, "select": SELECT, "per_page": PER_PAGE},
            # A documentação do OpenAlex fala em 10 req/s no polite pool, mas
            # em uso SUSTENTADO isso não se verifica: a 5 req/s levamos 429
            # com `x-ratelimit-remaining: 0` depois de ~800 requisições, e o
            # `mailto` não alterou o comportamento (testado em 2026-08-03).
            # O princípio A5 do DOC-02 é explícito — uma fonte que nos bloqueie
            # está perdida para sempre —, então recuamos para 1 req/s e
            # aceitamos ~5 h de coleta em vez de arriscar o acesso.
            rate_limit=RateLimit(requests_per_second=1.0, max_retries=12,
                                 backoff_base_s=4.0, backoff_max_s=1800),
            license_resolution=LicenseResolution(
                method="source_policy",
                evidence_url="https://docs.openalex.org/additional-help/faq",
                default_spdx="CC0-1.0",
                notes="Todos os dados do OpenAlex são CC0. Redistribuição permitida.",
            ),
        )

    def harvest(self, max_pages: int | None = None) -> AcquisitionManifest:
        m = self.manifest
        if m.completed_at:
            return m

        buffer: list[dict] = []
        pages = 0
        pending_count = 0
        shard_idx = len(list(self.out_dir.glob("part-*.parquet")))

        # Mesma disciplina de dois cursores do coletor do arXiv: o de
        # requisição avança a cada página; o durável, só após flush confirmado.
        request_cursor = m.resumable_cursor or "*"
        pending_cursor = request_cursor

        while True:
            if max_pages is not None and pages >= max_pages:
                break

            resp = self.http.get(
                ENDPOINT,
                params={
                    "filter": m.query_spec["filter"],
                    "select": SELECT,
                    "per-page": str(PER_PAGE),
                    "cursor": request_cursor,
                    "mailto": self.contact,
                },
                timeout=120,
            )
            data = resp.json()

            if m.expected_count is None:
                m.expected_count = data.get("meta", {}).get("count")

            results = data.get("results", [])
            for w in results:
                buffer.append(parse_work(w))

            pages += 1
            pending_count += len(results)

            nxt = data.get("meta", {}).get("next_cursor")
            request_cursor = nxt
            pending_cursor = nxt

            if len(buffer) >= FLUSH_EVERY:
                self._flush(buffer, shard_idx)
                shard_idx += 1
                buffer = []
                m.resumable_cursor = pending_cursor
                m.actual_count += pending_count
                pending_count = 0
                self.checkpoint()

            done = m.actual_count + pending_count
            if pages % 25 == 0 or not nxt:
                pct = f" ({100 * done / m.expected_count:.1f}%)" if m.expected_count else ""
                log.info("página %d · %d obras%s", pages, done, pct)

            if not nxt or not results:
                self._flush(buffer, shard_idx)
                m.resumable_cursor = None
                m.actual_count += pending_count
                m.mark_complete()
                self.checkpoint()
                log.info("Concluído: %d obras", m.actual_count)
                return m

        if buffer:
            self._flush(buffer, shard_idx)
        m.resumable_cursor = pending_cursor
        m.actual_count += pending_count
        self.checkpoint()
        return m

    def _flush(self, buffer: list[dict], idx: int) -> None:
        if not buffer:
            return
        path = self.out_dir / f"part-{idx:05d}.parquet"
        df = pl.DataFrame(buffer, schema=SCHEMA)
        tmp = path.with_suffix(".parquet.tmp")
        df.write_parquet(tmp, compression="zstd")
        tmp.replace(path)
        self.manifest.checksum_index[path.name] = canonical_hash(
            {"rows": len(df), "cols": sorted(df.columns)}
        )
        edges = int(df["n_references"].sum())
        log.info(
            "→ %s (%d obras, %d arestas de citação, %.1f MB)",
            path.name, len(df), edges, path.stat().st_size / 1e6,
        )


def harvest_arxiv_works(
    out_dir: Path,
    filter_expr: str | None = None,
    max_pages: int | None = None,
    contact: str = CONTACT,
) -> AcquisitionManifest:
    h = OpenAlexHarvester.resume_or_create(
        out_dir, lambda: OpenAlexHarvester.make_manifest(filter_expr), contact=contact
    )
    return h.harvest(max_pages=max_pages)
