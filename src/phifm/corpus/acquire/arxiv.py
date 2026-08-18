"""Coletor de metadados do arXiv via OAI-PMH.

Primeira etapa do Sprint S1 (DOC-02 §9): a espinha de metadados. É a chave de
junção que resolve categoria, licença, DOI e data de todas as outras fontes —
princípio A1, "metadado antes de conteúdo".

Correções descobertas em execução (2026-08-03), a serem levadas ao DOC-02 §3.1:
  1. O endpoint OAI-PMH do arXiv é ``https://oaipmh.arxiv.org/oai``.
     ``export.arxiv.org/oai2`` responde 301 e está obsoleto.
  2. O arXiv expõe o *set* ``physics``, que permite filtrar a família de Física
     NO SERVIDOR. Não é preciso coletar os ~2,7 M registros e filtrar depois:
     coletamos ~1,2 M diretamente. Menos tráfego para eles, menos tempo para nós.
  3. O formato ``arXiv`` (não ``oai_dc``) traz categorias, licença, DOI e
     journal-ref — exatamente os campos que A3 exige.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl

from phifm.core.schema.manifest import (
    AcquisitionManifest,
    HarvestMethod,
    LicenseResolution,
    RateLimit,
)
from phifm.corpus.acquire.base import CONTACT, ResumableHarvester

log = logging.getLogger(__name__)

ENDPOINT = "https://oaipmh.arxiv.org/oai"
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arx": "http://arxiv.org/OAI/arXiv/",
}

# Licença padrão do arXiv: concede ao arXiv o direito de distribuir, não a
# terceiros. Ver ADR-0001 §2 — direito D2 (treinar) vs D3 (redistribuir).
ARXIV_DEFAULT_LICENSE = "arXiv-perpetual-nonexclusive"

FLUSH_EVERY = 20_000  # registros por shard parquet

# ⚠️ O timeout padrão de 60 s do `PoliteSession` não serve para OAI-PMH em set
# grande. Medido em 2026-08-13, na primeira coleta do set `math`:
#
#   Identify                 0,3 s   200
#   ListSets                 0,2 s   200
#   ListRecords set=math   183,0 s   503 + Retry-After: 0
#
# O endpoint estava saudável; montar o conjunto de resultados do `math` é que
# leva 183 s. Com 60 s a resposta NUNCA chegava, então o coletor registrava
# "Falha de rede" e gastava as 10 tentativas sem baixar um único registro —
# tratando como erro de rede o que era o servidor pedindo paciência.
#
# O 503 com `Retry-After` é o mecanismo de controle de fluxo do protocolo, e o
# `PoliteSession` já sabe obedecê-lo. Só precisava conseguir receber a resposta.
TIMEOUT_OAI = 300

# Schema explícito. A inferência do Polars falha quando um campo opcional
# (doi, journal_ref, msc_class...) é nulo nas primeiras N linhas e string
# depois. Declarar o contrato é mais correto que aumentar infer_schema_length:
# o schema é parte da especificação, não um detalhe do leitor.
SCHEMA: dict[str, pl.DataType] = {
    "arxiv_id": pl.Utf8,
    "oai_identifier": pl.Utf8,
    "datestamp": pl.Utf8,
    "created": pl.Utf8,
    "updated": pl.Utf8,
    "title": pl.Utf8,
    "abstract": pl.Utf8,
    "authors": pl.List(pl.Utf8),
    "categories": pl.List(pl.Utf8),
    "primary_category": pl.Utf8,
    "license": pl.Utf8,
    "doi": pl.Utf8,
    "journal_ref": pl.Utf8,
    "comments": pl.Utf8,
    "msc_class": pl.Utf8,
    "acm_class": pl.Utf8,
}


def _text(node: ET.Element | None) -> str | None:
    return node.text.strip() if node is not None and node.text else None


def parse_record(rec: ET.Element) -> dict | None:
    """Extrai um registro do formato ``arXiv``.

    Registros deletados (``status="deleted"``) são pulados: o arXiv usa
    ``deletedRecord=persistent``, então eles aparecem no fluxo e precisam ser
    tratados explicitamente em vez de virarem linhas vazias.
    """
    header = rec.find("oai:header", NS)
    if header is None:
        return None
    if header.get("status") == "deleted":
        return None

    meta = rec.find("oai:metadata/arx:arXiv", NS)
    if meta is None:
        return None

    authors = []
    for a in meta.findall("arx:authors/arx:author", NS):
        keyname = _text(a.find("arx:keyname", NS)) or ""
        forenames = _text(a.find("arx:forenames", NS)) or ""
        name = f"{forenames} {keyname}".strip()
        if name:
            authors.append(name)

    cats = (_text(meta.find("arx:categories", NS)) or "").split()

    return {
        "arxiv_id": _text(meta.find("arx:id", NS)),
        "oai_identifier": _text(header.find("oai:identifier", NS)),
        "datestamp": _text(header.find("oai:datestamp", NS)),
        "created": _text(meta.find("arx:created", NS)),
        "updated": _text(meta.find("arx:updated", NS)),
        "title": _text(meta.find("arx:title", NS)),
        "abstract": _text(meta.find("arx:abstract", NS)),
        "authors": authors,
        "categories": cats,
        "primary_category": cats[0] if cats else None,
        # ── A3: licença resolvida POR REGISTRO, não presumida ──────────────
        "license": _text(meta.find("arx:license", NS)) or ARXIV_DEFAULT_LICENSE,
        "doi": _text(meta.find("arx:doi", NS)),
        "journal_ref": _text(meta.find("arx:journal-ref", NS)),
        "comments": _text(meta.find("arx:comments", NS)),
        "msc_class": _text(meta.find("arx:msc-class", NS)),
        "acm_class": _text(meta.find("arx:acm-class", NS)),
    }


class ArxivOAIHarvester(ResumableHarvester):
    """Coletor OAI-PMH retomável para a família de Física do arXiv."""

    @staticmethod
    def make_manifest(
        set_spec: str = "physics",
        from_date: str | None = None,
        until_date: str | None = None,
    ) -> AcquisitionManifest:
        return AcquisitionManifest(
            source_name="arxiv_metadata",
            harvest_method=HarvestMethod.OAI_PMH,
            endpoint=ENDPOINT,
            query_spec={
                "verb": "ListRecords",
                "metadataPrefix": "arXiv",
                "set": set_spec,
                "from": from_date,
                "until": until_date,
            },
            # DOC-02 §8.2: o arXiv pede 1 requisição a cada 3 segundos.
            rate_limit=RateLimit(requests_per_second=1 / 3, max_retries=10, backoff_max_s=600),
            license_resolution=LicenseResolution(
                method="per_record",
                evidence_url="https://info.arxiv.org/help/license/index.html",
                default_spdx=ARXIV_DEFAULT_LICENSE,
                notes=(
                    "Licença lida do campo <license> de cada registro. Ausência "
                    "significa a licença padrão do arXiv, que NÃO permite "
                    "redistribuição por terceiros (ADR-0001 §2, direito D3)."
                ),
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

        # ── DOIS cursores, e confundi-los é um bug silencioso ──────────────
        # `request_cursor` avança a CADA página: é o que dirige o laço.
        # `m.resumable_cursor` avança só após flush confirmado: é o ponto de
        # retomada durável. Usar o segundo para montar a requisição refaria a
        # mesma página até o flush seguinte — foi exatamente o defeito
        # encontrado no smoke test de 2026-08-03 (amplificação de 16×).
        request_cursor = m.resumable_cursor
        pending_cursor = request_cursor

        while True:
            if max_pages is not None and pages >= max_pages:
                log.info("Limite de %d páginas atingido — parando (retomável)", max_pages)
                break

            params = (
                {"verb": "ListRecords", "resumptionToken": request_cursor}
                if request_cursor
                else {
                    "verb": "ListRecords",
                    "metadataPrefix": "arXiv",
                    **({"set": m.query_spec["set"]} if m.query_spec.get("set") else {}),
                    **({"from": m.query_spec["from"]} if m.query_spec.get("from") else {}),
                    **({"until": m.query_spec["until"]} if m.query_spec.get("until") else {}),
                }
            )

            resp = self.http.get(ENDPOINT, params=params, timeout=TIMEOUT_OAI)
            try:
                root = ET.fromstring(resp.content)
            except ET.ParseError as exc:
                self.record_failure("F-XML-PARSE", str(exc))
                raise

            err = root.find("oai:error", NS)
            if err is not None:
                code = err.get("code", "unknown")
                if code == "noRecordsMatch":
                    log.info("Nenhum registro corresponde à consulta")
                    break
                self.record_failure(f"F-OAI-{code}", err.text or "", retryable=False)
                raise RuntimeError(f"Erro OAI-PMH: {code} — {err.text}")

            records = root.findall("oai:ListRecords/oai:record", NS)
            for rec in records:
                parsed = parse_record(rec)
                if parsed:
                    buffer.append(parsed)

            pages += 1
            pending_count += len(records)

            token_el = root.find("oai:ListRecords/oai:resumptionToken", NS)
            token = _text(token_el)

            if token_el is not None and m.expected_count is None:
                complete = token_el.get("completeListSize")
                if complete and int(complete) > 0:
                    m.expected_count = int(complete)
                elif pages == 1:
                    # Registrado uma vez, para ninguém procurar um percentual que
                    # não pode existir. Verificado em 2026-08-13: o OAI do arXiv
                    # não envia `completeListSize`, então não há total declarado —
                    # nem para o progresso, nem para conferir completude. O sinal
                    # de fim é o `resumptionToken` ausente, que é o do protocolo.
                    log.info("o arXiv não declara completeListSize — sem percentual "
                             "de progresso; o fim é o resumptionToken ausente")

            # ── ordem crítica (DOC-08 §7.2) ────────────────────────────────
            # O cursor NÃO avança antes de os dados estarem duráveis. Se o
            # flush falhar, retomamos da última escrita confirmada e
            # reprocessamos as páginas seguintes — entrega "ao menos uma vez",
            # com duplicatas que a dedup exata (DOC-04 §5.1) remove.
            # A ordem inversa perderia registros em silêncio.
            request_cursor = token
            pending_cursor = token

            if len(buffer) >= FLUSH_EVERY:
                self._flush(buffer, shard_idx)
                shard_idx += 1
                buffer = []
                m.resumable_cursor = pending_cursor
                m.actual_count += pending_count
                pending_count = 0
                self.checkpoint()

            done = m.actual_count + pending_count
            pct = f" ({100 * done / m.expected_count:.1f}%)" if m.expected_count else ""
            log.info("página %d · %d registros%s", pages, done, pct)

            if not token:
                log.info("Fluxo OAI encerrado pelo servidor")
                self._flush(buffer, shard_idx)
                m.resumable_cursor = None
                m.actual_count += pending_count
                m.mark_complete()
                self.checkpoint()
                return m

        # Parada por limite de páginas: consolidar antes de sair.
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
        tmp.replace(path)  # escrita atômica: nunca existe shard pela metade

        from phifm.core.schema.manifest import canonical_hash

        # ⚠️ `checksum_index` é hash de FORMA, não de conteúdo — ver o comentário
        # do campo em `core/schema/manifest.py`. Mantido para não mudar a
        # identidade de manifestos já gravados.
        self.manifest.checksum_index[path.name] = canonical_hash(
            {"rows": len(df), "cols": sorted(df.columns)}
        )
        # O índice de CONTEÚDO, que é o que o G1.5 exige: BLAKE3 dos bytes do
        # arquivo que acabou de ser gravado atomicamente.
        from phifm.core.schema.reprodutibilidade import hash_arquivo

        self.manifest.hash_conteudo[path.name] = hash_arquivo(path)
        log.info("→ %s (%d registros, %.1f MB)", path.name, len(df), path.stat().st_size / 1e6)


def harvest_physics(
    out_dir: Path,
    set_spec: str = "physics",
    from_date: str | None = None,
    until_date: str | None = None,
    max_pages: int | None = None,
    contact: str = CONTACT,
) -> AcquisitionManifest:
    h = ArxivOAIHarvester.resume_or_create(
        out_dir,
        lambda: ArxivOAIHarvester.make_manifest(set_spec, from_date, until_date),
        contact=contact,
    )
    return h.harvest(max_pages=max_pages)
