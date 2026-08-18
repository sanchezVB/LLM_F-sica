"""Manifesto de aquisição — o contrato que torna a coleta auditável.

Especificado em DOC-02 §8.1. Impõe os princípios A1 (metadado antes de
conteúdo), A3 (licença resolvida antes da coleta) e A4 (coleta idempotente e
retomável) do plano de aquisição.

Regras impostas aqui, não por convenção:
  - Um lote com `failures` não-vazio nunca é marcado como concluído sem revisão.
  - `resumable_cursor` é persistido a cada flush; interrupção nunca custa mais
    que um lote.
  - Ausência de `license_resolution` bloqueia o lote.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from blake3 import blake3
from pydantic import BaseModel, Field


def canonical_hash(obj: Any) -> str:
    """BLAKE3 sobre a serialização canônica — a identidade de tudo (DOC-01 P4)."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return blake3(payload.encode("utf-8")).hexdigest()


def utcnow() -> datetime:
    return datetime.now(UTC)


class HarvestMethod(StrEnum):
    OAI_PMH = "oai_pmh"
    REST_API = "rest_api"
    BULK_S3 = "bulk_s3"
    HF_DATASET = "hf_dataset"
    DUMP_ARCHIVE = "dump_archive"
    DIRECT_DOWNLOAD = "direct_download"


class RateLimit(BaseModel):
    """Cortesia é inegociável (DOC-02, princípio A5).

    Uma fonte que nos bloqueie está perdida permanentemente, e o custo disso
    excede qualquer ganho de velocidade.
    """

    requests_per_second: float = Field(gt=0)
    burst: int = 1
    backoff_base_s: float = 2.0
    backoff_max_s: float = 300.0
    max_retries: int = 8
    respect_retry_after: bool = True


class LicenseResolution(BaseModel):
    """Como a licença da fonte foi determinada (A3)."""

    method: Literal["per_record", "source_policy", "manual", "unresolved"]
    evidence_url: str | None = None
    default_spdx: str | None = None
    notes: str = ""

    @property
    def blocks_harvest(self) -> bool:
        return self.method == "unresolved"


class FailureRecord(BaseModel):
    """Nada é descartado em silêncio (DOC-03 §9)."""

    at: datetime = Field(default_factory=utcnow)
    cursor: str | None = None
    code: str
    message: str
    retryable: bool = True


class AcquisitionManifest(BaseModel):
    """Emitido ANTES de qualquer byte ser baixado."""

    manifest_id: str = ""
    schema_version: str = "0.1.0"

    source_name: str
    harvest_method: HarvestMethod
    endpoint: str
    query_spec: dict[str, Any] = Field(default_factory=dict)
    rate_limit: RateLimit
    license_resolution: LicenseResolution

    started_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None
    # ⚠️ DENOMINADOR DE PROGRESSO, na unidade de quem coleta. NÃO é promessa de
    # completude, e nada no código compara `actual_count` com ele.
    #
    # Dividir um pelo outro é erro de leitura. Medido em 2026-08-13 nos
    # manifestos em disco:
    #
    #   coletor              expected_count      actual_count
    #   arXiv OAI                      None         1.595.422
    #   OpenAlex snapshot       510.372.821         4.613.751
    #
    # No snapshot, `expected` são as obras VARRIDAS (o OpenAlex inteiro) e
    # `actual` são as GUARDADAS (as que casaram com o arXiv). A razão dá 0,9% e
    # sugere catástrofe onde houve filtragem correta.
    #
    # No arXiv fica `None` porque o OAI-PMH deles não envia `completeListSize` —
    # verificado no set `math` em 2026-08-13. A completude ali vem do protocolo:
    # o `resumptionToken` ausente na última página é o sinal de fim.
    expected_count: int | None = None
    # Registros efetivamente GRAVADOS por este manifesto. Esta unidade é estável
    # entre coletores, diferente da de cima.
    actual_count: int = 0
    bytes_downloaded: int = 0
    requests_made: int = 0

    output_uri: str = ""
    # ⚠️ NÃO é checksum de conteúdo, apesar do nome. Os coletores gravam aqui
    # `canonical_hash({"rows": n, "cols": [...]})` — o hash da FORMA. Dois arquivos
    # de conteúdo diferente com as mesmas linhas e colunas hasheiam igual.
    #
    # O DOC-02 §8.1 especifica "mapa doc_id → BLAKE3, endereçado por conteúdo", e a
    # implementação divergiu da especificação sob um nome que promete o contrário.
    # Descoberto em 2026-08-17 ao construir o manifesto raiz do G1.5: a verificação
    # profunda acusou 878 parquets "alterados" que estavam intactos.
    #
    # O campo fica como está para não invalidar o `manifest_id` das cinco coletas já
    # feitas — reescrevê-lo mudaria a identidade de manifestos que atestam coletas
    # concluídas. `hash_conteudo` abaixo é o índice de verdade, e o construtor do
    # manifesto raiz computa o seu próprio quando este está ausente.
    checksum_index: dict[str, str] = Field(default_factory=dict)
    # Índice de CONTEÚDO: caminho relativo → BLAKE3 dos bytes. Vazio nos manifestos
    # anteriores a 2026-08-17; o `scripts/manifesto_corpus.py` computa e grava o
    # dele nesse caso, e declara que foi computado depois da coleta.
    hash_conteudo: dict[str, str] = Field(default_factory=dict)
    hash_algo: str = "blake3"
    resumable_cursor: str | None = None
    failures: list[FailureRecord] = Field(default_factory=list)

    pipeline_git_sha: str = "unknown"
    tool_version: str = "phifm.corpus.acquire 0.1.0"

    def model_post_init(self, _ctx: Any) -> None:
        if not self.manifest_id:
            self.manifest_id = canonical_hash(
                {
                    "source": self.source_name,
                    "endpoint": self.endpoint,
                    "query": self.query_spec,
                    "schema": self.schema_version,
                }
            )

    # ── invariantes ────────────────────────────────────────────────────────

    @property
    def can_start(self) -> tuple[bool, str]:
        if self.license_resolution.blocks_harvest:
            return False, "A3: licença não resolvida — coleta bloqueada"
        return True, ""

    @property
    def can_complete(self) -> tuple[bool, str]:
        unresolved = [f for f in self.failures if f.retryable]
        if unresolved:
            return False, f"{len(unresolved)} falha(s) recuperável(is) pendente(s)"
        return True, ""

    def mark_complete(self, force: bool = False) -> None:
        ok, why = self.can_complete
        if not ok and not force:
            raise RuntimeError(f"Lote não pode ser concluído: {why}")
        self.completed_at = utcnow()

    # ── persistência ───────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)  # escrita atômica: retomada nunca vê manifesto parcial

    @classmethod
    def load(cls, path: Path) -> AcquisitionManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
