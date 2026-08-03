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
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from blake3 import blake3
from pydantic import BaseModel, Field


def canonical_hash(obj: Any) -> str:
    """BLAKE3 sobre a serialização canônica — a identidade de tudo (DOC-01 P4)."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return blake3(payload.encode("utf-8")).hexdigest()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class HarvestMethod(str, Enum):
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
    expected_count: int | None = None
    actual_count: int = 0
    bytes_downloaded: int = 0
    requests_made: int = 0

    output_uri: str = ""
    checksum_index: dict[str, str] = Field(default_factory=dict)
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
