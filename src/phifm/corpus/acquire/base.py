"""Infraestrutura de coleta: cortesia, retomada e falha explícita.

Implementa os princípios A4 (idempotente, retomável) e A5 (cortesia
inegociável) do DOC-02 §1, e a política de limites do DOC-02 §8.2.
"""

from __future__ import annotations

import logging
import random
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from phifm.core.schema.manifest import (
    AcquisitionManifest,
    FailureRecord,
    RateLimit,
)

log = logging.getLogger(__name__)

CONTACT = "phifm-corpus@localhost"  # sobrescrito por PHIFM_CONTACT (.env)
REPO = "https://github.com/sanchezVB/LLM_F-sica"


def user_agent(contact: str = CONTACT) -> str:
    """DOC-02 §8.2 — identificação com contato.

    Ser identificável protege o projeto: um coletor anônimo que incomode é
    bloqueado sem aviso; um identificado recebe um e-mail primeiro.
    """
    return f"PhiFM-Corpus/0.1 (Physics foundation model research; +{REPO}; {contact})"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


@dataclass
class Throttle:
    """Limitador de taxa com recuo exponencial e jitter."""

    limit: RateLimit
    _last: float = field(default=0.0, repr=False)

    def wait(self) -> None:
        interval = 1.0 / self.limit.requests_per_second
        elapsed = time.monotonic() - self._last
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last = time.monotonic()

    def backoff(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None and self.limit.respect_retry_after:
            # O servidor disse quanto esperar. Obedecer, com folga.
            return min(retry_after + 1.0, self.limit.backoff_max_s)
        delay = min(self.limit.backoff_base_s**attempt, self.limit.backoff_max_s)
        return delay * (0.5 + random.random())  # jitter contra sincronização


class PoliteSession:
    """Sessão HTTP que respeita limites, obedece Retry-After e nunca falha calada."""

    def __init__(self, limit: RateLimit, contact: str = CONTACT):
        self.throttle = Throttle(limit)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": user_agent(contact), "Accept-Encoding": "gzip, deflate"}
        )
        self.requests_made = 0
        self.bytes_downloaded = 0

    def get(
        self,
        url: str,
        params: dict | None = None,
        timeout: int = 60,
        headers: dict | None = None,
    ) -> requests.Response:
        """`headers` existe para o coletor de snapshot, que precisa de `Range`.

        Passar por aqui em vez de chamar `requests` direto mantém de graça o
        tratamento de 429/503 com Retry-After, o recuo exponencial, a cortesia
        e a contagem de bytes — que é o que alimenta o manifesto.
        """
        limit = self.throttle.limit
        last_exc: Exception | None = None

        for attempt in range(limit.max_retries):
            self.throttle.wait()
            try:
                r = self.session.get(url, params=params, timeout=timeout, headers=headers)
                self.requests_made += 1
                self.bytes_downloaded += len(r.content)

                # 503 + Retry-After é o mecanismo de controle de fluxo do OAI-PMH.
                # Não é erro: é o servidor pedindo para esperar.
                if r.status_code in (429, 503):
                    ra = r.headers.get("Retry-After")
                    delay = self.throttle.backoff(attempt, float(ra) if ra and ra.isdigit() else None)
                    log.info("HTTP %s — aguardando %.1fs (tentativa %d)", r.status_code, delay, attempt + 1)
                    time.sleep(delay)
                    continue

                # ⚠️ 4xx que não seja 429 é DEFINITIVO: repetir não muda a
                # resposta. Medido em 2026-08-10 na auditoria do S3b — papers
                # sem fonte no arXiv davam 404 e cada um custava seis tentativas
                # com recuo exponencial, ~25 s desperdiçados por paper ausente.
                #
                # Pior que o desperdício: o log enchia de "Falha de rede" para
                # algo que não é falha de rede, e sim ausência legítima do
                # recurso. Confundir os dois esconde problema de rede real.
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    r.raise_for_status()

                r.raise_for_status()
                return r

            except requests.HTTPError as exc:
                # Já sabemos que é definitivo (a guarda acima só deixa passar
                # 4xx não-429); propagar sem repetir.
                if exc.response is not None and 400 <= exc.response.status_code < 500 \
                        and exc.response.status_code != 429:
                    raise
                last_exc = exc
                delay = self.throttle.backoff(attempt)
                log.warning("HTTP %s — aguardando %.1fs", exc, delay)
                time.sleep(delay)

            except requests.RequestException as exc:
                last_exc = exc
                delay = self.throttle.backoff(attempt)
                log.warning("Falha de rede (%s) — aguardando %.1fs", exc, delay)
                time.sleep(delay)

        raise RuntimeError(f"Esgotadas {limit.max_retries} tentativas para {url}") from last_exc


class ResumableHarvester:
    """Base de coletor retomável.

    O estado vive no manifesto, não em memória: matar o processo e reiniciar
    retoma exatamente de onde parou, sem duplicar nem pular registros.
    """

    def __init__(self, manifest: AcquisitionManifest, out_dir: Path, contact: str = CONTACT):
        self.manifest = manifest
        self.out_dir = out_dir
        self.manifest_path = out_dir / "_manifest.json"
        self.contact = contact
        self.http = PoliteSession(manifest.rate_limit, contact)

        ok, why = manifest.can_start
        if not ok:
            raise RuntimeError(f"Coleta bloqueada — {why}")

        out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest.pipeline_git_sha = git_sha()

    @classmethod
    def resume_or_create(cls, out_dir: Path, factory, contact: str = CONTACT):
        """Retoma um manifesto existente ou cria um novo (A4)."""
        mp = out_dir / "_manifest.json"
        if mp.exists():
            m = AcquisitionManifest.load(mp)
            if m.completed_at:
                log.info("Lote já concluído em %s — nada a fazer", m.completed_at)
            else:
                log.info("Retomando de cursor=%s (%d registros já coletados)",
                         (m.resumable_cursor or "início")[:40], m.actual_count)
            return cls(m, out_dir, contact)
        return cls(factory(), out_dir, contact)

    def checkpoint(self) -> None:
        self.manifest.requests_made = self.http.requests_made
        self.manifest.bytes_downloaded = self.http.bytes_downloaded
        self.manifest.save(self.manifest_path)

    def record_failure(self, code: str, message: str, retryable: bool = True) -> None:
        """Falhas são registradas, nunca engolidas (DOC-03 §9)."""
        self.manifest.failures.append(
            FailureRecord(cursor=self.manifest.resumable_cursor, code=code,
                          message=message[:500], retryable=retryable)
        )
        self.checkpoint()
