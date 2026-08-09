"""Regressão: o coletor OAI não pode reemitir a mesma página.

Defeito real encontrado no smoke test de 2026-08-03. `m.resumable_cursor` só
avança após um flush confirmado (para durabilidade, DOC-08 §7.2), mas era ele
que montava a requisição — então a mesma página era pedida repetidamente até
o flush seguinte, com amplificação de 16×.

Duas consequências, e a segunda é a grave:
  1. 13× de trabalho desperdiçado;
  2. 13× de carga desnecessária no servidor do arXiv, o que viola o princípio
     A5 (cortesia) do DOC-02 e é o caminho mais rápido para sermos bloqueados.

O teste usa um servidor OAI falso, sem rede.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.corpus.acquire.arxiv import ArxivOAIHarvester  # noqa: E402

PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <ListRecords>
    {records}
    {token}
  </ListRecords>
</OAI-PMH>"""

REC = """<record>
  <header><identifier>oai:arXiv.org:{id}</identifier><datestamp>2024-06-01</datestamp></header>
  <metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/">
    <id>{id}</id><created>2024-06-01</created>
    <title>T{id}</title><abstract>A{id}</abstract>
    <authors><author><keyname>Silva</keyname><forenames>A</forenames></author></authors>
    <categories>gr-qc</categories>
  </arXiv></metadata>
</record>"""


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content
        self.status_code = 200
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:  # pragma: no cover
        pass


class FakeOAIServer:
    """Servidor OAI mínimo: 3 páginas, 2 registros cada, com resumptionToken."""

    def __init__(self, pages: int = 3, per_page: int = 2):
        self.pages, self.per_page = pages, per_page
        self.requests: list[str | None] = []

    def get(self, url, params=None, timeout=60):
        token = (params or {}).get("resumptionToken")
        self.requests.append(token)
        idx = int(token.split(":")[1]) if token else 0

        start = idx * self.per_page
        recs = "".join(REC.format(id=f"2406.{start + i:05d}") for i in range(self.per_page))
        nxt = (
            f"<resumptionToken>tok:{idx + 1}</resumptionToken>"
            if idx + 1 < self.pages
            else "<resumptionToken/>"
        )
        return FakeResponse(PAGE.format(records=recs, token=nxt).encode())


@pytest.fixture
def harvester(tmp_path, monkeypatch):
    server = FakeOAIServer()
    h = ArxivOAIHarvester(ArxivOAIHarvester.make_manifest("physics"), tmp_path)
    monkeypatch.setattr(h.http, "get", server.get)
    monkeypatch.setattr(h.http.throttle, "wait", lambda: None)  # sem espera no teste
    return h, server, tmp_path


def test_cada_pagina_e_pedida_uma_unica_vez(harvester):
    """O cursor de REQUISIÇÃO avança a cada página, não a cada flush."""
    h, server, _ = harvester
    h.harvest()
    assert server.requests == [None, "tok:1", "tok:2"], (
        f"Páginas reemitidas: {server.requests}. O cursor de requisição não avançou."
    )


def test_sem_amplificacao_de_registros(harvester):
    """3 páginas × 2 registros = 6 linhas, 6 ids únicos. Razão exata de 1,00×."""
    import polars as pl

    h, _, out = harvester
    m = h.harvest()
    df = pl.read_parquet(out / "*.parquet")
    assert df.height == 6, f"esperado 6 linhas, obtido {df.height}"
    assert df["arxiv_id"].n_unique() == 6, "amplificação detectada"
    assert m.actual_count == 6


def test_conclusao_limpa_zera_o_cursor(harvester):
    """Lote concluído não deixa cursor pendente — senão a retomada refaria trabalho."""
    h, _, _ = harvester
    m = h.harvest()
    assert m.completed_at is not None
    assert m.resumable_cursor is None
    assert m.failures == []


def test_licenca_ausente_recebe_padrao_do_arxiv(harvester):
    """A3: licença nunca fica nula. Ausência = licença padrão, não redistribuível."""
    import polars as pl

    h, _, out = harvester
    h.harvest()
    df = pl.read_parquet(out / "*.parquet")
    assert df["license"].null_count() == 0
    assert (df["license"] == "arXiv-perpetual-nonexclusive").all()


def test_coleta_bloqueada_sem_licenca_resolvida(tmp_path):
    """A3 imposto em código: manifesto sem resolução de licença não inicia."""
    from phifm.core.schema.manifest import LicenseResolution

    m = ArxivOAIHarvester.make_manifest("physics")
    m.license_resolution = LicenseResolution(method="unresolved")
    with pytest.raises(RuntimeError, match="A3"):
        ArxivOAIHarvester(m, tmp_path)
