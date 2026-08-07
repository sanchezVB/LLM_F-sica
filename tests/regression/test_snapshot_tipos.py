"""O caminho de ESCRITA do coletor de snapshot, que o teste de fumaça não viu.

Regressão de 2026-08-07, e o modo de falha vale mais que o defeito.

O defeito: `publication_date` é `date32[day]` no parquet e chega como
`datetime.date`; pela API é texto JSON, e o `SCHEMA` compartilhado diz
`pl.Utf8`. O polars aborta o `DataFrame` inteiro:

    ComputeError: could not append value: 2017-01-01 of type: date to the
    builder

O modo de falha: o teste de fumaça rodou com `--max-particoes 2`, e as duas
primeiras partições do manifesto são de 2016 e têm **zero** obras com origem no
arXiv. O buffer ficou vazio, `_flush` retornou na primeira linha, e o teste
passou sem exercitar nenhuma escrita. Verificou-se que nada acontecia.

Daí a forma destes testes: eles constroem um registro com os tipos que o
parquet realmente entrega e vão até o parquet no disco. Não dependem de rede,
então não dependem de sorte na amostragem de partições.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.corpus.acquire.openalex import SCHEMA  # noqa: E402
from phifm.corpus.acquire.openalex_snapshot import (  # noqa: E402
    COLUNAS,
    OpenAlexSnapshotHarvester,
    _normalizar_tipos,
)


def registro_do_parquet() -> dict:
    """Um registro como `parse_work` o devolve a partir do parquet.

    A diferença que importa está em `publication_date`: `datetime.date`, não
    `str`.
    """
    return {
        "openalex_id": "W4221163828",
        "arxiv_id": "2203.00339",
        "doi": "10.1063/5.0090861",
        "title": "Um título qualquer",
        "publication_year": 2022,
        "publication_date": dt.date(2022, 3, 1),
        "type": "article",
        "language": "en",
        "is_oa": True,
        "oa_status": "green",
        "cited_by_count": 7,
        "n_references": 62,
        "referenced_works": ["W123", "W456"],
        "openalex_field": "Computer Science",
        "openalex_topic": "Machine Learning",
    }


def test_data_do_parquet_quebra_o_schema_sem_normalizacao():
    """Guarda o defeito no lugar: sem a conversão, o polars recusa."""
    with pytest.raises(Exception, match="(?i)date|schema|append"):
        pl.DataFrame([registro_do_parquet()], schema=SCHEMA)


def test_normalizar_tipos_converte_data_para_iso():
    r = _normalizar_tipos(registro_do_parquet())
    assert r["publication_date"] == "2022-03-01"
    assert isinstance(r["publication_date"], str)


def test_normalizar_tipos_cobre_datetime_tambem():
    """Genérico de propósito: o schema do snapshot já mudou uma vez sem avisar,
    e um `datetime` novo em outro campo faria o mesmo estrago."""
    r = _normalizar_tipos({**registro_do_parquet(),
                           "outro_campo": dt.datetime(2022, 3, 1, 12, 30)})
    assert r["outro_campo"].startswith("2022-03-01T12:30")


def test_normalizar_tipos_nao_mexe_no_resto():
    original = registro_do_parquet()
    r = _normalizar_tipos(dict(original))
    for chave in ("openalex_id", "arxiv_id", "publication_year", "is_oa",
                  "referenced_works", "n_references"):
        assert r[chave] == original[chave], chave


def test_flush_escreve_parquet_legivel(tmp_path):
    """O caminho completo de escrita, que é o que o teste de fumaça não fez.

    Vai até o arquivo no disco e lê de volta — inclusive o `checksum_index`,
    porque um shard sem checksum não é rastreável pelo DOC-08.
    """
    class Falso(OpenAlexSnapshotHarvester):
        def __init__(self, out_dir):
            self.out_dir = out_dir
            self.manifest = OpenAlexSnapshotHarvester.make_manifest()

    h = Falso(tmp_path)
    h._flush([_normalizar_tipos(registro_do_parquet()) for _ in range(3)], 0)

    destino = tmp_path / "part-00000.parquet"
    assert destino.exists()
    assert not list(tmp_path.glob("*.tmp")), "o temporário da escrita atômica ficou para trás"

    df = pl.read_parquet(destino)
    assert len(df) == 3
    assert df["arxiv_id"][0] == "2203.00339"
    assert df["publication_date"][0] == "2022-03-01"
    assert df["referenced_works"][0].to_list() == ["W123", "W456"]
    assert h.manifest.checksum_index["part-00000.parquet"], "shard sem checksum"


def test_flush_vazio_nao_cria_arquivo(tmp_path):
    """Partição sem obras do arXiv é o caso COMUM: 0,5% de aproveitamento."""
    class Falso(OpenAlexSnapshotHarvester):
        def __init__(self, out_dir):
            self.out_dir = out_dir
            self.manifest = OpenAlexSnapshotHarvester.make_manifest()

    h = Falso(tmp_path)
    h._flush([], 0)
    assert not list(tmp_path.glob("*.parquet"))


def test_colunas_pedidas_batem_com_o_que_parse_work_usa():
    """Se alguém acrescentar um campo ao parser sem pedir a coluna, o registro
    sai com `None` silencioso — e a contagem final continua plausível."""
    assert set(COLUNAS) >= {
        "id", "doi", "title", "publication_year", "publication_date", "type",
        "referenced_works", "cited_by_count", "open_access", "primary_topic",
        "language", "locations",
    }
