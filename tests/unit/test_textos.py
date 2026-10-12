"""O texto completo por artigo (`phifm.retrieval.textos`). O que se tranca: ler um artigo
devolve o texto DELE, de qualquer grupo e de qualquer arquivo; o duplicado fica com o
primeiro; a construção retoma; e um índice que não bate com os arquivos é recusado."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.retrieval import textos as t  # noqa: E402


def _origem(tmp_path: Path) -> Path:
    origem = tmp_path / "origem"
    origem.mkdir()
    pl.DataFrame({"arxiv_id": [f"a{i}" for i in range(10)],
                  "texto": [f"texto de a{i}" for i in range(10)]}
                 ).write_parquet(origem / "part-00000.parquet")
    # a3 aparece de novo, com outro texto: vale o primeiro
    pl.DataFrame({"arxiv_id": ["b0", "a3", "b1"], "texto": ["texto de b0", "OUTRO", "texto de b1"]}
                 ).write_parquet(origem / "part-00001.parquet")
    return origem


def test_le_cada_artigo_de_qualquer_grupo_e_arquivo(tmp_path):
    man = t.construir(_origem(tmp_path), tmp_path / "destino", por_grupo=4)
    assert man["artigos"] == 12 and man["por_grupo"] == 4
    textos = t.Textos(tmp_path / "destino")
    assert len(textos) == 12
    for a in [f"a{i}" for i in range(10)] + ["b0", "b1"]:
        assert a in textos and textos.ler(a) == f"texto de {a}"
    assert "c9" not in textos and textos.ler("c9") is None


def test_a_construcao_retoma_sem_regravar_o_que_ja_existe(tmp_path):
    origem, destino = _origem(tmp_path), tmp_path / "destino"
    t.construir(origem, destino, por_grupo=4)
    antes = (destino / "part-00000.parquet").stat().st_mtime_ns
    (destino / "part-00001.parquet").unlink()
    (destino / t.NOME_INDICE).unlink()
    gravados = []
    t.construir(origem, destino, por_grupo=4, ao_gravar=lambda nome, n, total: gravados.append(
        (nome, n, total)))
    assert (destino / "part-00000.parquet").stat().st_mtime_ns == antes
    assert gravados == [("part-00000.parquet", 10, 10), ("part-00001.parquet", 2, 12)]
    assert t.Textos(destino).ler("b1") == "texto de b1"


def test_sem_manifesto_diz_o_que_rodar(tmp_path):
    with pytest.raises(FileNotFoundError, match="indexar_textos"):
        t.Textos(tmp_path)


def test_indice_que_nao_bate_com_os_arquivos_e_recusado(tmp_path):
    destino = tmp_path / "destino"
    t.construir(_origem(tmp_path), destino, por_grupo=4)
    idx = pl.read_parquet(destino / t.NOME_INDICE)
    idx.with_columns(pl.when(pl.col("arxiv_id") == "a1").then(5).otherwise(pl.col("linha"))
                     .cast(pl.Int32).alias("linha")).write_parquet(destino / t.NOME_INDICE)
    with pytest.raises(RuntimeError, match="linha errada"):
        t.Textos(destino).ler("a1")
