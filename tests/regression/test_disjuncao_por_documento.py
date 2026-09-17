"""Disjunção por PARTE não é disjunção por DOCUMENTO.

Medido em 2026-09-17: o RedPajama de Física tem 6.778 `arxiv_id` em duas linhas,
cópias byte a byte, todas em partes diferentes (33+34 e 32+34). O preparador de fatias
do ΦEnc só excluía as PARTES do treino; uma avaliação na 33 com a 34 no treino sairia
com `disjunto_do_treino: true` e documentos do treino dentro.

Nenhuma fatia já preparada é afetada — nenhuma dessas partes está num treino. O teste
trava a guarda que impede a próxima.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import polars as pl
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.training.pretrain.dados import ids_de, recusar_documentos_do_treino  # noqa: E402


def _parte(caminho: Path, ids: list[str]) -> Path:
    pl.DataFrame({"arxiv_id": ids, "texto": [f"texto {i}" for i in ids]}).write_parquet(caminho)
    return caminho


def test_parte_sem_documento_do_treino_passa(tmp_path):
    treino = ids_de([_parte(tmp_path / "part-00001.parquet", ["a", "b"])])
    recusar_documentos_do_treino(_parte(tmp_path / "part-00002.parquet", ["c", "d"]), treino)


def test_parte_com_COPIA_de_documento_do_treino_e_RECUSADA(tmp_path):
    treino = ids_de([_parte(tmp_path / "part-00033.parquet", ["x", "y"])])
    aval = _parte(tmp_path / "part-00034.parquet", ["y", "z"])
    with pytest.raises(ValueError, match="1 documentos"):
        recusar_documentos_do_treino(aval, treino)


def test_ids_de_nenhuma_parte_e_vazio():
    assert ids_de([]) == set()


def test_o_preparador_CHAMA_a_guarda_antes_de_ler_o_texto():
    """Pela AST: a guarda tem de vir antes do `read_parquet` do texto no laço."""
    arvore = ast.parse((RAIZ / "scripts" / "preparar_dados_phienc.py").read_text(
        encoding="utf-8"))
    linhas_guarda, linhas_leitura = [], []
    for no in ast.walk(arvore):
        if isinstance(no, ast.Call):
            nome = getattr(no.func, "id", None) or getattr(no.func, "attr", None)
            if nome == "recusar_documentos_do_treino":
                linhas_guarda.append(no.lineno)
            if nome == "read_parquet" and any(
                    isinstance(k.value, ast.List) and any(
                        getattr(e, "value", None) == "texto" for e in k.value.elts)
                    for k in no.keywords if k.arg == "columns"):
                linhas_leitura.append(no.lineno)
    assert linhas_guarda and linhas_leitura
    assert min(linhas_guarda) < min(linhas_leitura)
