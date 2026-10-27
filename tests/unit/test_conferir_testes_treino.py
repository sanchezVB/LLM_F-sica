"""O CI não pode aceitar importorskip ou a ausência de uma suíte obrigatória."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "conferir_testes_treino", Path(__file__).resolve().parents[2]
    / "scripts/conferir_testes_treino.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
conferir = _mod.conferir


def _relatorio(tmp_path, corpo):
    p = tmp_path / "resultado.xml"
    p.write_text(f"<testsuites><testsuite>{corpo}</testsuite></testsuites>", encoding="utf-8")
    return p


def test_aceita_casos_de_todos_os_alvos_inclusive_classes(tmp_path):
    p = _relatorio(tmp_path, '<testcase classname="tests.regression.test_a" name="um"/>'
                   '<testcase classname="tests.regression.test_b.Classe" name="dois"/>')
    assert conferir(p, ["tests/regression/test_a.py", "tests/regression/test_b.py"]) == 2


@pytest.mark.parametrize("elemento", ["skipped", "error", "failure"])
def test_recusa_salto_e_falha_mesmo_com_outro_teste_aprovado(tmp_path, elemento):
    p = _relatorio(tmp_path, '<testcase classname="tests.test_a" name="ok"/>'
                   f'<testcase classname="tests.test_a" name="ruim"><{elemento}/></testcase>')
    with pytest.raises(ValueError, match="saltados ou reprovados"):
        conferir(p, ["tests/test_a.py"])


def test_recusa_zero_testes(tmp_path):
    with pytest.raises(ValueError, match="nenhum teste"):
        conferir(_relatorio(tmp_path, ""), ["tests/test_a.py"])


def test_recusa_alvo_omitido_sem_confundir_prefixos(tmp_path):
    p = _relatorio(tmp_path, '<testcase classname="tests.test_a_extra" name="ok"/>')
    with pytest.raises(ValueError, match="alvo obrigatório"):
        conferir(p, ["tests/test_a.py"])


def test_recusa_arquivo_ausente(tmp_path):
    with pytest.raises(FileNotFoundError):
        conferir(tmp_path / "ausente.xml", ["tests/test_a.py"])
