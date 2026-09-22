"""Um zip do PowerShell não vira sete arquivos com barra invertida no nome.

O `extractall` do Python faz exatamente isso, sem erro, e foi o que sumiu com o
`model.safetensors` do encoder do caminho B no Colab (2026-09-22). Ver
`phifm.core.io.zips`.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.core.io.zips import descompactar_normalizado  # noqa: E402


def _zip_do_powershell(caminho: Path) -> Path:
    """Como o `Compress-Archive` do Windows PowerShell 5.1 grava: `pasta\\arquivo`."""
    with zipfile.ZipFile(caminho, "w") as z:
        z.writestr("modelo\\model.safetensors", b"pesos")
        z.writestr("modelo\\config.json", b"{}")
    return caminho


def test_a_BARRA_INVERTIDA_vira_pasta(tmp_path):
    arquivo = _zip_do_powershell(tmp_path / "m.zip")
    destino = tmp_path / "saida"
    escritos = descompactar_normalizado(arquivo, destino)

    assert (destino / "modelo" / "model.safetensors").read_bytes() == b"pesos"
    assert (destino / "modelo" / "config.json").exists()
    assert len(escritos) == 2
    # E nenhum arquivo com barra invertida no nome, que é o sintoma do defeito.
    assert not [p for p in destino.rglob("*") if "\\" in p.name]


def test_o_extractall_CRU_produz_o_defeito(tmp_path):
    """O dente: se esta asserção parar de valer, o módulo virou supérfluo — e aí é
    para apagá-lo, não para confiar que o `extractall` passou a traduzir."""
    if sys.platform.startswith("win"):
        pytest.skip("no Windows as duas barras são separadoras; o defeito é do Linux")
    arquivo = _zip_do_powershell(tmp_path / "m.zip")
    destino = tmp_path / "cru"
    with zipfile.ZipFile(arquivo) as z:
        z.extractall(destino)
    assert not (destino / "modelo" / "model.safetensors").exists()
    assert [p for p in destino.iterdir() if "\\" in p.name]


def test_o_zip_NORMAL_continua_funcionando(tmp_path):
    arquivo = tmp_path / "n.zip"
    with zipfile.ZipFile(arquivo, "w") as z:
        z.writestr("modelo/model.safetensors", b"pesos")
    descompactar_normalizado(arquivo, tmp_path / "saida")
    assert (tmp_path / "saida" / "modelo" / "model.safetensors").read_bytes() == b"pesos"


def test_entrada_que_ESCAPA_do_destino_e_recusada(tmp_path):
    arquivo = tmp_path / "mau.zip"
    with zipfile.ZipFile(arquivo, "w") as z:
        z.writestr("../fora.txt", b"x")
    with pytest.raises(ValueError, match="fora de"):
        descompactar_normalizado(arquivo, tmp_path / "saida")
    assert not (tmp_path / "fora.txt").exists()
