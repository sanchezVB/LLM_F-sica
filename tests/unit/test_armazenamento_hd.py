"""O plano nao pode escrever fora da raiz de dados indicada pelo usuario."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from phifm.core.armazenamento import planejar_armazenamento


def _checkout(tmp_path):
    hd = tmp_path / "HD"
    repo = hd / "projeto"
    repo.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='phifm'\n")
    return hd, repo


def test_todos_caches_e_venv_sao_no_hd_sem_criar_diretorios(tmp_path):
    hd, repo = _checkout(tmp_path)
    p = planejar_armazenamento(hd, repo)
    assert Path(p["venv"]).is_relative_to(hd)
    assert not Path(p["venv"]).exists()
    assert not (hd / "cache").exists()
    for caminho in p["diretorios"]:
        assert Path(caminho).is_relative_to(hd)
    for nome in ("TEMP", "TMP", "PIP_CACHE_DIR", "UV_CACHE_DIR", "HF_HOME",
                 "HF_HUB_CACHE", "HF_DATASETS_CACHE", "HF_XET_CACHE", "TORCH_HOME",
                 "XDG_CACHE_HOME", "PYTHONPYCACHEPREFIX", "NLTK_DATA"):
        assert Path(p["variaveis"][nome]).is_relative_to(hd)
    assert p["variaveis"]["PIP_CONFIG_FILE"] == os.devnull
    assert p["variaveis"]["PIP_REQUIRE_VIRTUALENV"] == "true"
    assert {"PIP_TARGET", "PIP_PREFIX", "TRANSFORMERS_CACHE"} <= set(p["remover_variaveis"])


def test_checkout_fora_do_hd_e_recusado(tmp_path):
    hd, repo = _checkout(tmp_path)
    with pytest.raises(ValueError, match="fora do HD"):
        planejar_armazenamento(repo, hd)


@pytest.mark.parametrize("relativo", ["cache", ".venv-treino-cpu", "models", "data/processed"])
def test_link_para_ssd_e_recusado_antes_de_criar_cache(tmp_path, relativo):
    hd, repo = _checkout(tmp_path)
    ssd = tmp_path / "SSD"
    ssd.mkdir()
    destino = (hd if relativo == "cache" else repo) / relativo
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.symlink_to(ssd, target_is_directory=True)
    with pytest.raises(ValueError, match="fora do HD"):
        planejar_armazenamento(hd, repo)
    assert not list(ssd.iterdir())


def test_raiz_atalho_e_recusada(tmp_path):
    hd, repo = _checkout(tmp_path)
    atalho = tmp_path / "atalho"
    atalho.symlink_to(hd, target_is_directory=True)
    with pytest.raises(ValueError, match="link/junction"):
        planejar_armazenamento(atalho, repo)


def test_raiz_relativa_e_recusada(tmp_path):
    _, repo = _checkout(tmp_path)
    with pytest.raises(ValueError, match="absolutos"):
        planejar_armazenamento(Path("hd"), repo)


def test_volume_sistema_e_recusado(tmp_path):
    hd, repo = _checkout(tmp_path)
    # O Windows tem drive (C:); em POSIX o drive e vazio. A regra e testada
    # com um valor nao vazio para nao confundir ausencia de SystemDrive.
    if hd.drive:
        with pytest.raises(ValueError, match="volume do sistema"):
            planejar_armazenamento(hd, repo, volume_sistema=hd.drive)
    else:
        assert planejar_armazenamento(hd, repo, volume_sistema="C:")["raiz_hd"] == str(hd)


def test_cli_cria_apenas_destinos_validados(tmp_path):
    hd, repo = _checkout(tmp_path)
    script = Path(__file__).resolve().parents[2] / "src/phifm/core/armazenamento.py"
    r = subprocess.run([sys.executable, "-B", "-I", str(script), "--raiz-hd", str(hd),
                        "--repositorio", str(repo), "--criar"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    p = json.loads(r.stdout)
    assert all(Path(caminho).is_dir() for caminho in p["diretorios"])
    assert not Path(p["venv"]).exists(), "planejar nao instala nem cria venv"
