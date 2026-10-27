"""Plano explícito de armazenamento local; não detecta o meio físico do disco.

O dono identifica o HD. O plano impede que checkout, venv e destinos conhecidos
escapem dessa raiz por caminho, link simbólico ou junction. Não copia credenciais
nem instala programas; o PowerShell aplica as variáveis à sessão corrente.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _abaixo(caminho: Path, raiz: Path) -> Path:
    resolvido = caminho.resolve()
    if not resolvido.is_relative_to(raiz):
        raise ValueError(f"destino fora do HD informado: {caminho} -> {resolvido}")
    if resolvido.exists() and not resolvido.is_dir():
        raise ValueError(f"o destino precisa ser um diretório: {caminho}")
    return resolvido


def planejar_armazenamento(
    raiz_hd: Path, repositorio: Path, *, volume_sistema: str | None = None,
) -> dict:
    """Valida TODOS os caminhos antes de permitir qualquer criação de diretório.

    `volume_sistema` vem de SystemDrive no Windows. Recusá-lo é uma precaução;
    outra letra de unidade não prova que o disco seja um HD. Essa identificação
    continua sendo fornecida pelo usuário, não inferida por este código.
    """
    raiz_hd, repositorio = Path(raiz_hd), Path(repositorio)
    if not raiz_hd.is_absolute() or not repositorio.is_absolute():
        raise ValueError("informe caminhos absolutos para o HD e o checkout")
    raiz = raiz_hd.resolve(strict=True)
    if not raiz.is_dir():
        raise ValueError("a raiz do HD precisa ser um diretório existente")
    # A raiz em si também não pode ser um atalho para um disco desconhecido.
    if raiz != raiz_hd.absolute():
        raise ValueError("a raiz do HD contém link/junction; informe o destino real")
    if volume_sistema and raiz.drive.casefold() == volume_sistema.rstrip("\\/").casefold():
        raise ValueError("a raiz indicada está no volume do sistema; informe o HD de dados")
    repo = _abaixo(repositorio, raiz)
    if not (repo / "pyproject.toml").is_file():
        raise ValueError(f"checkout não encontrado: {repo / 'pyproject.toml'}")

    cache = raiz / "cache" / "phifm"
    venv = repo / ".venv-treino-cpu"
    destinos = {
        "TEMP": cache / "tmp", "TMP": cache / "tmp", "TMPDIR": cache / "tmp",
        "PIP_CACHE_DIR": cache / "pip", "UV_CACHE_DIR": cache / "uv",
        "UV_PYTHON_INSTALL_DIR": cache / "uv-python",
        "UV_TOOL_DIR": cache / "uv-tools", "UV_TOOL_BIN_DIR": cache / "uv-bin",
        "HF_HOME": cache / "huggingface", "HF_HUB_CACHE": cache / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": cache / "huggingface" / "hub",
        "HF_DATASETS_CACHE": cache / "huggingface" / "datasets",
        "HF_XET_CACHE": cache / "huggingface" / "xet",
        "HF_ASSETS_CACHE": cache / "huggingface" / "assets",
        "TORCH_HOME": cache / "torch", "TORCH_EXTENSIONS_DIR": cache / "torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor", "TRITON_CACHE_DIR": cache / "triton",
        "XDG_CACHE_HOME": cache / "xdg", "PYTHONPYCACHEPREFIX": cache / "pycache",
        "PYTHONUSERBASE": cache / "python-user", "NLTK_DATA": repo / "ferramentas" / "nltk_data",
    }
    # O projeto usa estes destinos relativos ao checkout para dados e modelos.
    conhecidos = [venv, repo / "models", repo / "data", repo / "ferramentas",
                  repo / "checkpoints", repo / "outputs"]
    for caminho in conhecidos + list(destinos.values()):
        _abaixo(caminho, raiz)
    # Links em níveis abaixo de data/models também podem redirecionar downloads.
    for base in conhecidos[1:]:
        if base.exists():
            for diretorio, pastas, _ in os.walk(base, followlinks=False):
                for nome in pastas:
                    _abaixo(Path(diretorio) / nome, raiz)

    variaveis = {nome: str(_abaixo(caminho, raiz)) for nome, caminho in destinos.items()}
    variaveis.update({
        "PHIFM_HD_ROOT": str(raiz), "PIP_CONFIG_FILE": os.devnull,
        "PIP_REQUIRE_VIRTUALENV": "true", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONNOUSERSITE": "1",
    })
    return {
        "raiz_hd": str(raiz), "repositorio": str(repo), "venv": str(venv),
        "diretorios": sorted({str(p) for p in destinos.values()}),
        "variaveis": variaveis,
        # Evita que opções herdadas redirecionem instalação/cache para o SSD.
        "remover_variaveis": ["PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PYTHONPATH",
                              "TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE",
                              "PYTORCH_PRETRAINED_BERT_CACHE"],
        "meio_fisico": "HD informado pelo usuário; tipo físico não detectado",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raiz-hd", type=Path, required=True)
    p.add_argument("--repositorio", type=Path, required=True)
    p.add_argument("--criar", action="store_true")
    a = p.parse_args()
    try:
        plano = planejar_armazenamento(a.raiz_hd, a.repositorio,
                                       volume_sistema=os.environ.get("SYSTEMDRIVE"))
        if a.criar:
            for caminho in plano["diretorios"]:
                Path(caminho).mkdir(parents=True, exist_ok=True)
            # Reconfere a resolução após criar os diretórios.
            plano = planejar_armazenamento(a.raiz_hd, a.repositorio,
                                           volume_sistema=os.environ.get("SYSTEMDRIVE"))
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(plano, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
