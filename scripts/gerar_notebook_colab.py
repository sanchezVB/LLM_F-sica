#!/usr/bin/env python3
"""Monta o `.ipynb` do Colab a partir das células versionadas em `colab/`.

    PYTHONPATH=src python scripts/gerar_notebook_colab.py
    PYTHONPATH=src python scripts/gerar_notebook_colab.py --pasta /content/drive/MyDrive/outra

O notebook é ARTEFATO: quem manda é o `.py` ao lado das células, que é o que a suíte
lê. Gerar em vez de editar o `.ipynb` à mão evita o que já aconteceu com células do
Kaggle copiadas: a cópia diverge do original em silêncio.

## ⚠️ O SHA vai gravado no notebook

O notebook baixa o código do commit exato, e não de `main`. Um notebook que baixa
`main` mede uma versão do código que ninguém registrou — e o resultado deixa de dizer
o que foi rodado. Regenere depois de commitar mudanças no caminho de treino.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "colab"))

from phifm.core.console import utf8  # noqa: E402

utf8()

import t2eq_emb_modernbert as celulas  # noqa: E402

REPO = "sanchezVB/LLM_F-sica"
SAIDA = RAIZ / "colab" / "t2eq_emb_modernbert.ipynb"


def sha_do_repositorio() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True,
                       text=True, check=True)
    return r.stdout.strip()


def preencher(texto: str, sha: str, pasta: str) -> str:
    return (texto.replace("__SHA__", sha).replace("__REPO__", REPO)
            .replace("__PASTA__", pasta)
            .replace("__HASHES__", json.dumps(celulas.HASHES, indent=4))
            .replace("__REGRA__", celulas.REGRA))


def notebook(sha: str, pasta: str) -> dict:
    cabecalho = [
        "# Passo 1 do caminho B — ModernBERT-base como ΦEmb\n",
        "\n",
        "Gerado por `scripts/gerar_notebook_colab.py`; **não edite aqui** — o fonte "
        "é `colab/t2eq_emb_modernbert.py`.\n",
        "\n",
        f"Código: commit `{sha[:7]}`.\n",
        "\n",
        "**Antes de rodar:** Ambiente de execução → Alterar o tipo → **T4 GPU**, e "
        f"suba `pares_treino.parquet` e `pares_validacao.parquet` para `{pasta}/pares`.\n",
    ]
    celulas_json = [{"cell_type": "markdown", "metadata": {}, "source": cabecalho}]
    for fonte in celulas.CELULAS:
        linhas = preencher(fonte, sha, pasta).strip("\n").splitlines(keepends=True)
        celulas_json.append({
            "cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": linhas})
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": celulas_json,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pasta", default=celulas.PASTA_PADRAO,
                   help="pasta do Drive com `pares/` e onde os runs vão")
    p.add_argument("--saida", type=Path, default=SAIDA)
    p.add_argument("--sha", default=None, help="por omissão, o HEAD do repositório")
    a = p.parse_args()

    sha = a.sha or sha_do_repositorio()
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    a.saida.write_text(json.dumps(notebook(sha, a.pasta), indent=1,
                                  ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  {a.saida}  ·  commit {sha[:7]}  ·  dados em {a.pasta}/pares")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
