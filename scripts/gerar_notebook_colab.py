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

import importlib  # noqa: E402

REPO = "sanchezVB/LLM_F-sica"
PADRAO = "t2eq_emb_modernbert"

# ⚠️ O módulo das células é ESCOLHIDO, e não fixo: a secundária do caminho B tem duas
# células iguais que diferem só na `__VARIANTE__`, e um gerador por notebook faria as
# duas divergirem em silêncio — a mesma armadilha das células do Kaggle copiadas.
TITULOS = {
    "t2eq_emb_modernbert": "Passo 1 do caminho B — ModernBERT-base como ΦEmb",
    "t2eq_cpt_emb": "Secundária do caminho B — o CPT ajustado como ΦEmb",
    "t1h_bases_pequenas": "T1h — bases pequenas no lugar do MiniLM-L6",
}


def sha_do_repositorio() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAIZ, capture_output=True,
                       text=True, check=True)
    return r.stdout.strip()


def preencher(texto: str, sha: str, pasta: str, celulas, variante: str | None) -> str:
    texto = (texto.replace("__SHA__", sha).replace("__REPO__", REPO)
             .replace("__PASTA__", pasta)
             .replace("__HASHES__", json.dumps(celulas.HASHES, indent=4))
             .replace("__REGRA__", celulas.REGRA))
    if hasattr(celulas, "HASHES_ENCODER"):
        texto = texto.replace("__HASHES_ENCODER__",
                              json.dumps(celulas.HASHES_ENCODER, indent=4))
    if hasattr(celulas, "BASES"):
        texto = texto.replace("__BASES__", json.dumps(celulas.BASES, indent=4))
    if variante:
        texto = texto.replace("__VARIANTE__", variante)
    # ⚠️ Marcador que sobra vira string literal no notebook, e o erro só aparece
    # depois de a sessão subir e o Drive montar.
    for marcador in ("__SHA__", "__REPO__", "__PASTA__", "__HASHES__",
                     "__HASHES_ENCODER__", "__REGRA__", "__VARIANTE__",
                     "__BASES__"):
        if marcador in texto:
            raise SystemExit(
                f"o marcador {marcador} sobrou na célula. Ou falta um argumento "
                "(--variante?), ou a célula não devia tê-lo.")
    return texto


def notebook(sha: str, pasta: str, celulas, nome: str,
             variante: str | None) -> dict:
    braco = f" · braço {variante}" if variante else ""
    encoder = (f"\n\nE suba o encoder do braço para `{pasta}/encoders/"
               f"phienc-cpt-{variante}` (a pasta, ou o `.zip` dela).\n"
               if variante else "\n")
    cabecalho = [
        f"# {TITULOS.get(nome, nome)}{braco}\n",
        "\n",
        "Gerado por `scripts/gerar_notebook_colab.py`; **não edite aqui** — o fonte "
        f"é `colab/{nome}.py`.\n",
        "\n",
        f"Código: commit `{sha[:7]}`.\n",
        "\n",
        "**Antes de rodar:** Ambiente de execução → Alterar o tipo → **T4 GPU**, e "
        f"suba `pares_treino.parquet` e `pares_validacao.parquet` para `{pasta}/pares`.",
        encoder,
    ]
    celulas_json = [{"cell_type": "markdown", "metadata": {}, "source": cabecalho}]
    for fonte in celulas.CELULAS:
        linhas = preencher(fonte, sha, pasta, celulas,
                           variante).strip("\n").splitlines(keepends=True)
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
    p.add_argument("--celulas", default=PADRAO, choices=sorted(TITULOS),
                   help="o módulo de `colab/` com as células")
    p.add_argument("--variante", default=None,
                   help="o braço, quando as células têm `__VARIANTE__`")
    p.add_argument("--pasta", default=None,
                   help="pasta do Drive com `pares/` e onde os runs vão")
    p.add_argument("--saida", type=Path, default=None)
    p.add_argument("--sha", default=None, help="por omissão, o HEAD do repositório")
    a = p.parse_args()

    celulas = importlib.import_module(a.celulas)
    pasta = a.pasta or celulas.PASTA_PADRAO
    sufixo = f"-{a.variante}" if a.variante else ""
    saida = a.saida or RAIZ / "colab" / f"{a.celulas}{sufixo}.ipynb"

    sha = a.sha or sha_do_repositorio()
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(notebook(sha, pasta, celulas, a.celulas, a.variante),
                                indent=1, ensure_ascii=False) + "\n",
                     encoding="utf-8")
    print(f"  {saida}  ·  commit {sha[:7]}  ·  dados em {pasta}/pares")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
