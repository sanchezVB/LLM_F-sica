#!/usr/bin/env python3
"""Sobe o pacote da 8B e roda os notebooks no Kaggle pela CLI (a sessão de
`python -m kaggle auth login` desta máquina), sem passar pela web.

    # parte 1: o dataset g2-8b e o notebook de preparo (CPU, internet, sem cota de GPU)
    .venv\\Scripts\\python.exe scripts\\publicar_g2_8b.py preparo --enviar
    # parte 2: uma sessão das medidas (GPU T4 x2: GASTA COTA; só com o OK do dono)
    .venv\\Scripts\\python.exe scripts\\publicar_g2_8b.py medida --sessao 1 --cota 30 --enviar
    # o andamento e a volta dos arquivos pequenos (JSON e logs; nunca o GGUF)
    .venv\\Scripts\\python.exe scripts\\publicar_g2_8b.py estado preparo
    .venv\\Scripts\\python.exe scripts\\publicar_g2_8b.py baixar preparo

As lições do `publicar_kaggle.py` valem aqui: `python -m kaggle`, nunca o executável
solto; `create` só na 1ª vez, depois `version`; esperar o dataset ficar pronto antes do
`kernels push`; a GPU se pede por `machine_shape` (o `enable_gpu` é ignorado). Tudo
privado. Na parte 2, a imagem fica fixada na da 1ª sessão (`original`): o driver e a
imagem entram no hash, e as sessões seguintes têm de rodar na mesma.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PACOTE = RAIZ / "data/processed/kaggle_g2_8b"
NOTEBOOKS = RAIZ / "data/processed/kaggle_notebook"
RETORNO = RAIZ / "data/processed/kaggle_g2_8b_retorno"
SLUGS = {"dados": "phifm-g2-8b", "estado": "phifm-g2-8b-estado",
         "preparo": "phifm-g2-8b-preparo", "medida": "phifm-g2-8b-medida"}
TITULOS = {"dados": "PhiFM G2 8B", "estado": "PhiFM G2 8B Estado",
           "preparo": "PhiFM G2 8B Preparo", "medida": "PhiFM G2 8B Medida"}
# só o que é pequeno volta: nunca o Q8_0 (8,7 GB) nem os logits (3,6 GB)
PEQUENOS = r"(\.json|\.log|\.jsonl|ESTADO\.json|estado_[0-9a-f]+\.zip\.bin)$"


def kaggle(*args: str, saida: bool = True) -> str:
    r = subprocess.run([sys.executable, "-m", "kaggle", *args], cwd=RAIZ, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    texto = "\n".join(x for x in (r.stdout + r.stderr).split("\n") if "outdated" not in x).strip()
    if saida and texto:
        print(texto)
    if r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args[:2])} saiu com {r.returncode}")
    return texto


def usuario() -> str:
    m = re.search(r"username:\s*(\S+)", kaggle("config", "view", saida=False))
    if not m:
        raise SystemExit("não há sessão do Kaggle: rode `python -m kaggle auth login`")
    return m.group(1).lower()


def dataset_existe(ref: str) -> bool:
    return ref.split("/", 1)[1] in kaggle("datasets", "list", "--mine", "-s",
                                          ref.split("/", 1)[1], saida=False)


def subir_dataset(pasta: Path, ref: str, titulo: str, mensagem: str) -> None:
    """Cria (1ª vez) ou versiona o dataset e espera o Kaggle processá-lo."""
    (pasta / "dataset-metadata.json").write_text(json.dumps(
        {"title": titulo, "id": ref, "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    if dataset_existe(ref):
        kaggle("datasets", "version", "-p", str(pasta), "--dir-mode", "zip", "-m", mensagem)
    else:
        kaggle("datasets", "create", "-p", str(pasta), "--dir-mode", "zip")
    for _ in range(90):
        if "ready" in kaggle("datasets", "status", ref, saida=False).lower():
            print(f"dataset {ref}: pronto")
            return
        time.sleep(20)
    raise SystemExit(f"o dataset {ref} não ficou pronto em 30 min")


def empurrar(nome: str, celula: str, meta: dict) -> None:
    pasta = NOTEBOOKS / SLUGS[nome]
    pasta.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": celula.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (pasta / f"{SLUGS[nome]}.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    (pasta / "kernel-metadata.json").write_text(json.dumps(
        {"id": meta["id"], "title": TITULOS[nome], "code_file": f"{SLUGS[nome]}.ipynb",
         "language": "python", "kernel_type": "notebook", "is_private": True,
         "enable_internet": True, "competition_sources": [], **meta}, indent=2), encoding="utf-8")
    kaggle("kernels", "push", "-p", str(pasta))


def celula_do_pacote(nome: str) -> str:
    nb = json.loads((PACOTE / f"g2_8b_{nome}.ipynb").read_text(encoding="utf-8"))
    return "".join(nb["cells"][0]["source"])


def preparo(enviar: bool) -> None:
    dono = usuario()
    man = json.loads((PACOTE / "MANIFESTO.json").read_text(encoding="utf-8"))
    print(f"pacote g2-8b: {len(man['arquivos'])} arquivos; protocolo {man['protocolo_g2_sha256'][:12]}…")
    if not enviar:
        print("(sem --enviar: nada foi enviado)")
        return
    subir_dataset(PACOTE, f"{dono}/{SLUGS['dados']}", TITULOS["dados"],
                  f"protocolo {man['protocolo_g2_sha256'][:12]}")
    empurrar("preparo", celula_do_pacote("preparo"),
             {"id": f"{dono}/{SLUGS['preparo']}", "enable_gpu": False,
              "docker_image_pinning_type": "latest",
              "dataset_sources": [f"{dono}/{SLUGS['dados']}"], "kernel_sources": []})


def medida(sessao: int, cota: float, enviar: bool) -> None:
    dono = usuario()
    man = json.loads((PACOTE / "MANIFESTO.json").read_text(encoding="utf-8"))
    if not man.get("kl_8b_pronta"):
        raise SystemExit("o pacote não tem a KL da 8B aprovada: rode a parte 1, traga o "
                         "g2_kl_8B.json e reempacote")
    fase = "sanidade" if sessao == 1 else "teste"
    celula = celula_do_pacote("medida")
    for velho, novo in ((r'^FASE = "sanidade"', f'FASE = "{fase}"'), (r"^SESSAO = 1\b", f"SESSAO = {sessao}"),
                        (r"^COTA_RESTANTE_H = 30\.0", f"COTA_RESTANTE_H = {cota:.2f}")):
        celula, n = re.subn(velho, novo, celula, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"não achei a linha {velho} na célula")
    fontes = [f"{dono}/{SLUGS['dados']}"] + ([f"{dono}/{SLUGS['estado']}"] if sessao > 1 else [])
    print(f"sessão {sessao} ({fase}), cota informada {cota} h, entradas {fontes} + "
          f"{dono}/{SLUGS['preparo']}")
    if not enviar:
        print("(sem --enviar: nada foi enviado)")
        return
    empurrar("medida", celula,
             {"id": f"{dono}/{SLUGS['medida']}", "enable_gpu": True, "machine_shape": "NvidiaTeslaT4",
              "docker_image_pinning_type": "original", "dataset_sources": fontes,
              "kernel_sources": [f"{dono}/{SLUGS['preparo']}"]})


def estado(nome: str) -> None:
    kaggle("kernels", "status", f"{usuario()}/{SLUGS[nome]}")


def baixar(nome: str) -> Path:
    destino = RETORNO / nome
    destino.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{usuario()}/{SLUGS[nome]}", "-p", str(destino), "-o",
           "--file-pattern", PEQUENOS)
    print(f"→ {destino}")
    return destino


def main() -> int:
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="acao", required=True)
    a1 = sub.add_parser("preparo")
    a1.add_argument("--enviar", action="store_true")
    a2 = sub.add_parser("medida")
    a2.add_argument("--sessao", type=int, required=True)
    a2.add_argument("--cota", type=float, required=True, help="horas de GPU que restam")
    a2.add_argument("--enviar", action="store_true")
    for nome in ("estado", "baixar"):
        s = sub.add_parser(nome)
        s.add_argument("notebook", choices=("preparo", "medida"))
    a = p.parse_args()
    if a.acao == "preparo":
        preparo(a.enviar)
    elif a.acao == "medida":
        medida(a.sessao, a.cota, a.enviar)
    elif a.acao == "estado":
        estado(a.notebook)
    else:
        baixar(a.notebook)
    return 0


if __name__ == "__main__":
    sys.exit(main())
