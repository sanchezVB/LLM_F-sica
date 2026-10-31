#!/usr/bin/env python3
"""Monta o pacote da Qwen3-8B-Base para o Kaggle (DOC-12 §3.7.5): o dataset `g2-8b`.

    .venv\\Scripts\\python.exe scripts\\empacotar_g2_8b.py

Sai `data/processed/kaggle_g2_8b/` (e o .zip ao lado, para subir pela web), fora do git:

- `repo.zip.bin`: os arquivos do repositório que as medidas usam, BYTE A BYTE (o
  congelamento confere o SHA-256 dos bytes; um clone no Linux teria outros fins de linha).
  Vai num arquivo só, e com extensão `.bin`, porque o Kaggle descompacta todo `.zip` que
  sobe e descarta arquivos vazios (o mesmo cuidado do `empacotar_kaggle.py`);
- `modelo.json`: a revisão fixada da Qwen3-8B-Base, o SHA-256 de cada arquivo daqui e o
  Q8_0 convertido aqui, inteiro e tensor a tensor (`g2_gguf_8B.json`);
- `binarios.json`: os assets oficiais do llama.cpp b11159 (CPU para a KL, CUDA 12.8 para
  as medidas), com o SHA-256 da API de releases;
- `requisitos*.txt`: as versões congeladas dos pacotes (o congelamento recusa outras) e as
  da conversão local (transformers, tokenizers, numpy…);
- os dois notebooks (`kaggle/g2_8b_preparo.py` e `kaggle/g2_8b_medida.py`) e o MANIFESTO,
  com o SHA-256 de cada arquivo do pacote e de cada arquivo dentro do `repo.zip.bin`.

Antes de fechar, extrai o `repo.zip.bin` numa pasta temporária e confere que ele passa no
`exigir_aceite_e_congelamento`. Sobrescreve, não apaga.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
PACOTE = RAIZ / "data/processed/kaggle_g2_8b"
CELULAS = {"g2_8b_preparo.ipynb": RAIZ / "kaggle/g2_8b_preparo.py",
           "g2_8b_medida.ipynb": RAIZ / "kaggle/g2_8b_medida.py"}
KL_8B = "data/processed/avaliacao/g2_kl_8B.json"

# o que as medidas leem, relativo à raiz do repositório
ARQUIVOS = (
    "src/phifm/**/*.py",
    "scripts/medir_g2.py", "scripts/medir_g2_mmlu.py", "scripts/medir_g2_ifeval.py",
    "scripts/medir_g2_evalplus.py", "scripts/registrar_kl_g2.py",
    "kaggle/g2_evalplus.py",
    "configs/eval/preregistration/*", "configs/eval/g2_ifeval_dev20.jsonl",
    "data/processed/avaliacao/g2_particao.json", "data/processed/avaliacao/g2_exemplos.json",
    "data/processed/avaliacao/g2_autoteste.json", "data/processed/avaliacao/g2_acessos.jsonl",
    "data/processed/avaliacao/g2_downloads.json", "data/processed/avaliacao/g2_gguf_8B.json",
    "data/processed/avaliacao/g2_vazao_1.7B.json",
    "data/processed/avaliacao/g2_teste_1.7B_base_veredito.json",   # as amostras esperadas
    KL_8B,
    "data/raw/benchmarks/ugphysics/*/en.jsonl",
    "data/raw/benchmarks/olympiadbench/OlympiadBench/*/*.parquet",
    "data/raw/benchmarks/phybench/PHYBench-fullques_v1.json",
    "data/raw/benchmarks/scibench/*.json",
    "data/raw/benchmarks/mmlu/all/*.parquet",
    "data/raw/benchmarks/ifeval/ifeval_input_data.jsonl",
    "data/raw/benchmarks/evalplus-oficial/*.jsonl",
    "ferramentas/ifeval-oficial/**/*", "ferramentas/nltk_data/**/*",
    "ferramentas/llama.cpp-conversor-b11159/**/*",
    "models/base/gguf/kl_dev.txt",
)

HF = {"repositorio": "Qwen/Qwen3-8B-Base", "revisao": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"}
BINARIOS = {
    "tag": "b11159", "commit": "6b790a9c291b5d7af3312bbf9f0c558aa023b13e",
    "base_url": "https://github.com/ggml-org/llama.cpp/releases/download/b11159/",
    "cpu": {"llama-b11159-bin-ubuntu-x64.tar.gz":
            "31ec911b59ea271264d8a7dcfec095a41c92155db2786a43d5f358d5fe00500d"},
    "cuda": {"llama-b11159-bin-ubuntu-cuda-12.8-x64.tar.gz":
             "5840b0f87ec10400d7eea2f36138a45ddd9bdd6a794eb81d20a7f150d09c66bd",
             "cudart-llama-b11159-bin-ubuntu-cuda-12.8-x64.tar.gz":
             "f79ae034d707096f825be4b512366253950d3fa4895921a8c2577d4271ff8a18"},
}
# As versões que o congelamento confere (g2.PACOTES_DO_CORRETOR e PACOTES_DA_MEDIDA), mais
# as dependências leves do evalplus, que vai sem as pesadas (vllm, openai…), como aqui.
CONGELADOS = ("sympy", "antlr4-python3-runtime", "mpmath", "langdetect", "nltk", "absl-py",
              "immutabledict", "tree-sitter", "tree-sitter-python")
SEM_DEPENDENCIAS = ("evalplus",)
LEVES = ("wget", "tempdir", "appdirs", "multipledispatch", "termcolor", "fire")
# As da conversão local do Q8_0 da 8B (.venv-treino, g2_gguf_8B.json); o torch fica o da
# imagem (a conversão bf16 → f32 → Q8_0 não depende da versão dele).
CONVERSAO = {"transformers": "5.0.0", "tokenizers": "0.22.2", "safetensors": "0.8.0",
             "numpy": "2.5.1", "sentencepiece": "0.2.2", "huggingface-hub": "1.31.0"}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while bloco := f.read(1 << 24):
            h.update(bloco)
    return h.hexdigest()


def arquivos_do_repo() -> list[Path]:
    achados: set[Path] = set()
    for padrao in ARQUIVOS:
        casa = [p for p in RAIZ.glob(padrao) if p.is_file() and "__pycache__" not in p.parts]
        if not casa and padrao != KL_8B:
            raise SystemExit(f"nada casa com {padrao}")
        achados.update(casa)
    return sorted(achados)


def notebook(celula: str) -> dict:
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": celula.splitlines(keepends=True)}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def conferir_congelamento(zip_repo: Path) -> str:
    """O `repo.zip.bin`, extraído, tem de passar na mesma trava que a medida usa."""
    with tempfile.TemporaryDirectory() as d:
        with zipfile.ZipFile(zip_repo) as z:
            z.extractall(d)
        codigo = ("import sys; sys.path.insert(0, 'src'); from phifm.eval import g2; "
                  "print(g2.exigir_aceite_e_congelamento())")
        r = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", codigo], cwd=d,
                           capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"o repo do pacote NÃO passa no congelamento:\n{r.stderr[-2000:]}")
    return r.stdout.strip()


def main() -> int:
    gguf = json.loads((RAIZ / "data/processed/avaliacao/g2_gguf_8B.json").read_text(encoding="utf-8"))
    for campo in ("safetensors_locais", "mapa_q8_0"):
        if campo not in gguf:
            raise SystemExit(f"g2_gguf_8B.json sem '{campo}'")
    PACOTE.mkdir(parents=True, exist_ok=True)
    repo = {}
    zip_repo = PACOTE / "repo.zip.bin"
    with zipfile.ZipFile(zip_repo, "w", zipfile.ZIP_DEFLATED) as z:
        for p in arquivos_do_repo():
            rel = p.relative_to(RAIZ).as_posix()
            dados = p.read_bytes()
            z.writestr(rel, dados)
            repo[rel] = hashlib.sha256(dados).hexdigest()
    congelado = conferir_congelamento(zip_repo)
    print(f"congelamento do pacote: ok ({congelado[:12]}…)")
    (PACOTE / "modelo.json").write_text(json.dumps(
        {**HF, "arquivos": gguf["safetensors_locais"], "q8_0_local": gguf["gguf"]["q8_0"],
         "mapa_q8_0": gguf["mapa_q8_0"]}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (PACOTE / "binarios.json").write_text(json.dumps(BINARIOS, indent=1) + "\n", encoding="utf-8")
    # pyarrow e pyyaml: sem versão fixa (não decidem resultado), só para garantir que existam
    (PACOTE / "requisitos.txt").write_text(
        "".join(f"{p}=={version(p)}\n" for p in CONGELADOS + LEVES) + "pyarrow\npyyaml\n",
        encoding="utf-8")
    (PACOTE / "requisitos_sem_dependencias.txt").write_text(
        "".join(f"{p}=={version(p)}\n" for p in SEM_DEPENDENCIAS), encoding="utf-8")
    (PACOTE / "requisitos_conversao.txt").write_text(
        "".join(f"{p}=={v}\n" for p, v in CONVERSAO.items()), encoding="utf-8")
    for nome, celula in CELULAS.items():
        (PACOTE / nome).write_text(json.dumps(notebook(celula.read_text(encoding="utf-8")),
                                              ensure_ascii=False, indent=1), encoding="utf-8")
    livro = (RAIZ / "data/processed/avaliacao/g2_acessos.jsonl").read_bytes()
    # dataset-metadata.json é do CLI (publicar_g2_8b.py) e não sobe como dado
    arquivos = {p.relative_to(PACOTE).as_posix(): {"bytes": p.stat().st_size, "sha256": _sha(p)}
                for p in sorted(PACOTE.iterdir()) if p.is_file()
                and p.name not in ("MANIFESTO.json", "dataset-metadata.json")}
    manifesto = {"pacote": "g2_8b", "protocolo_g2_sha256": congelado,
                 "celulas_sha256": {n: _sha(c) for n, c in CELULAS.items()},
                 "kl_8b_pronta": KL_8B in repo and json.loads(
                     (RAIZ / KL_8B).read_text(encoding="utf-8")).get("passa") is True,
                 "livro_fotografado": {"sha256": hashlib.sha256(livro).hexdigest(),
                                       "linhas": livro.count(b"\n")},
                 "arquivos": arquivos, "repo": repo}
    (PACOTE / "MANIFESTO.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=1) + "\n",
                                           encoding="utf-8")
    shutil.make_archive(str(PACOTE), "zip", root_dir=PACOTE)
    total = sum(v["bytes"] for v in arquivos.values())
    print(f"pacote: {PACOTE} ({len(arquivos)} arquivos, {len(repo)} no repo.zip.bin, "
          f"{total / 1e6:.1f} MB); zip ao lado; KL da 8B no pacote: {manifesto['kl_8b_pronta']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
