#!/usr/bin/env python3
"""Baixa o que a medida do G2 usa. O dono autorizou cada item: os benchmarks e os modelos
base numa primeira leva, e o resto numa segunda, as duas em 2026-10-19. Tudo vai para o HD
e fica fora do git.

    .venv-treino\\Scripts\\python.exe scripts\\baixar_g2.py [--so benchmarks|modelos|github]
    .venv\\Scripts\\python.exe scripts\\baixar_g2.py --so nltk

O que cada opção baixa:
- benchmarks: os do G2 e os do dev (MMLU validation, MBPP), com HumanEval+ e MBPP+.
- modelos: as duas bases.
- github: o conversor do llama.cpp na tag do binário local (b11159) e o código oficial de
  correção do IFEval.
- nltk: o tokenizador de frases que esse código usa. Roda na venv onde o nltk está.

Cada repositório está FIXADO numa revisão, o commit do dia da autorização. Quem baixar de
novo recebe os mesmos bytes, ou o script recusa. Cada arquivo é conferido contra o hash que
a origem declara: o SHA-256 do LFS, ou o SHA-1 de blob do git. O manifesto
(`avaliacao/g2_downloads.json`, versionado) guarda o tamanho e o SHA-256 de cada um. O
conteúdo dos benchmarks NÃO vai ao git (DOC-11 §8.1).

Destinos: `data/raw/benchmarks/<nome>/`, `models/base/<nome>/` e `ferramentas/<nome>/`. Não
use a pasta `benchmarks/` da raiz: o `.gitignore` a deixa entrar no git.

O GPQA não está aqui: o dono decidiu não usá-lo (DOC-12 §3.7).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
BENCH = RAIZ / "data/raw/benchmarks"
MODELOS = RAIZ / "models/base"
FERRAMENTAS = RAIZ / "ferramentas"
NLTK_DATA = FERRAMENTAS / "nltk_data"
MANIFESTO = RAIZ / "data/processed/avaliacao/g2_downloads.json"

# (nome, repositório, tipo, revisão fixada, arquivos) — exatamente a lista autorizada.
ITENS = [
    ("olympiadbench", "Hothan/OlympiadBench", "dataset", "91184b52131e", [
        "OlympiadBench/OE_TO_physics_en_COMP/OE_TO_physics_en_COMP.parquet",
        "OlympiadBench/OE_TO_physics_zh_CEE/OE_TO_physics_zh_CEE.parquet"]),
    ("phybench", "Eureka-Lab/PHYBench", "dataset", "d6d91c787b7a", [
        "PHYBench-fullques_v1.json"]),
    ("scibench", "xw27/scibench", "dataset", "93931252bc1b", [
        "class.json", "class_sol.json", "fund.json", "fund_sol.json", "thermo.json",
        "thermo_sol.json"]),
    ("ugphysics", "UGPhysics/ugphysics", "dataset", "523e71bc4e33", [
        f"{m}/en.jsonl" for m in (
            "AtomicPhysics", "ClassicalElectromagnetism", "ClassicalMechanics",
            "Electrodynamics", "GeometricalOptics", "QuantumMechanics", "Relativity",
            "SemiconductorPhysics", "Solid-StatePhysics", "StatisticalMechanics",
            "TheoreticalMechanics", "Thermodynamics", "WaveOptics")]),
    ("mmlu", "cais/mmlu", "dataset", "c30699e8356d", [
        "all/dev-00000-of-00001.parquet", "all/test-00000-of-00001.parquet",
        "all/validation-00000-of-00001.parquet"]),
    ("humaneval", "openai/openai_humaneval", "dataset", "7dce6050a7d6", [
        "openai_humaneval/test-00000-of-00001.parquet"]),
    ("ifeval", "google/IFEval", "dataset", "966cd89545d6", ["ifeval_input_data.jsonl"]),
    # Segunda leva (2026-10-19): o código do G2.3 e o dev de código do treino.
    ("humanevalplus", "evalplus/humanevalplus", "dataset", "d32357cf319e", ["test.jsonl"]),
    ("mbppplus", "evalplus/mbppplus", "dataset", "b2d74c91837c", [
        "data/test-00000-of-00001-d5781c9c51e02795.parquet"]),
    ("mbpp", "google-research-datasets/mbpp", "dataset", "4bb6404fdc6c", [
        f"full/{s}-00000-of-00001.parquet" for s in ("train", "validation", "test", "prompt")]),
    ("Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B-Base", "model", "ea980cb0a6c2", None),
    ("Qwen3-8B-Base", "Qwen/Qwen3-8B-Base", "model", "49e3418fbbbc", None),
]

# (nome, repositório do GitHub, commit fixado, pasta, filtro dos caminhos). Com `pasta`, a
# listagem é só daquela pasta, sem recursão (o google-research é grande demais para a
# árvore inteira); sem ela, a árvore inteira do commit, filtrada.
GITHUB = [
    # O conversor da MESMA tag do binário local (b11159), para converter a base e o ΦGen com
    # a mesma receita. Desde esta versão ele é um pacote: o script, `conversion/` e `gguf-py/`.
    ("llama.cpp-conversor-b11159", "ggml-org/llama.cpp",
     "6b790a9c291b5d7af3312bbf9f0c558aa023b13e", None,
     lambda p: (p in ("convert_hf_to_gguf.py", "LICENSE") or p.startswith("conversion/")
                or (p.startswith("gguf-py/") and "/tests/" not in p)
                or p.startswith("requirements/requirements-convert"))),
    # A correção oficial do IFEval, sem os testes e sem o executor de linha de comando.
    ("ifeval-oficial", "google-research/google-research",
     "e6890f85757dd84e27ca6df2dd30651dafad28e0", "instruction_following_eval",
     lambda p: p.rsplit("/", 1)[-1] in ("evaluation_lib.py", "instructions.py",
                                        "instructions_registry.py", "instructions_util.py",
                                        "requirements.txt", "README.md")),
]
# O tokenizador de frases que o `instructions_util` do IFEval chama pelo nltk.
NLTK_RECURSOS = ("punkt_tab",)


def _sha256(arq: Path) -> str:
    h = hashlib.sha256()
    with arq.open("rb") as f:
        while bloco := f.read(1 << 24):
            h.update(bloco)
    return h.hexdigest()


def _blob_git(arq: Path) -> str:
    """O SHA-1 que o git dá ao conteúdo: `blob <tamanho>\\0` + os bytes."""
    h = hashlib.sha1(f"blob {arq.stat().st_size}\0".encode())
    with arq.open("rb") as f:
        while bloco := f.read(1 << 24):
            h.update(bloco)
    return h.hexdigest()


def _ler_manifesto() -> dict:
    return (json.loads(MANIFESTO.read_text(encoding="utf-8")) if MANIFESTO.exists()
            else {"o_que_e": "o que a medida do G2 baixou, com autorização do dono em "
                             "2026-10-19; conteúdo fora do git", "itens": {}})


def _gravar_manifesto(manifesto: dict) -> None:
    MANIFESTO.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")


def baixar_github(manifesto: dict) -> None:
    """Cada arquivo vem do `raw` no commit fixado e é conferido pelo SHA-1 de blob que a
    API do GitHub declara."""
    import requests

    for nome, repo, commit, pasta, filtro in GITHUB:
        api = f"https://api.github.com/repos/{repo}"
        if pasta:
            lista = requests.get(f"{api}/contents/{pasta}", params={"ref": commit},
                                 timeout=60).json()
            arvore = [{"path": x["path"], "sha": x["sha"], "size": x["size"]}
                      for x in lista if x["type"] == "file"]
        else:
            r = requests.get(f"{api}/git/trees/{commit}", params={"recursive": 1},
                             timeout=60).json()
            if r.get("truncated"):
                raise SystemExit(f"{repo}: a árvore veio truncada; use `pasta`")
            arvore = [x for x in r["tree"] if x["type"] == "blob"]
        escolhidos = sorted((x for x in arvore if filtro(x["path"])), key=lambda x: x["path"])
        destino = FERRAMENTAS / nome
        registro = {"repositorio": f"github.com/{repo}", "tipo": "codigo", "revisao": commit,
                    "arquivos": {}}
        for x in escolhidos:
            local = destino / x["path"]
            local.parent.mkdir(parents=True, exist_ok=True)
            corpo = requests.get(f"https://raw.githubusercontent.com/{repo}/{commit}/{x['path']}",
                                 timeout=120)
            corpo.raise_for_status()
            local.write_bytes(corpo.content)
            if local.stat().st_size != x["size"] or _blob_git(local) != x["sha"]:
                raise SystemExit(f"{repo}/{x['path']}: o blob do git não bate")
            registro["arquivos"][x["path"]] = {"bytes": x["size"], "sha256": _sha256(local)}
        manifesto["itens"][nome] = registro
        _gravar_manifesto(manifesto)
        print(f"✅ {nome}: {len(escolhidos)} arquivos, "
              f"{sum(x['size'] for x in escolhidos) / 1e6:.2f} MB → {destino}", flush=True)


def baixar_nltk(manifesto: dict) -> None:
    """O nltk baixa do índice dele, que não declara hash: o manifesto grava o SHA-256 do zip
    que chegou, e quem baixar de novo compara com ele."""
    import nltk

    registro = {"repositorio": "nltk_data", "tipo": "dados", "revisao": f"nltk {nltk.__version__}",
                "arquivos": {}}
    anterior = manifesto["itens"].get("nltk_data", {}).get("arquivos", {})
    for recurso in NLTK_RECURSOS:
        if not nltk.download(recurso, download_dir=str(NLTK_DATA), quiet=True):
            raise SystemExit(f"nltk: {recurso} não baixou")
        for zip_ in sorted(NLTK_DATA.rglob(f"{recurso}.zip")):
            caminho = zip_.relative_to(NLTK_DATA).as_posix()
            sha = _sha256(zip_)
            if caminho in anterior and anterior[caminho]["sha256"] != sha:
                raise SystemExit(f"nltk: {caminho} mudou desde a primeira vez ({sha[:12]}…)")
            registro["arquivos"][caminho] = {"bytes": zip_.stat().st_size, "sha256": sha}
    manifesto["itens"]["nltk_data"] = registro
    _gravar_manifesto(manifesto)
    print(f"✅ nltk_data → {NLTK_DATA} (defina NLTK_DATA para essa pasta)", flush=True)


def baixar(so: str | None) -> None:
    manifesto = _ler_manifesto()
    if so == "nltk":
        baixar_nltk(manifesto)
        return
    if so == "github":
        baixar_github(manifesto)
        return
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    for nome, repo, tipo, rev_curta, arquivos in ITENS:
        e_modelo = tipo == "model"
        if so and so != ("modelos" if e_modelo else "benchmarks"):
            continue
        info = api.model_info(repo) if e_modelo else api.dataset_info(repo)
        arvore = {a.path: a for a in api.list_repo_tree(repo, repo_type=tipo, recursive=True,
                                                        revision=info.sha) if hasattr(a, "size")}
        if not info.sha.startswith(rev_curta):
            # O repositório mudou depois da autorização: baixa a revisão fixada, não a nova.
            print(f"⚠️ {repo} está em {info.sha[:12]}; uso a revisão fixada {rev_curta}")
            revisao = next(c.commit_id for c in api.list_repo_commits(repo, repo_type=tipo)
                           if c.commit_id.startswith(rev_curta))
            arvore = {a.path: a for a in api.list_repo_tree(repo, repo_type=tipo,
                                                            recursive=True, revision=revisao)
                      if hasattr(a, "size")}
        else:
            revisao = info.sha
        destino = (MODELOS if e_modelo else BENCH) / nome
        destino.mkdir(parents=True, exist_ok=True)
        lista = arquivos or sorted(arvore)
        registro = {"repositorio": repo, "tipo": tipo, "revisao": revisao, "arquivos": {}}
        for caminho in lista:
            meta = arvore[caminho]
            t0 = time.perf_counter()
            local = Path(hf_hub_download(repo, caminho, repo_type=tipo, revision=revisao,
                                         local_dir=destino))
            if local.stat().st_size != meta.size:
                raise SystemExit(f"{repo}/{caminho}: {local.stat().st_size} bytes, o Hugging "
                                 f"Face declara {meta.size}")
            sha = _sha256(local)
            lfs = getattr(meta, "lfs", None)
            if lfs is not None and sha != lfs.sha256:
                raise SystemExit(f"{repo}/{caminho}: SHA-256 {sha[:16]}… e o LFS declara "
                                 f"{lfs.sha256[:16]}…")
            if lfs is None and _blob_git(local) != meta.blob_id:
                raise SystemExit(f"{repo}/{caminho}: o blob do git não bate")
            registro["arquivos"][caminho] = {"bytes": meta.size, "sha256": sha}
            print(f"   {nome}/{caminho} · {meta.size / 1e6:,.1f} MB · conferido · "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)
        manifesto["itens"][nome] = registro
        _gravar_manifesto(manifesto)
        print(f"✅ {nome} → {destino}", flush=True)
    if so is None:
        baixar_github(manifesto)
    print(f"manifesto → {MANIFESTO.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--so", choices=("benchmarks", "modelos", "github", "nltk"))
    a = p.parse_args()
    baixar(a.so)
    return 0


if __name__ == "__main__":
    sys.exit(main())
