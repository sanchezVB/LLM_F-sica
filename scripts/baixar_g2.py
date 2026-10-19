#!/usr/bin/env python3
"""Baixa o que a medida do G2 usa: os benchmarks e os modelos base, com o dono tendo
autorizado cada item (2026-10-19). Tudo no HD e fora do git.

    .venv-treino\\Scripts\\python.exe scripts\\baixar_g2.py [--so benchmarks|modelos]

Cada repositório está FIXADO numa revisão (o commit do Hugging Face no dia da autorização):
quem baixar de novo recebe os mesmos bytes, ou o script recusa. Cada arquivo é conferido
contra o hash que o Hugging Face declara — o SHA-256 do LFS, ou o SHA-1 de blob do git — e
o manifesto (`avaliacao/g2_downloads.json`, versionado) guarda tamanho e SHA-256 de cada
um. O conteúdo dos benchmarks NÃO vai ao git (DOC-11 §8.1; o GPQA ainda proíbe divulgar).

Destinos: `data/raw/benchmarks/<nome>/` e `models/base/<nome>/`. Não use a pasta
`benchmarks/` da raiz: o `.gitignore` a deixa entrar no git.

O GPQA não está aqui: ele exige que o dono aceite termos no Hugging Face, por conta própria.
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
        "all/dev-00000-of-00001.parquet", "all/test-00000-of-00001.parquet"]),
    ("humaneval", "openai/openai_humaneval", "dataset", "7dce6050a7d6", [
        "openai_humaneval/test-00000-of-00001.parquet"]),
    ("ifeval", "google/IFEval", "dataset", "966cd89545d6", ["ifeval_input_data.jsonl"]),
    ("Qwen3-1.7B-Base", "Qwen/Qwen3-1.7B-Base", "model", "ea980cb0a6c2", None),
    ("Qwen3-8B-Base", "Qwen/Qwen3-8B-Base", "model", "49e3418fbbbc", None),
]


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


def baixar(so: str | None) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    manifesto = (json.loads(MANIFESTO.read_text(encoding="utf-8")) if MANIFESTO.exists()
                 else {"o_que_e": "o que a medida do G2 baixou, com autorização do dono em "
                                  "2026-10-19; conteúdo fora do git", "itens": {}})
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
        MANIFESTO.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"✅ {nome} → {destino}", flush=True)
    print(f"manifesto → {MANIFESTO.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--so", choices=("benchmarks", "modelos"))
    a = p.parse_args()
    baixar(a.so)
    return 0


if __name__ == "__main__":
    sys.exit(main())
