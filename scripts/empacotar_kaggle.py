#!/usr/bin/env python3
"""Empacota o mínimo para um experimento de GPU rodar no Kaggle.

    .venv/Scripts/python.exe scripts/empacotar_kaggle.py --experimento t1a
    .venv/Scripts/python.exe scripts/empacotar_kaggle.py --experimento t1c

Produz `data/processed/kaggle_<exp>/`, para subir como Kaggle Dataset. Os nomes,
slugs e a lista de arquivos vêm de `phifm.core.kaggle`, para que este script e o
`publicar_kaggle.py` não possam discordar.

## T1a — ΦEmb, 211 MB e não 2,68 GB

O `pares_treino.parquet` inteiro tem 6,56 M de arestas e 2,68 GB. O T1a usa
**400 mil pares** — o volume do campeão do G1.1, escolhido por medição: o treino de
1,5 M empatou estatisticamente com ele (p=0,950) e o de 511 negativos também
(p=0,636). Subir o conjunto inteiro seria pagar 13× de banda por dados que a
medição diz não comprar nada.

## T1c — ΦRank de base diferente

Vão os negativos minerados do recuperador de verdade (o RRF top-50, já sem os
co-citados), a validação, e **dois modelos**: o ΦEmb campeão e o ΦRank de MiniLM
que serve de controle.

⚠️ Os negativos vão INTEIROS, 247 MB, com os 43,77 negativos por grupo. Cortar para
os 7 que o treino usa economizaria banda e **quebraria a comparação**: o controle
foi treinado sorteando 7 de 43,77, e sortear 7 de uma lista podada daria outros 7.
O experimento é de uma variável — a base —, então tudo o mais fica byte a byte igual.

⚠️ E os modelos vão porque não são públicos: o `phiemb-minilm-melhor` é o campeão do
G1.1 treinado aqui, e o resultado de referência do T1b (nDCG 0,1584) saiu dele. Usar
o `-t4-melhor` no lugar trocaria o recuperador junto com o reranqueador.

## O que sempre vai, e por que cada coisa

| arquivo | para quê |
|---|---|
| `phifm_src.zip.bin` | o pacote `phifm` + os scripts, para o notebook não reimplementar nada |
| `MANIFESTO.json` | hashes BLAKE3 e proveniência, para o que roda lá ser o que está aqui |

⚠️ O código vai como ZIP e não copiado no notebook. Um notebook que reimplementa o
laço de treino é um segundo laço para manter em sincronia, e a divergência entre os
dois seria invisível: os dois rodariam, com resultados diferentes, e nada apontaria
qual está certo. O manifesto existe para provar que o que rodou lá é este código.

## O que NÃO vai

Pesos públicos. `all-MiniLM-L6-v2`, `gte-base` e `physbert_cased` são baixados do
HuggingFace no próprio Kaggle, que tem rede — subir 90 MB de cada seria desperdício.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.kaggle import EXPERIMENTOS, Experimento, obter  # noqa: E402
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    git_sha_curto,
    hash_arquivo,
)

log = logging.getLogger("empacotar")

# Tudo do pacote, menos o que não roda lá nem faz sentido carregar.
EXCLUIR_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

# ⚠️ `.zip.bin`, nao `.zip`. O Kaggle DESCOMPACTA arquivos .zip no upload: medido em
# 2026-08-24, `phifm_src.zip` chegou no dataset como o diretorio `phifm_src/`, o
# notebook morreu em FileNotFoundError aos 26 s e — pior — o hash do FONTE deixou de
# ser conferivel, porque o arquivo que o manifesto descreve nao existia mais.
#
# Com uma extensao que o Kaggle nao reconhece como arquivo, ele guarda os bytes como
# estao e a conferencia por blake3 volta a valer. O `zipfile` abre pelo conteudo e
# nao pela extensao, entao nada mais muda.
SUFIXO_ZIP = ".zip.bin"


def _zipar_fonte(raiz: Path, destino: Path, scripts: tuple[str, ...]) -> int:
    n = 0
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((raiz / "src" / "phifm").rglob("*.py")):
            if any(p in EXCLUIR_DIRS for p in f.parts):
                continue
            z.write(f, f.relative_to(raiz / "src").as_posix())
            n += 1
        # Os scripts também, porque são o ponto de entrada de verdade — e assim o
        # notebook chama exatamente o que roda aqui.
        for nome in scripts:
            origem = raiz / "scripts" / nome
            if not origem.exists():
                raise SystemExit(
                    f"{origem} não existe, e o notebook do experimento a chama. "
                    "Zipar sem ela daria ModuleNotFoundError na GPU, depois do "
                    "upload inteiro.")
            z.write(origem, f"scripts/{nome}")
            n += 1
    return n


def _zipar_modelos(raiz: Path, destino: Path, modelos: tuple[str, ...]) -> int:
    """Zipa diretórios de modelo preservando só o nome final na raiz do ZIP.

    `models/phiemb-minilm-melhor/...` entra como `phiemb-minilm-melhor/...`, que é
    o que o notebook espera em `MODELOS / "phiemb-minilm-melhor"`.

    ⚠️ `estado_rank.pt` e `estado_treino.pt` ficam FORA. São o estado do otimizador
    para retomar treino, chegam a centenas de MB, e nada no notebook os lê — os
    modelos vão para inferência. Um `-melhor/` não os tem, mas o diretório de
    trabalho tem, e um dia alguém vai passar o errado.
    """
    n = 0
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in modelos:
            d = raiz / rel
            if not d.is_dir():
                raise SystemExit(
                    f"{d} não é um diretório. O experimento declara este modelo e "
                    "sem ele o notebook para no assert de pesos.")
            for f in sorted(d.rglob("*")):
                if not f.is_file() or f.suffix == ".pt":
                    continue
                z.write(f, (Path(d.name) / f.relative_to(d)).as_posix())
                n += 1
    return n


def _montar_t1a(exp: Experimento, raiz: Path, out: Path, a) -> dict:
    tr = pl.scan_parquet(a.pares / "pares_treino.parquet").head(a.max_pares).collect()
    tr.write_parquet(out / "pares_treino.parquet", compression="zstd")
    shutil.copy2(a.pares / "pares_validacao.parquet", out / "pares_validacao.parquet")
    n_py = _zipar_fonte(raiz, out / f"phifm_src{SUFIXO_ZIP}", exp.scripts)
    return {"max_pares": a.max_pares, "linhas_treino": tr.height, "modulos_python": n_py}


def _montar_t1c(exp: Experimento, raiz: Path, out: Path, a) -> dict:
    origem = a.negativos
    if not origem.exists():
        raise SystemExit(
            f"{origem} não existe. Rode scripts/minerar_do_recuperador.py e depois "
            "scripts/filtrar_cocitacao.py — treinar sobre os negativos NÃO filtrados "
            "ensina o reranqueador a rebaixar co-citados, que são relevantes.")
    shutil.copy2(origem, out / origem.name)
    shutil.copy2(a.pares / "pares_validacao.parquet", out / "pares_validacao.parquet")
    n_mod = _zipar_modelos(raiz, out / f"modelos{SUFIXO_ZIP}", exp.modelos)
    n_py = _zipar_fonte(raiz, out / f"phifm_src{SUFIXO_ZIP}", exp.scripts)
    grupos = pl.scan_parquet(origem).select(pl.len()).collect().item()
    return {"grupos": grupos, "modulos_python": n_py, "arquivos_de_modelo": n_mod,
            "modelos": list(exp.modelos),
            "negativos": str(origem).replace("\\", "/")}


MONTADORES = {"t1a": _montar_t1a, "t1c": _montar_t1c}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experimento", default="t1a", choices=sorted(EXPERIMENTOS))
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    p.add_argument("--out", type=Path, default=None,
                   help="por omissão, o `pacote` declarado pelo experimento")
    p.add_argument("--max-pares", type=int, default=400_000,
                   help="T1a: volume do campeão do G1.1; mais que isso a medição "
                        "diz que não compra nada (p=0,950)")
    p.add_argument("--negativos", type=Path,
                   default=Path("data/processed/negativos_dificeis/"
                                "pares_do_recuperador_limpos.parquet"),
                   help="T1c: negativos do recuperador de verdade, já sem co-citados")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s",
                        stream=sys.stdout)

    exp = obter(a.experimento)
    raiz = Path(__file__).resolve().parents[1]
    out = a.out or (raiz / exp.pacote)
    out.mkdir(parents=True, exist_ok=True)

    extra = MONTADORES[exp.nome](exp, raiz, out, a)

    # ⚠️ O conteudo do pacote e uma lista DECLARADA, e o que nao esta nela sai.
    #
    # Antes o manifesto era montado de `iterdir()`, entao qualquer arquivo obsoleto
    # no diretorio entrava na atestacao e subia para o Kaggle. Aconteceu em
    # 2026-08-24: ao renomear o fonte para `.zip.bin`, o `phifm_src.zip` antigo
    # ficou, e o manifesto passou a descrever OS DOIS — 175 KB de codigo velho
    # atestados como se fizessem parte do pacote.
    #
    # Um manifesto que descreve o que sobrou no disco em vez do que a etapa produziu
    # nao atesta nada.
    poupados = {"MANIFESTO.json", "dataset-metadata.json"}
    for f in sorted(out.iterdir()):
        if f.is_file() and f.name not in exp.arquivos and f.name not in poupados:
            log.warning("removendo arquivo que nao pertence ao pacote: %s", f.name)
            f.unlink()
    arquivos = [out / n for n in exp.arquivos]
    faltando = [f.name for f in arquivos if not f.exists()]
    if faltando:
        raise SystemExit(f"o pacote ficou sem {faltando} — nao vou gravar um "
                         "manifesto que descreve menos do que o notebook exige")
    manifesto = {
        "experimento": exp.nome,
        "git_sha": git_sha_curto(),
        **extra,
        "hash_algo": "blake3",
        "arquivos": {f.name: {"blake3": hash_arquivo(f), "bytes": f.stat().st_size}
                     for f in arquivos},
        "nota": ("Subir como Kaggle Dataset. O notebook confere estes hashes antes "
                 "de treinar: rodar sobre dados que não são estes produziria um "
                 "número incomparável com os medidos aqui."),
    }
    (out / "MANIFESTO.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(v["bytes"] for v in manifesto["arquivos"].values())
    print()
    print("=" * 68)
    for nome, v in manifesto["arquivos"].items():
        print(f"  {nome:34s} {v['bytes']/1e6:8.1f} MB  {v['blake3'][:12]}…")
    print(f"  {'TOTAL':34s} {total/1e6:8.1f} MB")
    print("=" * 68)
    print(f"  -> {out}")
    print(f"  {exp.nome} · git {manifesto['git_sha']} · "
          f"{manifesto['modulos_python']} módulos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
