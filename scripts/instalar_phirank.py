#!/usr/bin/env python3
"""Instala um ΦRank treinado no Kaggle como o reranqueador do sistema.

    .venv\\Scripts\\python.exe scripts\\instalar_phirank.py --de <dir> \\
        --para models/phirank-physbert-melhor --nota "T1c, p=0,0062"

## Por que isto é um script e não um `cp`

Um diretório de pesos que aparece em `models/` sem proveniência é um modelo que
ninguém sabe de onde veio nem por que está ali. Este script grava o manifesto de
etapa junto, com o que a medição diz e o que ela **não** diz.

## ⚠️ O `phirank.json` do modelo de PhysBERT tem uma linha errada

Ele diz `"inicializado do MiniLM, a mesma base do ΦEmb campeão"` — literal, porque
até 2026-09-03 essa string era fixa em `rerank.py` para qualquer base. O modelo veio
de `thellert/physbert_cased`, e o campo `"base"` do mesmo arquivo diz isso
corretamente.

O arquivo **não é corrigido** na instalação: ele é o que a execução produziu, e
reescrever a proveniência de um artefato depois do fato é pior que registrar a
divergência. Quem manda é o manifesto de etapa, que aponta o erro pelo nome.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.core.schema.reprodutibilidade import (  # noqa: E402
    Entrada,
    gravar_manifesto_etapa,
)

EXIGIDOS = ("config.json", "model.safetensors", "tokenizer.json",
            "tokenizer_config.json")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--de", type=Path, required=True,
                   help="diretório `-melhor` baixado da saída do notebook")
    p.add_argument("--para", type=Path, required=True)
    p.add_argument("--nota", required=True,
                   help="a medição que justifica a instalação, com o p")
    p.add_argument("--copiar", action="store_true",
                   help="copia os pesos; sem isto só (re)grava o manifesto sobre o "
                        "que já está em --para")
    a = p.parse_args()
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")

    if a.copiar:
        if not a.de.is_dir():
            raise SystemExit(f"{a.de} não é um diretório")
        a.para.mkdir(parents=True, exist_ok=True)
        for f in sorted(a.de.iterdir()):
            if f.is_file():
                shutil.copy2(f, a.para / f.name)

    faltando = [n for n in EXIGIDOS if not (a.para / n).exists()]
    if faltando:
        raise SystemExit(
            f"{a.para} está sem {faltando}. Um diretório de modelo incompleto falha "
            "no `from_pretrained` na hora da avaliação, longe daqui.")

    meta = json.loads((a.para / "phirank.json").read_text(encoding="utf-8"))
    melhor = json.loads((a.para / "melhor.json").read_text(encoding="utf-8"))
    base = meta.get("base") or melhor.get("base")
    if not base:
        raise SystemExit(
            f"{a.para}/phirank.json não declara a `base`. Sem ela o modelo entra no "
            "sistema sem se saber de que pré-treino ele partiu — que é justamente o "
            "que o T1c mediu.")

    # ⚠️ A divergência do `phirank.json`, registrada e não corrigida. Ver a docstring.
    desvio = meta.get("desvio_de_especificacao", "")
    divergencia = None
    if "MiniLM" in desvio and "MiniLM" not in base:
        divergencia = (
            f"`phirank.json` traz desvio_de_especificacao={desvio!r}, que afirma "
            f"MiniLM, mas o campo `base` diz {base!r}. A string era fixa em "
            "rerank.py até 2026-09-03 e valia para qualquer base. O arquivo NÃO foi "
            "reescrito: manda este manifesto.")

    gravar_manifesto_etapa(
        etapa="phirank_do_sistema",
        descricao=f"ΦRank do sistema, de {base}",
        raiz=a.para,
        entradas=[Entrada(caminho=str(a.de), nota="saída do notebook do Kaggle")],
        parametros={
            "script": "scripts/instalar_phirank.py",
            "base": base,
            "justificativa_medida": a.nota,
            "acerto_top1_no_grupo": melhor.get("acerto_top1"),
            "passo_do_checkpoint": melhor.get("passo"),
            "criterio_do_checkpoint": melhor.get("criterio"),
            "config_do_treino": meta.get("config"),
            "divergencia_de_proveniencia": divergencia,
            "o_que_a_medicao_NAO_diz": (
                "que este ΦRank generalize para outro recuperador, outro benchmark "
                "ou outro domínio. Foi medido sobre a fusão RRF deste ΦEmb, em "
                "2.000 consultas do nosso benchmark de citação, cuja noção de "
                "relevância é 'foi citado' — proxy enviesada e sem juízo humano."),
        },
        registros=melhor.get("passo", 0))

    tam = sum(f.stat().st_size for f in a.para.iterdir() if f.is_file())
    print("=" * 70)
    print(f"  {a.para}")
    print(f"  base   : {base}")
    print(f"  passo  : {melhor.get('passo')} · acerto@1 no grupo "
          f"{melhor.get('acerto_top1')}")
    print(f"  bytes  : {tam/1e6:.1f} MB")
    print(f"  nota   : {a.nota}")
    if divergencia:
        print(f"  ⚠️ {divergencia}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
