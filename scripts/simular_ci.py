#!/usr/bin/env python3
"""Roda o que o CI roda, sobre SÓ os arquivos versionados. Antes de empurrar.

    .venv\\Scripts\\python.exe scripts\\simular_ci.py

## Por que este script existe

O CI deste projeto ficou vermelho em **todas as 80 execuções** desde que foi criado
(2026-08-09) até 2026-08-25. Ficou verde em 19687c8, voltou a vermelho no commit
seguinte, e eu empurrei cinco commits sem olhar — o usuário é que perguntou "e porque
todos os run do github continuam dando failed, ta certo isso?".

A causa da segunda vez: um teste meu lia `models/isphysics-clf/model.pkl`, que tem
20 MB e é **gitignored**. Na minha máquina o arquivo existe e o teste passa. No CI o
arquivo não existe e o teste falha. E o commit que introduziu a falha foi justamente
o que documentava a lição de não fazer isso.

`pytest` local não tem como pegar essa classe de erro, porque ele vê o disco inteiro.
Este script vê o que o CI vê: um `git archive` do commit, que por construção contém
apenas arquivos rastreados.

## O que ele NÃO iguala, e é honesto dizer

A versão do Python. O CI usa 3.12 (o `requirements.lock` fixa `numpy==2.5.1`, que
exige >= 3.12); esta venv pode ser outra. O script AVISA quando difere em vez de
fingir paridade — ele testa visibilidade de arquivo, não compatibilidade de versão.

Também não reinstala as dependências: usa as desta venv. Um `requirements.lock`
quebrado passa aqui e falha lá, e para isso não há atalho local barato.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Espelha `.github/workflows/ci.yml`, na mesma ordem. Cada passo é
# (nome, argumentos-depois-do-python) — sempre `-m`, para usar o interpretador
# desta venv em vez de depender de um executável no PATH.
PASSOS: tuple[tuple[str, list[str]], ...] = (
    ("ruff", ["-m", "ruff", "check", "src/", "tests/", "scripts/"]),
    ("fronteiras de módulo",
     ["-m", "importlinter.cli", "lint-imports", "--config", "pyproject.toml"]),
    ("testes", ["-m", "pytest", "tests/", "-q"]),
    ("suíte golden (bloqueante)",
     ["-m", "pytest", "tests/golden/", "-q", "--tb=short"]),
    ("cobertura de verify/",
     ["-m", "pytest", "tests/", "-q", "--cov=src/phifm/verify",
      "--cov-report=term-missing", "--cov-fail-under=80"]),
)

PY_DO_CI = (3, 12)


def _exportar(ref: str, destino: Path) -> int:
    """Extrai em `destino` só os arquivos que `ref` rastreia.

    ⚠️ `git archive`, não `shutil.copytree` com uma lista de exclusão. Uma lista de
    exclusão escrita à mão erra exatamente onde importa: ela não sabe o que o
    `.gitignore` ignora hoje, e o dia em que alguém acrescentar uma linha lá o
    simulador deixa de simular sem avisar. O `git archive` pergunta ao git.
    """
    tar = destino.parent / "arquivo.tar"
    subprocess.run(["git", "archive", "--format=tar", "-o", str(tar), ref],
                   cwd=RAIZ, check=True)
    with tarfile.open(tar) as t:
        membros = t.getmembers()
        t.extractall(destino, filter="data")
    tar.unlink()
    return sum(1 for m in membros if m.isfile())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", default="HEAD",
                   help="commit a simular. O padrão é HEAD, que é o que o push "
                        "manda; use `git stash` antes se quiser testar o índice")
    p.add_argument("--parar-no-primeiro", action="store_true",
                   help="como o CI, que aborta no primeiro passo vermelho. Sem "
                        "isto roda todos, que é mais útil antes de empurrar")
    a = p.parse_args()

    # O console do Windows entrega cp1252 e este script imprime ✅ e ⚠️.
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")

    sha = subprocess.run(["git", "rev-parse", "--short", a.ref], cwd=RAIZ,
                         capture_output=True, text=True, check=True).stdout.strip()
    sujo = subprocess.run(["git", "status", "--porcelain"], cwd=RAIZ,
                          capture_output=True, text=True, check=True).stdout.strip()

    print("=" * 74)
    print(f"  simulando o CI sobre {a.ref} = {sha}, só com arquivos versionados")
    if sujo:
        print(f"  ⚠️ a árvore tem {len(sujo.splitlines())} mudanças NÃO commitadas, e "
              "elas não entram\n     nesta simulação — o CI também não as veria")
    minha = sys.version_info[:2]
    if minha != PY_DO_CI:
        print(f"  ⚠️ Python {minha[0]}.{minha[1]} aqui, {PY_DO_CI[0]}.{PY_DO_CI[1]} "
              "no CI. Este script testa visibilidade de\n     ARQUIVO, não "
              "compatibilidade de versão — uma falha só de versão passa aqui")
    print("=" * 74)

    tmp = Path(tempfile.mkdtemp(prefix="simular_ci_"))
    try:
        arvore = tmp / "arvore"
        arvore.mkdir()
        n = _exportar(a.ref, arvore)
        print(f"  {n} arquivos versionados exportados\n")

        # `PYTHONPATH=src`, como todo passo do workflow: o lock instala as
        # dependências, não o próprio pacote.
        ambiente = {**os.environ, "PYTHONPATH": "src", "PYTHONUTF8": "1"}
        resultados: list[tuple[str, int]] = []
        for nome, args in PASSOS:
            print(f"── {nome} " + "─" * max(4, 68 - len(nome)))
            r = subprocess.run([sys.executable, *args], cwd=arvore, env=ambiente)
            resultados.append((nome, r.returncode))
            print(f"   {'✅' if r.returncode == 0 else '❌'} {nome} "
                  f"(saída {r.returncode})\n")
            if r.returncode and a.parar_no_primeiro:
                break
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    vermelhos = [n for n, c in resultados if c]
    print("=" * 74)
    for nome, codigo in resultados:
        print(f"  {'✅' if codigo == 0 else '❌'} {nome}")
    if vermelhos:
        print()
        print(f"  {len(vermelhos)} passo(s) que o CI reprovaria: "
              f"{', '.join(vermelhos)}")
        print()
        print("  Se um teste passa com `pytest` direto e falha aqui, ele depende de")
        print("  arquivo NÃO versionado — `models/`, `data/processed/`, um cache. A")
        print("  correção é o teste parar de depender do artefato, não o teste sair.")
        print("=" * 74)
        return 1
    print(f"\n  o CI passaria em {sha}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
