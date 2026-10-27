#!/usr/bin/env python3
"""Exige testes de treino executados, sem saltos, no JUnit da execução corrente.

O pytest pode sair com 0 mesmo quando módulos inteiros usam ``importorskip``.
O CI chama este conferidor depois do pytest, preservando também o status dele.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def conferir(relatorio: Path, alvos: list[str]) -> int:
    """Devolve a contagem aprovada; recusa relatório vazio, falhas e alvos ausentes."""
    raiz = ET.parse(relatorio).getroot()
    casos = list(raiz.iter("testcase"))
    if not casos:
        raise ValueError("o relatório não contém nenhum teste executado")
    problemas = [c for c in casos if any(c.find(t) is not None
                                       for t in ("skipped", "failure", "error"))]
    if problemas:
        nomes = ", ".join(f"{c.get('classname', '')}::{c.get('name', '')}"
                          for c in problemas[:10])
        raise ValueError(f"{len(problemas)} testes saltados ou reprovados: {nomes}")
    classes = {c.get("classname", "") for c in casos}
    for alvo in alvos:
        modulo = alvo.removesuffix(".py").replace("\\", "/").replace("/", ".")
        if not any(c == modulo or c.startswith(modulo + ".") for c in classes):
            raise ValueError(f"nenhum teste executado do alvo obrigatório: {alvo}")
    return len(casos)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("relatorio", type=Path)
    p.add_argument("alvos", nargs="+", help="arquivos de teste obrigatórios, relativos à raiz")
    args = p.parse_args()
    try:
        n = conferir(args.relatorio, args.alvos)
    except (OSError, ET.ParseError, ValueError) as e:
        print(f"treino CPU não validado: {e}", file=sys.stderr)
        return 1
    print(f"treino CPU: {n} testes aprovados, sem saltos; todos os alvos executados")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
