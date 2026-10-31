#!/usr/bin/env python3
"""Traz para cá o estado de uma sessão da 8B no Kaggle (kaggle/g2_8b_medida.py).

    .venv\\Scripts\\python.exe scripts\\trazer_g2_8b.py data\\processed\\kaggle_g2_8b_retorno\\estado

A pasta é a `estado/` do Output da sessão (ESTADO.json e o zip do estado). Confere o
SHA-256 do zip e de cada arquivo (`g2_kaggle.ler_estado`) e então:
- as gerações e os logs vão para `g2_saidas/` e os agregados da 8B para
  `data/processed/avaliacao/` (só os arquivos da 8B: nada daqui é sobrescrito);
- o pacote do EvalPlus gerado lá vai para `data/processed/kaggle_g2_evalplus_8B/` (o daqui
  é o da 1,7B);
- o livro de acessos daqui ganha as linhas da 8B que o do Kaggle tem e o daqui não
  (`g2_kaggle.juntar_livros`): toda linha do Kaggle que não é da 8B tem de existir aqui.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_kaggle  # noqa: E402

EVALPLUS_KAGGLE = "data/processed/kaggle_g2_evalplus/"
EVALPLUS_AQUI = "data/processed/kaggle_g2_evalplus_8B/"


def destino_de(rel: str) -> str:
    return EVALPLUS_AQUI + rel[len(EVALPLUS_KAGGLE):] if rel.startswith(EVALPLUS_KAGGLE) else rel


def trazer(origem: Path, raiz: Path = RAIZ) -> dict:
    manifesto, conteudo = g2_kaggle.ler_estado(origem)
    livro_aqui = raiz / g2_kaggle.LIVRO
    antes = livro_aqui.read_bytes() if livro_aqui.exists() else b""
    junto = g2_kaggle.juntar_livros(antes, conteudo.pop(g2_kaggle.LIVRO))
    escritos = []
    for rel, dados in sorted(conteudo.items()):
        alvo = raiz / destino_de(rel)
        if "8B" not in alvo.name and not rel.startswith(EVALPLUS_KAGGLE) \
                and not alvo.name.startswith("etapa_"):
            raise SystemExit(f"o estado traz um arquivo que não é da 8B: {rel}")
        alvo.parent.mkdir(parents=True, exist_ok=True)
        alvo.write_bytes(dados)
        if alvo.suffix == ".jsonl":
            g2_kaggle.aparar_jsonl(alvo)   # o zip pode ter sido feito com a medida no ar
        escritos.append(destino_de(rel))
    livro_aqui.write_bytes(junto)
    novas = len(g2_kaggle.registros(junto)) - len(g2_kaggle.registros(antes))
    return {"sessao": manifesto["sessao"], "arquivos": escritos, "linhas_novas_no_livro": novas}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("origem", type=Path)
    r = trazer(p.parse_args().origem)
    print(f"sessão {r['sessao']}: {len(r['arquivos'])} arquivos; {r['linhas_novas_no_livro']} "
          "linhas novas no livro de acessos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
