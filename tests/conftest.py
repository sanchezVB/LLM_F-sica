"""Ajudantes compartilhados pela suíte.

## `so_codigo` — e por que ele existe

Um teste que afirma a AUSÊNCIA de um padrão no código-fonte reprova, com a mesma
facilidade, o **comentário que explica por que aquele padrão foi um erro**. Isso já
aconteceu **quatro vezes** neste repositório:

| onde | o teste buscava | e achava em |
|---|---|---|
| `test_kaggle_t1a.py` | `"InfoNCE"` | a nota sobre por que não usar `DataParallel` |
| `test_kaggle_t1c.py` | `"check=False"` | a docstring que conta que ele foi um erro |
| `test_kaggle_t1c.py` | `"phifm_src.zip"` | o comentário sobre o Kaggle descompactar `.zip` |
| `test_phirank_do_sistema.py` | a string de proveniência antiga | o comentário que a corrige |

A correção errada, nas quatro, seria apagar o comentário — perdendo justamente a
lição que ele carrega. `ast.unparse` sobre a árvore devolve o código sem nenhum `#`,
o que resolve a classe inteira.

⚠️ **Strings e docstrings sobrevivem** ao `unparse`, porque são expressões. Para
asserções de ausência isso erra na direção segura: o teste reclama de mais, nunca de
menos. Quando o padrão proibido também aparece numa docstring, a asserção precisa ser
mais específica — não menos.
"""

from __future__ import annotations

import ast
from pathlib import Path


def so_codigo(fonte: str) -> str:
    """O fonte sem comentários, para asserções de AUSÊNCIA."""
    return ast.unparse(ast.parse(fonte))


def so_codigo_de(caminho: Path) -> str:
    return so_codigo(caminho.read_text(encoding="utf-8"))
