"""Ajudantes compartilhados pela suíte.

## `so_codigo` — e por que ele existe

Um teste que afirma a AUSÊNCIA de um padrão no código-fonte reprova, com a mesma
facilidade, o **comentário ou a docstring que explicam por que aquele padrão foi um
erro**. Isso já aconteceu **cinco vezes** neste repositório:

| onde | o teste buscava | e achava em |
|---|---|---|
| `test_kaggle_t1a.py` | `"InfoNCE"` | a nota sobre por que não usar `DataParallel` |
| `test_kaggle_t1c.py` | `"check=False"` | a docstring que conta que ele foi um erro |
| `test_kaggle_t1c.py` | `"phifm_src.zip"` | o comentário sobre o Kaggle descompactar `.zip` |
| `test_phirank_do_sistema.py` | a string de proveniência antiga | o comentário que a corrige |
| `test_config_encoder.py` | `"flash_attention_2"` | a docstring que explica que a T4 não a tem |

A correção errada, nas cinco, seria apagar a explicação — perdendo justamente a lição
que ela carrega.

⚠️ **A quinta aconteceu depois de este arquivo existir**, e por causa de uma ressalva
que estava escrita aqui: `ast.unparse` remove `#` mas **preserva docstrings**, porque
elas são expressões. Eu documentei isso e caí nisso. Então `so_codigo` passou a
remover as docstrings também — uma ressalva conhecida que continua mordendo não é
ressalva, é bug.

O que **ainda** sobrevive: strings comuns, no meio do código. Uma asserção de ausência
sobre um padrão que também apareça numa string literal precisa ser mais específica —
não menos.
"""

from __future__ import annotations

import ast
from pathlib import Path


class _SemDocstrings(ast.NodeTransformer):
    """Remove a docstring de módulo, classe, função e função assíncrona."""

    def _limpar(self, node):  # type: ignore[no-untyped-def]
        self.generic_visit(node)
        corpo = node.body
        if (corpo and isinstance(corpo[0], ast.Expr)
                and isinstance(corpo[0].value, ast.Constant)
                and isinstance(corpo[0].value.value, str)):
            # `pass` no lugar, para um corpo que ficaria vazio continuar válido.
            node.body = corpo[1:] or [ast.Pass()]
        return node

    visit_Module = _limpar
    visit_ClassDef = _limpar
    visit_FunctionDef = _limpar
    visit_AsyncFunctionDef = _limpar


def so_codigo(fonte: str, sem_docstrings: bool = True) -> str:
    """O fonte sem comentários e (por omissão) sem docstrings.

    Para asserções de AUSÊNCIA. Ver a tabela das cinco ocorrências na docstring do
    módulo, e por que o padrão é remover as docstrings também.
    """
    arvore = ast.parse(fonte)
    if sem_docstrings:
        arvore = _SemDocstrings().visit(arvore)
        ast.fix_missing_locations(arvore)
    return ast.unparse(arvore)


def so_codigo_de(caminho: Path, sem_docstrings: bool = True) -> str:
    return so_codigo(caminho.read_text(encoding="utf-8"), sem_docstrings)
