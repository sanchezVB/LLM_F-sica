"""UTF-8 na saída, chamado no IMPORT e não depois do `parse_args`.

## ⚠️ O defeito que isto conserta, medido em 2026-09-10

Seis scripts deste repositório reconfiguravam a saída para UTF-8 **depois** de
`parse_args()`. O console do Windows entrega cp1252, e `Φ` não existe em cp1252 —
então `--help`, que o argparse imprime e que sai **antes** daquela linha, morria com
`UnicodeEncodeError`.

Um script cujo `--help` quebra é um script que ninguém consegue descobrir. E o erro
não aparece em nenhuma execução normal: só quando alguém pede ajuda.

O projeto já conhecia a metade tardia deste problema — `publicar_kaggle.py` tem a
nota de que sem o `reconfigure` ele levanta *"DEPOIS de já ter escrito os arquivos —
o trabalho fica feito e a saída diz que falhou"*. O que faltava era notar que o
argparse imprime antes de tudo.

## Por que no import

`main()` não é o primeiro código a rodar: o argparse pode imprimir ajuda ou erro de
uso e sair sem nunca chamar nada nosso. O único ponto garantidamente anterior é o
import.
"""

from __future__ import annotations

import contextlib
import sys


def utf8() -> None:
    """Põe `stdout` e `stderr` em UTF-8, sem levantar se não der.

    `contextlib.suppress` e não `try/except/pass` por gosto do linter, e porque
    um fluxo que não aceita `reconfigure` (um pipe já embrulhado, um teste que
    captura) não é motivo para derrubar o programa — o pior que acontece é a
    saída sair com `?` no lugar de `Φ`.
    """
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")
