"""O nome do argumento de dtype no `from_pretrained` MUDOU entre versões do `transformers`.

Sem torch no import: a escolha do nome é testável na suíte rápida.

## ⚠️ Por que isto existe (2026-09-16)

O conserto do fp16 de 2026-09-03/11 passa `dtype=torch.float32` ao `from_pretrained`
(ver `test_dtype_fp32.py`). É o nome do argumento no `transformers` NOVO — o 5.0.0 da
imagem do Kaggle, onde o T1f rodou com ele. O `transformers` 4.48.3 da `.venv-treino`
de então só conhecia `torch_dtype` (conferido na fonte de `PreTrainedModel.from_pretrained`).

Na 4.48 o `dtype` desconhecido segue para o construtor do modelo. O
`ModernBertModel` o recusa com `TypeError` — foi assim que apareceu, no ensaio local do
ajuste de embedding do ΦEnc. E onde não quebra, o conserto do fp16 simplesmente não
acontece localmente.

O corte em 4.56 é o da troca de nome no `transformers`; os dois lados dele usados neste
projeto estão conferidos (4.48.3 local e 5.0.0 no Kaggle).

⚠️ Desde 2026-09-23 a `.venv-treino` local também é 5.0.0 (a 4.48.3 ficou em
`.venv-treino-tf4`). A escolha pelo número da versão continua, porque é ela que deixa
as duas venvs rodarem o mesmo código — e a reserva existe para reproduzir medições antigas.
"""
from __future__ import annotations

from packaging.version import Version

VERSAO_DO_DTYPE = Version("4.56.0")


def nome_do_argumento_de_dtype(versao: str) -> str:
    """`"dtype"` a partir do 4.56; `"torch_dtype"` antes."""
    return "dtype" if Version(versao) >= VERSAO_DO_DTYPE else "torch_dtype"


def kwargs_fp32() -> dict:
    """`{<nome certo nesta versão>: torch.float32}`, para `from_pretrained(**...)`.

    ⚠️ fp32 EXPLÍCITO porque uma base fp16 com AMP dá `Attempting to unscale FP16
    gradients` no primeiro passo — ver `test_dtype_fp32.py`.
    """
    import torch
    import transformers

    return {nome_do_argumento_de_dtype(transformers.__version__): torch.float32}
