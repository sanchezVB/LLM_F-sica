"""ΦFM — modelos de fundação de Física.

## O cache do HuggingFace vai para o HD, definido NO IMPORT deste pacote

O `.env` da raiz declara `HF_HOME=D:/LLMFísica/cache/huggingface`, e nada o carregava.
Nesta máquina isso não fazia estrago: `C:\\Users\\User\\.cache\\huggingface` é uma
JUNCTION para essa mesma pasta do HD. Aqui a variável deixa de depender da junction —
que não existe num clone novo nem em outra máquina.

⚠️ **Correção de 2026-09-17.** A primeira versão desta nota dizia "6,8 GB de modelos no
SSD, quase todos duplicatas do HD". Era falso: as "duplicatas" eram os MESMOS arquivos
vistos pelos dois caminhos (mesmo ID de arquivo). Foi descoberto antes de apagar o
resto — apagar "as cópias do SSD" teria apagado os únicos exemplares. Uma cópia de
1,2 GB do ModernBERT-base já tinha sido apagada pelo caminho do C:, e ela estava no HD;
o modelo segue carregando da cópia que veio de `.hf_cache`.

O `huggingface_hub` lê o caminho do cache quando é IMPORTADO, então a variável tem de
ser definida antes — e os scripts CHAMAM `cache_hf_no_hd()` antes do `transformers`
(`tests/regression/test_cache_hf_no_hd.py`).

`setdefault`, e não atribuição: uma variável já definida no ambiente manda. Sem `.env`
(no Kaggle, no Mac, num clone limpo) nada muda.
"""
from __future__ import annotations

import os
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[2]


def hf_home_do_env(caminho_env: Path) -> str | None:
    """O valor de `HF_HOME` no `.env`, ou `None`. Só esta chave: o resto não é daqui."""
    if not caminho_env.is_file():
        return None
    for linha in caminho_env.read_text(encoding="utf-8", errors="replace").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, _, valor = linha.partition("=")
        if chave.strip().removeprefix("export ").strip() == "HF_HOME":
            valor = valor.strip().strip("\"'")
            return valor or None
    return None


def cache_hf_no_hd() -> str | None:
    """Define `HF_HOME` a partir do `.env`, sem sobrescrever. Devolve o valor em vigor.

    ⚠️ Os scripts a CHAMAM antes de importar o `transformers`, e não só importam o
    pacote: o `ruff` ordena os imports em grupos — bibliotecas de fora antes das do
    projeto — e desfez em 2026-09-17 a ordem que dependia só do import. Uma chamada
    entre os imports quebra o bloco que o isort reordena.
    """
    valor = hf_home_do_env(_RAIZ / ".env")
    if valor:
        os.environ.setdefault("HF_HOME", valor)
    return os.environ.get("HF_HOME")


cache_hf_no_hd()
