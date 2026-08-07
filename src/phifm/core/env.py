"""Carregamento do `.env` e o contato obrigatório de coleta.

**Por que este módulo existe.** O `base.py` documentava
``CONTACT = "phifm-corpus@localhost"  # sobrescrito por PHIFM_CONTACT (.env)``
e os scripts de coleta liam `os.environ`, mas **nada carregava o `.env`** e
`python-dotenv` não está no lock. O resultado era pior que um erro: a coleta
partia identificada como ``phifm-corpus@localhost`` — anônima na prática — sem
nenhum aviso. Exatamente o cenário contra o qual a docstring de
`user_agent()` argumenta.

Encontrado em 2026-08-06, antes da primeira coleta nesta máquina.

Sem dependência nova: o formato do `.env` aqui é `CHAVE=valor` com `#` de
comentário, e isso são vinte linhas de leitura.
"""

from __future__ import annotations

import os
from pathlib import Path

PLACEHOLDER = "seu-email@exemplo.com"
FALLBACK = "phifm-corpus@localhost"


def raiz_do_projeto() -> Path:
    return Path(__file__).resolve().parents[3]


def carregar_env(caminho: Path | None = None, sobrescrever: bool = False) -> dict[str, str]:
    """Lê o `.env` para `os.environ`. Devolve o que foi lido.

    Variável já presente no ambiente **vence** o arquivo por padrão: quem
    exportou na linha de comando quis aquilo, e um arquivo em disco não deve
    desfazer isso silenciosamente.
    """
    caminho = caminho or raiz_do_projeto() / ".env"
    lidos: dict[str, str] = {}
    if not caminho.exists():
        return lidos

    for linha in caminho.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        chave, _, valor = linha.partition("=")
        chave, valor = chave.strip(), valor.strip().strip("'\"")
        if not chave:
            continue
        lidos[chave] = valor
        if sobrescrever or chave not in os.environ:
            os.environ[chave] = valor
    return lidos


def contato_obrigatorio() -> str:
    """Contato de coleta, ou `RuntimeError` explicando o que falta.

    Falha **antes** da primeira requisição, não durante. Princípio A5 do
    DOC-02: uma fonte que nos bloqueie está perdida para sempre, e o arXiv não
    tem substituto. Coletar 18 mil vezes sob identificação de placeholder é
    apostar o Sprint S1 inteiro para economizar a edição de uma linha.
    """
    carregar_env()
    contato = (os.environ.get("PHIFM_CONTACT") or "").strip()

    if not contato or contato in (PLACEHOLDER, FALLBACK):
        raise RuntimeError(
            f"PHIFM_CONTACT não configurado (valor: {contato or 'vazio'!r}).\n"
            f"Editar {raiz_do_projeto() / '.env'} com um e-mail real.\n"
            "O arXiv exige contato identificável (DOC-02 §8.2): é o que permite "
            "a eles pedirem redução de taxa em vez de simplesmente bloquear o IP."
        )
    local, _, dominio = contato.partition("@")
    if not local or "." not in dominio or dominio.startswith(".") or dominio.endswith("."):
        raise RuntimeError(
            f"PHIFM_CONTACT não parece um e-mail: {contato!r}. "
            "Um endereço inválido é equivalente a nenhum — ninguém consegue avisar."
        )
    return contato
