"""O contato de coleta não pode voltar a ser placeholder em silêncio.

Regressão de 2026-08-06: `base.py` documentava que `PHIFM_CONTACT` vinha do
`.env`, os coletores liam `os.environ`, e **nada carregava o arquivo**. A
coleta partia como `phifm-corpus@localhost` sem um aviso — anônima na prática,
que é o cenário contra o qual a docstring de `user_agent()` argumenta e o que o
princípio A5 do DOC-02 trata como risco existencial: o arXiv não tem
substituto, e quem bloqueia um IP anônimo não manda e-mail antes.

O teste que importa aqui é o de FALHA: garantir que o caminho silencioso está
fechado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.core.env import (  # noqa: E402
    FALLBACK,
    PLACEHOLDER,
    carregar_env,
    contato_obrigatorio,
    raiz_do_projeto,
)


def test_carrega_pares_do_arquivo(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# comentário\n"
        "PHIFM_CONTACT=alguem@exemplo.org\n"
        "\n"
        'COM_ASPAS="valor"\n'
        "SEM_IGUAL\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PHIFM_CONTACT", raising=False)
    lidos = carregar_env(env)
    assert lidos["PHIFM_CONTACT"] == "alguem@exemplo.org"
    assert lidos["COM_ASPAS"] == "valor", "aspas fazem parte do formato, não do valor"
    assert "SEM_IGUAL" not in lidos


def test_ambiente_vence_o_arquivo(tmp_path, monkeypatch):
    """Quem exportou na linha de comando quis aquilo; um arquivo em disco não
    desfaz isso sem avisar."""
    env = tmp_path / ".env"
    env.write_text("PHIFM_CONTACT=do-arquivo@exemplo.org\n", encoding="utf-8")
    monkeypatch.setenv("PHIFM_CONTACT", "do-ambiente@exemplo.org")
    carregar_env(env)
    import os
    assert os.environ["PHIFM_CONTACT"] == "do-ambiente@exemplo.org"


def test_arquivo_ausente_nao_explode(tmp_path):
    assert carregar_env(tmp_path / "nao-existe") == {}


@pytest.mark.parametrize("valor", ["", PLACEHOLDER, FALLBACK, "   "])
def test_placeholder_e_recusado(monkeypatch, valor):
    """O caminho silencioso tem de estar fechado."""
    monkeypatch.setenv("PHIFM_CONTACT", valor)
    monkeypatch.setattr("phifm.core.env.carregar_env", lambda *a, **k: {})
    with pytest.raises(RuntimeError, match="PHIFM_CONTACT"):
        contato_obrigatorio()


@pytest.mark.parametrize("valor", ["sem-arroba", "a@sem-ponto", "@exemplo.com"])
def test_endereco_invalido_e_recusado(monkeypatch, valor):
    """Endereço inválido equivale a nenhum: ninguém consegue avisar."""
    monkeypatch.setenv("PHIFM_CONTACT", valor)
    monkeypatch.setattr("phifm.core.env.carregar_env", lambda *a, **k: {})
    with pytest.raises(RuntimeError):
        contato_obrigatorio()


def test_email_real_passa(monkeypatch):
    monkeypatch.setenv("PHIFM_CONTACT", "alguem@exemplo.org")
    monkeypatch.setattr("phifm.core.env.carregar_env", lambda *a, **k: {})
    assert contato_obrigatorio() == "alguem@exemplo.org"


def test_user_agent_carrega_o_contato():
    """O contato precisa chegar ao cabeçalho — é lá que o arXiv o lê."""
    from phifm.corpus.acquire.base import user_agent

    ua = user_agent("alguem@exemplo.org")
    assert "alguem@exemplo.org" in ua
    assert "PhiFM-Corpus" in ua
    assert "github.com" in ua, "o repo identifica o projeto além do e-mail"


def test_env_deste_projeto_esta_configurado():
    """Guarda de ambiente: se o `.env` local regredir para o placeholder, este
    teste diz isso antes de uma coleta de três horas descobrir."""
    env = raiz_do_projeto() / ".env"
    if not env.exists():
        pytest.skip(".env não existe nesta cópia — ver SETUP.md §2")
    conteudo = env.read_text(encoding="utf-8")
    assert PLACEHOLDER not in conteudo, "o .env ainda tem o e-mail de exemplo"
