"""4xx definitivo não se repete; 429 e 5xx se repetem.

Regressão de 2026-08-10, encontrada na auditoria do S3b. Papers sem fonte no
arXiv devolvem 404, e o `PoliteSession` repetia seis vezes com recuo exponencial
— ~25 s desperdiçados por paper ausente, num laço de 200.

O desperdício é o menor dos problemas. O log enchia de "Falha de rede" para algo
que **não é falha de rede**, e sim ausência legítima do recurso. Confundir os
dois esconde problema de rede de verdade no meio do ruído.

A distinção que o teste fixa:

    404, 403, 400   definitivo   -> levanta na hora
    429             transitório  -> repete (é controle de fluxo)
    503, 500        transitório  -> repete
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.core.schema.manifest import RateLimit  # noqa: E402
from phifm.corpus.acquire.base import PoliteSession  # noqa: E402


class RespostaFalsa:
    def __init__(self, codigo: int):
        self.status_code = codigo
        self.headers = {}
        self.content = b""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


def sessao(monkeypatch, codigos: list[int]) -> tuple[PoliteSession, list]:
    """Sessão cujo `get` devolve `codigos` em ordem, contando as chamadas."""
    s = PoliteSession(RateLimit(requests_per_second=1000.0, max_retries=6,
                                backoff_base_s=1.0, backoff_max_s=1.0))
    chamadas = []

    def falso_get(url, params=None, timeout=None, headers=None):
        chamadas.append(url)
        i = min(len(chamadas) - 1, len(codigos) - 1)
        return RespostaFalsa(codigos[i])

    monkeypatch.setattr(s.session, "get", falso_get)
    monkeypatch.setattr("time.sleep", lambda _: None)
    return s, chamadas


@pytest.mark.parametrize("codigo", [400, 403, 404, 410, 422])
def test_4xx_definitivo_nao_repete(monkeypatch, codigo):
    s, chamadas = sessao(monkeypatch, [codigo])
    with pytest.raises(requests.HTTPError):
        s.get("https://exemplo/x")
    assert len(chamadas) == 1, f"{codigo} foi repetido {len(chamadas)} vezes"


def test_429_repete():
    """429 é controle de fluxo, não ausência — repetir é o comportamento certo."""
    # Sem monkeypatch de sleep aqui: o caminho do 429 usa `continue`, não exceção.
    s = PoliteSession(RateLimit(requests_per_second=1000.0, max_retries=3,
                                backoff_base_s=1.0, backoff_max_s=1.0))
    chamadas = []

    def falso_get(url, params=None, timeout=None, headers=None):
        chamadas.append(url)
        return RespostaFalsa(429)

    s.session.get = falso_get
    import time as _t
    orig, _t.sleep = _t.sleep, lambda _: None
    try:
        with pytest.raises(RuntimeError, match="Esgotadas"):
            s.get("https://exemplo/x")
    finally:
        _t.sleep = orig
    assert len(chamadas) == 3, "429 devia ter sido repetido até o limite"


def test_503_repete(monkeypatch):
    """503 com Retry-After é o mecanismo de controle de fluxo do OAI-PMH."""
    s, chamadas = sessao(monkeypatch, [503])
    with pytest.raises(RuntimeError, match="Esgotadas"):
        s.get("https://exemplo/x")
    assert len(chamadas) == 6


def test_500_repete(monkeypatch):
    """5xx é do servidor e pode passar — diferente de 4xx, que é do pedido."""
    s, chamadas = sessao(monkeypatch, [500])
    with pytest.raises((RuntimeError, requests.HTTPError)):
        s.get("https://exemplo/x")
    assert len(chamadas) > 1, "5xx não devia ser tratado como definitivo"


def test_404_depois_sucesso_nao_mascara(monkeypatch):
    """Um 404 levanta na primeira, mesmo que a próxima resposta fosse 200.

    Parece contraintuitivo mas é correto: se o recurso não existe, não há o que
    esperar. Repetir 'até dar certo' num 404 é o mesmo que negar a resposta.
    """
    s, chamadas = sessao(monkeypatch, [404, 200])
    with pytest.raises(requests.HTTPError):
        s.get("https://exemplo/x")
    assert len(chamadas) == 1


def test_200_passa_direto(monkeypatch):
    s, chamadas = sessao(monkeypatch, [200])
    r = s.get("https://exemplo/x")
    assert r.status_code == 200
    assert len(chamadas) == 1
    assert s.requests_made == 1
