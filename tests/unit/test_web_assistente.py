"""A página com o assistente, com busca e assistente FALSOS, e o modelo sob demanda.

O que se tranca:

1. sem assistente, a página é a de antes: `/api/perguntar` não existe;
2. com assistente, a pergunta volta com texto, fontes numeradas, quais foram citadas e o
   que o portão tirou — e a busca do assistente passa pela trava da rota de busca;
3. as guardas contra outra origem: `Host` estranho → 403; corpo que não é JSON → 415;
4. o modelo sob demanda sobe na primeira chamada, cai sozinho depois de ocioso, e nunca
   derruba um servidor que ele não subiu.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.rag.assistente import Fonte, Resposta  # noqa: E402
from phifm.rag.llm import ModeloSobDemanda  # noqa: E402
from phifm.retrieval.indice import Resultado  # noqa: E402
from phifm.serving.web import _BuscaTravada, criar_servidor  # noqa: E402


class BuscaFalsa:
    vetores = np.zeros((3, 4))

    def buscar(self, consulta, k=10, excluir=None):
        return [Resultado(1, 0.9, "2001.00001", "Título", 2020, "quant-ph", ["A"])]


class AssistenteFalso:
    def __init__(self):
        self.busca = BuscaFalsa()
        self.modelo = None
        self.perguntas = []

    def responder(self, pergunta):
        self.perguntas.append(pergunta)
        self.busca.buscar("x")
        fontes = [Fonte(1, "2001.00001", "Um", 2020, "Resumo um.", 0.8),
                  Fonte(2, "2001.00002", "Dois", 2021, "Resumo dois.", float("nan"))]
        return Resposta(pergunta, "consulta", "É 42 [1].", fontes, [1], [7],
                        ["Uma frase longa sem fonte nenhuma para conferir aqui."])


@pytest.fixture
def servidor():
    criados = []

    def fazer(assistente=None):
        s = criar_servidor(BuscaFalsa(), porta=0, assistente=assistente, ocioso_min=10)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        criados.append(s)
        return f"http://127.0.0.1:{s.server_address[1]}"

    yield fazer
    for s in criados:
        s.shutdown()
        s.server_close()


def _post(url, corpo, tipo="application/json", host=None):
    req = urllib.request.Request(url, data=corpo, method="POST",
                                 headers={"Content-Type": tipo, **({"Host": host} if host else {})})
    with urllib.request.urlopen(req) as r:
        return r.status, json.loads(r.read())


def _abas_ocultas(base) -> bool:
    with urllib.request.urlopen(base + "/") as r:
        return 'id="abas" hidden' in r.read().decode("utf-8")


def test_as_abas_vem_no_html_so_com_assistente(servidor):
    assert _abas_ocultas(servidor()) and not _abas_ocultas(servidor(AssistenteFalso()))


def test_sem_assistente_a_pagina_e_so_a_busca(servidor):
    base = servidor()
    with urllib.request.urlopen(base + "/api/estado") as r:
        assert json.loads(r.read())["assistente"] is False
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base + "/api/perguntar", json.dumps({"pergunta": "oi?"}).encode())
    assert e.value.code == 404


def test_pergunta_volta_com_fontes_citadas_e_o_que_o_portao_tirou(servidor):
    assistente = AssistenteFalso()
    base = servidor(assistente)
    with urllib.request.urlopen(base + "/api/estado") as r:
        assert json.loads(r.read()) == {"assistente": True, "modelo_no_ar": False,
                                        "ocioso_min": 10}
    status, d = _post(base + "/api/perguntar", json.dumps({"pergunta": " Qual é? "}).encode())
    assert status == 200 and assistente.perguntas == ["Qual é?"]
    assert d["texto"] == "É 42 [1]." and d["removidas"] == [7]
    assert [(f["numero"], f["citada"]) for f in d["fontes"]] == [(1, True), (2, False)]
    assert d["fontes"][1]["escore"] is None          # NaN não vira JSON inválido
    assert d["fontes"][0]["link"] == "https://arxiv.org/abs/2001.00001"
    assert len(d["frases_sem_fonte"]) == 1
    assert isinstance(assistente.busca, _BuscaTravada)


def test_outra_origem_e_recusada(servidor):
    base = servidor(AssistenteFalso())
    corpo = json.dumps({"pergunta": "oi?"}).encode()
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base + "/api/perguntar", corpo, tipo="text/plain")
    assert e.value.code == 415
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base + "/api/perguntar", corpo, host="malicioso.example:80")
    assert e.value.code == 403
    req = urllib.request.Request(base + "/", headers={"Host": "rebinding.example"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 403


def test_pergunta_vazia_ou_json_ruim(servidor):
    base = servidor(AssistenteFalso())
    for corpo in (json.dumps({"pergunta": "  "}).encode(), b"{nao json", b"[1, 2]"):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base + "/api/perguntar", corpo)
        assert e.value.code == 400


class ModeloLocalFalso:
    def __init__(self, ja_no_ar=False):
        self.ligado, self.subiu, self.chamadas, self.paradas = ja_no_ar, False, 0, 0

    def no_ar(self):
        return self.ligado

    def iniciar(self, log=None):
        self.ligado = self.subiu = True

    def gerar(self, sistema, usuario, **kw):
        self.chamadas += 1
        return "ok"

    def parar(self):  # como o de verdade: só derruba o que ele mesmo subiu
        self.paradas += 1
        if self.subiu:
            self.ligado = self.subiu = False


def test_modelo_sobe_na_primeira_chamada_e_cai_depois_de_ocioso():
    m = ModeloLocalFalso()
    s = ModeloSobDemanda(m, ocioso_s=0.2, intervalo_s=0.05)
    assert not m.ligado
    assert s.gerar("s", "u") == "ok" and m.ligado
    time.sleep(0.5)
    assert not m.ligado
    assert s.gerar("s", "u") == "ok" and m.ligado     # sobe de novo
    s.parar()
    assert not m.ligado


def test_modelo_que_ja_estava_no_ar_nao_e_derrubado():
    m = ModeloLocalFalso(ja_no_ar=True)
    s = ModeloSobDemanda(m, ocioso_s=0.1, intervalo_s=0.05)
    s.gerar("s", "u")
    time.sleep(0.3)
    s.parar()
    assert m.ligado and not m.subiu


# ── a resposta em fluxo ─────────────────────────────────────────────────────


class AssistenteEmFluxo(AssistenteFalso):
    def responder_em_fluxo(self, pergunta):
        r = self.responder(pergunta)
        yield "fontes", (r.consulta, r.fontes)
        for pedaco in ("É 42 ", "[1]", " e ", "[9]."):   # [9] inventada: o fim a tira
            yield "pedaco", pedaco
        yield "fim", r


def _post_fluxo(url, pergunta):
    req = urllib.request.Request(url, method="POST",
                                 data=json.dumps({"pergunta": pergunta, "fluxo": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        assert r.headers["Content-Type"].startswith("application/x-ndjson")
        return [json.loads(linha) for linha in r.read().decode("utf-8").splitlines() if linha]


def test_fluxo_manda_fontes_pedacos_e_o_fim_que_passou_pelo_portao(servidor):
    base = servidor(AssistenteEmFluxo())
    eventos = _post_fluxo(base + "/api/perguntar", "Qual é?")
    tipos = [e["tipo"] for e in eventos]
    assert tipos == ["fontes", "pedaco", "pedaco", "pedaco", "pedaco", "fim"]
    assert all(not f["citada"] for f in eventos[0]["fontes"])   # ainda não se sabe
    assert "".join(e["texto"] for e in eventos if e["tipo"] == "pedaco") == "É 42 [1] e [9]."
    fim = eventos[-1]
    assert fim["texto"] == "É 42 [1]." and fim["removidas"] == [7]
    assert [f["citada"] for f in fim["fontes"]] == [True, False]


def test_sem_pedir_fluxo_a_resposta_vem_inteira(servidor):
    status, d = _post(servidor(AssistenteEmFluxo()) + "/api/perguntar",
                      json.dumps({"pergunta": "Qual é?"}).encode())
    assert status == 200 and d["texto"] == "É 42 [1]."


def test_modelo_sob_demanda_em_fluxo_sobe_e_libera_a_trava():
    class Falso(ModeloLocalFalso):
        def gerar_em_fluxo(self, sistema, usuario, **kw):
            yield from ("a", "b")

    m = Falso()
    s = ModeloSobDemanda(m, ocioso_s=60, intervalo_s=60)
    assert list(s.gerar_em_fluxo("s", "u")) == ["a", "b"] and m.ligado
    gen = s.gerar_em_fluxo("s", "u")
    next(gen)
    gen.close()                                   # o navegador desconectou no meio
    assert s.gerar("s", "u") == "ok"              # a trava foi solta
    s.parar()


def test_cliente_le_o_stream_do_llama_server():
    """O formato do `stream` do `/completion`: uma linha `data: {…}` por pedaço, e o
    último com `stop: true`."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from phifm.rag.llm import ModeloLocal

    class Falso(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            corpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert corpo["stream"] is True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for d in ({"content": "Olá", "stop": False}, {"content": ", mundo", "stop": False},
                      {"content": "", "stop": True}):
                self.wfile.write(f"data: {json.dumps(d)}\n\n".encode())

        def log_message(self, *a):
            return

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Falso)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        m = ModeloLocal(porta=srv.server_address[1])
        assert list(m.gerar_em_fluxo("s", "u")) == ["Olá", ", mundo"]
    finally:
        srv.shutdown()
        srv.server_close()
