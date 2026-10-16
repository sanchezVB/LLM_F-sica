"""O juiz independente (DOC-13 §9.6): o pacote, o cliente de conversa e a leitura."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from phifm.eval import assistente_bracos as b
from phifm.eval import juiz_independente as ji


def test_a_chave_e_do_texto_que_o_juiz_le_sem_marcas_e_sem_pontas():
    k = ji.chave("P?", "G.", "A resposta é 3 [1].")
    assert k == ji.chave("P?", "G.", "  A resposta é 3.\n") and len(k) == 24
    assert k != ji.chave("P?", "G.", "A resposta é 4 [1].")
    assert k != ji.chave("Outra?", "G.", "A resposta é 3 [1].")


def test_o_pacote_tem_cada_texto_uma_vez_e_nada_alem_do_que_o_juiz_le():
    grupos = {"teste_A2": [("P1?", "G1.", "sim [1]."), ("P2?", "G2.", "não [2].")],
              "teste_A4": [("P1?", "G1.", "sim [3]."), ("P2?", "G2.", "talvez.")],
              "selados": [("P2?", "G2.", "talvez.")]}
    pacote, contagem = ji.empacotar(grupos)
    assert [e["resposta"] for e in pacote] == ["sim.", "não.", "talvez."]
    assert all(set(e) == {"chave", "pergunta", "gabarito", "resposta"} for e in pacote)
    assert contagem == {"teste_A2": {"textos": 2, "novos_no_pacote": 2},
                        "teste_A4": {"textos": 2, "novos_no_pacote": 1},
                        "selados": {"textos": 1, "novos_no_pacote": 0}}
    assert pacote[0]["chave"] == ji.chave("P1?", "G1.", "sim [1].")


class ModeloQueJulga:
    def __init__(self):
        self.chamadas = []

    def gerar(self, sistema, usuario, **kw):
        self.chamadas.append((sistema, usuario, kw))
        v = "certo" if "sim" in usuario else "errado"
        return json.dumps({"nucleo": "x", "motivo": "m", "veredicto": v})


def test_julga_com_o_prompt_da_regra_e_retoma_sem_refazer(tmp_path):
    pacote, _ = ji.empacotar({"g": [("P1?", "G1.", "sim."), ("P2?", "G2.", "não."),
                                    ("P3?", "G3.", "")]})
    modelo, saida = ModeloQueJulga(), tmp_path / "v.jsonl"
    feitos = ji.julgar_pacote(modelo, pacote, saida, limite=1)
    assert len(feitos) == 1 and len(modelo.chamadas) == 1
    feitos = ji.julgar_pacote(modelo, pacote, saida)
    assert [feitos[e["chave"]]["veredicto"] for e in pacote] == ["certo", "errado", "absteve"]
    assert len(modelo.chamadas) == 2                     # o 1º não se repete; o vazio não vai
    sistema, usuario, kw = modelo.chamadas[0]
    assert sistema == b.SISTEMA_JUIZ and usuario == "Question: P1?\nGabarito: G1.\nAnswer:\nsim."
    assert kw["temperatura"] == 0.0 and kw["semente"] is not None
    assert ji.ler(saida) == feitos and ji.julgar_pacote(modelo, pacote, saida) == feitos
    assert len(modelo.chamadas) == 2


def test_o_cliente_de_conversa_manda_sistema_e_usuario_e_le_o_conteudo():
    recebido = {}

    class Servidor(BaseHTTPRequestHandler):
        def do_POST(self):
            recebido["caminho"] = self.path
            recebido["corpo"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            dados = json.dumps({"choices": [{"message": {"content": "  ok \n"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(dados)))
            self.end_headers()
            self.wfile.write(dados)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Servidor)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    modelo = ji.ModeloDeConversa(f"http://127.0.0.1:{srv.server_port}")
    assert modelo.gerar("S", "U", max_tokens=260, temperatura=0.0, semente=17) == "ok"
    srv.server_close()
    assert recebido["caminho"] == "/v1/chat/completions"
    c = recebido["corpo"]
    assert c["messages"] == [{"role": "system", "content": "S"},
                             {"role": "user", "content": "U"}]
    assert (c["max_tokens"], c["temperature"], c["seed"], c["stream"]) == (260, 0.0, 17, False)


def test_entre_os_juizes_conta_quem_deu_certo_sozinho():
    regra = ["certo"] * 6 + ["parcial", "errado", "absteve", "ilegivel"]
    outro = ["certo"] * 5 + ["parcial", "certo", "errado", "absteve", "errado"]
    r = ji.entre_juizes(regra, outro)
    assert (r["n"], r["so_a_regra_deu_certo"], r["so_o_independente_deu_certo"]) == (10, 1, 1)
    assert r["concordancia_certo"] == 0.8
    assert r["matriz_regra_x_independente"]["ilegivel"]["errado"] == 1
    assert r["matriz_regra_x_independente"]["certo"]["certo"] == 5


def test_a_conferencia_e_robusta_so_com_o_mesmo_desfecho_em_todo_eixo():
    regra = {"ganho_no_corpo": True, "nao_piora_o_primario": False}
    assert ji.conferir(True, regra, dict(regra))["leitura"] == "robusto"
    r = ji.conferir(True, regra, {"ganho_no_corpo": True, "nao_piora_o_primario": True})
    assert r["leitura"].startswith("FRÁGIL")
    assert r["eixos"]["nao_piora_o_primario"] == {
        "juiz_da_regra": False, "juiz_independente": True, "mesmo_desfecho": False}
    sem = ji.conferir(False, regra, dict(regra))
    assert sem["leitura"].startswith("SEM VALOR") and sem["eixos"] is None
