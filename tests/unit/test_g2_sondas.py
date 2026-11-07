"""As sondas de contaminação do G2 (S1 e S2): os cortes, o critério de 13 fichas, a retomada,
os agregados e o lacre, com fichas SINTÉTICAS e um servidor de mentira (nada de benchmark)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.eval import g2_sondas as s  # noqa: E402

# ── os cortes ────────────────────────────────────────────────────────────────


def test_corte_s1_exige_metade_de_32():
    prompt, verdadeira = s.corte_s1(list(range(70)))
    assert prompt == list(range(35)) and verdadeira == list(range(35, 70))
    assert s.corte_s1(list(range(63))) is None            # metade de 31 fichas
    assert s.corte_s1(list(range(64)))[0] == list(range(32))
    # a tabela inteira: vale a partir de 64 fichas de enunciado
    assert [n for n in range(0, 80) if s.corte_s1(list(range(n)))] == list(range(64, 80))


def test_corte_s2_quarenta_por_cento_e_resto_de_13():
    prefixo = [900, 901]
    prompt, verdadeira = s.corte_s2(prefixo, list(range(100)))
    assert prompt == prefixo + list(range(40)) and verdadeira == list(range(40, 100))
    assert s.corte_s2(prefixo, list(range(20))) is None    # 8 mostradas, resto 12
    assert s.corte_s2(prefixo, list(range(21))) is not None  # 8 mostradas, resto 13: vale
    assert s.corte_s2(prefixo, list(range(2))) is None     # 1 mostrada, resto 1
    assert s.corte_s2(prefixo, []) is None
    # a tabela de 0 a 30 fichas: vale a partir de 21; mostradas = round(0,4·n), sem empate
    assert [n for n in range(31) if s.corte_s2(prefixo, list(range(n)))] == list(range(21, 31))
    assert all(round(0.4 * n) == (4 * n + 5) // 10 for n in range(200_000))


# ── o critério ───────────────────────────────────────────────────────────────


def test_criterio_de_13_fichas_em_qualquer_posicao():
    verdadeira = list(range(100, 160))
    gerado = [1, 2, 3] + list(range(130, 143)) + [7]       # 13 fichas da verdadeira, deslocadas
    assert s.suspeito(gerado, verdadeira)
    assert not s.alinhado(gerado, verdadeira)              # não na mesma posição
    assert s.maior_trecho_comum(gerado, verdadeira) == 13
    quase = [1] + list(range(130, 142)) + [7]              # só 12 seguidas
    assert not s.suspeito(quase, verdadeira)
    assert s.alinhado(list(range(100, 113)), verdadeira)
    # cópia do prompt: o 13-grama casado também está no prompt (frase repetida no enunciado)
    assert s.copia_do_prompt(gerado, verdadeira, list(range(125, 150)))
    assert not s.copia_do_prompt(gerado, verdadeira, list(range(125, 140)))
    assert not s.copia_do_prompt(quase, verdadeira, list(range(100, 160)))   # nem suspeito


# ── o pedido e o servidor ────────────────────────────────────────────────────


def test_pedido_guloso_sem_cache_e_o_codigo_no_hash():
    p = s.pedido({"a.py": "x" * 64})
    assert (p["temperature"], p["top_p"], p["top_k"], p["min_p"], p["repeat_penalty"]) == (0.0, 1.0, 0, 0.0, 1.0)
    assert p["cache_prompt"] is False
    assert p["sondas"]["codigo"] == {"a.py": "x" * 64} and p["sondas"]["n_fichas"] == 13
    assert "codigo" not in s.pedido()["sondas"]
    assert s.parametros()["S1"] == {"gera": 48, "metade_minima": 32}
    assert s.parametros()["S2"]["gera"] == 64 and s.parametros()["S2"]["fracao_da_solucao"] == 0.4


def test_gerar_manda_fichas_e_tira_a_chave_das_sondas(monkeypatch):
    visto = {}

    def post(url, rota, corpo, tempo_max_s=600):
        visto.update(url=url, rota=rota, corpo=corpo)
        return {"tokens": [5, 6, 7]}

    monkeypatch.setattr(s, "_post", post)
    assert s.gerar("http://x", [1, 2], 48, s.pedido({"a.py": "h"})) == [5, 6, 7]
    c = visto["corpo"]
    assert visto["rota"] == "/completion" and "sondas" not in c
    assert (c["prompt"], c["n_predict"], c["stream"], c["return_tokens"], c["cache_prompt"]) == ([1, 2], 48, False, True, False)


def test_sondar_com_servidor_de_mentira(monkeypatch):
    # fichas = códigos dos caracteres; a "geração" devolve o que o teste mandar
    monkeypatch.setattr(s, "fichas", lambda url, texto: [ord(c) for c in texto])
    continuacoes = {}
    monkeypatch.setattr(s, "gerar", lambda url, prompt, n, pedido: continuacoes["v"][:n])
    enunciado = "a" * 40 + "b" * 40
    continuacoes["v"] = [ord("b")] * 48
    r = s.sondar("x", enunciado, None, "S1", {})
    assert r["aplicavel"] and r["suspeito"] and r["n_prompt"] == 40 and r["n_gerado"] == 48
    assert r["copia_do_prompt"] is False and r["maior_trecho"] == 40
    assert s.sondar("x", enunciado, None, "S2", {}) == {"aplicavel": False, "motivo": "sem_solucao"}
    assert s.sondar("x", enunciado, "  ", "S2", {})["motivo"] == "sem_solucao"
    assert s.sondar("x", enunciado, ["lista"], "S2", {})["motivo"] == "sem_solucao"   # não é texto
    assert s.sondar("x", "a" * 63, None, "S1", {}) == {"aplicavel": False, "motivo": "metade_curta"}
    assert s.sondar("x", enunciado, "c" * 20, "S2", {}) == {"aplicavel": False, "motivo": "resto_curto"}
    solucao = "c" * 30 + "d" * 50
    continuacoes["v"] = [ord("z")] * 64
    r2 = s.sondar("x", enunciado, solucao, "S2", {})
    assert r2["aplicavel"] and not r2["suspeito"] and r2["n_verdadeira"] == 48
    assert r2["n_prompt"] == len(enunciado) + len(s.SEPARADOR_S2) + 32
    # a S2 que reproduz o resto da solução é suspeita, alinhada e com trecho longo
    continuacoes["v"] = [ord("d")] * 64
    r3 = s.sondar("x", enunciado, solucao, "S2", {})
    assert r3["suspeito"] and r3["alinhado"] and r3["maior_trecho"] == 48
    # prompt + geradas acima do contexto do slot: não se aplica
    teto = r2["n_prompt"] + 63
    assert s.sondar("x", enunciado, solucao, "S2", {}, teto=teto)["motivo"] == "contexto"
    assert s.sondar("x", enunciado, solucao, "S2", {}, teto=teto + 1)["aplicavel"]


# ── a retomada ───────────────────────────────────────────────────────────────


def test_aparar_tira_so_a_cauda_da_queda(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_bytes(b'{"a": 1}\n{"b": 2}\n' + b"\x00" * 50)
    assert s.aparar(p) == 50 and p.read_bytes() == b'{"a": 1}\n{"b": 2}\n'
    p.write_bytes(b'{"a": 1}\n{"b": 2')                    # linha cortada
    assert s.aparar(p) == 7 and p.read_bytes() == b'{"a": 1}\n'
    p.write_bytes(b'{"a": 1}\n{"b": 2}')                   # completa, só perdeu o "\n"
    assert s.aparar(p) == 0 and p.read_bytes() == b'{"a": 1}\n{"b": 2}\n'
    assert s.aparar(p) == 0 and s.aparar(tmp_path / "nao_existe.jsonl") == 0


def _ambiente_de_mentira(monkeypatch, tmp_path, respostas):
    """O medir_g2_sondas com servidor, hash, sonda e livro de mentira."""
    import medir_g2_sondas as m
    from phifm.eval import g2, g2_geracao

    aval = tmp_path / "aval"
    (aval / "g2_saidas").mkdir(parents=True)
    (aval / "g2_kl_1.7B.json").write_text(json.dumps({"gguf": {"q8_0": {"sha256": "q"}}}),
                                         encoding="utf-8")
    monkeypatch.setattr(m, "AVAL", aval)
    monkeypatch.setattr(m, "SAIDAS", aval / "g2_saidas")

    class Servidor:
        def __init__(self, gguf, slots, log):
            assert slots == 1
            self.url = "http://x"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def no_ar(self):
            return True

    livro = []
    monkeypatch.setattr(g2_geracao, "ServidorG2", Servidor)
    monkeypatch.setattr(g2_geracao, "hash_da_configuracao", lambda srv, gguf, gpu, pedido: "h1")
    monkeypatch.setattr(s, "contexto_do_slot", lambda url: 32768)
    monkeypatch.setattr(g2, "exigir_aceite_e_congelamento", lambda: "ok")
    monkeypatch.setattr(g2, "medida_aberta", lambda *a, **k: bool(livro))
    monkeypatch.setattr(g2, "registrar_acesso", lambda *a, **k: livro.append((a, k)))
    chamadas = []

    def sondar(url, enunciado, solucao, sonda, pedido, teto=None):
        chamadas.append((enunciado, sonda))
        r = respostas.get((enunciado, sonda), {"aplicavel": True, "suspeito": False,
                                               "alinhado": False, "maior_trecho": 3})
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(s, "sondar", sondar)
    itens = [SimpleNamespace(bench="ugphysics", id="u1", enunciado="u1", solucao="x"),
             SimpleNamespace(bench="phybench", id="p1", enunciado="p1", solucao="y")]
    return m, itens, livro, chamadas


def test_rodar_retoma_so_o_que_falta_e_o_lacre(monkeypatch, tmp_path):
    respostas = {("u1", "S2"): OSError("caiu"),
                 ("p1", "S1"): {"aplicavel": True, "suspeito": True, "alinhado": True,
                                "maior_trecho": 40}}
    m, itens, livro, chamadas = _ambiente_de_mentira(monkeypatch, tmp_path, respostas)
    saida = m.SAIDAS / "sondas_teste_1.7B_base.jsonl"
    # uma linha de outro hash e uma cauda de queda: a 1ª é ignorada, a 2ª aparada
    saida.write_bytes(b'{"bench": "ugphysics", "item_id": "u1", "sonda": "S1", "aplicavel": true, '
                      b'"suspeito": true, "alinhado": true, "maior_trecho": 50, "hash_config": "h0"}\n'
                      + b"\x00" * 9)
    r = m.rodar("teste", "1.7B", "base", itens)
    assert len(chamadas) == 4 and r["faltam"] == 1 and r["registros"] == 3   # a S2 do u1 falhou
    assert livro[0][0][2] == "medida" and livro[0][1] == {"nota": "sondas"}
    # o lacre: o agregado versionado não traz ids; o arquivo fora do git tem os bytes do hash
    agg = json.loads((m.AVAL / "g2_sondas_teste_1.7B_base.json").read_text(encoding="utf-8"))
    assert "listas" not in agg and "u1" not in json.dumps(agg) and "p1" not in json.dumps(agg)
    lacrado = (m.SAIDAS / "sondas_teste_1.7B_base_listas.json").read_bytes()
    assert hashlib.sha256(lacrado).hexdigest() == agg["sha256_listas"]
    assert json.loads(lacrado)["suspeitos"] == {"phybench": {"S1": ["p1"]}}
    # a retomada: só a sonda que falhou volta; o livro ganha uma 'retomada'
    respostas.pop(("u1", "S2"))
    chamadas.clear()
    r = m.rodar("teste", "1.7B", "base", itens)
    assert chamadas == [("u1", "S2")] and r["faltam"] == 0 and r["registros"] == 4
    assert livro[1][0][2] == "retomada"
    # a linha de erro e as execuções ficam contadas (as linhas do hash, inclusive a de erro)
    assert r["linhas_de_erro"] == 1 and sum(r["execucoes"].values()) == 5
    erro = next(d for d in map(json.loads, saida.read_text(encoding="utf-8").splitlines())
                if "erro" in d)
    assert erro["erro"] == "OSError" and erro["ordem"] == 2 and "execucao" in erro


def test_servidor_morto_para_a_rodada_depois_de_3_erros(monkeypatch, tmp_path):
    respostas = {(e, s): ConnectionRefusedError("recusada") for e in ("u1", "p1") for s in ("S1", "S2")}
    m, itens, _, chamadas = _ambiente_de_mentira(monkeypatch, tmp_path, respostas)
    r = m.rodar("teste", "1.7B", "base", itens)
    assert len(chamadas) == 3 and r["faltam"] == 4 and r["linhas_de_erro"] == 3


def test_dev_leva_as_listas_e_nao_toca_o_livro(monkeypatch, tmp_path):
    m, itens, livro, _ = _ambiente_de_mentira(monkeypatch, tmp_path, {})
    r = m.rodar("dev", "1.7B", "base", itens)
    assert not livro and r["faltam"] == 0 and "listas" in r
    assert not (m.SAIDAS / "sondas_dev_1.7B_base_listas.json").exists()


def test_braco_so_base_ate_a_emenda_do_phigen(monkeypatch):
    import medir_g2_sondas as m

    monkeypatch.setattr(sys, "argv", ["medir_g2_sondas.py", "teste", "--braco", "phigen"])
    with pytest.raises(SystemExit):
        m.main()


# ── os agregados ─────────────────────────────────────────────────────────────


def test_agregados_e_lista_lacrada():
    def reg(b, i, sonda, susp, al=False, copia=False, trecho=13):
        return {"bench": b, "item_id": i, "sonda": sonda, "aplicavel": True, "suspeito": susp,
                "alinhado": al, "copia_do_prompt": copia, "maior_trecho": trecho if susp else 5}

    def nao(b, i, sonda, motivo):
        return {"bench": b, "item_id": i, "sonda": sonda, "aplicavel": False, "motivo": motivo}

    regs = [reg("ugphysics", "u1", "S1", True, al=True, trecho=40),
            reg("ugphysics", "u1", "S2", True, copia=True),
            reg("ugphysics", "u2", "S1", False),
            nao("ugphysics", "u2", "S2", "resto_curto"),
            nao("ugphysics", "u3", "S1", "metade_curta"),
            nao("ugphysics", "u3", "S2", "sem_solucao"),
            reg("phybench", "p1", "S1", False),
            reg("phybench", "p1", "S2", True)]
    r = s.agregar(regs)
    ug = r["por_benchmark"]["ugphysics"]
    assert ug["S1"] == {"itens": 3, "aplicaveis": 2, "suspeitos": 1, "alinhados": 1,
                        "so_copia_do_prompt": 0, "trecho_longo": 1,
                        "nao_aplicaveis": {"metade_curta": 1}, "fracao": 0.5}
    assert ug["S2"]["aplicaveis"] == 1 and ug["S2"]["so_copia_do_prompt"] == 1
    assert ug["S2"]["nao_aplicaveis"] == {"resto_curto": 1, "sem_solucao": 1}
    assert ug["uniao"] == {"itens": 3, "com_sonda_aplicavel": 2, "sem_sonda_aplicavel": 1,
                           "suspeitos": 1, "fracao": 0.3333, "fracao_dos_aplicaveis": 0.5,
                           "trecho_longo": 1}
    assert r["fracao_no_nulo"] == 1.0 and r["fracao_no_nulo_por_sonda"] == {"S1": 0.0, "S2": 1.0}
    assert r["listas"]["suspeitos"] == {"phybench": {"S2": ["p1"]},
                                        "ugphysics": {"S1": ["u1"], "S2": ["u1"]}}
    assert r["listas"]["trecho_longo"] == {"ugphysics": {"S1": ["u1"]}}
    assert r["total_suspeitos"] == 2
    # o lacre: o hash é o dos bytes canônicos e não depende da ordem dos registros
    assert r["sha256_listas"] == hashlib.sha256(s.canonico(r["listas"])).hexdigest()
    assert s.agregar(regs[::-1])["sha256_listas"] == r["sha256_listas"]
