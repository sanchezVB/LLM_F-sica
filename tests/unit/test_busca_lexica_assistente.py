"""A busca fundida com o BM25 no assistente (DOC-13 §9.8/§9.9) e a apuração da regra."""
from __future__ import annotations

import polars as pl

from phifm.eval import assistente_bracos as b
from phifm.rag import passagens as ps
from phifm.rag.assistente import SISTEMA_HYDE, SISTEMA_RESPOSTA, Assistente
from phifm.retrieval.indice import Resultado


class BuscaDupla:
    """Densa devolve x1, x2; a fundida devolve x2, x3 — e anota o que recebeu."""

    def __init__(self):
        self.hibridas = []

    def buscar(self, consulta, k=10, excluir=None):
        return [Resultado(i + 1, 0.9, a, f"T {a}", 2020, "quant-ph", ["A"])
                for i, a in enumerate(("x1", "x2"))][:k]

    def buscar_hibrido(self, consulta, lexica, consulta_lexica, k=10, c=60):
        self.hibridas.append((consulta, lexica, consulta_lexica))
        return [Resultado(i + 1, 0.03, a, f"T {a}", 2020, "quant-ph", ["A"])
                for i, a in enumerate(("x2", "x3"))][:k]


class ModeloQueTraduz:
    def __init__(self):
        self.chamadas = []

    def gerar(self, sistema, usuario, **kw):
        self.chamadas.append(sistema)
        if sistema == ps.SISTEMA_TRADUCAO:
            return "what is the substrate temperature"
        if sistema == SISTEMA_HYDE:
            return "Hypothetical title. Abstract."
        return "É 650 K [1]."


def _spine(tmp_path):
    pl.DataFrame({"arxiv_id": ["0704.0001", "x1", "x2", "x3"], "title": ["t0", "t1", "t2", "t3"],
                  "abstract": ["r0", "r1", "r2", "r3"], "year": [2007, 2020, 2021, 2022]}
                 ).write_parquet(tmp_path / "spine.parquet")
    return tmp_path / "spine.parquet"


def test_sem_lexica_nada_muda_nem_traduz(tmp_path):
    modelo, busca = ModeloQueTraduz(), BuscaDupla()
    a = Assistente(busca, modelo, _spine(tmp_path), k=2, semente=7)
    r = a.responder("qual a temperatura do substrato?")
    assert [f.arxiv_id for f in r.fontes] == ["x1", "x2"] and busca.hibridas == []
    assert ps.SISTEMA_TRADUCAO not in modelo.chamadas


def test_com_lexica_a_busca_e_a_fundida_com_a_pergunta_em_ingles(tmp_path):
    modelo, busca = ModeloQueTraduz(), BuscaDupla()
    a = Assistente(busca, modelo, _spine(tmp_path), k=2, semente=7, lexica="L")
    r = a.responder("qual a temperatura do substrato?")
    assert [f.arxiv_id for f in r.fontes] == ["x2", "x3"]
    assert busca.hibridas == [("Hypothetical title. Abstract.", "L",
                               "what is the substrate temperature")]
    assert modelo.chamadas == [SISTEMA_HYDE, ps.SISTEMA_TRADUCAO, SISTEMA_RESPOSTA]


def test_com_lexica_e_passagens_a_traducao_e_uma_so(tmp_path):
    modelo, busca = ModeloQueTraduz(), BuscaDupla()
    passo = ps.Passagens(modelo, lambda a: None, semente=7)
    a = Assistente(busca, modelo, _spine(tmp_path), k=2, semente=7, lexica="L", passagens=passo)
    a.responder("qual a temperatura do substrato?", com_passagens=True)
    assert modelo.chamadas.count(ps.SISTEMA_TRADUCAO) == 1


def test_A5_reaproveita_A2_quando_as_fontes_batem_e_mede_a_traducao(tmp_path):
    modelo, busca = ModeloQueTraduz(), BuscaDupla()
    a = Assistente(busca, modelo, _spine(tmp_path), k=2, semente=7, lexica="L")
    itens = [{"estrato": "primario", "ordem": i, "arxiv_id": p, "pergunta": f"P{i}?"}
             for i, p in enumerate(("x2", "x1"))]
    a2 = {("primario", "x2"): b.RespostaDoBraco("primario", 0, "x2", "A2", "de A2 [1].",
                                                fontes=["x2", "x3"], posicao_p=1, consulta="h"),
          ("primario", "x1"): b.RespostaDoBraco("primario", 1, "x1", "A2", "de A2 [1].",
                                                fontes=["x1", "x2"], posicao_p=1, consulta="h")}
    regs = {r.arxiv_id: r for r in b.rodar_busca_lexica_sobre_a2(a, itens, a2,
                                                                 tmp_path / "a5.jsonl")}
    assert regs["x2"].igual_a_A and regs["x2"].texto == "de A2 [1]."
    assert not regs["x1"].igual_a_A and regs["x1"].posicao_p is None   # x1 saiu das fontes
    assert all(r.braco == "A5" and r.consulta_passagens == "what is the substrate temperature"
               for r in regs.values())
    assert modelo.chamadas.count(SISTEMA_RESPOSTA) == 1                 # só o que mudou
    assert all(c == "h" for c, _, _ in busca.hibridas)                  # a consulta de A′


def test_a_regra_da_9_9_adota_so_com_o_IC_acima_de_3_pontos_e_o_tempo():
    ids = [("primario", f"p{i:03d}") for i in range(500)]
    v = {"A2": {k: "certo" if i < 400 else "errado" for i, k in enumerate(ids)}}
    v["A5"] = dict(v["A2"])
    for k in ids[400:440]:                                   # +40 de 500: +0,08
        v["A5"][k] = "certo"
    rs = [b.RespostaDoBraco(e, i, a, br, "x", posicao_p=1, segundos_recuperacao=14.0)
          for i, (e, a) in enumerate(ids) for br in ("A2", "A5")]
    r = b.comparar_busca_lexica(v, rs, tempo_a_s=31.0)
    assert r["A5_menos_A2"] == 0.08 and r["decisao"] == "ADOTA"
    assert r["tempo"]["regra_s"] == 45.0
    assert b.comparar_busca_lexica(v, rs, tempo_a_s=50.0)["decisao"] == "FICA O SISTEMA ATUAL"
    for k in ids[420:440]:                                   # +20 de 500: o IC desce de 0,03
        v["A5"][k] = "errado"
    assert b.comparar_busca_lexica(v, rs, tempo_a_s=31.0)["decisao"] == "FICA O SISTEMA ATUAL"
