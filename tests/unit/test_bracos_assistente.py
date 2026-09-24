"""Os braços, o juiz e a regra da medida do assistente (DOC-13 §9.1), com modelo e busca
FALSOS. O que se tranca:

1. B troca a ÚLTIMA fonte por P, na posição sorteada, e é A quando a busca já trouxe P;
2. C não recebe fonte nenhuma;
3. o juiz não vê as marcas [n] (ele não pode saber que houve fontes);
4. a regra: R conta `juiz_errou` como acerto e tira `item_invalido` do denominador; os
   três desfechos de cada eixo; I2 e I3 anulam a decisão.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import assistente_bracos as b  # noqa: E402
from phifm.rag.assistente import SISTEMA_HYDE, SISTEMA_RESPOSTA, Assistente, Fonte  # noqa: E402
from phifm.retrieval.indice import Resultado  # noqa: E402


class ModeloFalso:
    def __init__(self):
        self.chamadas = []

    def gerar(self, sistema, usuario, **kw):
        self.chamadas.append((sistema, usuario, kw))
        if sistema == SISTEMA_HYDE:
            return "Hypothetical title. Hypothetical abstract."
        if sistema == SISTEMA_RESPOSTA:
            return "A resposta é 42 [1]. Outra coisa [9]."
        if sistema == b.SISTEMA_MEMORIA:
            return "De memória, é 41."
        return '{"motivo": "ok", "veredicto": "certo"}'


class BuscaFalsa:
    def __init__(self, ids):
        self.ids = ids

    def buscar(self, consulta, k=10, excluir=None):
        return [Resultado(i + 1, 0.9 - i / 10, a, f"Título {a}", 2020, "quant-ph", ["A"])
                for i, a in enumerate(self.ids[:k])]


def _assistente(tmp_path, ids_busca):
    ids = [f"2001.{i:05d}" for i in range(10)]
    pl.DataFrame({"arxiv_id": ids, "title": [f"Título {a}" for a in ids],
                  "abstract": [f"Resumo de {a}." for a in ids],
                  "year": [2020] * 10}).write_parquet(tmp_path / "spine.parquet")
    modelo = ModeloFalso()
    return Assistente(BuscaFalsa(ids_busca), modelo, tmp_path / "spine.parquet", k=6,
                      semente=7), modelo, ids


def _fonte(i, a):
    return Fonte(i, a, f"T {a}", 2020, f"R {a}", 0.5)


def test_b_troca_a_ultima_fonte_por_p_na_posicao_sorteada():
    fa = [_fonte(i, f"x{i}") for i in range(1, 7)]
    p = _fonte(6, "P")
    fb, igual = b.fontes_de_b(fa, p)
    assert not igual
    pos = b.posicao_sorteada("P", 6)
    assert [f.arxiv_id for f in fb].index("P") + 1 == pos
    assert "x6" not in [f.arxiv_id for f in fb]            # saiu a ÚLTIMA
    assert [f.arxiv_id for f in fb if f.arxiv_id != "P"] == [f"x{i}" for i in range(1, 6)]
    assert [f.numero for f in fb] == list(range(1, 7))      # renumeradas


def test_b_e_a_quando_a_busca_ja_trouxe_p():
    fa = [_fonte(i, f"x{i}") for i in range(1, 7)]
    fb, igual = b.fontes_de_b(fa, _fonte(6, "x3"))
    assert igual and fb is fa


def test_posicao_sorteada_e_fixa_e_cobre_as_seis():
    assert b.posicao_sorteada("2101.00001") == b.posicao_sorteada("2101.00001")
    assert {b.posicao_sorteada(f"2101.{i:05d}") for i in range(200)} == set(range(1, 7))


def test_rodar_item_os_tres_bracos(tmp_path):
    assistente, modelo, ids = _assistente(tmp_path, [f"2001.{i:05d}" for i in range(6)])
    item = {"estrato": "primario", "ordem": 3, "arxiv_id": "2001.00008",
            "pergunta": "Qual é a massa do bóson?"}
    ra, rb, rc = b.rodar_item(assistente, item)
    assert ra.posicao_p is None and not rb.igual_a_A
    assert rb.posicao_p == b.posicao_sorteada("2001.00008", 6)
    assert ra.removidas == [9]                              # o portão rodou
    usuarios = [u for s, u, _ in modelo.chamadas if s == SISTEMA_RESPOSTA]
    assert "2001.00008" not in usuarios[0] and "2001.00008" in usuarios[1]
    assert "2001.00005" not in usuarios[1]                  # a 6ª de A saiu em B
    memoria = [u for s, u, _ in modelo.chamadas if s == b.SISTEMA_MEMORIA]
    assert memoria and "Sources" not in memoria[0] and "Portuguese" in memoria[0]
    assert rc.fontes == [] and rc.texto == "De memória, é 41."
    assert all(kw.get("semente") == 7 for _, _, kw in modelo.chamadas)


def test_rodar_item_b_igual_a_a_nao_gera_de_novo(tmp_path):
    assistente, modelo, ids = _assistente(tmp_path, [f"2001.{i:05d}" for i in range(6)])
    item = {"estrato": "primario", "ordem": 0, "arxiv_id": "2001.00002",
            "pergunta": "Qual é a massa?"}
    ra, rb, _ = b.rodar_item(assistente, item)
    assert rb.igual_a_A and rb.texto == ra.texto and ra.posicao_p == 3
    assert sum(1 for s, _, _ in modelo.chamadas if s == SISTEMA_RESPOSTA) == 1


def test_rodar_e_retomavel(tmp_path):
    assistente, modelo, ids = _assistente(tmp_path, [f"2001.{i:05d}" for i in range(6)])
    itens = [{"estrato": "primario", "ordem": i, "arxiv_id": f"2001.{i:05d}",
              "pergunta": f"Pergunta {i}?"} for i in range(4)]
    cache = tmp_path / "bracos.jsonl"
    assert len(b.rodar(assistente, itens, cache, limite=2)) == 6
    n = len(modelo.chamadas)
    assert len(b.rodar(assistente, itens, cache)) == 12
    assert len(b.rodar(assistente, itens, cache)) == 12
    assert len(modelo.chamadas) > n


def test_juiz_nao_ve_citacoes_e_le_o_formato():
    visto = []

    class Juiz:
        def gerar(self, sistema, usuario, **kw):
            visto.append(usuario)
            return 'Claro. {"motivo": "bate", "veredicto": "parcial"}'

    v, m = b.julgar(Juiz(), "P?", "G.", "É 42 [1][2]. E mais [3-4].")
    assert (v, m) == ("parcial", "bate")
    assert "[" not in visto[0].split("Answer:")[1]
    assert b.julgar(Juiz(), "P?", "G.", " [1] ")[0] == "absteve"
    assert b.ler_veredicto("acho que errado, não, certo")[0] == "certo"
    assert b.ler_veredicto("sei lá")[0] == "ilegivel"


def test_chave_do_julgamento_ignora_citacoes():
    assert (b.chave_do_julgamento("p", "g", "É 42 [1].")
            == b.chave_do_julgamento("p", "g", "É 42 [3]."))


def test_kappa():
    assert b.kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert b.kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0


def test_i3_decide_por_certo_contra_o_resto():
    h = ["certo"] * 50 + ["parcial"] * 25 + ["errado"] * 25
    j = ["certo"] * 50 + ["errado"] * 25 + ["parcial"] * 25   # só troca parcial × errado
    r = b.apurar_i3(h, j)
    assert r["kappa_certo"] == 1.0 and r["passa"]
    assert r["kappa_4_categorias"] < 0.6


def test_amostra_i3_itens_diferentes_por_braco():
    rs = [b.RespostaDoBraco("primario", i, f"id{i}", x, "t") for i in range(200) for x in "ABC"]
    am = b.amostra_i3(rs)
    assert len(am) == 100
    assert sum(r.braco == "A" for r in am) == 50 and sum(r.braco == "B" for r in am) == 50
    assert len({r.arxiv_id for r in am}) == 100


def test_amostra_r_so_nao_certos_do_primario_ate_100():
    v = {("primario", f"i{k}"): ("certo" if k % 2 else "errado") for k in range(300)}
    v[("pos_corte", "z")] = "errado"
    am = b.amostra_r(v)
    assert len(am) == 100 and all(k[0] == "primario" and v[k] != "certo" for k in am)
    assert len(b.amostra_r({("primario", "a"): "parcial", ("primario", "b"): "certo"})) == 1


def test_acerto_b_com_r_conta_juiz_errou_e_tira_invalido():
    certo = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    classe = [None] * 6 + ["juiz_errou", "item_invalido", "modelo_errou", "juiz_errou"]
    ponto, (lo, hi) = b.acerto_b_com_r(certo, classe, n_bootstrap=200)
    assert ponto == round(8 / 9, 4) and lo <= ponto <= hi


def test_acerto_b_com_r_por_amostra_usa_as_proporcoes():
    certo = np.array([1] * 6 + [0] * 4)
    classe = [None] * 6 + ["juiz_errou", "modelo_errou", None, None]   # 2 de 4 revisados
    ponto, _ = b.acerto_b_com_r(certo, classe, n_bootstrap=50)
    assert ponto == round((6 + 1 + 2 * 0.5) / 10, 4)


def _veredictos(n, acerto_a, acerto_b, acerto_c):
    ids = [("primario", f"i{k}") for k in range(n)]
    return {x: {k: ("certo" if j < round(p * n) else "errado") for j, k in enumerate(ids)}
            for x, p in (("A", acerto_a), ("B", acerto_b), ("C", acerto_c))}


@pytest.mark.parametrize("acerto_a,acerto_b,gerador,busca", [
    (0.95, 0.97, "BASTA", "NÃO LIMITA"),
    (0.60, 0.80, "LIMITA", "LIMITA"),
    (0.85, 0.90, "NÃO DECIDIDO", "NÃO DECIDIDO"),
])
def test_decidir_os_desfechos(acerto_a, acerto_b, gerador, busca):
    v = _veredictos(500, acerto_a, acerto_b, 0.2)
    r = b.decidir(v, {}, {"passa": True})
    assert r["situacao"] == "válido"
    assert (r["gerador"], r["busca"]) == (gerador, busca)


def test_decidir_i2_e_i3_anulam():
    v = _veredictos(500, 0.9, 0.95, 0.9)
    assert b.decidir(v, {}, {"passa": True})["situacao"].startswith("INVÁLIDO")
    v = _veredictos(500, 0.9, 0.95, 0.2)
    r = b.decidir(v, {}, {"passa": False})
    assert r["situacao"].startswith("NÃO DECIDIDO") and r["gerador"] == "—"


def test_decidir_r_entra_no_acerto_b():
    v = _veredictos(500, 0.85, 0.85, 0.2)
    nao_certos = [k for k, x in v["B"].items() if x != "certo"]
    revisao = dict.fromkeys(nao_certos, "juiz_errou")
    r = b.decidir(v, revisao, {"passa": True})
    assert r["acerto_B_com_R"] == 1.0 and r["gerador"] == "BASTA"
    assert r["lacuna_da_busca"] == 0.0     # a lacuna é pelo juiz, sem R
