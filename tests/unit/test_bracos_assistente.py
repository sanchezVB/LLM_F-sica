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


# ── o braço A′ da §9.2 ──────────────────────────────────────────────────────


class AssistenteReordenadoFalso:
    """`recuperar` devolve as fontes que o teste mandar, por consulta."""

    def __init__(self, fontes_por_consulta):
        self.fontes_por_consulta, self.respondidas = fontes_por_consulta, []

    def recuperar(self, consulta):
        return self.fontes_por_consulta[consulta]

    def responder_com(self, pergunta, fontes, consulta=""):
        from phifm.rag.assistente import Resposta

        self.respondidas.append(consulta)
        return Resposta(pergunta, consulta, "Nova [1].", fontes, [1], [], [])


def test_reordenado_reaproveita_A_quando_as_fontes_sao_as_mesmas(tmp_path):
    fa = [_fonte(i, f"x{i}") for i in range(1, 7)]
    trocadas = [_fonte(1, "P")] + fa[:5]
    itens = [{"estrato": "primario", "ordem": 0, "arxiv_id": "P", "pergunta": "q0"},
             {"estrato": "primario", "ordem": 1, "arxiv_id": "Q", "pergunta": "q1"}]
    ra = {("primario", "P"): b.RespostaDoBraco("primario", 0, "P", "A", "Velha [2].",
                                                [f.arxiv_id for f in fa], consulta="c0"),
          ("primario", "Q"): b.RespostaDoBraco("primario", 1, "Q", "A", "Igual [1].",
                                                [f.arxiv_id for f in fa], consulta="c1")}
    ass = AssistenteReordenadoFalso({"c0": trocadas, "c1": fa})
    regs = {r.arxiv_id: r for r in b.rodar_reordenado(ass, itens, ra, tmp_path / "a2.jsonl")}
    assert ass.respondidas == ["c0"]                     # só onde as fontes mudaram
    assert regs["P"].texto == "Nova [1]." and regs["P"].posicao_p == 1
    assert regs["Q"].igual_a_A and regs["Q"].texto == "Igual [1]."
    assert {r.braco for r in regs.values()} == {b.BRACO_REORDENADO}
    assert len(b.rodar_reordenado(ass, itens, ra, tmp_path / "a2.jsonl")) == 2   # retoma


def _caso_reordenado(n, certos_a, certos_a2, t_a=30.0, t_rec=14.0):
    ids = [("primario", f"i{k}") for k in range(n)]
    v = {"A": {k: ("certo" if j < certos_a else "errado") for j, k in enumerate(ids)},
         b.BRACO_REORDENADO: {k: ("certo" if j < certos_a2 else "errado")
                              for j, k in enumerate(ids)}}
    rs = [b.RespostaDoBraco("primario", j, k[1], "A", "t", segundos=t_a)
          for j, k in enumerate(ids)]
    rs += [b.RespostaDoBraco("primario", j, k[1], b.BRACO_REORDENADO, "t", segundos=20.0,
                             segundos_recuperacao=t_rec) for j, k in enumerate(ids)]
    return v, rs


def test_comparar_reordenado_e_pareado_por_item():
    r = b.comparar_reordenado(*_caso_reordenado(10, 5, 7))
    assert (r["acerto_A"], r["acerto_A2"], r["A2_menos_A"]) == (0.5, 0.7, 0.2)
    assert (r["ganha"], r["perde"]) == (2, 0)
    assert r["tempo_mediano_s"] == 44.0          # mediana de A + mediana da recuperação


def test_a_regra_da_9_2_adota_so_com_o_IC_inteiro_acima_de_3_pontos_e_dentro_do_tempo():
    assert b.REGRA_9_2_ACEITA and b.LIMIAR_ADOCAO_REORDENADO == 0.03
    assert b.comparar_reordenado(*_caso_reordenado(500, 300, 350))["decisao"] == "ADOTA"
    # +2 pontos: ganho real, mas o IC não fica inteiro acima de 0,03
    assert (b.comparar_reordenado(*_caso_reordenado(500, 300, 310))["decisao"]
            == "FICA O SISTEMA ATUAL")
    # ganho grande, mas estoura os 60 s
    assert (b.comparar_reordenado(*_caso_reordenado(500, 300, 350, t_a=50.0))["decisao"]
            == "FICA O SISTEMA ATUAL")


def test_decidir_relata_o_gerador_tambem_SEM_a_revisao_R():
    """R é do Claude desde 2026-10-04: se a decisão depender dela, tem de aparecer."""
    v = _veredictos(500, 0.85, 0.85, 0.2)
    nao_certos = [k for k, x in v["B"].items() if x != "certo"]
    r = b.decidir(v, dict.fromkeys(nao_certos, "juiz_errou"), {"passa": True})
    assert (r["gerador"], r["gerador_sem_R"]) == ("BASTA", "LIMITA")


# ── o estrato "corpo" (DOC-13 §9.5) ──────────────────────────────────────────


def test_a_passagem_toma_o_lugar_do_resumo_de_p_onde_p_ja_veio():
    fa = [_fonte(i, f"x{i}") for i in range(1, 7)]
    p = Fonte(6, "x3", "T x3", 2020, "A PASSAGEM", float("nan"))
    fb = b.fontes_com_passagem(fa, p)
    assert [f.arxiv_id for f in fb] == [f.arxiv_id for f in fa]     # mesma ordem, mesmas 6
    assert fb[2].resumo == "A PASSAGEM" and fb[2].numero == 3
    assert fa[2].resumo == "R x3"                                   # as de A′ não mudam


def test_a_passagem_entra_na_posicao_sorteada_quando_p_nao_veio():
    fa = [_fonte(i, f"x{i}") for i in range(1, 7)]
    fb = b.fontes_com_passagem(fa, Fonte(6, "P", "T P", 2020, "A PASSAGEM", float("nan")))
    ids = [f.arxiv_id for f in fb]
    assert ids.index("P") + 1 == b.posicao_sorteada("P", 6) and "x6" not in ids
    assert [f.numero for f in fb] == list(range(1, 7))


def test_item_do_corpo_tres_bracos_e_a_passagem_so_em_bp(tmp_path):
    assistente, modelo, ids = _assistente(tmp_path, [f"2001.{i:05d}" for i in range(6)])
    item = {"estrato": "corpo", "ordem": 0, "arxiv_id": "2001.00002",
            "pergunta": "Qual é a espessura do alvo?"}
    ra, rb, rc = b.rodar_item_corpo(assistente, item, "O alvo tem 460 nm de espessura.")
    assert (ra.braco, rb.braco, rc.braco) == (b.BRACO_REORDENADO, b.BRACO_PASSAGEM, "C")
    assert ra.posicao_p == 3 and rb.posicao_p == 3 and ra.fontes == rb.fontes
    respostas = [u for s, u, _ in modelo.chamadas if s == SISTEMA_RESPOSTA]
    assert len(respostas) == 2
    assert "460 nm" not in respostas[0] and "Resumo de 2001.00002." in respostas[0]
    assert "460 nm" in respostas[1] and "Resumo de 2001.00002." not in respostas[1]
    assert sum(1 for s, _, _ in modelo.chamadas if s == SISTEMA_HYDE) == 1   # uma consulta só
    memoria = [u for s, u, _ in modelo.chamadas if s == b.SISTEMA_MEMORIA]
    assert len(memoria) == 1 and "Sources" not in memoria[0]


def _respostas_do_corpo(n=60):
    saida = []
    for i in range(n):
        for braco in (b.BRACO_REORDENADO, b.BRACO_PASSAGEM, "C"):
            saida.append(b.RespostaDoBraco("corpo", i, f"a{i:03d}", braco, f"texto {braco} {i}",
                                           posicao_p=1 if (i % 3 == 0 or braco == "Bp") else None))
    return saida


def test_amostra_i3c_tem_20_de_cada_braco_de_itens_diferentes():
    amostra = b.amostra_i3c(_respostas_do_corpo())
    assert len(amostra) == 2 * b.N_I3C_POR_BRACO
    assert sum(r.braco == b.BRACO_REORDENADO for r in amostra) == b.N_I3C_POR_BRACO
    assert len({r.arxiv_id for r in amostra}) == len(amostra)
    assert [r.arxiv_id for r in amostra] == [r.arxiv_id for r in b.amostra_i3c(_respostas_do_corpo())]


def test_resumo_do_corpo_as_leituras_e_as_duas_guardas():
    respostas = _respostas_do_corpo(60)
    ids = [("corpo", f"a{i:03d}") for i in range(60)]
    v = {b.BRACO_PASSAGEM: dict.fromkeys(ids, "certo"),                      # Bp acerta tudo
         b.BRACO_REORDENADO: {k: ("certo" if i < 6 else "absteve") for i, k in enumerate(ids)},
         "C": dict.fromkeys(ids, "errado")}
    r = b.resumo_corpo(v, respostas)
    assert r["gerador_com_a_passagem"]["desfecho"] == "BASTA"
    assert r["lacuna_do_corpo"]["Bp_menos_A2"] == 0.9 and r["lacuna_do_corpo"]["perde"] == 0
    assert r["p_nas_fontes_de_A2"]["itens"] == 20
    assert r["guardas"]["I2c"]["valida"] is True
    # A′ acertou 2 dos 20 itens em que só tinha o resumo de P (i = 0 e 3): 0,10 ≤ 0,20
    assert r["guardas"]["fora_do_resumo"]["acerto_de_A2_neles"] == 0.1
    assert r["guardas"]["fora_do_resumo"]["valida"] is True
    # se o resumo responde, a guarda cai
    v[b.BRACO_REORDENADO] = dict.fromkeys(ids, "certo")
    assert b.resumo_corpo(v, respostas)["guardas"]["fora_do_resumo"]["valida"] is False
    # e C perto de Bp derruba a outra
    v["C"] = dict.fromkeys(ids, "certo")
    assert b.resumo_corpo(v, respostas)["guardas"]["I2c"]["valida"] is False


# ── as passagens sob demanda (DOC-13 §9.6) ───────────────────────────────────


class PassoFalso:
    """Anexa uma passagem à 2ª fonte — ou nada, se `vazio`."""

    def __init__(self, vazio=False):
        self.vazio = vazio

    def anexar_com_registro(self, pergunta, fontes, consulta=None):
        from dataclasses import replace

        from phifm.rag.passagens import Escolhido

        if self.vazio:
            return list(fontes), [], "the question"
        esc = [Escolhido(fontes[1].arxiv_id, 4, "Setup", "O alvo tem 460 nm.", 10, 1.0)]
        return ([replace(f, passagens=("O alvo tem 460 nm.",)) if i == 1 else f
                 for i, f in enumerate(fontes)], esc, "the question")


def test_item_das_passagens_quatro_bracos_e_so_A3_le_o_trecho_escolhido(tmp_path):
    assistente, modelo, ids = _assistente(tmp_path, [f"2001.{i:05d}" for i in range(6)])
    item = {"estrato": "corpo", "ordem": 0, "arxiv_id": "2001.00002",
            "pergunta": "Qual é a espessura do alvo?"}
    ra, r3, rb, rc = b.rodar_item_passagens(assistente, PassoFalso(), item, "A PASSAGEM DE OURO.")
    assert [r.braco for r in (ra, r3, rb, rc)] == [b.BRACO_REORDENADO, b.BRACO_PASSAGENS,
                                                   b.BRACO_PASSAGEM, "C"]
    assert ra.fontes == r3.fontes == rb.fontes and ra.consulta == r3.consulta
    assert r3.passagens == [["2001.00001", 4]] and ra.passagens == []
    assert r3.consulta_passagens == "the question" and ra.consulta_passagens == ""
    assert r3.segundos >= r3.segundos_recuperacao >= 0
    prompts = [u for s, u, _ in modelo.chamadas if s == SISTEMA_RESPOSTA]
    assert len(prompts) == 3
    assert "460 nm" not in prompts[0] and "460 nm" in prompts[1] and "460 nm" not in prompts[2]
    assert "A PASSAGEM DE OURO." in prompts[2] and "A PASSAGEM DE OURO." not in prompts[1]
    assert "Resumo de 2001.00001." in prompts[1]          # o resumo fica, com o trecho embaixo
    assert sum(1 for s, _, _ in modelo.chamadas if s == SISTEMA_HYDE) == 1


def test_A3_sobre_o_cache_de_A2_reaproveita_a_resposta_quando_nada_e_anexado(tmp_path):
    assistente, modelo, ids = _assistente(tmp_path, [f"2001.{i:05d}" for i in range(6)])
    itens = [{"estrato": "primario", "ordem": i, "arxiv_id": ids[i], "pergunta": f"P{i}?"}
             for i in range(2)]
    a2 = {("primario", it["arxiv_id"]): b.RespostaDoBraco(
        "primario", it["ordem"], it["arxiv_id"], b.BRACO_REORDENADO, "resposta de A2 [1].",
        fontes=ids[:6], posicao_p=1, consulta="hyde") for it in itens}

    def fontes_de(lista):
        return [_fonte(i, a) for i, a in enumerate(lista, start=1)]

    regs = b.rodar_passagens_sobre_a2(assistente, PassoFalso(vazio=True), itens, a2, fontes_de,
                                      tmp_path / "vazio.jsonl")
    assert all(r.igual_a_A and r.texto == "resposta de A2 [1]." and r.braco == b.BRACO_PASSAGENS
               for r in regs)
    assert not any(s == SISTEMA_RESPOSTA for s, _, _ in modelo.chamadas)

    regs = b.rodar_passagens_sobre_a2(assistente, PassoFalso(), itens, a2, fontes_de,
                                      tmp_path / "com.jsonl", limite=1)
    assert len(regs) == 1 and not regs[0].igual_a_A and regs[0].passagens == [[ids[1], 4]]
    regs = b.rodar_passagens_sobre_a2(assistente, PassoFalso(), itens, a2, fontes_de,
                                      tmp_path / "com.jsonl")
    assert len(regs) == 2                                  # retomou, sem refazer o primeiro
    assert sum(1 for s, _, _ in modelo.chamadas if s == SISTEMA_RESPOSTA) == 2


def _caso_das_passagens(n=100, certos_a2=3, certos_a3=45, t_a3=55.0):
    ids = [("corpo", f"a{i:03d}") for i in range(n)]
    v = {b.BRACO_REORDENADO: {k: "certo" if i < certos_a2 else "absteve"
                              for i, k in enumerate(ids)},
         b.BRACO_PASSAGENS: {k: "certo" if i < certos_a3 else "errado" for i, k in enumerate(ids)},
         b.BRACO_PASSAGEM: dict.fromkeys(ids, "certo"), "C": dict.fromkeys(ids, "errado")}
    respostas = []
    for i, (_, a) in enumerate(ids):
        pos = 1 if i % 2 == 0 else None
        respostas += [
            b.RespostaDoBraco("corpo", i, a, b.BRACO_REORDENADO, "x", posicao_p=pos, segundos=50.0),
            b.RespostaDoBraco("corpo", i, a, b.BRACO_PASSAGENS, "y", posicao_p=pos, segundos=t_a3,
                              segundos_passagens=4.0),
            b.RespostaDoBraco("corpo", i, a, b.BRACO_PASSAGEM, "z", posicao_p=1),
            b.RespostaDoBraco("corpo", i, a, "C", "w")]
    return v, respostas


def test_o_ganho_no_corpo_passa_so_com_o_IC_inteiro_acima_do_limiar():
    r = b.resumo_passagens(*_caso_das_passagens())
    assert r["ganho_no_corpo"]["A3_menos_A2"] == 0.42 and r["ganho_no_corpo"]["passa"] is True
    assert r["ganho_no_corpo"]["perde"] == 0 and r["tempo"]["passa"] is True
    assert r["o_teto"]["fracao_do_teto_colhida"] == 0.45
    assert all(g["valida"] for g in r["guardas"].values())
    # +0,17 em 100 itens: o IC desce abaixo de 0,15, e não passa
    assert b.resumo_passagens(*_caso_das_passagens(certos_a3=20))["ganho_no_corpo"]["passa"] is False
    # e o tempo reprova sozinho
    assert b.resumo_passagens(*_caso_das_passagens(t_a3=71.0))["tempo"]["passa"] is False


def _caso_do_primario(n=500, perde=0, ganha=0):
    ids = [("primario", f"p{i:03d}") for i in range(n)]
    a2 = {k: "certo" if i < 400 else "errado" for i, k in enumerate(ids)}
    a3 = dict(a2)
    for k in ids[:perde]:
        a3[k] = "errado"
    for k in ids[400:400 + ganha]:
        a3[k] = "certo"
    respostas = [b.RespostaDoBraco("primario", i, a, b.BRACO_PASSAGENS, "y",
                                   igual_a_A=i % 5 == 0, passagens=[] if i % 5 == 0 else [[a, 1]])
                 for i, (_, a) in enumerate(ids)]
    return {b.BRACO_REORDENADO: a2, b.BRACO_PASSAGENS: a3}, respostas


def test_nao_piora_e_o_IC_inteiro_acima_de_menos_3_pontos():
    r = b.nao_piora(*_caso_do_primario())
    assert r["A3_menos_A2"] == 0.0 and r["passa"] is True
    assert r["respostas_iguais_as_de_A2"] == 100 and r["com_passagem_anexada"] == 400
    assert b.nao_piora(*_caso_do_primario(perde=10, ganha=6))["passa"] is True     # −0,008
    assert b.nao_piora(*_caso_do_primario(perde=25))["passa"] is False             # −0,05


def test_a_regra_da_9_6_adota_so_com_os_tres_eixos_e_fica_pendente_sem_o_primario():
    corpo = b.resumo_passagens(*_caso_das_passagens())
    passa, nao = b.nao_piora(*_caso_do_primario()), b.nao_piora(*_caso_do_primario(perde=25))
    assert b.decidir_passagens(corpo, passa, {"passa": True})["decisao"] == "ADOTA"
    assert b.decidir_passagens(corpo, nao, {"passa": True})["decisao"] == "FICA O SISTEMA ATUAL"
    assert b.decidir_passagens(corpo, None, {"passa": True})["decisao"].startswith("PENDENTE")
    assert b.decidir_passagens(corpo, passa, {"passa": False})["decisao"] == "—"
    lento = b.resumo_passagens(*_caso_das_passagens(t_a3=71.0))
    assert b.decidir_passagens(lento, passa, {"passa": True})["decisao"] == "FICA O SISTEMA ATUAL"


def test_amostra_i3_da_9_6_troca_os_bracos_e_o_sorteio():
    v, respostas = _caso_das_passagens(60)
    amostra = b.amostra_i3c(respostas, bracos=(b.BRACO_PASSAGENS, b.BRACO_REORDENADO), salto=6)
    assert sorted({r.braco for r in amostra}) == sorted([b.BRACO_PASSAGENS, b.BRACO_REORDENADO])
    assert len(amostra) == 40 and len({r.arxiv_id for r in amostra}) == 40


# ── a emenda da §9.6: A″ com a instrução candidata (A‴) ──────────────────────


class PassoQueRepete:
    """Devolve sempre a mesma seleção — a que o braço A″ gravou, ou outra, se `mudou`."""

    def __init__(self, mudou=False):
        self.mudou, self.consultas = mudou, []

    def anexar_com_registro(self, pergunta, fontes, consulta=None):
        from dataclasses import replace

        from phifm.rag.passagens import Escolhido

        self.consultas.append(consulta)
        esc = [Escolhido(fontes[0].arxiv_id, 9 if self.mudou else 4, "s", "O alvo tem 460 nm.",
                         10, 1.0)]
        return ([replace(fontes[0], passagens=("O alvo tem 460 nm.",)), *fontes[1:]], esc,
                consulta)


def test_A4_so_troca_a_instrucao_e_recusa_selecao_diferente_da_de_A3(tmp_path):
    from phifm.rag.assistente import SISTEMA_RESPOSTA_V2

    ids = [f"2001.{i:05d}" for i in range(10)]
    pl.DataFrame({"arxiv_id": ids, "title": [f"Título {a}" for a in ids],
                  "abstract": [f"Resumo de {a}." for a in ids],
                  "year": [2020] * 10}).write_parquet(tmp_path / "spine.parquet")

    class ModeloV2(ModeloFalso):
        def gerar(self, sistema, usuario, **kw):
            self.chamadas.append((sistema, usuario, kw))
            return "As fontes não trazem a espessura."

    modelo = ModeloV2()
    assistente = Assistente(None, modelo, tmp_path / "spine.parquet", k=6, semente=7,
                            sistema=SISTEMA_RESPOSTA_V2)
    item = {"estrato": "corpo", "ordem": 0, "arxiv_id": ids[2], "pergunta": "Qual a espessura?"}
    a3 = {("corpo", ids[2]): b.RespostaDoBraco(
        "corpo", 0, ids[2], b.BRACO_PASSAGENS, "resposta de A3 [1].", fontes=ids[:6],
        posicao_p=3, consulta="hyde", segundos=50.0, segundos_recuperacao=30.0,
        segundos_passagens=2.0, consulta_passagens="what is the thickness",
        passagens=[[ids[0], 4]])}

    def fontes_de(lista):
        return [_fonte(i, x) for i, x in enumerate(lista, start=1)]

    passo = PassoQueRepete()
    (r4,) = b.rodar_instrucao_sobre_a3(assistente, passo, [item], a3, fontes_de,
                                       tmp_path / "a4.jsonl")
    assert r4.braco == b.BRACO_PASSAGENS_V2 and r4.passagens == [[ids[0], 4]]
    assert r4.fontes == ids[:6] and r4.consulta == "hyde"
    assert passo.consultas == ["what is the thickness"]        # a consulta de A″, sem traduzir
    sistema, usuario, _ = modelo.chamadas[-1]
    assert sistema == SISTEMA_RESPOSTA_V2 and "460 nm" in usuario
    assert r4.segundos >= 32.0 and r4.segundos_passagens == 2.0  # o tempo de A″ até a resposta
    with pytest.raises(RuntimeError, match="não é a que A″ gravou"):
        b.rodar_instrucao_sobre_a3(assistente, PassoQueRepete(mudou=True), [item], a3,
                                   fontes_de, tmp_path / "outro.jsonl")


def test_A4_no_teste_responde_todo_item_mesmo_sem_passagem(tmp_path):
    """Com outra instrução o prompt muda mesmo sem passagem: nada de A′ é reaproveitado."""
    assistente, modelo, ids = _assistente(tmp_path, [f"2001.{i:05d}" for i in range(6)])
    itens = [{"estrato": "primario", "ordem": i, "arxiv_id": ids[i], "pergunta": f"P{i}?"}
             for i in range(2)]
    a2 = {("primario", it["arxiv_id"]): b.RespostaDoBraco(
        "primario", it["ordem"], it["arxiv_id"], b.BRACO_REORDENADO, "resposta de A2 [1].",
        fontes=ids[:6], posicao_p=1, consulta="hyde", segundos_recuperacao=20.0)
        for it in itens}

    def fontes_de(lista):
        return [_fonte(i, a) for i, a in enumerate(lista, start=1)]

    regs = b.rodar_passagens_sobre_a2(
        assistente, PassoFalso(vazio=True), itens, a2, fontes_de, tmp_path / "a4.jsonl",
        braco=b.BRACO_PASSAGENS_V2, reaproveitar=False)
    assert [r.braco for r in regs] == [b.BRACO_PASSAGENS_V2] * 2
    assert not any(r.igual_a_A for r in regs) and all(r.passagens == [] for r in regs)
    assert all(r.segundos >= 20.0 and r.segundos_recuperacao == 20.0 for r in regs)
    assert all(r.consulta_passagens == "the question" for r in regs)
    assert sum(1 for s, _, _ in modelo.chamadas if s == assistente.sistema) == 2


def test_o_eixo_do_teste_e_a_decisao_valem_para_o_braco_da_emenda():
    v, respostas = _caso_do_primario(perde=10, ganha=6)
    v4 = {b.BRACO_REORDENADO: v[b.BRACO_REORDENADO], b.BRACO_PASSAGENS_V2: v[b.BRACO_PASSAGENS]}
    r4 = [b.RespostaDoBraco(r.estrato, r.ordem, r.arxiv_id, b.BRACO_PASSAGENS_V2, r.texto,
                            passagens=r.passagens) for r in respostas]
    prim = b.nao_piora(v4, r4, braco=b.BRACO_PASSAGENS_V2)
    assert prim["braco"] == "A4" and prim["A4_menos_A2"] == -0.008 and prim["passa"] is True
    assert prim["veredictos"]["A4"]["certo"] == 396 and prim["veredictos"]["A2"]["certo"] == 400
    assert prim["com_passagem_anexada"] == 400

    # A″ reprova o ganho no corpo; o candidato é A‴, e são os eixos DELE que decidem.
    corpo = b.resumo_passagens(*_caso_das_passagens(certos_a3=10))
    assert corpo["ganho_no_corpo"]["passa"] is False
    passa = {"ganho_no_corpo": {"passa": True}, "tempo": {"passa": True}}
    d = b.decidir_passagens(corpo, prim, {"passa": True}, candidato=passa)
    assert (d["candidato"], d["decisao"]) == ("A4", "ADOTA")
    lento = {"ganho_no_corpo": {"passa": True}, "tempo": {"passa": False}}
    assert b.decidir_passagens(corpo, prim, {"passa": True}, candidato=lento)["decisao"] \
        == "FICA O SISTEMA ATUAL"
    assert b.decidir_passagens(corpo, prim, {"passa": True})["candidato"] == "A3"


def test_comparar_instrucao_e_pareada_contra_A2_e_contra_A3():
    v, respostas = _caso_das_passagens(100, certos_a2=3, certos_a3=45)
    ids = sorted(v[b.BRACO_PASSAGENS])
    v[b.BRACO_PASSAGENS_V2] = {k: "certo" if 2 <= i < 49 else "absteve"
                               for i, k in enumerate(ids)}                # 47 certos
    respostas += [b.RespostaDoBraco("corpo", i, a, b.BRACO_PASSAGENS_V2, "v", segundos=48.0)
                  for i, (_, a) in enumerate(ids)]
    r = b.comparar_instrucao(v, respostas)
    assert r["acerto_A4"]["valor"] == 0.47 and r["ganho_no_corpo"]["A4_menos_A2"] == 0.44
    assert r["ganho_no_corpo"]["passa"] is True and r["tempo"]["passa"] is True
    assert (r["A4_menos_A3"]["valor"], r["A4_menos_A3"]["ganha"], r["A4_menos_A3"]["perde"]) \
        == (0.02, 4, 2)
