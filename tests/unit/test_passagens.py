"""Passagens sob demanda (`phifm.rag.passagens`), com modelo FALSO. O que se tranca:

1. a escolha é lexical, pela consulta em inglês, e respeita `k` e o orçamento de tokens;
2. trechos vizinhos do mesmo artigo viram um bloco só, na ordem do texto;
3. as fontes voltam na mesma ordem e com o mesmo número — só ganham as passagens;
4. sem passagens, o prompt é byte a byte o das medidas anteriores.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.rag import passagens as p  # noqa: E402
from phifm.rag.assistente import (  # noqa: E402
    ROTULO_PASSAGENS,
    SISTEMA_HYDE,
    SISTEMA_RESPOSTA,
    Assistente,
    Fonte,
    bloco_de_fontes,
)
from phifm.retrieval.indice import Resultado  # noqa: E402
from phifm.retrieval.trechos import Trecho  # noqa: E402


def _t(texto, n=10, secao="Methods"):
    return Trecho(texto, secao, "corpo", n)


class ModeloFalso:
    def __init__(self, traducao="what is the substrate temperature during growth"):
        self.traducao, self.chamadas = traducao, []

    def gerar(self, sistema, usuario, **kw):
        self.chamadas.append((sistema, usuario, kw))
        if sistema == p.SISTEMA_TRADUCAO:
            return self.traducao
        if sistema == SISTEMA_HYDE:
            return "Hypothetical title. Hypothetical abstract."
        return "A temperatura é 650 K [1]."

    def gerar_em_fluxo(self, sistema, usuario, **kw):
        self.chamadas.append((sistema, usuario, kw))
        yield "A temperatura é 650 K [1]."


def test_palavras_minusculas_numeros_e_virgula_decimal():
    assert p.palavras("The Substrate was held at 0,35 keV, T_c = 92 K.") == [
        "the", "substrate", "was", "held", "at", "0.35", "kev", "92"]


def test_bm25_premia_a_palavra_rara_e_zera_quem_nao_casa():
    docs = [p.palavras(x) for x in ("the film was grown on a substrate",
                                    "the film was annealed in oxygen",
                                    "the detector was calibrated with krypton")]
    s = p.bm25("substrate of the film", docs)
    assert s.argmax() == 0            # `substrate` só está no primeiro
    assert s[1] > 0 and s[1] < s[0]   # `film` e `the` casam, e valem menos
    assert p.bm25("neutrino", docs).tolist() == [0.0, 0.0, 0.0]


def test_escolher_respeita_k_e_o_orcamento_e_pula_quem_nao_casa():
    trechos = {
        "a": [_t("introduction to thin films"),
              _t("the substrate temperature was 650 K during growth", n=40),
              _t("growth rate and substrate temperature are discussed", n=40)],
        "b": [_t("the substrate was cleaned before growth", n=40), _t("neutrino oscillations")]}
    consulta = "substrate temperature during growth"
    todos = p.escolher(consulta, trechos, k=9, orcamento=1000)
    assert [(e.arxiv_id, e.posicao) for e in todos][0] == ("a", 1)
    assert ("b", 1) not in {(e.arxiv_id, e.posicao) for e in todos}   # não casa com nada
    assert len(p.escolher(consulta, trechos, k=2, orcamento=1000)) == 2
    # com 80 tokens cabem dois trechos de 40; o terceiro melhor fica de fora
    assert sum(e.n_tokens for e in p.escolher(consulta, trechos, k=9, orcamento=80)) == 80
    assert p.escolher(consulta, {}, k=9) == []


def test_um_trecho_maior_que_o_orcamento_nao_entra_e_nao_trava_os_seguintes():
    trechos = {"a": [_t("substrate temperature growth " * 3, n=5000),
                     _t("the substrate temperature was 650 K", n=30)]}
    esc = p.escolher("substrate temperature", trechos, k=9, orcamento=1440)
    assert [e.posicao for e in esc] == [1]


def test_vizinhos_viram_um_bloco_na_ordem_do_texto():
    esc = [p.Escolhido("a", 7, "s", "sete.", 5, 3.0), p.Escolhido("b", 2, "s", "dois.", 5, 2.5),
           p.Escolhido("a", 3, "s", "três.", 5, 2.0), p.Escolhido("a", 6, "s", "seis.", 5, 1.0)]
    assert p.em_blocos(esc) == {"a": ["três.", "seis. sete."], "b": ["dois."]}


def test_anexar_mantem_ordem_e_numero_e_so_poe_passagem_em_quem_tem_texto():
    textos = {"x2": [_t("the substrate temperature was 650 K during growth"),
                     _t("samples were shipped by mail")]}
    modelo = ModeloFalso()
    ps = p.Passagens(modelo, textos.get, k=9, semente=7)
    fontes = [Fonte(1, "x1", "T1", 2020, "R1", 0.9), Fonte(2, "x2", "T2", 2021, "R2", 0.8)]
    novas = ps.anexar("qual a temperatura do substrato no crescimento?", fontes)
    assert [(f.numero, f.arxiv_id, f.resumo) for f in novas] == [(1, "x1", "R1"), (2, "x2", "R2")]
    assert novas[0].passagens == ()
    assert novas[1].passagens == ("the substrate temperature was 650 K during growth",)
    sistema, usuario, kw = modelo.chamadas[0]
    assert sistema == p.SISTEMA_TRADUCAO and "substrato" in usuario and kw["semente"] == 7
    # com a consulta dada, o modelo não é chamado de novo
    ps.anexar("…", fontes, consulta="substrate temperature")
    assert len(modelo.chamadas) == 1


class TextosFalsos:
    def __init__(self):
        self.lidos = []

    def ler(self, arxiv_id):
        self.lidos.append(arxiv_id)
        if arxiv_id == "sem":
            return None
        return (f"\\section{{Setup}} The sample {arxiv_id} was grown at 650 K on a sapphire "
                "substrate and then annealed in oxygen for two hours before the measurement.")


def test_trechos_do_texto_corta_uma_vez_e_guarda_os_ultimos():
    textos = TextosFalsos()
    de = p.TrechosDoTexto(textos, lambda xs: [len(x.split()) for x in xs], guardados=2)
    ts = de("a")
    assert ts and "650 K" in ts[0].texto and ts[0].secao == "Setup"
    assert de("a") is ts and textos.lidos == ["a"]          # do cache, sem ler de novo
    assert de("sem") is None and de("sem") is None and textos.lidos == ["a", "sem"]
    de("a"), de("b")                                        # "sem" é o mais antigo: sai
    assert de("sem") is None and textos.lidos == ["a", "sem", "b", "sem"]


def test_traducao_vazia_cai_na_pergunta_como_veio():
    ps = p.Passagens(ModeloFalso(traducao="  "), lambda a: None)
    assert ps.consulta("What is the substrate temperature?") == "What is the substrate temperature?"


def test_o_bloco_sem_passagens_e_o_das_medidas_anteriores():
    fontes = [Fonte(1, "2001.00001", "Título", 2020, "Resumo.", 0.9),
              Fonte(2, "2001.00002", "Outro", None, "Outro resumo.", 0.8)]
    assert bloco_de_fontes(fontes) == (
        "[1] Título (2020, arXiv:2001.00001)\nResumo.\n\n"
        "[2] Outro (n.d., arXiv:2001.00002)\nOutro resumo.")


def test_o_bloco_com_passagens_as_poe_debaixo_do_resumo_do_mesmo_numero():
    fontes = [Fonte(1, "2001.00001", "Título", 2020, "Resumo.", 0.9,
                    passagens=("primeira\n  passagem", "segunda"))]
    assert bloco_de_fontes(fontes) == (
        "[1] Título (2020, arXiv:2001.00001)\nResumo.\n"
        f"{ROTULO_PASSAGENS}\n- primeira passagem\n- segunda")


class BuscaFalsa:
    def buscar(self, consulta, k=10, excluir=None):
        return [Resultado(i + 1, 0.9 - i / 10, a, f"Título {a}", 2020, "quant-ph", ["A"])
                for i, a in enumerate(("x1", "x2"))][:k]


def test_o_assistente_so_usa_passagens_quando_recebe_o_passo(tmp_path):
    pl.DataFrame({"arxiv_id": ["0704.0001", "x1", "x2"], "title": ["t0", "t1", "t2"],
                  "abstract": ["r0", "r1", "r2"], "year": [2007, 2020, 2021]}
                 ).write_parquet(tmp_path / "spine.parquet")
    textos = {"x2": [_t("the substrate temperature was 650 K during growth")]}
    modelo = ModeloFalso()
    sem = Assistente(BuscaFalsa(), modelo, tmp_path / "spine.parquet", k=2, semente=7)
    r = sem.responder("qual a temperatura do substrato?")
    assert all(f.passagens == () for f in r.fontes)
    assert not any(s == p.SISTEMA_TRADUCAO for s, _, _ in modelo.chamadas)

    modelo = ModeloFalso()
    com = Assistente(BuscaFalsa(), modelo, tmp_path / "spine.parquet", k=2, semente=7,
                     passagens=p.Passagens(modelo, textos.get, semente=7))
    r = com.responder("qual a temperatura do substrato?")
    assert r.fontes[1].passagens and not r.fontes[0].passagens
    prompt = next(u for s, u, _ in modelo.chamadas if s == SISTEMA_RESPOSTA)
    assert ROTULO_PASSAGENS in prompt and "650 K" in prompt
    # o fluxo da página entrega as fontes já com as passagens
    eventos = dict(com.responder_em_fluxo("qual a temperatura do substrato?"))
    assert eventos["fontes"][1][1].passagens and eventos["fim"].fontes[1].passagens
