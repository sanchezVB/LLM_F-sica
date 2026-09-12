"""O PB-Formula e as duas armadilhas que quase o mataram ou o tornariam inválido.

⚠️ **A primeira quase o matou.** A calibração do filtro rodou em 2.000 documentos e
achou 27 itens — 0,040% das formas. Eu estava a um passo de reportar que o DOC-11
§6.3 não é viável. Medido em quatro tamanhos:

    2.000 docs ·  68.079 formas ·     27 em ≥2 docs (0,040%)
   20.000 docs · 678.929 formas ·  1.654 em ≥2 docs (0,244%)

10× mais documentos deram **61× mais itens**, porque uma colisão exige as DUAS
pontas na amostra: amostrar 1/400 do corpus encontra ~1/160.000 dos pares.

⚠️ **A segunda o tornaria inválido**: um pool cujo teto não é 1,0. O T1b2 pagou
isso — 62% de alvos duplicados, teto real 0,7562 de nDCG@10, e a tabela publicada
comparava modelos contra um máximo inalcançável.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval.benchmarks.formula import (  # noqa: E402
    MIN_CARACTERES,
    Item,
    Ocorrencia,
    conferir_pool,
    e_conteudo,
    montar_itens,
    teto_do_pool,
)

EQ = r"F_{\mu\nu}=\partial_\mu A_\nu-\partial_\nu A_\mu"


def _oc(doc: str, forma: str = "h1", latex: str = EQ) -> Ocorrencia:
    return Ocorrencia(documento=doc, forma=forma, latex=latex)


# ── o filtro de conteúdo ────────────────────────────────────────────────────

def test_simbolo_solto_NAO_e_item():
    """⚠️ `extrair_equacoes` aceita `\\alpha` de propósito — ele alimenta o
    tokenizer. Medido: `\\alpha` aparece em 714 de 2.000 documentos. Recuperar
    "os documentos que contêm alpha" não mede nada."""
    for nao in (r"\alpha", r"\sigma", r"\beta", "x", r"\Delta L"):
        assert not e_conteudo(nao), nao


def test_equacao_de_conteudo_E_item():
    assert e_conteudo(EQ)
    assert e_conteudo(r"\langle\bar{q}q\rangle=-(0.24\pm0.01\rm{GeV})^3")


def test_exige_RELACAO_e_nao_so_comprimento():
    """Uma expressão longa sem relação é um termo, não uma afirmação."""
    longa = r"\int\frac{d^4k}{(2\pi)^4}\frac{1}{k^2-m^2+i\epsilon}"
    assert len(longa) >= MIN_CARACTERES
    assert not e_conteudo(longa)
    assert e_conteudo(longa + "=0")


def test_exige_COMPRIMENTO_e_nao_so_relacao():
    """`x=0` tem relação e não discrimina documento nenhum."""
    assert not e_conteudo("x=0")
    assert not e_conteudo(r"\theta=\frac{\pi}{4}")


# ── a montagem dos itens ────────────────────────────────────────────────────

def test_a_consulta_NUNCA_esta_entre_os_alvos():
    """Se o documento da consulta é alvo, o item é trivial: a equação está nele."""
    itens, _ = montar_itens([_oc("A"), _oc("B"), _oc("C")])
    assert len(itens) == 1
    it = itens[0]
    assert it.documento_da_consulta not in it.alvos
    assert it.alvos == frozenset({"A", "B", "C"}) - {it.documento_da_consulta}


def test_o_Item_RECUSA_consulta_entre_os_alvos():
    """A guarda fica no tipo, não só em quem monta."""
    with pytest.raises(ValueError, match="está entre os alvos"):
        Item(consulta=EQ, documento_da_consulta="A",
             alvos=frozenset({"A", "B"}), forma="h1")


def test_forma_em_UM_documento_so_nao_vira_item():
    itens, est = montar_itens([_oc("A"), _oc("A", latex=EQ + " ")])
    assert itens == []
    assert est.descartadas_um_documento_so == 1


def test_forma_COMUM_DEMAIS_e_descartada():
    """⚠️ Uma identidade de livro-texto em mil papers é notação compartilhada, não
    conteúdo que discrimina. O filtro de comprimento já pega quase tudo; este teto
    existe para o corpus inteiro, onde ele passa a morder."""
    ocs = [_oc(f"doc{i}") for i in range(30)]
    itens, est = montar_itens(ocs, max_documentos=20)
    assert itens == []
    assert est.descartadas_comuns_demais == 1
    itens, _ = montar_itens(ocs, max_documentos=40)
    assert len(itens) == 1


def test_a_consulta_e_a_grafia_MAIS_CURTA_e_nao_um_sorteio():
    """⚠️ Determinístico sem semente, e a grafia mais curta é a menos anotada — a
    que menos entrega o alvo por casamento de string. Um sorteio poderia escolher
    a grafia idêntica à do alvo e tornar o item trivial sem ninguém ver."""
    curta = "a=b+c" + "x" * 40
    longa = "a = b + c" + "x" * 60
    itens, _ = montar_itens([
        Ocorrencia("A", "h1", longa), Ocorrencia("B", "h1", curta)])
    assert itens[0].consulta == curta
    assert itens[0].documento_da_consulta == "B"


def test_o_teto_por_DOCUMENTO_limita_o_n_efetivo():
    """⚠️ A Falha 2 do artigo: equações do mesmo paper não são observações
    independentes. Com 35 documentos, tratar 4.117 linhas como independentes deu
    um intervalo 11x estreito demais. Um paper com 400 equações dominaria."""
    ocs = []
    for f in range(10):
        ocs += [Ocorrencia("dominante", f"h{f}", "z" * 50 + "=0"),
                Ocorrencia(f"outro{f}", f"h{f}", "z" * 60 + "=0")]
    itens, est = montar_itens(ocs, max_por_documento=3)
    do_dominante = [i for i in itens if i.documento_da_consulta == "dominante"]
    assert len(do_dominante) <= 3
    assert est.descartadas_teto_por_documento >= 1


def test_a_montagem_e_DETERMINISTICA():
    """Sem semente e sem depender da ordem de iteração de dict."""
    ocs = [_oc("C"), _oc("A"), _oc("B"),
           Ocorrencia("D", "h2", "y" * 50 + "=1"),
           Ocorrencia("E", "h2", "y" * 55 + "=1")]
    a, _ = montar_itens(ocs)
    b, _ = montar_itens(list(reversed(ocs)))
    assert [(i.forma, i.consulta, i.documento_da_consulta) for i in a] == \
           [(i.forma, i.consulta, i.documento_da_consulta) for i in b]


# ── o teto do pool, a lição do T1b2 ─────────────────────────────────────────

def test_o_teto_do_pool_e_1_quando_os_alvos_estao_dentro():
    itens, _ = montar_itens([_oc("A"), _oc("B")])
    assert teto_do_pool(itens, {"A", "B", "C"}) == 1.0
    conferir_pool(itens, {"A", "B", "C"})


def test_conferir_pool_LEVANTA_quando_o_teto_cai():
    """⚠️ O defeito que o T1b2 pagou: um pool com teto 0,7562 comparava modelos
    contra um máximo inalcançável, e a tabela publicada parecia válida."""
    itens, _ = montar_itens([_oc("A"), _oc("B")])
    with pytest.raises(ValueError, match="teto do pool"):
        conferir_pool(itens, {"A", "Z"})


def test_a_mensagem_do_teto_DIZ_quantos_itens_e_quais():
    itens, _ = montar_itens([_oc("A"), _oc("B")])
    try:
        conferir_pool(itens, {"A"})
    except ValueError as e:
        assert "1 de 1" in str(e)
        assert "máximo inalcançável" in str(e)
    else:
        pytest.fail("não levantou")


# ── as estatísticas, que impedem um corte silencioso ────────────────────────

def test_as_estatisticas_DIZEM_o_que_foi_descartado():
    """Sem isto, um corte agressivo demais reduz o benchmark em silêncio e ninguém
    sabe se o número é pequeno por causa do corpus ou do filtro."""
    ocs = [_oc("A"), _oc("B"), Ocorrencia("C", "h9", "z" * 50 + "=0")]
    _, est = montar_itens(ocs)
    d = est.como_dict()
    assert d["ocorrencias"] == 3 and d["formas"] == 2
    assert d["descartadas_um_documento_so"] == 1
    assert d["itens"] == 1


def test_a_nota_REGISTRA_a_licao_da_amostra_pequena():
    """⚠️ Ela quase matou o benchmark, e é contraintuitiva o bastante para
    precisar estar onde alguém a leia — não só no commit."""
    _, est = montar_itens([_oc("A"), _oc("B")])
    nota = est.como_dict()["nota"]
    assert "0,040%" in nota and "0,244%" in nota
    assert "duas pontas na amostra" in nota
    assert "quadraticamente" in nota
