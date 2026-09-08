"""A sonda de estrutura tensorial: os itens, a régua e o piso.

O DOC-05 §11.2 nomeia "uma sonda de estrutura tensorial" e não a especifica.
`eval/sonda_tensorial.py` é a especificação, e estes testes fixam as três coisas
que a fazem medir estrutura em vez de forma de string:

  1. a sonda é de **dois lados** — separar objetos diferentes E identificar
     renomeação de índice; um lado só é fácil de satisfazer pelo motivo errado;
  2. o **piso de superfície** é computável sem modelo, e é 0 de 72 — então
     qualquer taxa acima de 0,5 é evidência contra a superfície;
  3. empate exato não é acerto, porque empate é o colapso que a sonda pega.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.eval.sonda_tensorial import (  # noqa: E402
    CARREGADORES,
    FAMILIAS,
    Item,
    Resultado,
    itens,
    margem_de_superficie_por_familia,
    piso_de_superficie,
    similaridade_de_superficie,
)


def test_a_SUPERFICIE_reprova_a_sonda_inteira():
    """⚠️ A propriedade que torna a sonda conservadora, e a razão de ela existir.

    `T^{\\mu\\nu}` → `T_{\\mu\\nu}` é UM caractere; `T^{\\mu\\nu}` →
    `T^{\\alpha\\beta}` muda dois símbolos. Por semelhança de string o
    estrutural é o mais parecido — o oposto do que o item pede. Se um dia este
    teste começar a passar com taxa alta, os itens ficaram fáceis pelo motivo
    errado.
    """
    d = piso_de_superficie()
    assert d["itens"] == 72
    assert d["taxa"] == 0.0, (
        f"a superfície passou a acertar {d['taxa']:.0%} dos itens; a sonda "
        "deixou de ser conservadora")
    assert d["margem_media"] < 0
    for f, v in d["por_familia"].items():
        assert v["taxa"] == 0.0, f"a família {f} virou resolvível por string"
        assert v["margem_media"] < 0


def test_o_acaso_fica_ENTRE_os_dois_regimes():
    """A escala de leitura: 0,0 é superfície, 0,5 é moeda, >0,5 é estrutura.

    É o que permite ler o SINAL da distância até 0,5 em vez de precisar de um
    braço de controle para saber de que lado o modelo está.
    """
    assert piso_de_superficie()["taxa"] < 0.5


def test_a_sonda_tem_os_DOIS_lados_em_cada_item():
    """Um item precisa das três expressões. Com duas, um modelo que colapsa tudo
    passaria no lado da invariância e o número teria a cara de uma medição."""
    for it in itens():
        assert it.base != it.estrutural, "sem o lado da separação"
        assert it.base != it.renomeado, "o renomeado não renomeia nada"
        assert it.estrutural != it.renomeado


def test_o_carregador_e_o_MESMO_nos_tres():
    """Se o carregador variasse dentro do trio, a diferença entre os vetores
    incluiria a diferença de contexto e o número não seria sobre a expressão."""
    for it in itens():
        base, estrutural, renomeado = it.textos()
        for texto, expr in ((base, it.base), (estrutural, it.estrutural),
                            (renomeado, it.renomeado)):
            assert expr in texto
            assert texto.replace(expr, "{}") == it.carregador


def test_o_renomeado_preserva_a_ESTRUTURA_de_indice():
    """A invariância que se testa é de NOME de índice, não de estrutura.

    Se o renomeado mudasse `^` para `_` ou o número de índices, ele seria um
    segundo estrutural — e o item passaria a medir outra coisa.
    """
    for it in itens():
        assert it.base.count("^") == it.renomeado.count("^"), it.base
        assert it.base.count("_") == it.renomeado.count("_"), it.base
        assert it.base.count("{") == it.renomeado.count("{"), it.base


def test_todas_as_familias_tem_o_mesmo_peso():
    """Uma família com o dobro de itens dominaria a taxa agregada, e a quebra por
    família não avisaria — ela mostra taxas, não pesos."""
    tamanhos = {f: len(v) for f, v in FAMILIAS.items()}
    assert len(set(tamanhos.values())) == 1, tamanhos
    assert len(itens()) == sum(tamanhos.values()) * len(CARREGADORES)


def test_EMPATE_exato_nao_e_acerto():
    """⚠️ Empate é o modelo dando o mesmo vetor para os três — o colapso que a
    sonda de dois lados existe para pegar. Contá-lo como acerto daria taxa 1,0 a
    um modelo que não distingue nada."""
    r = Resultado()
    r.somar("x", 0.5, 0.5)
    assert r.acertos == [0]
    assert r.margens == [0.0]
    r.somar("x", 0.6, 0.5)
    assert r.acertos == [0, 1]


def test_a_quebra_por_familia_sai_com_taxa_e_margem():
    r = Resultado()
    r.somar("a", 0.9, 0.1)   # acerto, margem +0,8
    r.somar("a", 0.1, 0.9)   # erro,   margem -0,8
    r.somar("b", 0.7, 0.2)   # acerto, margem +0,5
    d = r.como_dict()
    assert d["itens"] == 3
    assert d["taxa"] == pytest.approx(2 / 3, abs=1e-4)
    assert d["por_familia"]["a"]["itens"] == 2
    assert d["por_familia"]["a"]["taxa"] == 0.5
    assert d["por_familia"]["a"]["margem_media"] == 0.0
    assert d["por_familia"]["b"]["taxa"] == 1.0
    assert len(r.acertos) == 3, "o vetor por item existe, para o pareado"


def test_sem_itens_o_resultado_DIZ_isso():
    assert Resultado().como_dict()["erro"]


def test_a_nota_avisa_do_piso():
    r = Resultado()
    r.somar("a", 1.0, 0.0)
    nota = r.como_dict()["nota"]
    assert "DOIS LADOS" in nota
    assert "SUPERF" in nota.upper()


def test_a_similaridade_de_superficie_e_bem_comportada():
    assert similaridade_de_superficie("abc", "abc") == pytest.approx(1.0)
    assert similaridade_de_superficie("", "abc") == 0.0
    assert 0.0 <= similaridade_de_superficie(r"T^{\mu}", r"T_{\mu}") <= 1.0


def test_a_DIFICULDADE_de_cada_familia_sai_junto_da_taxa():
    """⚠️ Uma família presa no zero é lida como "o modelo não sabe isso" quando
    pode ser "a superfície empurra forte demais nesta família".

    Calibrado em 2026-09-08: as duas famílias com margem de superfície branda
    (−0,039 e −0,046) separaram quatro encoders de 0,00 a 0,50; as duas com
    margem forte (−0,078 e −0,152) ficaram no zero para todos. A ordem foi exata,
    e a margem sai das strings — sem modelo.
    """
    d = margem_de_superficie_por_familia()
    assert set(d) == set(FAMILIAS)
    assert all(v < 0 for v in d.values()), d
    # A ordem medida na calibração, que é o que dá sentido à leitura por família.
    assert d["posicao_do_indice"] > d["operador_contra_campo"], d
    assert d["valencia"] > d["operador_contra_campo"], d
    assert d["contracao"] == min(d.values()), d

    r = Resultado()
    r.somar("valencia", 0.9, 0.1)
    assert r.como_dict()["por_familia"]["valencia"]["margem_de_superficie"] == (
        d["valencia"])


def test_familias_ambiguas_ficaram_de_FORA():
    """⚠️ `g_{\\mu\\nu}` contra `g_{\\nu\\mu}`: a métrica é simétrica, então são o
    MESMO objeto e o item não tem resposta certa. Um item discutível vira ruído
    com cara de medida, e este teste é o que impede alguém de "enriquecer" a
    sonda com trocas de ordem de índices de mesma valência."""
    for it in itens():
        assert not (sorted(it.base) == sorted(it.estrutural)
                    and it.familia != "operador_contra_campo"), (
            f"{it.base} e {it.estrutural} são anagramas — provável troca de "
            "ordem de índices, cuja resposta certa depende da simetria do tensor")


def test_o_item_e_hasheavel_e_congelado():
    """Os itens entram em conjuntos e chaves de cache no script do modelo."""
    a = itens()[0]
    assert isinstance(a, Item)
    assert len({a, a}) == 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.base = "outro"  # type: ignore[misc]
