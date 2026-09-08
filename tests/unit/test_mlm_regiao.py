"""A medida que discrimina a hipótese do DOC-07 §2.3, e o que a rigaria.

A hipótese é que mascarar equações INTEIRAS ensina o modelo a fechar uma equação,
não a completar um símbolo isolado. Se ela vale, o braço tratado ganha nos tokens
de equação e empata na prosa — então a medida tem de separar as duas regiões, e um
número agregado não distinguiria nada.

⚠️ E há uma forma de rigar isso que parece natural: avaliar com a política de
máscara do braço TRATADO (equação inteira para `[MASK]`). Ela testaria o tratado
na distribuição em que ele treinou e o controle numa estranha a ele — o tratado
ganharia por construção e o número teria a cara de uma medição.

Estes testes fixam que a máscara da avaliação é neutra, que os dois braços vêem as
MESMAS posições, e que a atribuição de acerto a região não pode se desalinhar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.eval.mlm_regiao import (  # noqa: E402
    FRACAO_MASCARA,
    Contagem,
    fracao_de_equacao,
    posicoes_mascaradas,
)


def test_a_mascara_da_avaliacao_IGNORA_as_marcas():
    """⚠️ A asserção central. Ver a docstring do módulo.

    `posicoes_mascaradas` nem recebe as marcas — não é que ela as ignore por
    disciplina, é que não tem como olhá-las. Uma assinatura que aceitasse as
    marcas convidaria alguém a "melhorar" a avaliação usando-as.
    """
    import inspect

    params = set(inspect.signature(posicoes_mascaradas).parameters)
    assert "marcas" not in params and "bit_math" not in params, (
        "a escolha das posições passou a poder ver as marcas de equação; é assim "
        "que a avaliação vira a política de treino do braço tratado")
    assert params == {"n_tokens", "semente", "indice", "fracao", "proibidas"}


def test_os_dois_bracos_veem_as_MESMAS_posicoes():
    """Sem isto, comparar dois modelos é comparar duas proporções soltas.

    Com as mesmas máscaras, o McNemar pareado sobre "acertou este token" enxerga
    o que duas taxas não enxergam — é o mesmo argumento que o resto do projeto já
    usa em `eval/statistics`.
    """
    a = posicoes_mascaradas(8192, semente=17, indice=42)
    b = posicoes_mascaradas(8192, semente=17, indice=42)
    assert np.array_equal(a, b)
    # E sequências diferentes recebem máscaras diferentes.
    c = posicoes_mascaradas(8192, semente=17, indice=43)
    assert not np.array_equal(a, c)


def test_a_semente_e_o_indice_nao_colidem_por_soma():
    """⚠️ `default_rng(semente + indice)` faria (17,5) e (18,4) darem a MESMA
    máscara, e sequências vizinhas receberiam máscaras correlacionadas."""
    a = posicoes_mascaradas(4096, semente=17, indice=5)
    b = posicoes_mascaradas(4096, semente=18, indice=4)
    assert not np.array_equal(a, b), (
        "a semente e o índice estão sendo somados; pares diferentes colidem")


def test_a_fracao_mascarada_e_a_declarada():
    n = 10_000
    p = posicoes_mascaradas(n, semente=17, indice=0)
    assert p.size == round(n * FRACAO_MASCARA)
    assert p.size == np.unique(p).size, "posição repetida gastaria orçamento"
    assert np.all(np.diff(p) > 0), "as posições saem ordenadas"


def test_posicoes_proibidas_ficam_de_fora():
    """Mascarar preenchimento gasta orçamento em posições que nenhum modelo tem
    como acertar: a taxa cai igual nos dois braços, com ruído a mais."""
    proibidas = np.arange(0, 500, dtype=np.int64)
    p = posicoes_mascaradas(1000, semente=17, indice=0, proibidas=proibidas)
    assert p.size > 0
    assert not np.isin(p, proibidas).any()
    # Com quase tudo proibido, ainda devolve algo em vez de estourar.
    q = posicoes_mascaradas(10, semente=17, indice=0,
                            proibidas=np.arange(9, dtype=np.int64))
    assert q.size == 1
    # E com TUDO proibido devolve vazio, sem levantar.
    assert posicoes_mascaradas(
        5, semente=17, indice=0, proibidas=np.arange(5, dtype=np.int64)).size == 0


def test_a_contagem_separa_as_regioes_e_calcula_a_vantagem():
    c = Contagem()
    # 4 tokens de equação, 3 acertos; 6 de prosa, 3 acertos.
    c.somar(certo=np.array([1, 1, 1, 0, 1, 1, 1, 0, 0, 0]),
            em_equacao=np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]))
    d = c.como_dict()
    assert d["tokens_equacao"] == 4 and d["tokens_prosa"] == 6
    assert d["acuracia_equacao"] == pytest.approx(0.75)
    assert d["acuracia_prosa"] == pytest.approx(0.5)
    assert d["acuracia_total"] == pytest.approx(0.6)
    assert d["vantagem_em_equacao"] == pytest.approx(0.25)
    # E o vetor por token existe, para o pareado.
    assert len(c.acertos) == 10 and len(c.e_equacao) == 10


def test_formas_diferentes_LEVANTAM():
    """Um desalinhamento atribuiria o acerto à região errada — e o número sairia
    com a cara certa, que é o pior tipo de erro nesta medida."""
    c = Contagem()
    with pytest.raises(ValueError, match="região errada|regiao errada"):
        c.somar(certo=np.array([1, 0, 1]), em_equacao=np.array([1, 0]))


def test_a_nota_diz_que_o_teste_e_CONSERVADOR_para_a_hipotese():
    """Se o ganho aparecesse só com máscara por span, seria memorização de
    formato e não capacidade — e quem lê o resultado precisa saber que a prova
    foi feita do lado difícil."""
    d = Contagem()
    d.somar(np.array([1]), np.array([1]))
    nota = d.como_dict()["nota"]
    assert "UNIFORME" in nota
    assert "CONSERVADOR" in nota
    assert "por construção" in nota


def test_fracao_de_equacao_avisa_quando_nao_ha_equacao():
    """Se a avaliação rodar sobre texto sem equação, a diferença entre regiões
    não pode aparecer por mais correta que a hipótese seja."""
    assert fracao_de_equacao(np.zeros(100, dtype=np.uint8), bit_math=1) == 0.0
    marcas = np.zeros(100, dtype=np.uint8)
    marcas[:37] = 1
    assert fracao_de_equacao(marcas, bit_math=1) == pytest.approx(0.37)
    assert fracao_de_equacao(np.empty(0, dtype=np.uint8), bit_math=1) == 0.0
