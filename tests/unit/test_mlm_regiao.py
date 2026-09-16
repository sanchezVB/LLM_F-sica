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
    diferenca_das_diferencas,
    diferenca_de_acuracia,
    equacao_da_prova,
    fracao_de_equacao,
    ler_pela_regra,
    posicoes_mascaradas,
    sequencias_sorteadas,
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


def test_a_LINHA_DE_BASE_sem_tratamento_acompanha_o_resultado():
    """⚠️ A vantagem em equação já é grande SEM tratamento nenhum.

    Medido em 2026-09-10, primeira execução da medida num modelo real: o
    ModernBERT-base, que nunca viu mascaramento consciente de equação, dá
    acurácia 0,8765 em equação contra 0,7480 em prosa — **+0,1286**.

    LaTeX é redundante: fechado um `\frac{`, o `}{` e o `}` vêm quase de graça.
    Então uma `vantagem_em_equacao` positiva no braço tratado **não é evidência**
    da hipótese do DOC-07 §2.3 — o que a testa é a diferença ENTRE braços.

    Sem este número ao lado, alguém lê +0,13 como sucesso do mascaramento por
    span quando é propriedade do formato.
    """
    c = Contagem()
    c.somar(certo=np.array([1, 1, 0, 1]), em_equacao=np.array([1, 1, 0, 0]))
    d = c.como_dict()
    assert d["linha_de_base_sem_tratamento"] == pytest.approx(0.1286)
    assert "DIFERENÇA ENTRE BRAÇOS" in d["nota"]
    assert "0,1286" in d["nota"]


# ── a ablação do §2.3 (2026-09-16) ──────────────────────────────────────────



def _braco(rng, n_seq, acc_eq, acc_pr, n_eq=40, n_pr=60):
    """`por_sequencia` sintético com acurácias verdadeiras conhecidas."""
    return [[i, int(rng.binomial(n_eq, acc_eq)), n_eq,
             int(rng.binomial(n_pr, acc_pr)), n_pr] for i in range(n_seq)]


def test_as_sequencias_sao_SORTEADAS_e_nao_um_prefixo():
    """⚠️ `range(n)` media o começo da fatia — amostra por conglomerado."""
    idx = sequencias_sorteadas(50_000, 2_000, 17)
    assert idx.tolist() != list(range(2_000))
    assert idx.max() > 40_000, "um sorteio de 2.000 em 50.000 não fica no começo"
    assert (np.diff(idx) > 0).all()
    assert idx.tolist() == sequencias_sorteadas(50_000, 2_000, 17).tolist()
    with pytest.raises(ValueError, match="reduza"):
        sequencias_sorteadas(10, 11, 17)


def test_o_registro_por_sequencia_soma_os_totais():
    c = Contagem()
    c.somar(np.array([1, 0, 1, 1]), np.array([1, 1, 0, 0]), indice=7)
    c.somar(np.array([0, 1]), np.array([0, 1]), indice=9)
    assert c.por_sequencia == [[7, 1, 2, 2, 2], [9, 1, 1, 0, 1]]
    assert sum(r[2] for r in c.por_sequencia) == c.total_equacao
    assert sum(r[4] for r in c.por_sequencia) == c.total_prosa


def test_melhorar_TUDO_por_igual_NAO_e_evidencia_do_2_3():
    """⚠️ A lição da correção da regra. Token de equação já é mais fácil sem
    tratamento; um tratado que ganha +0,05 em equação E em prosa não ensinou a
    relação prosa–fórmula, e a quantidade tem de sair ZERO — com o IC cruzando."""
    rng = np.random.default_rng(1)
    controle = _braco(rng, 800, 0.80, 0.70)
    tratado = _braco(rng, 800, 0.85, 0.75)
    r = diferenca_das_diferencas(controle, tratado, n_boot=2_000)
    assert r["cruza_zero"], r
    assert abs(r["diferenca_das_diferencas"]) < 0.02
    # e no entanto a acurácia em equação SUBIU — o que a regra antiga leria
    assert r["tratado"]["acuracia_equacao"] > r["controle"]["acuracia_equacao"] + 0.03


def test_ganhar_SO_em_equacao_sai_positivo_com_IC_acima_de_zero():
    rng = np.random.default_rng(2)
    controle = _braco(rng, 800, 0.80, 0.70)
    tratado = _braco(rng, 800, 0.86, 0.70)
    r = diferenca_das_diferencas(controle, tratado, n_boot=2_000)
    assert r["ic95"][0] > 0, r
    assert 0.03 < r["diferenca_das_diferencas"] < 0.09


def test_o_ponto_e_a_diferenca_das_VANTAGENS_agregadas():
    """A mesma conta de `Contagem.como_dict`, e não média por sequência."""
    controle = [[0, 9, 10, 1, 10], [1, 50, 100, 30, 100]]
    tratado = [[0, 10, 10, 1, 10], [1, 60, 100, 30, 100]]
    r = diferenca_das_diferencas(controle, tratado, n_boot=200)
    van_c = 59 / 110 - 31 / 110
    van_t = 70 / 110 - 31 / 110
    assert r["diferenca_das_diferencas"] == round(van_t - van_c, 5)


def test_bracos_com_sequencias_DIFERENTES_sao_recusados():
    rng = np.random.default_rng(3)
    controle = _braco(rng, 20, 0.8, 0.7)
    tratado = [[i + 1, *linha[1:]] for i, linha in enumerate(_braco(rng, 20, 0.8, 0.7))]
    with pytest.raises(ValueError, match="MESMAS sequências"):
        diferenca_das_diferencas(controle, tratado, n_boot=100)


def test_o_bootstrap_e_PAREADO_as_mesmas_reamostras_nos_dois_bracos():
    """Braços idênticos dão diferença das diferenças EXATAMENTE zero em toda
    reamostra — só acontece se as reamostras forem as mesmas nos dois."""
    rng = np.random.default_rng(4)
    b = _braco(rng, 100, 0.8, 0.7)
    r = diferenca_das_diferencas(b, [list(x) for x in b], n_boot=500)
    assert r["ic95"] == [0.0, 0.0]


def test_diferenca_de_acuracia_da_checagem_2():
    rng = np.random.default_rng(5)
    controle = [[i, int(rng.binomial(30, 0.3)), 30] for i in range(500)]
    tratado = [[i, int(rng.binomial(30, 0.5)), 30] for i in range(500)]
    r = diferenca_de_acuracia(controle, tratado, n_boot=1_000)
    assert r["ic95"][0] > 0.15


def test_a_regra_NAO_le_a_primaria_se_uma_checagem_reprova():
    primaria = {"ic95": [0.01, 0.05]}
    boa = {"diferenca": 0.2, "ic95": [0.15, 0.25]}
    ruim = {"diferenca": 0.01, "ic95": [-0.01, 0.03]}
    assert ler_pela_regra(primaria, ruim, 0.54)["desfecho"] == "RUN INVÁLIDO"
    assert ler_pela_regra(primaria, boa, 0.45)["desfecho"] == "RUN INVÁLIDO"
    assert ler_pela_regra(primaria, boa, 0.54)["desfecho"] == "TRATADO À FRENTE"


def test_os_TRES_desfechos_e_o_empate_NAO_e_o_negativo():
    boa = {"diferenca": 0.2, "ic95": [0.15, 0.25]}
    assert ler_pela_regra({"ic95": [-0.05, -0.01]}, boa, 0.54)["desfecho"] == "CONTROLE À FRENTE"
    empate = ler_pela_regra({"ic95": [-0.01, 0.02]}, boa, 0.54)
    assert empate["desfecho"] == "NÃO DECIDIDO"
    assert "NÃO é o negativo" in empate["leitura"]


def test_a_equacao_da_prova_e_INTEIRA_e_deterministica():
    n = 128
    ids = np.arange(5, 5 + n, dtype=np.int64)
    ide = np.full(n, -1, dtype=np.int32)
    disp = np.zeros(n, dtype=bool)
    ide[40:70], disp[40:70] = 0, True       # 30 tokens: cabe em 38 e passa de 20
    ide[80:85], disp[80:85] = 1, True       # 5 tokens: abaixo do piso
    pos = equacao_da_prova(ide, disp, ids, {0, 1, 2, 3, 4}, 17, 9)
    assert pos.tolist() == list(range(40, 70))
    assert pos.tolist() == equacao_da_prova(ide, disp, ids, {0, 1, 2, 3, 4}, 17, 9).tolist()


def test_a_equacao_da_prova_e_VAZIA_sem_display_elegivel():
    n = 128
    ids = np.arange(5, 5 + n, dtype=np.int64)
    ide = np.full(n, -1, dtype=np.int32)
    ide[40:70] = 0                           # matemática, mas inline
    assert equacao_da_prova(ide, np.zeros(n, dtype=bool), ids, {0, 1, 2, 3, 4},
                            17, 0).size == 0
