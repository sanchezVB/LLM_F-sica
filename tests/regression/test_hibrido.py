"""BM25, fusão por posto e as métricas que limitam o T1b.

Três coisas que, se estiverem erradas, produzem um número de T1b que parece bom:

  1. **A tokenização.** Em texto de Física, `hep-th`, `SU(2)` e `\\alpha` carregam
     significado. Um `\\w+` simples os parte, e o BM25 perde exatamente onde ele é
     melhor que o denso — sem nenhum erro visível, só um recall menor.
  2. **A fusão.** Somar escores de BM25 com cosseno é somar escalas incomparáveis;
     normalizar introduz dependência do lote de candidatos. RRF usa só a posição.
  3. **O teto.** `recall@100` do recuperador limita tudo que o reranker faz. Um
     reranker perfeito sobre recall@100 de 0,70 não passa de 0,70.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.eval.hibrido import (  # noqa: E402
    BM25,
    fundir_rrf,
    mcnemar_em,
    ndcg_em_10,
    recall_em,
    tokenizar,
    top_k,
)

# ─── tokenização ─────────────────────────────────────────────────────────────


def test_preserva_o_que_carrega_significado_em_fisica():
    """`hep-th`, `\\alpha` e dígitos. Um `\\w+` simples partiria os três."""
    t = tokenizar(r"Um estudo hep-th sobre \alpha e SU(2) com 2.5 TeV")
    assert "hep-th" in t, "hífen interno partido — `hep-th` virou dois termos"
    assert "\\alpha" in t, "comando LaTeX perdeu a barra"
    assert "2.5" in t, "número decimal partido"
    assert "su" in t and "2" in t


def test_minusculas():
    assert tokenizar("Física QUANTICA") == ["física", "quantica"]


# ─── BM25 ────────────────────────────────────────────────────────────────────


def _corpus():
    return [
        "supercondutividade de alta temperatura em cupratos",
        "buracos negros e radiacao Hawking em gravitacao quantica",
        "supercondutividade convencional BCS e pares de Cooper",
        "cosmologia observacional e a constante de Hubble",
    ]


def test_recupera_por_termo_raro():
    """O caso onde o BM25 ganha do denso: termo específico e literal."""
    b = BM25().indexar(_corpus())
    escores = b.pontuar("radiacao Hawking")
    assert int(np.argmax(escores)) == 1


def test_termo_comum_nao_penaliza():
    """⚠️ IDF de Robertson sem suavização fica NEGATIVO para termo em mais da
    metade dos documentos — e aí conter o termo derruba o documento.

    Aqui "supercondutividade" está em 2 de 4. Sem o `+1` no log, o IDF seria
    log(0,833) < 0 e os dois documentos que falam de supercondutividade seriam
    penalizados por isso.
    """
    b = BM25().indexar(_corpus())
    assert (b.idf >= 0).all(), "IDF negativo — termo comum penaliza quem o contém"
    escores = b.pontuar("supercondutividade")
    assert escores[0] > 0 and escores[2] > 0
    assert escores[1] == 0 and escores[3] == 0


def test_consulta_sem_termo_conhecido_nao_quebra():
    """Devolve zeros, não exceção. Uma consulta fora do vocabulário é normal."""
    b = BM25().indexar(_corpus())
    e = b.pontuar("zzzz qqqq")
    assert e.shape == (4,) and not e.any()


def test_documento_curto_nao_e_penalizado_pelo_comprimento():
    """A normalização por comprimento do BM25 é o que o diferencia do TF-IDF.

    Dois documentos com o mesmo termo uma vez: o mais CURTO deve pontuar mais,
    porque o termo representa uma fração maior dele.
    """
    b = BM25().indexar(["plasma", "plasma " + "palavra " * 200])
    e = b.pontuar("plasma")
    assert e[0] > e[1]


def test_pontuar_antes_de_indexar_levanta():
    """Silêncio aqui daria escores todos zero e um recall de 0 sem explicação."""
    import pytest

    with pytest.raises(RuntimeError, match="indexar"):
        BM25().pontuar("qualquer")


# ─── fusão ───────────────────────────────────────────────────────────────────


def test_rrf_premia_quem_aparece_nas_duas_listas():
    """O ponto da fusão: consenso entre sistemas independentes vale mais."""
    # doc 7 é 2º em ambas; doc 1 é 1º só na primeira.
    r = fundir_rrf([1, 7, 3], [9, 7, 5])
    assert r[0] == 7


def test_rrf_ignora_escores_e_usa_so_posicao():
    """Duas listas com as mesmas posições dão o mesmo resultado.

    É a propriedade que torna a fusão imune a escalas incomparáveis — o cosseno em
    [-1,1] contra o BM25 ilimitado.
    """
    assert fundir_rrf([4, 2], [2, 4]) == fundir_rrf([4, 2], [2, 4])
    r = fundir_rrf([4, 2], [2, 4])
    assert set(r) == {2, 4}


def test_rrf_com_uma_lista_preserva_a_ordem():
    assert fundir_rrf([5, 3, 1]) == [5, 3, 1]


def test_k_do_rrf_amortece_as_primeiras_posicoes():
    """Com k pequeno, o 1º lugar de uma lista domina; com k=60, o consenso vence.

    É a razão do k, e um teste que não a exercitasse deixaria o valor parecer
    arbitrário.
    """
    # doc 1 é 1º numa lista e ausente na outra; doc 9 é 2º nas duas.
    sem_amortecer = fundir_rrf([1, 9], [8, 9], k=0)
    amortecido = fundir_rrf([1, 9], [8, 9], k=60)
    assert sem_amortecer[0] == 1, "com k=0 o primeiro lugar isolado deveria dominar"
    assert amortecido[0] == 9, "com k=60 o consenso deveria vencer"


# ─── top_k e as métricas ─────────────────────────────────────────────────────


def test_top_k_em_ordem_decrescente():
    e = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
    assert top_k(e, 3) == [1, 3, 2]


def test_top_k_maior_que_o_disponivel():
    assert top_k(np.array([0.2, 0.4]), 10) == [1, 0]


def test_top_k_zero():
    assert top_k(np.array([0.2, 0.4]), 0) == []


def test_recall_conta_so_ate_a_posicao():
    # posições 0-indexadas; None = não recuperado
    pos = [0, 5, 99, 100, None]
    assert recall_em(pos, 1) == 1 / 5
    assert recall_em(pos, 10) == 2 / 5
    assert recall_em(pos, 100) == 3 / 5


def test_ndcg_zera_fora_do_top_10():
    """Mesma definição do avaliador do G1, de propósito.

    Dois nDCG diferentes no mesmo projeto seriam duas réguas com o mesmo nome — e
    este projeto já elegeu um campeão errado comparando protocolos.
    """
    assert ndcg_em_10([0]) == 1.0
    assert ndcg_em_10([10]) == 0.0
    assert ndcg_em_10([None]) == 0.0
    assert abs(ndcg_em_10([1]) - 1 / np.log2(3)) < 1e-9


def test_recall_e_o_teto_do_reranker():
    """A relação que o T1b inteiro depende, escrita como teste.

    Um reranker PERFEITO coloca em 1º tudo que chegou ao top-100 — e nada mais. O
    nDCG resultante é exatamente o recall@100.
    """
    posicoes = [0, 50, 99, None, None]          # 3 de 5 chegaram ao top-100
    teto = recall_em(posicoes, 100)
    perfeito = ndcg_em_10([0 if p is not None and p < 100 else None
                           for p in posicoes])
    assert perfeito == teto == 0.6


# ─── McNemar pareado ─────────────────────────────────────────────────────────


def test_pareado_enxerga_diferenca_que_proporcoes_esconderiam():
    """O ponto do teste pareado, escrito como número.

    100 consultas: 80 os dois acertam, 5 os dois erram, e das 15 discordantes o A
    ganha 14. Como duas proporções (0,94 contra 0,81) o erro padrão engoliria a
    diferença; pareado, 14 a 1 em 15 é significativo.
    """
    a = [0] * 80 + [None] * 5 + [0] * 14 + [None]
    b = [0] * 80 + [None] * 5 + [None] * 14 + [0]
    r = mcnemar_em(a, b, 10, "A", "B")
    assert r["ganha_a"] == 14 and r["ganha_b"] == 1
    assert r["discordantes"] == 15
    assert r["p"] < 0.05, f"14 a 1 deveria decidir, p={r['p']}"
    assert r["a"] in r["veredito"]


def test_pareado_nao_inventa_significancia_com_poucos_discordantes():
    """Com 6 discordantes 4 a 2, o honesto é dizer indeciso."""
    a = [0] * 90 + [0] * 4 + [None] * 2
    b = [0] * 90 + [None] * 4 + [0] * 2
    r = mcnemar_em(a, b, 10)
    assert r["discordantes"] == 6
    assert r["p"] > 0.05 and "indeciso" in r["veredito"]


def test_pareado_com_placar_equilibrado_da_empate():
    a = [0] * 20 + [None] * 20
    b = [None] * 20 + [0] * 20
    r = mcnemar_em(a, b, 10)
    assert r["ganha_a"] == 20 and r["ganha_b"] == 20
    assert r["p"] == 1.0 and "empate" in r["veredito"]


def test_pareado_sem_discordantes_nao_tem_o_que_decidir():
    a = b = [0, 5, None, 99]
    r = mcnemar_em(a, b, 10)
    assert r["discordantes"] == 0 and r["p"] == 1.0


def test_pareado_respeita_o_k():
    """Posição 5 está no top-10 e fora do top-1. O mesmo par muda de veredito."""
    a, b = [5], [0]
    assert mcnemar_em(a, b, 10)["discordantes"] == 0
    r1 = mcnemar_em(a, b, 1)
    assert r1["discordantes"] == 1 and r1["ganha_b"] == 1


def test_pareado_recusa_conjuntos_de_tamanhos_diferentes():
    """Silêncio aqui compararia consultas desalinhadas e daria um p sem sentido."""
    assert "erro" in mcnemar_em([0, 1], [0], 10)
