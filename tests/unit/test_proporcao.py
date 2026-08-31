"""O intervalo de Wilson, e o caso `k = 0` que é a razão de ele existir aqui.

A tentação é usar `p ± z·√(p(1−p)/n)`, que é a fórmula que todo mundo lembra. Ela
erra exatamente no regime das medições deste projeto — taxas pequenas, amostras
médias — e o erro tem a forma pior possível: com zero observações contrárias ela dá
largura **zero**, transformando "não vi nenhum" em "não existe nenhum".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.eval.statistics.proporcao import (  # noqa: E402
    n_para_meia_largura,
    wilson,
)


def test_zero_sucessos_nao_da_intervalo_de_largura_zero():
    """A razão de ser deste módulo.

    A normal ingênua daria [0, 0] — certeza absoluta de taxa nula a partir de
    nenhuma evidência contrária. Wilson dá um teto positivo, e é ele que permite
    dizer "a taxa é menor que 1,9%, não que ela é zero".
    """
    baixo, alto = wilson(0, 200)
    # `approx` e nao `== 0.0`: com k=0 o limite inferior e zero na algebra e
    # 1,7e-18 em ponto flutuante. Exigir igualdade exata aqui seria testar o modo
    # de arredondamento da FPU, nao a estatistica.
    assert baixo == pytest.approx(0.0, abs=1e-12)
    assert alto > 0.0
    assert alto == pytest.approx(0.0188, abs=1e-4)


def test_todos_sucessos_tambem_nao_colapsa():
    baixo, alto = wilson(200, 200)
    assert alto == 1.0
    assert baixo < 1.0


def test_nunca_sai_de_zero_um():
    """A normal ingênua sai. Wilson não pode, para nenhum (k, n)."""
    for n in (1, 3, 7, 20, 200, 5000):
        for k in (0, 1, n // 2, n - 1, n):
            if not 0 <= k <= n:
                continue
            baixo, alto = wilson(k, n)
            assert 0.0 <= baixo <= alto <= 1.0, (k, n, baixo, alto)


def test_mais_dados_aperta_o_intervalo():
    """Na mesma taxa observada, dobrar o n tem de estreitar."""
    larguras = [wilson(k, n)[1] - wilson(k, n)[0]
                for k, n in ((5, 100), (10, 200), (20, 400), (50, 1000))]
    assert larguras == sorted(larguras, reverse=True), larguras


def test_e_simetrico_no_complemento():
    """`wilson(k, n)` e `wilson(n−k, n)` têm de ser reflexos em torno de 0,5.

    Não é decoração: uma implementação que erra um sinal quebra essa simetria, e o
    erro passaria despercebido em taxas pequenas, que é onde este módulo é usado.
    """
    for k, n in ((3, 40), (17, 91), (0, 12)):
        b1, a1 = wilson(k, n)
        b2, a2 = wilson(n - k, n)
        assert b1 == pytest.approx(1 - a2, abs=1e-12)
        assert a1 == pytest.approx(1 - b2, abs=1e-12)


def test_sem_observacao_nenhuma_o_intervalo_e_a_reta_inteira():
    """Devolver (0, 0) para n=0 afirmaria uma taxa; (0, 1) afirma ignorância."""
    assert wilson(0, 0) == (0.0, 1.0)


def test_contagens_impossiveis_levantam():
    """Silenciar `k > n` daria um número plausível a partir de dados corrompidos."""
    with pytest.raises(ValueError, match="mais sucessos que tentativas"):
        wilson(11, 10)
    with pytest.raises(ValueError, match="negativas"):
        wilson(-1, 10)


def test_dimensionar_a_amostra_bate_com_o_intervalo():
    """`n_para_meia_largura` serve para escolher o n ANTES de ver os dados.

    O contrato é frouxo de propósito — ela usa a aproximação normal, que difere de
    Wilson —, mas não pode estar errada por muito: um `n` que dá o dobro da largura
    pedida faria alguém amostrar metade do necessário.
    """
    n = n_para_meia_largura(0.035, p_esperado=0.05)
    baixo, alto = wilson(round(0.05 * n), n)
    assert (alto - baixo) / 2 == pytest.approx(0.035, abs=0.012)


def test_dimensionar_cresce_quando_se_pede_precisao():
    assert (n_para_meia_largura(0.01) > n_para_meia_largura(0.02)
            > n_para_meia_largura(0.05))


def test_dimensionar_rejeita_pedido_impossivel():
    with pytest.raises(ValueError, match="meia_largura"):
        n_para_meia_largura(0.0)
    with pytest.raises(ValueError, match="p_esperado"):
        n_para_meia_largura(0.03, p_esperado=1.5)
