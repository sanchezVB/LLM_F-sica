"""Intervalo de confiança para uma proporção, sem depender de scipy.

## Por que Wilson e não a normal ingênua

O intervalo "normal" `p ± z·√(p(1−p)/n)` erra exatamente onde este projeto precisa
dele: em taxas pequenas e amostras médias. Com `k = 0` ele dá largura **zero** — a
afirmação de que a taxa é 0% com certeza absoluta, a partir de nenhuma observação
contrária. Com `k = 3` em `n = 200` ele desce abaixo de zero.

E é justamente esse o regime das medições daqui: taxa de falso positivo do
classificador (1,5–13,6%), contaminação de benchmark, fração de documentos com
equação mutilada. Um intervalo que colapsa em zero transformaria "não vi nenhum" em
"não existe nenhum".

Wilson (1927) resolve o caso `k = 0` com um limite superior positivo e nunca sai de
[0, 1]. Brown, Cai & DasGupta (2001) o recomendam como padrão para uso geral.

## O que ele NÃO cobre

Independência. Se as observações vierem de linhas do mesmo documento, ou de papers do
mesmo autor, o `n` efetivo é menor que o `n` contado e o intervalo é otimista. Este
repositório já pagou por confundir os dois: `val.head(500)` eram 500 linhas e **35
documentos**, e o intervalo de ±0,159 que isso implicava era largo o bastante para
conter tanto o resultado bonito quanto o honesto.
"""
from __future__ import annotations

import math

# Quantil 0,975 da normal padrão. Literal em vez de `scipy.stats.norm.ppf` para este
# módulo não arrastar scipy — ele é chamado de scripts que rodam na venv leve.
Z_95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Intervalo de Wilson para `k` sucessos em `n` tentativas.

    >>> [round(x, 4) for x in wilson(0, 200)]
    [0.0, 0.0188]
    >>> [round(x, 4) for x in wilson(10, 200)]
    [0.0274, 0.0896]
    """
    if n < 0 or k < 0:
        raise ValueError(f"k={k}, n={n}: contagens não podem ser negativas")
    if k > n:
        raise ValueError(f"k={k} > n={n}: mais sucessos que tentativas")
    if n == 0:
        # Sem observação nenhuma, o intervalo é a reta inteira. Devolver (0, 0)
        # afirmaria uma taxa de zero, que é o erro que este módulo existe para não
        # cometer.
        return (0.0, 1.0)
    p = k / n
    d = 1.0 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    margem = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centro - margem), min(1.0, centro + margem))


def n_para_meia_largura(meia_largura: float, p_esperado: float = 0.05,
                        z: float = Z_95) -> int:
    """Quantas observações para o intervalo ter aproximadamente esta meia-largura.

    Serve para escolher o tamanho da amostra ANTES de olhar os dados, que é a única
    hora em que essa escolha é honesta.

    >>> n_para_meia_largura(0.035, 0.05)
    149
    """
    if not 0 < meia_largura < 1:
        raise ValueError(f"meia_largura={meia_largura} fora de (0, 1)")
    if not 0 <= p_esperado <= 1:
        raise ValueError(f"p_esperado={p_esperado} fora de [0, 1]")
    # Aproximação normal, que basta para dimensionar: n = z²p(1−p)/m².
    return max(1, math.ceil(z * z * p_esperado * (1 - p_esperado)
                            / (meia_largura * meia_largura)))
