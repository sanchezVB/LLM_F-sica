"""As partes puras do protocolo do PB-Formula: estrato primário, pool e escore por documento.

Separadas do script para entrarem na suíte rápida — a mesma divisão que
`benchmarks/formula.py` tem em relação a `montar_pb_formula.py`.

A regra que estas funções servem está em `scripts/avaliar_pb_formula.py`, escrita
antes de qualquer número.
"""
from __future__ import annotations

import numpy as np

# O estrato que o DOC-11 §6.3-medido propõe como primário, e o limiar de
# "documento quase igual". Ver a regra no script.
ESTRATO_PRIMARIO = "notacional"
MAX_FORMAS_EM_COMUM = 5


def itens_primarios(grafia: np.ndarray, formas_em_comum: np.ndarray,
                    estrato: str = ESTRATO_PRIMARIO,
                    max_comum: int = MAX_FORMAS_EM_COMUM) -> np.ndarray:
    """Máscara do conjunto primário: variação notacional E sem documento quase igual.

    ⚠️ Os dois cortes existem por medição, não por gosto. No corpus inteiro, 53,1%
    dos itens têm todos os alvos com a grafia idêntica à da consulta e mais 37,0% só
    diferem em marcação: um casador de string resolve nove de cada dez, e a média
    sobre o benchmark inteiro premiaria exatamente isso. E 36,3% ligam a consulta a
    um documento com o qual ela divide 5+ equações — aí o que se recupera é o
    documento parecido, não a equação.
    """
    return (grafia == estrato) & (formas_em_comum < max_comum)


def escore_por_documento(escores: np.ndarray, doc_de_cada: np.ndarray,
                         n_docs: int) -> np.ndarray:
    """Do escore por EQUAÇÃO para o escore por DOCUMENTO, pelo máximo.

    ⚠️ Máximo, e não média: o documento é relevante porque contém AQUELA equação, e
    a média diluiria o acerto no resto do artigo — um documento de 200 equações
    perderia para um de 3 pela quantidade, que não é o que a tarefa pergunta.
    """
    if escores.ndim == 1:
        escores = escores[None, :]
    saida = np.full((escores.shape[0], n_docs), -np.inf, dtype=np.float32)
    for i in range(escores.shape[0]):
        np.maximum.at(saida[i], doc_de_cada, escores[i])
    return saida


def metricas_por_item(ordem: np.ndarray, alvos: list[set[int]],
                      ks: tuple[int, ...] = (1, 10, 50)) -> dict:
    """recall@k por item (acerta se QUALQUER alvo está no top-k) e o posto do melhor.

    Devolve as séries por item, e não só as médias: o bootstrap pareado precisa
    delas, e uma média sem a série não permite comparar dois sistemas nos mesmos
    itens.
    """
    postos, acertos = [], {k: [] for k in ks}
    for i, alvo in enumerate(alvos):
        if not alvo:
            raise ValueError(f"item {i} sem alvo no pool: o teto não é 1,0")
        posicoes = [int(np.flatnonzero(ordem[i] == d)[0]) + 1 for d in alvo]
        melhor = min(posicoes)
        postos.append(melhor)
        for k in ks:
            acertos[k].append(1.0 if melhor <= k else 0.0)
    p = np.asarray(postos, dtype=np.float64)
    return {"posto": p, "rr": 1.0 / p,
            **{f"recall_{k}": np.asarray(v, dtype=np.float64)
               for k, v in acertos.items()}}
