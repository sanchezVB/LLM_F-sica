"""Quais checkpoints o SISTEMA usa — num lugar só.

## ⚠️ Por que isto existe

Até 2026-09-08 o recuperador do sistema era a string
`models/phiemb-minilm-melhor` **copiada como default em quatro scripts**
(`avaliar_t1b.py`, `minerar_do_recuperador.py`, `minerar_negativos.py`) e o
reranqueador era `models/phirank-physbert-melhor` copiado em outros dois. Trocar o
recuperador significava editar quatro lugares e acertar todos.

E a armadilha específica: **há um quinto lugar que NÃO deve mudar.** O
`avaliar_encoders.py` lista o `phiemb-minilm-melhor` entre os candidatos do G1 —
lá ele é um **ponto da curva de volume**, evidência de uma medição passada, e não
"o recuperador do sistema". Trocar aquele caminho junto com os outros apagaria uma
comparação.

Uma constante torna essa diferença explícita: quem quer o recuperador do sistema
importa daqui; quem quer um checkpoint específico escreve o caminho e diz por quê.

## O que este módulo NÃO faz

Não escolhe o melhor modelo, não mede nada, e não é lido em tempo de treino. Ele
declara o que está **instalado**, e a instalação passa por
`scripts/instalar_phiemb.py` / `scripts/instalar_phirank.py`, que gravam o
manifesto de etapa com a medição que a justifica.

Sem torch, sem polars, sem rede: é só o contrato de nomes.
"""

from __future__ import annotations

# O recuperador denso do sistema.
#
# ⚠️ Este NÃO é o melhor ΦEmb medido. Em 2026-09-07 o `phiemb-minilm-3m-sorteado-
# melhor` deu nDCG@10 0,6026 contra 0,5246 deste — +0,078 — e ainda assim o
# sistema aponta para o antigo, de propósito: a referência do T1b/T1c (nDCG 0,1666
# do ΦRank, p=0,0062) foi medida com ESTE recuperador, e trocar os dois ao mesmo
# tempo mediria duas coisas. A troca é uma etapa deliberada, com remedição da
# cadeia depois — ver o §"a decisão que isto abre" no ESTADO.md.
RECUPERADOR = "models/phiemb-minilm-melhor"

# O reranqueador do sistema. Entrou em 2026-09-03: PhysBERT vence a fusão RRF
# (nDCG 0,1666 contra 0,1576, p=0,0062), e o `gte-base`, do mesmo tamanho, empata
# (p=0,637) — o mecanismo é pré-treino em Física, não diversidade de base.
RERANQUEADOR = "models/phirank-physbert-melhor"
