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
# Trocado em 2026-09-08, depois da curva de volume da T1a: o run de 6 M de pares
# sorteados (650.162 documentos citados) dá nDCG@10 **0,6223** no protocolo de teto
# 1,0, contra 0,5246 do `phiemb-minilm-melhor` que estava aqui — **+0,098**. Supera
# o GTE-large de 335M em todas as quatro métricas, a 1/14,8 dos parâmetros.
#
# ⚠️ Instalado em caminho NOVO, e não sobre o antigo. O `phiemb-minilm-melhor`
# continua em `models/` porque é um PONTO DA CURVA em `avaliar_encoders.py` —
# sobrescrevê-lo apagaria a evidência de que 400 mil pares sobre MiniLM dão 0,5246.
#
# A cadeia foi remedida em 2026-09-08 (T1b2) e o BM25 saiu; em 2026-09-10 (T1e) o
# reranqueador saiu também. **A composição do sistema é o ΦEmb servindo o top-10.**
RECUPERADOR = "models/phiemb-do-sistema"

# ⚠️ O ΦRank NÃO está mais na composição. Isto é o checkpoint do T1c.
#
# Ele entrou no sistema em 2026-09-03 com evidência boa: PhysBERT vence a fusão RRF
# (nDCG 0,1666 contra 0,1576, p=0,0062) e o `gte-base`, do mesmo tamanho, empata
# (p=0,637) — o mecanismo é pré-treino em Física, e esse resultado continua de pé.
#
# O que caiu foi o estágio, não a base. Depois de o recuperador melhorar 0,098
# (T1a 6 M), o reranqueador deixou de acrescentar à cadeia, e cinco medições
# pareadas concordam:
#
#     T1d  ΦRank antigo   @100   p=0,139   empate contra o ΦEmb sozinho
#     T1d  ΦRank retreinado @100 p=0,379   empate
#     T1e  @50                   p=0,166   empate
#     T1e  @100                  p=0,139   empate
#     T1e  @200                  p=0,138   empate
#
# E o bootstrap pareado sobre o nDCG@10 por consulta — que é o teste que casa com
# a pergunta, porque o McNemar mede pertencimento e é cego para ordenação — dá
# +0,0091 com IC 95% de [−0,0010, +0,0191]: também não estabelece.
#
# O mecanismo, medido no T1e: dobrar o conjunto de candidatos de 100 para 200 deu
# ao reranqueador 195 alvos novos e ele trouxe **+1** para o top-10. E das 421
# consultas com o alvo no top-10 dos dois, ele o **desce** em 167 e sobe em 141.
#
# A constante fica porque `avaliar_t1b.py` precisa de um default para remedir a
# cadeia histórica. Ela nomeia um ponto da curva, não uma peça do sistema — a
# mesma distinção que o `phiemb-minilm-melhor` tem em `avaliar_encoders.py`.
RERANQUEADOR = "models/phirank-physbert-melhor"
