# Plano — indexar trechos do texto completo para o assistente

**Status:** **Etapa 0 autorizada pelo dono em 2026-10-09** — o piloto e o estrato "corpo", com a barra de
5 pontos no desenvolvimento para construir. As etapas 1 e 2 continuam condicionadas ao piloto.
**Contexto:** [DOC-13 §3](DOC-13-recuperacao-embeddings-rag.md) (regras de chunking), §9.1–§9.3
(a medida do assistente e o que já foi tentado na busca), [ADR-0004](../adr/ADR-0004-phigen-para-o-assistente.md).

---

## 1. O problema que isto ataca

O assistente acerta **0,806** das perguntas; com o artigo certo garantido entre as fontes,
**0,944**. Os ~14 pontos de diferença são da busca, e a ordem dos resultados já foi
esgotada (§9.3): o ΦRank nos 50 primeiros colhe 0,827 de um teto de 0,847 no
desenvolvimento. O que sobra é o artigo certo não estar perto da consulta — o índice só
conhece `título. resumo`, um vetor por artigo.

A proposta do DOC-13 §3 é indexar **trechos do texto completo**: o resultado que o resumo
diz de um jeito, a introdução e a conclusão dizem de outros.

---

## 2. Os fatos, medidos em 2026-10-09

| | |
|---|---|
| texto completo já em disco | `data/processed/redpajama_fisica` — **828.601 artigos**, 12 GB, LaTeX a partir de `\section{Introduction}` (sem preâmbulo e sem o resumo) |
| **cobertura do índice** | **52,0%** dos 1.594.338 artigos |
| cobertura por ano | 62–83% até 2022 · **30% em 2023 · 5% em 2024 · ~1% em 2025–26** (o RedPajama foi colhido no início de 2023) |
| tamanho | média de **16.800 tokens** por artigo (mediana 13.600; 2,97 caracteres por token no tokenizer do encoder) |
| estrutura | 86% têm seção de introdução; 79%, de conclusão / resumo / discussão |
| o encoder do sistema | MiniLM de 384 dimensões, treinado e medido com **192 tokens** por texto |
| o índice de hoje | 1,22 GB (um vetor fp16 por artigo); indexar levou 57 min na RX 7600 → **466 textos/s** |
| a máquina | 16 GB de RAM, 8 GB de VRAM, **230 GB livres** no HD |

**O teto do ganho, antes de qualquer trabalho** (teste, primário, 500 perguntas):

- o sistema de hoje deixa o artigo certo fora das fontes em **91** perguntas;
- só **49** desses artigos têm texto completo em disco (54%);
- então trechos podem resgatar **no máximo 49 perguntas = +9,8 pontos** em "artigo certo
  entre as fontes" — e resgatar todas é irreal. Com a conversão vista no ΦRank (13,2
  pontos de busca viraram 10,6 de acerto), o teto em acerto é ~**+8 pontos**; uma
  expectativa honesta, metade disso: **~+4 pontos**, perto do limiar de adoção de 3;
- no estrato pós-corte (artigos de 2025 em diante) o ganho é **zero**: 1 de 150 tem texto.

⚠️ **E o teste não enxerga o principal.** As 650 perguntas saem de um fato do RESUMO — é o
desenho da medida. O que trechos compram de verdade é responder o que **não** está no
resumo (um valor de tabela, um detalhe do método), e para isso não há pergunta nenhuma no
conjunto. Medir só nas perguntas atuais pode reprovar uma coisa útil.

---

## 3. As opções

| | o quê | vetores | disco | GPU | o que exige de novo |
|---|---|---|---|---|---|
| **A · tudo** | todos os trechos de 192 tokens (passo 160) dos 828 mil artigos | **~87 M** (105 por artigo) | **~67 GB** | **~52 h** locais | busca aproximada (o índice não cabe nos 16 GB de RAM): FAISS ou quantização própria |
| **B · leve** | só introdução e conclusão: ~4 trechos do começo + ~3 do fim | **~5,8 M** | **~4,5 GB** | **~3,5 h** | nada: cabe na RAM, busca exata como hoje |
| **C · parar** | ficar com 0,806 | — | — | — | — |

Trechos de 512 tokens (o alvo do DOC-13) não servem com este encoder: ele lê 192. Em 512
seriam 31 M de trechos, e o encoder leria só os primeiros 37% de cada um.

Nas três, o custo em dinheiro é **US$ 0** — tudo local. O que custa dinheiro ou banda é
estender a cobertura (§6).

---

## 4. O caminho que eu recomendo: três etapas, cada uma decide a seguinte

### Etapa 0 · piloto — medir antes de construir (~1 dia meu · ~2 h de GPU · ~2 GB temporários)

A pergunta do piloto: **com trechos, o artigo certo sobe?** Sem indexar o corpus.

1. **O cortador de trechos**, com as regras do DOC-13 §3 (nunca partir uma equação; preferir
   fronteira de seção), e a auditoria M1: zero cortes dentro de ambiente matemático em
   10.000 amostras.
2. **Os concorrentes de verdade.** Para cada uma das 150 perguntas de desenvolvimento, os
   300 primeiros artigos da busca atual (os que hoje ficam na frente do artigo certo) que
   têm texto completo, mais o próprio artigo certo: ~23 mil artigos, ~2,4 M de trechos.
3. **Quatro representações comparadas no mesmo conjunto**, pela posição do artigo certo:
   o resumo (hoje) · o melhor trecho do artigo · o melhor entre introdução e conclusão ·
   o resumo combinado com o melhor trecho.
4. O ΦRank por cima do melhor candidato, como no produto.

**Sai do piloto:** quantas das perguntas em que a busca falha hoje seriam resgatadas, por
opção (A ou B). É exploratório, nas perguntas de desenvolvimento; nada toca o teste.

⚠️ O piloto é otimista num ponto, declarado: um artigo que hoje está além da 300ª posição
só concorre com os 300 da frente, não com o corpus inteiro.

### Etapa 1 · construir — só se o piloto passar a barra

Barra, **aceita pelo dono em 2026-10-09**: a mesma da §9.2 — pelo menos **5 pontos** de ganho no
desenvolvimento sobre o sistema de hoje. Abaixo disso, a opção é C.

- Se a opção **B** já der o ganho: construir B (~3,5 h de GPU, 4,5 GB). É o resultado mais
  provável de valer a pena, porque não exige infraestrutura nova.
- Se só **A** der: aí sim os ~52 h e os 67 GB, e a busca aproximada. Vale uma segunda
  pergunta ao dono antes — é uma semana de máquina.

### Etapa 2 · confirmar no teste — a regra que já existe

O braço A″ nos 650 itens, contra o A′ de hoje, com o mesmo juiz: adota se o IC 95% do
ganho ficar inteiro acima de **0,03**, com até **60 s** por pergunta (§9.2). ~5 h de GPU.

### Em paralelo, e eu recomendo: as perguntas que faltam (~1 dia meu · 0 do dono)

Um estrato novo, **"corpo"**: ~150 perguntas escritas por mim a partir do **texto
completo**, cuja resposta **não** está no resumo, com as mesmas guardas e o mesmo juiz.
Sem ele, a medida só vê o benefício pequeno (§2) e é cega para o grande. Com ele, dá para
responder "os trechos servem para quê" com número. O dono não julga nada; a ressalva de
sempre vale (calibrado contra o Claude).

---

## 5. O que o dono decide

| decisão | minha proposta |
|---|---|
| Fazer a etapa 0 (piloto)? | **Sim** — 2 h de GPU para não apostar 52 h às cegas · ✅ aceito |
| A barra para construir | 5 pontos no desenvolvimento, como na §9.2 · ✅ aceito |
| Escrever o estrato "corpo"? | **Sim** — é o que mede o benefício real · ✅ aceito |
| Estender a cobertura além de 52% (§6)? | **Não agora** — só depois de o piloto dizer que trechos funcionam |

---

## 6. O que fica de fora, e quanto custaria

- **Os artigos sem texto completo (48% do índice; quase tudo de 2023 em diante).** Baixar as
  fontes LaTeX do arXiv é possível, mas é download grande (centenas de GB, com custo de
  banda no acesso em lote) e precisa do aval do dono, com arquivo, origem e tamanho ditos
  antes. Sem isso, o assistente continua respondendo sobre artigos recentes só pelo resumo.
- **Trechos de 512 tokens e Matryoshka** (DOC-13 §8) pedem outro encoder; o do sistema lê
  192 tokens e não foi treinado para truncar dimensões.
- **O texto do peS2o** (18 GB em disco) é de artigos sem `arxiv_id` no esquema atual: não
  casa com o índice sem um trabalho de junção que este plano não inclui.

## 7. Os riscos, sem enfeite

| risco | por quê | o que o piloto diz |
|---|---|---|
| **O encoder não entender trecho de corpo** | foi treinado só com `título. resumo`; LaTeX de corpo, com macros e citações, é outra distribuição | direto: se o melhor trecho não superar o resumo, para aqui |
| **Ganho abaixo do limiar** | o teto é +9,8 em busca, e só metade dos erros tem texto | direto |
| A consulta não parecer com trecho | a consulta é um *resumo* hipotético — desenhada para casar com resumos | direto; se for o caso, a saída é uma consulta hipotética no formato de trecho |
| Concluir "não serve" por causa do teste | as perguntas atuais vêm do resumo | só o estrato "corpo" resolve |
| A opção A não caber na máquina | 67 GB de vetores contra 16 GB de RAM | não diz — é o motivo de B vir primeiro |
