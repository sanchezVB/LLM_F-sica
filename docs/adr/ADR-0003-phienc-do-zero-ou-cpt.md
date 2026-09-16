# ADR-0003 — O ΦEnc ainda deve ser treinado do zero?

**Status:** Proposto (2026-09-16) — **aguarda decisão do dono do projeto**
**Contexto:** [DOC-07 §2](../02-models/DOC-07-familia-de-modelos.md) (ΦEnc) e §14 (OQ-4), [DOC-00 D-01](../00-foundations/DOC-00-project-charter.md) (o dissenso registrado), [DOC-05 §8 e §11.2](../01-data/DOC-05-tokenizer.md)
**Não substitui nada ainda.** Se aceito, revisa a linha "Physics Encoder — treino do zero" da tabela de decisões do DOC-07.

---

## 1. A pergunta

O DOC-07 faz do ΦEnc **o único modelo do programa treinado do zero**, e o DOC-00
guarda a razão: o dissenso à D-01 — *"um modelo do zero, com tokenizer nativo de
Física e arquitetura nativa de Física, poderia adquirir vieses indutivos que nenhum
modelo por CPT consegue"* — é atacado *"no tier de encoders, onde podemos pagar para
respondê-la"*.

O ΦEnc do zero se apoiava em três ingredientes que só um modelo próprio teria. Dois
deles foram testados num proxy de 48 M a 0,6 B tokens entre 2026-09-11 e 2026-09-16.
Este ADR registra o que isso muda, e propõe um caminho.

---

## 2. A evidência

| ingrediente do ΦEnc do zero | teste | resultado | ressalva |
|---|---|---|---|
| **tokenizer nativo** — a regra de pré-tokenização de LaTeX (DOC-05 §8) | T2a, A×E, bits por byte, três instrumentos | **a regra CUSTOU**: A − E = +0,047 [+0,043; +0,050] bit/byte, a favor do tokenizer SEM ela | 0,6 B, uma semente, spike em A |
| **objetivo nativo** — mascarar equações inteiras (DOC-07 §2.3), medida primária | diferença das diferenças de MLM por região | **−0,0040** [−0,0058; −0,0022]: negativo pela regra; concentrado em display (exploratória) | a prova favorece o controle, e a regra nomeou isso antes |
| **objetivo nativo**, secundária de recuperação | pool do G1, 2.000 pares de citação | **nDCG@10 0,0171 → 0,1391** (8×), recall@1 p = 3,6×10⁻³⁶; **não é geometria** (anisotropia igual, e centrar não fecha a diferença) | MLM cru; ninguém mediu se sobrevive ao ajuste contrastivo |
| objetivo nativo, secundária de estrutura | sonda tensorial, 72 itens | 0,333 → 0,375, p = 0,70: sem diferença | poder baixo |
| **contexto de 8.192** | — | não é exclusivo de treino do zero: o ModernBERT-base (149 M, Warner et al., 2024) já tem | — |

E uma evidência vizinha, do lado da recuperação: **a base importa mais que o volume de
ajuste** (T1f). O GTE-base ajustado com 400 mil pares dá nDCG@10 0,6094 contra 0,5462
do MiniLM com os mesmos pares, e empata com o nosso MiniLM ajustado com 6 milhões.

---

## 3. O que a evidência diz, e o que ela não diz

**O argumento do tokenizer nativo perdeu o apoio.** A única regra específica de Física
no tokenizer piorou o modelo a 0,6 B.

**O objetivo do §2.3 deu um resultado dividido, e ele NÃO é argumento a favor de
treinar do zero.** As marcas de equação saem de offsets de CARACTERE
(`mascaramento.marcar_equacoes`), não de ids de token: o mesmo tratamento se aplica a
um pré-treino continuado de qualquer base, com qualquer tokenizer. Um resultado
positivo dele favoreceria usá-lo, não usá-lo do zero.

**Então o que sobra para o treino do zero é o dissenso em si** — viés indutivo nativo —,
e ele só se responde com o confronto que o DOC-07 §14 promete: ΦEnc do zero contra um
encoder geral adaptado, no mesmo corpus.

**E a pergunta que mais importa para o SISTEMA ainda não foi feita.** O ganho de 8× na
recuperação é de MLM cru, agregado por média. O recuperador do sistema é um bi-encoder
AJUSTADO por pares de citação, e o T1f mostrou que o ajuste pode apagar ou ampliar
diferenças de base. Se o ganho do tratamento não sobreviver ao ajuste, ele não vale
nada para a busca de Física.

**O que isto NÃO é:** um veredito sobre o ΦEnc de 150 M. São proxies de 48 M a 0,6 B,
3,3× abaixo do planejado para a ablação, com uma semente por braço.

---

## 4. As opções

| | o quê | custo | responde |
|---|---|---|---|
| **A** | ΦEnc-150M do zero como o DOC-07 descreve, com tokenizer E e `p_equacao` 0,6 | US$ 25–90 (DOC-07 §2.4) | nada sozinho: sem o confronto, "funcionou" não diz se foi o treino do zero |
| **B** | pré-treino continuado do ModernBERT-base no corpus de Física, com `p_equacao` 0,6 | ~US$ 3–9 — *estimativa* pela conversão do DOC-07 §2.4 com 3 B tokens (2,7×10¹⁸ FLOPs) | um encoder de Física de 150 M com o objetivo que ajudou a recuperação, pelo caminho da D-01 |
| **C** | **ajustar os dois braços de 48 M como ΦEmb** (receita do T1a, 400 mil pares) e comparar no G1 | ~1–2 h de T4 — *estimativa*: o MiniLM de 23 M levou 36 min. ⚠️ A conferir antes: o `train_embedding.py` nunca ajustou a arquitetura do ΦEnc | **se o ganho do §2.3 sobrevive ao ajuste** — o que decide se ele importa para o sistema |
| **D** | A e B lado a lado | A + B | o dissenso da D-01, como o DOC-07 §14 promete |

---

## 5. Recomendação (não decidida)

**Primeiro C.** É o experimento mais barato e o único que liga o resultado do §2.3 ao
produto. Cabe na cota gratuita da semana seguinte (restam ~2 h nesta).

- **Se o ganho sobreviver ao ajuste:** seguir com **B**. É o caminho mais barato para um
  encoder de Física de 150 M com o objetivo que funcionou, e é o que a D-01 manda para
  todo o resto do programa. A só entra como D, se o dissenso for uma pergunta que o
  projeto queira responder por si.
- **Se o ganho sumir no ajuste:** **pausar o ΦEnc.** O caminho da recuperação passa a
  ser base geral forte ajustada, na direção do T1f, e a verba do ΦEnc vai para onde a
  medição apontar.

**O que mudaria esta recomendação:** uma segunda semente que desfaça o ganho de
recuperação; a ablação a 2 B contradizendo a de 0,6 B; ou C mostrando que o ajuste
AMPLIA a diferença — aí a pergunta do treino do zero volta a valer o confronto D.

---

## 6. O que fica publicado, qualquer que seja a decisão

O DOC-07 §2.3 promete publicar o negativo. Os dois lados vão no §2.3-medido: o negativo
da primária **e** a discordância da recuperação, com as ressalvas de cada um. Publicar
só o negativo seria esconder a metade que mais importa para a busca; publicar só o
ganho seria esconder a regra.
