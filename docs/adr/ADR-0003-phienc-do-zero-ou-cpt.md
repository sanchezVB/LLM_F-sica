# ADR-0003 — O ΦEnc ainda deve ser treinado do zero?

**Status:** Proposto (2026-09-16) — **aguarda decisão do dono do projeto**. A opção C foi executada no mesmo dia (§7), o passo 1 do caminho B em 2026-09-17 (§8), e as três opções foram medidas na mesma régua em 2026-09-23 (§9): **o GTE-base cru ajustado (0,5964) domina o CPT tratado (0,5373) e o ModernBERT cru (0,5270)**.
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

---

## 7. A opção C, executada (2026-09-16)

| | controle (`p_equacao` 0,0) | tratado (`p_equacao` 0,6) | tratado − controle |
|---|---|---|---|
| nDCG@10 **antes** do ajuste (MLM cru) | 0,0171 | 0,1391 | +0,122 |
| **nDCG@10 depois** do ajuste | 0,3872 | **0,4712** | **+0,084 [+0,071; +0,097]** |
| recall@1 depois | 0,2215 | 0,2985 | 106 × 260 discordantes, McNemar p = 4,7×10⁻¹⁶ |
| recall@10 depois | 0,5890 | 0,6740 | |

Os dois braços de 48 M ajustados como ΦEmb com os hiperparâmetros do T1f, 200 mil
pares sorteados dos mesmos bytes do T1a, medidos na mesma sessão pelo protocolo do G1.
Regra escrita antes em `kaggle/t2eq_emb.py`.

**Tratado à frente: o ganho sobrevive ao ajuste.** Encolhe de +0,122 para +0,084, e
fica grande. Pela recomendação da §5, o próximo passo é **B** — pré-treino continuado
do ModernBERT-base no corpus de Física, com `p_equacao` 0,6.

⚠️ Uma semente por braço, 48 M, 0,6 B, metade da receita de pares. O que mudaria isto
continua escrito na §5.

---

## 8. O passo 1 do caminho B, medido (2026-09-17): a barra é 0,5270

Antes de gastar ~7 h de acelerador por braço num pré-treino continuado, a pergunta que
faltava: **quanto o ModernBERT-base faz SEM pré-treino nenhum em Física?** Mesmo ajuste,
mesmos 200 mil pares (blake3 conferido contra o pacote do T1a), mesmos hiperparâmetros,
e os três braços medidos na mesma sessão, na mesma GPU, pelo protocolo do G1.

| braço | parâmetros | recall@1 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| controle (ΦEnc do zero, `p_equacao` 0,0) | 48 M | 0,2215 | 0,5890 | 0,3391 | 0,3872 |
| tratado (ΦEnc do zero, `p_equacao` 0,6) | 48 M | 0,2985 | 0,6740 | 0,4203 | 0,4712 |
| **ModernBERT-base, sem Física** | **150 M** | **0,3540** | **0,7200** | **0,4774** | **0,5270** |

**Desfecho pela regra (`kaggle/t2eq_emb.py` · REGRA_MODERNBERT): MODERNBERT À FRENTE**,
+0,0558 [+0,0426; +0,0685] sobre o tratado, por bootstrap pareado por item.

### O que isto muda, e o que não muda

**Não desfaz o §2.3.** O tratado continua à frente do controle por +0,084 [+0,071;
+0,097], na mesma medição. O mascaramento de equações ajuda — só que ajudar um encoder
de 48 M treinado em 0,6 B tokens não basta para alcançar uma base geral de 150 M
treinada em 2 T.

**Fecha o caminho A, na prática.** Um ΦEnc-150M do zero teria de superar, com o nosso
corpus e o nosso orçamento, um modelo que já está pronto e de graça. Nada aqui sugere
que isso aconteça: a distância que o treino do zero teria de cobrir é a soma de três
ordens (tokens, parâmetros, e agora +0,056 de desvantagem medida).

**Dá ao caminho B um alvo numérico e uma pergunta mais limpa.** A barra é **0,5270** no
protocolo do G1, e os dois braços do CPT partem DESTA base — então a comparação passa a
ser dentro da família, com uma variável (`p_equacao`), em vez de atravessar tamanho,
tokenizer e volume de pré-treino.

⚠️ **A comparação com o ModernBERT NÃO é ablação de uma variável**, e a regra dizia isso
antes: 3× os parâmetros, outro tokenizer, 2 T tokens de pré-treino geral. É a barra do
produto.

⚠️ E ela é de recuperação, que é a secundária do §2.3. A primária — previsão de token de
equação — segue negativa a 0,6 B (−0,0040).

### Execução

Rodou no **Google Colab** (T4, 2,56 h), fora da cota do Kaggle, com o notebook
`colab/t2eq_emb_modernbert.ipynb` gerado do fonte versionado. O checkpoint veio no
formato do `transformers` 5.16 e **não abre na 4.48** — foi medido no venv paralelo com
5.0. Os dois braços de 48 M reproduziram exatamente os números de 2026-09-16 (0,3872 e
0,4712), que foram medidos na 4.48: confirmação, no nível da métrica, de que a troca de
versão não muda resultado.

Artefato: `data/processed/avaliacao/t2eq_emb_comparacao.json`.

### A decisão do dono, agora com número

1. **Seguir com B** — os dois braços de CPT do ModernBERT-base (~7 h de T4 cada, em duas
   placas), e ver se o `p_equacao` 0,6 tira o encoder de 0,5270 para cima. O pacote de
   dados e a célula estão prontos.
2. **Parar aqui** e usar o ModernBERT-base ajustado como recuperador, aceitando 0,5270 —
   que já supera tudo o que temos em 48 M, e é de graça.
3. **Nem B nem parar:** ir para uma base ainda mais forte, na direção do T1f (GTE-base
   ajustado dá 0,6094 no mesmo protocolo, com 400 mil pares).

⚠️ A opção 3 tem o melhor número da tabela e **não é comparável a estas três linhas**:
400 mil pares contra 200 mil. Comparar exigiria rodar o GTE-base a 200 mil, ou os três a
400 mil.

## 9. As três opções na mesma régua (2026-09-23)

A §8 deixou a opção 3 fora da tabela por não ser comparável ("400 mil pares contra 200
mil"). Ela foi medida a 200 mil pares (`t2eq_emb_gte`), os dois braços do caminho B foram
treinados e ajustados, e os quatro encoders foram medidos **na mesma sessão**, no
protocolo do G1, com as regras escritas antes (`colab/t2eq_cpt_emb.py` e
`kaggle/t2eq_emb.py` · REGRA_GTE):

| @200k pares, mesma receita | recall@1 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| ModernBERT-base CRU ajustado (a barra, opção 2) | 0,3540 | 0,7200 | 0,4774 | 0,5270 |
| CPT controle (`p_equacao` 0,0) | 0,3635 | 0,7175 | 0,4823 | 0,5297 |
| CPT tratado (`p_equacao` 0,6) — opção 1 | 0,3620 | 0,7290 | 0,4876 | 0,5373 |
| **GTE-base CRU ajustado — opção 3** | **0,4170** | **0,7880** | **0,5443** | **0,5964** |

A barra remedida deu 0,5270 de novo: o avaliador não derivou em seis dias.

### O que cada regra decidiu

- **Entre os braços** (a ciência, uma variável): tratado − controle **+0,0076
  [+0,0013; +0,0139]** → **TRATADO À FRENTE**. O ganho de recuperação do §2.3 se repete a
  150 M — mas **11× menor** que a 48 M (+0,084), com o IC encostando no zero. O recall@1
  empata (63 × 60, p=0,86): o ganho está abaixo do topo.
- **Contra a barra**: tratado − ModernBERT cru **+0,0103 [+0,0032; +0,0176]** → **CPT
  ACIMA DA BARRA**.
- **A base**: GTE − ModernBERT **+0,0694 [+0,0580; +0,0812]** → **GTE À FRENTE**.

### A leitura conjunta, que nenhuma regra isolada dá

**A escolha da base vale ~7× o pré-treino continuado inteiro.** Os 0,4 B tokens de
Física, com o objetivo do §2.3, compraram +0,010 sobre o ModernBERT-base; trocar o
ModernBERT pelo GTE, sem pré-treino nenhum, compra +0,069. O GTE cru ajustado fica
**+0,059** acima do melhor braço do CPT (diferença de estimativas pontuais; não é
comparação pré-registrada, e não precisa ser — a distância é 5× a largura dos ICs).

A frase da regra "o encoder do sistema passa a ser candidato a trocar" foi escrita antes
de o GTE entrar na mesma régua. Com ele dentro, **o CPT sobre o ModernBERT-base não é
candidato a nada no produto**: a opção 3 domina as opções 1 e 2.

### ⚠️ As ressalvas, e uma delas pesa contra o efeito do §2.3

1. **A assimetria de spike favorece o tratado.** O CPT controle teve 2 spikes de norma,
   com rollback de 182 lotes e LR pela metade por 500 passos; o tratado rodou limpo. Com
   um efeito de +0,0076 e o limite inferior do IC em +0,0013, a assimetria é do tamanho
   que poderia explicar parte dele. Não foi medida.
2. **A primária do CPT, a de MLM, deu NÃO DECIDIDO** (−0,00039 [−0,0016; +0,0009]).
   O §2.3 chega a 150 M com uma secundária positiva marginal e uma primária nula.
3. **Uma semente por braço, e 200 mil pares** — metade da receita do T1a.

### O que isto deixa para a decisão do dono

1. **O ΦEnc próprio fecha para o produto.** Do zero (caminho A) já tinha fechado na §8;
   o pré-treino continuado (caminho B) rende +0,010 sobre uma base que perde por 0,069
   para outra base pronta.
2. **O objetivo do §2.3 fica com evidência fraca e positiva.** Vale publicar como está:
   +0,084 a 48 M, +0,0076 a 150 M, primária nula nas duas escalas — o efeito encolhe com a
   base, que é o padrão de um viés que a base forte já resolve.
3. **A pergunta que continua aberta é a do encoder do sistema**, e ela já está rodando:
   o T1g mede o GTE-base a 1 M de pares contra o ΦEmb do sistema (MiniLM@6M, 0,6223).
4. **Um CPT com o §2.3 sobre o GTE-base** é a combinação que esta tabela sugere, e ela
   NÃO está pronta: o laço de pré-treino só conhece o ModernBERT, e o GTE é BERT. Pelo
   tamanho do ganho medido (+0,008 entre braços), é difícil que pague a engenharia.

Artefatos: `data/processed/avaliacao/t2eq_cpt_emb_comparacao.json`,
`t2eq_emb_gte_contra_barra.json`, `t2eq_cpt_ablacao.json`.
