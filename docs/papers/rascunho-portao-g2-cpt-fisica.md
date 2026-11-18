# Antes de treinar: um portão pré-registrado para o pré-treino continuado em Física, as duas bases medidas, e uma sonda de contaminação que mede competência

**Rascunho.** Não submetido, não revisado por ninguém além de mim. Ver §10 antes de usar
qualquer número daqui, e §11 antes de usar qualquer citação.

**Autoria:** o trabalho experimental é de Vinicius; este rascunho foi escrito por
Claude (Anthropic) a partir das medições do repositório. A ordem e a forma da
autoria são decisão do primeiro.

---

## Resumo

Queríamos saber se um pré-treino continuado (CPT) em Física melhora o raciocínio de um
modelo base em problemas de Física difíceis, sem custar capacidade geral e sem decorar o
teste. Antes de medir qualquer base e antes de qualquer treino, escrevemos a regra que
decide: o **portão G2**. Ela fixa quatro benchmarks abertos (2.528 itens de teste), a
partição, o motor de inferência, o corretor e a estatística. Depois congelamos tudo por
hash, com um livro de acessos que permite um acesso ao teste por medida. Medimos as duas
bases do plano, a **Qwen3-1.7B-Base** e a **Qwen3-8B-Base**, por inteiro. O ΦGen, o modelo
treinado, não existe: a decisão de treiná-lo está em aberto.

Três achados saíram das bases, antes de qualquer treino:

1. **Uma sonda de contaminação pré-registrada mede competência, não memória.** A sonda
   que mostra o enunciado e 40% da solução e procura 13 fichas idênticas na continuação
   marca **37% (1,7B) e 49% (8B)** do PHYBench, que foi publicado depois dos dados das duas
   bases. No dev, ela acompanha o tamanho do resto da solução e o tipo da resposta, não o
   acerto. A regra de "memorização nova" com essa sonda inventaria memorização em **16%**
   do PHYBench num placebo entre as duas bases. Com a sonda trocada antes do teste (13
   fichas no enunciado, ≥32 fichas na solução), o placebo dá **2%**. A sonda do enunciado
   mostra contaminação herdada real: 1% no PHYBench, contra 40–47% no OlympiadBench e
   20–52% no SciBench.
2. **A geração gulosa do `llama-server` não se repete com vários slots.** Com 4 slots,
   duas rodadas com os mesmos prompts deram gerações diferentes em **42%** das sondas. Com
   1 slot, servidor novo e a mesma ordem, a rodada refeita deu as **mesmas 785 gerações,
   ficha a ficha**. A ordem dos pedidos também conta.
3. **A régua exige ~+12 pontos.** Com as bases medidas, a regra passa com 80% de chance a
   partir de **+12,2 (1,7B) e +12,1 (8B)** pontos de ganho no macro de Física, como a
   simulação com premissas tinha previsto. É mais do que toda a distância entre as duas
   bases (~7,8 pontos). Os CPTs publicados que chegaram a +12 o fizeram em matemática,
   sobre bases fracas no domínio, com 50 a 500 B fichas.

---

## 1. A pergunta, e por que um portão

O plano do projeto previa um ΦGen: o CPT de uma base Qwen3 num corpus de Física. A
pergunta que o justifica é estreita: **raciocínio em Física sem a fonte no prompt**. Uma
medida anterior do projeto já mostrou que, no assistente de literatura, o gerador não é
o que limita: com o artigo certo no prompt, o modelo aberto acerta 19 de cada 20
perguntas, e o que falta é a busca.

Um CPT tem muitos graus de liberdade que podem fabricar um ganho: o prompt, os exemplos,
o extrator da resposta, o corretor, o teto de fichas, o checkpoint escolhido e o próprio
teste olhado mais de uma vez. Por isso a regra veio antes:

- tudo o que se escolhe é escolhido em material que nunca é teste;
- depois, tudo é congelado por hash, e mudar algo exige uma emenda registrada;
- cada braço tem um acesso ao teste por medida, e o livro de acessos é versionado;
- até o veredito, só se leem agregados e contadores automáticos.

## 2. O protocolo

### 2.1 Benchmarks e recortes

O **G2.1 decide** com 4 benchmarks abertos:

| Benchmark | Recorte | Itens de teste | k |
|---|---|---|---|
| UGPhysics-en | resposta única de tipo limpo, sem duplicata exata | 1.999 | 3 |
| OlympiadBench | `OE_TO_physics_en_COMP`, em 113 agrupamentos | 236 | 8 |
| PHYBench | `fullques_v1`, os 100 com gabarito | 100 | 16 |
| SciBench-física | class, fund e thermo | 193 | 8 |

O **G2.3** mede o que o CPT pode estragar:
- o MMLU sem as matérias de Física (13.402 itens);
- o EvalPlus (HumanEval+ e MBPP+, 542 problemas);
- o IFEval (541 prompts), num envelope de texto corrido para modelo base.

O desenvolvimento usou só material que nunca é teste: o UG-dev, o OlympiadBench em chinês,
os `_sol` do SciBench e as partes de dev do MMLU e do MBPP.

### 2.2 Partição, lacre e acessos

O UGPhysics foi partido **por grupos**: componentes conexos de itens que compartilham um
13-grama de palavras do enunciado. Assim, irmãos quase idênticos nunca ficam um no dev e
outro no teste. Ficaram 899 itens de dev, 1.999 de teste e 1.582 numa **reserva lacrada**,
em 3.948 grupos. As listas de ids foram para o git antes de qualquer saída de modelo.

O **livro de acessos** (`g2_acessos.jsonl`) recusa um segundo acesso ao teste com a mesma
chave (tamanho, braço, medida, hash da configuração). Ele aceita a retomada de uma medida
interrompida e uma repetição por falha operacional. Uma linha que não é JSON (a cauda de
uma queda de energia) para tudo, e o registro vai ao disco antes da primeira geração.

### 2.3 Geração e correção

- **Formato.** Modelo base, texto corrido, sem template de chat. Exemplos fixos no formato
  `Problem / Solution`, cada solução terminando em `The final answer is $\boxed{…}$`.
  Nenhum exemplo é irmão (13-grama) de um item de teste.
- **Decodificação.** T=0,6 e top-p 0,95, sem top-k, sem min-p e sem penalidades, até 4.096
  fichas. A semente de cada amostra é a mesma nos dois braços. A nota do item é a média
  das k amostras (avg@k).
- **Correção** por regras, sobre o verificador simbólico do projeto, com tolerância
  numérica por benchmark. O autoteste do gabarito exclui do teste o item cujo gabarito
  não passa contra si mesmo, ou cujo gabarito perturbado não falha. Por isso o G2.1
  mede 1.917, 231, 99 e 193 itens.
- **Auditoria cega do corretor.** Dois juízes independentes julgaram 200 gerações de dev
  sem ver o veredito do corretor, e os vereditos foram gravados por hash antes. A
  concordância foi de **99,5%** (199 de 200), com **κ = 0,977**. O desacordo é de notação:
  o corretor não reconhece um símbolo renomeado.

### 2.4 O motor

O `llama-server` do llama.cpp, na tag b11159, com GGUF Q8_0 convertido por nós a partir
dos safetensors da revisão fixada. A KL média do Q8_0 contra o F16 tem de ficar ≤0,01
nats. Ela deu 0,00317 na 1,7B e 0,00174 na 8B, e o Q8_0 da 8B convertido no Kaggle é
idêntico byte a byte ao convertido aqui. O tokenizer do ΦGen tem de ser idêntico ao da
base; as duas bases têm o mesmo.

### 2.5 A regra

**O G2.1 passa** só com as três condições:
1. o limite de baixo do IC 95% de Δ_macro (a média simples dos 4 deltas) é ≥ +10,0;
2. nenhum dos quatro deltas tem o IC inteiro abaixo de 0;
3. sem os itens de memorização nova (§4), o limite de baixo continua ≥ +10,0.

O IC sai de um bootstrap pareado, estratificado por benchmark, que reamostra os
agrupamentos (os grupos de 13-grama no UG, os 113 do OlympiadBench, os itens nos outros
dois), com 10.000 réplicas e semente fixa.

**O G2.3** pede não-inferioridade demonstrada no MMLU (limite de baixo ≥ −2). No EvalPlus e
no IFEval, o ΦGen é desclassificado se o delta for < −2 com o IC inteiro abaixo de 0.

### 2.6 Congelamento e emendas

O protocolo foi congelado em 2026-10-24, depois do dev e da auditoria do corretor, e
emendado quatro vezes até aqui. Cada emenda grava os hashes anteriores, e a configuração
da geração da 1,7B nunca mudou:
1. o parser do verificador e o leitor de jsonl;
2. o código e os dados do G2.3 entram no congelamento;
3. o executável e o rótulo da GPU vêm do ambiente, para rodar a 8B no Kaggle;
4. o código das sondas de contaminação, e a troca da sonda da condição 3 (§4.3).

## 3. As bases medidas

A 1,7B rodou numa AMD RX 7600 local (Vulkan). A 8B rodou no Kaggle, em 2×T4 (CUDA, a
mesma tag), em 6 sessões e ~34 h de cota, com o estado das medidas passado de uma sessão
para a outra num zip com hash.

**G2.1**, avg@k:

| Benchmark | Itens | 1,7B: acerto | 1,7B: truncado | 8B: acerto | 8B: truncado |
|---|---|---|---|---|---|
| UGPhysics | 1.917 | 10,0% | 5,4% | 17,8% | 2,8% |
| OlympiadBench | 231 | 4,3% | 16,0% | 12,3% | 8,8% |
| PHYBench | 99 | 0,1% | 17,8% | 1,1% | 9,0% |
| SciBench | 193 | 10,6% | 9,8% | 25,3% | 5,4% |
| **Macro** | | **~6,3** | | **~14,1** | |

**G2.3:**

| Medida | 1,7B | 8B |
|---|---|---|
| MMLU sem Física (13.402) | 62,72% | 76,56% |
| IFEval, strict médio de 16 amostras | 35,28% | 54,57% |
| EvalPlus, pass@1 plus nas 542 | 52,49% | 64,06% |

As sanidades passaram:
- **o MMLU de 57 matérias** dá 62,63 e 76,78, contra 62,63 e 76,89 do relatório do Qwen3 (folga de ±3);
- **a média dos 4 do EvalPlus na gulosa** dá 54,70 e 68,45, contra 52,70 e 67,65 (folga de ±5).

Duas das 542 tarefas do EvalPlus não passam nem com a solução canônica no avaliador
oficial 0.3.1. Elas ficam como falha fixa nos dois braços, e o pass@1 nas 540 é relatado
ao lado: 52,69% e 64,30%.

## 4. Contaminação: a sonda de 13 fichas na solução mede competência

### 4.1 As sondas, como foram pré-registradas

Duas sondas gulosas, sem exemplos, nos 2.528 itens:
- **S1:** o modelo recebe a primeira metade das fichas do enunciado (metade de ≥32 fichas)
  e gera 48;
- **S2:** o modelo recebe o enunciado e os primeiros 40% das fichas da solução, e gera 64.

O item é **suspeito** se alguma sequência de 13 fichas da geração aparece na continuação
verdadeira. O **PHYBench**, publicado depois dos dados das duas bases, é o **nulo**: a
fração de suspeitos nele estima o acaso. Na condição 3 do G2.1, **memorização nova** é um
sinal no ΦGen num item sem sinal na base.

### 4.2 O que o dev mostrou

Na 1,7B-base, no dev (UG-dev, OlympiadBench em chinês, `_sol` do SciBench):

| Benchmark | S1: suspeitos / aplicáveis | S2: suspeitos / aplicáveis |
|---|---|---|
| UGPhysics (300) | 24 / 218 | 117 / 295 |
| OlympiadBench-zh (115) | 2 / 102 | 22 / 115 |
| SciBench `_sol` (33) | 7 / 22 | 26 / 33 |

Lidos 40 casos, os 13-gramas casados da S2 são quase sempre pedaços de LaTeX. Em fórmula,
13 fichas são poucos símbolos, e quem sabe continuar a conta os escreve igual à solução.
A cópia de trechos do próprio prompt explica pouco (16 dos 117 suspeitos do UG). No UG:
- a S2 marca 48% dos itens de resposta EX, 38% dos NV e 30% dos outros;
- por tercil do tamanho do resto da solução, ela marca 19%, 41% e 59%;
- ela não acompanha o acerto da própria base: na 1,7B, os suspeitos acertam 0% (n=44) e os
  outros, 8,2% (n=61); na 8B, 12,2% contra 17,4%.

O problema é a condição 3. Se o sinal da S2 acompanha a capacidade de continuar a conta,
um ΦGen que aprendeu Física ganha sinais novos justamente nos itens em que melhorou, e a
condição 3 tira esses itens. Numa simulação, com ganho verdadeiro de +13 e as marcas novas
caindo 3× mais nos itens que melhoraram, a regra passaria em 39% das vezes, contra 84% com
as marcas ao acaso.

### 4.3 A decisão, antes do teste

Decidimos, antes de qualquer sonda no teste, que a condição 3 usa uma **lista
híbrida**: a S1 com 13 fichas e a S2 só com um trecho comum de **≥32 fichas**. As listas de
13 fichas continuam lacradas e relatadas. A troca entrou como emenda de regra no
pré-registro.

### 4.4 O teste

| Benchmark | Base | S1 | S2 | União, 13 fichas | Híbrida |
|---|---|---|---|---|---|
| UGPhysics | 1,7B | 11,4% | 37,3% | 40,7% | 13,0% |
| | 8B | 13,5% | 44,5% | 48,1% | 15,8% |
| OlympiadBench | 1,7B | 40,4% | 36,4% | 63,6% | 44,1% |
| | 8B | 46,8% | 43,2% | 70,8% | 51,3% |
| **PHYBench (nulo)** | 1,7B | **1,0%** | **37,0%** | 38,0% | **7,0%** |
| | 8B | **1,0%** | **49,0%** | 49,0% | **5,0%** |
| SciBench | 1,7B | 20,0% | sem solução | 15,0% | 15,0% |
| | 8B | 51,7% | sem solução | 38,9% | 38,9% |

As frações da S1 e da S2 são sobre as sondas aplicáveis; as da união e da híbrida, sobre os itens.

- **No nulo, a S2 de 13 fichas não se distingue do acaso.** Ela marca 37% na 1,7B, o mesmo
  que no UG e no OlympiadBench, e sobe para 49% na 8B só com o modelo maior, sem memória
  nenhuma.
- **A S1 tem sinal.** Ela fica em 1% no nulo nas duas bases e é alta onde os enunciados
  circulam: 40–47% no OlympiadBench. Ela cresce com a escala onde há memória: no SciBench,
  feito de problemas de livro-texto, vai de 20% para 52%, com 37 casamentos de ≥32 fichas
  na 8B contra 2 na 1,7B.

### 4.5 O placebo

A condição 3 aplicada ao par, com a 8B no lugar do ΦGen e a 1,7B no da base, marcaria como
memorização nova:

| Benchmark | Com a híbrida | Com 13 fichas |
|---|---|---|
| UGPhysics | 5,6% | 13,3% |
| OlympiadBench | 9,7% | 10,6% |
| **PHYBench (nulo)** | **2,0%** | **16,0%** |
| SciBench | 25,9% | 25,9% |

O PHYBench nenhum dos dois modelos viu. Nele, a lista de 13 fichas inventa memorização nova
em 16% dos itens, e a híbrida em 2%.

### 4.6 O lacre por compromisso de hash

"Até o veredito, nada se lê por item": a lista de ids suspeitos é informação por item.
Ela foi gravada fora do git, num JSON canônico, e o agregado versionado leva só os
contadores e o SHA-256 desses bytes. No veredito, a lista é aberta e conferida contra o
hash gravado antes do treino. As contagens da lista híbrida e do placebo acima foram
feitas por um contador automático, que só imprime totais.

Cada sonda grava também o hash das fichas que mostrou e das que esperava, e o agregado
grava o hash do conjunto. A 8B tem de ter sondado exatamente as fichas da 1,7B, porque o
tokenizer é o mesmo, e a conferência da etapa no Kaggle exige isso. O hash bateu.

## 5. A geração gulosa do `llama-server` só se repete com 1 slot e a mesma ordem

As sondas comparam, item a item, a lista do ΦGen com a da base. Isso pede que a medida se
repita. Medido no dev da 1,7B (RX 7600, Vulkan, b11159):

| Configuração | Gerações iguais | Sinais iguais |
|---|---|---|
| 4 slots, duas rodadas com os mesmos prompts | 58% | 774 de 785 |
| 1 slot, servidor novo, **mesma ordem**, o dev inteiro | **785 de 785** | 785 de 785 |
| 1 slot, outra ordem (1 sonda a cada 3) | 234 de 262 | 262 de 262 |
| 1 slot, ordem inversa | 37 de 40 | — |
| 1 slot, cada sonda depois de um pedido longo | 33–34 de 40 | — |

- **Com vários slots, o servidor junta no mesmo lote pedidos de slots diferentes**, e a conta
  da geração muda conforme a companhia.
- **Com 1 slot, o estado que um pedido deixa ainda mexe no arredondamento do seguinte.**
  Todas as diferenças aparecem depois da 8ª ficha.
- **Desligar o cache de prompt em RAM não resolve.**

A receita adotada é 1 slot, um pedido por vez, sempre na mesma ordem, a ~2,3× o tempo de 4
slots. Uma retomada depois de uma queda sobe um servidor novo, o que equivale a uma ordem
diferente dali em diante; cada linha grava a marca da execução. As medidas amostradas e
agregadas (G2.1, IFEval) toleram os 4 slots, porque o ruído não muda o agregado em média.

## 6. O poder, com as bases medidas

O pré-registro estimou o poder com premissas e pediu o recálculo com a base medida, sem
mudar a regra. Simulamos um ΦGen sobre a nota medida de cada item de cada base, aplicamos a
regra como está (o mesmo bootstrap) e contamos quantas vezes ela passa, em 150 réplicas por
ganho. O ganho foi simulado de duas formas:
- **uniforme** nos itens, que é o limite otimista;
- **concentrado** em 30% dos itens, sorteados, que é como um modelo ganha de verdade.

No concentrado, o teto de nota 1 corta parte do ganho nos itens fáceis. Por isso o eixo é
o **ganho efetivo**: o Δ_macro médio que saiu.

| Cenário | 80% de poder na 1,7B | 80% de poder na 8B |
|---|---|---|
| Ganho uniforme | +11,3 | +11,4 |
| **Ganho concentrado** | **+12,2** | **+12,1** |
| Concentrado, com marcas de memorização nova ao acaso (taxas do placebo) | +12,2 | +12,6 |
| Gulosa (k=1), concentrado | +13,5 | +13,5 |

- **A previsão se confirma.** Com o ganho concentrado, o erro-padrão do macro perto do
  limiar fica em 0,63–0,78 ponto, e a regra passa com 80% de chance a partir de ~+12,
  contra +12,0 a +12,6 previstos. Com +10 exato, ela passa em 0–1,3% das vezes.
- **A condição 3 com a lista híbrida custa pouco:** nada na 1,7B e +0,5 ponto na 8B.
- **A amostragem a T=0,6 vale o que custa.** A gulosa quase dobra o erro-padrão.

## 7. O que isso diz sobre treinar

Para passar com 80% de chance, o ΦGen precisa de ~+12 pontos sobre a própria base: a 1,7B
iria de ~6,3 a ~18,5, e a 8B de ~14,1 a ~26. O ΦGen-1,7B teria de superar a 8B-base.

A literatura que achamos (protocolos diferentes, e quase toda de matemática):
- **Os +12 de CPT vieram de bases fracas no domínio, com 50 a 500 B fichas.** Exemplos:
  Minerva 8B, MATH +12,6 com 164 B; Llemma 7B, +13,5 com 200 B; InternLM2-Math-7B, +12,9
  com 125 B. Com até 4 B fichas de texto, foram +1,6 a +2,0 (MathPile).
- **Sobre uma base forte, o mesmo CPT rende pouco.** O MathCode-Pile dá +17,0 na Llama-3-8B
  e +2,4 na DeepSeekMath-7B; a Qwen2-Math ganha +6,2 na 7B. A Qwen3-8B-Base já faz 60,8 no
  MATH.
- **A matemática não transfere para a Física.** No UGPhysics, a Qwen2.5-Math-7B-Instruct
  fica +0,14 acima da Qwen2.5-7B-Instruct.
- **Em benchmark de olimpíada, o CPT move 1 a 3 pontos.**
- **Os saltos grandes em raciocínio científico vieram de dados de pergunta e resposta, ou de
  destilação** (MAmmoTH2: TheoremQA +12,1), e não de texto cru.

Não achamos CPT publicado com o antes e o depois nos quatro benchmarks do G2.1. O
registro de decisão do projeto sobre o treino (o ADR-0006, ainda proposto) recomenda não
treinar o ΦGen agora. Pesam a barra sem precedente que achamos, a computação gratuita que
só chega a piloto (a não ser por uma TPU que exige portar o treino para JAX) e quatro
pré-requisitos que não custam GPU.

## 8. Limites

- **Não há gente na correção nem no julgamento.** O corretor é por regras, auditado por
  dois juízes Claude às cegas; nenhum físico revisou nada.
- **Os recortes dos benchmarks são nossos:** o UGPhysics só com resposta única e sem o juiz
  GPT-4o, o OlympiadBench só em inglês, o PHYBench só com gabarito. Nenhum número daqui se
  compara aos dos artigos dos benchmarks.
- **Q8_0 em vez de BF16, e avg@k a T=0,6** em vez do pass@1 guloso. As duas escolhas são
  declaradas e valem nos dois braços.
- **As sondas olham fichas, não paráfrases.** Uma memória parafraseada passa sem sinal.
- **O nulo é um benchmark só, de 100 itens.** A hipótese de que o PHYBench não está nos
  dados das bases vem da data de publicação, não de uma auditoria dos dados, que são privados.
- **O placebo da condição 3 é um par de bases**, não um CPT. Ele mostra a ordem de grandeza
  do ruído da regra, não o comportamento de um ΦGen.
- **A reprodutibilidade do §5 foi medida num motor e numa GPU.** No CUDA do Kaggle, só
  conferimos que as fichas sondadas eram as mesmas, não que a geração se repete.

## 9. Reprodução

Os itens dos benchmarks, as gerações e as listas de suspeitos ficam fora do git. O resto
está no repositório:
- **Protocolo:** `docs/03-evaluation/DOC-12-harness-protocolo-estatistico.md` §3.7, com o histórico
  completo no §3.7.4 (itens 1–32); `configs/eval/preregistration/g2.yaml`;
  `configs/eval/preregistration/g2_congelamento.json`.
- **Travas e regra:** `src/phifm/eval/g2.py`.
- **Geração e correção:** `src/phifm/eval/g2_geracao.py`, `g2_prompt.py`, `g2_itens.py`,
  `g2_corretor.py`; `scripts/medir_g2.py`, `medir_g2_mmlu.py`, `medir_g2_ifeval.py`,
  `medir_g2_evalplus.py`.
- **Sondas:** `src/phifm/eval/g2_sondas.py`, `scripts/medir_g2_sondas.py`.
- **Poder:** `scripts/poder_g2_1.py`.
- **A 8B no Kaggle:** `kaggle/g2_8b_preparo.py`, `kaggle/g2_8b_medida.py`,
  `scripts/publicar_g2_8b.py`, `scripts/trazer_g2_8b.py`.

## 10. De onde vem cada número

Todos estão em `data/processed/avaliacao/`:
- **G2.1:** `g2_teste_1.7B_base_veredito.json` e `g2_teste_8B_base_veredito.json`.
- **MMLU:** `g2_mmlu_teste_*_base.json`.
- **IFEval:** `g2_ifeval_teste_*_base.json`.
- **EvalPlus:** `g2_evalplus_teste_*_base.json`.
- **KL:** `g2_kl_1.7B.json`, `g2_kl_8B.json` e `g2_kl_8B_cuda.json`.
- **Auditoria do corretor:** `g2_auditoria.json`.
- **Sondas:** `g2_sondas_dev_1.7B_base.json`, `g2_sondas_teste_1.7B_base.json` e
  `g2_sondas_teste_8B_base.json`.
- **Poder:** `g2_poder_g2_1.json`.

Os números do dev sobre tipo de resposta, tercil e acerto (§4.2) e as contagens da lista
híbrida e do placebo (§4.4, §4.5) vieram de contagens automáticas, descritas no DOC-12
§3.7.4 itens 28, 30 e 31. A reprodutibilidade (§5) está no item 28. A simulação de 39%
contra 84% (§4.2) é da revisão adversarial descrita no item 28, e não tem script versionado.

## 11. Citações

Conferidas na fonte primária em 2026-11-18:
- **Qwen3 Technical Report**, arXiv 2505.09388: MATH 60,80 (8B-Base, Tab. 6) e 43,50 (1.7B-Base, Tab. 8); 36 T fichas para todos os tamanhos.
- **UGPhysics**, arXiv 2502.00334v4, Tab. 5: 22,69 contra 22,55.
- **MathCoder2**, arXiv 2410.08196, Tab. 2: Llama-3-8B 21,4 → 38,4; DeepSeekMath-7B 36,2 → 38,6.

Lidas por um agente de pesquisa e **não conferidas uma a uma** (conferir antes de usar):
- Minerva (2206.14858), Llemma (2310.10631), InternLM2-Math (2402.06332),
  DeepSeekMath (2402.03300), Qwen2.5-Math (2409.12122), MathPile (2312.17120);
- OctoThinker (2506.20512), MAmmoTH2 (2405.03548), SciLitLLM (2408.15545),
  AstroLLaMA (2409.19750), OmniScience (2503.17604);
- Ibrahim et al. (2403.08763), Llama-3-SynE (2407.18743), SciGLM (2401.07950) e
  Biderman et al. (2405.09673).
