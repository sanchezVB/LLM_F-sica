# Quando a supervisão e a avaliação saem da mesma estrutura: cinco falhas medidas em recuperação de Física supervisionada por citação

**Rascunho.** Não submetido, não revisado por ninguém além de mim. Ver §11 antes de
usar qualquer número daqui, e §12 antes de usar qualquer citação.

**Autoria:** o trabalho experimental é de Vinicius; este rascunho foi escrito por
Claude (Anthropic) a partir das medições do repositório. A ordem e a forma da
autoria são decisão do primeiro.

---

## Resumo

Grafos de citação são a fonte barata de supervisão para recuperação científica: a
citação de A para B é um rótulo de relevância que ninguém precisou anotar. Nós
construímos um sistema de recuperação de Física sobre 6,56 M de arestas de citação do
arXiv e um encoder de 23 M de parâmetros, e medimos cinco falhas que compartilham uma
causa: **a supervisão e a avaliação são derivadas da mesma estrutura**, e cada atalho
na derivação abre um vazamento que se parece com um resultado. A segunda delas tem
duas consequências independentes, e a segunda dessas pegou o resultado positivo deste
próprio artigo (§4.2).

As cinco, com o número que as expõe:

1. **Negativos difíceis minerados do próprio recuperador invertem o reranqueador.**
   Minerar "top-K recuperado menos o positivo" ensina *escore alto do recuperador ⇒
   negativo*. O reranqueador resultante anticorrelaciona com o recuperador em 83% das
   consultas (Spearman entre posição na fusão e escore: **+0,179**). Corrigido, vai a
   **−0,466** e 0% — e então não acrescenta nada, porque concorda.
2. **`head()` num parquet agrupado por documento mede uma fração dos documentos que
   parece medir, e ainda derruba o teto.** 500 linhas eram **35 documentos**; o
   intervalo de 95% do acerto@1 era **±0,159**, largo o bastante para conter tanto o
   resultado bonito quanto o honesto. Uma conclusão publicável ("o modelo não lê a
   consulta") saiu de **16** documentos e se inverteu com 457. E 62% dos itens tinham
   **alvo repetido**, o que dá cosseno idêntico e desempate arbitrário: o teto de um
   modelo perfeito era **0,7562**, não 1,0 — foi o que fabricou o "empate com o
   GTE-large" da tabela do §1 desta versão anterior (§4.2).
3. **Divisão treino/validação por posição não divide** quando 400 mil pares contêm
   17.844 documentos citados distintos. 49,6% das âncoras da "validação" já estavam no
   treino, e o modelo aprendeu identidade de paper em vez de relevância de par: nDCG
   da composição de **0,139 para 0,020** sobre documentos inéditos.
4. **9,1% dos negativos minerados são co-citados com o positivo**, e treinar um
   reranqueador a rebaixá-los é ensiná-lo a rebaixar o que é relevante — com a perda
   descendo normalmente durante o treino.
5. **Corpora de texto pleno extraídos de PDF têm as equações removidas**, e o
   diagnóstico intuitivo para isso satura no corpus bom. Presença de ambiente de
   equação: **0,0%** num corpus extraído de PDF contra **84,9%** no mesmo domínio
   construído do fonte LaTeX.

Nenhuma das cinco aparece na perda de treino. Todas produzem um número que parece
comparável. Quatro delas produziram, no nosso caso, um resultado que eu acreditei
antes de medir de novo.

O diagnóstico da falha 1 fez uma predição, e nós a testamos duas vezes.

**Na primeira (§10) ela se confirmou:** um cross-encoder de base diferente acrescenta
algo — mas só se a base for de domínio. Um modelo de recuperação forte do mesmo
tamanho empata (p = 0,637); o encoder de Física vence (p = 0,0062). E o encoder de
Física é, ele mesmo, um recuperador ruim neste benchmark.

**Na segunda (§10.2) ela se confirmou de um jeito que apagou o resultado da
primeira.** Melhorado o recuperador em 0,098, o mesmo reranqueador deixou de
acrescentar qualquer coisa — cinco medições pareadas, três profundidades, nenhuma o
separa do recuperador sozinho. Dobrar o conjunto de candidatos deu-lhe 195 alvos
novos e ele promoveu **um**. O estágio saiu do sistema pela regra registrada antes.

A lição que sobra é sobre transferência: **um resultado de reranking medido contra um
recuperador fraco não transfere para um recuperador melhor**, e o intervalo entre as
duas medições foi de quatro meses e uma linha de código.

---

## 1. Introdução

Modelos de recuperação de domínio precisam de pares (consulta, documento relevante).
Anotação humana é caro; o grafo de citação é grátis. A prática é conhecida — SPECTER
(Cohan et al., 2020) treina representações de documento científico com sinal de
citação — e a nossa motivação era a mesma: construir um recuperador de Física sem
anotar nada.

O ajuste fino funciona, e **menos do que a primeira versão deste rascunho
afirmava**. Todos os números abaixo estão no protocolo de **teto 1,0000** (ver §4.2:
a versão anterior desta tabela usava um protocolo cujo teto era 0,7562, e o "empate
com o GTE-large" era artefato dele).

| modelo | r@1 | r@10 | nDCG@10 |
|---|---|---|---|
| SciBERT, 110 M (base do ajuste) | 0,149 | 0,383 | 0,2537 |
| PhysBERT, 110 M (domínio) | 0,222 | 0,491 | 0,3507 |
| **MiniLM-L6, 23 M — SEM ajuste** | 0,314 | 0,664 | **0,4761** |
| nosso, 23 M, 400 mil arestas | 0,348 | 0,720 | **0,5246** |
| GTE-large, 335 M (geral) | 0,414 | 0,764 | **0,5788** |
| nosso, 23 M, 6 M de arestas | 0,432 | 0,831 | **0,6223** |

Três leituras que a tabela anterior não permitia. Todos os `p` são de **McNemar
exato pareado** sobre "o alvo chegou ao top-k", nas mesmas 2.000 consultas:

1. **A linha que faltava era a mais importante.** O MiniLM-L6 **sem nenhum ajuste**
   já bate o PhysBERT, e com folga (k=1: 368 × 185, p = 5,7e-15). Então o "+0,190
   sobre o PhysBERT" que a versão anterior atribuía ao ajuste fino por citação era,
   em sua maior parte, **o modelo base** — um encoder de sentença genérico contra um
   encoder de domínio que não foi treinado para similaridade. O ajuste por citação
   acrescenta **+0,049** de nDCG@10 sobre esse ponto de partida (0,4761 → 0,5246), e
   isso é real: k=1 p = 3,8e-05, k=10 p = 4,1e-11. Real, e **cinco vezes menor do que
   o anunciado**.
2. **Com 400 mil arestas o GTE-large vence, e não é empate.** k=1: 132 × 215,
   p = 9,8e-06; k=10 p = 0,0042.
3. **Bater o GTE-large exige 15× mais dado, e só em parte das métricas.** 6 M de
   arestas dão 0,6223 contra 0,5788 — no top-10 é decisivo (206 × 73, p = 7,6e-16),
   e no **top-1 é empate** (188 × 153, p = 0,065). A vitória existe e é parcial.

> ⚠️ **Estes cinco testes pareados não estavam no artefato de avaliação**, que pareia
> tudo contra **um** sistema de referência — o melhor dos nossos. Eu havia escrito a
> tabela acima com as diferenças de nDCG e sem os `p`, o que num artigo sobre rigor
> de medição é o erro que ele denuncia. Os testes foram calculados depois, das
> posições por consulta que o cache guarda; a limitação do artefato está registrada
> na §12.6.

Esse é o resultado positivo, e ele não é a contribuição deste artigo. A contribuição
é o que aconteceu **entre** a primeira versão desse número e essa: **cinco medições
que eu reportei e depois refutei — a desta tabela inclusive — e uma recomendação de
gasto que se refutou junto.** São seis retratações a partir de cinco falhas, porque a
Falha 2 produziu duas (§4.1 e §4.2).

O fio comum: quando o rótulo de treino, o negativo de treino e o rótulo de avaliação
saem todos do mesmo grafo, **cada atalho na derivação de um deles contamina os
outros**, e a contaminação é invisível porque a perda desce e a métrica sobe.

---

## 2. Montagem

**Corpus.** 1,59 M de registros de metadados do arXiv, filtrados para Física por um
classificador com acurácia 0,954 (falso positivo 2,4–3,7% por domínio). O grafo de
citação vem do OpenAlex e dá **6,56 M de arestas**; a validação usa **88.807
documentos citados distintos**.

**Modelos.** Recuperador denso: `all-MiniLM-L6-v2` ajustado com InfoNCE. Léxico: BM25.
Fusão: Reciprocal Rank Fusion (Cormack et al., 2009), k = 60. Reranqueador:
cross-encoder par a par, inicializado da mesma base do recuperador.

**Métrica e teste.** nDCG@10 e recall@k. As comparações entre sistemas são **McNemar
exato** sobre pares discordantes de "o alvo chegou ao top-k". Isto importa mais do que
parece: com 1.000 consultas, a diferença de 0,0091 de nDCG entre duas variantes tem
p = 0,636 — duas proporções independentes teriam sugerido uma diferença que o teste
pareado não sustenta.

**Intervalos de proporção.** Wilson (1927). A normal ingênua dá largura zero quando
não há observação contrária, o que transformaria "não vi nenhum" em "não existe
nenhum" — e é exatamente o regime das taxas medidas aqui.

**Computação.** Uma GPU de consumo de 8 GB (DirectML) e a cota gratuita de T4 do
Kaggle. A restrição é relevante: ela forçou que cada experimento fosse justificado por
uma medição anterior, o que é a razão de as falhas terem sido encontradas.

---

## 3. Falha 1 — negativos do recuperador invertem o reranqueador

### 3.1 O que foi feito

Mineração de negativos difíceis é padrão: DPR (Karpukhin et al., 2020) usa negativos
do BM25, ANCE (Xiong et al., 2021) reamostra do próprio índice em treino. Nós
mineramos do índice denso: para cada âncora, os top-K recuperados **menos** o
positivo conhecido.

### 3.2 O que isso ensina

O rótulo passa a ser predito pela posição no recuperador. Todo negativo está no
top-K; o positivo, por construção do conjunto, com frequência **não** está. O
cross-encoder aprende a regra mais simples que separa os dois: *se o recuperador
gostou, é negativo.*

Medido, sobre 1.000 consultas:

| | negativos minerados do índice | negativos do RRF top-50 real |
|---|---|---|
| Spearman(posição na fusão, escore) | **+0,179** | **−0,466** |
| consultas com rho > 0 | **83%** | **0%** |

Com posição menor = melhor, o sinal desejado é negativo. A primeira coluna é um
reranqueador que **prefere a cauda do recuperador**.

### 3.3 A correção, e por que ela não bastou

A correção é montar o grupo de treino com a **distribuição exata da avaliação**: os
50 candidatos que a fusão RRF de fato produz, incluindo o positivo quando ele está
lá. "Estar no topo" deixa de predizer o rótulo.

O reranqueador consertado é forte dentro do grupo — acerto@1 de **0,498 ± 0,045**
contra 0,125 do acaso e 0,20–0,25 do próprio RRF nos mesmos grupos — e **não acrescenta
nada ao sistema**:

| sistema | r@1 | r@10 | nDCG@10 | McNemar vs fusão (k=10) |
|---|---|---|---|---|
| BM25 | 0,067 | 0,236 | 0,1399 | p = 0,00073 (fusão vence) |
| denso | 0,055 | 0,233 | 0,1327 | p = 0,00018 (fusão vence) |
| **fusão RRF** | 0,068 | 0,271 | **0,1584** | — |
| fusão + reranqueador | 0,064 | 0,254 | 0,1493 | p = 0,118 (empate) |

A explicação é a própria correção: o reranqueador parte da **mesma base** do
recuperador, e agora concorda com ele (−0,466). Um reranqueador que re-deriva a ordem
do recuperador não tem informação nova para dar, por bem treinado que esteja.

> **A lição não é "não minere do recuperador".** É que o grupo de treino tem de ter a
> distribuição do grupo de inferência, e que **um reranqueador cuja base é a do
> recuperador é redundante por construção** — o que é uma predição testável.
>
> ⚠️ **Ela foi testada duas vezes, e a formulação acima está errada por um termo.**
> A §10 confirma a primeira metade: um cross-encoder de base diferente bate a fusão.
> A §10.2 mostra que a segunda estava mal enunciada — não é a **base** que decide,
> é **o quanto o recuperador já é bom**. Quando ele melhorou 0,098, o mesmo
> reranqueador de base diferente parou de acrescentar qualquer coisa.

### 3.4 O que isso acrescenta ao que já se sabia

O RocketQA (Qu et al., 2021) foi lido integralmente para esta seção, e a distinção se
sustenta em três eixos.

**O que o RocketQA diz.** Minerar os melhores colocados como negativos "is likely to
bring false negatives (i.e., unlabeled positives), since the annotators can only
annotate a few top-retrieved passages". A correção deles: *"utilize a well-trained
cross-encoder to remove top-retrieved passages that are likely to be false
negatives"*. No pipeline, o cross-encoder é o **filtro** e o que se treina é o
**dual-encoder** — o recuperador.

**Onde o nosso caso difere.**

| | RocketQA | aqui |
|---|---|---|
| o que se treina | o **recuperador** (dual-encoder) | o **reranqueador** (cross-encoder), que roda *depois* do recuperador |
| natureza do defeito | **ruído de rótulo**: o negativo é, de fato, relevante | **o critério de seleção correlaciona com o escore do modelo a ser reordenado** |
| o que conserta | filtrar o conjunto de **negativos** | trocar a origem do **positivo**, que passa a vir do mesmo top-K |

O terceiro eixo é o que mais importa: **o nosso defeito sobreviveria à correção
deles**. Ainda que todo negativo minerado fosse genuinamente irrelevante — zero falso
negativo, o filtro do RocketQA perfeito —, o rótulo continuaria predito pela posição,
porque o **positivo** está ausente do top-K por construção do conjunto. Filtrar
negativos não move essa correlação; mudar de onde vem o positivo, sim.

E o sintoma é outro: uma **inversão mensurável de correlação** (Spearman +0,179, 83%
das consultas), não uma queda de precisão. Custa um Spearman diagnosticar, e nenhuma
curva de perda mostra.

**Onde o nosso caso NÃO difere, e a §6 é deles.** O problema de falso negativo da §6
— co-citados rotulados como negativos — é exatamente o "unlabeled positives" do
RocketQA. Ali não há distinção a reivindicar: a contribuição da §6 é um *proxy* de
grafo barato (co-citação) no lugar de uma passagem de cross-encoder, mais a taxa
medida. Ver §6.

---

## 4. Falha 2 — n efetivo em dados agrupados

O parquet de pares vem **agrupado por documento citado**. Cada documento aparece 14 a
22 vezes. Então:

```
val.head(  200) ->  200 linhas ·  16 documentos
val.head(  500) ->  500 linhas ·  35 documentos
val.sample(500) ->  500 linhas · 259 documentos
```

Linhas do mesmo documento não são observações independentes. Com 35 documentos, o
acerto@1 de 0,364 tinha intervalo de 95% de **±0,159** — e a divisão contaminada e a
honesta reportaram o mesmo número **porque as duas mediam as mesmas três dezenas de
papers**.

> ⚠️ **Isto é confirmação, não observação nova.** É o problema de *unidade de
> análise* da inferência com agrupamento, e a referência de prática é Cameron &
> Miller (2015) — ver §12.2. O que a §4 acrescenta é estreito e específico: que
> **nomear as linhas de "grupos" no código esconde a diferença**, e que nenhum teste
> pega isso porque nada está errado, só mal-nomeado.

### 4.1 A conclusão que isso produziu e destruiu

Investigando por que o reranqueador não ganhava, medi o escore com a consulta
substituída por string vazia. Resultado: 0,355 com consulta vazia contra 0,390 com a
real. Conclusão registrada: *"o modelo não lê a consulta"*.

Amostra: **16 documentos**.

Com 457 documentos, os mesmos dois números viram **0,371 contra 0,229**, um ganho de
**+0,143 ± 0,059**. O modelo lê a consulta. E havia, no mesmo relatório, um resultado
que contradizia a conclusão — trocar a consulta por outra dava 0,160, próximo do
acaso — que eu li e não conectei.

> A falha não foi calcular mal. Foi chamar as linhas de "grupos" no código e nas
> tabelas. Um nome que confunde a unidade de amostragem com a unidade de observação
> esconde a diferença entre 500 e 35, e nenhum teste pega isso porque nada está
> errado — só mal-nomeado.

**A correção** é uma função de quatro linhas que devolve `(amostra, documentos
distintos)` e obriga o segundo valor a atravessar a métrica até o relatório, junto do
intervalo de confiança que ele implica.

### 4.2 ⚠️ A segunda consequência, que pegou o resultado positivo deste artigo

O intervalo largo é o efeito óbvio de poucos documentos distintos. Há um segundo, e
ele não é sobre incerteza — é sobre o **teto**.

O pool de candidatos do nosso portão saía de `val.head(2000)`. Num parquet agrupado
por documento citado, isso dá **62% de itens com alvo repetido**: dois candidatos com
o **texto idêntico**. Texto idêntico produz vetor idêntico, cosseno idêntico, e o
desempate do `argsort` é arbitrário. Um modelo **perfeito** nesse pool tem
nDCG@10 = **0,7562**, não 1,0.

As consequências:

- **toda comparação fica comprimida** contra um teto 24% mais baixo, e diferenças
  entre modelos bons encolhem junto;
- a primeira versão da tabela do §1 reportava **0,4657 contra 0,4628 do GTE-large** e
  concluía "empata estatisticamente". No protocolo de teto 1,0000, os mesmos dois
  modelos dão **0,5246 contra 0,5788**: o GTE-large **vence por 0,054**. O empate era
  a régua, não o modelo;
- e um portão do programa "passou" por **+0,003** nesse regime, número que depois se
  revelou ruído de desempate.

**A correção** é sortear o pool em vez de cortar, desduplicar pelo **texto** (é o
texto idêntico que produz cosseno idêntico, não o identificador), e **imprimir o teto
do protocolo junto de cada resultado**, recusando o relatório se o teto ficar abaixo
de 0,999.

> A lição é mais estreita e mais dura que a da §4.1: um número de avaliação sem o
> **teto do próprio protocolo** ao lado não diz se o limite é o modelo ou a régua. E
> um artigo sobre armadilhas de medição publicou uma tabela medida com a régua torta
> — a dele mesmo.

---

## 5. Falha 3 — divisão por posição não divide

A primeira divisão treino/validação separava as **últimas 8.000 linhas**, com um
comentário dizendo que era para evitar vazamento. Medido depois:

| | |
|---|---|
| âncoras da "validação" já vistas no treino | **49,6%** |
| documentos citados já vistos no treino | **39,2%** |

A causa é um número que o repositório já tinha medido três dias antes: os 400 mil
pares têm apenas **17.844 documentos citados distintos**. Cortar por posição num
conjunto assim não separa nada — os mesmos papers caem dos dois lados.

O modelo aprendeu "este paper específico é positivo" em vez de "este par é
relevante". Reportou acerto@1 de 0,370 e, sobre documentos inéditos, derrubou o nDCG
da composição de **0,139 para 0,020**. O conjunto de validação real tem 88.807
citados distintos, dos quais o reranqueador tinha visto 4,8%.

**A correção** é dividir por documento citado, com uma verificação que **levanta** se
qualquer documento aparecer dos dois lados. A métrica cai — o número honesto é menor
que o inflado, e é o que serve para decidir.

> ⚠️ **Também confirmação.** A comunidade de sistemas de recomendação tem literatura
> direta sobre vazamento por estratégia de divisão — Ji et al. (2023), Meng et al.
> (2020). O mecanismo deles é *temporal* e o nosso é *posicional num conjunto
> derivado de grafo*: instâncias diferentes da mesma classe. Ver §12.2. O que sobra
> de próprio aqui é a magnitude medida no caso de citação.

---

## 6. Falha 4 — co-citação como falso negativo

Dois papers citados juntos por um terceiro são um sinal clássico de relevância —
Small (1973) introduziu a co-citação exatamente como medida de relação entre dois
documentos. Na nossa mineração, **9,1% dos negativos** (62.646 de 688.136) eram
co-citados com o positivo.

E a taxa **subiu** quando o recuperador melhorou: com os candidatos minerados do
recuperador de 2026-09-08, **12,51%** (98.331 de 786.063). Um recuperador melhor traz
ao topo mais documentos topicamente próximos, e proximidade tópica é o que faz dois
papers serem citados juntos — então **o filtro importa mais à medida que o
recuperador melhora**, não menos. Treinar o reranqueador a rebaixá-los é ensiná-lo a rebaixar o que é
relevante, e a perda de treino desce normalmente enquanto isso acontece — porque, do
ponto de vista da perda, o rótulo é o rótulo.

**Isto é o "unlabeled positives" do RocketQA, e não reivindicamos distinção.** Qu et
al. (2021) nomeiam o problema e o corrigem com uma passagem de cross-encoder sobre os
candidatos. O que esta seção acrescenta é operacional: **um proxy de grafo custa uma
junção e nenhuma GPU**, e a taxa dele é mensurável contra um controle — 9,1% nos
negativos minerados contra **0,1%** num controle de negativos sorteados, razão de
**212×**. Onde existe grafo de citação, o proxy dá a maior parte do efeito pelo preço
de um `join`.

**Ressalva que precisa ser dita:** co-citação aproxima relevância, não a define. Um
paper relacionado que ninguém citou junto com o positivo continua passando como
negativo, e **esse residual não é medido**. O filtro remove uma classe de falso
negativo que sabemos nomear; não sabemos o tamanho do resto — e é precisamente o
resto que o cross-encoder do RocketQA alcançaria.

---

## 7. Falha 5 — a extração de PDF remove as equações, e o diagnóstico satura

Para pré-treino com objetivo consciente de equações, o corpus precisa ter equações.
Medimos quatro fatias do mesmo domínio, 3.000 documentos cada, sorteados:

| fatia | ch/doc | LaTeX % | `$…$` % | **ambiente de equação %** | seq/doc |
|---|---|---|---|---|---|
| resumos do arXiv (referência) | 1.123 | 21,8 | 26,9 | 0,0 | 0,9 |
| arXiv via fonte LaTeX | 49.212 | 100,0 | 99,6 | **84,9** | 1.158,3 |
| páginas web de matemática | 16.009 | 78,5 | 85,2 | 5,1 | 57,7 |
| texto pleno extraído de PDF | 28.605 | 16,3 | 18,2 | **0,0** | 1,0 |

Zero por cento contra 84,9%. O corpus extraído de PDF é 28,6 mil caracteres por
documento de Física **sem uma única equação em display**.

### 7.1 O diagnóstico intuitivo, e por que ele engana

A assinatura natural para "a equação foi arrancada" é o **operador órfão**: um ` = `
sem operando de um dos lados, como em `"where  =  is the rest energy"`. Ele funciona,
e **satura no corpus bom**: acusa 81,3% dos documentos do corpus de fonte LaTeX,
contra 3,0% da referência, e a inspeção mostra matemática intacta. A causa é banal —
em `$Z_{\rm max}$ = 15 kpc`, o caractere antes de ` = ` é `$`.

Um diagnóstico que dispara mais no corpus íntegro que no mutilado é pior que nenhum,
e foi com ele que eu concluí, em primeira instância, que o problema era resolúvel
apenas comprando acesso ao fonte. O discriminante que funciona é a presença do
ambiente, não a ausência do operando.

### 7.2 O que isso custou

Com base na primeira medição, recomendei ao usuário comprar acesso em massa ao arXiv
como **pré-requisito** do projeto. Um corpus de 835.379 documentos de Física
construído do fonte LaTeX estava no disco havia dezessete dias.

> A regra que o repositório aplica a GPU — medir antes de gastar cota — eu não
> apliquei a dinheiro. É a falha mais barata de evitar das cinco, e a única que
> teria custado ao usuário diretamente.

---

## 8. Duas alavancas que não moveram nada

Não é falha, é resultado nulo, e vale registrar porque as duas são o que se faria por
reflexo:

| variação | nDCG@10 | vs campeão | McNemar |
|---|---|---|---|
| **campeão: 400 mil pares, 127 negativos** | **0,4579** | — | — |
| 400 mil pares, **511 negativos** (GradCache) | 0,4486 | −0,0093 | p = 0,636 |
| **1,5 M pares**, 127 negativos | 0,4520 | −0,0059 | p = 0,950 |

Quadruplicar os negativos in-batch e multiplicar os dados por 3,75 nominalmente
**pioram**, e os testes pareados dizem empate nos dois casos. O enunciado honesto não
é "piorou": é que **nenhuma das duas compra nada mensurável nesta escala**, e o
orçamento de computação delas foi gasto sem retorno.

Um resultado adjacente do mesmo tipo, em tokenização: BPE contra Unigram em 200 mil
resumos de Física, mesmo vocabulário e mesmas regras de pré-tokenização. Bostrom &
Durrett (2020) favorecem Unigram em linguagem natural, e não havia evidência para
LaTeX. **BPE ganha**, com margem seis vezes maior em equações (13%) que em prosa
(2%), e mecanismo observável: o Unigram aprendeu 0 de 3 sequências LaTeX de teste como
token único; o BPE, 2 de 3.

---

## 9. O que junta as cinco

Em cada uma, a mesma estrutura:

| | o atalho | o que ele acopla | o sintoma que não aparece |
|---|---|---|---|
| 1 | negativos = top-K menos o positivo | critério do negativo ↔ escore do modelo | perda desce, correlação inverte |
| 2 | `head()` para amostrar | unidade de amostragem ↔ unidade de observação | métrica plausível, intervalo enorme |
| 3 | divisão por posição de linha | treino ↔ validação | métrica **sobe** |
| 4 | negativo = não-citado | rótulo ↔ estrutura de citação parcial | perda desce |
| 5 | corpus por conveniência de formato | conteúdo ↔ pipeline de extração | contagem de tokens correta, sinal ausente |

O denominador é o acoplamento. Grafos de citação são atraentes porque um único objeto
— o grafo — fornece rótulo positivo, critério de negativo e verdade de avaliação. É
essa economia que abre os cinco vazamentos: **não há uma segunda fonte contra a qual
conferir**.

**A prática que sobreviveu**, em três itens que custam pouco:

1. **Reportar sempre o n efetivo** ao lado da métrica, e o intervalo que ele implica.
   Onde a métrica agrega por grupo, o n é o número de grupos distintos.
2. **Medir a correlação entre o escore do reranqueador e a posição do recuperador.**
   Um Spearman. Ele expõe a Falha 1 antes de qualquer avaliação de ponta a ponta.
3. **Fazer a divisão levantar exceção**, não avisar. Um `logging.warning` de
   vazamento é lido depois de o resultado já ter sido reportado.
4. ⚠️ **Escolher o teste que casa com o que a peça faz.** Nós pré-registramos
   McNemar sobre "o alvo chegou ao top-k" para julgar um reranqueador — e esse teste
   mede **pertencimento**, sendo cego para ordenação *dentro* do top-k, que é metade
   do trabalho de um reranqueador. Um modelo que ordenasse o top-10 perfeitamente
   mudaria muito o nDCG e nada o recall@10.

   Registrar a regra antes é necessário e não basta: **a regra pode estar
   pré-registrada e mesmo assim ser o instrumento errado.** A correção é barata —
   calcular também um pareado sobre a métrica contínua (bootstrap sobre o nDCG por
   consulta) — e no nosso caso os dois concordaram (§10.2), o que é sorte, não
   método.

---

## 10. A predição da §3.3, testada duas vezes — e o que a segunda apagou

A explicação da §3.3 — redundância informacional entre reranqueador e recuperador —
faz uma predição falsificável: um cross-encoder de base **diferente** deve bater a
fusão. Nós a testamos com regra de decisão registrada antes de medir (McNemar em
k = 10 contra a fusão da mesma execução, limiar de Bonferroni 0,025 por serem duas
variantes) e as quatro leituras possíveis escritas de antemão.

    variante   base                       params  acc@1  +ΦRank       Δ  disc  p(k=10)
    controle   MiniLM-L6 (= a do ΦEmb)       23M  0,498  0,1483  -0,0093   229   0,1458
    gte        gte-base (geral forte)       109M  0,510  0,1530  -0,0046   220   0,6371
    phys       physbert (Física)            109M  0,566  0,1666  +0,0090   237   0,0062

    fusão RRF = 0,1576 nas três · 2.000 consultas · profundidade 50 · teto 0,4495

**A predição se confirma, e de forma mais estreita do que ela afirmava.** Base
diferente não basta: `gte-base`, um modelo de recuperação forte com o **mesmo número
de parâmetros** que o vencedor, empata com a fusão (p = 0,637). O que paga é
**pré-treino no domínio**.

As duas variantes compartilham tamanho (109 M), arquitetura (`BertModel`), os seis
hiperparâmetros, os dados de treino, a semente e o protocolo de avaliação. A única
diferença é o corpus de pré-treino, e é isso que sustenta a leitura causal.

⚠️ O braço `gte` rodou numa execução separada, e a combinação só é legítima porque o
**controle saiu byte a byte idêntico nas duas** — mesmos `sistemas`, mesmo pareado,
p = 0,14584 sobre 229 discordantes em ambas.

### 10.1 A assimetria que não estava prevista

O encoder de domínio é um **recuperador ruim** neste benchmark (nDCG **0,3507**
contra 0,5246 do nosso ajuste fino de 400 mil arestas, e 0,4761 do MiniLM-L6 **sem
ajuste algum** — números do protocolo de teto 1,0000, ver §1 e §4.2) e a **melhor
base de reranqueador** das três testadas. Pré-treino de domínio não produziu um bi-encoder
competitivo aqui e produziu um cross-encoder competitivo.

Não temos explicação mecanística para a assimetria, e não vamos inventar uma. A
hipótese barata de testar é que o cross-encoder pode usar interação termo a termo
entre consulta e documento, onde vocabulário de domínio rende, enquanto o bi-encoder
precisa comprimir o documento num vetor antes de ver a consulta. É especulação até
alguém medir.

---

### 10.2 A mesma predição, testada de novo — e o estágio saiu do sistema

O resultado do §10 foi medido contra o recuperador de então. Depois dele, o
recuperador melhorou: 6 M de arestas em vez de 400 mil, **+0,098 de nDCG@10** no
protocolo do portão. E a explicação da §3.3 faz uma segunda predição, que a primeira
formulação não separava: se o valor do reranqueador vem do que o recuperador deixa
para trás, **um recuperador melhor deve encolher esse valor**.

Encolheu até desaparecer. Cinco medições pareadas, com regra registrada antes de
cada uma:

| medição | reranqueador | profundidade | vence o recuperador sozinho? |
|---|---|---|---|
| 2026-09-08 | PhysBERT (o do §10) | 100 | não, p=0,139 |
| 2026-09-09 | PhysBERT retreinado nos negativos novos | 100 | não, p=0,379 |
| 2026-09-10 | PhysBERT (o do §10) | 50 | não, p=0,166 |
| 2026-09-10 | " | 100 | não, p=0,139 |
| 2026-09-10 | " | 200 | não, p=0,138 |

#### O mecanismo, e ele é mais forte que os cinco empates

Dobrar o conjunto de candidatos de 100 para 200 deu ao reranqueador **195 alvos
novos** — consultas cujo documento certo passou a estar no conjunto. Ele trouxe
**+1** para o top-10.

Não é "traz menos". É ~nenhum, de 195 oportunidades. **O reranqueador não consegue
promover um documento que o recuperador colocou entre a posição 100 e a 200**, o que
é a redundância da §3.3 medida diretamente: consertados os negativos, o Spearman
entre escore do reranqueador e posição do recuperador é **−0,466**, e um modelo que
concorda com o recuperador não tem por que discordar dele em lugar nenhum.

E há um segundo achado, que o critério pré-registrado não teria pego. Das **421**
consultas com o alvo no top-10 nos dois sistemas, o reranqueador **desce** o alvo em
167 e **sobe** em 141. Ele não ordena melhor dentro do top-10 — ordena um pouco
pior. Todo o ganho aparente de nDCG (+0,0091) vem de pertencimento, +27 líquido, e
**nada** de ordenação.

#### O que isto corrige na §3.3

A frase era "um reranqueador cuja **base** é a do recuperador é redundante por
construção". O §10 refutou metade: base diferente **venceu**, com p=0,0062.

A formulação certa é sobre **regime**, não sobre base: *o valor de um reranqueador é
limitado pelo que o recuperador deixa para trás, e some quando o recuperador melhora
o bastante*. O reranqueador do §10 não piorou — o recuperador subiu por baixo dele.

**A consequência prática é desconfortável e generalizável: um estágio de reordenação
tem de ser re-medido a cada mudança do recuperador, e um resultado de reranking
publicado contra um recuperador fraco não transfere.** O nosso ganho de p=0,0062
tinha quatro meses e continuou verdadeiro sobre aquele recuperador; sobre o
seguinte, não.

O estágio saiu da composição pela regra registrada antes. O sistema passou a ser
`recuperador → top-10`, e o cross-encoder de 109 M de parâmetros deixou de existir
em serviço.

---

## 11. ⚠️ Limitações, sem atenuação

1. **Um domínio, um encoder base, um grafo.** Física, MiniLM-L6, OpenAlex. Nada aqui
   estabelece que as taxas transferem. A Falha 1 é a única com mecanismo geral
   argumentável, e mesmo ela foi observada uma vez.
2. **Benchmark próprio, sem juízo humano de relevância.** "Relevante" = "citado". Isso
   é uma proxy conhecidamente enviesada — favorece papers citáveis, campos com cultura
   de citação densa, e não captura relevância que ninguém citou. O nDCG de 0,1584 da
   fusão não é comparável a nDCG de benchmarks com anotação.
3. **O resultado positivo do §1 encolheu quando a régua foi consertada.** Com 400 mil
   arestas o GTE-large **vence** (0,5788 contra 0,5246); bater ele exige 6 M de
   arestas, e mesmo então o pareado em recall@1 dá p = 0,065. E o ajuste fino por
   citação vale **+0,049** sobre o MiniLM-L6 sem ajuste, não os +0,190 que a versão
   anterior atribuía a ele — a maior parte daquela margem era o modelo base. Ver
   §4.2.
4. **A §6 não mede o falso negativo residual**, e a §7 não mede a taxa de
   contaminação do corpus de fonte LaTeX (o julgamento humano da amostra está
   pendente, com alvo pré-comprometido de 200 documentos).
5. **`lr` não foi re-ajustado** entre escalas de modelo no experimento da §10, o que
   é uma variável não controlada declarada antes de ver o resultado.
6. **A §4 e a §5 são confirmação, não observação nova.** A conferência de
   referências de 2026-09-08 achou literatura anterior para as duas — inferência com
   agrupamento (Cameron & Miller, 2015) e vazamento por estratégia de divisão em
   sistemas de recomendação (Ji et al., 2023; Meng et al., 2020). O que resta de
   próprio é a magnitude no caso de citação e a observação de nomenclatura da §4.
   Ver §12.2.
7. **Veículo realista: workshop.** Um artigo de armadilhas com validade externa de um
   domínio não é contribuição de conferência principal, e apresentá-lo como tal seria
   o mesmo erro de calibração que ele denuncia.

---

## 12. Referências — conferidas uma por uma em 2026-09-08

A versão anterior desta seção dizia: *"escrevi as referências abaixo de memória.
Nenhuma foi conferida contra a fonte."* Foram conferidas agora, cada uma contra a
fonte primária ou contra DBLP/ACL Anthology/JSTOR. **A conferência achou três coisas
substantivas**, e nenhuma delas era um número de página.

### ⚠️ 12.1 Uma citação minha afirmava o CONTRÁRIO do que o trabalho conclui

Eu havia escrito: *"Ali et al. (2024) — eficiência de tokenizer correlaciona com
desempenho a jusante."*

O trabalho conclui o oposto sobre essas métricas. Treinando 24 LLMs de 2,6 B
parâmetros, ele relata que **fertilidade e paridade não são sempre preditivas do
desempenho a jusante**, tornando-as um *proxy* questionável.

O erro é meu e a direção importa: a conclusão de Ali et al. **sustenta** o desenho do
bake-off deste programa (fertilidade é proxy, o modelo treinado é que decide) em vez
de o dispensar. Uma citação invertida num artigo sobre rigor de medição seria a pior
forma de se desmentir, e é exatamente por isso que a seção anterior existia.

### ⚠️ 12.2 A §4 e a §5 têm literatura anterior. As duas passam a ser confirmação

A seção anterior previa isto e chamava de "informação útil, não problema". As duas
existem:

**Para a §4 (n efetivo em dados agrupados)** — é o problema de *unidade de análise*
da inferência com agrupamento, e a referência de prática é Cameron & Miller (2015).
A contribuição da §4 não é notar que linhas agrupadas não são independentes; é a
observação estreita de que **nomear as linhas de "grupos" no código esconde a
diferença**, e que nenhum teste pega isso porque nada está errado — só mal-nomeado.

**Para a §5 (divisão por posição)** — a comunidade de sistemas de recomendação tem
literatura direta sobre vazamento por estratégia de divisão: Ji et al. (2023) e Meng
et al. (2020). O mecanismo deles é *temporal* e o nosso é *posicional num conjunto
derivado de grafo*, o que os torna instâncias diferentes da mesma classe — e a §5
tem de se posicionar assim, não como observação nova.

O que sobra de próprio na §5 é a magnitude medida no caso de citação: 49,6% das
âncoras da "validação" já vistas no treino, e o nDCG da composição caindo de 0,139
para 0,020 sobre documentos inéditos.

### ⚠️ 12.3 A §6 estava sem a referência que a funda

A §6 afirma que "dois papers citados juntos por um terceiro são um sinal clássico de
relevância" e não citava a origem. É Small (1973), que introduziu a co-citação
exatamente como medida de relação entre dois documentos.

### 12.4 Confirmado: de onde vem o k = 60 do RRF

Conferido no PDF do artigo, não de memória:

> *"where k = 60 was fixed during a pilot investigation and not altered during
> subsequent validation"*

e, na validação:

> *"k = 60 was near-optimal, but that the choice was not critical"*

Então o k = 60 é herança de um piloto, e **os próprios autores dizem que a escolha
não é crítica**. Quem repetir este trabalho não precisa defender o 60; precisa
declarar que o usou.

### 12.5 As referências, com a forma conferida

**Recuperação e negativos difíceis**

- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion
  outperforms condorcet and individual rank learning methods. *SIGIR '09*, 758–759.
- Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., &
  Yih, W. (2020). Dense passage retrieval for open-domain question answering.
  *EMNLP 2020*, 6769–6781.
- Xiong, L., Xiong, C., Li, Y., Tang, K.-F., Liu, J., Bennett, P. N., Ahmed, J., &
  Overwijk, A. (2021). Approximate nearest neighbor negative contrastive learning
  for dense text retrieval. *ICLR 2021*. arXiv:2007.00808.
- Qu, Y., Ding, Y., Liu, J., Liu, K., Ren, R., Zhao, W. X., Dong, D., Wu, H., &
  Wang, H. (2021). RocketQA: An optimized training approach to dense passage
  retrieval for open-domain question answering. *NAACL-HLT 2021*, 5835–5847.
  **É o vizinho mais próximo da §3** — as três contribuições dele são negativos
  entre lotes, *amostragem de negativos difíceis desruidada* e aumento de dados, e
  a segunda usa um cross-encoder para filtrar falso negativo. Continua merecendo
  leitura integral antes de a §3.4 afirmar distinção.
- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework:
  BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.

**Citação como sinal**

- Small, H. (1973). Co-citation in the scientific literature: A new measure of the
  relationship between two documents. *Journal of the American Society for
  Information Science*, 24(4), 265–269.
- Cohan, A., Feldman, S., Beltagy, I., Downey, D., & Weld, D. S. (2020). SPECTER:
  Document-level representation learning using citation-informed transformers.
  *ACL 2020*, 2270–2282.

**Vazamento por estratégia de divisão** (ver §12.2)

- Ji, Y., Sun, A., Zhang, J., & Li, C. (2023). A critical study on data leakage in
  recommender system offline evaluation. *ACM Transactions on Information Systems*,
  41(3), Article 75, 1–27.
- Meng, Z., McCreadie, R., Macdonald, C., & Ounis, I. (2020). Exploring data
  splitting strategies for the evaluation of recommendation models. *RecSys 2020*,
  681–686.

**Inferência com dados agrupados** (ver §12.2)

- Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust
  inference. *Journal of Human Resources*, 50(2), 317–372.
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical
  inference. *Journal of the American Statistical Association*, 22(158), 209–212.
- Brown, L. D., Cai, T. T., & DasGupta, A. (2001). Interval estimation for a
  binomial proportion. *Statistical Science*, 16(2), 101–133.
- McNemar, Q. (1947). Note on the sampling error of the difference between
  correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.

**Tokenização**

- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare
  words with subword units. *ACL 2016*, 1715–1725.
- Kudo, T. (2018). Subword regularization: Improving neural network translation
  models with multiple subword candidates. *ACL 2018*, 66–75.
- Bostrom, K., & Durrett, G. (2020). Byte pair encoding is suboptimal for language
  model pretraining. *Findings of EMNLP 2020*, 4617–4624.
- Tao, C., Liu, Q., Dou, L., Muennighoff, N., Wan, Z., Luo, P., Lin, M., & Wong, N.
  (2024). Scaling laws with vocabulary: Larger models deserve larger vocabularies.
  *NeurIPS 2024*. arXiv:2407.13623.
- Ali, M., Fromm, M., Thellmann, K., Rutmann, R., Lübbering, M., Leveling, J., et al.
  (2024). Tokenizer choice for LLM training: Negligible or crucial? *Findings of
  NAACL 2024*, 3907–3924. **Ver §12.1** — o achado é que fertilidade e paridade
  **não** são sempre preditivas do desempenho a jusante.

**Modelos e corpora**

- Hellert, T., Montenegro, J., & Pollastro, A. (2024). PhysBERT: A text embedding
  model for physics scientific literature. *APL Machine Learning*, 2(4), 046105.
  arXiv:2408.09574. É a referência do `thellert/physbert_cased` usado como alvo do §1.
- Lo, K., Wang, L. L., Neumann, M., Kinney, R., & Weld, D. S. (2020). S2ORC: The
  Semantic Scholar Open Research Corpus. *ACL 2020*, 4969–4983.
- Soldaini, L., & Lo, K. (2023). *peS2o (Pretraining Efficiently on S2ORC) Dataset*.
  Allen Institute for AI. ODC-By. Derivado do S2ORC (Lo et al., 2020); é o corpus
  de texto pleno da §7.
- Weber, M., Fu, D. Y., Anthony, Q., Oren, Y., Adams, S., Alexandrov, A., et al.
  (2024). RedPajama: An open dataset for training large language models. *NeurIPS
  2024, Datasets and Benchmarks Track*. arXiv:2411.12372. É a fonte da fatia arXiv
  da §7 (`togethercomputer/RedPajama-Data-1T`, revisão `398f9257`).

### 12.6 O que ainda falta, e agora é específico

1. ~~Ler o RocketQA integralmente~~ — **feito**. A §3.4 foi reescrita com o que o
   artigo diz de fato, e a distinção se sustenta em três eixos (o que se treina, a
   natureza do defeito, o que conserta). O terceiro é o que decide: **o nosso defeito
   sobreviveria à correção deles**, porque filtrar negativos não move uma correlação
   que vem da ausência do positivo. Em contrapartida, a **§6 é o problema deles** e
   passou a dizer isso.
2. **Reescrever a §4 e a §5** com o enquadramento do §12.2 — confirmação num
   domínio novo, não observação nova. A magnitude medida continua sendo nossa.
3. **Procurar vazamento posicional especificamente em conjuntos derivados de grafo
   de citação.** A busca encontrou a literatura de recomendação (temporal) e não
   encontrou o caso posicional-em-grafo; ausência de resultado numa busca não é
   ausência de literatura, e esta é a lacuna que restou.
4. ⚠️ **O avaliador de encoders pareia tudo contra UM sistema de referência.** Os
   cinco testes pareados da tabela do §1 não estavam no artefato: ele compara cada
   modelo com o melhor dos nossos e com mais ninguém. Foi possível calculá-los das
   posições por consulta que o cache guarda, mas o artefato de avaliação deveria
   trazê-los. É o mesmo defeito estrutural que o avaliador da composição tinha até
   2026-09-08, quando uma regra pré-registrada ficou sem o teste que nomeava.
5. ⚠️ **Há duas convenções de "posição" no repositório**, e elas se encontram numa
   função compartilhada. O avaliador da composição usa 0-based (`lista.index`); o
   cache do avaliador de encoders usa **1-based** (`x == 1` é o topo). O
   `mcnemar_em` espera 0-based, então alimentá-lo com o cache calcula
   silenciosamente o **top-(k−1)** — aconteceu na primeira tentativa de produzir os
   `p` acima, e só apareceu porque o teste em k=1 devolveu "0 discordantes" para
   modelos com recall@1 de 0,31 e 0,22, o que é impossível. Um número um pouco
   errado não teria acusado.


---

## 13. Reprodutibilidade

Todo número deste rascunho vem de um artefato versionado no repositório: cada etapa
grava um manifesto com hashes BLAKE3 das entradas e saídas, os parâmetros e o SHA do
commit. Os scripts que produzem cada tabela:

| seção | script |
|---|---|
| §1, §8 | `scripts/avaliar_encoders.py` |
| §3 | `scripts/minerar_do_recuperador.py`, `scripts/avaliar_t1b.py` |
| §4 | `src/phifm/training/amostragem.py` |
| §5 | `scripts/train_rerank.py` (`_dividir_por_documento`) |
| §6 | `scripts/filtrar_cocitacao.py` |
| §7 | `scripts/medir_equacoes_mutiladas.py` |
| §8 (tokenizador) | `scripts/bakeoff_tokenizer.py` |
| §10 | `kaggle/t1c_phirank.py` |
| §10.2 | `kaggle/t1d_phirank_denso.py`, `kaggle/t1e_profundidade.py` |
| §10.2 (o mecanismo) | `scripts/diagnosticar_teto.py` |

O que **não** é redistribuível: o corpus. Os resumos do arXiv seguem a licença de
cada submissão, e a licença padrão do arXiv concede ao arXiv o direito de distribuir,
não a terceiros. Os scripts de coleta e os manifestos permitem reconstruir; os bytes
não acompanham.
