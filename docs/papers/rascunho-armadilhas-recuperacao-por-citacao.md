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
na derivação abre um vazamento que se parece com um resultado.

As cinco, com o número que as expõe:

1. **Negativos difíceis minerados do próprio recuperador invertem o reranqueador.**
   Minerar "top-K recuperado menos o positivo" ensina *escore alto do recuperador ⇒
   negativo*. O reranqueador resultante anticorrelaciona com o recuperador em 83% das
   consultas (Spearman entre posição na fusão e escore: **+0,179**). Corrigido, vai a
   **−0,466** e 0% — e então não acrescenta nada, porque concorda.
2. **`head()` num parquet agrupado por documento mede uma fração dos documentos que
   parece medir.** 500 linhas eram **35 documentos**; o intervalo de 95% do acerto@1
   era **±0,159**, largo o bastante para conter tanto o resultado bonito quanto o
   honesto. Uma conclusão publicável ("o modelo não lê a consulta") saiu de **16**
   documentos e se inverteu com 457.
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

O diagnóstico da falha 1 fez uma predição, e nós a testamos (§10): um cross-encoder
de base diferente do recuperador deve acrescentar algo. **Acrescenta — mas só se a
base for de domínio.** Um modelo de recuperação forte do mesmo tamanho empata
(p = 0,637); o encoder de Física vence (p = 0,0062). E o encoder de Física é, ele
mesmo, um recuperador ruim neste benchmark, o que torna a assimetria o achado mais
interessante do conjunto.

---

## 1. Introdução

Modelos de recuperação de domínio precisam de pares (consulta, documento relevante).
Anotação humana é caro; o grafo de citação é grátis. A prática é conhecida — SPECTER
(Cohan et al., 2020) treina representações de documento científico com sinal de
citação — e a nossa motivação era a mesma: construir um recuperador de Física sem
anotar nada.

O ajuste fino funcionou. Um MiniLM-L6 de 23 M de parâmetros treinado em 400 mil
arestas de citação bate o PhysBERT (110 M, pré-treinado em Física) por **+0,190 de
nDCG@10** (pareado, p = 0,0000) e **empata estatisticamente com o GTE-large**, um
modelo geral de 335 M:

| modelo | r@1 | r@10 | nDCG@10 |
|---|---|---|---|
| nosso, 23 M | 0,262 | 0,708 | **0,4657** |
| GTE-large, 335 M (geral) | 0,278 | 0,677 | 0,4628 |
| PhysBERT, 110 M (domínio) | 0,146 | 0,425 | 0,2752 |
| SciBERT, 110 M (base) | 0,109 | 0,328 | 0,2074 |

Esse é o resultado positivo, e ele não é a contribuição deste artigo. A contribuição
é o que aconteceu **entre** a primeira versão desse número e essa: quatro medições que
eu reportei e depois refutei, e uma quinta que refutou uma recomendação de gasto.

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
> recuperador é redundante por construção** — o que é uma predição testável, e o
> teste está em andamento.

### 3.4 O que isso acrescenta ao que já se sabia

RocketQA (Qu et al., 2021) documenta falsos negativos na mineração de negativos
difíceis e propõe filtragem por um modelo cross-encoder. O nosso achado é vizinho e
distinto: o problema aqui não é o negativo estar errado, é o **critério de seleção do
negativo ser correlacionado com o escore do modelo que vai ser reordenado**. O
sintoma é uma inversão mensurável de correlação, não uma queda de precisão — e a
inversão é diagnosticável com um Spearman que custa nada.

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

---

## 6. Falha 4 — co-citação como falso negativo

Dois papers citados juntos por um terceiro são um sinal clássico de relevância. Na
nossa mineração, **9,1% dos negativos** (62.646 de 688.136) eram co-citados com o
positivo. Treinar o reranqueador a rebaixá-los é ensiná-lo a rebaixar o que é
relevante, e a perda de treino desce normalmente enquanto isso acontece — porque, do
ponto de vista da perda, o rótulo é o rótulo.

**Ressalva que precisa ser dita:** co-citação aproxima relevância, não a define. Um
paper relacionado que ninguém citou junto com o positivo continua passando como
negativo, e **esse residual não é medido**. O filtro remove uma classe de falso
negativo que sabemos nomear; não sabemos o tamanho do resto.

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

---

## 10. A predição da §3.3, testada: é domínio, não diversidade

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

O encoder de domínio é um **recuperador ruim** neste benchmark (nDCG 0,2752 contra
0,4657 do nosso ajuste fino — a margem de +0,190 do §1) e a **melhor base de
reranqueador** das três testadas. Pré-treino de domínio não produziu um bi-encoder
competitivo aqui e produziu um cross-encoder competitivo.

Não temos explicação mecanística para a assimetria, e não vamos inventar uma. A
hipótese barata de testar é que o cross-encoder pode usar interação termo a termo
entre consulta e documento, onde vocabulário de domínio rende, enquanto o bi-encoder
precisa comprimir o documento num vetor antes de ver a consulta. É especulação até
alguém medir.

---

## 11. ⚠️ Limitações, sem atenuação

1. **Um domínio, um encoder base, um grafo.** Física, MiniLM-L6, OpenAlex. Nada aqui
   estabelece que as taxas transferem. A Falha 1 é a única com mecanismo geral
   argumentável, e mesmo ela foi observada uma vez.
2. **Benchmark próprio, sem juízo humano de relevância.** "Relevante" = "citado". Isso
   é uma proxy conhecidamente enviesada — favorece papers citáveis, campos com cultura
   de citação densa, e não captura relevância que ninguém citou. O nDCG de 0,1584 da
   fusão não é comparável a nDCG de benchmarks com anotação.
3. **O resultado positivo do §1 é um empate, não uma vitória.** 0,4657 contra 0,4628
   com 1/14,8 dos parâmetros é um resultado de eficiência. Ler como superioridade
   seria exatamente o tipo de coisa que este artigo documenta.
4. **A §6 não mede o falso negativo residual**, e a §7 não mede a taxa de
   contaminação do corpus de fonte LaTeX (o julgamento humano da amostra está
   pendente, com alvo pré-comprometido de 200 documentos).
5. **`lr` não foi re-ajustado** entre escalas de modelo no experimento da §10, o que
   é uma variável não controlada declarada antes de ver o resultado.
6. **Veículo realista: workshop.** Um artigo de armadilhas com validade externa de um
   domínio não é contribuição de conferência principal, e apresentá-lo como tal seria
   o mesmo erro de calibração que ele denuncia.

---

## 12. ⚠️ Citações — TODAS precisam ser verificadas antes de qualquer submissão

Escrevi as referências abaixo de memória. **Nenhuma foi conferida contra a fonte.**
Volume, página, ano e a própria existência do que é afirmado precisam ser verificados
um por um; uma citação errada num artigo sobre rigor de medição é a pior forma de se
desmentir.

**Razoavelmente confiante no conteúdo, a conferir na forma:**

- Cormack, Clarke & Büttcher (2009) — Reciprocal Rank Fusion; a origem do k = 60.
- Karpukhin et al. (2020) — DPR; negativos difíceis do BM25.
- Xiong et al. (2021) — ANCE; negativos reamostrados do próprio índice.
- Qu et al. (2021) — RocketQA; falsos negativos na mineração e filtragem por
  cross-encoder. **É o vizinho mais próximo da §3** e merece leitura integral antes de
  a §3.4 afirmar distinção.
- Cohan et al. (2020) — SPECTER; representação de documento científico com sinal de
  citação.
- Sennrich, Haddow & Birch (2016) — BPE. Kudo (2018) — Unigram.
- Bostrom & Durrett (2020) — Unigram contra BPE em linguagem natural.
- Robertson & Zaragoza (2009) — BM25.
- Wilson (1927); Brown, Cai & DasGupta (2001) — intervalos de proporção.
- McNemar (1947).

**Menos confiante, verificar antes de citar:**

- PhysBERT (2024) — o encoder de Física usado como alvo do §1. Tenho o identificador
  do modelo (`thellert/physbert_cased`, 109 M, `BertModel`) mas **não** a referência
  bibliográfica com segurança.
- peS2o — o corpus de texto pleno da §7. Conheço o dataset; não tenho autor e ano com
  segurança.
- RedPajama-1T — a fatia arXiv da §7 (`togethercomputer/RedPajama-Data-1T`, revisão
  `398f9257`). Mesmo caso.
- Tao et al. (2024) — vocabulário ótimo cresce com o tamanho do modelo. Citado no
  documento de projeto; não verificado.
- Ali et al. (2024) — eficiência de tokenizer correlaciona com desempenho a jusante.
  Mesmo caso.

**Falta procurar:** trabalho anterior sobre (a) n efetivo em avaliação de recuperação
com dados agrupados, e (b) vazamento por divisão posicional em conjuntos derivados de
grafo. As duas são elementares o bastante para que exista literatura, e se existir a
§4 e a §5 deixam de ser contribuição e passam a ser confirmação — o que é uma
informação útil e não um problema.

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

O que **não** é redistribuível: o corpus. Os resumos do arXiv seguem a licença de
cada submissão, e a licença padrão do arXiv concede ao arXiv o direito de distribuir,
não a terceiros. Os scripts de coleta e os manifestos permitem reconstruir; os bytes
não acompanham.
