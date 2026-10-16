# Recuperação de artigos de Física do arXiv sob restrição severa de computação: da supervisão por citação a um assistente com fontes

Vinicius Sanchez

*Pesquisa independente*

*Versão 0.8 — 16 de outubro de 2026*

---

## Resumo

> Sob orçamento de computação nulo — uma GPU de consumo de 8 GB e cotas gratuitas de aceleradores em nuvem —, este trabalho constrói um sistema de recuperação de artigos de Física do arXiv supervisionado por citação e, sobre ele, um assistente que responde perguntas citando os artigos recuperados. Corrigido um defeito de amostragem que impunha teto de 0,7562 ao nDCG@10 da avaliação, um bi-encoder de 23 milhões de parâmetros ajustado com 6 milhões de pares de citação sorteados atinge 0,6223, contra 0,5788 de um modelo geral de 335 M; sortear os pares, em vez de tomá-los na ordem do arquivo, vale +0,020 sem custo, e uma base geral de 109 M alcança o mesmo nível com um sexto dos pares, a mais de quatro vezes o custo de embutir. No assistente, com regras registradas antes de cada medição, um modelo aberto de 8 bilhões de parâmetros acerta 0,976 das perguntas com o artigo certo entre as fontes e 0,700 com o que a busca traz: o limite é a busca. Dois componentes que a tarefa substituta, de recuperar o documento citado, havia retirado — um reordenador de base pré-treinada em Física e a fusão com o BM25 — voltam e levam o acerto a 0,870. Quando a resposta está no corpo do artigo, e não no resumo, o sistema acerta 0,033; com o trecho certo, 0,940. Nenhum humano julgou as respostas, e cada modelo foi treinado uma vez.

**Palavras-chave:** recuperação de informação científica; supervisão por citação; recuperação densa; reordenação; geração aumentada por recuperação; avaliação de ponta a ponta; reprodutibilidade.

---

## 1. Introdução

Modelos de linguagem de propósito geral apresentam, em Física, modos de falha próprios — incoerência dimensional, deriva algébrica em derivações longas, mistura silenciosa de convenções —, e uma resposta frequente é especializá-los com texto do domínio. A restrição é de recurso. Sob a relação de computação ótima de Hoffmann et al. [1], um modelo de 8 bilhões de parâmetros requer cerca de 160 bilhões de tokens; modelando estágio a estágio o funil de aquisição da literatura de Física legalmente adquirível a custo zero, obtêm-se entre 15 e 30 bilhões após deduplicação e triagem de licença — uma escassez de cinco a dez vezes.

Este trabalho investe onde o dado do domínio é excedente: a recuperação de artigos. O grafo de citação fornece milhões de pares de supervisão gratuitos, um recuperador de dezenas de milhões de parâmetros cabe numa GPU de consumo, e um sistema que encontra a literatura relevante é útil por si e é o componente de que um sistema de geração precisa para citar fontes. Sobre o recuperador foi construído esse sistema: um assistente que responde perguntas de Física com um modelo aberto, citando os artigos que a busca encontra. Ele permite medir cada componente da busca na tarefa final, e não só na tarefa substituta em que o recuperador foi treinado e escolhido.

Uma segunda restrição decorre da primeira. Com orçamento nulo, cada experimento precisa ser justificado por uma medição anterior, e medições anteriores passam a ser reexaminadas com frequência. Parte substancial dos achados deste trabalho, inclusive os negativos, originou-se desse reexame — a começar pelo protocolo de avaliação, cujo defeito invertia o resultado principal, e terminando na tarefa final, que reverteu duas decisões tomadas na substituta.

As contribuições são as seguintes.

1. Um procedimento de construção de corpus de Física a custo zero, atestado por uma cadeia de hashes sobre 52,40 GB, com a caracterização quantitativa do funil de filtragem e da integridade da notação matemática de cada fonte (Seções 3.1 a 3.3 e 4.1 a 4.3).
2. A remedição de um protocolo de recuperação densa que tinha teto, e a separação experimental entre amostragem, volume de pares e base do recuperador, com seis bases gerais na mesma régua (Seções 4.4 e 4.5).
3. Uma comparação de bases de reordenação com controle de tamanho, arquitetura, dados, semente e hiperparâmetros, em que só a base pré-treinada em Física melhora a fusão — e a constatação de que o ganho do estágio deixa de ser estabelecido quando o recuperador melhora, na tarefa substituta (Seção 4.6).
4. Um conjunto de avaliação de recuperação por equação construído do próprio corpus, com estratos de variação de grafia (Seção 4.8).
5. Uma avaliação de ponta a ponta do assistente, com regras registradas antes, que separa o gerador da busca e reverte duas decisões da tarefa substituta: o reordenador e a fusão com o BM25 voltam à cadeia e levam o acerto das respostas de 0,700 a 0,870 (Seções 3.8, 4.9 e 5.2).
6. Uma auditoria de oito ocorrências de amostragem dependente da ordem dos dados, num repositório em que rótulo, negativo e verdade de avaliação saem do mesmo grafo (Seção 5.3).

O trabalho não alega um modelo de Física em estado da arte. A Seção 4.4 reporta um resultado, retira-o por defeito de protocolo identificado depois das medições e reporta a remedição. A avaliação do assistente usa perguntas escritas e respostas julgadas por modelos de linguagem, sem julgamento humano das respostas (Seção 6). O pré-treinamento de encoders no domínio — tokenização de LaTeX, mascaramento de equações e pré-treinamento continuado — é objeto de um trabalho complementar [2], cujo resultado para o recuperador está resumido na Seção 4.7.

---

## 2. Trabalhos relacionados

### 2.1 Encoders científicos e gerais

O SciBERT [3] é o baseline convencional para texto científico, mas seu corpus é composto de 82% de literatura biomédica e 18% de ciência da computação, sem fatia declarada de Física; superá-lo em Física não constitui evidência de especialização. O SPECTER [4] introduz o grafo de citação como sinal de relacionamento entre documentos científicos, abordagem que este trabalho adota; o SciNCL [5] e o SPECTER2 [6] a refinam, com amostragem de vizinhos no grafo e treinamento multitarefa. Nenhum dos três foi medido aqui (Seção 6). O INDUS [7] cobre as cinco áreas científicas da NASA — ciências da Terra, astrofísica, ciências planetárias, heliofísica e ciências biológicas e físicas. O PhysBERT [8] é o competidor de mesmo domínio: um modelo de embedding pré-treinado sobre 1,2 milhão de artigos de Física do arXiv.

Modelos gerais de embedding treinados por aprendizado contrastivo em larga escala — GTE [9], BGE [10], E5 [11] — formam uma segunda barra de comparação, frequentemente omitida em trabalhos de domínio. A arquitetura de bi-encoder segue Reimers e Gurevych [12], e a base do sistema é o all-MiniLM-L6-v2 [13], um MiniLM [14] obtido por destilação de auto-atenção e já ajustado contrastivamente em larga escala, inclusive com pares de citação do S2ORC [15].

### 2.2 Recuperação densa e mineração de negativos

O DPR [16] estabelece o arcabouço de bi-encoder para recuperação densa, com negativos difíceis provenientes do BM25. O ANCE [17] reamostra negativos do próprio índice durante o treinamento, e o RocketQA [18] documenta falsos negativos nessa mineração e propõe filtrá-los com um cross-encoder. A Seção 4.6 reporta um modo de falha vizinho e distinto: o rótulo do negativo não está errado, mas o critério de seleção do negativo se correlaciona com o escore do modelo que será reordenado. O sintoma não é queda de precisão, e sim inversão de correlação, sem manifestação na função de perda.

### 2.3 Corpora

O peS2o [19] é um corpus de artigos científicos de acesso aberto derivado do S2ORC [15], com texto integral extraído de PDF. O RedPajama [20] inclui uma fatia do arXiv construída a partir do fonte LaTeX. O OpenWebMath [21] extrai texto matemático da web preservando a notação. A Seção 4.3 compara as três quanto à integridade da notação.

### 2.4 Geração aumentada por recuperação e sua avaliação

A geração aumentada por recuperação [22] condiciona a resposta de um modelo de linguagem a documentos recuperados. A consulta hipotética do HyDE [23] — o modelo escreve um documento que responderia à pergunta, e a busca usa esse texto — aproxima uma pergunta curta da forma dos documentos em que o recuperador foi treinado. O ALCE [24] avalia a geração com citações em correção e em qualidade das citações. Respostas abertas exigem julgamento, e modelos de linguagem usados como juízes [25] concordam com avaliadores humanos em parte das tarefas, com vieses conhecidos; por isso a concordância do juiz, medida pelo κ de Cohen [26], é aqui parte do protocolo, com limiar registrado antes.

---

## 3. Materiais e métodos

### 3.1 Índice de metadados

Os metadados do arXiv foram coletados pelo protocolo OAI-PMH restrito ao conjunto `physics`, com cursor durável por lote e retomada idempotente. As obras do OpenAlex [27] foram lidas do instantâneo público por faixas de bytes HTTP, sobre 13 das 189 colunas, sem materialização em disco. O casamento entre as fontes usa o campo de localizações do OpenAlex. Três decisões de esquema, determinadas empiricamente, afetam a cobertura: extrair o identificador arXiv do DOI recupera 1,5% dos registros, contra 98,5% pelo campo de localizações (medido em 200 obras); restringir à localização primária excluiria 1,44 milhão de registros cuja localização primária não é o arXiv; e uma expressão regular de identificadores anteriores a 2007 truncava 41,5% do acervo.

### 3.2 Classificação de domínio

O rótulo de Física é a presença de qualquer categoria da família de Física no registro, e não o prefixo da categoria primária: artigos de outras áreas com listagem cruzada em Física pertencem aos dois conjuntos. Dos 1.041.652 registros coletados em ciência da computação, economia e biologia quantitativa, 72.919 têm listagem cruzada em Física, e todos constam do índice; a conferência antecede a coleta dos negativos de matemática e não foi repetida sobre eles. O classificador é linear sobre representação esparsa, com perda `modified_huber`, treinado sobre 300.000 documentos positivos e 190.210 negativos de quatro domínios do arXiv — ciência da computação, economia, matemática e biologia quantitativa. A avaliação principal é por exclusão de domínio: treina-se omitindo um domínio negativo inteiro e mede-se a taxa de falso positivo nele.

### 3.3 Fontes de texto integral

A fatia arXiv do RedPajama [20] foi selecionada por pertinência exata: retêm-se os documentos cujo identificador consta do índice, sem classificador. O OpenWebMath [21] e o peS2o [19], que não trazem identificador arXiv, foram filtrados pelo classificador com limiar de 0,9. Todas as fontes são fixadas por revisão explícita do repositório de origem. A integridade da notação de cada fatia foi medida em 3.000 documentos sorteados por fatia, com quatro indicadores: fração de documentos com qualquer marcação LaTeX, com matemática em linha e com ambiente de equação em display, e número de sequências de controle LaTeX por documento.

### 3.4 Modelos e treinamento

**Bi-encoder.** A base do sistema é o all-MiniLM-L6-v2 [13], de 23 milhões de parâmetros: um MiniLM [14] de 6 camadas já ajustado pelos autores do Sentence-Transformers [12] com cerca de 1,17 bilhão de pares, dos quais cerca de 169 milhões de citação do S2ORC [15] (Seção 6); no restante do texto, MiniLM-L6. O ajuste usa perda contrastiva InfoNCE [28] sobre pares do grafo de citação — o citante como âncora, o citado como positivo —, com 127 negativos no lote, 192 tokens e agregação por média. Do conjunto de 6.564.111 pares e 667.304 documentos citados sorteiam-se, com semente registrada, subconjuntos de 200 mil a 6 milhões de pares, os mesmos para todas as bases que os usam. Uma variante usa 511 negativos por caching de gradiente [29].

**Bases gerais.** Com a mesma receita, foram ajustados o GTE-base [9] (109 M) com 200 mil, 400 mil e 1 milhão de pares; quatro bases de 33 M — gte-small [9], bge-small [10], e5-small [11] e all-MiniLM-L12-v2 — com 200 mil, e as duas melhores com 1 milhão; e o ModernBERT-base [30] (150 M), modelo de linguagem mascarada sem ajuste contrastivo prévio, com 200 mil. Foram medidos sem ajuste o GTE-large [9] (335 M), o PhysBERT [8] e o SciBERT [3], este também ajustado com 400 mil pares.

**Recuperação léxica e fusão.** BM25 [31] sobre índice esparso dos mesmos documentos; fusão recíproca de postos [32] com k = 60.

**Cross-encoder de reordenação.** Modelo par a par treinado sobre grupos de 8 candidatos amostrados da distribuição exata da avaliação, isto é, dos 50 primeiros candidatos da fusão. Três bases foram comparadas com os demais hiperparâmetros, os dados, a semente e o protocolo idênticos: MiniLM-L6 (23 M), GTE-base (109 M) e PhysBERT (109 M).

### 3.5 Protocolo de avaliação

A tarefa é recuperar o documento citado a partir do texto do documento citante — título e resumo, nos dois lados. As métricas são recall@k, MRR e nDCG@10.

Dois protocolos são usados. O de comparação de encoders avalia dentro de um conjunto de 2.000 pares de validação, sorteados com semente e deduplicados por texto, e interrompe a execução se houver âncora ou alvo de texto repetido — a condição que derruba abaixo de 1,0 o nDCG@10 de um recuperador perfeito (Seção 4.4); o teto é gravado no artefato ao lado da métrica. O de sistema usa 2.000 consultas contra o universo de 88.807 documentos citados distintos.

Três propriedades do protocolo condicionam a leitura e são retomadas na Seção 6: todos os modelos são lidos a 192 tokens, com agregação por média e sem prefixo de instrução; a divisão entre treinamento e validação é por documento citante; e o ponto de verificação de cada modelo ajustado é o de maior nDCG@10 numa avaliação interna sobre 1.000 pares que são os primeiros do mesmo sorteio dos 2.000 do protocolo, de modo que não há conjunto de teste separado. Braços de uma comparação são medidos na mesma sessão e com o mesmo código.

### 3.6 Análise estatística

Comparações sobre o evento "o alvo alcançou as k primeiras posições" usam o teste de McNemar exato [33] sobre pares discordantes, com os mesmos itens para todos os sistemas. Diferenças de nDCG@10 usam bootstrap pareado por consulta; intervalos para proporções, o método de Wilson [34]. A correção de Bonferroni é aplicada onde foi registrada antes da coleta, junto com a regra de leitura de cada desfecho possível: na comparação de bases de reordenação e nas de bases de 33 M. Nas demais, p-valores e intervalos são nominais.

"Empate" designa a não rejeição da hipótese nula — intervalo de 95% que cobre zero, ou p > 0,05 —, e não equivalência demonstrada. Os testes condicionam nos modelos treinados e medem a variância de amostragem dos itens, não a de treinamento; cada modelo foi treinado uma única vez. Onde a tabela reporta o McNemar em recall@1 ao lado de uma diferença de nDCG@10, o intervalo do nDCG@10 não foi calculado.

### 3.7 Reprodutibilidade

Cada etapa do pipeline grava um manifesto com os hashes BLAKE3 das saídas, os manifestos de entrada, os parâmetros e o commit, e uma cadeia de hashes em três níveis cobre, a partir de uma raiz, 52,40 GB em 35 etapas e 1.295 arquivos. A primeira versão da cadeia omitia o peS2o inteiro, sem que nenhuma verificação acusasse; hoje uma verificação exige que toda fonte declarada no filtro tenha etapa. A cadeia atesta os bytes, mas não torna a coleta refazível: duas coletas OAI-PMH do mesmo conjunto, com onze horas de intervalo, diferiram em 357 registros.

### 3.8 O assistente e sua avaliação

**O sistema.** O assistente responde perguntas de Física, em português ou em inglês. Um modelo aberto, o Qwen3-8B [35], quantizado a 4 bits e servido localmente na mesma GPU de 8 GB, escreve primeiro o título e o resumo de um artigo hipotético que responderia à pergunta [23]; o recuperador de 6 milhões de pares da Seção 4.5 busca esse texto entre os 1.594.338 artigos do índice; o título e o resumo das seis primeiras fontes entram no prompt, numerados; e o modelo responde citando-as como [n]. Um portão remove citações a fontes que não foram fornecidas. Se a fonte citada sustenta a frase não é verificado.

**Itens.** Cada item parte de um artigo P sorteado das âncoras de validação, fora do treinamento do recuperador e dentro do índice. Um segundo modelo de linguagem, o Claude (Anthropic), lê só o resumo de P e escreve, em português, uma pergunta que um físico faria e cuja resposta é um fato do resumo, com um gabarito curto. O teste tem 500 itens primários e 150 de artigos publicados depois do lançamento do Qwen3, que o modelo não pode ter lido; outros 200, disjuntos, servem de desenvolvimento. Antes de qualquer resposta existir, o autor revisou 40 perguntas sorteadas, com o critério, registrado, de refazer o gerador de perguntas se menos de 32 fossem válidas: as 40 foram. Dois estratos à parte, de 150 e de 100 perguntas, foram escritos a partir de uma passagem do corpo do artigo que o resumo não contém.

**Braços e juiz.** Três braços com o mesmo modelo e o mesmo prompt: A, o sistema; B, as mesmas fontes com P garantido entre elas, no lugar da sexta; C, sem fontes. Um juiz automático — o próprio Qwen3-8B, com o gabarito, sem saber o braço e sem as marcas de citação — classifica cada resposta como certa, parcial, errada ou abstenção, e só certa conta. O juiz é validado em 100 respostas julgadas às cegas, pelo κ de Cohen [26] entre certa e o resto, com mínimo de 0,6. Por decisão do autor, registrada antes de qualquer veredicto do juiz ser lido, esse julgamento e a revisão dos erros do braço B foram feitos pelo Claude, às cegas e com os veredictos selados por hash antes de o juiz rodar; nenhum humano julgou respostas. Uma das decisões foi conferida por um segundo juiz, de outra família e três vezes maior, o Mistral Small 3.2 [36], validado contra os mesmos veredictos.

**Regras.** Todas registradas antes da medição a que se aplicam, com intervalo de 95% por bootstrap pareado por item. O gerador basta se o intervalo do acerto de B ficar inteiro acima de 0,90; a busca limita se o de B − A ficar inteiro acima de 0,05; e as perguntas só medem o que dizem se C ficar abaixo de B − 0,10. Um componente novo da busca é adotado se o intervalo do ganho de acerto sobre o sistema vigente ficar inteiro acima de 0,03 e o tempo mediano por pergunta não passar de 60 s. Os candidatos são escolhidos nas perguntas de desenvolvimento — os de ordem e de índice só iam ao teste com ganho de 5 pontos na fração de perguntas com P entre as seis fontes —, e o teste decide uma vez por candidato.

---

## 4. Resultados

### 4.1 Corpus

**Tabela 1.** Índice de metadados construído, contra a estimativa de planejamento.

| Grandeza | Medido | Estimado no planejamento |
|---|---|---|
| Registros do arXiv (conjunto `physics`) | 1.595.422 | 1.200.000 |
| Obras do OpenAlex com localização no arXiv (de 510.372.821 varridas) | 4.613.751 | — |
| Taxa de casamento com o índice | 99,1% (1.581.098) | 98,5% |
| Arestas de citação | acima de 22,7 milhões | dezenas de milhões |
| Com referência de periódico declarada (campo `journal-ref`) | 740.823 (46,4%) | — |
| Fração redistribuível | 14,8% (235.795) | 25–35% |

Na Tabela 1, a fração redistribuível é a estimativa que mais errou, na direção desfavorável. Ela muda em degrau: no índice, passa de cerca de 5% para mais de 20% entre 9 e 10 de novembro de 2020, o que é compatível com uma mudança no formulário de submissão do arXiv, não verificada. É um limite inferior: corrigido o reconhecimento da dedicação ao domínio público, a fração passa de 14,77% a 14,88% numa coleta de conferência. O campo `journal-ref` é declarado pelo autor e não atesta revisão por pares.

**Tabela 2.** Fatias de texto após filtragem para Física. Tokens estimados como caracteres divididos por 4; nenhuma fatia foi tokenizada por inteiro.

| Fonte | Documentos aceitos | Tokens estimados |
|---|---|---|
| RedPajama, fatia arXiv [20] | 835.379 | 10,54 × 10⁹ |
| OpenWebMath [21] | 860.521 | 2,62 × 10⁹ |
| peS2o [19] | 5.526.331 de 38.972.211 (14,18%) | 14,60 × 10⁹ |
| Total | 7.222.231 | 27,75 × 10⁹ |

O total da Tabela 2 situa-se na metade superior da faixa de 15 a 30 bilhões de tokens estimada como adquirível, com ressalvas: a faixa supunha deduplicação e triagem de licença, e o total não tem deduplicação entre fontes — o peS2o e a fatia RedPajama podem conter os mesmos artigos — nem exclusão das licenças não comerciais. Dos documentos aceitos do peS2o, 65,6% são só resumos (7,9% dos tokens), e o texto integral dessa fatia não contém equações em display (Seção 4.3). Na fatia RedPajama, os documentos distintos são 828.601: 6.778 identificadores aparecem duas vezes.

### 4.2 Classificação de domínio

O classificador final atinge acurácia de 0,954, com taxa de falso positivo agregada de 4,6% sobre os quatro domínios negativos do treinamento; a taxa por domínio do modelo final não foi medida.

**Tabela 3.** Avaliação por exclusão de domínio, com modelos de 120.000 documentos por classe. "FP interno" é a taxa de falso positivo nos domínios presentes no treinamento; "FP externo", no domínio omitido.

| Domínio omitido | FP interno | FP externo | Razão | Precisão externa |
|---|---|---|---|---|
| Ciência da computação | 3,7% | 9,8% | 2,6 | 0,907 |
| Economia | 3,3% | 8,5% | 2,6 | 0,988 |
| Matemática | 2,4% | 35,4% | 14,6 | 0,731 |
| Biologia quantitativa | 3,1% | 31,2% | 10,2 | 0,907 |
| Estatística | 3,0% | 2,9% | 1,0 | — |

A Tabela 3 mostra que a degradação sob domínio não visto é função da proximidade. Sem negativos de matemática, o classificador aceita como Física 35,4% dos resumos de matemática no limiar de 0,5 e 13,6% no de 0,9 usado na filtragem; elevar o limiar a 0,999 reduz a taxa a 10,0% e estanca, porque a perda `modified_huber` satura as probabilidades. O OpenWebMath é majoritariamente matemático, mas é texto de web, e a taxa nele não foi medida.

### 4.3 Integridade da notação matemática

**Tabela 4.** Indicadores de notação matemática por fatia, 3.000 documentos sorteados por fatia.

| Fatia | Caracteres/doc. | LaTeX (%) | Matemática em linha (%) | Ambiente de equação (%) | Sequências de controle/doc. |
|---|---|---|---|---|---|
| Resumos do arXiv (referência) | 1.123 | 21,8 | 26,9 | 0,0 | 0,9 |
| RedPajama, arXiv (fonte LaTeX) | 49.212 | 100,0 | 99,6 | 84,9 | 1.158,3 |
| OpenWebMath (web) | 16.009 | 78,5 | 85,2 | 5,1 | 57,7 |
| peS2o, texto integral (de PDF) | 28.605 | 16,3 | 18,2 | 0,0 | 1,0 |

A fatia extraída de PDF tem, na Tabela 4, 28,6 mil caracteres por documento e nenhum ambiente de equação em display. A inspeção de trechos ilustra dois mecanismos: a equação em display é removida — num documento, restou o dois-pontos sem referente entre "if:" e "where" —, e a matemática em linha é achatada em texto — `p^*` aparece como `p *`, e `T_{eff}` como `T eff`. A fração de equações removidas não foi medida.

O indicador intuitivo de remoção, o operador órfão — um sinal de igualdade sem operando de um dos lados —, não discrimina: acusa 81,3% dos documentos da fatia de fonte LaTeX, 50,7% da extraída de PDF e 3,0% da referência, mas cresce com o comprimento do documento, que difere entre as fatias por um fator de até 44, e dispara no fonte LaTeX porque, em `$Z_{\rm max}$ = 15 kpc`, o delimitador que antecede o sinal não é aceito como operando. O indicador que separa as duas fatias de texto integral é o ambiente de equação, 84,9% contra 0,0% dos documentos — específico de formato, e não validado fora do fonte LaTeX.

Marcação preservada não implica completude. Em 298 artigos de Física, depois de expandidas as macros definidas pelo autor, a fatia de fonte LaTeX tem 16,6% menos expressões matemáticas distintas que o fonte original, com intervalo de reamostragem de artigos de [12,9%; 20,8%] — um déficit de contagem, e não uma contagem de equações ausentes, medido em blocos contíguos de identificadores e contra a versão corrente do arXiv, posterior ao instantâneo de 2023 da fatia.

### 4.4 Defeito de protocolo e remedição

A primeira avaliação colocava o bi-encoder ajustado com 400 mil pares em paridade com o GTE-large — nDCG@10 de 0,4657 contra 0,4628; recall@1 com p = 0,114 —, e uma auditoria posterior invalidou o protocolo.

A avaliação é feita dentro do conjunto: a resposta correta da âncora *i* é o positivo *i*, e a tarefa só é bem posta se os textos forem distintos. A amostra era tomada pelas primeiras 2.000 linhas do arquivo de validação, cuja ordem não é neutra — no comprimento de âncora, o primeiro bloco situa-se no percentil 94 — e que continham apenas 1.147 positivos distintos, com 62% das linhas ocupadas por um positivo repetido, um deles 28 vezes. Textos idênticos recebem similaridade idêntica, o desempate é arbitrário, e âncoras repetidas partilham uma única ordenação. A Tabela 5 mede o teto que isso impunha.

**Tabela 5.** Desempenho de um recuperador perfeito sob cada esquema de amostragem do conjunto de avaliação.

| Esquema de amostragem | recall@1 | nDCG@10 |
|---|---|---|
| Primeiras 2.000 linhas (protocolo original) | 0,5235 | 0,7562 |
| Amostra aleatória de 2.000 | 0,9364 | 0,9761 |
| Amostra aleatória com deduplicação por texto | 1,0000 | 1,0000 |

Contra um teto de 0,7562, a paridade decidida em 0,0029 de nDCG@10 estava abaixo do ruído de desempate. O mesmo esquema afetava o treinamento: os primeiros 400 mil pares do arquivo cobrem 17.844 documentos citados distintos, e uma amostra aleatória de mesmo tamanho, 191.300 — 10,7 vezes mais diversidade pelo mesmo custo. A Tabela 6 refaz a comparação no protocolo corrigido.

**Tabela 6.** Remedição no protocolo corrigido: 2.000 pares sorteados e deduplicados, teto de 1,0000 medido e gravado.

| Modelo | Parâmetros | nDCG@10 | recall@1 | recall@10 | MRR |
|---|---|---|---|---|---|
| Bi-encoder ajustado, 6 M pares sorteados | 23 M | **0,6223** | **0,4315** | **0,8305** | **0,5636** |
| GTE-large [9] (geral) | 335 M | 0,5788 | 0,4140 | 0,7640 | 0,5293 |
| Bi-encoder ajustado, 400 mil pares (primeiras linhas) | 23 M | 0,5265 | 0,3560 | 0,7185 | 0,4773 |
| all-MiniLM-L6-v2 (sem ajuste neste trabalho) | 23 M | 0,4761 | 0,3135 | 0,6640 | 0,4279 |
| SciBERT ajustado, 400 mil pares (primeiras linhas) | 110 M | 0,4746 | 0,3070 | 0,6705 | 0,4263 |
| PhysBERT [8] (domínio) | 109 M | 0,3507 | 0,2220 | 0,4910 | 0,3170 |
| SciBERT [3] (sem ajuste) | 110 M | 0,2537 | 0,1490 | 0,3825 | 0,2275 |

No protocolo corrigido, o veredito contra o modelo geral inverteu-se duas vezes: o modelo de 400 mil pares ficava 0,052 abaixo do GTE-large, e não 0,003 acima, e o treinamento com pares sorteados e volume crescente (Seção 4.5) levou a margem a +0,044. Em recall@10, a diferença é significativa para qualquer tabela de discordância compatível com as marginais (p < 10⁻⁵); para nDCG@10 e MRR, o intervalo não foi calculado; em recall@1, o teste pareado não a estabelece (188 contra 153 discordantes, p = 0,065). A comparação vale para entradas truncadas a 192 tokens e para um acervo que o modelo ajustado viu no treinamento e o geral não (Seção 6). Sobre o modelo de mesmo domínio, a margem é de 0,272.

O ganho do ajuste por citação é inverso ao ponto de partida: com 400 mil pares, o SciBERT sobe 0,221 e termina abaixo até do MiniLM-L6 sem ajuste, que com os mesmos pares sobe 0,050. Isso sugere que o ajuste ensina sobretudo a tarefa, que uma base contrastiva geral já sabe — com a ressalva de que a base do MiniLM-L6 já viu pares de citação (Seção 6) e de que o SciBERT foi ajustado com lote menor.

### 4.5 Amostragem, volume e base do recuperador

**Tabela 7.** Curva de volume do bi-encoder de base MiniLM-L6, uma variável por execução, no protocolo da Tabela 6. A margem é em relação ao GTE-large (0,5788).

| Pares de treinamento | Documentos citados distintos | nDCG@10 | Margem |
|---|---|---|---|
| 400 mil, primeiras linhas | 17.844 | 0,5265 | −0,0522 |
| 400 mil, sorteados | 191.198 | 0,5462 | −0,0326 |
| 1,5 milhão, sorteados | 390.856 | 0,5780 | −0,0008 |
| 3 milhões, sorteados | 518.635 | 0,6026 | +0,0238 |
| 6 milhões, sorteados | 650.162 | 0,6223 | +0,0435 |

As duas primeiras linhas da Tabela 7 formam a ablação mais limpa do trabalho — mesma base, mesmo número de pares, mesmo lote, mesmos 3.125 passos, mesma semente —, e a única variável é a amostragem: +0,0196 de nDCG@10, com recall@1 de 104 contra 71 discordantes (p = 0,0153), sem custo de computação. Por duplicação de volume, os ganhos são de +0,017, +0,025 e +0,020; a execução de 6 milhões vence a de 3 milhões com p = 0,0012 e não mostra saturação, mas já cobre 650.162 dos 667.304 documentos citados. Nem 511 negativos no lote em vez de 127 (0,5272 contra 0,5246) nem ler 256 ou 384 tokens em vez de 192 (p entre 0,08 e 0,84) produziram ganho detectável.

**Tabela 8.** nDCG@10 por base e por volume de pares de ajuste, no protocolo da Tabela 6. Em cada coluna, todas as bases viram os mesmos pares. Valores de colunas diferentes vêm de sessões diferentes; as comparações do texto são pareadas entre modelos medidos na mesma sessão. "Custo" é o tempo de embutir 2.000 textos relativo ao MiniLM-L6, após aquecimento. O MiniLM-L6 a 1,5 e a 3 milhões está na Tabela 7.

| Base | Parâmetros | Custo | 200 mil | 400 mil | 1 milhão | 6 milhões |
|---|---|---|---|---|---|---|
| MiniLM-L6 [13] | 23 M | 1,00 | 0,5289 | 0,5462 | — | **0,6223** |
| all-MiniLM-L12-v2 | 33 M | 1,76 | 0,5404 | — | — | — |
| e5-small [11] | 33 M | 1,76 | 0,5402 | — | — | — |
| bge-small [10] | 33 M | 1,75 | 0,5692 | — | 0,6012 | — |
| gte-small [9] | 33 M | 1,76 | 0,5706 | — | 0,6014 | — |
| GTE-base [9] | 109 M | 4,2 | **0,5964** | **0,6094** | **0,6211** | — |
| ModernBERT-base [30] | 150 M | — | 0,5270 | — | — | — |

A Tabela 8 mostra que a base pesa tanto quanto o volume, e que o custo de servir decide entre os dois. Com 400 mil pares, trocar o MiniLM-L6 pelo GTE-base acrescenta +0,0633 de nDCG@10, com recall@1 de 219 contra 104 discordantes (p = 1,5 × 10⁻¹⁰), e ainda fica abaixo do recuperador de 6 milhões em recall@10 (0,8010 contra 0,8305; p ≤ 0,033). Com 1 milhão de pares, na mesma sessão que o recuperador do sistema e com regra registrada antes, o GTE-base não se distingue dele: −0,0012 [−0,011; +0,008], com recall@1 de 148 contra 144 discordantes (p = 0,861). Embutir o universo de 88.807 documentos custa, porém, 954 segundos com o GTE-base e 217 com o MiniLM-L6, razão que incide também em serviço, e a regra manteve o sistema. O ModernBERT-base, sem ajuste contrastivo prévio, fica no nível do MiniLM-L6, 0,069 abaixo do GTE-base [2].

Bases do porte do MiniLM-L6 poderiam trazer o ganho de base sem o custo. Com 200 mil pares e intervalo de 98,75% (Bonferroni sobre quatro comparações), o gte-small e o bge-small ficam acima do MiniLM-L6, por +0,042 [+0,028; +0,055] e +0,040 [+0,028; +0,054]; o e5-small e o all-MiniLM-L12-v2 não se separam dele. Levadas a 1 milhão de pares, as duas melhores ficam abaixo do recuperador do sistema, −0,021 [−0,032; −0,010] nas duas, com intervalo de 97,5%: em recall@1 não se separam dele (p = 0,51), e em recall@10 marcam 0,7915 e 0,7945 contra 0,8305. Nenhuma base geral medida supera o recuperador instalado — a de 109 M o alcança a mais de quatro vezes o custo, e as de 33 M ficam abaixo a 1,8 vez —, e a busca por outra base foi encerrada por custo, sem que nenhuma fosse levada aos 6 milhões de pares.

A primeira medição da troca de base quase registrou um falso negativo: com 256 candidatos, o valor padrão do avaliador, deu +0,039 com p = 0,161, contra +0,0633 com p = 1,5 × 10⁻¹⁰ nos 2.000 do protocolo. Um valor padrão de linha de comando não é um protocolo.

### 4.6 Fusão e reordenação

Com o recuperador de 400 mil pares, a fusão com o BM25 supera os dois recuperadores isolados (nDCG@10 de 0,1584 contra 0,1399 e 0,1327; McNemar em k = 10, p = 0,00073 e 0,00018, mil consultas), e um cross-encoder de base MiniLM-L6 não lhe acrescenta ganho mensurável (p = 0,118).

O trajeto até esse cross-encoder contém o modo de falha mais informativo do trabalho. A mineração inicial de negativos difíceis tomava os *K* primeiros resultados do índice denso, excluída a citação verdadeira; como todo negativo assim obtido está no topo e o positivo frequentemente não, o rótulo passa a ser predizível pela posição, e o modelo aprende a preferir a cauda, com nDCG@10 de 0,0179, abaixo da ordenação aleatória. Seis hipóteses alternativas foram descartadas, e amostrar os negativos da distribuição exata da avaliação inverte a correlação (Tabela 9).

**Tabela 9.** Correlação de Spearman entre a posição do candidato na fusão e o escore do cross-encoder, por estratégia de mineração de negativos. Posição menor é melhor colocação, de modo que o sinal desejado é negativo.

| Estratégia de mineração | Spearman | Consultas com correlação positiva |
|---|---|---|
| K primeiros do índice denso, menos o positivo | +0,179 | 83% |
| Amostragem dos 50 primeiros da fusão real | −0,466 | 0% |

O reordenador corrigido passa a concordar com o recuperador, o que se esperaria tanto de redundância com a base quanto de competência. A hipótese da redundância foi testada pela predição que faz — uma base pré-treinada em corpus distinto deveria acrescentar informação —, com a regra (McNemar em k = 10 contra a fusão, limiar de Bonferroni de 0,025, 2.000 consultas) e as quatro leituras possíveis registradas antes (Tabela 10).

**Tabela 10.** Cross-encoders idênticos exceto pela base, sobre a fusão com o recuperador de 400 mil pares. Duas mil consultas, profundidade 50, fusão de referência com nDCG@10 de 0,1576, teto do conjunto de candidatos de 0,4495. "Acerto@1" é medido em grupos de 8 candidatos de 481 documentos de validação, no último passo do treinamento (±0,045); nos pontos de verificação efetivamente avaliados, vale 0,520 para o GTE-base e 0,608 para o PhysBERT. "Diferença" é em relação à fusão.

| Base | Parâmetros | Acerto@1 | nDCG@10 com reordenação | Diferença | Discordantes | p (k = 10) |
|---|---|---|---|---|---|---|
| MiniLM-L6 (a do recuperador) | 23 M | 0,498 | 0,1483 | −0,0093 | 229 | 0,1458 |
| GTE-base (geral) [9] | 109 M | 0,510 | 0,1530 | −0,0046 | 220 | 0,6371 |
| PhysBERT (Física) [8] | 109 M | 0,566 | 0,1666 | +0,0090 | 237 | 0,0062 |

Na Tabela 10, só a base pré-treinada em Física supera a fusão ao limiar registrado (140 contra 97 discordantes), com o ganho na cauda do top-10: em k = 1 há empate (p = 0,930). GTE-base e PhysBERT compartilham tamanho, arquitetura, hiperparâmetros, dados, semente e protocolo, o que exclui tamanho e arquitetura; não separam o corpus de pré-treinamento do objetivo — contrastivo no GTE, de linguagem mascarada no PhysBERT — nem do vocabulário, e o contraste entre as duas não foi testado de forma pareada.

**O estágio com o recuperador melhorado.** Trocar o recuperador de 400 mil pares pelo de 6 milhões, na mesma sessão e com as mesmas 2.000 consultas, elevou o recall@100 do bi-encoder de 0,5160 para 0,6325 e o nDCG@10 da cadeia completa em apenas 0,0003; contra o valor histórico, medido com outra profundidade, o ganho reportado seria sete vezes maior. A fusão deixa de somar: empata com o bi-encoder isolado em k = 10 (p = 0,949) e perde dele em recall@100 (0,6065 contra 0,6325), e, com orçamento igual de candidatos, 200 do bi-encoder alcançam 0,7300, contra 0,6835 da união de 100 de cada recuperador. Uma regra registrada antes da execução seguinte retirou o BM25. Na tarefa do assistente, uma fusão com o BM25 consultado de outro modo foi medida e adotada (Seção 4.9).

O ganho da reordenação sobre a fusão encolheu na direção predita — em recall@10, de +0,0225 (p = 0,0081) para +0,0150 (p = 0,086), diferença entre os dois ganhos não testada —, e retreinar o reordenador com negativos minerados do recuperador novo não o recuperou (92 contra 82 discordantes contra o antigo, p = 0,495). Contra o bi-encoder isolado, nenhuma das duas cadeias vence em k = 10 (p = 0,139 e 0,379), nem em nenhuma profundidade de candidatos (Tabela 11).

**Tabela 11.** Reordenação sobre o bi-encoder de 6 milhões de pares, sem fusão, por profundidade. Duas mil consultas; @50 e @100 obtidos dos mesmos escores de @200. "Teto" é o recall do conjunto de candidatos.

| Sistema | recall@1 | recall@10 | Teto | nDCG@10 |
|---|---|---|---|---|
| Bi-encoder | 0,0655 | 0,2810 | — | 0,1585 |
| + reordenação @50 | 0,0670 | 0,2930 | 0,5210 | 0,1664 |
| + reordenação @100 | 0,0675 | 0,2945 | 0,6325 | 0,1676 |
| + reordenação @200 | 0,0670 | 0,2950 | 0,7300 | 0,1673 |

Na Tabela 11, dobrar os candidatos de 100 para 200 sobe o teto em 195 consultas e o recall@10 em uma: o reordenador leva 20 desses alvos aos dez primeiros, e os candidatos adicionais expulsam de lá outros 19. O bootstrap pareado sobre o nDCG@10 não estabelece ganho em nenhuma profundidade (@100: +0,0091 [−0,0010; +0,0191]); ele e a decomposição a seguir foram acrescentados depois de o McNemar sobre o top-10, registrado de antemão, se mostrar o instrumento errado para um estágio que atua dentro dele. Entre as 421 consultas com o alvo no top-10 dos dois sistemas, o reordenador rebaixa o alvo em 167 e o eleva em 141 (p = 0,154): o ganho nominal vem de trazer alvos ao top-10, e não de ordená-los.

São quatro medições pareadas não independentes, todas com estimativa pontual favorável e nenhuma estabelecida — ganho não estabelecido, e não ganho nulo; estabelecer o ganho nominal exigiria cerca de 6.640 consultas. O estágio, que custa 109 M de parâmetros em serviço, foi retirado da cadeia avaliada neste protocolo. A conclusão valeu só para a tarefa substituta: no assistente, o mesmo reordenador foi medido de novo e adotado (Seção 4.9).

### 4.7 Pré-treinamento no domínio contra escolha da base

Se treinar ou pré-treinar continuamente um encoder no domínio compensaria a escolha de uma base geral é a pergunta do trabalho complementar [2]. Com os mesmos 200 mil pares e a receita da Tabela 8, um encoder de 48 M treinado do zero com 0,6 bilhão de tokens de Física chega a 0,4712, contra 0,5270 do ModernBERT-base sem pré-treinamento no domínio; o pré-treinamento continuado deste com 0,4 bilhão de tokens acrescenta +0,0103 [+0,0032; +0,0176]; e trocá-lo pelo GTE-base acrescenta +0,0694 [+0,0580; +0,0812]. Com uma semente por braço, e com a diferença entre bases reunindo tamanho, contexto e o estágio contrastivo prévio do GTE, a consequência para o recuperador é a da Seção 4.5: a escolha da base pesa mais que o pré-treinamento no domínio que o orçamento permite.

### 4.8 Recuperação por equação

O conjunto foi desenhado sob a hipótese, comum na defesa de sistemas híbridos em domínios matemáticos, de que a recuperação densa perde o casamento simbólico que a busca léxica preserva. A consulta é uma equação de um documento da fatia de fonte LaTeX, e os alvos são os demais documentos com a mesma forma canônica, que desfaz só variação de superfície — ambiente, rótulos, espaçamento, sinônimos como `\le` e `\leq`; das 202.365.265 equações de 828.601 documentos resultaram 374.739 itens.

Pouca variação notacional chega ao gabarito: em 53,1% dos itens todos os alvos têm grafia idêntica à da consulta, em 37,0% diferem só por marcação superficial, e só em 7,8% nenhum alvo coincide. E 36,3% dos itens ligam a consulta a um documento com cinco ou mais equações em comum. Antes de qualquer modelo medido, a medida primária foi restringida ao estrato notacional sem documento quase igual (24.508 itens).

**Tabela 12.** Recuperação por equação no estrato notacional. Dois mil itens sorteados, 20.000 documentos no conjunto de candidatos (838.198 equações), escore do documento igual ao máximo sobre suas equações, os quatro sistemas sobre o mesmo conjunto, teto de 1,0. Diferença em recall@10 contra o BM25, por bootstrap pareado por item.

| Sistema | recall@1 | recall@10 | recall@50 | MRR | recall@10 − BM25 |
|---|---|---|---|---|---|
| BM25 [31], equação inteira | 0,7115 | 0,9300 | 0,9790 | 0,7902 | — |
| **Bi-encoder do sistema** (23 M, 6 M de pares) | **0,8595** | **0,9550** | 0,9765 | **0,8966** | **+0,0250 [+0,0115; +0,039]** |
| ModernBERT-base ajustado (200 mil pares) | 0,8295 | 0,9500 | 0,9695 | 0,8760 | +0,0200 [+0,0065; +0,034] |
| GTE-base ajustado (400 mil pares) | 0,8005 | 0,9200 | 0,9545 | 0,8450 | −0,0100 [−0,0245; +0,005] |

Pela medida primária registrada, recall@10, o bi-encoder fica +0,0250 acima do BM25 no estrato notacional (Tabela 12); em recall@1, não registrada como primária, a distância é de 14,8 pontos. A hipótese não se sustenta para dois dos três modelos densos em recall@10 — o GTE-base empata —, e em recall@50 nenhum supera o BM25. Há assimetrias nos dois sentidos: o BM25 lê a equação inteira, e os modelos densos, 192 tokens; mas o BM25 usa tokenização de texto, que descarta operadores e estrutura, e não é um casador simbólico. Nos estratos de grafia idêntica e superficial, a vantagem do bi-encoder em recall@1 é menor (+0,053 e +0,065, contra +0,148). O conjunto passou a ser reportado como medida em que o sistema vai bem, e não como teste do que a recuperação densa perde. Uma primeira execução foi anulada, porque cada sistema montou um conjunto de candidatos diferente (Tabela 15).

---

### 4.9 O assistente: a busca limita, e a tarefa final desfaz duas decisões

**Tabela 13.** O assistente no teste primário: 500 perguntas sobre um fato do resumo. Intervalos de 95% por bootstrap pareado por item. Juiz calibrado contra o Claude, sem julgamento humano.

| Braço | O que o modelo recebe | Acerto | IC 95% |
|---|---|---|---|
| A · sistema | as seis fontes que a busca trouxe | 0,700 | [0,660; 0,740] |
| B · fonte garantida | as mesmas, com P garantido | 0,944 | [0,924; 0,964] |
| B, depois da revisão dos erros | — | 0,976 | [0,962; 0,988] |
| C · sem fontes | só a pergunta | 0,168 | [0,136; 0,202] |

Pelas regras registradas, o gerador basta e a busca limita (Tabela 13). Com o artigo certo no prompt, o modelo aberto acerta 19 de cada 20 perguntas, e o sistema perde 0,244 [0,206; 0,282] porque a busca traz P entre as seis fontes em só 0,686 delas. O desfecho do gerador não depende da revisão dos erros — sem ela, B fica em 0,944, também acima de 0,90 —, e as perguntas exigem a literatura: de memória, 0,168. Nos 150 itens posteriores ao modelo, B se mantém (0,967) e C cai (0,133), de modo que a resposta vem da fonte. A primeira versão do juiz foi reprovada na validação, com κ de 0,52, por baixar respostas certas por forma ou por informação a mais; o prompt foi corrigido e validado em 100 respostas de itens disjuntos, com κ de 0,76. Para o programa, a consequência é que um gerador próprio de Física não se justifica por este uso: o investimento vai para a busca.

**Tabela 14.** Componentes da busca medidos na tarefa do assistente, cada um contra o sistema vigente naquele passo. "Tarefa substituta" é o que a Seção 4.6 concluiu para consultas de contexto de citação. Acerto das respostas no teste primário, pela regra da Seção 3.8; as linhas de desenvolvimento medem a fração de perguntas com P entre as seis fontes e não tocaram o teste.

| Componente | Tarefa substituta | Assistente | Decisão |
|---|---|---|---|
| Reordenador de base PhysBERT nos 50 primeiros | ganho não estabelecido; retirado | 0,700 → 0,806, +0,106 [+0,072; +0,142] | adotado |
| Reordenar mais fundo, ler 10 fontes, 3 consultas hipotéticas | — | desenvolvimento: 0 a +0,027 | não testado |
| Trechos do texto completo no índice, ao lado dos resumos | — | desenvolvimento: +0,007 | não testado |
| Trechos do texto completo das fontes, anexados ao prompt | — | corpo: +0,520 [+0,420; +0,620]; resumo: −0,028 [−0,052; −0,004] | não adotado |
| Fusão com o BM25, consultado com a pergunta em inglês | fusão com o BM25 empata com o bi-encoder; retirada | 0,806 → 0,870, +0,064 [+0,034; +0,094] | adotado |

Duas das cinco decisões invertem o que a tarefa substituta havia concluído (Tabela 14). O reordenador retirado na Seção 4.6 — o mesmo modelo, sem retreino — eleva a fração de perguntas com P entre as seis fontes de 0,686 para 0,818 e o acerto da resposta em 10,6 pontos, com 68 perguntas ganhas e 15 perdidas, e o mesmo efeito nos itens posteriores ao modelo (+0,100 [+0,040; +0,160]). Uma fusão com o BM25 sobre título e resumo, consultado com a pergunta traduzida para o inglês pelo próprio modelo, e não com o texto de um documento, leva P às seis fontes em 0,892 e o acerto a 0,870. A margem é fina: o limite inferior do intervalo passa o limiar de 0,03 por 0,4 ponto, e as perguntas, escritas lendo o resumo, repetem dele 72% das palavras, o que favorece a busca por palavra. Nas perguntas do corpo, que repetem menos, o ganho de posição foi de 0,04 [0,00; 0,09].

O que não se transferiu também informa. Reordenar mais fundo, ler dez fontes ou combinar três consultas hipotéticas não passa de 2,7 pontos no desenvolvimento, e cortar os pares do reordenador em 256 tokens custa 6: a ordem estava esgotada. Indexar trechos do texto completo ao lado dos resumos rende 0,7 ponto, porque o melhor trecho de um artigo concorrente se parece mais com a consulta do que o resumo dele (+0,029 de escore mediano), e o de P quase nada (+0,004): com dezenas de trechos por artigo, cresce mais a chance de um concorrente parecer relevante do que a de P.

**O que o resumo não diz.** Em 150 perguntas escritas a partir do corpo de artigos, o sistema acerta 0,033 [0,007; 0,067], o mesmo que sem fontes (0,060); com a passagem certa no lugar do resumo de P, 0,940 [0,900; 0,973]. Ter P entre as fontes não ajuda: com o resumo dele no prompt, são 4 acertos em 94. Um passo que corta o texto completo das seis fontes e anexa os nove trechos de maior BM25 contra a pergunta em inglês — que põe o trecho de origem entre os 12 primeiros em 0,85 das vezes, contra 0,33 do recuperador denso, treinado em resumos — leva o acerto em 100 perguntas novas do corpo de 0,000 a 0,500, e a 0,520 com uma instrução que cobra a coisa específica pedida. Nas perguntas do resumo, porém, o passo com a instrução custa 2,8 pontos [−5,2; −0,4], sobretudo onde um trecho do próprio P foi anexado: a resposta passa a usar o corpo do artigo, e o gabarito foi escrito do resumo. A regra pedia o intervalo inteiro acima de −0,03, e o passo ficou como opção. Um segundo juiz, validado contra os veredictos às cegas com κ de 0,914, chega ao mesmo desfecho (−0,024 [−0,052; +0,004]).

Quando as fontes não trazem a resposta, o modelo afirma algo em 51% das vezes (74 de 145 no estrato do corpo), mas raramente inventa: só 5 de 79 afirmações contêm número ausente das fontes. O erro é de atribuição — um valor que está nas fontes, de outra grandeza ou de outro artigo —, e um portão mecânico de números não o pegaria.

## 5. Discussão

### 5.1 Domínio, capacidade e força do recuperador

O resultado da Seção 4.6 é assimétrico de modo não previsto. O PhysBERT fica abaixo, como recuperador, até do MiniLM-L6 sem ajuste — 0,3507 contra 0,4761 na Tabela 6 — e é a melhor base de reordenação entre as três testadas. Pré-treinamento de domínio, nas condições medidas, não produziu um bi-encoder competitivo e produziu um cross-encoder competitivo. Não há explicação medida; a hipótese de menor custo de teste é que o cross-encoder explora interação termo a termo entre consulta e documento, em que vocabulário de domínio é diretamente utilizável, enquanto o bi-encoder comprime o documento antes de ver a consulta.

Na tarefa substituta, o valor desse reordenador mostrou-se condicional à fraqueza do recuperador: com o recall@100 passando de 0,516 a 0,633, ele deixou de acrescentar informação mensurável e, dentro do top-10, rebaixou o alvo nominalmente mais vezes do que o elevou. Na tarefa do assistente, sobre o mesmo recuperador forte, foi o componente de maior ganho medido (Seção 4.9). A avaliação de um encoder de domínio apenas como recuperador, prática comum, teria descartado o PhysBERT, cujo valor apareceu em outro papel e noutra tarefa.

Para a recuperação de primeira etapa, as Tabelas 7 e 8 dizem que amostragem, volume e base contam, nessa ordem de custo: sortear os pares é grátis, o volume custa tempo de treinamento uma vez, e a base custa em toda consulta servida. Com um conjunto de pares quase esgotado, o próximo ganho teria de vir de outra fonte de pares ou de uma base mais forte pagando o custo de servir.

### 5.2 Tarefa substituta e tarefa final

Recuperar o documento citado a partir do citante é uma tarefa substituta conveniente: o gabarito é gratuito, o conjunto é grande e a métrica é estável. Ela serviu para treinar e escolher o recuperador. Para retirar componentes da cadeia, não serviu: as duas decisões desse tipo que ela tomou — retirar o reordenador e retirar o BM25 — foram revertidas quando medidas no assistente, com regras registradas antes (Tabela 14).

Três diferenças entre as tarefas são candidatas, e nenhuma foi isolada experimentalmente. A consulta: no protocolo de citação, é o título e o resumo inteiros do citante, que já têm a forma dos documentos e diluem as palavras raras; no assistente, é uma pergunta curta e específica, cujas palavras raras o BM25 aproveita. O alvo: o documento citado é um entre vários que o citante poderia ter citado; a pergunta tem um artigo de origem e um fato. E o que se mede: no protocolo, a posição de um documento citado no top-10; no assistente, a presença do artigo de origem entre as seis fontes lidas — e 17% deles estavam entre a 7ª e a 50ª posição, a faixa que o reordenador recebe.

A lição é de procedimento. Uma tarefa substituta escolhe entre variantes de um componente; manter ou retirar o componente é decisão da tarefa final, medida com o sistema inteiro. E a separação entre gerador e busca, por braços que diferem só na presença da fonte, mostrou onde investir antes de qualquer gasto: com o gerador aberto acertando 0,976 com a fonte certa, nenhum pré-treinamento de um gerador de Física se justifica por este uso — o mesmo padrão que o trabalho complementar [2] mediu para encoders, em que a escolha da base pesou mais que o pré-treinamento no domínio.

### 5.3 Vieses de amostragem em conjuntos derivados de grafos de citação

Uma auditoria identificou oito ocorrências distintas de amostragem dependente da ordem dos dados no mesmo repositório. As cinco que tocam este trabalho estão na Tabela 15; as outras três, na preparação e na avaliação do pré-treinamento, estão no trabalho complementar [2]. A direção do viés não é constante: a terceira elevou a métrica, e a quarta a rebaixou.

**Tabela 15.** Ocorrências de amostragem dependente da ordem dos dados, com o efeito medido.

| Local | Efeito medido | Consequência |
|---|---|---|
| Amostra de revisão do corpus de texto integral | 400 documentos, todos nos 2 primeiros de 277 arquivos — 0,67% do corpus, só resumos | Invalidou a amostra destinada a estimar contaminação |
| Avaliação sobre arquivo agrupado | 500 linhas correspondiam a 35 documentos; intervalo de 95% de ±0,159 | Duas conclusões mutuamente contraditórias |
| Divisão treino/validação por posição | 49,6% das âncoras da validação presentes no treino | nDCG@10 de 0,139 para 0,020 sobre documentos inéditos |
| Conjunto de avaliação e de treinamento do recuperador | Teto de nDCG@10 de 0,7562; 10,7 vezes menos documentos distintos no treinamento | Inverteu o veredito contra o modelo geral (Seção 4.4) |
| Conjunto de candidatos da recuperação por equação | Sorteio sobre uma lista cuja ordem mudava entre execuções: quatro sistemas viram 841.101, 843.127, 836.422 e 831.048 equações | Comparação pareada entre sistemas que não viram os mesmos distratores; medição anulada (Seção 4.8) |

Duas observações parecem generalizáveis. A primeira é que o defeito é invisível por construção: nenhuma ocorrência gera exceção ou se manifesta na função de perda, e todas produzem valores dentro da faixa esperada. A última foi exposta por acaso, quando os sistemas passaram a rodar em processos distintos; num processo só, teriam partilhado o conjunto, e o resultado sairia certo pelo motivo errado.

A segunda é que a correção adequada não é trocar o esquema de amostragem num ponto — o código de reordenação já sorteava, e a justificativa não se propagou ao de embedding —, mas verificar a cada execução que o conjunto de avaliação não contém consulta ou alvo repetido, com o teto gravado no artefato. Trabalhos que derivam do mesmo grafo o rótulo positivo, o critério de negativo e a verdade de avaliação estão estruturalmente expostos a esta classe de defeito; recomenda-se reportar, ao lado de cada métrica agregada por grupo, o número de grupos distintos medidos.

### 5.4 Implicações práticas

Cinco padrões emergiram, nenhum relativo a arquitetura.

Medir a fonte antes de adquiri-la. A fatia com fonte LaTeX já estava em disco havia dezessete dias quando se recomendou a aquisição paga de acesso ao fonte do arXiv como pré-requisito; faltava a medição da Tabela 4, de custo desprezível.

Medir o teto do conjunto de avaliação a cada execução. O defeito da Seção 4.4 não tinha sintoma: os números caíam na faixa esperada, e todos os modelos eram submetidos ao mesmo protocolo. Só o teto de um recuperador perfeito o expôs.

Empregar comparação pareada e medir os braços na mesma sessão. Com 256 candidatos, o erro padrão é de cerca de ±0,031, contra diferenças de interesse da ordem de 0,004; o teste pareado sobre os mesmos itens é o que torna essas diferenças acessíveis. E um valor histórico carrega os parâmetros da execução que o produziu — na Seção 4.6, a profundidade de candidatos —, o que teria multiplicado por sete o ganho reportado.

Comparar as bases candidatas antes de investir numa delas. A comparação entre bases custou 1 hora e 17 minutos de acelerador, contra cerca de 20 horas do pré-treinamento continuado, e mostrou uma diferença maior que a que ele produziu [2].

Medir na tarefa final antes de retirar um componente. Duas retiradas decididas na tarefa substituta foram desfeitas no assistente, onde levaram o acerto de 0,700 a 0,870 (Seção 4.9); medi-las de novo custou cerca de dez horas de máquina local.

---

## 6. Limitações

**Domínio e relevância.** Um domínio e um grafo de citação: Física e OpenAlex. Consultas e candidatos são títulos e resumos de artigos do arXiv. A definição operacional de relevância — relevante equivale a citado — favorece artigos citáveis e áreas de citação densa, e não captura relevância não citada; não há julgamento humano, e os valores de nDCG não se comparam aos de benchmarks anotados. Nenhum resultado foi medido em conjunto externo publicado, e os modelos de mesma técnica — SPECTER [4], SciNCL [5] e SPECTER2 [6] — não foram medidos.

**Seleção e divisão.** Não há conjunto de teste separado para o recuperador: o ponto de verificação de cada modelo ajustado é escolhido em 1.000 pares que são os primeiros do sorteio de 2.000 que dá o veredito, e as decisões de receita e de base foram tomadas no mesmo conjunto. Os valores dos modelos ajustados são otimistas, mais nas execuções longas; na de 6 milhões, o pico fica 0,007 acima do último ponto — pouco contra a margem de 0,044 sobre o GTE-large, mas da ordem das diferenças lidas como empate. A divisão é por documento citante, e os candidatos podem ocorrer como positivos no treinamento: é ajuste sobre acervo fixo, e não generalização a documentos inéditos, e a fração de candidatos vistos, não medida, cresce com o volume.

**Exposição prévia à tarefa.** O all-MiniLM-L6-v2 foi treinado com cerca de 169 milhões de pares de citação do S2ORC, que cobre o arXiv; a tarefa coincide em forma, e possivelmente em instâncias, com esses dados, e a sobreposição não foi medida. A linha sem ajuste da Tabela 6 não é, por isso, uma medição sem exposição à tarefa, e risco análogo vale para o all-MiniLM-L12-v2 e para as bases GTE, BGE e E5, cujos dados não foram auditados.

**Sementes e configuração de leitura.** Todo modelo foi treinado uma vez; duas execuções da mesma receita em máquinas distintas diferiram de 0,002 a 0,008 de nDCG@10, e diferenças dessa ordem — a da Tabela 10, os empates da Tabela 8 — não foram replicadas. Todos os modelos são lidos a 192 tokens e sem prefixo de instrução; o GTE-large não foi medido com os 512 tokens para que foi treinado, e algumas bases de 33 M foram treinadas com prefixo.

**Bases.** A busca por base foi encerrada por custo, sem que nenhuma alternativa fosse levada a 6 milhões de pares. Base é variável composta — tamanho, contexto e pré-treinamento contrastivo diferem juntos.

**Reordenação.** A comparação da Tabela 10 foi medida com o recuperador antigo, controla tamanho e arquitetura, mas não separa corpus, objetivo e vocabulário da base, e a taxa de aprendizado não foi reajustada entre tamanhos de modelo. O filtro de co-citação remove uma classe de falso negativo — 9,1% dos negativos minerados eram co-citados com o positivo —, e o residual não foi medido.

**Recuperação por equação.** A consulta é uma equação transcrita de um artigo, e não uma consulta de usuário. A restrição ao estrato notacional, registrada antes de qualquer modelo medido, foi decidida depois da medição da grafia dos itens; o gabarito só contém variação de superfície; o controle léxico não é um casador simbólico; os valores valem para 20.000 documentos candidatos, 2,4% do corpus; e a base sem ajuste não foi medida.

**Assistente.** Nenhum humano julgou uma resposta: o juiz, o mesmo modelo que responde, foi calibrado contra o Claude, que também escreveu as perguntas e os gabaritos e revisou os erros do braço B; o segundo juiz, de outra família, conferiu uma decisão e foi validado contra os mesmos veredictos. As perguntas têm o estilo de um modelo, saem de um fato de um resumo ou de uma passagem e repetem muitas palavras do texto de origem, o que favorece a busca por palavra; com perguntas de usuários, os ganhos podem ser menores. O teste primário decidiu quatro vezes. O gabarito vem só do texto de origem, e outro artigo pode responder certo com outro valor. O portão de citações não verifica se a fonte sustenta a frase, e 48% do índice não tem texto completo.

**Corpus.** Os tokens da Tabela 2 são estimados, sem deduplicação entre fontes. A taxa de contaminação do corpus filtrado não está medida — as taxas de falso positivo foram medidas em resumos, e o corpus é de texto integral; uma amostra estratificada de 200 resumos e 200 textos integrais está em julgamento humano, com o escore do classificador oculto.

**Reprodução.** A reexecução completa do pipeline não reproduz hoje os artefatos avaliados: o conjunto de pares, reconstruído com o código atual, conserva só 2,1% das âncoras de validação, porque o critério de embaralhamento mudou. Os arquivos avaliados foram restaurados byte a byte, e todos os números referem-se a eles.

---

## 7. Conclusão

Sob orçamento de computação nulo, foi possível construir um corpus de Física de cerca de 27,75 bilhões de tokens e um índice de 1,6 milhão de registros, verificáveis por uma cadeia de hashes, e um recuperador de 23 M que, ajustado por citação com 6 milhões de pares, fica acima de um modelo geral de 335 M em nDCG@10 (+0,044), recall@10 e MRR, com as ressalvas de seleção do ponto de verificação e de divisão por documento citante declaradas na Seção 6.

O caminho até esse resultado ensina mais que o número. O protocolo original tinha teto de 0,7562 e invertia o veredito, e a mesma amostragem pela ordem do arquivo tirava do treinamento nove de cada dez documentos distintos; corrigi-la valeu +0,020 sem custo. Uma base geral de 109 M alcança com 1 milhão de pares o que o volume produziu com 6 milhões, a mais de quatro vezes o custo de servir. E, na recuperação por equação, o bi-encoder supera o controle léxico no estrato de variação notacional — variação que, no gabarito, é só de superfície.

No assistente construído sobre esse recuperador, um modelo aberto de 8 bilhões de parâmetros acerta 19 de cada 20 perguntas quando o artigo certo está entre as fontes, e o limite é a busca. Medidos nessa tarefa, com regras registradas antes, o reordenador de base pré-treinada em Física e a fusão com o BM25 — os dois retirados na tarefa substituta — voltaram à cadeia e levaram o acerto de 0,700 a 0,870. O que o resumo não diz o sistema não responde (0,033), e o passo que o buscaria no texto completo custa nas perguntas que o resumo responde: ficou opcional.

O resultado metodológico de maior alcance é que nenhum dos defeitos que mudaram conclusões tinha sintoma além do próprio número. Em conjuntos em que rótulo, negativo e verdade de avaliação saem do mesmo grafo, a unidade de amostragem e o teto do conjunto precisam ser verificados a cada execução, e não assumidos por convenção; e uma tarefa substituta escolhe variantes de um componente, mas não decide se ele fica na cadeia.

Permanecem em aberto um conjunto de teste separado para o recuperador, a variância entre sementes, a comparação com SPECTER2 e SciNCL num conjunto externo, a contaminação do corpus filtrado e o julgamento humano das respostas do assistente, com perguntas de usuários reais.

---

## Disponibilidade de dados e código

O código e a documentação de projeto são públicos sob licença permissiva. Cada etapa do pipeline registra manifesto com hashes das entradas e saídas, parâmetros e commit, o que permite reconstrução a partir das fontes públicas, com as ressalvas da Seção 6. O corpus não acompanha o código: os registros do arXiv seguem a licença de cada submissão, e a licença padrão concede direito de distribuição ao arXiv, não a terceiros. As perguntas e os gabaritos da avaliação do assistente não são publicados, para que o conjunto de teste continue de teste; versionam-se os hashes e os resultados agregados.

**Nota de autoria.** O trabalho experimental, as decisões de projeto e a implementação são do autor. A redação deste manuscrito contou com assistência de um modelo de linguagem (Claude, Anthropic) a partir dos artefatos de medição do repositório; o autor revisou o texto e responde por ele. Na avaliação do assistente, um modelo da mesma família escreveu as perguntas, calibrou o juiz e revisou os erros, como descrito na Seção 3.8. As referências foram conferidas contra as respectivas fontes.

---

## Referências

[1] Hoffmann, J.; Borgeaud, S.; Mensch, A.; et al. Training Compute-Optimal Large Language Models. *Advances in Neural Information Processing Systems 35* (NeurIPS 2022), 2022.

[2] Sanchez, V. Pré-treinamento de encoders para texto de Física em LaTeX sob restrição severa de computação: tokenização, mascaramento de equações e escolha da base. Manuscrito em preparação, 2026.

[3] Beltagy, I.; Lo, K.; Cohan, A. SciBERT: A Pretrained Language Model for Scientific Text. *Proceedings of EMNLP-IJCNLP 2019*, 2019.

[4] Cohan, A.; Feldman, S.; Beltagy, I.; Downey, D.; Weld, D. S. SPECTER: Document-level Representation Learning using Citation-informed Transformers. *Proceedings of ACL 2020*, p. 2270–2282, 2020.

[5] Ostendorff, M.; Rethmeier, N.; Augenstein, I.; Gipp, B.; Rehm, G. Neighborhood Contrastive Learning for Scientific Document Representations with Citation Embeddings. *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing* (EMNLP 2022), p. 11670–11688, 2022.

[6] Singh, A.; D'Arcy, M.; Cohan, A.; Downey, D.; Feldman, S. SciRepEval: A Multi-Format Benchmark for Scientific Document Representations. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (EMNLP 2023), p. 5548–5566, 2023.

[7] Bhattacharjee, B.; Trivedi, A.; Muraoka, M.; et al. INDUS: Effective and Efficient Language Models for Scientific Applications. *Proceedings of EMNLP 2024, Industry Track*, 2024. arXiv:2405.10725.

[8] Hellert, T.; Montenegro, J.; Pollastro, A. PhysBERT: A text embedding model for physics scientific literature. *APL Machine Learning*, v. 2, n. 4, art. 046105, 2024. DOI 10.1063/5.0238090.

[9] Li, Z.; Zhang, X.; Zhang, Y.; Long, D.; Xie, P.; Zhang, M. Towards General Text Embeddings with Multi-stage Contrastive Learning. arXiv:2308.03281, 2023.

[10] Xiao, S.; Liu, Z.; Zhang, P.; Muennighoff, N.; Lian, D.; Nie, J.-Y. C-Pack: Packed Resources For General Chinese Embeddings. *Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval* (SIGIR 2024), p. 641–649, 2024.

[11] Wang, L.; Yang, N.; Huang, X.; Jiao, B.; Yang, L.; Jiang, D.; Majumder, R.; Wei, F. Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv:2212.03533, 2022.

[12] Reimers, N.; Gurevych, I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP-IJCNLP 2019*, p. 3982–3992, 2019.

[13] Sentence-Transformers. all-MiniLM-L6-v2 (ficha do modelo). Hugging Face, https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2. Acesso em 1º de outubro de 2026.

[14] Wang, W.; Wei, F.; Dong, L.; Bao, H.; Yang, N.; Zhou, M. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *Advances in Neural Information Processing Systems 33* (NeurIPS 2020), 2020.

[15] Lo, K.; Wang, L. L.; Neumann, M.; Kinney, R.; Weld, D. S. S2ORC: The Semantic Scholar Open Research Corpus. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (ACL 2020), p. 4969–4983, 2020.

[16] Karpukhin, V.; Oguz, B.; Min, S.; Lewis, P.; Wu, L.; Edunov, S.; Chen, D.; Yih, W. Dense Passage Retrieval for Open-Domain Question Answering. *Proceedings of EMNLP 2020*, p. 6769–6781, 2020.

[17] Xiong, L.; Xiong, C.; Li, Y.; Tang, K.-F.; Liu, J.; Bennett, P. N.; Ahmed, J.; Overwijk, A. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. *International Conference on Learning Representations* (ICLR 2021), 2021.

[18] Qu, Y.; Ding, Y.; Liu, J.; Liu, K.; Ren, R.; Zhao, W. X.; Dong, D.; Wu, H.; Wang, H. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. *Proceedings of NAACL-HLT 2021*, 2021.

[19] Soldaini, L.; Lo, K. peS2o (Pretraining Efficiently on S2ORC) Dataset. Relatório técnico, Allen Institute for AI, 2023. Licença ODC-By.

[20] Weber, M.; Fu, D.; Anthony, Q.; et al. RedPajama: an Open Dataset for Training Large Language Models. *Advances in Neural Information Processing Systems 37, Datasets and Benchmarks Track* (NeurIPS 2024), 2024. arXiv:2411.12372.

[21] Paster, K.; Dos Santos, M.; Azerbayev, Z.; Ba, J. OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text. *International Conference on Learning Representations* (ICLR 2024), 2024.

[22] Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.; Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.; Rocktäschel, T.; Riedel, S.; Kiela, D. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems 33* (NeurIPS 2020), p. 9459–9474, 2020.

[23] Gao, L.; Ma, X.; Lin, J.; Callan, J. Precise Zero-Shot Dense Retrieval without Relevance Labels. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics* (ACL 2023), v. 1: Long Papers, p. 1762–1777, 2023.

[24] Gao, T.; Yen, H.; Yu, J.; Chen, D. Enabling Large Language Models to Generate Text with Citations. *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (EMNLP 2023), p. 6465–6488, 2023.

[25] Zheng, L.; Chiang, W.-L.; Sheng, Y.; et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track* (NeurIPS 2023), 2023.

[26] Cohen, J. A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, v. 20, n. 1, p. 37–46, 1960.

[27] Priem, J.; Piwowar, H.; Orr, R. OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv:2205.01833, 2022.

[28] van den Oord, A.; Li, Y.; Vinyals, O. Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748, 2018.

[29] Gao, L.; Zhang, Y.; Han, J.; Callan, J. Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup. *Proceedings of the 6th Workshop on Representation Learning for NLP* (RepL4NLP 2021), p. 316–321, 2021.

[30] Warner, B.; Chaffin, A.; Clavié, B.; et al. Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. *Proceedings of ACL 2025*, 2025. arXiv:2412.13663.

[31] Robertson, S.; Zaragoza, H. The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, v. 3, n. 4, 2009.

[32] Cormack, G. V.; Clarke, C. L. A.; Büttcher, S. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *Proceedings of the 32nd International ACM SIGIR Conference*, p. 758–759, 2009.

[33] McNemar, Q. Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, v. 12, n. 2, p. 153–157, 1947.

[34] Wilson, E. B. Probable Inference, the Law of Succession, and Statistical Inference. *Journal of the American Statistical Association*, v. 22, n. 158, p. 209–212, 1927.

[35] Yang, A.; et al. Qwen3 Technical Report. arXiv:2505.09388, 2025.

[36] Mistral AI. Mistral-Small-3.2-24B-Instruct-2506 (ficha do modelo). Hugging Face, https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506. Acesso em 16 de outubro de 2026.
