# Construção de um sistema de recuperação de literatura de Física sob restrição severa de computação: corpus, tokenização, representação e verificação

Vinicius Sanchez

*Pesquisa independente*

*Versão 0.2 — 7 de setembro de 2026*

---

## Resumo

> A escassez de texto em domínios científicos estreitos limita a aplicação direta de leis de escala a modelos de fundação especializados. Estimativas do corpus de Física legalmente adquirível situam-se entre 15 e 30 bilhões de tokens após filtragem, contra os cerca de 160 bilhões que um modelo de 8 bilhões de parâmetros exigiria sob a relação de computação ótima de Hoffmann et al. Este trabalho descreve a construção, sob orçamento de computação nulo — uma GPU de consumo de 8 GB e cotas gratuitas de aceleradores em nuvem —, de um sistema de recuperação de literatura de Física, e reporta tanto os resultados obtidos quanto os que permanecem em aberto. Foram montados um índice de metadados de 1.595.422 registros do arXiv associado a 4.613.751 obras do OpenAlex com taxa de casamento de 99,1%, um classificador de domínio com acurácia de 0,954 e um corpus de treinamento de 27,75 bilhões de tokens. Um bi-encoder de 23 milhões de parâmetros, ajustado por supervisão de citação, supera o PhysBERT (109 M) em 0,190 de nDCG@10 e iguala estatisticamente um modelo geral de 335 M com 1/14,8 dos parâmetros. A composição de recuperação léxica e densa por fusão recíproca de postos supera cada recuperador isolado (p < 0,001). O principal resultado positivo é sobre reordenação: entre três cross-encoders idênticos exceto pelo corpus de pré-treinamento da base, apenas o pré-treinado em Física melhora a fusão (nDCG@10 de 0,1666 contra 0,1576; p = 0,0062), enquanto um modelo geral forte de mesmo tamanho e arquitetura empata (p = 0,637) — e o modelo de domínio é, isoladamente, o pior recuperador do conjunto. Reporta-se ainda que a presença de ambiente de equação, e não a ausência de operandos, é o discriminante válido para integridade de notação matemática em corpora (84,9% contra 0,0% entre duas fatias do mesmo domínio). Uma auditoria identificou seis ocorrências independentes de amostragem por posição em dados ordenados, uma das quais impunha teto de 0,7562 ao nDCG@10 do protocolo de avaliação principal; os resultados de recuperação afetados são reportados como provisórios e estão em remedição.

**Palavras-chave:** recuperação de informação científica; supervisão por citação; modelos de embedding de domínio; construção de corpus; tokenização de LaTeX; reprodutibilidade.

---

## 1. Introdução

Modelos de linguagem de propósito geral apresentam, em Física, modos de falha distintos dos observados em tarefas factuais: incoerência dimensional, deriva algébrica ao longo de derivações longas, ausência de redução correta em casos-limite e mistura silenciosa de convenções de sinal e de sistema de unidades. Nenhum desses modos é corrigível apenas pelo acréscimo de texto de domínio, porque nenhum deles é penalizado pelo objetivo de predição do próximo token, que premia plausibilidade local.

A restrição que organiza este trabalho, contudo, é de recurso. Sob a relação de computação ótima estabelecida por Hoffmann et al. [1], um modelo de 8 bilhões de parâmetros requer aproximadamente 160 bilhões de tokens de treinamento. Modelando o funil de aquisição e filtragem estágio a estágio para a literatura de Física legalmente adquirível a custo zero, obtêm-se entre 39 e 73 bilhões de tokens brutos, ou entre 15 e 30 bilhões após deduplicação e triagem de licença — uma escassez de cinco a dez vezes.

A resposta usual a essa restrição é ampliar a aquisição. A adotada aqui é distinta, e é sustentada pela literatura recente: treinamento a partir de inicialização aleatória é justificável apenas onde o dado é excedente — encoders na faixa de 10⁸ parâmetros —, e capacidades acima disso devem ser obtidas por pré-treinamento continuado sobre uma base geral forte. A evidência é consistente. O Galactica [2], única tentativa de larga escala de treinar um modelo científico de fundação a partir do zero, foi retirado de circulação três dias após o lançamento, com alucinação de citações entre os defeitos determinantes. Em contraste, Minerva [3], Llemma [4] e DeepSeekMath [5] foram todos obtidos por pré-treinamento continuado sobre bases gerais, e todos reportam ganhos substanciais em raciocínio quantitativo.

Uma segunda restrição, metodológica, decorre da primeira: com orçamento de computação nulo, cada experimento precisa ser justificado por uma medição anterior. Esse regime tem um efeito colateral relevante para os resultados aqui reportados — medições anteriores passam a ser reexaminadas com frequência, e parte substancial dos achados deste trabalho, inclusive os negativos, originou-se desse reexame.

As contribuições são as seguintes.

1. Um procedimento de construção de corpus de Física a custo zero, com o índice de metadados atestado por uma cadeia de hashes sobre 21,79 GB, e a caracterização quantitativa do funil de filtragem (Seções 3.1 a 3.3 e 4.1).
2. Um discriminante medido para integridade de notação matemática em corpora, e a demonstração de que o diagnóstico intuitivo para o mesmo fim satura no corpus íntegro (Seção 4.3).
3. Evidência empírica sobre a escolha de algoritmo de tokenização para texto em LaTeX, que não localizamos na literatura (Seção 4.4).
4. A separação experimental entre domínio e capacidade como propriedade relevante da base de um cross-encoder de reordenação, com controle de tamanho, arquitetura, dados, semente e hiperparâmetros (Seção 4.8).
5. Uma auditoria de vieses de amostragem em conjuntos derivados de grafos de citação, com o efeito medido de cada ocorrência (Seção 5.2).

O trabalho não alega um modelo de Física em estado da arte. A Seção 4.5 reporta um resultado de recuperação e, em seguida, o retira: uma auditoria do protocolo, posterior às medições, identificou um teto de 0,7562 no nDCG@10 alcançável por um modelo perfeito. Os números afetados são apresentados como históricos, e o critério de aceitação correspondente permanece aberto.

---

## 2. Trabalhos relacionados

### 2.1 Encoders científicos

O SciBERT [6] é o baseline convencional para tarefas de linguagem em texto científico, mas seu corpus de treinamento é composto majoritariamente de literatura biomédica, com participação marginal de Física. Superá-lo em tarefas de Física não constitui, portanto, evidência de especialização. O SPECTER [7] introduz o uso do grafo de citação como sinal de relacionamento entre documentos científicos, abordagem que este trabalho adota. O INDUS [8] cobre Ciências da Terra, heliofísica, ciências planetárias e astrofísica, mas não as subáreas de altas energias, matéria condensada e informação quântica. O PhysBERT [9] é o competidor de mesmo domínio: um modelo de embedding pré-treinado sobre 1,2 milhão de artigos de Física do arXiv.

Modelos gerais de embedding treinados com aprendizado contrastivo em larga escala, como o GTE [10], constituem uma segunda barra de comparação frequentemente omitida em trabalhos de domínio. A arquitetura de bi-encoder empregada aqui segue Reimers e Gurevych [11], e o modelo base do sistema principal é o MiniLM [12], obtido por destilação de auto-atenção.

### 2.2 Recuperação densa e mineração de negativos

O DPR [13] estabelece o arcabouço de bi-encoder duplo para recuperação densa e emprega negativos difíceis provenientes de BM25. O ANCE [14] reamostra negativos do próprio índice durante o treinamento, argumentando que negativos amostrados no lote produzem gradientes pouco informativos. O RocketQA [15] documenta a presença de falsos negativos nessa mineração e propõe filtragem por um cross-encoder.

A Seção 4.7 reporta um modo de falha vizinho e distinto: o problema não é o rótulo do negativo estar incorreto, mas o critério de seleção do negativo ser correlacionado com o escore do modelo que será reordenado. O sintoma não é queda de precisão, e sim inversão de correlação, sem qualquer manifestação na função de perda.

### 2.3 Corpora e tokenização

O peS2o [16] é um corpus de artigos científicos de acesso aberto derivado do S2ORC, com texto integral extraído de PDF. O RedPajama [17] inclui uma fatia do arXiv construída a partir do fonte LaTeX. O OpenWebMath [18] extrai texto matemático da web preservando notação. A Seção 4.3 compara as três quanto à integridade da notação.

Em tokenização, a codificação por pares de bytes [19] e o modelo de unigramas [20] são as duas alternativas dominantes. Bostrom e Durrett [21] reportam vantagem do modelo de unigramas em linguagem natural, atribuída a melhor alinhamento morfológico. Não localizamos evidência publicada equivalente para texto em LaTeX, e a Seção 4.4 apresenta a nossa.

---

## 3. Materiais e métodos

### 3.1 Aquisição do índice de metadados

Os metadados do arXiv foram coletados pelo protocolo OAI-PMH restrito ao conjunto `physics`, com cursor durável por lote e retomada idempotente. As obras do OpenAlex [22] foram obtidas do instantâneo público, lido por faixas de bytes HTTP sobre 13 das 189 colunas disponíveis, sem materialização em disco.

O casamento entre as duas fontes é feito pelo campo de localizações do registro do OpenAlex. Três decisões de esquema foram determinadas empiricamente e são reportadas por afetarem substancialmente a cobertura: o casamento pelo campo de identificadores externos do arXiv recupera 1,5% dos registros, contra 98,5% pelo campo de localizações; restringir à localização primária exclui 1,44 milhão de registros revisados por pares; e uma expressão regular de identificadores em formato anterior a 2007 truncava 41,5% do acervo.

### 3.2 Classificação de domínio

O rótulo de Física é derivado da presença de qualquer categoria da família de Física no registro, e não do prefixo da categoria primária, uma vez que artigos submetidos a outras áreas com listagem cruzada em Física pertencem a ambos os conjuntos. A validação dessa regra é exata: dos 72.919 registros negativos com listagem cruzada em Física, todos os 72.919 constam do índice.

O classificador é linear sobre representação esparsa, com perda `modified_huber`, treinado sobre amostra estratificada de 300.000 documentos positivos e 190.210 negativos, estes provenientes de quatro domínios do arXiv: ciência da computação, economia, matemática e biologia quantitativa. A avaliação principal é por exclusão de domínio — treina-se omitindo um domínio negativo inteiro e mede-se a taxa de falso positivo nele.

### 3.3 Fontes de texto integral

Três fatias públicas foram filtradas para Física pelo classificador da Seção 3.2, com limiar de decisão de 0,9: a fatia arXiv do RedPajama [17], o OpenWebMath [18] e o peS2o [16]. Todas as fontes são fixadas por revisão explícita do repositório de origem; a versão inicial da implementação baixava de referência móvel, o que impossibilita reprodução exata.

A integridade da notação matemática de cada fatia foi medida em 3.000 documentos sorteados por fatia, sobre quatro indicadores: fração de documentos contendo qualquer marcação LaTeX, fração contendo delimitadores de matemática em linha, fração contendo ambiente de equação em display, e número de sequências matemáticas por documento.

### 3.4 Modelos e treinamento

**Bi-encoder.** MiniLM-L6 [12] de 23 milhões de parâmetros, ajustado com perda contrastiva InfoNCE [23] sobre pares extraídos do grafo de citação, com lote de 128 e portanto 127 negativos no lote, comprimento máximo de 192 tokens e agregação por média. O conjunto de treinamento contém 6.564.111 pares e 667.304 documentos citados distintos. Uma variante foi treinada com 511 negativos no lote por caching de gradiente [24].

**Recuperação léxica.** BM25 [25] sobre índice esparso, com os mesmos documentos.

**Cross-encoder de reordenação.** Modelo par a par treinado sobre grupos de 8 candidatos amostrados da distribuição exata da avaliação, isto é, os 50 primeiros candidatos produzidos pela fusão. Três bases foram comparadas mantendo idênticos os seis hiperparâmetros restantes, os dados, a semente e o protocolo: MiniLM-L6 (23 M), GTE-base (109 M) [10] e PhysBERT (109 M) [9].

**Encoder pré-treinado a partir do zero.** Arquitetura bidirecional de 150 milhões de parâmetros, vocabulário de 40.960, comprimento de sequência de 8.192 e objetivo de modelagem de linguagem mascarada a 30% — taxa adotada seguindo os resultados de Warner et al. [26] —, sem predição de sentença seguinte, com agendamento de taxa de aprendizado do tipo *warmup–stable–decay*. A adição específica de domínio, sujeita a ablação, consiste em mascarar equações inteiras em display em uma fração dos exemplos, mantendo igual entre os braços o orçamento total de tokens mascarados.

### 3.5 Protocolo de avaliação

A tarefa de avaliação é a recuperação do documento citado a partir do texto do documento citante. As métricas são recall@k e nDCG@10. A composição de sistemas usa fusão recíproca de postos [27] com k = 60, sobre profundidade 50.

Os conjuntos de avaliação são amostrados uniformemente com semente registrada, com deduplicação por texto do documento alvo. A necessidade dessa deduplicação é discutida na Seção 4.5.

### 3.6 Análise estatística

Comparações entre sistemas são feitas por teste de McNemar exato [28] sobre pares discordantes do evento "o documento alvo alcançou as k primeiras posições", com os mesmos itens submetidos a todos os sistemas. Intervalos para proporções usam o método de Wilson [29], escolhido porque a aproximação normal produz intervalo de largura nula na ausência de observações contrárias, regime frequente nas taxas medidas aqui.

Onde há múltiplas variantes comparadas contra o mesmo controle, aplica-se correção de Bonferroni, com o limiar registrado antes da coleta dos dados, junto com a regra de leitura de cada desfecho possível.

Nas medições de degradação de equações, o bootstrap reamostra artigos e não equações, porque as equações de um mesmo artigo compartilham o tratamento que o pipeline de extração lhes aplicou; tratá-las como independentes produziria intervalo artificialmente estreito.

### 3.7 Reprodutibilidade

Cada etapa do pipeline grava um manifesto com os hashes BLAKE3 das entradas e saídas, os parâmetros e o identificador do commit. Uma cadeia de hashes em três níveis cobre, a partir de uma única raiz, o manifesto de cada etapa e, através dele, cada arquivo. A verificação superficial confere os manifestos; a verificação profunda relê os dados e é a única capaz de detectar alteração no conteúdo dos arquivos.

---

## 4. Resultados

### 4.1 Corpus

**Tabela 1.** Índice de metadados construído, contra a estimativa de planejamento.

| Grandeza | Medido | Estimado no planejamento |
|---|---|---|
| Registros do arXiv (conjunto `physics`) | 1.595.422 | 1.200.000 |
| Tamanho em disco | 674 MB (422 bytes/registro) | 516–686 bytes/registro |
| Obras do OpenAlex processadas | 4.613.751 | — |
| Taxa de casamento com o índice | 99,1% (1.581.098) | 98,5% |
| Arestas de citação | acima de 22,7 milhões | dezenas de milhões |
| Revisados por pares | 740.823 (46,4%) | — |
| Fração redistribuível | 14,8% (235.795) | 25–35% |

A fração redistribuível é a única estimativa de planejamento refutada, e a discrepância é ampla. Ela depende fortemente da época de publicação: 0,0% até 2004, 36,2% entre 2020 e 2024, e 48,8% entre 2025 e 2029.

**Tabela 2.** Fatias de texto integral após filtragem para Física.

| Fonte | Documentos aceitos | Tokens estimados |
|---|---|---|
| RedPajama, fatia arXiv [17] | 835.379 | 10,54 × 10⁹ |
| OpenWebMath [18] | 860.521 | 2,62 × 10⁹ |
| peS2o [16] | 5.526.331 de 38.972.211 (14,18%) | 14,60 × 10⁹ |
| Total | 7.222.231 | 27,75 × 10⁹ |

O total situa-se na metade superior da faixa de 15 a 30 bilhões de tokens estimada como necessária, a custo monetário nulo. A Seção 4.3 mostra que o volume, entretanto, não é o fator limitante.

### 4.2 Classificação de domínio

O classificador final atinge acurácia de 0,954, com taxa de falso positivo entre 2,4% e 3,7% em cada um dos quatro domínios negativos representados no treinamento.

**Tabela 3.** Avaliação por exclusão de domínio. "FP interno" é a taxa de falso positivo em domínios presentes no treinamento; "FP externo", no domínio omitido.

| Domínio omitido | FP interno | FP externo | Razão | Precisão externa |
|---|---|---|---|---|
| Ciência da computação | 3,7% | 9,8% | 2,6 | 0,907 |
| Economia | 3,3% | 8,5% | 2,6 | 0,988 |
| Matemática | 2,4% | 35,4% | 14,6 | 0,731 |
| Biologia quantitativa | 3,1% | 31,2% | 10,2 | 0,907 |
| Estatística | 3,0% | 2,9% | 1,0 | — |

O resultado que orienta o desenho é a assimetria entre domínios: a degradação sob domínio não visto não é uniforme, mas função da proximidade. Sem negativos de matemática, a filtragem do OpenWebMath — corpus majoritariamente matemático — admitiria cerca de 35% de conteúdo não pertinente. Elevar o limiar de decisão de 0,5 para 0,999 reduz essa taxa a 10,0% e estanca, porque a perda `modified_huber` satura as probabilidades.

Duas previsões formuladas antes da medição foram refutadas. A primeira supunha que negativos de matemática seriam pouco informativos, uma vez que o rótulo depende de uma marcação de listagem cruzada ausente do texto; treinando exclusivamente sobre artigos de primária matemática, a acurácia é de 0,830 contra 0,5 do acaso, o que indica que o sinal está no texto. A segunda supunha que estatística seria um domínio confundível, pelo vocabulário compartilhado com física estatística; a razão medida é 1,0, e o domínio foi excluído do treinamento final para preservar cota de amostragem para os domínios efetivamente confundíveis.

### 4.3 Integridade da notação matemática

**Tabela 4.** Indicadores de notação matemática por fatia de corpus, 3.000 documentos sorteados por fatia.

| Fatia | Caracteres/doc. | LaTeX (%) | Matemática em linha (%) | Ambiente de equação (%) | Sequências/doc. |
|---|---|---|---|---|---|
| Resumos do arXiv (referência) | 1.123 | 21,8 | 26,9 | 0,0 | 0,9 |
| RedPajama, arXiv (fonte LaTeX) | 49.212 | 100,0 | 99,6 | 84,9 | 1.158,3 |
| OpenWebMath (web) | 16.009 | 78,5 | 85,2 | 5,1 | 57,7 |
| peS2o, texto integral (de PDF) | 28.605 | 16,3 | 18,2 | 0,0 | 1,0 |

A fatia de texto integral extraída de PDF apresenta 28,6 mil caracteres por documento de Física sem qualquer equação em display. A inspeção de trechos confirma remoção, e não codificação alternativa: em um documento, a equação situada entre "if:" e "where" foi eliminada, restando o dois-pontos sem referente; em outro, `p^*` aparece como `p *` e `T_{eff}` como `T eff`, com os índices achatados em símbolos isolados pela extração.

O achado metodológico refere-se ao diagnóstico, não ao corpus. O indicador intuitivo para detectar remoção de equação é o operador órfão — um sinal de igualdade ou desigualdade sem operando de um dos lados. Ele apresenta comportamento invertido: acusa 81,3% dos documentos da fatia construída a partir do fonte LaTeX, contra 3,0% da referência, em ambos os casos com notação íntegra à inspeção. A causa é elementar: em `$Z_{\rm max}$ = 15 kpc`, o caractere que antecede o sinal de igualdade é um delimitador de matemática, não aceito como operando pela expressão regular. Um indicador que dispara com maior frequência no corpus íntegro do que no degradado é inutilizável. O discriminante válido é a presença do ambiente de equação, com separação de 84,9 contra 0,0 pontos percentuais entre duas fatias do mesmo domínio.

Um segundo experimento quantifica a perda da fatia construída a partir do fonte LaTeX em relação ao fonte original. Sobre 298 artigos de Física, a fração de equações ausentes é de 16,6%, com intervalo de confiança de 95% em [12,9%; 20,8%]. A decomposição entre ausência e discordância de notação é necessária: 97% das equações que não casaram utilizavam macros definidas pelo autor, e contabilizá-las como perda superestima a degradação. Uma medição anterior, restrita a 103 artigos, produzia intervalo [9,9%; 19,1%], que cruza o limiar de decisão de 10% e não sustenta conclusão.

### 4.4 Tokenização

**Tabela 5.** Tokenizadores treinados sobre 200.000 resumos do arXiv e avaliados em 5.000 reservados. "Fertilidade" é a razão entre tokens e palavras; as colunas de razão são relativas ao tokenizador de referência (variante F). "Regras" indica a presença das regras de pré-tokenização propostas. "LaTeX unitário" conta, entre três sequências LaTeX de teste, quantas são representadas por um único token.

| Variante | Algoritmo | Vocabulário | Regras | Fertilidade | Razão | Tokens/eq. | Razão | LaTeX unitário |
|---|---|---|---|---|---|---|---|---|
| A | BPE | 40.960 | sim | 0,9620 | 0,674 | 7,31 | 0,732 | 2/3 |
| B | Unigrama | 40.960 | sim | 0,9816 | 0,688 | 8,36 | 0,837 | 0/3 |
| C | BPE | 32.768 | sim | 0,9973 | 0,699 | 7,49 | 0,750 | 2/3 |
| D | BPE | 65.536 | sim | 0,8966 | 0,629 | 6,97 | 0,698 | 2/3 |
| E | BPE | 40.960 | não | 1,3245 | 0,929 | 9,46 | 0,947 | 0/3 |
| F | referência | 151.643 | — | 1,4263 | 1,000 | 9,99 | 1,000 | 0/3 |

Todas as variantes preservam o texto na decodificação e mantêm razão de fertilidade abaixo de 1,25 nas 35 subáreas medidas.

Três resultados merecem registro. Primeiro, as regras de pré-tokenização propostas são a variável de maior efeito: as variantes A e E diferem apenas por elas, e A produz 27% menos tokens em prosa e 23% menos em equações; E é a única variante que não atinge a meta de fertilidade em prosa.

Segundo, e contrariando a expectativa derivada de Bostrom e Durrett [21], a codificação por pares de bytes supera o modelo de unigramas em texto de Física, com margem seis vezes maior em equações (13%) do que em prosa (2%). O mecanismo é observável: o modelo de unigramas não reteve nenhuma das três sequências LaTeX de teste como token único, enquanto a codificação por pares de bytes reteve duas. A poda iterativa do primeiro remove essas unidades; a fusão incremental do segundo as preserva. O resultado é restrito a resumos e ao tamanho de vocabulário testado, e não se transfere automaticamente a texto integral.

Terceiro, o vocabulário de 40.960 não é ótimo pela métrica intrínseca: a ordenação é monotônica em toda a faixa testada, com 65.536 superando 40.960 e este superando 32.768. A métrica intrínseca, contudo, não captura o compromisso relevante: a matriz de embedding corresponde a 16% dos parâmetros do modelo de 150 M com vocabulário de 40.960, e a cerca de 24% com 65.536, isto é, parâmetros deslocados de camadas para embedding. Nenhum número desta seção decide a escolha; o que eles justificam é a inclusão do vocabulário maior na avaliação extrínseca.

Um resultado negativo restringe a leitura dos três anteriores: a meta de fertilidade em equações, fixada em 0,65 vez o tokenizador de referência, não é atingida por nenhuma variante, sendo 0,698 o melhor valor. A causa provável é o corpus de avaliação — resumos contêm matemática em linha curta, e o alvo foi calibrado supondo texto integral com equações em display, precisamente o que a Seção 4.3 mostra ausente dos resumos.

### 4.5 Recuperação densa: resultado provisório e defeito de protocolo

**Tabela 6.** Recuperação de documento citado, 2.000 candidatos, protocolo idêntico para todos os modelos. Valores reportados como históricos; ver o texto.

| Modelo | Parâmetros | nDCG@10 | recall@1 | recall@10 |
|---|---|---|---|---|
| Bi-encoder ajustado (este trabalho) | 23 M | 0,4657 | 0,262 | 0,708 |
| GTE-large [10] (geral) | 335 M | 0,4628 | 0,278 | 0,677 |
| Bi-encoder ajustado, execução local | 23 M | 0,4579 | 0,254 | 0,700 |
| PhysBERT [9] (domínio) | 109 M | 0,2752 | 0,146 | 0,425 |
| SciBERT [6] (base) | 110 M | 0,2074 | 0,109 | 0,328 |

A leitura registrada à época da medição tem duas partes. A superioridade sobre o modelo de mesmo domínio é ampla e significativa: 0,190 de nDCG@10 sobre o PhysBERT, com teste pareado a p < 10⁻⁴. A comparação com o modelo geral, ao contrário, é de paridade: a diferença de 0,0029 no nDCG@10 não é sustentada pelo teste pareado em recall@1, que apresenta 196 discordantes contra 165 em favor do modelo geral, com p = 0,114. A afirmação defensável é de eficiência paramétrica — um modelo de 23 M iguala um de 335 M, com 1/14,8 dos parâmetros — e não de superioridade.

Uma auditoria posterior invalidou o protocolo que produziu esta tabela. A avaliação é feita dentro do lote: calcula-se a matriz de similaridade entre âncoras e positivos, e a resposta correta da linha *i* é a coluna *i*. A tarefa só é bem posta se as colunas forem distintas, e duas condições violavam isso. A amostra era tomada pelas primeiras *n* linhas do arquivo, cuja ordem não é neutra — a mediana do comprimento da âncora decresce de aproximadamente 1.180 para 870 caracteres do início ao fim, e o primeiro bloco de 2.000 situa-se no percentil 94. Mais grave, esse bloco continha apenas 1.147 textos positivos distintos, com 62% das linhas ocupadas por um positivo repetido e um deles ocorrendo 28 vezes; textos idênticos produzem similaridade idêntica, e o desempate da ordenação é arbitrário. Adicionalmente, âncoras repetidas compartilham uma única ordenação, de modo que no máximo uma delas pode ocupar a primeira posição.

**Tabela 7.** Desempenho de um recuperador perfeito sob cada esquema de amostragem do conjunto de avaliação.

| Esquema de amostragem | recall@1 | nDCG@10 |
|---|---|---|
| Primeiras 2.000 linhas (usado na Tabela 6) | 0,5235 | 0,7562 |
| Amostra aleatória de 2.000 | 0,9364 | 0,9761 |
| Amostra aleatória com deduplicação por texto | 1,0000 | 1,0000 |

O recall@1 de 0,2620 da Tabela 6 foi obtido contra um teto de 0,5235. A comparação entre modelos permanece justa, pois todos foram submetidos ao mesmo protocolo, mas a margem não sobrevive: a paridade discutida acima decide-se em 0,0029 de nDCG@10, e o ruído de desempate arbitrário em 62% dos itens é de magnitude superior. Pelo mesmo motivo, parte dos pares discordantes do teste de McNemar reflete desempate arbitrário, e não discordância entre modelos.

O mesmo defeito afeta a amostragem do conjunto de treinamento. Sobre os 6.564.111 pares disponíveis, com 667.304 documentos citados distintos, tomar os primeiros 400.000 pares fornece 17.844 documentos distintos, enquanto uma amostra aleatória de mesmo tamanho fornece 191.300 — fator de 10,7 em diversidade, ao mesmo custo de computação. O modelo da Tabela 6 foi treinado sob a primeira condição.

Em consequência, os valores da Tabela 6 são reportados como históricos. A correção implementada consiste em amostragem com semente, deduplicação por texto e uma verificação que interrompe a avaliação caso o teto do conjunto não seja 1,0, com o teto registrado no artefato de saída ao lado da métrica. A remedição está em curso e exige, além da reavaliação, um novo treinamento.

### 4.6 Escala do treinamento contrastivo

**Tabela 8.** Efeito de duas variações de escala, medidas sob o protocolo da Tabela 6 e sujeitas à mesma ressalva.

| Configuração | nDCG@10 | Diferença | McNemar | Discordantes |
|---|---|---|---|---|
| 400 mil pares, 127 negativos (referência) | 0,4579 | — | — | — |
| 400 mil pares, 511 negativos [24] | 0,4486 | −0,0093 | p = 0,636 | 218 |
| 1,5 milhão de pares, 127 negativos | 0,4520 | −0,0059 | p = 0,950 | 256 |

Nenhuma das duas variações produz ganho detectável, e ambas apresentam estimativa pontual inferior à referência. O enunciado sustentado pelos testes pareados não é de piora, mas de ausência de efeito mensurável nesta escala.

Quanto ao número de negativos, o contraste entre 127 e 511 cobre a faixa entre a média e o extremo superior da prática corrente. O resultado não contradiz o ganho medido de 7 para 127 negativos, mas restringe sua extrapolação: a curva satura entre 127 e 511.

Quanto ao volume de dados, o treinamento com 1,5 milhão de pares foi interrompido em 38% por estabilização da métrica de validação. À luz da Seção 4.5, essa interrupção deve ser reconsiderada: os pares adicionais provinham de apenas 67.232 documentos distintos, e o esgotamento de um conjunto pequeno de documentos é indistinguível, pela métrica observada, de saturação de dados.

### 4.7 Sistema híbrido

**Tabela 9.** Composição de recuperadores. Mil consultas, universo de 88.807 documentos citados distintos, profundidade 50. O teste pareado compara cada sistema à fusão.

| Sistema | recall@1 | recall@10 | recall@50 | nDCG@10 | McNemar (k = 10) |
|---|---|---|---|---|---|
| BM25 [25] | 0,067 | 0,236 | 0,401 | 0,1399 | p = 0,00073 |
| Bi-encoder | 0,055 | 0,233 | 0,428 | 0,1327 | p = 0,00018 |
| Fusão recíproca de postos [27] | 0,068 | 0,271 | 0,446 | 0,1584 | — |
| Fusão + cross-encoder (base MiniLM) | 0,064 | 0,254 | 0,446 | 0,1493 | p = 0,118 |

A fusão supera ambos os recuperadores isolados com significância. O cross-encoder inicializado a partir da mesma base do recuperador denso não produz ganho mensurável.

O trajeto até esse resultado contém o modo de falha mais informativo do trabalho. A mineração inicial de negativos difíceis tomava os *K* primeiros resultados do índice denso, excluída a citação verdadeira. Como todo negativo assim obtido está no topo do índice, enquanto o positivo frequentemente não está, o critério de rotulagem torna-se predizível pela posição no recuperador, e o modelo aprende a regra mais simples que separa as classes.

**Tabela 10.** Correlação de Spearman entre a posição do candidato na fusão e o escore atribuído pelo cross-encoder, por estratégia de mineração de negativos. Posição menor indica melhor colocação, de modo que o sinal desejado é negativo.

| Estratégia de mineração | Spearman | Consultas com correlação positiva |
|---|---|---|
| K primeiros do índice denso, menos o positivo | +0,179 | 83% |
| Amostragem dos 50 primeiros da fusão real | −0,466 | 0% |

O modelo treinado sob a primeira estratégia prefere sistematicamente a cauda do recuperador e atinge nDCG@10 de 0,0179, inferior à ordenação aleatória. Corrigida a distribuição do grupo de treinamento para coincidir com a da avaliação, a correlação inverte-se e o modelo passa a concordar com o recuperador — e é precisamente essa concordância que explica a ausência de ganho: um reordenador que reproduz a ordenação já produzida pela fusão não acrescenta informação.

O limite superior disponível é amplo: o recall@50 de 0,446 corresponde ao nDCG@10 que um reordenador perfeito alcançaria, contra 0,1584 observado — fator de 2,8 dentro dos candidatos já recuperados.

Seis hipóteses alternativas foram testadas e descartadas no diagnóstico: erro de inferência do backend de atenção (diferença máxima de 4 × 10⁻⁶ contra a implementação de referência, com ordenação idêntica); negativos provenientes de população distinta (todos os minerados são documentos citados); grau de citação como atalho (o positivo tem grau médio 113,6 contra 8,5 dos negativos, mas a correlação de Spearman entre grau e escore é +0,056, com p = 0,17); atalho de formato de superfície (14 atributos, melhor AUC de 0,575); artefato de desempate na ordenação (permutar a posição do positivo produz valores idênticos); e insensibilidade do modelo ao texto da consulta, hipótese formulada sobre 16 documentos e refutada com 457, com efeito medido de +0,143 ± 0,059.

### 4.8 Efeito do corpus de pré-treinamento da base do reordenador

O diagnóstico da Seção 4.7 produz uma predição falseável: se o reordenador não acrescenta informação por redundância com o recuperador, então uma base pré-treinada em corpus distinto deveria acrescentá-la. A predição foi testada com regra de decisão registrada antes da coleta — teste de McNemar em k = 10 contra a fusão da mesma execução, limiar de Bonferroni de 0,025 por serem duas variantes, 2.000 consultas — e com as quatro leituras possíveis escritas de antemão, inclusive a de nenhuma variante superar o controle.

**Tabela 11.** Cross-encoders idênticos exceto pela base. Duas mil consultas, profundidade 50, fusão de referência com nDCG@10 de 0,1576 nas três execuções, teto do conjunto de 0,4495. "Acerto@1" refere-se ao grupo de 8 candidatos do treinamento; "Diferença" é em relação à fusão.

| Base | Parâmetros | Acerto@1 | nDCG@10 com reordenação | Diferença | Discordantes | p (k = 10) |
|---|---|---|---|---|---|---|
| MiniLM-L6 (a mesma do recuperador) | 23 M | 0,498 | 0,1483 | −0,0093 | 229 | 0,1458 |
| GTE-base (geral forte) [10] | 109 M | 0,510 | 0,1530 | −0,0046 | 220 | 0,6371 |
| PhysBERT (Física) [9] | 109 M | 0,566 | 0,1666 | +0,0090 | 237 | 0,0062 |

Apenas a base pré-treinada no domínio supera a fusão. A leitura correspondente, registrada previamente, é de que o mecanismo é conhecimento de domínio, e não diversidade de base nem capacidade. O que sustenta a interpretação causal é o controle: GTE-base e PhysBERT compartilham número de parâmetros, arquitetura, os seis hiperparâmetros de treinamento, os dados, a semente e o protocolo de avaliação, diferindo exclusivamente no corpus de pré-treinamento.

**Tabela 12.** Composição completa contra a fusão isolada.

| Métrica | Fusão | Fusão + reordenação de domínio |
|---|---|---|
| recall@1 | 0,0700 | 0,0690 |
| recall@10 | 0,2675 | 0,2890 |
| recall@50 | 0,4495 | 0,4495 |
| nDCG@10 | 0,1576 | 0,1666 |

No teste pareado em k = 10, a fusão vence em 97 casos e o sistema com reordenação em 140, sobre 237 discordantes, com p = 0,0062, abaixo do limiar de Bonferroni. Em k = 1 há empate (p = 0,930): o ganho concentra-se na cauda das dez primeiras posições, comportamento esperado de reordenação.

Duas ressalvas de execução são declaradas. A combinação das duas execuções é legítima porque o braço de controle produziu resultados idênticos campo a campo em ambas, com p = 0,14584 sobre 229 discordantes nos dois casos; sem essa verificação, a comparação seria entre protocolos e não entre bases. E a primeira execução da variante geral falhou por defeito de implementação, e não do modelo: os pesos distribuídos em precisão de 16 bits eram carregados nessa precisão, e o escalonador de gradiente exige pesos-mestres em 32 bits. O defeito permaneceu latente porque as outras duas bases são distribuídas em 32 bits.

### 4.9 Pré-treinamento do encoder: estado

O objetivo de mascaramento consciente de equações é a única contribuição de objetivo de treinamento proposta, e sua ablação não foi executada. Reporta-se aqui o que foi verificado.

O corpus de pré-treinamento está preparado: 2.001.270.262 tokens de 143.810 documentos, em 244.295 sequências de 8.192 tokens, com 9 de 44 partições sorteadas por semente registrada. A conferência sobre 300 sequências amostradas do binário efetivamente lido pelo treinamento indica 37,8% de tokens matemáticos contra 38,8% no corpus bruto, 25,0% em display contra 25,2%, taxa efetiva de mascaramento de 0,3000 contra 0,3000 requerida, e fração tratada de 0,903.

A restrição a equações em display, e não a toda notação matemática, decorre de medição: em 120 documentos, as equações em display têm mediana de 79 tokens, e as sequências em linha, mediana de 7 — o que corresponde a uma variável isolada, e não a uma equação. Tratar toda notação faria a ablação comparar condições substancialmente equivalentes.

A correção do laço de treinamento é verificada pela perda inicial: um modelo de linguagem mascarada não treinado prediz distribuição uniforme, de modo que a entropia cruzada inicial deve igualar o logaritmo do tamanho do vocabulário. O valor medido é 10,7343, contra ln(40.960) = 10,6204. Esse é o único indicador que distingue um mascaramento correto de um que apague os alvos ou os derive da entrada já mascarada — condições que não produzem nenhum outro sintoma observável.

Um defeito no marcador de equações ilustra o ponto. A expressão regular de detecção usava âncora de início de cadeia em conjunto com casamento a partir de posição arbitrária; a âncora continua referindo-se ao início real da cadeia, de modo que nenhuma equação iniciada após o primeiro caractere era detectada. O efeito medido foi de 0 documentos tratados em 120, contra 91,7% de documentos contendo equações em display segundo um segundo instrumento, e foi a discordância entre os dois instrumentos que localizou o erro. Sem o contador de fração tratada, a ablação teria sido executada comparando duas condições aleatórias e reportado ausência de efeito.

A avaliação do encoder não existe, e é o fator limitante atual. As três avaliações previstas — recuperação de Física, modelagem de linguagem mascarada em texto denso em equações e sondagem de estrutura tensorial — não foram implementadas. Uma perda de treinamento baixa não constitui veredito sobre a hipótese.

---

## 5. Discussão

### 5.1 Domínio, e não capacidade nem diversidade

O resultado da Seção 4.8 é assimétrico de modo não previsto. O modelo pré-treinado em Física é o pior recuperador entre os avaliados na Tabela 6, com nDCG@10 de 0,2752 contra 0,4657, e é a melhor base de reordenação entre as três testadas. Pré-treinamento de domínio, nas condições medidas, não produziu um bi-encoder competitivo e produziu um cross-encoder competitivo.

A implicação operacional é direta: para reordenação, convém partir de um modelo do domínio; para recuperação de primeira etapa, convém ajustar um modelo geral pequeno. Não dispomos de explicação mecanística para a assimetria. A hipótese de menor custo de teste é que o cross-encoder pode explorar interação termo a termo entre consulta e documento, regime em que vocabulário de domínio é diretamente utilizável, enquanto o bi-encoder deve comprimir o documento em um vetor antes de observar a consulta. Trata-se de especulação até que seja medida.

O resultado também qualifica a prática, comum em trabalhos de domínio, de avaliar um encoder especializado apenas como recuperador. Sob essa avaliação isolada, o PhysBERT seria descartado; seu valor no sistema aparece somente em outro papel.

### 5.2 Vieses de amostragem em conjuntos derivados de grafos de citação

Uma auditoria identificou seis ocorrências independentes de amostragem por posição em dados ordenados, todas no mesmo repositório.

**Tabela 13.** Ocorrências identificadas, com o efeito medido de cada uma.

| Local | Efeito medido | Consequência |
|---|---|---|
| Amostra de revisão do corpus de texto integral | 400 documentos, todos nos 2 primeiros de 277 arquivos — 0,67% do corpus, exclusivamente resumos | Invalidou a amostra destinada a estimar contaminação |
| Avaliação sobre arquivo agrupado | 500 linhas correspondiam a 35 documentos; intervalo de 95% de ±0,159 | Duas conclusões mutuamente contraditórias |
| Divisão treino/validação por posição | 49,6% das âncoras da validação presentes no treino | nDCG@10 de 0,139 para 0,020 sobre documentos inéditos |
| Seleção das primeiras 8 de 44 partições | Partições omitidas continham 57% mais equações em display | Viés no corpus de pré-treinamento, detectado antes do uso |
| Instrumento de conferência do item anterior | 200 linhas provenientes de um único arquivo | Reproduziu o valor enviesado que existia para refutar |
| Conjunto de avaliação principal | Teto de nDCG@10 de 0,7562 | Invalidou a medição do critério de aceitação (Seção 4.5) |

Duas observações parecem generalizáveis. A primeira é que o defeito é invisível por construção: nenhuma das seis ocorrências gera exceção, nenhuma se manifesta na função de perda, e todas produzem valores dentro da faixa esperada — a terceira eleva a métrica. A quinta é particularmente instrutiva, por haver ocorrido dentro do instrumento escrito para detectar a quarta.

A segunda é que a correção adequada não é a substituição pontual do esquema de amostragem. O caminho de código de reordenação já empregava amostragem aleatória, com a justificativa documentada em comentário, e essa justificativa não se propagou ao caminho de embedding. A correção adotada é uma verificação que interrompe a execução quando o teto do conjunto de avaliação difere de 1,0, com o teto registrado no artefato de saída — isto é, uma propriedade verificada a cada execução, e não uma convenção documentada.

Trabalhos que derivam simultaneamente o rótulo positivo, o critério de negativo e a verdade de avaliação de um mesmo grafo estão estruturalmente expostos a esta classe de defeito, por não disporem de uma segunda fonte contra a qual conferir. Recomenda-se reportar, ao lado de qualquer métrica agregada por grupo, o número de grupos distintos efetivamente medidos e o intervalo que ele implica.

### 5.3 Implicações práticas

Três padrões emergiram, nenhum relativo a arquitetura.

Medir a fonte antes de adquiri-la. A fatia de corpus que viabiliza o objetivo de treinamento específico de domínio já se encontrava em disco havia dezessete dias quando se recomendou a aquisição paga de acesso ao fonte do arXiv como pré-requisito. O que faltava não era dado nem orçamento, mas a medição da Tabela 4, de custo desprezível.

Medir a vazão real antes de planejar. A vazão medida em acelerador T4 foi de 181,6 pares por segundo, contra 20 a 26 na GPU local — aproximadamente 69.700 contra 7.700 tokens por segundo. Isso converte um pré-treinamento de 20 bilhões de tokens de 37 dias para cerca de 80 horas, alterando a viabilidade, e não apenas o cronograma.

Empregar comparação pareada. Com 256 candidatos, o erro padrão é de ±0,031 e a margem mínima detectável, de aproximadamente 0,061, contra uma diferença de interesse de 0,004; nem 4.000 candidatos alterariam essa conclusão. O teste pareado sobre os mesmos itens é o que torna a diferença acessível, e é indispensável em regimes de baixo orçamento, nos quais o tamanho do conjunto de avaliação é limitado.

---

## 6. Limitações

Um domínio, um modelo base e um grafo de citação. Física, MiniLM-L6 e OpenAlex. Nada aqui estabelece que as taxas medidas se transfiram a outros domínios ou fontes.

O conjunto de avaliação é próprio e não dispõe de julgamento humano de relevância. A definição operacional adotada — relevante equivale a citado — é uma aproximação com viés conhecido: favorece artigos citáveis e áreas com cultura de citação densa, e não captura relevância não citada. Os valores de nDCG deste trabalho não são comparáveis aos de benchmarks com anotação humana.

Os resultados das Tabelas 6 e 8 são históricos, pelo defeito de protocolo da Seção 4.5, e a remedição não está concluída.

A hipótese de mascaramento consciente de equações não foi testada. Código e dados estão prontos; a ablação não foi executada e a avaliação correspondente não existe.

A taxa de contaminação do corpus filtrado não está medida. As taxas de falso positivo que justificam o limiar de decisão foram medidas sobre resumos do arXiv, ao passo que o corpus filtrado é de texto integral, com outra distribuição. Uma amostra estratificada de 200 resumos e 200 textos integrais está preparada, e o julgamento humano está pendente.

A Seção 4.7 não mede o falso negativo residual. O filtro de co-citação remove uma classe nomeável de falso negativo — 9,1% dos negativos minerados eram co-citados com o positivo —, mas o tamanho do residual permanece desconhecido.

A taxa de aprendizado não foi reajustada entre escalas de modelo no experimento da Seção 4.8, o que constitui variável não controlada, declarada antes da observação do resultado.

Os parâmetros das etapas já executadas do pipeline são reconstruídos a partir do código, e não capturados durante a execução.

---

## 7. Conclusão

Este trabalho reporta a construção de um sistema de recuperação de literatura de Física sob orçamento de computação nulo, com resultados positivos, negativos e em aberto discriminados.

Estão estabelecidos: um corpus de 27,75 bilhões de tokens e um índice de metadados de 1,59 milhão de registros, construídos a custo nulo e verificáveis por uma cadeia de hashes; um classificador de domínio com acurácia de 0,954 e taxa de falso positivo caracterizada por proximidade de domínio; um discriminante válido para integridade de notação matemática em corpora; evidência empírica de que a codificação por pares de bytes supera o modelo de unigramas em texto de Física; e a demonstração, com controle de tamanho e arquitetura, de que o corpus de pré-treinamento da base determina se um cross-encoder acrescenta informação a uma fusão de recuperadores — com a assimetria de que o melhor modelo para reordenar é o pior para recuperar.

Permanecem em aberto os resultados de recuperação densa, cujo protocolo apresentava teto e cuja remedição está em curso, e a hipótese de mascaramento consciente de equações, cujo teste depende da construção da avaliação correspondente.

O resultado metodológico de maior alcance é o da Seção 5.2. Seis ocorrências independentes da mesma falha de amostragem, em um único repositório, sugerem que conjuntos derivados de grafos de citação exigem verificação ativa da unidade de amostragem, e não apenas convenção documentada.

---

## Disponibilidade de dados e código

O código e a documentação de projeto são públicos sob licença permissiva. Cada etapa do pipeline registra manifesto com hashes das entradas e saídas, parâmetros e identificador de commit, o que permite reconstrução a partir das fontes públicas.

O corpus não acompanha o código. Os registros do arXiv seguem a licença de cada submissão, e a licença padrão concede direito de distribuição ao arXiv, não a terceiros; a fração medida como redistribuível é de 14,8%.

**Nota de autoria.** O trabalho experimental, as decisões de projeto e a implementação são do autor. A redação deste manuscrito contou com assistência de um modelo de linguagem (Claude, Anthropic) a partir dos artefatos de medição do repositório; o autor revisou o texto e responde por ele. As referências a seguir foram conferidas contra as respectivas fontes.

---

## Referências

[1] Hoffmann, J.; Borgeaud, S.; Mensch, A.; et al. Training Compute-Optimal Large Language Models. *Advances in Neural Information Processing Systems 35* (NeurIPS 2022), 2022.

[2] Taylor, R.; Kardas, M.; Cucurull, G.; Scialom, T.; Hartshorn, A.; Saravia, E.; Poulton, A.; Kerkez, V.; Stojnic, R. Galactica: A Large Language Model for Science. arXiv:2211.09085, 2022.

[3] Lewkowycz, A.; Andreassen, A.; Dohan, D.; Dyer, E.; Michalewski, H.; Ramasesh, V.; Slone, A.; Anil, C.; Schlag, I.; Gutman-Solo, T.; Wu, Y.; Neyshabur, B.; Gur-Ari, G.; Misra, V. Solving Quantitative Reasoning Problems with Language Models. *Advances in Neural Information Processing Systems 35* (NeurIPS 2022), 2022.

[4] Azerbayev, Z.; Schoelkopf, H.; Paster, K.; Dos Santos, M.; McAleer, S.; Jiang, A. Q.; Deng, J.; Biderman, S.; Welleck, S. Llemma: An Open Language Model for Mathematics. *International Conference on Learning Representations* (ICLR 2024), 2024.

[5] Shao, Z.; Wang, P.; Zhu, Q.; et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.

[6] Beltagy, I.; Lo, K.; Cohan, A. SciBERT: A Pretrained Language Model for Scientific Text. *Proceedings of EMNLP-IJCNLP 2019*, 2019.

[7] Cohan, A.; Feldman, S.; Beltagy, I.; Downey, D.; Weld, D. S. SPECTER: Document-level Representation Learning using Citation-informed Transformers. *Proceedings of ACL 2020*, p. 2270–2282, 2020.

[8] Bhattacharjee, B.; Trivedi, A.; Muraoka, M.; et al. INDUS: Effective and Efficient Language Models for Scientific Applications. *Proceedings of EMNLP 2024, Industry Track*, 2024. arXiv:2405.10725.

[9] Hellert, T.; Montenegro, J.; Pollastro, A. PhysBERT: A text embedding model for physics scientific literature. *APL Machine Learning*, v. 2, n. 4, art. 046105, 2024. DOI 10.1063/5.0238090.

[10] Li, Z.; Zhang, X.; Zhang, Y.; Long, D.; Xie, P.; Zhang, M. Towards General Text Embeddings with Multi-stage Contrastive Learning. arXiv:2308.03281, 2023.

[11] Reimers, N.; Gurevych, I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *Proceedings of EMNLP-IJCNLP 2019*, p. 3982–3992, 2019.

[12] Wang, W.; Wei, F.; Dong, L.; Bao, H.; Yang, N.; Zhou, M. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *Advances in Neural Information Processing Systems 33* (NeurIPS 2020), 2020.

[13] Karpukhin, V.; Oguz, B.; Min, S.; Lewis, P.; Wu, L.; Edunov, S.; Chen, D.; Yih, W. Dense Passage Retrieval for Open-Domain Question Answering. *Proceedings of EMNLP 2020*, p. 6769–6781, 2020.

[14] Xiong, L.; Xiong, C.; Li, Y.; Tang, K.-F.; Liu, J.; Bennett, P. N.; Ahmed, J.; Overwijk, A. Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. *International Conference on Learning Representations* (ICLR 2021), 2021.

[15] Qu, Y.; Ding, Y.; Liu, J.; Liu, K.; Ren, R.; Zhao, W. X.; Dong, D.; Wu, H.; Wang, H. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering. *Proceedings of NAACL-HLT 2021*, 2021.

[16] Soldaini, L.; Lo, K. peS2o (Pretraining Efficiently on S2ORC) Dataset. Relatório técnico, Allen Institute for AI, 2023. Licença ODC-By.

[17] Weber, M.; Fu, D.; Anthony, Q.; et al. RedPajama: an Open Dataset for Training Large Language Models. *Advances in Neural Information Processing Systems 37, Datasets and Benchmarks Track* (NeurIPS 2024), 2024. arXiv:2411.12372.

[18] Paster, K.; Dos Santos, M.; Azerbayev, Z.; Ba, J. OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text. *International Conference on Learning Representations* (ICLR 2024), 2024.

[19] Sennrich, R.; Haddow, B.; Birch, A. Neural Machine Translation of Rare Words with Subword Units. *Proceedings of ACL 2016*, v. 1, p. 1715–1725, 2016.

[20] Kudo, T. Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. *Proceedings of ACL 2018*, v. 1, p. 66–75, 2018.

[21] Bostrom, K.; Durrett, G. Byte Pair Encoding is Suboptimal for Language Model Pretraining. *Findings of EMNLP 2020*, p. 4617–4624, 2020.

[22] Priem, J.; Piwowar, H.; Orr, R. OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv:2205.01833, 2022.

[23] van den Oord, A.; Li, Y.; Vinyals, O. Representation Learning with Contrastive Predictive Coding. arXiv:1807.03748, 2018.

[24] Gao, L.; Zhang, Y.; Han, J.; Callan, J. Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup. *Proceedings of the 6th Workshop on Representation Learning for NLP* (RepL4NLP 2021), p. 316–321, 2021.

[25] Robertson, S.; Zaragoza, H. The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, v. 3, n. 4, 2009.

[26] Warner, B.; Chaffin, A.; Clavié, B.; et al. Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. *Proceedings of ACL 2025*, 2025. arXiv:2412.13663.

[27] Cormack, G. V.; Clarke, C. L. A.; Büttcher, S. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *Proceedings of the 32nd International ACM SIGIR Conference*, p. 758–759, 2009.

[28] McNemar, Q. Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, v. 12, n. 2, p. 153–157, 1947.

[29] Wilson, E. B. Probable Inference, the Law of Succession, and Statistical Inference. *Journal of the American Statistical Association*, v. 22, n. 158, p. 209–212, 1927.
