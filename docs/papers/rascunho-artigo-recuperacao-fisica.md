# Construção de um sistema de recuperação de literatura de Física sob restrição severa de computação: corpus, tokenização, representação e verificação

Vinicius Sanchez

*Pesquisa independente*

*Versão 0.5 — 23 de setembro de 2026*

---

## Resumo

> A escassez de texto em domínios científicos estreitos limita a aplicação direta de leis de escala a modelos de fundação especializados. Estimativas do corpus de Física legalmente adquirível situam-se entre 15 e 30 bilhões de tokens após filtragem, contra os cerca de 160 bilhões que um modelo de 8 bilhões de parâmetros exigiria sob a relação de computação ótima de Hoffmann et al. Este trabalho descreve a construção, sob orçamento de computação nulo — uma GPU de consumo de 8 GB e cotas gratuitas de aceleradores em nuvem —, de um sistema de recuperação de literatura de Física, e reporta os resultados positivos, negativos e em aberto. Foram montados um índice de metadados de 1.595.422 registros do arXiv associado a 4.613.751 obras do OpenAlex, com taxa de casamento de 99,1%, um classificador de domínio com acurácia de 0,954 e um corpus de Física de 27,75 bilhões de tokens, atestado por uma cadeia de hashes sobre 52,40 GB. Corrigido um defeito de protocolo que impunha teto de 0,7562 ao nDCG@10, um bi-encoder de 23 milhões de parâmetros ajustado por supervisão de citação com 6 milhões de pares sorteados atinge nDCG@10 de 0,6223, contra 0,5788 de um modelo geral de 335 M; trocar a base por um modelo geral de 109 M produz com 400 mil pares o que o volume produziu com 6 milhões (empate pareado, p = 0,908). Entre três cross-encoders idênticos exceto pelo corpus de pré-treinamento da base, apenas o pré-treinado em Física melhora a fusão de recuperadores (p = 0,0062); com o recuperador melhorado, contudo, nenhuma de cinco medições pareadas estabelece ganho do estágio de reordenação, que foi retirado do sistema. Em encoders de 48 M treinados com 0,6 bilhão de tokens, regras de pré-tokenização específicas para LaTeX pioraram a modelagem em 0,047 bit por byte — resultado que apenas o terceiro de três instrumentos de viés conhecido pôde decidir —, e mascarar equações inteiras não melhorou a predição de tokens de equação (diferença das diferenças de −0,0040), mas melhorou a recuperação após ajuste contrastivo em 0,084 de nDCG@10 [0,071; 0,097]. A mesma receita de ajuste aplicada a uma base geral de 150 M sem qualquer pré-treinamento no domínio, contudo, alcança 0,5270 contra 0,4712 do braço tratado. O pré-treinamento continuado dessa base com 0,4 bilhão de tokens de Física e o mesmo objetivo repete o ganho de recuperação onze vezes menor (+0,0076 [+0,0013; +0,0139]), com a medida primária de modelagem nula; e uma segunda base geral, sem pré-treinamento no domínio, supera a primeira em 0,069 — cerca de sete vezes o que o pré-treinamento continuado acrescentou —, o que retira, sob este orçamento, a justificativa tanto para treinar o encoder do zero quanto para pré-treiná-lo continuamente. Num conjunto de recuperação por equação construído do próprio corpus, o bi-encoder de 23 M supera o BM25 justamente no estrato que exige variação notacional (recall@1 de 0,860 contra 0,712), contra a premissa de que a recuperação densa perde o casamento simbólico. Reportam-se ainda o discriminante válido para integridade de notação matemática em corpora, uma auditoria de oito ocorrências independentes de amostragem dependente da ordem dos dados e um mecanismo de detecção de instabilidade cujo acionamento dependia da variável sob teste.

**Palavras-chave:** recuperação de informação científica; supervisão por citação; modelos de embedding de domínio; construção de corpus; tokenização de LaTeX; mascaramento de equações; reprodutibilidade.

---

## 1. Introdução

Modelos de linguagem de propósito geral apresentam, em Física, modos de falha distintos dos observados em tarefas factuais: incoerência dimensional, deriva algébrica ao longo de derivações longas, ausência de redução correta em casos-limite e mistura silenciosa de convenções de sinal e de sistema de unidades. Nenhum desses modos é corrigível apenas pelo acréscimo de texto de domínio, porque nenhum deles é penalizado pelo objetivo de predição do próximo token, que premia plausibilidade local.

A restrição que organiza este trabalho, contudo, é de recurso. Sob a relação de computação ótima estabelecida por Hoffmann et al. [1], um modelo de 8 bilhões de parâmetros requer aproximadamente 160 bilhões de tokens de treinamento. Modelando o funil de aquisição e filtragem estágio a estágio para a literatura de Física legalmente adquirível a custo zero, obtêm-se entre 39 e 73 bilhões de tokens brutos, ou entre 15 e 30 bilhões após deduplicação e triagem de licença — uma escassez de cinco a dez vezes.

A resposta usual a essa restrição é ampliar a aquisição. A adotada aqui é distinta, e é sustentada pela literatura recente: treinamento a partir de inicialização aleatória é justificável apenas onde o dado é excedente — encoders na faixa de 10⁸ parâmetros —, e capacidades acima disso devem ser obtidas por pré-treinamento continuado sobre uma base geral forte. A evidência é consistente. O Galactica [2], única tentativa de larga escala de treinar um modelo científico de fundação a partir do zero, foi retirado de circulação três dias após o lançamento, com alucinação de citações entre os defeitos determinantes. Em contraste, Minerva [3], Llemma [4] e DeepSeekMath [5] foram todos obtidos por pré-treinamento continuado sobre bases gerais, e todos reportam ganhos substanciais em raciocínio quantitativo. A Seção 5.5 reexamina essa premissa no próprio nível de encoders, à luz das ablações da Seção 4.9 — e a conclusão a que chega é mais forte que a da literatura citada: sob este orçamento, nem o treinamento a partir do zero nem o pré-treinamento continuado pagam o que a simples escolha da base geral paga.

Uma segunda restrição, metodológica, decorre da primeira: com orçamento de computação nulo, cada experimento precisa ser justificado por uma medição anterior. Esse regime tem um efeito colateral relevante para os resultados aqui reportados — medições anteriores passam a ser reexaminadas com frequência, e parte substancial dos achados deste trabalho, inclusive os negativos, originou-se desse reexame.

As contribuições são as seguintes.

1. Um procedimento de construção de corpus de Física a custo zero, atestado por uma cadeia de hashes sobre 52,40 GB, e a caracterização quantitativa do funil de filtragem (Seções 3.1 a 3.3, 3.7 e 4.1).
2. Um discriminante medido para integridade de notação matemática em corpora, e a demonstração de que o diagnóstico intuitivo para o mesmo fim satura no corpus íntegro (Seção 4.3).
3. Evidência empírica sobre tokenização de texto em LaTeX, em duas camadas que apontam em sentidos opostos: a métrica intrínseca favorece regras de pré-tokenização específicas, e a modelagem de linguagem, medida em bits por byte por três instrumentos de viés conhecido, as rejeita (Seções 4.4 e 4.9.1).
4. A remedição de um protocolo de recuperação densa que tinha teto, e a separação experimental entre volume, diversidade e base na qualidade do recuperador (Seções 4.5 e 4.6).
5. A separação entre domínio e capacidade como propriedade da base de um cross-encoder de reordenação, com controle de tamanho, arquitetura, dados, semente e hiperparâmetros — e a demonstração de que o ganho do estágio desaparece quando o recuperador melhora (Seções 4.7 e 4.8).
6. Uma ablação do mascaramento de equações inteiras em duas escalas — do zero a 48 M e em pré-treinamento continuado a 150 M —, com o efeito sobre a recuperação encolhendo onze vezes da primeira para a segunda, e a demonstração de que a escolha da base geral vale cerca de sete vezes o pré-treinamento continuado inteiro (Seções 4.9.2 e 4.9.3).
7. Um conjunto de avaliação de recuperação por equação, construído do próprio corpus com estratos de variação notacional, e a refutação da premissa de que recuperação densa perde o casamento simbólico que a busca léxica preserva (Seção 4.10).
8. Uma auditoria de vieses de amostragem em conjuntos derivados de grafos de citação, de instrumentos de medição que, registrados antes da coleta, eram inválidos para a pergunta, e de mecanismos automáticos cujo acionamento dependia da própria variável sob teste (Seções 5.2 a 5.4).

O trabalho não alega um modelo de Física em estado da arte. A Seção 4.5 reporta um resultado de recuperação, retira-o por defeito de protocolo identificado após as medições e reporta a remedição. Os resultados de pré-treinamento da Seção 4.9 são de substitutos de 48 M e de um pré-treinamento continuado de 0,4 bilhão de tokens sobre uma base de 150 M, com uma semente por braço.

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

Em tokenização, a codificação por pares de bytes [19] e o modelo de unigramas [20] são as duas alternativas dominantes. Bostrom e Durrett [21] reportam vantagem do modelo de unigramas em linguagem natural, atribuída a melhor alinhamento morfológico. Não localizamos evidência publicada equivalente para texto em LaTeX, e as Seções 4.4 e 4.9.1 apresentam a nossa.

A comparação extrínseca de tokenizadores por modelos mascarados exige pontuar texto com um modelo que não é autorregressivo. A pseudo-verossimilhança de Salazar et al. [30] mascara um token por vez; Kauf e Ivanova [31] mostram que ela superestima a probabilidade de palavras divididas em vários tokens e propõem mascarar também os tokens seguintes da mesma palavra. A Seção 3.8 descreve como esse viés, e o viés oposto, foram usados para cercar o resultado.

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

**Bi-encoder.** MiniLM-L6 [12] de 23 milhões de parâmetros, ajustado com perda contrastiva InfoNCE [23] sobre pares extraídos do grafo de citação, com lote de 128 e portanto 127 negativos no lote, comprimento máximo de 192 tokens e agregação por média. O conjunto de treinamento contém 6.564.111 pares e 667.304 documentos citados distintos, do qual se sorteiam, com semente registrada, subconjuntos de 400 mil, 1,5 milhão, 3 milhões e 6 milhões de pares. Duas variantes alteram uma única variável cada: 511 negativos no lote por caching de gradiente [24], e a base GTE-base (109 M) [10] no lugar do MiniLM-L6, com os mesmos 400 mil pares e hiperparâmetros.

**Recuperação léxica.** BM25 [25] sobre índice esparso, com os mesmos documentos.

**Cross-encoder de reordenação.** Modelo par a par treinado sobre grupos de 8 candidatos amostrados da distribuição exata da avaliação, isto é, os 50 primeiros candidatos produzidos pela fusão. Três bases foram comparadas mantendo idênticos os seis hiperparâmetros restantes, os dados, a semente e o protocolo: MiniLM-L6 (23 M), GTE-base (109 M) [10] e PhysBERT (109 M) [9].

**Encoder pré-treinado a partir do zero.** A arquitetura planejada é bidirecional, de 150 milhões de parâmetros, com vocabulário de 40.960, comprimento de sequência de 8.192 e objetivo de modelagem de linguagem mascarada a 30% — taxa adotada seguindo os resultados de Warner et al. [26] —, sem predição de sentença seguinte, com agendamento de taxa de aprendizado do tipo *warmup–stable–decay*. A adição específica de domínio, sujeita a ablação, consiste em mascarar uma equação em display inteira em uma fração `p_equacao` dos exemplos que contêm equação elegível, mantendo igual entre os braços o orçamento total de tokens mascarados. Equações cortadas pela janela, no início ou no fim, não são elegíveis.

Os experimentos de pré-treinamento usam substitutos reduzidos da mesma arquitetura: 48 milhões de parâmetros, contexto de 1.024 tokens, lote lógico de 64 sequências e orçamento de 0,6 bilhão de tokens por braço, treinados em acelerador T4 a cerca de 20.500 tokens por segundo, em pouco mais de 8 horas por braço. Nessa escala, a matriz de embedding corresponde a 43,7% dos parâmetros. Os braços são exportados e medidos localmente.

**Pré-treinamento continuado.** O mesmo objetivo, aplicado ao ModernBERT-base (150 M) [26] com o tokenizador dele: 0,4 bilhão de tokens por braço — o que cabe numa sessão de 9 horas —, contexto de 1.024, lote lógico de 64 sequências, taxa de aprendizado de pico de 10⁻⁴, `p_equacao` de 0,0 no controle e 0,6 no tratado, e todos os demais valores iguais, conferidos por comparação da árvore sintática das células. A fatia de treinamento foi retokenizada com o tokenizador da base (423 milhões de tokens), e a de avaliação vem de uma partição que nenhum braço viu. Os dois braços treinam em duas T4 com paralelismo de dados, em que o lote lógico é dividido entre os processos e a máscara de cada micro-passo é função apenas de (semente, índice do micro-passo); um teste confirma que dois processos terminam com os mesmos pesos que um, a menos de 10⁻⁵. A vazão medida foi de 15.200 a 15.900 tokens por segundo nas duas placas, contra 15.600 projetados a partir do custo relativo medido em CPU.

**Bases gerais para o ajuste contrastivo.** Com os mesmos 200 mil pares e os hiperparâmetros da Tabela 10, foram ajustadas, além dos braços de pré-treinamento, duas bases gerais sem pré-treinamento no domínio: o ModernBERT-base e o GTE-base (109 M) [10]. As duas diferem entre si apenas na base.

### 3.5 Protocolo de avaliação

A tarefa de avaliação é a recuperação do documento citado a partir do texto do documento citante. As métricas são recall@k, MRR e nDCG@10.

Dois protocolos são usados. O protocolo de comparação de encoders avalia dentro de um conjunto de 2.000 pares de validação, sorteados com semente e deduplicados por texto, e interrompe a execução se o nDCG@10 de um recuperador perfeito sobre o conjunto diferir de 1,0; o teto é gravado no artefato ao lado da métrica. A necessidade dessa verificação é discutida na Seção 4.5. O protocolo de sistema usa 2.000 consultas contra o universo completo de 88.807 documentos citados distintos, com a composição por fusão recíproca de postos [27] com k = 60.

Braços de uma mesma comparação são medidos na mesma sessão e com o mesmo código. Comparar contra um valor histórico, medido com outra versão do avaliador, é tratado como defeito de desenho (Seção 4.7).

### 3.6 Análise estatística

Comparações entre sistemas sobre o evento "o documento alvo alcançou as k primeiras posições" são feitas por teste de McNemar exato [28] sobre pares discordantes, com os mesmos itens submetidos a todos os sistemas. Diferenças de nDCG@10 usam bootstrap pareado por consulta. Intervalos para proporções usam o método de Wilson [29], escolhido porque a aproximação normal produz intervalo de largura nula na ausência de observações contrárias, regime frequente nas taxas medidas aqui.

Onde há múltiplas variantes comparadas contra o mesmo controle, aplica-se correção de Bonferroni, com o limiar registrado antes da coleta dos dados, junto com a regra de leitura de cada desfecho possível.

O bootstrap reamostra sempre a unidade que carrega a dependência, e não a observação elementar: artigos, e não equações, nas medições de degradação de equações, porque as equações de um mesmo artigo compartilham o tratamento que o pipeline de extração lhes aplicou; documentos, e não posições, nos bits por byte; sequências, e não tokens, na modelagem mascarada por região. Tratar as unidades menores como independentes produziria intervalo artificialmente estreito.

### 3.7 Reprodutibilidade

Cada etapa do pipeline grava um manifesto com os hashes BLAKE3 das entradas e saídas, os parâmetros e o identificador do commit. Uma cadeia de hashes em três níveis cobre, a partir de uma única raiz, o manifesto de cada etapa e, através dele, cada arquivo. A verificação superficial confere os manifestos; a verificação profunda relê os dados e é a única capaz de detectar alteração no conteúdo dos arquivos.

A primeira versão da cadeia omitia uma fonte inteira: o peS2o, com 18,34 GB, não constava da lista de etapas, e a raiz atestava 21,79 GB de um corpus de cerca de 40 GB. Nenhuma verificação acusava, porque o construtor percorre apenas as etapas que conhece e reporta sucesso sobre elas. A cadeia atual cobre 52,40 GB em 35 etapas e 1.295 arquivos, e uma verificação exige que toda fonte declarada no filtro tenha etapa correspondente — uma propriedade conferida entre dois trechos de código, e não por convenção.

### 3.8 Avaliação do encoder pré-treinado

**Bits por byte.** A comparação entre tokenizadores usa a informação atribuída pelo modelo ao texto dividida pelo número de bytes, e não pelo número de tokens; a acurácia de predição mascarada não compara vocabulários, pela razão discutida na Seção 5.3. Um modelo mascarado não fornece verossimilhança exata, e cada aproximação tem viés de direção conhecida. Três instrumentos foram usados, cada um com a regra de leitura registrada antes do respectivo resultado: (1) mascarar 15% dos tokens, que favorece o tokenizador de tokens mais curtos, porque esconde pedaços menores cercados de fragmentos visíveis da mesma palavra; (2) mascarar unidades de texto delimitadas por cortes presentes nas duas segmentações e pontuar cada token independentemente, que favorece o tokenizador de tokens mais longos, porque a soma das entropias marginais é maior ou igual à entropia conjunta e a folga cresce com o número de tokens por unidade; e (3) a pseudo-verossimilhança da esquerda para a direita dentro de cada unidade, adaptada de Kauf e Ivanova [31], que recompõe a verossimilhança conjunta pela regra da cadeia e deixa um resíduo leve a favor do tokenizador de tokens mais longos. Os instrumentos 1 e 2 cercam o valor; o 3 decide.

**Modelagem mascarada por região.** Tokens de equação e de prosa são mascarados uniformemente a 15%, nas mesmas posições para os dois braços, sobre 2.000 sequências sorteadas de uma partição que nenhum braço viu, com contexto de 1.024. A medida primária do mascaramento de equações é a diferença das diferenças: a vantagem de acurácia em equação sobre prosa no braço tratado, menos a mesma vantagem no controle. A diferença simples não serve, porque tokens de LaTeX são localmente redundantes: o ModernBERT-base, que nunca viu mascaramento de equações, acerta 0,8765 em tokens de equação contra 0,7480 em prosa, uma vantagem de 0,1286 sem tratamento algum.

**Recuperação.** O protocolo de comparação de encoders da Seção 3.5, aplicado ao encoder com agregação por média, sem ajuste, e novamente após ajuste contrastivo como bi-encoder.

**Detecção de instabilidade.** Uma perda acima de μ + 4σ da janela de 100 passos anteriores, ou uma norma de gradiente acima de dez vezes a mediana móvel, aciona o retorno ao último ponto de verificação, o salto dos lotes suspeitos e 500 passos com taxa de aprendizado reduzida à metade; três acionamentos em 5.000 passos interrompem o treinamento. No braço tratado, a perda do critério é calculada apenas sobre os alvos do sorteio uniforme, e não sobre os da equação inteira, pela razão discutida na Seção 5.4.

**Sonda de estrutura tensorial.** Cada item é um trio sobre o mesmo texto: uma expressão base (`T^{\mu\nu}`), uma variante estrutural com os mesmos símbolos e estrutura diferente (`T_{\mu\nu}`) e uma variante renomeada com a mesma estrutura e símbolos diferentes (`T^{\alpha\beta}`). O item é acertado quando a base é mais próxima da renomeada que da estrutural. São 72 itens, em 4 famílias, 6 tensores e 3 textos. Por semelhança de superfície a variante estrutural é a mais próxima, e um modelo de trigramas de caracteres acerta 0 de 72; a escala resultante tem a superfície em 0, o acaso em 0,5 e a estrutura acima dele. Nenhum encoder existente avaliado alcança 0,5: ModernBERT-base 0,222, SciBERT 0,208, PhysBERT 0,028 e MiniLM-L6, o único treinado contrastivamente, 0,000.

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

O total situa-se na metade superior da faixa de 15 a 30 bilhões de tokens estimada como necessária, a custo monetário nulo. A Seção 4.3 mostra que o volume, entretanto, não é o fator limitante. Os 835.379 documentos da fatia RedPajama são linhas: 6.778 identificadores aparecem duas vezes, em cópias idênticas byte a byte distribuídas entre partições diferentes da coleta, e os documentos distintos são 828.601. A estimativa de tokens da fatia está inflada na mesma proporção, 0,8%. Nenhuma das partições envolvidas foi usada nos treinamentos da Seção 4.9.

Uma fatia adicional, de matemática e ciência da computação da mesma fonte RedPajama, foi coletada e mantida separada: 687.907 documentos, 44,9 bilhões de caracteres e cerca de 11,2 bilhões de tokens, sem duplicação interna e sem sobreposição com a fatia de Física — os 179.441 documentos com listagem cruzada em Física já pertenciam a ela e foram excluídos. Uma sonda sobre 5 de 100 fragmentos sorteados havia previsto esses volumes com erro inferior a 2%. A fatia dobra o LaTeX íntegro disponível, de 42,15 para 87,0 bilhões de caracteres, mas mais da metade desse total passaria a vir de fora do domínio. Nenhuma medição deste trabalho avalia esse compromisso, e por isso as fatias não são misturadas.

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

A métrica intrínseca também superestima o efeito disponível. No corpus de treinamento, de texto integral, a variante E gasta 15.673 tokens por documento contra 13.916 da A — 12,6% a mais, e não os 37,7% que a razão de fertilidade medida em resumos sugere. E a avaliação extrínseca, na Seção 4.9.1, inverte o sinal: com o mesmo orçamento de tokens, o modelo treinado sem as regras é o melhor.

### 4.5 Recuperação densa: defeito de protocolo e remedição

**Tabela 6.** Recuperação de documento citado, 2.000 candidatos, protocolo idêntico para todos os modelos. Valores históricos, obtidos sob o protocolo defeituoso descrito a seguir.

| Modelo | Parâmetros | nDCG@10 | recall@1 | recall@10 |
|---|---|---|---|---|
| Bi-encoder ajustado (este trabalho) | 23 M | 0,4657 | 0,262 | 0,708 |
| GTE-large [10] (geral) | 335 M | 0,4628 | 0,278 | 0,677 |
| Bi-encoder ajustado, execução local | 23 M | 0,4579 | 0,254 | 0,700 |
| PhysBERT [9] (domínio) | 109 M | 0,2752 | 0,146 | 0,425 |
| SciBERT [6] (base) | 110 M | 0,2074 | 0,109 | 0,328 |

A leitura registrada à época da medição tinha duas partes: superioridade ampla sobre o modelo de mesmo domínio, de 0,190 de nDCG@10 sobre o PhysBERT, e paridade com o modelo geral, com diferença de 0,0029 no nDCG@10 não sustentada pelo teste pareado em recall@1 (196 discordantes contra 165 em favor do modelo geral, p = 0,114).

Uma auditoria posterior invalidou o protocolo que produziu esta tabela. A avaliação é feita dentro do lote: calcula-se a matriz de similaridade entre âncoras e positivos, e a resposta correta da linha *i* é a coluna *i*. A tarefa só é bem posta se as colunas forem distintas, e duas condições violavam isso. A amostra era tomada pelas primeiras *n* linhas do arquivo, cuja ordem não é neutra — a mediana do comprimento da âncora decresce de aproximadamente 1.180 para 870 caracteres do início ao fim, e o primeiro bloco de 2.000 situa-se no percentil 94. Mais grave, esse bloco continha apenas 1.147 textos positivos distintos, com 62% das linhas ocupadas por um positivo repetido e um deles ocorrendo 28 vezes; textos idênticos produzem similaridade idêntica, e o desempate da ordenação é arbitrário. Adicionalmente, âncoras repetidas compartilham uma única ordenação, de modo que no máximo uma delas pode ocupar a primeira posição.

**Tabela 7.** Desempenho de um recuperador perfeito sob cada esquema de amostragem do conjunto de avaliação.

| Esquema de amostragem | recall@1 | nDCG@10 |
|---|---|---|
| Primeiras 2.000 linhas (usado na Tabela 6) | 0,5235 | 0,7562 |
| Amostra aleatória de 2.000 | 0,9364 | 0,9761 |
| Amostra aleatória com deduplicação por texto | 1,0000 | 1,0000 |

O recall@1 de 0,2620 da Tabela 6 foi obtido contra um teto de 0,5235. A comparação entre modelos era justa, pois todos foram submetidos ao mesmo protocolo, mas a margem não sobrevivia: a paridade decidia-se em 0,0029 de nDCG@10, e o ruído de desempate arbitrário em 62% dos itens é de magnitude superior.

O mesmo defeito afetava a amostragem do conjunto de treinamento. Sobre os 6.564.111 pares disponíveis, tomar os primeiros 400.000 fornece 17.844 documentos citados distintos, enquanto uma amostra aleatória de mesmo tamanho fornece 191.300 — fator de 10,7 em diversidade, ao mesmo custo de computação. O modelo da Tabela 6 foi treinado sob a primeira condição.

**Tabela 8.** Remedição no protocolo corrigido: 2.000 pares sorteados e deduplicados, teto de 1,0000 medido e gravado.

| Modelo | Parâmetros | nDCG@10 | recall@1 | recall@10 | MRR |
|---|---|---|---|---|---|
| Bi-encoder ajustado, 6 M pares sorteados | 23 M | **0,6223** | **0,4315** | **0,8305** | **0,5636** |
| GTE-large [10] (geral) | 335 M | 0,5788 | 0,4140 | 0,7640 | 0,5293 |
| Bi-encoder ajustado, 400 mil pares (primeiras linhas) | 23 M | 0,5265 | 0,3560 | 0,7185 | 0,4773 |
| MiniLM-L6 [12] (geral, sem ajuste) | 23 M | 0,4761 | 0,3135 | 0,6640 | 0,4279 |
| SciBERT ajustado, 400 mil pares (primeiras linhas) | 110 M | 0,4746 | 0,3070 | 0,6705 | 0,4263 |
| PhysBERT [9] (domínio) | 109 M | 0,3507 | 0,2220 | 0,4910 | 0,3170 |
| SciBERT [6] (sem ajuste) | 110 M | 0,2537 | 0,1490 | 0,3825 | 0,2275 |

Sob o protocolo corrigido, o veredito contra o modelo geral inverteu-se duas vezes. Remedido, o melhor bi-encoder então disponível — 1,5 milhão de pares tomados das primeiras linhas, 0,5442 — ficava 0,035 abaixo do GTE-large, e não 0,003 acima; o novo treinamento com pares sorteados e volume crescente (Seção 4.6) levou-a a +0,044. O bi-encoder de 23 M supera o GTE-large de 335 M em nDCG@10, recall@10 e MRR; em recall@1 a estimativa pontual lhe é favorável, mas o teste pareado não a estabelece (188 contra 153 discordantes, p = 0,065). Sobre o modelo de mesmo domínio, a margem é de 0,272 de nDCG@10.

### 4.6 Volume, diversidade e base do recuperador

**Tabela 9.** Curva de volume, uma variável por execução, protocolo da Tabela 8. A margem é em relação ao GTE-large (0,5788).

| Pares de treinamento | Documentos citados distintos | nDCG@10 | Margem |
|---|---|---|---|
| 400 mil, primeiras linhas | 17.844 | 0,5265 | −0,0523 |
| 400 mil, sorteados | 191.198 | 0,5462 | −0,0326 |
| 1,5 milhão, sorteados | 390.856 | 0,5780 | −0,0008 |
| 3 milhões, sorteados | 518.635 | 0,6026 | +0,0238 |
| 6 milhões, sorteados | 650.162 | 0,6223 | +0,0435 |

A primeira linha contra a segunda é a ablação mais limpa do trabalho — mesma base, mesmos 400 mil pares em número, mesmo lote, mesmos 3.125 passos, mesma semente —, e a única variável é a amostragem: +0,0197 de nDCG@10, com teste pareado em recall@1 de 104 contra 71 (p = 0,0153), sem custo adicional de computação. Os ganhos por duplicação de volume são de +0,032, +0,025 e +0,020. Contra a execução de 3 milhões, a de 6 milhões vence com p = 0,0012.

A execução de 6 milhões é a primeira com sinal de saturação: 15 avaliações consecutivas de validação abaixo do próprio pico, com queda de 1,0% até o fim, contra no máximo uma avaliação e queda indistinguível de ruído nas demais. Uma estatística anterior, a fração do treinamento em que o pico ocorre, sugeria o mesmo e era inválida — a execução de 400 mil tinha o pico ainda mais cedo; ela mede onde a grade de avaliação calha de capturar o máximo. O conjunto de pares está quase esgotado (650.162 de 667.304 documentos), de modo que ganho adicional teria de vir de outra fonte de pares, de outra base ou de mais parâmetros.

Duas leituras da versão anterior deste trabalho são revistas. O treinamento de 1,5 milhão de pares interrompido em 38% por estabilização da métrica usava as primeiras linhas, e os pares adicionais provinham de 67.232 documentos; sorteados, 1,5 milhão de pares alcançam 0,5780, contra 0,5442 das primeiras linhas, e a execução completa atinge o pico a 97% do caminho. A estabilização era esgotamento de documentos, e não de dados. Quanto ao número de negativos, a variante de 511 negativos obtém 0,5272 contra 0,5246 da execução de referência com 127, ambas com as primeiras linhas e remedidas no protocolo corrigido, sem ganho detectável, o que mantém a leitura de saturação entre 127 e 511.

**Tabela 10.** Base contra volume. Os dois primeiros modelos diferem apenas na base; a terceira linha é o bi-encoder de 6 milhões de pares. O pareado compara o GTE-base de 400 mil pares (primeiro número) ao modelo da linha, exceto na última, que compara o GTE-base de 1 milhão ao bi-encoder de 6 milhões, medidos na mesma sessão.

| Modelo | Pares | nDCG@10 | recall@1 | recall@10 | Pareado em recall@1 |
|---|---|---|---|---|---|
| GTE-base [10] ajustado (109 M) | 400 mil | 0,6094 | 0,4300 | 0,8010 | — |
| MiniLM-L6 ajustado (23 M) | 400 mil | 0,5462 | 0,3725 | 0,7395 | 219 × 104, p = 1,5 × 10⁻¹⁰ |
| MiniLM-L6 ajustado (23 M) | 6 milhões | 0,6223 | 0,4315 | 0,8305 | 147 × 150, p = 0,908 |
| GTE-base ajustado (109 M) | 1 milhão | 0,6211 | 0,4335 | 0,8180 | 148 × 144, p = 0,861 |

Com a base geral de 109 M, 400 mil pares produzem o que o volume produziu com quinze vezes mais pares. O efeito é coerente com um padrão observado antes: o ganho do ajuste por citação é inverso ao ponto de partida — o SciBERT sobe 0,221 e termina abaixo do MiniLM-L6, que sobe 0,049 (Tabela 8) —, o que sugere que o ajuste ensina sobretudo a tarefa de recuperação, que uma base contrastiva geral já sabe. No universo completo de 88.807 documentos, o GTE-base sem ajuste algum já empata em recall@10 com o bi-encoder de 6 milhões de pares (0,2755 contra 0,2810, p = 0,545), embora perca em profundidade (recall@200 de 0,6450 contra 0,7300).

A base, contudo, não foi trocada no sistema. Embutir o mesmo universo custa 954 segundos com o GTE-base e 217 com o MiniLM-L6, uma razão de 4,4 que se repete em serviço; pagar esse custo por um empate com o recuperador instalado não se justifica, e a combinação de base e volume, estimada em 39 horas de T4, não foi executada.

Um ponto intermediário foi medido depois, com regra registrada antes: ajustado com 1 milhão de pares, o GTE-base empata com o bi-encoder de 6 milhões — nDCG@10 de 0,6211 contra 0,6223, diferença de −0,0012 [−0,011; +0,008] por bootstrap pareado por item — e supera o MiniLM-L6 de 1,5 milhão por +0,043 [+0,033; +0,054]. Com um sexto do volume, a base alcança o recuperador instalado e não o passa. A curva do GTE-base cresce devagar — 0,5964 a 200 mil pares, 0,6094 a 400 mil, 0,6211 a 1 milhão —, e a regra manteve o sistema como está: sem margem, o custo de serviço decide.

A primeira medição da Tabela 10 quase registrou um falso negativo. Ela foi feita com 256 candidatos, valor padrão do avaliador, e produziu +0,039 com p = 0,161; com os 2.000 do protocolo, +0,0633 com p = 1,5 × 10⁻¹⁰. Com 256 candidatos a tarefa é muito mais fácil — o MiniLM-L6 sem ajuste marca 0,732 contra 0,476 —, e as duas escalas não se comparam. A discrepância foi percebida por comparação com o valor histórico do bi-encoder, que a execução com 2.000 reproduziu ao milésimo.

Um último fator do recuperador foi testado e descartado. Com truncamento a 192 tokens, 63,8% das âncoras são truncadas e 28,2% do texto é descartado; o bi-encoder já treinado, lendo 256 ou 384 tokens, não altera o recall em nenhuma profundidade (p entre 0,08 e 0,84) e custa 2,9 vezes mais para embutir. O resultado não exclui que treinar com contexto maior ajude, mas remove a razão de esperá-lo.

### 4.7 Sistema híbrido

**Tabela 11.** Composição de recuperadores com o bi-encoder de 400 mil pares. Mil consultas, universo de 88.807 documentos citados distintos, profundidade 50. O teste pareado compara cada sistema à fusão.

| Sistema | recall@1 | recall@10 | recall@50 | nDCG@10 | McNemar (k = 10) |
|---|---|---|---|---|---|
| BM25 [25] | 0,067 | 0,236 | 0,401 | 0,1399 | p = 0,00073 |
| Bi-encoder | 0,055 | 0,233 | 0,428 | 0,1327 | p = 0,00018 |
| Fusão recíproca de postos [27] | 0,068 | 0,271 | 0,446 | 0,1584 | — |
| Fusão + cross-encoder (base MiniLM) | 0,064 | 0,254 | 0,446 | 0,1493 | p = 0,118 |

A fusão supera ambos os recuperadores isolados com significância. O cross-encoder inicializado a partir da mesma base do recuperador denso não produz ganho mensurável.

O trajeto até esse resultado contém o modo de falha mais informativo do trabalho. A mineração inicial de negativos difíceis tomava os *K* primeiros resultados do índice denso, excluída a citação verdadeira. Como todo negativo assim obtido está no topo do índice, enquanto o positivo frequentemente não está, o critério de rotulagem torna-se predizível pela posição no recuperador, e o modelo aprende a regra mais simples que separa as classes.

**Tabela 12.** Correlação de Spearman entre a posição do candidato na fusão e o escore atribuído pelo cross-encoder, por estratégia de mineração de negativos. Posição menor indica melhor colocação, de modo que o sinal desejado é negativo.

| Estratégia de mineração | Spearman | Consultas com correlação positiva |
|---|---|---|
| K primeiros do índice denso, menos o positivo | +0,179 | 83% |
| Amostragem dos 50 primeiros da fusão real | −0,466 | 0% |

O modelo treinado sob a primeira estratégia prefere sistematicamente a cauda do recuperador e atinge nDCG@10 de 0,0179, inferior à ordenação aleatória. Corrigida a distribuição do grupo de treinamento para coincidir com a da avaliação, a correlação inverte-se e o modelo passa a concordar com o recuperador — e é precisamente essa concordância que explica a ausência de ganho: um reordenador que reproduz a ordenação já produzida pela fusão não acrescenta informação.

Seis hipóteses alternativas foram testadas e descartadas no diagnóstico: erro de inferência do backend de atenção (diferença máxima de 4 × 10⁻⁶ contra a implementação de referência, com ordenação idêntica); negativos provenientes de população distinta (todos os minerados são documentos citados); grau de citação como atalho (o positivo tem grau médio 113,6 contra 8,5 dos negativos, mas a correlação de Spearman entre grau e escore é +0,056, com p = 0,17); atalho de formato de superfície (14 atributos, melhor AUC de 0,575); artefato de desempate na ordenação (permutar a posição do positivo produz valores idênticos); e insensibilidade do modelo ao texto da consulta, hipótese formulada sobre 16 documentos e refutada com 457, com efeito medido de +0,143 ± 0,059.

**Tabela 13.** A mesma composição com o recuperador antigo (400 mil pares) e o novo (6 milhões), medidos na mesma sessão, com o mesmo código e as mesmas 2.000 consultas, profundidade 100. O reordenador é o de base PhysBERT da Seção 4.8.

| Sistema | recall@100, antigo | nDCG@10, antigo | recall@100, novo | nDCG@10, novo |
|---|---|---|---|---|
| BM25 | 0,4540 | 0,1340 | 0,4540 | 0,1340 |
| Bi-encoder | 0,5160 | 0,1323 | 0,6325 | 0,1585 |
| Fusão | 0,5380 | 0,1594 | 0,6065 | 0,1643 |
| Fusão + reordenação | 0,5380 | 0,1685 | 0,6065 | 0,1688 |

O BM25 idêntico nos dois braços é o controle de que o protocolo não mudou entre eles. O recuperador ganhou 0,117 de recall@100; a cadeia completa, 0,0003 de nDCG@10. Os braços na mesma sessão eram parte do desenho, e pagaram: comparado ao valor histórico de 0,1666, o ganho reportado seria de 0,0022, sete vezes o efeito real, por deriva de código e protocolo entre as duas medições.

Com o recuperador novo, a fusão deixa de somar. Contra o bi-encoder isolado em k = 10, ela vencia com o recuperador antigo (p = 1,5 × 10⁻⁸) e empata com o novo (p = 0,949); e seu recall@100 passa a ser inferior ao do bi-encoder isolado, isto é, a fusão desloca candidatos bons para fora do conjunto que o reordenador vê. A regra registrada antes da execução — o BM25 permanece apenas se a cadeia com fusão vencer a cadeia sem fusão — retirou-o da composição. Uma comparação independente, de orçamento igual de candidatos, confirma: o bi-encoder com 200 candidatos alcança recall de 0,7300, contra 0,6835 da união de 100 candidatos de cada recuperador. Em todas as profundidades medidas, o BM25 acrescenta cerca de metade do que os mesmos candidatos adicionais do bi-encoder acrescentariam.

O que falta ao recuperador é modelo, e não tarefa. Entre as 735 consultas cujo alvo não está nos 100 primeiros, a mediana do posto do alvo é 396 entre 88.807 documentos, e apenas 10 consultas (0,5%) têm o alvo além do posto 10.000 para os dois recuperadores, que têm vieses opostos.

### 4.8 Efeito do corpus de pré-treinamento da base do reordenador

O diagnóstico da Seção 4.7 produz uma predição falseável: se o reordenador não acrescenta informação por redundância com o recuperador, então uma base pré-treinada em corpus distinto deveria acrescentá-la. A predição foi testada com regra de decisão registrada antes da coleta — teste de McNemar em k = 10 contra a fusão da mesma execução, limiar de Bonferroni de 0,025 por serem duas variantes, 2.000 consultas — e com as quatro leituras possíveis escritas de antemão, inclusive a de nenhuma variante superar o controle.

**Tabela 14.** Cross-encoders idênticos exceto pela base, sobre a fusão com o recuperador antigo. Duas mil consultas, profundidade 50, fusão de referência com nDCG@10 de 0,1576 nas três execuções, teto do conjunto de 0,4495. "Acerto@1" refere-se ao grupo de 8 candidatos do treinamento; "Diferença" é em relação à fusão.

| Base | Parâmetros | Acerto@1 | nDCG@10 com reordenação | Diferença | Discordantes | p (k = 10) |
|---|---|---|---|---|---|---|
| MiniLM-L6 (a mesma do recuperador) | 23 M | 0,498 | 0,1483 | −0,0093 | 229 | 0,1458 |
| GTE-base (geral forte) [10] | 109 M | 0,510 | 0,1530 | −0,0046 | 220 | 0,6371 |
| PhysBERT (Física) [9] | 109 M | 0,566 | 0,1666 | +0,0090 | 237 | 0,0062 |

Apenas a base pré-treinada no domínio supera a fusão. A leitura correspondente, registrada previamente, é de que o mecanismo é conhecimento de domínio, e não diversidade de base nem capacidade. O que sustenta a interpretação causal é o controle: GTE-base e PhysBERT compartilham número de parâmetros, arquitetura, os seis hiperparâmetros de treinamento, os dados, a semente e o protocolo de avaliação, diferindo exclusivamente no corpus de pré-treinamento.

**Tabela 15.** Composição completa contra a fusão isolada, com o recuperador antigo.

| Métrica | Fusão | Fusão + reordenação de domínio |
|---|---|---|
| recall@1 | 0,0700 | 0,0690 |
| recall@10 | 0,2675 | 0,2890 |
| recall@50 | 0,4495 | 0,4495 |
| nDCG@10 | 0,1576 | 0,1666 |

No teste pareado em k = 10, a fusão vence em 97 casos e o sistema com reordenação em 140, sobre 237 discordantes, com p = 0,0062, abaixo do limiar de Bonferroni. Em k = 1 há empate (p = 0,930): o ganho concentra-se na cauda das dez primeiras posições, comportamento esperado de reordenação.

Duas ressalvas de execução são declaradas. A combinação das duas execuções é legítima porque o braço de controle produziu resultados idênticos campo a campo em ambas, com p = 0,14584 sobre 229 discordantes nos dois casos; sem essa verificação, a comparação seria entre protocolos e não entre bases. E a primeira execução da variante geral falhou por defeito de implementação, e não do modelo: os pesos distribuídos em precisão de 16 bits eram carregados nessa precisão, e o escalonador de gradiente exige pesos-mestres em 32 bits. O defeito permaneceu latente porque as outras duas bases são distribuídas em 32 bits.

**O estágio com o recuperador melhorado.** A célula que mediu a Tabela 13 registrava uma predição: o recall do recuperador é o teto do reordenador, e um recuperador melhor deixa menos a reordenar. O ganho marginal da reordenação sobre a fusão caiu de +0,0091 (p = 0,0081) para +0,0045 (p = 0,086). Retreinar o reordenador com negativos minerados do recuperador novo — a hipótese de que o ganho encolheu por desalinhamento de distribuição — não o recuperou: o novo empata com o antigo (92 contra 82 discordantes em k = 10, p = 0,495). E pelo critério registrado antes, nenhuma das duas cadeias vence o bi-encoder isolado em k = 10 (141 contra 168, p = 0,139; 157 contra 174, p = 0,379).

**Tabela 16.** Reordenação sobre o bi-encoder novo, sem fusão, por profundidade. Duas mil consultas; @50 e @100 obtidos dos mesmos escores de @200. "Teto" é o recall do conjunto de candidatos.

| Sistema | recall@1 | recall@10 | Teto | nDCG@10 |
|---|---|---|---|---|
| Bi-encoder | 0,0655 | 0,2810 | — | 0,1585 |
| + reordenação @50 | 0,0670 | 0,2930 | 0,5210 | 0,1664 |
| + reordenação @100 | 0,0675 | 0,2945 | 0,6325 | 0,1676 |
| + reordenação @200 | 0,0670 | 0,2950 | 0,7300 | 0,1673 |

Dobrar os candidatos de 100 para 200 alterou o desfecho de 39 consultas, divididas em 19 e 20 (p = 1,0). O teto subiu em 195 consultas e o recall@10, em uma: o reordenador não traz ao topo um alvo situado entre as posições 100 e 200 do recuperador. O bootstrap pareado sobre o nDCG@10 por consulta, com 20.000 reamostras, não estabelece ganho em nenhuma profundidade (@100: +0,0091 [−0,0010; +0,0191]). E a decomposição contraria a função do estágio: entre as 421 consultas com o alvo entre os dez primeiros nos dois sistemas, o reordenador rebaixa o alvo em 167 e o eleva em 141 (p = 0,154); todo o ganho nominal vem de trazer alvos ao top-10, e nada de ordená-los melhor.

São cinco medições pareadas, com dois reordenadores, e em nenhuma o estágio vence o recuperador isolado. Estabelecer o ganho nominal de 0,0091, se ele existir, exigiria cerca de 6.640 consultas, ou 4,7 horas de T4 por braço, para um estágio que custa 109 M de parâmetros em serviço. O estágio foi retirado, e o sistema entrega os dez primeiros resultados do bi-encoder.

### 4.9 Pré-treinamento do encoder

O corpus de pré-treinamento está preparado: 2.001.270.262 tokens de 143.810 documentos, em 244.295 sequências de 8.192 tokens, com 9 de 44 partições sorteadas por semente registrada. A conferência sobre 300 sequências amostradas do binário efetivamente lido pelo treinamento indica 37,8% de tokens matemáticos contra 38,8% no corpus bruto, 25,0% em display contra 25,2%, taxa efetiva de mascaramento de 0,3000 contra 0,3000 requerida, e fração tratada de 0,903 a 8.192 tokens.

A restrição a equações em display, e não a toda notação matemática, decorre de medição: em 120 documentos, as equações em display têm mediana de 79 tokens, e as sequências em linha, mediana de 7 — o que corresponde a uma variável isolada, e não a uma equação. Tratar toda notação faria a ablação comparar condições substancialmente equivalentes.

A correção do laço de treinamento é verificada pela perda inicial: um modelo de linguagem mascarada não treinado prediz distribuição uniforme, de modo que a entropia cruzada inicial deve igualar o logaritmo do tamanho do vocabulário. O valor medido é 10,7343, contra ln(40.960) = 10,6204. Esse é o único indicador que distingue um mascaramento correto de um que apague os alvos ou os derive da entrada já mascarada — condições que não produzem nenhum outro sintoma observável.

Um defeito no marcador de equações ilustra o ponto. A expressão regular de detecção usava âncora de início de cadeia em conjunto com casamento a partir de posição arbitrária; a âncora continua referindo-se ao início real da cadeia, de modo que nenhuma equação iniciada após o primeiro caractere era detectada. O efeito medido foi de 0 documentos tratados em 120, contra 91,7% de documentos contendo equações em display segundo um segundo instrumento, e foi a discordância entre os dois instrumentos que localizou o erro. Sem o contador de fração tratada, a ablação teria sido executada comparando duas condições aleatórias e reportado ausência de efeito.

As duas primeiras ablações a seguir usam os substitutos de 48 M da Seção 3.4, e a terceira, o pré-treinamento continuado da base de 150 M; todas têm uma semente por braço e são medidas sobre partições que nenhum braço viu. Os braços foram treinados em imagem de nuvem com `transformers` 5.0 e medidos localmente com a versão 4.48; como a versão antiga ignora chaves desconhecidas da configuração e usaria valores padrão, os modelos foram reexportados localmente, com diferença de logits nula contra a exportação original.

#### 4.9.1 Regras de pré-tokenização de LaTeX

As variantes A e E da Tabela 5 compartilham vocabulário, arquitetura e contagem de parâmetros, e diferem apenas nas regras de pré-tokenização que tornam `\frac`, `\begin{…}`, `^{` e `_{` unidades atômicas. É o único par de variantes sem confundidor de capacidade: a 48 M, com a matriz de embedding a 43,7% dos parâmetros, variantes de vocabulário diferente teriam contagens diferentes. O protocolo iguala tokens, e não texto; como E gasta 12,6% mais tokens por documento, A vê 12,5% mais texto pelo mesmo custo, e é essa a vantagem sob teste.

**Tabela 17.** Bits por byte (menor é melhor) por três instrumentos, cada um com a regra de leitura registrada antes do próprio resultado. Intervalos por bootstrap pareado por documento.

| Instrumento (Seção 3.8) | Documentos | Viés favorece | A (com regras) | E (sem regras) | A − E |
|---|---|---|---|---|---|
| 1 · 15% dos tokens mascarados | 2.000 | E | 0,6653 | 0,5878 | +0,078 [+0,074; +0,083] |
| 2 · unidades comuns, pontuação independente | 2.000 | A | 1,3096 | 1,3680 | −0,051 [−0,056; −0,046] |
| 3 · unidades comuns, da esquerda para a direita | 1.000 | A, resíduo leve | 0,8969 | 0,8500 | +0,047 [+0,043; +0,050] |

Os instrumentos 1 e 2 anulam-se, cada um a favor do braço que favorece, com intervalos estreitos de sinais opostos: um instrumento enviesado com poder de sobra encontra o viés com a mesma confiança com que encontraria o efeito. O instrumento 3 cai dentro do intervalo que os dois cercam, como a teoria prevê, e a queda do instrumento 2 para o 3 é maior em E (−0,52) que em A (−0,41), o que corresponde ao mecanismo nomeado: E, com mais tokens por unidade, pagava mais a folga entre entropias marginais e conjunta. Pela regra registrada, E à frente no instrumento 3, apesar do resíduo a favor de A, rejeita as regras de pré-tokenização a esta escala. Unidades de mais de 8 tokens são 6% das unidades e 53% dos bytes: a diferença entre os tokenizadores concentra-se em LaTeX longo sem espaços.

As medidas secundárias, registradas como incapazes de reverter a primária, não a acompanham. Na recuperação sem ajuste, A fica à frente (nDCG@10 de 0,0296 contra 0,0171; E − A = −0,0125 [−0,020; −0,005]; recall@1 com McNemar p = 0,004), com os dois braços perto do piso. Na sonda tensorial não há diferença (0,417 contra 0,333; 19 contra 13 discordantes, p = 0,38). Uma assimetria de execução favorece E e não foi controlada: o braço A sofreu uma instabilidade de perda que acionou retorno ao ponto de verificação anterior, descartando cerca de 3,3% do orçamento, seguido de 500 passos com taxa de aprendizado reduzida à metade; E treinou sem incidentes.

#### 4.9.2 Mascaramento de equações inteiras

O controle é o braço E da Seção 4.9.1 (`p_equacao` = 0); o tratado difere dele apenas em `p_equacao` = 0,6, o que foi conferido por comparação da árvore sintática das duas células de treinamento. A 1.024 tokens, 42% das janelas não contêm nenhuma equação em display que comece nelas, e a fração tratada, de 0,903 a 8.192 tokens, cai para cerca de 0,55; o valor de 0,6 foi escolhido antes do treinamento para que cerca de um terço dos exemplos mascarasse uma equação inteira. Uma equação típica de 75 tokens corresponde a cerca de 17% do orçamento de mascaramento de uma janela de 1.024.

**Tabela 18.** Checagens de manipulação e medida primária. Duas mil sequências sorteadas, as mesmas posições e regiões nos dois braços. Intervalos por bootstrap pareado por sequência.

| Medida | Controle | Tratado | Tratado − controle |
|---|---|---|---|
| Fração de exemplos com equação elegível (limiar ≥ 0,50) | — | 0,5381 | — |
| Acurácia com a equação inteira mascarada (104.141 tokens) | 0,0704 | 0,1984 | +0,128 [+0,122; +0,134] |
| Acurácia em tokens de equação, máscara uniforme (108.388 tokens) | 0,8694 | 0,8643 | −0,0051 |
| Acurácia em prosa, máscara uniforme (199.474 tokens) | 0,6944 | 0,6933 | −0,0011 |
| **Vantagem em equação (diferença das diferenças)** | +0,1750 | +0,1710 | **−0,0040 [−0,0058; −0,0022]** |

O tratamento foi absorvido — reconstruir uma equação inteira escondida vai de 7% para 20% — e não se transferiu: sob máscara uniforme, o tratado acerta menos tokens de equação que o controle, e a perda em equação excede a perda em prosa. Pela regra, o desfecho é negativo, e o efeito é pequeno: 0,4 ponto, contra 12,8 da checagem. Um viés da medida havia sido nomeado antes do resultado e aponta para o controle: o tratado viu tokens de equação mascarados sobretudo em blocos inteiros, enquanto a prova uniforme esconde tokens isolados com os vizinhos visíveis, o regime que o controle praticou durante todo o treinamento. A diferença das diferenças cancela um custo geral de distribuição, mas não um custo concentrado na região de equação. Em análise exploratória, posterior ao resultado, o negativo concentra-se nas equações em display (−0,0055) e não nas em linha (−0,0015, intervalo cruzando zero). A assimetria de execução, desta vez, é contra o tratado: uma instabilidade no passo 8.075 descartou 76 lotes e reduziu a taxa de aprendizado à metade no início do decaimento.

**Tabela 19.** Medidas secundárias: recuperação sem ajuste, recuperação após ajuste contrastivo e sonda tensorial.

| Medida | Controle | Tratado | Diferença |
|---|---|---|---|
| nDCG@10, sem ajuste (agregação por média) | 0,0171 | 0,1391 | +0,122 [+0,109; +0,135] |
| recall@1, sem ajuste | 0,0055 | 0,0785 | McNemar p = 3,6 × 10⁻³⁶ |
| nDCG@10, após ajuste com 200 mil pares | 0,3872 | **0,4712** | **+0,084 [+0,071; +0,097]** |
| recall@1, após ajuste | 0,2215 | 0,2985 | 106 × 260, p = 4,7 × 10⁻¹⁶ |
| recall@10, após ajuste | 0,5890 | 0,6740 | — |
| Sonda tensorial (72 itens) | 0,333 | 0,375 | 12 × 15, p = 0,70 |

**Tabela 20.** O mesmo ajuste contrastivo sobre uma base geral de 150 M sem qualquer pré-treinamento no domínio, com os mesmos 200 mil pares, os mesmos hiperparâmetros e os três braços medidos na mesma sessão.

| Braço | Parâmetros | Pré-treinamento | recall@1 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| Controle | 48 M | 0,6 B tokens de Física, do zero | 0,2215 | 0,5890 | 0,3391 | 0,3872 |
| Tratado | 48 M | idem, com mascaramento de equações | 0,2985 | 0,6740 | 0,4203 | 0,4712 |
| ModernBERT-base [26] | 150 M | cerca de 2 T tokens gerais | **0,3540** | **0,7200** | **0,4774** | **0,5270** |

A base geral supera o braço tratado em 0,0558 de nDCG@10, com intervalo de [+0,0426; +0,0685] por bootstrap pareado por item — a leitura registrada antes da coleta para esse desfecho. O mascaramento de equações ajuda, e não compensa a diferença de escala: um encoder de 48 M treinado com 0,6 bilhão de tokens do domínio não alcança um de 150 M treinado com cerca de 2 trilhões de tokens gerais, medido na tarefa do domínio.

A comparação não é de uma variável, e isso estava declarado antes: diferem o tamanho, o tokenizador e o volume de pré-treinamento. Ela não testa a hipótese do mascaramento — testa se o artefato que a hipótese produz, na escala que o orçamento permite, compete com o que existe de graça.

Um fator de oito na recuperação sem ajuste, entre modelos que diferem em um único hiperparâmetro, exigia descartar primeiro a explicação geométrica. A anisotropia dos dois braços é igual (cosseno médio de 0,972 e 0,971), centralizar os vetores não fecha a diferença (0,021 contra 0,143) e o braço A da Seção 4.9.1, também sem tratamento, fica em 0,030. A pergunta relevante para o sistema, porém, era se a diferença sobrevive ao ajuste contrastivo, que no recuperador pode apagar ou ampliar diferenças de base (Seção 4.6). Os dois braços foram ajustados como bi-encoders com os hiperparâmetros da Tabela 10, sobre os mesmos 200 mil pares sorteados (120.002 documentos citados), e medidos localmente na mesma sessão, com a regra registrada antes. O tratado fica à frente: o ganho encolhe de +0,122 para +0,084, e permanece de 22% em nDCG@10.

O resultado da ablação é, portanto, dividido. O mascaramento de equações inteiras não melhora a predição de tokens de equação e melhora a representação agregada para recuperação, antes e depois do ajuste. O mecanismo não foi medido. Os valores absolutos, com bases de 48 M e metade dos pares, não se comparam com os das Tabelas 8 a 10.

#### 4.9.3 Pré-treinamento continuado a 150 M

A Tabela 20 delimitou a pergunta: se o objetivo de mascaramento de equações tem valor, ele deve aparecer no pré-treinamento continuado da base que venceu, e não num modelo treinado do zero. Os dois braços partem do ModernBERT-base e diferem apenas em `p_equacao` (Seção 3.4). As regras de leitura, primária e secundária, foram registradas antes do treinamento.

A primeira execução do braço tratado foi interrompida pelo detector de instabilidade no passo 4.489, após três acionamentos, com a norma do gradiente normal nos três; a Seção 5.4 mostra que os acionamentos correspondiam à composição dos lotes, e não a instabilidade. Com o critério corrigido, o braço foi refeito do zero e treinou os 6.103 passos sem nenhum acionamento. O controle treinou os 6.103 passos com dois acionamentos pelo critério de norma — 7,2 e 7,7 contra medianas móveis de 0,64 e 0,61 —, que descartaram 182 lotes e reduziram a taxa de aprendizado à metade por 500 passos. Essa assimetria, desta vez, favorece o tratado.

**Tabela 21.** Pré-treinamento continuado: checagens de manipulação e medida primária. Duas mil sequências sorteadas de partição não vista, contexto de 1.024, as mesmas posições nos dois braços. Intervalos por bootstrap pareado por sequência.

| Medida | Controle | Tratado | Tratado − controle |
|---|---|---|---|
| Fração de exemplos com equação elegível (limiar ≥ 0,50) | — | 0,549 | — |
| Acurácia com a equação inteira mascarada (104.624 tokens) | 0,0266 | 0,1999 | +0,173 [+0,167; +0,179] |
| Acurácia em equação, máscara uniforme | 0,9032 | 0,9030 | — |
| Acurácia em prosa, máscara uniforme | 0,7783 | 0,7784 | — |
| **Vantagem em equação (diferença das diferenças)** | +0,1249 | +0,1245 | **−0,00039 [−0,00161; +0,00087]** |

Pela regra, o desfecho primário é não decidido. O tratamento foi absorvido com mais força que no substituto — reconstruir a equação inteira escondida vai de 2,7% para 20,0%, contra 7% para 20% a 48 M — e, sob máscara uniforme, os dois braços acertam equação e prosa igualmente até a terceira casa decimal. O negativo de −0,0040 do substituto não se repete: encolhe dez vezes, e o intervalo passa a cobrir zero.

**Tabela 22.** Os mesmos 200 mil pares e a mesma receita de ajuste contrastivo sobre quatro bases, medidas na mesma sessão, no protocolo da Seção 3.5.

| Base do ajuste | Pré-treinamento no domínio | recall@1 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| ModernBERT-base [26] | nenhum | 0,3540 | 0,7200 | 0,4774 | 0,5270 |
| ModernBERT-base, controle | 0,4 B tokens | 0,3635 | 0,7175 | 0,4823 | 0,5297 |
| ModernBERT-base, tratado | 0,4 B tokens, com mascaramento de equações | 0,3620 | 0,7290 | 0,4876 | 0,5373 |
| **GTE-base [10]** | **nenhum** | **0,4170** | **0,7880** | **0,5443** | **0,5964** |

Pelas regras registradas, três desfechos. Entre os braços, o tratado fica à frente: +0,0076 de nDCG@10 [+0,0013; +0,0139], com recall@1 empatado (63 contra 60 discordantes, p = 0,86) — o ganho de recuperação do substituto se repete, onze vezes menor que os +0,084 de lá, e com o limite inferior do intervalo próximo de zero. Contra a base sem pré-treinamento, o tratado fica acima por +0,0103 [+0,0032; +0,0176]. E o GTE-base supera o ModernBERT-base, ambos sem qualquer pré-treinamento no domínio, por +0,0694 [+0,0580; +0,0812]. O ModernBERT-base medido nesta sessão reproduz o valor de seis dias antes, 0,5270, com o mesmo avaliador.

Lidos juntos, os três dizem que 0,4 bilhão de tokens de Física com o objetivo específico de domínio compraram +0,010 sobre a base de partida, e que trocar a base de partida, sem pré-treinamento algum, compra +0,069. O GTE-base sem pré-treinamento fica 0,059 acima do melhor braço — diferença entre estimativas pontuais, não registrada como comparação, e cinco vezes a largura dos intervalos da tabela.

### 4.10 Recuperação por equação

Um argumento recorrente para sistemas híbridos em domínios matemáticos é que a recuperação densa perde o casamento simbólico exato que a busca léxica preserva. Para medi-lo no corpus próprio, foi construído um conjunto de avaliação de recuperação por equação, com gabarito gratuito: a consulta é uma equação de um documento, e os alvos são os demais documentos que contêm a mesma forma canônica de conteúdo. Das 202.365.265 equações extraídas de 828.601 documentos, resultaram 374.739 itens.

O nome do conjunto promete variação notacional, e a medição da grafia dos itens desmente a promessa: 53,1% dos itens têm alvo com grafia idêntica byte a byte à da consulta, 37,0% diferem apenas por marcação superficial — espaço, rótulos, `\nonumber`, alinhamento, espaçamento fino e pontuação final —, e só 7,8% exigem variação notacional de fato, como `\le` contra `\leq` ou `\gamma_{\mu}` contra `\gamma_\mu`. Além disso, 36,3% dos itens ligam a consulta a um documento com o qual ela partilha cinco ou mais equações — a mesma obra em duas versões, ou trabalhos companheiros. A medida primária foi então restringida, antes de qualquer modelo medido, ao estrato notacional sem documento quase igual (24.508 itens), com os demais estratos como controle.

**Tabela 23.** Recuperação por equação no estrato notacional. Dois mil itens sorteados, 20.000 documentos no conjunto de candidatos (838.198 equações), escore do documento igual ao máximo sobre suas equações, os quatro sistemas sobre o mesmo conjunto de candidatos, teto de 1,0. Diferença em recall@10 contra o BM25, por bootstrap pareado por item.

| Sistema | recall@1 | recall@10 | MRR | recall@10 − BM25 |
|---|---|---|---|---|
| BM25 [25], equação inteira | 0,7115 | 0,9300 | 0,7902 | — |
| **Bi-encoder do sistema** (23 M, 6 M de pares) | **0,8595** | **0,9550** | **0,8966** | **+0,0250 [+0,0115; +0,039]** |
| ModernBERT-base ajustado (200 mil pares) | 0,8295 | 0,9500 | 0,8760 | +0,0200 [+0,0065; +0,034] |
| GTE-base ajustado (400 mil pares) | 0,8005 | 0,9200 | 0,8450 | −0,0100 [−0,0245; +0,005] |

A premissa não se sustenta para estes modelos: o bi-encoder de 23 M supera o BM25 justamente no estrato que exige variação notacional, por 15 pontos em recall@1, apesar de uma assimetria a favor do BM25 — que lê a equação inteira, enquanto os modelos densos leem 192 tokens (mediana de 81, com 10% acima de 238). Os estratos de controle confirmam que o corte mede o que promete: a vantagem do bi-encoder em recall@1 cresce monotonicamente com a variação notacional, de +0,053 na grafia idêntica para +0,065 na superficial e +0,148 na notacional.

Uma primeira execução desta medição foi anulada: os sistemas foram medidos em processos separados, e cada processo montou um conjunto de candidatos diferente (Seção 5.2).

---

## 5. Discussão

### 5.1 Domínio, capacidade e força do recuperador

O resultado da Seção 4.8 é assimétrico de modo não previsto. O modelo pré-treinado em Física fica abaixo, como recuperador, até de um modelo geral de 23 M sem ajuste — nDCG@10 de 0,3507 contra 0,4761 na Tabela 8 — e é a melhor base de reordenação entre as três testadas. Pré-treinamento de domínio, nas condições medidas, não produziu um bi-encoder competitivo e produziu um cross-encoder competitivo.

Não dispomos de explicação mecanística para a assimetria. A hipótese de menor custo de teste é que o cross-encoder pode explorar interação termo a termo entre consulta e documento, regime em que vocabulário de domínio é diretamente utilizável, enquanto o bi-encoder deve comprimir o documento em um vetor antes de observar a consulta. Trata-se de especulação até que seja medida.

A continuação da Seção 4.8 qualifica a implicação operacional que a versão anterior deste trabalho extraía. O valor da reordenação é condicional à fraqueza do recuperador: com um recuperador cujo recall@100 subiu de 0,516 para 0,633, o mesmo reordenador de domínio deixou de acrescentar informação mensurável, e, dentro dos dez primeiros resultados, rebaixou o alvo nominalmente mais vezes do que o elevou. O controle de domínio contra capacidade da Tabela 14 continua válido como comparação; o que não se sustenta é tomá-lo como justificativa para manter o estágio. Para a recuperação de primeira etapa, a Tabela 10 indica partir da base geral mais forte disponível, com a ressalva de que a base de 109 M comprou com 400 mil pares o mesmo que o volume comprou com 6 milhões, ao custo de 4,4 vezes a inferência.

O resultado também qualifica a prática, comum em trabalhos de domínio, de avaliar um encoder especializado apenas como recuperador. Sob essa avaliação isolada, o PhysBERT seria descartado; seu valor no sistema apareceu somente em outro papel, e só enquanto o recuperador era fraco.

### 5.2 Vieses de amostragem em conjuntos derivados de grafos de citação

Uma auditoria identificou oito ocorrências independentes de amostragem dependente da ordem dos dados, todas no mesmo repositório.

**Tabela 24.** Ocorrências identificadas, com o efeito medido de cada uma.

| Local | Efeito medido | Consequência |
|---|---|---|
| Amostra de revisão do corpus de texto integral | 400 documentos, todos nos 2 primeiros de 277 arquivos — 0,67% do corpus, exclusivamente resumos | Invalidou a amostra destinada a estimar contaminação |
| Avaliação sobre arquivo agrupado | 500 linhas correspondiam a 35 documentos; intervalo de 95% de ±0,159 | Duas conclusões mutuamente contraditórias |
| Divisão treino/validação por posição | 49,6% das âncoras da validação presentes no treino | nDCG@10 de 0,139 para 0,020 sobre documentos inéditos |
| Seleção das primeiras 8 de 44 partições | Partições omitidas continham 57% mais equações em display | Viés no corpus de pré-treinamento, detectado antes do uso |
| Instrumento de conferência do item anterior | 200 linhas provenientes de um único arquivo | Reproduziu o valor enviesado que existia para refutar |
| Conjunto de avaliação principal e conjunto de treinamento | Teto de nDCG@10 de 0,7562; 10,7 vezes menos documentos distintos no treinamento | Inverteu o veredito do critério de aceitação (Seção 4.5) |
| Avaliação de modelagem mascarada por região | As 2.000 primeiras sequências de uma partição cobriam cerca de 4% dos documentos, na ordem de ingestão | Detectada antes da medição da Seção 4.9.2 |
| Conjunto de candidatos da recuperação por equação | Sorteio sobre uma lista cuja ordem mudava entre execuções: quatro sistemas, medidos em processos separados, viram 841.101, 843.127, 836.422 e 831.048 equações | Comparação pareada entre sistemas que não viram os mesmos distratores; medição anulada (Seção 4.10) |

Duas observações parecem generalizáveis. A primeira é que o defeito é invisível por construção: nenhuma das oito ocorrências gera exceção, nenhuma se manifesta na função de perda, e todas produzem valores dentro da faixa esperada — a terceira eleva a métrica. A quinta é particularmente instrutiva, por haver ocorrido dentro do instrumento escrito para detectar a quarta, e a sétima e a oitava, por terem ocorrido depois que as anteriores estavam catalogadas. A oitava foi exposta por uma correção: a medição passou a usar um processo por sistema para contornar o esgotamento de memória de vídeo, e só então cada processo passou a sortear seu próprio conjunto — com todos no mesmo processo, os sistemas teriam partilhado o conjunto por acidente, e o resultado sairia certo pelo motivo errado.

A segunda é que a correção adequada não é a substituição pontual do esquema de amostragem. O caminho de código de reordenação já empregava amostragem aleatória, com a justificativa documentada em comentário, e essa justificativa não se propagou ao caminho de embedding. A correção adotada é uma verificação que interrompe a execução quando o teto do conjunto de avaliação difere de 1,0, com o teto registrado no artefato de saída — isto é, uma propriedade verificada a cada execução, e não uma convenção documentada. No treinamento, a mesma correção valeu +0,020 de nDCG@10 sem custo de computação (Seção 4.6).

Trabalhos que derivam simultaneamente o rótulo positivo, o critério de negativo e a verdade de avaliação de um mesmo grafo estão estruturalmente expostos a esta classe de defeito, por não disporem de uma segunda fonte contra a qual conferir. Recomenda-se reportar, ao lado de qualquer métrica agregada por grupo, o número de grupos distintos efetivamente medidos e o intervalo que ele implica.

### 5.3 Instrumentos registrados antes da coleta, e inválidos para a pergunta

Registrar a regra de decisão antes dos dados protege contra escolher a leitura depois do resultado; não protege contra escolher o instrumento errado. Isso ocorreu três vezes neste trabalho, e em nenhuma o instrumento produzia erro ou valor fora da faixa esperada.

Primeiro, a acurácia de predição mascarada foi registrada como medida primária da comparação de tokenizadores, por ter o maior número de observações pareadas. Ela não compara vocabulários: um tokenizador que divide o texto em unidades menores deixa mais redundância local após o mascaramento e acerta mais sem ser modelo melhor. Num ensaio com modelos de 30 passos, a variante E marcou acurácia maior (0,0237 contra 0,0195) e bits por byte piores (2,890 contra 2,761). O viés apontava contra a hipótese sob teste, de modo que um empate teria sido lido como refutação honesta.

Segundo, o teste de McNemar sobre pertencimento aos k primeiros foi registrado para julgar um reordenador, que atua dentro dos k primeiros: um reordenador que ordenasse perfeitamente as dez primeiras posições não alteraria o recall@10. O bootstrap sobre o nDCG@10 por consulta e a decomposição da Seção 4.8 foram acrescentados depois, declarados como tal, e confirmaram a leitura.

Terceiro, o desempate registrado para os bits por byte era a pseudo-verossimilhança de um token por vez [30], que carrega, mais forte, o mesmo viés do instrumento que devia desempatar [31]. Foi substituída antes de executada.

A esses se soma o quase falso negativo da Seção 4.6, produzido por um valor padrão de linha de comando: um padrão não é um protocolo. A prática adotada em consequência é registrar, ao lado de cada regra, a direção do viés de cada instrumento, e, quando não há instrumento sem viés, cercar o valor com instrumentos de vieses opostos antes de decidir por um terceiro (Seção 4.9.1).

### 5.4 Mecanismos automáticos que dependem da variável sob teste

O pré-treinamento continuado da Seção 4.9.3 expôs uma classe de defeito distinta das duas anteriores: não um instrumento de medida errado, mas um mecanismo automático do próprio treinamento — ou da cadeia que leva o modelo à medida — cujo comportamento dependia da variável sob teste.

O detector de instabilidade julgava a perda média do lote. No braço tratado, parte dos alvos são equações inteiras escondidas, muito mais difíceis que os alvos do sorteio uniforme, e a quantidade delas por lote varia com o sorteio do tratamento. A perda do tratado oscilava com a composição do lote — o desvio padrão da perda registrada foi de 0,077, contra 0,037 no controle —, e o critério de μ + 4σ passou a ler composição como instabilidade. Como o fluxo de dados e a máscara são funções determinísticas de (semente, passo), os lotes dos três acionamentos puderam ser reproduzidos sem GPU: tinham de 3,0 a 4,1 desvios padrão acima da média de alvos de equação inteira em 300 passos sorteados, e dois deles ultrapassavam o máximo da amostra. O detector, portanto, intervinha — com retorno a ponto de verificação, salto de lotes e taxa de aprendizado reduzida — preferencialmente no braço tratado, e por causa do tratamento. Uma assimetria de execução desse tipo não é ruído: correlaciona-se com a variável. A correção julga a perda apenas sobre os alvos uniformes, cuja composição não depende do tratamento; no controle, os dois sinais coincidem. Refeito com ela, o braço tratado treinou sem acionamentos.

Dois defeitos na cadeia até a medida tinham a mesma propriedade de silêncio. O exportador de modelos nomeava os tokens especiais pelos identificadores da convenção dos tokenizadores próprios do projeto; na base de pré-treinamento continuado esses identificadores correspondem a tokens comuns, e o modelo exportado declarava o sinal de pontuação `#` como token de máscara. Pesos, logits e a ida e volta pelo disco conferiam com diferença nula, porque cada conferência comparava o artefato consigo mesmo — e a medida primária da Tabela 21 esconde tokens com o token de máscara declarado. O defeito foi encontrado ao ler o registro de uma exportação local, antes de qualquer medida. O comparador da ablação, por sua vez, fora escrito para o experimento da Seção 4.9.2 e carregava três constantes dele: a escala, a regra e uma ressalva sobre a assimetria de execução daquele experimento. Aplicado ao pré-treinamento continuado, o artefato saiu declarando a escala errada e uma assimetria contra o tratado, quando a deste experimento era a favor. Uma ressalva falsa tem a forma do cuidado e o conteúdo errado, e por isso não costuma ser conferida.

A prática adotada em consequência é tratar como parte do desenho experimental todo mecanismo que atua de modo diferente conforme o braço — e verificar, antes de comparar, que nenhum deles lê um sinal que o tratamento altera.

### 5.5 O encoder de domínio deve ser treinado do zero?

A premissa da Introdução — treinamento a partir do zero justificável no nível de encoders — apoiava-se, para o encoder deste trabalho, em três ingredientes que apenas um modelo próprio teria: tokenizador nativo, objetivo nativo e contexto longo. Os resultados da Seção 4.9, na escala de substituto, enfraquecem os três. A regra de tokenização nativa piorou a modelagem. O objetivo de mascaramento de equações ajudou a recuperação, mas as marcas de equação são derivadas de posições de caractere, e não de identificadores de token: o tratamento aplica-se igualmente ao pré-treinamento continuado de qualquer base, com qualquer tokenizador, e um resultado positivo dele favorece usá-lo, e não usá-lo a partir do zero. E contexto de 8.192 tokens já existe em encoders gerais abertos [26]. Soma-se a evidência da Tabela 10, de que a base importa mais que o volume de ajuste.

A comparação de referência foi então executada, e é o resultado mais consequente desta seção: a mesma base geral, ajustada como bi-encoder sem qualquer pré-treinamento no domínio, supera o braço tratado em 0,0558 de nDCG@10 (Tabela 20). O encoder treinado do zero perde para o que já existe de graça, na tarefa para a qual foi treinado.

Isso fecha, na prática, a opção de treinar o encoder a partir do zero sob este orçamento. Um modelo próprio de 150 M teria de cobrir três desvantagens somadas — volume de pré-treinamento, número de parâmetros e a diferença de 0,0558 agora medida —, e nenhuma evidência deste trabalho sugere que o objetivo específico de domínio, sozinho, as cubra: ele vale +0,0840, medido na mesma escala e no mesmo protocolo.

Restava o pré-treinamento continuado dessa base com o objetivo de mascaramento de equações, e ele foi executado (Seção 4.9.3): dois braços a partir do ModernBERT-base, uma variável, 0,4 bilhão de tokens cada, cerca de 7,5 horas por braço em duas T4. O objetivo mostrou-se real e pequeno. Na recuperação, o tratado supera o controle por +0,0076, e a base de partida por +0,0103; na predição de tokens, nada.

A resposta à pergunta desta seção, porém, veio de uma medição que o desenho original não previa. A base do pré-treinamento continuado foi escolhida por compartilhar a arquitetura planejada para o encoder próprio e ter contexto de 8.192 tokens, e não por ter sido comparada, na mesma régua, às demais bases gerais disponíveis. Quando o GTE-base foi ajustado nos mesmos 200 mil pares, superou o ModernBERT-base por 0,069 — cerca de sete vezes o que o pré-treinamento continuado inteiro acrescentou. O melhor encoder desta escala não passou por pré-treinamento no domínio algum.

Isso fecha as duas opções de encoder próprio sob este orçamento. Treinar do zero perde para a base geral sem pré-treinamento; pré-treinar continuamente a base escolhida rende menos que escolher outra base. O padrão do objetivo de domínio através das escalas é também informativo: +0,084 sobre bases de 48 M treinadas do zero, +0,0076 sobre uma de 150 M com 2 trilhões de tokens gerais, e medida primária nula nas duas. O efeito encolhe à medida que a base se fortalece, o que é o padrão esperado de uma lacuna que a base forte já cobre. O argumento do viés indutivo nativo permanece sem teste direto, e os resultados aqui não indicam que valha o seu preço.

### 5.6 Implicações práticas

Cinco padrões emergiram, nenhum relativo a arquitetura.

Medir a fonte antes de adquiri-la. A fatia de corpus que viabiliza o objetivo de treinamento específico de domínio já se encontrava em disco havia dezessete dias quando se recomendou a aquisição paga de acesso ao fonte do arXiv como pré-requisito. O que faltava não era dado nem orçamento, mas a medição da Tabela 4, de custo desprezível.

Medir a vazão real antes de planejar. A vazão medida em acelerador T4 foi de 181,6 pares por segundo, contra 20 a 26 na GPU local — aproximadamente 69.700 contra 7.700 tokens por segundo. Isso converte um pré-treinamento de 20 bilhões de tokens de 37 dias para cerca de 80 horas, alterando a viabilidade, e não apenas o cronograma.

Empregar comparação pareada. Com 256 candidatos, o erro padrão é de ±0,031 e a margem mínima detectável, de aproximadamente 0,061, contra uma diferença de interesse de 0,004; nem 4.000 candidatos alterariam essa conclusão. O teste pareado sobre os mesmos itens é o que torna a diferença acessível, e é indispensável em regimes de baixo orçamento, nos quais o tamanho do conjunto de avaliação é limitado.

Medir os braços na mesma sessão. Um valor histórico carrega a versão do avaliador que o produziu; a Tabela 13 mostra um erro de sete vezes evitado por essa única decisão de desenho, tomada antes do resultado.

Comparar as bases candidatas antes de investir numa delas. O pré-treinamento continuado custou cerca de 20 horas de sessão de T4, incluída a execução interrompida pelo detector, sobre uma base escolhida por argumento; a comparação entre bases, feita depois e no mesmo protocolo, custou 1 hora e 17 minutos e mostrou uma diferença sete vezes maior que o ganho do pré-treinamento. A ordem inversa teria poupado o investimento, ou o dirigido para a base certa.

---

## 6. Limitações

Um domínio e um grafo de citação: Física e OpenAlex. Nada aqui estabelece que as taxas medidas se transfiram a outros domínios ou fontes.

O conjunto de avaliação é próprio e não dispõe de julgamento humano de relevância. A definição operacional adotada — relevante equivale a citado — é uma aproximação com viés conhecido: favorece artigos citáveis e áreas com cultura de citação densa, e não captura relevância não citada. Os valores de nDCG deste trabalho não são comparáveis aos de benchmarks com anotação humana.

Os resultados das Tabelas 6 e 11 são históricos, pelo defeito de protocolo da Seção 4.5 e pela troca de recuperador da Seção 4.7. A comparação controlada da Tabela 14 foi medida com o recuperador antigo, e o ganho que ela mostra não se reproduz no sistema atual.

A comparação da Tabela 20 não é de uma variável: tamanho, tokenizador e volume de pré-treinamento diferem simultaneamente, e os três braços usam 200 mil pares — metade da receita das Tabelas 9 e 10, de modo que nenhum dos três valores se compara com os de lá. Ela responde se o artefato produzido pela hipótese compete com o que existe publicamente, e não qual dos três fatores produz a diferença.

Os resultados de pré-treinamento das Seções 4.9.1 e 4.9.2 são de substitutos de 48 M a 0,6 bilhão de tokens, 3,3 vezes abaixo do orçamento preparado para a ablação, e os da Seção 4.9.3, de um pré-treinamento continuado de 0,4 bilhão de tokens. Todos têm uma semente por braço, e cada ablação tem uma assimetria de execução não controlada — a favor de E na Seção 4.9.1, contra o tratado na Seção 4.9.2 e a favor do tratado na Seção 4.9.3. Nesta última, com um efeito de recuperação de +0,0076 e o limite inferior do intervalo em +0,0013, a assimetria é da ordem de grandeza que poderia explicar parte do efeito; ela não foi medida. A medida primária de modelagem mascarada tem viés declarado a favor do controle nas duas escalas. O encoder próprio de 150 M, treinado do zero, não foi treinado.

A Tabela 22 compara bases gerais com uma variável, a 200 mil pares — metade da receita das Tabelas 9 e 10. A ordem entre duas bases a esse volume pode não se manter com mais pares. A base vencedora foi levada a 1 milhão de pares e empatou com o recuperador do sistema (Tabela 10); a comparação com a mesma base a 6 milhões, que isolaria o volume, não foi executada.

O conjunto de recuperação por equação usa como consulta uma equação transcrita verbatim de um artigo, e não uma consulta escrita por um usuário; a vantagem medida do bi-encoder não se transfere, sem nova medição, a consultas que misturem notação e linguagem natural. A medida primária usa um sorteio de 2.000 itens de um estrato de 24.508, e a restrição a esse estrato, embora registrada antes de qualquer modelo medido, foi decidida depois da medição da grafia dos itens.

A taxa de contaminação do corpus filtrado não está medida. As taxas de falso positivo que justificam o limiar de decisão foram medidas sobre resumos do arXiv, ao passo que o corpus filtrado é de texto integral, com outra distribuição. Uma amostra estratificada de 200 resumos e 200 textos integrais está em julgamento humano, com o escore do classificador oculto até cada decisão.

O efeito de acrescentar a fatia de matemática e ciência da computação ao corpus de um modelo de Física não está medido.

A Seção 4.7 não mede o falso negativo residual. O filtro de co-citação remove uma classe nomeável de falso negativo — 9,1% dos negativos minerados eram co-citados com o positivo —, mas o tamanho do residual permanece desconhecido.

A taxa de aprendizado não foi reajustada entre escalas de modelo no experimento da Seção 4.8, o que constitui variável não controlada, declarada antes da observação do resultado.

A reexecução completa do pipeline não reproduz hoje os artefatos avaliados. O conjunto de pares de citação, reconstruído com o código atual, conserva apenas 2,1% das âncoras de validação, porque o critério de embaralhamento mudou depois da construção; os arquivos avaliados foram restaurados byte a byte, e todos os números deste trabalho referem-se a eles. A entrada bruta do OpenAlex que fornece contagens de referências e de citações não está mais disponível localmente, e uma reexecução da construção do índice gravou valores nulos sobre 14.052.319 referências com término bem-sucedido; o dano foi revertido por cópia de segurança, e uma verificação passou a recusar regressões desse tipo antes da gravação. Os parâmetros das etapas executadas antes dessas correções são reconstruídos a partir do código, e não capturados durante a execução.

---

## 7. Conclusão

Este trabalho reporta a construção de um sistema de recuperação de literatura de Física sob orçamento de computação nulo, com resultados positivos, negativos e em aberto discriminados.

Estão estabelecidos: um corpus de 27,75 bilhões de tokens de Física e um índice de metadados de 1,59 milhão de registros, construídos a custo nulo e verificáveis por uma cadeia de hashes sobre 52,40 GB; um classificador de domínio com acurácia de 0,954 e taxa de falso positivo caracterizada por proximidade de domínio; um discriminante válido para integridade de notação matemática em corpora; um bi-encoder de 23 M, ajustado por citação, que supera um modelo geral de 335 M em nDCG@10, recall@10 e MRR sob um protocolo de teto verificado; a evidência de que a base do recuperador vale, a 400 mil pares, quinze vezes o volume de ajuste; e a demonstração, com controle de tamanho e arquitetura, de que o corpus de pré-treinamento da base determina se um cross-encoder acrescenta informação a uma fusão de recuperadores — acompanhada da demonstração de que esse acréscimo desaparece quando o recuperador melhora.

Na escala de substituto, dois resultados de pré-treinamento são negativos — regras de pré-tokenização de LaTeX pioram a modelagem, e mascarar equações inteiras não melhora a predição de tokens de equação — e um é positivo: o mesmo mascaramento melhora a recuperação em 0,084 de nDCG@10 após ajuste contrastivo.

E dois resultados delimitam os três. A mesma receita de ajuste, aplicada a uma base geral de 150 M sem qualquer pré-treinamento no domínio, produz 0,5270 de nDCG@10 contra 0,4712 do braço tratado; e o pré-treinamento continuado dessa base com o mesmo objetivo repete o ganho de recuperação, onze vezes menor, enquanto outra base geral, também sem pré-treinamento no domínio, a supera por um valor sete vezes maior. O objetivo específico de domínio é real e pequeno, e encolhe à medida que a base se fortalece. Sob este orçamento, a decisão que mais pesa num encoder de recuperação de Física é a escolha da base geral, e ela custa uma fração do que custa qualquer pré-treinamento.

Na recuperação por equação, o resultado é o oposto da premissa com que o conjunto foi desenhado: o bi-encoder de 23 M supera a busca léxica justamente onde a notação varia, e a vantagem cresce com a variação.

A base geral mais forte, ajustada com um sexto dos pares, empata com o recuperador atual do sistema e custa 4,4 vezes mais para embutir; o sistema permanece como está. Permanecem em aberto a contaminação do corpus filtrado e o efeito de acrescentar texto de domínios vizinhos.

Os resultados metodológicos de maior alcance são os das Seções 5.2 a 5.4. Oito ocorrências da mesma falha de amostragem, três instrumentos registrados de antemão e inválidos para a pergunta e um mecanismo de segurança que intervinha preferencialmente no braço tratado, por causa do tratamento — nenhum deles com qualquer sintoma além do próprio número — sugerem que protocolos de baixo orçamento exigem verificação ativa da unidade de amostragem, da direção de viés de cada instrumento e da independência, em relação à variável sob teste, de todo mecanismo que atua sobre os braços, e não apenas convenção documentada ou registro prévio.

---

## Disponibilidade de dados e código

O código e a documentação de projeto são públicos sob licença permissiva. Cada etapa do pipeline registra manifesto com hashes das entradas e saídas, parâmetros e identificador de commit, o que permite reconstrução a partir das fontes públicas, com as ressalvas da Seção 6.

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

[30] Salazar, J.; Liang, D.; Nguyen, T. Q.; Kirchhoff, K. Masked Language Model Scoring. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (ACL 2020), p. 2699–2712, 2020.

[31] Kauf, C.; Ivanova, A. A. A Better Way to Do Masked Language Model Scoring. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics* (ACL 2023), v. 2: Short Papers, p. 925–935, 2023.
