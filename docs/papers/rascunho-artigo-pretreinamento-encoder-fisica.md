# Pré-treinamento de encoders para texto de Física em LaTeX sob restrição severa de computação: tokenização, mascaramento de equações e escolha da base

Vinicius Sanchez

*Pesquisa independente*

*Versão 0.1 — 2 de outubro de 2026*

---

## Resumo

> Sob a relação de computação ótima, um modelo de linguagem de Física de 8 bilhões de parâmetros exigiria de cinco a dez vezes mais texto do que o domínio oferece; a premissa corrente é treinar do zero só encoders, onde o dado sobra, e obter o resto por pré-treinamento continuado de uma base geral. Este trabalho testa a premissa no próprio nível de encoders, sob orçamento de computação nulo, com três ablações de uma variável e uma semente por braço. Em substitutos de 48 milhões de parâmetros treinados com 0,6 bilhão de tokens de Física, a pré-tokenização proposta para LaTeX — que o experimento não separa da ausência de fronteira de palavra — piora a modelagem em 0,047 bit por byte, resultado que só o terceiro de três instrumentos de viés conhecido pôde decidir. Mascarar equações inteiras não melhora a predição de tokens de equação e melhora a recuperação após ajuste contrastivo em 0,084 de nDCG@10; ainda assim, uma base geral de 150 M sem pré-treinamento no domínio supera o braço tratado em 0,056. O pré-treinamento continuado dessa base com 0,4 bilhão de tokens e o mesmo objetivo acrescenta +0,010, sugestivo e não estabelecido, enquanto trocá-la por outra base geral, já treinada contrastivamente, acrescenta +0,069. Sob este orçamento, a escolha da base pesou mais que qualquer pré-treinamento no domínio. Reportam-se ainda três instrumentos registrados de antemão e inválidos para a pergunta, e um detector de instabilidade cujo acionamento dependia da variável sob teste.

**Palavras-chave:** tokenização de LaTeX; modelagem de linguagem mascarada; mascaramento de equações; pré-treinamento continuado; bits por byte; reprodutibilidade.

---

## 1. Introdução

Sob a relação de computação ótima de Hoffmann et al. [1], um modelo de 8 bilhões de parâmetros requer cerca de 160 bilhões de tokens de treinamento, e o funil de aquisição da literatura de Física legalmente adquirível a custo zero rende entre 15 e 30 bilhões após deduplicação e triagem de licença [2] — uma escassez de cinco a dez vezes. A resposta sustentada pela literatura recente é treinar a partir de inicialização aleatória apenas onde o dado é excedente — encoders na faixa de 10⁸ parâmetros — e obter capacidades maiores por pré-treinamento continuado sobre uma base geral forte. O Galactica [3], a tentativa mais visível de treinar um modelo científico de fundação a partir do zero, teve a demonstração pública retirada três dias após o lançamento, com a geração de artigos fictícios, por vezes atribuídos a autores reais, entre os defeitos apontados [4]; Minerva [5], Llemma [6] e DeepSeekMath [7] foram obtidos por pré-treinamento continuado sobre bases gerais, e todos reportam ganhos substanciais em raciocínio quantitativo.

Para um encoder de Física, a premissa apoiava treinar do zero com base em três ingredientes que só um modelo próprio teria: um tokenizador nativo para LaTeX, um objetivo de pré-treinamento nativo — mascarar equações inteiras — e contexto longo. Este trabalho testa os três sob orçamento de computação nulo, com uma GPU de consumo de 8 GB e cotas gratuitas de aceleradores em nuvem, e mede o resultado onde ele importa para o sistema que o motiva: a recuperação de artigos de Física por supervisão de citação, descrita no trabalho complementar [2].

As contribuições são as seguintes.

1. Evidência sobre tokenização de texto em LaTeX em duas camadas que apontam em sentidos opostos: a métrica intrínseca favorece a pré-tokenização proposta, e a modelagem de linguagem, medida em bits por byte, a rejeita (Seções 4.1 e 4.3).
2. Um procedimento para comparar tokenizadores com modelos mascarados, que não fornecem verossimilhança exata: cercar o valor com dois instrumentos de vieses opostos e decidir por um terceiro (Seções 3.5 e 4.3).
3. Uma ablação do mascaramento de equações inteiras em duas escalas — do zero a 48 M e em pré-treinamento continuado a 150 M —, com o efeito sobre a recuperação muito menor na segunda e a medida primária de modelagem negativa na primeira e não decidida na segunda (Seções 4.4 e 4.5).
4. A constatação de que, sob este orçamento, a escolha da base geral vale mais que o pré-treinamento no domínio inteiro (Seções 4.5 e 5.1).
5. Três instrumentos registrados antes da coleta e inválidos para a pergunta, e mecanismos automáticos — um detector de instabilidade, um exportador, uma ressalva herdada — cujo comportamento dependia da variável sob teste (Seções 5.2 e 5.3).

Os resultados são de substitutos de 48 M e de um pré-treinamento continuado de 0,4 bilhão de tokens sobre uma base de 150 M, com uma semente por braço. O encoder próprio de 150 M planejado não foi treinado, e nada aqui é veredito sobre ele.

---

## 2. Trabalhos relacionados

**Tokenização.** A codificação por pares de bytes [8] e o modelo de unigramas [9] são as duas alternativas dominantes. Bostrom e Durrett [10] reportam vantagem do modelo de unigramas em desempenho a jusante em linguagem natural, atribuída a melhor alinhamento morfológico. Não localizamos evidência publicada equivalente para texto em LaTeX; a Seção 4.1 apresenta uma comparação intrínseca, e a Seção 4.3 mostra, para outra variável do tokenizador, que a métrica intrínseca não antecipa a extrínseca.

**Pontuação de modelos mascarados.** Comparar tokenizadores extrinsecamente com modelos mascarados exige pontuar texto com um modelo que não é autorregressivo. A pseudo-verossimilhança de Salazar et al. [11] mascara um token por vez; Kauf e Ivanova [12] mostram que ela superestima a probabilidade de palavras divididas em vários tokens e propõem mascarar também os tokens seguintes da mesma palavra. A Seção 3.5 descreve como esse viés, e o viés oposto, foram usados para cercar o resultado.

**Encoders e taxa de mascaramento.** O ModernBERT [13] é um encoder geral com contexto de 8.192 tokens, treinado com cerca de 2 trilhões de tokens e taxa de mascaramento de 30%, acima dos 15% convencionais; Wettig et al. [14] mostram que mascarar 40% supera 15% em modelos do porte do BERT-large. O GTE [15] é um modelo de embedding geral treinado contrastivamente em larga escala. Em Física, o PhysBERT [16] foi pré-treinado do zero sobre 1,2 milhão de artigos do arXiv; o SciBERT [17], sobre literatura biomédica e de computação.

---

## 3. Materiais e métodos

### 3.1 Corpus de pré-treinamento

O texto vem da fatia arXiv do RedPajama [18], construída a partir do fonte LaTeX e selecionada por pertinência exata ao índice de artigos de Física de [2]: 828.601 documentos distintos, dos quais 84,9% contêm ambiente de equação em display, contra 0,0% na fatia de texto integral extraída de PDF, em que as equações foram removidas. O corpus de pré-treinamento tem 2.001.270.262 tokens de 143.810 documentos, em 244.295 sequências de 8.192 tokens, com 9 de 44 partições sorteadas por semente registrada. A conferência sobre 300 sequências amostradas do binário efetivamente lido pelo treinamento indica 37,8% de tokens matemáticos contra 38,8% no corpus bruto, 25,0% em display contra 25,2%, taxa efetiva de mascaramento de 0,3000 contra 0,3000 requerida, e fração tratada de 0,903 a 8.192 tokens.

A restrição do tratamento a equações em display, e não a toda notação matemática, decorre de medição: em 120 documentos, as equações em display têm mediana de 79 tokens, e as sequências em linha, de 7 — o que corresponde a uma variável isolada, e não a uma equação. Tratar toda notação faria a ablação comparar condições substancialmente equivalentes.

Uma fatia adicional, de matemática e ciência da computação da mesma fonte, foi coletada e mantida separada: 687.907 documentos e cerca de 11,2 bilhões de tokens, sem sobreposição com a fatia de Física. Ela dobra o LaTeX íntegro disponível, de 42,15 para 87,0 bilhões de caracteres, mas mais da metade desse total passaria a vir de fora do domínio; nenhuma medição deste trabalho avalia esse compromisso, e as fatias não foram misturadas.

### 3.2 Tokenizadores

Seis tokenizadores foram treinados sobre 200.000 resumos do arXiv e avaliados em 5.000 reservados: quatro variantes de codificação por pares de bytes e uma de unigramas com a pré-tokenização proposta — sequências de LaTeX como `\frac`, `\begin{…}`, `^{` e `_{` isoladas como pré-tokens e nenhuma fronteira de palavra —, uma de pares de bytes com a pré-tokenização genérica, por palavra e pontuação, e um tokenizador geral de referência (Tabela 1).

### 3.3 Encoder planejado e substitutos

O encoder planejado, que não foi treinado (Seção 5.1), é bidirecional, de 142 milhões de parâmetros (150 M nominais), com vocabulário de 40.960, comprimento de sequência de 8.192 e objetivo de modelagem de linguagem mascarada a 30%, como no ModernBERT, sem predição de sentença seguinte, com agendamento de taxa de aprendizado do tipo *warmup–stable–decay*. A adição específica de domínio, sujeita a ablação, é mascarar uma equação em display inteira numa fração `p_equacao` dos exemplos que contêm equação elegível, mantendo igual entre os braços o orçamento total de tokens mascarados. Equações cortadas pela janela, no início ou no fim, não são elegíveis.

Os experimentos usam substitutos reduzidos da mesma arquitetura: 48 milhões de parâmetros, contexto de 1.024 tokens, lote lógico de 64 sequências e orçamento de 0,6 bilhão de tokens por braço, treinados em acelerador T4 a cerca de 20.500 tokens por segundo, em pouco mais de 8 horas por braço. Nessa escala, a matriz de embedding corresponde a 43,7% dos parâmetros. Os braços são exportados e medidos localmente.

### 3.4 Pré-treinamento continuado

O mesmo objetivo foi aplicado ao ModernBERT-base [13] (150 M), com o tokenizador dele: 0,4 bilhão de tokens por braço — o que cabe numa sessão de 9 horas —, contexto de 1.024, lote lógico de 64 sequências, taxa de aprendizado de pico de 10⁻⁴, `p_equacao` de 0,0 no controle e 0,6 no tratado, e todos os demais valores iguais, conferidos por comparação da árvore sintática das células de treinamento. A fatia de treinamento foi retokenizada com o tokenizador da base (423 milhões de tokens), e a de avaliação vem de uma partição que nenhum braço viu. Os braços treinam em duas T4 com paralelismo de dados, em que a máscara de cada micro-passo é função apenas de (semente, índice do micro-passo); um teste confirma que dois processos terminam com os mesmos pesos que um, a menos de 10⁻⁵. A vazão medida foi de 15.200 a 15.900 tokens por segundo nas duas placas, contra 15.600 projetados a partir do custo relativo medido em CPU.

### 3.5 Avaliação

**Bits por byte.** A comparação entre tokenizadores usa a informação atribuída pelo modelo ao texto dividida pelo número de bytes, e não de tokens; a acurácia de predição mascarada não compara vocabulários, pela razão da Seção 5.2. Um modelo mascarado não fornece verossimilhança exata, e cada aproximação tem viés de direção conhecida. Três instrumentos foram usados, cada um com a regra de leitura registrada antes do respectivo resultado: (1) mascarar 15% dos tokens, que favorece o tokenizador de tokens mais curtos, porque esconde pedaços menores cercados de fragmentos visíveis da mesma palavra; (2) mascarar unidades de texto delimitadas por cortes presentes nas duas segmentações e pontuar cada token independentemente, que favorece o tokenizador de tokens mais longos, porque a soma das entropias marginais é maior ou igual à entropia conjunta e a folga cresce com o número de tokens por unidade; e (3) a pseudo-verossimilhança da esquerda para a direita dentro de cada unidade, adaptada de Kauf e Ivanova [12], que recompõe a verossimilhança conjunta pela regra da cadeia e deixa um resíduo leve a favor do tokenizador de tokens mais longos. Os instrumentos 1 e 2 cercam o valor; o 3 decide.

**Modelagem mascarada por região.** Tokens de equação e de prosa são mascarados uniformemente a 15%, nas mesmas posições para os dois braços, sobre 2.000 sequências sorteadas de uma partição que nenhum braço viu, com contexto de 1.024. A medida primária do mascaramento de equações é a diferença das diferenças: a vantagem de acurácia em equação sobre prosa no braço tratado, menos a mesma vantagem no controle. A diferença simples não serve, porque tokens de LaTeX são localmente redundantes: o ModernBERT-base, que nunca viu mascaramento de equações, acerta 0,8765 em tokens de equação contra 0,7480 em prosa, uma vantagem de 0,1286 sem tratamento algum (medida com contexto de 512, em outra fatia).

**Recuperação.** O protocolo de comparação de encoders de [2]: recuperar o documento citado a partir do citante, dentro de 2.000 pares de validação sorteados e deduplicados, com teto de 1,0 verificado, entradas truncadas a 192 tokens e agregação por média. Cada encoder é medido sem ajuste e de novo depois de ajustado como bi-encoder, com perda contrastiva e lote de 128, sobre os mesmos 200 mil pares de citação sorteados (120.002 documentos citados). O ponto de verificação reportado é escolhido numa avaliação interna sobre os primeiros 1.000 dos mesmos 2.000 pares, e não há conjunto de teste separado (Seção 6).

**Detecção de instabilidade.** Uma perda acima de μ + 4σ da janela de 100 passos anteriores, ou uma norma de gradiente acima de dez vezes a mediana móvel, aciona o retorno ao último ponto de verificação, o salto dos lotes suspeitos e 500 passos com taxa de aprendizado reduzida à metade; três acionamentos em 5.000 passos interrompem o treinamento. No braço tratado do pré-treinamento continuado, a perda do critério é calculada apenas sobre os alvos do sorteio uniforme, e não sobre os da equação inteira, pela razão da Seção 5.3; o braço tratado de 48 M foi treinado com o critério original.

**Sonda de estrutura tensorial.** Cada item é um trio sobre o mesmo texto: uma expressão base (`T^{\mu\nu}`), uma variante estrutural com os mesmos símbolos e estrutura diferente (`T_{\mu\nu}`) e uma variante renomeada com a mesma estrutura e símbolos diferentes (`T^{\alpha\beta}`). O item é acertado quando a base é mais próxima da renomeada que da estrutural. São 72 itens, em 4 famílias, 6 tensores e 3 textos. Por semelhança de superfície a variante estrutural é a mais próxima, e um modelo de trigramas de caracteres acerta 0 de 72; a escala resultante tem a superfície em 0, o acaso em 0,5 e a estrutura acima dele. Nenhum encoder existente avaliado alcança 0,5: ModernBERT-base 0,222, SciBERT 0,208, PhysBERT 0,028 e all-MiniLM-L6-v2, o único treinado contrastivamente, 0,000.

### 3.6 Análise estatística

O bootstrap reamostra sempre a unidade que carrega a dependência, e não a observação elementar: documentos, e não posições, nos bits por byte; sequências, e não tokens, na modelagem mascarada por região; itens, na recuperação. Tratar as unidades menores como independentes produziria intervalo artificialmente estreito. Comparações de recuperação sobre o evento "o alvo alcançou as k primeiras posições" usam o teste de McNemar exato [19] sobre pares discordantes. Os intervalos condicionam nos modelos treinados e medem a variância de amostragem, e não a de treinamento: cada braço foi treinado uma vez.

---

## 4. Resultados

### 4.1 Tokenização: a métrica intrínseca

**Tabela 1.** Tokenizadores treinados sobre 200.000 resumos do arXiv e avaliados em 5.000 reservados. "Fertilidade" é a razão entre tokens e palavras; as colunas de razão são relativas ao tokenizador de referência (variante F). "Regras" indica a pré-tokenização proposta, em oposição à genérica; a fertilidade é medida sobre o documento inteiro, matemática incluída. "LaTeX unitário" conta, entre três sequências LaTeX de teste, quantas são representadas por um único token.

| Variante | Algoritmo | Vocabulário | Regras | Fertilidade | Razão | Tokens/eq. | Razão | LaTeX unitário |
|---|---|---|---|---|---|---|---|---|
| A | BPE | 40.960 | sim | 0,9620 | 0,674 | 7,31 | 0,732 | 2/3 |
| B | Unigrama | 40.960 | sim | 0,9816 | 0,688 | 8,36 | 0,837 | 0/3 |
| C | BPE | 32.768 | sim | 0,9973 | 0,699 | 7,49 | 0,750 | 2/3 |
| D | BPE | 65.536 | sim | 0,8966 | 0,629 | 6,97 | 0,698 | 2/3 |
| E | BPE | 40.960 | não | 1,3245 | 0,929 | 9,46 | 0,947 | 0/3 |
| F | referência | 151.643 | — | 1,4263 | 1,000 | 9,99 | 1,000 | 0/3 |

Todas as variantes da Tabela 1 preservam o texto na decodificação e mantêm razão de fertilidade abaixo de 1,25 nas 35 subáreas medidas. Três resultados merecem registro.

Primeiro, a pré-tokenização é a variável de maior efeito: A produz 27% menos tokens por palavra e 23% menos por trecho matemático que E. As duas diferem, porém, em mais que as regras de LaTeX — nas variantes A a D não há fronteira de palavra, e o BPE funde através de espaços, o que explica fertilidades abaixo de um token por palavra; em resumos, onde poucas palavras contêm sequência de controle, é essa segunda diferença que responde pela maior parte do ganho. E é a única variante que não atinge a meta de fertilidade.

Segundo, a codificação por pares de bytes produz menos tokens que o modelo de unigramas em texto de Física, com margem seis vezes maior em equações (13%) do que no documento inteiro (2%). O resultado é de compressão, e não contraria Bostrom e Durrett [10], cuja vantagem do modelo de unigramas é medida em desempenho a jusante; essa comparação extrínseca não foi feita aqui. Um indício do mecanismo: o modelo de unigramas não reteve nenhuma das três sequências LaTeX de teste como token único, enquanto a codificação por pares de bytes reteve duas — compatível com a poda iterativa do primeiro remover essas unidades e a fusão incremental do segundo preservá-las, hipótese que três sequências não testam.

Terceiro, a métrica intrínseca não pode escolher o tamanho do vocabulário: em BPE treinado sobre o mesmo corpus, as fusões do vocabulário menor são prefixo das do maior, e a fertilidade decresce com o vocabulário por construção. A medição acrescenta a magnitude — 6,8% menos tokens de 40.960 para 65.536. O compromisso relevante está fora da métrica: a matriz de embedding corresponde a 22,1% dos parâmetros do encoder planejado com vocabulário de 40.960 (31,5 M de 142,4 M) e a 31,2% com 65.536. A comparação extrínseca entre tamanhos não foi executada: a 48 M ela carregaria um confundidor de capacidade.

Dois resultados restringem a leitura. A meta de fertilidade em equações, fixada em 0,65 vez o tokenizador de referência, não é atingida por nenhuma variante (melhor valor, 0,698), provavelmente porque resumos contêm só matemática em linha curta. E a métrica intrínseca superestima o efeito disponível: no corpus de treinamento, de texto integral, a variante E gasta 15.673 tokens por documento contra 13.916 da A — 12,6% a mais, e não os 37,7% que a fertilidade medida em resumos sugere.

### 4.2 Verificação do laço de treinamento

Um modelo de linguagem mascarada não treinado prediz distribuição quase uniforme, de modo que a entropia cruzada inicial deve ficar próxima do logaritmo do tamanho do vocabulário, com pequeno excesso devido à inicialização. O valor medido é 10,7343, contra ln(40.960) = 10,6204 — 1% acima. É o único indicador que distingue um mascaramento correto de um que apague os alvos ou os derive da entrada já mascarada, condições que não produzem nenhum outro sintoma observável.

Um defeito no marcador de equações ilustra por que contadores de manipulação são parte do desenho. A expressão regular de detecção usava âncora de início de cadeia em conjunto com casamento a partir de posição arbitrária; a âncora continua referindo-se ao início real da cadeia, de modo que nenhuma equação iniciada após o primeiro caractere era detectada. O efeito foi de 0 documentos tratados em 120, contra 91,7% de documentos com equação em display segundo um segundo instrumento, e foi a discordância entre os dois que localizou o erro. Sem o contador de fração tratada, a ablação teria comparado duas condições aleatórias e reportado ausência de efeito.

Os braços foram treinados em imagem de nuvem com `transformers` 5.0 e medidos localmente com a versão 4.48; como a versão antiga ignora chaves desconhecidas da configuração e usaria valores padrão, os modelos foram reexportados localmente, com diferença de logits nula contra a exportação original.

### 4.3 Pré-tokenização de LaTeX

As variantes A e E da Tabela 1 têm o mesmo algoritmo, o mesmo tamanho de vocabulário, a mesma arquitetura e a mesma contagem de parâmetros, e diferem na pré-tokenização em dois pontos que este experimento não separa: A isola as sequências de LaTeX como pré-tokens e não impõe fronteira de palavra; E usa a segmentação genérica. O que se compara é a pré-tokenização proposta como um todo contra a genérica. As variantes C e D ficam fora porque, a 48 M, vocabulários de tamanhos diferentes dão contagens de parâmetros diferentes. O protocolo iguala tokens, e não texto: como E gasta 12,6% mais tokens por documento, A vê 12,5% mais texto pelo mesmo custo, e é essa a vantagem sob teste.

**Tabela 2.** Bits por byte (menor é melhor) por três instrumentos, cada um com a regra de leitura registrada antes do próprio resultado. As colunas A e E são agregadas sobre todos os bytes escondidos; a coluna A − E é a média das diferenças por documento, com intervalo por bootstrap pareado por documento, e por isso não coincide com a subtração das duas colunas.

| Instrumento (Seção 3.5) | Documentos | Viés favorece | A (com regras) | E (sem regras) | A − E |
|---|---|---|---|---|---|
| 1 · 15% dos tokens mascarados | 2.000 | E | 0,6653 | 0,5878 | +0,078 [+0,074; +0,083] |
| 2 · unidades comuns, pontuação independente | 2.000 | A | 1,3096 | 1,3680 | −0,051 [−0,056; −0,046] |
| 3 · unidades comuns, da esquerda para a direita | 1.000 | A, resíduo leve | 0,8969 | 0,8500 | +0,047 [+0,043; +0,050] |

Na Tabela 2, os instrumentos 1 e 2 anulam-se, cada um a favor do braço que favorece, com intervalos estreitos de sinais opostos: um instrumento enviesado com poder de sobra encontra o viés com a mesma confiança com que encontraria o efeito. O instrumento 3 cai dentro do intervalo que os dois cercam, como a teoria prevê, e a queda do instrumento 2 para o 3 é maior em E (−0,52) que em A (−0,41), o que corresponde ao mecanismo nomeado: E, com mais tokens por unidade, pagava mais a folga entre entropias marginais e conjunta. Pela regra registrada, E à frente no instrumento 3, apesar do resíduo a favor de A, rejeita a esta escala a pré-tokenização proposta — o conjunto das regras de LaTeX e da ausência de fronteira de palavra. Unidades de mais de 8 tokens são 6% das unidades e 53% dos bytes; não foi medido quanto delas é LaTeX longo sem espaços e quanto é prosa em que os tokens de A atravessam espaços.

As medidas secundárias, registradas como incapazes de reverter a primária, não a acompanham. Na recuperação sem ajuste, A fica à frente (nDCG@10 de 0,0296 contra 0,0171; E − A = −0,0125 [−0,020; −0,005]; recall@1 com McNemar p = 0,004), com os dois braços perto do piso. Na sonda tensorial não há diferença (0,417 contra 0,333; 19 contra 13 discordantes, p = 0,38). Uma assimetria de execução favorece E e não foi controlada: o braço A sofreu um acionamento do detector que descartou cerca de 3,3% do orçamento e reduziu a taxa de aprendizado à metade por 500 passos; E treinou sem incidentes.

### 4.4 Mascaramento de equações inteiras

O controle é o braço E da Seção 4.3 (`p_equacao` = 0); o tratado difere dele apenas em `p_equacao` = 0,6, conferido por comparação da árvore sintática das duas células de treinamento. A 1.024 tokens, 42% das janelas não contêm nenhuma equação em display que comece nelas, e a fração tratada, de 0,903 a 8.192 tokens, cai para cerca de 0,55; o valor de 0,6 foi escolhido antes do treinamento para que cerca de um terço dos exemplos mascarasse uma equação inteira. Uma equação típica de 75 tokens corresponde a cerca de um quarto do orçamento de mascaramento de uma janela de 1.024 (307 tokens).

**Tabela 3.** Checagens de manipulação e medida primária. Duas mil sequências sorteadas, as mesmas posições e regiões nos dois braços. Intervalos por bootstrap pareado por sequência.

| Medida | Controle | Tratado | Tratado − controle |
|---|---|---|---|
| Fração de exemplos com equação elegível (limiar ≥ 0,50) | — | 0,5381 | — |
| Acurácia com a equação inteira mascarada (104.141 tokens) | 0,0704 | 0,1984 | +0,128 [+0,122; +0,134] |
| Acurácia em tokens de equação, máscara uniforme (108.388 tokens) | 0,8694 | 0,8643 | −0,0051 |
| Acurácia em prosa, máscara uniforme (199.474 tokens) | 0,6944 | 0,6933 | −0,0011 |
| **Vantagem em equação (diferença das diferenças)** | +0,1750 | +0,1710 | **−0,0040 [−0,0058; −0,0022]** |

O tratamento foi absorvido — reconstruir uma equação inteira escondida vai de 7% para 20% (Tabela 3) — e não se transferiu: sob máscara uniforme, o tratado acerta menos tokens de equação que o controle, e a perda em equação excede a perda em prosa. Pela regra, o desfecho é negativo, e o efeito é pequeno: 0,4 ponto, contra 12,8 da checagem. Um viés da medida havia sido nomeado antes do resultado e aponta para o controle: o tratado viu tokens de equação mascarados sobretudo em blocos inteiros, enquanto a prova uniforme esconde tokens isolados com os vizinhos visíveis, o regime que o controle praticou durante todo o treinamento. Em análise exploratória, posterior ao resultado, o negativo concentra-se nas equações em display (−0,0055) e não nas em linha (−0,0015, intervalo cruzando zero). A assimetria de execução, desta vez, é contra o tratado: no passo 8.075 o detector acionou pelo critério de perda, descartou 76 lotes e reduziu a taxa de aprendizado à metade no início do decaimento. Esse braço foi treinado antes da correção da Seção 5.3, e o acionamento é provavelmente do mesmo tipo — composição do lote, e não instabilidade —, o que não foi verificado.

**Tabela 4.** Medidas secundárias: recuperação sem ajuste, recuperação após ajuste contrastivo e sonda tensorial.

| Medida | Controle | Tratado | Diferença |
|---|---|---|---|
| nDCG@10, sem ajuste (agregação por média) | 0,0171 | 0,1391 | +0,122 [+0,109; +0,135] |
| recall@1, sem ajuste | 0,0055 | 0,0785 | McNemar p = 3,6 × 10⁻³⁶ |
| nDCG@10, após ajuste com 200 mil pares | 0,3872 | **0,4712** | **+0,084 [+0,071; +0,097]** |
| recall@1, após ajuste | 0,2215 | 0,2985 | 106 × 260, p = 4,7 × 10⁻¹⁶ |
| recall@10, após ajuste | 0,5890 | 0,6740 | — |
| Sonda tensorial (72 itens) | 0,333 | 0,375 | 12 × 15, p = 0,70 |

Um fator de oito na recuperação sem ajuste (Tabela 4), entre modelos que diferem em um único hiperparâmetro, exigia descartar primeiro a explicação geométrica. A anisotropia dos dois braços é igual (cosseno médio de 0,972 e 0,971), centralizar os vetores não fecha a diferença (0,021 contra 0,143), e o braço A da Seção 4.3, também sem tratamento, fica em 0,030. A pergunta relevante para o sistema, porém, era se a diferença sobrevive ao ajuste contrastivo, que no recuperador pode apagar ou ampliar diferenças de base [2]. Ajustados os dois braços com a mesma receita e medidos na mesma sessão, com a regra registrada antes, o tratado fica à frente: o ganho encolhe de +0,122 para +0,084, e permanece de 22% em nDCG@10.

O resultado da ablação é, portanto, dividido: o mascaramento de equações inteiras não melhora a predição de tokens de equação e melhora a representação agregada para recuperação, antes e depois do ajuste. O mecanismo não foi medido. Restava saber se o artefato compete com o que existe de graça (Tabela 5).

**Tabela 5.** O mesmo ajuste contrastivo sobre uma base geral de 150 M sem qualquer pré-treinamento no domínio, com os mesmos 200 mil pares, os mesmos hiperparâmetros e os três braços medidos na mesma sessão.

| Braço | Parâmetros | Pré-treinamento | recall@1 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|
| Controle | 48 M | 0,6 B tokens de Física, do zero | 0,2215 | 0,5890 | 0,3391 | 0,3872 |
| Tratado | 48 M | idem, com mascaramento de equações | 0,2985 | 0,6740 | 0,4203 | 0,4712 |
| ModernBERT-base [13] | 150 M | cerca de 2 T tokens gerais | **0,3540** | **0,7200** | **0,4774** | **0,5270** |

Na Tabela 5, a base geral supera o braço tratado em 0,0558 de nDCG@10, com intervalo de [+0,0426; +0,0685] por bootstrap pareado por item — a leitura registrada antes da coleta para esse desfecho. A comparação não é de uma variável, e isso estava declarado antes: diferem o tamanho, o tokenizador e o volume de pré-treinamento. Ela não testa a hipótese do mascaramento; testa se o artefato que a hipótese produz, na escala que o orçamento permite, compete com o que existe publicamente. Não compete.

### 4.5 Pré-treinamento continuado e escolha da base

A Tabela 5 delimitou a pergunta: se o mascaramento de equações tem valor, ele deve aparecer no pré-treinamento continuado da base que venceu. Os dois braços partem do ModernBERT-base e diferem apenas em `p_equacao` (Seção 3.4), com as regras de leitura, primária e secundária, registradas antes do treinamento.

A primeira execução do braço tratado foi interrompida pelo detector de instabilidade no passo 4.489, após três acionamentos com a norma do gradiente normal nos três; a Seção 5.3 mostra que eles correspondiam à composição dos lotes, e não a instabilidade. Com o critério corrigido, o braço foi refeito do zero e treinou os 6.103 passos sem acionamento. O controle treinou os 6.103 passos com dois acionamentos pelo critério de norma — 7,2 e 7,7 contra medianas móveis de 0,64 e 0,61 —, que descartaram 182 lotes e reduziram a taxa de aprendizado à metade por 500 passos. Essa assimetria, desta vez, favorece o tratado.

**Tabela 6.** Pré-treinamento continuado: checagens de manipulação e medida primária. Duas mil sequências sorteadas de partição não vista, contexto de 1.024, as mesmas posições nos dois braços. Intervalos por bootstrap pareado por sequência.

| Medida | Controle | Tratado | Tratado − controle |
|---|---|---|---|
| Fração de exemplos com equação elegível (limiar ≥ 0,50) | — | 0,549 | — |
| Acurácia com a equação inteira mascarada (104.624 tokens) | 0,0266 | 0,1999 | +0,173 [+0,167; +0,179] |
| Acurácia em equação, máscara uniforme | 0,9032 | 0,9030 | — |
| Acurácia em prosa, máscara uniforme | 0,7783 | 0,7784 | — |
| **Vantagem em equação (diferença das diferenças)** | +0,1249 | +0,1245 | **−0,00039 [−0,00161; +0,00087]** |

Pela regra, o desfecho primário da Tabela 6 é não decidido. O tratamento foi absorvido com mais força que no substituto — reconstruir a equação inteira escondida vai de 2,7% para 20,0%, contra 7% para 20% a 48 M — e, sob máscara uniforme, os dois braços acertam equação e prosa igualmente até a terceira casa decimal. O negativo de −0,0040 do substituto não se repete: encolhe dez vezes, e o intervalo passa a cobrir zero.

**Tabela 7.** Os mesmos 200 mil pares e a mesma receita de ajuste contrastivo sobre quatro bases, medidas na mesma sessão.

| Base do ajuste | Pré-treinamento no domínio | recall@1 | recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| ModernBERT-base [13] | nenhum | 0,3540 | 0,7200 | 0,4774 | 0,5270 |
| ModernBERT-base, controle | 0,4 B tokens | 0,3635 | 0,7175 | 0,4823 | 0,5297 |
| ModernBERT-base, tratado | 0,4 B tokens, com mascaramento de equações | 0,3620 | 0,7290 | 0,4876 | 0,5373 |
| **GTE-base [15]** | **nenhum** | **0,4170** | **0,7880** | **0,5443** | **0,5964** |

Pelas regras registradas, a Tabela 7 tem três desfechos. Entre os braços, a estimativa favorece o tratado: +0,0076 de nDCG@10 [+0,0013; +0,0139], intervalo de 95% sem correção para as três comparações, com recall@1 sem diferença (63 contra 60 discordantes, p = 0,86). O sinal coincide com o do substituto, com um onze avos da magnitude; com uma semente por braço, a assimetria de execução a favor do tratado e a escolha do ponto de verificação em parte do conjunto de avaliação, o resultado é sugestivo e não estabelece o ganho. Contra a base sem pré-treinamento, o tratado fica acima por +0,0103 [+0,0032; +0,0176]. E o GTE-base — que, ao contrário do ModernBERT-base, já passou por pré-treinamento contrastivo de recuperação em larga escala — supera o ModernBERT-base por +0,0694 [+0,0580; +0,0812]; nenhum dos dois recebeu pré-treinamento neste corpus. O ModernBERT-base medido nesta sessão reproduz o valor de seis dias antes, 0,5270, com o mesmo avaliador.

Lidos juntos, os três dizem que 0,4 bilhão de tokens de Física com o objetivo específico de domínio compraram +0,010 sobre a base de partida, e que trocar a base de partida, sem pré-treinamento algum, compra +0,069. A diferença entre as bases reúne tamanho, contexto e, sobretudo, o estágio contrastivo prévio do GTE, e não isola nenhum deles. Levado a 1 milhão de pares, o GTE-base não se distingue do recuperador do sistema de [2], um MiniLM de 23 M ajustado com 6 milhões, e custa mais de quatro vezes mais para embutir.

---

## 5. Discussão

### 5.1 O encoder de domínio deve ser treinado do zero?

A premissa da Introdução apoiava-se em três ingredientes que só um modelo próprio teria. Os resultados, na escala de substituto, enfraquecem os três. A pré-tokenização nativa piorou a modelagem. O objetivo de mascaramento de equações ajudou a recuperação, mas as marcas de equação são derivadas de posições de caractere, e não de identificadores de token: o tratamento aplica-se igualmente ao pré-treinamento continuado de qualquer base, com qualquer tokenizador, e um resultado positivo dele favorece usá-lo, e não usá-lo a partir do zero. E contexto de 8.192 tokens já existe em encoders gerais abertos [13].

A comparação de referência é o resultado mais consequente: a base geral, ajustada sem pré-treinamento no domínio, supera o braço tratado em 0,0558 de nDCG@10 (Tabela 5). Isso retira, na prática, a justificativa para treinar o encoder a partir do zero sob este orçamento. Um modelo próprio de 150 M igualaria a base em parâmetros, mas não em volume de pré-treinamento — 2 bilhões de tokens preparados contra cerca de 2 trilhões; a diferença de 0,0558 foi medida no substituto de 48 M, confunde tamanho, tokenizador e volume, e não se sabe quanto dela restaria a 150 M. Nenhuma evidência deste trabalho sugere que o objetivo específico de domínio, que vale +0,084 na escala do substituto, cubra essa distância de volume.

O pré-treinamento continuado foi executado, e o objetivo mostrou-se pequeno: +0,0076 sobre o controle e +0,0103 sobre a base de partida na recuperação, e nada na predição de tokens. A resposta à pergunta desta seção, porém, veio de uma medição que o desenho original não previa. A base do pré-treinamento continuado foi escolhida por compartilhar a arquitetura planejada e ter contexto longo, e não por ter sido comparada, na mesma régua, às demais bases gerais. O GTE-base, ajustado nos mesmos pares, superou-a por 0,069 — cerca de sete vezes o que o pré-treinamento continuado acrescentou. O melhor encoder desta escala não passou por pré-treinamento de linguagem mascarada no domínio; é, porém, um modelo de embedding já treinado contrastivamente, e a diferença mede o conjunto base, e não um fator isolado.

O padrão do objetivo de domínio através das escalas é informativo: +0,084 sobre bases de 48 M treinadas do zero, +0,0076 sobre uma de 150 M com 2 trilhões de tokens gerais, com a medida primária negativa na primeira e não decidida na segunda. O efeito é menor sobre a base mais forte, o que é compatível com uma lacuna que a base forte já cobre; com dois pontos, de uma semente cada, que diferem também em tokenizador, volume e regime, o padrão é uma hipótese, e não um resultado. O argumento do viés indutivo nativo permanece sem teste direto, e os resultados aqui não indicam que valha o seu preço.

### 5.2 Instrumentos registrados antes da coleta, e inválidos para a pergunta

Registrar a regra de decisão antes dos dados protege contra escolher a leitura depois do resultado; não protege contra escolher o instrumento errado. Isso ocorreu três vezes no programa, e em nenhuma o instrumento produzia erro ou valor fora da faixa esperada.

Primeiro, a acurácia de predição mascarada foi registrada como medida primária da comparação de tokenizadores, por ter o maior número de observações pareadas. Ela não compara vocabulários: um tokenizador que divide o texto em unidades menores deixa mais redundância local após o mascaramento e acerta mais sem ser modelo melhor. Num ensaio com modelos de 30 passos, a variante E marcou acurácia maior (0,0237 contra 0,0195) e bits por byte piores (2,890 contra 2,761). O viés apontava contra a hipótese sob teste, de modo que um empate teria sido lido como refutação honesta.

Segundo, o desempate registrado para os bits por byte era a pseudo-verossimilhança de um token por vez [11], que carrega, mais forte, o mesmo viés do instrumento que devia desempatar [12]. Foi substituída antes de executada.

Terceiro, no sistema de recuperação, o teste de McNemar sobre pertencimento aos k primeiros foi registrado para julgar um reordenador, que atua dentro dos k primeiros [2]. A prática adotada em consequência é registrar, ao lado de cada regra, a direção do viés de cada instrumento e, quando não há instrumento sem viés, cercar o valor com instrumentos de vieses opostos antes de decidir por um terceiro, como na Seção 4.3.

### 5.3 Mecanismos automáticos que dependem da variável sob teste

O pré-treinamento continuado expôs uma classe de defeito distinta: não um instrumento de medida errado, mas um mecanismo automático do próprio treinamento — ou da cadeia que leva o modelo à medida — cujo comportamento dependia da variável sob teste.

O detector de instabilidade julgava a perda média do lote. No braço tratado, parte dos alvos são equações inteiras escondidas, muito mais difíceis que os alvos do sorteio uniforme, e a quantidade delas por lote varia com o sorteio do tratamento. A perda do tratado oscilava com a composição do lote — desvio padrão de 0,077, contra 0,037 no controle —, e o critério de μ + 4σ passou a ler composição como instabilidade. Como o fluxo de dados e a máscara são funções determinísticas de (semente, passo), os lotes dos três acionamentos puderam ser reproduzidos sem GPU: tinham de 3,0 a 4,1 desvios padrão acima da média de alvos de equação inteira em 300 passos sorteados, e dois deles ultrapassavam o máximo da amostra. O detector intervinha, portanto, preferencialmente no braço tratado, e por causa do tratamento. Uma assimetria de execução desse tipo não é ruído: correlaciona-se com a variável. A correção julga a perda apenas sobre os alvos uniformes, cuja composição não depende do tratamento; no controle, os dois sinais coincidem.

Dois defeitos na cadeia até a medida tinham a mesma propriedade de silêncio. O exportador de modelos nomeava os tokens especiais pelos identificadores da convenção dos tokenizadores próprios do projeto; na base de pré-treinamento continuado esses identificadores correspondem a tokens comuns, e o modelo exportado declarava o sinal `#` como token de máscara. Pesos, logits e a ida e volta pelo disco conferiam com diferença nula, porque cada conferência comparava o artefato consigo mesmo — e a medida primária da Tabela 6 esconde tokens com o token de máscara declarado. O defeito foi encontrado ao ler o registro de uma exportação local, antes de qualquer medida. O comparador da ablação, por sua vez, fora escrito para o experimento da Seção 4.4 e carregava três constantes dele: a escala, a regra e uma ressalva sobre a assimetria de execução daquele experimento. Aplicado ao pré-treinamento continuado, o artefato saiu declarando a escala errada e uma assimetria contra o tratado, quando a deste experimento era a favor. Uma ressalva falsa tem a forma do cuidado e o conteúdo errado, e por isso não costuma ser conferida.

A prática adotada em consequência é tratar como parte do desenho experimental todo mecanismo que atua de modo diferente conforme o braço, e verificar, antes de comparar, que nenhum deles lê um sinal que o tratamento altera.

### 5.4 Amostragem dependente da ordem dos dados

Três das oito ocorrências de amostragem dependente da ordem dos dados auditadas no programa [2] ocorreram na preparação e na avaliação do pré-treinamento. Selecionar as primeiras 8 de 44 partições do corpus teria deixado de fora partições com 57% mais equações em display, viés detectado antes do uso. O instrumento escrito para conferir essa seleção lia 200 linhas de um único arquivo e reproduziu o valor enviesado que existia para refutar. E as 2.000 primeiras sequências de uma partição, previstas para a avaliação de modelagem mascarada por região, cobriam cerca de 4% dos documentos, na ordem de ingestão; o defeito foi detectado antes da medição da Seção 4.4, e a avaliação passou a sortear.

### 5.5 Implicações práticas

Medir a vazão real, e a do modelo certo, antes de planejar. No ajuste contrastivo de um bi-encoder pequeno, a vazão medida em acelerador T4 foi de cerca de 69.700 tokens por segundo, contra 7.700 na GPU local. A extrapolação dessa vazão para o pré-treinamento, cerca de 80 horas para 20 bilhões de tokens, não se confirmou: o substituto de 48 M mediu 20.500 tokens por segundo, e a base de 150 M, cerca de 15.600 em duas placas, isto é, de 270 a 360 horas para o mesmo volume.

Comparar as bases candidatas antes de investir numa delas. O pré-treinamento continuado custou cerca de 20 horas de sessão de T4, incluída a execução interrompida pelo detector, sobre uma base escolhida por argumento; a comparação entre bases, feita depois e no mesmo protocolo, custou 1 hora e 17 minutos e mostrou uma diferença sete vezes maior que o ganho do pré-treinamento. A ordem inversa teria poupado o investimento, ou o dirigido para a base certa.

---

## 6. Limitações

A comparação de tokenizadores das Seções 4.1 e 4.3 não isola as regras de LaTeX: o braço que as contém também não tem fronteira de palavra. A ablação que as isolaria — segmentação genérica com e sem as regras — não foi executada, e a comparação extrínseca entre codificação por pares de bytes e modelo de unigramas, nem a entre tamanhos de vocabulário, tampouco.

Os resultados das Seções 4.3 e 4.4 são de substitutos de 48 M a 0,6 bilhão de tokens, 3,3 vezes abaixo do orçamento preparado para a ablação, e os da Seção 4.5, de um pré-treinamento continuado de 0,4 bilhão de tokens. Todos têm uma semente por braço, e cada ablação tem uma assimetria de execução não controlada — a favor de E na Seção 4.3, contra o tratado na Seção 4.4 e a favor do tratado na Seção 4.5. Nesta última, com um efeito de recuperação de +0,0076 e o limite inferior do intervalo em +0,0013, a assimetria é da ordem de grandeza que poderia explicar parte do efeito; ela não foi medida. A medida primária de modelagem mascarada tem viés declarado a favor do controle nas duas escalas. O encoder próprio de 150 M não foi treinado.

As medidas de recuperação herdam o protocolo de [2]: o ponto de verificação é escolhido em metade do conjunto que dá o veredito, a divisão entre treinamento e validação é por documento citante, e todos os modelos são lidos a 192 tokens. Diferenças pequenas, como os +0,0076 e +0,0103 da Tabela 7, são da ordem desse viés de seleção.

A comparação da Tabela 5 não é de uma variável: tamanho, tokenizador e volume de pré-treinamento diferem simultaneamente. A Tabela 7 compara bases a 200 mil pares, e a ordem entre bases a esse volume pode não se manter com mais pares; base é variável composta — o GTE-base tem 109 M, contexto de 512 e pré-treinamento contrastivo geral, possivelmente com exposição a dados de citação não auditada; o ModernBERT-base, 150 M, contexto de 8.192 e apenas modelagem mascarada.

A sonda tensorial usa itens sintetizados, e não ocorrências do corpus. O efeito de acrescentar a fatia de matemática e ciência da computação ao corpus de pré-treinamento não está medido.

---

## 7. Conclusão

Sob orçamento de computação nulo, os três ingredientes que justificariam treinar do zero um encoder de Física foram testados, e nenhum se sustentou como justificativa. A pré-tokenização proposta para LaTeX, favorecida pela métrica intrínseca, piorou a modelagem em 0,047 bit por byte, num resultado que só um terceiro instrumento, cercado por dois de vieses opostos, pôde decidir. Mascarar equações inteiras não melhorou a predição de tokens de equação e melhorou a recuperação após ajuste em 0,084 de nDCG@10 — e, ainda assim, o encoder de 48 M treinado do zero com esse objetivo ficou 0,056 abaixo de uma base geral de 150 M sem pré-treinamento no domínio.

O pré-treinamento continuado dessa base com 0,4 bilhão de tokens de Física e o mesmo objetivo repetiu o sinal do ganho de recuperação, pequeno e não estabelecido, enquanto outra base geral, já treinada contrastivamente, a superou por 0,069 — cerca de sete vezes os +0,010 que o pré-treinamento continuado acrescentou. Sob este orçamento, escolher a base geral pesa mais que qualquer pré-treinamento no domínio e custa uma fração dele.

O resultado metodológico de maior alcance é que três instrumentos registrados de antemão eram inválidos para a pergunta, e que um mecanismo de segurança intervinha preferencialmente no braço tratado, por causa do tratamento — nenhum deles com sintoma além do próprio número. Registro prévio não substitui a verificação ativa da direção de viés de cada instrumento e da independência, em relação à variável sob teste, de todo mecanismo que atua sobre os braços.

---

## Disponibilidade de dados e código

O código e a documentação de projeto são públicos sob licença permissiva, com manifestos de hashes das entradas e saídas de cada etapa. O corpus não acompanha o código: os registros do arXiv seguem a licença de cada submissão.

**Nota de autoria.** O trabalho experimental, as decisões de projeto e a implementação são do autor. A redação deste manuscrito contou com assistência de um modelo de linguagem (Claude, Anthropic) a partir dos artefatos de medição do repositório; o autor revisou o texto e responde por ele. As referências foram conferidas contra as respectivas fontes.

---

## Referências

[1] Hoffmann, J.; Borgeaud, S.; Mensch, A.; et al. Training Compute-Optimal Large Language Models. *Advances in Neural Information Processing Systems 35* (NeurIPS 2022), 2022.

[2] Sanchez, V. Construção de um sistema de recuperação de artigos de Física do arXiv sob restrição severa de computação: corpus, supervisão por citação e verificação de protocolo. Manuscrito em preparação, 2026.

[3] Taylor, R.; Kardas, M.; Cucurull, G.; Scialom, T.; Hartshorn, A.; Saravia, E.; Poulton, A.; Kerkez, V.; Stojnic, R. Galactica: A Large Language Model for Science. arXiv:2211.09085, 2022.

[4] Heaven, W. D. Why Meta's latest large language model survived only three days online. *MIT Technology Review*, 18 de novembro de 2022.

[5] Lewkowycz, A.; Andreassen, A.; Dohan, D.; Dyer, E.; Michalewski, H.; Ramasesh, V.; Slone, A.; Anil, C.; Schlag, I.; Gutman-Solo, T.; Wu, Y.; Neyshabur, B.; Gur-Ari, G.; Misra, V. Solving Quantitative Reasoning Problems with Language Models. *Advances in Neural Information Processing Systems 35* (NeurIPS 2022), 2022.

[6] Azerbayev, Z.; Schoelkopf, H.; Paster, K.; Dos Santos, M.; McAleer, S.; Jiang, A. Q.; Deng, J.; Biderman, S.; Welleck, S. Llemma: An Open Language Model for Mathematics. *International Conference on Learning Representations* (ICLR 2024), 2024.

[7] Shao, Z.; Wang, P.; Zhu, Q.; et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.

[8] Sennrich, R.; Haddow, B.; Birch, A. Neural Machine Translation of Rare Words with Subword Units. *Proceedings of ACL 2016*, v. 1, p. 1715–1725, 2016.

[9] Kudo, T. Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. *Proceedings of ACL 2018*, v. 1, p. 66–75, 2018.

[10] Bostrom, K.; Durrett, G. Byte Pair Encoding is Suboptimal for Language Model Pretraining. *Findings of EMNLP 2020*, p. 4617–4624, 2020.

[11] Salazar, J.; Liang, D.; Nguyen, T. Q.; Kirchhoff, K. Masked Language Model Scoring. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (ACL 2020), p. 2699–2712, 2020.

[12] Kauf, C.; Ivanova, A. A. A Better Way to Do Masked Language Model Scoring. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics* (ACL 2023), v. 2: Short Papers, p. 925–935, 2023.

[13] Warner, B.; Chaffin, A.; Clavié, B.; et al. Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. *Proceedings of ACL 2025*, 2025. arXiv:2412.13663.

[14] Wettig, A.; Gao, T.; Zhong, Z.; Chen, D. Should You Mask 15% in Masked Language Modeling? *Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics* (EACL 2023), p. 2985–3000, 2023.

[15] Li, Z.; Zhang, X.; Zhang, Y.; Long, D.; Xie, P.; Zhang, M. Towards General Text Embeddings with Multi-stage Contrastive Learning. arXiv:2308.03281, 2023.

[16] Hellert, T.; Montenegro, J.; Pollastro, A. PhysBERT: A text embedding model for physics scientific literature. *APL Machine Learning*, v. 2, n. 4, art. 046105, 2024. DOI 10.1063/5.0238090.

[17] Beltagy, I.; Lo, K.; Cohan, A. SciBERT: A Pretrained Language Model for Scientific Text. *Proceedings of EMNLP-IJCNLP 2019*, 2019.

[18] Weber, M.; Fu, D.; Anthony, Q.; et al. RedPajama: an Open Dataset for Training Large Language Models. *Advances in Neural Information Processing Systems 37, Datasets and Benchmarks Track* (NeurIPS 2024), 2024. arXiv:2411.12372.

[19] McNemar, Q. Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, v. 12, n. 2, p. 153–157, 1947.
