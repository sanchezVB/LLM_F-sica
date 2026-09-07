# ΦFM — uma família de foundation models para Física sob restrição de orçamento: projeto, medições e estado em setembro de 2026

**Rascunho de trabalho.** Não submetido, não revisado externamente. Ver §14 antes de
usar qualquer número, e §16 antes de usar qualquer citação.

**Autoria.** O trabalho experimental, as decisões de projeto e o repositório são de
Vinicius Sanchez. Este rascunho foi redigido por Claude (Anthropic) a partir dos
artefatos de medição versionados do repositório. Ordem e forma da autoria são decisão
do primeiro.

**Estado do documento:** v0.1 · 2026-09-07 · corresponde ao commit `8cc7a48`.

---

## Resumo

Física é um domínio pobre em dados pelos padrões de LLMs de fronteira: toda a
literatura legalmente adquirível está na ordem de 10¹⁰–10¹¹ tokens, contra 10¹³–10¹⁴
de um modelo de fronteira. Isso torna "treinar um LLM de Física do zero" o padrão
errado para todos os orçamentos exceto os maiores — e é a restrição que organiza este
programa. Descrevemos o projeto e a execução do **ΦFM**, uma família de foundation
models para Física construída como uma **escada de degraus independentes com portões
falseáveis**, sob a restrição de que cada degrau precisa ser entregável sozinho e
quase todos precisam custar aproximadamente zero em GPU.

Reportamos o estado após treze meses de projeto e cinco semanas de execução medida.
**O que está fechado:** um corpus de **27,75 bilhões de tokens** de Física montado a
custo zero, uma espinha de metadados de **1.595.422** registros do arXiv casada com
**4,61 M** de obras do OpenAlex a **99,1%**, um classificador de domínio com acurácia
**0,954**, um barramento de verificação mecânica com cinco de seis verificadores, e um
sistema de recuperação híbrido cuja composição completa — BM25 + denso → fusão RRF →
reranqueador — vence a fusão isolada com **p = 0,0062**. **O que está aberto:** o
portão G1 está **sob remedição** desde 2026-09-06, porque o protocolo que o julgou
tinha teto de **0,7562** de nDCG@10 em vez de 1,0; e o ΦEnc — o único modelo treinado
do zero, e o portador da hipótese central do programa — tem dados e código prontos e
**nenhuma avaliação**.

Duas contribuições são independentes do desfecho dos portões. A primeira é empírica:
para reordenar, o que paga é **pré-treino no domínio**, não diversidade de base nem
capacidade — um encoder de Física que é um recuperador *ruim* neste benchmark
(nDCG 0,2752 contra 0,4657) é a **melhor base de reranqueador** das três testadas, e
um modelo geral forte do mesmo tamanho empata (p = 0,637). A segunda é metodológica:
catalogamos **seis ocorrências independentes** de uma mesma armadilha de amostragem
neste repositório, com o custo medido de cada uma, incluindo uma que impunha teto ao
critério de sucesso do projeto e outra que fez o treino ver **10,7× menos** documentos
distintos do que podia com o mesmo orçamento.

**Palavras-chave:** foundation models científicos, recuperação de informação em
Física, supervisão por citação, corpus científico, verificação mecânica,
reprodutibilidade.

---

## 1. Introdução

### 1.1 O problema

Modelos de linguagem de propósito geral falham em Física de maneiras estruturalmente
distintas de como falham em perguntas factuais. O projeto nomeia dez modos de falha
(deriva simbólica, incoerência dimensional, cegueira a casos-limite, fragilidade
numérica, colapso de notação, alucinação de citações, alucinação de convenção,
analfabetismo gráfico, ingenuidade experimental, extrapolação confiante) e a
consequência de projeto que deles se extrai é específica: **cinco desses modos não são
corrigíveis adicionando mais texto de Física.** Deriva simbólica, incoerência
dimensional, cegueira a limites, alucinação de convenção e extrapolação confiante
exigem (i) um verificador dentro do laço de treino e (ii) dados explicitamente
construídos que demonstrem o comportamento de checagem.

Sobre isso incide uma restrição de recurso que não é negociável e que definiu a forma
do programa:

> **O muro de tokens.** Restringindo ao que é efetivamente adquirível a custo zero e
> modelando o funil de filtragem estágio a estágio, a literatura de Física disponível
> é de **39–73 B tokens brutos → 15–30 B de treino**. Um modelo de 8 B parâmetros
> compute-ótimo por Chinchilla pede **160 B**. A escassez é de 5–10×.

A leitura convencional dessa restrição é "consiga mais dados". A leitura que este
programa adota é diferente: **treino do zero é autorizado apenas onde o dado é
excedente** — encoders, onde 15–30 B tokens sobram para um modelo de 150 M — e tudo
acima disso é construído por *continual pretraining* de uma base geral forte, mais RL
com recompensa verificável e aumento por ferramentas. Não é um compromisso para
economizar; é o que a literatura mostra. Todo modelo científico aberto de ponta dos
últimos quatro anos — Minerva, Llemma, DeepSeekMath, Qwen-Math — saiu de CPT sobre uma
base geral, enquanto a única tentativa em larga escala de treinar um foundation model
científico do zero, o Galactica, foi retirada do ar três dias após o lançamento.

### 1.2 A segunda restrição: orçamento

Este programa foi executado numa GPU de consumo de 8 GB via DirectML e na cota
gratuita de Tesla T4 do Kaggle. O orçamento total gasto em computação até o estado
descrito aqui é de **US$ 0**; o custo em dinheiro de toda a Fase 1 — do corpus bruto
aos shards prontos, cinco documentos de projeto — foi **inferior a US$ 60**, e nenhum
deles em GPU.

Isso não é uma nota de rodapé sobre pobreza. É uma **condição de método**, e afirmamos
que ela melhorou o trabalho: quando cada experimento precisa ser justificado por uma
medição anterior, medições anteriores passam a ser lidas com atenção — e é lendo-as
com atenção que se encontram as falhas catalogadas na §12. Quatro das seis armadilhas
de amostragem que reportamos foram encontradas porque alguém precisou justificar um
gasto de horas de GPU gratuita.

### 1.3 Contribuições

1. **Um programa completo de projeto**, em 19 documentos e 1 ADR cobrindo 20
   pipelines, escrito antes do código e confrontado com ele. §4 descreve o mecanismo
   que mantém os dois alinhados, e §11 reporta quantas afirmações do projeto
   sobreviveram ao confronto — inclusive as que não sobreviveram.
2. **Um corpus de Física de 27,75 B tokens montado a custo zero**, com a espinha de
   metadados atestada por um único hash de Merkle sobre 21,79 GB (§5, §5.7).
3. **Um discriminante medido para integridade matemática de corpus** (§5.4). O
   diagnóstico intuitivo — "operador órfão" — **satura no corpus bom**: acusa 81,3% do
   corpus íntegro contra 3,0% da referência. O discriminante que funciona é presença
   de ambiente de equação: **84,9% contra 0,0%**.
4. **Evidência de tokenização em LaTeX que não existia** (§6): BPE bate Unigram em
   Física com margem seis vezes maior em equações (13%) que em prosa (2%), e com
   mecanismo observável.
5. **O achado de reranking** (§8.5): o que faz um cross-encoder acrescentar informação
   a uma fusão não é diversidade de base nem capacidade, é **pré-treino no domínio** —
   e o modelo de domínio é, ele mesmo, um recuperador ruim no mesmo benchmark.
6. **Um catálogo de seis ocorrências de uma mesma armadilha de amostragem** (§12.1),
   com o custo medido de cada uma. É a contribuição que menos gostaríamos de ter.

### 1.4 O que este artigo *não* alega

Não alegamos um modelo de Física estado da arte. O portão G1 **não está passado**, e
a §8.2 descreve por quê em detalhe, incluindo um defeito de protocolo descoberto no
dia anterior à redação deste rascunho que invalida a medição que parecia mais próxima
de fechá-lo. Não alegamos que o mascaramento consciente de equações funcione: ele é a
hipótese central do ΦEnc, o código está escrito e verificado, os dados estão prontos,
e **a ablação não rodou**. Um artigo que apresentasse qualquer uma dessas duas coisas
como resultado cometeria exatamente o erro que a §12 cataloga.

---

## 2. Posicionamento

### 2.1 Encoders científicos — a competição real do Tier 1

| Modelo | Params | Realidade do domínio |
|---|---|---|
| SciBERT (Beltagy et al., 2019) | 110 M | **82% biomédico, 18% CS** — praticamente zero Física |
| SPECTER / SPECTER2 (Cohan et al., 2020) | 110 M | Multi-domínio, enviesado a biomédico |
| INDUS (NASA/IBM, 2024) | 110–368 M | Astro e Ciências da Terra; não hep/cond-mat/quant-ph |
| **PhysBERT** (Hellert et al., 2024) | 110 M | **Competidor direto** — mesmo domínio |
| E5 / BGE / GTE / Qwen3-Embedding | 0,1–8 B | Geral, e surpreendentemente forte em zero-shot |

O posicionamento do programa é explícito quanto a uma prática que considera
defeituosa: **superar o SciBERT não é resultado científico em Física.** O SciBERT mal
viu Física, e usá-lo como baseline principal é escolher um adversário fraco por
convenção. As barras reais são o PhysBERT (mesmo domínio) e os embedders gerais
modernos, contra os quais papers de domínio rotineiramente deixam de comparar.

Essa decisão de posicionamento foi tomada antes de medir, e a medição a validou de
forma mais dura do que o documento previa: no nosso benchmark, o **MiniLM-L6 genérico
de 23 M de parâmetros supera o PhysBERT de 109 M específico de Física** em nDCG@10
(0,370 contra 0,275). Ser treinado **para** embedding importa mais que ser treinado
**em** Física. Um trabalho que só comparasse com o SciBERT teria reportado uma vitória
confortável e falsa.

### 2.2 Generativos científicos — a lição que organiza o Tier 2

| Modelo | Receita | Lição |
|---|---|---|
| Galactica (2022) | **Do zero**, 106 B tokens de ciência | Retirado em 3 dias. Pretraining científico do zero em escala de fronteira é faminto de dados e sem ancoragem |
| Minerva (2022) | PaLM + CPT, 38,5 B tokens | A receita de CPT funciona; **preservar LaTeX na ingestão é decisivo** |
| Llemma (2024) | Code Llama + CPT, 55 B | Uma base de *código* é ponto de partida melhor para raciocínio formal |
| DeepSeekMath (2024) | Coder-Base + 120 B + GRPO | Mineração agressiva + RL supera contagem de parâmetros |
| AstroLLaMA (2023–24) | LLaMA + CPT em astro-ph | **Cautelar:** CPT de pequena escala só sobre resumos entrega pouco |
| DeepSeek-R1 / série o | RL com recompensa verificável | RLVR é a técnica de maior alavancagem para raciocínio STEM |

Duas dessas linhas são operacionais para nós. A de Minerva — preservar LaTeX na
ingestão — vira o critério que decide qual fatia de corpus serve ao ΦEnc (§5.4). A de
AstroLLaMA é a razão pela qual o ΦEnc exige 15–30 B tokens de **texto pleno** e não
pode ser treinado sobre os 0,33 B de títulos e resumos que a espinha oferece.

### 2.3 A oportunidade estrutural

Física é **mecanicamente verificável**, e quase ninguém explora isso. Análise
dimensional, equivalência simbólica em CAS, redução em casos-limite e leis de
conservação são todos verificadores executáveis. Um único **barramento de verificação**
pode servir simultaneamente à filtragem de dados, às recompensas de RL, à correção de
benchmarks e à auto-checagem em inferência — o que torna estruturalmente impossível o
treino divergir da avaliação, porque os dois consultam o mesmo objeto. §7 reporta o
estado dele: cinco verificadores de seis.

---

## 3. Visão geral do sistema

O programa é organizado em uma escada de degraus, cada um entregável de forma
independente:

| Degrau | Entrega | Custo | Portão | Estado |
|---|---|---|---|---|
| **T0** Corpus | `PhysCorpus-Open` + tokenizer | US$ 0 | corpus reconstruível de um hash | espinha pronta, tokenizer não iniciado |
| **T1** Representação | ΦEnc / ΦEmb / ΦRank | US$ 35–120 | G1.1–G1.5 | **parcial, sob remedição** |
| **T2** Raciocínio | ΦGen-1,5B via CPT+SFT+RLVR, ΦRAG | US$ 300–600 acum. | G2.1–G2.5 | não iniciado |
| **T2c** Escala | ΦGen-8B | US$ 1.100–2.260 acum. | competitivo com abertos médios | não iniciado |
| **T3** Fronteira | ΦGen-32B, ΦMM, ΦAgent | 150–600k GPU-h | financiamento externo | não iniciado |

O repositório implementa os documentos, não o contrário: `configs/` é uma árvore Hydra
onde vive **todo** hiperparâmetro, `src/phifm/models/` contém `nn.Module` puros sem
consciência de paralelismo, e o sharding é aplicado em `src/phifm/training/`. As
fronteiras de import são impostas em CI. A suíte tem **618 testes** (13 saltados) na
venv principal, mais 14 que dependem de torch e rodam na venv de treino.

---

## 4. Método: como o programa decide

Quatro mecanismos de decisão foram adotados antes da execução, e é a eles que
atribuímos a maior parte dos achados negativos deste artigo.

### 4.1 Portões falseáveis, não metas vagas

Metas do tipo "estado da arte em Física" são explicitamente rejeitadas como portão. O
portão G1 do Tier 1 é:

| Critério | Limiar |
|---|---|
| G1.1 | ΦEmb supera o PhysBERT em ≥ 5 pontos de nDCG@10 |
| G1.2 | ΦEmb supera o melhor embedder **geral** com ≤ 1/10 dos parâmetros |
| G1.3 | ΦEnc supera SciBERT e INDUS em ≥ 4 de 5 tarefas de classificação/NER |
| G1.4 | ΦOCR ≥ 0,92 em recuperação de equações em PDFs reservados |
| G1.5 | Corpus reprodutível ponta a ponta a partir de **um** hash de manifesto |

O valor de um portão assim é que ele pode **reprovar**, e reprovou. O G1.2 foi a
razão de o GTE-large (335 M) entrar na comparação: com apenas o MiniLM-L6 de 23 M no
papel de "genérico", o G1.2 parecia passar, e a cláusula de tamanho (≤ 1/10 dos
parâmetros do rival) dá razão 1/1 contra um rival do mesmo tamanho — não fecha nada. A
correção impediu uma afirmação falsa, e o custo dela foi descobrir que estávamos
atrás em vez de na frente.

### 4.2 Pré-registro da regra de decisão

Nos experimentos comparativos, a regra de leitura é escrita **antes** de medir,
incluindo as leituras que correspondem a resultado nulo. O experimento T1c (§8.5)
registrou: McNemar exato em k = 10 contra a fusão da mesma execução, limiar de
Bonferroni 0,025 por serem duas variantes, 2.000 consultas, e as **quatro** leituras
possíveis escritas de antemão — inclusive a de nenhuma variante vencer.

O pré-registro pegou um erro que ele mesmo teria cometido. Quando um dos braços morreu
por um defeito nosso (§8.5), o script imprimiu a leitura "o mecanismo é conhecimento
de domínio, não diversidade" — uma conclusão que *exige* o braço ausente ter produzido
um número e perdido. A lógica olhava só `venceram` e nunca `falhas`. O buraco não
estava na regra: estava na implementação dela. Hoje qualquer braço ausente torna o
mecanismo **INCONCLUSIVO**, e há um teste que **executa** a lógica nos quatro desfechos
em vez de procurar o texto da conclusão.

### 4.3 Medir antes de gastar cota

A regra operacional do repositório é que nenhuma hora de GPU é gasta sem uma medição
que a justifique. Ela funcionou consistentemente para computação — e **falhou uma vez
para dinheiro**, com o custo documentado na §5.4: recomendamos ao dono do projeto
comprar acesso em massa ao arXiv como pré-requisito do ΦEnc, quando um corpus de
835.379 documentos de Física construído do fonte LaTeX estava no disco havia dezessete
dias. A regra existia; ela não havia sido aplicada à classe de recurso certa.

### 4.4 Painel de estado de verificação, por afirmação

Toda afirmação consequente do repositório carrega estado de verificação explícito,
distinguindo **confrontado com execução** de **escrito e nunca posto à prova**.
No estado descrito aqui, quatro dos vinte documentos estão confrontados e dezessete não.

Esse painel existe porque o modo de falha que mais preocupa este programa não é errar
— é **perder de vista o que foi medido e o que foi suposto**. Houve um episódio
concreto: três cópias da mesma tabela de estado (no README, no índice de documentos e
no DOC-00 §11) divergiram, e as três marcavam 19 de 20 documentos como "em revisão" ao
lado da afirmação "corpus de projeto completo". Uma cópia é uma fonte; três são um
convite ao desacordo. Hoje a tabela vive num lugar só.

---

## 5. O corpus

### 5.1 Espinha de metadados (Sprint S1)

| | Medido | O plano previa |
|---|---|---|
| Registros do arXiv (set `physics`) | **1.595.422** · 0 falhas | 1,2 M |
| Tamanho em disco | 674 MB · **422 bytes/registro** | 516–686 bytes |
| Obras do OpenAlex (snapshot) | **4.613.751** · 0 falhas | — |
| Casamento com a espinha | **99,1%** (1.581.098) | 98,5% |
| Arestas de citação | **22,7 M+** | "dezenas de milhões" |
| Revisados por pares | 740.823 (46,4%) | — |
| **Fração redistribuível** | **14,8%** (235.795) | 25–35%, refutado |

O S1 custou **~3 GB de disco**, não os ~150 GB orçados. O snapshot do OpenAlex tem
725 GB; nós lemos **13 das 189 colunas** por faixa de bytes HTTP, e ele nunca toca o
disco. O tempo medido foi 5,6 h contra as ~2 h que o documento estimava — a estimativa
era otimista porque o corpus dobrara para 725 GB e nada daquilo é uma transferência
sequencial única.

Três achados de aquisição merecem registro porque cada um teria custado um subconjunto
grande do corpus:

- **A chave de junção não está onde a documentação sugere.** Casar arXiv com OpenAlex
  por `ids.arxiv` dá **1,5%** de cobertura; por `locations`, **98,5%**.
- **`primary_location` exclui publicados.** Filtrar por ele perderia 1,44 M de
  registros revisados por pares.
- **IDs antigos truncados por regex** afetavam **41,5%** do acervo, e o defeito só é
  visível em identificadores pré-2007.

A previsão de tamanho do documento de aquisição se confirmou com folga: 422
bytes/registro em parquet zstd contra 516–686 previstos.

### 5.2 Classificador de Física (Sprint S2)

Um classificador `is_physics` treinado sobre 600 mil documentos, com rótulo derivado
da regra **autoritativa** da espinha e não de prefixos de categoria — porque um `cs.LG`
com cross-list em `quant-ph` está nos dois conjuntos. A validação disso é limpa: dos
72.919 negativos com cross-list de Física, **exatamente** 72.919 estão na espinha e
**zero** ficaram fora.

Resultado final, com negativos de quatro domínios: **acurácia 0,954**, falso positivo
de **2,4%–3,7%** em cada domínio.

O experimento que decidiu o desenho é deixa-um-domínio-de-fora — treinar sem um
domínio negativo inteiro e testar nele:

| domínio omitido | FP dentro | FP fora | piora | precisão fora |
|---|---|---|---|---|
| cs | 3,7% | 9,8% | 2,6× | 0,907 |
| econ | 3,3% | 8,5% | 2,6× | 0,988 |
| **math** | 2,4% | **35,4%** | **14,6×** | 0,731 |
| **q-bio** | 3,1% | **31,2%** | **10,2×** | 0,907 |

Sem negativos de `math`, filtrar o OpenWebMath admitiria ~35% de conteúdo matemático
como Física — e o OpenWebMath é feito de matemática. Foi essa medição que justificou
coletar 774.063 registros de `math`, e **o número só existiu depois de a coleta
começar**.

Duas previsões nossas foram testadas com resultado oposto ao esperado, e as duas valem
registro:

- **"Os negativos de `math` serão quase inúteis, porque o rótulo depende de uma flag
  de cross-list que não está no texto."** Falso. Treinando só em papers de primária
  `math.*`, a acurácia é **83%** contra 50% do acaso: um `math.AP` cross-listado em
  `math-ph` fala de operadores de Schrödinger e equações de fluidos; um que não é fala
  de análise abstrata. O rótulo **está** no texto.
- **"`stat` é vizinho próximo, porque física estatística e estatística compartilham
  vocabulário."** Falso. Omitir `stat` do treino dá **2,9% de FP contra 3,0% dentro do
  domínio — piora de 1,0×**, contra 14,6× do `math`. O vocabulário compartilhado
  existe; não basta para confundir o classificador. `stat` **não** entra no treino, e o
  valor da medição foi diagnóstico.

### 5.3 As fatias (Sprint S3)

| fonte | documentos | tokens | custo |
|---|---|---|---|
| RedPajama-arXiv (fatia de Física) | 835.379 | **10,54 B** | US$ 0 |
| OpenWebMath (filtrado) | 860.521 | **2,62 B** | US$ 0 |
| peS2o (filtrado) | 5.526.331 de 38.972.211 (14,18%) | **14,60 B** | US$ 0 |
| **total** | | **27,75 B** | **US$ 0** |

Isso põe o corpus na metade superior da faixa de 15–30 B que o ΦEnc exige. Volume
deixa de ser o gargalo — e é exatamente aí que o gargalo real aparece (§5.4).

Um teste de fumaça sobre **um** arquivo, antes dos 22, custou 29 minutos e evitou
perder um dia inteiro. Ele encontrou três defeitos:

1. **O leitor tentava parquet num `.json.gz`.** A exceção era capturada como AVISO e o
   resumo saía com código 0. Os 22 arquivos falhariam igual e a coleta produziria um
   diretório vazio com um resumo satisfeito. Hoje, zero registros **vistos** levanta
   exceção: taxa de aceitação zero pode ser legítima; não ter lido nada nunca é.
2. **O repositório publica v1 e v2 da mesma coleção.** O filtro pegava as duas — os
   mesmos papers duas vezes, ~31 h para um corpus com metade duplicada. E duplicação em
   pré-treino não é desperdício, é dano.
3. **A retomada guardava ordinais sem guardar a lista.** Ao restringir para v2, "a
   unidade 1 já feita" passou a apontar para um arquivo nunca processado.

### 5.4 Integridade matemática: o discriminante que funciona

O objetivo de treino do ΦEnc (§9) exige que o corpus **tenha equações**. Medimos
quatro fatias do mesmo domínio, 3.000 documentos cada, sorteados:

| fatia | ch/doc | LaTeX % | `$…$` % | **ambiente de equação %** | seq/doc |
|---|---|---|---|---|---|
| resumos do arXiv (referência) | 1.123 | 21,8 | 26,9 | 0,0 | 0,9 |
| **RedPajama-arXiv** (fonte LaTeX) | 49.212 | 100,0 | 99,6 | **84,9** | 1.158,3 |
| OpenWebMath (páginas web) | 16.009 | 78,5 | 85,2 | 5,1 | 57,7 |
| **peS2o texto pleno** (extraído de PDF) | 28.605 | 16,3 | 18,2 | **0,0** | 1,0 |

O corpus extraído de PDF tem **28,6 mil caracteres por documento de Física sem uma
única equação em display**. O que sobra no lugar da matemática é diagnóstico:

```
"a pair of elements ρ, σ ∈ Σ are said to be orthogonal if: where {0} is..."
"The stationary solution p * to Eq. (7) satisfies p * = W · p *"
"the dependence of __ as a function of normal pressures __"
```

No primeiro, a equação em display entre "if:" e "where" foi apagada inteira — o
dois-pontos aponta para o nada. No segundo, `p^*` virou `p *` e `T_{eff}` virou
`T eff`: os índices foram achatados em tokens soltos pela extração de PDF.

**O achado metodológico é sobre o diagnóstico, não sobre o corpus.** A assinatura
intuitiva para "a equação foi arrancada" é o **operador órfão** — um ` = ` sem operando
de um dos lados. Ele funciona, e **satura no corpus bom**: acusa **81,3%** dos
documentos do corpus de fonte LaTeX contra **3,0%** da referência, e a inspeção mostra
matemática perfeita. A causa é banal: em `$Z_{\rm max}$ = 15 kpc`, o caractere antes de
` = ` é `$`, que o regex não aceita como operando.

> Um diagnóstico que dispara mais no corpus íntegro que no mutilado é pior que nenhum.
> O discriminante que funciona é **presença do ambiente**, não ausência do operando:
> 84,9% contra 0,0%.

Foi com o diagnóstico saturado que concluímos, em primeira instância, que comprar
acesso ao fonte do arXiv passava de conveniência a **pré-requisito da hipótese central
do projeto**, cotado em US$ 100–180. Duas coisas estavam erradas. A cotação: o bucket
é *requester pays* e o conjunto de fonte é ~2,9 TB; a US$ 0,09/GB de egresso, o fonte
inteiro para fora da AWS passa de **US$ 400** (dezenas de dólares filtrando dentro). E
o pré-requisito: não é — o RedPajama-arXiv, no disco, resolve o problema por US$ 0.

### 5.5 Degradação do RedPajama (S3b): quando a estimativa pontual não decide

Se o RedPajama-arXiv é construído do fonte LaTeX, quanto ele perde em relação ao fonte
original? A resposta mudou cinco vezes, e só a última vale — não por ser a última, mas
por ser a única que mede a população que o critério pergunta e declara um intervalo:

| medição | degradação | por que estava errada |
|---|---|---|
| n = 6, sem macros | 19,6% | macros do autor contadas como perda |
| n = 6, com macros | 2,6% | amostra de 6 papers |
| n = 199 | 27,4% | número único; mistura perda com notação |
| decomposto | 13,4% ausência + 14,0% notação | fonte = tarball, não documento |
| montagem corrigida | 13,4% + 13,0% | população = arXiv inteiro, não Física |
| **n = 298, só Física** | **16,6% ausência**, IC 95% **[12,9%–20,8%]** | ← este |

Cada correção ensinou algo que não era cosmético:

- **Macros do autor.** A fonte escreve `\Ecal_\mu`, o RedPajama escreve
  `\mathcal{E}_\mu`. Das equações que **não** casavam, 97% usavam macro; das que
  casavam, 20%.
- **O número único mandava gastar por engano.** "Ausente" é a equação que o RedPajama
  não tem — só isso justifica pagar. "Discordante" é a que está lá com outra notação, e
  pagar por ela seria comprar a solução de um problema nosso.
- **Eu media o arXiv inteiro e chamava de Física.** 52% da amostra não estava na
  espinha. O paper que mais contribuía para a perda é de **teoria de grafos**, e as
  "equações perdidas" eram tabelas de ciclos de permutação num apêndice.
- **A estimativa pontual não decidia.** Com 103 papers o IC era [9,9%; 19,1%] —
  cruzava o limiar de 10% por 0,1 ponto. "97% provável" não é como se autoriza um
  gasto. Com 298 papers de Física: [12,9%; 20,8%], P(> 10%) = 100%. O bootstrap
  reamostra **papers**, não equações, porque as equações de um paper compartilham o
  destino que o pipeline lhe deu.

Uma hipótese foi testada e **rejeitada**: o RedPajama não trunca por tamanho fixo. Não
há acúmulo no topo da distribuição, e o pior paper tem 52 mil caracteres contra um
máximo de 313 mil. Não existe o atalho de mirar só os papers longos.

### 5.6 Licenciamento

A fração redistribuível medida é **14,8%**, contra 25–35% que o projeto supunha — e
ela depende fortemente da época: **0,0% aberto até 2004**, 36,2% em 2020–2024, **48,8%
em 2025–2029**. Isso dimensiona o `PhysCorpus-Open`, a entrega do degrau T0, e é a
razão de o corpus completo não acompanhar o código: os scripts de coleta e os
manifestos permitem reconstruir; os bytes não são nossos para distribuir.

### 5.7 Reprodutibilidade por manifesto (portão G1.5)

```
hash raiz  bbd73a7a26ac8e8b03b7bb9c142bbb47d459d7a32d643fec63b477cf25cf5fb7
33 etapas · 976 arquivos · 21,79 GB · verificação profunda: OK
```

Uma cadeia de Merkle em três níveis: o hash raiz cobre o hash de cada manifesto de
etapa, que cobre o BLAKE3 de cada arquivo. A distinção entre verificação **rasa**
(milissegundos, pega manifesto mexido) e **profunda** (relê os 21,79 GB, é a única que
pega parquet mexido) é declarada, porque chamar o resultado da rasa de "corpus
verificado" seria ausência de erro lida como sucesso.

**O G1.5 está meio fechado, e a metade aberta é honesta.** O corpus neste disco é
atestado por um hash, com proveniência por etapa. Mas refazer a coleta do zero **não**
reproduz os mesmos bytes: o OAI-PMH do arXiv filtra por *datestamp*, e o datestamp muda
quando um autor publica versão nova. É a semântica da fonte, não defeito do coletor, e
a solução em uso é guardar e hashear a camada bruta.

Dois defeitos que o G1.5 expôs, e nenhum era do manifesto:

1. **As fatias baixavam de `resolve/main/`** — alvo móvel. Uma refeitura futura poderia
   produzir outro corpus sem erro e sem aviso. Tivemos sorte: o OpenWebMath está fixo
   numa revisão anterior à nossa coleta. Sorte não é reprodutibilidade.
2. **O `checksum_index` das coletas não era checksum.** Era o hash da **forma**
   (`{linhas, colunas}`). Dois parquets de conteúdo completamente diferente, com as
   mesmas linhas e colunas, hasheavam igual — sob um nome que promete o contrário, e
   assim desde 2026-08-06. Foi descoberto porque a verificação profunda acusou **878
   parquets "alterados" que estavam intactos**. A resposta certa a um alarme não é
   assumir adulteração nem silenciar o alarme: é descobrir o que ele compara.

Treze testes cobrem esse subsistema, e cada um **estraga** algo e exige que o
verificador acuse: byte trocado, manifesto adulterado para "legalizar" a fraude, raiz
editada à mão, arquivo apagado, arquivo a mais. Mais dois protegem propriedades sem as
quais o resto não vale: **idempotência** (construir duas vezes dá o mesmo hash —
confirmado nos 21,79 GB reais) e **etapa declarada e ausente é erro**, porque um
manifesto de corpus incompleto que confere é o pior resultado possível.

---

## 6. Tokenizer: evidência que não existia para LaTeX

Bake-off de seis variantes, 200.000 resumos do arXiv para treinar, 5.000 reservados
para medir, em CPU, custo US$ 0:

```
var  algo      vocab    §8     fert  ×Qwen3  tok/eq  ×Qwen3  LaTeX 1-tok
A    BPE      40.960   sim   0,9620   0,674    7,31   0,732      2/3
B    Unigram  40.960   sim   0,9816   0,688    8,36   0,837      0/3
C    BPE      32.768   sim   0,9973   0,699    7,49   0,750      2/3
D    BPE      65.536   sim   0,8966   0,629    6,97   0,698      2/3
E    BPE      40.960   NÃO   1,3245   0,929    9,46   0,947      0/3
F    Qwen3   151.643    —    1,4263   1,000    9,99   1,000      0/3
```

Round-trip 100% e consistência numérica em todas as seis; 35 subáreas medidas, pior
razão 1,215 contra o limite de 1,25.

**Três leituras, e duas contrariam o próprio documento de projeto.**

1. **As regras de pré-tokenização estão vindicadas, e o teste era do documento.** O
   protocolo estipulava: *"se E empatar com A, a seção está errada e o documento precisa
   ser revisado."* Não empatou. A e E são idênticas exceto pelas regras de
   pré-tokenização, e A dá 27% menos tokens em prosa e 23% menos em equações. E é a
   única variante que falha a meta de fertilidade.
2. **BPE bate Unigram em LaTeX — a evidência que faltava.** O documento declarava que
   Bostrom & Durrett (2020) mostram vantagem do Unigram em linguagem natural e que
   *não há evidência publicada para LaTeX/Física*. Produzir essa evidência era a
   contribuição prometida, e ela saiu contra a previsão implícita: **BPE ganha**, com
   margem seis vezes maior em equações (13%) que em prosa (2%). E o mecanismo é
   observável, não só o número: o Unigram aprendeu **0 de 3** sequências LaTeX de teste
   como token único; o BPE aprendeu **2 de 3** (`\frac` e `\partial`). A poda iterativa
   do Unigram não retém essas unidades; a fusão do BPE retém.
3. **V = 40.960 não é o melhor, e a métrica intrínseca não pode decidir.** A ordem é
   monotônica nas três métricas — 32k → 41k → 64k, quanto maior melhor em toda a faixa
   testada. **Mas a fertilidade não vê a troca que decide:** a embedding é 16% do
   modelo de 150 M com V = 40.960, e seria ~24% com 65.536 — parâmetros gastos em
   embedding em vez de camadas. Nenhum número desta seção autoriza mudar a decisão de
   vocabulário; o que ele autoriza é **incluir V = 65.536 no bake-off de verdade**, em
   vez de tratar 40.960 como decidido.

Um negativo honesto, que reportamos porque ele restringe a leitura das outras três
linhas: **a meta de fertilidade em equações falha em todas as seis variantes.** A
melhor é D com 0,698 contra um teto de 0,65. Provavelmente é o corpus: resumos têm
matemática inline curta, e o alvo pode ter sido calibrado supondo texto pleno com
equações em display — justamente o que a §5.4 mostra que a fatia de resumos não tem.

---

## 7. O barramento de verificação

Cinco verificadores de seis estão implementados: **simbólico** (equivalência via CAS),
**dimensional**, **numérico**, **de limites** e **de invariantes** (leis de
conservação). Falta o **sandbox** de execução, que exige gVisor ou Firecracker — a
alternativa de `exec()` com builtins restritos está descartada no próprio documento de
projeto como trivialmente evadível, então não há atalho local.

Os defeitos encontrados nesse subsistema são instrutivos porque **nenhum deles produz
uma exceção**:

| defeito | consequência |
|---|---|
| `SIGALRM` não existe no Windows | O `AttributeError` era engolido por um `except` genérico, **nenhuma** expressão parseava, e o barramento devolvia `INCONCLUSIVE` em tudo |
| `split_symbols` parte identificadores | `hbar → a*b*h*r`, `eps → e*p*s`. Passava despercebido porque **os dois lados sofriam a mesma mutilação** |
| Namespace do SymPy colide com Física | `E` era o número de Euler; `Q` e `N` eram objetos do SymPy; `gamma` e `beta`, funções especiais |
| Precisão float64 no verificador numérico | Reprovava toda resposta correta |
| **Subscrito nomeado virava produto** | `E_{cin}` → `E_{c}*(i*n)`, com `i` e `n` vazando como grandezas livres |

O último merece destaque porque a gravidade não é o subscrito ilegível — é o
**colapso**. Produto comuta, nome não: `\rho_{xy}` e `\rho_{yx}` viravam o **mesmo
símbolo**, e a resistividade Hall é antissimétrica (ρ_xy = −ρ_yx). O mesmo para
`g_{\mu\nu}` contra `g_{\nu\mu}`. O parser executava por baixo exatamente a
transformação que o documento de ingestão lista como **rejeitada** — "normalizar
posição de índices" — e que o canonicalizador se recusa a fazer de propósito.

A correção ficou em duas camadas, e a razão é estrutural: a normalização é de ingestão,
mas `verify/symbolic.parse` aplica **a mesma função** — importada, não copiada —
porque quatro dos cinco consumidores do barramento (rollout de RLVR, gabarito de
benchmark, admissão de sintético e auto-checagem em inferência) entregam LaTeX que
nunca passou pelo pipeline de ingestão. Normalizar apenas na ingestão deixaria sem
proteção justamente o RLVR, onde símbolo trocado vira gradiente.

Ficam declaradamente de fora: índice contravariante (`T^{\mu}_{\nu}`), subscrito com
operador (`k_{n+1}`) e `\mathbf`. Corrigir o primeiro exige representar tensor com
índices no schema — decisão de dados, não de parser.

---

## 8. Modelos de representação

### 8.1 ΦEmb — supervisão por citação

O par de citação é a fonte barata de supervisão: a citação de A para B é um rótulo de
relevância que ninguém precisou anotar. Treinamos um `all-MiniLM-L6-v2` de 23 M de
parâmetros com InfoNCE sobre pares extraídos do grafo (6.564.111 pares de treino;
88.807 documentos citados distintos na validação).

O que a montagem mediu, e que é resultado limpo de passagem:

| Lote | pares/s | negativos por âncora |
|---|---|---|
| 8 (o que o SciBERT aguentava) | 4,1 | 7 |
| 32 | 11,6 | 31 |
| **128 (escolhido)** | **16,2** | **127** |

O **ΦEmb/MiniLM de 23 M bate o ΦEmb/SciBERT de 110 M** com p = 0,0054. O menor com 127
negativos no lote vence o maior com 7: negativos in-batch são limite de **qualidade**
do contrastivo, não só de velocidade, e agora está medido em vez de citado.

Uma ressalva de leitura que vale além deste projeto: **a perda não é comparável entre
lotes.** O MiniLM registra ~1,5 contra 0,3–0,5 do SciBERT, e isso não é pior — InfoNCE
com 128 negativos tem piso mais alto que com 8 (ln 128 ≈ 4,85 contra ln 8 ≈ 2,08).

### 8.2 O portão G1, e por que ele está sob remedição

Esta subseção reporta um resultado e depois o retira. Os dois passos são o conteúdo.

**A medição de 2026-08-27**, com 2.000 candidatos:

| modelo | params | nDCG@10 | recall@1 | recall@10 |
|---|---|---|---|---|
| ΦEmb-T4 (nosso) | 23 M | **0,4657** | 0,262 | **0,708** |
| GTE-large (genérico) | 335 M | 0,4628 | **0,278** | 0,677 |
| ΦEmb/MiniLM local | 23 M | 0,4579 | 0,254 | 0,700 |
| PhysBERT (alvo do G1.1) | 109 M | 0,2752 | 0,146 | 0,425 |
| SciBERT (base) | 110 M | 0,2074 | 0,109 | 0,328 |

A leitura registrada na época: **G1.1 passou de verdade** (+0,190 de nDCG sobre o
PhysBERT, pareado com p = 0,0000 — margem grande, significativa, sem ressalva);
**G1.2 passou pela letra e a margem não é evidência de superioridade** (+0,0029 sobre
o GTE-large, com o pareado em recall@1 dando empate e o GTE à frente na contagem:
196 discordantes contra 165, p = 0,114). A afirmação que sobrevivia a escrutínio era
de **eficiência de parâmetros** — um modelo de 23 M empata com um de 335 M, 1/14,8 do
tamanho — e não "batemos o GTE-large", que +0,003 não sustenta.

**Em 2026-09-06 descobrimos que o protocolo que produziu essa tabela tinha teto.**

O G1 avalia recuperação dentro do lote: `sim = âncoras @ positivos.T`, e a resposta
certa da linha *i* é a **coluna** *i*. Isso só é tarefa bem posta se cada coluna for
distinta. Duas coisas quebravam isso. Primeiro, a amostra era `val.head(n)`, e a ordem
do arquivo não é neutra — a mediana do comprimento da âncora cai de ~1.180 para ~870 do
início ao fim, e o primeiro bloco de 2.000 fica no percentil 94. Pior: em `head(2000)`
havia apenas **1.147 textos positivos distintos**, com **62% das linhas** num positivo
repetido e um deles aparecendo **28 vezes**. Textos byte-idênticos dão cosseno idêntico
e o desempate do `argsort` é arbitrário. Segundo, âncoras repetidas compartilham um
ranking, então no máximo uma delas pode ter posto 1.

Teto de um modelo **perfeito**, medido:

| pool | recall@1 | nDCG@10 |
|---|---|---|
| `head(2000)` — o que o G1 usou | **0,5235** | **0,7562** |
| `sample(2000)` | 0,9364 | 0,9761 |
| sorteio + desduplicação | **1,0000** | **1,0000** |

O G1 reportou recall@1 de 0,2620 contra um teto de 0,5235. **Metade do que parecia
limitação do modelo era o protocolo.**

O empate era *justo* entre modelos — todos sofriam igual —, e é por isso que a tabela
parecia válida. O que o teto destruía era a **margem**: o G1.2 se decidiu em +0,003 de
nDCG@10, e ruído arbitrário em 62% dos itens não deixa +0,003 sobreviver. Pelo mesmo
motivo, parte dos discordantes do McNemar era moeda, não discordância entre modelos.

**E o mesmo defeito estava no treino.** Medido em `pares_treino.parquet` (6.564.111
linhas, 667.304 documentos citados distintos):

| pares | `head` → documentos | sorteio → documentos | razão |
|---|---|---|---|
| 20.000 | 984 | 17.837 | **18,1×** |
| 400.000 | 17.844 | 191.300 | **10,7×** |
| 1.500.000 | 67.232 | 390.966 | 5,8× |

O run que produziu o campeão atual do G1 treinou com **17.844** documentos citados
distintos, onde o sorteio do mesmo tamanho daria **191.300**. Mesma GPU, mesmo tempo.

**O que isto não diz.** Não diz que o ΦEmb é melhor ou pior do que se pensava. Diz que
**não se sabe**: os números vieram de um protocolo com teto de 0,76, e o modelo veio de
um treino com 1/10,7 da diversidade disponível. A remedição no pool corrigido responde
a primeira metade; a segunda exige treinar de novo — 36 minutos de T4 gratuita.

O conserto: uma função de preparação de pool que sorteia com semente, desduplica por
texto e **levanta exceção** se o teto não for 1,0. O avaliador do treino usa o mesmo
pool, porque é ele que elege os checkpoints "melhor". E `teto_do_pool` passa a sair no
artefato versionado, porque um nDCG@10 de 0,4657 não diz, sozinho, se o modelo é
mediano ou se o protocolo é.

### 8.3 Duas alavancas de escala, ambas planas

Testadas de forma independente, mesma base e mesmo protocolo de 2.000 candidatos:

| variação | nDCG@10 | vs. campeão | McNemar |
|---|---|---|---|
| **campeão: 400 mil pares, 127 neg.** | **0,4579** | — | — |
| 400 mil pares, **511 neg.** (GradCache) | 0,4486 | −0,0093 | p = 0,636, empate |
| **1,5 M pares**, 127 neg. | 0,4520 | −0,0059 | p = 0,950, empate |

Nenhuma move a agulha, e as duas nominalmente **pioram**. Os testes pareados dizem
empate, então o enunciado honesto não é "piorou": é **não há ganho detectável**, com
218 e 256 discordantes.

Sobre os negativos: 511 contra 127 é a faixa alta da literatura contra a média, e ainda
assim nada. Isso não contradiz o achado de que 127 > 7 — contradiz a extrapolação de
que mais é sempre melhor. A curva satura entre 127 e 511, não entre 7 e 127.

Sobre os dados: o run de 1,5 M foi interrompido em 38% por platô medido. **E a §8.2
recomenda suspeitar desse platô**: ele foi justificado por "256 mil pares a mais,
metade inéditos, e o nDCG oscilou sem tendência" — mas esses pares saíam de **67 mil**
documentos, e exaurir um conjunto pequeno de documentos se parece exatamente com um
platô de dados. É a primeira conclusão do repositório a ser reaberta por uma medição
posterior sem ter sido refutada diretamente.

### 8.4 O sistema híbrido (T1b)

1.000 consultas, universo de 88.807 documentos citados, profundidade 50:

| sistema | r@1 | r@10 | r@50 | nDCG@10 | McNemar vs. fusão (k=10) |
|---|---|---|---|---|---|
| BM25 | 0,067 | 0,236 | 0,401 | 0,1399 | p = 0,00073 (fusão vence) |
| ΦEmb | 0,055 | 0,233 | 0,428 | 0,1327 | p = 0,00018 (fusão vence) |
| **ΦEmb + BM25 (RRF)** | 0,068 | 0,271 | 0,446 | **0,1584** | — |
| + ΦRank (base MiniLM) | 0,064 | 0,254 | 0,446 | 0,1493 | p = 0,118 (empate) |

**A fusão vence os dois recuperadores isolados com significância.** O reranqueador
inicializado da mesma base do recuperador **não acrescenta nada mensurável**.

O caminho até esse reranqueador contém o defeito mais instrutivo do projeto. A
mineração de negativos difíceis tomava o top-K do denso menos a citação verdadeira — o
que rotula NEGATIVO tudo que o recuperador põe no topo. O modelo aprendeu a regra mais
simples que separa os dois: *se o recuperador gostou, é negativo*. Medido:

| | negativos do índice | negativos do RRF top-50 real |
|---|---|---|
| Spearman(posição na fusão, escore) | **+0,179** | **−0,466** |
| consultas com rho > 0 | **83%** | **0%** |

Com posição menor = melhor, o sinal desejado é negativo: a primeira coluna é um
reranqueador que **prefere a cauda do recuperador**, e cujo nDCG era 0,0179 — pior que
ordem aleatória. A correção é montar o grupo de treino com a distribuição **exata** da
avaliação. Feito isso, "estar no topo" deixa de predizer o rótulo, o reranqueador
concorda com o recuperador (−0,466)... **e por isso mesmo não acrescenta nada.** Ele
re-deriva a ordem que a fusão já produziu.

Onde está a folga: `recall@50` de 0,446 é o teto de um reranqueador perfeito, contra
0,1584 de hoje — **2,8× dentro dos candidatos que já recuperamos**. A folga existe e é
grande; o que faltava era um modelo com conhecimento **além** do ΦEmb para explorá-la.

Seis hipóteses foram medidas e descartadas nesse diagnóstico, e ficam registradas para
não serem reinvestigadas: inferência incorreta do backend `sdpa` (dif. máx. 4e-6 contra
`eager`, ordenação idêntica); negativos de outra população (100% dos minerados são
documentos citados); grau de citação como atalho (positivo 113,6 contra 8,5, mas
Spearman(grau, escore) = +0,056, p = 0,17); atalho de formato (14 features, melhor AUC
0,575); desempate pelo índice 0 no `argsort` (embaralhar dá números idênticos); e "o
modelo não lê a consulta" — que veio de uma amostra de **16** documentos e se inverteu
com 457 (ler a consulta vale +0,143 ± 0,059).

### 8.5 T1c — é domínio, não diversidade

O diagnóstico da §8.4 faz uma predição falsificável: se o reranqueador empata por
**redundância informacional** com o recuperador, um cross-encoder de base **diferente**
deve bater a fusão. Testamos com duas variantes de 109 M contra o controle de 23 M,
onde **só `--base` muda** — os outros seis hiperparâmetros são byte a byte os do
controle, com um teste que falha se algum escorregar.

```
variante   base                       params  acc@1  +ΦRank       Δ  disc  p(k=10)
controle   MiniLM-L6 (= a do ΦEmb)       23M  0,498  0,1483  -0,0093   229   0,1458
gte        gte-base (geral forte)       109M  0,510  0,1530  -0,0046   220   0,6371
phys       physbert (Física)            109M  0,566  0,1666  +0,0090   237   0,0062

fusão RRF = 0,1576 nas três · 2.000 consultas · profundidade 50 · teto 0,4495
```

**A leitura pré-registrada que se aplica é a terceira das quatro: só a variante de
domínio vence ⇒ o mecanismo é conhecimento de domínio, não diversidade de base.**

O que faz disto uma afirmação de mecanismo e não uma coincidência: `gte-base` e
`physbert_cased` têm o **mesmo tamanho** (109 M), a **mesma arquitetura** (`BertModel`),
os **mesmos seis hiperparâmetros**, os **mesmos dados**, a **mesma semente** e o
**mesmo protocolo**. A única diferença entre as duas é o corpus de pré-treino.
**Diversidade de base não basta. Capacidade não basta.**

A composição completa, pela primeira vez, bate a fusão sozinha:

| | fusão | + ΦRank de PhysBERT |
|---|---|---|
| recall@1 | 0,0700 | 0,0690 |
| **recall@10** | 0,2675 | **0,2890** |
| recall@50 | 0,4495 | 0,4495 |
| **nDCG@10** | 0,1576 | **0,1666** |

No pareado em k = 10 a fusão ganha 97 e o reranqueador ganha **140**, com 237
discordantes (p = 0,0062, abaixo do limiar de Bonferroni de 0,025 registrado antes de
medir). Em k = 1 é empate (p = 0,930) — o ganho é na cauda do top-10, não no primeiro
lugar, que é o que se espera de reordenação.

#### 8.5.1 A assimetria, que é o achado mais interessante

**O PhysBERT é um recuperador ruim neste benchmark**: nDCG 0,2752 contra 0,4657 do
ΦEmb — perde por 0,190, e foi exatamente essa a medição que fechou o G1.1. **E é a
melhor base de reranqueador das três.**

Pré-treino de domínio não fez um bi-encoder bom aqui, e fez um cross-encoder bom. A
assimetria não era prevista, e é ela que dá uma receita operacional: **para reordenar,
partir de um modelo do domínio; para recuperar, ajustar um modelo geral pequeno.**

Não temos explicação mecanística e não vamos inventar uma. A hipótese barata de testar
é que o cross-encoder pode usar interação termo a termo entre consulta e documento,
onde vocabulário de domínio rende, enquanto o bi-encoder precisa comprimir o documento
num vetor **antes** de ver a consulta. É especulação até alguém medir.

#### 8.5.2 Duas ressalvas de execução, declaradas

**As duas corridas são combináveis, e há evidência disso.** O braço `gte` rodou numa
execução separada, e o controle saiu **byte a byte idêntico** nas duas — mesmos
sistemas, mesmo pareado, p = 0,14584 sobre 229 discordantes em ambas. É isso que
licencia comparar `gte` com `phys`; sem essa verificação, a comparação seria entre
protocolos e não entre bases.

**A primeira corrida do `gte` morreu por defeito nosso, não dele.** `thenlper/gte-base`
guarda os pesos em fp16; o `transformers` recente carrega no dtype do checkpoint por
padrão; e o `GradScaler` recusa desescalar gradientes fp16, porque AMP exige
pesos-mestres em fp32. Ficou invisível até ali porque MiniLM e PhysBERT são fp32 —
**qualquer base fp16 quebraria** —, e o custo foi um braço inteiro do experimento.

#### 8.5.3 O que entrou no sistema, e o que a proveniência dizia de errado

O ΦRank de PhysBERT foi instalado como componente do sistema, com o manifesto de etapa
gravado junto dos pesos, registrando a medição que o justifica **e** o que ela não diz
— que este reranqueador generalize para outro recuperador, outro benchmark ou outro
domínio **não foi medido**.

Duas notas de honestidade sobre esse passo. A primeira: o `phirank.json` do modelo saiu
do treino afirmando *"inicializado do MiniLM, a mesma base do ΦEmb campeão"* — a string
era fixa no treinador e valia para qualquer base. O arquivo do modelo instalado **não
foi reescrito**: é o que a execução produziu, e reescrever proveniência depois do fato é
pior que registrar a divergência. A segunda: as duas linhas de configuração que põem o
reranqueador no sistema são uma linha cada, e uma linha se reverte sem nada quebrar —
o sistema voltaria a compor com um reranqueador que a medição diz ser no-op, a métrica
cairia 0,009, e nenhum teste ficaria vermelho. Por isso há um teste que **lê o fonte** e
tranca as duas.

### 8.6 O que a T4 gratuita destravou

| | RX 7600 (DirectML) | Tesla T4 (CUDA) |
|---|---|---|
| vazão | 20–26 pares/s | **181,6 pares/s** |
| 400 mil pares | ~13 h, 3 mortes de processo | **36 min, sem queda** |
| precisão | fp32 obrigatório | fp16 + GradScaler |
| VRAM | estourava no lote 128 | 1.508 MB de 16 GB |

**7 a 9× mais rápido**, e o número que muda o plano do programa: um par são duas
sequências de 192 tokens, então 181,6 pares/s ≈ **69.700 tokens/s**, contra ~7.700
localmente.

```
20 B tokens ÷  7.700 tokens/s = 37 DIAS      (GPU local)
20 B tokens ÷ 69.700 tokens/s = ~80 HORAS    (T4 gratuita)
```

80 h são menos de três semanas da cota gratuita de 30 h/semana, em sessões de até 9 h
com retomada. **O ΦEnc sai de "impossível nesta máquina" para "questão de agendar"** —
e é estimativa conservadora, porque o lote ficou em 128 de propósito para isolar o
dispositivo e a T4 usou 1,5 dos seus 16 GB.

Cinco defeitos separaram "o notebook pronto" de "o notebook rodando", e **nenhum era do
treino** — todos eram diferença entre o ambiente local e o remoto: conta sem verificação
por telefone (a imagem de CPU era fixada em silêncio), campos de configuração que a
documentação apresenta como ligadores de GPU e que são mortos, descompactação
automática de `.zip` no upload (o hash do código deixou de ser conferível), mudança de
layout do diretório de entrada, e subprocesso que não herda `sys.path`. E dois defeitos
de **visibilidade**, que foram os caros: `check=False` no subprocesso do treino
transformava treino morto em notebook `COMPLETE` — uma execução terminou "com sucesso"
sem nenhum modelo —, e a API devolveu log de **0 bytes em três execuções seguidas**,
transformando diagnóstico em adivinhação a quatro execuções de custo.

---

## 9. ΦEnc — a hipótese central do programa

### 9.1 A hipótese

O ΦEnc é o único modelo do programa treinado **do zero**, autorizado porque nesta
escala (150 M) o dado é excedente. O objetivo de treino é MLM com taxa de mascaramento
de 30%, sem NSP, mais **uma** adição específica de Física, a ser ablacionada:

> **Mascaramento consciente de equações.** Em uma fração dos exemplos, mascarar uma
> **equação inteira** em vez de tokens aleatórios, forçando a reconstrução a partir do
> contexto em prosa. Hipótese: ensina a relação entre a descrição verbal de um fenômeno
> e sua expressão formal — exatamente a competência que a recuperação de Física exige.
> Se não ajudar, é descartado e **o negativo é publicado**.

É a única contribuição de arquitetura de treino que o programa propõe, e a §5.4 existe
porque ela **não é possível sobre texto de que as equações foram removidas**.

### 9.2 Três decisões de desenho, e a medição de cada uma

**Orçamento igual entre os braços.** `n_alvo = round(taxa × mascaráveis)` vale para os
dois braços; no tratado escolhe-se uma equação que caiba e o resto é aleatório. Sem
isso, a ablação mediria "consciente de equações" **e** "mascara 6× mais".

**Só equações em display.** Medido em 120 documentos do RedPajama-arXiv:

```
tipo      por doc   tokens: p10  mediana  p90   p99     max
display      40,6            39       79  214   678  19.587
inline      260,9             4        7   19    39     103
```

A mediana de 7 tokens do inline é uma **variável** (`$\rho$`), não uma equação. A
primeira versão tratava qualquer `$…$`, e a ablação teria medido o nada. 91,7% dos
documentos têm display, e 99,8% delas cabem no orçamento.

**A equação vai inteira para `[MASK]`, sem o esquema 80/10/10** — registrado como
escolha, não achado: se o tratamento ganhar, "ganhou porque mascara 100% com `[MASK]`"
segue sendo explicação viva, e a ablação que a mata está escrita na docstring.

### 9.3 O defeito que teria invalidado a ablação inteira

O padrão de display era `re.compile(r"^(?:...)")` usado com `DISPLAY.match(texto, i)`.
`Pattern.match(s, pos)` **já** ancora em `pos`, mas o `^` continua se referindo ao
início **real** da string — então o padrão nunca casava para nenhuma equação que não
começasse no caractere 0.

Medido: **0 de 120** documentos tratados, com `recaida_sem_equacao` em 120, contra
91,7% de documentos com display medidos por **outra** sonda, que fatiava a string antes
de casar. Foi a **discordância entre as duas sondas** que localizou o erro.

> Sem o contador `fracao_tratada`, a ablação teria rodado inteira comparando aleatório
> com aleatório e reportado empate — e o empate teria sido publicado como refutação da
> hipótese central do programa.

Depois do conserto: fração tratada de 0,883 com `p_equacao = 1,0`.

### 9.4 A verificação que prova que o MLM está certo

```
perda inicial 10,7343   contra   ln(40.960) = 10,6204
```

Um MLM não treinado prevê uniforme, então a entropia cruzada inicial **é** `ln(V)`. Se
o mascaramento apagasse os alvos, se o `-100` estivesse no lugar errado, ou se os alvos
viessem da entrada já mascarada, esse número sairia diferente e **nenhum outro sintoma
apareceria**. Está travado num teste de regressão.

Um segundo defeito foi achado por um teste que falhou. O agendamento WSD recebia
`frac_warmup = 0,03` e calculava o warmup como `total × 0,03`. Com isso, estender o
treino de 10 mil para 200 mil passos alongaria o warmup de 300 para 6.000 — e a taxa de
aprendizado do passo 5.000, **que já havia sido dado com o pico**, passaria a valer
8,3e-4. O agendamento deixava de ser extensível, que é o argumento inteiro para
preferir WSD a cosseno. Hoje `passos_warmup` é absoluto, derivado uma vez pelo plano e
guardado no checkpoint. *A tentação era ajustar o número esperado no teste.*

### 9.5 Os dados: 2,00 B tokens preparados e conferidos

**2.001.270.262 tokens** de 143.810 documentos, em 244.295 sequências de 8.192,
ocupando 6,0 GB, preparados a ~1,0 milhão de tokens/s em 35 minutos.

Por que 2 B e não mais: no proxy de 48 M, 1 B por braço custa 3,2–5,3 h de T4, e a
ablação inteira (controle + tratado) cabe em 6,4–10,7 h — dentro da cota gratuita de
30 h/semana, com cada braço numa sessão de 9 h. Preparar além disso seria preparar dado
que não dá para treinar de graça.

A conferência, antes de declarar pronto, medida em 300 sequências sorteadas **do
binário** — que é o que o treino lê:

| | preparado | corpus cru |
|---|---|---|
| matemática | **37,8%** | 38,8% |
| display | **25,0%** | 25,2% |
| `fracao_tratada` | **0,903** | — |
| taxa efetiva de mascaramento | **0,3000** | pedida 0,3000 |
| faixa de ids | [2; 40.958] | vocabulário 40.960, dentro |

**O viés que a conferência pegou.** A primeira preparação usou as **8 primeiras** de 44
partes do corpus:

```
partes 0-7  (usadas)      math 32,4%   display 18,8%
partes 8-43 (NÃO usadas)  math 42,7%   display 29,5%
```

As partes que ficaram de fora tinham **57% mais equação em display**. As partes não são
intercambiáveis. O viés ia **contra** a hipótese — menos equação enfraquece o
tratamento —, que é o lado seguro; um viés que ajuda não deixa de ser viés. Consertado
com partes sorteadas por semente, **a lista** gravada no manifesto (com sorteio, saber
que 9 partes entraram não permite reconstruir quais) e uma guarda que levanta se a
retomada usar outra semente.

Antes disso, a preparação levava 16 horas, das quais **98% em uma única função** —
O(spans × tokens) em Python puro, 188 milhões de iterações por documento. Vetorizada
com `searchsorted`: de 33 mil para ~1,0 milhão de tokens/s, com **0 divergências em
1.181.249 tokens** contra a implementação lenta.

### 9.6 O gargalo atual: não existe avaliação

**Esta é a lacuna que impede qualquer alegação sobre a hipótese central.** O protocolo
pede três avaliações do ΦEnc — recuperação de Física, MLM em texto denso em equações, e
uma sonda de estrutura tensorial — e **nenhuma das três existe**. Um `phienc.json` com
perda baixa não é veredito sobre o mascaramento consciente de equações: é evidência de
que o laço de treino funciona, que é outra coisa.

---

## 10. Custo e infraestrutura

O modelo de custo é **processar localmente, alugar apenas a GPU**: o corpus bruto
(10–20 TB potenciais) nunca sai do disco local, e só os shards tokenizados (20–80 GB)
subiriam para um pod alugado. Armazenamento recorrente cai de US$ 110–350/mês para
~US$ 0, mais ~US$ 1–3/mês só para checkpoints.

Quatro conclusões de infraestrutura que valem mais que a escolha da placa:

- **A100 deixou de ser boa compra.** A H100 custa 1,5× por hora e entrega 3× mais
  trabalho: é duas vezes mais barata por unidade de computação.
- **RTX 4090 e H100 empatam em custo por FLOP.** A decisão não é preço — é se o modelo
  cabe em 24 GB.
- **Não é preciso pagar egresso do arXiv.** Fatias já processadas são gratuitas: ~10–15
  B tokens de Física por US$ 0.
- **OCR não é necessário no começo.** O arXiv fornece fonte LaTeX; OCR entra para teses
  e livros, depois do G1.

Único custo não recorrente relevante identificado: um HD externo de 8–16 TB
(~US$ 150–250), ou o disco que já se tenha.

---

## 11. Estado consolidado

| Frente | Estado | Evidência |
|---|---|---|
| S1 · espinha de metadados | **completo** | 1,59 M + 4,61 M; junção 99,1% |
| S2 · classificador de Física | **completo** | acurácia 0,954; FP 2,4–3,7% |
| S3 · fatias do corpus | **completo** | **27,75 B tokens**, custo zero |
| Barramento de verificação | **5 de 6** | falta `sandbox` (gVisor/Firecracker) |
| G1.5 · corpus por um hash | *metade fechada* | 21,79 GB por **um** hash; refazer do zero depende de fonte mutável |
| T1a · ΦEmb na T4 | **medido** | 181,6 pares/s; destrava o ΦEnc |
| T1b · busca híbrida | **composição completa** | RRF vence os isolados (p < 0,001) |
| T1c · ΦRank de domínio | **fechado** | PhysBERT vence a fusão (p = 0,0062); `gte-base` empata (p = 0,637) |
| **ΦEmb / portão G1** | **SOB REMEDIÇÃO** | protocolo tinha teto de 0,7562; treino via 1/10,7 dos documentos |
| ΦEnc · dado | **destravado**, US$ 0 | ambiente de equação em 84,9% vs. 0,0% |
| ΦEnc · dados preparados | **2,00 B tokens** | 244.295 sequências; partes sorteadas |
| ΦEnc · código | *escrito, não treinado* | fumaça em CPU: 10,7343 vs. ln(V) = 10,6204 |
| **ΦEnc · avaliação** | **NÃO EXISTE** | é o gargalo |
| Revisão humana do peS2o | *amostra refeita, julgamento pendente* | 200 resumo + 200 texto pleno, estratificado |
| Tokenizer próprio | *bake-off feito, treino não* | §11.2 do protocolo não executado |
| PhysBench | *não iniciado* | — |

Dos cinco critérios do portão G1: **G1.1 tinha sido considerado passado e volta a
aberto** com a remedição; **G1.2 não passou**; **G1.3 e G1.4 não foram tocados** (o
ΦEnc não foi treinado, o ΦOCR foi adiado por decisão); **G1.5 está meio fechado**.

**Nenhum degrau do Tier 2 foi iniciado.**

---

## 12. Discussão

### 12.1 Uma armadilha, seis ocorrências, um custo medido

A contribuição metodológica deste artigo não é uma técnica; é um catálogo. Seis vezes,
de forma independente, este repositório amostrou dados ordenados por posição em vez de
por sorteio, e seis vezes o resultado foi um número plausível e errado:

| # | onde | o que mediu de verdade | custo |
|---|---|---|---|
| 1 | peS2o amostrado no começo do arquivo | 400 documentos, todos nos 2 primeiros de 277 parquets — **0,67% do corpus**, e 100% resumos | invalidou a revisão humana que decide a confiança no corpus |
| 2 | `val.head(500)` num parquet agrupado | 500 linhas = **35 documentos**; IC de ±0,159 | duas conclusões que se anularam |
| 3 | `pares_treino` cortado por posição | **49,6%** das âncoras da "validação" já estavam no treino | nDCG da composição caiu de 0,139 para 0,020 sobre documentos inéditos |
| 4 | 8 primeiras de 44 partes do RedPajama | partes omitidas tinham **57% mais** equação em display | viés no corpus do ΦEnc, pego antes de treinar |
| 5 | `head(200)` **dentro do conferidor** | 200 linhas do primeiro arquivo | quase reportou "o viés persistiu" sobre o próprio conserto |
| 6 | `val.head(2000)` no protocolo do portão | teto de nDCG@10 de **0,7562**, não 1,0 | invalidou a medição do critério de sucesso do projeto |

Duas observações que consideramos generalizáveis.

**A primeira: o defeito é invisível por construção.** Nenhuma das seis produz exceção,
nenhuma aparece na perda de treino, e todas produzem um número na faixa esperada. A
ocorrência 3 fez a métrica **subir**. A ocorrência 5 aconteceu dentro do código escrito
para detectar a ocorrência 4, e devolveu exatamente o número enviesado que existia para
refutar.

**A segunda: o conserto correto já estava escrito no repositório.** O caminho de
reranking já sorteava, **com a explicação do porquê escrita no código**, e a explicação
nunca foi aplicada ao caminho de embedding. Uma lição documentada em um módulo não se
propaga sozinha para os outros; o que propaga é uma guarda que **levanta exceção**.

É a razão de o conserto da ocorrência 6 não ser "passar a sortear", mas **uma guarda
que recusa avaliar num pool cujo teto não seja 1,0**, com o teto gravado no artefato de
saída ao lado da métrica.

### 12.2 Negativos que valem tanto quanto os positivos

Seis resultados nulos ou contrários à previsão foram medidos e registrados, e todos
economizaram trabalho futuro:

1. Lote maior no contrastivo (511 negativos) **não** melhora — empate com p = 0,636.
2. 3,75× mais dados de treino **não** melhora — empate com p = 0,950.
3. `stat` **não** é vizinho próximo do domínio de Física (piora de 1,0×), contra a
   suspeita explícita de quem escreveu isto.
4. Negativos de `math` **não** são inúteis apesar de o rótulo depender de uma flag
   ausente do texto (83% de acurácia intra-domínio).
5. `gte-base` — modelo geral forte, do mesmo tamanho do vencedor — **empata** com a
   fusão (p = 0,637); é o braço que transforma "base diferente funciona" em "**domínio**
   funciona".
6. A meta de fertilidade em equações **falha em todas as seis** variantes de tokenizer.

Cada um desses foi barato de medir e caro de supor. O item 5 é o que separa uma
afirmação de correlação de uma afirmação de mecanismo, e custou um braço extra de
experimento.

### 12.3 O que este programa sugere sobre construir modelos de domínio com pouco dinheiro

Três padrões emergiram, e nenhum é sobre arquitetura.

**Medir a fonte antes de comprá-la.** O corpus que destravou a hipótese central do
programa estava no disco havia dezessete dias enquanto se recomendava gastar centenas
de dólares para obtê-lo. O que faltava não era dado nem dinheiro: era um script de 80
linhas medindo presença de ambiente de equação nas quatro fatias já baixadas.

**A cota gratuita muda o plano, não só o cronograma.** 181,6 pares/s contra 20–26
transformaram o ΦEnc de "37 dias, impossível" em "80 horas, questão de agendar". Grande
parte do que este programa entregou é consequência de ter medido a vazão real antes de
projetar o cronograma sobre a estimada.

**Comparação pareada é o que separa resultado de ruído em regime de baixo orçamento.**
Com 256 candidatos, o erro padrão é ±0,031 e a margem mínima detectável ~0,061 — e a
diferença que interessava era 0,004. Nem 4.000 candidatos resolveriam. O que resolve é
medir os modelos nos **mesmos itens** e olhar só os discordantes: um placar de 32 a 8 em
40 discordantes é evidência que o teste não pareado descarta.

---

## 13. Trabalhos relacionados, e o que é novo aqui

O uso de grafo de citação como supervisão de recuperação científica é conhecido
(SPECTER). A mineração de negativos difíceis a partir do índice é padrão (DPR, ANCE), e
o problema de falsos negativos nela é documentado (RocketQA), que propõe filtragem por
um cross-encoder. **O nosso achado da §8.4 é vizinho e distinto:** o problema não é o
negativo estar errado, é o **critério de seleção do negativo ser correlacionado com o
escore do modelo que será reordenado**. O sintoma não é queda de precisão — é uma
**inversão de correlação mensurável**, diagnosticável com um Spearman que custa nada.

O achado da §8.5 — domínio, não diversidade, com a assimetria bi/cross-encoder — não
temos conhecimento de trabalho anterior que o isole com um controle de mesmo tamanho e
mesma arquitetura. É a alegação de novidade mais forte deste artigo, e a que mais
precisa de busca bibliográfica antes de submissão.

Para as §12.1 e §5.4, esperamos que **exista** literatura anterior: n efetivo em dados
agrupados e vazamento por divisão posicional são elementares o bastante. Se existir,
essas seções deixam de ser contribuição e passam a ser confirmação — o que é informação
útil, não um problema.

---

## 14. Ameaças à validade

1. **Um domínio, um encoder base, um grafo.** Física, MiniLM-L6, OpenAlex. Nada aqui
   estabelece que as taxas transferem.
2. **Benchmark próprio, sem juízo humano de relevância.** "Relevante" = "citado" é uma
   proxy conhecidamente enviesada: favorece papers citáveis e campos com cultura de
   citação densa, e não captura relevância que ninguém citou. Os valores de nDCG deste
   artigo **não são comparáveis** a benchmarks com anotação humana.
3. **O portão G1 está sob remedição, e os números da §8.2 são reportados como
   histórico.** Qualquer uso deles como resultado atual seria incorreto.
4. **A hipótese central do programa não foi testada.** O ΦEnc tem dados e código; a
   ablação não rodou e a avaliação não existe.
5. **A taxa de contaminação do corpus filtrado não está medida.** Os falsos positivos
   de 1,5–13,6% que justificam o limiar foram medidos em **resumos do arXiv**; o corpus
   é texto pleno, outra distribuição. A amostra estratificada (200 resumos + 200 textos
   plenos) está preparada e o **julgamento humano está pendente** — é a tarefa que só o
   dono do projeto pode fazer.
6. **A §8.4 não mede o falso negativo residual.** O filtro de co-citação remove uma
   classe de falso negativo que sabemos nomear (9,1% dos negativos minerados eram
   co-citados com o positivo); não sabemos o tamanho do resto.
7. **A taxa de aprendizado não foi reajustada entre escalas** no experimento da §8.5 —
   variável não controlada, declarada antes de ver o resultado.
8. **Os parâmetros das etapas já executadas são reconstruídos do código**, não
   capturados na execução; cada manifesto carrega essa marca.
9. **Veículo realista.** O conteúdo da §12 é um artigo de armadilhas com validade
   externa de um domínio — material de workshop, não de trilha principal. Apresentá-lo
   como mais do que é seria o mesmo erro de calibração que ele denuncia.

---

## 15. Trabalho futuro, em ordem de prioridade

1. **Concluir a remedição do G1** no pool corrigido, e **retreinar o ΦEmb com pares
   sorteados** — 36 minutos de T4 gratuita, e a única forma de saber se os 10,7× de
   diversidade perdida importavam.
2. **Construir a avaliação do ΦEnc** (recuperação de Física, MLM em texto denso em
   equações, sonda de estrutura tensorial). É o gargalo declarado: sem ela, treinar o
   ΦEnc produz um número sem veredito.
3. **Rodar a ablação do mascaramento consciente de equações** — controle e tratado, 1 B
   tokens por braço, 6,4–10,7 h de cota gratuita. Publicar o resultado **qualquer que
   seja o sinal**.
4. **Apurar a revisão humana do peS2o** com a amostra estratificada já preparada,
   fechando a única afirmação sobre o corpus que depende de julgamento.
5. **Reabrir o platô de 1,5 M pares** (§8.3) com pares sorteados, agora que se sabe que
   ele saía de 67 mil documentos.
6. **Implementar `verify/sandbox`** sobre gVisor ou Firecracker — o sexto verificador.
7. **Incluir V = 65.536 no bake-off de verdade** do tokenizer, em vez de tratar 40.960
   como decidido.

---

## 16. Reprodutibilidade e disponibilidade

Todo número deste artigo vem de um artefato versionado: cada etapa grava um manifesto
com hashes BLAKE3 das entradas e saídas, os parâmetros e o SHA do commit. Os scripts
que produzem cada tabela:

| seção | script |
|---|---|
| §5.1 | `scripts/harvest_arxiv.py`, `scripts/harvest_openalex_snapshot.py`, `scripts/build_spine.py` |
| §5.2 | `scripts/train_classifier.py`, `scripts/avaliar_transferencia.py` |
| §5.3 | `scripts/filtrar_hf.py`, `scripts/coletar_redpajama.py` |
| §5.4 | `scripts/medir_equacoes_mutiladas.py` |
| §5.5 | `scripts/auditar_latex.py` |
| §5.7 | `scripts/manifesto_corpus.py --verificar --profundo` |
| §6 | `scripts/bakeoff_tokenizer.py` |
| §7 | `src/phifm/verify/`, `tests/golden/` |
| §8.1–8.3 | `scripts/train_embedding.py`, `scripts/avaliar_encoders.py` |
| §8.4 | `scripts/minerar_do_recuperador.py`, `scripts/avaliar_t1b.py` |
| §8.5 | `kaggle/t1c_phirank.py`, `scripts/train_rerank.py` |
| §9 | `scripts/preparar_dados_phienc.py`, `scripts/train_phienc.py` |
| §12.1 | `src/phifm/training/amostragem.py` |

**O que não é redistribuível:** o corpus. Os resumos do arXiv seguem a licença de cada
submissão, e a licença padrão do arXiv concede ao arXiv o direito de distribuir, não a
terceiros. A fração medida como redistribuível é de **14,8%**. Os scripts de coleta e os
manifestos permitem reconstruir; os bytes não acompanham.

O código e os vinte documentos de projeto são públicos sob licença permissiva. A
intenção de release dos pesos é **aberta** (Apache-2.0/MIT), decidida no Stage-Gate 0.

---

## 17. Referências — nenhuma foi conferida

**Esta seção não é uma bibliografia; é uma lista de tarefas.** As referências abaixo
foram escritas de memória e **nenhuma foi verificada contra a fonte**. Volume, página,
ano e a própria existência do que é afirmado precisam ser conferidos um por um. Uma
citação errada num artigo sobre rigor de medição é a pior forma possível de se
desmentir.

**Razoavelmente confiante no conteúdo, a conferir na forma:**

- Beltagy, Lo & Cohan (2019) — SciBERT.
- Cohan et al. (2020) — SPECTER; representação de documento científico por sinal de
  citação.
- Cormack, Clarke & Büttcher (2009) — Reciprocal Rank Fusion; a origem do k = 60.
- Karpukhin et al. (2020) — DPR; negativos difíceis do BM25.
- Xiong et al. (2021) — ANCE; negativos reamostrados do próprio índice.
- Qu et al. (2021) — RocketQA. **É o vizinho mais próximo da §8.4** e merece leitura
  integral antes de a §13 afirmar distinção.
- Robertson & Zaragoza (2009) — BM25.
- Hoffmann et al. (2022) — Chinchilla; a relação `D* ≈ 20N` da §1.1.
- Taylor et al. (2022) — Galactica. Lewkowycz et al. (2022) — Minerva. Azerbayev et al.
  (2024) — Llemma. Shao et al. (2024) — DeepSeekMath/GRPO.
- Sennrich, Haddow & Birch (2016) — BPE. Kudo (2018) — Unigram.
- Bostrom & Durrett (2020) — Unigram contra BPE em linguagem natural.
- Wilson (1927); Brown, Cai & DasGupta (2001) — intervalos de proporção.
- McNemar (1947).

**Menos confiante, verificar antes de citar:**

- Hellert et al. (2024) — PhysBERT. Temos o identificador do modelo
  (`thellert/physbert_cased`, 109 M, `BertModel`) mas **não** a referência bibliográfica
  com segurança.
- INDUS (NASA/IBM, 2024) — citado como arXiv:2405.10725; não verificado.
- peS2o — o corpus de texto pleno da §5.4. Conhecemos o dataset; não temos autor e ano
  com segurança.
- RedPajama-Data-1T — a fatia arXiv (`togethercomputer/RedPajama-Data-1T`); mesmo caso.
- OpenWebMath; mesmo caso.
- Tao et al. (2024) — vocabulário ótimo cresce com o tamanho do modelo.
- Ali et al. (2024) — eficiência de tokenizer correlaciona com desempenho a jusante.
- ModernBERT — citado como fonte para a taxa de mascaramento de 30%; não verificado.
- Nougat (Blecher et al., 2023) — OCR de PDF acadêmico.

**Falta procurar:** trabalho anterior sobre (a) n efetivo em avaliação de recuperação
com dados agrupados, (b) vazamento por divisão posicional em conjuntos derivados de
grafo, e (c) a assimetria bi-encoder / cross-encoder sob pré-treino de domínio.
