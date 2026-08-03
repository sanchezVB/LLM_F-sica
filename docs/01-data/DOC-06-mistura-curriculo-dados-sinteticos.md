# DOC-06 — Mistura de Dados, Currículo e Motor de Dados Sintéticos

**Projeto:** ΦFM — Phi Foundation Models para Física
**Status:** `RASCUNHO v0.1` — aguardando revisão do Stage-Gate 5
**Cobre:** mistura de dados, curriculum learning, geração sintética verificada; **encerra a Fase 1**
**Depende de:** [DOC-02 §2](DOC-02-aquisicao-corpus.md), [DOC-04 §4](DOC-04-filtragem-dedup-descontaminacao.md), [DOC-05](DOC-05-tokenizer.md), [DOC-00 §2](../00-foundations/DOC-00-project-charter.md)
**Data:** 2026-08-03

---

## 1. Três problemas, um documento

Este documento resolve as três últimas questões antes de existir um shard de treino:

| # | Problema | Por que é difícil |
|---|---|---|
| **1** | **Mistura** — com que peso amostrar cada fonte e cada subárea? | Amostragem proporcional ao tamanho faria `astro-ph` e `cond-mat` dominarem, e faria o material pedagógico — que é minúsculo em volume e enorme em valor — desaparecer |
| **2** | **Ordem** — em que sequência apresentar os dados? | A literatura de curriculum learning para pretraining é **muito mais fraca do que se costuma admitir**. §4 separa o que replica do que não replica |
| **3** | **Lacunas** — como cobrir o que a literatura não contém? | Mecânica Clássica, Eletromagnetismo de graduação e Termodinâmica são estruturalmente sub-representados (DOC-02 §2); e os modos de falha F1–F3 exigem **comportamento de verificação demonstrado**, que quase nunca é escrito |

O problema 3 é o coração do documento e ocupa as §5 a §7.

---

## 2. Mistura de dados

### 2.1 Por que proporcional ao tamanho está errado

| Fonte | Tokens | Fração bruta | Valor pedagógico |
|---|---|---|---|
| RedPajama-arXiv (Física) | 10–14 B | ~35% | Médio — prosa de pesquisa, densa e elíptica |
| Teses | 6–12 B | ~25% | **Alto** — contêm as derivações longas que papers omitem |
| NTRS + OSTI | 5–11 B | ~20% | Variável — muito relatório burocrático |
| StackExchange | 0,3–0,6 B | ~1,5% | **Muito alto** — pergunta e resposta explicativa |
| OpenStax + livros abertos | ~0,01 B | **0,03%** | **Máximo** — texto pedagógico canônico |

Amostragem proporcional daria ao OpenStax **0,03%** do treino. É o material mais bem escrito do corpus inteiro para *ensinar* Física, e o modelo praticamente não o veria.

Aqui entra a distinção estabelecida no DOC-04 §4.1: **impacto de pesquisa e valor pedagógico são eixos separados e frequentemente anticorrelacionados.** A mistura é onde essa separação se paga — os dois escores entram com pesos diferentes.

### 2.2 Métodos

| Método | Como | Custo | Veredito |
|---|---|---|---|
| Heurístico manual | Pesos escolhidos por julgamento | US$ 0 | Baseline; é o que a maioria dos projetos faz e nunca valida |
| **DoReMi** (Xie et al., 2023) ✅ | Treina um modelo de referência e um *proxy*; usa Group DRO para achar pesos que minimizam a perda de pior caso entre domínios | ~2 treinos pequenos | ★ Principiado e, **na nossa escala, barato** |
| Adaptativo online (Skill-It, ODM) | Ajusta pesos durante o treino | Complexidade alta | Adiado — ganho incerto, custo de engenharia alto |

**Selecionado: DoReMi em escala reduzida, com o heurístico como controle.**

Na escala de fronteira o DoReMi é caro e por isso raramente usado. Na nossa não é: dois modelos de ~50 M em 5 B tokens custam **~US$ 5** (mesma aritmética do bake-off do DOC-05 §11). Um método que a maioria dos laboratórios não pode pagar em escala grande é trivialmente pagável na nossa — é uma vantagem estrutural de operar pequeno, e deve ser explorada.

### 2.3 Mistura proposta (hipótese inicial, a ser otimizada)

| Componente | Peso alvo | Épocas | Comentário |
|---|---|---|---|
| Papers arXiv (estratificado pelas 23 subáreas) | 34% | 1–2 | **Estratificado, não proporcional** — `hep-th` e `gr-qc` não podem sumir |
| Teses e dissertações | 14% | 1 | Derivações longas |
| Relatórios de agência (NTRS, OSTI, NIST) | 8% | 1 | Filtrado agressivamente por valor pedagógico |
| Journals CC BY (SCOAP³, DOAJ, SciPost) | 8% | 1–2 | Revisado por pares |
| **Sintético verificado** (§5) | **15%** | 1 | ★ Teto rígido — ver §6 |
| StackExchange (Física, Astro, Math) | 7% | 2–3 | Explicação dialógica |
| Matemática de apoio (`math.*`) | 6% | 1 | Álgebra, análise, EDP, geometria diferencial |
| Código científico | 5% | 1 | SymPy, NumPy, astropy, FEniCS |
| **Livros abertos e clássicos de DP** | **2%** | **4–6** | ★ Upsampling agressivo — minúsculo e insubstituível |
| Texto geral em inglês | 1% | 1 | Antídoto contra degradação de linguagem natural |

Os pesos são **hipótese**, não decisão. O DoReMi os revisa e a §9 exige que a revisão seja medida.

### 2.4 Política de épocas

Muennighoff et al. (2023): até ~4 épocas equivalem quase a dado novo; o retorno decai depois e é praticamente nulo por volta de 16.

**Aplicação por fonte, não global.** O orçamento de repetição é atribuído individualmente: fontes pequenas e excelentes (OpenStax, sintético verificado) repetem 4–6 vezes; fontes grandes e medianas repetem uma vez.

> **Risco associado.** Repetir uma fonte pequena 6 vezes aumenta a chance de **memorização literal**. Se essa fonte tiver qualquer sobreposição com material de benchmark, a memorização vira contaminação efetiva mesmo tendo passado pela descontaminação do DOC-04 §6. **Mitigação:** fontes com fator de repetição ≥ 4 passam por descontaminação em limiar mais estrito, e a memorização é medida diretamente (extração de sufixo em amostra) antes do treino longo.

---

## 3. Empacotamento e comprimento de sequência

Detalhe de baixo perfil com consequência direta em Física.

| Decisão | Escolha | Justificativa |
|---|---|---|
| Comprimento de treino | ΦEnc **8.192**; ΦGen inicia em 4.096 | Derivações são longas; contexto curto as trunca sistematicamente |
| Empacotamento | *Best-fit packing* com **máscara intra-documento** | Concatenação ingênua deixa o modelo atender ao documento anterior — ruído. A máscara em bloco diagonal elimina isso a custo desprezível |
| **Fronteira de corte** | ★ **Nunca no meio de uma equação ou de uma derivação** | Ver abaixo |

> **A regra de corte é específica de Física e importa mais do que parece.** Truncar no meio de um bloco `align` ensina ao modelo que derivações terminam abruptamente — reforçando exatamente o modo de falha **F1** (deriva simbólica), em que o modelo abandona uma derivação no meio. O chunking respeita fronteiras de seção e de ambiente matemático, ainda que ao custo de desperdiçar algum *padding*. Preferimos gastar 2–3% de tokens em preenchimento a ensinar um comportamento errado.

---

## 4. Curriculum learning: o que replica e o que não

### 4.1 Avaliação honesta da literatura

A tentação é projetar um currículo elaborado do fácil ao difícil. A evidência não sustenta isso para pretraining:

| Técnica | Evidência | Veredito |
|---|---|---|
| **Annealing de qualidade** — dados de máxima qualidade no **fim** do treino | **Forte.** Adotado por Llama 3, OLMo 2, MiniCPM, DeepSeek. Replica de forma consistente | ✅ **Adotar** |
| **Estágios de comprimento de contexto** — treinar curto, estender depois | **Forte.** Prática padrão; reduz custo de atenção nas fases iniciais | ✅ **Adotar** |
| **Currículo de dificuldade** (fácil → difícil) em pretraining | **Fraca e inconsistente.** Muitos ganhos publicados não replicam; a ordenação de dificuldade costuma ser arbitrária | ⚠️ **Testar, não presumir** |
| Currículo de dificuldade em **SFT e RL** | Moderada a boa | ✅ Adotar — mas é assunto do DOC-09/DOC-10, não deste |

**Posição adotada:** os dois primeiros são incorporados por terem evidência sólida. O terceiro entra como **ablação barata** (~US$ 10 na escala de 50 M), não como premissa. Se não produzir ganho mensurável, é descartado e o resultado negativo é publicado (DOC-00 §6.5).

Projetar um currículo de dificuldade complexo por parecer cientificamente sofisticado, sem evidência, seria exatamente o tipo de decisão que este programa se comprometeu a não tomar.

### 4.2 A fase de annealing

Últimos ~10–15% dos tokens de treino, com mistura reponderada:

| Componente no annealing | Peso |
|---|---|
| Livros abertos e material pedagógico | 20% |
| Sintético verificado de alta confiança | 25% |
| StackExchange de alta pontuação | 15% |
| Papers de topo de escore pedagógico | 30% |
| Restante | 10% |

Simultaneamente, decaimento da taxa de aprendizado até próximo de zero. É a fase que mais determina o comportamento final do modelo por unidade de computação — e, por ser curta, é barata de ablacionar.

---

## 5. O motor de dados sintéticos

### 5.1 Por que é obrigatório, e não um complemento

Dois problemas que **nenhuma quantidade de literatura resolve**:

**Problema A — a lacuna estrutural.** Ninguém publica preprints sobre Mecânica Lagrangiana de graduação ou sobre a lei de Gauss. É conhecimento consolidado. O arXiv é o registro da *fronteira*, não do *cânone*. Um modelo treinado só nele terá um perfil bizarro: forte em teoria de cordas, fraco em polias.

**Problema B — comportamento de verificação nunca é escrito.** Um físico, ao terminar uma derivação, confere dimensões, toma limites e checa casos especiais. Esse processo é **quase inteiramente ausente do texto publicado** — o autor o executa e publica apenas o resultado. Os modos de falha **F1, F2, F3 e F10** decorrem diretamente dessa ausência: o modelo nunca viu ninguém verificar.

Dados sintéticos verificados são a única forma de fornecer as duas coisas.

### 5.2 Taxonomia dos geradores

| # | Gerador | Método | Verificação | Ataca | Custo |
|---|---|---|---|---|---|
| **G1** | **Problemas com solução por CAS** | SymPy gera Lagrangiana/Hamiltoniana/configuração de campo, deriva as equações de movimento simbolicamente, resolve | **Por construção** | Lacuna A | US$ 0 |
| **G2** | **Derivações passo a passo** | Cada passo verificado por equivalência simbólica com o anterior | Barramento de verificação | F1 | US$ 0 |
| **G3** | **Análise dimensional** | Gera equações corretas e incorretas dimensionalmente, com a explicação de por quê | Álgebra de unidades | **F2** | US$ 0 |
| **G4** | **Redução em caso-limite** | Parte de resultado conhecido, toma o limite, verifica a redução (`v/c→0`, `ℏ→0`, `T→0`, `r→∞`) | CAS + numérico | **F3** | US$ 0 |
| **G5** | **Conversão de unidades e Fermi** | Cadeias de conversão e estimativas de ordem de grandeza com hipóteses explícitas | Numérico | F4 | US$ 0 |
| **G6** | **Reescrita pedagógica** | LLM reescreve seção densa de paper como explicação didática | ⚠️ **Não verificável mecanicamente** | Lacuna A | ~US$ 15–40 |
| **G7** | **Ancorado em simulação** | Resolve numericamente EDO/EDP reais; gera perguntas cuja resposta é a saída do solver | Por construção | F9 | US$ 0 |
| **G8** | **Erro injetado** ★ | Toma derivação correta e injeta erro específico: inversão de sinal, fator 2, índice trocado | **Sabemos onde pusemos o erro** | **F1** | US$ 0 |

### 5.3 G8 merece destaque

O gerador de erro injetado resolve dois problemas de uma vez:

1. **Dado de treino** que ensina o modelo a *localizar* o primeiro passo incorreto de uma derivação — a habilidade que F1 exige e que nenhum corpus natural contém, porque ninguém publica derivações erradas anotadas.
2. **Dado de benchmark** — é exatamente a tarefa "verificação de derivação" que o DOC-00 §3.4 identificou como lacuna de todos os benchmarks públicos, e que o DOC-11 vai construir.

O rótulo é perfeito por construção: sabemos qual passo corrompemos e como. Não há ambiguidade de anotação, não há custo de rotulagem humana, e a dificuldade é controlável — erros sutis (fator 2, sinal) versus grosseiros (índice trocado, termo perdido).

**Regra de higiene inegociável:** as partições de treino e de benchmark do G8 são geradas a partir de **derivações-fonte disjuntas**, com separação registrada no manifesto. Gerar as duas do mesmo conjunto seria autocontaminação — construir o próprio vazamento.

### 5.4 A regra que governa todo o motor

> **Nenhum dado sintético entra no corpus sem passar pelo barramento de verificação** (`src/phifm/verify/`).
>
> Dado sintético não verificado é apenas saída de modelo. É assim que ocorre colapso de distribuição, e é a diferença entre um motor de dados e uma máquina de alucinação em escala.

G1–G5, G7 e G8 são verificáveis **mecanicamente** e são a espinha dorsal do motor. G6 não é — e por isso tem tratamento separado na §6.

### 5.5 Geração local vence API por uma ordem de grandeza

Para o G6, gerar 200 M tokens:

| Rota | Vazão | Tempo | **Custo** |
|---|---|---|---|
| API comercial | — | — | **US$ 100–400** |
| Modelo aberto de 8B em H100 alugada (vLLM) | ~4.000 tok/s | ~14 h | **~US$ 39** |
| Modelo aberto de 8B quantizado em RTX 4090 | ~1.200 tok/s | ~46 h | **~US$ 16** |

**Alugar GPU e gerar localmente é 6–25× mais barato que API para geração em massa.** A API só se justifica onde a qualidade do anotador é decisiva e o volume é pequeno — que é exatamente o caso da anotação pedagógica do DOC-04 §4.2 (50 mil documentos, US$ 5–50), e não o caso aqui.

---

## 6. Colapso de modelo: o risco real e os limites que o contêm

Shumailov et al. (2024, *Nature*) mostram que modelos degeneram quando treinados recursivamente sobre suas próprias gerações. É o risco central de qualquer motor de dados sintéticos, e precisa ser endereçado explicitamente, não mencionado de passagem.

**Por que a nossa configuração é estruturalmente diferente:**

| Condição do colapso | Nossa situação |
|---|---|
| Dados sintéticos **substituem** os reais | ❌ Nunca substituem — são **acrescentados**, com teto de 15% (Gerstgrasser et al., 2024, mostram que acumular evita o colapso) |
| Geração **recursiva** — o modelo treina na própria saída | ❌ G1–G5, G7, G8 vêm de **CAS e simuladores**, não de um LLM. Não há laço |
| Gerações **não verificadas** | ❌ Verificação mecânica obrigatória (§5.4) |
| Fração sintética não limitada | ❌ Teto rígido, imposto no construtor de mistura |

**Tetos impostos em código, não por convenção:**

| Limite | Valor |
|---|---|
| Sintético total na mistura | **≤ 15%** |
| Dentro do sintético: gerado por LLM e não verificável mecanicamente (G6) | **≤ 25%** |
| ⇒ G6 como fração do treino total | **≤ 3,75%** |
| Origem recursiva (saída de um modelo ΦFM) | **0% no Tier 1 e no Tier 2** |

O último é o mais importante. Autodestilação e auto-melhoria são técnicas legítimas, mas pertencem ao pós-treino com verificação (DOC-09/DOC-10), onde há recompensa verificável fechando o laço. **No pretraining não entram.**

---

## 7. Orçamento e cronograma

| Atividade | Recurso | Tempo | Custo |
|---|---|---|---|
| Implementar G1–G5, G7, G8 | CPU, SymPy/SciPy | 2–3 semanas de trabalho | **US$ 0** |
| Gerar sintético mecânico (~2–3 B tokens) | CPU, paralelo | ~1 semana em fundo | **US$ 0** |
| Verificar tudo pelo barramento | CPU | incluído | **US$ 0** |
| G6 — reescrita pedagógica (200 M tokens) | GPU alugada | ~46 h numa 4090 | **~US$ 16** |
| DoReMi — otimização de mistura | GPU alugada | 2 × 7 h | **~US$ 5** |
| Ablações de currículo | GPU alugada | ~4 × 7 h | **~US$ 10** |
| Construção e empacotamento dos shards | CPU local | ~1 dia | **US$ 0** |
| **Total** | | | **~US$ 31** |

**A Fase 1 inteira — DOC-02 a DOC-06, do corpus bruto aos shards prontos — custa menos de US$ 60**, somando isto ao bake-off de tokenizer (US$ 15) e à anotação pedagógica (US$ 5–50) do DOC-04.

O produto é o degrau **T0** do DOC-17A §8.2: `PhysCorpus-Open`, o tokenizer e os shards de treino — entrega publicável antes de qualquer modelo existir.

---

## 8. Riscos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Sintético mecânico é repetitivo e artificial demais | **Alta** | Médio | Diversidade de templates medida explicitamente; entropia de n-gramas monitorada; teto de 15% limita o dano |
| G6 introduz erros factuais de Física (não verificável) | Média | **Alto** | Teto de 3,75% do total; auditoria humana em amostra de 200; preferir reescrita **ancorada** ao texto-fonte, não geração livre |
| Pesos do DoReMi não transferem de 50 M para 8 B | Média | Médio | Xie et al. reportam boa transferência, mas em escalas maiores. Validar com uma ablação em 1,5 B antes do CPT do 8 B |
| Upsampling 6× causa memorização literal | Média | **Alto** — vira contaminação | Extração de sufixo medida antes do treino longo; descontaminação mais estrita para fontes com repetição alta |
| Annealing sobreajusta à mistura final | Baixa | Médio | Ablacionar duração do annealing (5% / 10% / 15%) |
| Currículo de dificuldade não entrega ganho | **Alta** | Baixo | **É um resultado válido.** Ablação custa US$ 10 e o negativo é publicável |

---

## 9. Critérios de aceite do Stage-Gate 5

- [ ] **F1** — Mistura otimizada por DoReMi; comparação contra o heurístico **medida**, não presumida
- [ ] **F2** — Nenhuma das 23 subáreas abaixo de 1,5% nem acima de 12% da mistura final
- [ ] **F3** — 100% do dado sintético aprovado pelo barramento de verificação; taxa de rejeição registrada por gerador
- [ ] **F4** — Tetos da §6 impostos **em código**, com teste de CI que falha o build se violados
- [ ] **F5** — Partições de treino e benchmark do G8 comprovadamente disjuntas por manifesto
- [ ] **F6** — Memorização medida por extração de sufixo em fontes com repetição ≥ 4
- [ ] **F7** — Ablação de currículo de dificuldade executada; resultado publicado, positivo ou negativo
- [ ] **F8** — Chunking verificado: zero cortes no meio de ambiente matemático em amostra de 10.000 sequências
- [ ] **F9** — Shards construídos, endereçados por conteúdo, reconstruíveis a partir de um único hash de manifesto (**fecha G1.5**)

---

## 10. Questões em aberto

| ID | Questão | Destino |
|---|---|---|
| OQ-22 | 15% é o teto certo para sintético, ou dá para subir com verificação forte? | Ablação com 10% / 15% / 25% na escala de 50 M |
| OQ-23 | Pesos do DoReMi obtidos em 50 M transferem para 8 B? | Validar em 1,5 B antes do CPT grande |
| OQ-24 | G8 deve variar a sutileza do erro ao longo do treino (currículo)? | Depende do resultado de F7 |
| OQ-25 | Material em alemão/russo/francês entra na mistura? (herdado de OQ-13) | Ablação barata; provável peso ≤ 1% |
| OQ-26 | Quantos tokens sintéticos de cada gerador? A distribuição interna dos 15% não está fixada | Otimizar junto com o DoReMi |

---

## 11. Referências

1. Xie, S. M. et al. (2023). *DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.* NeurIPS.
2. Muennighoff, N. et al. (2023). *Scaling Data-Constrained Language Models.* NeurIPS.
3. Shumailov, I. et al. (2024). *AI models collapse when trained on recursively generated data.* Nature 631.
4. Gerstgrasser, M. et al. (2024). *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data.* COLM.
5. Ding, H. et al. (2024). *Fewer Truncations Improve Language Modeling.* ICML.
6. OLMo Team (2024). *OLMo 2 Furious.* arXiv:2501.00656.
7. Hu, S. et al. (2024). *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies.* COLM.
8. Grattafiori, A. et al. (2024). *The Llama 3 Herd of Models.* arXiv:2407.21783.

---

**Fim do DOC-06.** Encerra a Fase 1. Revisão da §9 necessária antes do DOC-07 (Especificação da Família de Modelos).
