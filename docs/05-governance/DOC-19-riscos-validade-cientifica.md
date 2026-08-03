# DOC-19 — Registro de Riscos e Protocolo de Validade Científica

**Projeto:** ΦFM — Phi Foundation Models para Física
**Status:** `RASCUNHO v0.1` — aguardando revisão do Stage-Gate 18
**Cobre:** registro consolidado de riscos, pré-mortem, protocolo de validade; **encerra o corpus de projeto**
**Depende de:** todos os dezoito documentos anteriores
**Data:** 2026-08-03

---

## 1. Objetivo

Dois trabalhos que só podem ser feitos no fim, com o desenho completo à vista:

1. **Consolidar os riscos** dispersos em dezoito documentos e ordená-los por ameaça real ao programa.
2. **Definir o que tornaria as conclusões inválidas** — e declarar antecipadamente as alegações que este programa **não poderá** fazer.

O segundo é o mais valioso e o mais raro. Um programa que não sabe o que o refutaria não é um programa científico.

---

## 2. Os cinco riscos que podem matar o programa

De cerca de setenta riscos registrados nos documentos anteriores, cinco são existenciais.

### R-1 · O cronograma humano não é sustentado
**Probabilidade: alta · Impacto: fatal**

15–21 pessoa-meses. Para uma pessoa em meio período, **três anos**. É o risco mais provável de todo o programa e o menos técnico.

**Mitigação estrutural, já embutida no desenho:** degraus independentes (DOC-00 §5), cada um publicável. T0 entrega corpus e tokenizer; G1 entrega os encoders. **Parar em qualquer degrau produz uma contribuição completa, não um fragmento.** A resposta A do DOC-17 §4 — reduzir escopo a T0 + G1, 7–10 pessoa-meses — é o caminho recomendado para execução solo.

### R-2 · Bug no barramento de verificação
**Probabilidade: média · Impacto: crítico**

O verificador filtra dados, admite sintéticos, calcula recompensa de RL, corrige benchmark e checa saída em serving. Um bug afeta **os cinco simultaneamente**, e o sintoma é ausência de sintoma: tudo parece consistente porque tudo está errado da mesma forma.

**Mitigação:** cobertura ≥ 95%, 1.100 casos golden incluindo a categoria indecidível (DOC-10 §5), verificador reservado fora do laço de treino (DOC-09 §5.4), e discordância entre verificadores tratada como alarme e nunca resolvida por voto.

### R-3 · Esquecimento catastrófico no CPT
**Probabilidade: alta · Impacto: desclassificatório**

O critério G2.3 é o único explicitamente desclassificatório do programa. Um ΦGen melhor em Física e pior em tudo o mais não é uma contribuição — é um modelo estragado.

**Mitigação:** varredura de LR (DOC-08 §5.3, US$ 63), replay de 5% por proxy, avaliação de regressão a cada 5.000 passos com alarme automático.

### R-4 · Credibilidade do benchmark
**Probabilidade: média · Impacto: crítico**

Construímos o benchmark **e** o modelo. Um revisor competente questionará isso na primeira leitura, e estará certo em questionar.

**Mitigação:** publicar conjunto de dev, corretores e harness; reportar sempre ao lado de benchmarks que não controlamos; publicar o PhysBench **antes** dos resultados que o usam (DOC-18 §5.1); validação por físicos independentes com portão de 3%.

> **Um benchmark próprio em que só o nosso modelo vai bem não é evidência de nada.** Está escrito no DOC-11 §11 e é repetido aqui porque é o ponto onde a credibilidade do programa inteiro se decide.

### R-5 · Degradação silenciosa do corpus
**Probabilidade: média · Impacto: alto**

Se o LaTeX for degradado na ingestão (RedPajama ou nosso próprio parser), tudo a jusante herda o dano — e o sintoma só aparece meses depois, como desempenho medíocre sem causa aparente.

**Mitigação:** auditoria S3b antes de qualquer compromisso (DOC-02 §9), métricas de preservação de equação com meta de 0,95 (DOC-03 §10), casos golden do canonicalizador.

---

## 3. Pré-mortem: como este programa falha

Exercício deliberado. É dezembro de 2028, o programa fracassou. O que aconteceu?

| Cenário | Sintoma | Antídoto no desenho |
|---|---|---|
| **Morreu de infraestrutura** | Oito meses construindo MLOps, orquestração e dashboards. Nenhum modelo treinado | DOC-16 §1: nada é construído antes de a ausência doer |
| **Morreu no verificador** | Bug sutil aprovava identidades falsas de operadores. Dados, recompensa e benchmark corrompidos juntos. Descoberto no G2 | DOC-10 §5 e DOC-09 §5.4 |
| **Morreu de escopo** | ΦCode, ΦMath e ΦVis viraram treinos separados. Nenhum ficou bom; o tempo acabou | DOC-07 §1 é vinculante |
| **Morreu de credibilidade** | Publicado. Revisores apontaram benchmark autofavorável, baselines fracos e contaminação não reportada | DOC-12 §3.6 (pré-registro) e §9 (tabela padrão) |
| **Morreu de tédio** | Projeto solo, catorze meses sem resultado publicável. A motivação acabou antes do modelo | ★ **Três publicações antes de existir modelo generativo** (DOC-17 §5) |

> O último cenário é o mais subestimado em projetos solo de longa duração, e é a razão pela qual a escada de degraus não é apenas prudência financeira — **é desenho de sustentabilidade humana.**

---

## 4. Registro consolidado — riscos de alta probabilidade

Os riscos "alta probabilidade" merecem atenção operacional contínua, ainda que não sejam existenciais.

| Risco | Doc | Impacto | Estado do controle |
|---|---|---|---|
| G1.2 — bater embedders gerais com 1/10 dos parâmetros | DOC-07 §16 | Alto | Contexto 16× maior, Matryoshka, pares de citação, interação tardia |
| Filtros heurísticos enviesados contra Física teórica | DOC-04 §10 | Alto | `GOLD-PASS` estratificado; remoção por subárea é métrica bloqueante |
| Lacuna de Física básica não compensada | DOC-02 §10 | Alto | Motor sintético é item de primeira classe, não remendo |
| Reward hacking não detectado | DOC-09 §9 | Alto | Verificador reservado; auditoria de 200 rollouts |
| Itens sintéticos fisicamente absurdos | DOC-11 §11 | Alto | Validação por físicos, portão de 3% |
| Sobre-engenharia de MLOps | DOC-16 §7 | Alto | Regra do DOC-16 §1 |
| SymPy aprova identidade falsa de operadores | DOC-10 §12 | Alto | Anotação de tipo; `INCONCLUSIVE` sem ela |
| Análise dimensional vacuamente aprovada em unidades naturais | DOC-10 §12 | Médio | Dimensão de massa |
| RLVR enviesa para verificável; modelo medíocre em explicar | DOC-09 §9 | Médio | `PB-Concept` como eixo separado |
| Escolha post hoc da métrica favorável | DOC-12 §11 | Crítico | Pré-registro versionado |
| Sobreajuste adaptativo ao conjunto privado | DOC-11 §11 | Alto | Contador publicado |
| Modelo recita constantes em vez de consultar | DOC-14 §8 | Médio | Regra de ancoragem CODATA |
| Quantização degrada raciocínio mais que classificação | DOC-15 §8 | Médio | Ablação por tarefa |

---

## 5. Protocolo de validade científica

Operacionaliza as seis regras do DOC-00 §6.

| # | Regra | Mecanismo | Verificado por |
|---|---|---|---|
| 1 | Descontaminação antes de qualquer alegação | Três vetores (DOC-04 §6.2); relatório publicado | CI |
| 2 | Delta contra a base é a manchete | Formato de tabela obrigatório (DOC-12 §9) | Revisão |
| 3 | Baselines fortes, não convenientes | Modelo geral forte obrigatório na tabela | Revisão |
| 4 | Rigor estatístico | IC bootstrap, ≥3 sementes, testes pareados, FDR | CI |
| 5 | Resultados negativos são entregáveis | §6 | Revisão |
| 6 | Físico no laço | Protocolo do DOC-12 §6, Krippendorff ≥ 0,6 | Portão |

**Acrescentado por este documento:**

| 7 | **Pré-registro** antes de toda avaliação de marco | Arquivo versionado, commit precede a execução | CI |
| 8 | **Versionamento do verificador** — resultados nunca reinterpretados em silêncio | `verifier_id` em todo resultado | CI |

---

## 6. Resultados negativos como entregáveis

O programa contém ablações cujo resultado negativo é informativo e **será publicado**:

| Ablação | Resultado negativo significaria | Doc |
|---|---|---|
| Tokenizer próprio vs. Qwen3 | Tokenização específica de domínio não compensa em Física | DOC-05 §11 |
| Regex de pré-tokenização (variante E) | O achado central do DOC-05 §8 está errado | DOC-05 §11 |
| Mascaramento consciente de equações | Objetivo específico de domínio não ajuda encoders | DOC-08 §4 |
| Currículo de dificuldade | Confirma que a evidência fraca da literatura se sustenta | DOC-06 §4.1 |
| PRM denso vs. recompensa de resultado | Supervisão de processo não compensa, mesmo sendo gratuita aqui | DOC-10 §7 |
| DoReMi vs. mistura heurística | Otimização de mistura não transfere de 50 M | DOC-06 §9 |
| Interação tardia (ColBERT) | Custo de índice não se justifica em Física | DOC-13 §9 |

> Cada um custa entre US$ 5 e US$ 30. **Sete resultados negativos publicáveis por menos de US$ 150** — e um deles (a variante E) testa diretamente a afirmação mais forte de todo o corpus de projeto.
>
> Publicar ablações negativas é raro na área, e é justamente o que dá credibilidade às positivas.

---

## 7. As alegações que este programa NÃO poderá fazer

Declaradas antes de os resultados existirem, para que não sejam negociadas depois.

| Não podemos alegar | Por quê |
|---|---|
| **"O melhor modelo de Física já feito"** | Não temos acesso controlado ao desempenho em Física de modelos fechados de fronteira. Podemos alegar superioridade sobre **modelos abertos de porte comparável, nos benchmarks medidos** |
| **"O maior corpus de Física do mundo"** | Corpora privados de outros laboratórios não são verificáveis. Podemos alegar o **maior corpus de Física abertamente redistribuível**, se for o caso |
| **"Reprodutibilidade bit a bit"** | Não-determinismo de CUDA e ordem de redução do FSDP. Garantimos **reprodutibilidade estatística** (DOC-01 §8) |
| **"Modelo descontaminado"** | Não podemos descontaminar o Qwen3-8B-Base. Podemos alegar que **nossa contribuição incremental** está descontaminada (DOC-04 §6.4) |
| **"O modelo entende Física"** | O PhysBench mede desempenho em dezesseis tarefas. Podemos alegar **desempenho nessas tarefas**, não compreensão |
| **"O ganho vem do tokenizer / do corpus / do RLVR"** | Atribuição causal exige ablação. Sem ela, alegamos apenas o efeito agregado |
| **"Generaliza para toda a Física"** | Amostramos 23 subáreas com cobertura desigual. Alegações são **estratificadas por subárea**, sempre |
| **"Seguro para uso clínico/industrial/regulatório"** | Nunca foi avaliado para isso, e não será |

> Esta tabela é o compromisso mais importante do documento. Toda pressão futura — de revisores, de financiadores, de entusiasmo próprio — empurra na direção de alegar mais do que se mediu. **Escrever os limites antes de ter os resultados é a única defesa confiável contra isso.**

---

## 8. Condições de encerramento

Um programa precisa saber como termina — inclusive bem.

| Cenário | Condição | Ação |
|---|---|---|
| **Sucesso pleno** | G2 aprovado | Publicar tudo; avaliar financiamento para Tier 3 |
| **Sucesso parcial em G1** | G1 aprovado, G2 inviável por tempo | **Encerrar em G1.** Corpus, tokenizer e encoders são contribuição completa e publicável |
| **Sucesso parcial em T0** | Corpus e tokenizer prontos, modelos inviáveis | **Encerrar em T0.** `PhysCorpus-Open` e o estudo de tokenização são publicáveis por si |
| **Falha em G1** | ΦEmb não supera PhysBERT nem embedders gerais | **Parar e diagnosticar.** Custo afundado ~US$ 330, deliberadamente pequeno. Publicar como resultado negativo com análise de causa |
| **Falha em G2.3** | Esquecimento catastrófico irremediável | Recuar para o ΦEnc; publicar o achado sobre limites de CPT em domínio estreito |
| **Invalidação por verificador** | Bug descoberto após alegações | **Retratar publicamente**, reexecutar, republicar com o `verifier_id` correto |

> A última linha é uma obrigação, não uma opção. Se um bug no barramento invalidar resultados publicados, a retratação é imediata e pública. **Um programa construído sobre verificação que não se retrata quando a verificação falha não tem nada.**

---

## 9. Critérios de aceite do Stage-Gate 18

- [ ] **S1** — Registro de riscos revisado a cada portão, não apenas escrito uma vez
- [ ] **S2** — Os cinco riscos existenciais com controle ativo e verificável
- [ ] **S3** — Regras 7 e 8 do §5 implementadas em CI
- [ ] **S4** — Ablações negativas do §6 executadas e publicadas, independentemente do resultado
- [ ] **S5** — Tabela de alegações impossíveis (§7) revisada antes de cada publicação
- [ ] **S6** — Condições de encerramento aceitas pelo sponsor **antes** do início da execução

---

## 10. Encerramento do corpus de projeto

Dezenove documentos e um ADR, cobrindo os vinte pipelines solicitados, do primeiro princípio ao protocolo de retratação.

O que o corpus estabelece:

- **Física é um domínio pobre em dados** — e isso determina toda a estratégia (D-01).
- **Física é mecanicamente verificável** — e isso é a vantagem estrutural que o programa explora em cinco lugares com uma única implementação.
- **O fosso é corpus + verificador + benchmark**, não pesos. Pesos depreciam; os três apreciam.
- **O custo é de tempo, não de dinheiro** — ~US$ 2.000 e 15–21 pessoa-meses.
- **A escada tem degraus independentes** — parar em qualquer um produz contribuição completa.

O que ele deliberadamente **não** estabelece: que o programa vai funcionar. Cada portão é falseável, cada alegação tem limite declarado, e cada ablação pode sair negativa. **É o que o torna um programa de pesquisa em vez de um plano de negócios.**

---

**Fim do DOC-19. Fim do corpus de projeto.**
