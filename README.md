# ΦFM — Phi Foundation Models para Física

Programa de pesquisa para projetar e construir uma família de foundation models especializada **exclusivamente em Física** e na matemática aplicada que a sustenta, junto com o corpus, a infraestrutura de verificação, os benchmarks e o stack de serving necessários para tornar as alegações auditáveis.

**Status:** corpus de projeto completo (19 documentos + 1 ADR, cobrindo os 20 pipelines). **Sprint S1 concluído**, S2 com os negativos coletados, e o primeiro modelo — ΦEmb — treinado e medido contra o PhysBERT. O código implementa os documentos, não o contrário.

> **Toda afirmação consequente deste repositório tem estado de verificação explícito** no [painel do DOC-19 §6-B](docs/05-governance/DOC-19-riscos-validade-cientifica.md). Sem isso, ninguém distingue o que foi medido do que foi suposto — nós inclusive. O que está abaixo é medido; o que ainda não foi, está marcado.

---

## Começar em outra máquina

- **[ESTADO.md](ESTADO.md)** — onde estamos, o que fazer a seguir, decisões pendentes
- **[SETUP.md](SETUP.md)** — instalar, trazer os dados, retomar as coletas

## Documentos de projeto

Vinte documentos, escritos em nível de publicação, cada um revisado antes do próximo começar.

**O estado de cada um vive em [`docs/README.md`](docs/README.md)**, com legenda definida — e não aqui. Havia três cópias da mesma tabela de estado (aqui, no índice e no DOC-00 §11), e elas divergiram: as três marcavam 19 de 20 documentos como "em revisão" ao lado da afirmação "corpus de projeto completo". Uma cópia é uma fonte; três são um convite ao desacordo.

O que a legenda de lá distingue, e importa: **🟢 confrontado com execução** (afirmações checadas contra medida) contra **🟡 escrito e nunca posto à prova**. Quatro documentos são 🟢 hoje; dezessete são 🟡. Por afirmação, a granularidade que decide, o rastreio é o [painel do DOC-19 §6-B](docs/05-governance/DOC-19-riscos-validade-cientifica.md).

### Fase 0 — Fundamentos
| Doc | Título |
|---|---|
| [DOC-00](docs/00-foundations/DOC-00-project-charter.md) | Carta do Projeto, Posicionamento Científico e Roteiro |
| [DOC-01](docs/00-foundations/DOC-01-system-architecture.md) | Arquitetura do Sistema e Organização do Repositório |
| [ADR-0001](docs/adr/ADR-0001-decisoes-stage-gate-0.md) | Decisões do Stage-Gate 0 e análise jurídica do corpus |
| [DOC-17A](docs/05-governance/DOC-17A-orcamento-gpu-runpod.md) | Custo-benefício de GPU e armazenamento (RunPod) — *extrato antecipado* |

### Fase 1 — Dados
| Doc | Título |
|---|---|
| [DOC-02](docs/01-data/DOC-02-aquisicao-corpus.md) | Plano Mestre de Aquisição de Corpus |
| [DOC-03](docs/01-data/DOC-03-ingestao-parsing-normalizacao.md) | Ingestão, Parsing e Normalização |
| [DOC-04](docs/01-data/DOC-04-filtragem-dedup-descontaminacao.md) | Filtragem, Deduplicação e Descontaminação |
| [DOC-05](docs/01-data/DOC-05-tokenizer.md) | Projeto do Tokenizer e Vocabulário Físico-Matemático |
| [DOC-06](docs/01-data/DOC-06-mistura-curriculo-dados-sinteticos.md) | Mistura de Dados, Currículo e Dados Sintéticos |

**Fase 1 completa.** Custo total dos cinco documentos, do corpus bruto aos shards prontos: **< US$ 60**.

### Fase 2 — Modelos
| Doc | Título |
|---|---|
| [DOC-07](docs/02-models/DOC-07-familia-de-modelos.md) | Especificação da Família de Modelos |
| [DOC-08](docs/02-models/DOC-08-pretraining-cpt.md) | Pretraining e Continual Pretraining |
| [DOC-09](docs/02-models/DOC-09-pos-treino-sft-dpo-rlvr.md) | Pós-treino: SFT, DPO, RLVR, Destilação |
| [DOC-10](docs/02-models/DOC-10-raciocinio-verificacao-ferramentas.md) | Raciocínio, Verificação e Ferramentas |

**Fase 2 completa.** Quatro troncos treinados, não dez modelos. Barramento de verificação especificado: o ativo central custa **~US$ 50** em computação.

### Fase 3 — Avaliação
| Doc | Título |
|---|---|
| [DOC-11](docs/03-evaluation/DOC-11-physbench.md) | PhysBench — Suíte de Benchmarks |
| [DOC-12](docs/03-evaluation/DOC-12-harness-protocolo-estatistico.md) | Harness de Avaliação e Protocolo Estatístico |

### Fase 4 — Sistemas
| Doc | Título |
|---|---|
| [DOC-13](docs/04-systems/DOC-13-recuperacao-embeddings-rag.md) | Recuperação, Embeddings e RAG |
| [DOC-14](docs/04-systems/DOC-14-agentes-ferramentas.md) | Agentes e Ferramentas Científicas |
| [DOC-15](docs/04-systems/DOC-15-inferencia-serving.md) | Inferência e Serving |
| [DOC-16](docs/04-systems/DOC-16-deployment-mlops-monitoramento.md) | Deployment, MLOps e Monitoramento |

### Fase 5 — Governança
| Doc | Título |
|---|---|
| [DOC-17](docs/05-governance/DOC-17-orcamento-cronograma.md) | Orçamento Consolidado e Cronograma Mestre |
| [DOC-18](docs/05-governance/DOC-18-licenciamento-seguranca-etica-release.md) | Licenciamento, Segurança, Ética e Release |
| [DOC-19](docs/05-governance/DOC-19-riscos-validade-cientifica.md) | Riscos e Protocolo de Validade Científica |

Índice completo com ordem de leitura em [`docs/README.md`](docs/README.md).

---

## Estrutura do repositório

```
docs/          ← os 20 documentos de projeto + ADRs. A fonte de verdade.
configs/       ← árvore Hydra. TODO hiperparâmetro mora aqui, nenhum no código.
src/phifm/     ← código. S1 concluído, S2 parcial, ΦEmb treinado; o resto segue o desenho.
  core/          schema, linhagem, licenças, unidades, LaTeX  ← não importa de nada
  corpus/        aquisição → parsing → filtro → dedup → mistura
  verify/        ★ barramento de verificação — 5 de 6: simbólico, dimensional,
                   numérico, limites, invariantes. Falta `sandbox`, que exige
                   gVisor ou Firecracker (DOC-10 §3.6 descarta `exec()` restrito)
  models/        nn.Module puros, sem consciência de paralelismo
  training/      laços de treino; o sharding é aplicado AQUI, nunca em models/
  eval/ retrieval/ agents/ tools/ serving/ monitoring/
pipelines/     ← assets Dagster. Só orquestração; a lógica vive em src/.
tests/golden/  ← casos de Física congelados que o verificador nunca pode quebrar
infra/         ← docker, slurm, terraform, k8s, monitoramento
benchmarks/    ← dados e definições de tarefa do PhysBench
```

Mapa detalhado e as fronteiras de import impostas em CI: [`src/README.md`](src/README.md).

**Dados nunca entram no git.** O corpus vive no disco local, endereçado por conteúdo, com arquivo frio no Google Drive; só shards tokenizados (20–80 GB) sobem para a GPU alugada. Instalação em outra máquina: [SETUP.md](SETUP.md). Ver [DOC-17A §6.1](docs/05-governance/DOC-17A-orcamento-gpu-runpod.md#61-opção-de-custo-zero-processar-localmente-alugar-só-a-gpu).

---

## As três descobertas que moldam tudo

1. **Física é um domínio pobre em dados.** Toda a literatura de Física legalmente adquirível soma ~30–60 bilhões de tokens após deduplicação. Um modelo 8B compute-ótimo por Chinchilla precisa de 160 bilhões. → O tier generativo é construído por **continual pretraining**, não do zero. Treino do zero é autorizado apenas para encoders, onde o dado é excedente. *(DOC-00 §4)*

2. **A competição não é o SciBERT.** O SciBERT é 82% biomédico e praticamente não viu Física; superá-lo não é resultado. As barras reais são o **PhysBERT** (mesmo domínio) e os **embedders gerais modernos**, contra os quais papers de domínio rotineiramente deixam de comparar. *(DOC-00 §3.1)*

   ✅ **Medido, e mais forte do que o documento supunha.** O MiniLM-L6 — genérico, 23 M de parâmetros — bate o PhysBERT (109 M, específico de Física) nas três métricas. Ser treinado **para** embedding importa mais que ser treinado **em** Física. Ver a tabela abaixo.

3. **Física é mecanicamente verificável, e quase ninguém explora isso.** Análise dimensional, equivalência simbólica, casos-limite e leis de conservação são todos verificadores executáveis. Um único **barramento de verificação** serve à filtragem de dados, às recompensas de RL, à correção de benchmarks e à auto-checagem em inferência — o que torna estruturalmente impossível o treino divergir da avaliação. *(DOC-01 §1 P3, §2)*

---

## A escada

| Degrau | Entrega | **Custo** | Portão | Estado |
|---|---|---|---|---|
| **T0 — Corpus** | `PhysCorpus-Open` + tokenizer de Física — publicável sem nenhum modelo | **US$ 0** | Corpus reconstruível a partir de um único hash de manifesto | ⚠️ espinha pronta; tokenizer ⬜ |
| **T1 — Representação** | ΦEnc / ΦEmb / ΦRank | **US$ 35–120** | Superar o PhysBERT em ≥5 nDCG@10 **e** superar o melhor embedder geral com 1/10 dos parâmetros | ⚠️ **metade** — ver abaixo |
| **T2 — Raciocínio** | ΦGen-1,5B via CPT + SFT + RLVR, ΦRAG | **US$ 300–600** acum. | ≥ +10 pontos sobre o **próprio modelo base**, zero regressão geral, ≥0,95 de precisão de citação |
| **T2c — Escala** | ΦGen-8B | **US$ 1.100–2.260** acum. | Competitivo com abertos de porte médio |
| **T3 — Fronteira** | ΦGen-32B, ΦMM, ΦAgent | 150–600k GPU-h | Exige financiamento externo |

Detalhamento em [DOC-17A §8](docs/05-governance/DOC-17A-orcamento-gpu-runpod.md#8-escada-de-orçamento-mínimo). Cada degrau é uma entrega independente — dá para parar em qualquer um.

---

## O que já foi medido

### Sprint S1 — a espinha de metadados ✅

| | Medido | O plano dizia |
|---|---|---|
| Registros do arXiv (set `physics`) | **1.595.422** · 0 falhas | 1,2 M |
| Tamanho em disco | 674 MB · **422 bytes/registro** | 516–686 bytes |
| Obras do OpenAlex (snapshot) | **4.613.751** · 137 GB lidos · 0 falhas | — |
| Casamento com a espinha | **99,1%** (1.581.098) | 98,5% pela chave `locations` |
| Arestas de citação | **22,7 M+** | "dezenas de milhões" |
| **Fração redistribuível** | **14,8%** | 25–35% ❌ corrigido no ADR-0001 |

O S1 custou **~3 GB de disco**, não os ~150 GB orçados: o snapshot de 725 GB é lido por faixa de bytes HTTP, 13 das 189 colunas, e nunca toca o disco. Ver [`openalex_snapshot.py`](src/phifm/corpus/acquire/openalex_snapshot.py).

### ΦEmb — recuperação por citação ⚠️

256 candidatos, agregação por média, 192 tokens, protocolo idêntico para todos.
Checkpoint no passo 24.000 de 50.000 — **treino em curso**.

| Modelo | Params | recall@1 | recall@10 | MRR |
|---|---|---|---|---|
| **ΦEmb (nosso)** | 110 M | **0,402** | **0,910** | **0,588** |
| MiniLM-L6 (genérico) | 23 M | 0,398 | 0,805 | 0,531 |
| PhysBERT (alvo do G1) | 109 M | 0,285 | 0,645 | 0,403 |
| SciBERT (base do ΦEmb) | 110 M | 0,199 | 0,570 | 0,326 |

**O portão T1 não está passado, e a razão importa.** Ele exige duas coisas:

- *superar o PhysBERT* — ✅ com folga: +0,117 em recall@1, +0,265 em recall@10
- *superar o melhor embedder geral com 1/10 dos parâmetros* — ❌ **não.** A margem sobre o MiniLM em recall@1 é **+0,004**, contra um erro padrão de ±0,031 em 256 itens: empate estatístico. E o ΦEmb é **5× maior**, não 1/10 menor.

Onde o ΦEmb se separa de verdade é `recall@10` (+0,105) e MRR (+0,057) — ele não acerta mais na primeira posição, ele deixa o certo de fora menos vezes. Para busca isso vale, mas é afirmação diferente da que o portão pede.

Sugestão que sai da própria medição: aplicar o mesmo fine-tune de citação **sobre o MiniLM**. Ele parte de 0,398 em vez de 0,199, é 5× menor e treina 5× mais rápido — e atacaria justamente a metade do portão que falta.

### O que ainda não foi medido ⬜

| | Bloqueado por |
|---|---|
| **ΦEnc** (o único modelo do zero) | precisa de 15–30 B tokens de texto completo; temos **0,33 B** de títulos e resumos. Depende do Sprint S3 |
| Tokenizer próprio | DOC-05 inteiro ⬜ — nada executado |
| Preservação de LaTeX (critério C1) | exige a fonte LaTeX, que vem do S3 |
| PhysBench | DOC-11 ⬜ |

O S3 é o gargalo real: **~600 GB** contra 425 GB livres no disco. A saída é a mesma do S1 — processar em fluxo, filtrar para Física com a classificadora do S2, descartar o bruto sem gravar.

---

## Decisões do Stage-Gate 0 (resolvidas em 2026-08-03)

| | Decisão | Resolução |
|---|---|---|
| **Q1** | Perfil de computação | Portátil, com Perfil A (1 nó) como alvo; escala revisitada no Portão G1 |
| **Q2** | Livros sob copyright | Domínio público e licenças abertas para treino; obras sob copyright **apenas em avaliação** (`train_ok=False`) |
| **Q3** | Intenção de release | **Pesos abertos** sob licença permissiva (Apache-2.0/MIT) |
| **Q4** | Modelo base para CPT | **Qwen3-8B-Base**, provisório, a confirmar por bake-off no G1 |

Análise jurídica completa em [ADR-0001](docs/adr/ADR-0001-decisoes-stage-gate-0.md) — leitura obrigatória antes do DOC-02.

---

## Orçamento (do DOC-17A)

**Modelo de custo: processar localmente, alugar apenas a GPU.** O corpus bruto (10–20 TB) nunca sai do disco local; só os shards tokenizados (20–80 GB) sobem para o pod alugado. Armazenamento recorrente cai de US$ 110–350/mês para ~US$ 0, mais um bucket de ~US$ 1–3/mês só para checkpoints.

Quatro conclusões que valem mais que a escolha da placa:

- **A100 deixou de ser boa compra** — a H100 custa 1,5× por hora e entrega 3× mais trabalho; é duas vezes mais barata por unidade de computação.
- **RTX 4090 e H100 empatam em $/FLOP.** A decisão não é preço — é se o modelo cabe em 24 GB.
- **Não é preciso pagar egress do arXiv.** Fatias de arXiv já processadas (`RedPajama-Data-1T`, `proof-pile-2`, `peS2o`) são gratuitas no HuggingFace: ~10–15 B tokens de Física por US$ 0.
- **OCR não é necessário no começo** — o arXiv fornece fonte LaTeX. OCR só entra para teses e livros, depois do Portão G1.

Único custo não recorrente relevante: **HD externo de 8–16 TB (~US$ 150–250)**, ou o disco que você já tiver.
