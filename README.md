# ΦFM — Phi Foundation Models para Física

Programa de pesquisa para projetar e construir uma família de foundation models especializada **exclusivamente em Física** e na matemática aplicada que a sustenta, junto com o corpus, a infraestrutura de verificação, os benchmarks e o stack de serving necessários para tornar as alegações auditáveis.

**Status:** corpus de projeto completo (19 documentos + 1 ADR, cobrindo os 20 pipelines) e **execução iniciada** — Sprint S1 coletando o corpus. O código implementa os documentos, não o contrário.

---

## Começar em outra máquina

- **[ESTADO.md](ESTADO.md)** — onde estamos, o que fazer a seguir, decisões pendentes
- **[SETUP.md](SETUP.md)** — instalar, trazer os dados, retomar as coletas

## Documentos de projeto

Vinte documentos, escritos em nível de publicação, cada um revisado antes do próximo começar.

### Fase 0 — Fundamentos
| Doc | Título | Status |
|---|---|---|
| [DOC-00](docs/00-foundations/DOC-00-project-charter.md) | Carta do Projeto, Posicionamento Científico e Roteiro | 🟡 Em revisão |
| [DOC-01](docs/00-foundations/DOC-01-system-architecture.md) | Arquitetura do Sistema e Organização do Repositório | 🟡 Em revisão |
| [ADR-0001](docs/adr/ADR-0001-decisoes-stage-gate-0.md) | Decisões do Stage-Gate 0 e análise jurídica do corpus | 🟡 Em revisão |
| [DOC-17A](docs/05-governance/DOC-17A-orcamento-gpu-runpod.md) | Custo-benefício de GPU e armazenamento (RunPod) — *extrato antecipado* | 🟢 Entregue |

### Fase 1 — Dados
| Doc | Título | Status |
|---|---|---|
| [DOC-02](docs/01-data/DOC-02-aquisicao-corpus.md) | Plano Mestre de Aquisição de Corpus | 🟡 Em revisão |
| [DOC-03](docs/01-data/DOC-03-ingestao-parsing-normalizacao.md) | Ingestão, Parsing e Normalização | 🟡 Em revisão |
| [DOC-04](docs/01-data/DOC-04-filtragem-dedup-descontaminacao.md) | Filtragem, Deduplicação e Descontaminação | 🟡 Em revisão |
| [DOC-05](docs/01-data/DOC-05-tokenizer.md) | Projeto do Tokenizer e Vocabulário Físico-Matemático | 🟡 Em revisão |
| [DOC-06](docs/01-data/DOC-06-mistura-curriculo-dados-sinteticos.md) | Mistura de Dados, Currículo e Dados Sintéticos | 🟡 Em revisão |

**Fase 1 completa.** Custo total dos cinco documentos, do corpus bruto aos shards prontos: **< US$ 60**.

### Fase 2 — Modelos
| Doc | Título | Status |
|---|---|---|
| [DOC-07](docs/02-models/DOC-07-familia-de-modelos.md) | Especificação da Família de Modelos | 🟡 Em revisão |
| [DOC-08](docs/02-models/DOC-08-pretraining-cpt.md) | Pretraining e Continual Pretraining | 🟡 Em revisão |
| [DOC-09](docs/02-models/DOC-09-pos-treino-sft-dpo-rlvr.md) | Pós-treino: SFT, DPO, RLVR, Destilação | 🟡 Em revisão |
| [DOC-10](docs/02-models/DOC-10-raciocinio-verificacao-ferramentas.md) | Raciocínio, Verificação e Ferramentas | 🟡 Em revisão |

**Fase 2 completa.** Quatro troncos treinados, não dez modelos. Barramento de verificação especificado: o ativo central custa **~US$ 50** em computação.

### Fase 3 — Avaliação
| Doc | Título | Status |
|---|---|---|
| [DOC-11](docs/03-evaluation/DOC-11-physbench.md) | PhysBench — Suíte de Benchmarks | 🟡 Em revisão |
| [DOC-12](docs/03-evaluation/DOC-12-harness-protocolo-estatistico.md) | Harness de Avaliação e Protocolo Estatístico | 🟡 Em revisão |

### Fase 4 — Sistemas
| Doc | Título | Status |
|---|---|---|
| [DOC-13](docs/04-systems/DOC-13-recuperacao-embeddings-rag.md) | Recuperação, Embeddings e RAG | 🟡 Em revisão |
| [DOC-14](docs/04-systems/DOC-14-agentes-ferramentas.md) | Agentes e Ferramentas Científicas | 🟡 Em revisão |
| [DOC-15](docs/04-systems/DOC-15-inferencia-serving.md) | Inferência e Serving | 🟡 Em revisão |
| [DOC-16](docs/04-systems/DOC-16-deployment-mlops-monitoramento.md) | Deployment, MLOps e Monitoramento | 🟡 Em revisão |

### Fase 5 — Governança
| Doc | Título | Status |
|---|---|---|
| [DOC-17](docs/05-governance/DOC-17-orcamento-cronograma.md) | Orçamento Consolidado e Cronograma Mestre | 🟡 Em revisão |
| [DOC-18](docs/05-governance/DOC-18-licenciamento-seguranca-etica-release.md) | Licenciamento, Segurança, Ética e Release | 🟡 Em revisão |
| [DOC-19](docs/05-governance/DOC-19-riscos-validade-cientifica.md) | Riscos e Protocolo de Validade Científica | 🟡 Em revisão |

Índice completo com ordem de leitura em [`docs/README.md`](docs/README.md).

---

## Estrutura do repositório

```
docs/          ← os 20 documentos de projeto + ADRs. A fonte de verdade.
configs/       ← árvore Hydra. TODO hiperparâmetro mora aqui, nenhum no código.
src/phifm/     ← código. Sprints S1–S2 implementados; o resto segue o desenho.
  core/          schema, linhagem, licenças, unidades, LaTeX  ← não importa de nada
  corpus/        aquisição → parsing → filtro → dedup → mistura
  verify/        ★ o barramento de verificação (CAS, dimensional, numérico, limites)
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

3. **Física é mecanicamente verificável, e quase ninguém explora isso.** Análise dimensional, equivalência simbólica, casos-limite e leis de conservação são todos verificadores executáveis. Um único **barramento de verificação** serve à filtragem de dados, às recompensas de RL, à correção de benchmarks e à auto-checagem em inferência — o que torna estruturalmente impossível o treino divergir da avaliação. *(DOC-01 §1 P3, §2)*

---

## A escada

| Degrau | Entrega | **Custo** | Portão |
|---|---|---|---|
| **T0 — Corpus** | `PhysCorpus-Open` + tokenizer de Física — publicável sem nenhum modelo | **US$ 0** | Corpus reconstruível a partir de um único hash de manifesto |
| **T1 — Representação** | ΦEnc / ΦEmb / ΦRank | **US$ 35–120** | Superar o PhysBERT em ≥5 nDCG@10 **e** superar o melhor embedder geral com 1/10 dos parâmetros |
| **T2 — Raciocínio** | ΦGen-1,5B via CPT + SFT + RLVR, ΦRAG | **US$ 300–600** acum. | ≥ +10 pontos sobre o **próprio modelo base**, zero regressão geral, ≥0,95 de precisão de citação |
| **T2c — Escala** | ΦGen-8B | **US$ 1.100–2.260** acum. | Competitivo com abertos de porte médio |
| **T3 — Fronteira** | ΦGen-32B, ΦMM, ΦAgent | 150–600k GPU-h | Exige financiamento externo |

Detalhamento em [DOC-17A §8](docs/05-governance/DOC-17A-orcamento-gpu-runpod.md#8-escada-de-orçamento-mínimo). Cada degrau é uma entrega independente — dá para parar em qualquer um.

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
