# Corpus de documentos de projeto — ΦFM

Vinte documentos em nível de publicação, cada um revisado em um *stage-gate* antes do próximo começar. Esta é a fonte de verdade do projeto: **o código implementa os documentos, não o contrário.**

## Ordem de leitura

Para quem chega agora, leia nesta ordem:

1. **[DOC-00](00-foundations/DOC-00-project-charter.md)** — por que o projeto existe, contra quem compete, e por que a Física é um domínio pobre em dados
2. **[ADR-0001](adr/ADR-0001-decisoes-stage-gate-0.md)** — as quatro decisões fundadoras e a análise jurídica do corpus
3. **[DOC-01](00-foundations/DOC-01-system-architecture.md)** — como o sistema é montado
4. **[DOC-17A](05-governance/DOC-17A-orcamento-gpu-runpod.md)** — quanto custa e em que hardware
5. **[DOC-02](01-data/DOC-02-aquisicao-corpus.md)** — de onde vêm os dados

## Estado atual

### Fase 0 — Fundamentos
| Doc | Título | Cobre | Status |
|---|---|---|---|
| [DOC-00](00-foundations/DOC-00-project-charter.md) | Carta do Projeto, Posicionamento Científico e Roteiro | — | 🟡 Em revisão |
| [DOC-01](00-foundations/DOC-01-system-architecture.md) | Arquitetura do Sistema e Organização do Repositório | Pipelines 1–2 | 🟡 Em revisão |
| [ADR-0001](adr/ADR-0001-decisoes-stage-gate-0.md) | Decisões do Stage-Gate 0 e postura jurídica | — | 🟡 Em revisão |

### Fase 1 — Dados
| Doc | Título | Cobre | Status |
|---|---|---|---|
| [DOC-02](01-data/DOC-02-aquisicao-corpus.md) | Plano Mestre de Aquisição de Corpus | Pipeline 3 | 🟡 Em revisão |
| [DOC-03](01-data/DOC-03-ingestao-parsing-normalizacao.md) | Ingestão, Parsing e Normalização | Pipeline 4 | 🟡 Em revisão |
| [DOC-04](01-data/DOC-04-filtragem-dedup-descontaminacao.md) | Filtragem de Qualidade, Deduplicação e Descontaminação | Pipelines 5, 16 (parte) | 🟡 Em revisão |
| [DOC-05](01-data/DOC-05-tokenizer.md) | Projeto do Tokenizer e Vocabulário Físico-Matemático | Pipeline 6 | 🟡 Em revisão |
| [DOC-06](01-data/DOC-06-mistura-curriculo-dados-sinteticos.md) | Mistura de Dados, Currículo e Motor de Dados Sintéticos | — | 🟡 Em revisão |

### Fase 2 — Modelos
| Doc | Título | Cobre | Status |
|---|---|---|---|
| [DOC-07](02-models/DOC-07-familia-de-modelos.md) | Especificação da Família de Modelos | — | 🟡 Em revisão |
| [DOC-08](02-models/DOC-08-pretraining-cpt.md) | Infraestrutura de Pretraining e Continual Pretraining | Pipelines 7, 9 | 🟡 Em revisão |
| [DOC-09](02-models/DOC-09-pos-treino-sft-dpo-rlvr.md) | Pós-treino: SFT, DPO, RLVR, Destilação | Pipelines 8, 10 | 🟡 Em revisão |
| [DOC-10](02-models/DOC-10-raciocinio-verificacao-ferramentas.md) | Raciocínio, Verificação e Treino Integrado a Ferramentas | — | 🟡 Em revisão |

### Fase 3 — Avaliação
| Doc | Título | Cobre | Status |
|---|---|---|---|
| DOC-11 | PhysBench — Projeto da Suíte de Benchmarks | Pipelines 11, 16 | ⚪ |
| DOC-12 | Harness de Avaliação e Protocolo Estatístico | Pipeline 11 | ⚪ |

### Fase 4 — Sistemas
| Doc | Título | Cobre | Status |
|---|---|---|---|
| DOC-13 | Recuperação, Embeddings e Stack de RAG | Pipelines 17, 18 | ⚪ |
| DOC-14 | Framework de Agentes e Ferramentas Científicas | Pipelines 19, 20 | ⚪ |
| DOC-15 | Inferência e Serving | Pipeline 12 | ⚪ |
| DOC-16 | Deployment, MLOps, Monitoramento e Versionamento | Pipelines 13, 14, 15 | ⚪ |

### Fase 5 — Governança
| Doc | Título | Cobre | Status |
|---|---|---|---|
| [DOC-17A](05-governance/DOC-17A-orcamento-gpu-runpod.md) | Custo-benefício de GPU e Armazenamento (RunPod) | — | 🟢 Entregue |
| DOC-17 | Orçamento, Modelo de Custos e Cronograma Mestre | — | ⚪ |
| DOC-18 | Licenciamento, Segurança, Ética e Estratégia de Release | — | ⚪ |
| DOC-19 | Registro de Riscos e Protocolo de Validade Científica | — | ⚪ |

**Legenda:** 🟢 entregue · 🟡 em revisão · 🔵 em redação · ⚪ na fila

## Convenções

- **Idioma:** português. Permanecem em inglês termos consagrados (*tokenizer, pretraining, embedding, checkpoint, benchmark, pipeline, scaling law*) e **todo** identificador de código, campo de schema e chave de configuração.
- **Diagramas:** Mermaid embutido no Markdown, para renderizar direto no GitHub.
- **Referências:** numeradas ao final de cada documento, formato de publicação.
- **Decisões:** mudanças de rumo viram um ADR em [`adr/`](adr/), nunca uma edição silenciosa de um documento já revisado.
- **Estimativas:** toda estimativa numérica declara sua confiança e como verificá-la. Números não verificados nunca são apresentados como medidos.
