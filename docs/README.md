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

**Esta tabela é a única fonte do estado dos documentos.** O `README.md` da raiz e
a tabela de roteiro do DOC-00 §11 apontam para cá em vez de repetir — três cópias
do mesmo estado divergem, e divergiram: até 2026-08-10 as três diziam "🟡 Em
revisão" para 19 de 20 documentos, ao lado da afirmação "corpus de projeto
completo". As duas coisas não podiam ser verdade.

### Legenda, que antes não existia

| | Significa | Como se verifica |
|---|---|---|
| 🟢 | **Confrontado com execução.** Afirmações checadas contra medida, e as erradas corrigidas no próprio documento | o `git log` do arquivo mostra commit de auditoria |
| 🟡 | **Escrito e revisado, nunca confrontado.** Internamente consistente; nada nele foi posto à prova por medição | — |
| ⚪ | Rascunho ou extrato parcial | declarado no cabeçalho do documento |

O que 🟡 **não** significa: que o documento esteja errado. Significa que ninguém
sabe — e o [painel do §6-B do DOC-19](05-governance/DOC-19-riscos-validade-cientifica.md)
é onde isso é rastreado por afirmação, que é a granularidade que importa. Uma
marca única num documento de quarenta páginas não diz quase nada.

### Fase 0 — Fundamentos
| Doc | Título | Cobre | Status |
|---|---|---|---|
| [DOC-00](00-foundations/DOC-00-project-charter.md) | Carta do Projeto, Posicionamento Científico e Roteiro | — | 🟢 Confrontado |
| [DOC-01](00-foundations/DOC-01-system-architecture.md) | Arquitetura do Sistema e Organização do Repositório | Pipelines 1–2 | 🟡 Não confrontado |
| [ADR-0001](adr/ADR-0001-decisoes-stage-gate-0.md) | Decisões do Stage-Gate 0 e postura jurídica | — | 🟢 Confrontado — fração redistribuível caiu de 25–35% para **14,8%** |

### Fase 1 — Dados
| Doc | Título | Cobre | Status |
|---|---|---|---|
| [DOC-02](01-data/DOC-02-aquisicao-corpus.md) | Plano Mestre de Aquisição de Corpus | Pipeline 3 | 🟢 Confrontado — endpoint, volumes e custo do OpenAlex corrigidos |
| [DOC-03](01-data/DOC-03-ingestao-parsing-normalizacao.md) | Ingestão, Parsing e Normalização | Pipeline 4 | 🟡 Não confrontado |
| [DOC-04](01-data/DOC-04-filtragem-dedup-descontaminacao.md) | Filtragem de Qualidade, Deduplicação e Descontaminação | Pipelines 5, 16 (parte) | 🟡 Não confrontado |
| [DOC-05](01-data/DOC-05-tokenizer.md) | Projeto do Tokenizer e Vocabulário Físico-Matemático | Pipeline 6 | 🟡 Não confrontado |
| [DOC-06](01-data/DOC-06-mistura-curriculo-dados-sinteticos.md) | Mistura de Dados, Currículo e Motor de Dados Sintéticos | — | 🟡 Não confrontado |

### Fase 2 — Modelos
| Doc | Título | Cobre | Status |
|---|---|---|---|
| [DOC-07](02-models/DOC-07-familia-de-modelos.md) | Especificação da Família de Modelos | — | 🟡 Não confrontado |
| [DOC-08](02-models/DOC-08-pretraining-cpt.md) | Infraestrutura de Pretraining e Continual Pretraining | Pipelines 7, 9 | 🟡 Não confrontado |
| [DOC-09](02-models/DOC-09-pos-treino-sft-dpo-rlvr.md) | Pós-treino: SFT, DPO, RLVR, Destilação | Pipelines 8, 10 | 🟡 Não confrontado |
| [DOC-10](02-models/DOC-10-raciocinio-verificacao-ferramentas.md) | Raciocínio, Verificação e Treino Integrado a Ferramentas | — | 🟡 Não confrontado |

### Fase 3 — Avaliação
| Doc | Título | Cobre | Status |
|---|---|---|---|
| [DOC-11](03-evaluation/DOC-11-physbench.md) | PhysBench — Projeto da Suíte de Benchmarks | Pipelines 11, 16 | 🟡 Não confrontado |
| [DOC-12](03-evaluation/DOC-12-harness-protocolo-estatistico.md) | Harness de Avaliação e Protocolo Estatístico | Pipeline 11 | 🟡 Não confrontado |

### Fase 4 — Sistemas
| Doc | Título | Cobre | Status |
|---|---|---|---|
| [DOC-13](04-systems/DOC-13-recuperacao-embeddings-rag.md) | Recuperação, Embeddings e Stack de RAG | Pipelines 17, 18 | 🟡 Não confrontado |
| [DOC-14](04-systems/DOC-14-agentes-ferramentas.md) | Framework de Agentes e Ferramentas Científicas | Pipelines 19, 20 | 🟡 Não confrontado |
| [DOC-15](04-systems/DOC-15-inferencia-serving.md) | Inferência e Serving | Pipeline 12 | 🟡 Não confrontado |
| [DOC-16](04-systems/DOC-16-deployment-mlops-monitoramento.md) | Deployment, MLOps, Monitoramento e Versionamento | Pipelines 13, 14, 15 | 🟡 Não confrontado |

### Fase 5 — Governança
| Doc | Título | Cobre | Status |
|---|---|---|---|
| [DOC-17A](05-governance/DOC-17A-orcamento-gpu-runpod.md) | Custo-benefício de GPU e Armazenamento (RunPod) | — | 🟢 Entregue |
| [DOC-17](05-governance/DOC-17-orcamento-cronograma.md) | Orçamento, Modelo de Custos e Cronograma Mestre | — | 🟡 Não confrontado |
| [DOC-18](05-governance/DOC-18-licenciamento-seguranca-etica-release.md) | Licenciamento, Segurança, Ética e Estratégia de Release | — | 🟡 Não confrontado |
| [DOC-19](05-governance/DOC-19-riscos-validade-cientifica.md) | Registro de Riscos e Protocolo de Validade Científica | — | 🟢 Confrontado — carrega o painel de verificação §6-B |

## Convenções

- **Idioma:** português. Permanecem em inglês termos consagrados (*tokenizer, pretraining, embedding, checkpoint, benchmark, pipeline, scaling law*) e **todo** identificador de código, campo de schema e chave de configuração.
- **Diagramas:** Mermaid embutido no Markdown, para renderizar direto no GitHub.
- **Referências:** numeradas ao final de cada documento, formato de publicação.
- **Decisões:** mudanças de rumo viram um ADR em [`adr/`](adr/), nunca uma edição silenciosa de um documento já revisado.
- **Estimativas:** toda estimativa numérica declara sua confiança e como verificá-la. Números não verificados nunca são apresentados como medidos.
