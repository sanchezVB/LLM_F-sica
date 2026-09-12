# ΦFM — Phi Foundation Models para Física

Programa de pesquisa para projetar e construir uma família de foundation models especializada **exclusivamente em Física** e na matemática aplicada que a sustenta, junto com o corpus, a infraestrutura de verificação, os benchmarks e o stack de serving necessários para tornar as alegações auditáveis.

**Status (2026-09-12):** corpus de projeto completo (19 documentos + 1 ADR, cobrindo os 20 pipelines). **Sprints S1, S2 e S3 concluídos** — 38,96 B tokens no disco, atestados por um hash de manifesto. **ΦEmb treinado, medido e no sistema**; ΦRank treinado, medido e **removido** por não pagar o próprio custo; **ΦEnc ainda não existe**. O bake-off de tokenizer está rodando. O código implementa os documentos, não o contrário.

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

**Fase 2 completa** — os documentos. A arquitetura decide treinar **quatro troncos em
vez de dez modelos**; nenhum dos quatro foi treinado ainda. Barramento de verificação
especificado e implementado 5 de 6: o ativo central custa **~US$ 50** em computação.

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
src/phifm/     ← código. S1–S3 concluídos, ΦEmb no sistema, ΦRank removido por medição;
                 ΦEnc escrito e nunca treinado. 813 testes.
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

   ✅ **Confirmado pela coleta, na ponta baixa da faixa.** O corpus somou **38,96 B
   tokens** de fontes gratuitas, contra os 30–60 B que o documento estimava. E o
   número que importa é menor: só **21,74 B** têm LaTeX íntegro — o peS2o, que é
   14,60 B, tem ambiente de equação em **0,0%** dos documentos. Volume não é o
   gargalo; integridade matemática é.

2. **A competição não é o SciBERT.** O SciBERT é 82% biomédico e praticamente não viu Física; superá-lo não é resultado. As barras reais são o **PhysBERT** (mesmo domínio) e os **embedders gerais modernos**, contra os quais papers de domínio rotineiramente deixam de comparar. *(DOC-00 §3.1)*

   ✅ **Medido, e mais forte do que o documento supunha.** O MiniLM-L6 — genérico, 23 M de parâmetros — bate o PhysBERT (109 M, específico de Física) nas três métricas. Ser treinado **para** embedding importa mais que ser treinado **em** Física. Ver a tabela abaixo.

3. **Física é mecanicamente verificável, e quase ninguém explora isso.** Análise dimensional, equivalência simbólica, casos-limite e leis de conservação são todos verificadores executáveis. Um único **barramento de verificação** serve à filtragem de dados, às recompensas de RL, à correção de benchmarks e à auto-checagem em inferência — o que torna estruturalmente impossível o treino divergir da avaliação. *(DOC-01 §1 P3, §2)*

---

## A escada

| Degrau | Entrega | **Custo** | Portão | Estado |
|---|---|---|---|---|
| **T0 — Corpus** | `PhysCorpus-Open` + tokenizer de Física — publicável sem nenhum modelo | **US$ 0** | Corpus reconstruível a partir de um único hash de manifesto | 🟡 **corpus 38,96 B atestado por um hash; 5 tokenizers treinados, bake-off rodando** — ver abaixo |
| **T1 — Representação** | ΦEnc / ΦEmb / ΦRank | **US$ 35–120** | Superar o PhysBERT em ≥5 nDCG@10 **e** superar o melhor embedder geral com 1/10 dos parâmetros | 🟡 **ΦEmb passou os dois; ΦRank saiu do sistema; ΦEnc não existe** |
| **T2 — Raciocínio** | ΦGen-1,5B via CPT + SFT + RLVR, ΦRAG | **US$ 300–600** acum. | ≥ +10 pontos sobre o **próprio modelo base**, zero regressão geral, ≥0,95 de precisão de citação |
| **T2c — Escala** | ΦGen-8B | **US$ 1.100–2.260** acum. | Competitivo com abertos de porte médio |
| **T3 — Fronteira** | ΦGen-32B, ΦMM, ΦAgent | 150–600k GPU-h | Exige financiamento externo |

Detalhamento em [DOC-17A §8](docs/05-governance/DOC-17A-orcamento-gpu-runpod.md#8-escada-de-orçamento-mínimo). Cada degrau é uma entrega independente — dá para parar em qualquer um.

---

## O que já foi medido

### O corpus — 38,96 B tokens, atestados por um hash ✅

```
hash raiz  3113f0fed57c44ddb6ffbc955d329fb9c4afd88785162e71d5341e84b4674d84
35 etapas · 1295 arquivos · 52,40 GB
```

| fonte | tokens | ambiente de equação |
|---|---|---|
| RedPajama-arXiv Física | 10,54 B | **84,9%** |
| RedPajama-arXiv math+cs | 11,20 B | mesmo mecanismo |
| OpenWebMath | 2,62 B | — |
| peS2o | 14,60 B | **0,0%** — equações removidas na extração |
| **total** | **38,96 B** | **21,74 B com LaTeX íntegro** |

⚠️ **O portão do T0 pede o corpus reconstruível a partir do hash, e duas das seis
etapas NÃO se reconstroem** — descoberto tentando, em 2026-09-12. A tabela mestra
perdeu a entrada (`data/raw/openalex_works` não existe mais) e reexecutar gravaria
nulo por cima de 14 M de referências, com código de saída 0; os pares de citação
mudaram de embaralhamento e o `pares_validacao` reconstruído tem **2,1% das âncoras**
do atual — é outro benchmark. As duas ficam marcadas `parametros_reconstruidos=true`
de propósito, com a razão registrada. `redpajama_math_cs` é a primeira etapa com os
parâmetros **capturados na execução**.

### Sprint S1 — a tabela mestra de metadados ✅

| | Medido | O plano dizia |
|---|---|---|
| Registros do arXiv (set `physics`) | **1.595.422** · 0 falhas | 1,2 M |
| Tamanho em disco | 674 MB · **422 bytes/registro** | 516–686 bytes |
| Obras do OpenAlex (snapshot) | **4.613.751** · 137 GB lidos · 0 falhas | — |
| Casamento com a tabela mestra | **99,1%** (1.581.098) | 98,5% pela chave `locations` |
| Arestas de citação | **22,7 M+** | "dezenas de milhões" |
| **Fração redistribuível** | **14,8%** | 25–35% ❌ corrigido no ADR-0001 |

O S1 custou **~3 GB de disco**, não os ~150 GB orçados: o snapshot de 725 GB é lido por faixa de bytes HTTP, 13 das 189 colunas, e nunca toca o disco. Ver [`openalex_snapshot.py`](src/phifm/corpus/acquire/openalex_snapshot.py).

### ΦEmb — recuperação por citação ✅

⚠️ **A tabela que estava aqui vinha de um protocolo quebrado, e os números foram
retirados.** Ela reportava recall@1 0,402 num pool de 256 candidatos montado com
`val.head(2000)` — que tem **62% de alvos duplicados**. Colunas byte-idênticas têm
cosseno idêntico, o `argsort` desempata de forma arbitrária, e o teto de um modelo
**perfeito** naquele pool era **0,7562 de nDCG@10, não 1,0**. O empate estatístico
com o MiniLM que este README anunciava era, no protocolo corrigido, uma derrota.

Protocolo de hoje: **2.000 candidatos**, pool sorteado e **desduplicado** (alvo e
consulta únicos, teto **1,0000** verificado), média mascarada, 192 tokens, idêntico
para todos, todos medidos na mesma sessão.

| Modelo | Params | nDCG@10 | recall@1 | recall@10 |
|---|---|---|---|---|
| **ΦEmb do sistema (nosso, 6 M pares)** | **23 M** | **0,622** | **0,431** | **0,831** |
| GTE-base@400k (T1f, base trocada) | 109 M | 0,609 | 0,430 | 0,801 |
| GTE-large (melhor genérico) | 335 M | 0,579 | 0,414 | 0,764 |
| MiniLM-L6 (genérico, sem ajuste) | 23 M | 0,476 | 0,313 | 0,664 |
| PhysBERT (alvo do G1.1) | 109 M | 0,351 | 0,222 | 0,491 |
| SciBERT | 110 M | 0,254 | 0,149 | 0,382 |

**O portão T1 passou nas duas metades:**

- *superar o PhysBERT em ≥5 nDCG@10* — ✅ **+0,271**
- *superar o melhor embedder geral com ≤1/10 dos parâmetros* — ✅ **+0,044** sobre o
  GTE-large, com **1/14,8** do tamanho. ⚠️ Em recall@1 o pareado dá p=0,065, então
  essa métrica específica ainda não está estabelecida.

A sugestão que este README fazia — *"aplicar o mesmo fine-tune sobre o MiniLM"* —
**foi executada e é o campeão atual**. O ΦEmb de 110 M baseado em SciBERT ficou para
trás: o de 23 M baseado em MiniLM faz 0,622 contra 0,475.

### Três resultados NEGATIVOS, que custaram tanto quanto os positivos

| | Medido |
|---|---|
| **BM25 na composição** | 🔴 fora. A fusão RRF parou de somar (empate, p=0,95) e cobrava 0,026 de teto |
| **ΦRank (reordenador)** | 🔴 fora do sistema. Dobrar a profundidade move 1,95% das consultas (p=1,0); o recall@10 sobe **+1 consulta** de 195 oportunidades. A cadeia é `ΦEmb → top-10` |
| **Truncagem a 192 tokens** | 🔴 não custa nada. 63,8% das âncoras são truncadas e 28,2% dos tokens descartados, e ler 256 ou 384 **não muda o recall** (p=0,08–0,84) e custa 2,9× |

### Tokenizer — 5 variantes treinadas, bake-off rodando 🟡

O DOC-05 §11.1 rodou nas seis variantes (custo US$ 0, em CPU): a variante **A** dá
fertilidade 0,962 contra 1,426 do Qwen3 — **0,674×**, dentro da meta de ≤0,80×.
Round-trip 100% e `1.5` consistente em todas.

⚠️ Mas a §11.1 é **proxy**, e o §11.2 — treinar um encoder por variante — é o que
decide. O par A×E (com e sem o regex de pré-tokenização da §8) está rodando no
Kaggle. E o proxy **erra por 3×**: no corpus de treino real E gasta 13,6% mais
tokens por documento, não os 37,7% que a razão de fertilidade prometia.

### O que ainda não foi medido ⬜

| | Bloqueado por |
|---|---|
| **ΦEnc** (o único modelo do zero) | ✅ o dado deixou de bloquear — **38,96 B tokens** no disco, dos quais **21,74 B de LaTeX íntegro**. Falta a GPU: ~80 h de T4 contra 30 h/semana de cota. E depende do bake-off decidir o tokenizer |
| PhysBench | DOC-11 ⬜ — `eval/benchmarks/` e `eval/harness/` estão vazios |
| G1.3 (classificação/NER) e G1.4 (ΦOCR) | não tocados |
| Composição do corpus | ⚠️ `math`+`cs` dobraram o LaTeX íntegro, mas põem 45% de não-Física. O T1c mediu que domínio importa (p=0,0062); o trade-off **não foi medido** e as fatias ficam separadas para isso |

**O S3 deixou de ser gargalo.** O que era "~600 GB contra 425 GB livres" resolveu-se
processando em fluxo: o bruto nunca aterra. Os 93,8 GB da coleta mais recente
produziram 12 GB de parquet sem nunca ter 1 GB de bruto em disco.

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
