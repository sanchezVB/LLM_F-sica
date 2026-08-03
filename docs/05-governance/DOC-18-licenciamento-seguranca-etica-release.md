# DOC-18 — Licenciamento, Segurança, Ética e Estratégia de Release

**Projeto:** ΦFM — Phi Foundation Models para Física
**Status:** `RASCUNHO v0.1` — aguardando revisão do Stage-Gate 17
**Cobre:** postura legal consolidada, uso dual, model cards, estratégia de publicação
**Depende de:** [ADR-0001](../adr/ADR-0001-decisoes-stage-gate-0.md), [DOC-02](../01-data/DOC-02-aquisicao-corpus.md), [DOC-11](../03-evaluation/DOC-11-physbench.md), [DOC-14](../04-systems/DOC-14-agentes-ferramentas.md)
**Data:** 2026-08-03

---

## 1. Escopo

O ADR-0001 estabeleceu a postura jurídica do **corpus**. Este documento cobre o que sobra: licenciamento dos **artefatos que produzimos**, riscos de **uso dual**, documentação obrigatória e a sequência de publicação.

---

## 2. Licenciamento dos artefatos

| Artefato | Licença | Justificativa |
|---|---|---|
| Código (`src/`, `pipelines/`, `configs/`) | **Apache-2.0** | Permissiva com concessão explícita de patentes; padrão em ML |
| Pesos dos modelos | **Apache-2.0** | Decisão Q3 do Stage-Gate 0 |
| `PhysCorpus-Open` | **CC BY 4.0** | Subconjunto redistribuível (ADR-0001 §6) |
| PhysBench — conjunto público | **CC BY 4.0** | Benchmark precisa ser reutilizável para ter valor |
| PhysBench — conjunto privado | Não distribuído | DOC-11 §8.1 |
| Manifestos do corpus | **CC BY 4.0** | Permite reprodução sem redistribuir bytes |
| Documentos de projeto | **CC BY 4.0** | |

### 2.1 A cadeia de licenças precisa fechar

Três pontos de contaminação já identificados, e a verificação obrigatória:

| Ponto | Risco | Estado |
|---|---|---|
| Conteúdo CC BY-NC no treino | Incompatível com pesos Apache-2.0 | ✅ Excluído (ADR-0001 §4) |
| Pesos do Nougat (CC BY-NC-4.0) como base do ΦOCR | Derivado herdaria a cláusula NC | ✅ Base permissiva exigida (DOC-03 §6.2) |
| Ferramentas GPL acopladas ao código | Contaminaria a Apache-2.0 | ✅ Só processo externo (DOC-14 §8) |

> **Auditoria de licença é item de CI, não revisão manual.** Uma verificação automatizada de dependências e de proveniência de pesos roda antes de qualquer release. Os três pontos acima foram descobertos em momentos diferentes do projeto, e cada um teria sido caro se descoberto depois do treino.

---

## 3. Uso dual: avaliação franca

Física tem aplicações sensíveis. Ignorar isso seria irresponsável; exagerá-lo seria teatro de segurança. A avaliação honesta:

| Área | Risco real | Avaliação |
|---|---|---|
| **Física nuclear** | Conteúdo relacionado a armas | O corpus é literatura **aberta e revisada por pares**. Os obstáculos ao desenvolvimento de armas nucleares são material físsil, engenharia de precisão e capacidade industrial — **não acesso a texto de física nuclear**, que está em qualquer biblioteca universitária |
| **Física de plasmas e alta energia** | Idem | Mesma análise |
| **Materiais e química** | Precursores, síntese | Fora do escopo do corpus; ferramentas de química não integradas |
| **Execução de código** | Código malicioso gerado | **Risco real e mitigado** — sandbox Firecracker, sem rede (DOC-14 §4) |
| **Injeção indireta de prompt** | Agente manipulado por conteúdo | **Risco real e mitigado** — fronteira de confiança (DOC-14 §4) |
| **Desinformação científica** | Gerar física plausível e falsa | **Risco real e central** — é o modo de falha F6/F10, e o programa inteiro é construído contra ele |

**Conclusão:** os riscos genuínos deste sistema **não são de proliferação — são de correção e de segurança de execução.** Um modelo que produz Física errada com confiança e citações inventadas causa dano real: estudantes aprendendo errado, revisão por pares poluída, literatura contaminada.

> É por isso que precisão de citação é **portão** (G2.4), abstenção é **portão** (G2.5) e regressão geral é **desclassificatória** (G2.3). **A postura de segurança deste programa é a exigência de correção**, não uma camada de filtros aparafusada no fim.

### 3.1 O que não fazemos

- Não integramos ferramentas de síntese química ou projeto de dispositivos.
- Não damos ao agente ações com efeito colateral externo (DOC-14 §5).
- Não treinamos em material classificado ou de acesso restrito.
- Não implementamos filtros de tópico em Física — seriam ineficazes (o conteúdo está em livros-texto) e prejudicariam usuários legítimos.

---

## 4. Documentação obrigatória

### 4.1 Model card

Todo modelo publicado acompanha um card contendo, além do padrão da área:

| Seção específica deste programa |
|---|
| **Delta contra o modelo base** por benchmark — a métrica de manchete (G2.1) |
| **Taxas de contaminação** por benchmark |
| **Cobertura por subárea** — em que subáreas o modelo é forte e em quais é fraco |
| **Modos de falha conhecidos** — F1–F10, com o estado de cada um |
| **Limites de calibração** — ECE, comportamento de abstenção |
| **Versão do verificador** usada na avaliação |
| **Degradação por quantização**, por tarefa (DOC-15 §3) |

> A seção de **cobertura por subárea** é incomum e importante. Um modelo forte em `hep-th` e fraco em Física experimental é útil — desde que o usuário saiba. Publicar só o agregado esconde exatamente a informação que decide se o modelo serve ao caso de uso de quem o baixa.

### 4.2 Datasheet do corpus

Segue Gebru et al. (2021), com adições: composição por fonte e subárea, distribuição de licenças, taxas de filtragem por estágio, o que foi **excluído e por quê** (MIT OCW, Feynman Lectures, obras sob copyright), e o relatório de contaminação.

---

## 5. Estratégia de release

### 5.1 Sequência

| Ordem | Artefato | Momento | Racional |
|---|---|---|---|
| 1 | Documentos de projeto | **Já publicados** | Convidam crítica antes de o trabalho estar feito — quando corrigir ainda é barato |
| 2 | `PhysCorpus-Open` + datasheet | T0 | Contribuição independente de qualquer modelo |
| 3 | Tokenizer + estudo de tokenização | T0 | Primeiro resultado científico original |
| 4 | ΦEnc, ΦEmb, ΦRank + PhysBench-Retrieval | G1 | Primeira família de modelos |
| 5 | PhysBench completo (público) | G2 | Precisa preceder as alegações que o usam |
| 6 | ΦGen-1,5B + paper principal | G2 | Contribuição principal |
| 7 | ΦGen-8B | T2c | Extensão |

**O item 5 antes do 6 é deliberado.** Publicar o benchmark antes dos resultados que ele sustenta permite que terceiros o examinem antes de vê-lo usado a nosso favor. A ordem inversa convidaria à suspeita de que o benchmark foi ajustado ao modelo.

### 5.2 O que fica retido, e por quê

| Retido | Motivo |
|---|---|
| `PhysCorpus-Full` | Redistribuição não permitida para boa parte das fontes (ADR-0001 §2, direito D3) |
| PhysBench privado | Integridade da avaliação (DOC-11 §8) |
| `PhysEval-Restricted` | Obras sob copyright, uso exclusivo de avaliação |

Para o corpus completo, publicamos **manifestos e código de reconstrução** — um terceiro com acesso às mesmas fontes reconstrói bit a bit. É o máximo que a postura jurídica permite, e é substancialmente melhor que o comum na área.

### 5.3 Publicações previstas

| Paper | Contribuição | Momento |
|---|---|---|
| *PhysCorpus: um corpus aberto de Física com proveniência completa* | Corpus + metodologia de filtragem calibrada para Física | T0 |
| *Tokenização para Física: pré-tokenização e o custo de fragmentar LaTeX* | Primeiro estudo controlado do tema | T0 |
| *ΦEnc/ΦEmb: encoders de Física com contexto longo* | Superação de PhysBERT e de embedders gerais | G1 |
| *PhysBench: avaliação verificável e renovável de raciocínio físico* | 9 tarefas inéditas + regeneração contra contaminação | G2 |
| *ΦGen: continual pretraining com recompensa verificável em Física* | O sistema completo | G2 |

---

## 6. Ética de atribuição e de uso

**Atribuição.** O corpus é construído sobre o trabalho de centenas de milhares de cientistas. Licenças com exigência de atribuição (CC BY, CC BY-SA) são respeitadas nos manifestos e na documentação. Os autores não deram consentimento individual para treino — **e isso é declarado no datasheet**, não omitido. É a situação de todo modelo de linguagem treinado em literatura científica; a diferença é dizê-lo.

**Uso educacional.** O caso de uso principal é auxiliar estudo e pesquisa. Duas obrigações decorrem:

1. **Nunca apresentar resultado não verificado como certo.** Os guardrails do DOC-15 §6 existem para isso.
2. **Expor o traço de raciocínio** (DOC-14 §6) — um estudante precisa poder verificar, não só receber a resposta. Um modelo de Física que produz respostas opacas é pedagogicamente pior que um livro.

**Autoria científica.** O ΦFM é ferramenta, não coautor. Trabalhos que o utilizem devem declará-lo em métodos, conforme as políticas editoriais vigentes.

---

## 7. Riscos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Contaminação de licença descoberta após o treino | Média | **Alto** | Auditoria em CI (§2.1); schema com `train_ok` |
| Modelo usado para produzir Física plausível e falsa em escala | Média | **Alto** | Portões G2.4/G2.5; model card explícito sobre limites |
| Detentor de direitos contesta o uso de dados | Baixa | **Alto** | ADR-0001; corpus reconstruível a partir de metadados permite refiltrar |
| Escape do sandbox em uso público | Baixa | **Crítico** | Firecracker; auditoria; sem rede |
| Model card lido como marketing | Média | Médio | Cobertura por subárea e falhas conhecidas em destaque, não em rodapé |
| Benchmark publicado depois dos resultados gera suspeita | Baixa | Médio | Sequência do §5.1 |

---

## 8. Critérios de aceite do Stage-Gate 17

- [ ] **R1** — Auditoria automatizada de licença em CI, cobrindo dependências e proveniência de pesos
- [ ] **R2** — Model card completo, com as sete seções específicas do §4.1
- [ ] **R3** — Datasheet do corpus publicado, incluindo o que foi excluído e por quê
- [ ] **R4** — Parecer jurídico do ADR-0001 §7 obtido antes do release de pesos
- [ ] **R5** — Sequência de release respeitada; PhysBench público antes das alegações que o usam
- [ ] **R6** — Ausência de consentimento individual dos autores declarada no datasheet
- [ ] **R7** — Manifestos de reconstrução publicados e verificados por terceiro

---

## 9. Referências

1. Mitchell, M. et al. (2019). *Model Cards for Model Reporting.* FAccT.
2. Gebru, T. et al. (2021). *Datasheets for Datasets.* CACM.
3. Solaiman, I. et al. (2023). *Evaluating the Social Impact of Generative AI Systems.* arXiv:2306.05949.
4. Bommasani, R. et al. (2023). *The Foundation Model Transparency Index.* arXiv:2310.12941.
5. Greshake, K. et al. (2023). *Indirect Prompt Injection.* AISec.

---

**Fim do DOC-18.**
