# DOC-16 — Deployment, MLOps, Monitoramento e Versionamento

**Projeto:** ΦFM — Phi Foundation Models para Física
**Status:** `RASCUNHO v0.1` — aguardando revisão do Stage-Gate 15
**Cobre:** entregáveis solicitados **13** (deployment), **14** (monitoramento) e **15** (versionamento); **encerra a Fase 4**
**Depende de:** [DOC-01 §5.8, §8](../00-foundations/DOC-01-system-architecture.md), [DOC-12](../03-evaluation/DOC-12-harness-protocolo-estatistico.md), [DOC-15](DOC-15-inferencia-serving.md)
**Data:** 2026-08-03

---

## 1. O princípio de escala: MLOps proporcional ao problema

A tentação em MLOps é construir a plataforma que uma empresa de mil pessoas precisaria. **Seria o erro de alocação mais caro do programa** — meses de engenharia que não produzem um único resultado científico.

> **Regra: nenhum componente de infraestrutura é construído antes que a sua ausência tenha causado um problema real.**

O que **é** construído desde o início são as três coisas cuja ausência é irreparável depois:

| Construir agora | Por quê |
|---|---|
| **Versionamento e linhagem** | Retrofitar proveniência é impossível — o dado de origem já se perdeu |
| **Telemetria de qualidade** | Sem histórico não há como detectar degradação |
| **Reprodutibilidade** | Um resultado irreprodutível é um resultado perdido |

O que **não** é construído no Tier 1–2: malha de serviços, feature store, plataforma de A/B testing, orquestração multi-região, dashboards elaborados. Cada um desses vira necessário quando houver usuários reais, e não antes.

---

## 2. Versionamento — quatro eixos independentes

O pipeline 15 do briefing. O DOC-01 §8 estabeleceu a espinha; aqui ela é operacionalizada.

| Eixo | Mecanismo | Identidade |
|---|---|---|
| **Código** | git | SHA do commit |
| **Configuração** | Hydra → JSON canônico → BLAKE3 | Hash de experimento |
| **Dados** | Snapshot Iceberg + manifestos endereçados por conteúdo | ID de snapshot |
| **Modelo** | HF Hub privado + registro de metadados | Nome de artefato (DOC-01 §8) |

E um quinto, específico deste programa:

| **Verificador** | Versão semântica de `verify/` | `verifier_id` |
|---|---|---|

> **O eixo do verificador é o menos óbvio e o mais importante.** Quando o barramento muda, escores mudam. Um resultado registrado sob `phifm.verify 0.3.1` **nunca é comparado** a outro sob `0.4.0` sem reexecução explícita. Sem isso, uma melhoria no verificador apareceria como melhoria do modelo — o que seria um erro científico grave e completamente invisível.

**Nome de artefato como ponteiro completo:**
```
phifm-gen-1p5b-rlvr-a3f21c04-9b7e0d12-012000
     │    │    │     │        │        └── passo
     │    │    │     │        └── hash de config
     │    │    │     └── snapshot de dados
     │    │    └── estágio
     │    └── tamanho
     └── família
```

---

## 3. Promoção de modelo

Nenhum modelo vai a serving sem passar por portões automatizados. Estados:

```
treinado → avaliado → validado → candidato → publicado → depreciado
```

| Transição | Portão |
|---|---|
| treinado → avaliado | Suíte completa executada; manifesto de reprodução emitido |
| avaliado → validado | **Sem regressão > 2 pontos** em benchmarks gerais (**G2.3**); relatório de contaminação gerado |
| validado → candidato | Avaliação humana concluída (DOC-12 §6); model card escrito |
| candidato → publicado | Revisão de licença (ADR-0001); decisão explícita de release |
| publicado → depreciado | Sucessor publicado; versão anterior permanece acessível |

**Depreciar nunca significa apagar.** Um modelo publicado que sustentou um resultado permanece disponível, ou o resultado deixa de ser verificável. É a mesma disciplina de um artigo que não pode ser retirado silenciosamente.

---

## 4. Monitoramento

### 4.1 As três camadas

| Camada | O que mede | Ferramenta |
|---|---|---|
| **Infraestrutura** | Latência, taxa de erro, GPU, custo/hora | Prometheus + Grafana |
| **Modelo** | Distribuição de tokens, taxa de abstenção, comprimento de resposta, uso de ferramenta | OpenTelemetry |
| **Qualidade** ★ | Taxa de falha de verificação, precisão de citação, drift de consultas | Próprio + Evidently |

### 4.2 Os sinais que realmente importam

A maioria dos painéis de MLOps mede infraestrutura. Os quatro sinais abaixo medem **se o modelo continua fazendo Física**:

| Sinal | Interpretação de uma alta |
|---|---|
| **Taxa de falha de verificação em produção** | ★ O modelo está produzindo mais equações incoerentes. Alarme direto |
| **Precisão de citação** | Queda abaixo de 0,95 viola o portão G2.4 em produção |
| **Taxa de abstenção** | Subida súbita indica degradação de confiança; queda indica perda de calibração |
| **Drift de distribuição de consultas** | Usuários perguntando sobre subáreas que o corpus cobre mal |

O primeiro é possível porque **o verificador roda em serving** (DOC-15 §6). É um sinal de qualidade em tempo real que quase nenhum sistema de LLM tem, e sai de graça da decisão arquitetural de P3.

O último fecha o volante de dados do DOC-01 §3: `P14 → P03`. Consultas mal atendidas viram alvo da próxima rodada de aquisição de corpus. **É o que faz o sistema melhorar depois do deploy em vez de apodrecer.**

### 4.3 Detecção de regressão

Toda publicação de modelo dispara uma comparação automática contra o modelo anterior nas tarefas de portão. Regressão além do IC dispara alarme e bloqueia a promoção.

---

## 5. CI/CD

| Gatilho | O que roda | Onde |
|---|---|---|
| Todo push | `ruff`, `mypy --strict` em `core/` e `verify/`, testes unitários, **`import-linter`** | GitHub Actions |
| Todo push em `verify/` | **Suíte golden completa** + cobertura ≥ 95% | Actions |
| PR | Testes de integração; construção de modelo sob os três perfis de computação (P5) | Actions |
| Noturno | Pipeline de corpus em amostra pequena, ponta a ponta | Runner próprio |
| Manual | Avaliação completa; treino | Runner com GPU |

### 5.1 Estado real da imposição (auditado em 2026-08-09)

Até esta data **não havia workflow nenhum**. Este documento e o DOC-01 §4.3
descreviam um CI que não existia, e a §8 listava "P5 — `import-linter` ativo;
violação de camada quebra o build" como critério de aceite sem que houvesse
build para quebrar.

O que passou a ser imposto, e o que ainda não é:

| Verificação | Exigido por | Estado | Observação |
|---|---|---|---|
| `ruff` | DOC-01 §5.8 | ✅ **ativo** | 135 violações corrigidas na auditoria |
| **`import-linter`** | DOC-01 §4.3 | ✅ **ativo** | 3 contratos, todos passando |
| Testes | DOC-16 §5 | ✅ **ativo** | 258 |
| Suíte golden bloqueante | DOC-10 §5 | ✅ **ativa** | |
| Cobertura de `verify/` | DOC-10 §10 exige **95%** | ⚠️ **piso em 80%** | real: 82% |
| `mypy --strict` em `core/` e `verify/` | DOC-01 §8 | ❌ **não imposto** | 52 erros pendentes |
| Pipeline noturno de corpus | DOC-16 §5 | ❌ não implementado | exige runner com dados |

**Sobre o piso de cobertura.** O DOC-10 §10 exige 95% e o CI configura 80%.
Configurar 95% faria o build falhar de imediato; configurar 80% e continuar
afirmando 95% no documento seria pior. **Os dois números ficam declarados**,
e fechar a diferença é o critério **J2** do Stage-Gate 9 — que continua
válido, agora com o débito visível em vez de presumido cumprido.

**Sobre o `mypy --strict`.** Dos 52 erros, boa parte é ausência de stubs em
`sympy` e `mpmath`, já resolvida por `ignore_missing_imports`. O restante é
anotação faltando em código nosso, e é trabalho real. Fica como dívida
declarada, não como afirmação falsa.

> **A lição desta auditoria vale além do CI.** Um documento que descreve um
> controle inexistente é pior que um documento que não menciona o controle:
> o primeiro cria confiança injustificada. A regra derivada — e vale para
> todo o corpus de projeto — é que **todo controle descrito como ativo
> precisa ter um comando que o demonstre**, ou ser marcado explicitamente
> como pendente.

**O `import-linter` é o que impede o apodrecimento arquitetural.** Sem ele, o DAG de módulos do DOC-01 §4.3 vira sugestão, e em seis meses `verify/` importa de `training/` e o princípio P3 morreu sem ninguém notar.

**A suíte golden bloqueando mudanças em `verify/` é a segunda proteção mais importante.** Um bug ali é global (DOC-10 §1).

---

## 6. Custo operacional

| Componente | Custo mensal |
|---|---|
| Armazenamento frio (B2/R2) | US$ 60–300 |
| Bucket de checkpoints | US$ 1–3 |
| Serving serverless | US$ 5–40 |
| Índices de recuperação (VPS) | US$ 0–80 |
| Rastreio de experimentos (W&B gratuito ou MLflow próprio) | US$ 0–20 |
| CI (Actions gratuito + runner próprio) | US$ 0–10 |
| Monitoramento (Grafana Cloud gratuito) | US$ 0 |
| **Total recorrente** | **US$ 66–453/mês** |

O extremo inferior é o realista para o Tier 1: **~US$ 70/mês**, dominado por armazenamento. Sem usuários reais, quase tudo escala a zero.

---

## 7. Riscos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| **Sobre-engenharia de MLOps consome o cronograma** | **Alta** | **Alto** | Regra do §1: nada antes de a ausência doer |
| Mudança de verificador reinterpreta resultados antigos | Média | **Alto** | Eixo de versionamento próprio (§2) |
| Apodrecimento arquitetural por violação de camadas | **Alta** | Médio | `import-linter` em CI |
| Custo de armazenamento cresce sem controle | Média | Médio | Política de retenção; brutos em camada fria |
| Modelo publicado apagado quebra reprodutibilidade | Baixa | **Alto** | Depreciar ≠ apagar (§3) |
| Painel bonito que ninguém olha | **Alta** | Baixo | Alarmes acionáveis, não dashboards |

> O primeiro risco é o mais provável de todos neste documento. **Construir plataforma é mais confortável que fazer ciência**, e um programa de uma pessoa pode facilmente gastar um trimestre em infraestrutura que nunca será usada na escala que justificaria construí-la.

---

## 8. Critérios de aceite do Stage-Gate 15

- [ ] **P1** — Cinco eixos de versionamento operacionais, incluindo o do verificador
- [ ] **P2** — Portões de promoção automatizados; promoção manual impossível sem passar
- [ ] **P3** — Taxa de falha de verificação instrumentada em serving, com alarme
- [ ] **P4** — Volante de dados fechado: consultas mal atendidas alimentam a fila de aquisição
- [ ] **P5** — `import-linter` ativo; violação de camada quebra o build
- [ ] **P6** — Suíte golden bloqueia mudanças em `verify/`
- [ ] **P7** — Custo recorrente medido e dentro do envelope
- [ ] **P8** — Nenhum componente construído sem problema real que o justifique (§1)

---

## 9. Referências

1. Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS.
2. Mitchell, M. et al. (2019). *Model Cards for Model Reporting.* FAccT.
3. Gebru, T. et al. (2021). *Datasheets for Datasets.* CACM.
4. Apache Iceberg documentation — snapshot isolation e time travel.

---

**Fim do DOC-16.** Encerra a Fase 4.
