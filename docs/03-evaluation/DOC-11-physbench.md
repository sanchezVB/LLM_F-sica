# DOC-11 — PhysBench: Projeto da Suíte de Benchmarks

**Projeto:** ΦFM — Phi Foundation Models para Física
**Status:** `RASCUNHO v0.1` — aguardando revisão do Stage-Gate 10
**Cobre:** entregáveis solicitados **11** e **16**; abre a **Fase 3**
**Depende de:** [DOC-00 §2, §3.4, §6](../00-foundations/DOC-00-project-charter.md), [DOC-06 §5](../01-data/DOC-06-mistura-curriculo-dados-sinteticos.md), [DOC-10](../02-models/DOC-10-raciocinio-verificacao-ferramentas.md)
**Data:** 2026-08-03

---

## 1. Princípios

**Um benchmark é uma alegação sobre o que importa.** Escolher o que medir é escolher o que otimizar. Cinco princípios governam a suíte:

| # | Princípio | Consequência |
|---|---|---|
| **B1** | **Cada tarefa mapeia a um modo de falha nomeado** (F1–F10 do DOC-00 §2) | Nenhuma tarefa entra por ser interessante; entra por medir uma falha que sabemos existir |
| **B2** | **Não construir o que já existe** | MMLU-física, GPQA, OlympiadBench e SciBench são reaproveitados. Construímos apenas as lacunas |
| **B3** | **Correção mecânica onde possível; honestidade onde não é** | As tarefas se dividem em verificáveis e não verificáveis, e **não fingimos que as segundas são as primeiras** |
| **B4** | **Conjunto público de desenvolvimento + conjunto privado de teste** | O público será contaminado — é inevitável e declarado. O privado é o que sustenta alegações |
| **B5** | **Geração mecânica ⇒ benchmark renovável** | ★ Ver §7 — a propriedade que torna esta suíte estruturalmente diferente |

---

## 2. A suíte

Treze tarefas em quatro trilhas. ★ marca tarefas que **nenhum benchmark público cobre**.

| Tarefa | O que mede | Ataca | Correção | Origem dos itens |
|---|---|---|---|---|
| **Trilha A — Raciocínio verificável** | | | | |
| `PB-Solve` | Problemas com resposta fechada | — | CAS | Livros abertos + G1 |
| `PB-Derive` | Produzir derivação completa e correta | F1 | Passo a passo | G2 |
| ★ `PB-Verify` | **Localizar o primeiro passo incorreto** | **F1** | **Exata** | **G8** |
| ★ `PB-Dim` | Julgar coerência dimensional | **F2** | Mecânica | G3 |
| ★ `PB-Limit` | Verificar redução em caso-limite | **F3** | CAS + numérico | G4 |
| ★ `PB-Convention` | Mesma Física, convenção oposta | **F7** | CAS | DOC-03 §4 |
| `PB-Fermi` | Estimativa de ordem de grandeza | F4 | Faixa + rubrica |  Manual + G5 |
| `PB-Tool` | Uso correto de ferramenta | F4 | **Execução real** | G1, G5 |
| **Trilha B — Conceitual** | | | | |
| `PB-Concept` | Compreensão qualitativa | — | ⚠️ Juiz + rubrica | Manual + StackExchange |
| ★ `PB-Abstain` | Recusar quando é correto recusar | **F10** | Rótulo | Construído |
| **Trilha C — Literatura** | | | | |
| `PB-Retrieve` | Recuperar a literatura certa | — | **Grafo de citações** | OpenAlex |
| ★ `PB-Cite` | Responder com citações **verificáveis** | **F6** | DOI resolvível | Construído |
| `PB-Entity` | NER de partículas, detectores, materiais, métodos | F5 | Anotação | Manual + fraca |
| ★ `PB-Formula` | Recuperar documentos por equação | F5 | **Forma canônica** | DOC-03 §3 |
| **Trilha D — Multimodal e experimental** | | | | |
| ★ `PB-Plot` | Ler inclinação, expoente, barras de erro | **F8** | Exata (sintético) | matplotlib + figuras reais |
| ★ `PB-Experiment` | Propagação de incerteza, sistemáticos | **F9** | Numérica | Construído |

**Nove das dezesseis tarefas são inéditas.** As demais existem em forma parcial em benchmarks públicos, e nós as incluímos porque a **correção por equivalência de CAS** (§4) muda substancialmente os números medidos.

---

## 3. Trilha A — o núcleo verificável

### 3.1 `PB-Verify` — a tarefa carro-chefe

> Dada uma derivação de `n` passos contendo **exatamente um** erro, identificar **qual passo** está errado e **por quê**.

É a tarefa que o DOC-00 §3.4 apontou como a lacuna mais evidente de todos os benchmarks públicos, e a que o gerador **G8** produz com rótulo perfeito — sabemos qual passo corrompemos e como.

| Dimensão de controle | Valores |
|---|---|
| Tipo de erro | Sinal invertido · fator numérico · índice trocado · termo perdido · unidade errada · limite mal tomado · aplicação de teorema fora de hipótese |
| Sutileza | Grosseiro (índice trocado) → sutil (fator 2 numa integral gaussiana) |
| Comprimento | 5 a 40 passos |
| Subárea | Todas as 23 |

**Métrica:** acurácia de localização exata do passo; secundariamente, acurácia da classificação do tipo de erro.

**Por que importa mais do que parece:** localizar o primeiro passo incorreto é a habilidade que separa "gerar texto que parece uma derivação" de "fazer Física". E é justamente a capacidade que F1 mede como ausente.

### 3.2 `PB-Convention` — robustez a convenção

Cada item aparece em **pares**: a mesma questão física sob convenções opostas.

| Eixo | Variantes |
|---|---|
| Assinatura métrica | `(+,−,−,−)` vs `(−,+,+,+)` |
| Sistema de unidades | SI vs Gaussiano |
| Unidades naturais | `ℏ = c = 1` vs explícito |
| Sinal de Fourier | `e^{ikx}` vs `e^{−ikx}` |

**Métrica principal: consistência dentro do par.** Um modelo que acerta os dois é robusto; que acerta um e erra o outro está aplicando uma convenção memorizada; que erra os dois não sabe a física. **A consistência é mais informativa que a acurácia** — e nenhum benchmark público mede isso.

### 3.3 `PB-Abstain` — recusar corretamente

Quatro categorias de item onde a resposta certa é não responder:

| Categoria | Exemplo |
|---|---|
| Dados insuficientes | Faltam parâmetros para determinar o resultado |
| Fora do regime de validade | Pedir resultado newtoniano em regime relativístico |
| Premissa falsa | A pergunta assume um fenômeno que não ocorre |
| Referência não verificável | Pedir citação para uma alegação sem fonte |

**Métrica pareada:** taxa de abstenção correta **e** taxa de abstenção indevida em itens perfeitamente respondíveis. Reportar só a primeira permitiria a um modelo que se recusa a tudo parecer excelente. É o portão do critério **G2.5**.

---

## 4. A correção por equivalência simbólica é uma contribuição metodológica

A maior parte dos benchmarks de Física e matemática corrige por **casamento de string** ou por extração com regex de uma resposta final. Isso produz um erro sistemático mensurável:

| Resposta do modelo | Gabarito | String | **CAS** |
|---|---|---|---|
| `2\pi\sqrt{L/g}` | `2\pi(L/g)^{1/2}` | ❌ Errado | ✅ **Correto** |
| `\frac{mv^2}{2}` | `0.5 m v^2` | ❌ Errado | ✅ **Correto** |
| `\hbar\omega/2` | `\frac{1}{2}\hbar\omega` | ❌ Errado | ✅ **Correto** |

> **Hipótese a ser medida e publicada:** a correção por string **subestima sistematicamente** a acurácia em Física, e o viés **não é uniforme entre modelos** — modelos que imitam a formatação do gabarito são favorecidos sobre modelos que raciocinam melhor mas escrevem diferente.
>
> Rodar os benchmarks públicos existentes com os dois corretores e publicar a diferença é, por si só, um resultado. Custa quase nada e recalibra a leitura da literatura.

Toda a suíte usa `verify/symbolic` com fallback numérico (DOC-10 §3.1, §3.3), incluindo o tratamento de `INCONCLUSIVE`: um item que o verificador não consegue decidir é **excluído do denominador e reportado separadamente**, jamais contado como erro do modelo.

---

## 5. Trilha B — o problema honesto da Física conceitual

`PB-Concept` mede o que **não é mecanicamente verificável**: por que o céu é azul, o que acontece com a entropia neste processo, por que este argumento de simetria funciona.

**É o calcanhar de Aquiles da suíte, e a honestidade sobre isso importa mais que a solução.** O DOC-09 §9 registrou o risco de um modelo otimizado por RLVR ficar excelente em resolver e medíocre em explicar. Sem `PB-Concept` esse risco seria invisível — mas a tarefa depende de julgamento, e julgamento é ruidoso.

**Protocolo, com as limitações declaradas:**

| Elemento | Escolha |
|---|---|
| Rubrica | Explícita, com âncoras por nota, elaborada por física, não pelo juiz |
| Juízes | **Três LLMs distintos**, não um |
| Agregação | Mediana; itens com discordância alta são sinalizados |
| **Calibração humana** | ★ **200 itens julgados por físicos**; concordância juiz–humano **medida e publicada** |
| Limite declarado | Se a concordância for baixa, o resultado é reportado como **fraco**, não como número |

> **Nenhum resultado de `PB-Concept` é publicado sem a estatística de concordância humana ao lado.** Um número de LLM-juiz sem calibração é opinião apresentada como medida.

---

## 6. Trilha C — literatura, com gabarito de graça

### 6.1 `PB-Retrieve` e a supervisão que já temos

O gabarito de recuperação normalmente exige anotação cara. Nós temos de graça:

> **As referências reais de um paper são o gabarito de recuperação.** Dado o texto de um paper (com as citações removidas), o conjunto correto de documentos a recuperar é a sua própria bibliografia — já resolvida em DOIs pelo grafo do OpenAlex (DOC-02 §3.1).

Milhões de consultas com gabarito, custo zero. É a mesma intuição do SPECTER, aplicada à avaliação.

Complementado por consultas em linguagem natural escritas manualmente (~500), porque bibliografia é um proxy imperfeito de intenção de busca real.

### 6.2 `PB-Cite` — a cláusula Galactica

O modelo responde uma pergunta **com citações**. Métricas:

| Métrica | Definição | Limiar de portão |
|---|---|---|
| **Precisão de citação** | Citações cujo DOI **resolve** e cujo conteúdo **sustenta** a afirmação | **≥ 0,95** (critério **G2.4**) |
| Taxa de DOI alucinado | DOIs que não existem | **Deve ser 0** |
| Cobertura | Afirmações verificáveis que receberam citação | Reportada |

O DOI é resolvido contra o Crossref em tempo de avaliação. **Um DOI inventado é falha automática do item, sem parcialidade.** É a medida direta do defeito que retirou o Galactica do ar em três dias.

### 6.3 `PB-Formula`

Dada uma equação, recuperar os documentos do corpus que a contêm — sob variação notacional. O gabarito vem da **forma canônica** do DOC-03 §3: dois documentos que escrevem a mesma equação de formas diferentes têm a mesma `canonical_latex`.

Mede diretamente a capacidade que recuperação densa costuma perder: casamento simbólico exato.

---

## 7. A propriedade que torna esta suíte diferente: renovabilidade

Todo benchmark público morre por contaminação. GPQA, MMLU e MATH ficam progressivamente menos informativos à medida que entram nos corpora de treino do mundo, e não há conserto — os itens são fixos.

> **Seis das nossas tarefas — `PB-Verify`, `PB-Dim`, `PB-Limit`, `PB-Convention`, `PB-Plot` (sintético) e parte de `PB-Solve` — são geradas mecanicamente.** Podemos **regenerar o conjunto privado de teste** a partir de derivações-fonte novas, quantas vezes quisermos, a custo praticamente zero.
>
> Um modelo não pode ser treinado em itens que ainda não existem. **A degradação por contaminação, que é terminal para benchmarks estáticos, aqui é apenas um recálculo.**

Política operacional: o conjunto privado é regenerado a cada **seis meses**, ou imediatamente sempre que houver suspeita de vazamento. Cada geração recebe versão (`PhysBench-Verify-v2026.2`), e resultados são sempre reportados com a versão.

Esta propriedade decorre diretamente de termos construído o motor de dados sintéticos verificado (DOC-06 §5) antes do benchmark. Não é acidente — é retorno da mesma decisão arquitetural.

---

## 8. Higiene: separação, contaminação e dimensionamento

### 8.1 Separação de partições

| Regra | Imposição |
|---|---|
| Itens gerados mecanicamente vêm de **derivações-fonte disjuntas** das usadas em treino | Manifesto; teste de CI (**F5** do DOC-06 §9) |
| Conjunto **público de dev** (~200 itens/tarefa) é liberado | Declarado como "será contaminado; use só para desenvolvimento" |
| Conjunto **privado de teste** (~500–1.000 itens/tarefa) **nunca é publicado** | Só resultados agregados |
| Descontaminação pelos três vetores do DOC-04 §6.2 | Antes de qualquer alegação |

### 8.2 Dimensionamento estatístico

Para uma proporção próxima de 0,5:

| `n` | Erro-padrão | IC 95% |
|---|---|---|
| 200 | 0,035 | ±6,9 pontos |
| 500 | 0,022 | ±4,4 pontos |
| **1.000** | **0,016** | **±3,1 pontos** |

**Decisão: 500 itens por tarefa no mínimo; 1.000 nas tarefas que sustentam os portões G1 e G2.** Com 200 itens seria impossível distinguir uma diferença de 5 pontos de ruído — e o protocolo do DOC-00 §6.4 exige que diferenças dentro de ICs sobrepostos sejam reportadas como não significativas. Comparações pareadas (McNemar) recuperam poder e são o teste padrão da suíte.

### 8.3 O sobreajuste que ninguém declara

> Mesmo um conjunto **privado** é sobreajustado se você avaliar contra ele mil vezes durante o desenvolvimento. É o sobreajuste adaptativo, e ele é invisível porque nenhum item vazou.

**Contramedida:** o número de avaliações contra o conjunto privado é **contado e registrado**, e o total é publicado junto com os resultados. O desenvolvimento usa o conjunto público; o privado é consultado em marcos, não continuamente.

---

## 9. Orçamento

| Item | Custo |
|---|---|
| Tarefas geradas mecanicamente (6 tarefas) | **US$ 0** |
| Gabarito de recuperação pelo grafo de citações | **US$ 0** |
| `PB-Plot` sintético | **US$ 0** |
| Rubricas e itens manuais (`PB-Concept`, `PB-Fermi`, `PB-Experiment`) | ~3 semanas de trabalho |
| Anotação de `PB-Entity` | ~2 semanas, ou anotação fraca + verificação |
| Calibração de juiz LLM (200 itens × 3 juízes) | **~US$ 20** |
| **Validação por físicos** (amostra estratificada) | Tempo de especialista — ver §10 |
| Execução do harness | **~US$ 30** |
| **Total em computação** | **~US$ 50** |

O custo real do PhysBench não é dinheiro — é **tempo de especialista humano**.

---

## 10. Validação por físicos

O critério 6 do DOC-00 §6 exige físico no laço. Sem isso, a suíte mede o que nós *achamos* que é Física.

| Elemento | Especificação |
|---|---|
| Amostra | 300 itens, estratificados por tarefa e subárea |
| Revisores | ≥ 3 físicos, com pelo menos um por grande área (teórica, experimental, computacional) |
| O que julgam | O item está **correto**? É **bem posto**? O gabarito está **certo**? A dificuldade é a declarada? |
| Métrica | Taxa de erro nos itens; concordância entre revisores |
| **Portão** | **Taxa de erro > 3% reprova a tarefa**, que volta para revisão |

> Itens gerados mecanicamente **não são automaticamente corretos como problemas de Física**. O G8 garante que sabemos onde o erro foi injetado; não garante que a derivação-base seja fisicamente sensata, nem que o enunciado faça sentido. **Verificação mecânica e validade física são coisas diferentes**, e confundi-las seria o erro mais provável desta suíte.

---

## 11. Riscos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Itens sintéticos fisicamente absurdos, ainda que mecanicamente corretos | **Alta** | **Alto** | Validação por físicos (§10) com portão de 3% |
| `PB-Concept` com concordância juiz–humano baixa | **Alta** | Médio | Publicar como resultado fraco; nunca como número isolado (§5) |
| Sobreajuste adaptativo ao conjunto privado | Média | **Alto** | Contagem de avaliações publicada (§8.3) |
| Diversidade sintética insuficiente — modelo decora o gerador | Média | **Alto** | Entropia de templates medida; regeneração semestral (§7) |
| Viés de subárea (mais itens onde é fácil gerar) | **Alta** | Médio | Cota mínima por subárea; reportar sempre estratificado |
| Nosso próprio modelo favorecido pelo desenho do benchmark | Média | **Crítico para credibilidade** | Baselines fortes; validação independente; publicar itens e corretores do conjunto público |

> O último risco é o mais sério para a credibilidade. Construímos o benchmark **e** o modelo. A mitigação estrutural é publicar o conjunto de dev, os corretores e o harness completos, de modo que terceiros possam reproduzir e contestar — e reportar sempre ao lado de benchmarks públicos que não controlamos (GPQA, OlympiadBench). **Um benchmark próprio em que só o nosso modelo vai bem não é evidência de nada.**

---

## 12. Critérios de aceite do Stage-Gate 10

- [ ] **K1** — 16 tarefas especificadas, cada uma mapeada a um modo de falha nomeado (B1)
- [ ] **K2** — ≥ 500 itens por tarefa no privado; ≥ 1.000 nas tarefas de portão
- [ ] **K3** — Validação por físicos concluída; taxa de erro ≤ 3% em toda tarefa
- [ ] **K4** — Concordância juiz–humano medida e publicada para `PB-Concept`
- [ ] **K5** — Separação treino/benchmark provada por manifesto para todas as tarefas mecânicas
- [ ] **K6** — Descontaminação pelos três vetores executada; taxas publicadas
- [ ] **K7** — Conjunto público liberado com corretores e harness; privado retido
- [ ] **K8** — Regeneração demonstrada: gerar `v2` do conjunto privado e mostrar equivalência estatística
- [ ] **K9** — Comparação string vs. CAS publicada nos benchmarks públicos existentes (§4)
- [ ] **K10** — Contador de avaliações contra o privado implementado e visível

---

## 13. Questões em aberto

| ID | Questão | Destino |
|---|---|---|
| OQ-43 | Qual a concordância juiz–humano real em Física conceitual? | Medir em K4 — pode inviabilizar `PB-Concept` como métrica quantitativa |
| OQ-44 | Itens do G8 são distinguíveis de derivações humanas? Se sim, o modelo pode aprender o artefato | Estudo cego com físicos |
| OQ-45 | `PB-Plot` sintético transfere para figuras reais de papers? | Medir a diferença; se for grande, sintético sozinho é insuficiente |
| OQ-46 | Vale submeter o PhysBench a um consórcio externo para curadoria independente? | Depois do G1 — aumentaria muito a credibilidade |

---

## 14. Referências

1. Rein, D. et al. (2023). *GPQA: A Graduate-Level Google-Proof Q&A Benchmark.* arXiv:2311.12022.
2. He, C. et al. (2024). *OlympiadBench.* ACL.
3. Wang, X. et al. (2024). *SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models.* ICML.
4. Hendrycks, D. et al. (2021). *Measuring Massive Multitask Language Understanding* (MMLU). ICLR.
5. Recht, B. et al. (2019). *Do ImageNet Classifiers Generalize to ImageNet?* ICML.
6. Dwork, C. et al. (2015). *The reusable holdout: Preserving validity in adaptive data analysis.* Science 349.
7. Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS.
8. Taylor, R. et al. (2022). *Galactica.* arXiv:2211.09085.
9. Cohan, A. et al. (2020). *SPECTER.* ACL.

---

**Fim do DOC-11.** Revisão da §12 necessária antes do DOC-12 (Harness de Avaliação e Protocolo Estatístico).
