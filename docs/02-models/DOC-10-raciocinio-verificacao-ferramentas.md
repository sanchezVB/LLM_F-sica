# DOC-10 — Raciocínio, Verificação e Treino Integrado a Ferramentas

**Projeto:** ΦFM — Phi Foundation Models para Física
**Status:** `RASCUNHO v0.1` — aguardando revisão do Stage-Gate 9
**Cobre:** especificação do **barramento de verificação**, raciocínio de cadeia longa, modelo de recompensa de processo (PRM), raciocínio integrado a ferramentas; resolve **OQ-5**; **encerra a Fase 2**
**Depende de:** [DOC-01 §1 (P3), §2](../00-foundations/DOC-01-system-architecture.md), [DOC-06 §5](../01-data/DOC-06-mistura-curriculo-dados-sinteticos.md), [DOC-09 §5](DOC-09-pos-treino-sft-dpo-rlvr.md)
**Data:** 2026-08-03

---

## 1. O ativo central do programa

O DOC-01 §9 registrou que o fosso defensável do projeto é a tríade **corpus + barramento de verificação + benchmark** — e não os pesos, que depreciam a cada seis meses. Este documento especifica a peça do meio.

O barramento é usado em **quatro lugares**, com uma única implementação:

| Consumidor | Uso | Documento |
|---|---|---|
| `corpus.filter` | Verificar equações extraídas; sinal de qualidade | DOC-03 §5 |
| `corpus.mixture` | Portão de admissão do dado sintético | DOC-06 §5.4 |
| `training.rl` | **Função de recompensa** do RLVR | DOC-09 §5 |
| `eval.grading` | Correção de benchmark por equivalência | DOC-11, DOC-12 |
| `serving` | Auto-checagem em inferência | §6.3 |

> **A consequência que justifica o desenho.** Como é o mesmo código, é **estruturalmente impossível** que a recompensa de treino divirja do corretor de avaliação. A maioria dos projetos de LLM tem três implementações separadas de "esta resposta está certa" que divergem silenciosamente — e o sintoma é o clássico "o modelo melhorou no treino e não melhorou no benchmark". Aqui esse modo de falha não existe.
>
> O preço: **um bug em `verify/` é um bug global.** Daí o requisito de cobertura de testes mais estrito do repositório (§10).

---

## 2. O protocolo unificado

```python
# src/phifm/verify/bus.py  (especificação)

class Verdict(Enum):
    PASS         = "pass"          # verificado como correto
    FAIL         = "fail"          # verificado como incorreto
    INCONCLUSIVE = "inconclusive"  # ★ não foi possível decidir — ver §2.1
    ERROR        = "error"         # o próprio verificador falhou

class VerificationResult(BaseModel):
    verdict: Verdict
    confidence: float          # [0,1] — calibrada por verificador
    evidence: str              # traço legível: o que foi checado e como
    verifier_id: str           # identidade + versão, para lineage
    cost_ms: int
    counterexample: str | None # quando FAIL, o contraexemplo concreto

class Verifier(Protocol):
    id: str
    def applicable(self, claim: Claim) -> bool: ...
    def verify(self, claim: Claim, ctx: Context) -> VerificationResult: ...
```

### 2.1 `INCONCLUSIVE` é um veredito de primeira classe

A maioria dos códigos de verificação confunde **"eu chequei e está errado"** com **"eu não consegui checar"**. Em filtragem de dados a confusão é tolerável. **Em RL ela é destrutiva.**

Se `INCONCLUSIVE` for tratado como `FAIL`, o modelo recebe recompensa negativa por resolver problemas difíceis — aqueles cuja verificação simbólica não fecha. O gradiente resultante ensina exatamente a coisa errada: **evitar problemas difíceis**. Depois de algumas centenas de passos de GRPO, o modelo prefere responder o que é fácil de verificar.

**Regra:** `INCONCLUSIVE` mapeia para recompensa **neutra** (zero), nunca negativa. E a taxa de `INCONCLUSIVE` por tipo de problema é monitorada — se subir, o verificador é que precisa melhorar, não o modelo.

### 2.2 Há um limite matemático, e ele precisa ser declarado

> **Teorema de Richardson (1968).** Para expressões envolvendo racionais, π, `log`, `exp`, `sin` e valor absoluto, decidir se uma expressão é identicamente zero é **indecidível**.

Não é uma limitação do SymPy. É um resultado matemático: **não existe verificador simbólico completo.** Qualquer sistema que alegue decidir equivalência sempre está errado em algum caso, ou não termina.

É por isso que `INCONCLUSIVE` existe, que o timeout é obrigatório, e que o verificador numérico (§3.3) é o complemento indispensável do simbólico — não um plano B. Um projeto que não reconhecesse isso construiria um verificador que trava ou mente.

---

## 3. Os seis verificadores

### 3.1 `verify/symbolic` — equivalência algébrica

| | |
|---|---|
| Motor | SymPy, com timeout rígido (padrão 5 s) |
| Estratégia em cascata | (1) comparação estrutural após canonicalização (DOC-03 §3) → (2) `simplify(a - b) == 0` → (3) `a.equals(b)` → (4) delega ao numérico |
| Retorna `INCONCLUSIVE` quando | Timeout, `simplify` não conclui, símbolos não resolvidos |

**Armadilha específica de Física: o SymPy não sabe o que é operador.** Ele assume comutatividade por padrão. `A*B - B*A` simplifica para `0`, o que é **falso** para operadores quânticos e para matrizes.

Mitigação: anotação de tipo sempre que o contexto permitir (`Symbol('A', commutative=False)`), derivada dos marcadores do DOC-03 (`\hat{}`, `\mathbf{}`, contexto de bra-ket). Sem anotação confiável em expressão com operadores → **`INCONCLUSIVE`, nunca `PASS`**.

### 3.2 `verify/dimensional` — coerência de unidades

Álgebra sobre as sete dimensões-base do SI `[M L T I Θ N J]`, com suporte a sistemas alternativos.

| Sistema | Tratamento |
|---|---|
| SI | Direto |
| Gaussiano | Conversão de carga e campos; fator `4π` |
| **Naturais (`ℏ = c = 1`)** | ★ Ver abaixo |
| HEP (GeV como mestre) | Tudo em potências de energia |

> **O caso degenerado que precisa de tratamento próprio.** Em unidades naturais, a análise dimensional comum **colapsa**: comprimento, tempo e massa viram todos potências de energia, e o teste de coerência dimensional passa vacuamente em quase tudo. É o exato caminho do *reward hacking* por omissão de unidades identificado no DOC-09 §5.4.
>
> **Solução:** em unidades naturais, verificar **dimensão de massa** em vez de dimensão física — cada termo de uma equação precisa ter a mesma potência de energia. O sistema de unidades vem do campo `physics.conventions` do DOC-03 §4, e se ele for `null`, o verificador retorna `INCONCLUSIVE` em vez de escolher um sistema por conta própria.

**O problema mais difícil é resolução de símbolos.** Um `E` pode ser energia, campo elétrico ou módulo de Young — no mesmo documento. Estratégia em cascata: (1) unidades declaradas no contexto → (2) inferência pela estrutura da equação → (3) `INCONCLUSIVE`. A expectativa calibrada do DOC-03 §5 permanece: **20–35% das equações verificáveis**, não 100%.

### 3.3 `verify/numeric` — substituição aleatória

O complemento indispensável do simbólico, conforme §2.2.

| Parâmetro | Valor | Justificativa |
|---|---|---|
| Substituições independentes | **20** | Uma coincidência numérica em 20 pontos aleatórios é improvável; em 1, não |
| Precisão | `mpmath`, 50 dígitos | Ruído de ponto flutuante em 64 bits geraria falsos negativos |
| Domínio de amostragem | Reais e complexos, evitando singularidades e cortes de ramo conhecidos | |
| Tolerância | Relativa, `1e-30`, com **acordo exigido em todas as substituições** | Tolerância frouxa é vetor de reward hacking (DOC-09 §5.4) |

Cortes de ramo (`sqrt`, `log`, potências fracionárias) são a principal fonte de falso negativo: duas expressões podem ser equivalentes num ramo e não em outro. Mitigação: amostrar preferencialmente no ramo principal e sinalizar discordância isolada como `INCONCLUSIVE`, não `FAIL`.

### 3.4 `verify/limits` — redução em casos-limite

Ataca **F3**. Biblioteca de reduções canônicas, cada uma com o limite e o resultado esperado:

| Limite | Reduz de | Para |
|---|---|---|
| `v/c → 0` | Cinemática relativística | Newtoniana |
| `ℏ → 0` | Mecânica quântica | Clássica (Ehrenfest, WKB) |
| Campo fraco, `v ≪ c` | Relatividade Geral | Gravitação newtoniana |
| `T → ∞` | Fermi-Dirac / Bose-Einstein | Maxwell-Boltzmann |
| `T → 0` | Estatística quântica | Estado fundamental |
| `r → ∞` | Solução de Schwarzschild | Espaço-tempo plano |
| `N → ∞` | Física estatística | Limite termodinâmico |
| Acoplamento `→ 0` | Teoria interagente | Teoria livre |

Implementação: `sympy.limit()` com verificação numérica cruzada — se o limite simbólico não fechar, avaliar numericamente numa sequência convergente.

### 3.5 `verify/conservation` — invariantes

Energia, momento linear e angular, carga, número bariônico e leptônico, unitariedade (norma da função de onda). Para uma solução alegada, verificar que o invariante se conserva — frequentemente exigindo integração numérica da solução proposta.

### 3.6 `verify/sandbox` — execução isolada

Executa código gerado pelo modelo. **Código gerado por modelo é entrada não confiável** — a mesma postura que se aplicaria a input de usuário anônimo.

| Opção | Isolamento | Veredito |
|---|---|---|
| `exec()` com builtins restritos | ❌ Trivialmente evadível — técnicas de escape são amplamente documentadas | ✗ |
| Docker apenas | ⚠️ Kernel compartilhado; escape de container é classe real de vulnerabilidade | ✗ Insuficiente |
| **gVisor** | Kernel em espaço de usuário; superfície de syscall reduzida | ✅ |
| **Firecracker microVM** | Virtualização completa; isolamento mais forte | ✅ **Preferido** |

**Política de execução, sem exceção:** sem rede, sistema de arquivos somente leitura exceto um `tmpfs` descartável, limite de CPU e memória, timeout rígido, sem acesso a segredos ou variáveis de ambiente do host, processo derrubado ao fim.

---

## 4. Álgebra de resultados

Como combinar seis vereditos em um. **Não é um `AND`** — um `INCONCLUSIVE` envenenaria a conjunção e reintroduziria o problema do §2.1.

```
qualquer FAIL de verificador de alta confiança   →  FAIL
todos PASS                                        →  PASS (confiança = mín. das confianças)
mistura de PASS e INCONCLUSIVE                    →  PASS com confiança reduzida
todos INCONCLUSIVE                                →  INCONCLUSIVE
qualquer ERROR                                    →  ERROR (registrado, investigado, nunca silencioso)
```

**Discordância entre verificadores é um evento de primeira classe.** Se o simbólico diz `PASS` e o numérico diz `FAIL`, há um bug em um dos dois — e isso é registrado, alertado e investigado, jamais resolvido por voto majoritário. Voto majoritário aqui esconderia exatamente o tipo de defeito que o §1 alerta ser global.

---

## 5. Testes golden — a disciplina que sustenta tudo

`tests/golden/` contém casos de Física congelados. **Qualquer mudança de comportamento do barramento sobre eles quebra o CI**, mesmo que a mudança pareça uma melhoria.

Três categorias, e a terceira é a que costuma faltar:

| Categoria | Conteúdo | Verifica |
|---|---|---|
| **Sabidamente corretos** | ~500 pares de expressões equivalentes, das 23 subáreas | Verificador não gera falso negativo |
| **Sabidamente incorretos** | ~500 pares com erro conhecido: sinal, fator 2, índice trocado, unidade errada | Verificador não gera falso positivo |
| **Sabidamente indecidíveis** ★ | ~100 casos onde sabemos que o verificador **não pode** decidir (incluindo instâncias do §2.2) | ★ Verificador diz `INCONCLUSIVE` em vez de **chutar** |

A terceira categoria é a mais importante e a mais esquecida. Um verificador que chuta em caso indecidível é pior que um que se abstém — porque em RL o chute vira gradiente, e um gradiente errado com confiança alta é o pior insumo possível.

---

## 6. Raciocínio de cadeia longa

### 6.1 Alocação adaptativa de computação

O RLVR alonga naturalmente a cadeia de raciocínio (observado no DeepSeek-R1). O problema resultante é de alocação:

| Patologia | Efeito |
|---|---|
| **Excesso de raciocínio** em problema fácil | Desperdício de tokens; às vezes o modelo "raciocina" até sair de uma resposta correta |
| **Falta de raciocínio** em problema difícil | Erro por precipitação |

Meta: alocação **proporcional à dificuldade**. Implementação: incluir no SFT exemplos com cadeias curtas para problemas fáceis e longas para difíceis, e monitorar a correlação entre comprimento de cadeia e dificuldade real do item. Se a correlação for fraca, a alocação está mal aprendida.

### 6.2 Autoverificação dentro da cadeia

Treinar o modelo a executar, dentro do próprio raciocínio, o que um físico faz ao terminar uma conta:

```
[derivação]
→ "Verificando dimensões: [L T⁻²] dos dois lados. ✓"
→ "Limite v/c → 0: reduz à expressão newtoniana. ✓"
→ "Caso especial m₂ → ∞: recupera o problema de um corpo. ✓"
[resposta]
```

Os dados vêm de graça: os geradores G3 e G4 do DOC-06 produzem exatamente esses traços, com o resultado da verificação já conhecido.

### 6.3 O modelo chamando o verificador em inferência

A aresta pontilhada `verify → serving` do DOC-01 §2. Não é só treinar *com* o verificador — é o modelo **invocá-lo**:

```
modelo emite  →  <verify type="dimensional">F = m a²</verify>
harness executa o verificador
resultado volta ao contexto  →  FAIL: [M L T⁻²] ≠ [M L T⁻⁴]
modelo revisa  →  "Corrigindo: F = m a"
```

É raciocínio integrado a ferramentas aplicado à própria verificação, e transforma o barramento de artefato de treino em capacidade de tempo de execução. **O modelo não precisa acertar de primeira; precisa saber conferir.**

---

## 7. PRM — modelo de recompensa de processo

| | |
|---|---|
| Dados | Rótulos de passo **gratuitos** dos geradores G2 e G8 (DOC-09 §5.2) |
| Arquitetura | Cabeça de classificação sobre o ΦEnc, ou cabeça sobre o ΦGen |
| Saída | Probabilidade de que o passo `k` seja correto **e avance** para a solução |
| Uso | Recompensa densa no GRPO, **auxiliar** à recompensa de resultado |
| Custo de treino | **~US$ 20** |

> **Restrição de peso.** O PRM é um modelo aprendido e, portanto, hackeável — ao contrário dos verificadores mecânicos. Gao et al. (2023) documentam superotimização de modelos de recompensa. **A recompensa de resultado (R1, peso 1,00) permanece dominante; o PRM entra como auxiliar com peso ≤ 0,3.** Um PRM com peso alto seria trocar um sinal incorruptível por um corruptível.

A ablação de PRM denso vs. recompensa de resultado apenas responde **OQ-35**, e custa ~US$ 30.

---

## 8. Raciocínio integrado a ferramentas (TIR)

O ΦMath do DOC-07 §9 é isto: ΦGen que **formula, chama, interpreta e verifica**, em vez de calcular de cabeça.

| Ferramenta | Uso | Documento |
|---|---|---|
| SymPy | Álgebra simbólica, integração, EDO | DOC-14 |
| NumPy / SciPy | Numérico, otimização, EDP | DOC-14 |
| mpmath | Alta precisão | DOC-14 |
| Biblioteca de unidades | Conversão e propagação de incerteza | `core/units` |
| Barramento de verificação | Autoverificação (§6.3) | este doc |

**A métrica que revela se o TIR funciona é a taxa de sucesso da chamada** — o código gerado **executa**? Um modelo que escreve chamadas plausíveis mas sintaticamente inválidas, ou que inventa APIs, aprendeu a imitar TIR sem fazer TIR. Meta: **≥ 95% de chamadas executáveis**. É por isso que o DOC-09 §3.2 exige que todo traço de ferramenta no SFT tenha sido **realmente executado**.

---

## 9. OQ-5 — verificação formal em Lean/Isabelle é viável?

Avaliação honesta de uma ideia atraente.

**A favor:** provadores interativos dão garantia de correção que nenhum CAS oferece. O `mathlib` do Lean formalizou uma quantidade impressionante de matemática. Se derivações de Física pudessem ser formalizadas, teríamos verificação total.

**Contra, e é decisivo no horizonte deste programa:**

| Obstáculo | Detalhe |
|---|---|
| **Cobertura** | O `mathlib` é rico em matemática pura e **muito pobre em Física**. Não há formalização madura de mecânica lagrangiana, eletromagnetismo ou teoria de campos |
| **Natureza do raciocínio** | Derivação física **não é demonstração de teorema**. É cálculo com justificativa física: aproximações controladas, descarte de termos de ordem superior, hipóteses de regime. Assistentes de prova modernos servem mal a isso |
| **Grandezas dimensionais** | Tipagem dimensional em provadores é área ativa e imatura |
| **Custo de formalização** | Formalizar uma única derivação de livro-texto leva ordens de grandeza mais tempo que escrevê-la |

**Decisão: não viável no Tier 2.** O barramento mecânico (§3) entrega a maior parte do valor a uma fração ínfima do custo.

**Nicho onde faz sentido, e fica registrado para o Tier 3:** verificar as **identidades puramente matemáticas** usadas dentro de uma derivação física — uma identidade de integral, uma manipulação tensorial, uma expansão em série. Aí o `mathlib` tem cobertura real. Seria um sétimo verificador, opcional e de alta confiança, aplicável a um subconjunto pequeno.

**Spike de viabilidade recomendado no Tier 3:** duas semanas, US$ 0 de computação, formalizando 20 identidades matemáticas extraídas de derivações reais do corpus, medindo cobertura e esforço. Não antes.

---

## 10. Requisitos de qualidade de código

`verify/` carrega o regime mais estrito do repositório, e a justificativa está no §1: um bug aqui é global.

| Requisito | Valor |
|---|---|
| Cobertura de testes | **≥ 95%**, imposto em CI |
| Tipagem | `mypy --strict` |
| Casos golden | §5 — quebra o build se mudarem |
| Determinismo | Mesma entrada → mesma saída, sempre. Sem aleatoriedade não semeada |
| Versionamento | `verifier_id` inclui a versão; resultados registram qual versão os produziu |
| Desempenho | Latência p99 registrada; timeout obrigatório em todos os caminhos |

O requisito de versionamento é sutil e importa: quando o verificador muda, **resultados antigos não são silenciosamente reinterpretados**. Um dado sintético aprovado pela versão 1.2 fica marcado como tal, e uma mudança que altere vereditos exige reavaliação explícita.

---

## 11. Orçamento

| Item | Recurso | Custo |
|---|---|---|
| Implementação dos seis verificadores | 4–6 semanas de trabalho | **US$ 0** |
| Construção dos casos golden (1.100 casos) | ~1 semana, manual | **US$ 0** |
| Sandbox (gVisor/Firecracker) | Configuração | **US$ 0** |
| Treino do PRM | GPU alugada | **~US$ 20** |
| Ablação PRM denso vs. resultado (OQ-35) | GPU alugada | **~US$ 30** |
| Geração de traços de TIR | Coberto no DOC-09 §8 | — |
| **Total** | | **~US$ 50** |

O ativo mais valioso do programa custa **cinquenta dólares em computação** e alguns meses de engenharia cuidadosa. É a melhor relação valor/custo de todo o projeto — e é exatamente por isso que ele foi colocado no centro da arquitetura desde o DOC-01, antes de qualquer modelo existir.

---

## 12. Riscos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Bug no verificador contamina dados, recompensa **e** avaliação simultaneamente | Média | **Crítico** | Cobertura ≥ 95%; golden tests; verificador reservado do DOC-09 §5.4 |
| `INCONCLUSIVE` tratado como `FAIL` em algum caminho de código | Média | **Alto** | Teste explícito de cada consumidor; §2.1 |
| SymPy assume comutatividade e aprova identidade falsa de operadores | **Alta** | **Alto** | Anotação de tipo; `INCONCLUSIVE` sem anotação confiável; casos golden com operadores |
| Análise dimensional vacuamente aprovada em unidades naturais | **Alta** | Médio | Verificação de dimensão de massa (§3.2) |
| Escape do sandbox | Baixa | **Crítico** | Firecracker; sem rede; auditoria da configuração |
| PRM hackeado e domina a recompensa | Média | Alto | Peso ≤ 0,3; resultado permanece dominante |
| Taxa de `INCONCLUSIVE` alta demais torna o RLVR ineficiente | Média | Médio | Monitorar por tipo de problema; melhorar o verificador, não relaxar o critério |

---

## 13. Critérios de aceite do Stage-Gate 9

- [ ] **J1** — Seis verificadores implementados sob o protocolo único da §2
- [ ] **J2** — Cobertura ≥ 95% e `mypy --strict` limpos em `verify/`
- [ ] **J3** — 1.100 casos golden congelados, **incluindo a categoria indecidível**
- [ ] **J4** — Todo consumidor testado quanto ao tratamento correto de `INCONCLUSIVE`
- [ ] **J5** — Barramento exercitado em ≥ 10⁶ equações reais; taxas de `PASS`/`FAIL`/`INCONCLUSIVE` publicadas por subárea
- [ ] **J6** — Sandbox auditado: tentativa deliberada de escape falha
- [ ] **J7** — Discordância entre verificadores registrada e alertada, nunca resolvida por voto
- [ ] **J8** — Taxa de sucesso de chamada de ferramenta ≥ 95%
- [ ] **J9** — Decisão de OQ-5 registrada como ADR

---

## 14. Questões em aberto

| ID | Questão | Destino |
|---|---|---|
| OQ-39 | Qual a taxa real de `INCONCLUSIVE` por subárea? Alta demais inviabiliza RLVR em partes da Física | Medir em J5 — pode redirecionar esforço de engenharia |
| OQ-40 | Autoverificação em inferência (§6.3) compensa o custo de latência? | DOC-15 |
| OQ-41 | Vale um verificador de "razoabilidade física" (ordem de grandeza plausível)? | Depois do PhysBench |
| OQ-42 | Como verificar Física **conceitual**? (herdado de OQ-38) | DOC-12 — provavelmente não é verificável, e a avaliação precisa admitir isso |

---

## 15. Referências

1. Richardson, D. (1968). *Some Undecidable Problems Involving Elementary Functions of a Real Variable.* Journal of Symbolic Logic 33(4).
2. Lightman, H. et al. (2024). *Let's Verify Step by Step.* ICLR.
3. Gao, L., Schulman, J., Hilton, J. (2023). *Scaling Laws for Reward Model Overoptimization.* ICML.
4. DeepSeek-AI (2025). *DeepSeek-R1.* arXiv:2501.12948.
5. Gou, Z. et al. (2024). *ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving.* ICLR.
6. Snell, C. et al. (2024). *Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters.* arXiv:2408.03314.
7. Meurer, A. et al. (2017). *SymPy: symbolic computing in Python.* PeerJ CS.
8. Agache, A. et al. (2020). *Firecracker: Lightweight Virtualization for Serverless Applications.* NSDI.
9. The mathlib Community (2020). *The Lean Mathematical Library.* CPP.

---

**Fim do DOC-10.** Encerra a Fase 2. Revisão da §13 necessária antes do DOC-11 (PhysBench).
