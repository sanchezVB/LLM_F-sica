# DOC-05 — Projeto do Tokenizer e Vocabulário Físico-Matemático

**Projeto:** ΦFM — Phi Foundation Models para Física
**Status:** `RASCUNHO v0.1` — aguardando revisão do Stage-Gate 4
**Cobre:** entregável solicitado **6** (pipeline de tokenização)
**Depende de:** [DOC-03 §3](DOC-03-ingestao-parsing-normalizacao.md), [DOC-04](DOC-04-filtragem-dedup-descontaminacao.md), [ADR-0001](../adr/ADR-0001-decisoes-stage-gate-0.md) (Q4)
**Data:** 2026-08-03

---

## 1. Por que o tokenizer é uma decisão de arquitetura, não de pré-processamento

O tokenizer determina o que o modelo **pode** representar. É a única camada do sistema cujos erros são irrecuperáveis: nenhuma quantidade de treino conserta uma segmentação que destrói estrutura.

Considere o que um tokenizer de propósito geral faz com uma expressão tensorial trivial:

```
Entrada:   T^{\mu\nu} = \partial^\mu \phi \, \partial^\nu \phi - \eta^{\mu\nu} \mathcal{L}
```

| Tokenizer | Tokens | O que acontece |
|---|---|---|
| Genérico (regex tipo `cl100k`) | ~34 | `\` separado de `partial`; `mu` fragmentado; `^{` quebrado |
| Consciente de Física (alvo) | ~18 | `\partial`, `\mu`, `\nu`, `^{`, `}` são unidades atômicas |

Não é apenas custo. Quando `\partial` é fragmentado em `\` + `par` + `tial`, o modelo precisa **aprender** que essa sequência específica de três tokens significa derivada parcial — e precisa reaprender isso em cada variação de contexto. Com token atômico, a relação é dada de graça pela representação. É o modo de falha **F5** (colapso de notação) do DOC-00 §2, atacado na sua origem.

**Restrição operacional (DOC-17A §6.1):** o tokenizer precisa ser **congelado antes do primeiro upload grande de shards**. Retokenizar significa refazer o upload de 20–80 GB numa conexão doméstica. O tokenizer é, portanto, a última decisão reversível barata do pipeline de dados.

---

## 2. A decisão se bifurca: dois tokenizers, não um

A resolução Q4 (Qwen3-8B-Base como base de CPT) força caminhos diferentes para os dois tiers de modelo. Tratá-los como um único problema seria erro de projeto.

| | **ΦEnc / ΦEmb / ΦRank** | **ΦGen (1,5B, 8B)** |
|---|---|---|
| Origem | **Treinado do zero** (D-01 autoriza) | **CPT sobre Qwen3-8B-Base** |
| Liberdade de tokenizer | **Total** | Restrita — precisa ser compatível com os pesos existentes |
| Estratégia | Tokenizer próprio, do zero | **Extensão** do vocabulário do Qwen3 |
| Vocabulário | 32k–48k (§7) | 151.936 + ~1,5–3k tokens novos |
| Tratamento de dígitos | Escolha nossa (§5) | **Herdado** — não modificável |
| Regex de pré-tokenização | Escolha nossa (§8) | Herdado, com adendos |

> **Consequência de Q4 que merece registro explícito.** Estender um tokenizer significa **herdar** o tratamento de números, o regex de pré-tokenização e a política de espaços do modelo base. Podemos *adicionar* tokens (ganho puro), mas não podemos reestruturar como números já são segmentados sem destruir as representações numéricas que o Qwen3 levou trilhões de tokens para aprender. A liberdade de projeto do §5 aplica-se integralmente apenas ao ΦEnc.

Isso torna o ΦEnc mais interessante cientificamente: é onde de fato se responde se um tokenizer nativo de Física vale a pena — a questão em aberto **OQ-2** do DOC-01 §11, e evidência direta para a decisão do Stage-Gate 4 sobre treino do zero.

---

## 3. Os algoritmos, comparados

### 3.1 Esclarecimento necessário

**SentencePiece não é um algoritmo.** É uma biblioteca (Kudo & Richardson, 2018) que implementa BPE *e* Unigram, tratando a entrada como fluxo de bytes bruto — sem exigir pré-tokenização por espaço, e codificando o espaço como caractere (`▁`). A confusão entre "SentencePiece" e "Unigram" é generalizada e importa aqui, porque a escolha real é entre **BPE e Unigram**, e ambas podem ser implementadas em SentencePiece ou em `tokenizers` (HuggingFace).

### 3.2 Comparação

| Algoritmo | Como constrói o vocabulário | Prós | Contras | Adequação a LaTeX |
|---|---|---|---|---|
| **BPE** (Sennrich et al., 2016) | Funde iterativamente o par de símbolos adjacentes mais **frequente** | Determinístico; rápido; ecossistema maduro (`tiktoken`, HF fast); usado por GPT, Llama, Qwen | Fusão gulosa não otimiza objetivo global; segmentação pode ser subótima | **Boa** — sequências LaTeX frequentes viram tokens naturalmente |
| **WordPiece** (Schuster & Nakajima, 2012) | Funde o par que mais aumenta a **verossimilhança**: `score = freq(ab)/(freq(a)·freq(b))` | Melhor morfologia que BPE em linguagem natural | Preso a pré-tokenização por palavra inteira; marcadores `##`; ecossistema em declínio | **Ruim** — "palavra" é conceito mal definido em LaTeX; `##` polui expressões |
| **Unigram LM** (Kudo, 2018) | Começa com vocabulário grande e **poda** para maximizar verossimilhança sob modelo unigrama | Probabilístico; permite *subword regularization* (amostrar segmentações); unidades mais plausíveis (Bostrom & Durrett, 2020) | Treino mais lento; menos suporte em ferramentas de inferência | **Boa a muito boa** — a poda por verossimilhança tende a preservar sequências de controle inteiras |
| **Nível de caractere / byte** | Sem vocabulário aprendido | Sem perda; sem OOV | Sequências longuíssimas; custo quadrático de atenção proibitivo | ✗ Inviável |

### 3.3 Decisão

| Modelo | Algoritmo | Justificativa |
|---|---|---|
| **ΦGen** | **BPE** | Não é escolha — é o algoritmo do Qwen3. A extensão precisa ser BPE |
| **ΦEnc** | **A medir** (§11) — BPE e Unigram | Bostrom & Durrett (2020) mostram vantagem do Unigram em linguagem natural; **não há evidência publicada para LaTeX/Física**. Produzir essa evidência é barato e é contribuição própria |

**Byte-fallback é obrigatório nos dois casos.** Física escreve `∇`, `∂`, `ℏ`, `⟨ψ|`, `⊗`, `†`, `∮`, `ℝ`, `𝒪`, `Å`, `μ`. Sem byte-fallback, esses caracteres viram `<UNK>` e a informação é destruída de forma irreversível. Com byte-fallback, o pior caso é ineficiência, nunca perda. **Não negociável.**

---

## 4. Tokens LaTeX: derivados dos dados, não escritos à mão

### 4.1 Método

Uma lista curada à mão de "tokens importantes de Física" seria enviesada pelo autor da lista. O método correto usa o corpus que já temos:

1. Extrair **todas** as sequências de controle `\[a-zA-Z]+\*?` do corpus parseado (DOC-03).
2. Ordenar por frequência de documento (não por frequência bruta — evita que um único paper com 10.000 usos de uma macro pessoal domine).
3. Aplicar piso de frequência: presente em ≥ 500 documentos distintos.
4. Adicionar os construtos estruturais do §4.2, que são raros como string mas críticos como estrutura.

**Estimativa: 1.200–2.500 sequências de controle** passam o piso. Custo: uma varredura sobre dados já processados, **US$ 0**.

### 4.2 O que precisa ser atômico

| Classe | Exemplos | Por quê |
|---|---|---|
| **Letras gregas** | `\alpha` `\beta` `\gamma` `\mu` `\nu` `\psi` `\Psi` `\phi` `\varphi` `\epsilon` `\varepsilon` `\Omega` `\hbar` | Símbolos primitivos de Física. `\epsilon` e `\varepsilon` são **tokens distintos** — DOC-03 §3.2 |
| **Operadores diferenciais** | `\partial` `\nabla` `\Delta` `\square` `\mathrm{d}` | Núcleo de toda EDP e teoria de campos |
| **Operadores integrais e de soma** | `\int` `\iint` `\oint` `\sum` `\prod` `\lim` | |
| **Estruturais de fração e raiz** | `\frac` `\sqrt` `\binom` | |
| **Notação de Dirac** | `\langle` `\rangle` `\vert` `\dagger` `\otimes` `\oplus` | Toda a Mecânica Quântica |
| **Relações** | `\approx` `\propto` `\sim` `\equiv` `\leq` `\geq` `\neq` `\rightarrow` `\to` `\mapsto` | |
| **Decoradores** | `\hat` `\vec` `\bar` `\tilde` `\dot` `\ddot` `\mathbf` `\mathcal` `\mathbb` `\mathrm` | Distinção operador/vetor/escalar é semântica |
| **Ambientes** | `\begin{equation}` `\end{equation}` `\begin{align}` `\end{align}` `\begin{pmatrix}` … | Tokens compostos; delimitam blocos matemáticos |
| **Estruturais de índice** | `^{` `_{` `}` `{` `&` `\\` | ★ Ver §6 |
| **Espaçamento matemático** | `\,` `\;` `\!` `\quad` `\qquad` | Carregam sentido tipográfico em equações |
| **Constantes e unidades** | `\hbar` `\epsilon_0` `\mu_0` `\mathrm{eV}` `\mathrm{K}` | Alta frequência; merecem token próprio |

### 4.3 O que **não** deve ser atômico

| Tentação | Por que rejeitar |
|---|---|
| `^{\mu\nu}` como token único | Explosão combinatória — há centenas de combinações de índices. O modelo deve **compor** a partir de `^{`, `\mu`, `\nu`, `}` |
| Equações inteiras frequentes | Impede generalização; o modelo decoraria em vez de compor |
| Macros de autor (`\eps`, `\bra`) | Já foram expandidas no DOC-03 §2. Se sobreviveram, são ruído |

---

## 5. Números: o ataque direto ao modo de falha F4

Aplicável integralmente ao ΦEnc; herdado no ΦGen (§2).

### 5.1 O problema

Sem tratamento especial, o BPE aprende `2024` como token único (frequente em datas) e `2025` como dois tokens. O modelo perde qualquer noção de valor posicional. Erros de aritmética e de conversão de unidade — o modo de falha **F4** — decorrem diretamente disso.

### 5.2 Opções

| Estratégia | Usado por | Prós | Contras |
|---|---|---|---|
| Sem tratamento | Modelos antigos | Compressão máxima | ✗ Aritmética destruída |
| **Dígito único** | PaLM, Minerva, Llama 2 | Valor posicional perfeitamente consistente; só 10 tokens | Mais tokens por número |
| Grupos de até 3 dígitos | Llama 3, GPT-4 | Melhor compressão | 1.000 tokens de vocabulário; fronteiras inconsistentes em notação científica |
| Grupos de 3, da direita para a esquerda | Variantes recentes | Alinha com valor posicional | Complexidade de implementação |

### 5.3 Decisão para o ΦEnc: **dígito único**

Três razões específicas de Física:

1. **Notação científica é onipresente.** `1.602\times10^{-19}` tem mantissa, expoente e sinal que precisam de fronteiras consistentes. Agrupamento de 3 dígitos quebra de formas diferentes conforme o número de dígitos da mantissa.
2. **Ordem de grandeza é raciocínio de primeira classe.** Estimativa de Fermi é habilidade central (DOC-00 §3.4, item 8) e depende de manipular expoentes como objetos.
3. **O custo é pequeno.** Números representam ~2–4% do texto de Física. Dígito único aumenta a contagem total de tokens em ~1–2% — troca claramente favorável.

> **Nota sobre o ΦGen.** O Qwen3 usa seu próprio esquema. **Não o alteramos.** Alterá-lo destruiria representações numéricas que custaram trilhões de tokens. A compensação para F4 no ΦGen não é o tokenizer — é o **uso obrigatório de ferramentas** (DOC-14): o modelo não calcula, ele chama uma calculadora e verifica o resultado.

---

## 6. Expressões tensoriais e estrutura de índices

O caso mais difícil, e onde tokenizers genéricos falham de forma mais visível.

```
R^{\rho}_{\ \sigma\mu\nu} = \partial_\mu \Gamma^{\rho}_{\nu\sigma}
                          - \partial_\nu \Gamma^{\rho}_{\mu\sigma} + \ldots
```

**Requisitos de segmentação:**

1. `^{` e `_{` são **tokens estruturais únicos**, não `^` + `{`. A posição do índice — covariante vs. contravariante — é semântica em Relatividade Geral, e separar o marcador do delimitador espalha essa informação por dois tokens.
2. `}` é token único e fecha a estrutura.
3. `\ ` (contrabarra-espaço, usado para alinhar índices em tensores) precisa ser preservado, não colapsado pela normalização de espaços.
4. Cada índice grego é um token; a **sequência** de índices é composicional.
5. `\Gamma`, `\Lambda`, `\eta`, `\delta` são tokens próprios.

Com isso, `R^{\rho}_{\ \sigma\mu\nu}` fica em **9 tokens** com estrutura totalmente explícita, contra ~20 tokens embaralhados num tokenizer genérico.

**Consequência prevista, a ser medida:** a estrutura de índices exposta ao modelo deve melhorar tarefas de Relatividade Geral e Teoria de Campos. É uma hipótese testável, incluída no bake-off do §11 como métrica estratificada por subárea.

---

## 7. Tamanho de vocabulário

### 7.1 O trade-off

| Vocabulário maior | Vocabulário menor |
|---|---|
| ✅ Melhor compressão → mais conteúdo por token → treino mais barato | ✅ Matriz de embedding menor |
| ✅ Contexto efetivo maior | ✅ Softmax de saída mais rápido |
| ❌ Embedding e softmax crescem linearmente | ❌ Sequências mais longas |
| ❌ Tokens raros recebem pouco gradiente | ❌ Fragmentação de notação |

### 7.2 A conta que decide

O custo do vocabulário é `V × d` parâmetros na entrada (e outro tanto na saída, se não houver *weight tying*):

| Modelo | `d` | V = 32k | V = 48k | V = 100k | V = 152k |
|---|---|---|---|---|---|
| ΦEnc-150M | 768 | 24,6 M (**16%**) | 36,9 M (**25%**) | 76,8 M (**51%**) | 116,7 M (**78%**) |
| ΦGen-1,5B | 1536 | 49 M (3%) | 74 M (5%) | 154 M (10%) | 233 M (16%) |
| ΦGen-8B | 4096 | 131 M (2%) | 197 M (2%) | 410 M (5%) | 622 M (8%) |

**A coluna do ΦEnc é a resposta.** Com V = 100k, **metade do modelo seria a tabela de embeddings** — parâmetros que não fazem computação, só memória associativa. Para um encoder de 150M, é desperdício grosseiro.

### 7.3 O argumento que costuma ser esquecido

O Qwen3 usa 151.936 tokens porque cobre **mais de 100 idiomas**. Nós cobrimos **inglês + notação matemática**. O orçamento de vocabulário de um tokenizer monolíngue e de domínio único é radicalmente menor para a mesma qualidade de compressão.

**Decisão: ΦEnc com V = 40.960** (múltiplo de 1024, alinhamento favorável a GPU), assim distribuído:

| Categoria | Tokens | Observação |
|---|---|---|
| Bytes (fallback) | 256 | Obrigatório |
| Especiais | ~16 | `[CLS]`, `[SEP]`, `[MASK]`, `[PAD]`, sentinelas |
| Dígitos | 10 | §5 |
| Sequências de controle LaTeX | ~2.000 | §4, derivadas dos dados |
| Estruturais matemáticos | ~64 | `^{`, `_{`, `}`, `&`, `\\`, … |
| Unicode matemático | ~256 | ∇ ∂ ℏ ⟨ ⟩ ⊗ † ∮ ℝ 𝒪 … |
| **Aprendidos por BPE/Unigram** | **~38.400** | Vocabulário geral de Física em inglês |

**ΦGen: 151.936 + extensão (§9) ≈ 154.500.**

> Ressalva honesta: Tao et al. (2024) argumentam que o vocabulário ótimo cresce com o tamanho do modelo e que a maioria dos modelos é sub-vocabularizada. O argumento vale para modelos gerais. Para domínio único, a curva se desloca. **V = 40.960 é uma hipótese fundamentada, e o bake-off do §11 inclui V ∈ {32k, 41k, 64k} como eixo de ablação.**

---

## 8. Pré-tokenização: onde tokenizers genéricos quebram LaTeX

Antes do BPE/Unigram operar, o texto é fatiado por um regex de pré-tokenização. **É aqui que a maioria dos tokenizers destrói LaTeX**, e é a parte menos discutida do projeto de tokenizers.

O regex do `cl100k` (GPT-4) segmenta por espaço e por fronteiras letra/dígito/pontuação. Aplicado a `\frac{d^2x}{dt^2}`:

```
\  frac  {  d  ^  2  x  }  {  dt  ^  2  }        →  13 pré-tokens
```

A contrabarra é separada de `frac`, e o BPE nunca pode voltar a uni-las porque a pré-tokenização é uma barreira dura.

**Regra adicional obrigatória, com precedência máxima:**

```regex
\\[a-zA-Z]+\*?          # sequência de controle é pré-token atômico
|\\begin\{[a-z*]+\}     # abertura de ambiente
|\\end\{[a-z*]+\}       # fechamento de ambiente
|\^\{|_\{               # marcadores estruturais de índice
|\\[,;!]                # espaçamento matemático
|\\\\                   # quebra de linha em matriz/align
```

Sem isso, **nenhuma quantidade de treino de BPE produz tokens LaTeX atômicos** — a barreira de pré-tokenização os torna inalcançáveis. É a decisão mais consequente e menos visível deste documento.

Para o **ΦGen**, o regex do Qwen3 é herdado; a extensão adiciona as regras acima como pré-passo. Ganho menor que no ΦEnc, mas positivo.

---

## 9. Extensão de vocabulário para o CPT

### 9.1 O problema

Adicionar N tokens novos a um modelo pré-treinado cria N linhas de embedding sem significado. Inicializadas aleatoriamente, produzem ativações fora de distribuição, e a perda dispara nos primeiros passos — desperdiçando computação e, na pior hipótese, desestabilizando o modelo.

### 9.2 Métodos

| Método | Descrição | Qualidade |
|---|---|---|
| Aleatório | `N(0, σ)` | ✗ Ruim — spike de perda garantido |
| Média global | Média de todos os embeddings existentes | Baseline aceitável (Hewitt, 2021) |
| **Média da segmentação antiga** ✅ | O token novo `\alpha` é inicializado com a média dos embeddings de como o modelo **já** segmentava `\alpha` (`\` + `al` + `pha`) | ★ O token novo nasce onde o modelo já sabe o que aquele texto significa |
| FOCUS (Minixhofer et al., 2024) | Combinação ponderada por similaridade semântica com tokens sobrepostos | Melhor, mais complexo |

**Selecionado: média da segmentação antiga, com aquecimento de embeddings.**

### 9.3 Procedimento

1. Calcular, para cada token novo, a segmentação que o tokenizer original produzia.
2. Inicializar entrada **e** saída (o Qwen3 não amarra embeddings nos tamanhos maiores — verificar) com a média ponderada dessas linhas.
3. **Fase de aquecimento:** ~200–500 passos treinando **apenas** as linhas novas de embedding, com o resto do modelo congelado.
4. Descongelar e iniciar o CPT normal.

### 9.4 O teste de sanidade que não pode ser pulado

> **Medir a perplexidade num conjunto reservado de texto geral (não-Física), antes e depois da extensão, sem nenhum treino.**
>
> Se a inicialização estiver correta, a diferença é **desprezível** — o modelo estendido representa o mesmo texto quase tão bem quanto o original. Se a perplexidade disparar, a inicialização está errada e o CPT partiria de um modelo já danificado.

Custo: minutos. É a diferença entre detectar um erro de extensão agora ou descobri-lo depois de gastar 84 horas de H100 e não entender por que o critério **G2.3** (sem regressão em capacidade geral) falhou.

---

## 10. Corpus de treino do tokenizer

**Não é o corpus inteiro.** 5–20 GB de texto bastam; além disso as frequências estabilizam.

**Requisito crítico — amostragem estratificada.** As fusões do BPE são movidas por frequência. Uma amostra não estratificada seria dominada por `astro-ph` e `cond-mat`, que são as maiores subáreas, e o vocabulário resultante sub-representaria a notação de `hep-th` e `gr-qc`. Como o §6 argumenta que a notação tensorial é justamente onde há mais a ganhar, isso desfaria o principal ganho do projeto.

| Componente | Proporção | Justificativa |
|---|---|---|
| Papers, estratificados pelas 23 subáreas | 60% | Cobertura equilibrada, **não** proporcional ao corpus |
| Material pedagógico (livros abertos, StackExchange) | 15% | Prosa explicativa, sub-representada em papers |
| Matemática pura (`math.*`) | 10% | Notação de apoio |
| Código científico | 10% | Sintaxe de programação |
| Texto geral em inglês | 5% | Evita degradar linguagem natural |

A amostra reflete a **mistura-alvo de treino** (DOC-06), não a distribuição bruta do corpus. Tokenizer e mistura de dados são decisões acopladas.

---

## 11. Avaliação: métricas e o bake-off

### 11.1 Métricas intrínsecas

| Métrica | Definição | Meta |
|---|---|---|
| **Fertilidade** | Tokens ÷ palavras, em texto de Física reservado | **≤ 0,80× do Qwen3** |
| **Fertilidade em equações** ★ | Tokens ÷ equação, isolando ambientes matemáticos | **≤ 0,65× do Qwen3** |
| Taxa de compressão | Bytes ÷ token | Maior é melhor |
| **Fidelidade de round-trip** | `decode(encode(x)) == x` | **100%** — garantido por byte-fallback; regressão = bug crítico |
| Taxa de byte-fallback | Fração de tokens que caem em bytes | < 0,5% em texto de Física |
| Consistência numérica | `1.5` segmentado igual em todos os contextos | 100% |
| Estratificação por subárea | Fertilidade medida **por subárea** | Nenhuma subárea > 1,25× da mediana |

**Baselines obrigatórios:** Qwen3 (151.936), Llama 3 (128k), GPT-4 `o200k`, e o tokenizer do PhysBERT. Comparar apenas com tokenizers ruins seria o mesmo erro que o DOC-00 §3.1 denuncia.

> **Sobre o que a compressão significa.** Comprimir 20% melhor **não** aumenta o corpus — reduz a contagem de tokens do mesmo conteúdo. O ganho real é que o modelo vê **mais Física por unidade de computação**, já que o custo de treino é proporcional a tokens. Ali et al. (2024) mostram que eficiência de tokenizer correlaciona com desempenho a jusante. O ganho é real, mas é de eficiência, não de volume — e as contagens de tokens do DOC-04 §7 são medidas em tokenizer geral e precisarão ser reafirmadas neste.

### 11.2 O bake-off: a única medida que decide

Métricas intrínsecas são proxies. A questão real é: **qual tokenizer produz o melhor modelo?**

Podemos responder isso empiricamente, e é barato:

| Variante | Eixo testado |
|---|---|
| A | BPE, V=40.960, dígito único |
| B | Unigram, V=40.960, dígito único |
| C | BPE, V=32.768, dígito único |
| D | BPE, V=65.536, dígito único |
| E | BPE, V=40.960, **sem** regras de pré-tokenização do §8 |
| F | Qwen3 sem modificação (**controle**) |

**Protocolo:** treinar um encoder de ~50 M parâmetros em 5 B tokens para cada variante; avaliar em recuperação de Física, MLM em texto denso em equações, e uma sonda de estrutura tensorial.

**Custo:** 50 M × 5 B → `C = 6 × 5e7 × 5e9 = 1,5e18` FLOPs → ~7 h numa RTX 4090 → **~US$ 2,40 por variante**. **Seis variantes: ~US$ 15.**

A variante E é a mais importante: isola o efeito do regex de pré-tokenização, que é a decisão que a §8 afirma ser a mais consequente e menos visível. Se E empatar com A, a §8 está errada e o documento precisa ser revisado.

> **Não há evidência publicada comparando algoritmos de tokenização em texto de Física e LaTeX.** Por US$ 15 e alguns dias, produzimos essa evidência. É a primeira contribuição científica original do programa, e ela cabe inteira no degrau T0 do DOC-17A §8.2.

---

## 12. Riscos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Tokenizer congelado cedo demais, precisa mudar | Média | **Alto** — reupload de 20–80 GB | Bake-off (§11) **antes** de congelar; só então o upload |
| Extensão do Qwen3 desestabiliza o modelo base | Média | **Alto** — falharia G2.3 | Teste de sanidade de perplexidade (§9.4), custo de minutos |
| Vocabulário sub-representa subáreas menores | **Alta** | Médio | Amostra estratificada (§10); fertilidade medida por subárea |
| Regex de pré-tokenização quebra casos-limite de LaTeX | Média | Médio | Suíte golden com 500 expressões reais; round-trip 100% obrigatório |
| Byte-fallback mascara problema de cobertura | Baixa | Baixo | Taxa de fallback monitorada; > 0,5% dispara investigação |
| Ganho do tokenizer próprio não se materializa | Média | Médio | **É um resultado válido** — responde OQ-2 e informa o Stage-Gate 4. Resultado negativo é entregável (DOC-00 §6.5) |

---

## 13. Critérios de aceite do Stage-Gate 4

- [ ] **E1** — Fertilidade ≤ 0,80× do Qwen3 em texto de Física; ≤ 0,65× em equações
- [ ] **E2** — Round-trip 100% fiel em suíte golden de 500 expressões LaTeX reais
- [ ] **E3** — Nenhuma subárea com fertilidade > 1,25× da mediana
- [ ] **E4** — Bake-off das seis variantes concluído; escolha justificada por medição, não por preferência
- [ ] **E5** — Variante E avaliada; efeito do regex de pré-tokenização quantificado
- [ ] **E6** — Extensão do Qwen3 passa no teste de perplexidade (§9.4) com desvio desprezível
- [ ] **E7** — Tokenizer congelado, versionado por hash de conteúdo e registrado no manifesto
- [ ] **E8** — Contagens de tokens do DOC-04 §7 reafirmadas no tokenizer definitivo

---

## 14. Questões em aberto

| ID | Questão | Destino |
|---|---|---|
| OQ-18 | BPE ou Unigram para LaTeX? | §11 — mede e decide |
| OQ-19 | O Qwen3 amarra embeddings de entrada e saída no 8B? | Verificar antes de implementar §9.3 |
| OQ-20 | Vale um tokenizer separado para o ΦCode? | DOC-07 |
| OQ-21 | Regularização por amostragem de subpalavra (só possível com Unigram) ajuda em Física? | §11, se a variante B vencer |

---

## 15. Referências

1. Sennrich, R., Haddow, B., Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL.
2. Kudo, T. (2018). *Subword Regularization: Improving NMT Models with Multiple Subword Candidates.* ACL.
3. Kudo, T., Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer.* EMNLP.
4. Schuster, M., Nakajima, K. (2012). *Japanese and Korean Voice Search.* ICASSP.
5. Bostrom, K., Durrett, G. (2020). *Byte Pair Encoding is Suboptimal for Language Model Pretraining.* Findings of EMNLP.
6. Hewitt, J. (2021). *Initializing New Word Embeddings for Pretrained Language Models.* Technical note.
7. Minixhofer, B. et al. (2024). *FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models.* EMNLP.
8. Tao, C. et al. (2024). *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies.* NeurIPS.
9. Ali, M. et al. (2024). *Tokenizer Choice For LLM Training: Negligible or Crucial?* Findings of NAACL.
10. Lewkowycz, A. et al. (2022). *Solving Quantitative Reasoning Problems with Language Models* (Minerva). NeurIPS.

---

**Fim do DOC-05.** Revisão da §13 necessária antes do DOC-06 (Mistura de Dados, Currículo e Motor de Dados Sintéticos).
