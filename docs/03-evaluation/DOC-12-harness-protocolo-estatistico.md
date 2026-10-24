# DOC-12 — Harness de Avaliação e Protocolo Estatístico

**Projeto:** ΦFM — Phi Foundation Models para Física
**Status:** `RASCUNHO v0.1` — aguardando revisão do Stage-Gate 11
**Cobre:** entregável solicitado **11** (pipeline de avaliação); **encerra a Fase 3**
**Depende de:** [DOC-00 §6](../00-foundations/DOC-00-project-charter.md), [DOC-10](../02-models/DOC-10-raciocinio-verificacao-ferramentas.md), [DOC-11](DOC-11-physbench.md)
**Data:** 2026-08-03

---

## 1. A tese: o protocolo importa mais que os números

O DOC-11 define **o que** medir. Este define **como medir sem se enganar**.

A literatura de LLMs está cheia de comparações que não se sustentam: sementes únicas, ausência de intervalos de confiança, baselines convenientes, escolha post hoc da métrica que favorece, contaminação não reportada. Nada disso é fraude — é ausência de protocolo.

> Este programa faz alegações comparativas contra modelos publicados. **Uma alegação sem protocolo estatístico não é resultado, é anedota.** O DOC-00 §6 estabeleceu seis regras de integridade; este documento as torna executáveis e verificadas em CI.

---

## 2. Arquitetura do harness

```mermaid
flowchart LR
    A["configs/eval/*.yaml<br/>experimento imutável"] --> B[Runner]
    B --> C["Adaptador de modelo<br/>vLLM · SGLang · HF · API"]
    C --> D[Geração]
    D --> E["<b>Corretor</b><br/>verify/ · juiz · exato"]
    E --> F["<b>Estatística</b><br/>bootstrap · pareado · FDR"]
    F --> G["Relatório versionado<br/>+ manifesto de reprodução"]
    H[("Registro de<br/>contaminação")] --> G
    I[("Contador de<br/>avaliações privadas")] --> G

    classDef core fill:#1b4d3e,stroke:#4ade80,color:#e6fff4
    class E,F core
```

Construído sobre o **lm-evaluation-harness** da EleutherAI, estendido com nossos corretores. Não reimplementamos infraestrutura de avaliação (P6) — estendemos a que a comunidade já audita.

**Requisito estrutural:** o corretor (`E`) é **o mesmo `verify/`** usado como recompensa de RLVR. É a imposição em código do princípio P3 e a razão de `eval.grading` importar de `verify` no DAG de módulos do DOC-01 §4.3.

---

## 3. Protocolo estatístico

### 3.1 O que é reportado, sempre

| Elemento | Obrigatório |
|---|---|
| Média pontual | ✅ |
| **IC bootstrap de 95%** sobre os itens (10.000 reamostragens) | ✅ |
| Número de itens `n` | ✅ |
| Número de sementes | ✅ ≥ 3 quando há amostragem |
| Versão do conjunto de teste | ✅ |
| Taxa de contaminação | ✅ |
| Taxa de `INCONCLUSIVE` | ✅ |
| Configuração de decodificação | ✅ |

**Um número sem intervalo de confiança não é publicado.** Nem em tabela interna, nem em slide, nem em paper.

### 3.2 Variabilidade: duas fontes distintas

| Fonte | Como capturar |
|---|---|
| **Variabilidade de item** — quais problemas caíram no conjunto | Bootstrap sobre itens |
| **Variabilidade de amostragem** — o modelo é estocástico | ≥ 3 sementes; reportar média e desvio |

São independentes e ambas precisam ser reportadas. Reportar só sementes esconde que o conjunto pode ser pequeno demais; reportar só bootstrap esconde que o modelo é instável.

**Decodificação:** temperatura 0 (`pass@1` determinístico) como número principal, mais `avg@8` com temperatura 0,7 para modelos de raciocínio, onde a variância importa. **A configuração é fixada antes de rodar**, nunca escolhida depois.

### 3.3 Comparações pareadas

Comparar dois modelos nos mesmos itens é um problema **pareado**, e tratá-lo como independente desperdiça poder estatístico e infla o `n` necessário.

| Tipo de métrica | Teste |
|---|---|
| Binária (certo/errado) | **McNemar** sobre pares discordantes |
| Contínua (escore de rubrica) | Bootstrap pareado da diferença |
| Ranqueamento (recuperação) | Wilcoxon pareado |

### 3.4 Múltiplas comparações

Dezesseis tarefas × vários modelos gera muitas hipóteses. Sem correção, **algo parecerá significativo por acaso** — com 16 tarefas e α = 0,05, a chance de ao menos um falso positivo é ~56%.

**Correção de Benjamini-Hochberg (FDR)** sobre a família de testes de cada alegação. Valores `p` brutos e corrigidos são ambos reportados.

### 3.5 Tamanho de efeito, não só significância

> Uma diferença de 0,3 ponto pode ser estatisticamente significativa com `n` grande e **irrelevante na prática**. Uma diferença de 8 pontos pode não atingir significância com `n` pequeno e ser **claramente real**.

Toda alegação reporta a **diferença absoluta com IC**, e o `p` corrigido é informação secundária. Diferenças com ICs sobrepostos são descritas como **não significativas**, nunca como vitória — regra 4 do DOC-00 §6.

### 3.6 Pré-registro

Mecanismo emprestado de ensaios clínicos, e o mais eficaz contra autoengano:

> **Antes de rodar qualquer avaliação de um marco, registrar em arquivo versionado:** quais tarefas são a manchete, qual o limiar de sucesso, quais baselines entram, e qual a hipótese. O commit precede a execução.

Isso impossibilita escolher post hoc a métrica que ficou boa. `configs/eval/preregistration/` guarda esses arquivos, e o relatório final aponta para o hash do pré-registro. Se uma métrica não pré-registrada for reportada, ela é rotulada **exploratória** — legítima, mas não confirmatória.

### 3.7 A regra do portão G2 — ACEITA em 2026-10-19

> **Estado: ACEITA pelo dono em 2026-10-19, como escrita.** Na mesma data, o dono decidiu não depender do GPQA: o G2.2 fica em 3 de 4. A regra foi escrita antes de medir as bases e antes de qualquer treino. Os parâmetros estão em [`configs/eval/preregistration/g2.yaml`](../../configs/eval/preregistration/g2.yaml). As travas estão em [`src/phifm/eval/g2.py`](../../src/phifm/eval/g2.py): `REGRA_G2_ACEITA = True` e `PROTOCOLO_G2_SHA256 = None`. Nenhum item de teste é tocado antes do hash do congelamento, que é gravado depois do dev e antes da primeira geração no teste. Aceitar a regra não autoriza treinar.

A regra cobre o G2.1, o G2.2 e o G2.3 do [DOC-00](../00-foundations/DOC-00-project-charter.md). O G2.4 (citação no modo RAG) e o G2.5 (calibração) precisam de regras próprias.

#### 3.7.1 Objeto

- **O que se compara.** O **ΦGen-CPT** é o checkpoint escolhido no dev, ainda um modelo base. Ele é comparado com **a própria base**, no mesmo protocolo, no mesmo motor e com o mesmo hash de configuração.
- **Dois portões independentes:** o ΦGen-1,7B contra a Qwen3-1.7B-Base e o ΦGen-8B contra a Qwen3-8B-Base. O "1,5B" do charter é a Qwen3-1.7B-Base.
- **Escopo do G2.2.** Pela nota de escopo do DOC-00, o G2.2 vale só para a 8B.
- **Pós-treino fica de fora.** Um ΦGen pós-treinado (SFT/RLVR) precisa de regra própria, com um braço de controle (a base com o mesmo pós-treino), escrita antes do T2b.

#### 3.7.2 Benchmarks e recortes

O **G2.1 decide** com 4 benchmarks abertos, 2.528 itens de teste:

| benchmark | recorte | teste | k |
|---|---|---|---|
| UGPhysics-en | resposta única de tipo limpo (NV, EX, EQ, MC, TF, IN), sem duplicata exata: 4.480 itens | 1.999 | 3 |
| OlympiadBench | `OE_TO_physics_en_COMP`, todos os itens, em 113 agrupamentos | 236 | 8 |
| PHYBench | `fullques_v1`, os 100 com gabarito | 100 | 16 |
| SciBench-física | class 56, fund 71, thermo 66 | 193 | 8 |

O **G2.3 decide** com três medidas:
- o MMLU sem as 4 matérias de física: 13.402 itens em 53 matérias;
- o EvalPlus, HumanEval+ 164 ∪ MBPP+ 378, num total de 542 problemas;
- o IFEval, com 541 prompts.

**Relatados, não decidem:**
- o MMLU de 57 matérias e o MMLU-física (640 itens);
- o HumanEval original, com as mesmas gerações do HumanEval+;
- a decodificação gulosa nos 529 itens de física;
- o IFEval cru;
- o GPQA-diamond-física, se o GPQA vier.

**Desenvolvimento, nunca relatado como resultado:**
- o UG-dev (≈899);
- o OlympiadBench `OE_TO_physics_zh_CEE` (115), como dev do extrator;
- os 34 `_sol` do SciBench;
- o MMLU dev (285) e o validation (1.531);
- o MBPP sem as tarefas do MBPP+.

**Fora de tudo:**
- `OE_MM_*` e `TP_*`;
- o UGPhysics-zh e os itens de várias respostas do UG;
- os 400 itens do PHYBench sem gabarito;
- o SciBench multimodal;
- JEEBench, TheoremQA e PhysReason.

**Sem GPQA.** O GPQA não entra no G2.1. O agregado fica fixado com os 4 abertos e não muda se o GPQA chegar.
- **No G2.2** vale o número do charter: ≥3 vitórias entre os 5 benchmarks listados, com o GPQA não medido contando como **não vencido**. Na prática isso exige 3 de 4. Não há substituto.
- **Se o dono aceitar os termos** do GPQA **antes** de a 8B-base ser medida, ele entra no G2.2 como 5º benchmark: recorte *main* com domínio Physics (187 itens), 5-shot CoT com exemplos de Química e Biologia, k=8.
- **Se chegar depois** da medida da 8B-base, ele fica só relatado.

#### 3.7.3 Partição dev/teste, lacre e acessos

**Princípio.** Tudo o que se escolhe é escolhido em material que nunca é teste: prompt, exemplos, extrator, corretor, teto de fichas, LR e checkpoint. Depois, tudo é congelado por hash.

**UGPhysics, partição por grupos.**
- **O que é um grupo.** É um componente conexo de itens que compartilham ≥1 13-grama de palavras do enunciado. Só contam os 13-gramas presentes em 2 a 50 itens.
- **A matéria do grupo** é a do item de menor índice.
- **O sorteio**, com a semente 20261019:
  - 20% dos grupos de cada matéria vão para o **DEV**;
  - dos grupos restantes, ≈2.000 itens elegíveis vão para o **TESTE**, em proporção à matéria e sorteados por grupo;
  - o resto é a **RESERVA LACRADA**.
- **Nenhum grupo é dividido.** O código é `g2.particionar_por_grupos`.
- **Contagens e destino.** Gravada em 2026-10-19, antes de qualquer saída de modelo: dev 899, teste 1.999 e reserva 1.582, em 3.948 grupos; a subamostra do G2.2 tem 1.004 itens (sha256 `3198479e…` no `g2.yaml`). As listas de ids vão para `data/processed/avaliacao/g2_particao.json`, com hash, **no git antes de qualquer saída de modelo**.

**Os outros três ficam 100% no teste.** O dev de formato vem do OlympiadBench-zh e dos `_sol` do SciBench.

**Desenvolvimento sem olhar o teste:**
1. **Autoteste do gabarito**, automático e sem modelo. Cada gabarito, posto em `\boxed{}`, tem de dar PASS contra si mesmo. O gabarito perturbado tem de dar FAIL:
   - o numérico é deslocado por 3× a tolerância do próprio item;
   - o simbólico recebe um símbolo trocado e um +1.

   O item que falha sai dos dois braços, por id. A string de um gabarito que falhou só é olhada com o aval do dono.
2. **Extrator e corretor**, desenvolvidos com a 1,7B-base no dev (k=1), em no máximo 5 iterações. Alvos: ≤5% sem resposta e ≤3% truncado.
3. **Auditoria do corretor**, feita pelo Claude às cegas em 200 gerações do dev, com o hash dos veredictos gravado antes. Alvo: concordância ≥95% e κ ≥0,8. Se não chegar lá em 5 iterações, a medida para e o dono decide por emenda.
4. **Sanidade das outras medidas:**
   - o MMLU é validado no dev;
   - as soluções canônicas do código passam 100% e as quebradas falham;
   - o envelope do IFEval é testado em 20 instruções escritas fora do IFEval.

**Lacre.**
- **As gerações de teste** são gravadas com sha256.
- **Até o veredito do ΦGen** só se leem agregados e contadores automáticos (sem resposta, truncado, INCONCLUSIVE). Nada se lê por item.
- **Um defeito do corretor achado depois** é corrigido no dev, reaplicado às saídas gravadas **dos dois braços** e registrado como desvio.

**Livro de acessos**, em `data/processed/avaliacao/g2_acessos.jsonl`, conferido por `g2.registrar_acesso`:
- **Por tamanho:** 1 acesso para a base e 1 para o ΦGen.
- **Repetição:** 1 só, por falha operacional, com a parcial descartada e registrada.
- **A base é re-medida** se o hash da configuração mudar. Ela só vale no veredito com o **mesmo** hash do ΦGen.
- **Cada comparador do G2.2** tem 1 acesso.
- **A reserva do UG** só abre por emenda escrita.

#### 3.7.4 Protocolo de geração

**Física**
- **Formato.** Modelo base, texto corrido, sem template de chat. Cabeçalho fixo e exemplos no formato `Problem / Solution`, cada solução terminando em `The final answer is $\boxed{…}$`.
- **Decodificação.** T=0,6, top-p 0,95, sem top-k, sem min-p e sem penalidades. A semente da amostra j é `sha256('bench:id:j')`, a mesma nos dois braços. Máximo de 4.096 fichas.
- **Paradas:** `\n\nProblem:`, `\nProblem:`, `<|endoftext|>` e `<|im_end|>`.
- **Guarda de repetição** no cliente: uma janela de 32 fichas que aparece 4 vezes corta a geração, igual nos dois braços.
- **Extração.** O último `\boxed{}` de chaves balanceadas. Se não houver, o que vem depois de `final answer is`. Se não houver, a resposta é "sem resposta".
- **Nota.** A do item é a média das k amostras (avg@k). FAIL, sem resposta, truncado, INCONCLUSIVE e ERROR valem 0. A do benchmark é a média dos itens ×100.

**Exemplos few-shot**, fixos e sorteados com a semente; nenhum pode ser irmão (13-grama) de um item de teste:

| benchmark | exemplos |
|---|---|
| UGPhysics | 4 do UG-dev, de 4 matérias diferentes |
| OlympiadBench | 4 do UG-dev |
| PHYBench | 3 EX do UG-dev |
| SciBench | 3 `_sol` do próprio livro, sem os quase-duplicados de itens de teste (saem 3 no thermo) |

**Correção.** É um adaptador novo sobre `verify/`, com um comparador numérico com tolerância, que hoje falta.

| benchmark | critério |
|---|---|
| UGPhysics NV | valor completo com erro relativo ≤1e-2. Desvio declarado: o oficial compara só a mantissa, e esse critério fica relatado |
| OlympiadBench | \|Δ\| ≤ `error` (absoluto); o único item sem `error` usa 1e-2 relativo. Sem o atalho oficial de percentagem |
| PHYBench | equivalência simbólica, com variáveis positivas |
| SciBench | `math.isclose(rel_tol=0,05)`, com a escala 10^x da unidade convertida |

Cada comparação tem limite de 10 s. Estourar o limite dá INCONCLUSIVE.

**Correções de dev do corretor, registradas antes do congelamento (2026-10-19).** O autoteste do gabarito no UG-dev excluía **17,9%** dos itens, contra o limite de 5%. A regra manda consertar o parser no dev, e foi o que se fez, olhando só gabaritos de dev. Tudo vale igual nos dois braços e entra no hash do congelamento (`g2_corretor.py`, `g2_itens.py`). As correções:

1. **Tipo efetivo pela forma do gabarito no UGPhysics.** O rótulo do dataset erra:
   - 105 dos 396 itens "NV" do dev têm gabarito simbólico, que passa a ser corrigido como EX;
   - EQ do tipo `grandeza = número` vira NV pelo lado direito; EQ do tipo `grandeza = expressão` (`\kappa(p, T) = …`) vira EX pelo lado direito, porque é a definição da grandeza; EQ sem `=` vira EX;
   - no OlympiadBench, "Numerical" com gabarito simbólico vira "Expression";
   - EQ com várias equações separadas por vírgula vira várias respostas;
   - IN escrito como desigualdade é comparado como desigualdade (mesma estrição e f_resp/f_gab constante positiva).

   A linha de resposta do prompt segue o tipo efetivo.
2. **Gabarito com cálculo** (integral, soma, produtório, ∂, limite, ∇, bra-ket, diferencial explícito, barra de avaliação) **não é verificável por regra**. O parser devolve uma expressão sem o significado, por exemplo ∂S/∂p de um símbolo vira 0. O comparador devolve INCONCLUSIVE, e o autoteste exclui o item dos dois braços. Prefere-se excluir a aprovar por coincidência de lixo.
3. **Gabarito que o parser não lê** (termo espectral `{}^{3}P_{0}`) é comparado como texto normalizado.
4. **Camada de Física sobre o `parse` do verify:**
   - `\pi` vira π, porque o parser o devolvia como símbolo;
   - `e^{x}` simbólico vira exp;
   - `\mu (…)` deixa de ser lido como função;
   - `I` é símbolo, não a unidade imaginária;
   - `\dot{x}` vira um identificador próprio, para que x e ẋ não colapsem;
   - `\langle E\rangle` simples vira identificador;
   - `\operatorname{coth}` vira `\coth`;
   - módulo |X| vira √(X²), porque o `parse` do verify **descartava `\left|E\right|` em silêncio**;
   - `d` antes de comando deixa de ser lido como diferencial;
   - linha e estrela viram subscrito nomeado;
   - `\{ \}` e `[ ]` viram parênteses nas expressões;
   - graus e minutos de arco viram número;
   - `v^2_2` (sobrescrito antes do subscrito) vira `v_2^2`. Antes o subscrito se perdia, e um gabarito do OlympiadBench-zh virava **zero**, de modo que a resposta "0" passava;
   - nomes de símbolo são normalizados (`v_{2}` e `v_2` são o mesmo);
   - `\coth`, `\sech` e `\csch`, que o parser não conhece, viram recíprocas de `tanh`, `cosh` e `sinh`.
5. **Aquecimento do parser.** A primeira chamada ao parser LaTeX passava do teto de 5 s do verify no Windows, e a primeira resposta saía INCONCLUSIVE. O corretor agora aquece o parser na importação.
6. **Ordem da cascata e tempo.** A amostragem numérica vem antes do `simplify`, que só entra quando ela não decide. Antes, a mesma comparação saía PASS com a máquina livre e INCONCLUSIVE por tempo com a máquina carregada, e a correção não era reprodutível. O teto de 10 s é de relógio: quem estoura é refeito uma vez antes de virar INCONCLUSIVE. A correção da medida roda sem outra carga pesada na máquina.

7. **Revisão pelo estresse adversarial (2026-10-19).** Três agentes atacaram o corretor com respostas sintéticas sobre itens de dev: mais de 1.400 casos, achados reproduzidos e conferidos por um verificador cético. O que mudou:
   - **Falsos positivos**, que inflariam a nota:
     - O parser do SymPy, no modo padrão, devolvia só o prefixo lido: `A, B` virava A, e `X \to Y` virava X. Uma resposta com duas alternativas passava sempre que a primeira estava certa, em 28 de 31 casos. O G2 agora lê LaTeX em **modo estrito**, com parser próprio do adaptador; o do verify continua frouxo.
     - ±∞ casava com qualquer número, porque inf ≤ inf dá verdadeiro.
     - O número lido como coeficiente da frente de uma expressão simbólica: `2\sqrt{m/k}` passava contra 2. O gabarito agora é lido em modo estrito, e na resposta a unidade solta só vale se for unidade de fato (lista SI com prefixos), separada por espaço.
     - O lado esquerdo da resposta era descartado: `2M = …` passava, e `\omega = …` passava contra `\omega' = …`.
     - `f(0)` virava f·0 = 0.
     - δ(u) virava δ·u, linear no argumento.
     - Unidade com expoente elevava o número: `18,76\,\mathrm{m}^2` valia 351,9.
     - `\text{not } A` passava como A, e `5 N` como "No".
     - `\tan\frac{x}{4} e^{y}` era lido como tan(x·e^y/4).
   - **Falsos negativos:**
     - subscrito colado (`k_BT` é k_B·T);
     - variantes gregas (`\varepsilon` = `\epsilon`);
     - `\sqrt[n]{}`;
     - `m(g+a)^2` como m·(g+a)², não como função;
     - gabarito de definição `G = expr` (G uma grandeza): o que se corrige é o valor de G. Ele passa pelo lado direito, com G conferido, pela regra da equação ou isolando G na resposta. **Desvio declarado, para o dono decidir:** a regra da EQ sozinha (razão constante) reprovaria `E V^2 = \pi B u_0^2` contra `E = \pi B u_0^2/V^2`. A EQ que não é definição segue a regra à letra, só com razão constante;
     - desigualdade e `x \in` contra intervalo;
     - várias caixas e os separadores `;`, `and` e `\quad` nas respostas múltiplas;
     - separador de milhar, Unicode, siunitx, `\over`, `\cfrac`, `e^{número}`;
     - prosa e unidade em volta de expressão;
     - resposta em texto puro.
   - **Extração:**
     - a caixa de um problema novo inventado depois da resposta não vale;
     - "final answer is" depois da última caixa vence;
     - a frase final aceita o primeiro trecho matemático.
   - **Decisões declaradas:**
     - a unidade da resposta não é conferida, só o valor (a resposta em keV com o número certo passa contra GeV);
     - não há atalho de porcentagem (60,5% ≠ 0,605) em nenhum benchmark;
     - se o enunciado usa ε e ϵ como símbolos distintos, a distinção se perde.

8. **Segunda rodada do estresse (2026-10-20),** sobre a versão reescrita: 947 casos e 43 causas, quase todas confirmadas por verificador cético. Cada uma ganhou teste de regressão (`test_g2_corretor_estresse2.py`). As principais:
   - **Extração:** a frase "final answer is" só vence a caixa se começar depois do fim dela e trouxer de fato uma resposta; comentário com "not", "either", "correct" ou "larger than" não vale. O corte de continuação só age em cabeçalho de começo de linha, no eco do prompt e em "Solution:". "Refer to Problem 2.1" e "Question: is this consistent?" não cortam.
   - **Número:** o lado esquerdo é conferido também no caminho numérico. Qualificadores em `\text{}` ("less than", "not", "or zero") reprovam. Letra que é também símbolo de física (`30 T`, `7.617 kT`, `5.2 / g`, `m g h`) só vale como unidade se for a unidade declarada do item.
   - **Carregador do OlympiadBench:** `$a$, $b$` deixava um `$` solto em cada parte, e 4 itens de dev nunca passavam. Corrigido, e o autoteste passa a excluir qualquer gabarito com `$`.
   - **Torre de potências** (`10^{10^7}`): era calculada como inteiro (138 s) e deixava uma thread viva que contaminava os itens seguintes. Agora sai em 0,08 s.
   - **Leitura:** argumento de função sem parênteses vai até o primeiro operador (`\cos\theta \cdot \frac{ma}{M}` era cos de tudo); versores (`\hat r` = e_r); Unicode grego; `\operatorname{erf}(…)` vira pseudofunção.
   - **Decisões declaradas:**
     - coeficiente decimal arredondado numa resposta simbólica não passa (0,333·aT⁴ ≠ aT⁴/3; a tolerância simbólica é 1e-9);
     - nas respostas múltiplas, o casamento é em qualquer ordem, como no juiz oficial do OlympiadBench, e rótulos trocados não são conferidos;
     - de `X \pm δ`, vale o valor central;
     - a resposta numérica com unidade sem ambiguidade passa, mesmo que a unidade seja outra (não há conversão).

9. **Terceira rodada do estresse (2026-10-21),** dirigida ao código novo da segunda: 566 casos e 33 causas. Os dois atacantes terminaram, e os verificadores caíram no limite de sessão. Cada achado foi então reproduzido à mão antes da correção, e ganhou teste de regressão (`test_g2_corretor_estresse3.py`). A maior parte eram **regressões criadas pela segunda rodada**. O que mudou:
   - **Extração.** A frase "final answer is" é lida na sentença inteira:
     - não vale com condição, contrafactual ou pergunta antes dela ("Had we neglected recoil, the final answer is…");
     - não vale com "?" na mesma sentença, nem com correção na seguinte ("No, …");
     - o comentário depois do valor ("correct to three figures", "which is consistent") não a anula;
     - enchimento antes do valor ("therefore", "given by") não a anula;
     - com várias respostas, junta os n trechos da sentença, sem pedaço de unidade (`m/s$^2$`).

     O corte de continuação só age depois de já haver uma resposta. "Solution:", o eco parcial e "**Problem**" sem número não cortam.
   - **Número.**
     - A unidade declarada só sai do fim se sobrar um número puro: com unidade "a", `6/a` e `8 − 2a` não viram 6.
     - Lados vazios da cadeia (`\approx 13.06`) não contam. O lado do meio é comparado na tolerância do item. Uma expressão simbólica no meio (`v = \sqrt{2gh} \approx 80.4`) e `|v|`/`\text{speed}` no começo valem.
     - Texto em `\text{}` sai, salvo qualificador (lista negra: not, less, more, than, or, zero, between, at least…): unidade por extenso ("meters") e "°C" voltam a passar.
     - Unidade declarada comparada por tokens (`J \cdot s` = `J s`, `\AA` = Å). Letra com `/` ou `^` só vale de uma lista curta (m, s, K, N, J, W, V, A, C, Ω, Å): `h/g`, `g^{2}` e `ML^{2}` não passam.
     - `10^14` é 10¹⁴, não 10¹·4. Isso consertou também um gabarito de dev do OlympiadBench que o carregador lia errado.
     - Grupos SI depois da vírgula decimal (`48.189\,685`) se juntam.
     - Em `1.50 \pm 0.05 \times 10^{-3}`, a escala fica com o valor central.
   - **Leitura.**
     - A função nomeada usa uma cabeça reservada (`\operatorname`, `\mathrm{erf}`, `erf(` em texto), e não mais todo `F_…`. Assim, F_N(1+μ), F_0[…] e F_1(t) voltam a ser grandezas. A pseudofunção tem coeficientes num espaço de 10⁴ e um termo constante do nome; antes, cn = dn e f(0) = 1 para todo nome.
     - min e max são simétricos.
     - O argumento de função leva o subscrito aninhado (`\theta_{\mathrm{c}}`) e o expoente do átomo único: `\ln r^{2}` = ln(r²).
     - Os versores `\mathbf{i}`, `\hat{i}`, `\hat{x}` e `\mathbf{e}_r` = `\hat r` coincidem e não entram no argumento da função anterior.
     - Grega Unicode com índice colado (`ε0`) vira ε₀.
   - **Lados e intervalos.** `v(0)` não é `v`, e `B(a)` não é `B(r)`; só `v` contra `v(t)` dispensa os argumentos. Na desigualdade, a variável tem de ser grandeza solta: `\tan\alpha \le \mu` dá [tan α, ∞).
   - **V/F e letras.**
     - A polaridade oposta solta depois da primeira palavra reprova ("Yes; no", "No (yes for …)").
     - "Yes it is" e "No - the rule forbids it" passam.
     - No rótulo de alternativa, reprova a palavra que aponta para outra letra ("Maybe D", "or (B)"). O "not" ou "no" do texto da opção copiado não reprova.
   - **Texto em volta da resposta simbólica.** Só ", where …" e "i.e." saem, e só sem qualificador. "for", "with" e "since" ficam, e o parser estrito recusa: antes, "X for case 1, Y for case 2" passava pela primeira.
   - **Limitação declarada.** Bessel com índice (`J_{0}(kr)`) segue como produto, porque J₀·(…) também é corrente ou momento angular.

10. **Quarta rodada do estresse (2026-10-23):** 963 casos e 23 causas nos dois atacantes que terminaram (falsos positivos e extração). De novo, a maioria era regressão das rodadas 2 e 3. Cada achado ganhou teste de regressão (`test_g2_corretor_estresse4.py`). O que mudou:
   - **Decisão declarada: vale a ÚLTIMA caixa não vazia, como nos juízes oficiais do OlympiadBench e do UGPhysics.** A frase "final answer is" só é lida quando não há caixa. Nas rodadas 2 e 3, a frase depois da caixa podia vencê-la, e cada rodada achou furos nos dois sentidos: comentário hesitante vencendo a caixa certa, e correção legítima perdendo para a caixa intermediária. Com o few-shot terminando sempre em `The final answer is $\boxed{…}$`, a regra simples erra menos e erra igual nos dois braços. O custo: uma resposta que encaixota um resultado intermediário e dá o final só em prosa reprova. A auditoria das 200 gerações de dev (§3.7.3, item 3) mede quanto isso acontece.
   - **Caixa.** A caixa vazia e a de placeholder (`\boxed{}`, `\boxed{...}`, do eco do prompt) não contam. O envoltório que cobre a caixa inteira sai (`{D}`, `\textbf{…}`, `\mathbf{…}`).
   - **Corte de continuação.**
     - Só arma com uma resposta de fato: a linha canônica (frase e caixa na mesma linha), uma caixa que fecha o parágrafo, ou a frase que se lê sem caixa.
     - `Problem:` e o eco cortam sozinhos. O eco é a linha de resposta INTEIRA, como `g2_prompt` a gera, e um teste confere as duas listas: as linhas do OlympiadBench tinham deixado de casar.
     - `Question:`, `Example 3:`, `Exercise 1.2` e `Problem 2.` só cortam se depois vier `Solution:` ou um eco.
     - `Solution:` depois da resposta, com um enunciado inventado entre os dois, corta.
     - O corte é linear: uma geração degenerada com centenas de caixas abertas levava 15 s fora do teto.
   - **Frase sem caixa.**
     - Recusa só a condição que governa a oração dela ("Had we neglected recoil, …"); "Note that" e "does not depend" não recusam.
     - Aceita "≈", "E =" e rótulos "(1)" antes do valor.
     - Depois do valor, recusa outro valor ligado ao primeiro ("$13.06$ eV, or $12.75$ eV"), "if" e "unless" na sentença, e a retratação na sentença seguinte ("But…", "However…", "Oops"). "Actually, this agrees…" não conta.
     - Em texto puro, para no primeiro `$` ou em ", i.e.", e aceita `E = 13.06 eV` e uma expressão que começa por letra.
   - **Número.**
     - A cauda tem de ter a FORMA de unidade antes de ser comparada com a unidade do item. A comparação por tokens ignorava dígitos e operadores, e `130 \cdot 2 N`, `60–90 GeV` e `80.4^{2} m/s` passavam.
     - O `\text{}` volta a ser lista BRANCA, agora com as unidades por extenso. A lista negra da rodada 3 deixava sair "thousand", "negative", qualificador em chinês e símbolo em texto (`N_A`).
     - `\pm` reconhece a escala do central com `\cdot` e o `10^` nu.
     - O μ separado da letra da unidade (`$0.3\,\mu$m`) vale contra a unidade do item.
   - **Expressão.**
     - Resposta com cálculo (`\sum`, `\int`) reprova.
     - Torre de potências com expoente simbólico (3 níveis ou mais) é INCONCLUSIVE sem ser calculada; antes, estourava o teto e deixava a thread viva.
     - "whereas" não é "where", e a cláusula com desigualdade ou qualificador não sai.
     - Do outro lado da resposta, só letra grega, `\ell` ou letra solta valem como grandeza: `\arcsin\theta = X` não vale.
   - **V/F e letras.**
     - "No doubt", "No matter", "Correct answer is no", "False alarm" e "True only if" não decidem.
     - A oposta depois de "is" reprova ("Yes, the answer is no").
     - No rótulo de alternativa, a outra letra sozinha ("C. B") ou afirmada ("B is also correct", "B too") reprova.
   - **Segunda leva**, do atacante de falsos positivos refeito (300 casos):
     - `\text{upper bound} = 60` e `\text{error} = …`: o nome em texto do lado esquerdo não pode ser qualificador.
     - "Yes/No" e "Yes \& no" não decidem.
     - "Option D is also valid" reprova; o rótulo C seguido de "Option C" segue C.
     - O isolamento pela raiz confere a solução na equação da resposta: `\sqrt{G} = -X` não dá G = X².
     - Derivada não calculada na resposta (`\frac{Q\,d\sigma}{d\varepsilon}`) reprova.
     - `\not\in` não é intervalo.
     - A frase com alternativa entre as partes não decide ("$2$ and $8$ or $9$").
   - **O que ficou de fora.** O atacante de regressões caiu três vezes no limite de sessão. No lugar dele, a regressão foi conferida pela suíte das rodadas 1 a 3 (todas passam, com as duas mudanças de política anotadas) e pelo autoteste do dev contra a versão anterior. A recomendação é parar as rodadas aqui: cada rodada acha furos mais raros, e a auditoria das gerações reais de dev mede melhor o que falta.

11. **Autoteste do teste (2026-10-23, com o OK do dono).** Na primeira passada, o OlympiadBench-teste excluiu 22 de 236 itens (9,3%), acima do limite de 5%. A causa foi achada só por flags estruturais, sem olhar gabarito nem enunciado: em 16 itens Numerical, o campo `error` do benchmark vale 0. Com tolerância 0, a perturbação de 3× a tolerância é o próprio gabarito e passa; na medida, o comparador exigiria igualdade exata.
    - **Decisão declarada:** `error` 0 é tolerância não declarada, e vale o 1e-2 relativo do OlympiadBench sem `error` (`g2_itens.tolerancias_olympiad`). Vale igual para os dois braços.
    - **Resultado final** (`g2_autoteste.json`, todos abaixo do limite, sem morte nem reinício do trabalhador):

      | Benchmark | Itens | Excluídos | Taxa | Composição |
      |---|---|---|---|---|
      | UGPhysics-teste | 1.999 | 82 | 4,1% | 65 INCONCLUSIVE (cálculo), 12 perturbação que passa, 3 FAIL, 2 sem resposta |
      | OlympiadBench-teste | 236 | 5 | 2,1% | 4 INCONCLUSIVE e 1 perturbação que passa, em equações e expressões |
      | PHYBench | 100 | 1 | 1,0% | — |
      | SciBench-teste | 193 | 0 | 0% | — |

      Os ids excluídos saem dos dois braços.

12. **Iteração 1 do dev na GPU (2026-10-23, com o OK do dono).** Qwen3-1.7B-Base, k=1, 439 itens (300 do UG-dev sorteados, 115 do OlympiadBench-zh e os 24 do SciBench que não são exemplo), em 17 min na RX 7600. Agregados em `g2_dev_1.7B_base_it1.json`.

    | Benchmark | Acerto | Sem resposta | Truncado | Inconclusivo | Fichas médias |
    |---|---|---|---|---|---|
    | UGPhysics | 6,0% | 4,7% | 8,7% | 2,7% | 327 |
    | OlympiadBench-zh | 17,4% | 0% | 9,6% | 0% | 312 |
    | SciBench | 4,2% | 0% | 12,5% | 0% | 558 |

    O alvo de sem resposta (≤5%) foi cumprido. O de truncado (≤3%) não. O que se viu e o que mudou:
    - **Truncado.** Todos os 40 vieram da guarda de repetição, nenhum do teto de fichas. A maioria é laço de verdade do modelo. Pelo menos 5, porém, foram cortados no meio da caixa final: a guarda antiga cortava a janela de 32 fichas que aparecesse 4 vezes em QUALQUER ponto, e numa derivação a expressão longa aparece derivada, simplificada, retomada e posta na caixa. Isso pesa diferente conforme o estilo do braço. **Mudança:** a guarda corta a janela que volta 4 vezes em intervalos IGUAIS (±2 fichas), que é a assinatura do laço, ou 8 vezes de qualquer jeito. Simulada nas 40 truncadas com o tokenizador do GGUF: as 5 cortadas na caixa seguiriam gerando, e os laços continuam cortados, no máximo umas voltas depois.
    - **Sem resposta.** 2 dos 14 respondiam em texto, sem caixa ("The answer is True."). **Mudança:** a frase "The answer is X" (sem "final") também é lida quando não há caixa. Os outros 12 não deram resposta.
    - **Inconclusivo.** Os 8 eram itens que o autoteste já exclui (cálculo, gabarito sem leitura), mas a medida ainda os gerava e contava. **Mudança:** `medir_g2` não gera nem conta os ids excluídos. No teste, exige o autoteste da MESMA versão do corretor.
    - **FAIL.** Uma amostra de 40 FAIL foi conferida à mão: em todas, a extração pegou a resposta que o modelo deu, e a resposta estava errada. Nenhum defeito do corretor apareceu ali; a auditoria cega (item 3 do §3.7.3) mede isso com 200 gerações.

13. **Iteração 2 do dev (2026-10-24, com o OK do dono)**, com a guarda de laço, a frase "The answer is" e sem os itens excluídos. Outro sorteio de 300 do UG-dev. Agregados em `g2_dev_1.7B_base_it2.json`.

    | Benchmark | Acerto | Sem resposta | Truncado | Fichas médias |
    |---|---|---|---|---|
    | UGPhysics | 12,0% | 2,0% | 7,3% | 353 |
    | OlympiadBench-zh | 17,4% | 0% | 10,4% | 363 |
    | SciBench | 0% | 4,2% | 8,3% | 526 |

    - **Truncado.** As 36 truncadas pararam todas pela guarda nova, nenhuma pelo teto de 4.096 fichas, e nenhuma tinha resposta completa antes do corte. São laços do próprio modelo base, e não defeito do harness. Com a amostragem do protocolo (T=0,6, top-p 0,95, sem penalidades, igual nos dois braços), o alvo de ≤3% não é atingível pela base. **Decisão do dono (2026-10-24):** o laço vale 0, a amostragem fica como está, e o alvo de ≤3% passa a valer só para defeito do harness (teto de fichas, paradas, guarda). Se o ΦGen laçar menos, isso aparece como ganho real. O dono também aceitou as duas decisões declaradas desta etapa: vale a última caixa (item 10) e o `error` 0 do OlympiadBench é tolerância não declarada (item 11).

14. **Auditoria cega do corretor (2026-10-24, §3.7.3 item 3).**
    - **Amostra.** `scripts/auditar_g2.py sortear` escolheu 200 gerações de dev corrigidas (das iterações 1 e 2, sem truncadas e sem os itens excluídos), estratificadas: 130 do UG, 50 do OlympiadBench e 20 do SciBench. Elas foram corrigidas com o corretor atual (24 PASS, 167 FAIL, 9 sem resposta). Os ids, os veredictos e o SHA-256 deles foram gravados no commit `a821268`, ANTES do julgamento.
    - **Julgamento.** Dois juízes Claude independentes por lote de 10 leram só o enunciado, a linha de resposta, o gabarito, o critério e a geração, sem ver o veredito do corretor nem rodar código dele. Concordaram entre si nas 200, e não houve desempate.
    - **Resultado** (`g2_auditoria.json`): concordância de **99,5%** no PASS contra o resto (199 de 200) e **κ = 0,977**. O alvo (≥95% e κ ≥ 0,8) foi cumprido na primeira passada.
    - **O desacordo**, um só, é de notação: gabarito `v_m = \sqrt{2kT/m}` e resposta `v_p = \sqrt{2kT/M}`, a mesma velocidade mais provável com outro índice e massa maiúscula. Os juízes aprovaram, e o corretor reprova, porque compara símbolos (M ≠ m). **Limitação declarada:** o corretor não identifica símbolos renomeados. Isso vale igual nos dois braços.

15. **Congelamento (2026-10-24).** `scripts/congelar_g2.py`, depois da auditoria cumprida e com o autoteste da mesma versão do corretor.
    - **`g2_congelamento.json`** guarda o SHA-256 de:
      - o código da medida: corretor, itens, `subscritos.py`, `symbolic.py`, prompt, geração, correção isolada e `medir_g2.py`, mais as versões do sympy, antlr4 e mpmath;
      - os dados: partição, exemplos, autoteste e o envelope do IFEval, escrito agora (preâmbulo e 3 demonstrações sem restrições verificáveis);
      - o hash da configuração da geração da 1,7B (`e1535689…`).
    - **`g2.yaml`** recebeu a guarda de laço, os hashes dos exemplos e do envelope, o hash da configuração e o hash do congelamento (`230829b2…`).
    - **`PROTOCOLO_G2_SHA256`** = `daf804d7…`.
    - Daqui em diante, `medir_g2 teste` recusa se o g2.yaml, qualquer arquivo de código ou de dados do congelamento, ou uma versão de pacote mudar. Mudança só por emenda registrada aqui.

    - **Queda e retomada.** A primeira tentativa morreu às 07:31 sem erro em log nenhum, com o servidor e o Python juntos. A causa não foi achada: a outra sessão não encerrou nada, e o repositório não mata processo por nome. A retomada pulou as amostras feitas. Na agregação, a linha `erro` da requisição interrompida contava a amostra duas vezes; agora vale uma linha por amostra, e `medir_g2 dev --so-corrigir` refaz só a correção e o agregado, sem GPU.

Os defeitos do item 4 que estão no próprio `verify/` (módulo, π, carga a frio) ficam para uma correção lá, porque o verify também é a recompensa do RLVR.

**MMLU.** 5-shot com os 5 itens de dev da matéria, uma chamada por item.
- **Previsão.** A maior probabilidade entre as fichas ` A`, ` B`, ` C` e ` D`, a temperatura 0.
- **Guarda do instrumento.** O MMLU de 57 matérias da base tem de ficar a ±3 pontos de 62,63 (1,7B) e de 76,89 (8B). Fora disso, a medida para, e só se corrige bug.

**Código (EvalPlus).**
- **Geração.** Prompts oficiais para completação direta, 0-shot, k=10 a T=0,2, mais 1 gulosa. As paradas e o sanitizador são os do EvalPlus, por conjunto.
- **Acerto.** Passar em todos os testes *plus*.
- **Execução.** Num notebook de **CPU do Kaggle com a internet desligada** e sem segredos, que é uma exceção registrada ao DOC-10 §3.6. Nunca no Windows.
- **Sanidade.** A média gulosa da base fica a ±5 de 52,70 (1,7B) e de 67,65 (8B).

**IFEval.**
- **Envelope.** Texto corrido, sem fichas especiais: um preâmbulo curto e 3 demonstrações `# Query / # Answer` sem restrições verificáveis. O envelope é escrito uma vez e congelado por hash.
- **Geração.** k=16 a T=0,7, top-p 0,8 e top-k 20, mais 1 gulosa, com até 1.280 fichas.
- **Correção.** Pelo código oficial.
- **O que decide** é o *prompt-level strict* médio das amostras. O IFEval cru e guloso, como o Open LLM Leaderboard roda bases, é relatado.
- **Por que o envelope.** O template de chat mediria a familiaridade com `<|im_start|>`, que a base tem e o CPT apaga. O IFEval cru mede o hábito de emitir EOS.

#### 3.7.5 Motor e precisão

- **O motor.** É o llama.cpp `llama-server` na tag **b11159** (commit 6b790a9c2).
- **A conversão.** GGUF **Q8_0** estático, convertido por nós a partir dos safetensors da revisão fixada, com o `convert_hf_to_gguf.py` da mesma tag e sem imatrix. Nenhum GGUF de terceiros.
- **Onde roda.** O par 1,7B roda na RX 7600 local (Vulkan). O par 8B roda no Kaggle 2×T4 (CUDA, mesmo commit).
- **O hash da configuração.** Reúne commit, flags, sha256 do GGUF, GPU e imagem. Vai em todo resultado.
- **Antes do teste.** A KL média do Q8_0 contra o F16 em 50 prompts do dev tem de ficar ≤0,01 nats.
- **Tokenizer.** O do ΦGen tem de ser idêntico ao da base (sha256 `c0382117…`). Se mudar, a regra não se aplica.

#### 3.7.6 G2.1 — a regra

- **Os deltas.** Δ_b = acerto_ΦGen − acerto_base em cada benchmark (avg@k ×100). **Δ_macro** é a média simples dos 4 Δ_b, com peso 1/4 cada; a micro daria ~79% do peso ao UG.
- **O IC.** IC 95% por **bootstrap pareado, estratificado por benchmark**, reamostrando agrupamentos (os grupos de 13-grama no UG, os 113 do Oly, os itens no PHYBench e no SciBench). São 10.000 réplicas com a semente 20261019.
- **Desvio declarado.** Este IC da diferença pareada substitui o critério de "ICs sobrepostos" do §3.5.

**O G2.1 PASSA só com as três condições:**
1. o limite de baixo do IC 95% de Δ_macro é **≥ +10,0**;
2. nenhum dos quatro Δ_b tem o IC **inteiro abaixo de 0**;
3. sem os itens de **memorização nova** (sinal S1 ou S2 no ΦGen e não na base, ver §3.7.9), o limite de baixo continua ≥ +10,0.

Fora disso, NÃO PASSA. A leitura é "demonstrado abaixo" se o IC fica inteiro abaixo de +10, ou "não demonstrado" se o IC atravessa +10. Se mais da metade do ganho vier de itens em que a base não tinha resposta extraível, a publicação diz **"ganho de formato"**.

**Poder**, por simulação com premissas não medidas, que serão recalculadas com a base medida sem mudar a regra:
- o erro-padrão do macro fica entre 0,72 e 0,92 ponto;
- a regra passa com 80% de chance a partir de +12,0 a +12,6 de ganho verdadeiro, e em 2,5% das vezes com +10 exato;
- com decodificação gulosa, o erro-padrão seria 1,57–1,83 ponto.

#### 3.7.7 G2.3 — a regra

Δ = ΦGen − base, em pontos, com IC 95% por bootstrap pareado sobre os itens:

| medida | regra |
|---|---|
| MMLU sem física (13.402) | **não-inferioridade demonstrada**: passa só se o limite de baixo ≥ −2,0 |
| EvalPlus (542), pass@1 = média de c/10 | **desclassifica** se Δ̂ < −2,0 **e** o IC inteiro abaixo de 0 |
| IFEval no envelope (541), strict médio de 16 | **desclassifica** se Δ̂ < −2,0 **e** o IC inteiro abaixo de 0 |

- **Uma basta.** Uma medida que falhe desclassifica sozinha (DOC-19 R-3), e por isso não há correção de multiplicidade.
- **A assimetria é declarada.** A não-inferioridade nas três reprovaria entre 15% e 50% dos ΦGen sem regressão nenhuma. Por simulação:
  - a forma escolhida desclassifica um ΦGen sem regressão em 0,3–1,4% das vezes por medida;
  - ela pega uma regressão verdadeira de −3 em 87–92%;
  - uma de −2 passa metade das vezes.
- **Durante o treino**, a "regressão geral < 2" do DOC-08 §5.3 é medida só no dev (MMLU validation e MBPP fora do MBPP+).

#### 3.7.8 G2.2 — a regra (só na 8B)

**Forma restrita.** O ΦGen-8B é comparado com **bases abertas ≤32B no mesmo protocolo de base**.
- **Números publicados** nunca são comparador: vêm de outros protocolos e servem só de contexto.
- **Comparadores**, lista fechada, Apache-2.0 e sem termos a aceitar:
  - **Qwen3-14B-Base**;
  - **Qwen3-30B-A3B-Base**;
  - **Qwen2.5-32B**.

  Um modelo *gated* só entra se o dono aceitar os termos antes de medir o ΦGen-8B.
- **Modelos instruídos e de raciocínio** ≤32B **não são medidos**, e isso é declarado.

**Itens.** A subamostra fixa de 1.004 do UG-teste (`teste_g2_2`) e os testes inteiros dos outros três. Os comparadores rodam com k=1 no UG, 2 no Oly e no Sci, e 4 no PHY. O ΦGen reaproveita as próprias saídas do G2.1.

**A regra.**
- **Vitória por benchmark.** O ΦGen vence o benchmark b se, contra **cada** comparador, o **IC 99%** da diferença pareada fica inteiro acima de 0. O 99% é Bonferroni sobre os 5 benchmarks.
- **Comparador quantizado.** Contra um comparador medido abaixo de Q8_0, o IC tem de ficar acima de +1,0.
- **Aprovação.** O G2.2 **passa com ≥3 vitórias entre os 5**.
- **Custo estimado:** ~20 h de Kaggle, só depois de existir o ΦGen-8B congelado.

#### 3.7.9 Contaminação

**(A) A base não é descontaminável.** O G2.1 é um delta, e a contaminação herdada se cancela se o CPT não a ampliar. Duas sondas gulosas, sem exemplos, rodam nos 2.528 itens de física:
- **S1:** o modelo recebe a primeira metade do enunciado e gera 48 fichas;
- **S2:** o modelo recebe o enunciado e 40% da solução e gera 64 fichas.

O item é suspeito se houver ≥1 sequência de 13 fichas idêntica à verdadeira. O PHYBench, publicado depois dos dados da base, é o nulo. A lista de ids suspeitos é lacrada antes do treino.

**(B) O corpus do CPT — PRÉ-CONDIÇÃO.** Sem ela, o ΦGen não é medido. O corpus final é varrido contra **todos** os itens do G2 (dev, teste, reserva e exemplos). Sai o documento que:
- tiver ≥1 13-grama de palavras de um enunciado ou de uma solução. Frase-feita é o 13-grama presente em ≥3 itens de benchmark ou em >1.000 documentos;
- contiver a forma canônica de um gabarito com ≥8 fichas.

O vetor semântico só é relatado. As remoções são publicadas por benchmark.

**(C) Depois do treino.** As sondas S1 e S2 são refeitas no ΦGen. Item com sinal no ΦGen e sem sinal na base é **memorização nova**, e alimenta a condição 3 do G2.1.

#### 3.7.10 Limites declarados

- **Pessoas.** Não há físicos nem revisão humana. O corretor é por regras, auditado pelo Claude no dev. O §6 deste documento e o DOC-00 §6.6 não são cumpridos, e nada se alega além das métricas medidas.
- **Recortes dos benchmarks.**
  - UGPhysics: sem o juiz GPT-4o, só itens de resposta única, 1.999 dos 5.520.
  - PHYBench: 100 dos 500.
  - OlympiadBench: só o recorte de texto em inglês.
  - SciBench: 193 itens.

  Nenhum desses números se compara aos artigos.
- **Tamanho e privacidade do teste.**
  - Não há teste privado (§7); a partição, o lacre, o contador e a reserva fazem as vezes dele.
  - O DOC-11 §8.2 (≥1.000 itens por tarefa) não é cumprido por Oly, PHY e Sci isolados; o agregado tem 2.528 itens.
- **Desvios do protocolo do §3:**
  - Q8_0 em vez de BF16;
  - avg@k a T=0,6 como métrica principal, no lugar do pass@1 guloso do §3.2;
  - INCONCLUSIVE conta como erro, desvio do §4.1;
  - o IFEval roda num envelope do projeto.
- **G2.3 e G2.2.** O G2.3 é assimétrico. O G2.2 fica restrito a 3 bases abertas e, sem GPQA, vira 3 de 4.
- **Portão anterior.** O G1 não foi vencido (G1.3 impossível pelo ADR-0003, G1.4 adiado, G1.5 parcial). O Tier 2 não começa sem um ADR que o autorize ou que revise o G1.
- **Riscos maiores:**
  - o corretor é código novo e é o elo fraco;
  - a vazão não foi medida;
  - a 1,7B pode ficar no piso no PHYBench;
  - o agrupamento por 13-grama não pega irmãos parafraseados;
  - é provável que o G2.2 falhe contra o Qwen3-30B-A3B-Base e o Qwen2.5-32B, e o negativo é publicado.

#### 3.7.11 Verificar antes de congelar

1. **Vazão** (`llama-batched-bench`, local e no Kaggle). Abaixo da metade da suposta, o dono decide os cortes antes da base: primeiro o IFEval de k=16 para k=8, depois o UG de k=3 para k=1.
2. **Memória da RX 7600** com `-np 4` × 10.240 e KV q8_0, e flash attention com KV q8_0 nos dois builds.
3. **MMLU.** O argmax das 4 letras pelo `n_probs` do servidor tem de bater com o do transformers na CPU em ~50 itens do dev.
4. **KL** do Q8_0 contra o F16 ≤0,01 nats.
5. **Autoteste do gabarito.** Exclusão ≤5% por benchmark; acima disso, conserta-se o parser.
6. **Extrator e auditoria no dev**, com os alvos do §3.7.3, e as fichas médias medidas contra as premissas.
7. **Troca.** A taxa de troca em 200 itens do dev rodados duas vezes.
8. **Sanidade da 8B-base** em 300 itens do UG-dev, antes do teste.
9. **Código e IFEval:**
   - a sobreposição MBPP × MBPP+ por `task_id`;
   - as canônicas do EvalPlus no notebook sem internet;
   - o envelope em 20 instruções de fora.
10. **Partição e exemplos.** A partição e os exemplos são gravados com hash. Nenhum exemplo pode ser irmão de item de teste.
11. **Kaggle:**
    - o 2×T4 consome a cota como uma sessão só;
    - a sessão de 12 h em segundo plano retoma por item;
    - o binário guardado como dataset roda sem recompilar.

---

## 4. Correção

### 4.1 Verificável

Delegada ao `verify/` (DOC-10). Três regras:

1. **Equivalência de CAS, nunca casamento de string** (DOC-11 §4).
2. **`INCONCLUSIVE` sai do denominador** e é reportado separadamente — jamais contado como erro do modelo.
3. **Verificador reservado** (DOC-09 §5.4) roda em paralelo; divergência entre ele e o de treino é alerta de reward hacking.

### 4.2 Por juiz LLM

Só para o que não é verificável (`PB-Concept`, qualidade de explicação).

| Regra | Valor |
|---|---|
| Juízes | ≥ 3 modelos distintos |
| Rubrica | Explícita, com âncoras, escrita por física |
| Ordem | Randomizada; posições embaralhadas contra viés de posição |
| Cegamento | O juiz não sabe qual modelo produziu qual resposta |
| **Auto-preferência** | ★ **O ΦGen nunca julga a si mesmo, nem a outro ΦGen** |
| Calibração | Concordância com físicos humanos medida em 200 itens e **publicada** |

O viés de auto-preferência é documentado e forte: modelos preferem o próprio estilo. Usar o ΦGen como juiz do ΦGen inflaria os resultados de forma invisível.

---

## 5. Descontaminação e o relatório que a acompanha

Executada pelos três vetores do DOC-04 §6.2 — n-grama, embedding e **forma canônica de equação** — antes de qualquer alegação.

**O relatório de contaminação é publicado junto com os resultados**, contendo por benchmark: taxa de contaminação, número de itens removidos, e resultados **com e sem** os itens contaminados. Se a diferença entre as duas versões for grande, isso é a informação mais importante da tabela.

**A limitação herdada permanece declarada:** não é possível descontaminar o Qwen3-8B-Base. Por isso o critério **G2.1** — delta contra o próprio modelo base — é a métrica de manchete (DOC-04 §6.4).

---

## 6. Avaliação humana

O critério 6 do DOC-00 §6 exige físico no laço para qualquer alegação de capacidade.

| Elemento | Especificação |
|---|---|
| Amostra | 300 respostas, estratificadas por tarefa e subárea |
| Avaliadores | ≥ 3 físicos, com pelo menos um experimental |
| Cegamento | Duplo — avaliador não sabe o modelo; a ordem é randomizada |
| Instrumento | Rubrica com âncoras + campo livre para erros de Física |
| Concordância | **Alpha de Krippendorff** reportado; abaixo de 0,6 invalida a rodada |
| Portão | Nenhuma alegação de capacidade em Física é publicada sem esta etapa |

O campo livre é o mais valioso: erros que nenhuma métrica automática captura — uma hipótese física implícita e falsa, um raciocínio circular, uma resposta certa por motivo errado.

---

## 7. Avaliação durante o treino vs. avaliação final

| | Durante o treino | Final |
|---|---|---|
| Conjunto | **Público de dev** | **Privado de teste** |
| Frequência | A cada 1.000–5.000 passos | Em marcos |
| Custo | Subconjunto barato | Suíte completa |
| Uso | Diagnóstico, ajuste de mistura | **Alegações** |
| Contabilizado no contador do DOC-11 §8.3 | Não | **Sim** |

**Nunca ajustar hiperparâmetros olhando o conjunto privado.** Fazê-lo o converte em conjunto de validação e destrói sua função — silenciosamente, porque nenhum item vazou.

---

## 8. Reprodutibilidade de um resultado

Todo relatório carrega um **manifesto de reprodução**:

```yaml
result_id: blake3:9f2a...
preregistration: blake3:3c81...      # o pré-registro que precedeu a execução
model:      phifm-gen-1p5b-rlvr-a3f21c04-9b7e0d12-012000
code:       git 7a8c85b
harness:    lm-eval-harness 0.4.x + phifm.eval 0.1.0
benchmark:  PhysBench-v2026.2 (privado)
decoding:   {temperature: 0.0, max_tokens: 4096}
seeds:      [17, 42, 1337]
verifier:   phifm.verify 0.3.1
hardware:   1x H100 80GB SXM
command:    make eval EXPERIMENT=...
```

Um terceiro reproduz ou refuta a partir disso — princípio **P7** do DOC-01. O `verifier` versionado é essencial: uma mudança no barramento pode alterar escores, e resultados antigos nunca são silenciosamente reinterpretados (DOC-10 §10).

---

## 9. O formato padrão de relatório

Toda tabela publicada segue este formato. Exemplo ilustrativo:

| Modelo | PB-Verify | PB-Dim | GPQA-física | MMLU (regressão) |
|---|---|---|---|---|
| Qwen3-8B-Base *(baseline)* | 31,2 ±3,0 | 48,7 ±3,1 | 39,4 ±3,4 | 71,2 ±1,1 |
| **ΦGen-8B** | **52,8 ±3,2** | **74,1 ±2,7** | **48,9 ±3,4** | 70,6 ±1,1 |
| **Δ vs. base** | **+21,6** [17,2; 26,0] | **+25,4** [21,1; 29,7] | **+9,5** [4,8; 14,2] | **−0,6** [−1,9; +0,7] |
| Modelo geral forte ≤32B | 44,1 ±3,2 | 61,3 ±3,0 | 51,2 ±3,4 | 76,8 ±1,0 |

Quatro coisas que este formato força:

1. **A linha de delta contra a base é a manchete** — G2.1.
2. **A coluna de regressão está sempre visível** — G2.3 não pode ser esquecido.
3. **O baseline geral forte aparece**, mesmo quando nos vence (aqui vence em GPQA e MMLU) — regra 3 do DOC-00 §6.
4. **Todo número tem incerteza.**

> Note o exemplo: o modelo geral vence em GPQA-física. **Reportar isso é obrigatório.** Uma tabela em que o nosso modelo vence em tudo seria sinal de que os baselines foram escolhidos a dedo.

---

## 10. Custo

| Item | Custo |
|---|---|
| Execução da suíte completa (1 modelo) | ~US$ 8 |
| Baselines (5 modelos × suíte) | ~US$ 40 |
| Juízes LLM (calibração + `PB-Concept`) | ~US$ 25 |
| Avaliação contínua durante treinos | ~US$ 25 |
| **Total em computação** | **~US$ 100** |
| Avaliação humana | Tempo de especialista |

---

## 11. Riscos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Escolha post hoc da métrica favorável | **Alta** | **Crítico para credibilidade** | Pré-registro versionado (§3.6) |
| Conjunto privado vira validação por uso repetido | Média | **Alto** | Contador publicado; §7 |
| Baselines fracos inflam nossos resultados | Média | **Alto** | Modelo geral forte obrigatório na tabela (§9) |
| Juiz LLM com viés de auto-preferência | **Alta** | Alto | ΦGen nunca julga ΦGen (§4.2) |
| Concordância humana baixa invalida `PB-Concept` | **Alta** | Médio | Krippendorff < 0,6 invalida a rodada; reportado como fraco |
| Bug no verificador afeta treino e avaliação juntos | Média | **Crítico** | Verificador reservado; DOC-10 §12 |

---

## 12. Critérios de aceite do Stage-Gate 11

- [ ] **L1** — Nenhum número publicado sem IC, `n`, sementes e versão do conjunto
- [ ] **L2** — Pré-registro implementado; CI rejeita relatório sem hash de pré-registro
- [ ] **L3** — Testes pareados e correção FDR implementados e usados
- [ ] **L4** — Corretor é literalmente `verify/`; verificado por teste de importação
- [ ] **L5** — Relatório de contaminação gerado automaticamente com todo resultado
- [ ] **L6** — Protocolo de avaliação humana executado, com Krippendorff ≥ 0,6
- [ ] **L7** — Manifesto de reprodução emitido para todo resultado; reprodução verificada por terceiro
- [ ] **L8** — Contador de avaliações contra o conjunto privado ativo e publicado
- [ ] **L9** — Tabela padrão inclui baseline geral forte, mesmo quando ele vence

---

## 13. Questões em aberto

| ID | Questão | Destino |
|---|---|---|
| OQ-47 | `avg@8` ou `pass@1` como métrica principal para modelos de raciocínio? | Fixar antes do primeiro pré-registro |
| OQ-48 | Krippendorff ≥ 0,6 é limiar adequado para julgamento de Física? | Calibrar com a primeira rodada humana |
| OQ-49 | Publicar um leaderboard público exigiria infraestrutura de submissão? | DOC-18 — decisão de release |

---

## 14. Referências

1. Dietterich, T. (1998). *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms.* Neural Computation.
2. Benjamini, Y., Hochberg, Y. (1995). *Controlling the False Discovery Rate.* JRSS-B.
3. Efron, B., Tibshirani, R. (1994). *An Introduction to the Bootstrap.* Chapman & Hall.
4. Dwork, C. et al. (2015). *The reusable holdout.* Science 349.
5. Krippendorff, K. (2004). *Content Analysis: An Introduction to Its Methodology.*
6. Zheng, L. et al. (2023). *Judging LLM-as-a-Judge.* NeurIPS.
7. Panickssery, A. et al. (2024). *LLM Evaluators Recognize and Favor Their Own Generations.* NeurIPS.
8. Gao, L. et al. (2023). *A framework for few-shot language model evaluation* (lm-evaluation-harness).

---

**Fim do DOC-12.** Encerra a Fase 3.
