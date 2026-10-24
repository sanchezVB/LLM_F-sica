# ADR-0005 — Resposta ilegível e condição necessária na recompensa do RLVR

**Status:** **Aceito (2026-10-23)** pelo dono do projeto: a opção O6 para a recompensa e a correção da álgebra no próprio barramento, reescrevendo a golden que congelava o comportamento antigo.
**Contexto:** [DOC-10 §2.1 e §4](../02-models/DOC-10-raciocinio-verificacao-ferramentas.md) (o neutro e a álgebra de resultados), [DOC-09 §5.1 e §5.4](../02-models/DOC-09-pos-treino-sft-dpo-rlvr.md) (componentes da recompensa e reward hacking), [DOC-01 §1, P3](../00-foundations/DOC-01-system-architecture.md) (a recompensa de treino não diverge do corretor de avaliação).
**Revisa:** o que o DOC-10 §2.1 chama de neutro, a álgebra do DOC-10 §4, o R1 do DOC-09 §5.1 e a tabela do DOC-09 §5.4.

---

## 1. O problema

Uma resposta que o `parse` não lê dá INCONCLUSIVE em todos os verificadores, o barramento devolve INCONCLUSIVE, e o `REWARD` dá 0. Uma resposta errada vale −0,95. Medido no barramento padrão, com o gabarito `\frac{1}{2} m v^{2}`:

| resposta | veredito | recompensa |
|---|---|---:|
| certa | PASS | +0,77 |
| errada (`m v^{2}`) | FAIL | −0,95 |
| errada com `\,\text{J}`, com `]` no fim, com `\vec` ou com `\approx` | INCONCLUSIVE | **0** |
| errada com `\text{nao sei}` | INCONCLUSIVE | **0** |

E dois atalhos maiores, que não passam pela ilegibilidade, quando o contexto declara SI:

| resposta | vetor dos verificadores | barramento | recompensa |
|---|---|---|---:|
| errada com a dimensão certa (fator 2, sinal) | simbólico FAIL, numérico FAIL, dimensional PASS | ERROR (discordância) | **0** |
| errada `m v^{2} f(1)` | simbólico e numérico INCONCLUSIVE, dimensional PASS | **PASS** | **+0,67** |

O erro mais comum de um modelo (fator ou sinal trocado) saía de graça, e um PASS de condição necessária passava por equivalência.

## 2. Por que o tamanho da penalidade não importa

O GRPO (DOC-09 §5.3, grupos de 8) normaliza a vantagem dentro do grupo. Num grupo com 7 respostas erradas (−0,95) e uma com o atalho, a vantagem do atalho é **+√7 ≈ +2,65 para qualquer valor acima de −0,95**: 0 e −0,5 dão o mesmo número, igual ao de uma resposta certa. Só o sinal de (atalho − errada) conta. Uma penalidade constante por lote some na vantagem. **O atalho só fecha se a abstenção causada pela resposta valer exatamente um FAIL.**

E o 0 do DOC-10 §2.1 não é neutro no GRPO: num grupo de FAIL ele é o melhor da turma. O neutro de verdade é tirar a amostra do grupo.

## 3. As opções

| opção | fecha o atalho? | custo |
|---|---|---|
| O1 manter | não | — |
| O2 os verificadores dão FAIL para resposta ilegível | só a classe ilegível | quebra a golden e contamina o filtro de dados e a auto-checagem |
| O3 motivo estruturado e −1 só no RLVR | só a classe ilegível | no `parse` de hoje, −1 para 29% das notações certas testadas |
| O4 penalidade −0,5 | não (§2) | — |
| O5 só monitorar | não (§2) | — |
| **O6** O3 completada pelas medidas abaixo | **sim, inclusive os atalhos legíveis** | trabalho antes de ligar |

A comparação foi feita com três revisões independentes (atacante, vítima e arquitetura), e as três chegaram à O6.

## 4. O desenho

### 4.1 Dois tipos de afirmação

O barramento recebia do mesmo jeito duas coisas diferentes:

- **equivalência** — uma resposta contra um gabarito (RLVR, correção de benchmark). Os verificadores de equivalência decidem.
- **relação** — uma equação física, como `E = m c^2` (filtro de dados, auto-checagem). Comparar `E` e `m c²` como símbolos independentes não tem sentido, e o simbólico os reprovava.

`Claim.context["tipo"]` vale `"equivalencia"` (o padrão quando há `rhs`) ou `"relacao"`. Numa relação, o simbólico e o numérico não se aplicam, e as condições necessárias decidem.

### 4.2 Dois papéis de verificador

| papel | verificadores | o que um PASS prova |
|---|---|---|
| **equivalência** | simbólico, numérico | que a resposta é o gabarito |
| **condição necessária** | dimensional, limites, conservação | só que a resposta não viola aquela condição |

### 4.3 A álgebra (DOC-10 §4)

Em ordem, a primeira regra que casa decide:

| caso | resultado |
|---|---|
| verificadores de **equivalência** discordam (PASS e FAIL) | ERROR, discordância: há bug num deles |
| equivalência PASS e condição necessária FAIL | ERROR, contradição: o gabarito ou a tabela de dimensões está errado |
| qualquer FAIL (de qualquer papel) | **FAIL** — condição necessária aprovada não contradiz FAIL de equivalência |
| equivalência PASS | PASS (a confiança segue a regra de antes) |
| só condições necessárias decidiram PASS, **e havia verificador de equivalência aplicável** | **INCONCLUSIVE** — condição necessária não prova equivalência |
| só condições necessárias, sem verificador de equivalência aplicável (relação) | PASS |
| só ERROR | ERROR |
| o resto | INCONCLUSIVE |

"FAIL" é sempre FAIL de alta confiança (`high_confidence`, 0,9), como o DOC-10 §4 já pedia. Abaixo disso, o FAIL conta como abstenção: a conservação que só não conseguiu zerar a derivada (0,85) não prova nada. A função é `verify.bus.classificar`, e a recompensa do RLVR usa a mesma, para as duas nunca divergirem. O papel é atributo de classe de cada verificador; um verificador sem papel declarado é tratado como condição necessária e nunca rende +1.

### 4.4 Motivo estruturado

`VerificationResult.motivo` diz por que um verificador não decidiu. O veredito não muda; a recompensa lê o motivo. `VerifierBus.run` preenche `sem_motivo` em toda abstenção que chegar sem motivo, e uma regressão exige motivo em todas.

| motivo | quem causa | na recompensa do RLVR |
|---|---|---|
| `resposta_ilegivel` | a resposta: o `parse` não a lê (inclusive sem `\boxed{}`) | −1 |
| `resposta_incomparavel` | a resposta: relação, tupla ou booleano contra uma expressão | −1 |
| `resposta_indefinida` | a resposta: o gabarito avalia nos pontos sorteados e ela não (`/0`, `0^{-m}`) | −1 |
| `vocabulario` | a resposta: símbolo ou função que não estão no gabarito nem no contexto | −1 |
| `operadores_na_resposta` | a resposta: marcador de operador que o gabarito e o contexto não têm | −1 |
| `excecao_na_resposta` | a resposta: um verificador de equivalência quebrou com ela | −1 |
| `tempo_na_resposta` | a resposta: o gabarito avaliou e ela estourou o teto | −1 |
| `gabarito_ilegivel`, `gabarito_incomparavel` | o item | fora do grupo, com alerta |
| `operadores` | o item: o gabarito ou o contexto declaram operadores | fora do grupo, com alerta |
| `sem_convencao` | o item: falta sistema de unidades ou dimensão declarada | fora do grupo |
| `tempo`, `indecidivel` | o problema | fora do grupo |
| `discordancia`, `contradicao` | o barramento | fora do grupo, com alerta |
| `excecao`, `sem_motivo` | defeito de verificador | fora do grupo, com alerta |

Dois cuidados que a revisão do desenho tornou obrigatórios:

- **A aplicabilidade é função só do item.** O numérico se declarava inaplicável quando a RESPOSTA tinha um marcador de operador, e sumia do vetor sem deixar motivo. Agora ele fica e se abstém com `operadores_na_resposta`.
- **Função com nome de grandeza é produto.** `m(g + a)`, com `m` símbolo do gabarito, é m·(g + a). Lido como função, abria o atalho `m v^{2} m(1)`.

### 4.5 O contrato de resposta (`verify/resposta.py`)

Lê a resposta como o prompt pede, antes de ela ir ao barramento. Fechado na falha: o que ele não cobre segue como está.

- Lê a ÚLTIMA `\boxed{}` de chaves balanceadas; caixa dentro de caixa lê a de dentro. Sem caixa, a resposta não existe (−1).
- Tira `$…$`, `\(…\)`, `\[…\]` e a pontuação do fim, até estabilizar.
- Lê só o lado direito de `grandeza = expressão` e de `grandeza \approx expressão`, quando há exatamente dois lados de nível zero e o esquerdo é uma grandeza solta (uma letra ou uma grega, com subscrito e linha opcionais). `\approx` no início também sai.
- **Não** tira unidade (decisão do dono de 2026-10-19: resposta com unidade dá `None`, e o prompt a pede sem unidade), **não** tira decoração (`\vec`, `\hat`) e **não** lê cadeias (`K = A = B`), `\propto` ou `\sim`.

O gabarito passa pelo mesmo contrato, sem exigir a caixa.

### 4.6 A recompensa do RLVR (`verify/recompensa.py`)

`recompensa(resultados)` lê o vetor de `VerifierBus.run`, não o veredito combinado. A primeira linha que casa decide:

| | caso | valor |
|---|---|---|
| 1 | nenhum verificador aplicável | `None`, alerta |
| 2 | discordância entre equivalências | `None`, alerta |
| 3 | contradição (equivalência PASS, condição FAIL) | `None`, alerta |
| 4 | qualquer FAIL decisivo | **−1** |
| 5 | equivalência PASS | **+1** |
| 6 | motivo do item vindo de um verificador de EQUIVALÊNCIA | `None`, alerta |
| 7 | motivo da resposta, de qualquer verificador | **−1** |
| 8 | `sem_motivo` ou `excecao` | `None`, alerta |
| 9 | o resto (tempo, indecidível, sem convenção) | `None` |

A linha 6 só olha a equivalência porque o dimensional sem sistema de unidades dá `sem_convencao` em todo item; se esse motivo vencesse a linha 7, o atalho da resposta ilegível voltava em todo item sem `unit_system`. O −1 não multiplica pela confiança: com −0,95 para um FAIL e −1 para outro, sete erradas a −0,95 ganhariam vantagem cheia contra uma a −1.

**`None` é contrato com o laço do GRPO**: a amostra sai inteira (perda, média, desvio e KL), e um grupo com menos de duas amostras pontuadas sai do passo.

**`avaliar(texto, gabarito, contexto)`** junta o contrato e o barramento. **`admitir(gabarito, contexto)`** decide se o item entra no RLVR: o gabarito tem de passar contra si mesmo com +1, **e o numérico tem de aprovar**. O PASS estrutural do simbólico, que o gabarito sempre tem contra si mesmo, não basta; sem o numérico, entravam itens em que só o `equals` reprovava, e toda resposta errada que o deixasse indeciso saía do grupo.

Isso é o **R1** do DOC-09 §5.1, agora a equivalência do simbólico e do numérico juntos (o R4 deixa de contar à parte). Quem montar a soma ponderada R1…R7 tem de dar à resposta cujo R1 é −1 por motivo da resposta o **pior** valor em cada componente aplicável.

### 4.7 Leituras corrigidas

No verify, para os cinco consumidores:

- **Equação contra expressão** dá `resposta_incomparavel`. O numérico comparava o booleano de dois `Eq` e aprovava qualquer par de equações; o simbólico relatava o `TypeError` como estouro de tempo. Exceção e tempo são motivos diferentes.
- **Unicode** (`½mv²`, `ω₀`, `ħ`, `×`) vira LaTeX, para ir à rota do ANTLR, que separa `m v`. O que sobrar fora do ASCII dá `None`.
- **Variantes gregas** (`\varepsilon` e `\epsilon`, e as de θ, φ, ρ e κ) são a mesma letra, no texto e nas chaves do contexto. `\varpi` fica de fora: viraria a constante π.
- **Números absurdos** (torre de expoentes inteiros, expoente de cinco dígitos, sessenta dígitos) dão `None` antes de qualquer avaliação. O numérico não tinha teto, e a torre ocupava gigabytes.
- **O numérico** avalia cada lado sob o seu próprio teto. Perdeu o piso absoluto de 1 na tolerância, com o qual `0` passava contra 1,87·10⁻⁶⁴. A formatação do contraexemplo levantava exceção, e a 1ª discordância contava como ponto indefinido.
- **O simbólico** semeia o RNG do SymPy antes do `equals`: o mesmo Claim dava INCONCLUSIVE numa chamada e FAIL na seguinte. Com operadores no item, não aprova pela comparação estrutural (`A*B` e `B*A` saíam iguais na rota do `parse_expr`). O contraexemplo não roda mais um `simplify` sem teto.
- **O dimensional** dá a `|v|` (e ao conjugado, à parte real e à imaginária) a dimensão do argumento, e trata função indefinida como indeterminada, e não como adimensional.
- **Os limites e a conservação** só dão FAIL quando houve o que reprovar: limite sem valor obtido, ou derivada não nula só porque o `simplify` não a zerou, viram abstenção.
- **Os marcadores de subscrito** são desfeitos também em nomes de função (`E_0(x)`).

## 5. O que muda e o que não muda

- `REWARD` continua como está, para os consumidores que leem o veredito combinado.
- A golden `test_expressao_ininteligivel_nao_vira_falha` continua valendo: o barramento dá INCONCLUSIVE para resposta ilegível. Quem dá −1 é a recompensa do RLVR.
- **Como entra no G2, por decisão do dono (2026-10-24): emenda registrada.** O protocolo do G2 foi congelado em `5bde25c` com o SHA-256 de `verify/symbolic.py`. Este trabalho só é mesclado depois de: a medida da base no teste terminar; a sessão do G2 medir no dev quantos veredictos mudam com o `symbolic.py` novo; o autoteste ser refeito; e as gerações já gravadas da base serem recorrigidas na CPU com o corretor novo, com o mesmo hash de geração. Assim a base e o ΦGen ficam corrigidos pelo mesmo corretor. O registro da emenda no DOC-12 é da sessão do G2.
- **O corretor do G2 não é editado, mas muda pelo `parse`.** Ele usa o `parse` do verify como último recurso para texto puro e o `normalizar_subscritos` no leitor estrito. As leituras do §4.7 (Unicode, variantes, números absurdos) podem mudar vereditos dele, e o hash do corretor (`g2.codigo_do_corretor()`, que cobre `verify/symbolic.py`) muda. O autoteste do dev tem de ser refeito depois do merge. O G2 mantém o motor de comparação próprio: é o verificador reservado do DOC-09 §5.4. Ele já dá FAIL para resposta ilegível, e nessa classe o RLVR passa a concordar com ele; para resposta certa com unidade ou decoração, o G2 lê e o RLVR dá −1, de propósito.

## 6. Testes

- **Reescrita:** `tests/golden/test_verifiers_fisica.py::test_discordancia_entre_verificadores_vira_erro_nao_voto`, que exigia ERROR para `m·v²` contra `F·d` em SI, só porque o dimensional aprovava. Passa a exigir FAIL, com o nome `test_condicao_necessaria_aprovada_nao_contradiz_falha_de_equivalencia`. Entram ao lado o erro de fator com a dimensão certa (FAIL), `m v² f(1)` em SI (não PASS) e a relação `F = m a` com `tipo = "relacao"` (PASS) contra `F = m a²` (FAIL).
- **Atualizada:** `tests/regression/test_parse_latex_fisica.py`, em que `\varepsilon_{0}` passa a ler `epsilon_0`.
- **Nova golden:** `tests/golden/test_recompensa_rlvr.py`, com a tabela do §4.6 linha a linha, cada motivo da resposta valendo −1, os atalhos do §1 de ponta a ponta (com e sem SI), as respostas certas valendo +1 e a admissão do item.
- **Nova regressão:** `tests/regression/test_adr_0005.py`, um teste por defeito do §4.7, mais a exigência de motivo em toda abstenção.
- A discordância verdadeira, entre dois verificadores de equivalência, segue congelada em `tests/golden/test_verifier_bus.py::test_discordancia_nunca_e_resolvida_por_voto`.

## 7. Riscos que ficam

- **Tempo do problema.** Uma resposta legível que faça o `simplify` estourar sai do grupo em vez de levar −1, quando o numérico também não decide. É o preço do DOC-10 §2.1. Os números absurdos, que eram a forma barata de provocá-lo, dão −1; a taxa por motivo é monitorada.
- **Domínio do numérico.** Ele amostra só em reais positivos (decisão de 2026-08-03, contra cortes de ramo). Uma resposta que só difere do gabarito no sinal (`|q| E` contra `q E`) tem o simbólico em FAIL e o numérico em PASS: discordância, fora do grupo. Declarar símbolos com sinal no contexto é a correção natural, e fica para depois.
- **Itens fora do RLVR.** O regex de operadores pega `\sigma` e `\gamma^`, e um gabarito que é equação não se compara: esses itens não são admitidos. É perda de cobertura, não de correção; medir quanto se perde.
- **Formato.** Até o SFT ensinar o formato, resposta certa com unidade, grau ou `±` leva −1. A amostragem por rejeição do SFT (DOC-09 §3) já seleciona o formato legível antes do RLVR.
- **Abstenção do R7.** `\text{não sei}` é resposta ilegível (−1). O R7 positivo só pode existir em item rotulado para abstenção, com o formato de abstenção reconhecido antes do contrato; fazer isso sem reabrir o atalho do §2 é trabalho de quem montar o R7.
- **Contradição por forma incoerente.** `\frac12 m v^2 + 0 \cdot m` e `\ln x - \ln y` passam na equivalência e reprovam no dimensional: contradição, fora do grupo, com alerta. Não ajuda resposta errada, mas tira o +1 de algumas certas.
- **Contrato compartilhado.** Um atalho que explore o próprio contrato ficaria invisível a quem usa o mesmo contrato. O G2 tem leitura própria, e a divergência entre a recompensa e o G2 deve ser medida no dev, por motivo, antes do RLVR.
