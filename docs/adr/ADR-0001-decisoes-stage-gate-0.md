# ADR-0001 — Decisões do Stage-Gate 0 e postura jurídica do corpus

**Status:** Aceito (2026-08-03) — com uma ressalva material que exige confirmação do sponsor (§4)
**Contexto:** [DOC-00 §9](../00-foundations/DOC-00-project-charter.md#9-stage-gate-0--decisões-do-sponsor-resolvidas-em-2026-08-03)
**Pré-requisito de leitura para:** DOC-02 (Plano Mestre de Aquisição de Corpus)

> **Isto não é parecer jurídico.** É análise de engenharia sobre restrições jurídicas, escrita para orientar decisões de arquitetura. Antes de qualquer publicação de pesos, o §7 exige parecer formal de advogado com atuação em propriedade intelectual.

---

## 1. Decisões registradas

| ID | Decisão | Escolha do sponsor |
|---|---|---|
| Q1 | Perfil de computação | Indefinido — arquitetura portátil, Perfil A como alvo de projeto |
| Q2 | Livros sob copyright | Domínio público e licenças abertas para treino; obras sob copyright somente em avaliação (`train_ok=False`) |
| Q3 | Intenção de release | Pesos abertos sob licença permissiva (Apache-2.0 / MIT) |
| Q4 | Modelo base para CPT | Qwen3-8B-Base, provisório, a confirmar por bake-off empírico no Portão G1 |

---

## 2. A distinção que decide o tamanho do corpus

Q2 e Q3 só produzem um plano coerente se três direitos distintos forem separados. Tratá-los como um só é o erro que, dependendo da direção, ou inviabiliza o projeto ou o expõe juridicamente.

| Direito | Pergunta | Situação típica no nosso corpus |
|---|---|---|
| **D1 — Acesso** | Podemos obter e ler este documento? | **Quase sempre sim.** O arXiv oferece acesso em massa legítimo (bucket S3 requester-pays); NASA NTRS, CERN Document Server e repositórios de teses são abertos. |
| **D2 — Treinar** | Podemos usar este documento para treinar um modelo? | **Juridicamente não assentado no mundo todo.** É o ponto em litígio ativo. Depende de jurisdição, de exceções de mineração de texto e dados (TDM) e de doutrina de uso legítimo. |
| **D3 — Redistribuir o corpus** | Podemos publicar os bytes deste documento? | **Frequentemente não.** A licença padrão do arXiv, por exemplo, concede ao *arXiv* o direito de distribuir — não a terceiros. |

**A consequência prática:**

> "Pesos abertos" (Q3) é uma afirmação sobre **o nosso artefato** — os pesos, que são obra nossa. **Não** exige que o corpus seja redistribuível. O que exige é uma posição defensável sobre **D2**.

Se D2 e D3 fossem confundidos e aplicássemos a regra de D3 ao treino, o corpus treinável cairia de **~30–50 bilhões de tokens para ~8–15 bilhões** — o que reprovaria o Tier 2 antes de começar, por falta de dado. Essa confusão é o risco número um deste ADR, e por isso ele existe.

**Postura adotada:**

1. **Treinar** (D2) sobre todo conteúdo legitimamente acessado cuja licença não proíba explicitamente o uso, excluindo obras marcadas `train_ok=False`.
2. **Publicar** (D3) pesos, tokenizer, benchmarks, código e **manifestos** — listas de identificadores (arXiv ID, DOI, bibcode) com os hashes e o código de processamento — mas **não os bytes** do corpus, salvo o subconjunto explicitamente livre (§6).
3. Registrar a licença de cada documento no campo `license` do `PhysicsDocumentRecord` (DOC-01 §6), de modo que qualquer nova postura possa ser reaplicada por reconstrução, sem recoleta.

O item 3 é o que torna esta decisão reversível. Se o parecer jurídico do §7 vier mais restritivo, refiltra-se o corpus a partir dos metadados e retreina-se — sem coletar nada de novo.

---

## 3. Realidade das licenças, fonte por fonte

Análise preliminar; a auditoria completa é o corpo do DOC-02.

| Fonte | Situação de licença | D2 (treinar) | D3 (redistribuir) |
|---|---|---|---|
| **arXiv — subconjunto CC** (CC BY, CC BY-SA, CC0) | Explicitamente aberto | ✅ | ✅ (respeitando atribuição/share-alike) |
| **arXiv — licença padrão** ("perpetual, non-exclusive license to distribute") | Concede direito de distribuição **ao arXiv**, não a terceiros | ⚠️ Sob argumento de TDM/uso legítimo | ❌ |
| **arXiv — CC BY-NC-SA** | Cláusula não-comercial | ❌ *(ver §4)* | ⚠️ Só não-comercial |
| **NASA NTRS, NIST, NOAA** | Obras do governo federal dos EUA são de domínio público (17 U.S.C. §105) | ✅ | ✅ |
| **CERN Document Server** | Política forte de acesso aberto; grande parte em CC BY 4.0 (SCOAP³) | ✅ | ✅ |
| **DESY, Fermilab, SLAC, LANL** | Misto — muitos relatórios em domínio público ou CC; obras de contratados exigem checagem individual | ⚠️ Caso a caso | ⚠️ |
| **DOAJ / journals de acesso aberto** | Predominantemente CC BY | ✅ | ✅ |
| **PMC Open Access subset** | Misto CC BY / CC BY-NC | ✅ para CC BY | ⚠️ |
| **Teses e dissertações (repositórios institucionais)** | Muito variável; frequentemente sem licença explícita | ⚠️ Caso a caso | ❌ por padrão |
| **OpenStax** | CC BY 4.0 | ✅ | ✅ |
| **LibreTexts** | Misto; boa parte CC BY-NC-SA | ⚠️ Filtrar página a página | ⚠️ |
| **MIT OpenCourseWare** | **CC BY-NC-SA 4.0** | ❌ *(ver §4)* | ❌ para uso comercial |
| **The Feynman Lectures (Caltech)** | Gratuito para leitura, **sem licença aberta**; termos vedam coleta automatizada | ❌ | ❌ |
| **Clássicos pré-1931** (Maxwell, Gibbs, Boltzmann, Poincaré, Einstein 1905–16, Eddington, Jeans) | Domínio público nos EUA | ✅ | ✅ |
| **Livros modernos sob copyright** (Jackson, Landau, Sakurai, Peskin, MTW, Ashcroft, Arfken, Boas) | Copyright ativo | ❌ **`train_ok=False`** | ❌ |
| **Physics StackExchange** | CC BY-SA 4.0 | ✅ | ✅ (share-alike) |

---

## 4. ⚠️ Ressalva material: a escolha de licença permissiva exclui fontes que você pediu

Esta é a consequência não óbvia da combinação Q2 + Q3, e precisa de confirmação explícita.

**O conflito.** Se treinarmos sobre conteúdo **CC BY-NC** (não-comercial) e publicarmos os pesos sob **Apache-2.0** (que autoriza uso comercial), estaremos — sob a leitura mais conservadora — habilitando uso comercial de material licenciado como não-comercial. A questão de se pesos de modelo constituem obra derivada do material de treino é juridicamente **não assentada**. A postura conservadora, coerente com Q3, é **excluir conteúdo NC do treino**.

**O que isso remove do corpus, entre as fontes que você listou nominalmente:**

| Fonte excluída | Licença | Perda estimada |
|---|---|---|
| **MIT OpenCourseWare** | CC BY-NC-SA 4.0 | ~0,3–0,8 B tokens de material didático de altíssima qualidade |
| **Parte do LibreTexts** | CC BY-NC-SA em boa fração | ~0,2–0,5 B tokens |
| **Notas de aula universitárias com cláusula NC** | Variado | ~0,3–1,0 B tokens |
| **Subconjunto NC do arXiv** | CC BY-NC-SA | ~0,5–1,5 B tokens |
| **Total** | | **~1,3–3,8 B tokens** |

> ### 📊 Medição real (2026-08-03) — substitui a estimativa acima para o arXiv
>
> O Sprint S1 coletou os metadados do arXiv e a distribuição de licenças deixou de ser estimativa. Numa amostra de **6.975 papers de Física** (datestamps de junho/2024):
>
> | Licença | Fração | Efeito |
> |---|---|---|
> | **CC BY 4.0** | **43,3%** | ✅ Treina e redistribui |
> | CC BY-SA | 1,3% | ✅ Treina e redistribui (share-alike) |
> | CC0 1.0 | 1,4% | ✅ Treina e redistribui |
> | **arXiv padrão** (não-exclusiva) | **45,8%** | ⚠️ Treina (D2), **não redistribui** (D3) |
> | CC BY-NC-ND | 5,5% | ❌ **Excluído do treino** |
> | CC BY-NC-SA | 2,7% | ❌ **Excluído do treino** |
>
> **Três consequências:**
>
> 1. **45,9% do arXiv de Física é redistribuível e compatível com Apache-2.0.** É bem mais que o suposto ao escrever o §6 deste ADR, e **aumenta materialmente o `PhysCorpus-Open`**.
> 2. **O custo da decisão Q3 é de 8,2%** dos papers, por cláusula NC. Fica dentro da faixa de 3–10% estimada no §4, na metade superior. A decisão se mantém, agora com número medido.
> 3. ⚠️ **A amostra é de 2024 e superestima o histórico.** A opção de licença CC foi introduzida pelo arXiv anos após o início do repositório, e a adesão cresceu com o tempo. A fração CC de todo o acervo será **menor** — provavelmente 25–35%. O número definitivo sai da coleta completa, em andamento.

Sobre um corpus de 30–50 B, é uma perda de **3% a 10%** — mas é material *pedagogicamente denso*, exatamente o tipo que mais contribui para explicação didática e resolução de problemas. Não é uma perda irrelevante.

**Três saídas, para sua decisão:**

| Opção | Efeito | Custo |
|---|---|---|
| **A. Manter Apache-2.0, excluir NC** *(padrão atual)* | Máxima defensabilidade jurídica e máxima adoção | Perde 1,3–3,8 B tokens de material didático |
| **B. Publicar os pesos sob CC BY-NC-SA** | Permite usar todo o material NC | Elimina uso comercial dos pesos; reduz muito a adoção e a citação |
| **C. Dois modelos** — `ΦGen-open` (Apache-2.0, sem NC) e `ΦGen-nc` (CC BY-NC-SA, corpus completo) | Melhor dos dois mundos; permite medir empiricamente o valor do material NC | Dobra o custo de CPT do Tier 2 (~+$1.700–3.400) e a complexidade de avaliação |

**Recomendação: opção A agora, com a opção C como experimento no Tier 2 se o orçamento permitir.** A comparação `ΦGen-open` vs. `ΦGen-nc` seria, por si só, uma contribuição publicável — *quanto vale, em pontos de benchmark, o material educacional com cláusula não-comercial?* Ninguém mediu isso, e nós teríamos o aparato para medir.

---

## 5. O mecanismo que torna Q2 seguro

A decisão Q2 — ingerir livros sob copyright para avaliação, jamais para treino — só é confiável se for imposta por código, não por processo.

O mecanismo, definido em DOC-01 §6:

```
license.train_ok = False
        ↓
roteamento físico para a partição eval-only do Iceberg
        ↓
o dataloader de treino não tem credencial de leitura dessa partição
        ↓
teste de CI falha o build se qualquer manifesto de shard de treino
referenciar um doc_id com train_ok=False
```

Três camadas independentes: separação física, separação de permissão e verificação em CI. Vazamento exige que as três falhem ao mesmo tempo. **É isso que permite usar Jackson e Sakurai como conjunto de avaliação de altíssima qualidade sem risco de contaminar os pesos** — e é uma vantagem real, porque os problemas desses livros são o melhor material de avaliação de Física que existe.

---

## 6. Estratégia de release em dois níveis

Decorre naturalmente de §2:

| Artefato | Nível | Licença | Conteúdo |
|---|---|---|---|
| **Pesos ΦFM** | Público | Apache-2.0 | Todos os modelos da família |
| **Código, configs, tokenizer** | Público | Apache-2.0 | Repositório completo |
| **PhysBench** | Público | CC BY 4.0 | Benchmarks e corretores |
| **Manifestos do corpus** | Público | CC BY 4.0 | IDs, hashes, decisões de filtro, código de reconstrução — permite reprodução por terceiros sem redistribuir bytes |
| **`PhysCorpus-Open`** | Público | CC BY 4.0 | O subconjunto **verdadeiramente livre** (CC BY / CC0 / domínio público / governo dos EUA) — estimativa de **8–15 B tokens**. Seria, ao que se sabe, o maior corpus de Física abertamente redistribuível já publicado |
| **`PhysCorpus-Full`** | Interno | — | Corpus completo de treino, não redistribuído |
| **`PhysEval-Restricted`** | Interno | — | Partição `train_ok=False`; livros sob copyright, uso exclusivo de avaliação |

O `PhysCorpus-Open` merece destaque: mesmo sendo um terço do corpus completo, **8–15 B tokens de Física abertamente redistribuível é um artefato de valor científico próprio**, e é publicável ainda no Tier 1 — antes de qualquer modelo estar pronto. É uma entrega de baixo risco e alto retorno reputacional.

---

## 7. Trabalho jurídico obrigatório antes do release

| Item | Quando | Por quê |
|---|---|---|
| Parecer sobre D2 na jurisdição brasileira | Antes do Tier 2 | A LDA 9.610/98 tem exceções estreitas e **não possui exceção explícita de TDM**; a regulação de IA em tramitação no Congresso pode alterar o quadro. Como o projeto é conduzido no Brasil, é a jurisdição que governa. |
| Parecer sobre a compatibilidade NC ↔ Apache-2.0 | Antes do Tier 2 | Confirma ou derruba a exclusão do §4 |
| Registro de opt-out de TDM | Contínuo | Alguns editores publicam sinalizações de recusa de TDM; respeitá-las fortalece a posição |
| Revisão de termos de uso por fonte | Durante o DOC-02 | Coleta automatizada pode violar termos de serviço mesmo quando o conteúdo é acessível — questão distinta de direito autoral |
| Model card com procedência do corpus | No release | Prática exigida pela comunidade e mitigadora de risco |

---

## 8. Gatilhos de revisão

Este ADR deve ser reaberto se:

- Decisão judicial relevante sobre treino de modelos em obras protegidas, no Brasil, EUA ou UE
- Aprovação de marco legal de IA no Brasil com disposições sobre TDM
- Mudança da postura do arXiv sobre uso do corpus para treino
- Sucesso em obter licenciamento com editoras (abriria 2–5 B tokens adicionais e alteraria o §3)
- Decisão de publicar o `PhysCorpus-Full` (exigiria reanálise completa de D3)

---

## 9. Consequências para o DOC-02

O DOC-02 fica autorizado a prosseguir sob estas regras:

1. Toda fonte é catalogada com licença resolvida em SPDX **antes** de qualquer coleta em massa.
2. Fontes NC entram no catálogo marcadas, coletadas mas **não** roteadas para treino (permitindo a opção C do §4 depois).
3. Livros sob copyright entram exclusivamente pela partição `train_ok=False`.
4. Toda coleta respeita `robots.txt`, rate limits e termos de serviço; fontes que exijam contorno técnico são excluídas do plano.
5. Nenhuma biblioteca-sombra é utilizada, em nenhuma hipótese — inviabilizaria o release de pesos e a publicação científica, que são os objetivos declarados do programa.

---

**Fim do ADR-0001.**
