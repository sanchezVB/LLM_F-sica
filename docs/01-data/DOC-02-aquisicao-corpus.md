# DOC-02 — Plano Mestre de Aquisição de Corpus

**Projeto:** ΦFM — Phi Foundation Models para Física
**Status:** `RASCUNHO v0.1` — aguardando revisão do Stage-Gate 1
**Cobre:** entregável solicitado **3** (pipeline de coleta de dados)
**Depende de:** [DOC-00](../00-foundations/DOC-00-project-charter.md), [DOC-01 §6](../00-foundations/DOC-01-system-architecture.md), [ADR-0001](../adr/ADR-0001-decisoes-stage-gate-0.md), [DOC-17A §8](../05-governance/DOC-17A-orcamento-gpu-runpod.md)
**Data:** 2026-08-03

---

## 1. Objetivo e princípios

Construir o maior corpus de Física com proveniência completa que possa ser montado **sem custo de aquisição e sem exposição jurídica**, e deixar registrado o caminho para expandi-lo caso surja orçamento ou licenciamento.

Cinco princípios de aquisição, derivados de DOC-01 §1 e ADR-0001 §9:

| # | Princípio | Consequência operacional |
|---|---|---|
| **A1** | **Metadado antes de conteúdo** | A espinha de metadados (arXiv, OpenAlex, INSPIRE, ADS) é coletada **primeiro**. Ela é a chave de junção que resolve categoria, licença, DOI e grafo de citações de todas as outras fontes. Sem ela, não há como filtrar para Física nem como aplicar o ADR-0001. |
| **A2** | **Gratuito e pré-processado antes de bruto e pago** | Fatias de corpus já limpas e livres existem. Usá-las primeiro dá 60–70% do corpus na primeira quinzena, a custo zero. |
| **A3** | **Licença resolvida em SPDX antes da coleta em massa** | Nenhuma fonte entra em coleta grande sem `license.spdx_id` determinado e `train_ok` decidido. |
| **A4** | **Coleta idempotente, retomável, endereçada por conteúdo** | Toda coleta pode ser interrompida e retomada sem duplicar nem perder trabalho. Obrigatório quando o processamento leva semanas numa máquina doméstica. |
| **A5** | **Cortesia e conformidade são inegociáveis** | `robots.txt`, limites de taxa declarados, `User-Agent` identificando o projeto com e-mail de contato, e recuo exponencial. Uma fonte que nos bloqueie é uma fonte perdida para sempre. |

---

## 2. Taxonomia de destino: o seletor de Física

A lista de áreas do briefing precisa virar um seletor executável. O mapeamento abaixo é o filtro primário de todas as fontes com metadados do arXiv, e o alvo do classificador (§6) para as fontes sem.

| Área do briefing | Categorias arXiv | Volume relativo |
|---|---|---|
| Mecânica Clássica / Analítica / Hamiltoniana / Lagrangiana | `physics.class-ph`, `nlin.CD`, `math.DS` | Pequeno no arXiv — **lacuna estrutural**, ver §7 |
| Relatividade Restrita e Geral, Gravitação | `gr-qc` | ~110 k papers |
| Mecânica Quântica, Informação Quântica | `quant-ph` | ~250 k |
| Teoria Quântica de Campos, Cordas | `hep-th` | ~160 k |
| Física de Partículas | `hep-ph`, `hep-ex`, `hep-lat` | ~250 k |
| Física Nuclear | `nucl-th`, `nucl-ex` | ~90 k |
| Física Estatística, Termodinâmica | `cond-mat.stat-mech` | ~60 k |
| Matéria Condensada | `cond-mat.*` (8 subcats) | ~350 k |
| Astrofísica e Cosmologia | `astro-ph.*` (6 subcats) | ~350 k |
| Óptica | `physics.optics` | ~40 k |
| Eletromagnetismo | disperso: `physics.class-ph`, `physics.optics`, `physics.app-ph` | — |
| Plasma | `physics.plasm-ph` | ~20 k |
| Instrumentação e Física Experimental | `physics.ins-det`, `physics.data-an`, `astro-ph.IM` | ~50 k |
| Métodos Numéricos, Ciência Computacional | `physics.comp-ph`, `math.NA` | ~50 k |
| Física Matemática | `math-ph` | ~60 k |
| Fluidos, Atômica, Química, Biofísica, Geofísica, Espacial, Aceleradores | `physics.*` restantes | ~200 k |
| **Matemática de apoio** | `math.AP` (EDP), `math.CA` (Análise Real/Complexa), `math.DG` (Geom. Diferencial), `math.OC` (Otimização/Variacional), `math.PR` (Probabilidade), `math.ST` (Estatística), `math.NA`, `math.RA`/`math.LA` (Álgebra Linear) | ~300 k |

**Nota crítica.** Mecânica Clássica, Eletromagnetismo de graduação e Termodinâmica básica são **sub-representados no arXiv**, porque são conhecimento consolidado e ninguém publica preprints sobre eles. Isso é uma lacuna estrutural do corpus que ataca diretamente vários modos de falha do DOC-00 §2. As fontes que a cobrem são livros abertos, notas de aula, StackExchange, clássicos de domínio público e dados sintéticos (DOC-06). **Elas não são complemento — são compensação obrigatória de um viés conhecido.**

---

## 3. Catálogo de fontes — Camada A: gratuito, imediato, licença resolvida

Prioridade máxima. Tudo aqui é obtível em semanas, sem custo e sem ambiguidade jurídica relevante.

### 3.1 Espinha de metadados (A1 — coletar primeiro)

| Fonte | Acesso | Volume | Licença | Por que primeiro |
|---|---|---|---|---|
| **arXiv metadata** | OAI-PMH em **`https://oaipmh.arxiv.org/oai`**, `metadataPrefix=arXiv`, `set=physics` | ~1,2 M registros, **~700 MB** em parquet | Metadados CC0 | Fornece **categoria, licença por paper, DOI, journal-ref, resumo**. É a chave de tudo. |
| **OpenAlex** | ⚠️ **API agora é cotada** — usar o snapshot S3, que continua livre | 250 M obras; snapshot ~330 GB | **CC0** | Grafo de citações, fields-of-study, links de OA, resolução de autores/instituições |
| **INSPIRE-HEP** | API REST (`inspirehep.net/api/literature`) | ~1,5 M registros | Metadados CC0 | Melhor metadado de HEP que existe; aponta para fulltext OA |
| **NASA ADS** | API com token gratuito | ~2 M registros de Física/astro | Metadados abertos | Cobertura de astronomia insuperável; resumos |
| **Unpaywall** | Snapshot / API gratuita | 50 M DOIs | CC0 | Descobre qual paper tem versão OA e onde |

**Custo: US$ 0. Tempo: ~2 h para o arXiv; 3–5 dias para o conjunto. Tamanho em disco: ~10–25 GB.**

> ### ⚙️ Correções apuradas em execução (2026-08-03)
>
> O Sprint S1 foi executado e cinco pontos desta seção estavam errados. Registrados aqui porque cada um teria custado tempo ou credibilidade:
>
> 1. **Endpoint.** `export.arxiv.org/oai2` responde **301** e está obsoleto. O correto é **`https://oaipmh.arxiv.org/oai`**. O `Identify` declara explicitamente: *"Metadata harvesting permitted through OAI interface"*.
> 2. **Filtragem no servidor.** O arXiv expõe o *set* **`physics`** (e subsets `physics:hep-th`, `physics:gr-qc`…). Não é preciso coletar 2,7 M registros e filtrar depois — coletamos **~1,2 M direto**. Menos tráfego para eles, menos tempo para nós.
> 3. **Formato.** `metadataPrefix=arXiv` (não `oai_dc`) traz categorias, **licença por registro**, DOI e journal-ref — exatamente os campos que o princípio A3 exige.
> 4. **Tamanho.** A espinha de metadados do arXiv ocupa **~516–686 bytes/registro** em parquet zstd, ou **~700 MB no total** — não os ~150 GB estimados. Aquele número pressupunha baixar o snapshot completo do OpenAlex (330 GB), o que é **desnecessário**: a API do OpenAlex permite filtrar a fatia de Física, e uma passagem já traz também as `referenced_works` (o grafo de citações do DOC-07 §3.1). **Consequência: o Sprint S1 cabe folgado em disco comum.**
> 5. **Semântica de `from`/`until`.** Filtram por **datestamp** (última modificação), não por data de criação. Uma fatia de um mês retorna também papers antigos com metadados atualizados. Irrelevante para a coleta completa; relevante para fatias.
>
> Também observado: este servidor retorna `completeListSize=0`, então **não há barra de progresso percentual** — o progresso é reportado em contagem absoluta.

> ### ⚠️ O OpenAlex passou a cobrar pela API (medido em 2026-08-03)
>
> Esta seção descrevia o OpenAlex como *"snapshot S3 público, sem chave"*. A **API** mudou de modelo e agora é cotada:
>
> ```
> "Insufficient budget. This request costs $0.0001 but you only have $0
>  remaining. Resets at midnight UTC."
> ```
>
> | | |
> |---|---|
> | Cota gratuita | **1.000 requisições/dia** (`x-ratelimit-limit: 1000`) |
> | Custo por requisição | US$ 0,0001 |
> | Necessário para 3,67 M obras | 18.336 requisições |
> | Pela cota gratuita | **18 dias** |
> | Pagando | **US$ 1,83** |
>
> **O snapshot S3 continua gratuito e sem cota**, e é a rota correta. Medição
> na conexão do projeto: 444 Mbps de download, ou seja, **~2 h para os 330 GB**
> — mais rápido que pagar a API. Processado em fluxo (baixa partição → filtra
> para arXiv → apaga → próxima), o pico de disco fica em ~10 GB, pelo mesmo
> princípio do DOC-03 §8.
>
> **Lição para o resto do plano:** o custo declarado de uma fonte é uma
> *medição com prazo de validade*, não um fato. Toda fonte da Camada A precisa
> ser reverificada no momento da coleta, e o plano precisa de rota alternativa
> onde houver dependência de API de terceiro.

> Limite de taxa do ADS: ~5.000 requisições/dia por token. Coletar 2 M registros em lotes de 2.000 leva ~1.000 requisições — trivial. A API do arXiv pede **1 requisição a cada 3 segundos**; use OAI-PMH em lote, não a API de busca.

### 3.2 Texto completo pré-processado e gratuito (A2 — o grosso do corpus)

| Fonte | Acesso | Tokens (fatia Física) | Licença | Observação |
|---|---|---|---|---|
| **RedPajama-Data-1T, fatia arXiv** | HuggingFace | **10–14 B** | Licenças arXiv originais | ~28 B tokens de LaTeX de todo o arXiv; filtrar por categoria via junção com §3.1 |
| **OpenWebMath** | HuggingFace | **2–4 B** | Web, mista | 14,7 B tokens de web matemática com LaTeX preservado; alta densidade de Física |
| **peS2o (AI2)** | HuggingFace | **6–10 B** | **ODC-BY** | ~40 B tokens derivados do S2ORC, já limpos; filtrar por field-of-study |
| **proof-pile-2 / AlgebraicStack** | HuggingFace | **1–3 B** | Mista, documentada | Matemática formal e código algébrico; relevante para o barramento de verificação |

**Custo: US$ 0. Tempo de download: 2–4 dias. Tamanho em disco: ~400–600 GB comprimido.**

> **Contrapartida honesta (viola parcialmente P2 do DOC-01).** Essas fatias foram processadas pelo pipeline de terceiros, não pelo nosso. Parte da estrutura de LaTeX pode ter sido degradada — exatamente o que o Minerva identificou como decisivo. **Protocolo obrigatório:** processar nós mesmos uma amostra aleatória de 2.000 papers a partir da fonte do arXiv e medir a taxa de preservação de equações contra a mesma amostra no RedPajama. Se a degradação exceder ~10%, o bulk pago do arXiv (~US$ 100–180 de egress, Camada C) passa a se justificar. **Custo da medição: US$ 0. Ela precede qualquer decisão de gastar.**

### 3.3 Domínio público — obras do governo dos EUA (17 U.S.C. §105)

Categoria mais limpa juridicamente de todo o plano: sem copyright, redistribuível, treinável.

| Fonte | Acesso | Volume | Conteúdo |
|---|---|---|---|
| **NASA NTRS** | API REST (`ntrs.nasa.gov/api/citations/search`) | ~500 k–1 M docs, **3–6 B tokens** | Astrofísica, propulsão, instrumentação, plasma, dinâmica orbital, relatórios técnicos de missões |
| **OSTI.GOV (DOE)** | API REST | ~500 k docs OA, **2–5 B tokens** | **Los Alamos, Fermilab, SLAC, Brookhaven, Oak Ridge, Livermore** — o equivalente do DOE ao NTRS. Frequentemente esquecido; é enorme |
| **NIST** | Publicações da NIST Technical Series | ~50 k docs, **0,2–0,5 B** | Metrologia, constantes, padrões, dados de referência |

**Custo: US$ 0. Tempo: 2–4 semanas (coleta paginada com cortesia). Muitos documentos são PDF → exigem parsing, ver DOC-03.**

> Ressalva: obras produzidas por **contratados** de laboratórios nacionais nem sempre são domínio público. O campo `license` deve registrar a evidência por documento; na dúvida, `redistributable=False`, `train_ok=True`.

### 3.4 Acesso aberto com licença CC explícita

| Fonte | Acesso | Volume | Licença |
|---|---|---|---|
| **SCOAP³** | `repo.scoap3.org`, API + bulk | ~60 k artigos, **0,6–1 B tokens** | **CC BY 4.0** |
| **SciPost Physics** | API aberta | ~5 k artigos | CC BY 4.0 |
| **DOAJ (fatia Física)** | API DOAJ + editores | ~150 k artigos, **1,5–2,5 B** | Predominantemente CC BY |
| **CERN Document Server** | OAI-PMH + REST | ~200 k registros, **1–2 B** | Mista; grande parte CC BY 4.0 |
| **HAL, Zenodo, arXiv-CC** | OAI-PMH / API | **0,5–1,5 B** | CC BY / CC0 |

> **SCOAP³ merece destaque.** É um consórcio que converteu para acesso aberto CC BY praticamente toda a produção de física de altas energias das principais revistas — **Physics Letters B, Nuclear Physics B, JHEP, European Physical Journal C**, e partes de **Physical Review D e PRL**. Ou seja: conteúdo **revisado por pares, publicado em revista de primeira linha, com licença aberta e redistribuível**. Isso resolve parcialmente e de forma legítima o acesso a APS, Elsevier e Springer que o briefing pedia — sem licenciamento, sem custo, sem ambiguidade. É a melhor relação valor/esforço de todo o catálogo.

### 3.5 Material pedagógico e conceitual (compensação da lacuna do §2)

| Fonte | Acesso | Volume | Licença |
|---|---|---|---|
| **Physics StackExchange** (+ Astronomy, Math, MathOverflow) | Dumps no Internet Archive | ~1,5 M posts, **0,3–0,6 B tokens** | **CC BY-SA 4.0** |
| **OpenStax** (University Physics I–III, College Physics, Astronomy) | Download direto, fonte CNXML | ~10 M tokens | **CC BY 4.0** |
| **Clássicos de domínio público** (pré-1931) | Internet Archive, Project Gutenberg | ~3–8 k volumes, **0,5–1,5 B** | Domínio público |
| **Notas de aula de autores, livremente distribuídas** | Sites institucionais | **0,05–0,2 B** | Caso a caso |

**Clássicos de domínio público a priorizar:** Maxwell (*Treatise on Electricity and Magnetism*), Gibbs (*Elementary Principles in Statistical Mechanics*), Boltzmann (*Vorlesungen über Gastheorie*), Poincaré, Einstein (artigos 1905–1916), Eddington (*The Mathematical Theory of Relativity*), Jeans, Lorentz, Rayleigh (*Theory of Sound*), Thomson & Tait, Whittaker (*Analytical Dynamics*), Routh, Lamb (*Hydrodynamics*), Love (*Elasticity*), Heaviside, Larmor.

> **Achado relevante:** vários textos canônicos modernos são **disponibilizados gratuitamente pelos próprios autores** e portanto são utilizáveis, ao contrário do que a lista do briefing sugere. Casos a verificar individualmente durante a execução: as notas de Teoria Quântica de Campos de **Srednicki**, as notas de **David Tong** (Cambridge DAMTP), as notas de Relatividade Geral de **Sean Carroll** (publicadas no arXiv como `gr-qc/9712019`), e diversos textos de 't Hooft. **Cada um exige verificação individual dos termos**, mas o conjunto pode render material de qualidade de livro-texto legitimamente.

---

## 4. Camada B: gratuito, porém trabalhoso

Entra depois do Portão G1. Alto volume, alta heterogeneidade, licenças ambíguas.

| Fonte | Volume estimado | Dificuldade | Postura de licença |
|---|---|---|---|
| **Teses e dissertações** — CORE.ac.uk, NDLTD, DART-Europe, e repositórios do MIT (DSpace), Caltech (THESIS), Harvard (DASH), Stanford, Berkeley (eScholarship), Cambridge (Apollo), Oxford (ORA), ETH (Research Collection) | ~150–300 k, **6–12 B tokens** | Alta — PDF, formatos díspares, muitos sem licença explícita | `train_ok=True`, `redistributable=False` por padrão |
| **Código científico** — The Stack v2 filtrado para NumPy/SciPy/SymPy/astropy/QuTiP/FEniCS/LAMMPS/GEANT4/CLASS/CAMB + notebooks | **3–8 B tokens** | Média | Só licenças permissivas |
| **Dados abertos** — CERN Open Data (CC0), GWOSC (LIGO), MAST/HEASARC (NASA), SDSS, Materials Project | Sobretudo numérico | Média | Predominantemente CC0/PD |
| **ESA, DESY, demais laboratórios** | **0,5–2 B** | Média | Caso a caso |

> **Distinção importante sobre "dados experimentais e simulações".** O briefing os lista como fonte de corpus. Eles **não são texto de pretraining** — são arrays numéricos. O valor deles é outro e é maior: (a) treinar e avaliar **uso de ferramentas** (o modelo consulta dados reais em vez de alucinar), (b) ancorar o ΦAgent, (c) construir o benchmark de raciocínio experimental do DOC-11. Tratá-los como tokens de pretraining seria desperdiçá-los. O plano de exploração está no DOC-14, não aqui.

---

## 5. Camadas C, D e E — adiado, só avaliação, excluído

### Camada C — pago ou negociado (adiado indefinidamente)

| Fonte | Situação | Custo | Veredito |
|---|---|---|---|
| **arXiv bulk S3** (`s3://arxiv`, requester-pays) | Fonte LaTeX íntegra, processada por nós | ~US$ 100–180 de egress | **Só se a medição do §3.2 reprovar o RedPajama** |
| **APS / Physical Review** | Programa de TDM para assinantes; parte já livre via SCOAP³ | Assinatura institucional | Adiado — SCOAP³ cobre a fatia de HEP |
| **Elsevier** (Physics Letters, Nuclear Physics) | API de TDM para assinantes; PLB e NPB já em CC BY via SCOAP³ | Assinatura | Adiado |
| **Springer Nature** (Nature Physics, EPJ) | API de TDM; EPJC já em CC BY via SCOAP³ | Assinatura | Adiado |
| **IOP** | *New Journal of Physics* é CC BY; resto por assinatura | Assinatura | NJP entra na Camada A; resto adiado |
| **AIP** (J. Appl. Phys., Phys. Fluids, Rev. Sci. Instrum.) | Assinatura | Assinatura | Adiado |
| **Science / AAAS**, **Wiley** (Annalen der Physik) | Assinatura | Assinatura | Adiado |

> Se houver vínculo institucional com acesso a essas bases, a rota correta é o **programa oficial de TDM do editor** — não coleta pela interface web, que viola termos de serviço mesmo com acesso legítimo. Isso exige convênio formal e está fora do caminho crítico.

### Camada D — somente avaliação (`train_ok=False`)

Ingeridos, indexados e usados **exclusivamente** como conjunto de avaliação, com a proteção tripla do ADR-0001 §5. **Obtidos apenas por vias legítimas** — exemplares próprios, acesso institucional ou de biblioteca.

Jackson · Griffiths (EM, MQ, Partículas) · Landau & Lifshitz (10 vols) · Goldstein · Sakurai · Messiah · Dirac · Weinberg (QFT, Gravitação, Cosmologia) · Peskin & Schroeder · Misner–Thorne–Wheeler · Reif · Kittel · Ashcroft & Mermin · Born & Wolf · Boas · Arfken & Weber · Morse & Feshbach · Courant & Hilbert · Shankar · Ballentine · Zangwill · Feynman Lectures

**Valor:** são a melhor fonte de problemas de avaliação de Física que existe. Alimentam o PhysBench (DOC-11) e nunca tocam os pesos.

### Camada E — excluído sem exceção

| Excluído | Motivo |
|---|---|
| **Bibliotecas-sombra** (LibGen, Sci-Hub, Z-Library, Anna's Archive) | ADR-0001 §9.5 — inviabilizaria release de pesos e publicação, que são os objetivos declarados |
| **MIT OpenCourseWare** | CC BY-NC-SA; incompatível com pesos Apache-2.0 (ADR-0001 §4) |
| **The Feynman Lectures online (Caltech)** | Sem licença aberta; termos vedam coleta automatizada |
| **Qualquer fonte que exija contornar paywall, CAPTCHA ou ToS** | A5 |

---

## 6. Filtragem para Física: o classificador de rótulo gratuito

Fontes como peS2o, OpenWebMath e The Stack são multi-domínio. Reduzi-las a Física exige um classificador — e existe uma fonte de rótulos perfeita e gratuita.

**Método:**

1. **Rótulos gratuitos.** Os ~2,7 M resumos do arXiv já vêm rotulados pelos próprios autores com categoria (§3.1). São dados de treino supervisionado de graça, com a taxonomia exata que queremos.
2. **Modelo.** Classificador `fastText` hierárquico sobre as 23 subáreas do §2, mais uma classe `não-física`. Treina em minutos na CPU, infere a ~10⁵ documentos/segundo. Um transformer daria talvez +2 pontos de F1 a um custo de inferência 1000× maior sobre 10⁸ documentos — **trade-off errado nesta etapa**.
3. **Calibração do limiar.** Não usar 0,5. Calibrar sobre um conjunto de validação anotado à mão (500 docs) para **precisão de 0,95**, aceitando revocação menor. Justificativa: o corpus é abundante e o custo de um documento irrelevante contaminando o treino é maior que o de perder um documento relevante.
4. **Rótulo suave preservado.** A probabilidade por subárea é gravada em `taxonomy.subfield[]`, não descartada. Ela vira o peso de amostragem do currículo no DOC-06 e o eixo de estratificação da avaliação no DOC-12.

**Custo: US$ 0. Tempo: 2 dias.** Alternativa considerada e rejeitada: filtro por lista de palavras-chave — rápido, mas revocação terrível em textos onde a Física é implícita (um paper de EDP sobre a equação de onda não diz "física" em lugar nenhum).

---

## 7. Orçamento consolidado de tokens

| Camada | Fonte | Tokens brutos | Custo |
|---|---|---|---|
| A | RedPajama-arXiv (Física) | 10–14 B | $0 |
| A | peS2o (Física) | 6–10 B | $0 |
| A | OpenWebMath (Física) | 2–4 B | $0 |
| A | proof-pile-2 / AlgebraicStack | 1–3 B | $0 |
| A | NASA NTRS | 3–6 B | $0 |
| A | OSTI.GOV (DOE) | 2–5 B | $0 |
| A | SCOAP³ + DOAJ + SciPost + NJP | 2–3,5 B | $0 |
| A | CERN CDS + INSPIRE fulltext | 1,5–3 B | $0 |
| A | NIST | 0,2–0,5 B | $0 |
| A | StackExchange (Física, Astro, Math) | 0,3–0,6 B | $0 |
| A | Clássicos de domínio público | 0,5–1,5 B | $0 |
| A | OpenStax + notas abertas | 0,06–0,2 B | $0 |
| B | Teses e dissertações | 6–12 B | $0 |
| B | Código científico | 3–8 B | $0 |
| B | ESA, DESY, demais laboratórios | 0,5–2 B | $0 |
| | **Bruto, antes de deduplicação** | **~39–73 B** | **$0** |

**Fator de deduplicação.** As fontes de artigo (RedPajama-arXiv, peS2o, SCOAP³, CDS, INSPIRE) **se sobrepõem fortemente** — são o mesmo paper visto por agregadores diferentes. A sobreposição esperada nesse bloco é de 40–55%. As fontes aditivas de verdade são NTRS, OSTI, teses, StackExchange, clássicos e código.

| | Estimativa |
|---|---|
| Bruto | 39–73 B |
| Após dedup exata + heurísticos + near-dedup | 20–38 B |
| Após qualidade por modelo + dedup semântica + descontaminação | **15–30 B** |
| Subconjunto redistribuível (`PhysCorpus-Open`) | **6–12 B** |

**Conclusão: 15–30 bilhões de tokens de Física, por US$ 0 de aquisição.** Suficiente com folga para todo o Tier 1 e para o CPT do ΦGen-1,5B em uma época (DOC-17A §8.2); o ΦGen-8B exige 2–4 épocas, dentro do que Muennighoff et al. (2023) indicam ser seguro, mas sem margem.

> **Correção (v0.2, após o DOC-04).** A estimativa original desta seção era de **22–42 B**. O [DOC-04 §7](DOC-04-filtragem-dedup-descontaminacao.md#7-o-funil-com-números) modelou o funil estágio a estágio e encontrou **15–30 B**. A diferença vem de dois estágios que esta seção não contabilizava separadamente: a deduplicação semântica (−5%) e o filtro de qualidade por modelo (−15%). O número do DOC-04 é o vigente; este ficou registrado para preservar o rastro da revisão.

---

## 8. Infraestrutura de coleta

### 8.1 Manifesto de aquisição

Toda coleta emite um `AcquisitionManifest` antes de qualquer byte ser baixado. Ele é o que torna A4 verificável e o que satisfaz G1.5 (DOC-00 §5).

```python
# src/phifm/corpus/acquire/manifest.py  (especificação)

class AcquisitionManifest(BaseModel):
    manifest_id: str            # BLAKE3 da config de coleta — identidade do lote
    source_name: str            # "arxiv_metadata", "nasa_ntrs", "scoap3", ...
    harvest_method: Literal["oai_pmh", "rest_api", "bulk_s3", "hf_dataset",
                            "dump_archive", "direct_download"]
    endpoint: str
    query_spec: dict            # filtros exatos: categorias, intervalo de datas, campos
    rate_limit: RateLimit       # req_per_sec, burst, backoff_base, max_retries
    started_at: datetime
    completed_at: datetime | None
    expected_count: int | None
    actual_count: int
    bytes_downloaded: int
    checksum_index_uri: str     # mapa doc_id → BLAKE3, endereçado por conteúdo
    license_resolution: LicenseResolution   # como a licença foi determinada (A3)
    failures: list[FailureRecord]           # o que falhou e por quê — nunca silencioso
    resumable_cursor: str | None            # token OAI-PMH ou offset de paginação
    pipeline_git_sha: str
```

**Regras impostas por código:**
- Um lote com `failures` não vazio **não** é marcado como concluído sem revisão explícita.
- `resumable_cursor` é persistido a cada N registros; interrupção nunca custa mais que N.
- `license_resolution` vazio bloqueia o lote — imposição direta de A3.

### 8.2 Limites de taxa e cortesia

| Fonte | Limite | Estratégia |
|---|---|---|
| arXiv OAI-PMH | 1 req / 3 s, respeitar `retry-after` | Lotes de 1.000 registros; ~2.700 requisições no total |
| NASA ADS | ~5.000 req/dia por token | Lotes de 2.000; concluir em 1–2 dias |
| INSPIRE-HEP | Sem limite publicado; ser conservador | 1 req/s, `User-Agent` identificado |
| NASA NTRS / OSTI | Sem limite publicado | 1–2 req/s, recuo exponencial em 429/503 |
| HuggingFace | Alto | Paralelismo de 4–8; usar `hf_transfer` |
| Internet Archive | Moderado | 1 req/s; preferir arquivos de dump a itens individuais |

`User-Agent` padrão para todo o projeto:
```
PhiFM-Corpus/0.1 (Physics foundation model research; +<url-do-repo>; <e-mail-de-contato>)
```

**Conformidade não negociável:** `robots.txt` respeitado por padrão via `urllib.robotparser`; recuo exponencial com jitter em 429/503; nenhuma fonte é coletada em paralelo além do que ela declara suportar. Uma fonte que nos bloqueie está perdida permanentemente, e o custo disso excede qualquer ganho de velocidade.

---

## 9. Cronograma de execução

Sequenciado por dependência e por relação valor/esforço. Tudo executável em máquina local, em segundo plano.

| Sprint | Semanas | Entrega | Disco | Custo |
|---|---|---|---|---|
| **S1 — Espinha de metadados** | 1 | arXiv, OpenAlex, INSPIRE, ADS, Unpaywall coletados e unidos | ~150 GB | $0 |
| **S2 — Classificador de Física** | 1 (paralelo) | fastText treinado e calibrado a 0,95 de precisão (§6) | < 1 GB | $0 |
| **S3 — Bulk pré-processado** | 2–3 | RedPajama-arXiv, peS2o, OpenWebMath, proof-pile-2 baixados e filtrados | ~600 GB | $0 |
| **S3b — Auditoria de LaTeX** | 3 | Medição de preservação de equações vs. fonte original (§3.2) — **decide se o bulk pago se justifica** | — | $0 |
| **S4 — Domínio público** | 3–5 | NTRS, OSTI, NIST | ~2–4 TB (PDF) | $0 |
| **S5 — CC BY** | 5–6 | SCOAP³, DOAJ, SciPost, NJP, CERN CDS | ~200 GB | $0 |
| **S6 — Pedagógico** | 6–7 | StackExchange, OpenStax, clássicos de domínio público | ~100 GB | $0 |
| **S7 — Teses** *(pós-G1)* | 8–12 | CORE, NDLTD, repositórios institucionais | ~5–10 TB | $0 |
| **S8 — Código** *(pós-G1)* | 10–12 | The Stack v2 filtrado, notebooks | ~200 GB | $0 |

**Marco ao fim do S6: `PhysCorpus-Raw v0.1` — 20–35 B tokens brutos, custo zero, tudo com proveniência e licença resolvidas.** É a entrada do DOC-03 (parsing) e do DOC-04 (dedup), e o insumo do degrau T0 do DOC-17A §8.2.

**Requisito de disco:** ~8–15 TB no pico (dominado por PDFs de NTRS/OSTI e teses). Confirma o HD externo de 16 TB do DOC-17A.

---

## 10. Riscos e mitigações

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| RedPajama degradou o LaTeX além do aceitável | Média | Alto — ataca P2 e o achado central do Minerva | Auditoria S3b **antes** de qualquer compromisso; rota de contingência é o bulk S3 por ~$180 |
| Sobreposição entre agregadores maior que 55% | Média | Médio — corpus menor que o previsto | Fontes aditivas (NTRS, OSTI, teses) são independentes e cobrem a diferença |
| Bloqueio por coleta agressiva | Baixa | Alto — fonte perdida permanentemente | A5 imposto em código; taxas conservadoras; contato identificado |
| Lacuna de Física básica não compensada (§2) | **Alta** | **Alto** — atinge os modos de falha F1–F3 | Reconhecido explicitamente; compensação por dados sintéticos verificados é item de primeira classe no DOC-06, não um remendo |
| Licenças de teses irresolvíveis em escala | Alta | Baixo | Padrão conservador: `train_ok=True`, `redistributable=False`; teses são Camada B, pós-G1 |
| Mudança de postura do arXiv sobre uso para treino | Baixa | Alto | Gatilho de revisão do ADR-0001 §8; corpus reconstruível a partir de metadados |
| Volume de PDFs excede o disco | Média | Médio | NTRS/OSTI processados em fluxo: baixa → extrai texto → descarta PDF, mantendo só o hash |

---

## 11. Critérios de aceite do Stage-Gate 1

- [ ] **B1** — Toda fonte da Camada A possui `license.spdx_id` resolvido e `train_ok` decidido **antes** da coleta em massa
- [ ] **B2** — Espinha de metadados coletada; ≥ 99% dos papers de Física do arXiv com categoria e licença resolvidas
- [ ] **B3** — Classificador de Física atinge precisão ≥ 0,95 no conjunto de validação anotado
- [ ] **B4** — Auditoria S3b concluída, com decisão registrada sobre o bulk pago do arXiv
- [ ] **B5** — `PhysCorpus-Raw v0.1` reconstruível ponta a ponta a partir dos manifestos de aquisição (satisfaz G1.5)
- [ ] **B6** — Nenhum documento da Camada E presente; verificado por auditoria automatizada de proveniência
- [ ] **B7** — Contagem real de tokens medida e comparada à estimativa do §7; desvios > 30% investigados e explicados

---

## 12. Questões em aberto

| ID | Questão | Destino |
|---|---|---|
| OQ-6 | Qual a taxa real de preservação de LaTeX no RedPajama-arXiv? | S3b — mede e decide |
| OQ-7 | Textos livremente distribuídos pelos autores (Srednicki, Tong, Carroll) — cada um permite uso em treino? | Verificação individual durante o S6 |
| OQ-8 | O acesso institucional do sponsor cobre APS/Elsevier/Springer com programa de TDM? | Depende de informação do sponsor; reabre a Camada C |
| OQ-9 | Vale processar PDFs do NTRS/OSTI antes ou depois do ΦOCR estar pronto? | DOC-03 — decisão de ordenação entre parsing e OCR |

---

**Fim do DOC-02.** Revisão da §11 necessária antes do DOC-03 (Ingestão, Parsing e Normalização).
