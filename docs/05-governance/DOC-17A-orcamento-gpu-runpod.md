# DOC-17A — Análise de Custo-Benefício de GPU e Armazenamento (RunPod)

**Status:** `RASCUNHO v0.2` — **extrato parcial**. Cobre apenas horas de GPU de treino; não inclui processamento de dados, verificador, avaliação nem baselines. **A referência financeira vigente é o [DOC-17](DOC-17-orcamento-cronograma.md).**
**Depende de:** [DOC-00 §4.2, §7](../00-foundations/DOC-00-project-charter.md), [DOC-01 §7](../00-foundations/DOC-01-system-architecture.md)
**Data:** 2026-08-03

> **Aviso sobre preços.** Os valores de RunPod flutuam semanalmente e diferem entre Community Cloud e Secure Cloud, por região e por disponibilidade. Os números abaixo são faixas aproximadas para calibrar a decisão. **A metodologia é durável; os preços não são** — confirme as tarifas vigentes antes de comprometer orçamento. Todos os valores em dólares.

---

## 1. A pergunta certa não é "qual GPU tem melhor custo-benefício"

A métrica intuitiva é `$/hora`. Ela é enganosa. A métrica correta para treino é:

$$\text{custo por experimento} = \frac{6 \cdot N \cdot D}{\text{TFLOP/s efetivos}} \times \frac{\text{preço}}{\text{hora}} \times \frac{1}{3600}$$

onde `TFLOP/s efetivos = pico denso BF16 × MFU`. Isso dá **$/PFLOP-hora**, que é comparável entre placas.

Mas há uma restrição binária que domina tudo:

> **Se o modelo não cabe na VRAM, o preço por FLOP é irrelevante — o job simplesmente não roda.** E se ele só cabe distribuído em várias GPUs sem NVLink, o MFU real desaba para 15–25% e o `$/PFLOP-hora` nominal vira ficção.

Portanto a ordem de decisão é: **(1) VRAM decide o que é possível → (2) interconexão decide se multi-GPU é viável → (3) $/PFLOP-hora decide entre os candidatos sobreviventes.**

---

## 2. Tabela comparativa

Pico denso BF16 com acumulação em FP32 — o número realista de treino, não o número de marketing (que costuma incluir esparsidade 2:1 e/ou acumulação em FP16, inflando por 2× a 4×). Placas *consumer* e *workstation* (Ada, Ampere, Blackwell RTX) reduzem o throughput pela metade quando a acumulação é em FP32; placas de datacenter (A100, H100, H200, B200, MI300X) não.

| GPU | VRAM | Banda | Pico BF16 denso (treino) | MFU típ. | **TFLOP/s efetivos** | RunPod $/h (aprox.) | **$/PFLOP-h** | Veredito |
|---|---|---|---|---|---|---|---|---|
| RTX A5000 | 24 GB | 768 GB/s | ~55 | 0,35 | ~19 | 0,16–0,26 | **8–14** | Barato, mas lento e pouca VRAM |
| **RTX 4090** | 24 GB | 1008 GB/s | ~165 | 0,35 | **~58** | 0,34–0,69 | **6–12** | ★ Melhor valor absoluto em job de 1 GPU |
| RTX 5090 | 32 GB | 1792 GB/s | ~210 | 0,35 | ~74 | 0,69–0,94 | 9–13 | Boa; +8 GB sobre a 4090 importa |
| RTX A6000 | 48 GB | 768 GB/s | ~77 | 0,35 | ~27 | 0,33–0,76 | 12–28 | VRAM sem FLOPs — só se precisar dos 48 GB |
| A40 | 48 GB | 696 GB/s | ~75 | 0,35 | ~26 | 0,35–0,44 | 13–17 | Idem A6000, um pouco mais barata |
| L40S | 48 GB | 864 GB/s | ~91 | 0,40 | ~36 | 0,79–1,03 | 22–29 | ✗ Caro para o que entrega em treino |
| **A100 80 SXM** | 80 GB | 2039 GB/s | 312 | 0,48 | ~150 | 1,64–1,89 | **11–13** | ✗ **Valor ruim hoje — ver §3** |
| **H100 80 SXM** | 80 GB | 3350 GB/s | 990 | 0,45 | **~445** | 2,39–2,99 | **5–7** | ★ Melhor $/FLOP com interconexão séria |
| H200 141 SXM | 141 GB | 4800 GB/s | 990 | 0,48 | ~475 | 3,19–3,99 | 7–8 | ★ VRAM que muda o que cabe em 1 GPU |
| B200 180 | 180 GB | 8000 GB/s | ~2250 | 0,42 | ~945 | 5,98–6,99 | 6–7 | Ótimo, mas ociosidade custa caro |
| **MI300X** | 192 GB | 5300 GB/s | ~1307 | 0,35 | ~457 | ~2,49 | **~5** | ★ Ver §5 — a opção assimétrica |

*MFU de placas consumer é penalizado por ausência de NVLink, menor banda e limitação térmica; de MI300X, pela maturidade do ROCm. São estimativas — o DOC-17 exigirá um micro-benchmark de validação antes de qualquer compromisso grande.*

---

## 3. Três conclusões contra-intuitivas

### 3.1 A A100 deixou de ser boa compra

A100 80 GB SXM entrega ~150 TFLOP/s efetivos a ~$1,89/h → **~$12,6/PFLOP-h**.
H100 80 GB SXM entrega ~445 TFLOP/s efetivos a ~$2,79/h → **~$6,3/PFLOP-h**.

**A H100 custa 1,5× mais por hora e entrega 3× mais trabalho.** É *duas vezes mais barata* por unidade de computação, e termina o run em um terço do tempo. A A100 só se justifica se for a única coisa disponível, ou se a diferença de preço na sua região for atípica. Regra prática: **nunca alugue A100 para treino se houver H100 no catálogo.**

### 3.2 A RTX 4090 empata com a H100 em $/FLOP — e isso é a favor da H100

RTX 4090: ~$6–12/PFLOP-h. H100 SXM: ~$5–7/PFLOP-h. Na prática **empatam**.

Como o custo por unidade de trabalho é o mesmo, a decisão se desloca para o que *não* está no preço:

| Dimensão | RTX 4090 | H100 SXM |
|---|---|---|
| VRAM | 24 GB | 80 GB |
| Multi-GPU | P2P/NCCL desabilitado em Ada consumer — escalonamento ruim | NVLink 900 GB/s |
| Tempo até o resultado | ~8× mais lento | referência |
| Disponibilidade em lote | alta (Community) | variável |

**Portanto:** 4090 para trabalho *embaraçosamente paralelo* de 1 GPU (OCR em lote, geração de embeddings, treino de encoder, ablações pequenas). H100 sempre que o modelo precisar de mais de 24 GB ou o prazo importar. Custa o mesmo; escolha pelo formato do trabalho, não pelo preço.

### 3.3 O maior desperdício de dinheiro não é a GPU errada — é o run perdido

Um CPT de 8B em 40B tokens custa ~$3.300 e leva ~300 horas em 4× H100. Um *loss spike* não detectado na hora 200, ou um bug no dataloader descoberto no fim, queima $2.200 de uma vez. Isso é maior que qualquer diferença entre placas nesta análise.

Consequência de projeto, já embutida no DOC-01 §8: checkpoint a cada ≤ 15 minutos, detecção automática de spike com rollback, e a regra de que **nenhuma configuração escala para um run longo antes de rodar 500 passos em escala reduzida com curva de loss saudável**. Isso vale mais dinheiro que a escolha de hardware.

---

## 4. Mapeamento carga de trabalho → GPU

Contas de memória para full fine-tune com AdamW em precisão mista (bf16 + mestre fp32): **16 bytes por parâmetro** (2 params bf16 + 2 grads bf16 + 4 mestre fp32 + 4 momento + 4 variância), mais ativações. Com otimizador de 8 bits (`AdamW8bit`), cai para ~10 bytes/param.

| Carga | Memória necessária | GPU recomendada | Custo estimado |
|---|---|---|---|
| **Parsing LaTeX, filtro, dedup** (CPU) | — | **Não usar GPU.** Servidor dedicado CPU (Hetzner AX-series, ~€60–120/mês) ou pods CPU do RunPod | ~$70–140/mês |
| **ΦOCR em lote** (~7,5M páginas) | < 10 GB | 1–2× **RTX 4090** interruptible | ~$180–900 |
| **Embeddings p/ dedup semântico** (~10¹⁰ tokens) | < 8 GB | 1× **RTX 4090** | ~$10–20 |
| **ΦEnc 400M** (MLM, ~40B tokens ×2–3 passes) | ~7 GB + ativações | 1× **H100** (60–180 h) ou 1× 4090 (460–1400 h) | ~$400–500 nos dois casos |
| **ΦEmb contrastivo** | batch grande domina → VRAM | 1× **H100 80 GB** ou 8× com GradCache | ~$300–600 |
| **ΦGen-1,5B CPT** | 24 GB (16 GB c/ 8-bit) | 1× **H100** ou 1× 5090 c/ 8-bit | ~$400–800 |
| **ΦGen-8B CPT full** (40B tokens) | **128 GB** + ativações | **4× H100 SXM** (~300 h) — ou ver §5 | **~$3.350** (~$1.700 spot) |
| **ΦGen-8B SFT** | 128 GB | 4× H100, poucas horas | ~$200–400 |
| **ΦGen-8B RLVR/GRPO** | política + referência + motor de rollout | **8× H100 SXM** | ~$1.700–4.200 |
| **Avaliação / serving** | 8B quantizado ≈ 6–10 GB | **RTX 4090** ou RunPod Serverless | ~$50–200/mês |

> **Nota técnica importante — não use LoRA para o CPT.** É tentador (8B em 4-bit cabe numa 4090 por $0,44/h). Mas Biderman et al. (2024), *"LoRA Learns Less and Forgets Less"*, mostram que LoRA fica sistematicamente atrás de full fine-tuning justamente em *continual pretraining para um domínio novo* — que é exatamente o nosso caso. LoRA é para adaptação de estilo e tarefa, não para absorver conhecimento de domínio. Economizar aqui compromete o Portão G2.1, que é o número de manchete do projeto. **Full fine-tune, sem exceção, para o CPT.**

---

## 5. A opção assimétrica: uma única GPU de VRAM enorme

Este é o achado mais acionável para uma equipe pequena.

Um CPT full de 8B precisa de ~128 GB de estado de otimizador. As configurações possíveis:

| Configuração | VRAM total | TFLOP/s | $/h | Horas p/ 40B tokens | **Custo total** | Complexidade |
|---|---|---|---|---|---|---|
| 4× H100 80 GB (NVLink) | 320 GB | ~1780 | ~11,16 | ~300 | **~$3.350** | FSDP multi-GPU, sharding, comunicação |
| 2× H100 80 GB | 160 GB | ~890 | ~5,58 | ~600 | **~$3.350** | FSDP, memória apertada |
| 1× H200 141 GB (+ AdamW8bit → 80 GB) | 141 GB | ~475 | ~3,59 | ~1.120 | **~$4.020** | **Nenhuma — é single-GPU** |
| **1× MI300X 192 GB** | 192 GB | ~457 | ~2,49 | ~1.170 | **~$2.910** | Single-GPU, mas atrito de ROCm |

**Leitura.** Uma única MI300X de 192 GB comporta o fine-tune completo de um modelo 8B **sem nenhum paralelismo**, pelo menor custo total da tabela. Isso elimina de uma vez: sharding FSDP, depuração de NCCL, dependência de topologia NVLink e falhas parciais de nó — a maior fonte de horas de engenharia perdidas em projetos pequenos.

**Contrapartida honesta:** o ecossistema ROCm tem defasagem em relação ao CUDA. PyTorch, vLLM e TorchTitan funcionam em ROCm, mas com menos cobertura de testes, kernels ocasionalmente mais lentos que o pico teórico sugere, e bibliotecas de nicho (alguns kernels do Flash Attention, `bitsandbytes`, alguns caminhos do veRL) exigindo contorno. O MFU de 0,35 que assumi já embute parte disso, e ainda assim ela vence em custo.

**Recomendação:** rodar um *spike* de 4 horas (~$10) em uma MI300X assim que o Tier 1 terminar, medindo tokens/s reais no nosso código. Se o número bater, o Tier 2 inteiro fica ~15% mais barato e substancialmente mais simples. Se não bater, cai-se para 4× H100 sem perda. **Custo do experimento: $10. Valor da informação: alto.** Este é exatamente o tipo de decisão que não deve ser tomada por intuição.

---

## 6. Armazenamento: o custo que passa despercebido

**Este é o erro mais caro que se comete com RunPod**, e é maior que qualquer diferença entre GPUs no Tier 1.

Volumes de rede do RunPod custam da ordem de **$0,05–0,07/GB/mês**. O corpus bruto deste projeto (tarballs do arXiv, PDFs de teses, relatórios de agências) fica em **10–20 TB**.

> 15 TB em volume de rede do RunPod ≈ **$750–1.000/mês** ≈ **$9.000–12.000/ano** — mais que todo o orçamento de GPU do Tier 1.

### Arquitetura correta de armazenamento

A observação decisiva é que **os três estágios do dado têm tamanhos radicalmente diferentes:**

| Estágio | Tamanho | Padrão de acesso | Onde colocar | Custo/mês |
|---|---|---|---|---|
| **Bruto** (tarballs, PDFs) | 10–20 TB | Lido 1–3 vezes, nunca no laço de treino | **Backblaze B2** (~$0,006/GB/mês) ou **Cloudflare R2** (~$0,015/GB/mês, egress zero) | **$60–300** |
| **Processado** (Parquet/Iceberg) | ~300–600 GB | Lido a cada reconstrução de mistura | **Cloudflare R2** — egress zero é decisivo aqui | **~$10** |
| **Shards tokenizados** | **~160–200 GB** | Lido a cada época, alta vazão | **NVMe local do pod** (incluído no preço da GPU); baixar uma vez no início | **$0** |
| **Checkpoints** (8B c/ otimizador ≈ 128 GB cada) | ~640 GB (5 retidos) | Escrito a cada 15 min, lido em retomada | Volume de rede RunPod (pequeno) + push assíncrono para R2 | **~$40** |

**Total: ~$110–350/mês**, contra $750–1.000/mês na abordagem ingênua. **Economia de ~$7.000–10.000/ano** por uma decisão de arquitetura que custa um dia de trabalho.

O ponto não óbvio: **texto comprime brutalmente.** 40 bilhões de tokens em `uint32` são apenas 160 GB. O dado *quente* de treino é pequeno; o dado *frio* de origem é que é grande. Confundir os dois é o que gera a conta de $12 mil.

> **Por que Cloudflare R2 e não S3:** o pipeline relê o corpus processado a cada reconstrução de mistura e a cada retreino do tokenizer — dezenas de TB de egress ao longo do projeto. No S3 isso custa $0,09/GB (≈ $900 por 10 TB baixados). No R2, **egress é gratuito**. Para um padrão de "escreve uma vez, lê muitas em outro provedor de compute", R2 é a escolha estruturalmente correta.

### 6.1 Opção de custo zero: processar localmente, alugar só a GPU

A tabela acima assume que o corpus vive na nuvem. **Ela não precisa assumir isso**, e para um orçamento apertado a arquitetura correta é outra.

A observação que libera tudo: **os estágios do dado diferem por duas ordens de grandeza em tamanho, e apenas o menor precisa estar perto da GPU.**

```
  [ Máquina local ]                                  [ Pod RunPod alugado por hora ]

  Bruto 10–20 TB   ──parse──▶  Processado ~400 GB  ──tokeniza──▶  Shards 20–160 GB
  (HD externo)                 (disco local)                       (upload 1×)  │
                                                                                ▼
                                                                       treino na GPU
                                                                                │
                                                                    checkpoints ◀┘
                                                                    (download ~1–15 GB)
```

**Todo o pipeline de dados — coleta, parsing de LaTeX, filtragem, deduplicação, treino do tokenizer — é trabalho de CPU, RAM e disco. Nada disso precisa de GPU.** Roda na sua máquina, em segundo plano, ao longo de semanas, a custo zero. O corpus bruto nunca sai do HD externo.

O que sobe para a nuvem são apenas os **shards tokenizados**: 20 B tokens em `uint32` = **80 GB**; um subconjunto inicial de 5 B tokens = **20 GB**. Upload único de 1 a 4 horas numa conexão doméstica típica. Depois disso, a GPU alugada lê do disco efêmero do próprio pod, que já está incluído no preço da hora.

| Item | Custo na abordagem "tudo na nuvem" | Custo na abordagem local |
|---|---|---|
| Corpus bruto 15 TB | $60–1.000/mês | **HD externo 16 TB, ~$200 uma única vez** (ou disco que você já tem) |
| Processado ~400 GB | ~$10/mês | **$0** — disco local |
| Shards de treino | ~$10/mês | **$0** — disco efêmero do pod, já incluído na hora de GPU |
| Checkpoints | ~$40/mês | **$0–5** — baixar ao fim de cada run; opcionalmente 1 volume de rede pequeno |
| **Total recorrente** | **$110–350/mês** | **≈ $0/mês** |

**Limitações honestas desta abordagem:**

1. **Retomada de run interrompido.** Sem volume de rede persistente, um pod spot que morre leva o checkpoint junto. Mitigação: gravar checkpoint a cada ~15 min direto para um bucket B2/R2 (o checkpoint de um modelo 150M é ~2 GB; de um 1,5B, ~18 GB — upload rápido). Custo de um bucket só para checkpoints: **~$1–3/mês**. Vale a pena; não abra mão disso.
2. **Retrabalho.** Se você precisar retokenizar com vocabulário diferente, refaz o upload. Mitigação: congelar o tokenizer antes do primeiro upload grande.
3. **Banda doméstica.** Upload de 80 GB a 50 Mbps leva ~3,5 h. Aceitável para evento único, ruim se for semanal. Mitigação: começar com o subconjunto de 5 B tokens (20 GB).

**Veredito: para orçamento restrito, processe localmente e alugue apenas a GPU.** O custo de armazenamento recorrente cai de $110–350/mês para essencialmente zero, ao preço de um HD externo e de alguma disciplina de checkpoint.

---

## 7. Táticas específicas de RunPod

| Tática | Ganho | Custo / risco |
|---|---|---|
| **Pods interruptible (spot)** | ~40–50% de desconto | Exige checkpoint ≤ 15 min + auto-resume — já obrigatório pelo DOC-01 §8. **Use para todo treino.** |
| **Community Cloud** para 1 GPU | 30–50% mais barato que Secure | Topologia e confiabilidade variáveis. Ótimo para OCR/embedding/encoder; **evite para multi-GPU** |
| **Secure Cloud** para multi-GPU | NVLink verificado, hardware de datacenter | ~1,3–1,5× o preço. **Obrigatório para o CPT de 8B** |
| **Verificar SXM vs PCIe** | H100 SXM tem NVLink 900 GB/s; PCIe não | H100 PCIe em FSDP de 4 GPUs pode perder 30–50% de MFU. **Sempre confirme SXM** |
| **Serverless** para avaliação/inferência | Paga só o que usa; escala a zero | Cold start. Ideal para o harness de avaliação, que é intermitente |
| **Reserva mensal** para o CPT | Costuma bater o preço horário em runs longos | Compromisso de prazo; só faz sentido depois do G1 |
| **Templates com imagem própria** | Elimina 10–20 min de setup por pod | Manter o Dockerfile do DOC-01 §5.8 versionado por digest |

**Comparação honesta com alternativas:** Vast.ai costuma ser mais barato que o Community Cloud do RunPod, com variância maior de confiabilidade e de topologia — aceitável para trabalho em lote reiniciável, ruim para runs longos. Lambda, Nebius, DataCrunch e Together têm preços competitivos de H100 com hardware mais previsível, e normalmente vencem em compromissos reservados de um mês ou mais. **Recomendação: RunPod como plataforma principal pela ergonomia (spot + serverless + volumes de rede em um só lugar), com cotação comparativa antes de qualquer compromisso acima de ~$5.000.**

---

## 8. Escada de orçamento mínimo

A estimativa original de $8–14 mil embutia três decisões que **não são necessárias** e que inflavam o custo por quase uma ordem de grandeza. Corrigidas, o programa muda de figura.

### 8.1 As três correções

| Suposição inflada | Correção | Economia |
|---|---|---|
| Corpus na nuvem | Processar localmente, subir só os shards (§6.1) | **$110–350/mês → ~$0** |
| Pipeline de OCR desde o início | **O arXiv fornece a fonte LaTeX**, não só PDF. OCR só é necessário para teses, livros e relatórios — tudo adiável para depois do Portão G1 | **$180–900 → $0** |
| Puxar o bulk do arXiv do S3 (egress pago) | Subconjuntos de arXiv já processados e **gratuitos** existem no HuggingFace: `RedPajama-Data-1T` (fatia arXiv, ~28 B tokens de LaTeX), `proof-pile-2`, `peS2o`. Filtrando para Física: **~10–15 B tokens, custo zero de egress** | **~$180 → $0** |

> **Consequência para o DOC-02.** O plano de aquisição precisa ser reordenado: **primeiro o que é gratuito e já processado** (RedPajama-arXiv, NASA NTRS que é domínio público, dumps do Physics StackExchange em CC BY-SA, OpenStax, clássicos de domínio público via Internet Archive), e só depois o bulk pago do arXiv, quando houver justificativa. Isso não é um rebaixamento do plano: são ~15–20 B tokens de Física por US$ 0, suficientes para todo o Tier 1 e para o CPT de 1,5 B.
>
> Contrapartida honesta: corpora pré-processados por terceiros violam parcialmente o princípio P2 (DOC-01 §1) — o pipeline de LaTeX deles não é o nosso, e parte da estrutura de equações pode já ter sido degradada. Estratégia: usá-los como **base inicial**, medir a qualidade da preservação de LaTeX contra uma amostra que processemos nós mesmos, e reprocessar do zero só se a medição justificar.

### 8.2 A escada, degrau a degrau

Cada degrau é uma entrega independente. Você pode parar em qualquer um.

| Degrau | Entrega | GPU | **Custo** | Wall-clock |
|---|---|---|---|---|
| **T0 — Corpus** | `PhysCorpus-Open` + tokenizer de Física. **Publicável sem nenhum modelo.** | nenhuma | **$0** | 3–6 semanas locais |
| **T1a — ΦEnc + ΦEmb** | Encoder 150M + modelo de embedding → **Portão G1: bater o PhysBERT** | 1× 4090 (129 h) ou 1× H100 (17 h) | **$25–90** | 1–5 dias |
| **T1b — ΦRank + índice** | Reranker + busca híbrida → sistema de recuperação de Física completo | 1× 4090 | **+$10–30** | 2–3 dias |
| **T2a — ΦGen-1,5B** | CPT sobre 15 B tokens → primeiro modelo generativo de Física | **1× H100** (84 h) | **+$120–240** | 4–7 dias |
| **T2b — SFT + RLVR no 1,5B** | Raciocínio verificado pelo barramento de verificação | 1× H100 em janelas | **+$235** | ~1 semana |
| **T2c — ΦGen-8B** | Modelo competitivo com abertos de porte médio | 4× H100 (150 h) ou 1× MI300X | **+$850–1.700** | 1–3 semanas |

**Marcos de custo acumulado:**

- **Até o Portão G1 (resultado publicável, bate o PhysBERT): US$ 25 – 120.**
- **Até um modelo generativo de Física funcional (T2b): US$ 300 – 600.** *(revisado pelo [DOC-09 §8](../02-models/DOC-09-pos-treino-sft-dpo-rlvr.md#8-orçamento): a varredura de coeficiente de KL e a geração de dados de SFT não estavam contabilizadas)*
- Até o ΦGen-8B (T2c): US$ 1.100 – 2.260.

Custos únicos e recorrentes fora da GPU: **HD externo de 8–16 TB, ~$150–250** (ou o disco que você já tiver), e **~$1–3/mês** de bucket B2/R2 exclusivo para checkpoints — este último não é opcional, é o que impede um pod spot interrompido de destruir dias de treino.

### 8.3 Configuração recomendada para o orçamento mínimo

| Etapa | O que alugar | Como |
|---|---|---|
| T0 | **nada** | Tudo local |
| T1a, T1b | **1× RTX 4090**, Community Cloud, interruptible | Mais barato por FLOP; o encoder cabe folgado em 24 GB |
| T2a em diante | **1× H100 SXM 80 GB**, interruptible | Pelo mesmo dinheiro que a 4090 por unidade de trabalho, entrega o resultado 8× mais rápido — e a 4090 não comporta um 1,5 B em full fine-tune sem contorções |
| T2c | **4× H100 SXM** ou **1× MI300X** | Decidir pelo spike de $10 do §5 |

O ponto do §3.2 aparece aqui na prática: **4090 e H100 custam o mesmo por trabalho realizado.** Portanto o critério não é preço — é se o modelo cabe. Abaixo de 24 GB, 4090. Acima, H100. Nunca A100.

---

## 9. O que ainda precisa ser medido

Esta análise é aritmética sobre especificações publicadas, não medição. Antes de comprometer valores acima de ~$1.000, o DOC-17 exigirá:

1. **Micro-benchmark de MFU** do nosso próprio código de treino em 4090, H100 e MI300X — tokens/s reais, não pico teórico.
2. **Medição de throughput do ΦOCR** em PDFs reais de Física (a estimativa de $180 vs $900 varia por 5×, e depende inteiramente da arquitetura escolhida no DOC-03).
3. **Teste de escalonamento FSDP** em 2/4/8 H100 para medir a eficiência real de multi-GPU antes de dimensionar o run de CPT.
4. **Taxa de interrupção observada** em pods spot na região escolhida, para calibrar a frequência de checkpoint.

Cada um custa menos de $30 e cada um pode alterar o orçamento em milhares. São executados no fim do Tier 1.

---

**Fim do DOC-17A.**
