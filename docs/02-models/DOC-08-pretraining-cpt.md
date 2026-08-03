# DOC-08 — Infraestrutura de Pretraining e Continual Pretraining

**Projeto:** ΦFM — Phi Foundation Models para Física
**Status:** `RASCUNHO v0.1` — aguardando revisão do Stage-Gate 7
**Cobre:** entregáveis solicitados **7** (pré-treinamento) e **9** (continual pretraining)
**Depende de:** [DOC-01 §7](../00-foundations/DOC-01-system-architecture.md), [DOC-06](../01-data/DOC-06-mistura-curriculo-dados-sinteticos.md), [DOC-07](DOC-07-familia-de-modelos.md), [DOC-17A](../05-governance/DOC-17A-orcamento-gpu-runpod.md)
**Data:** 2026-08-03

---

## 1. Duas cargas de trabalho que quase nada compartilham

O documento cobre dois regimes que são frequentemente tratados como um só, e não são:

| | **Pretraining do zero** (ΦEnc) | **Continual pretraining** (ΦGen) |
|---|---|---|
| Ponto de partida | Pesos aleatórios | Modelo já competente |
| Objetivo | MLM | Próximo token |
| Risco dominante | Não convergir | **Destruir o que já funciona** |
| Taxa de aprendizado | Alta (~1e-3) | **Baixa (~3e-5)** — §5.2 |
| Métrica que define sucesso | Perda de validação | **Delta contra a base, sem regressão** (G2.1 + G2.3) |
| Hardware | 1 GPU | 1–4 GPUs |
| Custo | US$ 25–90 | US$ 120–1.700 |

> **O risco assimétrico do CPT.** No pretraining do zero, um hiperparâmetro ruim custa uma execução. No CPT, ele **destrói capacidades que custaram trilhões de tokens** a quem treinou a base — e o dano é frequentemente invisível na curva de perda, porque a perda em texto de Física melhora enquanto a capacidade geral desmorona. É por isso que o critério G2.3 (sem regressão > 2 pontos em benchmarks gerais) é **desclassificatório**, não uma métrica secundária.

---

## 2. Paralelismo e a costura de portabilidade

Reafirmando o princípio **P5** (DOC-01 §1): `src/phifm/models/` define `nn.Module` puros, sem consciência de distribuição. Todo sharding é aplicado por `src/phifm/training/parallel/` no momento do wrap.

| Modelo | Estratégia | Config |
|---|---|---|
| ΦEnc-150M | GPU única; DDP se houver mais de uma | `dp_shard: 1, tp: 1` |
| ΦGen-1,5B | GPU única com **AdamW de 8 bits**, ou FSDP2 em 2 | `dp_shard: 1–2` |
| ΦGen-8B | **FSDP2** em 4× H100 — ou GPU única em MI300X (§3.2) | `dp_shard: 4, tp: 1` |
| Tier 3 (32B+) | FSDP2 + TP; migração para Megatron-Core | `dp_shard: 8, tp: 8, pp: 4` |

**Tensor parallel não é usado nos Tiers 1 e 2.** TP só compensa quando o modelo não cabe com FSDP puro ou quando a comunicação intra-nó é rápida o bastante para amortizar o all-reduce por camada. Em 4 GPUs com NVLink, FSDP2 puro é mais simples e igualmente rápido. Adicionar TP aqui seria complexidade sem retorno.

---

## 3. Memória: a conta que decide o hardware

### 3.1 Contabilidade por parâmetro

Full fine-tune com AdamW em precisão mista:

| Componente | Bytes/param | 1,5 B | 8 B |
|---|---|---|---|
| Parâmetros bf16 | 2 | 3,0 GB | 16 GB |
| Gradientes bf16 | 2 | 3,0 GB | 16 GB |
| Mestre fp32 | 4 | 6,0 GB | 32 GB |
| Momento fp32 (`m`) | 4 | 6,0 GB | 32 GB |
| Variância fp32 (`v`) | 4 | 6,0 GB | 32 GB |
| **Subtotal (16 B/param)** | **16** | **24 GB** | **128 GB** |
| Ativações (ctx 4096, checkpointing) | — | ~4–8 GB | ~12–24 GB |
| **Total** | | **~30 GB** | **~145 GB** |

Com **AdamW de 8 bits** (`m` e `v` quantizados), cai para 10 bytes/param: **15 GB** e **80 GB**.

### 3.2 Configurações viáveis

| Modelo | Configuração | Cabe? |
|---|---|---|
| ΦGen-1,5B | 1× RTX 4090 (24 GB) + AdamW8bit + checkpointing | ✅ Apertado |
| ΦGen-1,5B | **1× H100 (80 GB), AdamW pleno** | ✅ **Folgado — recomendado** |
| ΦGen-8B | 1× H100 (80 GB) | ❌ 145 GB não cabem |
| ΦGen-8B | 2× H100 FSDP2 | ⚠️ 72 GB/GPU — muito apertado |
| ΦGen-8B | **4× H100 FSDP2** | ✅ **~36 GB/GPU — recomendado** |
| ΦGen-8B | **1× MI300X (192 GB), sem paralelismo** | ✅ **Cabe inteiro — ver DOC-17A §5** |

A linha da MI300X é a mais interessante para uma equipe pequena: elimina FSDP, NCCL e dependência de topologia NVLink. O spike de US$ 10 do DOC-17A §5 decide.

---

## 4. Pretraining do zero — ΦEnc

| Hiperparâmetro | Valor | Justificativa |
|---|---|---|
| Otimizador | AdamW, `β = (0,9, 0,98)`, `ε = 1e-6` | `β₂ = 0,98` é padrão em encoders |
| Taxa de aprendizado de pico | 1e-3 | Modelo pequeno tolera LR alta |
| Agendamento | Warmup 3% → **trapezoidal** (WSD) | ★ Ver abaixo |
| Weight decay | 0,01 (sem decay em norms e bias) | |
| Batch (tokens) | ~2 M por passo | |
| Clipping de gradiente | 1,0 | |
| Precisão | bf16, mestre fp32 | |
| Máscara MLM | **30%** | ModernBERT: 15% do BERT é subótimo |
| Comprimento de sequência | 8.192, com unpadding | DOC-07 §2.1 |

> **Por que agendamento trapezoidal (Warmup-Stable-Decay) e não cosseno.** O cosseno exige fixar o número total de passos **antes** de começar. O WSD mantém LR constante e decai só no fim — o que permite (a) parar em qualquer ponto com um decay curto e obter um modelo utilizável, (b) retomar e estender o treino sem descartar o agendamento, e (c) fazer a fase de annealing do DOC-06 §4.2 coincidir com o decay. Para um projeto com orçamento incerto e execução em GPU interrompível, essa flexibilidade vale mais que o ganho marginal do cosseno.

---

## 5. Continual pretraining — ΦGen

A parte difícil.

### 5.1 O dilema central

```
LR alta demais  →  adaptação de domínio boa, capacidade geral destruída   →  falha G2.3
LR baixa demais →  capacidade geral preservada, nenhuma Física aprendida  →  falha G2.1
```

Não há como satisfazer os dois critérios por acaso. A LR de pico é **o hiperparâmetro de maior alavancagem de todo o programa**.

### 5.2 A receita

Ibrahim et al. (2024) estabelecem o protocolo canônico, com duas componentes:

**(1) Re-warmup seguido de re-decay.** Não continuar do agendamento antigo, nem começar com LR constante. Aquecer até um pico reduzido e decair.

| Parâmetro | Valor proposto | Observação |
|---|---|---|
| **LR de pico** | **3e-5** (faixa a varrer: 1e-5 a 1e-4) | ~10% da LR de pretraining original |
| Warmup | 1–2% dos passos | Curto — o modelo já está em bom ponto |
| Decay | Cosseno até 10% do pico | |
| Weight decay | Igual ao da base | Mudar isso desestabiliza |

**(2) Replay da distribuição original.** É a componente que a maioria dos projetos omite, e é o que preserva capacidade geral.

> **O problema prático: não temos os dados de pretraining do Qwen3.** São privados. Não dá para fazer replay do que não se tem.
>
> **Solução:** replay por *proxy* — uma amostra de corpus geral de alta qualidade (FineWeb-Edu ou equivalente) com caráter semelhante. Não é idêntico, mas Ibrahim et al. mostram que **replay aproximado captura a maior parte do benefício**. O DOC-06 §2.3 já reserva 1% para texto geral; **durante o CPT esse peso sobe para 5%**, mais os 5% de código que já estão na mistura.

Custo do replay: ~5% dos tokens de treino. É o seguro mais barato do programa contra a falha desclassificatória G2.3.

### 5.3 A varredura de LR não é opcional

**Protocolo:** quatro valores de LR de pico — 1e-5, 3e-5, 5e-5, 1e-4 — treinando o ΦGen-1,5B por ~1 B tokens cada.

| | |
|---|---|
| Custo por execução | `6 × 1,5e9 × 1e9 = 9e18` FLOPs → ~5,6 h numa H100 → ~US$ 16 |
| **Custo total da varredura** | **~US$ 63** |
| O que mede | Perda em Física **e** regressão em benchmarks gerais, simultaneamente |
| Critério de escolha | Maior ganho em Física **sujeito a** regressão geral < 2 pontos |

US$ 63 para calibrar o hiperparâmetro que decide entre passar e falhar o Portão G2 é o melhor uso de dinheiro do programa inteiro.

> **Ressalva honesta sobre transferência de escala.** A LR ótima varia com o tamanho do modelo, e não podemos aplicar μP (Maximal Update Parametrization) a um Qwen3 já treinado. A varredura em 1,5 B **não transfere exatamente** para 8 B. Mitigação: aplicar a regra empírica `LR ∝ 1/√(largura)` como ponto de partida, e validar com uma execução curta de 500 passos em 8 B antes do treino longo. Custo da validação: ~US$ 30. Alegar que a transferência é exata seria falso.

### 5.4 Ordem das fases

| Fase | Tokens | Contexto | LR | Mistura |
|---|---|---|---|---|
| 1 · CPT principal | 85% | 4.096 | Pico → 20% | DOC-06 §2.3 + replay 5% |
| 2 · Extensão de contexto | 3% | **32.768** | 20% → 15% | Documentos longos |
| 3 · **Annealing** | 12% | 4.096 (amostras longas mantidas) | 15% → ~0 | DOC-06 §4.2 |

O annealing vem **por último** e com LR próxima de zero — é a fase que mais determina o comportamento final por unidade de computação.

---

## 6. Estabilidade numérica

### 6.1 Detecção de spike com rollback automático

Um *loss spike* não detectado na hora 200 de uma execução de 300 horas queima US$ 2.200 (DOC-17A §3.3). A automação disso vale mais que qualquer escolha de hardware.

**Sinais monitorados a cada passo:** perda, norma do gradiente, norma dos pesos, norma das ativações por camada, razão LR/perda.

**Critério de spike:** perda excede `μ + 4σ` da janela móvel de 100 passos, **ou** norma do gradiente excede 10× a mediana móvel.

**Resposta automatizada:**
1. Interromper e registrar o batch ofensor (identificado por `(seed, step)` — ver §7.2)
2. Retomar do último checkpoint saudável
3. **Pular** os batches da janela suspeita
4. Reduzir a LR em 50% por 500 passos, depois restaurar
5. Se ocorrerem 3 spikes em 5.000 passos, **parar e exigir intervenção humana** — é sintoma de problema sistêmico, não de um batch ruim

O PaLM documenta esse procedimento executado manualmente. Automatizá-lo é engenharia de algumas centenas de linhas e evita perder execuções inteiras durante o sono.

### 6.2 Medidas preventivas

| Medida | Efeito | Custo |
|---|---|---|
| **z-loss** (`1e-4 · log²Z`) | Impede a partição do softmax de derivar; estabilizador padrão desde o PaLM | Desprezível |
| **QK-norm** | Normaliza query e key antes da atenção; elimina uma classe inteira de instabilidade | Pequeno |
| Clipping de gradiente em 1,0 | Contenção de emergência | Zero |
| fp32 em softmax, layernorm e perda | Evita underflow em bf16 | Pequeno |
| Batch grande | Gradiente menos ruidoso | — |

No CPT do ΦGen essas medidas só se aplicam se a base já as tiver — **adicionar QK-norm a um modelo que não foi treinado com ela mudaria a função e destruiria os pesos**. Verificar a arquitetura do Qwen3 antes de habilitar qualquer uma. Para o ΦEnc, que é do zero, todas entram desde o início.

---

## 7. Tolerância a falhas em GPU interrompível

O DOC-17A recomenda pods *spot* (40–50% de desconto). Isso só é seguro com a infraestrutura desta seção.

### 7.1 Checkpointing

| Requisito | Implementação |
|---|---|
| Frequência | **≤ 15 minutos** de trabalho em risco |
| Escrita | **Assíncrona** — o treino não bloqueia esperando I/O |
| Destino | NVMe local → upload assíncrono para bucket B2/R2 (~US$ 1–3/mês) |
| Retenção | Últimos 3 + 1 a cada 10% do treino |
| Conteúdo | Pesos, estado do otimizador, **estado do dataloader**, RNG, passo, config |
| Tamanho | 1,5 B: ~18 GB · 8 B: ~128 GB |

### 7.2 O detalhe que costuma ser feito errado: retomada do dataloader

Retomar os pesos é fácil. Retomar **a posição exata no fluxo de dados** é onde a maioria das implementações falha silenciosamente — e o efeito é revisitar ou pular dados, quebrando a política de épocas do DOC-06 §2.4 e tornando a execução irreprodutível.

**Solução: ordenação de dados puramente determinística a partir de `(seed, step)`.** O dataloader não guarda estado — ele **calcula** qual shard e qual offset correspondem ao passo `n`. Retomar do passo 60.000 produz exatamente a mesma sequência que produziria uma execução contínua.

Benefícios adicionais: identifica o batch ofensor de um spike pelo número do passo (§6.1), e permite reproduzir um bug de dados sem reexecutar nada antes dele.

### 7.3 Retomada automática

Detectar preempção (sinal do provedor, quando disponível), subir pod novo, baixar o último checkpoint, retomar. Meta: **< 10 minutos** de interrupção por preempção, sem intervenção humana.

---

## 8. Extensão de contexto longo

Fase 2 da §5.4. Derivações de Física são longas; treinar só em 4.096 significa nunca ver uma derivação completa.

| Método | Como | Veredito |
|---|---|---|
| **YaRN** (Peng et al., 2024) | Interpolação de RoPE por banda de frequência + escala de atenção | ✅ **Preferido** — melhor preservação de contexto curto |
| Position Interpolation | Comprime posições linearmente | Simples, degrada mais contexto curto |
| NTK-aware scaling | Ajusta a base do RoPE | Intermediário; sem treino adicional |
| Treinar longo desde o início | — | ❌ Atenção quadrática torna proibitivo |

**Dados da fase:** papers completos, teses, e derivações sintéticas longas do gerador G2 (DOC-06 §5.2).

> **Verificação obrigatória: regressão em contexto curto.** Extensão de contexto longo frequentemente degrada desempenho em contexto curto — e como a maior parte da avaliação é em contexto curto, isso apareceria como falha inexplicada dos portões. **Avaliar em 512, 2.048 e 4.096 antes e depois da fase 2.** Degradação > 1 ponto reverte a extensão.

---

## 9. Observabilidade

### 9.1 O que registrar

| Categoria | Métricas | Frequência |
|---|---|---|
| Otimização | Perda, norma do gradiente, LR, norma dos pesos | Todo passo |
| Vazão | Tokens/s, **MFU**, utilização de GPU, tempo de I/O | Todo passo |
| Memória | Pico alocado, fragmentação | A cada 100 passos |
| **Validação por subárea** ★ | Perda de validação **separada para cada uma das 23 subáreas** | A cada 1.000 passos |
| Validação por fonte | Perda por fonte de dados | A cada 1.000 passos |
| **Regressão geral** | MMLU, HumanEval, IFEval em subconjunto rápido | A cada 5.000 passos |
| Avaliação de Física | Proxy barato do PhysBench | A cada 5.000 passos |

### 9.2 As duas métricas que realmente comandam decisões

**Perda de validação por subárea.** Se `hep-th` estagna enquanto `astro-ph` continua melhorando, a mistura está errada — e isso é **corrigível durante o treino**, ajustando pesos de amostragem. Uma perda agregada esconde exatamente essa informação. É a métrica com maior valor por unidade de esforço de instrumentação.

**Regressão geral a cada 5.000 passos.** É o sistema de alarme do critério desclassificatório G2.3. Detectar esquecimento catastrófico no passo 20.000 custa US$ 200; detectá-lo no fim custa a execução inteira.

### 9.3 Metas de MFU — e o portão que elas impõem

| Configuração | MFU alvo |
|---|---|
| ΦEnc em 1× RTX 4090 | ≥ 35% |
| ΦGen-1,5B em 1× H100 | ≥ 40% |
| ΦGen-8B em 4× H100 FSDP2 | ≥ 40% |
| ΦGen-8B em 1× MI300X | ≥ 30% |

> **Regra dura: nenhuma execução longa começa com MFU abaixo da meta.** MFU 20% em vez de 40% significa pagar o dobro pelo mesmo resultado — em uma execução de US$ 1.700, são US$ 850 desperdiçados que uma hora de perfilamento teria evitado. O micro-benchmark do DOC-17A §9 é pré-requisito, não sugestão.

---

## 10. Orçamento

| Item | Custo |
|---|---|
| ΦEnc-150M | US$ 25–90 |
| Ablação de mascaramento por equação | ~US$ 5 |
| **Varredura de LR do CPT (4 valores em 1,5 B)** | **~US$ 63** |
| Validação de transferência de escala (500 passos em 8 B) | ~US$ 30 |
| ΦGen-1,5B — CPT completo | US$ 120–240 |
| Extensão de contexto (1,5 B) | ~US$ 15 |
| ΦGen-8B — CPT completo | US$ 850–1.700 |
| Micro-benchmarks de MFU | ~US$ 30 |
| **Total da Fase 2 (treino base)** | **US$ 1.140–2.170** |

Sem o ΦGen-8B, que é opcional e fica no degrau T2c: **US$ 290–470**.

---

## 11. Riscos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Esquecimento catastrófico (falha G2.3) | **Alta** | **Desclassificatório** | Varredura de LR + replay de 5% + monitoramento a cada 5.000 passos |
| LR de 1,5 B não transfere para 8 B | Média | Alto | Regra `1/√largura` + validação de 500 passos (§5.3) |
| Extensão de contexto degrada contexto curto | Média | Médio | Avaliação obrigatória antes/depois; reverter se > 1 ponto (§8) |
| Preempção causa perda de dados ou repetição | Média | Médio | Dataloader determinístico por `(seed, step)` (§7.2) |
| MFU abaixo da meta desperdiça metade do orçamento | Média | **Alto** | Portão de MFU antes de execução longa (§9.3) |
| Spike não detectado queima execução | Média | **Alto** | Rollback automatizado (§6.1) |
| Habilitar QK-norm/z-loss em base que não os tem | Baixa | **Alto** | Verificar arquitetura do Qwen3 antes; §6.2 |

---

## 12. Critérios de aceite do Stage-Gate 7

- [ ] **H1** — Varredura de LR concluída; escolha justificada por medição conjunta de ganho em Física e regressão geral
- [ ] **H2** — Replay de 5% implementado e verificado na mistura de CPT
- [ ] **H3** — Rollback automático de spike testado por injeção deliberada de batch corrompido
- [ ] **H4** — Retomada determinística comprovada: matar no passo `n` e retomar produz sequência de dados **idêntica**
- [ ] **H5** — MFU dentro da meta em todas as configurações, medido antes de qualquer execução longa
- [ ] **H6** — Perda de validação por subárea instrumentada e visível em painel
- [ ] **H7** — Avaliação de regressão geral rodando a cada 5.000 passos, com alarme automático
- [ ] **H8** — Extensão de contexto validada sem regressão de contexto curto > 1 ponto
- [ ] **H9** — Checkpoint assíncrono com upload para bucket verificado sob preempção real

---

## 13. Questões em aberto

| ID | Questão | Destino |
|---|---|---|
| OQ-31 | Qual corpus geral serve melhor como proxy de replay para o Qwen3? | Testar FineWeb-Edu e Dolma na varredura de LR |
| OQ-32 | O Qwen3 já usa QK-norm e z-loss? | Verificar antes de implementar §6.2 |
| OQ-33 | Vale fazer o annealing com LR ainda menor que 15% do pico? | Ablação curta durante a fase 3 |
| OQ-34 | Contexto de 32k é suficiente, ou derivações exigem 128k? | Medir distribuição de comprimento de derivações completas no corpus |

---

## 14. Referências

1. Ibrahim, A. et al. (2024). *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* TMLR.
2. Gupta, K. et al. (2023). *Continual Pre-Training of Large Language Models: How to (re)warm your model?* arXiv:2308.04014.
3. Chowdhery, A. et al. (2023). *PaLM: Scaling Language Modeling with Pathways.* JMLR.
4. Peng, B. et al. (2024). *YaRN: Efficient Context Window Extension of Large Language Models.* ICLR.
5. Hu, S. et al. (2024). *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (WSD schedule). COLM.
6. Dettmers, T. et al. (2022). *8-bit Optimizers via Block-wise Quantization.* ICLR.
7. Warner, B. et al. (2024). *ModernBERT.* arXiv:2412.13663.
8. Zhang, B., Sennrich, R. (2019). *Root Mean Square Layer Normalization.* NeurIPS.
9. Dehghani, M. et al. (2023). *Scaling Vision Transformers to 22 Billion Parameters* (QK-norm). ICML.

---

**Fim do DOC-08.** Revisão da §12 necessária antes do DOC-09 (Pós-treino: SFT, DPO, RLVR, Destilação).
