# DOC-15 — Inferência e Serving

**Projeto:** ΦFM — Phi Foundation Models para Física
**Status:** `RASCUNHO v0.1` — aguardando revisão do Stage-Gate 14
**Cobre:** entregável solicitado **12** (pipeline de inferência)
**Depende de:** [DOC-07](../02-models/DOC-07-familia-de-modelos.md), [DOC-13](DOC-13-recuperacao-embeddings-rag.md), [DOC-14](DOC-14-agentes-ferramentas.md), [DOC-17A](../05-governance/DOC-17A-orcamento-gpu-runpod.md)
**Data:** 2026-08-03

---

## 1. Três perfis de inferência, não um

Tratar inferência como um problema só levaria a superdimensionar tudo. O programa tem três cargas com requisitos incompatíveis:

| Perfil | Uso | Requisito dominante | Configuração |
|---|---|---|---|
| **Avaliação em lote** | Rodar o PhysBench, gerar dados sintéticos, amostragem por rejeição | **Vazão** — latência irrelevante | 1× 4090 spot, batch grande |
| **Interativo** | Usuário fazendo perguntas | **Latência do primeiro token** | Serverless ou 1× 4090 quente |
| **Agêntico** | ΦAgent com laços de ferramenta | **Reuso de prefixo** | SGLang, cache de KV |

**Nenhum deles exige uma H100 dedicada 24/7.** O modo de operação padrão é *scale-to-zero*: o serviço não existe quando ninguém o usa.

---

## 2. Escolha do motor

| Motor | Vantagem decisiva | Custo | Uso aqui |
|---|---|---|---|
| **vLLM** | PagedAttention, batching contínuo, ecossistema amplo | — | ✅ Padrão para lote e interativo |
| **SGLang** | **RadixAttention** — cache automático de prefixo em árvore | — | ✅ **Agêntico** — DOC-01 §5.6 |
| TensorRT-LLM | Menor latência absoluta | Ciclo de build lento, preso ao hardware | ⚠️ Só se houver SLA de produção |
| llama.cpp | Roda em CPU e Apple Silicon | Vazão baixa | ✅ **Demonstração local** — ver §5 |

O ganho do SGLang em uso agêntico é grande porque o ΦAgent reemite o mesmo prompt de sistema, o mesmo catálogo de ferramentas e os mesmos trechos recuperados em cada iteração do laço. Com RadixAttention, esse prefixo é computado uma vez.

---

## 3. Quantização

| Formato | Bits | Degradação típica | Uso |
|---|---|---|---|
| bf16 | 16 | referência | Avaliação de manchete |
| **AWQ / GPTQ** | 4 | Pequena, mas **precisa ser medida** | ✅ Serving |
| FP8 | 8 | Muito pequena, exige Hopper+ | ✅ Se houver H100 |
| GGUF Q4_K_M | ~4 | Pequena | ✅ Demonstração local |

> **Regra dura: nenhum número de benchmark é reportado a partir de modelo quantizado sem a comparação bf16 ao lado.** Quantização é decisão de implantação, não de capacidade, e misturar as duas produziria resultados irreproduzíveis por quem baixar os pesos em bf16.
>
> **Ressalva específica de Física:** a degradação por quantização não é uniforme entre tarefas. Raciocínio de múltiplos passos e aritmética são mais sensíveis que classificação — exatamente as capacidades que mais nos interessam. A ablação de quantização é medida **por tarefa do PhysBench**, não em agregado.

O ΦGen-1,5B em AWQ ocupa ~1,2 GB e roda confortavelmente numa 4090 junto com o ΦEmb e o ΦRank.

---

## 4. Decodificação especulativa

Um modelo rascunho pequeno propõe `k` tokens; o modelo alvo os verifica em paralelo. Aceleração típica de 1,5–3×, **sem alterar a distribuição de saída**.

Vantagem específica nossa: **temos o par natural pronto.** ΦGen-1,5B como rascunho e ΦGen-8B como alvo, treinados no mesmo corpus com o mesmo tokenizer — o que dá taxa de aceitação alta, porque os modelos concordam com frequência.

Alternativa sem segundo modelo: **decodificação especulativa por n-grama** sobre o próprio contexto, útil quando a resposta repete trechos do prompt (comum em RAG e em derivações que reafirmam equações).

**Fica para depois do ΦGen-8B existir.** Antes disso não há par.

---

## 5. Perfil de implantação

| Perfil | Infraestrutura | Custo | Quando |
|---|---|---|---|
| **Local** | llama.cpp / Ollama, GGUF Q4 | **US$ 0** | Demonstração, desenvolvimento, uso pessoal |
| **Serverless** | RunPod Serverless, scale-to-zero | **~US$ 5–40/mês** | ✅ **Padrão** — avaliação e uso intermitente |
| **Dedicado** | 1× 4090 contínua | ~US$ 250/mês | Só com demanda sustentada |
| **Produção** | Ray Serve / KServe, autoescala | ~US$ 500+/mês | Tier 3, se houver usuários reais |

> O perfil **local** é estrategicamente importante e costuma ser subestimado. Um ΦGen-1,5B em GGUF Q4 roda num laptop com 8 GB de RAM. Para um modelo de Física, isso significa que **um estudante ou pesquisador sem orçamento consegue usá-lo** — o que é, em boa medida, o ponto de publicar pesos abertos.

**Padrão: serverless com scale-to-zero.** O serviço custa quase nada quando ocioso, e o *cold start* de 10–30 s é aceitável para uso de pesquisa. Pagar GPU dedicada antes de haver demanda seria o erro de custo mais fácil de cometer nesta fase.

---

## 6. A camada de guardrails

O que é imposto **na saída**, independentemente do que o modelo gerou:

| Guardrail | Ação | Origem |
|---|---|---|
| **Verificação de citação** | DOI resolvido; citação inválida é removida e a afirmação marcada como não fundamentada | DOC-13 §6.2 |
| **Checagem dimensional** | Equações da resposta passam por `verify/dimensional`; incoerência é sinalizada ao usuário | DOC-10 §3.2 |
| **Declaração de convenção** | Se a resposta usa convenção específica, ela é declarada explicitamente | F7 |
| **Calibração de confiança** | Afirmações de baixa confiança recebem hedge; abstenção quando apropriado | F10, G2.5 |
| **Fronteira de confiança** | Conteúdo recuperado nunca vira instrução | DOC-14 §4 |

> **Os guardrails não são cosméticos.** A verificação de citação é o portão G2.4 operando em produção, e a checagem dimensional pega em tempo real uma fração dos erros de F2 que escaparam do treino. **É a mesma `verify/` do treino e da avaliação** — quarta aplicação do princípio P3.

---

## 7. Metas de desempenho

| Métrica | ΦGen-1,5B (AWQ, 1× 4090) | ΦGen-8B (AWQ, 1× H100) |
|---|---|---|
| Primeiro token (p50) | < 200 ms | < 300 ms |
| Vazão, 1 fluxo | ~60 tok/s | ~50 tok/s |
| Vazão, lote 32 | ~900 tok/s | ~1.400 tok/s |
| Ponta a ponta com RAG (p95) | < 2 s | < 3 s |
| Pedido agêntico (p95) | < 30 s | < 45 s |

Números a validar por medição; servem como alvo e como detector de regressão.

---

## 8. Riscos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Quantização degrada raciocínio mais que o esperado | **Alta** | Médio | Ablação **por tarefa**; bf16 para números de manchete |
| Cold start serverless frustra uso interativo | Média | Baixo | Documentado; instância quente opcional |
| Guardrail de citação adiciona latência inaceitável | Média | Médio | Cache de resolução de DOI; verificação assíncrona quando possível |
| Custo de GPU dedicada antes de haver demanda | **Alta** | Médio | Padrão scale-to-zero; dedicado exige justificativa |
| Divergência entre `verify/` de treino e de serving | Baixa | **Alto** | Mesmo pacote versionado; teste de importação em CI |

---

## 9. Critérios de aceite do Stage-Gate 14

- [ ] **O1** — Ablação de quantização medida **por tarefa** do PhysBench; degradação documentada
- [ ] **O2** — Nenhum número de manchete reportado a partir de modelo quantizado
- [ ] **O3** — Guardrail de citação ativo em serving, com a mesma taxa do PhysBench
- [ ] **O4** — Perfil local (GGUF) funcional e documentado em laptop de 8 GB
- [ ] **O5** — Scale-to-zero verificado; custo ocioso próximo de zero
- [ ] **O6** — `verify/` de serving é o mesmo pacote do de treino, verificado em CI
- [ ] **O7** — Metas de latência atingidas ou revisadas por medição

---

## 10. Referências

1. Kwon, W. et al. (2023). *Efficient Memory Management for Large Language Model Serving with PagedAttention* (vLLM). SOSP.
2. Zheng, L. et al. (2024). *SGLang: Efficient Execution of Structured Language Model Programs.* NeurIPS.
3. Lin, J. et al. (2024). *AWQ: Activation-aware Weight Quantization.* MLSys.
4. Frantar, E. et al. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR.
5. Leviathan, Y. et al. (2023). *Fast Inference from Transformers via Speculative Decoding.* ICML.

---

**Fim do DOC-15.**
