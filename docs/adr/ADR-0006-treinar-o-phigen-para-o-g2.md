# ADR-0006 — O ΦGen deve ser treinado agora, para o portão G2? E dá para fazê-lo sem custo?

**Status:** **Proposto (2026-11-18)**. A decisão é do dono do projeto.
**Contexto:** [DOC-12 §3.7](../03-evaluation/DOC-12-harness-protocolo-estatistico.md) (a regra do G2 e os itens 20–32 do §3.7.4), [DOC-07 §5](../02-models/DOC-07-familia-de-modelos.md) (o ΦGen), [DOC-08 §5](../02-models/DOC-08-pretraining-cpt.md) (a receita do CPT), [DOC-17 §2 e §7](../05-governance/DOC-17-orcamento-cronograma.md) e [DOC-17A](../05-governance/DOC-17A-orcamento-gpu-runpod.md) (custos), [DOC-19 §2 e §8](../05-governance/DOC-19-riscos-validade-cientifica.md) (riscos), [ADR-0001 §7](ADR-0001-decisoes-stage-gate-0.md), [ADR-0003](ADR-0003-phienc-do-zero-ou-cpt.md) e [ADR-0004](ADR-0004-phigen-para-o-assistente.md).
**É** o ADR que o DOC-12 §3.7.10 exige antes do Tier 2 ("O Tier 2 não começa sem um ADR que o autorize ou que revise o G1"). **Não revisa** a regra do G2.

---

## 1. A pergunta

O portão G2 está pronto: a regra foi congelada antes de qualquer treino, e a Qwen3-1.7B-Base e a Qwen3-8B-Base foram medidas por inteiro. Falta o ΦGen, o pré-treino continuado (CPT) de uma dessas bases em Física. O ADR-0004 decidiu que o ΦGen não se treina por causa do assistente; se for treinado, é pela pergunta do G2: raciocínio em Física sem a fonte no prompt.

**Vale treinar o ΦGen agora para tentar o G2? E, como o dono perguntou, dá para treinar sem custo?**

---

## 2. A evidência

### 2.1 O que o G2 exige, medido nas bases

| | Qwen3-1.7B-Base | Qwen3-8B-Base |
|---|---|---|
| G2.1, avg@k: UGPhysics · OlympiadBench · PHYBench · SciBench | 10,0 · 4,3 · 0,1 · 10,6 | 17,8 · 12,3 · 1,1 · 25,3 |
| **Macro de Física** (média simples dos 4) | **~6,3** | **~14,1** |
| G2.3: MMLU sem física · IFEval · EvalPlus | 62,72% · 35,28% · 52,49% | 76,56% · 54,57% · 64,06% |

- **A barra do G2.1.** Para passar com 80% de chance, o ΦGen precisa de **~+12 pontos de ganho efetivo no macro** sobre a própria base: +12,2 na 1,7B e +12,1 na 8B, +12,6 na 8B com a condição 3 (DOC-12 §3.7.4 item 32). A 1,7B teria de ir de ~6,3 a ~18,5, e a 8B de ~14,1 a ~26.
- **A escala dessa barra.** +12 é mais do que toda a distância entre as duas bases (~7,8 pontos), que vem de ~4,8× mais parâmetros com o mesmo pré-treino de 36 T fichas. O ΦGen-1,7B teria de superar a 8B-base.
- **E sem perder o resto.** Não-inferioridade de 2 pontos no MMLU; o EvalPlus e o IFEval desclassificam se Δ̂ < −2 com o IC inteiro abaixo de 0 (§3.7.7). E sem memorização nova (condição 3, com a lista híbrida da emenda 4).

### 2.2 O que a literatura mostra

Fontes primárias, lidas em 2026-11-18. As notas são as dos próprios artigos, em protocolos diferentes do G2, e quase todas são de matemática, não de Física: servem de indicação da ordem de grandeza, não de previsão.

| CPT | Base | Fichas de CPT | Ganho |
|---|---|---|---|
| Minerva 8B (arXiv 2206.14858) | PaLM 8B | 164 B | MATH 1,5 → 14,1 (**+12,6**) |
| Llemma 7B (arXiv 2310.10631) | Code Llama 7B | 200 B | MATH 4,5 → 18,0 (**+13,5**) |
| InternLM2-Math-7B (arXiv 2402.06332) | InternLM2-7B | 125 B | MATH 8,6 → 21,5 (**+12,9**) |
| DeepSeekMath 7B (arXiv 2402.03300) | DeepSeek-Coder 7B | 500 B | MATH **+23,7** |
| MathCoder2 (arXiv 2410.08196) | Llama-3-8B · **DeepSeekMath-7B** | ~58 B | MATH **+17,0** · **+2,4** |
| Qwen2-Math-7B (arXiv 2409.12122) | Qwen2-7B | corpus de ~700 B | MATH 44,2 → 50,4 (**+6,2**) |
| MathPile (arXiv 2312.17120) | Mistral-7B | 0,4–4 B | MATH **+1,6 a +2,0** |
| OctoThinker-3B (arXiv 2506.20512) | Llama-3.2-3B | ~220 B | OlympiadBench **+1,3 a +2,7** |
| MAmmoTH2-8B (arXiv 2405.03548) | Llama-3-8B | ~5 B de **QA** ×2 | TheoremQA **+12,1**, GPQA **+8,0** |

O que essas linhas dizem juntas:
- **Os +12 publicados vêm de bases fracas no domínio e de 50 a 500 B fichas.** Com até ~4 B fichas de texto, o ganho foi de +1,6 a +2,0.
- **Sobre uma base que já é forte, o mesmo CPT rende pouco.** O corpus que deu +17,0 na Llama-3-8B deu +2,4 na DeepSeekMath. A Qwen2-Math ganhou +6,2 na 7B. A Qwen3 é dessas: a Qwen3-8B-Base faz 60,8 no MATH (arXiv 2505.09388), acima da Qwen2.5-Math-7B (55,4), que é especializada.
- **Em benchmark de olimpíada, o CPT move de 1 a 3 pontos**, mesmo com ~220 B fichas.
- **A matemática não transfere para a Física.** No UGPhysics, a Qwen2.5-Math-7B-Instruct faz 22,69 contra 22,55 da Qwen2.5-7B-Instruct: +0,14 (arXiv 2502.00334, Tab. 5). Os autores escrevem que o CPT e o ajuste em matemática rendem "ganhos marginais" em Física, e que o treino precisa de conteúdo de Física.
- **CPT de ciência em 7–8B** mexe de −8 a +4 pontos em benchmarks de conhecimento (AstroLLaMA, arXiv 2409.19750; SciLitLLM sobre a Qwen2.5-7B, +2,6, arXiv 2408.15545).
- **Os saltos de +8 a +14 em raciocínio científico vieram de dado em formato de pergunta e resposta, ou de destilação**, não de texto cru (MAmmoTH2 acima; OmniScience, arXiv 2503.17604). Em 2025–2026, vieram de RL.
- **Esquecimento.** Com 5–30% de dado geral (replay) e a LR reaquecida e redecaída, as perdas gerais publicadas ficam entre 0 e −1,5 ponto (arXiv 2403.08763, 2407.18743). Sem isso, passam de 2: o SciGLM perdeu 10,8 no MBPP (arXiv 2401.07950). Em CPT, o LoRA aprende bem menos que o ajuste completo: no GSM8K, 0,202 contra 0,293 aos 20 B fichas, e o r=16 não aprendeu nada mensurável (arXiv 2405.09673).
- **Não há nenhum CPT publicado com o antes e o depois nos quatro benchmarks do G2.1.** Nem números publicados das duas bases Qwen3 neles.

### 2.3 O que a computação gratuita compra

O plano do repositório é um CPT de **15 B fichas** na 1,7B (ajuste completo, ~24 GB de memória, ~84 h de uma H100, US$ 120–240) e de 20 B na 8B (~128 GB, US$ 850–1.700), precedido de uma varredura obrigatória de 4 LRs de ~1 B fichas cada (DOC-07 §5.3, DOC-08 §5.3).

O que cabe sem dinheiro novo. As vazões da 1,7B são **estimativas**, calibradas na única medida do projeto: o CPT do ModernBERT-base em 2×T4, a 10,5–11,0% de MFU (ESTADO.md). Nenhum gerador foi treinado aqui.

| Caminho | 1 B fichas | 10 B fichas | Viável? |
|---|---|---|---|
| **Kaggle 2×T4**, 1,7B (Adam de 8 bits, FSDP nas 2 placas, checkpoint de ativações; o AdamW pleno não cabe) | 105–350 h de cota, **3,5–12 semanas** | 35–117 semanas | Só piloto, até ~0,5–1 B |
| Kaggle 2×T4, 8B | — | — | **Não**: nem o ajuste completo cabe |
| **TPU do Kaggle**, v5e-8, 1,7B em bf16 (20 h por semana, sessões de 9 h) | 4–9 h | **2–5 semanas** | Sim, **se** a conta tiver a v5e-8 e o treino for portado para JAX |
| Colab Pro (já pago), 1,7B em A100 ou L4 | 1,6–2,6 meses | 16–26 meses | Só complemento |
| TPU Research Cloud (Google), se concedido | horas | horas (1,7B) ou 1–2 dias (8B) | Não garantido |
| Referência alugada (premissas do DOC-17A) | US$ 18–22 | **~US$ 200** | Fora da restrição atual do dono |

- **A TPU do Kaggle** é o único caminho grátis na escala do plano. A documentação do Kaggle cita 20 h por semana e a v5e-8, mas se contradiz, e a conta do dono não foi conferida. Treinar nela exige portar o laço para JAX (MaxText ou EasyDeL): segundo a documentação lida em 2026-11-18, o PyTorch/XLA foi descontinuado em favor do TorchTPU. E a conversão HF ↔ Orbax da Qwen3-1.7B não está na tabela do MaxText (a 0,6B, a 4B e a 8B estão).
- **O TRC** é o único programa gratuito aberto a pesquisador independente: "qualquer um pode manifestar interesse", e os convites saem aos poucos, sem garantia. Os de créditos do Google, da AWS e da NVIDIA pedem vínculo de docente ou pós-graduando. A TPU é grátis por tempo limitado, num projeto do Google Cloud, mas o que não é TPU (VMs pequenas, armazenamento) se paga, ainda que pouco. Em troca, pede-se compartilhar a pesquisa (artigo, código aberto ou blog) e dar retorno ao Google.
- **O Kaggle anuncia +15 h por semana de GPU** para quem vincula uma conta Colab Pro, numa promoção experimental. A conta do dono não foi conferida.

### 2.4 As travas que não custam GPU

Antes de qualquer ΦGen medido, há trabalho e decisões que nenhuma GPU resolve:

1. **O G1 não foi vencido.** O G1.1 e o G1.2 passaram; o G1.3 ficou impossível pelo ADR-0003, o G1.4 foi adiado e o G1.5 está parcial. O DOC-17 §7 manda parar e diagnosticar antes de gastar no Tier 2. Essas falhas são da linha do encoder e da reprodutibilidade do corpus: nenhuma delas é sobre a pergunta do ΦGen.
2. **Os pareceres jurídicos.** O ADR-0001 §7 pede, "antes do Tier 2", um parecer sobre o direito de treinar no Brasil, onde a Lei 9.610/98 não tem exceção explícita de mineração de texto, e outro sobre a compatibilidade entre NC e Apache-2.0. Não há registro de que tenham sido feitos.
3. **A varredura de descontaminação do §3.7.9 B não tem código.** Ela é pré-condição: sem ela, o ΦGen não é medido.
4. **Não há código de CPT.** `src/phifm/training/cpt/` está vazio. O único laço de pré-treino treina encoder por MLM, em DDP, e aborta quando o run não cabe na sessão, em vez de gravar e retomar na seguinte.
5. **O corpus está incompleto.** Existem as fatias de Física do S3 (~27,75 B fichas estimadas por caracteres ÷ 4, nunca contadas com o tokenizer do Qwen3) e 11,2 B de LaTeX de math e cs. Faltam o corpus de replay (OQ-31 do DOC-08), o material didático e o sintético do DOC-06. O peS2o não tem equações. O RedPajama perde 16,6% delas, e a fatia provavelmente inclui os ~2,9% de artigos NC, o que não foi conferido.
6. **O tokenizer não pode mudar.** O DOC-05 §9 prevê estender o vocabulário para o CPT, mas o G2 exige o tokenizer idêntico ao da base (§3.7.5). Um ΦGen medido no G2 fica com o vocabulário do Qwen3.
7. **Medir o ΦGen exige uma emenda** (o braço e o GGUF dele), e cada tamanho tem um acesso só ao teste.

---

## 3. O que a evidência diz, e o que ela não diz

**Diz:**
- **Dá para treinar sem custo, mas só a 1,7B**, e em uma de duas escalas:
  - a de piloto, nas T4 do Kaggle, até ~0,5–1 B fichas, ao longo de semanas de cota;
  - a do plano, só pela TPU do Kaggle, se a conta a tiver e o treino for portado para JAX, ou pelo TRC, se concedido.

  A 8B não cabe no grátis, salvo pelo TRC.
- **Na escala de piloto, a literatura indica poucos pontos.** Em matemática de nível médio, foram de +1,6 a +2,0 com até 4 B fichas de texto. Esperar +12 num macro de Física difícil seria esperar mais do que qualquer resultado publicado que achamos nessa escala.
- **Mesmo na escala do plano, não achamos precedente para +12.** Os +12 publicados vieram de bases fracas no domínio, com 50 a 500 B fichas. Sobre bases fortes, o mesmo CPT deu de +2 a +6, e o CPT de matemática não transferiu para a Física.
- **Com a receita de CPT em texto cru e a computação gratuita, o resultado esperado do G2.1 é "não demonstrado".** É um resultado legítimo e publicável, mas não é o que o portão pede.

**Não diz:**
1. **Nada foi medido nesta régua.** Nenhum CPT publicado tem o antes e o depois nos quatro benchmarks do G2.1, e o projeto não tem âncora própria. O único análogo é o CPT do encoder no ADR-0003: 0,4 B fichas compraram +0,010 de nDCG@10, contra +0,069 de só trocar a base.
2. **As vazões da 1,7B em T4 e em TPU são estimativas**, com uma faixa de ~3×. Só encoders foram medidos.
3. **A oferta de TPU e o bônus do Colab na conta do dono** não foram conferidos.
4. **A rota com mais ganho por ficha não foi avaliada:** dados de Física com solução, como pergunta e resposta, derivações ou destilação. Ela esbarra no "ganho de formato" (§3.7.6), no risco de memorização (condição 3) e na fronteira entre CPT e pós-treino (§3.7.1), que pede regra própria com braço de controle.

---

## 4. As opções

| | O quê | Custo | O que compra |
|---|---|---|---|
| **A** | **Não treinar o ΦGen agora.** O G2 fica publicado como instrumento e linha de base: a regra congelada, as duas bases medidas, as sondas lacradas e o poder | US$ 0, sem cota | Um resultado completo, que já existe. O Tier 2 fica fechado até algo da §6 mudar |
| **B** | **Piloto grátis na 1,7B, só no dev.** ~0,5–1 B fichas, com regra de parada escrita antes, sem tocar o teste | US$ 0. Semanas de cota (T4) ou uma sessão (TPU). Semanas de engenharia: o código de CPT e a retomada entre sessões, ou o porte para JAX. As travas 2, 3 e 5 antes | A primeira medida "fichas → ganho" em Física numa régua congelada, que a literatura não tem: publicável mesmo se der pouco, que é o que a literatura indica |
| **C** | **O CPT do plano (10–15 B fichas) de graça**, pela TPU do Kaggle ou pelo TRC | US$ 0 se a TPU ou o TRC existirem. O porte para JAX e todas as travas | A tentativa real do G2.1, com resultado esperado abaixo do limiar |
| **D** | **Mudar a receita** para dados de Física com solução (pergunta e resposta, derivações, destilação) | Uma regra nova antes do treino (a fronteira com o pós-treino, o ganho de formato, a memorização) e as licenças desses dados | A rota com mais ganho por ficha na literatura. Muda o que o G2 mede |
| **E** | **Alugar GPU**, como o DOC-07 planejava | ~US$ 200 por 10 B fichas na 1,7B; ~US$ 1.700 por 20 B na 8B | Tempo. É contra a restrição atual do dono |

---

## 5. Recomendação

**Opção A.** Não treinar o ΦGen agora, e publicar o G2 como instrumento e linha de base.

Por quê:
- **Não achamos precedente para a barra.** O G2.1 pede +12 pontos num macro de Física difícil sobre uma base forte em STEM. Nos resultados publicados que achamos, o CPT só chegou a +12 em matemática, sobre bases fracas, e com dezenas de vezes mais fichas do que a computação gratuita compra.
- **O gratuito só chega ao piloto,** a não ser pela TPU do Kaggle ou pelo TRC, que ainda não estão confirmados.
- **Antes de qualquer ΦGen medido, há quatro travas que não custam GPU:** os pareceres jurídicos, o código da varredura, o código de CPT e o corpus que falta.
- **Treinar agora levaria, muito provavelmente, a "não demonstrado"**, depois de semanas de engenharia e de cota.
- **O que já existe é completo e publicável:** a regra congelada antes de qualquer treino, as duas bases medidas e a régua calibrada pelo poder.

**Duas ações de custo zero, que mantêm a opção C aberta:**
1. **Conferir no menu do Kaggle** qual TPU a conta recebe e se o vínculo do Colab Pro dá as +15 h por semana. É um clique, do dono.
2. **Decidir se pede o TRC.** O pedido é do dono: exige uma conta e um projeto no Google Cloud, onde o que não é TPU se paga, e o compromisso de compartilhar a pesquisa. A concessão não é garantida.

**Se o dono preferir treinar mesmo assim**, a ordem é esta:
1. os pareceres do ADR-0001 §7;
2. o código da varredura do §3.7.9 B e do CPT;
3. a opção B, com a regra de parada escrita antes e medida só no dev, sem gastar o acesso ao teste;
4. a C, só se o piloto mostrar no dev um ganho que projete o limiar.

---

## 6. O que reabriria esta decisão

- **Computação na escala do plano a custo zero:** o TRC concedido, ou a TPU v5e-8 confirmada na conta com o porte para JAX feito. A opção C passaria a custar só engenharia.
- **Um CPT publicado com o antes e o depois** em UGPhysics, OlympiadBench, PHYBench ou SciBench, sobre uma base forte, com ganho grande e fichas alcançáveis.
- **Um piloto (opção B) que mostre no dev** um ganho que projete ≥ +12 no macro.
- **A decisão de escrever a regra da rota com dados de solução** (opção D), com braço de controle (§3.7.1).

---

## 7. O que fica publicado, qualquer que seja a decisão

- **O protocolo:** o DOC-12 §3.7, o `g2.yaml`, o `g2_congelamento.json`, as 4 emendas e o livro de acessos.
- **As medidas das duas bases**, em agregados: G2.1, G2.3, as sondas S1/S2 com as listas lacradas por hash, e o poder recalculado.
- **Os achados de método:**
  - a S2 de 13 fichas mede competência, não memória (no nulo, 37% na 1,7B e 49% na 8B);
  - a geração gulosa do `llama-server` só repete ficha a ficha com 1 slot, servidor novo e a mesma ordem;
  - o lacre por compromisso de hash.
- **O código da medida e as gerações por hash.** As gerações ficam fora do git.

**Achados nos documentos de plano, para corrigir se o Tier 2 seguir:**
- o DOC-08 §5.2 decai a LR até 10% do pico, e o §5.4 a leva a ~0;
- o CPT da 8B tem 20 B fichas no DOC-07 §5.3 e 40 B no DOC-17A;
- o DOC-17A dá ~US$ 400–800 para o CPT de 1,5B na §4 e +US$ 120–240 na §8.2;
- a extensão de tokenizer do DOC-05 §9 é incompatível com o G2.
