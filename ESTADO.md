# Estado do projeto — 2026-08-31

Ponto de retomada para migração de máquina. Instalação em [SETUP.md](SETUP.md).

## Onde estamos

**Corpus de projeto:** completo. 19 documentos + 1 ADR, cobrindo os 20 pipelines.

| Sprint | Estado | Observação |
|---|---|---|
| **S1** · espinha de metadados | 🟢 completo | 1,59 M arXiv + 4,61 M obras; junção de **99,1%** |
| **S2** · classificador de Física | 🟢 completo | subárea + `is_physics`; acurácia **0,954** com os 4 domínios, FP 2,4–3,7% em cada |
| **S3** · fatias do HuggingFace | 🟢 **27,75 B tokens** | RedPajama 10,54 B + OpenWebMath 2,62 B + **peS2o 14,60 B**, custo zero. S3b: o RedPajama **degrada 16,6%** |
| **ΦEmb** | 🟢 G1.1 ✅ / G1.2 ✅ | nDCG@10 **0,4657** contra 0,4628 do GTE-large, com 1/14,8 dos parâmetros. ⚠️ A margem é +0,003 e o pareado em recall@1 dá EMPATE — o defensável é paridade a 1/15 do tamanho, não vitória |
| **G1.5** · corpus por um hash | 🟡 metade fechada | 21,79 GB verificáveis byte a byte por **um** hash; refazer do zero depende de uma fonte mutável, nomeada |
| Barramento de verificação | 🟢 5 de 6 | falta só `sandbox` — exige gVisor/Firecracker |
| **T1a** · ΦEmb na T4 | 🟢 medido | **181,6 pares/s** contra 20-26 aqui; 13 h viram 36 min. Destrava o ΦEnc: ~80 h de T4 em vez de 37 dias |
| **T1b** · busca híbrida | 🟡 fusão ✅ / ΦRank no-op | RRF entrega **nDCG 0,1584** e vence os isolados (p<0,001); o reranker empata (p=0,118) porque parte da **mesma base** do ΦEmb |
| **T1c** · ΦRank de base diferente | 🟢 **fechado: domínio** | PhysBERT vence a fusão (nDCG **0,1666** vs 0,1576, **p=0,0062**); `gte-base`, do MESMO tamanho, **empata** (p=0,637). Não é diversidade de base nem capacidade — é **pré-treino em Física** |
| **ΦEnc** · dado | 🟢 **destravado, US$ 0** | RedPajama-arXiv tem ambiente de equação em **84,9%** contra 0,0% do peS2o. ~10 B tokens de LaTeX íntegro no disco. A recomendação de comprar acesso ao arXiv estava errada — [ADR-0002](docs/adr/ADR-0002-fonte-latex-para-o-phienc.md) |
| **ΦEnc** · código | 🔴 não existe | `training/pretrain/` está vazio. É o gargalo agora, e não é de dinheiro |

Suíte: **501 testes** (9 saltados, os de torch), `PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/ -q`.
Os que dependem de torch rodam na venv de treino:
`.venv-treino/Scripts/python.exe -m pytest tests/regression/test_g1_criterios.py tests/regression/test_comparacao_pareada.py tests/regression/test_melhor_checkpoint.py tests/regression/test_gradcache.py tests/regression/test_estado_progresso.py -q`

## Itens de custo zero executados (2026-08-31)

O usuário pediu "execute todos os itens que nao tem custo $" sobre a lista de
recomendações. O que saiu disso, em ordem de importância:

### 1. ⚠️ A recomendação nº 1 estava errada, e ela custava dinheiro

Eu havia aberto a lista com "decida o arXiv (US$ 100–180), ou abandone a hipótese
central do ΦEnc". **O corpus com equações intactas estava no disco havia 17 dias.**
RedPajama-arXiv, fatia de Física: 835.379 documentos, 12 GB, ambiente de equação em
**84,9%** contra **0,0%** do peS2o. Volume MEDIDO: **42,15 G caracteres**, 50.451 por
documento, **10,54 B tokens** a ~4 ch/token — acima dos 5 B por variante que o
DOC-05 §11.2 pede.

⚠️ **E os 10,54 B já estavam na tabela de sprints acima**, desde a S3. O que ninguém
havia conferido era a integridade matemática da fatia — e a seção do peS2o dizia
"volume não é o gargalo; integridade matemática é" com a resposta na mesma tabela.

Registrado em [ADR-0002](docs/adr/ADR-0002-fonte-latex-para-o-phienc.md), com a
correção de método (o regex de operador órfão satura em texto LaTeX íntegro) e a
cotação de verdade do S3 (US$ 400+ para fora da AWS, dezenas de dólares filtrando
dentro).

### 2. T1c — o reranqueador de base diferente, rodando na cota gratuita

A predição da §3.3 do rascunho: se a redundância informacional anula o ΦRank, um
cross-encoder de base diferente do ΦEmb deve bater a fusão. Duas variantes de 109 M
(`thenlper/gte-base` e `thellert/physbert_cased`) contra o controle de 23 M,
**só `--base` muda** — os outros seis hiperparâmetros são byte a byte os do controle,
com teste que falha se algum escorregar.

Regra de decisão registrada ANTES: McNemar em k=10 contra a fusão da mesma execução,
limiar de Bonferroni **0,025** por serem duas variantes, 2.000 consultas (o controle
antigo tinha 1.000 e deu p=0,118 — faltava poder). As quatro leituras possíveis estão
escritas, inclusive a de nenhuma vencer.

`kaggle/t1c_phirank.py` · dataset `phifm-t1c-rerank-bases` (457,7 MB) ·
notebook `phifm-t1c-rerank`.

### 3. Rascunho do artigo metodológico

[docs/papers/rascunho-armadilhas-recuperacao-por-citacao.md](docs/papers/rascunho-armadilhas-recuperacao-por-citacao.md)
— cinco falhas medidas, com a tese de que compartilham uma causa: supervisão e
avaliação derivadas da mesma estrutura. ⚠️ As referências foram escritas de memória e
**nenhuma foi conferida**; a §12 lista quais são as mais frágeis.

### 4. Folha de revisão da amostra de contaminação

`scripts/folha_de_revisao.py` gera um HTML único, no HD, sem rede. O escore do
classificador fica **escondido até depois do julgamento** (viés de ancoragem), o alvo
é pré-comprometido em 200 documentos, e `scripts/apurar_revisao.py` apura com Wilson e
estratifica por faixa de escore. Esta é a tarefa que só o usuário pode fazer.

### 5. `scripts/simular_ci.py`

Roda os cinco passos do workflow sobre um `git archive` do commit — só arquivos
rastreados, que é o que o CI vê. É a classe de erro que deixou o CI vermelho e que
`pytest` local não pega, porque ele vê o disco inteiro.

### 6. DOC-05 absorveu o §11.1 medido

O documento seguia afirmando que não havia evidência de BPE vs Unigram em LaTeX (há,
é nossa, e contraria a previsão da §3.2) e tratando V=40.960 como decidido. A decisão
fica; agora com o número que a contraria escrito ao lado, e com a razão pela qual a
métrica intrínseca não pode decidi-la.

### E um conserto no publicador do Kaggle

`datasets create` volta ANTES de o dataset existir — a criação é assíncrona. O
`kernels push` seguinte não resolveu a fonte, avisou numa linha, criou o notebook
**sem dados** e disse "successfully pushed" com código 0. O `FALHAS_SILENCIOSAS`
pegou; agora espera `datasets status` dizer `ready`.

## T1c — o que o reranqueador precisa é DOMÍNIO, não diversidade (2026-09-03)

    variante   base                       params  acc@1  +ΦRank       Δ  disc  p(k=10)
    controle   MiniLM-L6 (= a do ΦEmb)       23M  0,498  0,1483  -0,0093   229   0,1458
    gte        gte-base (geral forte)       109M  0,510  0,1530  -0,0046   220   0,6371
    phys       physbert (Física)            109M  0,566  0,1666  +0,0090   237   0,0062

    fusão RRF = 0,1576 nas três · 2.000 consultas · profundidade 50 · teto 0,4495

**Leitura pré-registrada aplicada: "só a `phys` ⇒ o mecanismo é CONHECIMENTO DE
DOMÍNIO, não diversidade".** É o que saiu, e o pré-registro é de antes de medir.

O que faz disto uma afirmação de mecanismo, e não uma coincidência: `gte-base` e
`physbert_cased` têm o **mesmo tamanho** (109 M), a **mesma arquitetura**
(`BertModel`), os **mesmos seis hiperparâmetros**, os **mesmos dados**, a **mesma
semente** e o **mesmo protocolo**. A única diferença entre as duas é o corpus de
pré-treino. **Diversidade de base não basta. Capacidade não basta.**

⚠️ **As duas corridas são combináveis, e há evidência disso.** O `gte` rodou numa
execução separada (a primeira morreu num bug meu), e o controle saiu **byte a byte
idêntico** nas duas — `sistemas`, `pareado_contra_a_fusao` e `teto` iguais campo por
campo, p=0,14584 sobre 229 discordantes nas duas. É isso que licencia comparar `gte`
com `phys`; sem essa verificação a comparação seria entre protocolos, não entre bases.

### O contraponto, que é o achado mais interessante

O PhysBERT é um recuperador **ruim** neste benchmark: nDCG 0,2752 contra 0,4657 do
ΦEmb — perde por 0,190, e foi exatamente essa a medição do G1.1. E é a **melhor base
de reranqueador** das três.

Pré-treino de domínio não fez um bi-encoder bom aqui, e fez um cross-encoder bom. A
assimetria não era o que eu esperava, e é ela que dá uma receita: **para reordenar,
partir de um modelo do domínio; para recuperar, ajustar um modelo geral pequeno.**

### A predição do T1b se confirmou

**A predição do T1b se confirmou.** O diagnóstico dizia que o ΦRank empatava por
**redundância informacional** — partia do mesmo MiniLM do ΦEmb e re-derivava a ordem
que a fusão já tinha (Spearman −0,466). A predição: uma base com pré-treino diferente
deve bater a fusão. Bateu, com **p = 0,00625**, abaixo do limiar de Bonferroni de
**0,025** registrado antes de medir.

| | fusão | + PhysBERT |
|---|---|---|
| recall@1 | 0,0700 | 0,0690 |
| **recall@10** | 0,2675 | **0,2890** |
| recall@50 | 0,4495 | 0,4495 |
| **nDCG@10** | 0,1576 | **0,1666** |

No pareado k=10: a fusão ganha 97, o reranqueador ganha **140**, 237 discordantes. Em
k=1 é empate (p=0,930) — o ganho é na cauda do top-10, não no primeiro lugar, o que é
o que se espera de reordenação.

**O controle replicou o empate do T1b** com o dobro das consultas: p=0,146 sobre 229
discordantes, contra p=0,118 sobre 105 antes. É isso que mostra que a diferença é da
**base** e não do protocolo — as duas variantes rodaram no mesmo universo, com os
mesmos 2.000 sorteios e os mesmos seis hiperparâmetros.

Acerto@1 no grupo de 8: **0,566 ±0,044** contra 0,498 do controle e 0,125 do acaso.

### ⚠️ Erro 1 — o `gte` morreu por bug nosso, não dele

    ValueError: Attempting to unscale FP16 gradients.

`thenlper/gte-base` guarda os pesos em **fp16**, o `transformers` novo carrega no
dtype do checkpoint por padrão, e o `GradScaler` recusa desescalar gradientes fp16 —
o AMP exige pesos-mestres em fp32, porque é ele que faz a passagem em fp16 e o
scaler que desescala de volta. Um modelo já em fp16 não tem para onde.

Consertado com `dtype=torch.float32` explícito em `rerank.py`. Ficou invisível até
aqui porque MiniLM e PhysBERT são fp32 — **qualquer base fp16 quebraria**, e o custo
foi um braço inteiro do experimento.

### ⚠️ Erro 2 — o meu pré-registro confundia "perdeu" com "não rodou"

Com o `gte` ausente, o script imprimiu **"o mecanismo é CONHECIMENTO DE DOMÍNIO, não
diversidade"** — uma leitura que *exige* o `gte` ter produzido um número e perdido. A
lógica olhava só `venceram` e nunca `falhas`.

O pré-registro existia para impedir exatamente esse tipo de conclusão, e o buraco não
estava na regra: estava na implementação dela. Agora qualquer braço ausente torna o
mecanismo **INCONCLUSIVO**, e há um teste que **executa** a lógica nos quatro
desfechos em vez de procurar texto.

**O que está estabelecido:** uma base diferente vence a fusão.
**O que segue aberto:** *qual propriedade* da base faz isso — domínio ou capacidade
de pré-treino. Só o `gte` separa as duas, e ele está sendo refeito com o conserto.

### Vazão medida contra a minha estimativa

PhysBERT treinou a **35,4 exemplos/s** com **2.332 MB** de VRAM. Eu estimei 14–24/s
ao usuário — **pessimista por ~2×**. As avaliações caíram na faixa estimada (13 min o
de 23 M, ~40 min o de 109 M). Execução inteira: 1 h 42 min, dos quais ~50 min o
treino e ~40 min a avaliação da variante que sobreviveu.

E 2,3 GB num cartão de 16 GB dizem que `--grupos 2` subutiliza a placa. Aumentar
seria mais rápido e **quebraria a comparabilidade com o controle**, então fica —
`--grupos` é um dos seis parâmetros que o teste de uma-variável tranca.

## Bake-off do tokenizer — §11.1 medido, e dois achados contra o documento (2026-08-27)

    200.000 resumos do arXiv para treinar · 5.000 reservados para medir · CPU · US$ 0

    var  algo      vocab    §8     fert  ×Qwen3  tok/eq  ×Qwen3  LaTeX 1-tok
    A    BPE      40.960   sim   0,9620   0,674    7,31   0,732      2/3
    B    Unigram  40.960   sim   0,9816   0,688    8,36   0,837      0/3
    C    BPE      32.768   sim   0,9973   0,699    7,49   0,750      2/3
    D    BPE      65.536   sim   0,8966   0,629    6,97   0,698      2/3
    E    BPE      40.960   NÃO   1,3245   0,929    9,46   0,947      0/3
    F    Qwen3   151.643    —    1,4263   1,000    9,99   1,000      0/3

Round-trip 100% e `1.5` consistente em todas as seis. 35 subáreas medidas, pior
razão 1,215 contra o limite de 1,25 do §11.1.

### 1. A §8 está vindicada, e o teste era do próprio documento

O §11.2 estipula: *"Se E empatar com A, a §8 está errada e o documento precisa ser
revisado"*. **Não empatou.** A e E são idênticas exceto pelas regras de
pré-tokenização, e A dá 27% menos tokens em prosa e 23% menos em equações. E é a
**única variante que falha a meta de fertilidade** (0,929 contra o teto de 0,80).

### 2. BPE bate Unigram em LaTeX — evidência que não existia

O §3.3 declara: *"Bostrom & Durrett (2020) mostram vantagem do Unigram em linguagem
natural; **não há evidência publicada para LaTeX/Física**"*. Produzir essa evidência
era a contribuição própria prometida, e ela saiu: **BPE ganha**.

A margem é seis vezes maior em equações (13%) que em prosa (2%), e o mecanismo é
observável — o Unigram aprendeu **0 de 3** sequências LaTeX como token único, o BPE
aprendeu **2 de 3** (`rac` e `\partial`). A poda iterativa do Unigram não retém
essas unidades; a fusão do BPE retém. Não é só um número: é uma explicação.

### 3. ⚠️ V = 40.960 NÃO é o melhor, e isso contraria o §7

A ordem é monotônica nas três métricas: 32k (0,9973) → 41k (0,9620) → 64k (0,8966).
Quanto maior, melhor, em toda a faixa testada. Consistente com Tao et al. (2024),
que o próprio §7.3 cita como ressalva.

**Mas a métrica intrínseca não vê a troca que decide.** O DOC-07 §2.2 registra que a
embedding é 16% do modelo de 150M com V=40.960; em 65.536 seria ~24% — parâmetros
gastos em embedding em vez de camadas. Fertilidade não captura isso, e é precisamente
por essa razão que o §11.2 (treinar 50M em 5B tokens por variante) é o veredito.

**Nenhum número desta seção autoriza mudar o §7.** O que ele autoriza é incluir
V=65.536 no bake-off de verdade, em vez de tratar 40.960 como decidido.

### 4. A meta de fertilidade em equações falha em TODAS as variantes

O §11.1 pede ≤ 0,65× do Qwen3. A melhor é D com 0,698. Nenhuma alcança — negativo
honesto, e provavelmente ligado ao corpus: resumos têm matemática inline curta, e o
alvo pode ter sido calibrado supondo texto pleno com equações em display, que é
justamente o que o peS2o não tem (ver a seção das equações mutiladas).

Medido por `scripts/bakeoff_tokenizer.py` · `data/processed/tokenizer/bakeoff.json`.

## ⚠️ O peS2o tem as EQUAÇÕES REMOVIDAS — e isso NÃO bloqueia o ΦEnc

> **A primeira metade desta seção está certa; a conclusão foi SUPERADA em
> 2026-08-31.** O peS2o de texto pleno está mutilado, e isso é sólido. O que estava
> errado é o que eu concluí disso: que o acesso pago ao fonte do arXiv passava a
> pré-requisito. O **RedPajama-arXiv**, no disco desde 2026-08-14, é construído do
> **fonte LaTeX** e tem ambiente de equação em **84,9%** dos documentos contra
> **0,0%** do peS2o. 835.379 documentos de Física, 12 GB, custo US$ 0.
>
> A seção fica de pé porque a medição do peS2o é boa e a fatia continua imprópria
> para o objetivo do DOC-07 §2.3. Ver [ADR-0002](docs/adr/ADR-0002-fonte-latex-para-o-phienc.md)
> para a decisão que substitui a "decisão que isto força" no fim daqui, e a seção
> de 2026-08-31 no topo deste arquivo para o que a correção custou.

Medido antes de gastar cota de GPU:

    fonte                     operador órfão   órfãos/doc   com LaTeX
    peS2o texto pleno              50,2%          3,2         15,5%
    arXiv resumos                   3,0%          0,0         23,2%

"Operador órfão" é um ` = `, ` < ` ou ` > ` sem operando de um dos lados — a
assinatura de uma equação que foi arrancada do texto. Metade dos documentos de
texto pleno do peS2o tem pelo menos um, com média de 3,2 por documento. Nos resumos
do arXiv é 3,0% e praticamente zero por documento.

O que sobra no lugar da matemática, em documentos diferentes:

    "a pair of elements ρ, σ ∈ Σ are said to be orthogonal if: where {0} is..."
    "The stationary solution p * to Eq. (7) satisfies p * = W · p *"
    "the dependence of __ as a function of normal pressures __"

No primeiro, a equação em display entre "if:" e "where" foi **apagada inteira** — o
dois-pontos aponta para o nada. No segundo, `p^*` virou `p *` e `T_{eff}` virou
`T eff`: os índices foram achatados em tokens soltos pela extração de PDF.

### Por que isto bloqueia, e não é só ruído

**O DOC-07 §2.3 propõe mascaramento consciente de equações** — mascarar uma equação
inteira e reconstruí-la a partir da prosa. É a única adição específica de Física ao
objetivo de treino, e a hipótese que o ΦEnc existiria para testar. **Não é possível
sobre texto de que as equações foram removidas.**

E o DOC-05 inteiro — tokenizer nativo com ~2.000 sequências de controle, pré-tokenização
que preserva `rac{d^2x}{dt^2}`, tratamento estrutural de índices — pressupõe LaTeX
íntegro. Sobre peS2o, esse orçamento de vocabulário não tem o que representar.

### O que isso faz com o corpus de 27,75 B

Volume não é o gargalo; **integridade matemática é**. Dos 14,60 B do peS2o, a maior
parte do valor em tokens vem do texto pleno — que é justamente a fatia com as
equações mutiladas. Os resumos do arXiv têm matemática íntegra e são a fonte de
melhor qualidade que existe aqui, mas são ~1,1 mil caracteres por documento.

### ~~A decisão que isto força~~ — o que eu concluí, e por que estava errado

Eu escrevi aqui que o **arXiv pago** passava de "conveniência" a **pré-requisito da
hipótese central do ΦEnc**, e listei três alternativas ruins. Em 2026-08-31 abri as
recomendações ao usuário com isso, cotado em US$ 100–180.

**Duas coisas erradas.**

A cotação: o conjunto de fonte é ~2,9 TB (março de 2023) e os dois conjuntos somam
~9,2 TB (abril de 2025); o bucket é `requester pays` e o arXiv remete à tabela da
AWS. A US$ 0,09/GB de egresso, o fonte inteiro para fora da AWS passa de **US$ 400**.
Filtrando dentro da AWS e egressando só o `.tex`, dezenas de dólares. Nunca foi
US$ 100–180.

E o pré-requisito: **não é**. `scripts/medir_equacoes_mutiladas.py` agora mede as
quatro fatias, e a resposta estava no disco:

    fatia                          ch/doc   LaTeX%   $..$%   AMBIENTE%   seq/doc
    arXiv resumos (referência)      1.123     21,8    26,9       0,0        0,9
    RedPajama-arXiv                49.212    100,0    99,6      84,9    1.158,3
    OpenWebMath                    16.009     78,5    85,2       5,1       57,7
    peS2o texto pleno              28.605     16,3    18,2       0,0        1,0

⚠️ **E o meu diagnóstico saturava.** O "operador órfão" acusa **81,3%** do RedPajama,
contra 3,0% da referência — e os trechos crus mostram matemática perfeita. Em
`$Z_{\rm max}$ = 15 kpc` o caractere antes de ` = ` é `$`, que o regex não aceita
como operando. Um diagnóstico que dispara mais no corpus íntegro que no mutilado é
pior que nenhum. O discriminante que funciona é **presença de ambiente de equação**:
84,9% contra 0,0%.

**A regra deste repositório é medir antes de gastar cota. Eu a apliquei à GPU e não
ao dinheiro.** Ver [ADR-0002](docs/adr/ADR-0002-fonte-latex-para-o-phienc.md).

## G1.2 — passou pela letra, e o resultado honesto é PARIDADE (2026-08-27)

    2.000 candidatos · protocolo do veredito · o modelo treinado na T4

    modelo                        r@1     r@10   nDCG@10
    ΦEmb-T4 (23M)               0,262    0,708    0,4657
    GTE-large (genérico 335M)   0,278    0,677    0,4628
    ΦEmb/MiniLM local (23M)     0,254    0,700    0,4579
    PhysBERT (alvo do G1.1)     0,146    0,425    0,2752
    SciBERT (base)              0,109    0,328    0,2074

**G1.1: passou de verdade.** +0,190 de nDCG sobre o PhysBERT, pareado p=0,0000.
Margem grande, significativa, sem ressalva.

**G1.2: passou pela letra do critério, e a margem NÃO é evidência de superioridade.**
+0,0029 de nDCG sobre o GTE-large. E o pareado em recall@1 dá **empate com o GTE à
frente na contagem**: 196 discordantes contra 165, p=0,114.

Os dois modelos trocam qualidades:

    GTE-large é melhor em colocar a resposta em 1º   (r@1  0,278 vs 0,262)
    o nosso é melhor em trazê-la ao top-10           (r@10 0,708 vs 0,677)

O nDCG@10 divide a diferença e cai do nosso lado por três milésimos.

### O que os dois números juntos dizem

Este documento registrava "perde do GTE-large por 0,005". Agora é "+0,003". Um
deslocamento de 0,008 entre duas corridas da MESMA receita não é melhoria — é
ruído, e o que o par de medições estabelece é que **sempre foi empate**.

A afirmação que sobrevive a escrutínio: **um modelo de 23M parâmetros empata com um
de 335M neste benchmark**, 1/14,8 do tamanho. Eficiência de parâmetros, medida.
"Batemos o GTE-large" não é defensável com +0,003.

### As ressalvas que mantêm o G1 aberto

  · benchmark PRÓPRIO (pares de citação), não um reservado e publicado
  · G1.3 (ΦEnc em classificação/NER), G1.4 (ΦOCR) e G1.5 não são tocados aqui
  · nDCG@10 com UM relevante por consulta é 1/log2(1+pos); julgamentos graduados
    dariam outro número

## peS2o filtrado — o corpus dobra (2026-08-26)

    22 arquivos · 87,1 GB baixados · 0 falhas · concluído

    vistos            : 38.972.211
    aceitos           :  5.526.331  (14,18%)
    texto aceito      :     58,40 G caracteres
    ~tokens           :     14,60 B

Com isto o corpus sai de 13,15 B para **27,75 B tokens** — na metade superior da
faixa de 15–30 B que o DOC-07 §2 exige para o ΦEnc. Deixa de faltar dado.

Saída: `data/processed/pes2o_fisica/` · 277 parquets · 18 GB · no HD.

### Três defeitos achados por um teste de fumaça de UM arquivo

Rodar um arquivo antes dos 22 custou 29 minutos e evitou perder o dia inteiro:

**O leitor tentava parquet num `.json.gz`.** O peS2o v1 é `.json.gz`; a exceção era
capturada como AVISO e o resumo saía com código 0. Os 22 arquivos falhariam igual e
a coleta produziria um diretório vazio com um resumo satisfeito. Consertado com
leitura em FLUXO (os arquivos têm 1,6 a 7,1 GB comprimidos) e uma guarda: zero
registros VISTOS agora levanta. Taxa de aceitação zero pode ser legítima; não ter
lido nada nunca é.

**O repositório publica v1 E v2 da mesma coleção**: 22 arquivos e 100,7 GB de v1,
22 e 87,1 GB de v2. O filtro pegava as duas — os mesmos papers duas vezes, ~31 h
para um corpus com metade duplicada. E duplicação em pré-treino não é desperdício,
é dano: o S3b mediu que o RedPajama **degrada 16,6%**. `FONTES` passa a carregar um
prefixo (`data/v2/`), e `filtrar` RECUSA lista que abranja mais de uma versão.

**A retomada guardava ordinais sem guardar a lista.** Ao restringir para v2, a
"unidade 1 feita" deixou de ser `data/v1/train-00000` e virou `data/v2/train-00000`,
nunca processado. `assinatura_da_lista` sela os ordinais e `feitas()` levanta se
divergir — é o mesmo erro que `retomada.py` existe para impedir, um nível acima.

### A ressalva que o volume NÃO resolve

Os falsos positivos de 1,5–13,6% que justificam o limiar 0,9 foram medidos em
**resumos do arXiv**. Isto é texto completo de paper — outra distribuição, e a taxa
de contaminação aqui **não está medida**. O filtro separou 400 documentos em
`_amostra_para_revisao.json` para julgamento humano, justamente porque inventar o
número daqui seria extrapolar demais.

## T1a — o ΦEmb na T4, e o número que destrava o ΦEnc (2026-08-26)

    dispositivo: Tesla T4 (CUDA 7.5, 16 GB) · 400.000 pares · 3.125 passos

    | | RX 7600 (DirectML) | Tesla T4 (CUDA) |
    |---|---|---|
    | vazão            | 20-26 pares/s      | **181,6 pares/s** |
    | 400 mil pares    | ~13 h, 3 mortes    | **36 min, sem queda** |
    | precisão         | fp32 obrigatório   | fp16 + GradScaler |
    | VRAM             | estourava no 128   | 1.508 MB de 16 GB |

**7 a 9× mais rápido.** E o ganho de qualidade, no mesmo log:

                     recall@1   recall@10      MRR
    base (MiniLM)       0,270       0,666    0,403
    ΦEmb na T4          0,316       0,816    0,476
    ganho              +0,046      +0,150   +0,073

Melhor checkpoint no passo 3.000: nDCG@10 **0,5512**.

⚠️ **Medido com 1.000 candidatos**, e o veredito do G1 usa 2.000. Estes números NÃO
são comparáveis ao portão — o veredito local está pendente (a máquina está com o
peS2o, e RAM livre de 2 GB não comporta os dois).

### O que isso significa para o ΦEnc

Um par são duas sequências de 192 tokens, então 181,6 pares/s ≈ **69.700 tokens/s**
na T4 contra ~7.700 aqui:

    20 B tokens ÷  7.700 tokens/s = 37 DIAS      (RX 7600)
    20 B tokens ÷ 69.700 tokens/s = ~80 HORAS    (T4)

80 h são **menos de 3 semanas** da cota gratuita de 30 h/semana, em sessões de até
9 h com retomada a cada 100 passos. O ΦEnc sai de "impossível nesta máquina" para
"questão de agendar" — e é estimativa CONSERVADORA: o lote ficou em 128 de propósito
para isolar o dispositivo, e a T4 usou 1,5 dos 16 GB.

### Os cinco defeitos entre o notebook pronto e o notebook rodando

Nenhum era do treino. Todos eram diferença entre o ambiente local e o do Kaggle:

| defeito | sintoma |
|---|---|
| conta sem verificação por telefone | `machine_shape` aceito e ignorado; imagem de CPU fixada |
| `enable_gpu` / `accelerator` são campos mortos | quem liga a GPU é `machine_shape` (lido do SDK, não da doc) |
| Kaggle DESCOMPACTA `.zip` no upload | `phifm_src.zip` chegou como diretório; hash do código deixou de ser conferível |
| layout do input mudou com a imagem nova | `/kaggle/input/datasets/<dono>/<slug>/` em vez de `/kaggle/input/<slug>/` |
| subprocesso não herda `sys.path` da célula | `ModuleNotFoundError: No module named 'phifm'` |

E dois defeitos de VISIBILIDADE, que foram os caros:

`check=False` no subprocesso do treino transformava treino morto em notebook
`COMPLETE`. Uma execução terminou "com sucesso" sem nenhum modelo.

`kernels output` da API devolveu log de **0 bytes em três execuções seguidas**. Sem
log, o diagnóstico virava adivinhação — e adivinhar custou quatro execuções. A
correção (gravar a saída do treino em `/kaggle/working/treino.log`, com stderr no
mesmo arquivo) deu a causa exata na execução seguinte.

## T1b — a fusão fecha, e o reranker é um no-op (2026-08-24)

    1.000 consultas · universo de 88.807 documentos citados da validação · top-50

    sistema                      r@1    r@10    r@50   nDCG@10
    BM25                       0,067   0,236   0,401    0,1399
    ΦEmb                       0,055   0,233   0,428    0,1327
    ΦEmb+BM25 (RRF)            0,068   0,271   0,446    0,1584   <- entregável
    ΦEmb+BM25+ΦRank            0,064   0,254   0,446    0,1493

    PAREADO contra a fusão (McNemar exato, mesmas consultas):
      top-10  BM25              69 a 34  · a fusão vence (p=0,00073)
      top-10  ΦEmb              69 a 31  · a fusão vence (p=0,00018)
      top-10  +ΦRank            61 a 44  · empate (p=0,118)
      top-1   +ΦRank            31 a 27  · empate (p=0,694)

    -> data/processed/avaliacao/t1b_resultado.json

**O entregável do T1b é a fusão**, e ela vence os dois recuperadores isolados com
significância. O reranker não acrescenta nada mensurável: o pareado dá empate, e o
estimador de ponto inclina contra ele.

### O que foi consertado, e o que o conserto revelou

O ΦRank treinado com `minerar_negativos.py` **invertia o recuperador**. A mineração
tomava como negativo o top-K do denso menos a citação verdadeira, o que rotula
NEGATIVO tudo que o recuperador põe no topo — e o modelo aprendeu
`muito recuperado ⇒ não é a resposta`. Como no T1b os candidatos SÃO o top-50 do
recuperador, ele rebaixava justamente o que a fusão promovia: nDCG 0,0179, pior que
ordem aleatória, com r@1 de 0,007.

`minerar_do_recuperador.py` monta o grupo do RRF top-50 de verdade — a distribuição
exata da avaliação. Com isso "estar no topo" deixa de prever o rótulo, porque o
positivo também está no topo. O efeito é inequívoco:

    Spearman(posição na fusão, escore)   antigo      novo
                                        +0,179     -0,466
    consultas com rho > 0                  83%         0%

    escore médio por faixa       antigo      novo
      posições  0-4              -4,319    -0,106
      posições  5-14             -4,053    -1,443
      posições 15-29             -3,812    -2,249
      posições 30-49             -3,723    -2,726

O sinal virou: de preferir a cauda do recuperador para concordar com ele. E é
exatamente por concordar tanto que o reranker não ganha nada — ele **re-deriva a
ordem que a fusão já produziu**. No grupo de 8 ele é forte (acerto@1 0,498 ±0,045
contra ~0,20–0,25 do próprio RRF nos mesmos grupos), mas essa força não vira ganho
sobre 50 candidatos porque não é informação nova.

### Por que provavelmente é o ΦEnc que falta, e não a mecânica

O DOC-07 §4 pede o cross-encoder **inicializado do ΦEnc**. O ΦEnc não existe — exige
15–30 B tokens de texto completo e o S3 entregou 13,15 B sem treinar. O desvio
declarado usa o `all-MiniLM-L6-v2`, que é **a mesma base do ΦEmb**.

Um reranker que parte do mesmo modelo que o recuperador carrega o mesmo
conhecimento, e um cross-encoder só ganha do bi-encoder quando vê algo que o outro
não vê. O Spearman de -0,466 é a medida disso: ele concorda com o ΦEmb porque *é* o
ΦEmb, só com atenção cruzada. A mecânica de reranking está exercitada e correta; o
que falta é um encoder com conhecimento próprio.

### Os dois defeitos de medição que apareceram no caminho

**`head` num parquet agrupado.** `avaliar` usava `val.head(500)` e o arquivo vem
agrupado por documento citado: eram 35 papers repetidos ~14 vezes, não 500
observações. Com n efetivo 35 o acerto@1 de 0,364 tinha intervalo de ±0,159 e não se
separava da base de 0,198 — e a divisão contaminada e a honesta reportaram o mesmo
número porque as duas mediam os mesmos 35 papers. O mesmo `head` estava no treino,
cobrindo ~700 documentos em vez de ~9.000. Corrigido em `amostrar_por_documento`
(módulo sem torch, para o teste rodar na suíte rápida), e as métricas agora imprimem
o intervalo sobre o n efetivo.

**Chave mentindo no JSON.** `avaliar_t1b.py` gravava `recall_100` mesmo com
`--profundidade 50`. Valor certo, rótulo errado — o tipo de coisa que um relatório
futuro copia sem verificar.

### Hipóteses medidas e descartadas, para não serem reinvestigadas

| hipótese | medição que a matou |
|---|---|
| `sdpa` inferindo errado no DirectML | dif_max 4e-6 contra `eager` na CPU, ordenação idêntica |
| negativos de outra população | 100% dos minerados são documentos citados |
| grau de citação como atalho | positivo 113,6 contra 8,5 (13,3×) **mas** Spearman(grau, escore) = +0,056, p=0,17 |
| atalho de formato/superfície | 14 features, melhor AUC 0,575 (número de pontos) |
| desempate pelo índice 0 no `argsort` | embaralhar a posição do positivo dá números idênticos, como tem de ser num cross-encoder par a par |
| "o modelo não lê a consulta" | veio de amostra de **16** documentos; com 457, ler a consulta vale +0,143 ± 0,059 |

### Onde está a folga

`recall@50` de 0,446 é o teto: um reranker perfeito daria nDCG 0,446 contra os
0,1584 de hoje — **2,8× dentro dos candidatos que já recuperamos**. A folga existe e
é grande; o que não existe é um modelo com conhecimento além do ΦEmb para explorá-la.
Subir o recall seria a outra rota, e as duas alavancas baratas para isso já foram
medidas planas.

## G1.5 — o corpus por um hash, e o que ele prova

    hash raiz  bbd73a7a26ac8e8b03b7bb9c142bbb47d459d7a32d643fec63b477cf25cf5fb7
    33 etapas · 976 arquivos · 21,79 GB · verificação profunda ✅

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/manifesto_corpus.py --verificar --profundo
```

Cadeia de Merkle em três níveis: o hash raiz cobre o hash de cada manifesto de
etapa, que cobre o BLAKE3 de cada arquivo. Verificação **rasa** custa
milissegundos e pega manifesto mexido; **profunda** relê os 21,79 GB e é a única
que pega parquet mexido. A distinção está declarada porque chamar o resultado da
rasa de "corpus verificado" seria ausência de erro lida como sucesso.

### Por que 🟡 e não ✅

O critério tem duas metades, e só uma está fechada.

**Fechada:** o corpus neste disco é atestado por um hash, com proveniência por
etapa — entradas, parâmetros, contagem, git sha. Reconstruir a partir da camada
bruta que guardamos e conferir contra o manifesto funciona e está testado.

**Aberta:** refazer a coleta do zero **não** reproduz os mesmos bytes. O arXiv
OAI-PMH filtra por *datestamp*, e o datestamp muda quando um autor publica versão
nova — uma coleta refeita amanhã traz registros que a de hoje não tinha. É a
semântica da fonte, não defeito do coletor, e a solução é a que já está em uso:
guardar e hashear a camada bruta, que é a nossa cópia fixada.

Segunda ressalva: os parâmetros das etapas já executadas são **reconstruídos** do
código, não capturados na execução — cada manifesto carrega
`parametros_reconstruidos=True`. A marca cai quando cada etapa passar a gravar o
próprio manifesto ao terminar.

### Dois defeitos que o G1.5 expôs, e nenhum era do manifesto

**1. As fatias baixavam de `resolve/main/`.** Alvo móvel: uma refeitura futura
poderia produzir outro corpus sem erro e sem aviso. Agora fixam a revisão. Tivemos
sorte — o OpenWebMath está em `fde8ef8d` desde 2023-10-17, anterior à nossa
coleta. Sorte não é reprodutibilidade.

**2. O `checksum_index` das coletas não era checksum.** Era
`canonical_hash({"rows": n, "cols": [...]})` — o hash da **forma**. Dois parquets
de conteúdo completamente diferente, com as mesmas linhas e colunas, hasheiam
igual. O DOC-02 §8.1 especifica "mapa doc_id → BLAKE3, endereçado por conteúdo";
a implementação divergiu da especificação sob um nome que promete o contrário, e
ficou assim desde 2026-08-06.

Descoberto porque a verificação profunda acusou **878 parquets "alterados" que
estavam intactos**. A resposta certa a um alarme não é assumir adulteração nem
silenciar o alarme — é descobrir o que ele compara. Consertado: os coletores
gravam `hash_conteudo` (BLAKE3 real), o `checksum_index` fica com o comentário
dizendo o que é, e o construtor do raiz computa o seu próprio índice sobre os
bytes do disco.

### O que a suíte garante aqui

13 testes, e cada um **estraga** algo e exige que o verificador acuse: byte
trocado, manifesto adulterado para "legalizar" a fraude, raiz editada à mão,
arquivo apagado, arquivo a mais. Mais dois que protegem propriedades sem as quais
o resto não vale: **idempotência** (construir duas vezes o mesmo corpus dá o mesmo
hash — confirmado nos 21,79 GB reais) e **etapa declarada e ausente é erro**, não
silêncio, porque um manifesto de corpus incompleto que confere é o pior resultado
possível.

Quatro defeitos foram pegos por esses testes antes de eu declarar o G1.5 pronto,
incluindo um que fez a verificação profunda emitir **1.000 falsos positivos** — e
um verificador que dá mil alarmes falsos é um verificador que se aprende a
ignorar.


## Coletas — como retomar

Todas são **idempotentes e retomáveis**. Basta rodar de novo:

```bash
./scripts/run_harvest.sh arxiv          # macOS / Linux
.\scripts\run_harvest.ps1 snapshot      # Windows — ver o commit 7a28508
```

O `_manifest.json` em cada pasta guarda o cursor durável. A retomada refaz
apenas o lote pendente (entrega "ao menos uma vez"), e a duplicação é
removida pela dedup exata.

| Fonte | Coletado | Situação |
|---|---|---|
| arXiv | **1.595.422 registros**, 78 shards, 674 MB, 0 falhas | ✅ concluído 2026-08-07 05:44 UTC |
| OpenAlex API | 150 mil obras, 10,1 M arestas | ⏸️ substituído pelo snapshot |
| OpenAlex snapshot | em curso | ~5,8 h medidos, ver abaixo |

A previsão de tamanho do DOC-02 §3.1 se confirmou: **422 bytes/registro** em
parquet zstd, contra os 516–686 previstos, e 674 MB contra "~700 MB no total".
A espinha cabe folgado em disco comum, como o documento afirmava — e o número
de registros ficou em 1,59 M, entre o 1,2 M estimado para o set `physics` e o
2,7 M do acervo inteiro.

### OpenAlex: resolvido pelo snapshot (2026-08-06)

A API passou a ser cotada — 1.000 requisições/dia grátis, US$ 0,0001 cada.
Precisamos de 18.336. As rotas, com os números **medidos**:

| Rota | Custo | Tempo | |
|---|---|---|---|
| Esperar a cota | US$ 0 | 18 dias | |
| Pagar a API | US$ 1,83 | ~5 h | |
| **Snapshot** | **US$ 0** | **5,6 h** | ✅ implementado |

```bash
.\scripts\run_harvest.ps1 snapshot
```

O snapshot é livre e sem cota. A estimativa de ~2 h do DOC-02 era otimista —
o corpus dobrou para 725 GB e nada disso é uma transferência sequencial única.
Medido: 5,6 h e 155 GB, sendo os 155 GB o resultado de ler **só 13 das 189
colunas** por faixa de bytes HTTP. Ver a docstring de
[`openalex_snapshot.py`](src/phifm/corpus/acquire/openalex_snapshot.py) para a
progressão de 40 h a 5,6 h e o que dominava cada etapa.

Não é bloqueio imediato: os 10,1 M de arestas já coletados bastam para
começar a treinar o ΦEmb.

## Sessão de 2026-08-10 a 14 — os quatro passos, e o S2 fechado

Ordem de execução trocada em relação à de retorno: o passo 3 é **pré-requisito**
do 1 (sem ele o treino novo perde o pico igual ao anterior), e o 2 é o que
permite julgar o resultado do 1.

| Passo | Estado | Onde |
|---|---|---|
| 3. Guardar o melhor checkpoint | ✅ `817c427` | `training/embedding.py` |
| 2. Comparação pareada | ✅ `f7ed572` | `eval/encoders.py` |
| 1. ΦEmb sobre MiniLM | ✅ concluído 01:17 | `models/phiemb-minilm-melhor` |
| 4a. S3b — auditoria de LaTeX | ✅ **decidido**, ver abaixo | `data/processed/avaliacao/s3b_latex.json` |
| 4c. Classificador `is_physics` | ✅ treinado | `models/isphysics-clf` |
| 4b. RedPajama filtrado pelo spine | ✅ 835.379 docs, **10,54 B tokens** | `data/processed/redpajama_fisica` |
| 4d. OpenWebMath filtrado | ✅ 860.521 docs, **2,62 B tokens**, 114/114 | `data/processed/openwebmath_fisica` |
| 4d. peS2o | ⬜ não iniciado (42,7 h medidas) | — |

### S3b — a resposta é PAGAR, e o caminho até ela tem cinco correções

O número mudou cinco vezes. Só o último vale, e o que o torna confiável não é ser
o último: é ser o único que mede a população certa e declara um intervalo.

| medição | degradação | por que estava errada |
|---|---|---|
| n=6, sem macros | 19,6% | macros do autor contadas como perda |
| n=6, com macros | **2,6%** | amostra de 6 papers |
| n=199 | 27,4% | número único, mistura perda com notação |
| decomposto | 13,4% ausência + 14,0% notação | fonte = tarball, não documento |
| montagem corrigida | 13,4% + 13,0% | população = arXiv inteiro, não Física |
| **n=298, só Física** | **16,6% ausência**, IC [12,9%–20,8%] | ← este |

**Veredito: o RedPajama perde 16,6% das equações de Física, o IC 95% inteiro está
acima do limiar de 10%, e o bulk pago do arXiv (US$ 100–180) se justifica**
(DOC-02 §3.2).

O que cada correção ensinou, porque nenhuma foi cosmética:

**As macros do autor.** A fonte escreve `\Ecal_\mu`, o RedPajama escreve
`\mathcal{E}_\mu`. Das equações que NÃO casavam, 97% usavam macro; das que
casavam, 20%. Confirma o DOC-03 §2.2 ("60–80% dos papers definem macros"), que o
painel do DOC-19 marcava ⬜ nunca testado.

**O número único mandava gastar por engano.** "Ausente" é a equação que o
RedPajama não tem — só isso justifica pagar. "Discordante" é a que está lá com
outra notação, e pagar por ela seria comprar a solução de um problema nosso.
Reportar 27,4% juntava as duas.

**A fonte era o tarball, não o documento.** Eu concatenava todos os `.tex` do
pacote, inclusive rascunhos que o documento não inclui, e contava o texto depois
de `\end{document}`. O viés é assimétrico: inflar a fonte **aumenta** a
degradação, empurrando para gastar. Corrigido em `montar_documento`, que segue
`\input`/`\include` a partir do `\documentclass`.
**E não mudou nada** — a ausência ficou nos mesmos 13,4%. A hipótese estava
errada, e valia mais saber disso que acertar.

**Eu media o arXiv inteiro e chamava de Física.** O shard do RedPajama-arXiv é o
arXiv todo; 52% da amostra não estava no spine. O paper que mais contribuía para
a perda é de **teoria de grafos**, e as "equações perdidas" eram tabelas de ciclos
de permutação num apêndice — listas de inteiros como `(52,0,22,47,31,...)`.
Restringir a Física derruba a discordância de 13% para 9,1% (o comparador foi
escrito para notação de Física) e **sobe** a ausência.

**A estimativa pontual não decidia.** Com 103 papers o IC era [9,9%, 19,1%] —
cruzava o limiar por 0,1 ponto. "97% provável" não é como se autoriza um gasto.
Ampliar para 298 papers de Física fechou: [12,9%, 20,8%], P(>10%) = 100%.
O bootstrap reamostra **papers**, não equações: as equações de um paper
compartilham o destino que o pipeline lhe deu, e tratá-las como independentes
daria intervalo falsamente estreito.

Hipótese testada e **rejeitada**: o RedPajama não trunca por tamanho fixo (não há
acúmulo no topo da distribuição, e o pior paper tem 52 mil caracteres contra um
máximo de 313 mil). Não há atalho de mirar só os papers longos.

### Passo 4c — o classificador (histórico; o resultado final está acima)

`is_physics` treinado: **F1 0,972** nas duas classes, 600 mil documentos.

O rótulo negativo é a regra **autoritativa**, não a lista de prefixos. Negativo
não é "veio do conjunto cs/econ/q-bio" — conjunto do arXiv é por categoria, e um
`cs.LG` com cross-list em `quant-ph` está nos dois:

| conjunto | registros | com cross-list de Física |
|---|---|---|
| cs | 988.244 | 5,7% |
| econ | 16.984 | 5,5% |
| q-bio | 56.142 | **32,8%** |

Dos 72.919 negativos com cross-list de Física, **exatamente** 72.919 estão no
spine e **zero** ficaram fora — os conjuntos OAI-PMH batem com as listas de
categoria, o que valida as duas coletas de uma vez.

#### A fronteira é fuzzy por definição do corpus, e mede-se quanto

Medido em 2026-08-13: **72.872 dos 1.595.422 papers do spine (4,6%) têm primária
FORA da família de Física.** Mais da metade é matemática.

| arquivo primário | papers |
|---|---|
| math | 36.867 |
| cs | 17.578 |
| q-bio | 7.377 |
| eess, stat, q-fin | 6.157 |
| chao-dyn, solv-int | 2.614 ← Física legada, é a dívida do `PHYSICS_PREFIXES` |

Não é defeito: o DOC-02 §2 decide de propósito que "qualquer categoria da família
conta, não só a primária". Mas tem consequência que ninguém havia medido — o
classificador tem `math.AP` como **positivo** (quando há cross-list de Física) e
`math.AP` como **negativo** (quando não há), e a diferença é uma flag que **não
está no texto**.

Previ que isso tornaria os negativos de `math` quase inúteis. **Errado.** Treinando
só em papers de primária `math.*`, positivos contra negativos:

```
              precision    recall  f1-score
      fisica      0.832     0.827     0.829
  nao_fisica      0.828     0.833     0.830
    accuracy                          0.830
```

**83%**, contra 50% do acaso. O rótulo está no texto: um `math.AP` cross-listado em
`math-ph` fala de operadores de Schrödinger e equações de fluidos; um que não é
fala de análise abstrata. Os negativos de `math` vão ajudar.

O número também dá o teto: dentro de `math.*`, ~17% são intrinsecamente
confundíveis, e nenhum limiar remove isso.

Cruzamento das duas regras de rótulo nas fatias de `math` já coletadas
(158.452 registros): concordam em **94,19%**, e a discordância é **assimétrica** —
zero casos de o prefixo dizer Física e o spine não ter (o prefixo nunca é largo
demais), contra 9.207 casos de o spine ter e o prefixo não ver (todos
`math.AP`/`math.PR`/`math.DG`… com cross-list). A regra do spine é a mais
abrangente das duas, e é a que rotula.

#### Resultado final, com `math` coletado (2026-08-14 00h09)

774.063 registros de `math` coletados em 22 fatias, zero falhas. Retreino
estratificado: 300.000 física · 190.210 não-física, acurácia **0,954** (contra
0,972 sem `math` — a queda é saudável, o negativo ficou mais difícil).

Deixa-um-domínio-de-fora nos quatro domínios, 120 mil por classe:

| omitido | FP **dentro** | FP **fora** | piora | precisão fora |
|---|---|---|---|---|
| cs | 3,7% | 9,8% | 2,6× | 0,907 |
| econ | 3,3% | 8,5% | 2,6× | 0,988 |
| **math** | 2,4% | **35,4%** | 14,6× | 0,731 |
| **q_bio** | 3,1% | **31,2%** | 10,2× | 0,907 |

**A coluna que importa para produção é «FP dentro».** Os quatro domínios estão no
treino do modelo final, então ele tem **2,4%–3,7%** de falso positivo em cada um.
Era isso que a coleta comprava.

E «domínio não visto» **não é uniformemente catastrófico** — depende da proximidade.
Omitir `cs` custa 9,8%; omitir `math`, 35,4%. Os dois vizinhos próximos que o arXiv
tem (`math`, `q-bio`) estão agora dentro do treino.

Correção de uma afirmação minha: eu disse que o limiar "estanca em 10%". Verdade
para vizinho próximo não visto, mas a queda de 35,4% para 10,0% é de 3,5× — ajuda
muito, só não zera.

#### O que ainda pode dar errado no 4d, e não foi medido

1. **Texto de web.** Nenhum domínio do arXiv o representa, e o OpenWebMath é isso.
   Nenhum dado que temos responde, e extrapolar de resumos do arXiv para HTML de
   fórum e nota de aula é troca de distribuição maior que qualquer uma medida aqui.
2. **`stat` foi coletado e MEDIDO em 2026-08-14: não é vizinho próximo.**
   149.461 registros, 2,6% de contaminação (o set mais limpo de todos). Omitindo
   `stat` do treino, o falso positivo nele é **2,9% contra 3,0% dentro do
   domínio — piora de 1,0×**, contra 14,6× do `math`. A 0,9 de limiar, 0,4%.

   **Minha suspeita estava errada e o documento estava certo.** Eu argumentei que
   física estatística e estatística compartilhavam vocabulário demais (função de
   partição, entropia, Ising em `stat.ML`). O vocabulário compartilhado existe;
   não basta para confundir o classificador.

   **Decisão: `stat` NÃO entra no treino.** Com 5 domínios a cota estratificada
   cairia de 75 mil para 60 mil, tirando negativos de `cs` e `math` — que SÃO
   confundíveis — para dar lugar a um que o modelo já trata bem sem ter visto. O
   valor do `stat` foi diagnóstico. Reverter isto exige medir que a diluição
   compensa, o que não foi feito.

   Faltam `eess` e `q-fin`, e a prioridade deles caiu: o único vizinho próximo que
   a suspeita apontou não se confirmou. O
   `harvest_negativos.py` os descarta como "negativos limpos, mais negativo fácil
   não ensina fronteira" — **julgamento sem medição, idêntico em forma ao que fiz
   sobre `math`** e que custou 42,1%. Física estatística e estatística compartilham
   vocabulário (função de partição, entropia, Ising em `stat.ML`). Não afirmo que
   `stat` é próximo; registro que ninguém mediu e que agora há como medir.
3. **A cota de 75 mil descarta 87% do `cs` e do `math`.** Troquei volume por
   equilíbrio com base em medição a 80 mil. Se no tamanho real o `cs` piorar além
   dos 2,2% do proporcional, a cota está apertada demais.

**Recomendação para o 4d:** seguir, com limiar alto (≥0,9) e medindo contaminação
numa amostra da saída filtrada. A amostra é a única coisa que responde a pergunta
do texto de web — todo o resto é extrapolação.

#### ⚠️ Transferência de domínio: o 0,972 não transfere (histórico, pré-`math`)

Deixa-um-domínio-de-fora, treinar sem um domínio e testar nele:

| domínio omitido | FP dentro | FP no domínio omitido | piora |
|---|---|---|---|
| q-bio | 1,9% | **32,9%** | 17× |
| **math** | 2,7% | **42,1%** | **15,3×** |

O `math` é o pior, medido em 100 mil negativos (2026-08-13, com as fatias
2005–2020 já coletadas). Precisão desaba para 0,696, e a curva de limiar não
salva: a 0,999 ainda são **12,8%**.

O significado prático: **sem negativos de `math`, filtrar o OpenWebMath admitiria
~42% de conteúdo matemático como Física** — e o OpenWebMath é feito de matemática.
Corpus contaminado quase pela metade na fatia que mais importa. É o que justifica a
coleta, e o número só existiu depois de ela começar.

**17× mais falsos positivos.** E subir o limiar quase não ajuda: de 0,5 para
0,999 a taxa cai de 32,9% para 10,0% e **estanca** — `modified_huber` satura as
probabilidades, então o limiar tem pouca resolução. O piso de ~10% é estrutural.

> **Resolvido em 2026-08-14.** Estas duas linhas diziam que `math` faltava e que
> o q-bio era "o vizinho mais difícil possível". `math` foi coletado (774.063
> registros) e mostrou-se **pior** que o q-bio: 35,4% contra 31,2% de falso
> positivo como domínio omitido. Os dois estão agora no treino, e o resultado que
> vale está em §resultado final. Mantido aqui porque o caminho até o número
> importa — a afirmação sobre o q-bio era palpite meu, não medição.

### Passos 1–2 — o ΦEmb sobre MiniLM

| | recall@1 | recall@10 | MRR |
|---|---|---|---|
| MiniLM-L6 cru | 0,265 | 0,665 | 0,400 |
| **ΦEmb/MiniLM (pico, passo 2.800)** | **0,322** | **0,804** | **0,477** |
| último passo (3.125) | 0,303 | 0,809 | 0,469 |

O passo 3 pagou na primeira execução: o pico deu MRR 0,477 e o último passo
0,469. Sem guardar o melhor, 0,008 iriam embora.

| Lote | pares/s | negativos por âncora |
|---|---|---|
| 8 (o que o SciBERT aguentava) | 4,1 | 7 |
| **128 (escolhido)** | **16,2** | **127** |

⚠️ **A perda não é comparável entre lotes.** O MiniLM registra ~1,5 contra
0,3–0,5 do SciBERT, e isso não é pior: InfoNCE com 128 negativos tem piso mais
alto que com 8 (ln 128 ≈ 4,85 contra ln 8 ≈ 2,08).

### O portão G1 era julgado por métrica que ele não menciona

O DOC-00 §5 pede **nDCG@10**, eu media recall@1. E o G1.2 pede superar o melhor
embedder **geral** com **≤ 1/10 dos parâmetros** — cláusula relativa ao rival:
vencer o MiniLM de 23M com um modelo de 23M dá razão 1/1 e **não fecha nada**.
Entrou o GTE-large (335M) como genérico forte; contra ele o ΦEmb/MiniLM dá 1/14,6.

A comparação pareada mostrou o que duas proporções soltas descartavam: o que era
empate a n=256 (+0,004 contra ±0,031) virou **vitória sobre o MiniLM com p=0,029**
a n=2.000.

| n | erro padrão | margem mínima detectável |
|---|---|---|
| 256 | ±0,031 | ~0,061 |
| 2.000 | ±0,011 | ~0,022 |

#### O resultado, 2.000 candidatos

| modelo | params | nDCG@10 | recall@1 | recall@10 |
|---|---|---|---|---|
| GTE-large (genérico) | 335M | **0,463** | **0,278** | 0,677 |
| **ΦEmb/MiniLM** | **23M** | 0,458 | 0,254 | **0,700** |
| ΦEmb/SciBERT | 110M | 0,429 | 0,227 | 0,673 |
| MiniLM-L6 (genérico) | 23M | 0,370 | 0,205 | 0,560 |
| PhysBERT | 109M | 0,275 | 0,146 | 0,425 |
| SciBERT | 110M | 0,207 | 0,109 | 0,328 |

**G1.1 ✅** — +0,183 de nDCG@10 sobre o PhysBERT contra limiar de +0,05, pareado
com p=0,0000. Sem ambiguidade.

**G1.2 ❌** — o GTE-large ganha por **0,005** em nDCG@10 e vence o pareado em
recall@1 (205 a 157, p=0,0134). A cláusula de tamanho **fecha** (1/14,8, dentro do
1/10 exigido); é a métrica que não.

Foi para isto que o GTE-large entrou. Com só o MiniLM-L6 de 23M no papel de
"genérico", o G1.2 parecia passar. A correção impediu uma afirmação falsa, e o
custo dela foi descobrir que estamos 0,005 atrás em vez de na frente.

Onde estamos de fato: perdemos o primeiro lugar por 0,005 usando **1/14,8 dos
parâmetros**, e ganhamos em recall@10 (0,700 contra 0,677). O GTE acerta mais na
primeira posição; nós colocamos mais no top-10.

Resultado limpo de passagem: o **ΦEmb/MiniLM de 23M bate o ΦEmb/SciBERT de 110M**
com p=0,0054. O menor com 127 negativos no lote vence o maior com 7 — negativos no
lote são limite de **qualidade** do contrastivo, não só de velocidade, e agora está
medido em vez de citado.

~~O caminho para fechar o G1.2 é lote maior (a GPU tem 8 GB e o lote 128 foi o
teto medido), não modelo maior — o modelo maior já perdeu.~~

⚠️ **Isto foi medido em 2026-08-17 e está errado.** Ver
§"As duas alavancas de escala são planas" abaixo: lote maior foi testado e
**piorou**. A frase ficou aqui riscada em vez de apagada porque previsão errada
apagada é previsão que ninguém aprende a não repetir.

### As duas alavancas de escala são planas

Duas hipóteses de escala, testadas de forma independente, mesma base e mesmo
protocolo de 2.000 candidatos. As duas falharam:

| variação | nDCG@10 | contra o campeão | McNemar |
|---|---|---|---|
| **campeão: 400 mil pares, 127 neg** | **0,4579** | — | — |
| 400 mil pares, **511 neg** (GradCache) | 0,4486 | −0,0093 | p=0,636, empate |
| **1,5 M pares**, 127 neg | 0,4520 | −0,0059 | p=0,950, empate |

Nenhuma das duas move a agulha, e as duas nominalmente **pioram**. Os testes
pareados dizem "empate" nos dois casos, então o enunciado honesto não é "piorou":
é **não há ganho detectável**, com 218 e 256 discordantes.

**Sobre os negativos.** 511 contra 127 é a faixa alta da literatura contra a
média, e ainda assim nada. Isso não contradiz o achado de que 127 > 7 — contradiz
a extrapolação de que mais é sempre melhor. A curva satura entre 127 e 511, não
entre 7 e 127.

**Sobre os dados.** O treino de 1,5 M foi interrompido no passo 4.500 de 11.719
(38%), por platô medido, não por falha: os ganhos aconteceram até o passo 2.000, e
de 2.500 a 4.500 — 256 mil pares, metade deles inéditos para o run de 400 mil — o
nDCG@10 oscilou entre 0,528 e 0,542 sem tendência (medido entre 1.000 candidatos,
a avaliação interna do treino). No passo 4.200, com 537.600 pares vistos, MRR e
recall@1 estavam empatados com o campeão, que viu 358.400.

A ressalva de honestidade: 38% não é 100%. O que está medido é que **o pico deste
run** (passo 4.000) é indistinguível do campeão. Não está medido o que 1,5 M
pares completos fariam — mas 2.000 passos de platô é a evidência que justificou
trocar 13,2 h de treino por 1 h de avaliador.

**O que isto custou e o que compra.** Zero dólar de GPU, e elimina as duas
respostas mais baratas para o G1.2. O que resta não testado: base maior (o
ΦEnc-150M do DOC-07, US$ 25–90 alugado) ou supervisão diferente do par de citação.
As duas custam mais que zero, e são decisão do dono do projeto.

### Parâmetros do treino novo, medidos antes de lançar

| Lote | pares/s | negativos por âncora |
|---|---|---|
| 8 (o que o SciBERT aguentava) | 4,1 | 7 |
| 32 | 11,6 | 31 |
| **128 (escolhido)** | **16,2** | **127** |

Negativos no lote são o limite de **qualidade** do contrastivo, não só de
velocidade — 127 está na faixa de 64–256 da literatura, e é a metade do portão
T1 que faltava atacar. Mesmos 400 mil pares do treino anterior de propósito: a
comparação isola a mudança de base e de lote.

⚠️ **A perda não é comparável entre lotes.** O MiniLM registra ~1,5 contra
0,3–0,5 do SciBERT, e isso não é pior: InfoNCE com 128 negativos tem piso mais
alto que com 8 (ln 128 ≈ 4,85 contra ln 8 ≈ 2,08). Comparar perdas de lotes
diferentes é erro de leitura.

### Por que aumentar o conjunto de avaliação não bastava

| n | erro padrão | margem mínima detectável |
|---|---|---|
| 256 | ±0,031 | ~0,061 |
| 2.000 | ±0,011 | ~0,022 |
| 4.000 | ±0,008 | ~0,015 |

A margem entre ΦEmb e MiniLM era **0,004**. Nem com 4.000 candidatos ela sairia
do ruído. O que resolve é comparação **pareada**: os modelos são medidos nos
mesmos itens, e só os **discordantes** informam sobre a diferença. Placar de 32 a
8 em 40 discordantes é evidência que o teste não pareado descarta.

## O que fazer a seguir, em ordem

1. ~~Negativos de `math`~~ — ✅ 774.063 registros, 22 fatias, zero falhas
2. ~~Retreinar `is_physics` e medir transferência~~ — ✅ ver §resultado final
3. ~~**4d · OpenWebMath filtrado**~~ — ✅ 860.521 documentos, 2,62 B tokens, zero
   falhas, 114 de 114 unidades. Contaminação de química ~2,2% visível na
   distribuição de domínios. **peS2o não iniciado** (42,7 h medidas).
4. ~~**Medir se `stat` é vizinho próximo**~~ — ✅ **não é** (1,0×). A suspeita era
   minha, o documento estava certo. `math` segue o pior (42,1% de FP).
5. **Decidir sobre o bulk pago do arXiv** — US$ 100–180. A medição está fechada
   (16,6%, IC [12,9%–20,8%]); a decisão é de orçamento, não técnica.
6. ~~**4b · RedPajama filtrado pelo spine**~~ — ✅ 835.379 documentos,
   42.145.866.036 caracteres = **10,54 B tokens** (contagem exata; o estimado era
   10,56 B ±4%). Com o OpenWebMath, o corpus é **13,15 B tokens**.
7. **Fechar o G1.2** — ⚠️ as duas rotas baratas estão **descartadas por medição**:
   lote maior piorou (0,4486) e mais dados empataram (0,4520), ver §"As duas
   alavancas de escala são planas". O que resta pede dinheiro: base maior
   (ΦEnc-150M, US$ 25–90 alugado) ou supervisão diferente do par de citação. A
   decisão é do dono do projeto, e é a primeira do projeto que não tem versão de
   custo zero.
8. ~~**Fechar o G1.5**~~ — 🟡 metade fechada, ver §"G1.5 — o corpus por um hash".
   O que falta é capturar parâmetros na execução em vez de reconstruí-los, e isso
   se resolve etapa por etapa, de graça, quando cada uma rodar de novo.
9. **`verify/sandbox`** (DOC-10 §3.6) — o sexto verificador. Depende de gVisor
   ou Firecracker; `exec()` com builtins restritos está descartado no próprio
   documento como trivialmente evadível, então não há atalho local.

### A fatiagem do `math`, e por que ela existe

O set inteiro **não coleta**: o arXiv não monta o conjunto de resultados dentro do
timeout dele. Medido em 2026-08-13, com o endpoint saudável (`Identify` 0,3 s):

| requisição | |
|---|---|
| set inteiro | 503 após 183 s — dez tentativas, zero registros |
| fatia de 5 anos | 503 após 183 s |
| **fatia de 1 ano** | **200 após 56 s** |
| fatia de 1 mês | 200 após 40 s |

São 22 fatias, de `earliestDatestamp` = 2005-09-16 (lido do `Identify`) até hoje.
Manifesto por fatia: se 2019 cair, 2005–2018 não são refeitos.

⚠️ `from`/`until` filtram por **datestamp** — quando o metadado foi alterado — não
pela data de submissão. Não abre lacuna (cada registro tem um datestamp só, e
fatias contíguas particionam o set), mas **carrega a distribuição para o fim**: todo
paper já revisado migra para uma fatia recente. Extrapolar o total das fatias
antigas subestima muito.

E o `completeListSize` **não vem** do arXiv, então não há total declarado — nem
para progresso, nem para conferir completude. O sinal de fim é o do protocolo:
`resumptionToken` ausente na última página.

Dívidas registradas, nenhuma bloqueante:

- `PHYSICS_PREFIXES` (`normalize/spine.py`) não tem os arquivos legados
  (`adap-org`, `chao-dyn`, `patt-sol`, `solv-int`, `acc-phys`, `atom-ph`,
  `chem-ph`, `plasm-ph`, `supr-con`). Medido: custou **zero** até agora, porque o
  arXiv retroagiu cross-list atual em todos os 5.478 papers legados do spine. O
  rótulo do `is_physics` não depende dela de propósito.
- `montar_documento` não avalia condicionais (`\if…\else`) nem `\includeonly`.
  Ambos erram para o lado de incluir mais, o que **infla** a fonte — direção
  conservadora para a decisão de gastar.

O subscrito LaTeX saiu desta lista: está feito em `db637ed`, com o raciocínio
registrado abaixo.

## Achados desta sessão que alteraram documentos

| Achado | Impacto | Registrado em |
|---|---|---|
| Endpoint OAI do arXiv mudou | `export.arxiv.org` dá 301 | DOC-02 §3.1 |
| Set `physics` filtra no servidor | 1,2 M em vez de 2,7 M | DOC-02 §3.1 |
| Espinha ocupa ~700 MB, não 150 GB | Cabe em disco comum | DOC-02 §3.1 |
| `primary_location` exclui publicados | Perderia 1,44 M revisados por pares | DOC-02 |
| Chave de junção não está em `ids.arxiv` | 1,5% vs 98,5% de cobertura | `openalex.py` |
| IDs antigos truncados por regex | 41,5% do acervo | teste de regressão |
| **OpenAlex passou a cobrar** | 18 dias grátis ou US$ 1,83 | DOC-02 §3.1 |
| Licenças: 45,9% redistribuível em 2024 | Dimensiona o `PhysCorpus-Open` | ADR-0001 §4 |
| Precisão float64 no verificador | Reprovava toda resposta correta | `numeric.py` |

## Achados de 2026-08-06

| Achado | Impacto | Registrado em |
|---|---|---|
| `.gitignore` sem barra inicial | `corpus/` casava em qualquer nível e engoliu `src/phifm/corpus/` inteiro; `models/` e `checkpoints/` tinham o mesmo defeito | `.gitignore` |
| `SIGALRM` não existe no Windows | O `AttributeError` era engolido pelo `except` de `parse()`, **nenhuma** expressão parseava e o barramento devolvia `INCONCLUSIVE` em tudo | `symbolic.py` |
| `split_symbols` parte identificadores | `hbar → a*b*h*r`, `eps → e*p*s`, `kB → B*k`. Passava porque os dois lados sofriam a mesma mutilação | `symbolic.py` |
| Namespace do SymPy colide com Física | `E` era o número de Euler, `Q`/`N` eram objetos do SymPy, `gamma`/`beta` eram funções especiais | `symbolic.py` |
| `"_" in s` roteava para o LaTeX | `q*E_campo` virava `E_{c}` silenciosamente | `symbolic.py` |
| Termo dominante inverte no infinito | Perto de ponto finito domina a menor potência; no infinito, a maior | `limits.py` |
| **Nada carregava o `.env`** | `base.py` documentava `PHIFM_CONTACT` vindo do `.env`, os coletores liam `os.environ` e não havia carregador — a coleta sairia como `phifm-corpus@localhost`, anônima na prática, sem aviso | `core/env.py` |
| Suspensão do Windows | Equivalente do `caffeinate`: sem `SetThreadExecutionState`, o SO suspende um processo que só fala com a rede | `core/sistema.py` |
| **`Start-Process` não desacopla** | Processo criado por shell entra no job object dele e morre com ele. A 1ª coleta durou 8 min 40 s e morreu **sem traceback**. `Win32_Process.Create` por WMI escapa | `run_harvest.ps1` |
| Layout do snapshot do OpenAlex mudou | `data/works/` → `data/parquet/works/`; 330 GB/250 M obras → **725 GB/510 M** | `openalex_snapshot.py` |
| Snapshot: ~2 h era otimista | Medido **5,6 h** e 155 GB (21% dos bytes, por poda de colunas). Empata com a rota paga em tempo e ganha no custo — mas a margem é comparável, não de ordem de grandeza | `openalex_snapshot.py` |
| `publication_date` é `date` no parquet | Pela API é texto, e o `SCHEMA` compartilhado diz `Utf8`. O polars abortava o `DataFrame` inteiro. **O teste de fumaça não pegou porque as 2 partições sorteadas tinham 0 obras do arXiv — o caminho de escrita nunca rodou** | `test_snapshot_tipos.py` |
| Pool de conexões do `requests` é 10 | Com 16 threads em `prebuscar`, 6 conexões eram descartadas por faixa, cada uma custando um aperto de mão TCP+TLS novo — a latência que a paralelização eliminava, de volta pela porta dos fundos | `openalex_snapshot.py` |
| **Subscrito nomeado virava produto** | `E_{cin}` → `E_{c*(i*n)}`, com `i` e `n` vazando como grandezas livres. E como produto comuta, `\rho_{xy}` e `\rho_{yx}` colapsavam no **mesmo símbolo** | `core/latex/subscritos.py` |
| `v_{0}` e `v_0` eram símbolos diferentes | A forma com chaves não casava com `INEQUIVOCOS`, então a tabela de dimensões era ignorada em silêncio para a notação mais comum | `core/latex/subscritos.py` |

### Subscrito nomeado: o que o commit não conta

`db637ed` traz esta correção, mas a mensagem dele descreve o trabalho anterior
no `symbolic.py` (timeout no Windows, `split_symbols`, namespace `_FISICA`).
Optou-se por não reescrever o histórico, então o raciocínio fica aqui.

**A gravidade não é o subscrito ilegível, é o colapso.** Produto comuta, nome
não. `\rho_{xy}` e `\rho_{yx}` viravam o mesmo símbolo — e resistividade Hall é
antissimétrica (ρ_xy = −ρ_yx). O mesmo para `g_{\mu\nu}` contra `g_{\nu\mu}`.
Ou seja: o parser executava por baixo exatamente a transformação que o DOC-03
§3.2 lista como **rejeitada** — "Normalizar posição de índices" — e que o
canonicalizador se recusa a fazer de propósito. O tratamento preserva a ordem
escrita dos índices, o que é o que separa `g_mu_nu` de `g_nu_mu`.

**Por que a correção ficou em duas camadas, e não só na ingestão.** A
normalização em si é de ingestão (DOC-03 §3), e é lá que mora: produz o
identificador canônico e o mapa de volta ao LaTeX do autor, que é o que
popula o `context["dimensions"]`. Mas `verify/symbolic.parse` aplica **a mesma
função** — importada, não copiada — porque quatro dos cinco consumidores do
barramento (rollout de RLVR, gabarito de benchmark, admissão de sintético e
auto-checagem em inferência) entregam LaTeX que nunca passou pelo pipeline do
DOC-03. Só o filtro de dados passa. Normalizar apenas na ingestão deixaria sem
proteção justamente o RLVR, onde símbolo trocado vira gradiente.

**Achatar sozinho pioraria.** `parse_latex('E_cin')` devolve `E_{c}*(i*n)`, o
que colapsa `E_cin` com `E_cal` em `E_{c}`. Por isso o identificador é trocado
por marcador de subscrito numérico antes do parse — a única forma que o ANTLR
atravessa intacta — e a troca é desfeita nos símbolos depois.

**Fica de fora, declarado no módulo:** índice contravariante (`T^{\mu}_{\nu}`,
que o ANTLR já lia como `T**mu`, **descartando** o subscrito, antes desta
mudança); subscrito com operador (`k_{n+1}`); e `\mathbf`, que o DOC-03 §3.2
trata como semântico e não tipográfico. Corrigir o índice contravariante exige
representar tensor com índices no schema — decisão de dados, não de parser.

## Onde está cada coisa

## Espinha construída — 2026-08-07

```bash
PYTHONPATH=src .venv/Scripts/python.exe scripts/build_spine.py \
    --arxiv data/raw/arxiv_metadata --openalex data/raw/openalex_snapshot
```

| | |
|---|---|
| Registros únicos | **1.595.422** · 0,0% de duplicação por retomada |
| Revisados por pares | 740.823 (46,4%) |
| **Redistribuível** (`train_open`) | **235.795 (14,8%)** — dimensiona o `PhysCorpus-Open` |
| OpenAlex casado | 257.321, com a coleta em curso |

A fração publicável depende fortemente da época, confirmando o ADR-0001 §4:
**0,0% aberto até 2004**, 36,2% em 2020–2024, **48,8% em 2025–2029**.

Previsão de tamanho do DOC-02 §3.1 confirmada: **422 bytes/registro** brutos
contra 516–686 previstos.

## Onde está cada coisa

| | Local | Drive | GitHub |
|---|---|---|---|
| Código, docs, testes | ✅ | ✅ | ✅ desde 2026-08-07 — `src/phifm/corpus/` estava fora, ver achados |
| Manifestos | ✅ | ✅ | ✅ |
| Coletas brutas (285 MB) | ✅ | ✅ | ❌ por decisão |
| Espinha consolidada (150 MB) | ✅ | ✅ | ❌ |
| Classificador (59 MB) | ✅ | ✅ | ❌ |
| `.env` | ✅ | ❌ | ❌ recriar |

## Histórico reescrito em 2026-08-07

O `model.pkl` de 56 MB saiu do histórico com `git-filter-repo --path models
--invert-paths`, seguido de force-push. Resultado local: `.git` de **30 MB para
492 KB**, e um clone novo do GitHub traz **463 KB** com os 38 commits
preservados. Cópia de segurança em bundle foi feita antes.

### ⚠️ O force-push NÃO apaga o blob do GitHub

Isto precisa ficar registrado, porque é contraintuitivo e a documentação do
`filter-repo` não enfatiza:

```
$ gh api "repos/sanchezVB/LLM_F-sica/contents/models/subfield-clf/model.pkl?ref=b18743d"
AINDA LÁ: 58.594.000 bytes
```

Os commits antigos ficam **inalcançáveis, não apagados**. O GitHub continua
servindo qualquer um deles por SHA direto até rodar a coleta de lixo dele, e
isso não é acionável pelo lado de cá — só por **pedido ao GitHub Support**. O
tamanho que a API reporta para o repositório segue em ~30 MB por essa razão.

Consequência prática: quem clonar recebe 463 KB, que era o objetivo. Quem tiver
o SHA antigo ainda baixa o blob. Como o arquivo é um classificador treinado e
não um segredo, isso é higiene, não incidente — mas se algum dia entrar uma
credencial no histórico, **force-push não basta**: tem de rotacionar a
credencial e abrir chamado.

### Armadilha ao redor: `git fetch` traz o lixo de volta

Entre o `filter-repo` e o push eu rodei `git fetch origin` para conferir o que
seria substituído. O `.git` voltou de 492 KB para 30 MB na hora, porque o
remoto ainda anunciava a história antiga. Resolvido com:

```bash
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

Vale para qualquer clone que ainda tenha a história velha: não basta puxar, tem
de expirar o reflog e podar.
