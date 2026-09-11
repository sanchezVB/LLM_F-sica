# Estado do projeto — 2026-09-11

Ponto de retomada para migração de máquina. Instalação em [SETUP.md](SETUP.md).

## Onde estamos

**Corpus de projeto:** completo. 19 documentos + 1 ADR, cobrindo os 20 pipelines.

| Sprint | Estado | Observação |
|---|---|---|
| **S1** · tabela mestra de metadados | 🟢 completo | 1,59 M arXiv + 4,61 M obras; junção de **99,1%** |
| **S2** · classificador de Física | 🟢 completo | subárea + `is_physics`; acurácia **0,954** com os 4 domínios, FP 2,4–3,7% em cada |
| **S3** · fatias do HuggingFace | 🟢 **27,75 B tokens** | RedPajama 10,54 B + OpenWebMath 2,62 B + **peS2o 14,60 B**, custo zero. S3b: o RedPajama **degrada 16,6%** |
| **ΦEmb** | 🟢 **G1.1 ✅ / G1.2 ✅** | nDCG@10 **0,6223** contra 0,5788 do GTE-large — **+0,044** a 1/14,8 dos parâmetros, teto do protocolo **1,0000**. Supera nas **quatro** métricas; em recall@1 o pareado dá p=0,065, então esse ainda não é estabelecido |
| **T1a** · volume × diversidade | 🟢 **−0,052 → +0,044** | cinco runs, uma variável cada. Sinal de platô no 6 M: **15 avaliações consecutivas abaixo do pico** e queda de 1%, contra ≤1 ponto e ~0 nos outros três |
| **Recuperador do sistema** | 🟢 trocado e a cadeia remedida | `phiemb-do-sistema` (6 M), **+0,098** no G1. Mas a cadeia foi de 0,1685 para **0,1688** — **+0,0003** |
| **Base do recuperador** · T1f | 🟡 **montado, esperando cota** | o GTE-base zero-shot EMPATA (0,2755 contra 0,2810 de r@10, p=0,545) **sem uma linha do nosso dado**. A sonda está em `kaggle/t1f_base_gte.py`: GTE-base@400k contra o MiniLM@400k que já está no disco, **2h39** e **zero upload** (reusa o dataset do T1a, assinatura conferida `e7be008b295aab2f`). ⚠️ A primária é nDCG@10 e isso MUDOU de sentido: o T1e tirou o ΦRank, a cadeia virou ΦEmb → top-10 e **nada lê o fundo hoje** |
| **T1e** · profundidade | 🔴 **o ΦRank SAI do sistema** | dobrar o candidato move **1,95%** das consultas (19×20, p=1,0). O teto subiu 0,098 e o recall@10 subiu **+1 consulta** de 195 oportunidades. A cadeia é `ΦEmb → top-10` |
| **Truncagem 192** | 🔴 **não custa nada** | 63,8% das âncoras truncadas e 28,2% dos tokens descartados, e ler 256 ou 384 **não muda o recall** (pareado p=0,08–0,84) e custa 2,9× para embutir. Era a minha melhor aposta |
| **Teto do recuperador** | 🟢 **diagnosticado** | os 37% perdidos têm posto **mediano 396** de 88.807, e só **10 consultas** (0,5%) são inalcançáveis pelos dois métodos. É lacuna de modelo. `@100 → @200` vale **+0,098** de teto — cinco dobras de dado |
| **T1d** · ΦRank retreinado | 🔴 **fechado: dois negativos** | a hipótese da distribuição **não se sustentou** (empate, p=0,50) e o novo NÃO substitui. E a leitura independente: **nenhuma das duas cadeias vence o ΦEmb sozinho** (p=0,14 e p=0,38) — pelo critério pré-registrado, o estágio de reordenação não paga o próprio custo com este recuperador |
| **T1b2** · a cadeia | 🟢 **o BM25 SAIU da composição** | a regra pré-registrada decidiu; e em 2026-09-10 o ΦRank saiu também (T1e). A fusão RRF parou de somar (empate, p=0,95) e cobrava **0,026 de teto**. O ΦRank fica, com a evidência enfraquecida (p=0,0081 → **p=0,086**) |
| **G1.5** · corpus por um hash | 🟡 **o hash cobria 55% do corpus** | ⚠️ o peS2o (18,34 GB, 14,60 B tokens) estava FORA da tabela `ETAPAS` desde 2026-08-26, e o raiz atestava 21,79 GB de um corpus de ~40 GB. Nada acusava: o construtor só percorre a lista, e o relatório sai ✅ sobre o que ele conhece. Agora **40,13 GB · 34 etapas · 1.258 arquivos**, raiz `927ae486…`. Guarda nova: toda fonte de `filtrar_hf.FONTES` tem de ter entrada em `ETAPAS`, código contra código |
| **G1.5** · parâmetros capturados | 🟡 **os 5 scripts capturam; os artefatos são de antes** | `build_spine`, `build_pairs`, `train_classifier`, `coletar_redpajama` e `filtrar_hf` gravam o próprio manifesto ao terminar, com os argumentos da execução. ⚠️ E o `coletar_redpajama` tinha um elo SOLTO: capturava desde o começo e montava `Entrada(caminho=…)` **sem `manifesto_id`** — parecia bem atestado e a proveniência terminava na primeira aresta. `entrada_de()` centraliza a busca do id nas três formas do repositório |
| Barramento de verificação | 🟢 5 de 6 | falta só `sandbox` — exige gVisor/Firecracker |
| **T1a** · ΦEmb na T4 | 🟢 medido | **181,6 pares/s** contra 20-26 aqui; 13 h viram 36 min. Destrava o ΦEnc: ~80 h de T4 em vez de 37 dias |
| **T1b** · busca híbrida | 🟢 **composição completa** | RRF entrega nDCG 0,1576 e vence os isolados (p<0,001); com o ΦRank de PhysBERT vai a **0,1666** (p=0,0062). O reordenador **entrou no sistema** em 2026-09-03 |
| **T1c** · ΦRank de base diferente | 🟢 **fechado: domínio** | PhysBERT vence a fusão (nDCG **0,1666** vs 0,1576, **p=0,0062**); `gte-base`, do MESMO tamanho, **empata** (p=0,637). Não é diversidade de base nem capacidade — é **pré-treino em Física** |
| **ΦEnc** · dado | 🟢 **destravado, US$ 0** | RedPajama-arXiv tem ambiente de equação em **84,9%** contra 0,0% do peS2o. ~10 B tokens de LaTeX íntegro no disco. A recomendação de comprar acesso ao arXiv estava errada — [ADR-0002](docs/adr/ADR-0002-fonte-latex-para-o-phienc.md) |
| **ΦEnc** · código | 🟡 escrito, não treinado | mascaramento de equações, fluxo sem estado, detector de spike, laço WSD. Fumaça em CPU: perda inicial **10,7343** contra ln(40.960)=**10,6204** |
| **ΦEnc** · fatia de avaliação | 🟢 **51,7 M tokens, DISJUNTA** | `part-00022` — uma das 35 partes que o treino não usou. O manifesto declara `disjunto_do_treino` e as 9 excluídas; o avaliador de MLM **recusa** fatia que não declare |
| **ΦEnc** · dados | 🟢 **2,00 B tokens prontos** | 244.295 sequências de 8.192, 6,0 GB. Partes SORTEADAS. `fracao_tratada` **0,903**, taxa efetiva **0,3000** |
| **Revisão do peS2o** | 🟡 amostra REFEITA, julgamento pendente | a amostra anterior cobria **0,67%** do corpus e era 100% resumo. A nova é estratificada: 200 resumo + 200 texto pleno, sorteio uniforme sobre os 277 parquets |
| **ΦEnc** · avaliação | 🟢 **as três medidas rodaram em modelo real** | recuperação em 6 encoders, sonda tensorial em 4, e o MLM por região no ModernBERT-base: **+0,1286 de vantagem em equação SEM tratamento**, o que muda como a medida se lê. Falta o ΦEnc |
| **§11.2** · o bake-off | 🟡 **A×E montado, empacotado** | as seis variantes a 5 B são 263 h ≈ 8,8 semanas de cota. O que cabe é o **par limpo** A×E a 0,8 B: `kaggle/t2a_tokenizer.py`, dois braços de 7,5 h, 5,4 GB empacotados. C e D têm contagem de parâmetros diferente de A (a 48 M a embedding é 43,7%), então carregam confundidor de capacidade. ⚠️ **Empate a 0,8 B NÃO refuta a §8** — registra-se como não decidido |
| **§11.2** · o instrumento | 🟢 **bits por byte, e a acurácia saiu** | acurácia de MLM **não compara vocabulários**: quem parte em pedaços menores acerta mais sem ser melhor, e o viés aponta CONTRA a hipótese. Confirmado num ensaio real — E marcou acurácia maior (0,0237 contra 0,0195) e bits/byte pior (2,890 contra 2,761). `phifm.eval.bits_por_byte`, fumaça com o mesmo modelo contra si mesmo: Δ 0,00000 |
| **Proxy de fertilidade** | 🟢 **erra por 3×, medido** | E gasta **13,6%** mais tokens por documento no corpus de treino real, não os 37,7% da razão de fertilidade. A §11.1 mediu **resumos**, onde a matemática é *inline* e curta. E a §11.1-medido declarava a §8 "vindicada pelo teste que o §11.2 estipulou" — o §11.2 estipulou TREINAR MODELOS; corrigido |
Suíte: **771 testes** na venv rápida (17 saltados) + **21 na venv de treino**, `PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/ -q`.
Mais 9 do laço de pré-treino, que rodam na venv de treino:
`.venv-treino/Scripts/python.exe -m pytest tests/regression/test_laco_pretreino.py -q`
Os que dependem de torch rodam na venv de treino:
`.venv-treino/Scripts/python.exe -m pytest tests/regression/test_g1_criterios.py tests/regression/test_comparacao_pareada.py tests/regression/test_melhor_checkpoint.py tests/regression/test_gradcache.py tests/regression/test_estado_progresso.py -q`
E os do ΦEnc de ponta a ponta (exportador + corrente até o avaliador do G1):
`.venv-treino/Scripts/python.exe -m pytest tests/regression/test_exportar_phienc.py tests/regression/test_phienc_ate_a_recuperacao.py -q`

## O MLM por região rodou num modelo real, e a leitura dele muda (2026-09-10)

A medida nunca tinha visto um logit. Para rodá-la num modelo de prateleira faltavam
duas peças, agora feitas: o `preparar_dados_phienc.py` ganhou
`--especiais-do-tokenizer` (o default, que produziu os 2,00 B tokens, **não muda**) e
o avaliador passou a ler máscara e especiais **do tokenizer do modelo medido**, em
vez das constantes do projeto.

**ModernBERT-base**, 2.000 sequências da fatia disjunta, 39,4% de token de equação,
**192 s**:

| | |
|---|---|
| acurácia em **equação** | **0,8765** (60.781 tokens) |
| acurácia em **prosa** | 0,7480 (93.155 tokens) |
| **vantagem em equação** | **+0,1286** |

### ⚠️ E é isto que muda a leitura da medida

O ModernBERT-base **nunca viu mascaramento consciente de equação**. Mesmo assim
prevê token de equação muito melhor que prosa — LaTeX é redundante: fechado um
`rac{`, o `}{` e o `}` vêm quase de graça.

**Logo, uma `vantagem_em_equacao` positiva no braço tratado NÃO é evidência da
hipótese do DOC-07 §2.3.** Ela já é +0,13 sem tratamento nenhum. O que testa a
hipótese é a **diferença entre braços** dessa vantagem — uma diferença de diferenças.

Sem essa linha de base ao lado, o número sairia com a cara de sucesso do
mascaramento por span quando é propriedade do formato. A `Contagem.como_dict()`
passou a emitir `linha_de_base_sem_tratamento: 0.1286` junto de cada resultado, e um
teste fixa isso.

### E dois defeitos do mesmo tipo, no mesmo dia

`--emb` e `--modelo` eram `Path`, e no Windows `Path("answerdotai/ModernBERT-base")`
sai com barra invertida do `str()` — o `from_pretrained` não reconhece. Apareceu no
`diagnosticar_teto.py` de manhã e no `avaliar_phienc_mlm.py` à noite. **Onde um
script aceita "modelo", o tipo certo é `str`.**

O segundo custou 3 min de execução: o `.name` num `str` estourou **depois** do laço,
na hora de imprimir. O artefato já estava gravado — a ordem "gravar, depois
imprimir" é o que salvou, e é a mesma lição do `reconfigure(encoding)`.

## O GTE-base zero-shot EMPATA com o nosso ajustado no top-10 (2026-09-10)

Medido local, sem cota: os dois candidatos a base contra o recuperador do sistema,
no universo real de 88.807 e nas mesmas 2.000 consultas.

| modelo | r@1 | r@10 | r@100 | r@200 | mediana | custo |
|---|---|---|---|---|---|---|
| **ΦEmb do sistema** (23 M, 6 M pares) | 0,0655 | **0,2810** | **0,6325** | **0,7300** | 43 | 217 s |
| **GTE-base zero-shot** (109 M) | 0,0625 | 0,2755 | 0,5615 | 0,6450 | 64 | 954 s |
| MiniLM-L6 zero-shot (23 M) | 0,0520 | 0,2130 | 0,4555 | 0,5335 | 152 | 222 s |

| pareado contra o ΦEmb | k=10 | k=100 | k=200 |
|---|---|---|---|
| GTE-base zero-shot | **empate**, p=0,545 | ΦEmb vence, p=5,7e-15 | ΦEmb vence, p=2,2e-20 |
| MiniLM-L6 zero-shot | ΦEmb vence, p=2,7e-15 | ΦEmb vence, p=1,1e-70 | ΦEmb vence, p=3,5e-80 |

**Um modelo de prateleira, sem uma linha do nosso dado, empata com o nosso ajustado
em 6 M de pares — na métrica que o sistema serve.** Depois de o ΦRank sair, o
sistema entrega o top-10 do recuperador, e é ali que o empate acontece.

Ele perde fundo (recall@200 0,6450 contra 0,7300), mas o sistema não usa mais
profundidade.

### O que o ajuste por citação vale, agora medido no universo real

O MiniLM-L6 é a **nossa base**. De zero-shot para o ΦEmb de 6 M:

    recall@10    0,2130 -> 0,2810   (+0,068, p=2,7e-15)
    recall@100   0,4555 -> 0,6325   (+0,177)

O ajuste funciona, e funciona bem. O problema é que ele parte de um lugar baixo.

### ⚠️ E o padrão que isto fecha: o ganho do ajuste é INVERSO ao ponto de partida

| base | zero-shot (G1) | ajustado 400 mil | ganho |
|---|---|---|---|
| SciBERT, 110 M | 0,2537 | 0,4746 | **+0,221** |
| MiniLM-L6, 23 M | 0,4761 | 0,5246 | +0,049 |

Quem começa pior ganha mais e **ainda termina abaixo**. A leitura mecânica é que o
ajuste ensina sobretudo *"seja um encoder de recuperação"* — coisa que o GTE já
sabe. Se for isso, o ganho dele com os nossos pares pode ser **pequeno**, e o empate
de hoje não se converteria em vitória.

É especulação com mecanismo, não medição. E é exatamente o que o experimento
seguinte resolve.

### O experimento que decide, e por que ele é 400 mil e não 6 M

**GTE-base ajustado em 400 mil pares contra o nosso MiniLM ajustado em 400 mil.**
Uma variável — a base —, e o ponto de comparação já existe.

⚠️ Comparar GTE-base@400k contra o ΦEmb@6M seria mudar base *e* volume. E um
GTE-base@6M custa **~39 h de T4** (4,7× o custo por par, 15× os pares), o que é
inviável na cota gratuita. Então o run de 400 mil é uma **sonda** para decidir se as
39 h valem, não a resposta final.

**Custo: 2h39.** Não cabe nas ~1h30 que sobraram desta semana.

### E o custo de serviço, que entra na decisão

O GTE-base levou **954 s** para embutir o mesmo universo contra **217 s** do nosso —
**4,4×**. Hoje isso compra um empate no top-10. Mesmo que o ajuste o coloque à
frente, a margem tem de pagar 4,4× de inferência em serviço, para sempre.

## O ΦRank SAI do sistema, pela regra pré-registrada (2026-09-10)

O T1e mediu a curva de profundidade em uma passagem: 2.000 consultas, 200
candidatos, com @50 e @100 saindo **de graça** dos mesmos escores. 3h08.

| sistema | r@1 | r@10 | teto | nDCG@10 |
|---|---|---|---|---|
| ΦEmb | 0,0655 | 0,2810 | — | 0,1585 |
| ΦEmb+ΦRank @50 | 0,0670 | 0,2930 | 0,5210 | 0,1664 |
| ΦEmb+ΦRank @100 | 0,0675 | 0,2945 | 0,6325 | 0,1676 |
| ΦEmb+ΦRank @200 | 0,0670 | 0,2950 | **0,7300** | 0,1673 |

### 1. A regra: EMPATE, e empate tira o estágio

| confronto (k=10) | placar | discordantes | p |
|---|---|---|---|
| @100 contra @200 | 19 × 20 | **39** | **1,000** |
| @50 contra @200 | 51 × 55 | 106 | 0,771 |

⚠️ **O número que importa é o 39.** Dobrar o conjunto de candidatos mudou o
desfecho de **1,95% das consultas**, e as dividiu 19/20 — cara ou coroa.

### 2. E o mecanismo é mais forte que o empate

O teto subiu de 0,6325 para 0,7300: **195 consultas a mais** passaram a ter o alvo
entre os candidatos. O recall@10 foi de 0,2945 para 0,2950 — **+1 consulta**.

**O reordenador não consegue trazer para o top-10 um alvo que está entre a posição
100 e a 200 do recuperador.** Não é "traz menos": é ~nenhum, de 195 oportunidades.

É a predição da §3.3 do rascunho, confirmada em escala: consertados os negativos, o
ΦRank passou a **concordar** com o recuperador (Spearman −0,466), e *um reordenador
que re-deriva a ordem do recuperador não tem informação nova para dar*.

### 3. A segunda leitura, independente: nenhuma profundidade vence o ΦEmb

| cadeia | k=1 | k=10 |
|---|---|---|
| @50 | p=0,857 | p=0,166 |
| @100 | p=0,791 | p=0,139 |
| @200 | p=0,859 | p=0,138 |

### ⚠️ 4. E o teste que eu pré-registrei era o INSTRUMENTO ERRADO

McNemar sobre "o alvo chegou ao top-k" mede **pertencimento**. Um reordenador que
ordenasse o top-10 perfeitamente mudaria muito o nDCG e **nada** o recall@10 — então
o critério que escrevi é cego para metade do trabalho de um reordenador.

Calculei o teste que casa com a pergunta, bootstrap pareado sobre o nDCG@10 por
consulta (20.000 reamostras):

| cadeia | Δ nDCG | IC 95% | p |
|---|---|---|---|
| @50 | +0,0079 | [−0,0017, +0,0177] | 0,112 |
| @100 | +0,0091 | [−0,0010, +0,0191] | **0,080** |
| @200 | +0,0088 | [−0,0016, +0,0192] | 0,098 |

**Também não estabelece.** O intervalo inclui zero nas três.

E a decomposição mata o argumento de vez. Das **421** consultas com o alvo no top-10
nos dois sistemas, o ΦRank **desce** o alvo em 167 e **sobe** em 141 (p=0,154) — a
direção é *contra* ele. Todo o +0,0091 vem de pertencimento (+27 líquido: traz 168,
tira 141), e **nada** de ordenar melhor, que é o que um reordenador deveria fazer.

### O veredito, e o que ele custa

A regra dizia: empate → a profundidade fica em 100, e como o T1d já mostrou que o
ΦRank não está estabelecido em @100, **o estágio SAI**. Sai.

São **cinco medições pareadas** (T1d antigo, T1d novo, T1e @50/@100/@200), duas
delas com reordenadores diferentes, e em nenhuma o estágio vence o recuperador
sozinho. Mais o bootstrap, que também não. Mais a direção negativa dentro do top-10.

**A composição do sistema passa a ser `ΦEmb → top-10`.** O `phirank-physbert-melhor`
fica em `models/` como ponto da curva do T1c — é a evidência de que pré-treino em
Física é a base certa para um cross-encoder —, mas não está na cadeia.

⚠️ **Nota de reprodutibilidade:** o @100 do T1e deu **141 × 168** em k=10, idêntico
ao braço antigo do T1d medido no dia anterior, noutra sessão. A lição do T1b2 —
medir os braços na mesma sessão — é sobre **deriva de código e protocolo**, não sobre
ruído de execução: com o mesmo código e o mesmo protocolo, o número é o mesmo.

## A truncagem a 192 tokens não custa nada, e era a minha melhor aposta (2026-09-10)

Todos os cinco runs da T1a usaram `max_tokens=192`, e nunca foi variado. Medido: a
**mediana** de uma âncora é **228 tokens**, então **63,8% são truncadas** e **28,2%
de todo o texto** nunca chega ao modelo. Parecia o lever mais barato do recuperador.

A versão gratuita da pergunta — o ΦEmb **já treinado** lendo mais texto, no universo
real de 88.807 e nas mesmas 2.000 consultas, três execuções locais de 40 min:

| max_tokens | r@1 | r@10 | r@50 | r@100 | r@200 | r@1000 | custo |
|---|---|---|---|---|---|---|---|
| **192** | 0,0655 | 0,2810 | 0,5210 | 0,6325 | 0,7300 | 0,9045 | 217 s |
| 256 | 0,0680 | 0,2765 | 0,5125 | 0,6410 | 0,7280 | 0,9105 | 498 s |
| 384 | 0,0665 | 0,2795 | 0,5045 | 0,6290 | 0,7255 | 0,9090 | 625 s |

**Nada.** Os deltas ficam em ±0,005, trocam de sinal entre profundidades, e o pareado
não separa nenhum par:

| | k=10 | k=100 | k=200 |
|---|---|---|---|
| 256 contra 192 | p=0,349 | p=0,082 | p=0,728 |
| 384 contra 192 | p=0,839 | p=0,592 | p=0,431 |

E custa **2,9× mais** para embutir o mesmo universo (217 s → 625 s) — que é custo de
serviço, não só de avaliação.

O controle a 192 **reproduziu byte a byte** o `teto_do_recuperador.json` da manhã, o
que é o que autoriza ler a tabela.

### ⚠️ O que isto NÃO prova, e por que mesmo assim decide

Não prova que **treinar** a 384 não ajudaria. O modelo foi ajustado a 192: ele pode
ter aprendido a comprimir os primeiros 192 tokens e simplesmente não usar o resto, e
um treino a 384 poderia aprender a usá-lo. A evidência é contra a expectativa, não
contra a hipótese.

Mas ela **remove a razão de esperar** o ganho, e era essa expectativa que justificava
gastar cota. A previsão registrada antes de medir era: *"resumo científico é
carregado na frente — título e primeiras frases dão o tópico, e a cauda costuma ser
método e resultado"*. É o que o número diz.

**Decisão: a truncagem sai da frente da fila.** De lever mais promissor do
recuperador passou a ser o mais fraco dos não testados.

## Onde estão os 37%: é lacuna de modelo, não teto da tarefa (2026-09-10)

O T1d fechou dizendo que o trabalho seguinte é no recuperador. Antes de gastar T4,
a medição que decide *qual* trabalho: o posto verdadeiro do alvo entre os **88.807**
documentos, nas mesmas 2.000 consultas, com o BM25 como controle. Sete minutos na
RX 7600.

| faixa do posto | ΦEmb | BM25 |
|---|---|---|
| 1 | 131 (6,6%) | 120 (6,0%) |
| 2–10 | 431 (21,6%) | 337 (16,9%) |
| 11–100 | 703 (35,1%) | 451 (22,6%) |
| **101–1.000** | **544 (27,2%)** | 451 (22,6%) |
| 1.001–10.000 | 175 (8,8%) | 418 (20,9%) |
| **> 10.000** | **16 (0,8%)** | 223 (11,2%) |
| **mediana** | **43** | 161 |

### A resposta: o prêmio está perto, e é grande

Das **735 consultas** perdidas no top-100, a **mediana do posto é 396** — percentil
99,55 de 88.807. O modelo já as coloca quase no topo; elas só não passam do corte.

E o teto da tarefa é minúsculo: **10 consultas** (0,5% do total) têm o alvo além de
10.000 para os **dois** métodos. Dois recuperadores de viés oposto falhando juntos é
a evidência mais forte disponível sem julgamento humano — e ela diz que quase nada
aqui é irrecuperável.

**Os 37% são lacuna de modelo, não limite da tarefa.**

### ⚠️ A profundidade é o lever mais barato, e por uma margem absurda

| profundidade | recall |
|---|---|
| @10 | 0,2810 |
| @100 (o atual) | 0,6325 |
| **@200** | **0,7300** |
| @500 | 0,8460 |
| @1.000 | 0,9045 |

Ir de 100 para 200 candidatos sobe o teto em **+0,098**. Pela curva de volume
(+0,0196 por dobra), o mesmo ganho exigiria **cinco dobras de dado** — 32× os pares,
~294 h de T4. A profundidade custa **2× o tempo de reordenação e zero de treino**.

⚠️ Com a ressalva que o T1d impõe: profundidade só vale se houver reordenador para
explorá-la, e o T1d disse que ele não está estabelecido em @100. As duas coisas se
combinam numa pergunta só — **o reordenador ganha quando tem 200 candidatos em vez
de 100?** — e essa ainda não foi feita.

### E o BM25 sai de novo, por uma porta independente da regra do T1b2

A comparação de **orçamento igual**, que é a única honesta: "denso@100 + bm25@100"
custa até 200 candidatos, então o controle é `denso@200`.

| composição | candidatos | recall |
|---|---|---|
| denso@100 | 100 | 0,6325 |
| união(denso@100, bm25@100) | ≤200 | 0,6835 |
| **denso@200** | 200 | **0,7300** |

O denso sozinho vence por **+0,047** no mesmo orçamento, e o padrão se repete em toda
profundidade: o que o BM25 acrescenta é sempre **cerca de metade** do que os mesmos
candidatos extras do denso dariam (+0,051 contra +0,098; +0,026 contra +0,051; +0,010
contra +0,020).

Isto **não** era o que a regra do T1b2 testou — ela comparou a fusão RRF truncada em
100 contra o denso em 100. Cheguei a supor que o defeito da fusão fosse a truncagem,
e não a informação do BM25; a conta acima **refuta essa suposição**. O BM25 não tem
o que acrescentar em orçamento igual, e a decisão de tirá-lo se confirma por dois
caminhos independentes.

## O T1d fechou com DOIS negativos, e o segundo é maior (2026-09-10)

Treino de 50 min e duas avaliações de 85 min cada — **3h40**, contra as ~3h37 que a
projeção dava. Os dois reordenadores na mesma sessão, `--sem-fusao`, mesmas 2.000
consultas, universo de 88.807, teto de 0,6325.

### 1. A regra pré-registrada: EMPATE, e por regra o novo NÃO substitui

| | antigo × novo | p |
|---|---|---|
| top-1 | 48 × 41 | **0,525** |
| top-10 | 92 × 82 | **0,495** |

Nem sombra na direção da hipótese: o **antigo** está nominalmente à frente nos dois
k, e no nDCG@10 (**0,1676** contra 0,1632). A hipótese de que treinar o ΦRank na
distribuição que ele vê recuperaria o ganho perdido **não se sustenta**.

E o diagnóstico interno é coerente: no próprio grupo de treino o novo é
*ligeiramente melhor* — acerto@1 **0,508** contra 0,498, ambos ±0,045. Ele aprendeu
o que devia aprender e isso não chegou à cadeia. É o teto de novo.

### 2. ⚠️ A leitura independente, que vale mais: a reordenação não paga o custo

O critério estava escrito antes: *"se NENHUMA das duas cadeias vencer o ΦEmb sozinho
em k=10, o estágio de reordenação não paga o próprio custo com este recuperador"*.

| braço | ΦEmb × ΦEmb+ΦRank (k=10) | p |
|---|---|---|
| novo | 157 × 174 | 0,379 |
| antigo | 141 × 168 | **0,139** |

**Nenhuma vence.** Pelo critério pré-registrado, o estágio de reordenação — que
custa 85 min de T4 por avaliação e 109 M de parâmetros em serviço — **não está
estabelecido** contra simplesmente entregar o top-10 do recuperador.

### O que isso NÃO diz, e a conta que resolve

Não diz que o efeito é zero. O nDCG@10 sobe **+0,0091** (0,1585 → 0,1676) e o
pareado é nominalmente favorável (168 contra 141). O que se pode afirmar é
**"não estabelecido com 2.000 consultas"**, não "não existe".

E dá para dizer quanto custaria estabelecer. Com 309 discordantes em 2.000 consultas
e 54,4% deles a favor da cadeia, 80% de poder a α=0,05 exige **1.026 discordantes**
— cerca de **6.640 consultas**, ou **4,7 h de T4 por braço**.

Essa é a decisão que sobra: 4,7 h para estabelecer (ou refutar) um ganho de 0,0091
num estágio que custa 109 M de parâmetros em serviço. **A conta não fecha a favor de
medir**; fecha a favor de gastar as mesmas horas no recuperador, onde 37% das
consultas ainda não recebem o alvo no top-100.

## O BM25 saiu da composição pela regra pré-registrada (2026-09-08)

A pergunta do run, escrita na célula **antes** de rodar: o reordenador vai melhor
sobre o top-100 da fusão RRF ou sobre o top-100 do ΦEmb sozinho? E a regra:

> se `ΦEmb+ΦRank` VENCER `ΦEmb+BM25+ΦRank` no pareado, o BM25 sai da composição.
> Se EMPATAR, ele sai também — não paga o próprio custo nem os 0,026 de teto que
> cobra. Só fica se vencer.

As duas cadeias na MESMA execução, mesmas 2.000 consultas, mesmo reordenador em
memória, universo de 88.807:

| sistema | r@1 | r@10 | r@100 | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0,060 | 0,229 | 0,454 | 0,1340 |
| ΦEmb | 0,066 | 0,281 | **0,632** | 0,1585 |
| ΦEmb+BM25 (RRF) | 0,072 | 0,282 | 0,607 | 0,1643 |
| ΦEmb+BM25+ΦRank | 0,068 | 0,297 | 0,607 | **0,1688** |
| **ΦEmb+ΦRank (sem fusão)** | 0,068 | 0,294 | **0,632** | 0,1676 |

**A cadeia com fusão fica 0,0012 de nDCG à frente — e isso não é vencer.** No
top-10 ela acerta 594 das 2.000 contra 589: um líquido de **5 consultas**.

### ⚠️ O pareado da regra NÃO foi calculado pela rodada

Os oito pareados do artefato têm a **fusão** como referência, porque era a pergunta
original do T1b ("o reranker acrescenta algo à fusão?"). O confronto que a regra
nomeia — cadeia contra cadeia — não estava lá. Duas comparações contra um terceiro
sistema não são a comparação entre elas.

Deu para decidir por aritmética, e é uma aritmética fechada: o McNemar exato só olha
os discordantes, e para um líquido de 5 o menor `p` possível é o caso extremo 5 a 0,
que dá **0,0625**. Qualquer outra estrutura de discordantes dá mais. Logo a fusão
**não podia vencer sob nenhuma estrutura** — a regra resolvia sem o número.

Mas isso foi sorte do tamanho do efeito, não desenho. O `avaliar_t1b.py` agora
calcula `confronto_das_cadeias` diretamente, e um teste guarda tanto a existência do
confronto quanto o argumento aritmético em forma executável.

### O que sai disto, concretamente

1. **A composição do sistema é `ΦEmb → ΦRank`.** O BM25 sai. Ele não estava em
   nenhum caminho de serviço — só no avaliador e no minerador —, então a mudança é
   de projeto e de default, não de deploy.
2. **O minerador de negativos mudou de default junto** — e isto é o ponto que
   importa. Todo o argumento do `minerar_do_recuperador.py` é *treinar na
   distribuição do teste*; a distribuição do teste acabou de mudar. Continuar
   minerando da fusão seria repetir o defeito de 2026-08-24 com outra roupa, e é a
   causa mecânica provável de o ganho do ΦRank ter encolhido. `--com-fusao`
   reproduz os negativos antigos e existe só para isso.
3. **O teto subiu 0,026** sem custo nenhum: 0,6065 → **0,6325**. Ainda assim 37%
   das consultas não recebem o alvo no top-100, e nenhuma melhora de reordenação
   alcança essas.

Custo do run: **10.938 s** (3h02) — 16,6 s de BM25, 177 s de embutir o universo,
**10.744 s de reordenar** (98%), agora com duas passagens em vez de uma.

### E os negativos novos já estão minerados — com 23% mais grupos

Rodou local, na RX 7600 por DirectML, em **7 minutos** (444–489 sequências/s). O
grupo agora vem do top-50 do ΦEmb, que é a composição decidida:

| | RRF, 2026-08-24 | **ΦEmb, 2026-09-08** |
|---|---|---|
| recall@50 (âncoras que produzem grupo) | 0,476 | **0,546** |
| grupos de 30.000 âncoras | 14.289 | **16.391** |
| negativos por grupo | — | 47,96 |
| posição média do alvo no candidato | — | 12,48 (p50 7, p90 35) |

**+14,7% de grupos de treino, de graça.** Um reranker só age quando o alvo está no
conjunto de candidatos, então a melhora do recuperador aparece duas vezes: mais
grupos utilizáveis *e* na distribuição que o modelo vai ver na avaliação.

⚠️ O 0,476 de agosto veio do `registros: 14289` do manifesto por arquivo daquela
rodada. Eu havia escrito "~13.290", estimando de 0,443 × 30.000 — e o 0,443 da
docstring do minerador é de outra medição, não daquela execução. O número estava
no disco e eu derivei em vez de ler.

### E a co-citação passou a remover MAIS, não menos

O `filtrar_cocitacao.py` sobre os negativos densos remove **12,51%** (98.331 de
786.063, sobrando 41,96 por grupo). Sobre os da fusão, em agosto, removia **9,1%**.

Faz sentido e é a favor do filtro: um recuperador melhor traz ao top-50 mais
documentos topicamente próximos, e proximidade tópica é justamente o que faz dois
papers serem citados juntos. **O filtro importa mais agora, não menos** — e sem
ele o ΦRank aprenderia a afastar o que a literatura agrupa, que foi o defeito
medido em 2026-08-18.

⚠️ **Duas correções no minerador saíram desta rodada.** Ele gravava o resumo num
nome fixo (`_do_recuperador.json`) e a primeira execução **sobrescreveu em
silêncio o resumo de 2026-08-24** — o diretório não é versionado, então aquele
número só sobreviveu porque estava citado na docstring do próprio script. O resumo
agora leva o nome do parquet, e os rótulos `posicao_do_alvo_no_rrf` e o `por_que`
deixaram de afirmar "RRF" quando a composição é o denso.

⚠️ **E a mineração não é reprodutível bit a bit.** Duas execuções com a mesma
semente deram 16.362 e 16.391 grupos: a soma em float na GPU não é associativa,
documentos de cosseno quase igual trocam de lugar, e perto da posição 50 isso
decide se o alvo entra no grupo. 0,18%. Inofensivo para negativos (são entrada de
treino), mas **o hash do parquet não bate entre execuções** e nenhum número deste
script serve de medida.

Falta passar por `filtrar_cocitacao.py` antes de treinar: co-citados com o
positivo continuam entrando como negativo.

## O exportador do ΦEnc, e o defeito que ele pegou em 2 minutos (2026-09-08)

O bloqueio era real: o `laco.py` grava `torch.save({"modelo": state_dict, ...})`, e
`AutoModel.from_pretrained` precisa de um `config.json` ao lado dos pesos. O
`scripts/exportar_phienc.py` fecha isso, com quatro recusas — configuração vinda do
run e nunca de um nome, tokenizer conferido contra o `vocab` da configuração,
`strict=True` no load, e run inacabado ou com spike exigindo `--mesmo-assim`.

### ⚠️ O avaliador de recuperação do §11.2 NÃO era código novo

`scripts/avaliar_encoders.py` já carrega um diretório com `AutoModel`, faz média
mascarada e roda o protocolo do G1 com teto e McNemar. Escrever um segundo daria
duas réguas com o mesmo nome. O que faltava era só o exportador — e o §11.2 passa
a ser `avaliar_encoders.py --modelo "variante A=models/phienc-variante-A"`.

### E a medida discrimina MLM puro, sem estágio contrastivo

A dúvida era se seis encoders sem treino contrastivo cairiam todos no ruído. Os
números que já estavam no `g1_resultado.json` respondem: **SciBERT 0,2537 e
PhysBERT 0,3507** — MLM puro, média mascarada, zero contrastivo, **0,097 de
separação** entre encoders que diferem no domínio do pré-treino. É a comparação
análoga à do §11.2, e tem faixa de sobra. Nenhum estágio de adaptação é necessário,
o que poupa ~6 h de T4 no bake-off.

## A sonda de estrutura tensorial: especificada, e calibrada antes de servir (2026-09-08)

O §11.2 nomeia "uma sonda de estrutura tensorial" e **não a especifica**. A
especificação está em `src/phifm/eval/sonda_tensorial.py`, desenhada para testar o
que a **§8** afirma ser "a decisão mais consequente e menos visível" do documento —
que `^{` e `_{` precisam ser pré-tokens atômicos.

Cada item é um trio, sobre o mesmo carregador de texto:

| | expressão | o que muda |
|---|---|---|
| base | `T^{\mu\nu}` | — |
| estrutural | `T_{\mu\nu}` | mesmos símbolos, estrutura diferente |
| renomeado | `T^{\alpha\beta}` | estrutura igual, símbolos diferentes |

O item passa quando `sim(base, renomeado) > sim(base, estrutural)`: **separar**
objetos diferentes E **identificar** renomeação de índice. Medir um lado só é fácil
de satisfazer pelo motivo errado — quem colapsa tudo passa na invariância, quem
decora string passa na separação. 4 famílias × 6 tensores × 3 carregadores = 72 itens.

### ⚠️ A superfície reprova a sonda inteira: 0 de 72

`T^{\mu\nu}` → `T_{\mu\nu}` é **um caractere**; o renomeado muda dois símbolos.
Por semelhança de string o estrutural é o mais parecido — o oposto do que o item
pede. Semelhança de trigramas de caractere acerta **zero**, com margem negativa nas
quatro famílias. Isso dá uma escala que a maioria das sondas não tem: 0,0 é
superfície, **0,5 é moeda**, acima de 0,5 é estrutura — e o acaso fica *entre* os
dois regimes, então o sinal já diz de que lado o modelo está.

### E nenhum encoder existente chega a 0,5

| modelo | taxa | margem |
|---|---|---|
| piso (só superfície) | 0,0000 | −0,0784 |
| **ModernBERT-base** (MLM) | **0,2222** | −0,0017 |
| SciBERT (MLM puro) | 0,2083 | −0,0068 |
| PhysBERT (MLM, Física) | 0,0278 | −0,0271 |
| MiniLM-L6 (**contrastivo**) | 0,0000 | −0,1794 |

**A sonda discrimina** — 0,00 a 0,22 entre quatro encoders, e por família até 0,50
de faixa. Mas **todos estão do lado da superfície**, então o que ela mede no regime
de hoje é *quanto o modelo resiste ao empurrão da string* — que é exatamente a
pergunta da §8.

E o padrão tem mecanismo: o **único modelo treinado contrastivamente é o mais preso
à superfície de todos**, enquanto os de MLM puro são os menos presos. Treino
contrastivo de sentença empurra para semelhança de forma; MLM não.

### ⚠️ E um critério meu que estava errado, registrado no código

Duas famílias (`contracao`, `operador_contra_campo`) deram 0,0000 nos quatro
modelos, e as outras duas variaram até 0,50. Escrevi um `familia_tem_faixa()` com
critério estrutural para explicar isso — e ele **reprovava a `posicao_do_indice`,
que é justamente a que discrimina melhor**.

O predetor certo é a **margem de superfície da família**, computável das strings sem
modelo, e a ordem é exata: −0,039 e −0,046 discriminam; −0,078 e −0,152 ficam presas
no zero. Mas um limiar ajustado a quatro pontos seria escolher família pelo
resultado com cara de critério. Então **nada é cortado**: a margem sai ao lado de
cada taxa, e quem lê vê qual família estava presa no piso.

### O teste de ponta a ponta pegou um `TypeError` que teria custado 44 h

Um ΦEnc de brinquedo (2 camadas, vocab 512) atravessando laço → exportador →
avaliador levou 2,5 min e falhou: o `PreTrainedTokenizerFast` embrulhado declarava
`token_type_ids` por default e `ModernBertModel.forward()` **não aceita** esse
argumento. O artefato passava em toda conferência de pesos e morria na primeira
linha de quem fosse usá-lo.

Corrigido no `model_input_names`, e a exportação agora confere também
`modelo(**tokenizer(texto))` — uma terceira conferência, porque pesos idênticos e
logits idênticos não dizem nada sobre a compatibilidade entre os dois artefatos.

## O recuperador melhorou 0,098 e a cadeia melhorou 0,0003 (2026-09-08)

Os dois recuperadores medidos na MESMA sessão, mesmo commit, mesmas 2.000
consultas, universo de 88.807. O BM25 sai **idêntico** nos dois braços
(nDCG@10 0,1340, recall@100 0,4540) — é o controle que prova que o protocolo não
mudou entre eles.

| sistema | recall@100 | nDCG@10 | | recall@100 | nDCG@10 |
|---|---|---|---|---|---|
| | **recuperador NOVO** | | | **ANTIGO** | |
| BM25 | 0,4540 | 0,1340 | | 0,4540 | 0,1340 |
| ΦEmb sozinho | **0,6325** | 0,1585 | | 0,5160 | 0,1323 |
| ΦEmb+BM25 (RRF) | 0,6065 | 0,1643 | | 0,5380 | 0,1594 |
| **+ ΦRank** | 0,6065 | **0,1688** | | 0,5380 | **0,1685** |

**O recuperador sozinho ganhou muito** — nDCG@10 +0,026, recall@100 **+0,117**. **A
cadeia inteira ganhou +0,0003.** Praticamente nada.

### ⚠️ A fusão RRF parou de somar, e agora custa

No braço novo o **recall@100 da fusão (0,6065) é MENOR que o do ΦEmb sozinho
(0,6325)**. Misturar o BM25 — cujo recall@100 é 0,4540 — desloca candidatos bons
do top-100 e **derruba o teto do reranker em 0,026**.

E o pareado confirma a virada:

| RRF contra o ΦEmb sozinho | top-1 | top-10 |
|---|---|---|
| com o recuperador ANTIGO | RRF vence, **p=0,0017** | RRF vence, **p=1,5e-08** |
| com o recuperador NOVO | empate, p=0,203 | empate, **p=0,949** |

A fusão era um ganho grande quando o ΦEmb era pior que o BM25 em parte do espectro.
Com o ΦEmb novo ela **não acrescenta nada** e cobra teto. Isto é o achado mais
acionável do dia.

### O ΦRank encolheu, como a hipótese pré-registrada previa

| | ganho marginal sobre a fusão | pareado no top-10 |
|---|---|---|
| recuperador ANTIGO | +0,0091 | vence, **p=0,0081** |
| recuperador NOVO | +0,0045 | empate, **p=0,086** |

Exatamente o previsto na célula, escrito **antes** do número: o `recall@100` do
recuperador é o teto do reranker, e um recuperador melhor deixa menos para
reordenar. **O critério pré-registrado para decidir CONTRA o ΦRank era a cadeia
completa ficar abaixo da fusão** — e ela não ficou (0,1688 > 0,1643). Ele fica, com
a evidência enfraquecida.

E há uma causa mecânica provável: o ΦRank instalado foi treinado com negativos
difíceis **minerados pelo recuperador antigo** (`minerar_do_recuperador.py`). A
distribuição de treino dele não é mais a distribuição que ele vê. Retreiná-lo com
negativos do recuperador novo é o experimento seguinte, e é barato.

### ⚠️ Os dois braços salvaram a conclusão de um erro de 7×

O número histórico é **0,1666**. O recuperador ANTIGO, medido nesta sessão, dá
**0,1685**. Se eu tivesse comparado 0,1688 contra o 0,1666 de agosto, teria
reportado **+0,0022** da troca do recuperador — sete vezes o efeito real de
**+0,0003**. A diferença era deriva de código e protocolo entre agosto e hoje.

Era a decisão de desenho central do T1b2, escrita antes de rodar, e ela pagou.

### E duas estimativas minhas erradas, na direção oposta

Estimei ~50 min para os dois braços. No meio do run, lendo o log, "corrigi" para
2h52 por braço. **O real foi 76 min por braço** — 18 s de BM25, 172 s de embutir o
universo, 73 min de reordenar (96% do custo).

A segunda estimativa errou porque eu li o `5309.0s` do log do Kaggle como tempo
de setup do braço 1, quando é **tempo cumulativo do notebook**: naquele instante o
braço 1 já havia terminado e o que carregava pesos era o braço 2. Corrigir sobre um
carimbo mal lido é pior que não corrigir.

Custo real gravado em `t1b2_resultado.json`; o `avaliar_t1b.py` agora grava
`custo_segundos` sozinho.

## 6 M de pares: supera o GTE-large nas quatro métricas (2026-09-08)

A curva de volume completa, uma variável por vez, protocolo de teto **1,0000** e
2.000 candidatos:

| run | documentos citados | nDCG@10 | margem vs GTE-large |
|---|---|---|---|
| 400 mil `head` | 17.844 | 0,5265 | −0,0523 |
| 400 mil sorteado | 191.198 | 0,5462 | −0,0326 |
| 1,5 M sorteado | 390.856 | 0,5780 | −0,0008 |
| 3 M sorteado | 518.635 | 0,6026 | +0,0238 |
| **6 M sorteado** | **650.162** | **0,6223** | **+0,0435** |
| GTE-large (335M) | — | 0,5788 | — |

### Agora as quatro métricas, e a ressalva de uma delas

| | ΦEmb 6M (23M) | GTE-large (335M) | |
|---|---|---|---|
| nDCG@10 | **0,6223** | 0,5788 | **+0,044** |
| recall@10 | **0,8305** | 0,7640 | **+0,067** |
| MRR | **0,5636** | 0,5293 | **+0,034** |
| recall@1 | **0,4315** | 0,4140 | +0,018 — ⚠️ pareado **p=0,065** |

O recall@1 virou a nosso favor em número, mas o **pareado não estabelece**
(188×153 discordantes, p=0,0654). O honesto: **superamos em nDCG@10, recall@10 e
MRR; em recall@1 lideramos sem significância**. Contra o run de 3 M, o de 6 M
vence com p=0,0012.

### Primeiro sinal de virada da curva

⚠️ **A estatística que estava aqui era a errada, e a conclusão sobreviveu à
troca.** A versão anterior comparava a *fração do treino* em que o pico caiu — 93,4%
no run de 6 M contra 99,8% no de 3 M — e concluía "é a primeira vez que a curva não
termina subindo". **É falso:** o run de 400 mil teve o pico a **89,6%**, ainda mais
cedo. A fração não mede saturação; mede onde a grade de avaliação (a cada 200 passos)
calha de pegar o máximo.

A estatística certa é **quanto a curva caiu depois do pico, e por quantos pontos**:

| run | passos | pico | pontos APÓS o pico | queda até o fim |
|---|---|---|---|---|
| 400 mil | 3.125 | 0,6147 | 1 | 0,0002 (0,03%) |
| 1,5 M | 11.718 | 0,6567 | 1 | 0,0010 (0,15%) |
| 3 M | 23.437 | 0,6804 | 0 | 0,0000 |
| **6 M** | 46.875 | **0,7022** | **15** | **0,0071 (1,01%)** |

Nesse recorte o sinal é muito mais forte do que a fração sugeria: o run de 6 M passou
**~3.000 passos e 15 avaliações consecutivas sem bater o próprio pico**, com queda de
1%. Os outros três nunca passaram de **um** ponto de avaliação abaixo do máximo, com
queda indistinguível de ruído.

Continua não sendo platô de dado provado — é o comportamento de **um** run. Mas agora
a evidência é a que a afirmação precisa, e não uma coincidência de grade.

Ganhos por dobra, no protocolo do portão: **+0,020 / +0,032 / +0,025 / +0,020**.
Ainda não decrescem monotonicamente, e o dado restante é pouco: 6 M de 6.564.111
pares, 650.162 de 667.304 documentos. **O próximo ganho não vem de mais pares
destes** — vem de outra fonte de pares, de outra base, ou de mais parâmetros.

## ⚠️ O recuperador do sistema mudou, e a cadeia está por remedir (2026-09-08)

`phifm.core.modelos.RECUPERADOR` aponta para `models/phiemb-do-sistema` (o run de
6 M). O anterior, `phiemb-minilm-melhor`, dava **0,5246**: a troca é **+0,098**.

**Instalado em caminho novo, não sobre o antigo.** O `phiemb-minilm-melhor`
continua em `models/` porque é um ponto da curva em `avaliar_encoders.py` —
sobrescrevê-lo apagaria a evidência de que 400 mil pares sobre MiniLM dão 0,5246.

✅ **A referência do T1b/T1c foi remedida** (ver a seção do T1b2 acima). O nDCG
**0,1666** de agosto era da fusão RRF sobre o recuperador ANTIGO; com o novo a
cadeia dá **0,1688**, e a razão para esperar menos ganho do ΦRank se confirmou —
o `recall@100` do recuperador é o teto do reranker, e um recuperador melhor deixa
menos para reordenar.

Custo medido: a avaliação da cadeia embute os **88.807** documentos do universo e
reordena 2.000×100 pares com um cross-encoder de 109M. Em CPU local isso é da
ordem de horas; o caminho é o mesmo do T1c — a T4 gratuita.

## O G1.2 passou — 3 M de pares sorteados superam o GTE-large (2026-09-07)

Quatro runs, uma variável por vez, todos no protocolo de teto **1,0000** e 2.000
candidatos:

| run | documentos citados | nDCG@10 | margem vs GTE-large |
|---|---|---|---|
| 400 mil `head` | 17.844 | 0,5265 | −0,0523 |
| 400 mil sorteado | 191.198 | 0,5462 | −0,0326 |
| 1,5 M sorteado | 390.856 | 0,5780 | −0,0008 |
| **3 M sorteado** | **518.635** | **0,6026** | **+0,0238** |
| GTE-large (335M) | — | 0,5788 | — |

### ⚠️ G1.2: PASSOU — e por que este "+0,024" não é o "+0,003" de agosto

| | ΦEmb 3M (23M) | GTE-large (335M) | |
|---|---|---|---|
| nDCG@10 | **0,6026** | 0,5788 | **+0,024 nós** |
| recall@10 | **0,8130** | 0,7640 | **+0,049 nós** |
| MRR | **0,5439** | 0,5293 | **+0,015 nós** |
| recall@1 | 0,4090 | 0,4140 | −0,005 — **empate** (pareado 170×180, p=0,63) |

Em agosto o G1.2 "passou" por **+0,003** num protocolo cujo teto era **0,7562**, com
62% dos itens carregando desempate arbitrário. Três diferenças fazem esta afirmação
ser de outra natureza:

1. o teto do protocolo é **1,0000**, medido e gravado no artefato;
2. a margem é **oito vezes** maior, e vem acompanhada de vitória em recall@10 e MRR;
3. o pareado em recall@1 é reportado, e diz **empate** — não uma vitória que não
   existe.

O honesto é: **um modelo de 23M supera um genérico de 335M em recuperação de
Física, exceto em recall@1, onde empata.** A cláusula de tamanho fecha com folga
(1/14,8 contra o 1/10 exigido).

### O que custou o `head`, medido

**0,076 de nDCG@10** entre o pior e o melhor run — e os dois usaram a mesma base,
o mesmo lote, o mesmo código. Decompondo:

    head -> sorteio, a 400 mil pares          +0,020
    400 mil -> 1,5 M sorteados                +0,032
    1,5 M -> 3 M sorteados                    +0,025

O primeiro termo era **de graça**: mesmo custo de GPU. Os outros dois foram
comprados com tempo de T4 que a cota já dava.

### A curva ainda não platôou, no terceiro run seguido

| run | passos | pico | fração do treino |
|---|---|---|---|
| 400 mil | 3.125 | 0,6147 | 89,6% |
| 1,5 M | 11.718 | 0,6567 | 97,3% |
| 3 M | 23.437 | **0,6804** | **99,8%** |

(avaliação interna de 1.000 candidatos.) O pico do run de 3 M está no penúltimo
ponto medido. O corpus tem **6.564.111** pares e **667.304** documentos citados:
usamos 3 M e 518.635. Ainda há **2,2×** de pares disponíveis, e um run do corpus
inteiro levaria ~9,3 h — no limite de uma sessão do Kaggle.

Curvas versionadas em `data/processed/avaliacao/t1a_curvas_de_treino.json`.

### A decisão que isto abre, e que não é minha

O recuperador do sistema é o `phiemb-minilm-melhor` (**0,5246**). O run de 3 M dá
**0,6026** — **+0,078**. Trocar melhoraria toda a cadeia híbrida, e **invalidaria a
referência do T1b/T1c**: o nDCG 0,1666 do ΦRank foi medido com o recuperador antigo,
e trocar os dois ao mesmo tempo mediria duas coisas. É por isso que o registro do
T1C fixa o antigo de propósito.

## 1,5 M de pares sorteados empatam com o GTE-large (2026-09-07)

Três runs, uma variável cada, no protocolo de teto 1,0 e 2.000 candidatos:

| run | documentos citados | nDCG@10 | margem vs GTE-large |
|---|---|---|---|
| 400 mil `head` | 17.844 | 0,5265 | −0,0523 |
| 400 mil sorteado | 191.198 | 0,5462 | −0,0326 |
| **1,5 M sorteado** | **390.856** | **0,5780** | **−0,0008** |
| GTE-large (335M) | — | 0,5788 | — |

**O `head` custava 0,052 de nDCG@10. Metade disso era amostragem, metade era volume.**

### O que é empate e o que não é

| | ΦEmb 1,5M (23M) | GTE-large (335M) | |
|---|---|---|---|
| nDCG@10 | **0,5780** | 0,5788 | −0,0008 — empate |
| recall@10 | **0,7850** | 0,7640 | **nós** |
| recall@1 | 0,3890 | **0,4140** | eles, pareado p=0,0083 |
| MRR | 0,5216 | **0,5293** | eles |

O modelo **acha o documento certo no top-10 mais vezes** que um genérico 14,8×
maior, e **o põe em primeiro menos vezes**. Isso não é uma vitória nem uma derrota
— é um perfil diferente, e dizer "empatamos" sem essa tabela esconderia metade.

**G1.2 continua NÃO PASSOU**, e é o correto: o critério pede *superar*, e −0,0008
não supera. Mas a ressalva que o ESTADO carregava desde agosto — "o defensável é
paridade a 1/15 do tamanho, não vitória" — passa a ser **verdade medida** em vez de
artefato de um protocolo com teto 0,7562.

⚠️ E vale lembrar como chegamos aqui: em agosto o G1.2 "passou" por +0,003 num
protocolo quebrado. Agora estamos a −0,0008 num protocolo de teto 1,0. Números
parecidos, significados opostos — a diferença é que este vem com o teto medido e o
pareado reportado.

### A curva ainda não platôou

    passo       0    1600    3200    4800    6400    8800   11400
    nDCG@10  0,5515  0,6073  0,6209  0,6292  0,6392  0,6515  0,6567

(na avaliação interna de 1.000 candidatos.) O pico está no passo **11.400 de
11.718** — 97% do caminho. 196,2 pares/s, 2h08 de T4 gratuita.

O corpus tem **6.564.111** pares e **667.304** documentos citados distintos: ainda
há 4,4× mais dado disponível. Um run de 3 M levaria ~4,3 h e um do corpus inteiro
~9,3 h — os dois cabem na cota de 30 h/semana, o segundo no limite de uma sessão.
É o caminho para superar em vez de empatar, e ele está medido, não suposto.

## O retreino da T1a com pares sorteados: +0,020 de graça (2026-09-07)

A ablação mais limpa do projeto até aqui — mesma base, mesmos 400.000 pares, mesmo
lote 128, mesmos 3.125 passos, mesma semente. **A única variável é a amostragem.**

| | documentos citados distintos | nDCG@10 | recall@1 |
|---|---|---|---|
| ΦEmb-T4 (`head`) | 17.844 | 0,5265 | 0,3560 |
| ΦEmb-T4 (sorteado) | **191.198** | **0,5462** | **0,3725** |
| | 10,7× | **+0,0197** | +0,0165 |

Pareado em recall@1 entre os dois: **104 a 71, p=0,0153**. O ganho é real, não ruído.
E custou **zero**: 3.125 passos a 211,8 pares/s, 36 min de T4 gratuita — o mesmo
que o run original.

O modelo novo é o melhor dos nossos por nDCG@10 (0,5462 contra 0,5442 do
ΦEmb/MiniLM 1,5M), mas os dois **empatam em recall@1** (163 discordantes,
p=0,4336). A troca de liderança vale na métrica do portão, não nas duas.

### O que ele NÃO resolve

O G1.2 continua vermelho. A distância até o GTE-large sai de −0,0346 para
**−0,0326**: fechou 0,002 de 0,035. O ganho de +0,020 foi sobre a versão `head` do
MESMO run, e o campeão anterior já era o de 1,5 M de pares — então a diversidade
melhorou o modelo sem mover o portão.

Contra o GTE-large o pareado segue decisivo: **132 a 215, p=0,0000**.

### E a curva não platôou

    passo      0    200    800   1600   2200   2800   3000
    nDCG@10 0,5515 0,5837 0,5930 0,6064 0,6146 0,6147 0,6146

Subindo até o fim, com o pico no passo 2.800 de 3.125. Isso fecha a suspeita
levantada ontem: o "platô medido" que interrompeu o run de 1,5 M em 38% era
artefato do prefixo — 256 mil pares novos tirados de 67 mil documentos se parecem
com platô de dados porque **eram** exaustão de documentos, não de dados.

**O experimento seguinte está definido:** 1,5 M de pares sorteados dão **390.966**
documentos distintos contra os 67.232 do prefixo. A 211,8 pares/s são ~2 h de T4 —
cabe numa sessão de 9 h e na cota de 30 h/semana. É o candidato real a fechar os
0,033.

## O G1.2 passava por +0,003 e, medido direito, perde por −0,035 (2026-09-06)

Remedição no pool corrigido, 2.000 candidatos, **teto do protocolo 1,0000**:

| modelo | params | nDCG@10 | recall@1 | recall@10 | MRR |
|---|---|---|---|---|---|
| GTE-large (genérico) | 335M | **0,5788** | 0,4140 | 0,7640 | 0,5293 |
| **ΦEmb/MiniLM 1,5M** (nosso) | 23M | **0,5442** | 0,3670 | 0,7425 | 0,4917 |
| ΦEmb/MiniLM+GC (nosso) | 23M | 0,5272 | 0,3620 | 0,7145 | 0,4790 |
| ΦEmb-T4 (nosso, era o campeão) | 23M | 0,5265 | 0,3560 | 0,7185 | 0,4773 |
| ΦEmb/MiniLM (nosso) | 23M | 0,5246 | 0,3480 | 0,7195 | 0,4738 |
| MiniLM-L6 (genérico) | 23M | 0,4761 | 0,3135 | 0,6640 | 0,4279 |
| ΦEmb/SciBERT (nosso) | 110M | 0,4746 | 0,3070 | 0,6705 | 0,4263 |
| PhysBERT (alvo do G1.1) | 109M | 0,3507 | 0,2220 | 0,4910 | 0,3170 |
| SciBERT (base do ΦEmb) | 110M | 0,2537 | 0,1490 | 0,3825 | 0,2275 |

### O que mudou de veredito

| | protocolo quebrado (teto 0,7562) | pool corrigido (teto 1,0) |
|---|---|---|
| GTE-large | 0,4628 | 0,5788 |
| nosso melhor | 0,4657 | 0,5442 |
| margem | **+0,003 a nosso favor** | **−0,035 contra** |

**G1.1: PASSOU** — +0,193 de nDCG@10 sobre o PhysBERT, contra um limiar de +0,05,
e o pareado em recall@1 dá p=0,0000. Passa com mais folga que antes.

**G1.2: NÃO PASSOU.** A vitória de +0,003 era artefato: com 62% dos itens carregando
desempate arbitrário, uma margem de três milésimos não significava nada. Medido num
pool de teto 1,0 a diferença é **doze vezes maior e no sentido oposto**.

O honesto agora é: **o ΦEmb bate o PhysBERT com folga e não bate o melhor embedder
geral**, a 1/14,8 dos parâmetros dele. Isso é um resultado defensável — só não é o
que o G1.2 pede.

### Duas coisas que a remedição também mostrou

**O campeão mudou.** O `PhiEmb-T4` (0,5265) deixou de ser o melhor dos nossos; o
**ΦEmb/MiniLM 1,5M** (0,5442) é. E ele é justamente o run que foi **interrompido em
38% por "platô medido"** — o que reforça a suspeita de que o platô era artefato do
prefixo: os 256 mil pares extras vinham de 67 mil documentos, e mais dados ainda
estavam ajudando.

**O ΦEmb/SciBERT de 110M (0,4746) empata com o MiniLM-L6 genérico de 23M (0,4761).**
Um modelo nosso, com quase 5× os parâmetros, não supera o genérico pequeno.

## ⚠️ O Portão G1 media contra um teto de 0,7562, e o treino via um prefixo (2026-09-06)

Auditoria da armadilha do `head()` depois da sexta ocorrência. Os conjuntos de
avaliação da T1b e da T1c estão limpos (`val.sample(n=..., seed=...)`), a divisão
treino/validação de `pairs.py` está limpa (embaralha por âncora e confere
vazamento), e o `classifier.py` embaralha antes de cortar. **O G1 não estava.**

### O protocolo tinha teto, e ninguém o havia medido

O G1 avalia recuperação **dentro do lote**: `sim = ancoras @ positivos.T`, e a
resposta certa da linha *i* é a **coluna** *i*. Isso só é tarefa bem posta se cada
coluna for distinta. Duas coisas quebravam isso:

1. **`amostra = val.head(n)`.** A ordem de `pares_validacao.parquet` não é neutra —
   a mediana do comprimento da âncora cai de ~1.180 para ~870 do início ao fim, e o
   primeiro bloco de 2.000 fica no **percentil 94**. Pior: em `head(2000)` havia só
   **1.147 textos positivos distintos**, com **62% das linhas** num positivo
   repetido e um deles aparecendo **28 vezes**. Textos byte-idênticos dão cosseno
   idêntico e o desempate do `argsort` é arbitrário: a diagonal cai num posto
   qualquer entre as 28.
2. **Âncoras repetidas.** Linhas com a mesma consulta compartilham **um** ranking,
   então no máximo uma delas pode ter posto 1.

Teto de um modelo **perfeito**, medido:

| pool | recall@1 | nDCG@10 |
|---|---|---|
| `head(2000)` — o que o G1 usou | **0,5235** | **0,7562** |
| `sample(2000)` | 0,9364 | 0,9761 |
| sorteio + desduplicação | **1,0000** | **1,0000** |

O G1 reportou recall@1 **0,2620** contra um teto de **0,5235**. Metade do que
parecia limitação do modelo era o protocolo.

O empate era **justo** entre modelos — todos sofriam igual —, e é por isso que a
tabela parecia válida. O que ele destruía era a **margem**: o G1.2 se decidiu em
**+0,003** de nDCG@10, e ruído arbitrário em 62% dos itens não deixa +0,003
sobreviver. Pelo mesmo motivo, parte dos discordantes do McNemar era moeda e não
discordância entre modelos.

Consertado com `preparar_pool` em `phifm.training.amostragem` — sorteio com
semente, desduplicação por texto, e uma guarda que **levanta** se o teto não for
1,0. O `avaliar` do treino usa o mesmo pool, porque é ele que elege os checkpoints
`-melhor`. `teto_do_pool` sai junto no artefato versionado: um nDCG@10 de 0,4657
não diz se o modelo é mediano ou se o protocolo é.

### E o treino via 10,7× menos documentos do que podia

Medido em `pares_treino.parquet` (6.564.111 linhas, **667.304** documentos citados
distintos):

| pares | `head` → documentos | sorteio → documentos | |
|---|---|---|---|
| 20.000 | 984 | 17.837 | **18,1×** |
| 400.000 | 17.844 | 191.300 | **10,7×** |
| 1.500.000 | 67.232 | 390.966 | 5,8× |

O run da T1a — 400 mil pares, que produziu o **campeão atual do G1** — treinou com
**17.844** documentos citados distintos onde o sorteio do mesmo tamanho daria
**191.300**. Mesma GPU, mesmo tempo.

`rerank.py` já tinha esse conserto, **com a explicação escrita no código**, e ela
nunca foi aplicada ao caminho do embedding. Agora os quatro pontos de corte
sorteiam, e o corte acontece no plano (`amostrar_do_plano`) porque coletar 6,5 M
pares com 1,0 GB de RAM livre matava o processo antes do primeiro passo.

**Vale suspeitar do platô que interrompeu o run de 1,5 M em 38%.** Ele foi
justificado por "256 mil pares a mais, metade inéditos, e o nDCG@10 oscilou sem
tendência" — mas esses pares saíam de **67 mil** documentos. Exaurir um conjunto
pequeno de documentos se parece exatamente com um platô de dados.

### O que isto NÃO diz

Que o ΦEmb é melhor ou pior do que se pensava. Diz que **não se sabe**: os números
antigos vieram de um protocolo com teto de 0,76 e o modelo veio de um treino com
1/10,7 da diversidade disponível. A remedição no pool corrigido responde a primeira
metade; a segunda exige treinar de novo — 36 min de T4 gratuita para a T1a.

## ⚠️ A amostra que decide a confiança no corpus cobria 0,67% dele (2026-09-05)

A revisão dos 400 documentos do peS2o ia medir a coisa errada. O filtro montava a
amostra de passagem:

    if len(f.amostra) < N_AMOSTRA and f.vistos % 97 == 0:

Isto **para** de amostrar assim que junta 400. Medido nos parquets prontos: os 400
documentos de `_amostra_para_revisao.json` estão todos em `part-00000` e
`part-00001`, e o último ocupa a posição **37.087 de 5.526.331 aceitos — 0,67%**.

E o peS2o filtrado não é uma população só:

| | documentos | dos docs | dos tokens | tamanho |
|---|---|---|---|---|
| resumo (s2ag) | 3.626.168 | 65,6% | **7,9%** | 1.280 B/doc |
| texto pleno (s2orc) | 1.900.163 | 34,4% | **92,1%** | 28.454 B/doc |

Os resumos vêm primeiro (`part-00000` a `part-00180`), então a amostra antiga é
**100% resumo**. Ela mediria a precisão do filtro nos 7,9% do corpus e nada nos
92,1%.

### E isso invertia a justificativa da própria medição

`apurar_revisao.py` dizia, com razão: os 1,5–13,6% de referência foram medidos em
**resumos do arXiv**, e o corpus agora é texto pleno — outra distribuição, a taxa não
transfere de graça. **A amostra que ia testar isso era de resumos.** O experimento
media o braço de controle e chamava de tratamento.

Sexta ocorrência de `head()` sobre dado ordenado neste repositório — depois do peS2o
amostrado no começo, do `val.head(500)` que eram 35 documentos, do `pares_treino`
cortado por posição (49,6% de vazamento), das 8 primeiras de 44 partes do RedPajama e
do `head(200)` sobre uma lista de parquets dentro do próprio conferidor.

### O conserto, nas duas pontas

`scripts/amostrar_para_revisao.py` sorteia uniforme **sem reposição dentro de cada
estrato**, sobre os 277 parquets prontos, e grava `(arquivo, linha)` de cada
documento. 200 + 200, embaralhados — julgar 200 resumos e depois 200 papers
confundiria o estrato com o cansaço de quem julga. Para o texto pleno a folha mostra
o início **e um trecho a 45% do documento**: num paper de 28 mil caracteres o começo é
a capa, e a contaminação aparece no corpo.

Dentro do filtro, `Filtragem.considerar` passa a ser um **reservatório** (algoritmo
R), então uma coleta futura já sai com amostra uniforme. A ressalva que ele não
resolve: com retomada, cada execução vê só os seus arquivos — para medir, o
amostrador de verdade é o script.

### Duas contas que também estavam erradas

A apuração multiplicava a taxa do peS2o pelos **27,75 B do corpus inteiro**,
atribuindo ao RedPajama-arXiv e ao OpenWebMath uma contaminação que ninguém mediu.
Agora traduz nos **14,60 B do peS2o** e diz o que não cobre.

E o número final passa a ser a **combinação ponderada por token** dos dois estratos,
não a taxa da amostra: com estratos de tamanhos iguais e pesos de 8% e 92%, a taxa da
amostra diz mais sobre quantos de cada um foram sorteados do que sobre o corpus. O
intervalo é a soma ponderada dos limites de Wilson — conservadora de propósito, porque
a alternativa normal dá largura zero quando um estrato sai com 0 falsos positivos.

## O corpus do ΦEnc está preparado e conferido (2026-09-05)

**2.001.270.262 tokens** de 143.810 documentos, 9 partes sorteadas de 44, 6,0 GB —
`tokens.u16.bin` + `marcas.u8.bin` em `data/processed/phienc_dados/`. A ~1.046 mil
tok/s, 35 min.

Por que 2 B e não mais: no proxy de 48 M, 1 B por braço custa **3,2–5,3 h** de T4, e
a ablação inteira (controle + tratado) cabe em 6,4–10,7 h — dentro da cota de 30 h
por semana, com cada braço numa sessão de 9 h. Preparar além disso seria preparar
dado que não dá para treinar de graça.

### A conferência, antes de dizer que está pronto

| | preparado | corpus cru |
|---|---|---|
| matemática | **37,8%** | 38,8% |
| display | **25,0%** | 25,2% |
| `fracao_tratada` | **0,903** | — |
| taxa efetiva | **0,3000** | pedida 0,3000 |
| ids | [2, 40958] | vocabulário 40.960 ✓ |

Medido em 300 sequências sorteadas do binário — que é o que o treino lê. E o trecho
decodificado é LaTeX de Física íntegro: `\begin{equation}`, `\label{BASForDC}`,
`\bar{c}_s`, `f^{a_1 a_2 b}`.

### ⚠️ O viés que a conferência pegou, e que eu quase gravei

A primeira preparação usou as **8 PRIMEIRAS** de 44 partes:

    partes 0-7  (usadas)      math 32,4%   display 18,8%
    partes 8-43 (NÃO usadas)  math 42,7%   display 29,5%

As que ficaram de fora tinham **57% mais equação em display**. As partes não são
intercambiáveis, e pegar as primeiras é `head()` no nível de arquivo — **quarta vez**
que este repositório tropeça nisso (o peS2o amostrado no começo e concluído sem
LaTeX; o `val.head(500)` que eram 35 documentos; o `pares_treino` cortado por posição
com 49,6% de vazamento).

O viés ia **contra** a hipótese do DOC-07 §2.3 — menos equação enfraquece o
tratamento —, que é o lado seguro. Um viés que ajuda não deixa de ser viés.

Consertado: partes sorteadas com semente, **a lista** no manifesto (com sorteio,
saber que 9 entraram não permite reconstruir quais), e retomar com outra semente
**levanta**. `--em-ordem` fica para reproduzir a preparação antiga, rotulado.

Depois do conserto o binário passou de 36,7% para **37,8%** de matemática contra os
38,8% do cru, e de 23,9% para **25,0%** de display contra 25,2%.

### ⚠️ E a QUINTA ocorrência da mesma armadilha, dentro do conferidor

A primeira versão da conferência usava `head(200)` sobre uma **lista** de parquets —
que lê as 200 primeiras linhas, **todas do primeiro arquivo**. A comparação entre
grupos virava comparação entre dois arquivos, e devolveu exatamente os 32,4% da
medição enviesada que ela existia para refutar.

Eu quase reportei isso como "o viés persistiu". O número que decide é o do binário
preparado contra o corpus cru, e lá a diferença é de 1,0 ponto em matemática e 0,2 em
display.

### Antes disso: a preparação levava 16 horas

`marcar_equacoes` era **98% do tempo** — O(spans × tokens) em Python puro, 188
milhões de iterações por documento. Vetorizado com `searchsorted`: **33 mil → 1.046
mil tok/s**, com 0 divergências em 1.181.249 tokens contra a implementação lenta.

## O código de pré-treino do ΦEnc existe (2026-09-03)

Quatro peças, separadas de propósito — só o laço importa torch, e as outras três
rodam na suíte rápida:

| peça | o que garante |
|---|---|
| `pretrain/mascaramento.py` | a hipótese do DOC-07 §2.3, com orçamento igual entre os braços |
| `pretrain/dados.py` | o fluxo SEM ESTADO do DOC-08 §7.2, puro em `(semente, passo)` |
| `pretrain/spike.py` | detecção e resposta do §6.1, e o WSD do §4 |
| `pretrain/laco.py` | a costura, o checkpoint e o rollback |
| `models/encoder/` | a config conferida contra o `transformers` |

### O número que prova que o MLM está certo

    perda inicial 10,7343   contra   ln(40.960) = 10,6204

Um MLM não treinado prevê uniforme, então a entropia cruzada inicial **é** `ln(V)`.
Se o mascaramento apagasse os alvos, se o `-100` estivesse no lugar errado, ou se os
alvos viessem da entrada já mascarada, esse número sairia diferente e **nenhum outro
sintoma apareceria**. Está travado num teste.

Junto: taxa efetiva 0,2998 contra os 30% pedidos, retomada no passo certo com o
plano WSD do disco, e o aviso de `fracao_tratada` baixa disparando.

### Três coisas que o desenho do mascaramento decidiu, e a medição de cada uma

**Orçamento igual entre os braços.** `n_alvo = round(taxa × mascaráveis)` vale para
os dois; no tratado escolhe-se uma equação que caiba e o resto é aleatório. Sem isso
a ablação mediria "consciente de equações" **e** "mascara 6× mais".

**Só equações em DISPLAY.** Medido em 120 documentos do RedPajama-arXiv:

    tipo      por doc   tokens: p10  mediana  p90   p99     max
    display      40,6            39       79  214   678  19.587
    inline      260,9             4        7   19    39     103

A mediana de 7 tokens do inline é uma **variável** (`$\rho$`), não uma equação. A
primeira versão tratava qualquer `$…$` e a ablação mediria o nada. 91,7% dos
documentos têm display, e 99,8% delas cabem no orçamento.

**A equação vai inteira para `[MASK]`, sem 80/10/10** — registrado como escolha, não
achado: se o tratamento ganhar, "ganhou porque mascara 100% com `[MASK]`" segue
sendo explicação viva, e a ablação que a mata está escrita na docstring.

### ⚠️ O bug que a sonda pegou, e que teria invalidado a ablação inteira

`DISPLAY` era `re.compile(r"^(?:...)")` usado com `DISPLAY.match(texto, i)`.
`Pattern.match(s, pos)` **já** ancora em `pos`, mas o `^` continua se referindo ao
início REAL da string — então o padrão nunca casava para nenhuma equação que não
começasse no caractere 0.

Medido: **0 de 120** documentos tratados, `recaida_sem_equacao` em 120, contra 91,7%
de documentos com display medidos por outra sonda que fatiava a string antes de
casar. Foi a **discordância entre as duas sondas** que localizou o erro. Sem a
`fracao_tratada` nos contadores, a ablação teria rodado inteira comparando aleatório
com aleatório e reportado empate.

Depois do conserto: `tratada 0,883` com `p_equacao=1,0`.

### E um defeito no agendamento, achado por um teste que falhou

O `lr_wsd` recebia `frac_warmup=0.03` e calculava o warmup como `total × 0,03`. Com
isso, estender o treino de 10 mil para 200 mil passos alongava o warmup de 300 para
6.000 — e a LR do passo 5.000, **que já havia sido dado com o pico**, passaria a
valer 8,3e-4. O agendamento deixava de ser extensível, que é o argumento inteiro do
DOC-08 §4 para preferir WSD a cosseno.

Agora `passos_warmup` é absoluto, derivado uma vez por `plano_wsd()` e guardado no
checkpoint. A tentação era ajustar o número esperado no teste.

### O que NÃO existe, e é o gargalo agora

**Avaliação.** O DOC-05 §11.2 pede recuperação de Física, MLM em texto denso em
equações e uma sonda de estrutura tensorial. Um `phienc.json` com perda baixa **não
é veredito** sobre a hipótese do §2.3 — é só evidência de que o laço funciona.

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

### 2. T1c — o reordenador de base diferente, rodando na cota gratuita

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

## O ΦRank de PhysBERT ENTROU no sistema (2026-09-03)

Pela primeira vez a composição inteira — BM25 + ΦEmb → RRF → ΦRank — bate a fusão
sozinha. O que mudou no repositório:

| onde | de | para |
|---|---|---|
| `models/` | — | `phirank-physbert-melhor/` (438,7 MB, passo 5.500, acerto@1 0,608) |
| `rerank.py` `BASE_PADRAO` | `all-MiniLM-L6-v2` | **`thellert/physbert_cased`** |
| `avaliar_t1b.py --rank` | `phirank-minilm-melhor` | **`phirank-physbert-melhor`** |
| `avaliar_t1b.py --lote-rank` | 32 | **8** — 32 estoura os 8 GB com um modelo de 109 M |

⚠️ O `--lote-rank 32` foi calibrado para o MiniLM e **OOM com o modelo novo**:
`Could not allocate tensor with 150994944 bytes` — 151 MB de UM tensor intermediário
(32 × 384 × 3.072 × 4 bytes). Quem rodasse o avaliador local depois da troca
receberia isso. Numa T4 de 16 GB o lote maior cabe e é mais rápido; o default existe
para funcionar em todo lugar.

`scripts/instalar_phirank.py` grava o manifesto de etapa junto dos pesos, com a
medição que justifica **e** o que ela não diz — que este ΦRank generalize para outro
recuperador, outro benchmark ou outro domínio não foi medido.

`tests/regression/test_phirank_do_sistema.py` tranca as duas linhas de configuração.
São uma linha cada, e uma linha se reverte sem nada quebrar: o sistema voltaria a
compor com um reordenador que a medição diz ser no-op, e a métrica cairia 0,009 sem
nenhum teste vermelho. Os testes leem o FONTE em vez de importar `rerank.py`, que
arrasta torch — um `importorskip` faria o guarda pular justamente no CI.

### ⚠️ Duas coisas que este passo revelou

**1. A proveniência gravada dentro do modelo estava errada.** O `phirank.json` do
modelo de PhysBERT saiu do treino afirmando *"inicializado do MiniLM, a mesma base do
ΦEmb campeão"* — a string era fixa em `rerank.py` e valia para qualquer base.
`scripts/train_rerank.py` já tinha sido parametrizado; **esta cópia dentro do
treinador escapou**, e é a que vai para dentro do diretório do modelo.

Consertado para `f"inicializado de {self.cfg.base}"`. O arquivo do modelo instalado
**não foi reescrito**: é o que a execução produziu, e reescrever proveniência depois
do fato é pior que registrar a divergência. O manifesto de etapa aponta o erro pelo
nome.

**2. A armadilha do teste-que-reprova-o-comentário chegou à quarta ocorrência.**
`test_kaggle_t1a` com "InfoNCE", `test_kaggle_t1c` com `check=False` e com
`phifm_src.zip`, e agora este com a string de proveniência antiga — quatro testes que
buscavam um padrão proibido e o achavam no comentário que explica por que ele é
proibido. A correção errada, nas quatro, seria apagar o comentário.

`tests/conftest.py` agora expõe `so_codigo()`, que passa o fonte por `ast.unparse` e
devolve o código sem nenhum `#`. Resolve a classe inteira, e a tabela das quatro
ocorrências está na docstring dele.

## T1c — o que o reordenador precisa é DOMÍNIO, não diversidade (2026-09-03)

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
de reordenador** das três.

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

No pareado k=10: a fusão ganha 97, o reordenador ganha **140**, 237 discordantes. Em
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

    hash raiz  927ae48695f6c9c66f7499e634a9b9c5c4d0e5e699ac63c5196cd94fd0c53469
    34 etapas · 1258 arquivos · 40,13 GB

### ⚠️ O hash anterior cobria 55% do corpus (achado em 2026-09-11)

O raiz de 2026-09-08 era `bbd73a7a…` sobre **33 etapas · 976 arquivos · 21,79 GB**,
e o ESTADO.md dizia "o corpus por um hash". O peS2o — filtrado em 2026-08-26,
**18,34 GB e 14,60 B tokens dos 27,75 B** — nunca foi acrescentado à tabela
`ETAPAS`, então o construtor não sabia que ele existia.

Nada acusou, e o motivo é o que importa: **o construtor só percorre a lista
declarada**. Uma etapa que não está lá não existe para ele, e o relatório de
verificação sai ✅ sobre o que ele conhece. Um verificador que passa não prova que
o que ficou de fora está íntegro — prova que ele não olhou.

A lista explícita é a decisão certa (um construtor que varre o disco atestaria o
que *está* lá, não o que *deveria*), e o preço dela é este: acrescentar uma fonte
sem acrescentar a etapa é silencioso. A guarda nova fecha esse flanco sem abrir
mão da lista — `test_manifesto_g1_5.py` exige que toda fonte de
`filtrar_hf.FONTES` tenha entrada em `ETAPAS`, **código contra código**, sem
depender de `data/processed/` existir. Teria pego isto em 2026-08-26.

| etapa | GB | arquivos | registros |
|---|---|---|---|
| spine | 0,76 | 1 | 1.595.422 |
| isphysics_clf | 0,02 | 2 | — |
| pares_citacao | 2,68 | 2 | 6.697.651 |
| redpajama_fisica | 12,56 | 46 | 835.379 |
| openwebmath_fisica | 3,57 | 47 | 860.521 |
| **pes2o_fisica** | **18,34** | **282** | **5.526.331** |

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
A tabela mestra cabe folgado em disco comum, como o documento afirmava — e o número
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
| Tabela mestra ocupa ~700 MB, não 150 GB | Cabe em disco comum | DOC-02 §3.1 |
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

## Tabela mestra construída — 2026-08-07

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
| Tabela mestra consolidada (150 MB) | ✅ | ✅ | ❌ |
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
