# Estado do projeto — 2026-08-17

Ponto de retomada para migração de máquina. Instalação em [SETUP.md](SETUP.md).

## Onde estamos

**Corpus de projeto:** completo. 19 documentos + 1 ADR, cobrindo os 20 pipelines.

| Sprint | Estado | Observação |
|---|---|---|
| **S1** · espinha de metadados | 🟢 completo | 1,59 M arXiv + 4,61 M obras; junção de **99,1%** |
| **S2** · classificador de Física | 🟢 completo | subárea + `is_physics`; acurácia **0,954** com os 4 domínios, FP 2,4–3,7% em cada |
| **S3** · fatias do HuggingFace | 🟢 13,15 B tokens | RedPajama 10,54 B + OpenWebMath 2,62 B, custo zero; peS2o não iniciado. S3b: o RedPajama **degrada 16,6%** |
| **ΦEmb** | 🟡 G1.1 ✅ / G1.2 ❌ | perde do GTE-large por 0,005. **As duas rotas baratas para fechar estão descartadas por medição** |
| **G1.5** · corpus por um hash | 🟡 metade fechada | 21,79 GB verificáveis byte a byte por **um** hash; refazer do zero depende de uma fonte mutável, nomeada |
| Barramento de verificação | 🟢 5 de 6 | falta só `sandbox` — exige gVisor/Firecracker |

Suíte: **355 testes** (5 saltados, os de torch), `PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/ -q`.
Os que dependem de torch rodam na venv de treino:
`.venv-treino/Scripts/python.exe -m pytest tests/regression/test_g1_criterios.py tests/regression/test_comparacao_pareada.py tests/regression/test_melhor_checkpoint.py tests/regression/test_gradcache.py tests/regression/test_estado_progresso.py -q`

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
