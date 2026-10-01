# Revisão do rascunho v0.5 — todos os achados

Apêndice de [`revisao-v0.5.md`](revisao-v0.5.md). Gerado a partir das saídas dos doze revisores. As linhas referem-se à **v0.5** (commit `cad11c8`). O estado de cada achado na v0.6 é: **aplicado** (o trecho citado foi alterado, ou a correção entrou em outro trecho), **em parte**, ou **pendente** (o texto da v0.6 não mudou; a decisão ou a medição é do autor).


## Resumo, Introdução e Conclusão

### L13 · maior · afirma mais que a evidência · aplicado

> trocar a base por um modelo geral de 109 M produz com 400 mil pares o que o volume produziu com 6 milhões (empate pareado, p = 0,908)

**Problema.** O Resumo declara empate a 400 mil pares com base só no recall@1. Na métrica primária do trabalho (nDCG@10), o GTE-base@400k fica 0,013 abaixo, e a medição posterior (T1g) mostrou que o empate em nDCG@10 só aparece a 1 M de pares. O Resumo diz 400 mil; o corpo e a Conclusão dizem 1 M.

**Evidência.** data/processed/avaliacao/t1f_base_n2000.json: GTE-base@400k tem nDCG@10 0,6094, recall@10 0,8010 e MRR abaixo do ΦEmb@6M, que tem 0,6223 e 0,8305. O único teste pareado é o McNemar em recall@1: 150 × 147, p = 0,9076. Não há bootstrap de nDCG@10 para esse par. Em t1g_comparacao.json, o GTE-base@1M fica a −0,00119 [−0,011; +0,0084] do sistema: o intervalo tem meia-largura de cerca de 0,0097, o que dá um erro padrão de cerca de 0,005. Com esse erro, uma diferença de −0,0129 teria intervalo aproximado de [−0,023; −0,003], que exclui zero. Isto é uma estimativa minha, não uma medição. O próprio corpo (linha 304) diz que 'com um sexto do volume [1 M], a base alcança o recuperador instalado', e a Conclusão (linha 643) repete 'um sexto dos pares'. O t1f_treino.json registra ainda: 'Empate aqui prova que a 400 mil não dá para ver'.

**Correção proposta.** Substituir por: "trocar a base por um modelo geral de 109 M produz, com 1 milhão de pares, o que o volume produziu com 6 milhões (nDCG@10 de 0,6211 contra 0,6223, diferença de −0,0012 [−0,011; +0,008])". A mesma correção vale para a linha 300 do corpo ('quinze vezes mais pares'), ou então rodar o bootstrap pareado de nDCG@10 do GTE-base@400k contra o ΦEmb@6M e reportá-lo.

### L13 · maior · falta controle ou limitação · aplicado

> supera o BM25 justamente no estrato que exige variação notacional (recall@1 de 0,860 contra 0,712)

**Problema.** O Resumo destaca a métrica não registrada que tem a maior margem (15 pontos) e omite a primária pré-registrada (2,5 pontos). A conclusão qualitativa se mantém, mas isso contraria a disciplina de registro prévio que o próprio artigo defende (Seções 3.6 e 5.3).

**Evidência.** A docstring de scripts/avaliar_pb_formula.py ('A REGRA, escrita antes') diz: 'PRIMÁRIA: recall@10 por item no conjunto primário'. Em pb_formula_resultado.json, a diferença em recall@10 contra o BM25 é +0,025 [+0,0115; +0,039]. O recall@1 (0,8595 contra 0,7115) não tem teste nem intervalo no artefato. A legenda da Tabela 23 também põe a diferença em recall@10 como a comparação.

**Correção proposta.** "...supera o BM25 justamente no estrato que exige variação notacional (recall@10, medida primária registrada antes: +0,025 [+0,012; +0,039]; recall@1 de 0,860 contra 0,712)"

### L635 · maior · inconsistência interna · aplicado

> a evidência de que a base do recuperador vale, a 400 mil pares, quinze vezes o volume de ajuste

**Problema.** A Conclusão afirma duas razões diferentes para o valor da base: 15× a 400 mil pares e 6× a 1 M. Só a segunda tem suporte na métrica primária.

**Evidência.** A linha 643 da mesma Conclusão diz: 'A base geral mais forte, ajustada com um sexto dos pares, empata com o recuperador atual'. Em t1g_comparacao.json, o empate em nDCG@10 vem a 1 M de pares (−0,00119 [−0,011; +0,0084]). A 400 mil, em t1f_base_n2000.json, o nDCG@10 é 0,6094 contra 0,6223 e só o recall@1 empata (p = 0,908).

**Correção proposta.** "a evidência de que a base do recuperador vale cerca de seis vezes o volume de ajuste: com 1 milhão de pares, o GTE-base empata em nDCG@10 com o bi-encoder de 6 milhões"

### L639 · maior · desatualizado após a v0.5 · aplicado

> Sob este orçamento, a decisão que mais pesa num encoder de recuperação de Física é a escolha da base geral

**Problema.** A frase generaliza além do que foi medido. Ela vale na comparação entre pré-treinamento e escolha de base, mas o sistema final foi decidido pelo volume de pares (MiniLM@6M), e a medição posterior mostra que bases melhores a 1 M não alcançam esse volume. Falta ao artigo o fechamento da busca por encoder.

**Evidência.** O T1h e o T1i vieram depois da v0.5 e não aparecem no artigo (grep por gte-small, bge-small e T1h/T1i: nenhuma ocorrência). Em t1h_comparacao.json, o gte-small e o bge-small superam o MiniLM-L6 a 200 mil pares por +0,042 e +0,040 (IC 98,75%, a 1,76× do custo). Em t1i_comparacao.json, a 1 M de pares eles ficam ABAIXO do sistema: 0,6014 e 0,6012 contra 0,6223, diferença −0,021 com IC 97,5% [−0,032; −0,010], leitura 'o volume ainda manda a 1 M'. O commit 62e42fc encerrou a busca por encoder ('nenhuma supera o MiniLM-L6@6M a um custo aceitável'), e o ADR-0003 registra que a condição de reabertura foi testada. Na Tabela 9, o volume de 400 mil para 6 M rende +0,0761 (0,5462 → 0,6223), mais que o efeito de base GTE−ModernBERT (+0,0694). O t1i_comparacao.json mede o custo do GTE-base em 4,16× (18,34 s contra 4,41 s), contra os 4,4 citados. O cabeçalho continua 'Versão 0.5 — 23 de setembro', embora o commit f04ac1f (T1g) tenha alterado o texto depois da v0.5.

**Correção proposta.** Trocar por: "Sob este orçamento, escolher a base geral pesa mais que qualquer pré-treinamento no domínio, e custa uma fração dele; o volume de pares de ajuste, contudo, pesa tanto quanto a base: bases pequenas que superam o MiniLM-L6 a 200 mil pares ficam abaixo dele a 1 milhão contra 6 milhões (−0,021 [−0,032; −0,010])." Acrescentar à Seção 4.6 um parágrafo com o T1h/T1i, dizer na linha 643 que a busca por encoder foi encerrada com seis bases e dois pré-treinamentos, e atualizar a versão e a data do cabeçalho.

### L13 · menor · afirma mais que a evidência · aplicado

> Entre três cross-encoders idênticos exceto pelo corpus de pré-treinamento da base

**Problema.** Só dois dos três braços são pareados em tamanho e arquitetura; o terceiro difere em tamanho.

**Evidência.** Tabela 14 (linhas 359–361): o MiniLM-L6 tem 23 M de parâmetros, contra 109 M do GTE-base e do PhysBERT. t1c_resultado.json registra só 'Só a base muda' nos hiperparâmetros. O corpo, na linha 363, restringe o controle a 'GTE-base e PhysBERT ... diferindo exclusivamente no corpus'.

**Correção proposta.** "Entre três cross-encoders que diferem apenas na base — dois deles de 109 M, com a mesma arquitetura, diferindo no corpus de pré-treinamento —, apenas o pré-treinado em Física melhora a fusão de recuperadores (p = 0,0062)"

### L13 · menor · clareza · aplicado

> cerca de sete vezes o que o pré-treinamento continuado acrescentou

**Problema.** O leitor do Resumo toma +0,0076 como referência do 'sete vezes' e obtém uma razão errada. O denominador correto (+0,0103 sobre a base de partida) não aparece no Resumo.

**Evidência.** Em t2eq_cpt_emb_comparacao.json, o ganho sobre a base é de +0,01027 (contra_a_barra) e o de GTE−ModernBERT é de 0,06935; 0,06935 / 0,01027 = 6,75. O único número de pré-treinamento continuado que o Resumo dá, porém, é +0,0076 (tratado − controle), e 0,06935 / 0,00757 = 9,2.

**Correção proposta.** "...supera a primeira em 0,069 — cerca de sete vezes os +0,010 que o pré-treinamento continuado acrescentou à base de partida —"

### L13 · menor · afirma mais que a evidência · aplicado

> com a medida primária de modelagem nula

**Problema.** 'Nula' sugere evidência de ausência de efeito; a regra registrada lê o resultado como não decidido.

**Evidência.** t2eq_cpt_ablacao.json: diferença das diferenças de −0,00039 [−0,00161; +0,00087], desfecho 'NÃO DECIDIDO', com a leitura "NÃO é o negativo ... é 'a 0,4 B não dá para ver'". O corpo, na linha 480, diz 'o desfecho primário é não decidido'.

**Correção proposta.** "...com a medida primária de modelagem não decidida (−0,0004 [−0,0016; +0,0009])"

### L13 · menor · clareza · aplicado

> um corpus de Física de 27,75 bilhões de tokens, atestado por uma cadeia de hashes sobre 52,40 GB

**Problema.** O Resumo sugere que os 52,40 GB são o corpus de Física de 27,75 B. Na verdade a cadeia inclui a fatia de matemática e computação, que o artigo mantém separada, e os 27,75 B são uma estimativa por caracteres.

**Evidência.** README.md, linhas 139–154: 'O corpus — 38,96 B tokens, atestados por um hash ... 35 etapas · 1295 arquivos · 52,40 GB', incluindo 'RedPajama-arXiv math+cs 11,20 B'. ESTADO.md, linha 23: '52,40 GB ... (com a fatia math+cs de 2026-09-12 dentro)'. A cadeia também cobre o spine, os pares e o classificador (ESTADO.md, linhas 3492–3499). A Tabela 2 chama os tokens de 'estimados' (cerca de 4 caracteres por token, ESTADO.md linha 2868), e a linha 167 reconhece 0,8% de duplicação na fatia RedPajama.

**Correção proposta.** "...e um corpus de Física de cerca de 27,75 bilhões de tokens (estimados), atestado, junto com o índice, os pares de citação e a fatia separada de matemática e ciência da computação, por uma cadeia de hashes sobre 52,40 GB"

### L13 · menor · número não confere · em parte

> um índice de metadados de 1.595.422 registros do arXiv

**Problema.** Há dois snapshots. O artigo usa 1.595.422 sem dizer qual é, e esse número não se reproduz a partir de nenhum artefato local. A cópia local tem 1.595.065 registros e 0,999 GB, portanto não é o arquivo de 0,76 GB atestado na cadeia. As percentagens (99,1%, 46,4%, 14,8%) são as mesmas nos dois.

**Evidência.** Localmente: data/raw/arxiv_metadata/_manifest.json tem actual_count 1.595.105; data/raw/harvest_arxiv.log termina com 'concluído: 1,595,105 registros'; os 77 shards somam 1.595.105 linhas e 1.595.065 arxiv_id únicos. data/S1_COMPLETO.json dá spine unique_records 1.595.065 e train_open 235.628. data/processed/spine.parquet tem 1.595.065 linhas, peer_reviewed 740.702 e train_open 235.628, em 0,999 GB. DOC-00, DOC-02, DOC-19 e ADR-0001 usam 1.595.065. Em sentido contrário: ESTADO.md (seção de 2026-08-07 e tabela da cadeia, 'spine | 0,76 | 1 | 1.595.422') e README.md usam 1.595.422, e a Tabela 1 traz os derivados desse snapshot (740.823; 235.795).

**Correção proposta.** Declarar o snapshot na Seção 3.1 ou na legenda da Tabela 1: "contagens do índice atestado pela raiz 3113f0fe… (1.595.422 registros); uma construção anterior do mesmo dia tinha 1.595.065". Se o snapshot atestado não puder ser confirmado, adotar 1.595.065, com 740.702 revisados por pares e 235.628 redistribuíveis, e manter 99,1%.

### L25 · menor · afirma mais que a evidência · aplicado

> a conclusão a que chega é mais forte que a da literatura citada

**Problema.** O resultado vale num regime diferente do da literatura e não o contradiz nem o supera; 'mais forte' extrapola a escala medida.

**Evidência.** A literatura citada (Minerva, Llemma, DeepSeekMath) é de decodificadores com bilhões de parâmetros. A evidência do artigo vem de encoders de até 150 M, com uma semente por braço, 0,4 B tokens de pré-treinamento continuado e 200 mil pares (Seções 4.9.3 e 6, linha 613).

**Correção proposta.** "...e a conclusão a que chega, restrita a encoders de até 150 M e a uma semente por braço, vai além da premissa: sob este orçamento, nem o treinamento a partir do zero nem o pré-treinamento continuado pagam o que a simples escolha da base geral paga"

### L32 · menor · clareza · aplicado

> o diagnóstico intuitivo para o mesmo fim satura no corpus íntegro

**Problema.** O indicador não satura, porque 81,3% não é o teto; ele dispara mais no corpus íntegro do que no degradado.

**Evidência.** Seção 4.3, linha 202: o operador órfão 'acusa 81,3% dos documentos da fatia construída a partir do fonte LaTeX, contra 3,0% da referência' e é descrito como de 'comportamento invertido'.

**Correção proposta.** "...e a demonstração de que o diagnóstico intuitivo para o mesmo fim se inverte, disparando mais no corpus íntegro que na referência (81,3% contra 3,0%)"

### L33 · menor · clareza · aplicado

> medida em bits por byte por três instrumentos de viés conhecido, as rejeita

**Problema.** A frase sugere que os três instrumentos rejeitam as regras; um deles as favorece. O Resumo está correto ('apenas o terceiro ... pôde decidir').

**Evidência.** Tabela 17 e t2a_bits_por_byte_AxE*.json: o instrumento 1 dá +0,078 contra as regras, o 2 dá −0,051 a favor delas e o 3 dá +0,047 contra. Só o 3 decide (linha 128: 'Os instrumentos 1 e 2 cercam o valor; o 3 decide').

**Correção proposta.** "...e a modelagem de linguagem, medida em bits por byte pelo instrumento que dois outros, de vieses opostos, cercam, as rejeita"

### L35 · menor · afirma mais que a evidência · aplicado

> a demonstração de que o ganho do estágio desaparece quando o recuperador melhora

**Problema.** O que se mediu foi a ausência de ganho estabelecido, com estimativa pontual positiva e poder insuficiente, e não o desaparecimento do ganho. A mesma frase aparece na linha 635.

**Evidência.** Corpo, linha 378: o ganho marginal caiu de +0,0091 (p = 0,0081) para +0,0045 (p = 0,086). Linha 389: bootstrap @100 de +0,0091 [−0,0010; +0,0191]. Linha 391: 'Estabelecer o ganho nominal de 0,0091, se ele existir, exigiria cerca de 6.640 consultas'. Depois da v0.5, em data/processed/avaliacao/assistente_busca_dev.json (exploratório, outra tarefa), reordenar os 50 primeiros com o ΦRank-PhysBERT eleva o recall de fonte em +0,093 [+0,047; +0,140] no estrato primário.

**Correção proposta.** "...e a demonstração de que, com o recuperador melhorado, o ganho do estágio deixa de ser detectável na recuperação de documentos citados (cinco medições pareadas; estimativa pontual de +0,009, com poder insuficiente)". Na linha 635: "acompanhada de que esse acréscimo deixa de ser detectável quando o recuperador melhora".

### L635 · menor · erro estatístico · aplicado

> que supera um modelo geral de 335 M em nDCG@10, recall@10 e MRR sob um protocolo de teto verificado

**Problema.** 'Supera' em três métricas se apoia só em estimativas pontuais. As diferenças provavelmente são significativas, mas o artefato não mostra isso.

**Evidência.** data/processed/avaliacao/g1_t1a6m.log e g1_resultado.json: o único teste pareado contra o GTE-large é o McNemar em recall@1 (188 × 153, p = 0,0654). Não há bootstrap nem teste para nDCG@10 (+0,0435), recall@10 (+0,0665) ou MRR (+0,0343). A Seção 3.6 diz que diferenças de nDCG@10 usam bootstrap pareado.

**Correção proposta.** Rodar e reportar o bootstrap pareado por consulta de nDCG@10 (e o McNemar em recall@10) do ΦEmb@6M contra o GTE-large. Até lá: "que fica à frente de um modelo geral de 335 M em nDCG@10 (+0,044), recall@10 e MRR nas estimativas pontuais, sob um protocolo de teto verificado".

### L635 · menor · número não confere · aplicado

> um índice de metadados de 1,59 milhão de registros

**Problema.** O valor foi truncado, não arredondado.

**Evidência.** 1.595.422 / 10⁶ = 1,595, que com duas casas arredonda para 1,60 (1.595.065 também dá 1,60).

**Correção proposta.** "um índice de metadados de 1,6 milhão de registros"

### L639 · menor · clareza · aplicado

> a supera por um valor sete vezes maior

**Problema.** Pela leitura natural, a razão sai errada. A contagem 'dois resultados' também não bate com o que vem em seguida.

**Evidência.** O antecedente imediato é o 'ganho de recuperação, onze vezes menor', isto é, +0,0076. Em t2eq_cpt_emb_comparacao.json, 0,0694 / 0,0076 = 9,2; o fator 7 só vale contra +0,0103 (tratado − base sem pré-treinamento). Além disso, 'E dois resultados delimitam os três' é seguido de três resultados.

**Correção proposta.** "...repete o ganho de recuperação, onze vezes menor, e acrescenta +0,010 sobre a base de partida; outra base geral, também sem pré-treinamento no domínio, a supera por 0,069, cerca de sete vezes esse acréscimo." Revisar também 'dois resultados delimitam os três'.

### L641 · menor · clareza · aplicado

> e a vantagem cresce com a variação

**Problema.** A afirmação vale só em recall@1, como o corpo diz na linha 510; a Conclusão omite a métrica.

**Evidência.** Em recall@1: identica 0,999 − 0,9465 = +0,053, superficial 0,941 − 0,8765 = +0,065, notacional 0,8595 − 0,7115 = +0,148 (pb_formula_identica.json, pb_formula_superficial.json, pb_formula_resultado.json). Na métrica primária recall@10: +0,002 (1,000 contra 0,998), −0,0065 (0,9775 contra 0,984) e +0,025. Não é monotônico, e o denso fica abaixo do BM25 no estrato superficial.

**Correção proposta.** "...e a vantagem em recall@1 cresce com a variação notacional (+0,053, +0,065 e +0,148)"

### Conferido e correto

- 0,954 de acurácia do classificador: isphysics_com_math.log, 'accuracy 0.954' sobre 73.531 exemplos; as taxas de falso positivo por domínio omitido (35,4% em matemática, que cai para 10,0% com limiar 0,999) batem com o corpo
- 99,1%: 1.581.098 / 1.595.422 = 99,10%, e 1.581.098 / 1.595.065 = 99,12%; bate com o README. O valor absoluto vem do README e não foi verificável localmente
- 27,75 B: ESTADO.md, linha 3273, '13,15 B para 27,75 B'; 13,15 + 14,60 = 27,75. Os componentes da Tabela 2 somam 27,76 só por arredondamento
- 0,6223 contra 0,5788: g1_resultado.json, ndcg_10 de 0,6223 (PhiEmb 6M sorteado) e 0,5788 (GTE-large); teto do protocolo 1,0
- 6 M de pares e 650.162 documentos: t1a6m_treino_sorteado.log, 'pares: 6,000,000 ... 650,162 documentos citados distintos'
- p = 0,908: t1f_base_n2000.json, McNemar em recall@1 do sistema contra o GTE-base@400k, 150 × 147, p = 0,9076
- p = 0,0062: t1c_resultado.json, variante PhysBERT, p_k10 = 0,00625, 237 discordantes, delta +0,009
- 0,047 bit/byte: t2a_bits_por_byte_AxE_l2r.json, diferença média de 0,046613 [0,0432; 0,0501], vencedor E
- −0,0040: t2eq_ablacao.json, diferença das diferenças −0,004 [−0,00581; −0,00216]
- 0,084 [0,071; 0,097]: t2eq_emb_comparacao.json, 0,08397 [0,07077; 0,0972]
- 0,5270 contra 0,4712: t2eq_emb_comparacao.json e ADR-0003 §8; ModernBERT − tratado = +0,05579 [0,04264; 0,06854]
- +0,0076 [+0,0013; +0,0139]: t2eq_cpt_emb_comparacao.json, 0,00757 [0,00128; 0,01388]
- 'onze vezes': 0,08397 / 0,00757 = 11,1
- 0,069: t2eq_cpt_emb_comparacao.json, GTE − ModernBERT = 0,06935 [0,05797; 0,08118]
- 'sete vezes' em relação a +0,0103: 0,06935 / 0,01027 = 6,75 (t2eq_cpt_emb_comparacao.json, contra_a_barra)
- 0,860 contra 0,712: pb_formula_resultado.json, recall@1 de 0,8595 (sistema) contra 0,7115 (BM25); com arredondamento a três casas dá 0,860 e 0,712; estrato notacional de 24.508 itens, 2.000 sorteados, teto 1,0
- 0,6 bilhão de tokens a 48 M: kaggle/t2a_tokenizer.py, TOKENS = 600_000_000
- 0,4 bilhão de tokens no pré-treinamento continuado: kaggle/t2eq_cpt.py, TOKENS = 400_000_000 (400 M / (1.024 × 64) = 6.103 passos, como no corpo)
- 15–30 B e 39–73 B: DOC-02, linha 292, e DOC-04, linhas 33–39 e 259 (estimativas de planejamento, rotuladas como tal no texto); 160 B para 8 B: DOC-00, linha 130; 'cinco a dez vezes' = 160/30 = 5,3 e 160/15 = 10,7
- Empate do GTE-base@1M (linha 643): t1g_comparacao.json, −0,00119 [−0,011; +0,0084], 148 × 144, p = 0,861; custo de 954 s contra 217 s = 4,4 (T1f)
- Coerência com o ADR-0003 aceito: o ΦEnc não é treinado, o ΦEmb continua o MiniLM@6M, e o §2.3 fica como resultado (+0,084 a 48 M, +0,0076 a 150 M). O Resumo e a Conclusão concordam com a decisão; a revisão do PB-Formula (DOC-11 §6.3, 'medida em que o sistema vai bem, não lacuna') já está refletida na leitura 'contra a premissa'
- Contribuições 1, 2, 4, 6, 7 e 8: as seções citadas existem e contêm o que é alegado (3.1–3.3, 3.7 e 4.1; 4.3; 4.5–4.6; 4.9.2–4.9.3; 4.10; 5.2 com as oito linhas da Tabela 24, 5.3 com três instrumentos e 5.4 com o detector). As ressalvas de redação estão nos achados
- 'nenhuma de cinco medições pareadas' é coerente com a linha 391 do corpo; os p-valores de t1b2_resultado.json e t1e_resultado.json (0,086, 0,139, 0,86 etc.) aparecem no corpo
- Intervalo da Conclusão para o pré-treinamento continuado e a assimetria de execução a favor do tratado: coerentes com t2eq_cpt_ablacao.json ('controle 2 spikes, tratado 0')

### Não verificável nesta máquina

- 1.595.422 registros: não se reproduz com nenhum artefato local (a coleta local tem 1.595.105 linhas e 1.595.065 ids únicos). Seria preciso o manifesto da etapa spine sob a raiz 3113f0fe… e o arquivo de 0,76 GB atestado, que estão na máquina Windows ou no Drive
- 4.613.751 obras do OpenAlex e 1.581.098 casados: o S1_COMPLETO.json local só tem 1,1 M de obras (coleta pela API, incompleta), e o spine local tem n_references em apenas 863.684 registros. Seria preciso o log e o manifesto do processamento do snapshot do OpenAlex e o spine juntado posterior
- 52,40 GB, 35 etapas e 1.295 arquivos: só há a alegação no README e no ESTADO. Seria preciso o manifesto raiz da cadeia e a verificação profunda sobre os dados
- 27,75 B tokens: estimativa por caracteres sobre um corpus que não está nesta máquina. Seria preciso o corpus filtrado (RedPajama, OpenWebMath, peS2o) e uma contagem com tokenizador
- Teto de 0,7562 do protocolo antigo: não há artefato .json/.log versionado com esse valor, só comentários em scripts/avaliar_encoders.py e src/phifm/training/embedding.py e o ESTADO. Seria preciso o pares_validacao original e o pool val.head(2000)
- Bootstrap de nDCG@10 do GTE-base@400k contra o ΦEmb@6M, e do ΦEmb@6M contra o GTE-large: exige os escores por item ou os checkpoints, que não estão aqui
- Execuções de treino (0,6 B e 0,4 B efetivamente consumidos, instabilidades): só as configurações do código foram conferidas; faltam os logs de treino da nuvem e os checkpoints
- PhysBERT e GTE-base diferirem 'exclusivamente no corpus de pré-treinamento': depende dos cartões dos modelos (tokenizador e objetivo de pré-treino), não verificáveis offline; a ressalva do achado sobre o MiniLM-L6 independe disso


## Corpus e métodos (§3.1–3.3, §3.7, §4.1)

### L74 · maior · número não confere · pendente

> truncava 41,5% do acervo

**Problema.** Há três valores para a mesma grandeza (41,5%, ~30%, 22,4%) e nenhum artefato versionado sustenta o do artigo. Sobre o índice de Física, a fração de identificadores anteriores a 2007 é 22,4%. O 41,5% pode vir de uma amostra inicial enviesada para obras antigas, mas isso não está registrado.

**Evidência.** A única ocorrência de 41,5% fora do rascunho é uma linha de tabela em ESTADO.md:4108, sem artefato. No ponto do código em que a regex vive (src/phifm/corpus/acquire/openalex.py:103-105) o comentário diz 'descarta em silêncio ~30% do acervo'. Medido aqui: identificadores em formato antigo (com '/') são 356.850 de 1.595.065 no índice (22,4%) e 223.585 de 1.100.000 nas obras do OpenAlex coletadas (20,3%).

**Correção proposta.** Substituir por um número medido e dizer sobre o quê: "uma expressão regular que não aceitava o formato de identificador anterior a 2007 truncava todos os identificadores desse formato, 22,4% do índice (356.850 registros)" — recalculando no instantâneo de 1.595.422. Se o 41,5% for mantido, citar a amostra em que foi medido.

### L84 · maior · inconsistência interna · aplicado

> Três fatias públicas foram filtradas para Física pelo classificador da Seção 3.2, com limiar de decisão de 0,9

**Problema.** A fatia RedPajama (10,54 B dos 27,75 B tokens) não passou pelo classificador nem por limiar: foi selecionada por pertinência exata do identificador arXiv ao índice. O texto descreve um método que não foi o usado, e atribui à fatia de LaTeX as taxas de falso positivo do classificador, que não se aplicam a ela.

**Evidência.** src/phifm/corpus/slices/redpajama.py, docstring '## O filtro é o spine, não um classificador': 'A pertinência é exata e autoritativa — nada de limiar, nada de probabilidade'. scripts/manifesto_corpus.py ETAPAS['redpajama_fisica']: descricao 'filtrada por casamento exato com a tabela mestra', parametros {'filtro': 'spine (exato)'}. Só openwebmath_fisica e pes2o_fisica têm 'script': 'scripts/filtrar_hf.py', 'limiar': 0.9.

**Correção proposta.** Substituir por: "Três fatias públicas foram filtradas para Física. A fatia arXiv do RedPajama [17] foi selecionada por pertinência exata: retêm-se os documentos cujo identificador arXiv consta do índice da Seção 3.1, sem classificador e sem limiar. O OpenWebMath [18] e o peS2o [16], que não trazem identificador arXiv, foram filtrados pelo classificador da Seção 3.2, com limiar de decisão de 0,9."

### L122 · maior · afirma mais que a evidência · aplicado

> Cada etapa do pipeline grava um manifesto com os hashes BLAKE3 das entradas e saídas, os parâmetros e o identificador do commit.

**Problema.** A frase vale para o código atual, não para os artefatos reportados: em quase todas as etapas o manifesto foi escrito depois, pelo construtor, com parâmetros lidos do código e o commit da construção. Ao menos um desses parâmetros contradiz o log da execução (400.000 contra 300.000 no classificador), e os 300.000 são os que o próprio artigo reporta na l. 80. A ressalva existe na Seção 6, mas a Seção 3.7 afirma o contrário sem remeter a ela.

**Evidência.** ESTADO.md:3469-3472 e 24: só redpajama_math_cs tem parâmetros capturados na execução; as demais etapas derivadas levam parametros_reconstruidos=True, 'os 5 scripts capturam; os artefatos são de antes'. manifesto_corpus.py:278-303 e 336-342: os manifestos das 28 coletas brutas e das derivadas são escritos pelo construtor da raiz, com git_sha = commit da construção. Caso concreto de parâmetro reconstruído errado: ETAPAS['isphysics_clf'] declara max_por_classe = 400_000 (padrão do script), mas isphysics_com_math.log mostra 'cota 75,000' e '300,000 física · 190,210 não-física', isto é, max_por_classe = 300.000 (classifier.py:186, cota = max_por_classe // 4). reprodutibilidade.py, classe Entrada: a entrada carrega o manifesto_id da etapa a montante, ou a nota 'sem manifesto a montante', não o BLAKE3 dos arquivos de entrada.

**Correção proposta.** "O manifesto de cada etapa registra os hashes BLAKE3 dos arquivos de saída, os identificadores dos manifestos de entrada, os parâmetros e um identificador de commit. No código atual a etapa o grava ao terminar; para os artefatos deste trabalho, executados antes disso, os manifestos foram escritos pelo construtor da cadeia, com parâmetros reconstruídos a partir do código e o commit da construção — só a fatia de matemática e ciência da computação tem parâmetros capturados na execução (Seção 6)." Corrigir max_por_classe para 300.000 em ETAPAS e reconstruir a raiz.

### L124 · maior · falta controle ou limitação · em parte

> A cadeia atual cobre 52,40 GB em 35 etapas e 1.295 arquivos

**Problema.** O artigo oferece a cadeia como atestado (resumo, contribuição 1, conclusão), mas não publica o hash raiz e o manifesto raiz não está no repositório: um leitor não tem contra o que conferir. Falta também a limitação que o código declara: a cadeia atesta os bytes que existem em disco e não torna a coleta refazível, porque o OAI-PMH do arXiv é mutável. A Seção 6 e a nota de disponibilidade ('permite reconstrução a partir das fontes públicas') não a mencionam.

**Evidência.** git ls-files | grep -i manifest: só phienc_dados_manifesto.json, manifesto_corpus.py, manifest.py e o teste; data/processed/MANIFESTO-RAIZ.json não é versionado (.gitignore: '/data/processed/*') nem existe nesta máquina. O hash raiz 3113f0fed57c…4d84 aparece só em ESTADO.md:3466 e README.md:144, não no artigo. manifesto_corpus.py, MUTAVEIS: 'arxiv OAI-PMH — uma coleta refeita traz registros que a de hoje não tinha'; reprodutibilidade.py: o hash 'Não prova que rodar o pipeline de novo produz os mesmos bytes'. No rascunho, grep por 'datestamp' e 'mutáve': zero ocorrências. Prova da mutabilidade: as duas coletas (1.595.065 e 1.595.422). Os 52,40 GB incluem a fatia de matemática e ciência da computação (cerca de 12,3 GB) e as coletas brutas; as três fatias de Física somam 12,56 + 3,57 + 18,34 = 34,47 GB.

**Correção proposta.** Acrescentar: "A raiz da cadeia é 3113f0fe…4d84 [hash completo], e o manifesto raiz e os manifestos de etapa acompanham o código. A cadeia atesta os bytes existentes; não torna a coleta refazível. O OAI-PMH do arXiv filtra por data de modificação, e duas coletas do mesmo conjunto feitas com onze horas de intervalo diferiram em 357 registros; a camada bruta coletada é a cópia fixada. Dos 52,40 GB, 34,47 GB são as três fatias de Física; o restante são a fatia de matemática e ciência da computação, os metadados brutos, os pares de citação e o classificador." Versionar MANIFESTO-RAIZ.json (cabe como evidência).

### L151 · maior · sem evidência versionada · pendente

> | Taxa de casamento com o índice | 99,1% (1.581.098) | 98,5% |

**Problema.** O número do resumo só tem apoio numa linha do README. As notas do repositório indicam que o índice atestado pela cadeia foi construído com a coleta parcial da API (257 mil casados; 14,05 M referências), de modo que os 99,1% descreveriam uma junção com o instantâneo completo que não está materializada no índice. Além disso, os 98,5% da coluna 'estimado no planejamento' são outra grandeza: a fração de obras do OpenAlex com identificador extraível, medida em 200 obras, e não a fração de registros do índice com obra casada.

**Evidência.** grep por 1.581.098 / 1581098: só README.md:172 (commit e29fb45, 2026-08-10) e o rascunho; nenhum .json ou .log. ESTADO.md:4182, na seção da construção com 1.595.422 registros: 'OpenAlex casado | 257.321, com a coleta em curso'. ESTADO.md:26: a tabela mestra tinha 14.052.319 referências, vindas de data/raw/openalex_works (API), que 'sumiu'. manifesto_corpus.py:91-92: 'A tabela mestra sai de `openalex_works` (a API), não do snapshot'. No único índice desta máquina (1.595.065), 863.684 registros casam (54,1%). Aritmética: 1.581.098/1.595.422 = 99,10%.

**Correção proposta.** Regerar a contagem na máquina do corpus e gravá-la em data/processed/avaliacao/. Na tabela: "Registros do índice com obra do OpenAlex casada (junção com o instantâneo) | 99,1% (1.581.098 de 1.595.422) | —", e dizer em nota qual artefato carrega essa junção (o índice ou o grafo usado na construção dos pares). Retirar os 98,5% da coluna de planejamento.

### L153 · maior · afirma mais que a evidência · aplicado

> | Revisados por pares | 740.823 (46,4%) | — |

**Problema.** A grandeza é 'registros com o campo journal-ref preenchido pelo autor', um substituto que o artigo não define. Ele subestima o publicado (outros 374.912 registros têm DOI sem journal-ref) e inclui anais e outros veículos sem revisão por pares verificada.

**Evidência.** src/phifm/corpus/normalize/spine.py:141: pl.col("journal_ref").is_not_null().alias("peer_reviewed"). No índice local: journal_ref preenchido em 740.702 registros (46,44%), idêntico à coluna peer_reviewed; DOI preenchido em 1.071.313 (67,2%); 374.912 registros têm DOI e não têm journal_ref.

**Correção proposta.** "| Com referência de periódico declarada (campo journal-ref) | 740.823 (46,4%) | — |", com nota: "Indicador aproximado de publicação: o campo é preenchido pelo autor; 67% dos registros têm DOI."

### L160 · maior · falta controle ou limitação · aplicado

> | Fonte | Documentos aceitos | Tokens estimados |

**Problema.** Os 27,75 B do resumo, da Tabela 2 e da conclusão são caracteres divididos por 4, não uma contagem de tokens por algum tokenizador. O artigo não diz isso em lugar nenhum. A única tokenização medida do projeto sugere razão menor que 4 em LaTeX, o que deslocaria a fatia RedPajama em cerca de 10%.

**Evidência.** scripts/filtrar_hf.py:73: print(f"≈ tokens : {f.caracteres_aceitos/4/1e9:.2f} B (a ~4 chars/token)"). ESTADO.md:4034: '42.145.866.036 caracteres = 10,54 B tokens'. sonda_dominios.json: chave 'math_mais_cs_tokens_a_4_chars'. grep no rascunho por 'caracteres por token', '4 caracteres' e 'chars': nenhuma ocorrência. Indício contrário: phienc_dados_manifesto.json registra 2.001.270.262 tokens em 143.810 documentos com o tokenizador do projeto; a 50,4 mil caracteres por documento (média da fatia) isso dá cerca de 3,6 caracteres por token (aproximado: o teto max_tokens = 2e9 pode ter cortado a última parte).

**Correção proposta.** Acrescentar à legenda da Tabela 2: "Os tokens são estimados como o número de caracteres dividido por 4; nenhuma fatia foi tokenizada por inteiro. Em LaTeX, o tokenizador da Seção 4.4 produziu cerca de 3,6 caracteres por token nas partes efetivamente tokenizadas, de modo que a estimativa da fatia RedPajama é conservadora." (conferir o 3,6 na máquina do corpus antes de afirmar). No resumo e na conclusão, escrever "cerca de 27,75 bilhões de tokens (estimados a 4 caracteres por token)".

### L164 · maior · afirma mais que a evidência · aplicado

> | peS2o [16] | 5.526.331 de 38.972.211 (14,18%) | 14,60 × 10⁹ |

**Problema.** Dos 5.526.331 documentos do peS2o, 3.626.168 (65,6%) são resumos, não texto integral. Isso é 50,2% dos 7.222.231 'documentos aceitos' da tabela. Em tokens o efeito é pequeno (7,9% do peS2o, cerca de 1,15 B), mas a contagem de documentos sob o rótulo 'texto integral' engana. O artigo só menciona resumos de passagem (l. 536 e 619), sem dar a composição.

**Evidência.** data/processed/avaliacao/revisao_pes2o_amostragem.json: estratos.resumo.documentos = 3626168 (fracao_de_documentos 0,6562; peso_em_tokens 0,0791); estratos['texto pleno'].documentos = 1900163 (0,3438; 0,9209). A Tabela 2 (l. 158) se intitula 'Fatias de texto integral', a Seção 3.3 'Fontes de texto integral', e a l. 619 diz 'o corpus filtrado é de texto integral'.

**Correção proposta.** Desdobrar a linha: "peS2o [16], texto integral | 1.900.163 | 13,45 × 10⁹" e "peS2o [16], apenas resumo | 3.626.168 | 1,15 × 10⁹", e acrescentar após a tabela: "Dos documentos aceitos do peS2o, 65,6% são resumos, que respondem por 7,9% dos tokens da fatia; o texto integral concentra 92,1%." Corrigir a l. 619 para "o corpus filtrado é, em tokens, majoritariamente de texto integral".

### L167 · maior · inconsistência interna · aplicado

> na metade superior da faixa de 15 a 30 bilhões de tokens estimada como necessária

**Problema.** Há três problemas na frase. (1) No próprio artigo a faixa de 15–30 B é o que se estimou adquirível, não o necessário (o necessário citado é 160 B). (2) A faixa foi definida após deduplicação e triagem de licença, e o total de 27,75 B é a soma simples de três fontes, sem triagem de licença (registros NC entram) e sem deduplicação entre fontes (peS2o e RedPajama contêm os mesmos artigos do arXiv em duas extrações). (3) Depois do ADR-0003 nenhum modelo do sistema é treinado sobre o corpus; os experimentos usaram 0,6 B e 0,4 B tokens.

**Evidência.** L. 13: 'Estimativas do corpus de Física legalmente adquirível situam-se entre 15 e 30 bilhões de tokens após filtragem'. L. 23: 'entre 15 e 30 bilhões após deduplicação e triagem de licença — uma escassez'. DOC-07:66: '15–30 B tokens são um excedente de 5–10× para um modelo de 150 M'. grep por 'partition|eval_only|train_ok|licen' em redpajama.py, coletar_redpajama.py, hf_filtrado.py, preparar_dados_phienc.py, pretrain/dados.py, pairs.py e build_pairs.py: zero ocorrências; ids_do_spine() usa todos os arxiv_id do índice. O índice local tem 47.007 registros eval_only (licenças NC). ADR-0003 (aceito em 2026-09-23): o ΦEnc não é treinado.

**Correção proposta.** Substituir por: "O total situa-se na metade superior da faixa de 15 a 30 bilhões de tokens estimada como adquirível, a custo monetário nulo. A comparação tem duas ressalvas: a faixa foi estimada após deduplicação e triagem de licença, e o total medido é a soma das três fatias, sem deduplicação entre fontes — o peS2o e a fatia RedPajama podem conter os mesmos artigos do arXiv em extrações distintas, sobreposição não medida — e sem exclusão das licenças não comerciais. Os experimentos de pré-treinamento da Seção 4.9 usaram 0,6 e 0,4 bilhão de tokens desse total."

### L74 · menor · clareza · aplicado

> o casamento pelo campo de identificadores externos do arXiv recupera 1,5% dos registros, contra 98,5% pelo campo de localizações

**Problema.** Os dois percentuais vêm de 200 obras (1,5% são 3 obras) e o tamanho da amostra não é dito. A rota de 1,5% é o DOI DataCite, não um 'campo de identificadores externos do arXiv', que não existe no registro. O 1,44 milhão é a diferença de contagens da API sobre o arXiv inteiro (não só Física), e 'revisados por pares' é inferência. A direção se confirma numa amostra 5.500 vezes maior, que pode ser citada.

**Evidência.** openalex.py:58-59 e 91-95: 'Medido em 2026-08-03 sobre 200 obras'; 'O arXiv ID NÃO está em `ids.arxiv` — esse campo não existe'; a rota de 1,5% é o 'DOI DataCite 10.48550/arXiv.<id>'. openalex.py:12-18: 2,23 M (primary_location) contra 3,67 M (locations), contagens da API sobre todas as obras com origem no arXiv, e 'publicados em revista' é a explicação dada, não uma medida. Remedição aqui em 1.100.000 obras (data/raw/openalex_works): rota por DOI 9.641 (0,88%); DOI ou localizações 1.093.590 (99,42%).

**Correção proposta.** "o identificador arXiv não existe como campo próprio do registro; extraí-lo do DOI recupera 0,9% das obras, contra 99,4% pela URL das localizações (medido em 1,1 milhão de obras); restringir o filtro à localização primária reduz as obras com origem no arXiv de 3,67 para 2,23 milhões, e as 1,44 milhão excluídas são as que têm outra localização primária, tipicamente a revista em que foram publicadas;"

### L78 · menor · clareza · aplicado

> dos 72.919 registros negativos com listagem cruzada em Física, todos os 72.919 constam do índice

**Problema.** A validação cobre três dos quatro domínios negativos e foi feita antes da coleta de matemática, o vizinho mais confundível. Falta o denominador. Sem isso, os 72.919 parecem contradizer os 179.441 da l. 169.

**Evidência.** tests/regression/test_negativos_contaminados.py:3-15 e classifier.py:120-133: medição de 2026-08-11 sobre cs (988.244), econ (16.984) e q-bio (56.142), 'Total: 72.919 de 1.041.652'. DOC-19:226: '`math` em coleta'. A l. 80 lista quatro domínios negativos, incluindo matemática; a l. 169 dá 179.441 listagens cruzadas para matemática e ciência da computação.

**Correção proposta.** "…dos 1.041.652 registros negativos de ciência da computação, economia e biologia quantitativa, 72.919 têm listagem cruzada em Física, e todos os 72.919 constam do índice. A verificação antecede a coleta dos negativos de matemática e não foi repetida sobre eles." (ou repeti-la e reportar)

### L84 · menor · clareza · pendente

> Todas as fontes são fixadas por revisão explícita do repositório de origem

**Problema.** As fatias RedPajama de Física e OpenWebMath deste trabalho foram baixadas pela 'versão inicial', de referência móvel; a revisão foi estabelecida depois, pela data da última modificação da origem. A frase sugere que só uma implementação anterior, sem consequência para os dados, tinha o defeito.

**Evidência.** git log -S: a fixação da revisão entrou em e206b63 (2026-08-17). cadeia_owm.log (commits de 2026-08-14 e 16) registra a filtragem do OpenWebMath, iniciada após o RedPajama ('RedPajama saiu; iniciando OpenWebMath'). ESTADO.md:3566-3569: 'As fatias baixavam de resolve/main/ … Tivemos sorte — o OpenWebMath está em fde8ef8d desde 2023-10-17'. manifesto_corpus.py, IMUTAVEIS: no RedPajama só o índice é fixado (398f9257); os fragmentos vêm de data.together.xyz/…/v1.0.0. O peS2o foi filtrado em 2026-08-26, já com a revisão fixada.

**Correção proposta.** "O código fixa cada fonte por revisão explícita. As fatias RedPajama de Física e OpenWebMath foram coletadas antes dessa fixação, de referência móvel; a revisão que lhes corresponde foi estabelecida a posteriori, pela data da última modificação do repositório de origem, anterior à coleta. No RedPajama fixa-se o índice de fragmentos, e os fragmentos são servidos de endereço versionado pela origem."

### L124 · menor · afirma mais que a evidência · pendente

> uma verificação exige que toda fonte declarada no filtro tenha etapa correspondente

**Problema.** A propriedade existe e passa, mas cobre só as duas fontes do filtro por classificador. As duas fatias vindas do coletor do RedPajama (Física, e matemática e ciência da computação — 2 das 4 fatias de texto e cerca de 25 dos 52 GB) não têm guarda equivalente, e o código o admite.

**Evidência.** tests/regression/test_manifesto_g1_5.py, test_TODA_fonte_do_filtrar_hf_tem_etapa_no_manifesto: compara filtrar_hf.FONTES (openwebmath, pes2o) com ETAPAS. manifesto_corpus.py:147-151: 'esta fatia vem do coletor do RedPajama, que a guarda não cobre'. Replicado aqui sem gravar nada: 7 ETAPAS, nenhuma fonte de FONTES faltando.

**Correção proposta.** "…e uma verificação exige que toda fonte declarada no filtro por classificador tenha etapa correspondente — uma propriedade conferida entre dois trechos de código, e não por convenção. As fatias obtidas pelo coletor do RedPajama não têm guarda equivalente; a sua inclusão na cadeia continua dependendo de declaração manual."

### L148 · menor · inconsistência interna · pendente

> | Registros do arXiv (conjunto `physics`) | 1.595.422 | 1.200.000 |

**Problema.** Existem duas coletas do mesmo conjunto, com cerca de 11 horas de diferença, que divergem em 357 registros únicos. O artigo usa a B de forma coerente, e os percentuais (14,8%; 46,4%) coincidem nas duas, mas o ADR de licenças e os documentos de dados citam a A. O artigo não diz a data da coleta nem que existe outra.

**Evidência.** Instantâneo A: data/S1_COMPLETO.json (unique_records 1595065, train_open 235628), data/raw/arxiv_metadata/_manifest.json (actual_count 1595105, completed_at 2026-08-06T18:59Z), o único spine.parquet desta máquina (1.595.065 linhas; 235.628 abertos; 740.702 com journal-ref), ADR-0001:162-167, DOC-00:109, DOC-02:53, DOC-19:153. Instantâneo B: ESTADO.md:3612 e 4179-4181 (1.595.422; 235.795; 740.823; concluído em 2026-08-07 05:44 UTC), README, sonda_dominios.json ('fisica_na_tabela_mestra': 1595422).

**Correção proposta.** Na legenda da Tabela 1: "Coleta OAI-PMH concluída em 7 de agosto de 2026. Uma coleta independente concluída onze horas antes retornou 1.595.065 registros únicos; a diferença de 357 decorre de o protocolo filtrar por data de modificação (Seção 3.7)." Alinhar ADR-0001, DOC-00, DOC-02 e DOC-19 ao instantâneo usado.

### L150 · menor · clareza · aplicado

> | Obras do OpenAlex processadas | 4.613.751 | — |

**Problema.** As obras processadas (varridas) foram 510 milhões; 4.613.751 são as retidas por terem localização no arXiv, em qualquer área. Só cerca de 1,58 milhão delas casa com o índice de Física, então 'associado a 4.613.751 obras' superestima a associação.

**Evidência.** src/phifm/core/schema/manifest.py:104-110: 'OpenAlex snapshot 510.372.821 [expected] 4.613.751 [actual] … `expected` são as obras VARRIDAS (o OpenAlex inteiro) e `actual` são as GUARDADAS (as que casaram com o arXiv)'. Resumo (l. 13): 'índice de metadados de 1.595.422 registros do arXiv associado a 4.613.751 obras do OpenAlex'.

**Correção proposta.** Na tabela: "Obras do OpenAlex com localização no arXiv (retidas de 510.372.821 varridas) | 4.613.751". No resumo: "um índice de 1.595.422 registros do arXiv, dos quais 99,1% casados com uma obra entre as 4.613.751 de origem arXiv do OpenAlex".

### L152 · menor · número não confere · pendente

> | Arestas de citação | acima de 22,7 milhões | dezenas de milhões |

**Problema.** O valor é um piso de uma coleta parcial e não define 'aresta': referência a qualquer obra do OpenAlex, ou aresta arXiv→arXiv resolvida dentro do corpus. Os artefatos dão 14,05 M, 22,7 M+ e 41,5 M para coletas de completude diferentes. O grafo é a supervisão do bi-encoder e merece um número definido.

**Evidência.** O '22,7 M+' vem do README de 2026-08-10 e de pairs.py:29 ('Com 22,7 M de arestas… ~8% das referências' resolvidas no corpus). data/S1_COMPLETO.json: citation_edges = 41.522.830 com só 1.100.000 obras coletadas (soma de n_references no índice local: 41.522.830, em 863.684 registros casados). ESTADO.md:26: 14.052.319 referências na tabela mestra atestada. ESTADO.md:3497: 6.697.651 pares.

**Correção proposta.** Reportar duas grandezas medidas no instantâneo completo: "Referências listadas nas obras casadas | N₁" e "Arestas resolvidas dentro do índice (arXiv→arXiv) | N₂", de que saem os 6.697.651 pares após o teto de 8 por âncora. N₁ e N₂ precisam ser regerados na máquina do corpus.

### L154 · menor · número não confere · pendente

> | Fração redistribuível | 14,8% (235.795) | 25–35% |

**Problema.** A dedicação ao domínio público da Creative Commons é redistribuível e está contada como não redistribuível. Com ela, a fração arredonda para 14,9%, não 14,8% (efeito de 0,10 ponto, na direção conservadora). O rótulo de versão errado (3.0 lido como 4.0) não altera a fração, porque as duas versões caem na mesma partição, mas deixaria a atribuição de licença errada em 2% de um eventual subconjunto aberto.

**Evidência.** src/phifm/core/licensing/registry.py, _RULES: nenhuma regra casa 'creativecommons.org/licenses/publicdomain/', que cai em UNKNOWN (NOASSERTION, redistributable=False, train_only). No índice local: 1.660 registros com essa licença (todos de 2008–2015), spdx NOASSERTION, partição train_only. (235.628+1.660)/1.595.065 = 14,88%; com os números do artigo, (235.795+1.660)/1.595.422 = 14,88%. A regra 'licenses/by(?![-a-z])' rotula by/3.0 como CC-BY-4.0: 4.618 registros (2,0% da partição aberta); by-nc-sa/3.0 vira CC-BY-NC-SA-4.0: 3.013.

**Correção proposta.** Corrigir o registro (regras para licenses/publicdomain e para a versão 3.0), reconstruir a contagem e reportar "14,9% (237.455)" na Tabela 1 e na l. 653. Ou manter e anotar: "limite inferior: 1.660 registros sob a dedicação ao domínio público da Creative Commons não foram reconhecidos pelo resolvedor de licenças e estão contados como não redistribuíveis".

### L156 · menor · clareza · aplicado

> 0,0% até 2004, 36,2% entre 2020 e 2024, e 48,8% entre 2025 e 2029

**Problema.** Os percentuais conferem, mas a apresentação por lustro esconde um degrau: a fração aberta sobe de cerca de 5% para cerca de 25% entre 9 e 10 de novembro de 2020 e depois cresce devagar. O lustro 2020–2024 faz a média entre um ano a 8% e quatro anos a 35–48%. O artigo fala em 'época de publicação' sem causa. O rótulo '2025 e 2029' descreve dados que terminam em agosto de 2026, e o ano é o da primeira submissão ao arXiv, não o de publicação.

**Evidência.** Índice local, por ano de `created`: 2019 4,2%; 2020 8,0%; 2021 34,9%; 2022 42,7%; 2023 45,8%; 2024 47,9%; 2025 49,1%; 2026 48,5%. Por mês: 2020-10 4,9%; 2020-11 19,9%; 2020-12 27,9%. Por dia: 02 a 08/11/2020 entre 2% e 10% (média ~5%); 09/11 9,6%; 10/11 22,4%; 11/11 27,9%; daí em diante 21–32%. Ano máximo no índice: 2026. Lustros locais: ≤2004 0 de 262.602; 2020–2024 36,17%; 2025+ 48,83%. ADR-0001 (l. ~168): 'A causa é uma transição abrupta, não um declínio suave'.

**Correção proposta.** "Ela depende da data de submissão, e a dependência é um degrau: 0,0% até 2007, entre 1% e 4% de 2008 a 2019, e um salto de cerca de 5% para cerca de 25% entre 9 e 10 de novembro de 2020, seguido de crescimento lento até 48–49% em 2025–2026. Um salto de um dia não é mudança de comportamento dos autores; é compatível com uma alteração nas opções de licença do formulário de submissão do arXiv, hipótese que este trabalho não verificou." (ou verificá-la e citar a fonte)

### L156 · menor · afirma mais que a evidência · aplicado

> A fração redistribuível é a única estimativa de planejamento refutada

**Problema.** Outras duas estimativas da tabela ficaram fora do previsto: a contagem de registros (+33%) e os bytes por registro (fora da faixa). Erraram na direção favorável, mas erraram. 'Tamanho em disco' sob o título 'Índice de metadados construído' refere-se à coleta bruta, não ao índice.

**Evidência.** Tabela 1: registros 1.595.422 contra 1.200.000 estimados (+33%; DOC-02:53 e DOC-19:153 registram o erro de +33%); 422 bytes/registro contra a faixa 516–686 (18% abaixo do limite inferior). Os 674 MB (ESTADO.md:3612, '78 shards, 674 MB') são da coleta bruta do arXiv; o índice construído tem 0,76 GB (ESTADO.md:3495).

**Correção proposta.** "A fração redistribuível é a única estimativa de planejamento que errou na direção desfavorável, e a discrepância é ampla; a contagem de registros ficou 33% acima do estimado e o tamanho por registro, abaixo da faixa prevista." Renomear a linha para "Tamanho em disco da coleta bruta do arXiv".

### L165 · menor · número não confere · aplicado

> | Total | 7.222.231 | 27,75 × 10⁹ |

**Problema.** As parcelas exibidas somam 27,76 e o total diz 27,75. É arredondamento das parcelas, mas na tabela aparece como soma que não fecha.

**Evidência.** 10,54 + 2,62 + 14,60 = 27,76. ESTADO.md:4034-4035: RedPajama exato 42.145.866.036/4 = 10,536 B e 'Com o OpenWebMath, o corpus é 13,15 B' (ou seja, o OpenWebMath está em torno de 2,61 B); 13,15 + 14,60 = 27,75. Documentos: 835.379 + 860.521 + 5.526.331 = 7.222.231 (confere).

**Correção proposta.** Acrescentar à legenda "o total é calculado antes do arredondamento das parcelas", ou exibir as parcelas com três casas (10,536; 2,61x; 14,600).

### L169 · menor · afirma mais que a evidência · aplicado

> havia previsto esses volumes com erro inferior a 2%

**Problema.** A frase é verdadeira para matemática e ciência da computação, mas a conferência que a própria sonda definiu como régua, a Física já em disco, errou 4,2%, e o erro de amostragem de 5 fragmentos em 100 não foi calculado. Citar só o acerto dá à sonda uma precisão que ela não demonstrou.

**Evidência.** data/processed/avaliacao/sonda_dominios.json: math_mais_cs_documentos 685200 (real 687.907, −0,4%); math_mais_cs_caracteres 44.644.575.500 (real 44,9 G, −0,6%); 'fisica_conferencia': 794100 contra 828.601 em disco, e a chave 'como_ler' diz que essa é 'a régua desta sonda' e que 'o erro de AMOSTRAGEM não está calculado'. ESTADO.md:1394: 'erro de −4,2%'.

**Correção proposta.** "Uma sonda sobre 5 de 100 fragmentos sorteados havia previsto esses volumes com erro inferior a 1%; a conferência da própria sonda, sobre a fatia de Física já em disco, errou 4,2%, e o erro de amostragem não foi calculado."

### Conferido e correto

- L72: coleta por OAI-PMH restrita ao conjunto physics, com cursor durável — data/raw/arxiv_metadata/_manifest.json (query_spec.set = 'physics', harvest_method oai_pmh); 13 colunas lidas do instantâneo do OpenAlex — COLUNAS em src/phifm/corpus/acquire/openalex_snapshot.py tem 13 entradas; leitura por faixa de bytes sem parquet inteiro em disco (docstring do mesmo módulo).
- L74: 1,44 milhão = 3,67 M (locations) − 2,23 M (primary_location), em openalex.py:12-18; o manifesto local da API tem expected_count = 3.667.221. A direção do 1,5% contra 98,5% se confirma em 1.100.000 obras locais: 0,88% pelo DOI, 99,42% por DOI ou localizações.
- L78: a regra de rótulo é a pertinência ao índice e não o prefixo da primária (classifier.py:120-140); o 72.919 de 72.919 consta do teste de regressão, do classificador e do DOC-19 (os negativos brutos não estão nesta máquina).
- L80: classificador linear (SGDClassifier, loss='modified_huber') sobre TF-IDF (classifier.py:366-395); 300.000 positivos e 190.210 negativos; quatro domínios (cs, econ, math, q_bio); avaliação por exclusão de domínio — tudo em data/processed/avaliacao/isphysics_com_math.log.
- L84: limiar 0,9 para OpenWebMath e peS2o (cadeia_owm.log: 'openwebmath filtrado · limiar 0.9'; ETAPAS); a fixação de revisão existe no código (hf_filtrado.py; REVISAO do índice do RedPajama = 398f9257…, igual à gravada em sonda_dominios.json).
- L86: 3.000 documentos por fatia, semente 17, e os quatro indicadores (pct_com_sequencia_latex, pct_com_math_inline, pct_com_ambiente_de_equacao, sequencias_latex_por_doc) — data/processed/avaliacao/equacoes_por_fatia.json.
- L122: cadeia em três níveis (raiz → hash do manifesto de etapa → BLAKE3 por arquivo) e distinção entre verificação rasa e profunda — src/phifm/core/schema/reprodutibilidade.py.
- L124: a guarda fonte×etapa existe (tests/regression/test_manifesto_g1_5.py) e passa — replicada aqui: 7 etapas derivadas, nenhuma fonte de filtrar_hf.FONTES sem etapa. Aritmética coerente com o histórico: 21,79 + 18,34 = 40,13 GB (commit 3d4da86, 'hash raiz 90beca3d sobre 40,13 GB'); 33 + 2 = 35 etapas; 976 + 282 + 37 = 1.295 arquivos; 21,79/40,13 = 54%.
- Tabela 1, aritmética no instantâneo do artigo: 674 MB/1.595.422 = 422 bytes; 235.795/1.595.422 = 14,78%; 740.823/1.595.422 = 46,43%; 1.581.098/1.595.422 = 99,10%.
- Tabela 1, conferência no único índice desta máquina (1.595.065 registros): redistribuível 235.628 (14,77%); journal-ref 740.702 (46,44%); eval_only 47.007; train_only 1.312.430 — idêntico a data/S1_COMPLETO.json e ao ADR-0001.
- L156: frações por época reproduzidas no índice local: 0 de 262.602 até 2004 (0,0%); 136.254 de 376.666 em 2020–2024 (36,17%); 86.597 de 177.334 em 2025–2026 (48,83%).
- Pista 2 confirmada: 'creativecommons.org/licenses/publicdomain/' cai em NOASSERTION/train_only (1.660 registros); by/3.0 recebe spdx CC-BY-4.0 (4.618 registros, na partição aberta); by-nc-sa/3.0 recebe CC-BY-NC-SA-4.0 (3.013). Só o primeiro altera a fração: 14,8% passa a 14,9%.
- Pista 3 confirmada nos dados: degrau da fração aberta de cerca de 5% para cerca de 25% entre 2020-11-09 e 2020-11-10 (por data de `created`). O artigo não atribui causa.
- Tabela 2: 835.379 linhas, 828.601 identificadores distintos, 6.778 repetidos (0,81%) — data/processed/avaliacao/pb_formula_montagem.json; 5.526.331/38.972.211 = 14,18%; 5.526.331 confirmado em revisao_pes2o_amostragem.json; soma de documentos = 7.222.231; 42,15 G caracteres/4 = 10,54 B; taxa de aceitação do OpenWebMath de 13,6% (cadeia_owm.log), coerente com 860.521 em cerca de 6,3 M.
- L167: as partes do RedPajama usadas no pré-treinamento (phienc_dados_manifesto.json: 35, 14, 02, 09, 03, 13, 00, 43, 28) não incluem 32, 33 nem 34, onde estão as cópias.
- L169: sonda sobre 5 de 100 fragmentos, semente 17 (sonda_dominios.json); 44,9/4 = 11,2 B; 42,15 + 44,9 = 87,05; 44,9/87,05 = 51,6% (mais da metade); 179.441 e 1.438.941 constam de ETAPAS e batem com os metadados da sonda (650.645 math + 932.221 cs menos a sobreposição).
- Pós-v0.5: T1h, T1i, o commit 62e42fc e o ADR-0003 não alteram nenhum número das linhas 70–87, 120–125 e 142–170; o único reflexo é o enquadramento da l. 167 (faixa 'necessária'), tratado nos achados. O índice de busca criado depois (1.594.338 artigos) difere do índice de metadados só pelo corte de comprimento mínimo do texto, não por outra coleta.

### Não verificável nesta máquina

- Cadeia de hashes (52,40 GB, 35 etapas, 1.295 arquivos, raiz 3113f0fe…): data/processed/MANIFESTO-RAIZ.json e os _manifesto_etapa.json não são versionados nem estão nesta máquina. É preciso rodar `scripts/manifesto_corpus.py --verificar --profundo` na máquina do corpus e versionar o manifesto raiz. Conferir também se a raiz continua válida depois da limpeza de 7,8 GB de 2026-09-25 (commit e1577f3).
- Instantâneo de 1.595.422 registros (235.795 redistribuíveis; 740.823 com journal-ref): aqui só existe o instantâneo de 1.595.065. É preciso o spine.parquet da máquina do corpus.
- Taxa de casamento de 99,1% (1.581.098) e as 4.613.751 obras: exigem os fragmentos de data/raw/openalex_snapshot. Falta também esclarecer qual fonte do OpenAlex alimentou o índice atestado: manifesto_corpus.py diz a API (openalex_works), ESTADO.md mostra o comando com o instantâneo, e o índice atestado tinha 14.052.319 referências.
- As 189 colunas do instantâneo do OpenAlex: exigem o esquema do parquet de origem.
- 72.919 de 72.919: exige data/raw/arxiv_negativos (cs, econ, q-bio, math).
- peS2o (38.972.211 vistos; 58,40 G caracteres), total do OpenWebMath (860.521; 2,62 B) e fatia de matemática e ciência da computação (687.907 documentos; 44,9 G caracteres; duplicação 1,000×; sobreposição zero com a Física): exigem as fatias em data/processed/. Para o OpenWebMath, o log versionado cobre só a execução retomada (233.079 aceitos em 1.717.282 vistos).
- Cópias 'idênticas byte a byte' dos 6.778 identificadores repetidos e sua localização nas partes 32–34: exigem a fatia redpajama_fisica.
- Sobreposição entre fontes (os mesmos artigos do arXiv no RedPajama e no peS2o) e, portanto, o total deduplicado em tokens: exige cruzar identificadores das duas fatias.
- Razão real de caracteres por token em cada fatia: exige tokenizar uma amostra de cada fatia com o tokenizador do projeto.
- Causa externa do degrau de 9–10 de novembro de 2020 (mudança das opções de licença no formulário de submissão do arXiv): exige fonte externa, por exemplo o registro de mudanças do arXiv; não foi consultada nesta revisão.
- Número definitivo de arestas de citação (referências totais e arestas arXiv→arXiv resolvidas): exige o grafo do instantâneo completo.


## Classificação de domínio e integridade da notação (§3.2, §4.2, §4.3)

### L173 · maior · afirma mais que a evidência · aplicado

> com taxa de falso positivo entre 2,4% e 3,7% em cada um dos quatro domínios negativos representados no treinamento

**Problema.** A frase atribui ao classificador final, e a cada domínio, um número que foi medido em outros modelos e agregado sobre três domínios. Nenhuma medição versionada dá o FP por domínio do modelo final; a que existe para o modelo final (4,6% agregado) contradiz a faixa, e a única por domínio (comentário de código) vai de 1,3% a 5,8%.

**Evidência.** data/processed/avaliacao/isphysics_com_math.log: o classificador final tem revocação de nao_fisica = 0,954 em 28.541 negativos de validação, isto é, FP agregado de 4,6%, fora da faixa 2,4–3,7%. Os 2,4–3,7% são a coluna 'FP dentro' de transferencia.json, e src/phifm/corpus/filter/classifier.py (avaliar_transferencia, linhas 291–327) mostra o que ela mede: modelos de exclusão de domínio (120 mil + 120 mil, não o modelo final), FP sobre `neg_dentro` = os três domínios restantes AGREGADOS, tomados por .head(120000) sem sorteio. A linha 'math 2,4%' é o FP em cs+econ+q_bio quando math é omitido, não o FP em math. O único FP por domínio registrado está em comentário do mesmo arquivo (linhas 170–174): estratificado cs 2,8%, econ 1,3%, q-bio 4,8%, math 5,8%.

**Correção proposta.** Substituir por: "O classificador final atinge acurácia de 0,954 na validação (73.531 documentos), com taxa de falso positivo agregada de 4,6% sobre os quatro domínios negativos do treinamento. A taxa por domínio do modelo final não foi medida; nos modelos de exclusão de domínio da Tabela 3, a taxa agregada sobre os domínios presentes no treinamento fica entre 2,4% e 3,7%." Se o FP por domínio do modelo final for medido, reportá-lo em tabela própria.

### L185 · maior · afirma mais que a evidência · aplicado

> admitiria cerca de 35% de conteúdo não pertinente

**Problema.** Três saltos: (i) 35,4% é a fração de resumos de matemática do arXiv aceitos, não a fração do conteúdo admitido; (ii) é extrapolada para páginas web, o que o log do experimento declara não poder fazer; (iii) usa o limiar 0,5, quando a filtragem foi feita a 0,9, onde o mesmo contrafactual dá 13,6%.

**Evidência.** transferencia.json (math): fp_fora = 0,3539 no limiar 0,5, medido em 120.000 resumos de matemática do arXiv; no limiar 0,9, que é o de produção (linha 84 do rascunho), fp = 0,1356. O próprio isphysics_com_math.log encerra com: 'Sobre texto de web não diz nada, e nenhum dado que temos diria'. A linha 619 do rascunho reconhece que as taxas foram medidas em resumos.

**Correção proposta.** "Sem negativos de matemática, o classificador aceita como Física 35,4% dos resumos de matemática do arXiv no limiar 0,5, e 13,6% no limiar de 0,9 usado na filtragem. O OpenWebMath é majoritariamente matemático, mas é texto de web, e a taxa nele não foi medida."

### L202 · maior · erro estatístico · aplicado

> acusa 81,3% dos documentos da fatia construída a partir do fonte LaTeX, contra 3,0% da referência

**Problema.** A fração de documentos com ao menos um órfão cresce com o comprimento, e a comparação 81,3% contra 3,0% é entre documentos 44 vezes mais longos. Texto íntegro do tamanho do peS2o daria 52%, praticamente os 50,7% medidos nele: o indicador não detecta a degradação do peS2o. A causa dada (delimitador `$`) existe também nos resumos e explica só o excesso de densidade do RedPajama. A conclusão de que o indicador é inutilizável se sustenta e fica mais forte, mas o texto não mostra o número do corpus degradado em que ela se apoia, e 'satura' (linha 32) não descreve 81,3%.

**Evidência.** equacoes_por_fatia.json: referência 1.123 caracteres/doc, RedPajama 49.212, peS2o 28.605 (50,7% com órfão, valor que o rascunho não mostra), OpenWebMath 16.009 (40,2%). Reprodução em revisao/orfao.py com o mesmo regex sobre 40.000 resumos de data/raw/arxiv_metadata: 2,8% com órfão a 1.053 caracteres; os mesmos resumos íntegros concatenados dão 35,0% a 16.009 caracteres, 52,3% a 28.605 e 72,3% a 49.212. Por mil caracteres, o JSON dá 0,036 (referência), 0,110 (peS2o), 0,220 (OWM), 0,437 (RedPajama).

**Correção proposta.** "Ele não discrimina: acusa 81,3% dos documentos da fatia de fonte LaTeX, 50,7% da fatia extraída de PDF e 3,0% da referência, mas a fração de documentos com ao menos uma ocorrência cresce com o comprimento. Resumos íntegros concatenados até 28,6 mil caracteres dão 52%, o mesmo que a fatia degradada. Por mil caracteres, a densidade é de 0,04 na referência, 0,11 no texto de PDF e 0,44 no fonte LaTeX, onde `$Z_{\rm max}$ = 15 kpc` dispara a expressão regular porque o delimitador não é aceito como operando." Incluir a coluna de órfãos na Tabela 4 e trocar 'satura' na linha 32.

### L202 · maior · afirma mais que a evidência · aplicado

> O discriminante válido é a presença do ambiente de equação

**Problema.** Quanto ao comprimento, o contraste é robusto: 0 em 3.000 documentos de 28,6 mil caracteres não se explica por tamanho. Mas o indicador mede formato, não integridade. Texto de PDF nunca contém `\begin{equation}`, preserve ou não as equações; a referência íntegra pontua igual ao corpus degradado; e o OpenWebMath pontua quase como o peS2o. A validade está apoiada em um único par de fatias em que formato e integridade estão perfeitamente confundidos, sem corpus de integridade conhecida em outro formato. 'Discriminante válido para integridade de notação matemática em corpora' (linhas 32 e 635) excede o que foi medido.

**Evidência.** scripts/medir_equacoes_mutiladas.py: AMBIENTE = \\begin\{(equation|align|eqnarray|gather|multline) — detecta marcação LaTeX de ambiente, não `$$…$$`, `\[…\]` nem matemática em Unicode. equacoes_por_fatia.json: referência íntegra = 0,0% (0,10% na minha amostra de 20.000 resumos); OpenWebMath, descrito no script como 'LaTeX preservado na extração' e com 85,2% de matemática em linha, = 5,1%; peS2o = 0,0%; RedPajama = 84,9%.

**Correção proposta.** "Entre as duas fatias de texto integral do mesmo domínio, a presença de ambiente de equação em LaTeX separa 84,9% de 0,0%. O indicador é específico de formato: vale 0,0% também nos resumos de referência, que não têm equações em display, e 5,1% no OpenWebMath, que usa outros delimitadores. Não foi validado contra um corpus de integridade conhecida fora do fonte LaTeX." Ajustar a contribuição 2 (linha 32) e a conclusão (linha 635) para 'um indicador de presença de equações em display em LaTeX'.

### L204 · maior · afirma mais que a evidência · aplicado

> a fração de equações ausentes é de 16,6%

**Problema.** O número não é a fração de equações ausentes: é um déficit líquido de contagem de expressões matemáticas distintas, em linha e em display, por artigo. Nenhuma equação foi verificada como ausente. Pode superestimar (diferenças de versão, de expansão de macros e de montagem alteram a contagem de formas distintas) e subestimar (formas só do RedPajama compensam as faltantes dentro do artigo). O estimador de razão é dominado por poucos artigos: o artigo típico perde 3,8%.

**Evidência.** src/phifm/eval/latex_audit.py, linhas 176–178: ausentes = Σ max(0, eq_fonte − eq_redpajama) / Σ eq_fonte, em que as contagens são de formas canônicas DISTINTAS (linhas 343–344), e src/phifm/core/latex/extrair.py extrai matemática em linha e em display (≥ 4 caracteres). s3b_latex.json recomputado: 9.255/55.780 = 0,1659; não casadas no total = 25,7% (o rascunho não informa); 23 artigos têm mais formas no RedPajama que no fonte (876); mediana do déficit por artigo = 3,8%, média 14,9%; 113 de 298 com déficit acima de 10%; 15 artigos concentram 41% do déficit, e um só (2209.14501, 1.524 → 488) concentra 11,2%.

**Correção proposta.** "Sobre 298 artigos de Física, 25,7% das expressões matemáticas distintas do fonte (em linha e em display) não têm correspondente canônico na fatia. Desse total, 16,6 pontos são déficit líquido de contagem — a fatia tem menos expressões distintas que o fonte — e 9,1 são discordância de forma. O déficit é concentrado: a mediana por artigo é de 3,8%, e 15 artigos respondem por 41% dele. O déficit de contagem é uma aproximação da perda de conteúdo, não uma contagem de equações ausentes."

### L204 · maior · falta controle ou limitação · aplicado

> Um segundo experimento quantifica a perda da fatia construída a partir do fonte LaTeX em relação ao fonte original

**Problema.** Parte do 'déficit' compara a versão de hoje do artigo com a versão que o RedPajama congelou em 2023: equações acrescentadas em revisão posterior contam como perda da fatia. O efeito é mensurável (cerca de 2 pontos na estimativa), e o artigo não o menciona. A conclusão qualitativa sobrevive: o intervalo sem os artigos revisados continua acima de 10%.

**Evidência.** latex_audit.py, linhas 66 e 291: o fonte é baixado de https://arxiv.org/e-print/<id> sem versão, isto é, a versão ATUAL; docs/adr/ADR-0002 §4: 'O RedPajama-1T é um instantâneo de 2023'. Cruzando s3b_latex.json com data/processed/spine.parquet: pelo campo `created` (data da versão corrente; só 166 de 298 caem no mês do identificador), 16 artigos têm versão de 2023-03-01 em diante — déficit de 32,5% neles contra 14,7% nos outros 282, e 20,9% do déficit total; pelo campo `updated`, são 36 artigos, 27,0% contra 14,1%, 31,4% do déficit. O maior contribuinte, 2209.14501, tem versão de 2023-05-22. Sem os 16: 14,7%, IC 95% [11,9%; 17,8%] pelo mesmo bootstrap; sem os 36: 14,1% [11,3%; 17,2%].

**Correção proposta.** Acrescentar: "O fonte de comparação é a versão corrente do arXiv, e a fatia é um instantâneo de 2023; 16 dos 298 artigos têm versão posterior ao instantâneo e concentram 21% do déficit. Excluídos, o déficit é de 14,7%, com intervalo [11,9%; 17,8%]." Melhor ainda: refazer a auditoria baixando a versão vigente na data do instantâneo e reportar esse número como principal.

### L204 · maior · erro estatístico · aplicado

> com intervalo de confiança de 95% em [12,9%; 20,8%]

**Problema.** A amostra é de conveniência: artigos submetidos em quatro ou cinco datas específicas, não um sorteio dos 835 mil documentos. O intervalo mede a variação entre artigos dentro desses blocos e não a incerteza sobre a fatia; o déficit varia de 11,4% a 28,4% entre blocos, variação que o intervalo não incorpora. Chamar o resultado de 'intervalo de confiança de 95%' para a fatia de Física excede o desenho.

**Evidência.** latex_audit.py, amostrar_redpajama (linhas 239–275): a amostra são os primeiros 64 MB do PRIMEIRO fragmento do RedPajama, filtrados pelo índice de Física. s3b_latex.json: os 298 identificadores formam cinco blocos contíguos — 1607.04516–04937 (148 artigos, déficit 13,8%), 2012.08681–08832 (54; 11,4%), 2203.01948–02041 (45; 17,3%), 2209.14481–14623 (49; 28,4%), 2005.143–144 (2; 2,8%). O bootstrap (linhas 102–124) reamostra artigos como se fossem sorteio independente; reproduzi [0,1294; 0,2084].

**Correção proposta.** "Sobre 298 artigos de Física tomados do início de um dos 100 fragmentos da fonte — cinco blocos contíguos de identificadores, de 2016 a 2022 —, o déficit é de 16,6%, com intervalo de reamostragem de artigos em [12,9%; 20,8%]. A amostra não é um sorteio da fatia, e o déficit varia de 11% a 28% entre os blocos; o intervalo não cobre essa variação." Para um intervalo de fato, sortear artigos entre fragmentos.

### L204 · maior · afirma mais que a evidência · aplicado

> 97% das equações que não casaram utilizavam macros definidas pelo autor

**Problema.** O 97% é apresentado, logo após 'Sobre 298 artigos de Física', como propriedade da amostra. Foi medido em um único artigo, fora da amostra e fora do domínio, antes da correção. Além disso, ele justifica a EXPANSÃO de macros, não a decomposição entre ausência e discordância que a frase diz justificar: são duas correções distintas, e a segunda permanece necessária depois da primeira.

**Evidência.** src/phifm/core/latex/macros.py (linhas 3–8) e latex_audit.py (linhas 325–328): 'Medido em 2026-08-10 no arXiv 1607.04520: casaram 571, 20% usam macro; NÃO casaram 239, 97%'. É UM artigo, com 49 macros. Esse identificador não está em data/processed/spine.parquet (não é Física pelo índice do projeto) nem entre os 300 de s3b_latex.json. Não há artefato de evidência além do comentário de código. E no resultado final as macros já são expandidas (latex_audit.py, linha 329) e a discordância residual é de 9,1%.

**Correção proposta.** "Duas correções de medida antecedem esse número. A primeira é expandir as macros definidas pelo autor antes de comparar: em um artigo examinado em detalhe, com 49 macros, 97% das 239 expressões que não casavam usavam macro, contra 20% das 571 que casavam. A segunda é separar déficit de contagem de discordância de forma, que persiste em 9,1% mesmo com as macros expandidas e pode ser resíduo do comparador."

### L78 · menor · clareza · aplicado

> dos 72.919 registros negativos com listagem cruzada em Física, todos os 72.919 constam do índice

**Problema.** A validação 'exata' cobre três dos quatro domínios negativos do treinamento; matemática, o domínio com mais listagem cruzada em Física, não entra na contagem. Além disso, o que ela valida é a completude do conjunto `physics` do OAI-PMH frente às listas de categoria, não a regra de rótulo em si.

**Evidência.** tests/regression/test_negativos_contaminados.py (linhas 11–19) e classifier.py (linhas 131–133): os 72.919 são de 1.041.652 negativos únicos de cs (988.244), econ (16.984) e q-bio (56.142), medidos em 2026-08-11, antes da coleta de matemática (774.063 registros) e de estatística. A linha 80 descreve quatro domínios, incluindo matemática.

**Correção proposta.** "A coleta é consistente com essa regra: dos 1.041.652 registros coletados em ciência da computação, economia e biologia quantitativa, 72.919 têm listagem cruzada em Física, e todos os 72.919 constam do índice. A mesma conferência não foi repetida para matemática." (ou repeti-la e atualizar o número).

### L80 · menor · falta controle ou limitação · pendente

> A avaliação principal é por exclusão de domínio

**Problema.** O leitor entende que a exclusão de domínio avalia a mesma receita estratificada descrita na frase anterior. Não avalia: os modelos da Tabela 3 têm outro tamanho (120 mil por classe) e outra composição de negativos, dominada pelo maior domínio restante. Isso condiciona a leitura de 'FP interno' e da assimetria entre domínios.

**Evidência.** classifier.py, avaliar_transferencia: os negativos de treino vêm de `carregar(treino_dom, n)` = ordenação por hash sobre os domínios juntos, .head(120000), ou seja, amostragem PROPORCIONAL, não a estratificada por cota do modelo final (montar_binario, cota de 75.000). Com os tamanhos de sonda_dominios.json (cs 932.221; math 650.645; q_bio 37.734; econ 16.055), omitindo math o treino negativo é ~94,5% cs. O próprio código registra (linhas 170–174) que a amostragem proporcional dá 18,9% de FP em q-bio contra 4,8% da estratificada.

**Correção proposta.** Acrescentar: "Os modelos da exclusão de domínio são treinados com 120.000 documentos por classe, e seus negativos são sorteados em conjunto dos domínios restantes, em proporção ao tamanho de cada um, e não pela cota estratificada do modelo final; a entrada é título e resumo."

### L167 · menor · erro de referência · aplicado

> A Seção 4.3 mostra que o volume, entretanto, não é o fator limitante.

**Problema.** A Seção 4.3 mostra que metade do volume (peS2o, 14,6 de 27,75 bilhões de tokens) não tem equações em display; não mostra que o volume não limita coisa alguma, porque nada ali é condicionado ao volume.

**Evidência.** A Seção 4.3 (linhas 189–204) reporta indicadores de notação e a auditoria do RedPajama; nenhuma medida relaciona volume de corpus a desempenho. Fora da faixa atribuída, mas a remissão é à seção sob revisão. Depois da v0.5, o ADR-0003 (ea60df2) registra que o encoder próprio não é treinado, de modo que o corpus de texto integral não alimenta o sistema final.

**Correção proposta.** "A Seção 4.3 mostra que mais da metade desse volume, a fatia extraída de PDF, não contém equações em display, de modo que o volume total superestima o texto utilizável para objetivos que dependem de notação."

### L180 · menor · clareza · pendente

> | Economia | 3,3% | 8,5% | 2,6 | 0,988 |

**Problema.** A coluna 'Precisão externa' não é comparável entre linhas: depende da razão positivos/negativos do teste, que varia de 1:1 a 7,5:1. Economia aparenta ser o melhor caso (0,988) com FP quase igual ao de ciência da computação (0,907). E o valor de estatística existe na evidência, mas está omitido.

**Evidência.** transferencia.json: n_teste_neg = 120.000 (cs, math), 16.055 (econ), 37.734 (q_bio), sempre contra 120.000 positivos. Precisão de econ = 114.046/(114.046+1.360) = 0,988 com FP de 8,5%; cs = 0,907 com FP de 9,8%. transferencia_stat.json: precisao_fora = 0,9701 para estatística, que a tabela mostra como '—'.

**Correção proposta.** Acrescentar coluna 'Negativos de teste' (120.000; 16.055; 120.000; 37.734; 120.000) e nota na legenda: "a precisão externa é calculada contra 120.000 positivos e não é comparável entre linhas com número diferente de negativos". Preencher estatística com 0,970, e registrar que essa linha vem de modelo treinado com os quatro domínios.

### L185 · menor · sem evidência versionada · pendente

> porque a perda `modified_huber` satura as probabilidades

**Problema.** A saturação é propriedade conhecida da perda (probabilidade = (clip(margem, −1, 1)+1)/2), e é compatível com o patamar, mas a relação causal é afirmada sem medição: o que os números mostram é que ~10% dos negativos de matemática recebem margem ≥ ~1, isto é, erro confiante.

**Evidência.** transferencia.json (math): FP de 0,1170 (0,95), 0,1029 (0,99), 0,0998 (0,999) — o patamar confere. Mas nenhum arquivo em data/processed/avaliacao mede a fração de negativos de matemática com probabilidade igual a 1,0; a explicação vem de docstring de classifier.py (linhas 272–275), escrita para o caso q-bio anterior.

**Correção proposta.** "...reduz essa taxa a 10,0% e estanca: cerca de 10% dos resumos de matemática recebem a probabilidade máxima, que a perda `modified_huber` trunca em 1, de modo que nenhum limiar os separa." Ou medir e reportar a fração com probabilidade igual a 1,0.

### L187 · menor · sem evidência versionada · pendente

> a acurácia é de 0,830 contra 0,5 do acaso

**Problema.** O número que refuta a primeira previsão não tem artefato de evidência; só o relato do projeto, sem tamanho de amostra nem suporte por classe.

**Evidência.** grep por 0,830/0.830 em data/processed/avaliacao, scripts e src: o único registro é um relatório colado em ESTADO.md (linhas 3760–3764). Não há log nem JSON versionado, nem script que reproduza o treino restrito a primária math.

**Correção proposta.** Versionar o log do experimento (com n por classe) ou qualificar: "em experimento exploratório, sem artefato versionado, a acurácia foi de 0,830".

### L191 · menor · falta controle ou limitação · pendente

> 3.000 documentos sorteados por fatia

**Problema.** A amostra é o início de uma partição sorteada, com piso de 2.000 caracteres, não um sorteio de 3.000 documentos da fatia. O contraste 84,9 contra 0,0 é grande demais para depender disso, mas os valores da Tabela 4 (em especial 84,9% e 1.158,3) são de uma partição de 44, numa fatia cuja heterogeneidade entre partições o próprio artigo documenta.

**Evidência.** medir_equacoes_mutiladas.py, _amostrar: sorteia 8 partições, lê `.head(n*4)` = as primeiras 12.000 linhas na ordem da lista, descarta documentos com menos de 2.000 caracteres (exceto na referência) e sorteia 3.000 do que sobra. conta_rp.log: partições do RedPajama têm 20.000 documentos, logo as 12.000 linhas vêm de UMA partição (random.Random(17).sample(range(44), 8)[0] = 33); no peS2o (277 partições, ~19.950 docs), da partição 267. ESTADO.md 2738–2741 e a linha 539 do rascunho registram que as partições do RedPajama são heterogêneas em equações (18,8% contra 29,5% em display).

**Correção proposta.** "3.000 documentos sorteados entre as 12.000 primeiras linhas de partições sorteadas de cada fatia (na prática, uma partição no RedPajama e no peS2o), com mínimo de 2.000 caracteres nas três fatias de texto integral." Idealmente, repetir sorteando documentos de todas as partições.

### L193 · menor · clareza · aplicado

> Sequências/doc.

**Problema.** O número 1.158,3 do RedPajama inclui comandos de estrutura e citação, não só matemática; em fonte LaTeX, 100% em 'LaTeX (%)' é trivial. O rótulo induz a ler a coluna como quantidade de matemática.

**Evidência.** medir_equacoes_mutiladas.py: CTRL = \\[a-zA-Z]+\*? ; 'sequencias_latex_por_doc' conta toda sequência de controle (\section, \cite, \ref, \begin), e 'LaTeX (%)' é a presença de qualquer uma. A linha 86 do rascunho chama a coluna de 'número de sequências matemáticas por documento'.

**Correção proposta.** Renomear a coluna para "Sequências de controle LaTeX/doc." e, na linha 86, trocar "sequências matemáticas" por "sequências de controle LaTeX (qualquer comando, matemático ou não)".

### L200 · menor · afirma mais que a evidência · aplicado

> A inspeção de trechos confirma remoção, e não codificação alternativa

**Problema.** Dois ou três trechos escolhidos ilustram, não confirmam. E o segundo exemplo do próprio parágrafo é codificação alternativa (matemática em linha achatada em texto), o oposto do que a frase afirma em geral. O que os trechos sugerem é remoção das equações em display e achatamento das em linha; a fração removida não foi medida.

**Evidência.** Os trechos existem apenas em ESTADO.md (linhas 3163–3169), três citações de documentos diferentes; não há artefato versionado (o script imprime 2 trechos por fatia em stdout). O segundo, 'The stationary solution p * to Eq. (7) satisfies p * = W · p *', mostra uma equação em linha que SOBREVIVEU em forma achatada. O exemplo `T eff` não aparece em nenhum dos trechos citados.

**Correção proposta.** "A inspeção de trechos ilustra dois mecanismos: a equação em display é removida — em um documento, a equação entre 'if:' e 'where' foi eliminada —, e a matemática em linha é achatada em texto, com `p^*` grafado `p *`. A fração de equações removidas na fatia não foi medida." Versionar os trechos com identificador do documento.

### L202 · menor · clareza · aplicado

> com separação de 84,9 contra 0,0 pontos percentuais

**Problema.** 84,9 e 0,0 são percentuais; a separação é que vale 84,9 pontos percentuais.

**Evidência.** equacoes_por_fatia.json: pct_com_ambiente_de_equacao = 84.9 (RedPajama) e 0.0 (peS2o): são porcentagens de documentos.

**Correção proposta.** "com 84,9% contra 0,0% dos documentos, uma separação de 84,9 pontos percentuais".

### L202 · menor · inconsistência interna · aplicado

> Um indicador que dispara com maior frequência no corpus íntegro do que no degradado

**Problema.** A fatia que serve de polo 'íntegro' na validação do discriminante é, dois parágrafos depois, a fatia que perde um sexto das expressões. Os dois sentidos — notação preservada no que existe, e completude frente ao fonte — não são distinguidos no texto.

**Evidência.** Linha 202: 'em ambos os casos com notação íntegra à inspeção' e 'corpus íntegro' para a fatia RedPajama; linha 169: 'LaTeX íntegro disponível'. Linha 204 e s3b_latex.json: a mesma fatia tem déficit de 16,6% e 25,7% de expressões sem correspondente no fonte; o veredito do experimento é 'RedPajama DEGRADA'.

**Correção proposta.** Distinguir os termos: usar "com marcação LaTeX preservada" para a fatia RedPajama nas linhas 169 e 202, e abrir a linha 204 com "Marcação preservada não implica completude:".

### L204 · menor · sem evidência versionada · pendente

> Uma medição anterior, restrita a 103 artigos, produzia intervalo [9,9%; 19,1%]

**Problema.** O intervalo da medição anterior não tem artefato versionado; não é possível conferi-lo nem saber a estimativa pontual correspondente.

**Evidência.** s3b_run.log, s3b_run2.log e s3b_run3.log registram execuções com 199, 197 e 298 artigos; nenhuma com 103. O intervalo [9,9%; 19,1%] só aparece em ESTADO.md (linha 3709) e o '103' em comentário de latex_audit.py (linha 72). s3b_latex.json foi sobrescrito pela execução final.

**Correção proposta.** Versionar o log daquela execução, ou marcar: "(execução não preservada; valor do registro do projeto)".

### L204 · menor · desatualizado após a v0.5 · aplicado

> que cruza o limiar de decisão de 10%

**Problema.** O limiar aparece sem definição: é o critério, fixado antes, para comprar o fonte LaTeX do arXiv. A regra mandava comprar, o projeto não comprou e usou a fatia assim mesmo, e depois da v0.5 o encoder a que a decisão servia deixou de ser treinado. O leitor recebe um limiar sem decisão, e uma medição cuja consequência o artigo não tira.

**Evidência.** s3b_latex.json, 'veredito': 'RedPajama DEGRADA — ... inteiramente acima do limiar de 10%; o bulk pago do arXiv (US$ 100–180) se justifica'. docs/adr/ADR-0002 (2026-08-31): decide NÃO comprar e usar o RedPajama. docs/adr/ADR-0003, aceito em 2026-09-23 12:03 (commit ea60df2, depois da v0.5 das 02:04): 'O ΦEnc próprio não é treinado'. O rascunho não diz em nenhum ponto que decisão o limiar de 10% governa (grep: a expressão só ocorre na linha 204).

**Correção proposta.** "O limiar de 10% havia sido fixado de antemão como critério para adquirir o fonte LaTeX diretamente do arXiv. O déficit medido o excede; a aquisição não foi feita, e a fatia foi usada como está nos pré-treinamentos da Seção 4.9. Como o encoder próprio não é treinado (Seção 4.9), a aquisição deixou de ser necessária."

### Conferido e correto

- Linha 80: 300.000 positivos e 190.210 negativos, quatro domínios (cs, econ, math, q_bio), cota estratificada de 75.000 — isphysics_com_math.log ('is_physics: 300,000 física · 190,210 não-física', 'negativos por domínio (cota 75,000): cs, econ, math, q_bio')
- Linha 80: classificador linear sobre representação esparsa com perda modified_huber — src/phifm/corpus/filter/classifier.py, make_pipeline (TF-IDF 1–2-gramas + SGDClassifier(loss='modified_huber'))
- Linha 173: acurácia 0,954 — isphysics_com_math.log (accuracy 0.954, 73.531 documentos de validação = 15% de 490.210)
- Tabela 3, todas as células numéricas das quatro primeiras linhas — transferencia.json: cs 0,0371/0,0979/0,907; econ 0,0326/0,0847/0,988; math 0,0242/0,3539/0,731; q_bio 0,0306/0,3121/0,907; razões recomputadas 2,64; 2,60; 14,65; 10,21
- Tabela 3, linha Estatística: 3,0% / 2,9% / razão 1,0 — transferencia_stat.json (0,0303; 0,0292; razão 0,96)
- Linha 185: limiar 0,999 reduz o FP em matemática omitida a 10,0% e a curva achata (11,7% a 0,95; 10,3% a 0,99; 10,0% a 0,999) — transferencia.json
- Linha 187: estatística excluída do treinamento final — isphysics_com_math.log lista só cs, econ, math, q_bio; a razão 1,0 confere
- Linha 84 (apoio): a 0,9 de limiar os FPs externos são 1,5% (cs), 1,1% (econ), 13,6% (math), 12,0% (q_bio) — transferencia.json
- Tabela 4, todas as 24 células — equacoes_por_fatia.json (1.123/21,8/26,9/0,0/0,9; 49.212/100,0/99,6/84,9/1.158,3; 16.009/78,5/85,2/5,1/57,7; 28.605/16,3/18,2/0,0/1,0), n = 3.000 por fatia, semente 17
- Linha 202: 81,3% (RedPajama) e 3,0% (referência) de documentos com operador órfão — equacoes_por_fatia.json; e o RedPajama tem de fato mais órfãos que o peS2o (81,3% contra 50,7%; 21,52 contra 3,14 por documento; 0,44 contra 0,11 por mil caracteres), de modo que a conclusão 'inutilizável' se mantém mesmo normalizada por comprimento
- Linha 202: o mecanismo `$…$ = ` — o regex ORFAO exige [\w\)\]] antes do operador; reproduzido em resumos do arXiv (ex.: '$V$ = 8.9', '$\sqrt{s}$ = 100 TeV')
- Linha 202: o contraste 84,9% contra 0,0% não é artefato de comprimento — resumos íntegros têm 0,10% com ambiente a 1,05 mil caracteres, o que daria ~3% a 28,6 mil; 0 em 3.000 no peS2o fica abaixo disso
- Linha 204: 298 artigos, 16,6%, IC [12,9%; 20,8%] — s3b_latex.json e s3b_run3.log; recomputei 9.255/55.780 = 0,1659 e o bootstrap (semente 17, 20.000 reamostras) devolve [0,1294; 0,2084]
- Linha 204: a amostra é restrita a Física pelo índice — s3b_run3.log ('388 de 900 são Física pelo spine'); os 298 identificadores constam todos de data/processed/spine.parquet
- Linha 204: o bootstrap reamostra artigos, não equações — latex_audit.py, _bootstrap_ausencia
- Linhas 167/169 (apoio): partições do RedPajama com 20.000 documentos e ~1 G de caracteres — conta_rp.log; sonda_dominios.json confere a extrapolação de math+cs (685.200 documentos, 44,6 G caracteres)
- Nenhum trecho das linhas 76–81, 171–188 e 189–205 cita resultado de recuperação alterado por T1h, T1i ou pelo commit 62e42fc; o único efeito pós-v0.5 nessas seções é o do ADR-0003 sobre a decisão ligada ao limiar de 10% (achado na linha 204)

### Não verificável nesta máquina

- Tabela 4 a partir dos dados: exige data/processed/redpajama_fisica, openwebmath_fisica, pes2o_fisica e pares/pares_validacao.parquet, ausentes desta máquina; conferi apenas contra equacoes_por_fatia.json e reproduzi a linha de referência com resumos de data/raw/arxiv_metadata (2,8% de órfãos contra 3,0%)
- Trechos citados na linha 200 ('if:'/'where', `p *`, `T eff`): exigem o peS2o filtrado; só há a transcrição em ESTADO.md, e `T eff` não aparece nela
- Acurácia 0,954 e Tabela 3 por reexecução: exigem data/raw/arxiv_negativos (cs, econ, math, q_bio, stat) e o modelo em models/isphysics-clf; conferi somente logs e JSON
- FP por domínio do classificador final (base do achado da linha 173): exigiria aplicar o modelo final a negativos de validação separados por domínio
- Os 72.919 negativos com listagem cruzada (linha 78): exigem data/raw/arxiv_negativos; e a mesma conferência para matemática não está registrada
- Acurácia 0,830 do treino restrito a primária math (linha 187): sem log, JSON ou script; exigiria os negativos de math e a rotina usada
- Intervalo [9,9%; 19,1%] da medição com 103 artigos (linha 204): a execução não foi preservada
- Confusor de versão em definitivo: exige o cache data/raw/arxiv_fontes (tarballs e redpajama_shard0_*.jsonl) e o histórico de versões de cada artigo no arXiv, além da data exata do instantâneo do RedPajama; usei `created`/`updated` do spine como aproximação
- Os 97% de macros no arXiv 1607.04520: exigem o tarball do artigo e o texto dele no RedPajama
- Fração de negativos de matemática com probabilidade igual a 1,0 (explicação por saturação, linha 185): exige os escores do modelo de exclusão de domínio
- Se o `.head()` do polars leu de fato só a primeira partição sorteada no peS2o e no RedPajama: inferido do código e dos tamanhos de partição em conta_rp.log; confirmar exige as fatias


## Tokenização (§3.4, §4.4, §4.9.1)

### L221 · crítico · afirma mais que a evidência · aplicado

> as variantes A e E diferem apenas por elas, e A produz 27% menos tokens em prosa

**Problema.** Os 27% 'em prosa' não são efeito das regras de LaTeX: são, quase inteiramente, efeito de A não ter fronteira de palavra. A afirmação 'as regras propostas são a variável de maior efeito' atribui às regras de LaTeX um ganho que vem de outra diferença de pré-tokenização. Pelo mesmo motivo, 'E é a única variante que não atinge a meta' reflete que E é a única variante própria com fronteira de palavra (a referência F também tem). A coluna 'Regras' da Tabela 5 (l. 208-217) esconde essa segunda diferença.

**Evidência.** Mesmo código da entrada anterior (scripts/bakeoff_tokenizer.py l. 140-152). Medição em 5.259 resumos do spine: só 0,82% das palavras contêm um casamento das regras de LaTeX (0,011 casamento por palavra; 5,7% dos bytes estão em $...$), de modo que tornar essas sequências atômicas pode economizar, no máximo, poucos centésimos de token por palavra — contra a diferença observada de 0,36 token por palavra (1,3245 − 0,9620). O piso de qualquer tokenizador com fronteira de palavra e dígito isolado é 1,290 token/palavra nessa amostra; A, B, C e D (0,90 a 1,00) estão abaixo do piso, o que só a fusão através de espaços explica. Coerente com isso, a vantagem de A cai de 37,7% em resumos (quase sem LaTeX) para 12,6% no fonte LaTeX (l. 229) — o contrário do que se esperaria se as regras de LaTeX fossem a causa.

**Correção proposta.** Substituir por: "Primeiro, a pré-tokenização é a variável de maior efeito: A produz 27% menos tokens por palavra e 23% menos por trecho matemático que E. As duas diferem, porém, em mais que as regras de LaTeX — nas variantes A a D não há fronteira de palavra, e o BPE funde através de espaços, o que explica fertilidades abaixo de um token por palavra; em resumos, onde menos de 1% das palavras contém sequência de controle, é essa segunda diferença que responde pela quase totalidade do ganho. E é a única variante que não atinge a meta de fertilidade (0,80 vez a referência)." Na legenda da Tabela 5, trocar a definição de 'Regras' por: "'Pré-tok.' indica a pré-tokenização proposta (sequências de LaTeX isoladas, sem fronteira de palavra) ou a genérica (por palavra e pontuação)."

### L407 · crítico · afirma mais que a evidência · aplicado

> diferem apenas nas regras de pré-tokenização

**Problema.** A e E diferem em DUAS variáveis: (i) as regras de LaTeX e (ii) a presença de fronteira de palavra/pontuação na pré-tokenização. Em A o BPE funde através de espaços (tokens multi-palavra); em E não pode. O experimento da Seção 4.9.1 testa o pacote 'regras de LaTeX + ausência de fronteira de palavra' contra 'regex genérico', e não isola as regras de LaTeX. A frase 'É a única diferença' e a conclusão que dela depende (l. 417, l. 229, resumo l. 13) não se sustentam como escritas. O mecanismo alternativo é plausível e não foi excluído: tokens multi-palavra são linhas raras da embedding num modelo de 48 M. Também é impreciso 'unidades atômicas': a regra garante fronteira de pré-token, não token único (Tabela 5: A tem 2/3, não 3/3).

**Evidência.** scripts/bakeoff_tokenizer.py, função construir(): variantes com regras (A, B, C, D) usam Split(PRE_TOK_LATEX) + Split(dígito) + ByteLevel(use_regex=False) (l. 140-146), isto é, NENHUMA divisão por espaço ou pontuação; a variante E usa Split(dígito) + ByteLevel(use_regex=True) (l. 151-152), o regex do GPT-2, que corta em palavra e pontuação. O comentário do próprio código (l. 148-150) admite: 'precisa de ALGUMA segmentação geral, senão o BPE recebe o documento inteiro como um pré-token' — que é exatamente o que A recebe entre dois comandos LaTeX ou dígitos. Os mesmos arquivos variante_A.json / variante_E.json são os usados no treino do T2a (kaggle/t2a_tokenizer.py l. 361; src/phifm/core/kaggle.py l. 475-477). Confirmação pelos números do próprio artigo: fertilidade de A = 0,9620 token por palavra \S+ (<1) e 7,074 bytes/token contra 6,81 bytes/palavra — só é possível com tokens que atravessam espaços. Medição minha em 5.259 resumos do spine (gather_every(300), mesma construção título+resumo): piso de pré-tokens por palavra sob a pré-tokenização de E = 1,290 (E mede 1,3245, 2,7% acima do piso); pré-tokens por palavra sob a de A = 0,111; 25% dos pré-tokens de A têm 2 ou mais palavras.

**Correção proposta.** Substituir por: "As variantes A e E da Tabela 5 têm o mesmo algoritmo, o mesmo tamanho de vocabulário, a mesma arquitetura e a mesma contagem de parâmetros, e diferem na pré-tokenização em dois pontos que este experimento não separa: A isola `\frac`, `\begin{…}`, `^{` e `_{` como pré-tokens e não impõe fronteira de palavra, de modo que o BPE pode fundir através de espaços; E usa a segmentação genérica por palavra e pontuação e não isola as sequências de LaTeX. O que se compara é, portanto, a pré-tokenização proposta como um todo contra a genérica." Registrar em Limitações que a ablação que isola as regras de LaTeX (regex genérico + regras, contra regex genérico) não foi executada.

### L225 · maior · número não confere · aplicado

> a matriz de embedding corresponde a 16% dos parâmetros do modelo de 150 M com vocabulário de 40.960, e a cerca de 24% com 65.536

**Problema.** Os dois percentuais estão errados e o próprio código do projeto já registra a correção. O erro é contra o argumento do artigo no sentido favorável (o compromisso é mais duro do que o texto diz), mas é número errado, e é inconsistente com os 43,7% da l. 98, que são calculados sobre o total real.

**Evidência.** PYTHONPATH=src .venv/bin/python: PHIENC_150M.parametros() = embedding 31.458.048, total 142.420.480, fração 0,2209; com vocab=65.536: embedding 50,33 M, total 161.319.424, fração 0,3120. A docstring de src/phifm/models/encoder/config.py (l. 19-37) diz literalmente: 'Os 16% do DOC-07 §2.2 estão errados ... O real é 22,1% → 31,2%' e explica a origem do 16% (V=32.000 dividido por 150 M nominais). Dividindo por 150 M nominais daria 21,0% e 33,6% (conta do auditor) — tampouco 16%/24%.

**Correção proposta.** "a matriz de embedding corresponde a 22,1% dos parâmetros do encoder planejado (31,5 M de 142,4 M) com vocabulário de 40.960, e a 31,2% (50,3 M de 161,3 M) com 65.536, isto é, 18,9 M de parâmetros a mais, todos em embedding".

### L225 · maior · inconsistência interna · aplicado

> o que eles justificam é a inclusão do vocabulário maior na avaliação extrínseca

**Problema.** A frase promete uma avaliação extrínseca do vocabulário de 65.536 que o artigo não entrega e que, depois do ADR-0003, não será feita. O leitor chega à Seção 4.9.1 esperando o braço D e encontra a explicação de por que ele foi excluído.

**Evidência.** A avaliação extrínseca (Seção 4.9.1, l. 407) treinou apenas A e E; docs/01-data/DOC-05-tokenizer.md l. 463: 'as seis variantes → só A e E'. Não há braço D em data/processed/avaliacao/ (só t2a_*AxE*). ADR-0003 (aceito em 2026-09-23, §10): 'o ΦEnc próprio não é treinado, por nenhum dos dois caminhos'.

**Correção proposta.** "Nenhum número desta seção decide a escolha. A comparação extrínseca entre tamanhos de vocabulário não foi executada: a 48 M ela carregaria um confundidor de capacidade (Seção 4.9.1), e o encoder próprio não foi treinado; a questão fica em aberto."

### L407 · maior · inconsistência interna · aplicado

> É o único par de variantes sem confundidor de capacidade

**Problema.** Os pares A×B e B×E também não têm confundidor de capacidade. A×E não é o único par limpo quanto à capacidade; é o par escolhido para a pergunta da pré-tokenização. Além disso 'compartilham vocabulário' deveria ser 'tamanho de vocabulário': os vocabulários são diferentes.

**Evidência.** Tabela 5 do próprio artigo (l. 212-216): A, B e E têm todos vocabulário de 40.960; kaggle/t2a_tokenizer.py l. 17: 'C tem 32.768, A/B/E têm 40.960, D tem 65.536'. Com o mesmo V e a mesma arquitetura, a contagem de parâmetros é idêntica (PROXY_BAKEOFF: 48.026.624).

**Correção proposta.** "Das três variantes de mesmo tamanho de vocabulário (A, B e E), e portanto de mesma contagem de parâmetros, treinou-se o par A×E, que responde pela pré-tokenização; o par A×B, que isolaria o algoritmo, não foi treinado. As variantes C e D ficam fora porque, a 48 M, com a matriz de embedding a 43,7% dos parâmetros, vocabulários de tamanhos diferentes dão contagens diferentes."

### L417 · maior · afirma mais que a evidência · aplicado

> rejeita as regras de pré-tokenização a esta escala

**Problema.** O resultado medido rejeita a pré-tokenização proposta como pacote, a 48 M e 0,6 B de tokens; não permite dizer qual dos dois componentes custa. A leitura 'a diferença concentra-se em LaTeX longo sem espaços' também fica em aberto: unidades comuns longas podem ser trechos de prosa em que os tokens de A atravessam espaços e por isso não há corte comum.

**Evidência.** data/processed/avaliacao/t2a_bits_por_byte_AxE_l2r.json: A 0,89688, E 0,85004, diferença +0,046613 [0,043213; 0,050061] — o número confere. O que não confere é o objeto rejeitado: scripts/bakeoff_tokenizer.py l. 140-152 mostra que o braço A traz as regras de LaTeX E a ausência de fronteira de palavra (ver entrada da l. 407). Além disso, os 6%/53% citados na mesma linha vêm de 200 documentos (docstring de src/phifm/eval/bits_por_byte.py l. 84-87), sem artefato versionado em data/processed/avaliacao/.

**Correção proposta.** Trocar por: "Pela regra registrada, E à frente no instrumento 3, apesar do resíduo a favor de A, rejeita a esta escala a pré-tokenização proposta — o conjunto das regras de LaTeX e da ausência de fronteira de palavra, que o experimento não separa." E trocar a última frase por: "Em 200 documentos, unidades de mais de 8 tokens são 6% das unidades e 53% dos bytes; não foi medido quanto delas é LaTeX sem espaços e quanto é prosa em que os tokens de A atravessam espaços." Ajustar no mesmo sentido a frase do Resumo (l. 13) e a l. 229.

### L96 · menor · desatualizado após a v0.5 · aplicado

> A arquitetura planejada é bidirecional, de 150 milhões de parâmetros

**Problema.** O parágrafo descreve no presente um modelo cujo treinamento foi formalmente descartado depois da v0.5 (o artigo só o diz na l. 613). E '150 milhões' é o nome nominal: a configuração dá 142,4 M, valor que importa para os percentuais de embedding da l. 225.

**Evidência.** docs/adr/ADR-0003-phienc-do-zero-ou-cpt.md, Status: 'Aceito (2026-09-23) ... o ΦEnc próprio não é treinado, por nenhum dos dois caminhos'; commit ea60df2. Contagem real da configuração: PHIENC_150M.parametros()['total'] = 142.420.480 (src/phifm/models/encoder/config.py; 22 camadas, d = 768, ffn 1.152, V = 40.960).

**Correção proposta.** "**Encoder pré-treinado a partir do zero.** A arquitetura que havia sido planejada, e que não foi treinada (Seção 5.5), é bidirecional, de 142 milhões de parâmetros (nominalmente 150 M), com vocabulário de 40.960, …"

### L102 · menor · desatualizado após a v0.5 · aplicado

> duas bases gerais sem pré-treinamento no domínio

**Problema.** Depois da v0.5 foram ajustadas outras quatro bases gerais pequenas com os mesmos 200 mil pares, e duas delas com 1 milhão. A Seção 3.4 (l. 90 e 102) lista só o ModernBERT-base e o GTE-base e não descreve esses braços, que fecham a pergunta da base.

**Evidência.** data/processed/avaliacao/t1h_comparacao.json (200 mil pares, n = 2.000): MiniLM-L6 0,5289; gte-small 0,5706 (+0,0417 [0,028; 0,055], IC 98,75%); bge-small 0,5692; e5-small 0,5402; MiniLM-L12 0,5404; custo 1,75-1,76×. data/processed/avaliacao/t1i_comparacao.json (1 M de pares): gte-small 0,6014 e bge-small 0,6012 contra 0,6223 do sistema (−0,021 [−0,032; −0,010], IC 97,5%). Commits 0f1d08f, dc7755d, 62e42fc ('a busca por encoder ENCERRADA').

**Correção proposta.** Acrescentar ao fim do parágrafo: "Quatro bases gerais do porte do MiniLM-L6 (gte-small, bge-small, e5-small e MiniLM-L12, 33 M) foram ajustadas com os mesmos 200 mil pares, e as duas melhores, com 1 milhão de pares, contra o recuperador do sistema." — e reportar os resultados (T1h, T1i) na Seção 4.6.

### L208 · menor · falta controle ou limitação · pendente

> relativas ao tokenizador de referência (variante F)

**Problema.** O artigo nunca diz qual é o tokenizador de referência contra o qual todas as razões da Tabela 5 e as duas metas (0,80 e 0,65) são calculadas. Sem o nome e a revisão, a tabela não é reproduzível.

**Evidência.** grep -i qwen no rascunho: nenhuma ocorrência. scripts/bakeoff_tokenizer.py l. 293-303: o controle F é o tokenizador do Qwen ('Qwen/Qwen3-8B', com queda para 'Qwen/Qwen2.5-7B' se o primeiro não carregar); vocab_size 151.643. O arquivo bakeoff.json, que registra qual dos dois foi de fato carregado, não está na máquina.

**Correção proposta.** Na legenda: "…relativas ao tokenizador de referência (variante F), o do Qwen3 [ref.], BPE de uso geral com 151.643 entradas, sem modificação." Conferir no bakeoff.json o campo `modelo` e acrescentar a referência à lista.

### L219 · menor · clareza · pendente

> mantêm razão de fertilidade abaixo de 1,25 nas 35 subáreas medidas

**Problema.** 'Razão de fertilidade' aqui é a razão entre a pior subárea e a mediana das subáreas da mesma variante, e não a coluna 'Razão' da Tabela 5 (relativa à referência F, onde F vale 1,000 e E 0,929). Com o mesmo nome na frase seguinte à tabela, o leitor confunde as duas.

**Evidência.** scripts/bakeoff_tokenizer.py l. 268-274: pior_subarea_razao = maior mediana de fertilidade por subárea (com 30 ou mais documentos) dividida pela mediana das medianas, dentro de cada variante. DOC-05 §11.1-medido: '35 subáreas medidas, pior razão 1,215 contra o limite de 1,25'.

**Correção proposta.** "Todas as variantes preservam o texto na decodificação, e em nenhuma a subárea de pior fertilidade passa de 1,25 vez a mediana das 35 subáreas com ao menos 30 documentos (pior caso, 1,215)."

### L221 · menor · clareza · aplicado

> E é a única variante que não atinge a meta de fertilidade em prosa

**Problema.** O artigo chama de 'em prosa' (l. 221 e 223) uma medida sobre o documento inteiro, com a matemática dentro, e 'palavra' é qualquer sequência sem espaço. A contaminação é pequena em resumos, mas a margem de 2% entre BPE e unigramas 'em prosa' (l. 223) é da mesma ordem. E a meta citada não tem valor declarado.

**Evidência.** scripts/bakeoff_tokenizer.py l. 92 e 236-239, 279: fertilidade = total de tokens / total de casamentos de \S+ sobre o documento inteiro (título + resumo), incluindo os trechos matemáticos. Na amostra de 5.259 resumos, 5,7% dos bytes estão em $...$ e 28,5% dos documentos têm ao menos um casamento das regras. A meta (0,80 vez a referência) está em docs/01-data/DOC-05-tokenizer.md §13 (E1) e não aparece no artigo; a de equações (0,65) aparece na l. 227.

**Correção proposta.** Na legenda da Tabela 5: "'Fertilidade' é a razão entre tokens e palavras (sequências sem espaço) no documento inteiro, matemática incluída; 'Tokens/eq.' é a média de tokens por trecho entre cifrões." No texto, trocar 'em prosa' por 'no documento inteiro' e escrever "a meta de fertilidade, fixada em 0,80 vez o tokenizador de referência".

### L223 · menor · afirma mais que a evidência · aplicado

> A poda iterativa do primeiro remove essas unidades; a fusão incremental do segundo as preserva.

**Problema.** O que foi observado é o desfecho em três cadeias; o mecanismo (poda contra fusão) é inferência não testada, apresentada no indicativo após 'o mecanismo é observável'. Também não está excluída uma diferença de configuração entre os treinadores (o de unigramas tem limite de comprimento de peça; o de BPE, não — relevante porque nas variantes A a D os pré-tokens abrangem várias palavras).

**Evidência.** scripts/bakeoff_tokenizer.py l. 173-177 e 214-226: o único dado é cobertura_1_token sobre três cadeias de teste (\frac, \partial, \begin{equation}): B 0/3, A 2/3. Não há medição de que essas peças estavam no vocabulário-semente do modelo de unigramas e foram podadas. As margens conferem: 1 − 0,9620/0,9816 = 2,0%; 1 − 7,31/8,36 = 12,6%; razão 6,3.

**Correção proposta.** "Um indício do mecanismo: o modelo de unigramas não reteve nenhuma das três sequências LaTeX de teste como token único, e a codificação por pares de bytes reteve duas. É compatível com a poda iterativa do primeiro remover essas unidades e a fusão incremental do segundo preservá-las, hipótese que três sequências não testam."

### L225 · menor · afirma mais que a evidência · aplicado

> a ordenação é monotônica em toda a faixa testada

**Problema.** A monotonia é propriedade do algoritmo, não achado empírico: o resultado não poderia ter sido outro, e 'o vocabulário de 40.960 não é ótimo pela métrica intrínseca' vale para qualquer tamanho finito. É apresentada como o terceiro de 'três resultados que merecem registro'. O que é empírico é a magnitude (6,8% menos tokens de 40.960 para 65.536; 3,7% a mais em 32.768).

**Evidência.** scripts/bakeoff_tokenizer.py: C, A e D são BPE treinados no mesmo corpus (semente 17), com a mesma pré-tokenização e o mesmo BpeTrainer, diferindo só em vocab_size (VARIANTES, l. 80-86). As fusões do vocabulário menor são prefixo das do maior, e um token criado por fusão posterior só participa de fusões ainda posteriores; logo o número de tokens de qualquer texto é não crescente em V.

**Correção proposta.** "Terceiro, a métrica intrínseca não pode escolher o tamanho do vocabulário: em BPE treinado sobre o mesmo corpus as fusões do vocabulário menor são prefixo das do maior, e a fertilidade decresce com o vocabulário por construção. O que a medição acrescenta é a magnitude — 6,8% menos tokens de 40.960 para 65.536."

### L229 · menor · sem evidência versionada · pendente

> 12,6% a mais, e não os 37,7%

**Problema.** O número do artigo é coerente com a célula do experimento, mas não é verificável nos artefatos versionados e diverge do que três documentos do repositório afirmam (13,6%). Convém dizer sobre que fatia foi medido e versionar a contagem.

**Evidência.** Os 15.673 e 13.916 tokens por documento só existem na docstring de kaggle/t2a_tokenizer.py l. 34-35 (A: 2.001.270.262 tokens de 143.810 documentos; E: 2.003.418.640 de 127.828); a aritmética confere (15.672,8/13.916,1 = 1,1262; 143.810/127.828 = 1,1250; 1,3245/0,9620 = 1,3768). Nenhum arquivo de data/processed/avaliacao/ sustenta os valores, as fatias não estão na máquina, e DOC-05 (l. 276, 390, 450), README.md l. 227 e ESTADO.md l. 57 dizem 13,6% (medição de 2026-09-11 a 0,9 B). A única razão de tokens versionada é a dos arquivos t2a_bits_por_byte_*: 19,5% a 19,7% sobre bytes idênticos, no prefixo dos documentos. As duas contagens por documento são, além disso, sobre conjuntos de documentos diferentes (143.810 contra 127.828).

**Correção proposta.** "Nas fatias de treinamento, de fonte LaTeX integral e 2,0 bilhões de tokens cada, a variante E gasta 15.673 tokens por documento (127.828 documentos) contra 13.916 da A (143.810) — 12,6% a mais, e não os 37,7% …"; versionar o manifesto das duas fatias em data/processed/avaliacao/ e alinhar DOC-05, README e ESTADO, que ainda dizem 13,6%.

### L229 · menor · afirma mais que a evidência · aplicado

> o modelo treinado sem as regras é o melhor

**Problema.** 'É o melhor' sem qualificação generaliza uma medida (bits por byte, instrumento 3) a que as secundárias não acompanham, como a própria Seção 4.9.1 reconhece. A frase da Seção 4.4 é mais forte que a da seção que ela cita.

**Evidência.** t2a_bits_por_byte_AxE_l2r.json: E à frente por 0,047 bit/byte; t2a_secundarias.json: A à frente em nDCG@10 (0,0296 contra 0,0171) e recall@1 (p = 0,0039); sonda sem diferença (p = 0,377). Uma semente por braço, 48 M, 0,6 B de tokens, e um retorno a ponto de verificação só no braço A (l. 419).

**Correção proposta.** "E a avaliação extrínseca, na Seção 4.9.1, inverte o sinal na medida primária: com o mesmo orçamento de tokens, o modelo treinado com a pré-tokenização genérica reconstrói o texto com 0,047 bit por byte a menos."

### L409 · menor · falta controle ou limitação · pendente

> Bits por byte (menor é melhor) por três instrumentos

**Problema.** A medida primária cobre apenas a janela inicial de cada documento (título, resumo, introdução), onde equações em display são mais raras que no corpo. Nem a Seção 3.8 nem a 4.9.1 dizem isso, e a generalização para 'modelagem do texto de Física' depende disso.

**Evidência.** scripts/avaliar_bits_por_byte.py, janela_comum() (l. 154-173): mede 'o maior prefixo de texto que cabe em contexto tokens em TODOS' os tokenizadores. Nos JSON: 1.175.305 bytes escondidos em 2.000 documentos a 15% ⇒ ~3,9 kB medidos por documento, contra 49.212 caracteres por documento na fatia (Tabela 4, l. 196) — cerca de 8% de cada documento, sempre o início. Nesse trecho E usa 19,7% mais tokens que A sobre bytes idênticos (307.304 contra 256.735 em t2a_bits_por_byte_AxE_unidade.json).

**Correção proposta.** Acrescentar à legenda: "Cada documento é medido no maior prefixo que cabe em 1.024 tokens nos dois tokenizadores — em média cerca de 3,9 kB, ou 8% do documento —, de modo que o corpo dos artigos, onde se concentram as equações em display, não é medido."

### L414 · menor · erro estatístico · aplicado

> | 1,3096 | 1,3680 | −0,051 [−0,056; −0,046] |

**Problema.** As colunas A e E e a coluna A − E estimam grandezas diferentes (razão de somas contra média de razões por documento); na linha 2 a diferença das colunas cai fora do IC de 95% da própria linha. O sinal não muda e nenhuma conclusão depende disso, mas a tabela é inconsistente à leitura.

**Evidência.** data/processed/avaliacao/t2a_bits_por_byte_AxE_unidade.json: A 1,30962, E 1,36795 (agregados sobre todos os bytes), diferenca_media −0,050878, ic95 [−0,056045; −0,045846]. 1,30962 − 1,36795 = −0,0583, fora do intervalo impresso. Causa em src/phifm/eval/bits_por_byte.py: como_dict() agrega bits/bytes sobre todos os documentos (l. 217-224), e bootstrap_pareado() usa a média não ponderada das diferenças por documento (l. 262-267). Instrumento 1: 0,0775 contra +0,078; instrumento 3: 0,0468 contra +0,047. Além disso, o instrumento 3 tem n = 1.000 e o 2, n = 2.000 (l. 417 compara as quedas −0,52 e −0,41 entre conjuntos diferentes; o valor do instrumento 2 nos mesmos 1.000 documentos não está versionado).

**Correção proposta.** Acrescentar à legenda da Tabela 17: "As colunas A e E são agregadas sobre todos os bytes escondidos; a coluna A − E é a média das diferenças por documento, com intervalo por bootstrap pareado (10.000 reamostras), e por isso não coincide com a subtração das duas colunas. O instrumento 3 foi medido nos primeiros 1.000 dos 2.000 documentos." Na l. 417, qualificar: "a queda do instrumento 2 (2.000 documentos) para o 3 (1.000)".

### L419 · menor · falta controle ou limitação · pendente

> Na recuperação sem ajuste, A fica à frente

**Problema.** A truncagem em 192 tokens corta boa parte dos textos no braço E e quase nenhum no braço A: A lê mais texto por documento. A vantagem de A na recuperação tem uma explicação que não é a qualidade da representação e que o texto não nomeia, embora nomeie a assimetria de execução que favorece E. Não foi possível medir aqui a fração truncada por braço (tokenizadores ausentes).

**Evidência.** data/processed/avaliacao/t2a_secundarias.json: protocolo max_tokens = 192 nos dois braços; os números conferem (nDCG@10 0,0296 / 0,0171; −0,01251 [−0,02005; −0,0052]; McNemar 27×9, p = 0,0039). src/phifm/eval/encoders.py l. 167-168 trunca por TOKENS do tokenizador de cada braço, e a l. 17 do mesmo módulo registra: 'Mesmo max_tokens | quem lê mais contexto leva vantagem de graça'. Com as fertilidades da Tabela 5 e ~159 palavras por título+resumo (amostra de 5.259 do spine), E gasta em média ~210 tokens por texto (acima de 192) e A ~153 (abaixo).

**Correção proposta.** Acrescentar após o parêntese: "A comparação trunca os dois braços em 192 tokens, e não em texto: A, que gasta menos tokens por palavra, lê mais de cada resumo, de modo que parte dessa vantagem pode ser de cobertura e não de representação; a fração de textos truncados em cada braço não foi medida."

### Conferido e correto

- Pista (1) REFUTADA no rascunho: o arquivo não contém nenhum caractere de controle (form feed 0x0C, backspace 0x08, tab etc.; contagem por Python sobre os bytes = 0) e a l. 407 traz `\frac`, `\begin{…}`, `^{` e `_{` íntegros. Todas as 16 contrabarras do arquivo estão nas l. 136, 202, 407 e 499 e estão corretas (\mu, \nu, \alpha, \beta, \rm, \frac, \begin, \nonumber, \le, \leq, \gamma). O defeito existia em ESTADO.md (3 ocorrências) e em src/phifm/eval/mlm_regiao.py, corrigido no commit df41c82, cuja mensagem diz 'O rascunho do artigo esta limpo'.
- L. 98 e 407: 43,7% de embedding a 48 M está CORRETO — PROXY_BAKEOFF (12 camadas, d = 512, ffn 768, V = 40.960): embedding 20.972.032 de 48.026.624 = 43,67%.
- Tabela 17, três linhas, conferem com data/processed/avaliacao/t2a_bits_por_byte_AxE.json (0,66534 / 0,58780; +0,078217 [0,073759; 0,082603]; 2.000 docs), _unidade.json (1,30962 / 1,36795; −0,050878 [−0,056045; −0,045846]; 2.000 docs) e _l2r.json (0,89688 / 0,85004; +0,046613 [0,043213; 0,050061]; 1.000 docs).
- Pista (5): a piora de 0,047 bit/byte é do terceiro de três instrumentos (PLL da esquerda para a direita, t2a_bits_por_byte_AxE_l2r.json, vencedor E); os instrumentos 1 e 2 têm sinais opostos com IC que não cruzam zero; o 3 cai entre −0,051 e +0,078.
- L. 417: quedas do instrumento 2 para o 3 — A: 1,30962 − 0,89688 = 0,413 (−0,41); E: 1,36795 − 0,85004 = 0,518 (−0,52).
- L. 409: bootstrap pareado por documento confere com src/phifm/eval/bits_por_byte.py (reamostra documentos, mesmos índices nos dois braços, 10.000 reamostras, semente 17); bytes escondidos idênticos nos dois braços nos instrumentos 2 e 3 (1.181.775 e 588.170).
- L. 419: secundárias conferem com t2a_secundarias.json e t2a_sonda_tensorial.json — nDCG@10 0,0296 / 0,0171; E − A = −0,01251 [−0,02005; −0,0052]; McNemar em recall@1 27×9, p = 0,0039; sonda 0,4167 / 0,3333, 19×13 discordantes, p = 0,3771.
- L. 419: 'cerca de 3,3% do orçamento' é coerente com 299 lotes descartados de 9.155 passos (ESTADO.md l. 1315; kaggle/t2a_tokenizer.py l. 345: 1 spike no passo 3.798, 1 rollback); 0,6 B / (64 × 1.024) = 9.155 passos, coerente com o lote e o contexto da l. 98.
- Tabela 5: as colunas de razão são aritmeticamente coerentes com as de valor (0,9620/1,4263 = 0,674; 0,9816 → 0,688; 0,9973 → 0,699; 0,8966 → 0,629; 1,3245 → 0,929; 7,31/9,99 = 0,732; 8,36 → 0,837; 7,49 → 0,750; 6,97 → 0,698; 9,46 → 0,947) e a tabela é idêntica à de docs/01-data/DOC-05-tokenizer.md §11.1-medido.
- L. 221: 27% menos tokens (1 − 0,9620/1,3245 = 27,4%) e 23% menos em equações (1 − 7,31/9,46 = 22,7%) conferem como aritmética; E (0,929) é a única acima da meta de 0,80.
- L. 223: margens BPE contra unigramas — 2,0% no documento e 12,6% em equações, razão 6,3 ('seis vezes') — conferem; LaTeX unitário 2/3 contra 0/3 confere com a tabela.
- L. 227: nenhuma variante atinge 0,65 em equações; melhor valor 0,698 (D). A Tabela 4 (l. 195) sustenta que resumos têm 0,0% de ambiente de equação; 'Tokens/eq.' mede, na prática, só trechos entre cifrões.
- L. 229 e 407: aritmética de 12,6% (15.673/13.916), 12,5% (143.810/127.828 documentos) e 37,7% (1,3245/0,9620) confere com kaggle/t2a_tokenizer.py l. 34-41.
- Pista (4) CONFIRMADA: 'fertilidade' conta o documento inteiro (título + resumo, matemática incluída) por \S+ (scripts/bakeoff_tokenizer.py l. 92, 236-239, 279).
- Pista (3) CONFIRMADA: a monotonia com o vocabulário é garantida por construção para C, A e D (mesmo corpus, mesma pré-tokenização, mesmo treinador BPE) e é apresentada como o terceiro de 'três resultados'.
- Pista (2) CONFIRMADA para a l. 225 (16%/24% errados; real 22,1%/31,2%) e REFUTADA para a l. 98 (43,7% correto).
- Legenda da Tabela 5: 200.000 documentos de treino e 5.000 reservados são os padrões do script (--corpus-docs 200000, --aval-docs 5000), com reserva por documento antes do treino; 'LaTeX unitário' sobre três sequências confere com CASOS_LATEX.
- L. 96: vocabulário 40.960, contexto 8.192, MLM a 30% sem NSP conferem com docs/02-models/DOC-07 §2.2-2.3 e com config.py. L. 98: 0,6 B de tokens a ~20.500 tokens/s dá 8,1 h, coerente com 'pouco mais de 8 horas' (DOC-05: A 8 h 16, E 8 h 08).
- Referências cruzadas das três partes existem e apontam para o conteúdo certo: Seções 3.4, 3.8, 4.3, 4.9.1 e Tabelas 5 e 17; citações [10], [12], [21], [23]-[26], [30], [31] batem com a lista de referências (Bostrom e Durrett 2020 para a expectativa a favor de unigramas; Kauf e Ivanova 2023 para o instrumento 3).

### Não verificável nesta máquina

- Valores brutos da Tabela 5 (fertilidade, tokens/eq., LaTeX unitário, 35 subáreas, pior razão 1,215): data/processed/tokenizer/bakeoff.json não está na máquina e a biblioteca `tokenizers` não está instalada na .venv; só foi possível conferir contra a transcrição em DOC-05 (alegação) e a aritmética. Seria preciso o bakeoff.json ou reexecutar scripts/bakeoff_tokenizer.py.
- Inspeção direta do vocabulário de variante_A.json para contar tokens multi-palavra (entradas com 'Ġ' interno) e medir quanto do ganho de A vem deles: os arquivos variante_*.json não estão na máquina. A conclusão sobre a ausência de fronteira de palavra vem do código, da fertilidade < 1 e do piso medido em 5.259 resumos; a ablação que separa as duas diferenças (regex genérico + regras de LaTeX) exigiria treinar um tokenizador e um encoder.
- 15.673 e 13.916 tokens por documento (e a divergência com os 13,6% de DOC-05/README/ESTADO): as fatias t2a_A e t2a_E e seus manifestos não estão na máquina.
- Fração de textos truncados em 192 tokens por braço na recuperação secundária, e 'unidades de mais de 8 tokens são 6% das unidades e 53% dos bytes' (medido em 200 documentos, sem artefato versionado): exigem os tokenizadores A e E e a fatia redpajama_fisica.
- Qual tokenizador de referência foi de fato carregado como F (Qwen3-8B ou a queda para Qwen2.5-7B): registrado no campo `modelo` do bakeoff.json, ausente.
- Limite de comprimento de peça do treinador de unigramas (padrão da biblioteca, não fixado no script) como possível assimetria contra o BPE sem limite: não verificável sem a biblioteca `tokenizers` e os vocabulários.
- Registros de treino dos braços A e E (vazão de ~20.500 tokens/s, duração, instabilidade no passo 3.798 e retorno ao ponto de verificação): os logs do Kaggle e os checkpoints não estão na máquina; conferido só contra ESTADO.md e comentários de código.
- Demais números da Seção 3.4 fora da tokenização, sem artefato em data/processed/avaliacao/: 6.564.111 pares e 667.304 documentos citados (só constantes em src/phifm/training/amostragem.py), 423 milhões de tokens retokenizados, vazão de 15.200 a 15.900 tokens/s contra 15.600 projetados, teste de paralelismo a menos de 10⁻⁵ (existe tests/regression/test_laco_ddp.py, não executado: exige torch), e os 'seis hiperparâmetros restantes' do cross-encoder.


## Recuperação densa (§3.5, §3.6, §4.5)

### L108 · maior · falta controle ou limitação · aplicado

> avalia dentro de um conjunto de 2.000 pares de validação, sorteados com semente e deduplicados por texto

**Problema.** A remedição corrigiu o teto (empates por texto repetido), não a separação treino/avaliação em nível de documento. Os candidatos do pool de avaliação (e as próprias consultas, como positivos de outras âncoras) foram, com alta probabilidade, vistos pelo bi-encoder ajustado durante o treino; os modelos gerais (GTE-large, PhysBERT) não viram esse acervo. O cenário é o de ajuste no domínio com acervo compartilhado (como no MS MARCO), o que é legítimo para um sistema sobre acervo fixo, mas o artigo não descreve a divisão, não mede a sobreposição e apresenta a vitória sobre o GTE-large sem essa qualificação. A taxa exata (um auditor anterior reportou ~99,9% dos candidatos presentes como positivos no treino) não é verificável nesta máquina.

**Evidência.** src/phifm/training/pairs.py:142-163 (`dividir`): a divisão é só por âncora (`val = pares.join(val_ids, on='arxiv_id', how='semi')`), e a única guarda confere `arxiv_id` (âncora). Nada impede que o documento citado de um par de validação seja positivo de outra âncora no treino, nem que a âncora de validação apareça como positivo no treino. `git log -- src/phifm/training/pairs.py`: a função não mudou depois de 2026-08-09; a remedição de 2026-09-06 mexeu em `amostragem.preparar_pool` (sorteio + deduplicação) e na amostragem do treino, não na divisão. ESTADO.md:2574-2576 declara a divisão 'limpa' por conferir só a âncora. Log t1a6m_treino_sorteado.log: o treino de 6 M cobre 650.162 dos 667.304 documentos citados; o universo de citados da validação tem 88.807 (teto_do_recuperador.json). O próprio rascunho (linha 538) relata que, no reordenador, passar a documentos inéditos derrubou o nDCG@10 de 0,139 para 0,020.

**Correção proposta.** Acrescentar à Seção 3.5: "A divisão entre treinamento e validação é feita por âncora: nenhum documento citante da validação aparece como âncora no treinamento. Os documentos citados não são separados: X% dos candidatos do conjunto de avaliação ocorrem como positivos de outras âncoras no treinamento, e Y% das âncoras de validação ocorrem ali como positivos. A avaliação mede, portanto, a recuperação de pares de citação inéditos sobre um acervo visto no ajuste, e não a generalização a documentos inéditos; os modelos gerais comparados não viram esse acervo." Medir X e Y, reportar a Tabela 8 também no subconjunto do pool cujo positivo e cuja âncora nunca aparecem no treinamento, e repetir a ressalva na Seção 6.

### L233 · maior · falta controle ou limitação · aplicado

> protocolo idêntico para todos os modelos

**Problema.** 'Protocolo idêntico' significa aqui que todos os modelos externos são lidos a 192 tokens, com média mascarada e sem prefixo, que é a configuração de treinamento do bi-encoder próprio e não a de uso do GTE-large (512 tokens). A sensibilidade ao comprimento foi medida apenas para o modelo treinado a 192, onde é nula por construção plausível; para um modelo treinado com entradas mais longas, ler os 28% de texto descartados pode ajudar. A afirmação central da seção (23 M supera 335 M por 0,044) fica sem o controle que a isolaria dessa escolha. A Seção 3.5 tampouco informa truncamento e agregação aplicados aos modelos de comparação.

**Evidência.** src/phifm/eval/encoders.py: `avaliar_um(..., max_tokens=192)` e agregação por média mascarada para todos os modelos; scripts/avaliar_encoders.py: `--max-tokens` padrão 192; cabeçalho de todas as tabelas dos logs g1_*: '2000 candidatos · média mascarada · 192 tokens'. O cartão do thenlper/gte-large indica truncamento a 512 tokens. O rascunho (linha 308) relata que a 192 tokens 63,8% das âncoras são truncadas e 28,2% do texto é descartado; truncagem_192/256/384.json medem só `models/phiemb-do-sistema`. A docstring de encoders.py registra a ressalva ('números publicados do PhysBERT podem usar protocolo próprio'), mas o artigo não.

**Correção proposta.** Medir o GTE-large (e o PhysBERT) no mesmo pool com 512 tokens e reportar a linha adicional na Tabela 8. Na Seção 3.5: "Todos os modelos, próprios e externos, são avaliados com truncamento a 192 tokens, agregação por média mascarada e sem prefixo de instrução — a configuração de treinamento do bi-encoder, e não necessariamente a de uso recomendada de cada modelo externo." Se a medição a 512 não for feita, registrar na Seção 6 que a comparação com o GTE-large vale para entradas truncadas a 192 tokens.

### L259 · maior · erro estatístico · aplicado

> Remedição no protocolo corrigido: 2.000 pares sorteados e deduplicados, teto de 1,0000 medido e gravado.

**Problema.** O checkpoint reportado na Tabela 8 (e nas Tabelas 9 e 10) é o máximo de até 234 avaliações sobre metade do próprio conjunto em que o veredito é medido; as bases gerais não passam por seleção alguma. É seleção no conjunto de teste, com viés otimista a favor dos modelos próprios e crescente com o número de avaliações (15 no treino de 400 mil, 234 no de 6 M). Pelo log, a ordem de grandeza é de até ~0,007 no pool de acompanhamento: não ameaça a margem de +0,044 sobre o GTE-large, mas é da ordem das diferenças lidas como 'empate' ou 'ganho por duplicação'. Além disso, o mesmo pool de 2.000 julgou todas as decisões de desenho (lote, negativos, volume, base: 13 modelos só no g1_resultado.json). A correção de 2026-09-06 trocou `head` por `preparar_pool` no acompanhamento, mas manteve o mesmo arquivo e a mesma semente do veredito.

**Evidência.** src/phifm/training/embedding.py:602 — `avaliar` monta o pool de acompanhamento com `preparar_pool(val, n, exigir_n=False)`, semente padrão SEMENTE_POOL=17, sobre o mesmo `pares_validacao.parquet` (133.540 linhas) do veredito; embedding.py:789-821 — `_talvez_melhor` grava o checkpoint `-melhor` quando o nDCG@10 desse pool sobe. t1a_curvas_de_treino.json: `n_candidatos: 1000`, `passos_aval: 200`, `semente: 17`. scripts/avaliar_encoders.py:108 usa `preparar_pool(val, 2000, 17)`: o pool de 1.000 do acompanhamento é, por construção, o prefixo do pool de 2.000 do veredito. g1_resultado.json: todos os modelos próprios avaliados são os diretórios `...-melhor`. t1a6m_treino_sorteado.log: 234 avaliações, 55 trocas de 'melhor'; o checkpoint eleito marca 0,702 no acompanhamento, as oito últimas avaliações ficam entre 0,693 e 0,699 e a final em 0,695. pairs.py só grava `pares_treino` e `pares_validacao` (FRACAO_VALIDACAO = 0,02); não existe conjunto de teste. DOC-12 §7: 'Durante o treino: público de dev; Final: privado de teste'.

**Correção proposta.** Declarar na Seção 3.5: "O checkpoint avaliado de cada treinamento é o de maior nDCG@10 em avaliações periódicas (a cada 200 passos) sobre os 1.000 primeiros pares do mesmo conjunto de 2.000; não há conjunto de teste separado, e os modelos gerais não passam por seleção." E remedir: os 20.078 pares deduplicados deixam um bloco disjunto (por exemplo, as linhas 2.000 a 4.000 do mesmo sorteio, ou outra semente com exclusão das 2.000 primeiras) que serve de teste sem novo treinamento; reportar a Tabela 8 nesse bloco, ou ao menos o checkpoint final ao lado do `-melhor`. Enquanto isso não for feito, incluir a ressalva na Seção 6.

### L266 · maior · erro de referência · aplicado

> MiniLM-L6 [12] (geral, sem ajuste)

**Problema.** A base do recuperador não é o MiniLM destilado da referência [12]: é um modelo de embedding já treinado com cerca de 1,17 bilhão de pares, dos quais ~169 milhões são pares de citação do S2ORC (que inclui o arXiv), isto é, a mesma tarefa e potencialmente os mesmos pares do conjunto de avaliação. Duas consequências não registradas: (i) o rótulo 'sem ajuste' é enganoso, e o 0,4761 dessa linha não é desempenho sem supervisão de citação; (ii) há risco de contaminação da avaliação por pares de citação vistos pela base, herdado pelos modelos ajustados sobre ela. O GTE [10] também foi treinado com pares de texto em larga escala cuja composição o cartão do modelo não detalha. A frase da Seção 4.6 de que o ajuste 'ensina a tarefa que uma base contrastiva geral já sabe' tem aqui uma explicação mais direta.

**Evidência.** O checkpoint usado é `sentence-transformers/all-MiniLM-L6-v2` (src/phifm/eval/encoders.py, CONCORRENTES; t1a_curvas_de_treino.json, `base`). O cartão do modelo (huggingface.co/sentence-transformers/all-MiniLM-L6-v2, README consultado nesta revisão) declara ajuste contrastivo a partir de `nreimers/MiniLM-L6-H384-uncased` com 1.170.060.424 pares, entre eles 'S2ORC Citation pairs (Abstracts)' 116.288.806, 'S2ORC Citation pairs (Titles)' 52.603.982, 'S2ORC (Title, Abstract)' 41.769.185 e 'SPECTER citation triplets' 684.100. No rascunho, `grep -n S2ORC` só devolve a linha 60 (peS2o) e a referência [16]; a linha 50 descreve a base como 'o MiniLM [12], obtido por destilação de auto-atenção'.

**Correção proposta.** Na Seção 3.4 e na linha 50: "A base é o all-MiniLM-L6-v2 [11, 12], um MiniLM de 6 camadas já ajustado contrastivamente pelos autores do sentence-transformers com cerca de 1,17 bilhão de pares, entre os quais ~169 milhões de pares de citação do S2ORC." Na Tabela 8, trocar o rótulo para "all-MiniLM-L6-v2 (geral, sem ajuste neste trabalho)". Na Seção 6: "A base do bi-encoder e, possivelmente, os modelos gerais comparados foram treinados com pares de citação do S2ORC, que cobre o arXiv; parte dos pares de avaliação pode ter sido vista por essas bases. A sobreposição não foi medida."

### L271 · maior · falta controle ou limitação · aplicado

> O bi-encoder de 23 M supera o GTE-large de 335 M em nDCG@10, recall@10 e MRR

**Problema.** Cada linha das Tabelas 8 a 10 é um único treinamento com uma única semente. McNemar e bootstrap por item cobrem a variabilidade de itens, não a de treinamento. A margem de +0,044 sobre o GTE-large é várias vezes maior que a variação entre execuções observada (0,002 a 0,008) e deve sobreviver; mas a limitação não está declarada para estas seções, e o protocolo estatístico do próprio projeto exige três sementes ou a declaração da falta.

**Evidência.** t1a_curvas_de_treino.json: `semente: 17` em todas as configurações da curva de volume; t1g_comparacao.json, ressalvas: 'Uma semente por modelo.'; t1h e t1i: 'Uma semente por base.' DOC-12 §3.1-3.2: 'Número de sementes ≥ 3 quando há amostragem' e 'reportar só bootstrap esconde que o modelo é instável'. Estimativa disponível da variação entre execuções da mesma receita: Tabela 6, 0,4657 (T4) contra 0,4579 (local) = 0,0078; no protocolo corrigido (g1_resultado.json), 0,5265 (`phiemb-minilm-t4-melhor`) contra 0,5246 (`phiemb-minilm-melhor`) = 0,0019. O rascunho declara 'uma semente por braço' só para as Seções 4.9.x (linhas 40, 403, 613).

**Correção proposta.** Acrescentar à Seção 3.6 e à Seção 6: "Todos os treinamentos contrastivos das Tabelas 8 a 10 têm uma semente por configuração; os testes pareados cobrem a variabilidade de itens, e não a de treinamento. A única estimativa desta última é a diferença entre duas execuções da mesma receita em máquinas distintas, de 0,002 a 0,008 de nDCG@10; diferenças dessa ordem entre configurações não são interpretadas." Idealmente, treinar mais duas sementes da configuração de 400 mil pares sorteados (3.125 passos cada) e reportar média e desvio.

### L108 · menor · clareza · aplicado

> interrompe a execução se o nDCG@10 de um recuperador perfeito sobre o conjunto diferir de 1,0

**Problema.** O que interrompe a execução é a presença de texto repetido; o teto é medido e gravado, não usado como condição de parada. As duas coisas coincidem por construção (o teto calculado vale 1,0 se e somente se não há repetição), de modo que a frase não está errada no efeito, mas descreve um mecanismo que o código não tem. A mesma frase aparece na linha 547. O 'recuperador perfeito' é perfeito apenas quanto a empates por texto idêntico: não considera que a âncora pode citar outros documentos presentes no pool.

**Evidência.** src/phifm/training/amostragem.py:206-212 — `preparar_pool` levanta AssertionError se houver âncora ou positivo repetido no pool. scripts/avaliar_encoders.py:151-158 — o teto é calculado depois por `teto_do_pool` e, se < 0,999, o script apenas imprime um aviso e grava o resultado. `teto_do_pool` (amostragem.py:216-240) só conta textos idênticos.

**Correção proposta.** "... e interrompe a execução se houver âncora ou documento-alvo repetido no conjunto; o desempenho de um recuperador perfeito quanto a empates — 1,0 quando não há repetição — é calculado e gravado no artefato ao lado da métrica."

### L108 · menor · falta controle ou limitação · pendente

> usa 2.000 consultas contra o universo completo de 88.807 documentos citados distintos

**Problema.** O protocolo de sistema usa exatamente o esquema 'amostra aleatória sem deduplicação' que a Tabela 7 mostra ter teto abaixo de 1,0 (âncoras repetidas compartilham uma ordenação), e a verificação de teto da Seção 3.5 não se aplica a ele. Além disso, os demais documentos citados pela mesma âncora estão no universo e contam como não relevantes, de modo que o recall@1 tem teto estrutural bem abaixo de 1,0 e as métricas com um relevante por consulta subestimam um modelo que acerta outra referência da mesma âncora. Afeta todos os sistemas por igual, mas não está declarado; 'universo completo' sugere o acervo inteiro.

**Evidência.** scripts/avaliar_t1b.py:184-191 — `pool = val.unique(subset=['arxiv_citado'])` (o universo são os citados da VALIDAÇÃO; o treinamento tem 667.304 citados distintos) e `linhas = val.sample(n=2000, seed=...)`, sem deduplicação de âncora, com um único alvo por linha. pairs.py: MAX_POR_ANCORA = 8; 133.540 pares de validação para no máximo ~20 mil âncoras distintas (amostragem.py:175: 20.078 pares após deduplicação), isto é, vários citados por âncora, todos dentro do universo. teto_do_recuperador.json confirma universo 88.807 e 2.000 consultas, mas não mede teto.

**Correção proposta.** "O protocolo de sistema usa 2.000 pares sorteados da validação como consultas contra os 88.807 documentos citados distintos do conjunto de validação. Cada consulta tem um único alvo: os outros documentos citados pela mesma âncora (até sete) permanecem no universo e contam como não relevantes, e âncoras podem repetir-se entre as consultas; o teto deste protocolo é inferior a 1,0 e não foi medido." Medir o teto com a função usada no outro protocolo, ou avaliar com todos os citados da âncora como relevantes.

### L110 · menor · inconsistência interna · pendente

> Braços de uma mesma comparação são medidos na mesma sessão e com o mesmo código.

**Problema.** A Tabela 8 e a curva da Tabela 9 não foram medidas numa mesma sessão: as linhas de comparação vêm de cache de sessões anteriores, entre as quais houve commits no avaliador (634c47d, 2a8e61d). A garantia real é mais fraca que a enunciada — mesma amostra, mesmo n e mesmo comprimento —, embora a reprodução posterior do valor do sistema indique que não houve deriva.

**Evidência.** g1_t1a6m.log (2026-09-08 11:13): 'cache: 12 modelo(s) já medidos no mesmo protocolo' — GTE-large, PhysBERT, SciBERT, MiniLM-L6 e os demais vêm do cache; só o modelo de 6 M é medido. O GTE-large foi medido em g1_pool_corrigido.log (2026-09-07 01:53-02:21). encoders.py `comparar`: o cache é invalidado por (digesto da amostra, n, max_tokens), não por versão de código ou de biblioteca. Atenuante: t1g e t1i remediram o sistema em sessão própria e reproduziram 0,6223.

**Correção proposta.** "Braços de uma mesma comparação são medidos sobre a mesma amostra, verificada por digesto, com os mesmos n e comprimento máximo; nas Tabelas 8 e 9 as posições por item dos modelos já medidos são reutilizadas de sessões anteriores do mesmo protocolo, e nas comparações com regra registrada (Tabelas 10 e seguintes) todos os braços são remedidos na mesma sessão."

### L114 · menor · inconsistência interna · aplicado

> Diferenças de nDCG@10 usam bootstrap pareado por consulta.

**Problema.** A afirmação principal da Seção 4.5 ('supera em nDCG@10, recall@10 e MRR') é a única da seção sem o teste que a Seção 3.6 promete, e a Tabela 8 não traz intervalos. O resultado quase certamente se confirma (a cota acima já estabelece recall@10), mas o texto reporta teste apenas para a métrica em que não há diferença.

**Evidência.** g1_resultado.json: o bloco `pareado` traz só McNemar em recall@1 (188×153, p = 0,0654, recalculado: 0,06543); não há intervalo para a diferença de nDCG@10, recall@10 ou MRR entre o modelo de 6 M e o GTE-large. Os bootstraps existentes são de T1g/T1h/T1i e T2eq. `g1_cache.json` (posições por item) não está versionado. DOC-12 §3.1: 'Um número sem intervalo de confiança não é publicado'. Cota calculável sem os itens: recall@10 de 1.661 contra 1.528 acertos em 2.000 dá diferença líquida de 133; no pior caso (472×339 discordantes) o McNemar exato dá p = 3,4×10⁻⁶.

**Correção proposta.** Calcular, a partir das posições por item, o bootstrap pareado da diferença de nDCG@10 e de MRR e o McNemar em k = 10 entre o bi-encoder de 6 M e o GTE-large, e escrever: "... supera o GTE-large em nDCG@10 (+0,044 [a; b], bootstrap pareado por item), em recall@10 (x contra y discordantes, p = ...) e em MRR (+0,034 [c; d])". Versionar o arquivo de posições por item que sustenta os testes.

### L243 · menor · clareza · em parte

> 196 discordantes contra 165 em favor do modelo geral

**Problema.** Os números e os p-valores estão corretos, mas 'discordantes' designa o total (361 e 341), e não cada lado do placar; 'N discordantes contra M' induz a ler 196 ou 188 como o total. O mesmo vale para '188 contra 153 discordantes' na linha 271.

**Evidência.** `git show 13313c9:data/processed/avaliacao/g1_resultado.json`: ganha_a (bi-encoder) = 165, ganha_b (GTE-large) = 196, discordantes = 361, p = 0,1142. g1_resultado.json atual: 188 e 153, discordantes = 341, p = 0,0654.

**Correção proposta.** Linha 243: "(361 pares discordantes, 196 a favor do modelo geral e 165 a favor do bi-encoder, p = 0,114)". Linha 271: "(341 pares discordantes, 188 a favor do bi-encoder e 153 a favor do GTE-large, p = 0,065)".

### L251 · menor · sem evidência versionada · pendente

> Primeiras 2.000 linhas (usado na Tabela 6) | 0,5235 | 0,7562

**Problema.** O número que dá título ao defeito (teto de 0,7562) e as duas primeiras linhas da Tabela 7 não têm artefato de medição versionado; estão registrados só em comentários de código e no relato do projeto. São coerentes entre si e com a fórmula de `teto_do_pool`, mas não reproduzíveis a partir da evidência versionada.

**Evidência.** `grep -rl '7562\|9364\|9761\|5235' data/processed/avaliacao` só devolve coincidências de dígitos em arquivos sem relação. Os valores 0,5235/0,7562, 0,9364/0,9761, '1.147 distintos', '62%', '28 vezes', 'mediana 1.180 → 870' e 'percentil 94' existem apenas em docstrings (amostragem.py, encoders.py, tests/regression/test_pool_g1.py) e no ESTADO.md. O único teto versionado como artefato é o do pool corrigido (g1_resultado.json, `teto_do_protocolo`: 1,0 nas quatro métricas). Checagem de coerência possível: com 1.147 positivos distintos, o teto de recall@1 só por positivos repetidos seria 1.147/2.000 = 0,5735, compatível com 0,5235 depois das âncoras repetidas.

**Correção proposta.** Gerar e versionar um artefato (por exemplo, `g1_teto_por_esquema.json`) com `teto_do_pool` aplicado a `val.head(2000)`, `val.sample(2000, seed=17)` e ao pool deduplicado, mais as contagens de positivos distintos, a fração de linhas com positivo repetido, a multiplicidade máxima e as medianas de comprimento; citar esse artefato no material suplementar.

### L255 · menor · afirma mais que a evidência · pendente

> o ruído de desempate arbitrário em 62% dos itens é de magnitude superior

**Problema.** A afirmação é plausível, mas não medida: a estimativa dá um ruído da mesma ordem de grandeza da margem de 0,0029 (cerca de 1,3 a 1,8 vez), e não uma ordem acima. O que a evidência estabelece com mais força é que o mesmo modelo passou de +0,003 a −0,052 ao ser remedido.

**Evidência.** Nenhum artefato mede a variância do nDCG@10 por desempate. Estimativa feita nesta revisão, supondo recuperador perfeito, 760 linhas sem repetição e 387 grupos somando 1.240 linhas (um deles de 28): desvio-padrão da média de nDCG@10 por desempate ≈ 0,0037 por modelo (≈ 0,005 na diferença entre dois modelos).

**Correção proposta.** "... a paridade decidia-se em 0,0029 de nDCG@10, e o ruído de desempate arbitrário em 62% dos itens é da mesma ordem (desvio-padrão estimado de 0,004 a 0,005 na diferença entre dois modelos); remedido no protocolo corrigido, o mesmo modelo fica 0,052 abaixo."

### L257 · menor · inconsistência interna · pendente

> uma amostra aleatória de mesmo tamanho fornece 191.300

**Problema.** Dois valores para a mesma grandeza em seções vizinhas (191.300 na 4.5, 191.198 na Tabela 9). Ambos têm origem identificável e o fator de 10,7 vale para os dois, mas o treinamento efetivamente executado usou 191.198.

**Evidência.** t1a_treino_sorteado.log: 'pares: 400,000 de 400,000 disponíveis · 191,198 documentos citados distintos'; Tabela 9 (linha 280) usa 191.198. O valor 191.300 vem da medição prévia em amostragem.py:95 (`amostrar_do_plano`, outro sorteio). 191.300/17.844 = 10,72 e 191.198/17.844 = 10,71.

**Correção proposta.** Usar 191.198 na linha 257 ("... fornece 191.198 — fator de 10,7 em diversidade"), ou explicitar: "191.300 numa amostra de conferência e 191.198 na amostra efetivamente usada no treinamento (Tabela 9)".

### L267 · menor · falta controle ou limitação · aplicado

> SciBERT ajustado, 400 mil pares (primeiras linhas)

**Problema.** Segundo o relato do projeto, o bi-encoder sobre SciBERT foi ajustado com lote 8 (7 negativos), não com a receita de 127 negativos das demais linhas. A Tabela 8 o apresenta como se diferisse apenas na base, e a leitura da linha 300 ('o ganho do ajuste é inverso ao ponto de partida — o SciBERT sobe 0,221 e termina abaixo do MiniLM-L6') fica confundida com o número de negativos.

**Evidência.** ESTADO.md:3886 e 3989: 'Lote 8 (o que o SciBERT aguentava) · 7 negativos por âncora' contra '128 (escolhido) · 127'; ESTADO.md:3935-3937: 'O menor com 127 negativos no lote vence o maior com 7'. A Seção 3.4 (linha 90) só descreve a receita do MiniLM-L6 (lote 128). O log de treino do modelo sobre SciBERT não está versionado.

**Correção proposta.** Rotular a linha como "SciBERT ajustado, 400 mil pares (primeiras linhas), lote 8" e acrescentar à Seção 3.4: "O ajuste sobre SciBERT usou lote 8 (7 negativos), limite de memória da placa local; não é comparável, como ablação de base, às execuções com lote 128." Qualificar a frase correspondente da Seção 4.6.

### L271 · menor · inconsistência interna · aplicado

> ficava 0,035 abaixo do GTE-large, e não 0,003 acima

**Problema.** O '+0,003' pertence ao modelo de 400 mil pares (o da Tabela 6) e o '−0,035' ao modelo de 1,5 M: a frase compara modelos diferentes. No mesmo modelo, a inversão é de +0,003 para −0,052 (400 mil) ou de −0,011 para −0,035 (1,5 M); além disso, a ordem entre os dois modelos próprios também se inverteu. A formulação 'o melhor então disponível' é defensável, mas o leitor entende que se trata do mesmo modelo. O pronome em 'levou-a a +0,044' não tem antecedente feminino.

**Evidência.** `git show 13313c9:data/processed/avaliacao/g1_resultado.json` (protocolo antigo): `phiemb-minilm-t4-melhor` (400 mil) 0,4657 = +0,0029 sobre o GTE-large (0,4628); `phiemb-minilm-1m5-melhor` (1,5 M) 0,4520 = −0,011. g1_pool_corrigido.log (protocolo corrigido): 1,5 M = 0,5442 (−0,035); 400 mil T4 = 0,5265 (−0,052).

**Correção proposta.** "Remedido, o modelo da Tabela 6 (400 mil pares, 0,5265) ficava 0,052 abaixo do GTE-large, e não 0,003 acima; o melhor bi-encoder então disponível passou a ser o de 1,5 milhão de pares das primeiras linhas (0,5442, antes 0,4520), ainda 0,035 abaixo. O novo treinamento com pares sorteados e volume crescente (Seção 4.6) levou a margem a +0,044."

### Conferido e correto

- Tabela 8, as sete linhas e quatro métricas, conferem com data/processed/avaliacao/g1_resultado.json: 6 M 0,6223/0,4315/0,8305/0,5636; GTE-large 0,5788/0,4140/0,7640/0,5293; 400 mil primeiras linhas 0,5265/0,3560/0,7185/0,4773; MiniLM-L6 0,4761/0,3135/0,6640/0,4279; SciBERT ajustado 0,4746/0,3070/0,6705/0,4263; PhysBERT 0,3507/0,2220/0,4910/0,3170; SciBERT 0,2537/0,1490/0,3825/0,2275.
- Tabela 6 confere com a versão histórica do artefato (git show 13313c9:data/processed/avaliacao/g1_resultado.json): 0,4657/0,262/0,708 (phiemb-minilm-t4-melhor); GTE-large 0,4628/0,2775/0,6765; execução local 0,4579/0,2535/0,700; PhysBERT 0,2752/0,1455/0,425; SciBERT 0,2074/0,1095/0,328; n = 2.000 para todos.
- Linha 243: +0,190 sobre o PhysBERT (0,4657 − 0,2752 = 0,1905) e +0,0029 sobre o GTE-large (0,4657 − 0,4628) conferem.
- McNemar exato recalculado com scipy.stats.binomtest: 196×165 → p = 0,1142 (texto: 0,114); 188×153 → p = 0,0654 (texto: 0,065). As contagens batem com as diferenças de recall@1 (35 itens = (0,4315 − 0,4140) × 2.000).
- Linha 271: 0,5442 − 0,5788 = −0,035; 0,6223 − 0,5788 = +0,0435 (+0,044); 0,6223 − 0,3507 = 0,272; 'inverteu-se duas vezes' (passa por +0,003, falha por −0,035, passa por +0,044) confere com os vereditos em git 13313c9, g1_pool_corrigido.log e g1_t1a6m.log.
- Superioridade pontual do modelo de 6 M sobre o GTE-large em nDCG@10, recall@10 e MRR confere; para recall@10 a diferença líquida de 133 itens dá p ≤ 3,4×10⁻⁶ no McNemar exato mesmo no pior caso de discordantes.
- Seção 3.5: o pool é sorteado com semente (17) e deduplicado por texto de âncora e de positivo (amostragem.preparar_pool); o log registra '133,540 pares · pool sorteado (semente 17) e desduplicado: 2,000 candidatos, alvo e consulta únicos'.
- Seção 3.5: o teto é gravado no artefato ao lado da métrica — g1_resultado.json traz `teto_do_protocolo` com recall_1, recall_10, ndcg_10 e mrr iguais a 1,0, e os logs imprimem 'teto do protocolo ... nDCG@10 1.0000 · recall@1 1.0000'.
- Seção 3.5: protocolo de sistema com 2.000 consultas e universo de 88.807 documentos (teto_do_recuperador.json, base_*.json, truncagem_*.json); fusão recíproca de postos com k = 60 (hibrido.py, K_RRF = 60).
- Seção 3.5: as métricas recall@k, MRR e nDCG@10 (com um relevante por consulta, 1/log2(1+posição) até a décima posição) são as implementadas em encoders.avaliar_um.
- Seção 3.6: McNemar exato bicaudal sobre discordantes nos mesmos itens (hibrido.mcnemar_em, binomial exata; o cache é chaveado pelo digesto da amostra); bootstrap pareado por item existe (bootstrap_pareado_itens, 10.000 reamostragens) e é usado em T1g/T1h/T1i/T2eq; intervalo de Wilson implementado em src/phifm/eval/statistics/proporcao.py; Bonferroni aplicado em T1c (alfa 0,025), T1h (IC de 98,75%, 4 comparações) e T1i (97,5%, 2 comparações).
- Seção 3.6: bootstrap por artigo, e não por equação, na medição de degradação de equações (latex_audit._bootstrap_ausencia reamostra papers; s3b_latex.json traz ausencia_ic95).
- Linha 245: a descrição do defeito (primeiras n linhas, similaridade idêntica em textos idênticos, desempate arbitrário, âncoras repetidas compartilhando uma ordenação) corresponde ao código anterior documentado em embedding.avaliar e ao log g1_n2000.log ('usando os 2000 primeiros'); 1.147 positivos distintos é coerente com o teto de recall@1 de 0,5235 (limite de 0,5735 só por positivos repetidos).
- Linha 255: recall@1 de 0,2620 contra teto de 0,5235 confere com a Tabela 6 e a Tabela 7; a comparação entre modelos era simétrica (mesmo pool para todos).
- Linha 257: fator 10,7 confere (191.300/17.844 = 10,72); os documentos distintos das execuções sorteadas conferem com os logs de treino: 191.198 (400 mil), 390.856 (1,5 M), 518.635 (3 M), 650.162 (6 M).
- O modelo de 6 M segue sendo o recuperador do sistema depois da v0.5: t1g_comparacao.json e t1i_comparacao.json remediram `models/phiemb-do-sistema` em sessão própria e reproduziram 0,6223/0,4315/0,8305/0,5636; T1h, T1i, o commit 62e42fc e o ADR-0003 não tornam obsoleto nenhum número das linhas 104-119 e 231-272.
- Pista (2) confirmada no código atual: o `-melhor` continua eleito por nDCG@10 sobre o mesmo arquivo de validação do veredito, agora via preparar_pool em vez de head (ver achado); pista (3) confirmada: uma semente (17) por configuração; pista (4): p-valores conferem; pista (5) confirmada: o artigo não registra o treino do all-MiniLM-L6-v2 com pares de citação do S2ORC.

### Não verificável nesta máquina

- Taxa de sobreposição em nível de documento entre o pool de avaliação e o treino (a pista de ~99,9% dos candidatos presentes como positivos no treino): exige pares_treino.parquet e pares_validacao.parquet (colunas arxiv_id e arxiv_citado) para cruzar positivos e âncoras do pool de 2.000 com os positivos e âncoras do treino de 6 M. Só a ausência da guarda no código foi confirmada.
- Tabela 7 (tetos 0,5235/0,7562 e 0,9364/0,9761) e as estatísticas da linha 245 (1.147 positivos distintos, 62% das linhas, 28 repetições, medianas de 1.180 e 870 caracteres, percentil 94): exigem pares_validacao.parquet para rodar teto_do_pool em val.head(2000) e val.sample(2000) e as contagens. Também os 20.078 pares após deduplicação.
- 6.564.111 pares, 667.304 documentos citados distintos e 17.844 distintos nas primeiras 400 mil linhas: exigem pares_treino.parquet.
- Intervalo de bootstrap para a diferença de nDCG@10 e de MRR e McNemar em k = 10 entre o bi-encoder de 6 M e o GTE-large: exigem as posições por item (g1_cache.json, não versionado) ou os checkpoints para remedir.
- Tamanho do viés de seleção do checkpoint `-melhor`: exige os checkpoints final e `-melhor` de cada execução medidos num bloco de validação disjunto do pool de acompanhamento. Também não foi possível confirmar que o sorteio do polars na sessão de Kaggle coincide linha a linha com o local (mesma semente, versão da biblioteca desconhecida); em qualquer caso os dois pools saem do mesmo arquivo.
- Efeito do truncamento a 192 tokens sobre o GTE-large e o PhysBERT: exige rodar esses modelos a 512 tokens no mesmo pool.
- Sobreposição entre os pares de avaliação e os pares de citação do S2ORC usados no treino do all-MiniLM-L6-v2 (e dos dados do GTE): exige a lista de pares do S2ORC do sentence-transformers e as datas e identificadores das âncoras de validação.
- Receita do bi-encoder sobre SciBERT (lote 8, número de pares): o log de treino não está versionado; a informação vem só do ESTADO.md.
- Variância real do nDCG@10 por desempate arbitrário no pool antigo: exige os embeddings dos modelos sobre val.head(2000); a estimativa de ~0,004 feita aqui supõe recuperador perfeito e uma estrutura de grupos plausível, não medida.
- Teto do protocolo de sistema (âncoras repetidas entre as 2.000 consultas e outros citados da mesma âncora no universo de 88.807): exige pares_validacao.parquet.


## Volume, diversidade e base (§4.6)

### L285 · maior · erro estatístico · aplicado

> Os ganhos por duplicação de volume são de +0,032, +0,025 e +0,020.

**Problema.** Apresentada como está, a sequência +0,032, +0,025, +0,020 parece um retorno decrescente por duplicação e reforça a leitura de saturação do parágrafo seguinte. Normalizada pelo número real de duplicações, a sequência é +0,017, +0,025, +0,020: aproximadamente constante (log-linear), sem tendência de queda dentro do intervalo medido.

**Evidência.** Tabela 9 e g1_resultado.json: 400 mil sorteados 0,54616 → 1,5 M 0,57800 (+0,0318) → 3 M 0,60262 (+0,0246) → 6 M 0,62231 (+0,0197). O primeiro passo é de 400 mil para 1,5 milhão: fator 3,75 (1,91 duplicação), e não 2. Por duplicação: 0,0318 / log2(3,75) = +0,0167. O T1h acrescenta um ponto abaixo: MiniLM-L6@200k 0,5289 (t1h_comparacao.json) → 400 mil 0,5462 = +0,017 (sessões distintas). O próprio ESTADO.md (linha 2339) lista '+0,020 / +0,032 / +0,025 / +0,020' e conclui 'Ainda não decrescem monotonicamente'.

**Correção proposta.** "Os ganhos entre degraus são de +0,032 (de 400 mil para 1,5 milhão, fator 3,75), +0,025 e +0,020 (duas duplicações); por duplicação de volume, +0,017, +0,025 e +0,020 — aproximadamente constantes, sem decréscimo monotônico no intervalo medido."

### L287 · maior · erro estatístico · aplicado

> A execução de 6 milhões é a primeira com sinal de saturação: 15 avaliações consecutivas de validação abaixo do próprio pico, com queda de 1,0% até o fim

**Problema.** A estatística 'número de avaliações abaixo do próprio máximo' tem o mesmo defeito que o parágrafo atribui à estatística anterior: mede onde a grade calhou de capturar um máximo ruidoso. Um máximo isolado de +0,003 a +0,007 acima dos vizinhos, com ruído de 0,003 entre avaliações, gera sozinho '15 avaliações abaixo do pico e queda de 1%'. A média do trecho posterior ao pico é superior à do trecho anterior, de modo que os dados versionados não mostram queda; mostram desaceleração (inclinação cerca de metade da de 3 M) e achatamento nas últimas 15 avaliações, que também aparece em 3 M e é compatível com o fim do agendamento da taxa de aprendizado. A frase seguinte ('de modo que ganho adicional teria de vir de outra fonte...') apoia-se nesse sinal.

**Evidência.** data/processed/avaliacao/t1a_curvas_de_treino.json, série 6m_sorteado (nDCG@10 interno, 1.000 candidatos, avaliação a cada 200 passos). Os números da frase conferem: pico 0,7022 no passo 43.800, 15 avaliações depois, final 0,6951, queda relativa 1,01%. Mas: (a) o pico é um ponto isolado: os dez maiores valores são 0,7022 (43.800), 0,6994, 0,6988, 0,6984, 0,6983, 0,6981..., e o desvio-padrão das diferenças entre avaliações consecutivas no fim da série é 0,0031; (b) as médias de blocos de 15 avaliações no fim da série SOBEM: 0,6902 → 0,6908 → 0,6939 → 0,6957 (o último bloco, o 'pós-pico', é o mais alto); (c) a inclinação por mínimos quadrados nas últimas 30 e 45 avaliações é positiva (+0,00016 e +0,00017 por avaliação, t = 2,6 e 5,7), e só as últimas 15 dão inclinação nula (−0,00006, t = −0,45). Para comparação, a execução de 3 M tem inclinação de +0,00025 e +0,00035 nas últimas 30 e 45 e também nula nas últimas 15 (t = 0,58).

**Correção proposta.** "A execução de 6 milhões é a que mais desacelera: na validação interna, a inclinação do nDCG@10 nas últimas 45 avaliações é cerca de metade da observada na execução de 3 milhões (+0,00017 contra +0,00035 por avaliação), e a média das 15 últimas (0,6957) supera a das 15 anteriores (0,6939) por 0,002. O máximo (0,7022, no passo 43.800) é um ponto isolado, 0,003 acima do segundo maior valor, com ruído de 0,003 entre avaliações consecutivas; as 15 avaliações seguintes ficam abaixo dele, mas isso é o esperado de um máximo ruidoso e não constitui queda. Duas estatísticas usadas antes — a fração do treinamento em que o pico ocorre e o número de avaliações abaixo do pico — medem onde a grade calha de capturar o máximo, e não saturação. O que limita o ganho adicional por esta via é o esgotamento dos pares (650.162 de 667.304 documentos já usados), e não um platô medido."

### L300 · maior · afirma mais que a evidência · aplicado

> Com a base geral de 109 M, 400 mil pares produzem o que o volume produziu com quinze vezes mais pares.

**Problema.** p = 0,908 em recall@1 é ausência de evidência de diferença numa métrica, não equivalência, e não se estende às outras duas colunas da Tabela 10. Em recall@10 o recuperador de 6 M é necessariamente superior (p ≤ 0,033), e em nDCG@10 o GTE-base de 400 mil pares fica 0,013 abaixo, sem teste. A frase generaliza um não-rejeitar em recall@1 para 'produzem o que o volume produziu' e contradiz o parágrafo da linha 304, segundo o qual é com 1 milhão de pares — e não com 400 mil — que a base alcança o sistema. O mesmo vale para 'pagar esse custo por um empate' na linha 302.

**Evidência.** t1f_base_n2000.json: GTE-base@400k nDCG@10 0,60945, recall@1 0,4300, recall@10 0,8010; sistema (6 M) 0,62231, 0,4315, 0,8305. O único teste entre os dois é McNemar em recall@1: 150 x 147, p = 0,9076 (reproduzido com scipy.stats.binomtest). Não há TOST nem margem de equivalência declarada em lugar algum (grep por 'equival|TOST|margem' nos scripts de comparação e na regra do T1f). O que os 297 discordantes permitem afirmar: diferença de recall@1 de −0,0015, IC 95% [−0,018; +0,015] (Wald pareado). Em nDCG@10, métrica primária do artigo, a diferença pontual é −0,0129 e não foi testada (as posições por item do T1f não estão versionadas); o erro-padrão do bootstrap do T1g para um par semelhante é cerca de 0,005, o que poria −0,0129 a ~2,6 erros-padrão de zero. Em recall@10 a diferença é de 59 itens em 2.000 (0,8010 contra 0,8305): como o sistema erra 339 itens e o GTE-base 398, há no máximo 737 discordantes, e o McNemar exato no pior caso (339 x 398) dá p = 0,033 — a diferença em recall@10 é significativa a 5% qualquer que seja a tabela de discordância. Além disso, a linha 304 diz que o GTE-base só 'alcança' o recuperador instalado a 1 milhão de pares (0,6211), e descreve 0,6094 → 0,6211 como crescimento.

**Correção proposta.** "Com a base geral de 109 M, 400 mil pares chegam perto do que o volume produziu com quinze vezes mais pares: em recall@1 os dois não se distinguem (147 contra 150 discordantes, p = 0,908; diferença de −0,0015, IC de 95% [−0,018; +0,015]), mas o recuperador de 6 milhões permanece à frente em recall@10 (0,8305 contra 0,8010, p ≤ 0,033 para qualquer tabela de discordância compatível com as marginais) e em nDCG@10 (0,6223 contra 0,6094, diferença não testada). Nenhuma margem de equivalência foi registrada; 'empate', aqui e adiante, significa que o teste pareado não rejeita a igualdade, e não equivalência demonstrada." Na linha 302, trocar 'por um empate com o recuperador instalado' por 'por um resultado que, no melhor caso, iguala o recuperador instalado em recall@1 e fica abaixo dele em recall@10'.

### L304 · maior · desatualizado após a v0.5 · aplicado

> a regra manteve o sistema como está: sem margem, o custo decide.

**Problema.** A §4.6 termina no ponto de 1 M do GTE-base e deixa em aberto 'outra base' (linha 287) e 'a combinação de base e volume' (linha 302). Dois experimentos pré-registrados posteriores respondem exatamente à pergunta que o parágrafo deixa suspensa (uma base pequena traria o ganho de base sem o custo de 4,4x?) e a resposta é negativa a 1 M. O ADR-0003 (linha 277) já registra o desfecho e o encerramento da busca. Sem isso, a seção descreve um estado do trabalho que não é mais o atual, e as linhas 615 e 643 (limitações e conclusão) herdam a lacuna.

**Evidência.** A última alteração do rascunho é f04ac1f (2026-09-23 10:11, T1g). Depois vieram cb49b0d/aaa03a9/0f1d08f (T1h, 23/09), 5dff577/327758c/dc7755d (T1i, 23-24/09) e 62e42fc (24/09, 'a busca por encoder ENCERRADA'). `grep -n -i 't1h|t1i|gte-small|bge-small|e5-small|L12'` no rascunho devolve zero linhas. data/processed/avaliacao/t1h_comparacao.json: MiniLM-L6@200k 0,5289; gte-small 0,5706 (+0,04168 [0,02843; 0,05501], IC 98,75%, McNemar 107x202, p=7,1e-8, custo 1,76x); bge-small 0,5692 (+0,04031 [0,02752; 0,05381], 112x197, p=1,5e-6, 1,75x); e5-small 0,5402 (+0,01133 [-0,00057; 0,02368], EMPATE); MiniLM-L12 0,5404 (+0,01151 [-0,00012; 0,02326], EMPATE). t1i_comparacao.json: gte-small@1M 0,6014 (-0,02087 [-0,0323; -0,0096], IC 97,5%), bge-small@1M 0,6012 (-0,02112 [-0,03233; -0,00998]); recall@1 147x135, p=0,513 nas duas; contra GTE-base@1M -0,01968 [-0,02926; -0,0100] e -0,01993 [-0,0294; -0,01039] (IC 95%); custo 1,77x; GTE-base 18,34 s contra 4,41 s = 4,16x. As regras (colab/t1h_bases_pequenas.py e colab/t1i_pequenas_1m.py, bloco REGRA) foram commitadas antes dos resultados (cb49b0d 13:40 < 0f1d08f 20:29; 5dff577 20:33 < dc7755d 06:32). Conferi que comparar_t1h.py e comparar_t1i.py passam alfa=0,05/4 e 0,05/2 a bootstrap_pareado_itens.

**Correção proposta.** Inserir depois da linha 304 (e ajustar 287, 302, 615 e 643 para remeter a ele):

"Restava saber se uma base do porte do MiniLM-L6 traria o ganho de base sem o custo. Com regra registrada antes, cinco bases de 23 a 33 M foram ajustadas com os mesmos 200 mil pares e medidas na mesma sessão, cada candidata contra o MiniLM-L6 de 200 mil pares (0,5289), com intervalo de 98,75% (Bonferroni sobre quatro comparações) por bootstrap pareado por item. O gte-small e o bge-small ficam acima, por +0,042 [+0,028; +0,055] e +0,040 [+0,028; +0,054] de nDCG@10, com recall@1 pareado de 202 contra 107 e 197 contra 112; o e5-small e o MiniLM-L12 não se separam do controle sob a correção (+0,011 [−0,001; +0,024] e +0,012 [−0,000; +0,023]). Embutir 2.000 textos, após aquecimento, custa entre 1,75 e 1,76 vez o MiniLM-L6 nas quatro, contra 4,2 vezes para o GTE-base medido do mesmo modo. Tomado como referência o GTE-base de 200 mil pares (0,5964, medido em outra sessão, diferença não pareada), as duas vencedoras recuperam cerca de 60% do ganho de base.

Levadas a 1 milhão de pares — os mesmos do GTE-base da Tabela 10 — e comparadas ao recuperador instalado com intervalo de 97,5% (Bonferroni sobre duas comparações), as duas ficam abaixo: 0,6014 e 0,6012 contra 0,6223, diferenças de −0,021 [−0,032; −0,010] em ambas. Em recall@1 não se separam do sistema (135 contra 147 discordantes, p = 0,51); a perda está abaixo da primeira posição (recall@10 de 0,7915 e 0,7945 contra 0,8305). Ficam também 0,020 abaixo do GTE-base de mesmo volume [−0,029; −0,010]. A 1 milhão de pares, portanto, o volume do recuperador instalado pesa mais que a troca por uma base pequena melhor: nenhuma das bases gerais medidas o supera — a de 109 M o alcança a mais de quatro vezes o custo de embutir, e as de 33 M ficam abaixo a 1,8 vez. A comparação que isolaria a base, uma das pequenas com os 6 milhões de pares (estimada em 17 horas de T4), não foi executada, e a extrapolação da curva — cerca de +0,013 por duplicação de pares entre 200 mil e 1 milhão — não é medida. Valem as ressalvas do bloco: uma semente por base, e bge-small e e5-small usados sem o prefixo para o qual foram treinados. A busca por outro encoder foi encerrada neste ponto."

Na linha 615, trocar 'a comparação com a mesma base a 6 milhões, que isolaria o volume, não foi executada' por um texto que cite também as duas bases de 33 M a 1 M; na linha 643, acrescentar que as bases pequenas ficam 0,021 abaixo a 1 M.

### L279 · menor · número não confere · aplicado

> | 400 mil, primeiras linhas | 17.844 | 0,5265 | −0,0523 |

**Problema.** Diferenças calculadas sobre valores arredondados; erro de uma unidade na quarta casa em dois lugares.

**Evidência.** g1_resultado.json: 0,5265462 − 0,5787665 (GTE-large) = −0,05222, que arredonda para −0,0522; o texto traz −0,0523, obtido dos valores já arredondados (0,5265 − 0,5788). Idem na linha 285: '+0,0197' para sorteado contra primeiras linhas, quando 0,5461634 − 0,5265462 = +0,01962 (arredonda para +0,0196). As outras quatro margens da Tabela 9 conferem com os valores não arredondados (−0,0326; −0,0008; +0,0238; +0,0435).

**Correção proposta.** Tabela 9, primeira linha: margem "−0,0522". Linha 285: "+0,0196 de nDCG@10".

### L280 · menor · inconsistência interna · pendente

> | 400 mil, sorteados | 191.198 | 0,5462 | −0,0326 |

**Problema.** Dois números diferentes para a mesma grandeza a 25 linhas de distância, sem aviso de que são sorteios distintos. Ambos são reais; o leitor não tem como saber.

**Evidência.** t1a_treino_sorteado.log linha 1: '400,000 de 400,000 disponíveis · 191,198 documentos citados distintos' (confere com a Tabela 9; t1f_treino.json repete 191198). A linha 256 do rascunho diz que 'uma amostra aleatória de mesmo tamanho fornece 191.300', número que vem do docstring de kaggle/t1a_phiemb.py e de tests/regression/test_amostragem_rerank.py (outra amostra). O mesmo ocorre com 1,5 M: 390.856 no log e na Tabela 9, 390.966 no docstring.

**Correção proposta.** Na linha 256, usar o valor do sorteio efetivamente treinado ("191.198 — fator de 10,7") ou acrescentar à legenda da Tabela 9: "As contagens são as do sorteio usado no treinamento; a estimativa da Seção 4.5 (191.300) vem de outro sorteio do mesmo tamanho."

### L300 · menor · número não confere · aplicado

> MiniLM-L6, que sobe 0,049 (Tabela 8)

**Problema.** O número citado não é o que a tabela referenciada produz. A leitura qualitativa não muda.

**Evidência.** Tabela 8 (linhas 265-266): bi-encoder ajustado com 400 mil pares (primeiras linhas) 0,5265; MiniLM-L6 sem ajuste 0,4761. Diferença: 0,0504 (g1_resultado.json: 0,52655 − 0,47611 = 0,05043). O valor 0,049 corresponde a outro checkpoint, o 'ΦEmb/MiniLM (23M)' de 0,5246 (0,52461 − 0,47611 = 0,0485), que não está na Tabela 8 e só aparece na linha 289. O SciBERT confere: 0,4746 − 0,2537 = 0,2209.

**Correção proposta.** Trocar por "que sobe 0,050 (Tabela 8)".

### L302 · menor · sem evidência versionada · aplicado

> uma razão de 4,4 que se repete em serviço

**Problema.** O 4,4 é a razão de embutir o universo de 88.807 documentos, medida uma vez. 'Que se repete em serviço' é afirmação sem medição; a medição posterior e mais controlada dá 4,2. A diferença não muda a decisão, mas o texto afirma como medido algo que foi inferido.

**Evidência.** 954 s (base_gte_base.json, custo_segundos 954,0) e 216,8 s (teto_do_recuperador.json): razão 4,40, conferida. Mas a única medição de custo feita com instrumento próprio (comparar_t1h.py / comparar_t1i.py: 2.000 textos, lote 16, após aquecimento) dá GTE-base 18,34 s contra 4,41 s do sistema = 4,16x (t1i_comparacao.json); ESTADO.md registra 4,13x no ensaio. O docstring de comparar_t1h.py explica que os segundos do avaliador incluem carga do modelo e aquecimento. Nenhum arquivo mede latência de serviço (consulta a consulta) para o GTE-base.

**Correção proposta.** "...uma razão de 4,4 (4,2 quando medida depois sobre 2.000 textos, após aquecimento), custo que incide também em serviço, a cada consulta e a cada reindexação;"

### L304 · menor · clareza · aplicado

> o GTE-base empata com o bi-encoder de 6 milhões — nDCG@10 de 0,6211 contra 0,6223, diferença de −0,0012 [−0,011; +0,008]

**Problema.** Os números estão corretos e 'empate' é o desfecho definido pela regra registrada (IC cruza zero). Diferentemente da linha 300, aqui o intervalo é estreito o bastante para sustentar uma afirmação de equivalência com margem explícita: o IC de 95% exclui diferenças maiores que 0,011 em qualquer direção. O texto ganharia em dizer isso, em vez de deixar 'empata' sem definição, e em avisar que a margem é lida depois, não registrada antes.

**Evidência.** t1g_comparacao.json: diferenca −0,00119, ic95 [−0,011; 0,00842], cruza_zero true, desfecho EMPATE. kaggle/t1g_gte_1m.py, REGRA (escrita em 2026-09-22, commit 2bde691, anterior ao resultado f04ac1f): 'EMPATE (IC cruza 0)'. A regra não declara margem de equivalência. recall@10 do mesmo par: 0,8180 contra 0,8305 (25 itens; não testado).

**Correção proposta.** "...o GTE-base não se separa do bi-encoder de 6 milhões — nDCG@10 de 0,6211 contra 0,6223, diferença de −0,0012 [−0,011; +0,008] por bootstrap pareado por item, o que limita a diferença a cerca de ±0,011 (margem lida do intervalo, e não registrada antes; a regra definia empate como intervalo que contém zero)..."

### L306 · menor · clareza · aplicado

> produziu +0,039 com p = 0,161; com os 2.000 do protocolo, +0,0633 com p = 1,5 × 10⁻¹⁰

**Problema.** Cada par 'diferença, p' mistura duas métricas: a diferença é de nDCG@10 e o p é do teste pareado em recall@1. Além disso, a explicação dada para o quase falso negativo ('a tarefa é muito mais fácil') omite a causa estatística direta: com n = 256 há só 256 consultas e 51 discordantes, isto é, pouco poder.

**Evidência.** t1f_base.json (n_candidatos 256): nDCG@10 0,8295 − 0,7908 = +0,0387; pareado em recall@1 31 x 20, 51 discordantes, p = 0,161. t1f_base_n2000.json: 0,6094 − 0,5462 = +0,0633; recall@1 219 x 104 (ESTADO.md linha 1470), p = 1,45e-10 (reproduzido). Em t1f_base.json os recalls são múltiplos de 1/256 (0,3125 = 80/256): o parâmetro n fixa também o número de consultas.

**Correção proposta.** "...e produziu +0,039 de nDCG@10, com pareado em recall@1 de 31 contra 20 (p = 0,161); com os 2.000 do protocolo, +0,0633, com 219 contra 104 (p = 1,5 × 10⁻¹⁰). Com 256 candidatos há também apenas 256 consultas — 51 discordantes, pouco poder — e a tarefa é muito mais fácil..."

### L308 · menor · afirma mais que a evidência · aplicado

> não altera o recall em nenhuma profundidade (p entre 0,08 e 0,84) e custa 2,9 vezes mais para embutir

**Problema.** 'Em nenhuma profundidade' vale para as três profundidades testadas, não para todas as disponíveis nos mesmos arquivos; em k = 50 o contexto de 384 tokens é pior com p = 0,007 sem correção, uma diferença de 0,0165 que excede o '±0,005' do relato. A conclusão substantiva (ler mais texto não ajuda) se mantém e até se reforça, mas o intervalo de p declarado não descreve o conjunto de comparações possíveis. O fator 2,9 é o de 384 tokens; com 256 o custo é 2,3 vezes.

**Evidência.** ESTADO.md linhas 1857-1862: só três profundidades foram testadas (k = 10, 100, 200), com p de 0,082 a 0,839 — é daí que vem o intervalo citado. Recalculei o McNemar exato a partir de postos_denso em truncagem_192.json, truncagem_256.json e truncagem_384.json (2.000 consultas): os seis valores citados conferem (0,349; 0,082; 0,728; 0,839; 0,592; 0,431). Nas profundidades não testadas: 384 tokens em k = 50, 87 x 54 a favor de 192 (recall@50 0,5210 → 0,5045), p = 0,0068; 256 tokens em k = 2.000, 5 x 16 a favor de 256, p = 0,027; 256 em k = 1.000, 12 x 24, p = 0,065. Com 18 testes (2 comprimentos x 9 profundidades), o limiar de Bonferroni é 0,0028 e nenhum o atinge. Custos: 216,7 s (192), 497,7 s (256), 625,1 s (384): 2,30x e 2,88x.

**Correção proposta.** "...o bi-encoder já treinado, lendo 256 ou 384 tokens, não melhora o recall nas profundidades testadas (10, 100 e 200; p entre 0,08 e 0,84); nas demais profundidades disponíveis, nenhuma diferença sobrevive à correção para 18 comparações (a menor, p = 0,007, é contra o contexto de 384 tokens em recall@50). O custo de embutir sobe 2,3 e 2,9 vezes, respectivamente."

### Conferido e correto

- Tabela 9, nDCG@10 das cinco linhas (0,5265; 0,5462; 0,5780; 0,6026; 0,6223) e GTE-large 0,5788: conferem com g1_resultado.json (0,52655; 0,54616; 0,57800; 0,60262; 0,62231; 0,57877).
- Tabela 9, margens −0,0326, −0,0008, +0,0238 e +0,0435: conferem com os valores não arredondados.
- Tabela 9, documentos citados distintos 191.198, 390.856, 518.635 e 650.162: conferem com a linha 1 de t1a_treino_sorteado.log, t1a15_, t1a3m_ e t1a6m_treino_sorteado.log.
- Linha 285, ablação de amostragem: 104 contra 71, p = 0,0153 (g1_t1a_sorteado.log linha 45; binomial exata recalculada = 0,01532); mesmos 3.125 passos nas duas execuções (t1a_treino_t4.log e t1a_treino_sorteado.log, ambos em T4).
- Linha 285, 6 M contra 3 M: 115 x 70, p = 0,0012 (g1_resultado.json; recalculado 0,001158).
- Linha 287, números brutos da frase de saturação: pico da execução de 6 M em 43.800, 15 avaliações depois, queda de 1,01%; nas outras, 1, 1 e 0 avaliações após o pico e quedas de 0,03%, 0,15% e 0 (t1a_curvas_de_treino.json). A interpretação é que está em questão, não a contagem.
- Linha 287, pico da execução de 400 mil mais cedo (0,896 do treinamento) que o da de 6 M (0,934): confere.
- Linha 289: 1,5 M sorteado 0,5780 contra 0,5442 das primeiras linhas; pico a 97% (0,9729); 511 negativos 0,5272 contra 0,5246 com 127 (g1_resultado.json e t1f_base_n2000.json).
- Tabela 10, linha GTE-base@400k: 0,6094 / 0,4300 / 0,8010 (t1f_base_n2000.json).
- Tabela 10, linha MiniLM-L6@400k: 0,5462 / 0,3725 / 0,7395; pareado 219 x 104, p = 1,5e-10 (contagens em ESTADO.md linha 1470; p recalculado 1,451e-10).
- Tabela 10, linha MiniLM-L6@6M: 0,6223 / 0,4315 / 0,8305; pareado 147 x 150, p = 0,908 (t1f_base_n2000.json: ganha_a 150, ganha_b 147, p 0,9076).
- Tabela 10, linha GTE-base@1M: 0,6211 / 0,4335 / 0,8180; pareado 148 x 144, p = 0,861 (t1g_comparacao.json).
- Linha 300, 'quinze vezes mais pares' (6 M / 400 mil) e SciBERT 'sobe 0,221' (0,4746 − 0,2537 = 0,2209) terminando abaixo do MiniLM-L6 sem ajuste (0,4746 < 0,4761).
- Linha 300, universo de 88.807 documentos: GTE-base sem ajuste recall@10 0,2755 contra 0,2810 do sistema; McNemar recalculado a partir de postos_denso (base_gte_base.json e teto_do_recuperador.json): 131 x 142, p = 0,545. recall@200 0,6450 contra 0,7300: confere (e a diferença em profundidade é real: 89 x 259, p = 2e-20).
- Linha 302: 954 s e 217 s (base_gte_base.json 954,0; teto_do_recuperador.json 216,8), razão 4,40; estimativa de 39 horas de T4 registrada em t1f_treino.json (sonda_de.custo_estimado_h_t4 = 39) e na regra do T1f.
- Linha 304: −0,0012 [−0,011; +0,008] e +0,043 [+0,033; +0,054] (t1g_comparacao.json: −0,00119 [−0,011; 0,00842] e 0,04312 [0,03277; 0,05369]); bootstrap pareado por item confirmado no código (scripts/avaliar_ablacao_secundarias.py, bootstrap_pareado_itens, 10.000 reamostras).
- Linha 304, 'regra registrada antes': kaggle/t1g_gte_1m.py com a REGRA entrou em 2bde691 (22/09 23:10), o comparador em 949060a (23/09 01:58) e o resultado em f04ac1f (23/09 10:11).
- Linha 304, curva do GTE-base: 0,5964 a 200 mil (t2eq_emb_gte_contra_barra.json), 0,6094 a 400 mil (t1f_base_n2000.json), 0,6211 a 1 M (t1g_comparacao.json); 'um sexto do volume' confere.
- Linha 306: com 256 candidatos, +0,039 (0,8295 − 0,7908) e p = 0,161 (31 x 20); MiniLM-L6 sem ajuste 0,732 contra 0,476 (t1f_base.json e t1f_base_n2000.json); o n = 2.000 reproduz 0,5462 e 0,6223.
- Linha 308: os seis p de truncagem em k = 10, 100 e 200 (0,349; 0,082; 0,728; 0,839; 0,592; 0,431) reproduzidos a partir dos postos; controle de 192 tokens idêntico ao teto_do_recuperador.json; 625,1 / 216,7 = 2,88.
- T1h e T1i (para o parágrafo novo): alfa de Bonferroni 0,05/4 e 0,05/2 efetivamente passado ao bootstrap em comparar_t1h.py e comparar_t1i.py; razões de custo 1,75-1,77x e 4,16x conferem com custo_s dos JSON; regras commitadas antes dos resultados.
- '~60% do ganho de base' (commit 0f1d08f): (0,5706 − 0,5289) / (0,5964 − 0,5289) = 62% para o gte-small e 60% para o bge-small, com a ressalva de que o 0,5964 vem de outra sessão.

### Não verificável nesta máquina

- 17.844 documentos citados distintos nos primeiros 400 mil pares (Tabela 9) e 67.232 nos primeiros 1,5 milhão (linha 289): t1a_treino_t4.log não registra a contagem; os valores só aparecem em docstrings (kaggle/t1a_phiemb.py, src/phifm/training/amostragem.py, tests/regression/test_amostragem_rerank.py) e no ESTADO.md. Seria preciso pares_treino.parquet para contar n_unique do documento citado em head(400.000) e head(1.500.000).
- 667.304 documentos citados distintos e 6.564.111 pares no conjunto de treinamento (linha 287): exigem pares_treino.parquet, ausente desta máquina.
- 63,8% das âncoras truncadas e 28,2% do texto descartado a 192 tokens (linha 308): exigem os pares e o tokenizador; nenhum artefato versionado guarda a medição, só o ESTADO.md (linha 1845).
- 'Interrompido em 38% por estabilização da métrica' (linha 289): o log daquele treinamento de 1,5 M com primeiras linhas não está em data/processed/avaliacao; só o ESTADO.md o afirma.
- nDCG@10 pareado entre GTE-base@400k e o recuperador de 6 M (necessário para testar a diferença de −0,0129 e para um TOST de verdade): t1f_base_n2000.json não guarda posições por item; seria preciso o checkpoint models/phiemb-gte-base-400k-melhor e o pares_validacao.parquet para reexecutar com bootstrap_pareado_itens.
- Tabela de discordância em recall@10 entre GTE-base@400k e o sistema: só foi possível limitar o p pelo pior caso (≤ 0,033) a partir das marginais; o valor exato exige as posições por item.
- Que as contagens idênticas do T1i (147 x 135, p = 0,513, para gte-small e bge-small) são coincidência e não troca de arquivos: o ESTADO.md afirma ter conferido os hashes dos pesos; os checkpoints não estão aqui.
- Estimativas de 39 horas (GTE-base a 6 M) e 17 horas (base pequena a 6 M) de T4: são extrapolações de vazão registradas nos scripts e no ESTADO.md, não medições; os logs de treinamento do T1g, T1h e T1i não estão versionados.
- Custo de serviço por consulta (latência) do GTE-base contra o MiniLM-L6: nenhum artefato o mede; só existem o custo de embutir o universo (954 s contra 217 s) e o de embutir 2.000 textos após aquecimento (4,16x).


## Sistema híbrido e reordenador (§4.7, §4.8)

### L332 · maior · afirma mais que a evidência · aplicado

> é precisamente essa concordância que explica a ausência de ganho

**Problema.** Uma correlação negativa moderada com a posição na fusão é esperada de qualquer reordenador competente, porque a posição na fusão carrega sinal de relevância; ρ = −0,466 não distingue 'redundante' de 'bom'. A afirmação causal exigiria o mesmo indicador para o reordenador que ganha (PhysBERT). E a predição derivada do diagnóstico — outra base acrescentaria informação — falhou para o GTE, de modo que a própria Seção 4.8 enfraquece a explicação que a 4.7 dá como certa.

**Evidência.** t1b_inversao_depois_do_conserto.log: média de −0,466 em 60 consultas, medida só para o reordenador de base MiniLM. Não há medição equivalente para os reordenadores de base GTE e PhysBERT em data/processed/avaliacao/. A Tabela 14 mostra que a base GTE, diferente da do recuperador, também não ganha (p=0,637), e a linha 363 conclui 'não diversidade de base'.

**Correção proposta.** "...a correlação inverte-se e o modelo passa a concordar com o recuperador. A hipótese de trabalho, testada na Seção 4.8, é que essa concordância reflita redundância com a base do recuperador; a medição não a estabelece, porque correlação negativa com a fusão é esperada de qualquer reordenador competente e o indicador não foi obtido para as outras bases." Se houver cota, medir o mesmo Spearman para os reordenadores de base GTE e PhysBERT e incluí-los na Tabela 12.

### L345 · maior · afirma mais que a evidência · aplicado

> por deriva de código e protocolo entre as duas medições

**Problema.** Os artefatos não mostram deriva de código: os recuperadores isolados reproduzem ao décimo de milésimo entre as duas versões. A diferença de 0,0019 que produz o 'sete vezes' é o efeito de um parâmetro declarado — profundidade 50 contra 100. A lição 'um valor histórico carrega a versão do avaliador' (linha 597) atribui a uma causa não medida o que tem causa identificável. O desenho na mesma sessão continua correto, mas pelo motivo certo.

**Evidência.** O valor histórico 0,1666 é de profundidade 50 (t1b_physbert_n2000.json, código 73088dc); o braço antigo da Tabela 13 é de profundidade 100 (t1b2_antigo.json, código 9a4ec94). Entre os dois arquivos, BM25 (0,06/0,2285/0,134) e bi-encoder (0,0555/0,2325/0,1323) são idênticos em recall@1, recall@10 e nDCG@10; só mudam as linhas que dependem da profundidade: fusão 0,1576→0,1594 e fusão+reordenação 0,1666→0,1685. scripts/avaliar_t1b.py:278-284 funde os `--profundidade` primeiros de cada lista (`fundir_rrf(ord_emb, ord_bm)[:a.profundidade]`).

**Correção proposta.** "Os braços na mesma sessão eram parte do desenho, e pagaram: o valor histórico de 0,1666 fora medido com profundidade 50, e o braço novo, com 100. Comparados diretamente, dariam um ganho de 0,0022, sete vezes o efeito real; a diferença é da profundidade — BM25 e bi-encoder isolados reproduzem-se exatamente entre as duas execuções." Na linha 597: "Um valor histórico carrega os parâmetros da execução que o produziu (aqui, a profundidade de candidatos); a Tabela 13 mostra um erro de sete vezes evitado...".

### L347 · maior · afirma mais que a evidência · aplicado

> A regra registrada antes da execução

**Problema.** A regra foi escrita antes da segunda execução, mas depois de vistos os resultados da primeira, sobre as mesmas consultas, e com desfecho assimétrico (empate retira). Não é registro cego. A decisão é defensável e confirmada pela comparação de orçamento igual, mas o leitor precisa saber a ordem dos fatos e que o teste nomeado foi substituído por um argumento de limite.

**Evidência.** A regra ('se empatar, ele sai também — não paga ... os 0,026 de teto que cobra') entra no repositório em f84f9bb (2026-09-08 14:55). A primeira execução do T1b2 (código 9a4ec94, 11:41; resultado versionado em b24436d, 14:45) já mostrara, nas mesmas 2.000 consultas, fusão contra bi-encoder com p=0,949 e recall@100 de 0,6065 contra 0,6325 — a diferença de 0,026 citada na própria regra. `git show 9a4ec94:kaggle/t1b2_cadeia.py` não contém a regra. O pareado que a regra nomeia não foi calculado (t1b2_sem_fusao.json só pareia contra a fusão; ESTADO.md: 'O pareado da regra NÃO foi calculado pela rodada'); a decisão saiu por aritmética: 594 contra 589 acertos em k=10, líquido de 5, p mínimo possível de 0,0625.

**Correção proposta.** "Vista a primeira execução, registrou-se, antes da segunda, a regra de que o BM25 permanece apenas se a cadeia com fusão vencer a cadeia sem fusão. A cadeia com fusão acerta 594 consultas em k = 10, e a sem fusão, 589; com saldo de cinco consultas, o menor p possível do teste exato é 0,0625, e a regra retirou o BM25 da composição."

### L355 · maior · número não confere · aplicado

> Acerto@1" refere-se ao grupo de 8 candidatos do treinamento

**Problema.** A coluna Acerto@1 da Tabela 14 reporta o último passo (0,566 e 0,510), enquanto as colunas vizinhas (nDCG, discordantes, p) são do ponto de verificação selecionado, cujo acerto@1 é 0,608 e 0,520. A diferença entre os dois pontos do PhysBERT (0,042) é do tamanho da diferença entre braços que a coluna pretende mostrar. Além disso, o grupo é de validação, e o intervalo de ±0,045 (481 documentos) não é informado: 0,498 e 0,510/0,520 são indistinguíveis.

**Evidência.** t1c_treino_physbert.log: 'melhor ate agora (acerto@1 0.608) -> phirank-phys-melhor' e, no passo final, 'CONCLUIDO · acerto@1 0.566 ±0.044'. t1c_treino_gte.log: melhor 0,520; final 0,510 ±0,045. Os modelos avaliados são os de sufixo -melhor (t1b_physbert_n2000.json: `rank: /kaggle/working/phirank-phys-melhor`; t1b_gte_n2000.json: `phirank-gte-melhor`). A medida é feita em 481 documentos de validação separados por documento citado (train_rerank.py, `_dividir_por_documento`), não no conjunto de treinamento.

**Correção proposta.** Na Tabela 14, usar o acerto@1 do ponto de verificação avaliado: GTE-base 0,520; PhysBERT 0,608 (o do MiniLM-L6 precisa ser conferido no registro do treino, que não está versionado). Na legenda: "'Acerto@1' é medido em grupos de 8 candidatos de 481 documentos de validação, separados do treinamento por documento citado (intervalo de 95% de ±0,045), no ponto de verificação selecionado por esse mesmo critério e depois avaliado."

### L363 · maior · afirma mais que a evidência · aplicado

> diferindo exclusivamente no corpus de pré-treinamento

**Problema.** Entre GTE-base e PhysBERT variam ao mesmo tempo pelo menos três fatores: corpus, objetivo de pré-treinamento (contrastivo de pares contra linguagem mascarada) e vocabulário/caixa — este último altera inclusive quanto do documento cabe nos 384 tokens. Uma base contrastiva de bi-encoder não é necessariamente boa inicialização de cross-encoder, de modo que 'geral forte' não isola 'capacidade'. O controle sustenta 'não é tamanho nem arquitetura'; não sustenta 'é o corpus'. A mesma frase aparece no resumo (linha 13), na contribuição 5 (linha 35) e na conclusão (linha 637). Falta o braço que isolaria o corpus: BERT-base (linguagem mascarada, geral) ou SciBERT.

**Evidência.** src/phifm/training/rerank.py: `self.tok = AutoTokenizer.from_pretrained(cfg.base)` — cada braço usa o tokenizador e o vocabulário da própria base (os logs t1c_treino_gte.log e t1c_treino_physbert.log baixam vocab.txt distintos; a base de Física é `physbert_cased`, a geral é o gte-base, sem caixa). O próprio rascunho descreve o GTE como modelo de 'aprendizado contrastivo em larga escala' (linha 50; ref. [10], 'Multi-stage Contrastive Learning'), enquanto o PhysBERT [9] é pré-treinado por linguagem mascarada. O que o código de fato iguala (conferido): os seis argumentos de kaggle/t1c_phirank.py, a divisão (677 citados reservados, 13.575/714 grupos), os 12.500 grupos sorteados (11.904 documentos) e o código de treino (git diff 73088dc..96434bc em rerank.py = só `dtype=torch.float32`).

**Correção proposta.** Substituir por: "GTE-base e PhysBERT compartilham número de parâmetros, arquitetura, hiperparâmetros de treinamento, dados, semente e protocolo de avaliação. Diferem na base pré-treinada como um todo — corpus, objetivo de pré-treinamento (contrastivo no GTE, linguagem mascarada no PhysBERT) e vocabulário —, de modo que o experimento exclui tamanho e arquitetura como explicação, mas não separa o corpus dos outros dois fatores. Uma base geral de linguagem mascarada do mesmo tamanho (BERT-base) seria o controle que falta." Ajustar no mesmo sentido o resumo ('idênticos exceto pela base pré-treinada'), a contribuição 5 e a conclusão, e acrescentar o ponto à Seção 6.

### L363 · maior · falta controle ou limitação · aplicado

> os dados, a semente e o protocolo de avaliação

**Problema.** O McNemar da Tabela 14 mede a variância de amostragem das consultas, não a de treinamento. Com +0,0090 de nDCG@10 e um único treinamento por base, não se sabe se a diferença entre bases excede a diferença entre sementes da mesma base; a oscilação de 0,04 entre pontos de verificação a 250 passos de distância sugere que essa variância não é desprezível. A limitação não está declarada para esta seção.

**Evidência.** Um treinamento por base, semente 17 (t1c_resultado.json, campo `fixo`). Dentro de uma mesma execução, o acerto@1 de validação do PhysBERT oscila de 0,606 (passo 6.000) a 0,566 (passo 6.250) — t1c_treino_physbert.log. A Seção 6 declara 'uma semente por braço' apenas para a Seção 4.9 (linha 613); para a 4.8 declara só a taxa de aprendizado.

**Correção proposta.** Acrescentar à Seção 6: "A comparação da Tabela 14 tem um treinamento por base, com uma semente; o teste pareado cobre a amostragem das consultas, não a variância entre treinamentos, que não foi medida." E, na Seção 4.8, depois da tabela: "Cada base foi treinada uma vez."

### L389 · maior · inconsistência interna · aplicado

> o reordenador não traz ao topo um alvo situado entre as posições 100 e 200 do recuperador

**Problema.** A frase contradiz o dado da frase anterior. O reordenador traz 20 dos 195 alvos novos (10,3%) ao top-10; o saldo é +1 porque os distratores adicionais expulsam 19 alvos que já estavam lá. A conclusão correta é 'ganhos e perdas se cancelam', e não 'o reordenador não alcança a faixa 100–200'.

**Evidência.** Recalculado de t1e_resultado.json (`posicoes`): das 195 consultas cujo alvo está entre as posições 100 e 199 do bi-encoder, 20 chegam aos dez primeiros com reordenação @200; outras 19, que estavam nos dez primeiros @100, saem. É o próprio 19×20 da frase anterior (`confronto_das_cadeias`, k=10, @100 contra @200). Como acrescentar candidatos só pode empurrar o alvo para baixo, os 20 ganhos vêm necessariamente da faixa 100–200.

**Correção proposta.** "O teto subiu em 195 consultas e o recall@10, em uma: o reordenador leva 20 desses 195 alvos aos dez primeiros, e os candidatos adicionais expulsam de lá 19 alvos que já estavam — o que a profundidade acrescenta, os distratores retiram."

### L391 · maior · erro estatístico · aplicado

> São cinco medições pareadas, com dois reordenadores

**Problema.** Não são cinco evidências independentes: são quatro configurações (uma duplicada), três delas com o mesmo modelo e os mesmos escores, todas na mesma amostra de consultas. Cinco testes não significativos, com poder de um terço e todos no mesmo sentido, são ausência de evidência, não evidência de ausência. O texto local diz 'não estabelece', mas a contagem 'cinco' sugere acúmulo, e as linhas 35, 524 e 637 já escrevem 'o ganho desaparece' e 'demonstração de que esse acréscimo desaparece', o que os dados não sustentam.

**Evidência.** t1d_resultado.json (antigo, @100) e t1e_resultado.json (@100) trazem o mesmo pareado, 141×168, p=0,13899022845302342; comparando `posicoes`, 0 de 2.000 consultas diferem — é a mesma medição repetida. @50/@100/@200 do T1e saem dos mesmos escores (campo `de_graca`). As cinco usam as mesmas 2.000 consultas. Todas são nominalmente favoráveis à reordenação: 168×141, 174×157, 150×126, 180×152. Recalculado: com 309 discordantes em 2.000 e 54,4% a favor, o McNemar exato tem poder de ~32% (simulação) para o efeito observado; com 80% de poder, a menor diferença detectável de recall@10 é ~0,025, contra 0,0135 observada. Bootstrap do nDCG@10 @100: +0,0091 [−0,0009; +0,0191], 3,8% das reamostras ≤ 0.

**Correção proposta.** "Dois reordenadores foram comparados ao recuperador isolado nas mesmas 2.000 consultas, um deles em três profundidades obtidas dos mesmos escores; em nenhuma configuração o estágio vence (p entre 0,14 e 0,38), embora todas as diferenças sejam nominalmente favoráveis a ele. O teste tem poder de cerca de um terço para o efeito observado e só detectaria, com 80% de poder, diferenças de recall@10 a partir de 0,025: o resultado é 'ganho não estabelecido', não 'ganho nulo'." No resumo, trocar 'nenhuma de cinco medições pareadas' por 'nenhuma das comparações pareadas (dois reordenadores, três profundidades)'; nas linhas 35 e 637, trocar 'desaparece' por 'deixa de ser estabelecido'.

### L391 · maior · desatualizado após a v0.5 · aplicado

> O estágio foi retirado, e o sistema entrega os dez primeiros resultados do bi-encoder.

**Problema.** Depois da v0.5 o mesmo reordenador reapareceu como candidato a voltar ao sistema, com ganho pareado claro sobre o recuperador novo em outra tarefa (perguntas em linguagem natural, top-6). Isso contradiz a generalização das linhas 524 e 526 ('o valor da reordenação é condicional à fraqueza do recuperador', 'só enquanto o recuperador era fraco') e desatualiza a descrição do que o sistema entrega. A retirada vale para o protocolo de citação; não é uma propriedade do estágio.

**Evidência.** Commit 4d6bcb2 (2026-09-29) e data/processed/avaliacao/assistente_busca_dev.json: em 200 perguntas de desenvolvimento do assistente, reordenar os 50 primeiros com o mesmo ΦRank-PhysBERT leva o recall do artigo de origem nas 6 fontes de 0,74 a 0,83 (+0,09 [+0,05; +0,135], 19 ganhos, 1 perda; no estrato primário, 0,733→0,827). O resultado é exploratório (`exploratorio: true`), sem confirmação em teste. scripts/explorar_busca_dev.py descreve o sistema de hoje como 'um resumo hipotético, os 6 primeiros'.

**Correção proposta.** "O estágio foi retirado da cadeia avaliada neste protocolo, que passa a entregar os dez primeiros resultados do bi-encoder. A conclusão é restrita a consultas de contexto de citação: numa medição exploratória posterior, com 200 perguntas em linguagem natural, o mesmo reordenador elevou de 0,74 para 0,83 a fração de perguntas com o artigo de origem entre as seis fontes entregues (+0,09 [+0,05; +0,135]), resultado ainda sem confirmação em conjunto de teste." Qualificar as linhas 524 e 526 no mesmo sentido.

### L323 · menor · erro de referência · pendente

> tomava os *K* primeiros resultados do índice denso, excluída a citação verdadeira

**Problema.** A limitação remete a um procedimento que a seção citada não descreve; o leitor não sabe que os negativos de treino foram filtrados por co-citação, nem de onde vem o 9,1%.

**Evidência.** A Seção 6 (linha 623) diz: 'A Seção 4.7 não mede o falso negativo residual. O filtro de co-citação remove ... 9,1% dos negativos minerados'. `grep co-cita` no rascunho só encontra essa linha: nem a Seção 4.7 nem a 3.4 mencionam o filtro. O código o usa (scripts/train_rerank.py, `--negativos` com os 'limpos'; kaggle/t1c_phirank.py treina com pares_do_recuperador_limpos.parquet).

**Correção proposta.** Acrescentar à linha 332, depois de 'Corrigida a distribuição do grupo de treinamento': "Em ambas as estratégias, os negativos co-citados com o positivo foram removidos antes do treinamento (9,1% dos minerados da fusão), por serem falsos negativos prováveis." E mencionar o filtro na descrição do cross-encoder na Seção 3.4.

### L325 · menor · clareza · pendente

> Correlação de Spearman entre a posição do candidato na fusão e o escore atribuído pelo cross-encoder

**Problema.** Os valores conferem, mas a tabela não diz que são médias de correlações por consulta sobre 60 consultas, nem que o 0,0179 é de outra amostra (300), diferente das mil da Tabela 11. 'Inferior à ordenação aleatória' é verdadeiro e fica mais forte com o valor de referência.

**Evidência.** t1b_inversao_do_reranker.log e t1b_inversao_depois_do_conserto.log: 'universo 88,807 · consultas 60', 'media +0.179 · consultas com rho>0: 83% de 60' e 'media -0.466 · 0% de 60'. O nDCG@10 de 0,0179 (linha 332) vem de t1b_resultado_reranker_invertido.json, com 300 consultas e teto de 0,4433; a ordenação aleatória de 50 candidatos daria 0,4433 × 4,5436/50 = 0,040. O script que gerou os dois logs não está no repositório (git log -S só encontra o texto em documentos).

**Correção proposta.** Legenda: "Média, sobre 60 consultas, da correlação de Spearman por consulta entre...". Linha 332: "...e atinge nDCG@10 de 0,0179 em 300 consultas, abaixo dos 0,040 esperados de uma ordenação aleatória dos mesmos 50 candidatos." Versionar o script do diagnóstico.

### L334 · menor · sem evidência versionada · pendente

> Seis hipóteses alternativas foram testadas e descartadas no diagnóstico

**Problema.** Seis resultados quantitativos do artigo apoiam-se apenas no relato do projeto. Os números do rascunho reproduzem o relato sem erro de transcrição; falta o artefato.

**Evidência.** Os valores (4×10⁻⁶; 113,6 contra 8,5; +0,056 com p=0,17; 14 atributos, AUC 0,575; 16 e 457 documentos; +0,143 ± 0,059) coincidem com a tabela do ESTADO.md (linhas 3447–3454), mas não há arquivo em data/processed/avaliacao/ nem script em scripts/ que os produza.

**Correção proposta.** Versionar o script e a saída de cada uma das seis medições, ou reduzir o parágrafo ao que tem artefato e declarar: "As seis medições foram feitas durante o diagnóstico e não têm artefato versionado; os valores são os registrados à época."

### L347 · menor · número não confere · pendente

> Em todas as profundidades medidas, o BM25 acrescenta cerca de metade

**Problema.** Há quatro profundidades medidas no arquivo; a razão é próxima de metade em três e de 0,7 na quarta (onde as diferenças são de 11 e 16 consultas).

**Evidência.** teto_do_recuperador.json, `orcamento_igual`: +0,051 contra +0,0975 (razão 0,52); +0,026 contra +0,051 (0,51); +0,010 contra +0,020 (0,50); e, na quarta composição, 0,9100−0,9045 = +0,0055 contra 0,9125−0,9045 = +0,0080 (0,69).

**Correção proposta.** "Nas três primeiras profundidades medidas (100, 200 e 500 candidatos do bi-encoder), o BM25 acrescenta cerca de metade do que os mesmos candidatos adicionais do bi-encoder acrescentariam; a 1.000, cerca de dois terços, sobre diferenças de 11 e 16 consultas."

### L355 · menor · falta controle ou limitação · pendente

> Cross-encoders idênticos exceto pela base

**Problema.** A linha do MiniLM vem de um treinamento anterior, em outra máquina e outra versão do código, possivelmente sem precisão mista; 'idênticos exceto pela base' só está verificado para o par GTE/PhysBERT. Como a leitura causal usa esse par, o efeito é pequeno, mas a legenda afirma mais do que os artefatos mostram.

**Evidência.** kaggle/t1c_phirank.py: `CONTROLE = MODELOS / "phirank-rrf-melhor"` — o controle MiniLM é um ponto de verificação pronto, apenas reavaliado ('O controle é reavaliado, não copiado'); só GTE e PhysBERT são treinados na sessão. t1b_resultado.json (agosto) registra o caminho local `models\\phirank-rrf-melhor`. Em rerank.py, precisão mista e atenção sdpa só valem em CUDA (`self.amp = self.dev.type == "cuda" and cfg.amp`). O registro de treino do controle não está versionado.

**Correção proposta.** Na legenda: "Cross-encoders treinados com a mesma receita e bases diferentes. GTE-base e PhysBERT foram treinados na mesma infraestrutura e com o mesmo código; o de base MiniLM-L6 é o modelo da Tabela 11, treinado antes com os mesmos hiperparâmetros e reavaliado aqui."

### L355 · menor · inconsistência interna · aplicado

> nas três execuções

**Problema.** A legenda fala em três execuções e o texto em duas. Além disso, o controle idêntico atesta apenas o avaliador (é um modelo fixo, não retreinado); o que licencia combinar os braços treinados é o código de treino ser o mesmo, e isso é verificável e não está dito.

**Evidência.** Linha 376: 'A combinação das duas execuções'. t1c_resultado.json (controle + PhysBERT, código 73088dc) e t1c_resultado_braco_gte.json (controle + GTE, código 96434bc) são duas execuções com três avaliações. O controle é idêntico nas duas (p=0,1458384109048898, 229 discordantes). O diff de código de treino entre as duas é uma linha (`dtype=torch.float32`), sem efeito em bases de 32 bits.

**Correção proposta.** Legenda: "...fusão de referência com nDCG@10 de 0,1576 nas três avaliações". Linha 376: "A combinação das duas execuções é legítima por duas verificações: o controle, que é um modelo fixo, produziu resultados idênticos campo a campo em ambas (p = 0,14584 sobre 229 discordantes), o que atesta o avaliador; e o código de treinamento difere entre elas em uma única linha, a que fixa a precisão de carga em 32 bits, sem efeito sobre bases já distribuídas nessa precisão."

### L363 · menor · falta controle ou limitação · aplicado

> Apenas a base pré-treinada no domínio supera a fusão.

**Problema.** A leitura 'domínio, não capacidade' depende do contraste entre PhysBERT e GTE, e um braço significativo ao lado de um não significativo não é teste desse contraste. Aqui o limite superior resolve a favor do rascunho; falta apenas reportá-lo.

**Evidência.** A Tabela 14 testa cada base contra a fusão; não há teste direto PhysBERT contra GTE-base, e os arquivos t1b_*_n2000.json não guardam posições por consulta. Limite calculável: recall@10 de 0,2890 contra 0,2635 dá saldo de 51 consultas; os discordantes entre os dois são no máximo 237+220=457; no pior caso (254×203) o McNemar exato dá p=0,019. Contra o MiniLM: saldo 66, no máximo 466 discordantes, p≤0,0026.

**Correção proposta.** Acrescentar: "O contraste direto entre as bases não foi calculado por consulta, mas admite limite: o saldo de 51 consultas em recall@10 a favor do PhysBERT sobre o GTE-base, com no máximo 457 discordantes, dá p ≤ 0,019 no teste exato; contra o MiniLM-L6, p ≤ 0,003."

### L363 · menor · clareza · aplicado

> os seis hiperparâmetros de treinamento, os dados, a semente

**Problema.** A semente é contada duas vezes: os argumentos fixos são seis com ela, cinco sem ela.

**Evidência.** t1c_resultado.json, campo `fixo`: 'max-grupos 12500, grupos 2, n-negativos 7, lr 2e-5, 384 tokens, semente 17' — seis itens, a semente incluída. A linha 94 repete 'os seis hiperparâmetros restantes, os dados, a semente'.

**Correção proposta.** "...os seis argumentos de treinamento (número de grupos, grupos por passo, negativos por grupo, taxa de aprendizado, comprimento máximo e semente), os dados e o protocolo de avaliação"; alinhar a linha 94.

### L378 · menor · erro estatístico · aplicado

> caiu de +0,0091 (p = 0,0081) para +0,0045 (p = 0,086)

**Problema.** A frase pareia uma diferença de nDCG com o p de outra estatística, e lê a passagem de 'significativo' a 'não significativo' como queda, sem testar a diferença entre os ganhos (+45 contra +30 consultas, nas mesmas 2.000). A queda é nominal.

**Evidência.** t1b2_antigo.json: 116×161 em k=10 (saldo +45 consultas, recall@10 +0,0225); t1b2_novo.json: 128×158 (saldo +30, +0,015). Os valores p são do McNemar sobre pertencimento ao top-10; as diferenças entre parênteses são de nDCG@10. Não há teste da diferença entre os dois ganhos; os artefatos do T1b2 não guardam posições por consulta.

**Correção proposta.** "O ganho da reordenação sobre a fusão em recall@10 passou de +0,0225 (161 contra 116 discordantes, p = 0,0081) para +0,0150 (158 contra 128, p = 0,086), e em nDCG@10, de +0,0091 para +0,0045; a diferença entre os dois ganhos não foi testada."

### L389 · menor · afirma mais que a evidência · aplicado

> a decomposição contraria a função do estágio

**Problema.** Os números conferem, mas 167×141 com p=0,154 é empate, não direção contrária; e trazer um alvo da posição 11–100 para os dez primeiros é exatamente o que um reordenador faz. A decomposição mostra onde está o ganho nominal, não que o estágio falha em sua função.

**Evidência.** Recalculado de t1e_resultado.json: entre as 421 consultas com alvo no top-10 nos dois sistemas, 167 descem, 141 sobem, 113 ficam (p=0,154; contribuição ao nDCG de −0,0004). As entradas no top-10 contribuem +0,0395 e as saídas, −0,0300.

**Correção proposta.** "A decomposição localiza o ganho nominal: entre as 421 consultas com o alvo entre os dez primeiros nos dois sistemas, o reordenador rebaixa o alvo em 167 e o eleva em 141 (p = 0,154, contribuição líquida de −0,0004 ao nDCG@10); todo o saldo vem de alvos que entram nos dez primeiros (+0,0395) menos os que saem (−0,0300)."

### L391 · menor · clareza · aplicado

> Estabelecer o ganho nominal de 0,0091, se ele existir, exigiria cerca de 6.640 consultas

**Problema.** O número está certo, mas refere-se ao teste de recall@10, não ao 'ganho nominal de 0,0091' de nDCG@10 citado na frase. É também poder calculado tomando o efeito observado como verdadeiro, o que convém dizer.

**Evidência.** Reproduzido: com 168/309 discordantes a favor, 80% de poder a α=0,05 exige 1.026 discordantes, isto é, 6.639 consultas; 6.640 × 5.096,6 s / 2.000 = 4,70 h (t1d_resultado.json, `total_s`). Essa conta é do McNemar em k=10 (diferença de recall@10 de 0,0135). Para o nDCG@10 (+0,0091, desvio-padrão por consulta de 0,2313, de t1e `posicoes`), 80% de poder exige ~5.070 consultas.

**Correção proposta.** "Tomando o efeito observado como verdadeiro (54,4% de 309 discordantes em k = 10), estabelecê-lo com 80% de poder exigiria cerca de 1.026 discordantes, ou 6.640 consultas — 4,7 horas de T4 por braço —, para um estágio que custa 109 M de parâmetros em serviço."

### Conferido e correto

- Tabela 11 inteira confere com t1b_resultado.json (1.000 consultas, universo 88.807, profundidade 50): BM25 0,067/0,236/0,401/0,1399; bi-encoder 0,055/0,233/0,428/0,1327; fusão 0,068/0,271/0,446/0,1584; fusão+cross-encoder 0,064/0,254/0,446/0,1493; McNemar k=10: p=0,00073, 0,00018 e 0,118.
- Linha 321: a fusão supera os dois recuperadores isolados em k=10 (69×34 e 69×31), e o cross-encoder de base MiniLM não ganha (61×44, p=0,118).
- Tabela 12: +0,179 com 83% de consultas positivas e −0,466 com 0% conferem com t1b_inversao_do_reranker.log e t1b_inversao_depois_do_conserto.log.
- Linha 332: nDCG@10 de 0,0179 confere com t1b_resultado_reranker_invertido.json e é de fato inferior à ordenação aleatória (0,040 esperado com teto 0,4433 e 50 candidatos).
- Tabela 13 inteira confere com t1b2_antigo.json, t1b2_novo.json e t1b2_resultado.json (2.000 consultas, profundidade 100, mesmo código 9a4ec94, reordenador phirank-physbert-melhor); BM25 idêntico nos dois braços.
- Linha 345: 0,6325−0,5160=0,1165 (≈0,117); 0,1688−0,1685=0,0003; 0,1688−0,1666=0,0022; razão ≈7.
- Linha 347: p=1,5×10⁻⁸ (143×62) com o recuperador antigo e p=0,949 (124×122) com o novo; recall@100 da fusão 0,6065 < 0,6325 do bi-encoder; 0,7300 contra 0,6835 no orçamento igual (teto_do_recuperador.json).
- Linha 349: 735 consultas fora dos 100 primeiros, mediana do posto 396, 10 consultas (0,5% das 2.000) além de 10.000 nos dois recuperadores — teto_do_recuperador.json; 735 também recalculado das posições do T1e.
- Linha 353: limiar de Bonferroni de 0,025, McNemar em k=10 contra a fusão da mesma execução e as quatro leituras estão no script do commit 73088dc (2026-08-31 19:49), anterior ao treino (log iniciado depois) e ao resultado versionado.
- Pista 1 (comparações múltiplas): p=0,0062486 (97×140, McNemar exato bilateral, recalculado) fica abaixo de 0,025 (duas variantes), de 0,0167 (três braços) e de 0,0083 (três braços × k=1 e k=10); resiste a Bonferroni até 8 testes.
- Tabela 14, colunas nDCG, Diferença, Discordantes e p: MiniLM 0,1483/−0,0093/229/0,1458; GTE 0,1530/−0,0046/220/0,6371; PhysBERT 0,1666/+0,0090/237/0,0062; fusão 0,1576 e teto 0,4495 nos três arquivos (t1b_controle_n2000, t1b_gte_n2000, t1b_physbert_n2000, t1c_resultado*.json).
- Pista 3 (controle no código), para o par GTE/PhysBERT: kaggle/t1c_phirank.py passa os mesmos argumentos (max-grupos 12500, grupos 2, n-negativos 7, lr 2e-5, max-tokens 384, semente 17) e o mesmo arquivo de negativos; os logs mostram a mesma divisão (677 citados reservados, 13.575/714 grupos), os mesmos 12.500 grupos (11.904 documentos), precisão mista e sdpa nos dois; o diff de rerank.py entre os dois códigos é só `dtype=torch.float32`; ambos carregam como BertForSequenceClassification.
- Tabela 15 e linha 374: 0,0700/0,0690; 0,2675/0,2890; 0,4495/0,4495; 0,1576/0,1666; 97×140 em 237 discordantes, p=0,0062; k=1 com p=0,930 (66×64).
- Linha 376: controle idêntico nas duas execuções (p=0,14584, 229 discordantes, em t1c_resultado.json e t1c_resultado_braco_gte.json); a falha do braço GTE é 'ValueError: Attempting to unscale FP16 gradients' (t1c_treino_gte_falhou.log).
- Linha 378: +0,0091 com p=0,0081 (116×161) e +0,0045 com p=0,086 (128×158); novo contra antigo 82×92, p=0,495; contra o bi-encoder 141×168 (p=0,139) e 157×174 (p=0,379) — t1b2_*.json e t1d_resultado.json. A predição do teto consta no script do commit 9a4ec94, e os critérios do T1d e do T1e constam nos commits fcd7279 e a81d9ff, anteriores aos resultados.
- Tabela 16 inteira confere com t1e_resultado.json, e recall@1, recall@10 e nDCG@10 foram recalculados das posições por consulta (0,1585; 0,1664; 0,1676; 0,1673; tetos 0,5210/0,6325/0,7300).
- Linha 389: 39 consultas divididas em 19 e 20 (p=1,0); teto +195 consultas; recall@10 +1 consulta; bootstrap @100 +0,0091 [−0,0010; +0,0191] reproduzido (obtive [−0,0009; +0,0192] com 20.000 reamostras; @50 e @200 também incluem zero); 421 consultas, 167 rebaixadas e 141 elevadas, p=0,154.
- Linha 391: 6.640 consultas (1.026 discordantes para 80% de poder, α=0,05) e 4,7 h de T4 reproduzidos a partir de 168/309 e do custo medido de 5.096,6 s por 2.000 consultas.
- Referências [25] (BM25), [27] (fusão recíproca de postos), [9] (PhysBERT) e [10] (GTE) correspondem ao uso feito nas Seções 4.7 e 4.8.
- T1h, T1i, o commit 62e42fc e o ADR-0003 não alteram números das Seções 4.7 e 4.8: o recuperador 'novo' de 6 milhões de pares continua sendo o do sistema. O que muda estas seções depois da v0.5 é o commit 4d6bcb2 (achado próprio).

### Não verificável nesta máquina

- Contagem de parâmetros (23 M, 109 M, 109 M), tamanho e caixa dos vocabulários e precisão dos pesos distribuídos (16 bits no gte-base, 32 nos outros): exigem os arquivos de configuração e os pesos das três bases, que não estão nesta máquina.
- Acerto@1 de 0,498 do controle MiniLM e os hiperparâmetros, a máquina e a precisão com que ele foi treinado: exigem models/phirank-rrf-melhor/phirank.json e o registro do treino de agosto, não versionados.
- As seis hipóteses descartadas da linha 334: exigem os scripts e as saídas do diagnóstico, que não estão no repositório; só pude confrontar com o ESTADO.md, que é alegação.
- Correlação de Spearman com a posição na fusão para os reordenadores de base GTE e PhysBERT (necessária para sustentar a explicação por redundância): exige os pontos de verificação e GPU.
- Teste pareado direto PhysBERT contra GTE-base e PhysBERT contra MiniLM: exige posições por consulta, que os arquivos t1b_*_n2000.json não guardam; só o limite superior de p foi calculado.
- Pareado da cadeia com fusão contra a cadeia sem fusão (a regra do BM25) e teste da diferença entre os ganhos com recuperador antigo e novo: t1b2_*.json não guardam posições por consulta.
- Reprodução da Tabela 12: o script não está versionado, e exigiria os dois reordenadores MiniLM e o bi-encoder.
- Que todos os negativos minerados são documentos citados, a fração de 9,1% removida pelo filtro de co-citação e que os grupos de treino vêm dos 50 primeiros da fusão: exigem pares_do_recuperador_limpos.parquet e o arquivo anterior ao filtro.
- Disjunção entre as 2.000 consultas de avaliação e os grupos de treino do reordenador: exige os pares de treino e de validação.
- Que o 'bi-encoder de 400 mil pares' das Tabelas 11, 14 e 15 (phiemb-minilm-melhor) é o modelo descrito: exige o diretório do modelo e seu manifesto.


## Pré-treinamento do encoder (§3.8, §4.9.2, §4.9.3)

### L435 · maior · falta controle ou limitação · aplicado

> uma instabilidade no passo 8.075 descartou 76 lotes e reduziu a taxa de aprendizado à metade no início do decaimento

**Problema.** O artigo chama de 'instabilidade' um acionamento que, pelo próprio diagnóstico da §5.4, provavelmente foi composição do lote — isto é, uma intervenção (rollback, salto de lotes, LR pela metade) causada pelo tratamento, no braço que sustenta os dois números de manchete da §4.9.2 (−0,0040 e +0,084). A direção é contra o tratado, então o +0,084 fica conservador; mas o desfecho primário 'controle à frente' por 0,4 ponto pode dever parte ao mecanismo, e não só ao viés da prova uniforme. Além disso, a §3.8 (l.134) descreve o critério corrigido como se fosse o método de todo braço tratado, o que não vale para a §4.9.2.

**Evidência.** git log: a correção do detector (adfae30, 'o detector confundia a COMPOSICAO do lote com spike') é de 2026-09-19; o braço tratado de 48 M já estava treinado e medido em 2026-09-16 (d77c92c), portanto rodou com o critério ANTIGO (perda média do lote, incluindo os alvos de equação inteira). ESTADO.md l.176 admite: 'O spike do §2.3 a 48 M (passo 8.075) também foi pelo critério de perda — 2,0419 contra 1,9361 — no braço tratado. O mecanismo é provavelmente o mesmo; não foi verificado pela composição do lote.' t2eq_ablacao.json só registra '1 spike (passo 8.075, rollback de 76 lotes, LR pela metade até 8.575)'. Nem a §4.9.2 nem a §5.4 (l.563–571) mencionam que o braço de 48 M foi atingido.

**Correção proposta.** Na l.435: 'A assimetria de execução, desta vez, é contra o tratado: no passo 8.075 o detector acionou pelo critério de perda, descartou 76 lotes e reduziu a taxa de aprendizado à metade no início do decaimento. Esse braço foi treinado antes da correção descrita na Seção 5.4, com a perda julgada sobre todos os alvos; o acionamento é provavelmente do mesmo tipo — composição do lote, e não instabilidade —, o que não foi verificado. A intervenção, portanto, pode correlacionar-se com o tratamento, e o negativo de 0,4 ponto deve ser lido com essa ressalva.' Na l.134, acrescentar: 'Essa correção vale para o braço tratado da Seção 4.9.3; o da Seção 4.9.2 foi treinado com o critério original.' E citar o caso na §5.4.

### L491 · maior · afirma mais que a evidência · aplicado

> o ganho de recuperação do substituto se repete, onze vezes menor que os +0,084 de lá

**Problema.** O intervalo é sobre consultas: fixa os dois modelos treinados e não contém a variância de semente do pré-treinamento continuado nem a do ajuste contrastivo (duas etapas estocásticas encadeadas, uma semente cada). Um efeito de +0,0076 com limite inferior em +0,0013, recall@1 nominalmente a favor do controle e uma assimetria de execução a favor do tratado não sustenta 'se repete' nem, na l.493, 'compraram +0,010'. O parágrafo diz 'limite inferior próximo de zero', mas a ressalva de que o intervalo não cobre o ruído de treino só aparece, parcialmente, na l.613, longe do número — e a frase forte é a que sobe para a contribuição 6 (l.36) e a conclusão (l.639, 'real e pequeno').

**Evidência.** t2eq_cpt_emb_comparacao.json: diferença 0,00757, ic95 [0,00128; 0,01388] por bootstrap sobre 2.000 itens; recall@1 controle 0,3635 > tratado 0,3620 (63×60, p=0,857, recalculado: 0,8570); ressalvas do próprio arquivo: 'Uma semente por braço' e 'o controle do CPT teve 2 spikes de norma com rollback de 182 lotes e o tratado nenhum — assimetria a favor do tratado'. Não há no repositório nenhuma repetição com outra semente (nem do pré-treinamento, nem do ajuste contrastivo) que estime o ruído de treino.

**Correção proposta.** 'Entre os braços, a estimativa pontual favorece o tratado: +0,0076 de nDCG@10 [+0,0013; +0,0139], com recall@1 empatado (63 contra 60 discordantes, p = 0,86). O intervalo é por reamostragem de consultas e não inclui a variância de semente do pré-treinamento nem a do ajuste, que não foram medidas; somada à assimetria de execução a favor do tratado, a diferença é compatível com o ganho do substituto, onze vezes menor em valor absoluto, e não o estabelece.' Ajustar no mesmo sentido as l.36, 493 e 639.

### L491 · maior · falta controle ou limitação · aplicado

> E o GTE-base supera o ModernBERT-base, ambos sem qualquer pré-treinamento no domínio, por +0,0694

**Problema.** O numerador do 'sete vezes' não é 'escolha da base geral' em sentido neutro: é a diferença entre uma base só-MLM e uma base que já passou por pré-treinamento contrastivo de recuperação em grande escala (cujos pares, segundo a publicação do GTE, incluem literatura científica — o que tornaria 'sem qualquer pré-treinamento no domínio' impreciso; a composição não é verificável neste repositório). O denominador é o ganho de um pré-treinamento MLM de 0,4 B tokens. A comparação mostra que o estágio contrastivo prévio vale mais que o MLM de domínio, não que 'trocar a base' em geral vale sete vezes; a §4.9.3 e a l.102 ('diferem entre si apenas na base') não declaram esse confundimento.

**Evidência.** Aritmética confere: 0,06935/0,01027 = 6,75 ('cerca de sete'), mesma métrica, mesmos 2.000 itens, mesmos 200 mil pares (t2eq_cpt_emb_comparacao.json). Mas o próprio artigo descreve o GTE como modelo 'treinado com aprendizado contrastivo em larga escala' (l.50) e diz que 'o ajuste ensina sobretudo a tarefa de recuperação, que uma base contrastiva geral já sabe' (l.300); base_gte_base.json mostra o GTE-base, sem ajuste algum, com recall@10 de 0,2755 no universo de 88.807. O ModernBERT-base é um modelo só de linguagem mascarada. A ressalva do arquivo de evidência ('uma variável, a BASE') só lista parâmetros e contexto.

**Correção proposta.** Na l.491: 'E o GTE-base — que, ao contrário do ModernBERT-base, já passou por pré-treinamento contrastivo de recuperação em larga escala, sobre pares que incluem texto científico — supera o ModernBERT-base por +0,0694 [+0,0580; +0,0812]; nenhum dos dois recebeu pré-treinamento neste corpus.' Na l.493, acrescentar: 'A diferença entre as bases reúne tamanho, contexto e, sobretudo, o estágio contrastivo prévio do GTE; não isola nenhum deles.'

### L130 · menor · inconsistência interna · aplicado

> o ModernBERT-base, que nunca viu mascaramento de equações, acerta 0,8765 em tokens de equação contra 0,7480 em prosa

**Problema.** O parágrafo descreve o protocolo ('2.000 sequências sorteadas… contexto de 1.024') e, na frase seguinte, cita uma linha de base medida em outro protocolo (contexto 512, outra fatia). O argumento (vantagem de equação sem tratamento) continua válido, mas o número não é do protocolo descrito.

**Evidência.** mlm_modernbert_base.json: protocolo.contexto = 512, sem 'amostragem: sorteada', dados phienc_aval_modernbert (9 partes excluídas), 153.936 tokens. As medidas das Tabelas 18 e 21 usam contexto 1.024, amostragem sorteada, ~307,9 mil tokens (t2eq_cpt_mlm_controle_uniforme.json: dados phienc_aval_modernbert_cpt, 2 partes excluídas).

**Correção proposta.** Acrescentar '(medido com contexto de 512, em fatia distinta)' após '0,1286', ou remedir a base no protocolo de 1.024.

### L399 · menor · clareza · aplicado

> a entropia cruzada inicial deve igualar o logaritmo do tamanho do vocabulário. O valor medido é 10,7343, contra ln(40.960) = 10,6204

**Problema.** O texto diz 'deve igualar' e apresenta um valor que não iguala, sem tolerância nem explicação (a inicialização aleatória não produz distribuição exatamente uniforme, e o excesso é esperado). A frase seguinte, 'único indicador… nenhum outro sintoma observável', é mais forte do que a evidência apresentada.

**Evidência.** ln(40.960) = 10,6204 (confere); 10,7343 − 10,6204 = 0,114 nat (1,1% acima).

**Correção proposta.** '…deve ficar próxima do logaritmo do tamanho do vocabulário, com pequeno excesso devido à inicialização. O valor medido é 10,7343, contra ln(40.960) = 10,6204 — 1% acima.'

### L423 · menor · número não confere · aplicado

> Uma equação típica de 75 tokens corresponde a cerca de 17% do orçamento de mascaramento de uma janela de 1.024.

**Problema.** 75 em 307 é 24%, não 17%; a medida do próprio treino (≈30% por exemplo tratado) é coerente com 24% e não com 17%. O erro está na fonte e foi copiado para o artigo.

**Evidência.** Taxa efetiva de mascaramento 0,3000 (l.395): orçamento = 1.024 × 0,30 = 307 tokens; 75/307 = 0,244. O mesmo cálculo a 8.192 dá 79/2.457 = 3,2%, que é o '~3%' de ESTADO.md l.1341 — de onde vem também o '~17%' ('75 tokens num orçamento de 307 é ~17%'). ESTADO.md l.1233 registra 9,7% de tokens de equação entre os mascarados com 32,3% de exemplos tratados, i.e. ~30% por exemplo tratado.

**Correção proposta.** 'Uma equação típica de 75 tokens corresponde a cerca de um quarto do orçamento de mascaramento de uma janela de 1.024 (307 tokens).'

### L466 · menor · clareza · pendente

> Os dois braços partem do ModernBERT-base e diferem apenas em `p_equacao`

**Problema.** Os braços diferem em p_equacao, na revisão do código do detector e no número de execuções. O texto da l.468 relata a re-execução, mas não diz que o controle ficou com o código anterior nem por que isso é inócuo; 'refeito do zero' também é ambíguo numa seção que contrapõe 'do zero' a 'continuado'.

**Evidência.** ESTADO.md l.70–83 e 163–174: o controle foi treinado antes da correção adfae30 e não foi refeito; o tratado foi refeito depois, com o detector corrigido (outra revisão do código) e após uma primeira execução interrompida no passo 4.489. A equivalência do critério no controle é argumentada ('a diferença é de arredondamento'), não medida. Os valores 7,2/7,7 e 0,64/0,61 dos acionamentos de norma (l.468) só constam do ESTADO.md.

**Correção proposta.** '…diferem em `p_equacao` (Seção 3.4) e, por força da correção da Seção 5.4, na revisão do detector: o controle não foi refeito, porque nele todos os alvos são uniformes e os dois critérios de perda coincidem.' E trocar 'refeito do zero' por 'reiniciado a partir da base'.

### L491 · menor · clareza · aplicado

> onze vezes menor que os +0,084 de lá

**Problema.** A razão compara diferenças absolutas de nDCG@10 em pontos de partida distintos, com volumes de pré-treinamento distintos (0,6 B contra 0,4 B) e regimes distintos (do zero contra continuado). O número está certo; a leitura 'o efeito encolhe à medida que a base se fortalece' (l.639) não é de uma variável.

**Evidência.** 0,08397/0,00757 = 11,09 (confere). Mas as bases de comparação diferem: controle a 0,3872 (48 M, do zero, 0,6 B tokens) contra 0,5297 (150 M, continuado, 0,4 B tokens). Em termos relativos: 21,7% contra 1,4%, razão de 15.

**Correção proposta.** '…onze vezes menor em valor absoluto que os +0,084 de lá (1,4% contra 22% em termos relativos), entre experimentos que diferem também em volume de pré-treinamento (0,4 contra 0,6 bilhão de tokens) e em regime.'

### L493 · menor · erro estatístico · aplicado

> cinco vezes a largura dos intervalos da tabela

**Problema.** A frase só vale (arredondando) para o intervalo mais estreito, e compara uma diferença não pareada com a largura de intervalos de outras diferenças pareadas. O intervalo dessa comparação poderia ter sido calculado diretamente.

**Evidência.** A Tabela 22 não tem intervalos; os do parágrafo anterior têm larguras 0,0126 (entre braços), 0,0144 (contra a barra) e 0,0232 (GTE contra barra): 0,0591/0,0126 = 4,7; /0,0144 = 4,1; /0,0232 = 2,5.

**Correção proposta.** '…diferença entre estimativas pontuais, não registrada como comparação, de quatro a cinco vezes a largura dos intervalos pareados entre os braços.' Ou reportar o bootstrap pareado GTE-base − tratado, marcado como exploratório.

### L493 · menor · desatualizado após a v0.5 · pendente

> O GTE-base sem pré-treinamento fica 0,059 acima do melhor braço

**Problema.** Consequência específica do T1h para esta seção (a ausência do T1h em geral já é achado conhecido): a 200 mil pares, uma base contrastiva de 23 M empata com o ModernBERT-base de 150 M com e sem pré-treinamento continuado, e uma de 33 M supera o melhor braço por ~0,033. Reforça a leitura da l.493 e a ressalva do achado sobre o estágio contrastivo; sessões diferentes, logo só indicativo.

**Evidência.** t1h_comparacao.json (posterior à v0.5, mesmos 200 mil pares, mesmo protocolo n=2000/semente 17/192 tokens, outra sessão): MiniLM-L6 de 22,7 M → nDCG@10 0,5289; gte-small de 33,4 M → 0,5706. Tabela 22: ModernBERT-base 0,5270, CPT controle 0,5297, CPT tratado 0,5373.

**Correção proposta.** Acrescentar à l.493, ao incorporar o T1h: 'Medidas posteriores no mesmo protocolo, em outra sessão, situam bases contrastivas de 23 M e 33 M em 0,529 e 0,571 com os mesmos 200 mil pares — ao nível do ModernBERT-base e acima do melhor braço do pré-treinamento continuado.'

### L639 · menor · clareza · aplicado

> a supera por um valor sete vezes maior

**Problema.** Na conclusão, 'sete vezes maior' fica sem denominador explícito e o antecedente gramatical mais próximo (o ganho entre braços) daria nove vezes. As l.36, 583 e 599 usam o denominador certo (o pré-treinamento continuado inteiro, +0,010).

**Evidência.** 0,06935/0,01027 = 6,75 (GTE − base contra tratado − base); mas a frase vem logo após 'repete o ganho de recuperação, onze vezes menor', cujo referente é +0,0076: 0,06935/0,00757 = 9,2.

**Correção proposta.** '…a supera por 0,069, cerca de sete vezes o que o pré-treinamento continuado inteiro acrescentou à base (+0,010).'

### Conferido e correto

- Tabela 18 (t2eq_ablacao.json e recálculo a partir de acertos_por_token em t2eq_mlm_*_uniforme.json): 0,8694/0,8643, 0,6944/0,6933, diferença das diferenças −0,0040 [−0,0058; −0,0022] (recalculado: −0,003996); 108.388 e 199.474 tokens; equação inteira 0,0704→0,1984, +0,128 [+0,122; +0,134], 104.141 tokens; fração tratada 0,5381. Mesmas posições e regiões nos dois braços (máscaras de região idênticas, mesmos índices sorteados).
- Tabela 21 (t2eq_cpt_ablacao.json e recálculo por token): 0,9032/0,9030, 0,7783/0,7784, −0,00039 [−0,00161; +0,00087] (recalculado: −0,000386); 0,0266→0,1999, +0,173 [+0,167; +0,179], 104.624 tokens; 'encolhe dez vezes' = 10,3.
- Tabela 19: nDCG@10 sem ajuste 0,0171/0,1391, +0,122 [+0,109; +0,135]; recall@1 0,0055/0,0785, p = 3,56e-36 (8×154, recalculado); após ajuste 0,3872/0,4712, +0,084 [0,071; 0,097] (0,08397 [0,07077; 0,0972]); 106×260, p = 4,66e-16 (recalculado); recall@10 0,589/0,674; sonda 0,333/0,375, 12×15, p = 0,701 (recalculado).
- Tabela 20: 0,2215/0,5890/0,3391/0,3872; 0,2985/0,6740/0,4203/0,4712; 0,3540/0,7200/0,4774/0,5270; ModernBERT − tratado = 0,0558 [0,0426; 0,0685]. 'Três braços medidos na mesma sessão' confirmado pela mensagem do commit 3249698.
- Tabela 22 e l.491: todos os valores das quatro bases; +0,0076 [+0,0013; +0,0139]; 63×60, p = 0,857; +0,0103 [+0,0032; +0,0176]; +0,0694 [+0,0580; +0,0812] (t2eq_cpt_emb_comparacao.json e t2eq_emb_gte_contra_barra.json).
- 'Onze vezes': 0,08397/0,00757 = 11,09. 'Sete vezes': 0,06935/0,01027 = 6,75 (mesma métrica, mesmos 2.000 itens, mesmos 200 mil pares). '+0,010' e '+0,069' da l.493; 0,059 = 0,5964 − 0,5373.
- 'Seis dias antes': barra do ModernBERT-base registrada em 2026-09-17 (3249698), sessão da secundária em 2026-09-22/23 (370f3e0); valor 0,5270 reproduzido.
- L.460: fator de oito (0,1391/0,0171 = 8,1); cossenos 0,972 e 0,9714; centralizado 0,0214 contra 0,1427; braço A 0,0296; ganho de 22% (0,084/0,3872 = 21,7%); 120.002 documentos citados (só em ESTADO.md).
- L.435: display −0,0055 [−0,0076; −0,0034]; em linha −0,0015 com intervalo cruzando zero (t2eq_exploratoria_regioes.json); '0,4 ponto contra 12,8'.
- L.130: 0,8765, 0,7480, 0,1286 (mlm_modernbert_base.json). L.136: ModernBERT-base 0,222, SciBERT 0,208, PhysBERT 0,028, MiniLM-L6 0,000, piso de superfície 0/72 (sonda_tensorial_calibracao.json).
- Pista 3: t2eq_cpt_spikes_lotes.json confirma os três acionamentos da 1ª execução do tratado a 150 M (passos 429, 1.699, 4.489) com z = 3,73, 3,00 e 4,09 e dois acima do máximo da referência (3.127) — bate com a §5.4. Essa execução foi descartada; os números reportados da §4.9.3 vêm da 2ª execução, sem acionamentos, de modo que o detector não contaminou o braço tratado de 150 M. A assimetria remanescente (2 acionamentos de norma no controle, 182 lotes) está declarada na l.468 e na l.613.
- L.423: fração tratada ~0,55 a 1.024 e 'cerca de um terço' (0,6 × 0,5381 = 0,32) são coerentes com o código (kaggle/t2eq_tratado.py) e com o valor medido.
- L.403 e l.613 declaram uma semente por braço; l.613 declara as três assimetrias de execução e que a da §4.9.3 'poderia explicar parte do efeito'.

### Não verificável nesta máquina

- O lote do passo 8.075 do braço tratado de 48 M: scripts/diagnosticar_spikes_lotes.py poderia reproduzir a composição (o fluxo é função de semente e passo), mas os dados de pré-treinamento não estão nesta máquina. Os limiares 2,0419 contra 1,9361 só constam do ESTADO.md.
- Os registros de treinamento (phienc.json, logs) dos quatro braços: passos dos acionamentos, normas 7,2/7,7 contra medianas 0,64/0,61, 76 e 182 lotes descartados, desvio padrão da perda 0,077 contra 0,037 — só no ESTADO.md e nas ressalvas dos JSON.
- A fração tratada de 0,549 da Tabela 21: o comparador a lê de mascaramento.fracao_tratada do treino, que não está versionado; o valor coincide, com três casas, com a previsão da sonda feita para outra fatia/tokenizador (0,549, em kaggle/t2eq_tratado.py), enquanto a de 48 M tem quatro casas (0,5381). Vale conferir no phienc.json do CPT tratado que não é constante herdada.
- L.395: 2.001.270.262 tokens, 143.810 documentos, 244.295 sequências, 37,8%/38,8%, 25,0%/25,2%, fração tratada 0,903; l.397: medianas 79 e 7 em 120 documentos; l.399: perda inicial 10,7343; l.401: 0 de 120 e 91,7% — exigem o corpus; constam apenas do ESTADO.md.
- L.403: reexportação com diferença de logits nula e versões do transformers (5.0 / 4.48) — exige os checkpoints.
- L.423: 'conferido por comparação da árvore sintática das duas células' e '42% das janelas sem equação em display' — não reexecutado.
- Composição dos dados de pré-treinamento contrastivo do GTE-base (se incluem pares científicos): afirmação da literatura, não verificável no repositório.
- Variância de semente do pré-treinamento e do ajuste contrastivo: não existe nenhuma repetição no repositório; não há como dimensionar o +0,0076 contra o ruído de treino.


## Recuperação por equação (§4.10)

### L497 · maior · afirma mais que a evidência · aplicado

> os alvos são os demais documentos que contêm a mesma forma canônica de conteúdo

**Problema.** Circularidade de construção: 'mesma equação' é definida pelo canonicalizador do projeto, que por desenho só colapsa variação de superfície de tokens. O estrato 'notacional' contém, portanto, apenas variantes que um normalizador de expressões regulares desfaz — justamente as que a tokenização em subpalavras de um encoder absorve. A premissa na forma forte (equivalência simbólica com distância léxica) não foi testada; documentos com a mesma equação em grafia mais distante entram como distratores. A linha 37 (contribuição 7) chama isso de 'refutação da premissa', o que um conjunto próprio com esse gabarito não sustenta.

**Evidência.** src/phifm/core/latex/canonical.py: a canonização só aplica ambientes/delimitadores, \label/\tag/\nonumber, espaçamento, \dfrac→\frac, \left/\right, \le→\leq, \ge→\geq, \ne→\neq, \to, \cdot condicional, chave unitária e espaço; a §2 do módulo REJEITA de propósito \epsilon↔\varepsilon, ordem de operandos, posição de índices e simplificação. Executado: equivalentes('\frac{a}{b}=c','a/b=c') = False; equivalentes('T = 2\pi\sqrt{\frac{L}{g}}','T=2\pi(L/g)^{1/2}') = False; equivalentes('x+1=y','1+x=y') = False. DOC-11 §6.3-medido parte 2, ponto 3: formas 'canonicamente equivalentes e lexicamente distantes ... praticamente não aparecem em pares de artigos reais — teriam de ser GERADAS'.

**Correção proposta.** Acrescentar após a frase: "'Mesma equação' significa aqui identidade sintática após a canonização do projeto, que desfaz apenas variação de superfície — ambiente, rótulos, espaçamento, `\left`/`\right`, `\dfrac`, sinônimos como `\le`/`\leq` e chaves unitárias — e deliberadamente não iguala `\frac{a}{b}` a `a/b`, nem reordena operandos. Variação notacional mais distante não está no gabarito, e documentos que a contenham contam como distratores." Na linha 37, trocar "a refutação da premissa de que recuperação densa perde o casamento simbólico" por "a constatação de que, nele, o bi-encoder supera o BM25 inclusive no estrato de variação notacional de superfície".

### L497 · maior · sem evidência versionada · aplicado

> Um argumento recorrente para sistemas híbridos em domínios matemáticos

**Problema.** A premissa que a seção diz refutar é atribuída à literatura sem referência, e a seção não cita nenhum trabalho de recuperação de fórmulas, área com tarefas e sistemas estabelecidos (estruturais e densos). Sem isso, a contribuição 7 refuta uma hipótese de desenho do próprio projeto, e não se sabe como o conjunto se compara aos existentes.

**Evidência.** grep no rascunho por ARQMath, NTCIR, Tangent, Approach0, 'formula retrieval', 'math-aware': nenhuma ocorrência; a frase não tem citação. A origem documentada da premissa é interna: DOC-11 §6.3 ('Mede diretamente a capacidade que recuperação densa costuma perder: casamento simbólico exato', hoje riscada) e a linha 641 do próprio rascunho ('a premissa com que o conjunto foi desenhado').

**Correção proposta.** "O conjunto foi desenhado sob a hipótese, adotada no projeto, de que a recuperação densa perde o casamento simbólico que a busca léxica preserva." Acrescentar à Seção 2 um parágrafo sobre recuperação de fórmulas (as tarefas ARQMath e NTCIR-12 MathIR e os sistemas estruturais como Tangent-S e Approach0, com as referências conferidas na fonte) e dizer que nenhum sistema estrutural foi medido aqui.

### L503 · maior · inconsistência interna · aplicado

> | Sistema | recall@1 | recall@10 | MRR | recall@10 − BM25 |

**Problema.** Na profundidade 50 nenhum modelo denso supera o BM25 e um deles fica significativamente abaixo; na primária, um dos três empata. A frase 'a premissa não se sustenta para estes modelos' (linha 510) vale para dois dos três em recall@10 e para nenhum em recall@50. A coluna existe na tabela do DOC-11 e foi retirada no artigo, que em outras seções trata o recall profundo do recuperador como o limite da cadeia.

**Evidência.** pb_formula_resultado.json traz recall_50, omitido da Tabela 23: BM25 0,9790; bi-encoder do sistema 0,9765; ModernBERT 0,9695; GTE-base 0,9545. Para o GTE-base: 91 erros contra 42 do BM25 em 2.000 itens, pior caso do teste exato p = 2,6×10⁻⁵, a favor do BM25; e a primária dele é −0,0100 [−0,0245; +0,005] ('cruza_zero': true). O desfecho do script é lido só no 'melhor_denso' (max entre três, sem correção de multiplicidade).

**Correção proposta.** Incluir a coluna recall@50 (0,9790; 0,9765; 0,9695; 0,9545) e reescrever: "Dois dos três modelos densos superam o BM25 em recall@10; o GTE-base empata (−0,0100 [−0,0245; +0,005]). Em recall@50 nenhum deles supera o BM25, e o GTE-base fica abaixo (0,9545 contra 0,9790). O desfecho registrado é lido no melhor dos três modelos; com correção de Bonferroni para três comparações o intervalo do bi-encoder do sistema continua excluindo zero." (conferir a última frase com o bootstrap a 98,33%).

### L510 · maior · falta controle ou limitação · aplicado

> A premissa não se sustenta para estes modelos

**Problema.** O controle léxico não é 'casamento simbólico exato': é um saco de nomes de macro e letras feito para texto, que descarta operadores, chaves, índices e estrutura, e depende do espaço entre letras. Ele erra a primeira posição em 5,35% dos itens cuja grafia é idêntica, onde um casador de cadeia faria 1,0. E o casador simbólico de fato — o hash da forma canônica do próprio projeto — faz 1,0 em todos os estratos por construção, porque é ele que define o gabarito. O que a Tabela 23 sustenta é 'o bi-encoder supera este BM25', não 'o denso não perde casamento simbólico'. A frase 'assimetria a favor do BM25' conta só a truncagem e omite as assimetrias contra ele.

**Evidência.** src/phifm/eval/hibrido.py: _TOKEN = r"\\?[A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)*|\d+(?:\.\d+)?". Executado: tokenizar('a = b + c - d') == tokenizar('a + b = c d') == ['a','b','c','d']; 'm a' vira ['m','a'] e 'ma' vira ['ma']; '\left( x+y \right)^2 = k T' vira ['\left','x','y','\right','2','k','t'] contra ['x','y','2','kt'] de '(x+y)^2=kT'. data/processed/avaliacao/pb_formula_identica.json: BM25 recall_1 = 0,9465 no estrato em que TODO alvo tem a grafia byte a byte da consulta (o denso faz 0,9990). k1=1,5 e b=0,75 não ajustados (docstring da classe BM25).

**Correção proposta.** Substituir por: "Neste conjunto, o bi-encoder de 23 M supera o controle léxico. O controle é um BM25 com tokenização de texto — conserva nomes de macro, letras e números, e descarta operadores, chaves e estrutura —, com k1 e b nos valores padrão; não é um casador simbólico. Ele acerta a primeira posição em 0,9465 dos itens de grafia idêntica, onde um casador de cadeia acertaria todos, e um casador pela forma canônica acertaria todos os itens dos três estratos por construção, pois é a forma canônica que define o gabarito. A truncagem a 192 tokens pesa contra os modelos densos; a tokenização pesa contra o BM25." Acrescentar, se possível, uma linha de controle com BM25 sobre tokens de LaTeX (operadores e chaves preservados).

### L510 · maior · desatualizado após a v0.5 · aplicado

> apesar de uma assimetria a favor do BM25

**Problema.** A §4.10 foi escrita antes do aceite e ainda se apresenta como teste da premissa geral. A especificação revista é mais contida que o texto: reporta o resultado como favorável ao sistema, reconhece que o conjunto não contém a variação dura que a premissa original visava e tira uma consequência de projeto (o estágio de casamento simbólico fica sem evidência a favor, não refutado). Nada disso está na seção. O rótulo 'Bi-encoder do sistema' continua correto depois do T1h/T1i e do commit 62e42fc.

**Evidência.** git ea60df2 (2026-09-23 12:03, dez horas depois da v0.5 em 5056f9d 02:04): DOC-11 §6.3 revisto e ACEITO — 'o benchmark é reportado como medida em que o sistema vai bem, e não como lacuna que justifique um componente'; adotados os pontos 1, 2 e 4, e o ponto 3 registra que a intenção original exigiria variação gerada (sintética), não executada. DOC-13 (mesmo commit): 'A perna de fórmula ... não se justifica por esse número. Ela teria de se justificar por consultas que usuários escrevem'.

**Correção proposta.** Acrescentar ao fim do parágrafo: "A especificação do conjunto foi revista em consequência: ele passa a ser reportado como medida de recuperação por equação em que o sistema vai bem, e não como teste do que a recuperação densa perde. Para esse teste seriam necessárias formas canonicamente equivalentes e lexicamente distantes, que quase não ocorrem entre pares de artigos reais e teriam de ser geradas. Para o sistema, a consequência é que um estágio de casamento simbólico não se justifica por este número; teria de se justificar por consultas escritas por usuários, que não foram medidas."

### L510 · maior · erro estatístico · aplicado

> por 15 pontos em recall@1

**Problema.** O número que vai ao resumo (0,860 contra 0,712) e ao texto é de uma métrica que não é a primária registrada e não traz intervalo nem teste. O efeito registrado é seis vezes menor (2,5 pontos em recall@10). A vantagem em recall@1 é robusta — significativa em qualquer configuração de discordantes —, mas o texto não diz que ela é exploratória nem a quantifica.

**Evidência.** scripts/avaliar_pb_formula.py (regra de c1b02f0, 2026-09-17): 'PRIMÁRIA: recall@10 por item ... IC por bootstrap pareado por item'. pb_formula_resultado.json só tem teste para recall@10: +0,0250 [0,0115; 0,039]. Para recall@1 (0,8595 − 0,7115 = 0,148) não há IC nem teste em nenhum artefato; as séries por item (.parcial.json) não estão versionadas. Limite calculado a partir das marginais: 577 erros do BM25 contra 281 do denso em 2.000 itens; no pior caso de discordância (erros disjuntos) o teste binomial exato dá p = 2,6×10⁻²⁴; IC95 não pareado [0,123; 0,173].

**Correção proposta.** "Pela medida primária registrada, recall@10, o bi-encoder fica +0,0250 [+0,0115; +0,039] acima do BM25. Em recall@1, medida não registrada como primária, a distância é de 14,8 pontos (0,8595 contra 0,7115; McNemar exato, p = …, com os discordantes do arquivo por item)." Rodar o McNemar sobre as séries por item e versionar o resultado; no resumo, dar o recall@10 com o intervalo ao lado do recall@1.

### L510 · maior · erro estatístico · aplicado

> cresce monotonicamente com a variação notacional

**Problema.** A monotonia só existe em recall@1, e o primeiro degrau (idêntica → superficial) não se distingue do ruído. Na métrica registrada a ordem não é monotônica, e em recall@50 o BM25 supera o bi-encoder do sistema no estrato superficial de forma significativa — dado omitido. Além disso, a previsão registrada falava em vantagem do BM25 nos controles, e em recall@1 o BM25 não tem vantagem em estrato nenhum: perde por 5 pontos até na grafia idêntica, o que indica fraqueza do controle e não confirmação do corte.

**Evidência.** pb_formula_identica.json, pb_formula_superficial.json, pb_formula_resultado.json. recall@1 denso − BM25: +0,0525, +0,0645, +0,148; a diferença entre os dois primeiros é 0,012 com erro-padrão ≈ 0,010 (amostras independentes de 2.000 itens), z ≈ 1,2. Na métrica registrada, recall@10: +0,0020 [0,0005; 0,004], −0,0065 [−0,015; 0,0015] (desfecho 'EMPATE' no arquivo), +0,0250 — não monotônica. Em recall@50: +0,001, −0,0115 (0,9865 contra 0,998: 27 erros contra 4; pior caso do teste exato p = 3,4×10⁻⁵, a favor do BM25), −0,0025. Previsão registrada no script: 'a vantagem do BM25 seja MAIOR neles'.

**Correção proposta.** "Nos estratos de controle, com 2.000 itens cada, a diferença em recall@10 é de +0,0020 [+0,0005; +0,004] na grafia idêntica e de −0,0065 [−0,015; +0,0015] na superficial, contra +0,0250 na notacional; em recall@1 a vantagem do bi-encoder é de +0,053, +0,065 e +0,148, e só o terceiro valor se distingue dos outros dois. Em recall@50 o BM25 fica à frente no estrato superficial (0,998 contra 0,9865). A previsão registrada — vantagem do BM25 nos controles — não se verificou em recall@1 em nenhum estrato."

### L497 · menor · clareza · pendente

> Das 202.365.265 equações extraídas de 828.601 documentos, resultaram 374.739 itens.

**Problema.** A frase salta de 202 milhões de equações para 375 mil itens sem dizer os filtros, que definem o que o conjunto mede: 'de conteúdo' não é definido, o teto por documento descartou mais formas (421 mil) do que as que viraram item, a consulta é sempre a grafia mais curta, e o 'recall@k' da Tabela 23 é taxa de acerto de qualquer alvo, com 1,3 alvo por item.

**Evidência.** pb_formula_montagem.json: de_conteudo 29.279.808 (14,5% das equações); formas 27.185.523; descartadas_um_documento_so 26.389.057 (97,1%); descartadas_comuns_demais 513; descartadas_teto_por_documento 421.214; itens 374.739 (27.185.523 − 26.389.057 − 513 − 421.214 confere); alvos_por_item média 1,298, máx. 19. formula.py: MIN_CARACTERES = 40, RELACAO, MAX_DOCUMENTOS = 20, MAX_ITENS_POR_DOCUMENTO = 3; consulta = grafia mais curta entre os documentos; 'recall@k = fração dos itens com ao menos UM alvo no top-k'.

**Correção proposta.** "Das 202.365.265 equações extraídas de 828.601 documentos, 29.279.808 são de conteúdo — contêm uma relação e têm ao menos 40 caracteres na forma canônica —, em 27.185.523 formas distintas, das quais 97,1% ocorrem em um documento só. Mantidas as formas presentes em 2 a 20 documentos, com no máximo três itens por documento de consulta, resultaram 374.739 itens, com 1,3 alvo em média. A consulta é a grafia mais curta da forma; recall@k é a fração de itens com ao menos um alvo entre os k primeiros."

### L499 · menor · clareza · aplicado

> 53,1% dos itens têm alvo com grafia idêntica byte a byte à da consulta

**Problema.** As três porcentagens do texto não somam 100% porque o estrato 'mista' (2,1%) foi omitido, e 'têm alvo com grafia idêntica' se lê como 'algum alvo', quando a definição é 'todos os alvos'.

**Evidência.** pb_formula_diagnostico.json: identica 0,5307; superficial 0,3698; mista 0,0212 (7.948 itens); notacional 0,0783. Soma das três frações citadas: 97,9%. formula.estrato_de_grafia: 'identica' exige que TODO alvo coincida; 'notacional', que NENHUM coincida.

**Correção proposta.** "em 53,1% dos itens todos os alvos têm grafia idêntica byte a byte à da consulta, em 37,0% todos diferem apenas por marcação superficial — …—, em 2,1% há alvos dos dois tipos, e só em 7,8% nenhum alvo coincide, de modo que o item exige variação notacional de fato"

### L501 · menor · falta controle ou limitação · aplicado

> 20.000 documentos no conjunto de candidatos (838.198 equações)

**Problema.** Os valores absolutos de recall valem para um conjunto de candidatos de 2,4% do corpus; no corpus inteiro (da ordem de 35 milhões de equações de conteúdo) seriam menores, e a diferença entre sistemas pode mudar. O tamanho usado difere do padrão do script registrado, e o critério da escolha não está documentado.

**Evidência.** pb_formula_montagem.json: ids_distintos_no_corpus 828.601, pool_documentos 314.161; 20.000/828.601 = 2,4%. scripts/avaliar_pb_formula.py: '--pool' default=5000 (também na versão pré-registrada c1b02f0); o help diz que o tamanho se calibra rodando só o BM25. Nenhum registro versionado da escolha de 20.000 foi encontrado em ESTADO.md ou no DOC-11.

**Correção proposta.** Acrescentar à legenda: "O conjunto de candidatos tem 2,4% dos documentos do corpus; os valores absolutos não se transferem ao corpus inteiro." E registrar no texto como os 20.000 foram escolhidos (se por calibração com o BM25, antes de medir os modelos densos, dizê-lo).

### L506 · menor · falta controle ou limitação · em parte

> **Bi-encoder do sistema** (23 M, 6 M de pares)

**Problema.** Não há vazamento do sinal avaliado: os modelos foram ajustados em título e resumo, quase sem equações, e são aplicados aqui fora da distribuição de treino. Isso precisa ser dito, porque (a) o leitor não sabe se os documentos foram vistos no treino e (b) sem o MiniLM-L6 não ajustado como controle, não se sabe se a vantagem vem do ajuste em Física ou da tokenização em subpalavras da base. A ordem dos três modelos (o de 23 M à frente do GTE-base de 109 M) sugere a segunda hipótese, não medida.

**Evidência.** src/phifm/training/pairs.py: os pares de treino são 'título + ". " + resumo' de artigos do arXiv (spine + OpenAlex). Medido em 20.000 linhas sorteadas de data/processed/spine.parquet com ocorrencias_do_documento: 265 (1,3%) têm alguma equação de conteúdo (relação e ≥ 40 caracteres canônicos). Os documentos do conjunto (fatia arXiv do RedPajama) são da mesma população de artigos; não há separação treino/avaliação por documento. Nenhum artefato mede o MiniLM-L6 sem ajuste neste conjunto.

**Correção proposta.** Acrescentar: "Os três modelos densos foram ajustados em pares de título e resumo, dos quais 1,3% contêm alguma equação de conteúdo; equação contra equação é, para eles, uso fora da distribuição de treino. Os artigos do conjunto pertencem à mesma população dos pares, sem separação por documento. A base MiniLM-L6 sem ajuste não foi medida, de modo que a vantagem não pode ser atribuída ao ajuste em Física."

### L510 · menor · sem evidência versionada · pendente

> (mediana de 81, com 10% acima de 238)

**Problema.** Os dois números não têm artefato versionado, e o texto não diz em qual tokenizador nem sobre qual população (consultas, equações indexadas ou ambas) foram medidos. Com três tokenizadores, a fração truncada difere por modelo.

**Evidência.** grep por '238' em data/processed/avaliacao/, scripts/ e src/: nenhum artefato nem código que produza esses valores; aparecem só em ESTADO.md (linha 654) e DOC-11 (linha 210), como prosa. O campo 'max_tokens': 192 está em pb_formula_resultado.json. Os três modelos densos têm tokenizadores diferentes (MiniLM, ModernBERT, GTE).

**Correção proposta.** Versionar a medição (um JSON com mediana, percentil 90 e fração acima de 192 por tokenizador, sobre as 838.198 equações indexadas e as 2.000 consultas) e escrever: "no tokenizador do bi-encoder do sistema, a mediana das equações indexadas é de 81 tokens e X% ultrapassam os 192 lidos".

### Conferido e correto

- Tabela 23, as quatro linhas (recall@1, recall@10, MRR) conferem com data/processed/avaliacao/pb_formula_resultado.json: BM25 0,7115/0,9300/0,7902; bi-encoder do sistema 0,8595/0,9550/0,8966; ModernBERT@200k 0,8295/0,9500/0,8760; GTE-base@400k 0,8005/0,9200/0,8450.
- Diferenças em recall@10 contra o BM25 e intervalos conferem com o mesmo arquivo: +0,0250 [0,0115; 0,039], +0,0200 [0,0065; 0,034], −0,0100 [−0,0245; 0,005]; bootstrap pareado por item, 2.000 itens.
- Protocolo da legenda confere: 2.000 itens, pool de 20.000 documentos, 838.198 equações indexadas, semente 17, max_tokens 192, teto 1,0, escore do documento = máximo sobre as equações, estrato notacional com max_formas_em_comum 5, 24.508 itens no estrato.
- 202.365.265 equações, 828.601 identificadores distintos e 374.739 itens conferem com pb_formula_montagem.json (835.379 linhas, 6.778 ids repetidos, coerente com a linha 167 do rascunho).
- 53,1% / 37,0% / 7,8% conferem com pb_formula_diagnostico.json (0,5307; 0,3698; 0,0783); 36,3% com ≥ 5 equações em comum confere (0,3633; 136.144 itens); 24.508 = notacional sem documento quase igual confere.
- A lista de marcação 'superficial' do texto (espaço, rótulos, \nonumber, alinhamento, espaçamento fino, pontuação final) corresponde à expressão regular _SUPERFICIAL/_PONTUACAO_FINAL de src/phifm/eval/benchmarks/formula.py; os exemplos \le/\leq e \gamma_{\mu}/\gamma_\mu são de fato colapsados por canonical.py (executado: classificados como 'notacional' e canonicamente iguais).
- Vantagens em recall@1 por estrato: 0,9990 − 0,9465 = +0,0525 (≈ +0,053), 0,9410 − 0,8765 = +0,0645 (≈ +0,065), 0,8595 − 0,7115 = +0,148; '15 pontos' é arredondamento de 14,8. Os valores do resumo (0,860 e 0,712) são arredondamentos corretos.
- A vantagem em recall@1 no estrato notacional é estatisticamente robusta mesmo sem o arquivo por item: 577 erros contra 281 em 2.000 itens; no pior caso de discordância o teste binomial exato dá p = 2,6×10⁻²⁴.
- 'antes de qualquer modelo medido': o diagnóstico de grafia é do commit d748495 (2026-09-17), a regra e o estrato primário foram registrados em c1b02f0 (2026-09-17 20:45), e os resultados são de 102bc91/41ed882 (2026-09-18). A regra registrada tem de fato recall@10 como primária e os estratos idêntica/superficial como secundária.
- Execução anulada (linha 512) e remissão à Seção 5.2: confere com o commit ca4a2d9 e com a última linha da Tabela 24 (841.101, 843.127, 836.422 e 831.048 equações); o conserto (pool ordenado, pool_sha na assinatura do cache) está no script.
- Os quatro sistemas foram medidos sobre o mesmo conjunto de candidatos: a assinatura do cache inclui pool_sha e o número de equações, e um sistema de outro pool é recusado (scripts/avaliar_pb_formula.py, carregar_cache).
- O índice não entrega o gabarito: entram todas as equações de conteúdo dos documentos do pool, não só as compartilhadas; o documento da consulta é mascarado por item; o estrato primário exclui documentos com 5 ou mais formas em comum.
- Sem vazamento do sinal avaliado no treino do bi-encoder: os pares são título + resumo (src/phifm/training/pairs.py), e só 1,3% de 20.000 resumos sorteados da spine contêm alguma equação de conteúdo.
- 'Bi-encoder do sistema (23 M, 6 M de pares)' continua correto depois da v0.5: o ADR-0003 aceito (§10), o T1h/T1i e o commit 62e42fc mantêm o MiniLM-L6 ajustado em 6 M de pares como encoder do sistema.
- 'assimetria a favor do BM25' quanto à truncagem é coerente com o código: os densos truncam consulta e equação em 192 tokens (max_tokens), o BM25 indexa a cadeia inteira.
- A limitação sobre consultas verbatim e sobre a escolha do estrato depois da medição da grafia existe na linha 617 (fora do trecho), coerente com o ponto 4 da especificação revista.

### Não verificável nesta máquina

- Recalcular as métricas e o bootstrap pareado: exigem data/processed/pb_formula/ (itens.parquet, itens_estratos.parquet, ocorrencias/*.parquet), os três checkpoints (models/phiemb-do-sistema, phiemb-gte-base-400k-melhor, phiemb-t2eq-modernbert-200k-melhor) e o cache por item pb_formula_resultado.parcial.json, que está no .gitignore. Nenhum deles está nesta máquina.
- McNemar exato de recall@1 e de recall@50 por estrato: exige as séries por item do .parcial.json; aqui só foi possível dar limites de pior caso a partir das marginais.
- 'mediana de 81, com 10% acima de 238' tokens: exige as equações do pool e os tokenizadores; não há artefato versionado.
- Reprodução da montagem (374.739 itens, estratos, 24.508): exige o corpus data/processed/redpajama_fisica (44 partes).
- Que os 36,3% de pares com cinco ou mais equações em comum sejam 'a mesma obra em duas versões, ou trabalhos companheiros': é interpretação; conferir exigiria amostrar os pares de documentos no corpus.
- Sobreposição exata entre os documentos do conjunto de candidatos e os artigos usados nos pares de treino do bi-encoder: exige os pares de treino e o pool (pool_sha 537cbd9f9294c673).
- Efeito de um BM25 com tokenização de LaTeX, de um casador de cadeia e do MiniLM-L6 sem ajuste: são medições novas, que exigem os mesmos dados e modelos.
- A existência, na literatura, do 'argumento recorrente' da linha 497 e as referências de recuperação de fórmulas sugeridas: precisam ser conferidas na fonte; não há acesso bibliográfico verificado nesta revisão.


## Discussão e Limitações (§5, §6)

### L575 · maior · desatualizado após a v0.5 · aplicado

> Soma-se a evidência da Tabela 10, de que a base importa mais que o volume de ajuste.

**Problema.** Como regra geral a frase não se sustenta: a Tabela 10 mostra troca de base ≈ quinze vezes mais pares (com o volume ligeiramente à frente), e as medições posteriores à v0.5 mostram que a vantagem de base vista a 200 mil pares não sobrevive a 1 M contra o sistema de 6 M. A mesma leitura reaparece na linha 524 ('a Tabela 10 indica partir da base geral mais forte disponível'), enquanto a decisão final do projeto foi manter a base mais fraca com mais volume.

**Evidência.** Tabela 10: GTE-base@400k = 0,6094 contra MiniLM@6M = 0,6223 (o volume fica à frente por 0,0129). t1i_comparacao.json: gte-small@1M −0,0209 [−0,0323; −0,0096] e bge-small@1M −0,0211 [−0,0323; −0,0100] contra o sistema, desfecho "ABAIXO", leitura "o volume ainda manda a 1 M: o sistema segura o lugar". t1h_comparacao.json: as mesmas bases ganham +0,042 e +0,040 a 200 mil pares. ADR-0003 §10: o ΦEmb continua o MiniLM a 6 M; commit 62e42fc: "nenhuma supera o MiniLM-L6@6M (0,6223) a um custo aceitável".

**Correção proposta.** "Soma-se a evidência da Tabela 10, de que a troca de base rende, a 400 mil pares, o que o volume rende com quinze vezes mais pares — sem que a base ultrapasse o volume: duas bases pequenas que venciam o MiniLM-L6 a 200 mil pares por +0,04 ficaram, a 1 milhão, 0,021 abaixo do recuperador de 6 milhões (IC de 97,5% excluindo zero)." Na linha 524, substituir 'indica partir da base geral mais forte disponível' por 'indica que base e volume são substituíveis nesta faixa, e que o custo de serviço decide entre eles'.

### L577 · maior · afirma mais que a evidência · aplicado

> O encoder treinado do zero perde para o que já existe de graça, na tarefa para a qual foi treinado.

**Problema.** A frase, e as linhas 579 ("Isso fecha, na prática, a opção") e 585 ("Isso fecha as duas opções de encoder próprio sob este orçamento. Treinar do zero perde para a base geral"), enunciam como resultado sobre 'o encoder do zero' o que foi medido num substituto três vezes menor, com 0,6 B tokens e uma semente. Além disso o substituto foi pré-treinado em MLM, não 'treinado para' recuperação. O que os dados sustentam é uma decisão de alocação de orçamento (ADR-0003 §10), não uma conclusão de que o treino do zero perde.

**Evidência.** Tabela 20 (linhas 452-454) e data/processed/avaliacao/t2eq_cpt_emb_comparacao.json: o que foi medido é um substituto de 48 M, 0,6 B tokens, uma semente, 200 mil pares (ressalvas do JSON: "Uma semente por braço; 200 mil pares, metade da receita"). docs/adr/ADR-0003 §3 diz textualmente: "O que isto NÃO é: um veredito sobre o ΦEnc de 150 M. São proxies de 48 M a 0,6 B, 3,3× abaixo do planejado para a ablação, com uma semente por braço." A própria §6 (linha 613) admite: "O encoder próprio de 150 M, treinado do zero, não foi treinado."

**Correção proposta.** "O substituto de 48 M treinado do zero com 0,6 bilhão de tokens perde, depois do mesmo ajuste, para uma base geral de 150 M que já existe de graça." E, na linha 585: "Com isso, nenhuma das duas opções de encoder próprio foi levada adiante sob este orçamento: o substituto treinado do zero ficou 0,0558 abaixo da base geral sem pré-treinamento, e o pré-treinamento continuado da base escolhida rendeu menos que a troca de base. São medições de uma semente por braço, a 200 mil pares; o encoder de 150 M do zero não foi treinado, e estes resultados não são um veredito sobre ele."

### L579 · maior · inconsistência interna · aplicado

> Um modelo próprio de 150 M teria de cobrir três desvantagens somadas

**Problema.** Contagem dupla. (i) Um modelo próprio de 150 M não teria desvantagem de número de parâmetros frente a uma base de 150 M. (ii) Os 0,0558 não são uma terceira desvantagem a somar: são o efeito medido das outras duas (parâmetros e volume) mais o tokenizador, no substituto de 48 M. O argumento que 'fecha' a opção soma a causa ao seu próprio efeito. O ADR-0003 §8 tem a mesma formulação ("a soma de três ordens"), de onde a frase foi herdada.

**Evidência.** Tabela 20: o braço tratado tem 48 M e 0,6 B tokens; o ModernBERT-base, 150 M e ~2 T tokens; a diferença de 0,0558 é exatamente o resultado dessa comparação (t2eq_cpt_emb_comparacao.json / ADR-0003 §8: +0,0558 [+0,0426; +0,0685]). A linha 458 declara que 'diferem o tamanho, o tokenizador e o volume de pré-treinamento'.

**Correção proposta.** "Um modelo próprio de 150 M igualaria a base em parâmetros, mas não em volume de pré-treinamento — 2 bilhões de tokens preparados contra cerca de 2 trilhões. A diferença de 0,0558 foi medida no substituto de 48 M e confunde tamanho, tokenizador e volume; não se sabe quanto dela restaria a 150 M. Nenhuma evidência deste trabalho sugere que o objetivo de domínio, que vale +0,0840 na escala do substituto e +0,0076 a 150 M, cubra três ordens de grandeza de volume."

### L583 · maior · falta controle ou limitação · aplicado

> O melhor encoder desta escala não passou por pré-treinamento no domínio algum.

**Problema.** A conclusão 'pré-treinar continuamente a base escolhida rende menos que escolher outra base' compara um ganho de MLM de domínio sobre uma base sem treino contrastivo com a troca por uma base que já é um modelo de embedding contrastivo. 'Base' aqui agrega arquitetura, tamanho, contexto e, sobretudo, pré-treino contrastivo geral em centenas de milhões de pares. O que a Tabela 22 mostra é que o pré-treino contrastivo geral vale mais que 0,4 B tokens de MLM de domínio nesta tarefa — e não que o pré-treino de domínio não paga sobre a melhor base (não testado), nem numa tarefa em que contexto além de 192 tokens importe. A §6 (linha 615) diz 'compara bases gerais com uma variável' sem nomear o que a variável agrega.

**Evidência.** data/processed/avaliacao/t2eq_emb_gte_contra_barra.json, ressalva: "uma variável, a BASE: mesmos pares, lote, 192 tokens e semente. O GTE-base tem 110 M contra 150 M e contexto de 512 contra 8.192". Bases: thenlper/gte-base (modelo de embedding já treinado contrastivamente em larga escala) contra ModernBERT-base (encoder só de MLM). ADR-0003 §9 item 4: "Um CPT com o §2.3 sobre o GTE-base é a combinação que esta tabela sugere, e ela NÃO está pronta". ADR-0003 §10 'O que reabriria': tarefa em que o contexto longo pese; "O dissenso da D-01 … Nada nesta decisão o responde".

**Correção proposta.** Na linha 583: "O melhor encoder desta escala não passou por pré-treinamento de linguagem mascarada no domínio; é, porém, um modelo de embedding já treinado contrastivamente em larga escala, e o ModernBERT-base não é. A diferença de 0,069 mede o conjunto base — arquitetura, tamanho, contexto e pré-treinamento contrastivo geral —, e não um fator isolado." Na §6, linha 615, acrescentar: "'Base' é uma variável composta: o GTE-base tem 109 M, contexto de 512 e pré-treinamento contrastivo geral; o ModernBERT-base, 150 M, contexto de 8.192 e apenas modelagem mascarada. O pré-treinamento continuado com o objetivo de equações não foi aplicado à base vencedora, e a tarefa trunca os textos em 192 tokens, de modo que o contexto longo não entra na medida."

### L585 · maior · inconsistência interna · aplicado

> e medida primária nula nas duas

**Problema.** A discussão resume como 'nula nas duas' uma medida primária que, pela regra pré-registrada, foi NEGATIVA a 48 M e não decidida a 150 M. O mesmo parágrafo conclui 'O efeito encolhe à medida que a base se fortalece, o que é o padrão esperado de uma lacuna que a base forte já cobre' a partir de dois pontos que diferem simultaneamente em tamanho, tokenizador, volume (0,6 B contra 0,4 B), treino do zero contra continuado, e uma semente cada, com a assimetria de execução invertida entre eles (linha 613).

**Evidência.** Tabela 18 (linha 433): diferença das diferenças a 48 M = −0,0040 [−0,0058; −0,0022], intervalo que exclui zero; linha 435: "Pela regra, o desfecho é negativo". Só a 150 M o desfecho é 'não decidido' (Tabela 21: −0,00039 [−0,00161; +0,00087]; t2eq_cpt_emb_comparacao.json, primaria_de_mlm_ja_medida.desfecho = "NÃO DECIDIDO"). ADR-0003 §2 também registra a 48 M "negativo pela regra".

**Correção proposta.** "…+0,084 sobre bases de 48 M treinadas do zero, +0,0076 sobre uma de 150 M com 2 trilhões de tokens gerais; a medida primária foi negativa na primeira (−0,0040 [−0,0058; −0,0022]) e não decidida na segunda. O efeito de recuperação é menor sobre a base mais forte, o que é compatível com uma lacuna que a base forte já cobre; com dois pontos, uma semente cada, que diferem também em tokenizador, volume e regime de treinamento, o padrão é uma hipótese, e não um resultado."

### L593 · maior · inconsistência interna · aplicado

> Isso converte um pré-treinamento de 20 bilhões de tokens de 37 dias para cerca de 80 horas

**Problema.** O parágrafo intitulado 'Medir a vazão real antes de planejar' projeta o pré-treinamento do encoder com a vazão de outro modelo (23 M) em outra tarefa, e a projeção foi desmentida por um fator de 3,4 a 4,5 pela vazão que o próprio trabalho mediu depois (Seção 3.4). A frase 'alterando a viabilidade' fica, portanto, apoiada num número superado. Há ainda o erro aritmético de 37 contra 30 dias.

**Evidência.** (a) Aritmética: 20e9 / 7.700 tokens/s = 2,60e6 s = 30,1 dias, não 37 (37 dias exigiriam 6.250 tokens/s); 20e9 / 69.700 = 79,7 h confere. (b) Os 69.700 tokens/s são a vazão do AJUSTE CONTRASTIVO do MiniLM de 23 M (181,6 pares/s × 2 × 192 tokens; ESTADO.md: "Um par são duas sequências de 192 tokens, então 181,6 pares/s ≈ 69.700 tokens/s"). A vazão de pré-treinamento efetivamente medida depois está no próprio artigo: linha 98, "cerca de 20.500 tokens por segundo" para o substituto de 48 M numa T4; linha 100, "15.200 a 15.900 tokens por segundo nas duas placas" para 150 M. Com elas, 20 B tokens custam 271 h (48 M) ou cerca de 356 h (150 M, duas T4), e não 80 h.

**Correção proposta.** "Medir a vazão real, e do modelo certo, antes de planejar. No ajuste contrastivo, a T4 mediu 181,6 pares por segundo contra 20 a 26 na GPU local, o que reduziu um treinamento de 13 horas a 36 minutos. A extrapolação dessa vazão para o pré-treinamento — cerca de 80 horas para 20 bilhões de tokens — não se confirmou: o substituto de 48 M mediu 20.500 tokens por segundo, e a base de 150 M, cerca de 15.600 em duas placas (Seção 3.4), isto é, de 270 a 360 horas para o mesmo volume. A vazão de um modelo não é a de outro."

### L603 · maior · falta controle ou limitação · aplicado

> ## 6. Limitações

**Problema.** O checkpoint de cada modelo é eleito pelo nDCG@10 sobre itens do mesmo pares_validacao.parquet, sorteados com a mesma semente do pool do veredito (o pool de 1.000 do acompanhamento é subconjunto dos 2.000 do veredito). Não há conjunto de teste separado: seleção de checkpoint, escolhas de receita (lote, negativos, truncamento, bases candidatas) e veredito usam os mesmos itens. O viés é otimista e não é igual entre braços: execuções longas (6 M de pares) têm muito mais avaliações entre as quais escolher o pico que as de 200 mil. Diferenças pequenas (−0,0012, +0,0076, +0,0103) são da ordem do que essa seleção pode produzir.

**Evidência.** src/phifm/training/embedding.py:602 (`amostra = preparar_pool(val, n, exigir_n=False)`) e 789-821 (`_talvez_melhor`: grava `<saida>-melhor` quando o nDCG@10 de validação sobe); src/phifm/training/amostragem.py:163-200 (SEMENTE_POOL = 17; embaralha com a semente, desduplica e faz `pool.head(n)`), de modo que o pool de acompanhamento é um prefixo do mesmo sorteio usado no veredito. t1f_treino.json: hiperparametros.n_candidatos_aval = 1000, semente 17, melhor.passo = 3000, criterio "nDCG@10 — a métrica do portão G1". Todos os modelos medidos são os '-melhor' (t1g/t1h/t1i/t2eq_cpt_emb_comparacao.json: caminhos '...-melhor'); protocolo.n = 2000, semente = 17 em todos. A §6 não menciona seleção de checkpoint.

**Correção proposta.** Acrescentar à §6: "Não há conjunto de teste separado. O checkpoint reportado de cada modelo é o de maior nDCG@10 em avaliações periódicas sobre o próprio conjunto de validação, com a mesma semente de sorteio do conjunto de 2.000 pares que dá o veredito; as decisões de receita e de base também foram tomadas sobre ele. Os valores absolutos são, portanto, otimistas, e o viés não é igual entre braços, porque execuções mais longas oferecem mais pontos de avaliação entre os quais escolher. Os intervalos reportados não incluem essa seleção, e diferenças da ordem de 0,01 de nDCG@10 devem ser lidas com essa ressalva."

### L605 · maior · falta controle ou limitação · aplicado

> Um domínio e um grafo de citação: Física e OpenAlex.

**Problema.** Faltam três limitações de validade externa e de vazamento: (1) a divisão é por âncora, de modo que os documentos-alvo da validação aparecem como positivos no treino (e uma âncora de validação pode aparecer como documento citado no treino) — o modelo ajustado já viu os candidatos, as bases gerais não, o que favorece os modelos ajustados e cresce com o volume de pares; (2) não há nenhuma medição em conjunto externo publicado (por exemplo SciDocs/SciRepEval ou a fatia científica do BEIR), de modo que nenhum número é comparável com a literatura, nem se sabe se o ajuste degrada o modelo fora da tarefa; (3) consultas e candidatos são artigos do arXiv com título+resumo, truncados em 192 tokens: livros, periódicos sem preprint, literatura anterior a 1991 e consultas escritas por usuários não estão representados.

**Evidência.** src/phifm/training/pairs.py:149-159: a divisão sorteia ÂNCORAS (`ancoras.head(n_val)`) e a única verificação é de âncora compartilhada (`tr.join(val.select("arxiv_id")…)`); não há restrição sobre `arxiv_citado`. Os pares são arXiv→arXiv (colunas arxiv_id/arxiv_citado). O treino cobre 650.162 de 667.304 documentos citados (linha 287) e o universo de avaliação tem 88.807 citados. Em data/processed/avaliacao/ não há nenhuma medição em benchmark externo (transferencia*.json é o classificador de domínio). O título e o Resumo (linha 13) falam em 'literatura de Física'.

**Correção proposta.** Acrescentar à §6: "A divisão entre treino e validação é por documento citante. Os documentos citados da validação não são excluídos do treino, onde aparecem como positivos de outras âncoras; os modelos ajustados já viram, portanto, os candidatos entre os quais são avaliados, e as bases gerais não. A fração de candidatos vistos não foi medida, e cresce com o volume de pares — o que pode inflar a curva da Tabela 9 e a vantagem sobre os modelos sem ajuste. Nenhum resultado foi medido em conjunto externo publicado: os valores não são comparáveis com os da literatura, e não se sabe se o ajuste por citação preserva o desempenho fora desta tarefa. Por fim, consultas e candidatos são títulos e resumos de artigos do arXiv, truncados em 192 tokens; 'literatura de Física' deve ser lido como 'artigos de Física no arXiv', e nada aqui cobre livros, periódicos sem preprint ou consultas formuladas por usuários."

### L607 · maior · falta controle ou limitação · aplicado

> O conjunto de avaliação é próprio e não dispõe de julgamento humano de relevância.

**Problema.** O all-MiniLM-L6-v2 é um modelo de embedding treinado pelos seus autores em mais de um bilhão de pares, entre eles pares de citação do S2ORC (título–título, resumo–resumo, título–resumo). A tarefa de avaliação deste artigo é exatamente recuperação de documento citado sobre título+resumo de artigos do arXiv, que o S2ORC cobre. Logo (i) a linha 'MiniLM-L6 (geral, sem ajuste)' da Tabela 8 (0,4761) não é zero-shot na tarefa, (ii) pares da validação podem ter sido vistos pela base antes de qualquer ajuste, e (iii) a leitura da linha 300 ('o ajuste ensina sobretudo a tarefa de recuperação, que uma base contrastiva geral já sabe') e a de 5.1/5.5 ('partir da base geral mais forte') ficam confundidas com contaminação. Risco análogo vale para as bases GTE, BGE e E5. A citação [12] (Wang et al., destilação) também descreve o modelo errado na linha 90.

**Evidência.** A base do bi-encoder é sentence-transformers/all-MiniLM-L6-v2, e não o MiniLM de [12]: src/phifm/eval/encoders.py:104 ("MiniLM-L6 (genérico 23M)": "sentence-transformers/all-MiniLM-L6-v2"), t1h_comparacao.json (bases.minilm-l6.base), t1f_base_n2000.json. grep por 'all-MiniLM' e por 'S2ORC' no rascunho: 'all-MiniLM' não ocorre; 'S2ORC' só na linha 60 (peS2o). A §6 não menciona a origem dos pesos de nenhuma base geral.

**Correção proposta.** Acrescentar à §6: "A base do bi-encoder do sistema é o all-MiniLM-L6-v2, um modelo de embedding treinado pelos seus autores com mais de um bilhão de pares, entre os quais pares de citação do S2ORC. A tarefa de avaliação deste trabalho — recuperar o documento citado a partir do citante, sobre título e resumo de artigos do arXiv — coincide em forma, e possivelmente em instâncias, com esses dados. Não medimos a sobreposição entre os pares de validação e os pares de citação vistos pela base. Consequentemente, as linhas 'sem ajuste' das Tabelas 8 e 22 não são medições zero-shot, a vantagem das bases contrastivas gerais sobre o PhysBERT e o ModernBERT-base pode incluir exposição prévia à tarefa e aos próprios pares, e o mesmo risco se aplica às bases GTE, BGE e E5, cujos dados de pré-treinamento incluem literatura científica." Corrigir também a identificação do modelo na Seção 3.4 (all-MiniLM-L6-v2, Reimers e Gurevych [11], sobre MiniLM [12]).

### L613 · maior · falta controle ou limitação · aplicado

> Todos têm uma semente por braço

**Problema.** A limitação de semente única é declarada só para o pré-treinamento. As Tabelas 9, 10, 14 e 22 comparam um modelo treinado por braço; McNemar e bootstrap pareado capturam apenas a variância de amostragem das consultas, condicionada àquele modelo, e não a variância entre treinamentos. Isso afeta diretamente as conclusões de 5.1 (domínio contra capacidade na Tabela 14, decidida por +0,0090 com um treinamento por base) e os 'empates' da Tabela 10.

**Evidência.** A frase está no parágrafo de pré-treinamento (4.9.1–4.9.3). Mas os modelos de recuperação e reordenação também têm uma única semente de treinamento: t1g_comparacao.json ressalvas "Uma semente por modelo."; t1h/t1i "Uma semente por base."; t1c_resultado.json fixo "… semente 17 — idênticos ao controle"; t1f_treino.json hiperparametros.semente = 17; linha 285 do artigo ("mesma semente").

**Correção proposta.** Acrescentar à §6: "Todo modelo deste trabalho — bi-encoders, cross-encoders e encoders pré-treinados — foi treinado uma única vez, com uma semente. Os testes pareados e os intervalos por bootstrap medem a incerteza de amostragem das consultas, condicionada ao modelo treinado, e não a variância entre treinamentos. Diferenças pequenas entre braços — em particular a da Tabela 14, de +0,0090, e os empates da Tabela 10 — não foram replicadas com outra semente."

### L524 · menor · afirma mais que a evidência · aplicado

> a base de 109 M comprou com 400 mil pares o mesmo que o volume comprou com 6 milhões

**Problema.** Mesma raiz do achado já conhecido no Resumo, propagada à Discussão: 'o mesmo' só está estabelecido em recall@1. Na métrica primária a diferença de −0,0129 é maior que a meia-largura típica dos intervalos pareados deste protocolo, e o empate em nDCG@10 só foi demonstrado para o GTE-base a 1 milhão.

**Evidência.** Tabela 10 / t1f: nDCG@10 0,6094 contra 0,6223 (−0,0129), recall@10 0,8010 contra 0,8305; o único teste pareado é em recall@1 (147 × 150, p = 0,908). Não encontrei em data/processed/avaliacao/t1f_*.json bootstrap pareado de nDCG@10 para esse par. Para escala: em t1i_comparacao.json uma diferença de −0,0209 tem intervalo [−0,032; −0,010], e em t1g uma de −0,0012 tem [−0,011; +0,008] — meia-largura de cerca de 0,010.

**Correção proposta.** "…com a ressalva de que a base de 109 M, com 400 mil pares, empatou em recall@1 com o volume de 6 milhões e ficou 0,013 abaixo em nDCG@10 (diferença não testada), alcançando-o apenas com 1 milhão de pares, ao custo de 4,4 vezes a inferência."

### L530 · menor · clareza · aplicado

> oito ocorrências independentes de amostragem dependente da ordem dos dados

**Problema.** (i) 'Independentes' é forte: a quinta ocorreu dentro do instrumento da quarta, e todas vêm de um único repositório e de um único autor de código — são oito instâncias, não oito observações independentes de uma taxa. (ii) O título da seção ('em conjuntos derivados de grafos de citação') e a generalização da linha 549 cobrem só três das oito; as demais são de corpus de texto integral, de partições de pré-treinamento e de conjunto de equações. (iii) Três linhas não têm artefato versionado que as sustente. Não há no texto a afirmação de que a amostra enviesada superestima como regra: a tabela mostra direções distintas (a terceira eleva a métrica, a sexta a rebaixa), o que está correto.

**Evidência.** As oito linhas da Tabela 24 estão enumeradas. Evidência versionada conferida: linha 1 (revisao_pes2o_amostragem.json), linha 4 (docstring de scripts/preparar_dados_phienc.py: display 18,8% contra 29,5% = +57%), linha 6 (Tabela 7; 191.300/17.844 = 10,7), linha 7 (src/phifm/eval/mlm_regiao.py: "~4% primeiros documentos"), linha 8 (docs/03-evaluation/DOC-11-physbench.md:231 com os quatro totais). Para as linhas 2, 3 e 5 (35 documentos e ±0,159; 49,6% e 0,139→0,020; 200 linhas de um arquivo) só há narrativa em ESTADO.md e em docs/papers/rascunho-armadilhas-recuperacao-por-citacao.md, sem artefato em data/processed/avaliacao. Das oito, apenas as linhas 2, 3 e 6 envolvem conjuntos derivados do grafo de citação.

**Correção proposta.** "Uma auditoria identificou oito ocorrências distintas de amostragem dependente da ordem dos dados, todas no mesmo repositório; três delas em conjuntos derivados do grafo de citação. A direção do viés não é constante: a terceira elevou a métrica, a sexta a rebaixou." Renomear a seção para 'Vieses de amostragem dependente da ordem dos dados' e indicar, para as linhas 2, 3 e 5 da Tabela 24, que os valores vêm do registro de trabalho e não de artefato reexecutável.

### L547 · menor · número não confere · aplicado

> uma verificação que interrompe a execução quando o teto do conjunto de avaliação difere de 1,0

**Problema.** A propriedade verificada com interrupção é a ausência de duplicatas no pool (que implica teto 1,0 neste protocolo); o teto medido só gera aviso. A descrição no artigo é mais forte que o código. Vale só para o protocolo de comparação de encoders: no protocolo de sistema o teto é o recall do recuperador (0,4495 na Tabela 14) e nada é interrompido.

**Evidência.** src/phifm/training/amostragem.py:201-209: o que interrompe é um AssertionError quando há texto de âncora ou de alvo repetido no pool. scripts/avaliar_encoders.py:151-157: o teto é medido por teto_do_pool, impresso e gravado; se `teto["ndcg_10"] < 0.999` o script apenas imprime "⚠️ teto abaixo de 1,0" e segue para `salvar(...)`.

**Correção proposta.** "A correção adotada é uma verificação que interrompe a execução quando o conjunto de avaliação contém consulta ou alvo de texto repetido — a condição que derruba o teto abaixo de 1,0 —, com o teto medido e registrado no artefato de saída."

### L615 · menor · desatualizado após a v0.5 · aplicado

> A ordem entre duas bases a esse volume pode não se manter com mais pares.

**Problema.** A limitação está redigida como possibilidade em aberto; depois da v0.5 ela foi medida (a vantagem a 200 mil pares não se manteve a 1 M contra o sistema) e a busca por encoder foi encerrada por decisão, com um teste declaradamente não executado. A §6 deve registrar o encerramento e o que ficou sem medir.

**Evidência.** t1h_comparacao.json e t1i_comparacao.json (2026-09-23/24); ADR-0003 §10, bloco 'O que reabriria esta decisão': "Testado em 2026-09-23/24 (T1h e T1i), e a busca encerrada pelo dono"; commit 62e42fc: "O teste restante (uma pequena a 6 M, ~17 h para ~+0,02) não será feito".

**Correção proposta.** "A ordem entre bases a 200 mil pares não é a ordem a volumes maiores: gte-small e bge-small superam o MiniLM-L6 a 200 mil pares e ficam abaixo do recuperador do sistema a 1 milhão. A busca por encoder foi encerrada por decisão de custo, e não por esgotamento: nenhuma base alternativa foi levada a 6 milhões de pares, de modo que a comparação com uma variável — a base, no volume do sistema — não existe para nenhuma delas."

### Conferido e correto

- §5.1: nDCG@10 0,3507 (PhysBERT) e 0,4761 (MiniLM-L6 sem ajuste) conferem com t1f_base_n2000.json (0,35068 e 0,47611); recall@100 0,516→0,633 confere com a Tabela 13 e com teto_do_recuperador.json (em_100 = 0,6325); razão de custo 4,4 confere com t1f_treino.json e t1g_comparacao.json (954 s / 217 s).
- §5.2, Tabela 24 linha 1: 400 documentos, 2 de 277 arquivos, 0,67%, 100% resumos — confere com revisao_pes2o_amostragem.json (amostra_anterior).
- §5.2, linha 4: 57% mais equações em display — confere (29,5% / 18,8% = 1,57; docstring de scripts/preparar_dados_phienc.py). Linha 6: fator 10,7 = 191.300/17.844 e teto 0,7562 (Tabela 7; comentário em scripts/avaliar_encoders.py). Linha 7: ~4% dos documentos (src/phifm/eval/mlm_regiao.py). Linha 8: 841.101, 843.127, 836.422 e 831.048 equações (docs/03-evaluation/DOC-11-physbench.md:231); 838.198 equações e 24.508 itens no estrato em pb_formula_resultado.json.
- §5.2: não há no texto afirmação geral de que a amostra enviesada/recente superestima; as únicas ocorrências de 'superestima' (linhas 64 e 229) tratam de pseudo-verossimilhança e de fertilidade, e a Tabela 24 mostra direções de viés distintas.
- §5.2: '+0,020 de nDCG@10 sem custo de computação' confere com a linha 285 (+0,0197).
- §5.4: os três acionamentos têm z = 3,73, 3,00 e 4,09 sobre 300 passos de referência, e dois deles (3.407 e 3.553 alvos) ultrapassam o máximo da amostra (3.127) — t2eq_cpt_spikes_lotes.json.
- §5.5: +0,0558 [+0,0426; +0,0685], +0,0840 [+0,071; +0,097], +0,0076 [+0,0013; +0,0139], +0,0103 [+0,0032; +0,0176] e +0,0694 [+0,0580; +0,0812] conferem com t2eq_cpt_emb_comparacao.json, t2eq_emb_gte_contra_barra.json e ADR-0003 §7–§9; 'cerca de sete vezes' = 0,0694/0,0103 = 6,7.
- §5.5: a frase 'O argumento do viés indutivo nativo permanece sem teste direto' é coerente com o ADR-0003 §10 ('Nada nesta decisão o responde').
- §5.6: erro de sete vezes da Tabela 13 (0,0022/0,0003 = 7,3); 20 h de T4 para o pré-treinamento continuado (2 × 7,5 h + 4.489/6.103 × 7,5 h ≈ 20,5 h); 1 h 17 da comparação entre bases consta do ESTADO.md; 80 h = 20e9/69.700 confere; ±0,031 e ~0,061 a 256 candidatos e ~0,015 a 4.000 constam da tabela de poder do ESTADO.md.
- §6, linha 613: 3,3× abaixo do orçamento (2,0 B / 0,6 B); direções das assimetrias de execução (a favor de E em 4.9.1, contra o tratado em 4.9.2, a favor do tratado em 4.9.3) conferem com as linhas 419, 435 e 468 e com as ressalvas de t2eq_cpt_emb_comparacao.json; 'O encoder próprio de 150 M não foi treinado' confere com o ADR-0003 §10.
- §6, linha 615: GTE-base a 1 milhão empata com o sistema — t1g_comparacao.json: −0,00119 [−0,011; +0,00842], desfecho EMPATE.
- §6, linha 625: taxa de aprendizado não reajustada, declarada antes — confere com a ressalva gravada em t1c_resultado.json.
- §6 já contém: benchmark próprio sem julgamento humano (607), comparação da Tabela 20 não é de uma variável (611), semente única no pré-treinamento (613), 200 mil pares na Tabela 22 (615), consulta verbatim na recuperação por equação (617), contaminação do corpus não medida (619), falso negativo residual (623), irreprodutibilidade do pipeline (627).

### Não verificável nesta máquina

- Fração dos candidatos (documentos citados) da validação que aparecem como positivos no treino, e âncoras de validação que aparecem como citadas no treino: exige pares_treino/pares_validacao, que não estão nesta máquina. O código (src/phifm/training/pairs.py) mostra apenas que nada impede a sobreposição.
- Sobreposição entre os pares de validação e os pares de citação do S2ORC usados no treinamento do all-MiniLM-L6-v2 (e a composição exata dos dados de pré-treinamento de GTE, BGE e E5): depende do cartão do modelo e dos dados externos; aqui só confirmei que a base usada é sentence-transformers/all-MiniLM-L6-v2.
- Tabela 24, linhas 2, 3 e 5 (500 linhas = 35 documentos, ±0,159; 49,6% de âncoras vazadas, nDCG 0,139→0,020; 200 linhas de um único arquivo): só constam de ESTADO.md e de docs/papers/rascunho-armadilhas-recuperacao-por-citacao.md, sem artefato em data/processed/avaliacao. A meia-largura ±0,159 é coerente com p = 0,364 e n = 35.
- §5.3: ensaio de 30 passos (acurácia 0,0237 contra 0,0195; 2,890 contra 2,761 bits por byte) — só em ESTADO.md, sem artefato versionado.
- §5.4: desvio padrão da perda de 0,077 (tratado) contra 0,037 (controle) — só em ESTADO.md; os registros de treinamento do pré-treinamento continuado não estão aqui. Idem para o defeito do token de máscara '#' no exportador e para as três constantes do comparador.
- §5.6: 'dezessete dias' em disco e a vazão de 181,6 pares/s contra 20–26 — alegações do ESTADO.md; o log t1a_treino_t4.log existe mas não foi conferido por limite de orçamento. A 'diferença de interesse de 0,004' da linha 595 só aparece no ESTADO.md (margem histórica entre ΦEmb e MiniLM) e não está em nenhuma tabela do artigo.
- Bootstrap pareado de nDCG@10 entre GTE-base@400k e MiniLM@6M (linha 524): exige os checkpoints; não há artefato com esse intervalo.
- Efeito real da seleção de checkpoint sobre o conjunto do veredito (tamanho do viés otimista por braço): exige as curvas de validação de todos os treinos e um conjunto de teste separado.
- §6, linha 627: 2,1% das âncoras conservadas na reconstrução e 14.052.319 referências anuladas — dependem do corpus e do índice, ausentes desta máquina.


## Lente estatística (artigo inteiro)

### L108 · maior · falta controle ou limitação · aplicado

> avalia dentro de um conjunto de 2.000 pares de validação, sorteados com semente e deduplicados por texto

**Problema.** O checkpoint reportado é o melhor de 15 a 234 avaliações num subconjunto do próprio conjunto do veredito; os modelos de referência sem ajuste (GTE-large, MiniLM) não passam por seleção alguma. É viés de seleção (winner's curse) não declarado. A ordem de grandeza (até ~0,007 no pool interno da execução de 6 M) não ameaça a margem de +0,0435 sobre o GTE-large, mas é comparável às diferenças que decidem os 'empates' (−0,0012) e ao efeito de +0,0076 da Seção 4.9.3; além disso o número de avaliações difere entre braços (234 no sistema, ~15 nas execuções curtas), de modo que o viés não é igual dos dois lados.

**Evidência.** src/phifm/training/embedding.py:602 e 789-821: o checkpoint `-melhor` é eleito pelo maior nDCG@10 em `preparar_pool(val, n_candidatos)`; src/phifm/training/amostragem.py:163-202: `SEMENTE_POOL = 17`, sorteio com essa semente, deduplicação e `pool.head(n)`; scripts/train_embedding.py:135 e scripts/avaliar_encoders.py:105-108 leem o MESMO `pares_validacao.parquet` com a mesma semente. kaggle/t1a_phiemb.py:229 e kaggle/t1g_gte_1m.py:140 usam n_candidatos=1000. Logo o conjunto de seleção são os 1.000 primeiros itens dos 2.000 do veredito. Todos os modelos medidos são `-melhor` (t1g_comparacao.json, t2eq_cpt_emb_comparacao.json). t1a_curvas_de_treino.json: a execução de 6 M tem 234 avaliações, pico 0,7022 (passo 43.800) contra 0,6951 no fim.

**Correção proposta.** Acrescentar à Seção 3.5: "O ponto de verificação reportado de cada modelo ajustado é o de maior nDCG@10 numa avaliação interna de 1.000 candidatos feita a cada 200 ou 500 passos; esses 1.000 itens são os primeiros do mesmo sorteio que produz os 2.000 do protocolo, de modo que metade do conjunto do veredito participou da escolha do ponto de verificação. Os modelos sem ajuste não passam por seleção. O viés é a favor dos modelos ajustados e cresce com o número de avaliações (15 a 234); na execução de 6 milhões, a distância entre o pico e o último ponto, no conjunto interno, é de 0,007." E repetir a ressalva nas Limitações. Idealmente, remedir os 1.000 itens que não participaram da seleção e reportar as margens só neles.

### L114 · maior · inconsistência interna · aplicado

> Diferenças de nDCG@10 usam bootstrap pareado por consulta.

**Problema.** A Seção 3.6 enuncia a regra, e os resultados de recuperação das Seções 4.5 e 4.6 não a seguem: a diferença relatada é de nDCG@10 e o p-valor é de outro evento (recall@1). Os dados do próprio projeto (T1i, T2eq-CPT) mostram que os dois instrumentos discordam em qualquer dos sentidos, logo um p de recall@1 não qualifica nem desqualifica uma diferença de nDCG@10. A seção também não diz qual das duas é a medida primária de cada comparação.

**Evidência.** A métrica de manchete é nDCG@10, mas o teste reportado ao lado dela é McNemar em recall@1 nas linhas 243, 271, 285 ("+0,0197 de nDCG@10, com teste pareado em recall@1 de 104 contra 71"), na Tabela 10 (coluna "Pareado em recall@1") e na linha 306. g1_resultado.json, t1f_base.json e t1f_base_n2000.json não contêm nenhum IC de nDCG (grep -c 'ic95|bootstrap' = 0). t1i_comparacao.json mostra a dissociação entre os dois instrumentos: gte-small contra o sistema tem nDCG@10 −0,0209 [−0,0323; −0,0096] (IC de 97,5%, exclui zero) e, nos mesmos itens, McNemar em recall@1 147×135, p = 0,513, 'empate'. t2eq_cpt_emb_comparacao.json mostra o inverso na Seção 4.9.3: nDCG +0,0076 com IC que exclui zero e recall@1 63×60, p = 0,857.

**Correção proposta.** Na Seção 3.6: "A medida primária de toda comparação de recuperadores é o nDCG@10, com intervalo por bootstrap pareado por item; o teste de McNemar em recall@1 é diagnóstico e não decide. Nas comparações anteriores ao T1g (Tabelas 8 a 10, exceto a última linha da Tabela 10), só o diagnóstico foi calculado à época, e os intervalos de nDCG@10 foram acrescentados depois." E calcular esses intervalos (a função `bootstrap_pareado_itens` já existe) para: 6 M contra GTE-large, sorteados contra primeiras linhas a 400 mil, 6 M contra 3 M, GTE-base 400 mil contra MiniLM 400 mil e contra MiniLM 6 M.

### L271 · maior · sem evidência versionada · aplicado

> O bi-encoder de 23 M supera o GTE-large de 335 M em nDCG@10, recall@10 e MRR

**Problema.** É o resultado de manchete do artigo e não tem intervalo nem teste na métrica em que é afirmado; o único teste reportado não rejeita a 5%. Pela ordem de grandeza dos erros padrão de comparações análogas com os mesmos 2.000 itens (≈0,005 a 0,006 em t1g/t1i), a diferença de 0,0435 quase certamente se sustenta — a lacuna é de relato, não de resultado, mas um revisor a apontará de imediato.

**Evidência.** t1f_base_n2000.json, bloco `pareado`: o único teste entre 'ΦEmb do sistema (6M)' e 'GTE-large' é McNemar em recall@1, 188×153, p = 0,0654, veredito gravado 'empate'. Nenhum arquivo de evidência traz IC para os +0,0435 de nDCG@10 (grep 'ic95' = 0 em g1_resultado.json e t1f_base_n2000.json).

**Correção proposta.** Calcular e reportar: "O bi-encoder de 23 M supera o GTE-large de 335 M em nDCG@10 por +0,0435 [IC de 95% por bootstrap pareado por item: a calcular]; em recall@1 a estimativa pontual lhe é favorável e o teste pareado não a estabelece (188 contra 153 discordantes, p = 0,065)." Sem o cálculo, trocar 'supera' por 'fica acima, em estimativa pontual,'.

### L304 · maior · afirma mais que a evidência · aplicado

> o GTE-base empata com o bi-encoder de 6 milhões

**Problema.** Em todo o artigo 'empate' significa 'o teste não rejeitou', isto é, ausência de evidência, mas é redigido como evidência de ausência ('empata', 'alcança') e sustenta decisões de sistema. A Seção 3.6 não define o termo, não declara nível, bilateralidade, número de reamostras nem método do intervalo, e não informa a sensibilidade do desenho. No caso do T1g o IC permite uma afirmação de não inferioridade com margem explícita; nos casos apoiados só em McNemar (linhas 300, 347, 374, 378) não há intervalo algum.

**Evidência.** kaggle/t1g_gte_1m.py:160-162: a regra registrada define "EMPATE (IC cruza 0)"; src/phifm (veredito gravado em t1g_comparacao.json): "empate — 292 discordantes não separam os dois". Nenhuma margem de equivalência, TOST ou poder aparece na regra nem na Seção 3.6 (linhas 114-118). Recalculado a partir do IC de t1g_comparacao.json ([−0,011; +0,00842]): erro padrão 0,00495, diferença mínima detectável a 80% de poder ≈ 0,014 de nDCG@10. O mesmo uso de 'empate' = p > 0,05 aparece nas linhas 300 (p = 0,545), 302, 347 (124×122, p = 0,949), 374 (66×64, p = 0,930), 378 (92×82, p = 0,495), 491 (63×60), 615 e 643.

**Correção proposta.** Acrescentar à Seção 3.6: "Os testes são bilaterais com α = 0,05; os intervalos são de 95%, pelo método do percentil, com 20.000 reamostras. 'Empate' designa, neste trabalho, a não rejeição da hipótese nula — o intervalo cobre zero ou p > 0,05 — e não equivalência demonstrada: nenhuma margem de equivalência foi registrada. Com 2.000 itens, o erro padrão de uma diferença pareada de nDCG@10 é de cerca de 0,005, e diferenças abaixo de 0,014 não são detectáveis com 80% de poder." Na linha 304: "o GTE-base não se distingue do bi-encoder de 6 milhões — diferença de −0,0012 [−0,011; +0,008]; o intervalo exclui uma perda maior que 0,011 e um ganho maior que 0,008".

### L363 · maior · erro estatístico · aplicado

> Apenas a base pré-treinada no domínio supera a fusão.

**Problema.** A interpretação causal ('conhecimento de domínio, e não capacidade') depende do contraste PhysBERT × GTE-base, mas a evidência é a de dois testes contra um terceiro: um significativo, outro não. A diferença entre significativo e não significativo não é, em si, um teste. Além disso há um único treinamento por base: a variância entre sementes do ajuste do reordenador não é medida, e um efeito de +0,009 de nDCG@10 é da ordem do que uma semente pode mover. O controle de variáveis é bom; o que falta é o teste direto e a ressalva de semente.

**Evidência.** t1c_resultado.json e t1c_resultado_braco_gte.json: cada base é testada apenas contra a fusão (PhysBERT 97×140, p = 0,0062; GTE-base 114×106, p = 0,637), em execuções separadas, com `semente 17` única. Não há teste pareado PhysBERT × GTE-base em nenhum arquivo; t1b_physbert_n2000.json e t1b_gte_n2000.json não guardam posições por consulta, de modo que ele não pode ser calculado dos artefatos versionados. Diferença pontual entre os dois reordenadores: 0,1666 − 0,1530 = 0,0136 de nDCG@10, sem intervalo.

**Correção proposta.** "Apenas a base pré-treinada no domínio supera a fusão ao limiar registrado. O contraste que a interpretação exige — PhysBERT contra GTE-base, diretamente — não foi testado de forma pareada: as duas bases foram medidas em execuções distintas e comparadas, cada uma, à fusão (diferença pontual de 0,0136 de nDCG@10 entre elas). Há um treinamento por base, com a mesma semente, e a variância entre sementes não foi medida. A leitura de que o mecanismo é conhecimento de domínio é a registrada previamente e é compatível com os dados; não está estabelecida por um teste direto." Se as posições por consulta das duas execuções puderem ser regeneradas, reportar McNemar em k = 10 e bootstrap de nDCG@10 entre as duas.

### L378 · maior · erro estatístico · aplicado

> caiu de +0,0091 (p = 0,0081) para +0,0045 (p = 0,086)

**Problema.** Dois problemas. (1) O p-valor é atribuído a uma diferença de nDCG@10 mas testa pertencimento ao top-10 — instrumento que a própria Seção 5.3 declara inadequado para reordenadores. (2) A afirmação 'caiu' compara um resultado significativo com um não significativo; a diferença entre os dois ganhos (a interação recuperador × reordenação) não foi testada, e a conta grosseira acima indica que ela não se distingue de zero. A predição registrada ('um recuperador melhor deixa menos a reordenar') fica, assim, compatível com os dados, não confirmada.

**Evidência.** t1b2_resultado.json: +0,0091 = 0,1685 − 0,1594 e +0,0045 = 0,1688 − 0,1643 são diferenças de nDCG@10; os p-valores são de McNemar em k = 10 (recall@10): antigo 116×161, p = 0,00808; novo 128×158, p = 0,0862. Em recall@10 os ganhos são +0,0225 e +0,0150. Saldo de discordantes: +45 contra +30; sob independência, z ≈ (45−30)/√(277+286) ≈ 0,63.

**Correção proposta.** "O ganho da reordenação sobre a fusão, em recall@10, é de +0,0225 com o recuperador antigo (116 contra 161 discordantes, p = 0,0081) e de +0,0150 com o novo (128 contra 158, p = 0,086); em nDCG@10, +0,0091 e +0,0045. A direção é a predita, mas a diferença entre os dois ganhos não foi testada, e o desenho não a estabelece." Se as posições por consulta do T1b2 estiverem disponíveis, reportar o bootstrap pareado da diferença das diferenças.

### L391 · maior · afirma mais que a evidência · aplicado

> São cinco medições pareadas, com dois reordenadores, e em nenhuma o estágio vence o recuperador isolado.

**Problema.** As 'cinco medições' não são cinco réplicas: uma está contada duas vezes, três compartilham os mesmos escores do mesmo reordenador sobre as mesmas 2.000 consultas, e todas usam o mesmo conjunto de consultas. O que os dados mostram é um efeito pontual consistentemente positivo (+0,008 a +0,009 de nDCG@10, limite inferior em −0,001) num desenho com cerca de 43% de poder. Isso é ausência de evidência com poder insuficiente, e a frase lê-se como cinco fracassos independentes. A decisão de retirar o estágio continua defensável pelo custo, mas não por 'não vence em nenhuma'.

**Evidência.** t1d_resultado.json (braço antigo, @100): ΦEmb 141 × ΦEmb+ΦRank 168, p = 0,139. t1e_resultado.json (@100): 141 × 168, p = 0,13899 — a mesma medição (mesmo reordenador, mesmas consultas, mesma profundidade). As demais: T1d novo 157×174 (p = 0,379); T1e @50 126×150 (p = 0,166); T1e @200 152×180 (p = 0,138). Nas cinco, o saldo é a favor do reordenador. Recalculado das posições por consulta de t1e_resultado.json: ΔnDCG@10 @50 +0,0079 [−0,0015; +0,0176], @100 +0,0091 [−0,0009; +0,0195], @200 +0,0088 [−0,0014; +0,0193]. Poder do desenho para o efeito nominal de 0,0091 com n = 2.000: ≈ 43%.

**Correção proposta.** "Em quatro medições pareadas não independentes — dois reordenadores e três profundidades sobre as mesmas 2.000 consultas —, o estágio fica acima do recuperador isolado em estimativa pontual (+0,008 a +0,009 de nDCG@10) e em nenhuma o intervalo exclui zero (@100: +0,0091 [−0,0010; +0,0191]). Com 2.000 consultas, o poder para detectar um efeito desse tamanho é de cerca de 43%: o resultado não demonstra que o estágio é inútil, e sim que o ganho, se existir, é pequeno demais para pagar 109 M de parâmetros em serviço."

### L491 · maior · afirma mais que a evidência · aplicado

> Entre os braços, o tratado fica à frente: +0,0076 de nDCG@10 [+0,0013; +0,0139]

**Problema.** O efeito é de uma medida secundária, com p ≈ 0,019 sem correção: passa em Bonferroni para duas comparações (0,025) e não passa para três (0,0167), e a família não é declarada. Somam-se: primária não decidida, recall@1 sem diferença, uma semente por braço, assimetria de execução a favor do tratado (admitida na linha 613) e seleção de checkpoint em metade do conjunto de teste, cuja magnitude é comparável ao efeito. O texto reconhece o limite inferior próximo de zero, mas 'fica à frente' e 'o ganho de recuperação do substituto se repete' afirmam mais do que isso sustenta. A comparação do melhor de dois braços contra a base tem viés de seleção leve, mas resiste à correção.

**Evidência.** t2eq_cpt_emb_comparacao.json: diferença 0,00757, IC95 [0,00128; 0,01388] → erro padrão 0,00321, z = 2,36, p ≈ 0,019 (recalculado). Três comparações lidas na mesma tabela com IC de 95% sem correção ('três desfechos'); a segunda é do 'melhor_braco' contra a base (campo `contra_a_barra.melhor_braco`), z = 2,79, p ≈ 0,005. A medida primária da mesma ablação é 'NÃO DECIDIDO' (−0,00039 [−0,00161; +0,00087]) e o recall@1 dá 63×60, p = 0,857. Os dois braços são checkpoints `-melhor`, escolhidos em 1.000 dos 2.000 itens do veredito (ver achado da linha 108). T1h e T1i, posteriores, já usam IC de 98,75% e 97,5% (campo `protocolo.ic`).

**Correção proposta.** "Entre os braços, a estimativa favorece o tratado: +0,0076 de nDCG@10 [+0,0013; +0,0139], IC de 95% sem correção para as três comparações desta tabela (p ≈ 0,02; não resistiria a Bonferroni para três), com recall@1 sem diferença (63 contra 60 discordantes, p = 0,86). O sinal coincide com o do substituto, com um onze avos da magnitude; com uma semente por braço, a assimetria de execução descrita acima e a escolha do ponto de verificação em parte do conjunto de avaliação, o resultado é sugestivo e não estabelece o ganho." E, na frase seguinte: "Contra a base sem pré-treinamento, o melhor dos dois braços — escolhido depois de medido — fica acima por +0,0103 [+0,0032; +0,0176]".

### L613 · maior · falta controle ou limitação · aplicado

> Todos têm uma semente por braço

**Problema.** Todos os ICs e p-valores do artigo condicionam no modelo treinado: capturam a variância de amostragem das consultas, não a de treinamento (semente, ordem dos lotes, amostra de pares). Para as Tabelas 9, 10, 14 e 22 isso não é dito em lugar nenhum, embora os artefatos do projeto o registrem. Diferenças de 0,001 a 0,02 entre execuções únicas (os degraus da curva de volume, o 'empate' do GTE-base, o efeito do reordenador) estão na faixa em que a variância entre sementes de um ajuste contrastivo costuma operar.

**Evidência.** A ressalva da linha 613 cobre só os pré-treinamentos das Seções 4.9.x (também linha 403). Para os recuperadores e reordenadores, o texto diz apenas 'mesma semente' (linhas 285 e 363). t1g_comparacao.json, `ressalvas`: "Uma semente por modelo."; t1i_comparacao.json: "Uma semente por base."; t1c_resultado.json: "semente 17". A Seção 3.6 (linhas 114-118) descreve os intervalos como 'por consulta' e não diz o que eles deixam de fora.

**Correção proposta.** Na Seção 3.6: "Os intervalos e testes condicionam nos modelos treinados: medem a variância de amostragem das consultas e não a de treinamento. Cada modelo deste trabalho foi treinado uma única vez, com a mesma semente, e a variância entre sementes não foi medida em nenhuma comparação." Nas Limitações, depois da linha 613: "O mesmo vale para os recuperadores e reordenadores das Tabelas 8 a 16 e 22: uma execução por modelo. Diferenças entre execuções da ordem de 0,01 de nDCG@10 ou menos — entre elas o empate da Tabela 10 e o ganho do reordenador de domínio — não devem ser lidas como estabelecidas contra a variância de treinamento."

### L116 · menor · inconsistência interna · aplicado

> Onde há múltiplas variantes comparadas contra o mesmo controle, aplica-se correção de Bonferroni

**Problema.** A frase é geral e a prática, na v0.5, é restrita à Tabela 14. Nas Tabelas 11 e 23 a conclusão resiste à correção (p ≤ 0,00073 na Tabela 11; z ≈ 2,9 para o segundo sistema da Tabela 23, pelos ICs do texto), então é problema de enunciado, não de resultado — exceto na Seção 4.9.3, tratada em achado próprio.

**Evidência.** O campo `alfa_bonferroni: 0.025` só existe em t1c_resultado*.json (Tabela 14). t1b_resultado.json (Tabela 11: três sistemas contra a fusão, em k = 1 e k = 10) não tem limiar corrigido; t2eq_cpt_emb_comparacao.json (Seção 4.9.3: três comparações) e t1g_comparacao.json usam `ic95`; a linha 501 (Tabela 23: três sistemas contra o BM25) declara IC sem correção. T1h/T1i, fora do texto, aplicam Bonferroni nos ICs.

**Correção proposta.** "A correção de Bonferroni é aplicada onde foi registrada antes da coleta: na comparação de bases do reordenador (Tabela 14, duas variantes, limiar de 0,025). Nas demais tabelas com mais de uma comparação contra o mesmo controle (Tabelas 11, 22 e 23 e Seção 4.9.3), os p-valores e intervalos são nominais, sem correção, e a família de comparações é a da própria tabela."

### L306 · menor · clareza · aplicado

> produziu +0,039 com p = 0,161

**Problema.** A frase lê-se como se p = 0,161 fosse o teste de +0,039 de nDCG@10; é McNemar em recall@1 com 51 discordantes. O 'quase falso negativo' é sobretudo um problema de poder (256 itens, 51 discordantes), o que reforça o achado sobre a ausência de análise de sensibilidade.

**Evidência.** t1f_base.json (n_candidatos = 256), bloco pareado: 'GTE-base@400k (T1f)' × 'MiniLM@400k (o nosso)', 31×20, 51 discordantes, p = 0,1608 — McNemar em recall@1. O p de 1,5 × 10⁻¹⁰ da mesma frase também é de recall@1 (219×104, recalculado 1,45 × 10⁻¹⁰).

**Correção proposta.** "Ela foi feita com 256 candidatos, valor padrão do avaliador, e produziu +0,039 de nDCG@10, com o pareado em recall@1 sobre apenas 51 discordantes (31 contra 20, p = 0,161); com os 2.000 do protocolo, +0,0633, com 219 contra 104 (p = 1,5 × 10⁻¹⁰)."

### L391 · menor · clareza · pendente

> exigiria cerca de 6.640 consultas

**Problema.** O número corresponde a 90% de poder com α = 0,05 bilateral, mas o texto não declara nem o poder nem o α, e o cálculo não está em código versionado. A 80% — a convenção mais comum — seriam cerca de 5.000 consultas.

**Evidência.** Recalculado: com o IC do texto ([−0,0010; +0,0191], erro padrão 0,00513, desvio por consulta 0,229), n = ((1,96 + z_poder)·dp/0,0091)² dá 4.984 a 80% de poder e 6.672 a 90%; com o desvio recalculado das posições de t1e_resultado.json (0,2313), 5.072 e 6.790. O número 6.640 não aparece em scripts/, src/ nem kaggle/ (só em ESTADO.md e em notas de t1f_base.json/t1a_curvas).

**Correção proposta.** "Estabelecer o ganho nominal de 0,0091, se ele existir, exigiria cerca de 6.700 consultas para 90% de poder com α = 0,05 bilateral (cerca de 5.000 para 80%), a partir do desvio padrão por consulta de 0,23 medido aqui".

### L419 · menor · afirma mais que a evidência · pendente

> Na sonda tensorial não há diferença (0,417 contra 0,333; 19 contra 13 discordantes, p = 0,38)

**Problema.** Com 72 itens, o teste não tem poder para distinguir uma diferença de 8 pontos percentuais de zero; 'não há diferença' é evidência de ausência afirmada a partir de ausência de evidência. O mesmo vale para a linha 446 (12×15, p = 0,70).

**Evidência.** Recalculado: binomtest(13, 32) → p = 0,377 (confere). São 72 itens (linha 446 e t2a_sonda_tensorial.json), diferença pontual de 0,083 (6 itens). Com 32 discordantes, o teste só rejeitaria a 5% com divisão de 23×9 ou mais extrema.

**Correção proposta.** "Na sonda tensorial a diferença não é estabelecida (0,417 contra 0,333; 19 contra 13 discordantes, p = 0,38): com 72 itens, o instrumento não distingue uma diferença de oito pontos de zero."

### Conferido e correto

- P-valores de McNemar exato recalculados com scipy.stats.binomtest a partir das contagens do texto, todos conferem: 196×165 → 0,1142 (linha 243); 188×153 → 0,0654 (linha 271); 104×71 → 0,0153 (linha 285); 219×104 → 1,45e-10, 147×150 → 0,908, 148×144 → 0,861 (Tabela 10); 97×140 → 0,00625 (Tabela 15); 92×82 → 0,495, 141×168 → 0,139, 157×174 → 0,379 (linha 378); 19×20 → 1,0 e 167×141 → 0,154 (linha 389); 63×60 → 0,857 (linha 491); 12×15 → 0,701 (Tabela 19); 19×13 → 0,377 (linha 419).
- Tabela 14: 229 discordantes com p = 0,1458 corresponde a 103×126 e 220 com p = 0,6371 a 106×114 (reconstruído por busca); confere com t1c_resultado.json e t1c_resultado_braco_gte.json (126×103 e 114×106 em t1b_controle_n2000.json e t1b_gte_n2000.json). Limiar de Bonferroni 0,025 gravado em `alfa_bonferroni`.
- Tabela 11: p = 0,00073 (69×34), 0,00018 (69×31) e 0,118 (61×44) conferem com t1b_resultado.json, n = 1.000.
- Linha 347: p = 1,5e-8 (143×62) e p = 0,949 (124×122) conferem com t1b2_resultado.json. Linha 374: k = 1, 66×64, p = 0,930 confere com t1b_physbert_n2000.json.
- Linha 285: 6 M contra 3 M, p = 0,0012 (115×70) confere com g1_t1a6m.log e g1_resultado.json.
- Linha 304 e Tabela 10: T1g, nDCG@10 0,6211 contra 0,6223, diferença −0,00119, IC95 [−0,011; +0,00842], e +0,0431 [0,0328; 0,0537] contra o MiniLM de 1,5 M conferem com t1g_comparacao.json; a regra 'EMPATE = IC cruza 0' está de fato escrita antes em kaggle/t1g_gte_1m.py.
- Linha 389: recalculei do zero, a partir das posições por consulta de t1e_resultado.json, o bootstrap pareado de nDCG@10 @100 (+0,0091 [−0,0009; +0,0195] com 20.000 reamostras e outra semente; o texto dá [−0,0010; +0,0191]), as 421 consultas com alvo no top-10 nos dois sistemas e a divisão 167 rebaixa × 141 eleva. Tudo confere dentro do ruído de Monte Carlo.
- Linha 491: +0,0076 [+0,0013; +0,0139], +0,0103 [+0,0032; +0,0176] e +0,0694 [+0,0580; +0,0812] conferem com t2eq_cpt_emb_comparacao.json; desfecho primário 'não decidido' (−0,00039 [−0,00161; +0,00087]) confere.
- Linha 287: '15 avaliações consecutivas abaixo do próprio pico, queda de 1,0%' confere com t1a_curvas_de_treino.json (pico 0,7022 no passo 43.800 de 46.875, avaliação a cada 200 passos, final 0,6951).
- O bootstrap reamostra a unidade pareada correta (item/consulta) nas comparações de recuperação, e os braços de cada comparação são medidos sobre os mesmos itens na mesma sessão (`todos_medidos_nesta_sessao: true` em t1g, t1i, t2eq_cpt_emb).
- A Seção 5.3 declara corretamente que o McNemar em k = 10 é instrumento inadequado para reordenador e que o bootstrap de nDCG@10 foi acrescentado depois dos dados.

### Não verificável nesta máquina

- Variância entre sementes de qualquer modelo: exige retreinar (checkpoints, pares e GPU ausentes). Nenhum artefato versionado tem mais de uma semente por braço.
- Magnitude real do viés de seleção de checkpoint no conjunto do veredito: exigiria remedir os modelos só nos 1.000 itens que não participaram da escolha (pares de validação e checkpoints ausentes). Também não pude confirmar que `models/phiemb-do-sistema` é o checkpoint de pico do passo 43.800 (inferido do padrão `-melhor` e de scripts/instalar_phiemb.py, não lido).
- Teste pareado direto PhysBERT × GTE-base como base do reordenador (Tabela 14): t1b_physbert_n2000.json e t1b_gte_n2000.json não guardam posições por consulta.
- Teste da interação recuperador antigo/novo × reordenação (linha 378): t1b2_resultado.json não guarda posições por consulta; a conta z ≈ 0,63 supõe independência e é só ordem de grandeza.
- IC por bootstrap dos +0,0435 de nDCG@10 do bi-encoder de 6 M contra o GTE-large, e dos degraus da Tabela 9: os escores por item não estão versionados.
- Parâmetros efetivos do bootstrap (número de reamostras, método do percentil, semente) em `bootstrap_pareado_itens`: não li a função; só a linha 389 do texto declara 20.000 reamostras.
- t1i_comparacao.json traz McNemar idêntico (147×135, p = 0,5125) para gte-small e bge-small contra o sistema. Os dois têm recall@1 de 0,4255, o que fixa o saldo em −12, mas a coincidência das duas contagens merece conferência nos escores por item antes de T1i entrar no texto.
- Tabela 23 (Seção 4.10) e Tabelas 17–18: não recalculei os intervalos; só conferi a coerência interna dos ICs citados.


## Referências e trabalhos relacionados

### L25 · maior · afirma mais que a evidência · aplicado

> O Galactica [2], única tentativa de larga escala de treinar um modelo científico de fundação a partir do zero, foi retirado de circulação três dias após o lançamento

**Problema.** Três problemas somados: (1) fato sem fonte — a retirada não está em [2]; (2) imprecisão — retirou-se a demo, não o modelo; (3) inferência indevida — alucinação de citações em uso público não é evidência de que treinar do zero falha, e o próprio [2] reporta ganhos quantitativos. 'Única tentativa' é universal sem verificação. Um revisor atacaria esse parágrafo, que sustenta a premissa da Introdução.

**Evidência.** A referência [2] é o próprio artigo do Galactica (arXiv:2211.09085), que não pode sustentar a retirada posterior. Pelo que se sabe do caso, o que saiu do ar três dias depois (novembro de 2022) foi a demonstração pública; os pesos continuaram disponíveis. O artigo [2] reporta, ao contrário do que o parágrafo sugere, resultados fortes em raciocínio matemático (supera modelos gerais maiores no MATH e em equações LaTeX). A frase anterior ('sustentada pela literatura recente: treinamento a partir de inicialização aleatória é justificável apenas onde o dado é excedente — encoders na faixa de 10⁸ parâmetros') não tem citação alguma.

**Correção proposta.** 'O Galactica [2], a tentativa mais visível de treinar um modelo científico de fundação a partir do zero, teve a demonstração pública retirada três dias após o lançamento [fonte jornalística], por alucinação de citações, embora o artigo reporte resultados competitivos em raciocínio matemático. Minerva [3], Llemma [4] e DeepSeekMath [5] seguiram o outro caminho — pré-treinamento continuado sobre bases gerais — e reportam ganhos substanciais.' Retirar 'A evidência é consistente' ou citar trabalhos que comparem diretamente as duas estratégias.

### L48 · maior · falta controle ou limitação · aplicado

> O SPECTER [7] introduz o uso do grafo de citação como sinal de relacionamento entre documentos científicos, abordagem que este trabalho adota.

**Problema.** O trabalho adota exatamente o sinal do SPECTER (pares do grafo de citação) e avalia em predição de citação, mas não cita nem mede os sucessores diretos dessa linha — SciNCL (Ostendorff et al., EMNLP 2022) e SPECTER2/SciRepEval (Singh et al., EMNLP 2023) —, nem o próprio SPECTER como baseline. São os competidores de mesma técnica; SciBERT e PhysBERT sem ajuste contrastivo por citação não ocupam esse lugar. Sem eles, a afirmação de que o bi-encoder supera as alternativas disponíveis fica sem o baseline mais pertinente.

**Evidência.** grep -i 'SciNCL|SPECTER2|SciRepEval|SciDocs' no rascunho: zero ocorrências. grep de identificadores de modelo em scripts/, src/, docs/adr, data/processed/avaliacao: aparecem all-MiniLM-L6-v2, thenlper/gte-*, thellert/physbert_cased, allenai/scibert_scivocab_uncased, ModernBERT-base, e5-small, bge-small; nenhum allenai/specter, allenai/specter2 ou malteos/scincl. As Tabelas 6 e 8 comparam contra SciBERT, PhysBERT, MiniLM e GTE apenas.

**Correção proposta.** Acrescentar à Seção 2.1: 'O SciNCL [n] e o SPECTER2 [n] refinam essa linha, com amostragem de vizinhos no grafo de citação e treinamento multitarefa, respectivamente.' E, ou medir SPECTER2 e SciNCL no protocolo da Tabela 8, ou declarar na Seção 6: 'Os modelos de embedding científico treinados por citação (SPECTER, SciNCL, SPECTER2) não foram medidos; a comparação da Tabela 8 não inclui o competidor de mesma técnica.'

### L223 · maior · afirma mais que a evidência · aplicado

> contrariando a expectativa derivada de Bostrom e Durrett [21], a codificação por pares de bytes supera o modelo de unigramas em texto de Física

**Problema.** O resultado medido (BPE produz menos tokens) não contraria Bostrom e Durrett, que não preveem menor fertilidade para o unigrama, e sim melhor desempenho a jusante. 'Supera' fica sem métrica declarada. Além disso, a linha 62 promete 'evidência equivalente' nas Seções 4.4 e 4.9.1, mas a 4.9.1 não tem braço de unigrama: não há nenhuma evidência extrínseca BPE × unigrama no trabalho. O próprio rascunho mostra, em 4.9.1, que a métrica intrínseca pode inverter de sinal na extrínseca.

**Evidência.** Resumo de Bostrom & Durrett na ACL Anthology (2020.findings-emnlp.414, consultado): a comparação é por desempenho em tarefas após ajuste fino ('fine-tuned task performance'), em inglês e japonês, com o unigrama 'matches or outperforms BPE' e melhor alinhamento morfológico; não há afirmação sobre contagem de tokens. No rascunho, a comparação BPE × unigrama é só a das variantes A e B da Tabela 5 (fertilidade 0,9620 × 0,9816; tokens/eq. 7,31 × 8,36), métrica intrínseca de compressão. A Seção 4.9.1 (linha 412) compara A e E — regras de pré-tokenização, ambas BPE; 'unigram' só ocorre nas linhas 62, 213 e 223.

**Correção proposta.** Linha 223: 'Segundo, a codificação por pares de bytes produz menos tokens que o modelo de unigramas em texto de Física, com margem seis vezes maior em equações (13%) do que em prosa (2%). O resultado é de compressão e não contraria Bostrom e Durrett [21], cuja vantagem do unigrama é medida em desempenho a jusante; essa comparação extrínseca não foi feita aqui.' Linha 62: 'Não localizamos evidência publicada equivalente para texto em LaTeX; a Seção 4.4 apresenta uma comparação intrínseca, e a Seção 4.9.1 mostra, para outra variável do tokenizador, que a métrica intrínseca não antecipa a extrínseca.'

### L300 · maior · falta controle ou limitação · aplicado

> o que sugere que o ajuste ensina sobretudo a tarefa de recuperação, que uma base contrastiva geral já sabe

**Problema.** A base 'geral' do sistema principal já foi treinada em cerca de 169 milhões de pares de citação do S2ORC, isto é, na mesma tarefa (citante → citado) e possivelmente sobre os mesmos artigos do arXiv que compõem a avaliação. Isso (a) explica de modo alternativo o ganho pequeno do ajuste (+0,049) e o resultado 'base importa mais que volume'; (b) torna impreciso o rótulo 'geral, sem ajuste' na Tabela 8; (c) abre possibilidade de sobreposição entre os pares de validação e os dados de treino da base, que não está medida. É a ressalva que um revisor de recuperação levantaria primeiro, e o texto não a declara.

**Evidência.** grep em scripts/, src/, docs/adr e data/processed/avaliacao: a base usada é 'sentence-transformers/all-MiniLM-L6-v2' (136 ocorrências). A ficha do modelo (huggingface.co/sentence-transformers/all-MiniLM-L6-v2, consultada) lista entre os dados de treino: 'S2ORC Citation pairs (Abstracts)' 116.288.806; 'S2ORC Citation pairs (Titles)' 52.603.982; 'S2ORC (Title, Abstract)' 41.769.185; 'SPECTER citation triplets' 684.100; total 1.170.060.424 pares. No rascunho, grep por 'all-MiniLM', 'S2ORC' (fora da linha 60/691) e 'pares de cita' não devolve nenhuma declaração disso; a Seção 6 (linhas 603–627) não o menciona; a Tabela 8 (linha 266) rotula a base como 'MiniLM-L6 [12] (geral, sem ajuste)'.

**Correção proposta.** Nomear o checkpoint (all-MiniLM-L6-v2) na Seção 3.4 e acrescentar à Seção 6: 'A base do bi-encoder (all-MiniLM-L6-v2) foi treinada pelos seus autores em mais de 1 bilhão de pares, entre os quais cerca de 169 milhões de pares de citação do S2ORC e os tripletos do SPECTER; ela não é ingênua quanto à tarefa de predição de citação, e a sobreposição entre esses pares e o conjunto de validação deste trabalho não foi medida. O ganho medido do ajuste é, portanto, ganho sobre uma base que já viu a tarefa.' Na Tabela 8, trocar 'geral, sem ajuste' por 'geral contrastivo, sem ajuste próprio'. Declarar o equivalente para o GTE [10] depois de conferir a tabela de dados de pré-treinamento do artigo.

### L48 · menor · afirma mais que a evidência · pendente

> O INDUS [8] cobre Ciências da Terra, heliofísica, ciências planetárias e astrofísica, mas não as subáreas de altas energias, matéria condensada e informação quântica.

**Problema.** A lista do rascunho omite 'biology' e, mais importante, 'physics', que os autores declaram cobrir. A negativa sobre altas energias, matéria condensada e informação quântica não foi conferida contra a composição do corpus do INDUS e, como está, contradiz o que o resumo afirma.

**Evidência.** Resumo de arXiv:2405.10725 (consultado): os domínios declarados são 'Earth science, biology, physics, heliophysics, planetary sciences and astrophysics'.

**Correção proposta.** 'O INDUS [8] declara cobrir Ciências da Terra, biologia, física, heliofísica, ciências planetárias e astrofísica, com corpus orientado às divisões científicas da NASA; não foi avaliado pelos autores em altas energias, matéria condensada ou informação quântica.' (Conferir a segunda oração contra a seção de dados do artigo antes de mantê-la.)

### L48 · menor · afirma mais que a evidência · aplicado

> seu corpus de treinamento é composto majoritariamente de literatura biomédica, com participação marginal de Física

**Problema.** 'Participação marginal de Física' atribui ao artigo um número que ele não dá. O que a fonte sustenta é que o corpus é de ciência da computação e biomedicina.

**Evidência.** Beltagy et al. (2019) descrevem o corpus como 1,14 milhão de artigos do Semantic Scholar, 18% de ciência da computação e 82% do domínio biomédico amplo; Física não é reportada como categoria.

**Correção proposta.** 'seu corpus de treinamento é composto de 82% de literatura biomédica e 18% de ciência da computação, sem fatia declarada de Física'.

### L60 · menor · erro de referência · aplicado

> O peS2o [16] é um corpus de artigos científicos de acesso aberto derivado do S2ORC

**Problema.** O S2ORC é nomeado e é a origem da fatia que responde por 14,60 dos 27,75 bilhões de tokens, e não tem referência. Dois trabalhos pertinentes às afirmações do artigo também faltam: o unarXive (Saier, Krause e Färber, 2023), corpus de texto integral do arXiv a partir do fonte LaTeX com fórmulas e rede de citação — diretamente comparável ao que as Seções 3.3 e 4.3 constroem —, e o BEIR (Thakur et al., 2021), que é o referente natural da ressalva da linha 606 e contém o SCIDOCS, tarefa de predição de citação.

**Evidência.** grep 'S2ORC' no rascunho: só as linhas 60 e 691 (título de [16]); não há referência a Lo et al., 'S2ORC: The Semantic Scholar Open Research Corpus', ACL 2020. grep 'unarXive|BEIR|MTEB': zero ocorrências; a linha 606 fala em 'benchmarks com anotação humana' sem citar nenhum.

**Correção proposta.** Citar Lo et al. (ACL 2020) junto ao S2ORC na linha 60; acrescentar em 2.3: 'O unarXive [n] oferece texto integral do arXiv a partir do fonte LaTeX com fórmulas e contexto de citação preservados; não foi usado aqui.'; e na linha 606: 'benchmarks com anotação humana, como os do BEIR [n]'.

### L64 · menor · erro de referência · pendente

> A pseudo-verossimilhança de Salazar et al. [30] mascara um token por vez; Kauf e Ivanova [31]

**Problema.** A lista é numérica por ordem de aparição, e as duas últimas entradas, acrescentadas depois, quebram a ordem. Estilos numéricos (ACL, IEEE, Vancouver) exigem renumeração.

**Evidência.** Script sobre o rascunho — ordem de primeira citação: 1,…,21,30,31,22,23,…,29. As referências [30] e [31] aparecem na linha 64, antes de [22] (linha 72).

**Correção proposta.** Renumerar: Salazar et al. passa a [22] e Kauf e Ivanova a [23], deslocando as atuais [22]–[29] para [24]–[31], com as citações do corpo (linhas 64, 128, 559 e 72–114, 162–164, 316–318, 454, 486, 505, 575) ajustadas.

### L90 · menor · erro de referência · aplicado

> MiniLM-L6 [12] de 23 milhões de parâmetros

**Problema.** A citação [12] cobre a destilação, não o modelo efetivamente usado como base nem o seu treinamento contrastivo. O leitor que seguir [12] não encontra o modelo das Tabelas 8–10.

**Evidência.** O checkpoint usado é sentence-transformers/all-MiniLM-L6-v2 (grep no código). A ficha do modelo indica como base 'nreimers/MiniLM-L6-H384-uncased'. O artigo [12] (Wang et al., 2020) descreve o método de destilação; o modelo de 6 camadas e 384 dimensões com ajuste contrastivo em 1,17 bilhão de pares é do projeto Sentence-Transformers, não de [12].

**Correção proposta.** 'all-MiniLM-L6-v2 (23 M), modelo do projeto Sentence-Transformers [11] obtido por ajuste contrastivo, em mais de 1 bilhão de pares, de um MiniLM [12] de 6 camadas' — e acrescentar à lista a ficha do modelo com data de acesso.

### L96 · menor · erro de referência · pendente

> taxa adotada seguindo os resultados de Warner et al. [26]

**Problema.** Atribui a [26] um resultado que [26] toma de outro trabalho. A configuração é de Warner et al.; a evidência é de Wettig et al.

**Evidência.** O ModernBERT usa taxa de mascaramento de 30%, mas como escolha de projeto apoiada em Wettig et al. (EACL 2023, 'Should You Mask 15% in Masked Language Modeling?'); o resultado empírico de que 15% é subótimo é desse trabalho. grep por 'Wettig' no rascunho: zero ocorrências.

**Correção proposta.** 'taxa adotada seguindo a configuração de Warner et al. [26], que se apoia em Wettig et al. [n]' — e acrescentar a referência.

### Conferido e correto

- Por script: as 31 referências são todas citadas no corpo e toda citação [n] do corpo existe na lista; não há órfãs, citações indefinidas nem entradas duplicadas.
- [1] Hoffmann et al., NeurIPS 2022: a relação de cerca de 20 tokens por parâmetro sustenta 'modelo de 8 bilhões requer aproximadamente 160 bilhões de tokens' (linha 23).
- [2] Galactica: autores e arXiv:2211.09085 corretos (a afirmação sobre a retirada é outro assunto, ver achado).
- [3] Minerva: lista de 14 autores e venue NeurIPS 2022 corretos; é pré-treinamento continuado do PaLM.
- [4] Llemma: autores conferidos em arXiv:2310.10631; venue ICLR 2024 confirmada (proceedings.iclr.cc e OpenReview 4WnqRR915j); pré-treinamento continuado do Code Llama, como o corpo afirma.
- [5] DeepSeekMath, arXiv:2402.03300, 2024: correto; pré-treinamento continuado sobre base geral.
- [6] SciBERT, EMNLP-IJCNLP 2019: correto; 110 M coerente com BERT-base.
- [7] SPECTER, ACL 2020, p. 2270–2282: correto; uso do grafo de citação como sinal corresponde ao trabalho.
- [8] INDUS: três primeiros autores, EMNLP 2024 Industry Track e arXiv:2405.10725 conferidos na página do arXiv.
- [9] PhysBERT: autores (Hellert, Montenegro, Pollastro) e '1,2 milhão de artigos de Física do arXiv' conferidos no resumo de arXiv:2408.09574; o identificador usado no código é thellert/physbert_cased.
- [10] GTE, arXiv:2308.03281, seis autores: correto; os checkpoints usados (thenlper/gte-base, thenlper/gte-large) são os desse artigo, e 109 M / 335 M conferem.
- [11] Sentence-BERT, EMNLP-IJCNLP 2019, p. 3982–3992: correto.
- [12] MiniLM, NeurIPS 2020, autores corretos (a adequação da citação ao checkpoint usado é o achado da linha 90).
- [13] DPR, EMNLP 2020, p. 6769–6781; [14] ANCE, ICLR 2021; [15] RocketQA, NAACL-HLT 2021: autores e venues corretos, e as afirmações da Seção 2.2 (negativos de BM25; negativos do próprio índice; filtragem de falsos negativos por cross-encoder) correspondem aos trabalhos.
- [16] peS2o (Soldaini e Lo, 2023, ODC-By): correto; 'derivado do S2ORC, texto extraído de PDF' corresponde; o total de 38.972.211 documentos é compatível com a versão v2.
- [17] RedPajama, NeurIPS 2024 D&B, arXiv:2411.12372: correto; a fatia arXiv é de fato construída do fonte LaTeX.
- [18] OpenWebMath, ICLR 2024: correto.
- [19] Sennrich et al., ACL 2016, p. 1715–1725; [20] Kudo, ACL 2018, p. 66–75: corretos.
- [21] Bostrom e Durrett, Findings of EMNLP 2020, p. 4617–4624: metadados conferidos na ACL Anthology; a descrição da linha 62 ('vantagem do unigrama em linguagem natural, atribuída a melhor alinhamento morfológico') é fiel.
- [22] OpenAlex, arXiv:2205.01833; [23] van den Oord et al., arXiv:1807.03748; [24] Gao et al., RepL4NLP 2021, p. 316–321; [25] Robertson e Zaragoza, FnTIR 3(4), 2009: corretos.
- [26] ModernBERT: arXiv:2412.13663 e três primeiros autores conferidos; '2 trilhões de tokens' e 'comprimento nativo de 8.192' conferidos no resumo; 150 M para a versão base e máscara de 30% conferem com o artigo.
- [27] Cormack, Clarke e Büttcher, SIGIR 2009, p. 758–759: correto; k = 60 é o valor do artigo.
- [28] McNemar 1947 e [29] Wilson 1927: volume, número e páginas corretos.
- [30] Salazar et al., ACL 2020, p. 2699–2712 e [31] Kauf e Ivanova, ACL 2023 Short Papers, p. 925–935: corretos; a descrição da linha 64 (a PLL superestima palavras de vários tokens; a correção mascara também os tokens seguintes da mesma palavra) é fiel.
- Tabela 5, linhas A e B: 0,9816/0,9620 = +2,0% em prosa e 7,31/8,36 = −12,6% em equações, coerente com '13%' e '2%' e 'seis vezes' da linha 223 (os números; a leitura é o achado).

### Não verificável nesta máquina

- Se os dados de pré-treinamento e ajuste do GTE [10] incluem pares do S2ORC ou de citação: o PDF de arXiv:2308.03281 foi baixado mas não pôde ser lido (sem pdftotext/poppler na máquina) e o orçamento de consultas web se esgotou. É preciso conferir a tabela de dados do apêndice antes de chamar o GTE de 'geral' sem ressalva (linhas 238, 264, 489, 583).
- Sobreposição entre os pares de validação do trabalho e os pares de citação do S2ORC vistos pelo all-MiniLM-L6-v2 (e eventualmente pelo GTE): exige os pares, que não estão nesta máquina.
- Se thellert/physbert_cased é o checkpoint final de embedding do PhysBERT ou só o de pré-treinamento mascarado — o que muda a justiça da comparação nas Tabelas 6 e 8; exigiria consultar a ficha do modelo.
- [9] volume, número de artigo (046105) e DOI 10.1063/5.0238090 na APL Machine Learning: a página do arXiv mostrou apenas o DOI do arXiv; conferido só de memória.
- [26] publicação nos Proceedings of ACL 2025: a página do arXiv não traz o campo de venue; conferido só de memória.
- Composição do corpus de treinamento do INDUS (se o ADS usado inclui altas energias, matéria condensada e informação quântica): exigiria ler a seção de dados do artigo.
- A data e as circunstâncias da retirada da demonstração do Galactica: conhecidas de memória (novembro de 2022), não conferidas em fonte primária nesta sessão.
