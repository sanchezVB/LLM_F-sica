# Revisão independente do rascunho v0.5, e o que mudou na v0.6

**Data:** 2026-10-01 · **Objeto:** `docs/papers/rascunho-artigo-recuperacao-fisica.md`, v0.5 (commit `cad11c8`) · **Resultado:** v0.6, no mesmo arquivo · **Todos os achados:** [`revisao-v0.5-achados.md`](revisao-v0.5-achados.md)

## 1. Veredito

Os números do artigo conferem com a evidência versionada: de cerca de 400 afirmações rastreadas até um arquivo de `data/processed/avaliacao/` ou até o código, 214 foram registradas como corretas e 183 geraram achado. Quase nenhum achado é erro de transcrição. O problema da v0.5 é outro: em vários pontos o texto **afirma mais do que o experimento mediu**, e três propriedades do protocolo que condicionam todos os resultados de recuperação não estavam declaradas.

Os seis pontos que mudam o que o artigo diz:

1. **A ablação de tokenização não isola as regras de LaTeX.** As variantes A e E não diferem "apenas pelas regras": A também não tem fronteira de palavra (`ByteLevel(use_regex=False)` contra `use_regex=True`, `scripts/bakeoff_tokenizer.py:140-152`), e o BPE funde através de espaços. É por isso que A tem fertilidade abaixo de um token por palavra. O experimento da §4.9.1 rejeita o pacote, e não as regras.
2. **O ponto de verificação é escolhido em metade do conjunto que dá o veredito.** `preparar_pool(val, 1000)` e `preparar_pool(val, 2000)` usam a mesma semente e o mesmo arquivo; o primeiro é o prefixo do segundo. Não existe conjunto de teste. A margem de +0,044 sobre o GTE-large sobrevive; as diferenças pequenas lidas como empate ou ganho (−0,0012, +0,0076, +0,0103) são da ordem do viés.
3. **A base do recuperador já viu a tarefa.** O `all-MiniLM-L6-v2` foi treinado pelos autores com cerca de 1,17 bilhão de pares, 169 milhões deles pares de citação do S2ORC, que cobre o arXiv. O rótulo "geral, sem ajuste" da Tabela 8 enganava, e a citação [12] descrevia outro modelo.
4. **A divisão treino/validação é só por âncora.** Os documentos citados da validação aparecem como positivos no treino. É ajuste sobre acervo fixo — legítimo para o sistema —, mas o artigo não dizia, e os modelos gerais comparados não viram esse acervo.
5. **O Resumo dizia 400 mil pares onde o corpo diz 1 milhão.** A 400 mil, o GTE-base empata só em recall@1; fica 0,013 abaixo em nDCG@10 e abaixo em recall@10 (p ≤ 0,033 em qualquer tabela de discordância).
6. **A v0.5 parava antes do fim da história.** T1h, T1i, o encerramento da busca por encoder, a revisão da especificação do PB-Formula e a volta exploratória do reordenador no assistente não estavam no texto.

Nenhum desses pontos derruba um resultado central. O bi-encoder de 6 milhões de pares continua acima do GTE-large; o que muda é o que se pode dizer sobre *por que* e *com que margem*.

## 2. Como a revisão foi feita, e o que ela não cobre

Doze revisores independentes, um por seção ou lente (estatística, referências), cada um instruído a achar o arquivo de evidência de cada número — o rascunho não cita nenhum pelo caminho — e a registrar também o que conferiu e está certo. Somente leitura; nesta máquina, sem o corpus, sem os pares de treino e sem os checkpoints.

**O que deu errado no processo, para calibrar a confiança.** O desenho previa um cético independente por seção, encarregado de refutar cada achado. O limite de uso do plano interrompeu a execução duas vezes, e essa etapa não rodou. No lugar dela:

- conferi pessoalmente, contra o código e os JSON, cerca de 40 achados — todos os críticos e os maiores que entraram no texto — e todos se confirmaram;
- quatro seções foram revisadas duas vezes, por revisores que não se viram, e os achados principais coincidem;
- os achados mais graves foram encontrados de forma independente por dois ou três revisores (o checkpoint, por três; o S2ORC, por três; a semente única, por três; os 400 mil pares, por três).

Os 105 achados menores **não** passaram por segunda conferência. Estão no apêndice com a evidência que o revisor apresentou; trate-os como alegação bem fundamentada, não como fato.

**Não cobertos por revisor:** a lente de "revisor hostil de um venue" e a lente de integridade do texto. A segunda eu fiz por script (§7); a primeira está em §4 como opinião minha, marcada como tal.

**Uma pista falsa minha, registrada.** Eu havia anotado que a linha 409 trazia `\frac` corrompido. Não trazia: o rascunho não tem nenhum byte de controle. O defeito existia, mas no `ESTADO.md` (três ocorrências) e numa docstring de `src/phifm/eval/mlm_regiao.py` — corrigidos no commit `cad11c8`.

## 3. O que mudou na v0.6

110 substituições de texto e quatro referências novas; o estado de cada achado está em [`revisao-v0.5-achados.md`](revisao-v0.5-achados.md). Dos 183 achados, 141 estão aplicados, 4 em parte e 38 pendentes. **Aplicado quer dizer que o texto passou a dizer a verdade — em muitos casos, declarando uma limitação. Não quer dizer que a medição que falta foi feita** (§5).

Por tema:

| Tema | O que o texto dizia | O que diz agora |
|---|---|---|
| Tokenização (§4.4, §4.9.1, Resumo, contribuição 3) | A e E "diferem apenas" pelas regras de LaTeX | Diferem em dois pontos que o experimento não separa; a ablação que isolaria as regras não foi feita |
| Embedding (§4.4) | 16% e ~24% dos parâmetros | 22,1% e 31,2% (o 16% era a conta de V = 32 mil; corrigido também no DOC-05 e no DOC-07) |
| Monotonia do vocabulário (§4.4) | Resultado empírico | Propriedade do BPE; a medição dá só a magnitude |
| BPE contra unigramas (§4.4) | "Contrariando Bostrom e Durrett" | Resultado de compressão, que não os contraria; a comparação extrínseca não foi feita |
| Protocolo (§3.5, §3.6, §6) | — | Checkpoint escolhido no conjunto do veredito; divisão por âncora; 192 tokens para todos; "empate" = não rejeição; uma semente em todo modelo |
| Base do recuperador (§3.4, Tabela 8, §6) | "MiniLM-L6 [12] (geral, sem ajuste)" | `all-MiniLM-L6-v2`, já treinado com pares de citação do S2ORC |
| 6 M contra GTE-large (§4.5) | "Supera" em três métricas | Fica acima; recall@10 significativo em qualquer discordância (p < 10⁻⁵); IC de nDCG@10 não calculado |
| Curva de volume (§4.6) | "+0,032, +0,025, +0,020 por duplicação" | O primeiro degrau é de 3,75×; por duplicação, +0,017, +0,025, +0,020 |
| Saturação a 6 M (§4.6) | "Primeira com sinal de saturação" | A média das 15 últimas avaliações supera a das 15 anteriores; o pico é um ponto isolado |
| Base contra volume (Resumo, §4.6, §5.1, §7) | 400 mil pares = 6 milhões | 1 milhão de pares alcança; 400 mil não |
| T1h e T1i (§4.6, §3.4, §6, §7) | Ausentes | Dois parágrafos novos; busca por encoder encerrada por custo |
| Diagnóstico do reordenador (§4.7) | A concordância "explica" a ausência de ganho | Hipótese de trabalho; o indicador não distingue redundante de competente |
| "Deriva de código" (§4.7, §5.6) | O valor histórico diferia por deriva | Diferia pela profundidade (50 contra 100); BM25 e bi-encoder reproduzem-se exatamente |
| Regra do BM25 (§4.7) | "Registrada antes da execução" | Registrada depois da primeira execução e antes da segunda (`b24436d` 14:45, `f84f9bb` 14:55) |
| Domínio contra capacidade (§4.8, Resumo, contribuição 5, §6) | "Diferindo exclusivamente no corpus" | Diferem também em objetivo de pré-treinamento e vocabulário; sem teste direto entre as duas bases |
| Acerto@1 da Tabela 14 | Valores do último passo | Legenda diz que são do último passo e dá os dos checkpoints avaliados (0,520 e 0,608) |
| Profundidade 100 → 200 (§4.8) | "Não traz ao topo um alvo entre 100 e 200" | Traz 20 e expulsa 19 (recalculado de `t1e_resultado.json`) |
| "Cinco medições" (§4.8, Resumo) | Cinco, e o ganho "desaparece" | Quatro, não independentes; ganho não estabelecido, e não nulo |
| Reordenador (§4.8, §5.1) | O valor é condicional à fraqueza do recuperador | Neste protocolo; no assistente, exploratório, +0,09 [+0,05; +0,14] |
| Detector no braço de 48 M (§3.8, §4.9.2) | "Uma instabilidade" | Acionamento pelo critério antigo, provavelmente composição do lote |
| Pré-treinamento continuado (§4.9.3, Resumo, §7) | "O ganho se repete"; primária "nula" | Sugestivo, não estabelecido; primária não decidida (e negativa a 48 M) |
| GTE contra ModernBERT (§4.9.3, §5.5, §6) | "Ambos sem pré-treinamento no domínio" | O GTE já tem pré-treinamento contrastivo; "base" é variável composta |
| Encoder do zero (§5.5) | "Perde"; "fecha as duas opções" | O substituto de 48 M perde; não é veredito sobre o de 150 M; a "soma de três desvantagens" contava a causa e o efeito |
| Vazão (§5.6) | 20 B tokens em 80 h (37 dias locais) | A extrapolação não se confirmou: 270 a 360 h com a vazão medida |
| Orçamento de máscara (§4.9.2) | 75 tokens = 17% de 307 | Um quarto |
| PB-Formula (§4.10, Resumo, contribuição 7, §7) | "Refutação da premissa"; 15 pontos em recall@1 | Primária registrada é recall@10 (+0,025); gabarito só de superfície; BM25 não é casador simbólico; recall@50 incluído; especificação revista |
| Classificador (§4.2) | FP de 2,4–3,7% "em cada domínio" | 4,6% agregado no modelo final; a faixa é dos modelos de exclusão |
| OpenWebMath (§4.2) | "Admitiria 35% de conteúdo não pertinente" | 35,4% dos resumos de matemática a 0,5; 13,6% no limiar usado; não medido em web |
| Operador órfão (§4.3, contribuição 2) | "Invertido"; "discriminante válido" | Cresce com o comprimento; o indicador de ambiente é específico de formato |
| Auditoria de LaTeX (§4.3) | "16,6% de equações ausentes, IC de 95%" | Déficit líquido de contagem; amostra de conveniência; versão corrente contra instantâneo de 2023 |
| RedPajama (§3.3) | Filtrado pelo classificador a 0,9 | Selecionado por pertinência exata ao índice |
| Corpus (§4.1, Tabelas 1 e 2) | "Revisados por pares"; tokens; "necessária" | Campo `journal-ref`; tokens a 4 caracteres; "adquirível"; 65,6% dos documentos do peS2o são só resumo; sem deduplicação entre fontes |
| Licença (§4.1) | Depende da "época" | É um degrau entre 9 e 10 de novembro de 2020 |
| Referências | 31 | 35: ficha do `all-MiniLM-L6-v2`, S2ORC, SciNCL, SciRepEval/SPECTER2 — **aguardam a sua conferência na fonte** |

## 4. Decisões que são suas

**Sobre o conteúdo.**

1. **Título.** Consultas e candidatos são títulos e resumos do arXiv. "Literatura de Física" promete mais; a §6 agora diz isso, mas o título continua. Sugestão: "…de artigos de Física do arXiv…".
2. **As razões de manchete** ("onze vezes", "sete vezes"). Estão aritmeticamente certas, mas comparam experimentos que diferem em volume, regime e tokenizador. Mantive com ressalva; cortar é opção.
3. **Galactica (§1).** Corrigi para "a demonstração pública foi retirada" — os pesos continuaram disponíveis —, mas a referência [2] é o próprio artigo, que não pode sustentar a retirada. Falta uma fonte.
4. **INDUS (§2.1).** O resumo do artigo declara cobrir "physics" e "biology"; o texto diz que não cobre subáreas de Física. Não alterei: depende da seção de dados do artigo.
5. **Taxa de 30% de máscara (§3.4).** Atribuída a Warner et al. [26]; a evidência empírica é de Wettig et al. (EACL 2023), que não está na lista.
6. **Ordem das referências.** [30] e [31] são citadas antes de [22]. Estilos numéricos pedem renumeração.
7. **Tabelas citadas só pela própria legenda:** 7, 11, 12, 15, 16, 17, 18, 19 e 23.

**Sobre o destino — opinião minha, não de um revisor.**

- **O artigo são vários.** Oito contribuições, 24 tabelas, cerca de 21 mil palavras. Um revisor vai perguntar qual é a tese. Leio três artigos: (a) o sistema de recuperação e a curva volume × base, que é o resultado mais sólido; (b) a auditoria metodológica das §5.2 a 5.4, que é o mais original e já tem um rascunho próprio (`rascunho-armadilhas-recuperacao-por-citacao.md`); (c) os negativos de tokenização e pré-treinamento, que depois desta revisão são mais fracos do que pareciam, porque a ablação A×E não isola o que dizia isolar.
- **Idioma.** Qualquer venue de PLN ou RI exige inglês. Em português, o caminho é relatório técnico no arXiv (cs.IR), que não exige revisão por pares e fixa a prioridade.
- **O que um revisor vai exigir e não há:** um baseline treinado por citação (SPECTER2 ou SciNCL), um conjunto externo (SciDocs ou SciRepEval), mais de uma semente e um conjunto de teste separado. Os dois últimos custam pouco (§5, itens 1 e 7).

## 5. Medições que faltam — só a máquina do corpus faz

Em ordem de retorno por custo. As quatro primeiras não exigem treinar nada.

1. **Conjunto de teste sem retreino.** Os 20.078 pares deduplicados deixam folga: remedir as Tabelas 8, 9 e 10 nos itens 1.000 a 1.999 do sorteio (que não participaram da seleção), ou nas linhas 2.000 a 4.000. Reportar o checkpoint final ao lado do `-melhor`.
2. **Intervalos de nDCG@10 que o texto promete e não tem.** `bootstrap_pareado_itens` já existe. Pares: 6 M × GTE-large; sorteado × primeiras linhas a 400 mil; 6 M × 3 M; GTE-base 400 mil × MiniLM 400 mil e × MiniLM 6 M. Versionar as posições por item.
3. **Sobreposição treino/validação por documento.** Fração dos candidatos da avaliação que ocorrem como positivos no treino, e fração das âncoras de validação que ocorrem lá; Tabela 8 também no subconjunto nunca visto.
4. **GTE-large e PhysBERT a 512 tokens** no mesmo conjunto, para a Tabela 8.
5. **Controle sem exposição à tarefa.** Medir `nreimers/MiniLM-L6-H384-uncased` (o MiniLM antes do ajuste contrastivo) como a linha "sem ajuste" de fato.
6. **Baseline de mesma técnica:** SPECTER2 e SciNCL no protocolo da Tabela 8.
7. **Variância de semente.** Duas sementes a mais da configuração de 400 mil pares sorteados (3.125 passos cada).
8. **Contraste direto PhysBERT × GTE-base** no reordenador: regenerar as posições por consulta e rodar o pareado. O controle que falta é uma base de linguagem mascarada geral do mesmo tamanho.
9. **PB-Formula.** McNemar de recall@1 por item; um BM25 com tokenização que preserve operadores e chaves; a base sem ajuste; e o artefato da distribuição de comprimento (81 e 238 não têm arquivo).
10. **T1i: conferir uma coincidência.** `gte-small` e `bge-small` têm exatamente o mesmo McNemar em recall@1 (147 × 135, p = 0,5125). Os dois têm recall@1 de 0,4255, o que força o saldo, mas não as duas contagens. Pode ser coincidência; vale um minuto. Retirei as contagens do texto e deixei só o p.
11. **Números do corpus sem artefato.** Os 99,1% (1.581.098) só existem numa linha do README; os 41,5% só no `ESTADO.md`, e a fração de identificadores antigos no índice é 22,4%; "acima de 22,7 milhões" de arestas é um piso de coleta parcial. Regerar e gravar em `data/processed/avaliacao/`.
12. **Auditoria de LaTeX com sorteio entre fragmentos** e com a versão do e-print vigente na data do instantâneo.
13. **Ablação que isola as regras de LaTeX:** segmentação genérica com e sem as regras. É a única que responde à pergunta original da §4.9.1.
14. **Outros números só registrados em prosa:** teto por esquema de amostragem (0,7562, 0,9761), as seis hipóteses da §4.7, a acurácia de 0,830, a execução de 103 artigos, os 15.673 e 13.916 tokens por documento (o artigo diz 12,6%; DOC-05, README e ESTADO dizem 13,6%).

## 6. Defeitos fora do artigo

| Onde | Defeito | Estado |
|---|---|---|
| `ESTADO.md`, `src/phifm/eval/mlm_regiao.py` | `\frac` gravado como form feed + `rac` | Corrigido (`cad11c8`) |
| `docs/02-models/DOC-07` §2.2, `docs/01-data/DOC-05` §11.1 | Embedding a 16% (e ~24%); o certo é 22,1% e 31,2%, como a docstring de `config.py` já dizia | Corrigido |
| `src/phifm/core/licensing/registry.py` | Sem regra para `licenses/publicdomain/` (1.660 registros caem em NOASSERTION e contam como não redistribuíveis: 14,8% deveria ser 14,9%); versão ignorada (`by/3.0` rotulado CC-BY-4.0) | **Aberto** — exige reconstruir a contagem |
| `scripts/manifesto_corpus.py`, `ETAPAS['isphysics_clf']` | `max_por_classe = 400_000` reconstruído do padrão; o log da execução mostra 300.000 | **Aberto** — exige reconstruir a raiz |
| `data/processed/MANIFESTO-RAIZ.json` | Não versionado; o hash raiz não aparece no artigo, que oferece a cadeia como atestado | **Aberto** |
| `docs/adr/ADR-0003` §8 | "A soma de três ordens" conta os 0,0558 como terceira desvantagem, e eles são o efeito das outras duas | **Aberto** — o artigo foi corrigido, o ADR não |
| ADR-0001, DOC-00, DOC-02, DOC-19 | Citam a coleta de 1.595.065 registros; o artigo usa a de 1.595.422 | **Aberto** |
| `SETUP.md`, `requirements.lock` | "414 testes" (são 1.070); `reportlab` fora do lock e sem menção ao extra `relatorio`, o que quebra dois testes numa instalação limpa | **Aberto** |

## 7. Integridade do texto da v0.6, por script

- Nenhum byte de controle; nenhuma sequência LaTeX corrompida.
- Tabelas 1 a 24 definidas uma vez cada, em ordem.
- 35 referências; toda citação do corpo existe na lista, e toda referência é citada.
- Toda "Seção X.Y" citada existe.
- `tests/regression/test_artigo_pdf.py`: 11 testes passam.
