# ADR-0004 — O ΦGen deve ser treinado para o assistente de literatura?

**Status:** **Proposto (2026-10-09)** — aguarda o aceite do dono do projeto.
**Contexto:** [DOC-07 §5](../02-models/DOC-07-familia-de-modelos.md) (ΦGen: CPT sobre Qwen3, 1,5 B e 8 B), [DOC-13 §6 e §9](../04-systems/DOC-13-recuperacao-embeddings-rag.md) (geração ancorada e a medida do assistente), [DOC-11 §5](../03-evaluation/DOC-11-physbench.md) (juiz automático e concordância humana).
**Não revisa** a linha "Physics Generator — modelo próprio" do DOC-07: restringe o motivo pelo qual ele seria treinado.

---

## 1. A pergunta

O assistente (`scripts/perguntar.py`, a aba "Perguntar" da página) responde perguntas de
Física citando os artigos que a busca encontra. Ele roda com um modelo ABERTO — o
Qwen3-8B, local — e não com o ΦGen, que não existe ainda. O ΦGen-1,5B custaria
US$ 120–240 de pré-treino continuado em Física.

**Vale treinar o ΦGen para melhorar o assistente?** Ele só mexeria em uma das duas
metades: no modelo que escreve, não na busca que escolhe o que ele lê.

---

## 2. A evidência

A medida foi desenhada para separar as duas metades, com a regra escrita e aceita antes
de qualquer item existir (DOC-13 §9.1). 500 perguntas no estrato que decide, cada uma
saída de um fato do resumo de um artigo P; três braços com o mesmo modelo e o mesmo prompt.

| braço | o que o modelo recebe | acerto | IC 95% |
|---|---|---|---|
| **B** · fonte garantida | as 6 fontes da busca, com P garantido entre elas | **0,944** | [0,924; 0,964] |
| B, depois da revisão dos erros (R) | — | **0,976** | [0,962; 0,988] |
| **A** · o sistema (antes do ΦRank) | o que a busca trouxe | **0,700** | [0,660; 0,740] |
| **A′** · o sistema de hoje, com o ΦRank | a busca reordenada | **0,806** | — |
| **C** · sem fontes | a pergunta sozinha | **0,168** | [0,136; 0,202] |

| eixo da regra | medida | limiar | desfecho |
|---|---|---|---|
| gerador | acerto de B depois de R: 0,976 [0,962; 0,988] | IC inteiro ≥ 0,90 | **BASTA** |
| busca | B − A: 0,244 [0,206; 0,282] | IC inteiro > 0,05 | **LIMITA** |

E a consequência já colhida: reordenar os 50 primeiros da busca com o ΦRank — que já
existia, custo de treino zero — levou o acerto de 0,700 a **0,806** (+0,106 [0,072; 0,142],
DOC-13 §9.2), e está adotado desde 2026-10-07.

---

## 3. O que a evidência diz, e o que ela não diz

**Diz:** com o artigo certo na mão, o modelo aberto acerta 19 de cada 20 perguntas. O teto
que um gerador melhor poderia buscar no assistente é de ~2 a 6 pontos (de 0,944–0,976 até
1). O que a busca deixa na mesa é de **14 pontos ainda hoje** (0,806 contra 0,944), depois
de já ter devolvido 10.

**Não diz:**

1. **Nada sobre o ΦGen como resolvedor de problemas.** As perguntas pedem um fato que
   está escrito no resumo — é leitura, não derivação. Os portões G2.1 e G2.3 (benchmarks
   fechados, sem artigo no prompt) são outra medida, e essa só se faz treinando.
2. **Nada sobre perguntas que exigem juntar vários artigos**, ou o texto completo.
3. **Não tem revisão humana das respostas.** O dono fez a I1 (40 de 40 perguntas válidas)
   e decidiu não julgar mais. O juiz automático foi calibrado contra o Claude, às cegas
   (κ 0,76 depois de um conserto; o primeiro juiz foi reprovado com κ 0,52), e a revisão
   dos erros também é do Claude, que escreveu as perguntas. O desfecho "BASTA" não depende
   dela — passa só com o juiz (0,944) —, mas o DOC-11 §5 pede gente, e não houve.
4. **O portão de citações não mede se a fonte sustenta a frase** (G2.4): só que a citação
   aponta para uma fonte fornecida.

---

## 4. As opções

| | o quê | custo | o que compra |
|---|---|---|---|
| **A** | **Não treinar o ΦGen por causa do assistente.** O investimento do assistente vai para a busca | US$ 0 agora | os 14 pontos que a busca ainda deixa |
| **B** | Treinar o ΦGen-1,5B assim mesmo | US$ 120–240 | no assistente, no máximo ~2–6 pontos; o resto do valor estaria no G2, que esta medida não toca |
| **C** | Antes de qualquer treino, medir um modelo geral MAIOR no braço B (o teste barato previsto na regra) | horas de GPU | só faria sentido se o gerador limitasse — e ele não limita |

---

## 5. Recomendação (não decidida)

**Opção A.** O assistente não é argumento para treinar o ΦGen. Se o ΦGen for treinado,
que seja pela pergunta do G2 — raciocínio em Física sem a fonte no prompt —, com a medida
dele escrita antes, como esta foi.

Para o assistente, a ordem do investimento fica:

1. o que ainda couber na ordem dos resultados (mais fundo que 50, mais fontes) — barato,
   em andamento nas perguntas de desenvolvimento;
2. **trechos do texto completo** (DOC-13 §3): 14% dos artigos certos não chegam nem aos 50
   primeiros, e isso é representação, não ordem;
3. o *entailment* das citações (G2.4), que hoje é o limite declarado do portão.

---

## 6. O que reabriria esta decisão

- Uma revisão humana que desminta o juiz: se o acerto de B, julgado por gente, ficar
  abaixo de 0,90.
- Um conjunto de perguntas que exija raciocínio sobre o texto (não um fato do resumo) em
  que o braço B caia abaixo de 0,90 — aí o gerador passa a limitar.
- A busca fechar a lacuna: com A perto de B, o que sobra é do gerador, e a conta do
  §3 muda de lado.

## 7. O que fica publicado, qualquer que seja a decisão

A regra e os desvios (DOC-13 §9.1 e §9.2), os agregados em
`data/processed/avaliacao/assistente_*.json` com os hashes dos itens, e o código da medida
(`phifm.eval.assistente`, `phifm.eval.assistente_bracos`, `scripts/medir_assistente.py`).
As perguntas, os gabaritos e as respostas ficam fora do git (DOC-11 §8.1).
