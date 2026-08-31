# ADR-0002 — De onde vem o LaTeX íntegro que o ΦEnc precisa

**Status:** Aceito (2026-08-31)
**Contexto:** [DOC-07 §2.3](../02-models/DOC-07-phienc.md) (mascaramento consciente de equações), [DOC-05 §11](../01-data/DOC-05-tokenizer.md), [ADR-0001 §2](ADR-0001-decisoes-stage-gate-0.md) (a separação D1/D2/D3)
**Substitui:** a recomendação de 2026-08-27 de comprar acesso em massa ao arXiv como pré-requisito

---

## 1. A decisão

**Usar a fatia de Física do RedPajama-arXiv, que já está no disco, como corpus de pré-treino do ΦEnc.** Não comprar acesso ao S3 do arXiv agora.

| | |
|---|---|
| Corpus escolhido | `data/processed/redpajama_fisica/` — 835.379 documentos, 42,15 G caracteres, 10,54 B tokens, 12 GB |
| Origem | `togethercomputer/RedPajama-Data-1T`, fatia arXiv, construída do **fonte LaTeX** |
| Filtro | casamento exato com a espinha de Física (`scripts/coletar_redpajama.py`) |
| Custo | **US$ 0** — baixado em 2026-08-14 |
| Alternativa rejeitada | arXiv S3 `requester pays`, agora, como pré-requisito |

---

## 2. ⚠️ Por que este ADR existe: eu recomendei um gasto sem medir a alternativa

Em 2026-08-27 medi o peS2o de texto pleno e encontrei as equações removidas na extração. A conclusão que registrei foi que o acesso pago ao fonte LaTeX *"passa de conveniência a pré-requisito da hipótese central do ΦEnc"*, e em 2026-08-31 abri a lista de recomendações com **"decida o arXiv, ou abandone explicitamente a hipótese do ΦEnc"**, cotado em US$ 100–180.

**A verificação que faltou levou um comando.** O corpus tem quatro fatias, e uma delas — o RedPajama-arXiv — é construída a partir do fonte LaTeX. Ela estava no disco desde 2026-08-14, com 835 mil documentos de Física.

Este projeto tem uma regra explícita de medir antes de gastar cota, e ela nasceu justamente de erros deste tipo. Eu a apliquei à GPU e não ao dinheiro.

---

## 3. A medição que decide

`scripts/medir_equacoes_mutiladas.py`, 3.000 documentos por fatia, sorteados (não `head()`), semente 17:

| fatia | ch/doc | LaTeX % | `$…$` % | **ambiente de equação %** | seq/doc |
|---|---|---|---|---|---|
| arXiv resumos (referência) | 1.123 | 21,8 | 26,9 | 0,0 | 0,9 |
| **RedPajama-arXiv** | 49.212 | **100,0** | **99,6** | **84,9** | **1.158,3** |
| OpenWebMath | 16.009 | 78,5 | 85,2 | 5,1 | 57,7 |
| peS2o texto pleno | 28.605 | 16,3 | 18,2 | **0,0** | 1,0 |

Um trecho cru do RedPajama, para não depender só de agregados:

```
The Shannon entropy associated to the probability of transition for each $xyz$
element gives its statistical complexity (in bits):

\begin{equation}
C_{\mu,xyz} = - P_{xyz} \log_2 P_{xyz}
\label{eq:complex}
\end{equation}
```

É o fonte como o autor escreveu: ambiente, corpo da equação e `\label`. É exatamente o que o DOC-07 §2.3 precisa mascarar.

### 3.1 E uma correção de método

A medição de 2026-08-27 usou como assinatura o **operador órfão** — ` = ` sem operando de um lado. Ela é válida em texto do qual a matemática foi arrancada, e **satura em texto onde o LaTeX está íntegro**: no RedPajama ela acusa 81,3%, contra 3,0% da referência, e os trechos crus mostram matemática perfeita. A causa é banal: em `$Z_{\rm max}$ = 15 kpc` o caractere antes de ` = ` é `$`, que não é `[\w\)\]]`, e o regex dispara.

O discriminante correto é a **presença de ambiente de equação**: 84,9% contra 0,0%. Nenhum ajuste de regex conserta um corpus sem equações, e nenhum regex de órfão distingue um corpus com elas.

---

## 4. Consequências

**O que destrava.** A hipótese central do ΦEnc é testável hoje, sem gastar. Volume medido: **42,15 G caracteres**, 50.451 por documento, **10,54 B tokens** a ~4 ch/token — acima dos 5 B por variante que o bake-off do DOC-05 §11.2 pede, e acima do que um encoder de 150 M consome com folga.

**O que continua valendo do arXiv pago.** Cobertura e frescor. O RedPajama-1T é um instantâneo de 2023, não tem o que saiu depois, e é a fatia arXiv **de todo o arXiv** filtrada pela nossa espinha — o que o filtro não pegou não está lá. Para o ΦEnc-150M isso não morde; para um Tier 2, mordia.

**A cotação que eu dei estava errada, nos dois sentidos.** O conjunto de fonte é ~2,9 TB (março de 2023) e os dois conjuntos somam ~9,2 TB (abril de 2025); o bucket é `requester pays` e o arXiv não publica preço, remetendo à tabela da AWS. A US$ 0,09/GB de egresso, o fonte inteiro para fora da AWS passa de **US$ 400**, não US$ 100–180. Mas filtrar dentro da AWS (`us-east-1`, egresso zero para EC2) e baixar só o `.tex` custaria **dezenas de dólares** — os tars são por mês, não por área, então não há como pedir só Física na origem.

**Se um dia for comprar, a rota é essa:** filtrar na nuvem, egressar o `.tex`. Não baixar 2,9 TB para o HD.

---

## 5. A rota gratuita pelo serviço de exportação, e o que os termos dizem

Para *complementar* com o que é posterior a 2023, sem S3:

- Os **termos de uso da API** limitam a *"no more than one request every three seconds, and limit requests to a single connection at a time"*. A 3 s por paper, 50 mil papers levam ~42 h de relógio, em segundo plano, sem GPU.
- A página de dados em massa descreve o *"export service crawling"* como **"recommended for new content or subset of content"** — que é literalmente este caso.
- Os termos **proíbem** *"store and serve arXiv e-prints … from your servers"* sem licenciamento, e proíbem *"attempt to circumvent rate limits"*. Treinar localmente não é servir; **redistribuir o corpus seria**, e isso é o D3 do ADR-0001 §2, que já está resolvido como "não redistribuímos".
- Para taxa maior os termos dizem *"please contact our support team"*.

**Recomendação:** se essa rota for usada em volume, escrever ao suporte do arXiv antes, descrevendo o uso e a taxa. Custa um e-mail e remove a ambiguidade entre a letra do limite e o espírito do "bulk vai pelo S3". O rascunho está em `docs/01-data/rascunho-email-arxiv.md` — **não enviado**: escrever em nome do usuário para um terceiro é decisão dele.

---

## 6. O que ainda não está medido, e é honesto listar

1. **Contaminação.** O RedPajama-arXiv de 2023 pode conter os papers que os nossos benchmarks usam. O `phifm.eval.contamination` existe; esta fatia ainda não passou por ele.
2. **Sobreposição com o peS2o.** As duas fatias vêm em parte dos mesmos papers, e treinar sobre a união duplicaria conteúdo com deduplicação incompleta.
3. **Qualidade do fonte.** Fonte LaTeX traz macros do autor, `\input` de arquivos ausentes, comentários e pacotes. Nada disso é texto de Física, e o DOC-05 §10 pede uma normalização que ainda não existe.
4. **Se o `\label` e o `\cite` devem ser mascarados** junto com a equação, ou tratados como ruído. É uma decisão de desenho do DOC-07 §2.3 que a existência do corpus torna urgente.

Nenhum dos quatro custa dinheiro. Os quatro custam medição, e são o trabalho que este ADR libera.
