# Rascunho — e-mail ao suporte do arXiv

**Status: NÃO ENVIADO.** Escrever a um terceiro em nome do usuário é decisão dele, não
minha. Este arquivo é o rascunho; enviar, ajustar ou descartar é com você.

**Quando enviar:** só se a rota do serviço de exportação for usada em volume (ordem
de dezenas de milhares de papers). Ver [ADR-0002 §5](../adr/ADR-0002-fonte-latex-para-o-phienc.md).
Para o corpus que já está no disco não há nada a perguntar.

**Para:** o canal de suporte indicado em <https://info.arxiv.org/help/contact.html>

**Por que perguntar em vez de só respeitar o limite.** Os termos de uso da API
limitam a uma requisição a cada três segundos, e a 3 s por paper a coleta cabe no
limite pela letra. Mas a página de dados em massa diz que download em massa de
texto pleno vai pelo S3, e os termos dizem "se o seu caso precisa de taxa maior,
contate o suporte". Um caso que respeita a letra e tensiona o espírito é exatamente
o tipo de coisa que se resolve com um e-mail, e não com uma interpretação própria.

---

## Rascunho (inglês)

> **Subject:** Permission check — LaTeX source harvesting for a physics-domain
> language model (non-redistributive, single connection, 1 req/3s)
>
> Hello,
>
> I am an independent researcher building a physics-domain encoder model. I would
> like to confirm that my intended use of the export service is acceptable to you
> before I start, rather than after.
>
> **What I would like to fetch.** The LaTeX e-print source for a subject-filtered
> subset of physics papers — on the order of 50,000 papers, selected from
> `astro-ph`, `cond-mat`, `gr-qc`, `hep-*`, `nucl-*`, `physics.*` and `quant-ph`.
>
> **Why source rather than PDF or the S3 bundles.** The model's central research
> question is whether an equation-aware pretraining objective helps in physics, so
> the LaTeX markup is the signal, not incidental formatting. The S3 bundles are
> organised by month rather than by subject, so obtaining a physics subset that way
> means transferring the full ~2.9 TB source set and discarding most of it. Fetching
> the subset directly is smaller for you and for me.
>
> **How I would fetch it.** One request every three seconds, a single connection at
> a time, no parallelism, no retries beyond the polite minimum, with a descriptive
> User-Agent and my e-mail address in it. At that rate the whole subset takes about
> 42 hours of wall-clock time, spread over several days.
>
> **What I will not do.** I will not redistribute the e-prints or any derivative
> corpus, and I will not store or serve them from any public server. The files stay
> on a single local disk and are used only as training input. If the model weights
> are ever published, the corpus will not be.
>
> **What I am asking.**
>
> 1. Is this use of the export service acceptable, or would you prefer I take the
>    S3 route even for a subject-limited subset?
> 2. If the export service is acceptable, is one request every three seconds the
>    rate you want, or would you prefer slower?
> 3. Is there an attribution or acknowledgement format you would like used in any
>    resulting publication?
>
> I am happy to adjust anything about the approach, including abandoning it, based
> on your answer. Thank you for maintaining arXiv, and for making bulk access
> possible at all.
>
> Best regards,
> [nome]
> [e-mail]
> [link do projeto, se houver]

---

## Notas sobre o rascunho

**O que ele faz de propósito:**

- **Pergunta antes**, e diz isso na primeira frase. Um pedido de permissão que chega
  depois da coleta é um aviso, não um pedido.
- **Dá o número.** "50.000 papers, 1 req/3 s, ~42 h" é verificável; "alguns papers"
  não é, e convida a uma resposta genérica.
- **Justifica o fonte** em vez de pedir e esperar. A razão real — o LaTeX é o sinal,
  não formatação — é também a razão pela qual o PDF não serve, e dizê-la evita a
  resposta "use o PDF".
- **Argumenta pelo lado deles**: o subconjunto por área é *menos* transferência que
  os 2,9 TB dos tars mensais. É verdade, e é o argumento mais forte que existe aqui.
- **Declara o D3 explicitamente** (não redistribuir), porque é a proibição literal
  dos termos e deixá-la implícita seria a omissão mais visível.
- **Oferece desistir.** Se a resposta for "não", a resposta útil é "obrigado", não
  uma negociação.

**O que ele não faz:**

- Não pede exceção ao limite de taxa. Pedir taxa maior transformaria uma pergunta de
  conformidade num pedido de favor, e o gargalo aqui não é tempo.
- Não menciona financiamento nem afiliação que não existam.
