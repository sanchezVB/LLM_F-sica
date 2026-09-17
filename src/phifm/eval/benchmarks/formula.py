"""PB-Formula — recuperar documentos por EQUAÇÃO, sob variação notacional.

O DOC-11 §6.3 especifica: *"dada uma equação, recuperar os documentos do corpus que
a contêm — sob variação notacional. O gabarito vem da forma canônica do DOC-03 §3:
dois documentos que escrevem a mesma equação de formas diferentes têm a mesma
`canonical_latex`."*

É o gabarito gratuito da Trilha C aplicado a fórmulas, e mede o que recuperação
densa costuma perder: casamento simbólico exato. Também é a única medida do projeto
que testa diretamente o que o regex de pré-tokenização do DOC-05 §8 existe para
servir.

## ⚠️ Um filtro PRÓPRIO, porque `extrair_equacoes` é generoso de propósito

`_relevante`, no extrator, aceita `\\alpha` — e a docstring dele diz isso
explicitamente: *"`abc` não é equação; `a=b` e `\\alpha` são"*. Está certo para
alimentar um tokenizer, que precisa ver toda a notação.

Para um benchmark, símbolo solto não é item. Medido em 2.000 documentos: as formas
canônicas mais espalhadas são `\\alpha` (714 documentos), `\\sigma` (526), `\\beta`
(480). Recuperar "os documentos que contêm `\\alpha`" não mede nada.

`e_conteudo` exige **relação** (`=`, `\\approx`, `\\propto`, …) e comprimento
mínimo da forma canônica. Com ≥40 caracteres, a notação comum desaparece: zero
formas em mais de 20 documentos.

## ⚠️ A lição que quase matou este benchmark: amostra pequena subestima a repetição

A primeira calibração rodou em 2.000 documentos e achou **27 itens** — 0,040% das
formas. Eu estava a um passo de reportar que o §6.3 do DOC-11 não é viável.

Medido em quatro tamanhos:

    2.000 docs ·  68.079 formas ·     27 em ≥2 docs (0,040%)
    5.000 docs · 170.438 formas ·    234 em ≥2 docs (0,137%)
   10.000 docs · 344.030 formas ·    514 em ≥2 docs (0,149%)
   20.000 docs · 678.929 formas ·  1.654 em ≥2 docs (0,244%)

**A taxa cresce com o corpus, e o número absoluto cresce mais que linearmente**: 10×
mais documentos deram 61× mais itens. A razão é elementar e fácil de esquecer — uma
colisão exige as DUAS pontas na amostra, então amostrar 1/400 do corpus encontra
~1/160.000 dos pares. Calibrar corte de benchmark em amostra pequena não subestima
um pouco: subestima quadraticamente.

## ⚠️ O que NÃO é defeito da canonização

Na calibração apareceram como formas distintas:

    F_{\\mu\\nu}=\\partial_\\mu A_\\nu-\\partial_\\nu A_\\mu
    F_{\\mu\\nu}=\\partial_\\muA_\\nu-\\partial_\\nuA_\\mu

A segunda é **LaTeX inválido**: o TeX leria `\\muA` como uma macro inexistente. É
texto corrompido — o RedPajama perde 16,6% das equações (S3b) —, e a canonização
está certa em não colapsá-las, porque não pode saber se `\\muA` é uma macro real.

Conferidos sete pares notacionais: ela colapsa `E = mc^2`/`E=mc^{2}`,
`\\frac`/`\\dfrac`, `a \\cdot b`/`a b`, `\\alpha_s`/`\\alpha_{s}`, `\\geq`/`\\ge` e
`\\left(x\\right)`/`(x)`. Seis de sete, e o sétimo não é variação legítima.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

# Relações que fazem de uma expressão uma AFIRMAÇÃO, e não um símbolo.
# `\sim` e `\propto` entram: "n \sim 100" é conteúdo, ainda que fraco — o corte de
# comprimento é que separa esse caso.
RELACAO = re.compile(
    r"=|\\approx|\\simeq|\\equiv|\\propto|\\leq|\\geq|\\sim|\\ll|\\gg|<|>")

# Caracteres mínimos da forma CANÔNICA. Medido: a 40, nenhuma forma de conteúdo
# aparece em mais de 20 documentos, então a notação comum já sai por aqui.
MIN_CARACTERES = 40

# ⚠️ Teto de documentos por forma. A 40 caracteres ele não morde em 20 mil
# documentos, e existe para o corpus inteiro: uma identidade de livro-texto que
# apareça em mil papers é notação compartilhada, não conteúdo que discrimina.
MAX_DOCUMENTOS = 20

# ⚠️ Teto de itens por documento de CONSULTA. Equações do mesmo paper não são
# observações independentes — é a Falha 2 do artigo (`n` efetivo em dados
# agrupados): com 35 documentos, tratar 4.117 linhas como independentes deu um
# intervalo 11x estreito demais. Um paper com 400 equações dominaria o benchmark.
MAX_ITENS_POR_DOCUMENTO = 3


def e_conteudo(canonica: str, min_caracteres: int = MIN_CARACTERES) -> bool:
    """A forma canônica é uma equação de CONTEÚDO, não notação inline?

    Ver o § sobre o filtro próprio: `extrair_equacoes` aceita `\\alpha` de
    propósito, e para um benchmark isso não é item.
    """
    c = canonica.strip()
    return len(c) >= min_caracteres and bool(RELACAO.search(c))


@dataclass(frozen=True)
class Ocorrencia:
    """Uma equação, onde ela apareceu, e a forma canônica dela."""

    documento: str
    forma: str          # hash canônico
    latex: str          # o bruto, como o documento escreveu


@dataclass(frozen=True)
class Item:
    """Uma consulta do benchmark e o gabarito dela.

    `consulta` é o LaTeX **bruto** de um documento, não a forma canônica — é assim
    que a equação aparece para quem busca. Os `alvos` são os OUTROS documentos que
    escrevem a mesma equação, possivelmente de outra forma. É essa diferença entre
    a grafia da consulta e a do alvo que faz o benchmark medir variação notacional
    em vez de casamento de string.
    """

    consulta: str
    documento_da_consulta: str
    alvos: frozenset[str]
    forma: str

    def __post_init__(self) -> None:
        if self.documento_da_consulta in self.alvos:
            raise ValueError(
                f"o documento da consulta ({self.documento_da_consulta}) está "
                "entre os alvos. O item seria trivial: a equação aparece nele.")
        if not self.alvos:
            raise ValueError("item sem alvo não é recuperável")


@dataclass
class Estatisticas:
    """O que a montagem descartou, e por quê. Sem isto, um corte agressivo demais
    reduz o benchmark em silêncio e ninguém sabe se o número é pequeno por causa
    do corpus ou do filtro."""

    ocorrencias: int = 0
    formas: int = 0
    descartadas_sem_conteudo: int = 0
    descartadas_um_documento_so: int = 0
    descartadas_comuns_demais: int = 0
    descartadas_teto_por_documento: int = 0
    itens: int = 0
    documentos_de_consulta: int = 0

    def como_dict(self) -> dict:
        return {
            **self.__dict__,
            "nota": (
                "⚠️ `descartadas_um_documento_so` é o número GRANDE e é esperado: "
                "equações de conteúdo são quase únicas. Medido, a repetição cresce "
                "com o corpus — 2.000 documentos dão 0,040% e 20.000 dão 0,244%, "
                "porque uma colisão exige as duas pontas na amostra. Calibrar este "
                "filtro em amostra pequena subestima quadraticamente."),
        }


def montar_itens(
    ocorrencias: list[Ocorrencia],
    *,
    max_documentos: int = MAX_DOCUMENTOS,
    max_por_documento: int = MAX_ITENS_POR_DOCUMENTO,
) -> tuple[list[Item], Estatisticas]:
    """Agrupa ocorrências por forma canônica e monta os itens com as guardas.

    ⚠️ A consulta sai do documento cuja grafia é **mais curta** entre as
    ocorrências da forma, e não de um sorteio. Duas razões: é determinístico sem
    precisar de semente, e a grafia mais curta é a menos anotada — a que menos
    entrega o alvo por casamento de string. O sorteio poderia escolher justamente
    a grafia idêntica à do alvo e tornar o item trivial sem ninguém ver.
    """
    est = Estatisticas(ocorrencias=len(ocorrencias))
    por_forma: dict[str, list[Ocorrencia]] = defaultdict(list)
    for o in ocorrencias:
        por_forma[o.forma].append(o)
    est.formas = len(por_forma)

    itens: list[Item] = []
    usados_por_documento: dict[str, int] = defaultdict(int)
    # Ordem determinística: forma, para a saída não depender de iteração de dict.
    for forma in sorted(por_forma):
        grupo = por_forma[forma]
        documentos = {o.documento for o in grupo}
        if len(documentos) < 2:
            est.descartadas_um_documento_so += 1
            continue
        if len(documentos) > max_documentos:
            est.descartadas_comuns_demais += 1
            continue
        # A grafia mais curta, com desempate pelo texto E PELO DOCUMENTO.
        #
        # ⚠️ O desempate pelo documento não é zelo: sem ele, duas ocorrências com
        # o LaTeX **byte a byte igual** empatam na chave e o `min` devolve a
        # primeira da lista — que depende da ordem em que as ocorrências chegaram.
        # O caso é comum, não raro: duas grafias idênticas da mesma equação em
        # documentos diferentes é exatamente o que um corpus grande produz. Pego
        # pelo teste de determinismo, não pela leitura.
        consulta = min(grupo, key=lambda o: (len(o.latex), o.latex, o.documento))
        if usados_por_documento[consulta.documento] >= max_por_documento:
            est.descartadas_teto_por_documento += 1
            continue
        alvos = documentos - {consulta.documento}
        if not alvos:
            est.descartadas_um_documento_so += 1
            continue
        itens.append(Item(consulta=consulta.latex,
                          documento_da_consulta=consulta.documento,
                          alvos=frozenset(alvos), forma=forma))
        usados_por_documento[consulta.documento] += 1

    est.itens = len(itens)
    est.documentos_de_consulta = len({i.documento_da_consulta for i in itens})
    return itens, est


ESTRATOS_DE_GRAFIA = ("identica", "superficial", "mista", "notacional")

# O que muda a STRING sem mudar a notação: espaço, rótulo, numeração, alinhamento,
# espaçamento fino e a pontuação que fecha a equação na frase.
_SUPERFICIAL = re.compile(
    r"\\(?:label|tag\*?)\{[^{}]*\}|\\nonumber\b|\\notag\b|\\[,;:!]|&|\s+")
_PONTUACAO_FINAL = re.compile(r"[.,;]+$")


def _sem_superficie(latex: str) -> str:
    return _PONTUACAO_FINAL.sub("", _SUPERFICIAL.sub("", latex))


def estrato_de_grafia(consulta: str, grafias_dos_alvos: list[str]) -> str:
    """Quanto da variação notacional um item EXIGE, pela grafia dos alvos.

    - `identica`: todo alvo escreve a equação byte a byte como a consulta;
    - `superficial`: todo alvo coincide com ela a menos de espaço, `\\label{…}`,
      `\\tag{…}`, `\\nonumber`/`\\notag`, `&` de alinhamento, espaçamento fino (`\\,` `\\;` `\\:`
      `\\!`) e pontuação final;
    - `mista`: algum alvo coincide nesse sentido, e algum não;
    - `notacional`: nenhum alvo coincide.

    ⚠️ Existe porque o nome do benchmark promete "sob variação notacional", e o corpus
    inteiro (2026-09-17, 374.739 itens) mediu outra coisa: **53,1%** `identica`, **37,0%**
    `superficial`, 2,1% `mista` e só **7,8%** `notacional`. Um casador de string sairia
    bem em nove de cada dez itens sem saber nada de notação, e a média sobre o benchmark
    inteiro esconderia isso. O recall tem de sair POR ESTRATO.

    ⚠️ E espaço não bastava: numa amostra do estrato que sobrava, a diferença de vários
    itens era só o texto do `\\label`, um `&=` no lugar de `=` ou o ponto final. Isso é
    marcação, não notação. `\\left(` contra `(` fica como notacional: é uma das
    variações que a canonização existe para colapsar.

    A grafia de cada documento é a mais curta que ele usa para a forma
    (`ocorrencias_do_documento`), então um alvo que também escreve a grafia da consulta,
    mas tem outra mais curta, conta como diferente. É um piso da coincidência, não o
    valor exato.
    """
    if not grafias_dos_alvos:
        raise ValueError("item sem alvo não tem estrato")
    if all(g == consulta for g in grafias_dos_alvos):
        return "identica"
    base = _sem_superficie(consulta)
    coincide = [_sem_superficie(g) == base for g in grafias_dos_alvos]
    if all(coincide):
        return "superficial"
    return "mista" if any(coincide) else "notacional"


def teto_do_pool(itens: list[Item], documentos_do_pool: set[str]) -> float:
    """Recall@k de um modelo PERFEITO neste pool. Tem de ser 1,0.

    ⚠️ Esta função existe por causa do defeito que o T1b2 pagou: o pool do G1 tinha
    62% de alvos duplicados, o teto real era 0,7562 de nDCG@10 em vez de 1,0, e a
    tabela publicada comparava modelos contra um máximo inalcançável. Um benchmark
    que não confere o próprio teto não sabe o que está medindo.

    Aqui o teto cai abaixo de 1,0 quando algum item tem TODOS os alvos fora do
    pool — aí nenhum modelo pode acertá-lo, e a média de recall fica limitada.
    """
    if not itens:
        return 0.0
    alcancaveis = sum(1 for i in itens if i.alvos & documentos_do_pool)
    return alcancaveis / len(itens)


def conferir_pool(itens: list[Item], documentos_do_pool: set[str]) -> None:
    """Levanta se o teto não é 1,0. A única porta de entrada honesta."""
    teto = teto_do_pool(itens, documentos_do_pool)
    if teto < 1.0:
        fora = [i.forma for i in itens if not (i.alvos & documentos_do_pool)]
        raise ValueError(
            f"teto do pool é {teto:.4f}, não 1,0: {len(fora)} de {len(itens)} "
            "itens têm todos os alvos fora do pool, e nenhum modelo pode "
            "recuperá-los. Um benchmark com teto < 1 compara modelos contra um "
            f"máximo inalcançável. Primeiras formas afetadas: {fora[:3]}")


@dataclass
class Resultado:
    """O que uma execução mede. `recall_em` é por k, como o G1 reporta."""

    modelo: str
    itens: int
    recall_em: dict[int, float] = field(default_factory=dict)
    posicoes: list[int | None] = field(default_factory=list)

    def como_dict(self) -> dict:
        return {
            "modelo": self.modelo, "itens": self.itens,
            "recall_em": {str(k): round(v, 4)
                          for k, v in sorted(self.recall_em.items())},
            "nota": (
                "recall@k = fração dos itens com ao menos UM alvo no top-k. O "
                "gabarito costuma ter 1 a 3 alvos, então recall@k satura rápido — "
                "compare k pequeno."),
        }


def ocorrencias_do_documento(texto: str, min_caracteres: int = MIN_CARACTERES,
                             ) -> tuple[dict[str, str], dict[str, int]]:
    """As equações de CONTEÚDO de um documento: `{forma: grafia mais curta}`, e contagens.

    Canoniza cada equação UMA vez (`canonical.forma` devolve a forma canônica e o hash
    juntos, com a mesma definição de `hash_canonico`). Dentro do documento fica só a
    grafia mais curta de cada forma, com desempate pelo texto: `montar_itens` escolhe a
    consulta pela grafia mais curta ENTRE documentos, e as repetições dentro de um
    documento só inflariam a memória da montagem no corpus inteiro.

    Uma exceção da canonização é CONTADA e pulada, não silenciada.

    ⚠️ **Mas o contador não mede perda, e esta docstring dizia que media.** A versão
    anterior ligava as falhas aos 16,6% de equações que o RedPajama perde (S3b), como se
    o número dissesse quanto do corpus o benchmark não enxerga. Não diz: `canonicalizar`
    é TOTAL — não tem nenhum `raise`, e devolve uma forma para qualquer cadeia. No corpus
    inteiro (202.365.265 equações, 2026-09-17) o contador deu 0, e deu 0 por
    construção. Ele só acusaria uma exceção inesperada de código. As equações que o
    RedPajama perdeu não chegam a `extrair_equacoes`, e nenhum contador daqui as vê.
    """
    from phifm.core.latex import canonical
    from phifm.core.latex.extrair import extrair_equacoes

    formas: dict[str, str] = {}
    cont = {"equacoes": 0, "falhas_canonizacao": 0, "de_conteudo": 0}
    for e in extrair_equacoes(texto):
        cont["equacoes"] += 1
        try:
            f = canonical.forma(e)
        except Exception:
            cont["falhas_canonizacao"] += 1
            continue
        if not e_conteudo(f.canonica, min_caracteres):
            continue
        cont["de_conteudo"] += 1
        atual = formas.get(f.hash)
        if atual is None or (len(e), e) < (len(atual), atual):
            formas[f.hash] = e
    return formas, cont
