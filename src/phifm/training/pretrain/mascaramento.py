"""Mascaramento MLM, com a variante consciente de equações do DOC-07 §2.3.

**É a única adição específica de Física ao objetivo de pré-treino**, e portanto a
razão científica de treinar o ΦEnc do zero em vez de ajustar um modelo existente. Se
ela não ajudar, o DOC-07 §2.3 manda descartá-la e publicar o negativo.

Este módulo é **numpy puro, sem torch**, de propósito: ele é o coração da hipótese e
os testes dele têm de rodar na suíte rápida. `rerank.py` importa torch e por isso
`amostragem.py` teve de ser extraído dele depois; aqui a separação nasce feita.

## As três decisões de desenho, e por que cada uma

### 1. Orçamento de máscara IGUAL entre as duas variantes

O erro fácil é mascarar uma equação de 200 tokens onde o aleatório mascararia 30. As
duas variantes passariam a diferir em **duas** coisas — "consciente de equações" e
"mascara 6× mais" — e nenhum resultado seria atribuível. Este repositório já perdeu
um experimento por trocar base e lote na mesma corrida.

Então: `n_alvo = round(taxa × n_mascaráveis)` vale para as duas. No exemplo tratado,
escolhe-se uma equação que **caiba** no orçamento, mascara-se ela inteira, e o resto
do orçamento é preenchido aleatoriamente. A conta de tokens mascarados fica igual; o
que muda é a *estrutura* do que se mascara.

### 2. A equação vai inteira para `[MASK]`, sem o 80/10/10

O ponto do tratamento é reconstruir a equação **a partir da prosa**. Deixar 10% dos
tokens dela intactos e 10% substituídos por tokens aleatórios entregaria pedaços da
resposta e diluiria exatamente o efeito que se quer medir. O 80/10/10 continua
valendo para o preenchimento aleatório do resto do orçamento.

⚠️ Isto é uma escolha, não um achado: ela torna o tratamento mais forte e também mais
distante do MLM padrão. Se o tratamento ganhar, "ganhou porque mascara 100% com
`[MASK]`" é uma explicação alternativa viva, e a ablação que a mata é rodar o
tratamento com 80/10/10 também. Está registrado aqui para não ser esquecido depois.

### 3. O tratamento usa equações em DISPLAY, e isso foi medido

A primeira versão tratava qualquer coisa entre `$…$` como equação. Medido em 120
documentos do RedPajama-arXiv, com o tokenizer da variante A:

    tipo      por doc   tokens: p10  mediana  p90   p99   max
    display      40,6            39       79  214   678  19.587
    inline      260,9             4        7   19    39     103

A mediana de **7 tokens** do inline é uma **variável** (`$\\rho$`), não uma equação.
Mascarar 7 tokens de uma variável não testa "reconstruir a expressão formal a partir
da prosa": é MLM comum num trecho meio LaTeX, e a ablação mediria o nada — e
reportaria empate.

Display tem material sobrando: **91,7% dos documentos** têm ao menos uma, **40,6 por
documento**, mediana de **79 tokens**, e **99,8% cabem** no orçamento de 2.457 tokens
(30% de 8.192). Então o tratamento seleciona só display, com piso de
`MIN_TOKENS_TRATAMENTO`, e os ~8% de documentos sem display recaem no aleatório —
contados, não escondidos.

O inline continua sendo **marcado**, porque a marcação serve para estatística e para
trabalho futuro; ele só não é elegível para o tratamento.

### 4. Os contadores existem porque o tratamento pode não acontecer

Um exemplo sorteado para tratamento pode não ter equação nenhuma, ou ter só equações
maiores que o orçamento. Nos dois casos ele **recai** no mascaramento aleatório — e
se isso acontecer em 90% dos exemplos, a "ablação de mascaramento de equações" estará
comparando aleatório com aleatório e reportando um empate que não significa nada.

`Contadores.fracao_tratada()` é a métrica que precisa ser registrada junto da perda.
Sem ela, o resultado nulo é indistinguível de tratamento ausente.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# ── Onde está a matemática ──────────────────────────────────────────────────
#
# ⚠️ A ORDEM importa: `$$…$$` antes de `$…$`, senão o segundo casa o primeiro `$$`
# como um par vazio e a expressão inteira vira duas equações degeneradas.
#
# Os ambientes vêm primeiro porque um `\begin{align}` pode conter `$` dentro (raro,
# mas acontece em texto de autor), e a alternância do `re` é gulosa pela ordem.
_AMBIENTES = (r"equation|align|eqnarray|gather|multline|displaymath|"
              r"flalign|alignat|split|array|cases|matrix|[bpvV]matrix|smallmatrix")

EQUACAO = re.compile(
    # \begin{align}…\end{align}, com o `*` opcional e o mesmo nome nas duas pontas
    r"\\begin\{(?P<amb>" + _AMBIENTES + r")\*?\}(?:.|\n)*?\\end\{\1\*?\}"
    # \[ … \]
    r"|\\\[(?:.|\n)*?\\\]"
    # $$ … $$  — ANTES do $…$
    r"|\$\$(?:.|\n)*?\$\$"
    # $ … $ inline. ⚠️ Sem quebra de linha DUPLA e com teto de tamanho: um `$` solto
    # no texto (moeda, ou um `\$` mal escapado) faria o casamento engolir parágrafos
    # inteiros e marcar prosa como equação — que é pior que não marcar nada, porque
    # o tratamento passaria a mascarar prosa acreditando mascarar matemática.
    r"|\$(?!\$)[^$\n]{1,400}?\$(?!\$)",
)

# O que abre uma equação em DISPLAY — as que expressam uma lei, não uma variável.
# ⚠️ SEM `^`, e isto e um bug que a sonda pegou.
#
# `Pattern.match(s, pos)` JA ancora em `pos` — mas o `^` continua se
# referindo ao inicio REAL da string, e por isso falha para todo `pos > 0`.
# Com `^` no padrao, `DISPLAY.match(texto, i)` nunca casava para nenhuma
# equacao que nao comecasse no caractere 0: medido, 0 de 120 documentos
# tratados, com `recaida_sem_equacao` em 120 — contra 91,7% de documentos
# com display medidos por uma sonda que fatiava a string antes de casar.
#
# Foi a discordancia entre as duas sondas que localizou o erro. Sem a
# `fracao_tratada` nos contadores, a ablacao teria rodado inteira
# comparando aleatorio com aleatorio e reportado empate.
DISPLAY = re.compile(r"(?:\\begin\{|\\\[|\$\$)")

# Comprimento mínimo, em caracteres, para uma marcação contar como matemática. `$x$`
# tem 3 e é uma variável.
MIN_CHARS_EQUACAO = 6

# Piso em TOKENS para uma equação ser elegível ao tratamento. Medido: 99,4% das
# display passam de 20, e o p90 do inline é 19 — o piso separa os dois regimes sem
# depender do tipo, e vale como segunda barreira caso o `DISPLAY` erre.
MIN_TOKENS_TRATAMENTO = 20


def spans_de_equacao(texto: str, min_chars: int = MIN_CHARS_EQUACAO,
                     ) -> list[tuple[int, int, bool]]:
    """`(início, fim, é_display)` de cada trecho de matemática, sem sobreposição.

    >>> spans_de_equacao(r"antes $E = mc^2$ depois")
    [(6, 16, False)]
    >>> spans_de_equacao("um $x$ só")
    []
    >>> spans_de_equacao(r"\\begin{equation} F = ma \\end{equation}")[0][2]
    True
    """
    achados: list[tuple[int, int, bool]] = []
    for m in EQUACAO.finditer(texto):
        i, f = m.span()
        if f - i < min_chars:
            continue
        # `finditer` já não devolve sobreposições, mas um ambiente aninhado
        # (`split` dentro de `equation`) pode ter sido casado pelo externo; a
        # checagem custa nada e garante a invariante que o resto do módulo assume.
        if achados and i < achados[-1][1]:
            continue
        achados.append((i, f, bool(DISPLAY.match(texto, i))))
    return achados


def marcar_equacoes(offsets: list[tuple[int, int]], texto: str,
                    min_chars: int = MIN_CHARS_EQUACAO,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """`(id_equacao, e_display)` por token; `id_equacao` é −1 fora de matemática.

    Os `offsets` são os que o `tokenizers` devolve com `encode(...).offsets`. Fazer
    esta marcação **na tokenização**, e não na hora de mascarar, é o que permite ao
    mascarador ser puro numpy: depois do empacotamento o texto original já não está
    disponível, e recuperá-lo por documento seria carregar o corpus duas vezes.

    ⚠️ Um token conta como "de equação" se **sobrepõe** o intervalo. Exigir contenção
    total perderia o token de fronteira que carrega o `$` ou o `\\begin`, que é
    justamente a parte que o modelo usaria para adivinhar que ali havia matemática.

    ⚠️ **E é vetorizado por `searchsorted`, não por laço duplo.** A primeira versão
    varria todos os tokens para cada span, e medido em 64 documentos ela era **98%
    do tempo de preparação do corpus** — 55 mil tok/s contra os 3,1 M tok/s do
    tokenizer em Rust, porque a mediana é 214 equações e 14,5 mil tokens por
    documento (188 milhões de iterações Python por documento). A 33 mil tok/s a
    preparação de 2 B tokens levaria **16 horas**.
    """
    n = len(offsets)
    marca = np.full(n, -1, dtype=np.int32)
    display = np.zeros(n, dtype=bool)
    spans = spans_de_equacao(texto, min_chars)
    if not spans or n == 0:
        return marca, display

    ini_tok = np.fromiter((a for a, _ in offsets), dtype=np.int64, count=n)
    fim_tok = np.fromiter((b for _, b in offsets), dtype=np.int64, count=n)
    # ⚠️ Tokens sem extensão (`(0, 0)`, os especiais) saem do cálculo e voltam com
    # −1. Eles quebrariam a monotonicidade que o `searchsorted` exige, e um `(0, 0)`
    # no FIM da sequência faria a busca binária responder qualquer coisa.
    reais = np.flatnonzero(ini_tok != fim_tok)
    if reais.size == 0:
        return marca, display
    a, b = ini_tok[reais], fim_tok[reais]

    # A monotonicidade é propriedade do tokenizer: ele emite tokens da esquerda
    # para a direita, sem sobreposição. Se um dia não for, a busca binária daria
    # respostas erradas em SILÊNCIO — então a checagem levanta em vez de degradar.
    if not (np.all(np.diff(a) >= 0) and np.all(np.diff(b) >= 0)):
        raise ValueError(
            "os offsets dos tokens não estão ordenados, e a marcação por busca "
            "binária depende disso. Um tokenizer que emite tokens fora de ordem "
            "precisa da varredura linear — que é 56× mais lenta, e por isso não "
            "está aqui como recaída silenciosa.")

    for k, (ini, fim, e_disp) in enumerate(spans):
        # tokens que SOBREPÕEM [ini, fim): os com `b > ini` e `a < fim`.
        lo = int(np.searchsorted(b, ini, side="right"))
        hi = int(np.searchsorted(a, fim, side="left"))
        if hi > lo:
            alvo = reais[lo:hi]
            marca[alvo] = k
            display[alvo] = e_disp
    return marca, display


@dataclass(frozen=True)
class ConfigMascara:
    """DOC-08 §4 manda 30%; o 80/10/10 é a convenção do BERT, não medição nossa."""

    taxa: float = 0.30
    # Fração dos exemplos que recebem o tratamento. 0,0 é o BRAÇO DE CONTROLE da
    # ablação do DOC-07 §2.3 — MLM padrão, sem nada de Física.
    p_equacao: float = 0.0
    p_mask: float = 0.8
    p_aleatorio: float = 0.1
    semente: int = 17

    def __post_init__(self) -> None:
        if not 0.0 < self.taxa < 1.0:
            raise ValueError(f"taxa={self.taxa} fora de (0, 1)")
        if not 0.0 <= self.p_equacao <= 1.0:
            raise ValueError(f"p_equacao={self.p_equacao} fora de [0, 1]")
        if self.p_mask + self.p_aleatorio > 1.0:
            raise ValueError(
                f"p_mask + p_aleatorio = {self.p_mask + self.p_aleatorio} > 1; o "
                "resto é a fração que fica intacta e não pode ser negativa")


@dataclass
class Contadores:
    """O que separa "o tratamento não ajudou" de "o tratamento não aconteceu"."""

    exemplos: int = 0
    sorteados_para_tratamento: int = 0
    tratados: int = 0
    recaida_sem_equacao: int = 0
    recaida_equacao_curta: int = 0
    recaida_equacao_grande: int = 0
    tokens_mascaraveis: int = 0
    tokens_mascarados: int = 0
    tokens_de_equacao_mascarados: int = 0
    _reservado: dict = field(default_factory=dict, repr=False)

    def fracao_tratada(self) -> float:
        """Dos sorteados, quantos de fato tiveram uma equação mascarada.

        ⚠️ Registrar isto junto da perda não é opcional. Se ficar baixo, a ablação
        está comparando aleatório com aleatório, e o empate que ela reportar não diz
        nada sobre a hipótese do DOC-07 §2.3.
        """
        return self.tratados / max(self.sorteados_para_tratamento, 1)

    def taxa_efetiva(self) -> float:
        return self.tokens_mascarados / max(self.tokens_mascaraveis, 1)

    def como_dict(self) -> dict:
        return {
            "exemplos": self.exemplos,
            "sorteados_para_tratamento": self.sorteados_para_tratamento,
            "tratados": self.tratados,
            "recaida_sem_equacao": self.recaida_sem_equacao,
            "recaida_equacao_curta": self.recaida_equacao_curta,
            "recaida_equacao_grande": self.recaida_equacao_grande,
            "fracao_tratada": round(self.fracao_tratada(), 4),
            "taxa_efetiva": round(self.taxa_efetiva(), 4),
            "tokens_mascarados": self.tokens_mascarados,
            "tokens_de_equacao_mascarados": self.tokens_de_equacao_mascarados,
            "nota": ("`fracao_tratada` baixa invalida a ablação do DOC-07 §2.3: o "
                     "braço tratado teria recaído em MLM aleatório."),
        }


def mascarar(ids: np.ndarray, id_equacao: np.ndarray, e_display: np.ndarray, *,
             cfg: ConfigMascara, rng: np.random.Generator, id_mask: int,
             n_vocab: int, ids_especiais: frozenset[int],
             contadores: Contadores | None = None,
             ) -> tuple[np.ndarray, np.ndarray]:
    """Devolve `(entrada, alvos)`; `alvos` é −100 onde não há perda.

    `−100` é a convenção do `CrossEntropyLoss` do PyTorch e do `labels` do
    HuggingFace: posição ignorada. Usar 0 seria treinar o modelo a prever `[PAD]`.
    """
    if ids.shape != id_equacao.shape:
        raise ValueError(f"ids {ids.shape} e id_equacao {id_equacao.shape} diferem")
    entrada = ids.copy()
    alvos = np.full(ids.shape, -100, dtype=np.int64)

    # Especiais nunca entram: mascarar `[CLS]` ensina a prever `[CLS]`, e mascarar
    # `[PAD]` gasta orçamento de perda em posições sem informação.
    mascaravel = ~np.isin(ids, list(ids_especiais))
    indices = np.flatnonzero(mascaravel)
    if contadores is not None:
        contadores.exemplos += 1
        contadores.tokens_mascaraveis += int(indices.size)
    if indices.size == 0:
        return entrada, alvos

    n_alvo = int(round(cfg.taxa * indices.size))
    if n_alvo == 0:
        return entrada, alvos

    escolhidos_equacao: np.ndarray = np.empty(0, dtype=np.int64)
    tratar = cfg.p_equacao > 0.0 and rng.random() < cfg.p_equacao
    if tratar:
        if contadores is not None:
            contadores.sorteados_para_tratamento += 1
        escolhidos_equacao = _escolher_equacao(
            id_equacao, e_display, mascaravel, n_alvo, rng, contadores)

    # O resto do orçamento vem de sorteio uniforme entre os mascaráveis que não
    # foram levados pela equação. É isto que mantém o orçamento IGUAL entre os dois
    # braços — ver a decisão 1 na docstring do módulo.
    restante = n_alvo - int(escolhidos_equacao.size)
    if restante > 0:
        livres = np.setdiff1d(indices, escolhidos_equacao, assume_unique=False)
        n = min(restante, livres.size)
        aleatorios = rng.choice(livres, size=n, replace=False) if n else np.empty(
            0, dtype=np.int64)
    else:
        aleatorios = np.empty(0, dtype=np.int64)

    # ── aplicar ─────────────────────────────────────────────────────────────
    # A equação inteira vira `[MASK]`, sem 80/10/10 — decisão 2 da docstring.
    if escolhidos_equacao.size:
        alvos[escolhidos_equacao] = ids[escolhidos_equacao]
        entrada[escolhidos_equacao] = id_mask

    if aleatorios.size:
        alvos[aleatorios] = ids[aleatorios]
        sorte = rng.random(aleatorios.size)
        vira_mask = aleatorios[sorte < cfg.p_mask]
        vira_lixo = aleatorios[(sorte >= cfg.p_mask)
                               & (sorte < cfg.p_mask + cfg.p_aleatorio)]
        entrada[vira_mask] = id_mask
        if vira_lixo.size:
            entrada[vira_lixo] = rng.integers(0, n_vocab, size=vira_lixo.size)
        # O resto fica intacto, de propósito: é o que impede o modelo de aprender
        # que "onde não há [MASK] a entrada é sempre confiável".

    if contadores is not None:
        contadores.tokens_mascarados += int(escolhidos_equacao.size + aleatorios.size)
        contadores.tokens_de_equacao_mascarados += int(escolhidos_equacao.size)
    return entrada, alvos


def _escolher_equacao(id_equacao: np.ndarray, e_display: np.ndarray,
                      mascaravel: np.ndarray, n_alvo: int,
                      rng: np.random.Generator,
                      contadores: Contadores | None) -> np.ndarray:
    """Uma equação de DISPLAY inteira que caiba em `n_alvo`, ou vazio com o motivo.

    ⚠️ Só display, e só acima de `MIN_TOKENS_TRATAMENTO`. Ver a decisão 3 na
    docstring do módulo: a mediana do inline é 7 tokens, e mascarar uma variável não
    testa a hipótese. O motivo de cada recaída é contado separadamente, porque "não
    tinha matemática" e "só tinha equação grande" pedem respostas diferentes.
    """
    # `e_display` restringe ANTES de tudo: um documento cheio de inline e sem
    # nenhuma display não é tratável, e chamar isso de "sem equação" seria mentira.
    de_display = id_equacao[(id_equacao >= 0) & e_display]
    if de_display.size == 0:
        if contadores is not None:
            contadores.recaida_sem_equacao += 1
        return np.empty(0, dtype=np.int64)

    # Só as que cabem no orçamento. Sortear primeiro e desistir depois enviesaria o
    # tratamento contra documentos com muitas equações grandes.
    candidatas, curtas, grandes = [], 0, 0
    for k in np.unique(de_display):
        pos = np.flatnonzero((id_equacao == k) & mascaravel)
        if pos.size < MIN_TOKENS_TRATAMENTO:
            curtas += 1
        elif pos.size > n_alvo:
            grandes += 1
        else:
            candidatas.append(pos)
    if not candidatas:
        if contadores is not None:
            # A recaída mais frequente é a que se reporta: se as duas contarem, a
            # soma passa a exceder o número de exemplos e a fração fica ilegível.
            if grandes >= curtas:
                contadores.recaida_equacao_grande += 1
            else:
                contadores.recaida_equacao_curta += 1
        return np.empty(0, dtype=np.int64)

    escolha = candidatas[int(rng.integers(0, len(candidatas)))]
    if contadores is not None:
        contadores.tratados += 1
    return escolha.astype(np.int64)
