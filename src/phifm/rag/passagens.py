"""Passagens sob demanda: os trechos do texto completo dos artigos que a busca JÁ achou.

O assistente só lia resumos, e a medida do corpo (DOC-13 §9.5) mostrou o custo: 0,033 de
acerto no que o resumo não diz, 0,940 quando a passagem certa está no prompt. Aqui ela é
procurada: o texto completo de cada fonte é cortado (`phifm.retrieval.trechos`), os trechos
das fontes competem entre si pela PERGUNTA, e os melhores entram no prompt, debaixo do
resumo do artigo de onde vieram. Não há índice de trechos do corpus: o que se corta são as
6 fontes de uma pergunta, na hora.

## A regra que escolhe — lexical, e com a pergunta em inglês

Desenvolvida nas 150 perguntas do corpo (`scripts/explorar_passagens.py`, 2026-10-12), nas
94 em que o artigo certo estava entre as fontes. Alcance = a passagem de onde a pergunta
saiu está entre os 12 primeiros trechos:

| consulta | ordem dos trechos | alcance |
|---|---|---|
| a pergunta, como veio (português) | ΦEmb (cosseno) | 0,33 |
| a consulta hipotética da busca | ΦEmb | 0,32 |
| a consulta hipotética da busca | BM25 | 0,70 |
| **a pergunta traduzida para inglês** | **BM25** | **0,85** |

O encoder foi treinado em `título. resumo` e não acha um detalhe de método; e a consulta
hipotética imita um resumo, então puxa a introdução. O que distingue a passagem são as
palavras RARAS da pergunta — o nome do instrumento, do parâmetro —, e para casá-las com um
texto em inglês a pergunta tem de estar em inglês: uma chamada curta do modelo (~1 s).

A frequência de documento do BM25 sai só dos trechos das fontes daquela pergunta. As
fontes são do mesmo assunto; o que é raro ENTRE elas é o que separa a passagem.

⚠️ O alcance é do desenvolvimento, em perguntas já olhadas e escritas pelo Claude. O que
vale é a medida da §9.6, em perguntas novas.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from phifm.retrieval.trechos import Trecho

SISTEMA_TRADUCAO = (
    "Translate the physics question into English. Keep every symbol, number, name and "
    "technical term exactly. Output only the translated question.")
LIMITE_TRECHO = 160      # o corte da auditoria M1 e das passagens do estrato "corpo"
K_PASSAGENS = 9          # trechos entregues por pergunta, somadas as fontes
# O orçamento é em TOKENS, não em trechos: uma equação maior que o limite sai num trecho
# só (`excede`), e nove desses estourariam o contexto do modelo sem aviso.
ORCAMENTO = K_PASSAGENS * LIMITE_TRECHO

_PALAVRA = re.compile(r"[a-z][a-z0-9]+|\d+(?:\.\d+)?")


def traduzir(modelo, pergunta: str, semente: int | None = None) -> str:
    """A pergunta em inglês, para as buscas por palavra (os trechos aqui; o artigo, na
    §9.8 do DOC-13). Se o modelo devolver vazio, vale a pergunta como veio."""
    texto = modelo.gerar(SISTEMA_TRADUCAO, pergunta, max_tokens=200, temperatura=0.3,
                         semente=semente)
    return " ".join(texto.split()) or pergunta


def palavras(texto: str) -> list[str]:
    """Minúsculas, palavras de duas letras ou mais e números (vírgula decimal vira ponto)."""
    return _PALAVRA.findall(texto.lower().replace(",", "."))


def bm25(consulta: str, docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """O escore BM25 de cada documento para a consulta. A frequência de documento sai de
    `docs` — aqui, os trechos das fontes de UMA pergunta."""
    n = len(docs)
    df = Counter(p for d in docs for p in set(d))
    media = sum(len(d) for d in docs) / max(n, 1)
    termos = set(palavras(consulta))
    s = np.zeros(n, dtype=np.float32)
    for j, d in enumerate(docs):
        tf = Counter(d)
        for t in termos & tf.keys():
            idf = np.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s[j] += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / media))
    return s


@dataclass
class Escolhido:
    arxiv_id: str
    posicao: int          # o índice do trecho dentro do artigo
    secao: str
    texto: str
    n_tokens: int
    escore: float


def escolher(consulta: str, trechos: dict[str, list[Trecho]], k: int = K_PASSAGENS,
             orcamento: int = ORCAMENTO) -> list[Escolhido]:
    """Os melhores trechos para a consulta, entre os de todos os artigos de `trechos`
    (arxiv_id → trechos, na ordem do artigo). No máximo `k`, e sem passar do `orcamento`
    de tokens; trecho que não casa com palavra nenhuma da consulta não entra. Devolve na
    ordem do escore; empate fica com quem vem antes."""
    todos = [(a, i, t) for a, ts in trechos.items() for i, t in enumerate(ts)]
    if not todos:
        return []
    s = bm25(consulta, [palavras(t.texto) for _, _, t in todos])
    saida: list[Escolhido] = []
    gasto = 0
    for j in np.argsort(-s, kind="stable"):
        if len(saida) >= k or s[j] <= 0:
            break
        a, i, t = todos[j]
        if gasto + t.n_tokens > orcamento:
            continue
        gasto += t.n_tokens
        saida.append(Escolhido(a, i, t.secao, t.texto, t.n_tokens, float(s[j])))
    return saida


def em_blocos(escolhidos: list[Escolhido]) -> dict[str, list[str]]:
    """arxiv_id → as passagens daquele artigo, na ordem do TEXTO, com os trechos vizinhos
    colados num bloco só: o modelo lê um parágrafo e a continuação dele, não dois recortes."""
    por_artigo: dict[str, list[Escolhido]] = {}
    for e in escolhidos:
        por_artigo.setdefault(e.arxiv_id, []).append(e)
    saida: dict[str, list[str]] = {}
    for a, es in por_artigo.items():
        es.sort(key=lambda e: e.posicao)
        blocos, anterior = [], None
        for e in es:
            if anterior is not None and e.posicao == anterior + 1:
                blocos[-1] = f"{blocos[-1]} {e.texto}"
            else:
                blocos.append(e.texto)
            anterior = e.posicao
        saida[a] = blocos
    return saida


class TrechosDoTexto:
    """`trechos_de` para o produto: lê o texto completo (`phifm.retrieval.textos.Textos`) e
    corta. Os últimos `guardados` artigos ficam cortados na memória — quem pergunta de novo
    sobre o mesmo assunto recebe as mesmas fontes."""

    def __init__(self, textos, contar: Callable[[list[str]], list[int]],
                 limite: int = LIMITE_TRECHO, guardados: int = 64):
        import threading

        self.textos, self.contar, self.limite, self.guardados = textos, contar, limite, guardados
        self._cache: dict[str, list[Trecho] | None] = {}
        # A página atende pedidos em threads: duas perguntas não mexem na fila juntas.
        self._trava = threading.Lock()

    def __call__(self, arxiv_id: str) -> list[Trecho] | None:
        from phifm.retrieval.trechos import cortar

        with self._trava:
            if arxiv_id in self._cache:
                self._cache[arxiv_id] = self._cache.pop(arxiv_id)   # vai para o fim da fila
                return self._cache[arxiv_id]
            texto = self.textos.ler(arxiv_id)
            trechos = (None if texto is None
                       else cortar(texto, self.contar, limite=self.limite, sobreposicao=0))
            self._cache[arxiv_id] = trechos
            while len(self._cache) > self.guardados:
                del self._cache[next(iter(self._cache))]
            return trechos


def contador(tok) -> Callable[[list[str]], list[int]]:
    """Textos → nº de tokens de cada um, pelo tokenizer do índice (o do corte medido)."""
    def contar(textos: list[str]) -> list[int]:
        if not textos:
            return []
        return [len(x) for x in tok(textos, add_special_tokens=False, verbose=False)["input_ids"]]

    return contar


def carregar_ou_avisar(modelo, tok, pasta, semente: int | None = None) -> Passagens | None:
    """O passo das passagens lendo o texto completo de `pasta`, ou None com um aviso se a
    cópia em grupos pequenos não existe: o assistente segue só com os resumos."""
    from phifm.retrieval.textos import Textos

    try:
        textos = Textos(pasta)
    except FileNotFoundError as e:
        print(f"⚠️ sem passagens do texto completo: {e}")
        return None
    return Passagens(modelo, TrechosDoTexto(textos, contador(tok)), semente=semente)


class Passagens:
    """O passo que o `Assistente` chama depois da busca. `trechos_de(arxiv_id)` devolve os
    trechos do artigo, ou None se ele não tem texto completo — de onde eles vêm (o HD, um
    cache, um arquivo de medida) não é assunto daqui."""

    def __init__(self, modelo, trechos_de: Callable[[str], list[Trecho] | None],
                 k: int = K_PASSAGENS, semente: int | None = None):
        self.modelo, self.trechos_de, self.k, self.semente = modelo, trechos_de, k, semente

    def consulta(self, pergunta: str) -> str:
        """A pergunta em inglês. Se o modelo devolver vazio, vale a pergunta como veio."""
        return traduzir(self.modelo, pergunta, self.semente)

    def escolher(self, consulta: str, ids: list[str]) -> list[Escolhido]:
        trechos = {a: ts for a in ids if (ts := self.trechos_de(a))}
        return escolher(consulta, trechos, self.k, self.k * LIMITE_TRECHO)

    def anexar_com_registro(self, pergunta: str, fontes: list, consulta: str | None = None
                            ) -> tuple[list, list[Escolhido], str]:
        """`anexar`, devolvendo também os trechos escolhidos e a consulta em inglês — é o
        que a medida guarda para dizer, depois, se a passagem de onde a pergunta saiu foi
        entregue, e com que palavras ela foi procurada."""
        consulta = consulta or self.consulta(pergunta)
        escolhidos = self.escolher(consulta, [f.arxiv_id for f in fontes])
        blocos = em_blocos(escolhidos)
        return ([replace(f, passagens=tuple(blocos.get(f.arxiv_id, ()))) for f in fontes],
                escolhidos, consulta)

    def anexar(self, pergunta: str, fontes: list, consulta: str | None = None) -> list:
        """As mesmas fontes, na mesma ordem e com o mesmo número, cada uma com as suas
        passagens. Fonte sem texto completo, ou sem trecho escolhido, volta como veio."""
        return self.anexar_com_registro(pergunta, fontes, consulta)[0]
