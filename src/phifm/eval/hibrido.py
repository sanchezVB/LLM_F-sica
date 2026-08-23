"""Busca híbrida: BM25 léxico + ΦEmb denso, fundidos, com ΦRank reordenando.

É a composição que o T1b entrega (DOC-01 §5 seleciona "Qdrant vetorial +
OpenSearch/BM25 lexical + ranqueamento híbrido fundido"). Aqui, na versão local e
sem custo: BM25 em processo sobre scipy esparso, os vetores densos que já estão em
cache, e fusão por posto.

    consulta ─┬─► BM25    ─► top-K léxico   ─┐
              └─► ΦEmb    ─► top-K denso    ─┴─► RRF ─► top-100 ─► ΦRank ─► top-10

## Por que os três, e o que cada um cobre

**BM25** acerta o que o denso erra: nome próprio, sigla, número. Uma consulta que
menciona `SU(2)` ou `Λ-CDM` casa lexicalmente mesmo que o embedding não tenha
aprendido o termo. É também o que funciona em documento que o modelo nunca viu.

**ΦEmb** acerta o que o BM25 erra: paráfrase, sinônimo, o paper que trata do mesmo
fenômeno com outro vocabulário.

**ΦRank** não recupera nada — ele só reordena o que os dois trouxeram. Se o
documento certo não está no top-100, nenhum reranker o traz de volta, e é por isso
que `recall@100 do recuperador` é a métrica que limita todo o resto.

## Por que fusão por POSTO (RRF) e não por escore

Somar escores exige que eles sejam comparáveis. Não são: o cosseno do ΦEmb vive em
[-1, 1] e o BM25 é ilimitado e depende do tamanho do corpus e da consulta.
Normalizar (min-max, z-score) introduz uma dependência do LOTE de candidatos — o
mesmo documento recebe escores diferentes conforme quem mais foi recuperado, o que
é uma forma sutil de vazamento entre consultas.

RRF usa só a POSIÇÃO: `1/(k + posto)`. Não precisa de normalização, é imune a
escalas e é o padrão da literatura de fusão. O `k=60` vem de Cormack et al. (2009)
e amortece o peso das primeiras posições — sem ele, um sistema que erra com
confiança domina a fusão.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)

# Tokenização. ⚠️ Mantém `\command`, dígitos e hífen interno, porque em texto de
# Física eles carregam significado: `SU(2)`, `Λ-CDM`, `\alpha`, `hep-th`. Um
# `\w+` simples partiria todos e o BM25 perderia exatamente onde ele é melhor que
# o denso.
_TOKEN = re.compile(r"\\?[A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)*|\d+(?:\.\d+)?")

# Amortecimento do RRF. Cormack et al. (2009). Ver a docstring do módulo.
K_RRF = 60


def tokenizar(texto: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(texto)]


@dataclass
class BM25:
    """BM25 Okapi sobre matriz esparsa.

    Implementado aqui em vez de trazer `rank_bm25` por dois motivos: são ~40 linhas
    de matemática conhecida, e a tokenização precisa ser a nossa — a da biblioteca
    partiria `hep-th` e `\\alpha`.

    `k1=1.5` e `b=0.75` são os valores padrão da literatura. Não foram ajustados
    para este corpus, e ajustá-los sem um conjunto de validação separado seria
    escolher hiperparâmetro no conjunto de teste.
    """

    k1: float = 1.5
    b: float = 0.75
    vocab: dict[str, int] = field(default_factory=dict)
    idf: np.ndarray | None = None
    # (documentos x termos), já com a saturação do BM25 aplicada por documento.
    matriz: sp.csr_matrix | None = None
    n_docs: int = 0

    def indexar(self, docs: list[str]) -> BM25:
        tokens = [tokenizar(d) for d in docs]
        self.n_docs = len(docs)
        comprimentos = np.array([len(t) for t in tokens], dtype=np.float32)
        media = float(comprimentos.mean()) if self.n_docs else 1.0

        self.vocab = {}
        linhas, colunas, valores = [], [], []
        df = Counter()
        for i, toks in enumerate(tokens):
            cont = Counter(toks)
            df.update(cont.keys())
            norm = self.k1 * (1 - self.b + self.b * len(toks) / max(media, 1e-9))
            for termo, tf in cont.items():
                j = self.vocab.setdefault(termo, len(self.vocab))
                linhas.append(i)
                colunas.append(j)
                # Saturação pré-computada: o peso de um termo no documento não
                # depende da consulta, então cabe no índice.
                valores.append(tf * (self.k1 + 1) / (tf + norm))

        self.matriz = sp.csr_matrix(
            (valores, (linhas, colunas)),
            shape=(self.n_docs, len(self.vocab)), dtype=np.float32)

        # IDF de Robertson com o `+1` de suavização, que impede valor negativo para
        # termo presente em mais da metade dos documentos. Sem ele, um termo muito
        # comum PENALIZA o documento que o contém, o que inverte o sentido.
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for termo, j in self.vocab.items():
            n = df[termo]
            self.idf[j] = math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))
        log.info("BM25: %s documentos, %s termos", f"{self.n_docs:,}",
                 f"{len(self.vocab):,}")
        return self

    def pontuar(self, consulta: str) -> np.ndarray:
        """Escore de todos os documentos para uma consulta."""
        if self.matriz is None or self.idf is None:
            raise RuntimeError("indexar() antes de pontuar()")
        js = [self.vocab[t] for t in set(tokenizar(consulta)) if t in self.vocab]
        if not js:
            return np.zeros(self.n_docs, dtype=np.float32)
        sub = self.matriz[:, js]
        return np.asarray(sub @ self.idf[js]).ravel()


def fundir_rrf(*listas: list[int], k: int = K_RRF) -> list[int]:
    """Funde listas ordenadas de índices por Reciprocal Rank Fusion.

    Recebe posições, não escores — ver a docstring do módulo para por que somar
    escores de sistemas diferentes é errado.
    """
    pontos: dict[int, float] = {}
    for lista in listas:
        for posto, doc in enumerate(lista, start=1):
            pontos[doc] = pontos.get(doc, 0.0) + 1.0 / (k + posto)
    return [d for d, _ in sorted(pontos.items(), key=lambda x: -x[1])]


def top_k(escores: np.ndarray, k: int) -> list[int]:
    """Índices dos `k` maiores, em ordem decrescente.

    `argpartition` e não `argsort`: ordenar 667 mil escores por consulta custaria
    ~20x mais, e só o topo importa.
    """
    k = min(k, len(escores))
    if k <= 0:
        return []
    parcial = np.argpartition(-escores, k - 1)[:k]
    return [int(i) for i in parcial[np.argsort(-escores[parcial])]]


def recall_em(posicoes: list[int | None], k: int) -> float:
    """Fração das consultas cujo documento certo apareceu até a posição `k`.

    ⚠️ É o TETO de tudo que vem depois. Um reranker perfeito sobre um recuperador
    com recall@100 de 0,70 não passa de 0,70 — e nenhuma melhora de reranking
    aparece nas consultas em que o documento certo nunca chegou.
    """
    if not posicoes:
        return 0.0
    return sum(1 for p in posicoes if p is not None and p < k) / len(posicoes)


def ndcg_em_10(posicoes: list[int | None]) -> float:
    """nDCG@10 com UM relevante por consulta: 1/log2(1+posição), 0 fora do top-10.

    Mesma definição do avaliador do G1 (`eval/encoders.py`), de propósito: dois
    nDCG diferentes no mesmo projeto seriam duas réguas com o mesmo nome.
    """
    if not posicoes:
        return 0.0
    return sum(0.0 if p is None or p >= 10 else 1.0 / math.log2(2 + p)
               for p in posicoes) / len(posicoes)
