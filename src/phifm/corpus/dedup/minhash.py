"""Deduplicação exata e aproximada (DOC-04 §5).

Dois estágios, na ordem que o DOC-04 §2 justifica — do mais barato ao mais
caro, para que o caro processe o menor conjunto possível:

    1. exata      hash do conteúdo normalizado · custo ~zero · remove 15–25%
    2. aproximada MinHash + LSH · custo alto de CPU · remove 20–35%

═══ Limiar 0,85, não 0,8 ═══

Corpora web usam 0,8. Física exige mais conservador, e a razão é concreta:
papers **legitimamente compartilham** trechos longos — seções de método
padronizadas, descrições do mesmo detector, condições experimentais idênticas.
Dois papers distintos do ATLAS compartilham páginas descrevendo o aparato. Com
0,8, o near-dedup removeria conteúdo genuíno.

═══ LSH é gerador de candidatos, não decisor ═══

A curva-S do LSH tem transição suave: com 16 bandas de 8 linhas o limiar
efetivo fica em ~0,71, o que gera candidatos demais em cima e de menos em
baixo. Por isso o LSH só **propõe** pares, e a decisão usa a estimativa de
Jaccard sobre as assinaturas completas, comparada a 0,85. Dois estágios, e o
segundo é exato até o erro de amostragem do MinHash (±1/√128 ≈ 2,8 pp).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from blake3 import blake3

log = logging.getLogger(__name__)

# ── parâmetros (DOC-04 §5.2) ──────────────────────────────────────────────
N_PERM = 128          # 1,59 M docs × 128 × 4 B ≈ 814 MB — cabe em 8 GB de RAM
SHINGLE = 5           # 5-gramas de palavras, padrão consolidado
LIMIAR = 0.85         # ver docstring
BANDAS = 16           # 16 × 8 = 128; limiar efetivo do LSH ≈ 0,71 (alta revocação)
LINHAS = N_PERM // BANDAS

_MERSENNE = (1 << 61) - 1  # primo de Mersenne: módulo rápido e sem viés
_MAX32 = (1 << 32) - 1

_NAO_PALAVRA = re.compile(r"\W+", re.UNICODE)


def normalizar(texto: str) -> str:
    """Normalização para COMPARAÇÃO — agressiva de propósito.

    Diferente do canonicalizador LaTeX (`core/latex/canonical`), aqui o alvo
    é prosa, e variação de caixa e pontuação é ruído puro. O texto original
    nunca é alterado: isto alimenta só o hash.

    **NFC é obrigatório.** `é` existe em duas codificações Unicode — um único
    ponto de código (U+00E9) ou `e` mais acento combinante (U+0065 U+0301).
    São *canonicamente equivalentes* pela norma, renderizam idêntico, e
    pipelines diferentes produzem uma ou outra. Sem NFC, o mesmo documento
    vindo de duas fontes teria hashes distintos e a dedup exata falharia em
    silêncio — exatamente no caso que ela existe para pegar.

    O que NÃO fazemos: **remover acentos**. Isso é mais agressivo que
    equivalência canônica e mudaria palavras, não codificação. Numa base
    majoritariamente em inglês o ganho seria marginal, e o risco de fundir
    termos distintos em outras línguas é real.
    """
    return _NAO_PALAVRA.sub(" ", unicodedata.normalize("NFC", texto).lower()).strip()


def hash_exato(texto: str) -> str:
    """Chave da dedup exata. 16 bytes: colisão em 10⁹ docs é ~10⁻²¹."""
    return blake3(normalizar(texto).encode("utf-8")).hexdigest(length=16)


def _shingles(texto: str, k: int = SHINGLE) -> np.ndarray:
    """k-gramas de palavras, como hashes de 32 bits, sem repetição."""
    palavras = normalizar(texto).split()
    if len(palavras) < k:
        # Documento curto demais para k-gramas: usa as palavras isoladas, ou
        # o texto inteiro. Sem isso, todo documento curto teria assinatura
        # vazia e colidiria com todos os outros — falso positivo em massa.
        pecas = palavras or [texto[:64]]
    else:
        pecas = [" ".join(palavras[i : i + k]) for i in range(len(palavras) - k + 1)]
    h = {int.from_bytes(blake3(p.encode("utf-8")).digest(length=4), "little") for p in pecas}
    return np.fromiter(h, dtype=np.uint64, count=len(h))


class MinHasher:
    """Gera assinaturas MinHash com permutações universais `(a·x + b) mod p`."""

    def __init__(self, n_perm: int = N_PERM, seed: int = 17):
        rng = np.random.default_rng(seed)
        self.n_perm = n_perm
        # `a` ímpar e não-nulo garante permutação; `dtype=uint64` evita overflow
        self.a = rng.integers(1, _MERSENNE, size=n_perm, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, _MERSENNE, size=n_perm, dtype=np.uint64)

    def assinatura(self, texto: str) -> np.ndarray:
        s = _shingles(texto)
        if s.size == 0:
            return np.full(self.n_perm, _MAX32, dtype=np.uint32)
        # (n_perm, n_shingles) → mínimo por linha
        h = (self.a[:, None] * s[None, :] + self.b[:, None]) % np.uint64(_MERSENNE)
        return (h.min(axis=1) & np.uint64(_MAX32)).astype(np.uint32)

    def assinaturas(self, textos: list[str], lote: int = 2000) -> np.ndarray:
        out = np.empty((len(textos), self.n_perm), dtype=np.uint32)
        for i, t in enumerate(textos):
            out[i] = self.assinatura(t)
            if i and i % (lote * 10) == 0:
                log.info("  assinaturas: %s/%s", f"{i:,}", f"{len(textos):,}")
        return out


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Estimativa de Jaccard: fração de posições em que as assinaturas batem.

    Erro-padrão ≈ 1/√n_perm ≈ 2,8 pp com 128 permutações. É o suficiente para
    decidir contra um limiar de 0,85 — a zona de incerteza fica em [0,82; 0,88]
    e cai do lado conservador por construção do §5.3.
    """
    return float(np.count_nonzero(a == b) / a.size)


@dataclass
class ResultadoDedup:
    exatos_removidos: int = 0
    proximos_removidos: int = 0
    clusters: dict[int, list[int]] = field(default_factory=dict)
    representantes: set[int] = field(default_factory=set)
    pares_avaliados: int = 0

    @property
    def total_removido(self) -> int:
        return self.exatos_removidos + self.proximos_removidos


def bandas_lsh(assinaturas: np.ndarray, bandas: int = BANDAS) -> list[dict[bytes, list[int]]]:
    """Agrupa em baldes por banda. Dois documentos são candidatos se caírem no
    mesmo balde em **ao menos uma** banda."""
    n, k = assinaturas.shape
    linhas = k // bandas
    tabelas: list[dict[bytes, list[int]]] = []
    for b in range(bandas):
        faixa = assinaturas[:, b * linhas : (b + 1) * linhas]
        balde: dict[bytes, list[int]] = defaultdict(list)
        for i in range(n):
            balde[faixa[i].tobytes()].append(i)
        tabelas.append(balde)
    return tabelas


def _uniao_find(n: int):
    pai = list(range(n))

    def acha(x: int) -> int:
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    def une(x: int, y: int) -> None:
        rx, ry = acha(x), acha(y)
        if rx != ry:
            pai[max(rx, ry)] = min(rx, ry)

    return acha, une


def agrupar(
    assinaturas: np.ndarray, limiar: float = LIMIAR, bandas: int = BANDAS
) -> tuple[dict[int, list[int]], int]:
    """LSH propõe candidatos; a estimativa de Jaccard decide.

    Devolve `{id_do_cluster: [índices]}` apenas para clusters com mais de um
    membro, e a contagem de pares efetivamente avaliados — que é a métrica de
    custo, e o que justifica o LSH existir.
    """
    n = assinaturas.shape[0]
    acha, une = _uniao_find(n)
    avaliados = 0
    vistos: set[tuple[int, int]] = set()

    for tabela in bandas_lsh(assinaturas, bandas):
        for membros in tabela.values():
            if len(membros) < 2:
                continue
            # Baldes gigantes são quase sempre artefato (documentos vazios ou
            # boilerplate). Avaliá-los é O(m²) e não agrega.
            if len(membros) > 500:
                log.warning("  balde com %d membros — provável boilerplate, pulado", len(membros))
                continue
            for x in range(len(membros)):
                for y in range(x + 1, len(membros)):
                    i, j = membros[x], membros[y]
                    par = (i, j) if i < j else (j, i)
                    if par in vistos:
                        continue
                    vistos.add(par)
                    avaliados += 1
                    if jaccard(assinaturas[i], assinaturas[j]) >= limiar:
                        une(i, j)

    grupos: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        grupos[acha(i)].append(i)
    return {r: m for r, m in grupos.items() if len(m) > 1}, avaliados
