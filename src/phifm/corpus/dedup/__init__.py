"""Deduplicação (DOC-04 §5)."""

from phifm.corpus.dedup.minhash import (
    LIMIAR,
    N_PERM,
    MinHasher,
    ResultadoDedup,
    agrupar,
    hash_exato,
    jaccard,
    normalizar,
)
from phifm.corpus.dedup.representante import Candidato, escolher, ganho_de_licenca

__all__ = [
    "LIMIAR", "N_PERM", "MinHasher", "ResultadoDedup", "agrupar",
    "hash_exato", "jaccard", "normalizar",
    "Candidato", "escolher", "ganho_de_licenca",
]
