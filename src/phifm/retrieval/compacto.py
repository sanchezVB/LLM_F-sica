"""Os vetores do índice em int8: metade da memória do float16, para o acervo poder crescer
(DOC-13 §8.1, ACEITA em 2026-10-17).

Hoje os 1,59 M vetores de 384 dimensões em float16 ocupam 1,22 GB, e a memória livre da
máquina de 16 GB desce a 0,3–0,5 GB durante as perguntas (DOC-13 §9.9, "a paginação,
resolvida"). Com ~2 M artigos a mais (o peS2o de fora do arXiv), o float16 passaria de
2,7 GB e a máquina voltaria a paginar. Em int8, o índice dobrado ocupa o mesmo de hoje.

## A conta

Quantização escalar simétrica, uma escala por dimensão: `v ≈ escala_d · round(v_d /
escala_d)`, com `escala_d` = o percentil `PERCENTIL` de |v_d| numa amostra fixa de
`N_AMOSTRA` vetores, dividido por 127, e o resto cortado em ±127. O cosseno vira
`q · (escala ⊙ v_int8)` = `(q ⊙ escala) · v_int8`: a consulta é escalada uma vez, e a
matriz é lida em int8, por blocos, convertida para float32 na hora — como o float16 já é
hoje (`Busca.pontuar`).

## O que NÃO está aqui

A troca na busca do produto (`phifm.retrieval.indice.Busca`). Ela só faz sentido com o
acervo maior, e passa pela medida de equivalência da §8.1 antes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# ── a regra (DOC-13 §8.1) ────────────────────────────────────────────────────
# A medida recusa enquanto isto for False. ACEITA pelo dono em 2026-10-17, como escrita.
REGRA_8_1_ACEITA = True
PERCENTIL = 99.99
N_AMOSTRA = 100_000
SEMENTE = 20261017
BLOCO = 200_000
NOME_INT8, NOME_ESCALA, NOME_MANIFESTO = "vetores_int8.npy", "escala_int8.npy", "compacto.json"


def escala(vetores: np.ndarray, n_amostra: int = N_AMOSTRA, semente: int = SEMENTE,
           percentil: float = PERCENTIL) -> np.ndarray:
    """A escala de cada dimensão, de uma amostra fixa de linhas (lidas em ordem: o `.npy`
    está mapeado do HD, e salto para trás custa)."""
    n = vetores.shape[0]
    idx = np.sort(np.random.default_rng(semente).choice(n, min(n_amostra, n), replace=False))
    amostra = np.abs(np.asarray(vetores[idx], dtype=np.float32))
    e = np.percentile(amostra, percentil, axis=0).astype(np.float32) / 127.0
    if not np.all(e > 0):
        raise ValueError(f"{int((e <= 0).sum())} dimensões com escala zero: vetores constantes?")
    return e


def quantizar(vetores: np.ndarray, e: np.ndarray, bloco: int = BLOCO) -> np.ndarray:
    saida = np.empty(vetores.shape, dtype=np.int8)
    for i in range(0, vetores.shape[0], bloco):
        v = np.asarray(vetores[i:i + bloco], dtype=np.float32) / e
        saida[i:i + bloco] = np.clip(np.rint(v), -127, 127).astype(np.int8)
    return saida


def pontuar(m: np.ndarray, e: np.ndarray, q: np.ndarray, bloco: int = BLOCO) -> np.ndarray:
    """O cosseno aproximado de `q` com cada documento, em float32, na ordem da tabela."""
    qs = (np.asarray(q, dtype=np.float32) * e).astype(np.float32)
    return np.concatenate([m[i:i + bloco].astype(np.float32) @ qs
                           for i in range(0, m.shape[0], bloco)])


def topo(s: np.ndarray, k: int) -> list[int]:
    """Os `k` de maior escore, na ordem — com o desempate estável de `Busca._resultados`."""
    k = min(k, s.size)
    idx = np.argpartition(-s, k - 1)[:k]
    return [int(j) for j in idx[np.argsort(-s[idx], kind="stable")]]


def _sha_arquivo(arq: Path) -> str:
    h = hashlib.sha256()
    with arq.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 22), b""):
            h.update(bloco)
    return h.hexdigest()


def converter(indice: Path, destino: Path) -> dict:
    """Grava a versão int8 do índice em `destino` (outra pasta: o índice do produto não é
    tocado). O manifesto amarra a conversão ao índice de origem."""
    man_origem = json.loads((indice / "manifesto.json").read_text(encoding="utf-8"))
    vetores = np.load(indice / "vetores.npy", mmap_mode="r")
    e = escala(vetores)
    m = quantizar(vetores, e)
    destino.mkdir(parents=True, exist_ok=True)
    np.save(destino / NOME_INT8, m)
    np.save(destino / NOME_ESCALA, e)
    man = {"o_que_e": "os vetores do índice em int8, escala por dimensão (DOC-13 §8.1)",
           "origem": str(indice), "origem_modelo_blake3": man_origem.get("modelo_blake3"),
           "documentos": int(m.shape[0]), "dimensao": int(m.shape[1]),
           "percentil": PERCENTIL, "n_amostra": N_AMOSTRA, "semente": SEMENTE,
           "bytes_float16": int(vetores.shape[0] * vetores.shape[1] * 2),
           "bytes_int8": int(m.nbytes), "sha256_int8": _sha_arquivo(destino / NOME_INT8)}
    (destino / NOME_MANIFESTO).write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man


def carregar(destino: Path, indice: Path) -> tuple[np.ndarray, np.ndarray]:
    """`(matriz int8, escala)`, conferidos contra o índice: mesmo modelo e mesmo tamanho."""
    man = json.loads((destino / NOME_MANIFESTO).read_text(encoding="utf-8"))
    origem = json.loads((indice / "manifesto.json").read_text(encoding="utf-8"))
    m = np.load(destino / NOME_INT8)
    if man["origem_modelo_blake3"] != origem.get("modelo_blake3") or m.shape[0] != man["documentos"]:
        raise ValueError("a versão int8 não é deste índice (outro modelo ou outro tamanho): "
                         "converta de novo.")
    return m, np.load(destino / NOME_ESCALA)


# ── a medida de equivalência ─────────────────────────────────────────────────


def sobreposicao(a: list[int], b: list[int]) -> float:
    return len(set(a) & set(b)) / len(a) if a else 1.0


MIN_SOBREPOSICAO_50 = 0.98        # dos 50 que vão ao ΦRank, em média
MAX_PERDAS_LIQUIDAS = 1           # itens em que o artigo certo sai dos 50, menos os que entram


def decidir(itens: list[dict]) -> dict:
    """`itens`: por pergunta, a sobreposição dos 50 da busca fundida e se o artigo certo
    está nos 50 com float16 (`p_exato`) e com int8 (`p_int8`)."""
    if not itens:
        raise ValueError("nenhum item medido")
    sob = float(np.mean([d["sobreposicao_50_fundida"] for d in itens]))
    perde = sum(1 for d in itens if d["p_exato"] and not d["p_int8"])
    ganha = sum(1 for d in itens if d["p_int8"] and not d["p_exato"])
    passa = sob >= MIN_SOBREPOSICAO_50 and perde - ganha <= MAX_PERDAS_LIQUIDAS
    return {"n": len(itens), "sobreposicao_media_50": round(sob, 4),
            "min_sobreposicao": MIN_SOBREPOSICAO_50, "perde_o_artigo": perde,
            "ganha_o_artigo": ganha, "max_perdas_liquidas": MAX_PERDAS_LIQUIDAS,
            "equivalente": passa}
