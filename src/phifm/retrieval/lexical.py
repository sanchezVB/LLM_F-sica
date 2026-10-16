"""Busca lexical (BM25) sobre título + resumo, na MESMA ordem do índice denso.

O índice denso acha o artigo pelo assunto; uma pergunta sobre um detalhe — um composto, uma
técnica, um número — parece pouco com o resumo como um todo, e o encoder a perde (DOC-13
§9.5: o artigo certo chega às fontes em 63% das perguntas do corpo, contra 82% nas do
resumo). Dentro de um artigo, a palavra rara da pergunta foi o que achou a passagem (§9.6:
BM25 0,85 contra 0,33 do encoder). Este índice testa a mesma ideia um nível acima: achar o
ARTIGO pelas palavras raras.

## O formato

Uma matriz esparsa documentos × termos em CSC (coluna por termo), com o PESO BM25 já
calculado em cada célula: pontuar uma consulta é somar as colunas dos termos dela. Os
documentos estão na ordem de `indice_busca/documentos.parquet`, e o manifesto guarda o
hash dessa ordem — fundir com o denso é somar posições, sem tabela de tradução.

Sem dependência nova: `scipy.sparse`, `numpy` e `polars`.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

K1, B = 1.2, 0.75
NOME_MATRIZ, NOME_TERMOS, NOME_MANIFESTO = "pesos_csc.npz", "termos.txt", "manifesto.json"
# A matriz em três arrays soltos, lidos do disco SOB DEMANDA (`mmap`): uma consulta toca só
# as colunas dos seus termos. Inteira na memória ela custava ~1 GB, e com o modelo e a
# busca densa juntos a máquina de 16 GB paginava (DOC-13 §9.9, 2026-10-14).
PARTES = {"indptr": "csc_indptr.npy", "indices": "csc_indices.npy", "data": "csc_data.npy"}
_PALAVRA = re.compile(r"[a-z][a-z0-9]+|\d+(?:\.\d+)?")
# Só as palavras de função mais comuns do inglês: tiradas do índice, encurtam a matriz em
# ~40% e não mudam ordem nenhuma (o idf delas é perto de zero). Nada de física aqui.
PALAVRAS_VAZIAS = frozenset((
    "the", "of", "and", "in", "to", "a", "is", "we", "for", "on", "with", "by", "that",
    "are", "this", "as", "at", "be", "from", "an", "which", "it", "can", "or", "these",
    "has", "have", "was", "were", "its", "their", "our", "also", "been", "not", "than",
    "but", "such", "into", "between", "both", "using", "used", "when", "where", "while",
    "there", "then", "thus", "however", "here", "two", "one"))


def termos(texto: str) -> list[str]:
    """As palavras como o BM25 as vê: minúsculas, letras e dígitos, número decimal com
    ponto (vírgula vira ponto), sem as palavras de função."""
    return [t for t in _PALAVRA.findall(texto.lower().replace(",", "."))
            if t not in PALAVRAS_VAZIAS]


def hash_da_ordem(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def construir(ids: list[str], textos, destino: Path, ao_lote=None) -> dict:
    """O índice de `ids`, na ordem dada. `textos` é um iterável de `(ids_do_lote,
    textos_do_lote)` na MESMA ordem — o chamador lê o spine em lotes, sem a tabela inteira
    na memória."""
    from scipy import sparse

    vocab: dict[str, int] = {}
    # Um par de arrays por LOTE, e não por documento: 1,6 milhão de arrays pequenos
    # custariam ~360 MB só de cabeçalho.
    partes_idx, partes_tf, celulas, comprimentos = [], [], [], []
    vistos = 0
    t0 = time.perf_counter()
    for lote_ids, lote_textos in textos:
        if list(lote_ids) != ids[vistos:vistos + len(lote_ids)]:
            raise RuntimeError("os textos não vêm na ordem dos ids do índice")
        idx_lote: list[int] = []
        tf_lote: list[int] = []
        for texto in lote_textos:
            ts = termos(texto or "")
            comprimentos.append(len(ts))
            cont: dict[int, int] = {}
            for t in ts:
                j = vocab.setdefault(t, len(vocab))
                cont[j] = cont.get(j, 0) + 1
            idx_lote.extend(cont.keys())
            tf_lote.extend(cont.values())
            celulas.append(len(cont))
        partes_idx.append(np.array(idx_lote, dtype=np.int32))
        partes_tf.append(np.array(tf_lote, dtype=np.float32))
        vistos += len(lote_ids)
        if ao_lote:
            ao_lote(vistos, len(vocab), time.perf_counter() - t0)
    if vistos != len(ids):
        raise RuntimeError(f"vieram {vistos} textos para {len(ids)} ids")

    n = len(ids)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(celulas, out=indptr[1:])
    indices = np.concatenate(partes_idx)
    tf = np.concatenate(partes_tf)
    del partes_idx, partes_tf
    dl = np.asarray(comprimentos, dtype=np.float32)
    avgdl = float(dl.mean())
    df = np.bincount(indices, minlength=len(vocab)).astype(np.float64)
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    norma = np.repeat(K1 * (1 - B + B * dl / avgdl), np.diff(indptr))
    pesos = idf[indices] * tf * (K1 + 1) / (tf + norma)
    del tf, norma
    matriz = sparse.csr_matrix((pesos, indices, indptr), shape=(n, len(vocab))).tocsc()
    del pesos, indices

    destino.mkdir(parents=True, exist_ok=True)
    gravar_partes(destino, matriz)
    termos_ordenados = [""] * len(vocab)
    for t, j in vocab.items():
        termos_ordenados[j] = t
    (destino / NOME_TERMOS).write_text("\n".join(termos_ordenados), encoding="utf-8")
    manifesto = {"documentos": n, "termos": len(vocab), "celulas": int(matriz.nnz),
                 "comprimento_medio": round(avgdl, 2), "k1": K1, "b": B,
                 "o_que_e": "BM25 sobre `título. resumo`, na ordem do índice denso",
                 "hash_da_ordem": hash_da_ordem(ids),
                 "palavras_vazias": len(PALAVRAS_VAZIAS),
                 "segundos": round(time.perf_counter() - t0, 1)}
    (destino / NOME_MANIFESTO).write_text(json.dumps(manifesto, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return manifesto


def gravar_partes(destino: Path, matriz) -> None:
    """A matriz CSC em três `.npy`, que `Lexica` abre sob demanda."""
    for nome, arq in PARTES.items():
        np.save(destino / arq, np.ascontiguousarray(getattr(matriz, nome)))


def converter_para_partes(pasta: Path) -> None:
    """O índice gravado como `.npz` (o formato até 2026-10-15), reescrito em partes."""
    from scipy import sparse

    gravar_partes(pasta, sparse.load_npz(pasta / NOME_MATRIZ).tocsc())


class Lexica:
    """O índice aberto. `pontuar(consulta)` devolve o escore BM25 de cada documento. A
    matriz fica no disco e só as colunas consultadas sobem (`mmap`); sem as partes, o
    `.npz` antigo vai inteiro para a memória."""

    def __init__(self, pasta: Path, ids: list[str] | None = None):
        man = pasta / NOME_MANIFESTO
        if not man.exists():
            raise FileNotFoundError(f"{pasta} não tem {NOME_MANIFESTO}: rode a indexação "
                                    "lexical (`scripts/explorar_busca_lexical.py --indexar`).")
        self.manifesto = json.loads(man.read_text(encoding="utf-8"))
        if ids is not None and hash_da_ordem(ids) != self.manifesto["hash_da_ordem"]:
            raise SystemExit("a ordem dos documentos do índice lexical não é a do índice "
                             "denso: fundir os dois somaria escores de artigos diferentes.")
        if all((pasta / arq).exists() for arq in PARTES.values()):
            self.indptr, self.indices, self.data = (
                np.load(pasta / PARTES[n], mmap_mode="r") for n in ("indptr", "indices", "data"))
        else:
            from scipy import sparse

            m = sparse.load_npz(pasta / NOME_MATRIZ).tocsc()
            self.indptr, self.indices, self.data = m.indptr, m.indices, m.data
        self.n = int(self.manifesto["documentos"])
        self.vocab = {t: j for j, t in enumerate(
            (pasta / NOME_TERMOS).read_text(encoding="utf-8").split("\n"))}

    def pontuar(self, consulta: str) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float32)
        for t in set(termos(consulta)):
            j = self.vocab.get(t)
            if j is None:
                continue
            a, b = int(self.indptr[j]), int(self.indptr[j + 1])
            s[self.indices[a:b]] += self.data[a:b]   # numa coluna cada linha aparece 1 vez
        return s


def carregar_ou_avisar(pasta: Path, ids: list[str]) -> Lexica | None:
    """O índice lexical, ou None com um aviso se ele não existir: a busca segue densa."""
    try:
        return Lexica(pasta, ids)
    except FileNotFoundError as e:
        print(f"⚠️ {e} A busca segue só com o encoder.")
        return None


def posicao(escores: np.ndarray, i: int) -> int:
    """A posição (1 = primeiro) do documento `i`: quantos têm escore MAIOR, mais um."""
    return int((escores > escores[i]).sum()) + 1


def rrf(*posicoes: np.ndarray, c: int = 60) -> np.ndarray:
    """Fusão por posição recíproca: Σ 1/(c + posição). Recebe as posições de TODOS os
    documentos em cada lista; devolve um escore (maior = melhor)."""
    return sum(1.0 / (c + p.astype(np.float64)) for p in posicoes)


def posicoes_de(escores: np.ndarray) -> np.ndarray:
    """A posição de cada documento (1 = primeiro), com empate desfeito pela ordem."""
    ordem = np.argsort(-escores, kind="stable")
    pos = np.empty(escores.size, dtype=np.int64)
    pos[ordem] = np.arange(1, escores.size + 1)
    return pos
