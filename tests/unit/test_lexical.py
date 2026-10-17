"""O índice lexical (BM25 sobre título + resumo): a fórmula, a ordem e a fusão."""
from __future__ import annotations

import math

import numpy as np
import pytest

from phifm.retrieval import lexical as lx

DOCS = {"a1": "Superconducting qubits. Coherence of transmon qubits is limited by TLS.",
        "a2": "Graphene growth. Graphene was grown on copper at 1000 C by CVD.",
        "a3": "MgB2 films. Films of MgB2 grown by pulsed laser deposition on sapphire.",
        "a4": "Dark matter. The dark matter halo of the galaxy and its rotation curve."}


def _construir(tmp_path, docs=DOCS, lote=2):
    ids = list(docs)
    lotes = [(ids[i:i + lote], [docs[a] for a in ids[i:i + lote]])
             for i in range(0, len(ids), lote)]
    return ids, lx.construir(ids, lotes, tmp_path)


def test_termos_minusculos_numero_com_ponto_e_sem_palavras_de_funcao():
    assert lx.termos("The MgB2 films of 0,35 nm, grown at 650 K") == [
        "mgb2", "films", "0.35", "nm", "grown", "650"]


def test_o_peso_guardado_e_o_bm25_da_formula(tmp_path):
    ids, man = _construir(tmp_path)
    idx = lx.Lexica(tmp_path, ids)
    s = idx.pontuar("graphene")
    # "graphene" só em a2, duas vezes; o peso é o BM25 escrito à mão
    n, df, tf = 4, 1, 2
    dl = len(lx.termos(DOCS["a2"]))
    idf = math.log1p((n - df + 0.5) / (df + 0.5))
    esperado = idf * tf * (lx.K1 + 1) / (tf + lx.K1 * (1 - lx.B + lx.B * dl / man["comprimento_medio"]))
    assert s[1] == pytest.approx(esperado, rel=1e-5)
    assert s[0] == s[2] == s[3] == 0
    assert man["documentos"] == 4 and man["celulas"] == len(idx.data)
    assert isinstance(idx.data, np.memmap)            # do disco, sob demanda


def test_o_npz_antigo_ainda_abre_e_vira_partes(tmp_path):
    from scipy import sparse

    ids, _ = _construir(tmp_path)
    novo = lx.Lexica(tmp_path, ids).pontuar("graphene copper")
    indptr = np.load(tmp_path / lx.PARTES["indptr"])
    m = sparse.csc_matrix((np.load(tmp_path / lx.PARTES["data"]),
                           np.load(tmp_path / lx.PARTES["indices"]), indptr),
                          shape=(4, len(indptr) - 1))
    sparse.save_npz(tmp_path / lx.NOME_MATRIZ, m, compressed=False)
    for arq in lx.PARTES.values():
        (tmp_path / arq).unlink()
    antigo = lx.Lexica(tmp_path, ids)
    assert not isinstance(antigo.data, np.memmap)
    assert np.array_equal(antigo.pontuar("graphene copper"), novo)
    lx.converter_para_partes(tmp_path)
    assert np.array_equal(lx.Lexica(tmp_path, ids).pontuar("graphene copper"), novo)


def test_a_palavra_rara_acha_o_artigo_e_termo_desconhecido_nao_quebra(tmp_path):
    ids, _ = _construir(tmp_path)
    idx = lx.Lexica(tmp_path, ids)
    s = idx.pontuar("What substrate was used for the MgB2 films grown by pulsed laser deposition?")
    assert int(np.argmax(s)) == 2 and lx.posicao(s, 2) == 1
    assert idx.pontuar("zzzz qqqq").sum() == 0


def test_recusa_ordem_diferente_da_do_indice_denso(tmp_path):
    ids, _ = _construir(tmp_path)
    with pytest.raises(SystemExit, match="ordem"):
        lx.Lexica(tmp_path, list(reversed(ids)))
    with pytest.raises(RuntimeError, match="ordem"):
        lx.construir(ids, [(["a2", "a1"], ["x", "y"])], tmp_path / "outro")


def test_posicoes_e_rrf():
    s = np.array([0.1, 0.9, 0.5, 0.5])
    assert lx.posicoes_de(s).tolist() == [4, 1, 2, 3] and lx.posicao(s, 3) == 2
    denso, lexico = np.array([1, 2, 3]), np.array([3, 1, 2])
    f = lx.rrf(denso, lexico)
    assert int(np.argmax(f)) == 1                     # 2º e 1º vence 1º e 3º
    assert f[0] == pytest.approx(1 / 61 + 1 / 63)


def test_melhor_janela_acha_o_trecho_e_devolve_None_sem_texto():
    longe = "graphene copper growth " * 100
    perto = "graphene copper growth " * 60 + "the MgB2 film was grown at 650 K on sapphire " \
        + "graphene copper growth " * 40
    s = lx.melhor_janela("MgB2 film grown on sapphire at 650 K", [longe, None, perto, ""])
    assert s[1] is None and s[2] > s[0] >= 0 and s[3] == 0


def test_ordem_com_texto_quem_nao_tem_texto_nao_perde_nem_ganha():
    # sem diferença nenhuma no ΦRank e no texto, a ordem é a da busca
    assert lx.ordem_com_texto([0.0] * 4, [1.0] * 4) == [0, 1, 2, 3]
    assert lx.ordem_com_texto([0.0] * 4, [None] * 4) == [0, 1, 2, 3]
    # o 4º tem o melhor trecho e o melhor ΦRank: sobe para o topo
    assert lx.ordem_com_texto([0.1, 0.2, 0.3, 0.9], [0.0, 0.0, 0.0, 5.0])[0] == 3
    # o 1º não tem texto: na lista do texto fica na posição da busca (1ª), e não afunda
    ordem = lx.ordem_com_texto([0.5, 0.4, 0.3, 0.2], [None, 0.3, 0.2, 0.1])
    assert ordem[0] == 0
