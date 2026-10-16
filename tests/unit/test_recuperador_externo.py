"""O recuperador contra os de fora (DOC-13 §9.12): configurações, texto, pool, ordenação e a
leitura — sem torch e sem modelo."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.eval import recuperador_externo as rx  # noqa: E402
from phifm.training.pairs import textos_de_documentos  # noqa: E402

# ── as configurações ─────────────────────────────────────────────────────────


def test_os_de_fora_como_os_autores_mandam_e_o_nosso_como_na_tabela_6():
    assert (rx.SPECTER2.pooling, rx.SPECTER2.formato, rx.SPECTER2.max_tokens,
            rx.SPECTER2.similaridade, rx.SPECTER2.adaptador) == (
        "cls", "sep", 512, "euclidiana", "allenai/specter2")
    assert (rx.SCINCL.pooling, rx.SCINCL.formato, rx.SCINCL.max_tokens) == ("cls", "sep", 512)
    assert (rx.NOSSO.pooling, rx.NOSSO.formato, rx.NOSSO.max_tokens, rx.NOSSO.similaridade) == (
        "media", "ponto", 192, "cosseno")
    t6 = rx.no_protocolo_da_tabela_6(rx.SPECTER2)
    assert (t6.pooling, t6.formato, t6.max_tokens, t6.similaridade) == (
        "media", "ponto", 192, "cosseno") and t6.adaptador == "allenai/specter2"
    chaves = [c.chave for c in rx.NA_NOSSA_TAREFA]
    assert len(set(chaves)) == len(chaves)
    assert set(rx.TABELA_6) == {rx.MINILM.caminho, rx.NOSSO.caminho}


def test_o_texto_ponto_e_byte_a_byte_o_de_treino():
    titulos = ["Spin  waves\n in films", None, "  A.B ", "x"]
    resumos = ["We  study\tmagnons. ", "Only abstract.", None, " y  z "]
    spine = pl.DataFrame({"arxiv_id": ["a", "b", "c", "d"], "title": titulos,
                          "abstract": resumos})
    esperado = dict(textos_de_documentos(spine.lazy().with_columns(
        pl.col("abstract").fill_null("") + pl.lit("x" * 130))).collect().iter_rows())
    for i, (t, r) in zip("abcd", zip(titulos, resumos, strict=True), strict=True):
        assert rx.texto(t, (r or "") + "x" * 130, "ponto") == esperado[i]
    assert rx.texto(" Spin  waves ", " We study. ", "sep", "[SEP]") == "Spin waves[SEP]We study."
    with pytest.raises(ValueError):
        rx.texto("t", "r", "outro")


# ── o pool ───────────────────────────────────────────────────────────────────


def _val_e_spine(n=40):
    ids = [f"p{i}" for i in range(n)]
    cit = [f"c{i}" for i in range(n)]
    tit = {x: f"Title {x}" for x in ids + cit}
    res = {x: f"Abstract of {x} " + "w" * 120 for x in ids + cit}
    val = pl.DataFrame({"arxiv_id": ids, "arxiv_citado": cit,
                        "ancora": [rx.texto(tit[x], res[x], "ponto") for x in ids],
                        "positivo": [rx.texto(tit[x], res[x], "ponto") for x in cit]})
    spine = pl.DataFrame({"arxiv_id": list(tit), "title": list(tit.values()),
                          "abstract": [res[x] for x in tit]})
    return val, spine


def test_pool_com_campos_mantem_o_pareamento_e_confere_o_texto():
    val, spine = _val_e_spine()
    pool = rx.pool_com_campos(val, spine.lazy(), n=30, semente=3)
    assert pool.height == 30
    for row in pool.iter_rows(named=True):
        assert row["titulo_a"] == f"Title {row['arxiv_id']}"
        assert row["titulo_p"] == f"Title {row['arxiv_citado']}"
    mexido = spine.with_columns(pl.when(pl.col("arxiv_id") == "c3").then(pl.lit("Other"))
                                .otherwise(pl.col("title")).alias("title"))
    with pytest.raises(ValueError, match="não batem"):
        rx.pool_com_campos(val, mexido.lazy(), n=40, semente=3)
    with pytest.raises(ValueError, match="fora do spine"):
        rx.pool_com_campos(val, spine.filter(pl.col("arxiv_id") != "p0").lazy(), n=40,
                           semente=3)


# ── a ordenação, as métricas e a leitura ─────────────────────────────────────


def test_posicoes_cosseno_e_euclidiana_diferem_em_vetores_nao_normalizados():
    va = np.array([[1.0, 0.0], [0.0, 1.0]])
    vp = np.array([[10.0, 1.0], [0.1, 0.1]])
    assert list(rx.posicoes(va, vp, "cosseno")) == [1, 2]
    # Pela distância, o candidato pequeno está mais perto das duas consultas.
    assert list(rx.posicoes(va, vp, "euclidiana")) == [2, 1]
    with pytest.raises(ValueError):
        rx.posicoes(va, vp, "manhattan")


def test_metricas_de_um_relevante_por_consulta():
    m = rx.metricas(np.array([1, 2, 11, 1]))
    assert m["recall_1"] == 0.5 and m["recall_10"] == 0.75
    assert m["ndcg_10"] == round((1 + 1 / np.log2(3) + 0 + 1) / 4, 4)
    assert m["mrr"] == round((1 + 0.5 + 1 / 11 + 1) / 4, 4)


def test_comparar_e_decidir():
    rng = np.random.default_rng(0)
    b = rng.integers(1, 30, 2000)
    melhor = np.maximum(1, b - 5)
    assert rx.comparar(b, b)["leitura"] == "nao_se_distingue"
    r = rx.comparar(melhor, b)
    assert r["leitura"] == "A_acima" and r["ic95"][0] > 0
    assert rx.comparar(b, melhor)["leitura"] == "A_abaixo"
    ok = rx.conferir_reproducao({c: v + 0.0001 for c, v in rx.TABELA_6.items()})
    ruim = rx.conferir_reproducao({c: v + 0.01 for c, v in rx.TABELA_6.items()})
    assert ok["passa"] and not ruim["passa"]
    acima = {"x": {"leitura": "A_acima"}, "y": {"leitura": "A_acima"}}
    assert rx.decidir(ok, acima) == "supera_os_dois"
    assert rx.decidir(ruim, acima) == "invalida"
    assert rx.decidir(ok, {**acima, "y": {"leitura": "A_abaixo"}}) == "abaixo_de_algum"
    assert rx.decidir(ok, {**acima, "y": {"leitura": "nao_se_distingue"}}) == \
        "nao_se_distingue_de_algum"


def test_comparar_por_consulta_so_nas_consultas_comuns():
    a = {("Cite", "q1"): 0.9, ("Cite", "q2"): 0.8, ("CoView", "q1"): 0.7}
    b = {("Cite", "q1"): 0.5, ("Cite", "q2"): 0.4, ("CoRead", "q9"): 0.1}
    r = rx.comparar_por_consulta(a, b)
    assert r["n_consultas"] == 2 and r["A_menos_B"] == 0.4 and r["leitura"] == "A_acima"
    with pytest.raises(ValueError):
        rx.comparar_por_consulta({("x", "1"): 1.0}, {("y", "1"): 1.0})


def test_a_regra_e_proposta_e_o_script_recusa_sem_aceite():
    import avaliar_recuperador_externo as script

    assert rx.REGRA_9_12_ACEITA is False
    with pytest.raises(SystemExit, match="PROPOSTA"):
        script._recusa_sem_aceite()
    assert rx.SCIREPEVAL_COMMIT == "e04594e6401cb68fb4ddf6daa801fb219cccc0bb"
