"""A amostragem que decide se a métrica do ΦRank mede algo.

Este arquivo existe por causa de um erro que custou uma semana de conclusões
erradas. `avaliar` usava `val.head(500)`, e o parquet de pares vem **agrupado por
documento citado** — então as 500 linhas eram 35 papers repetidos ~14 vezes.

Linhas do mesmo documento não são observações independentes. O n efetivo era 35, o
intervalo de 95% do acerto@1 era ±0,159, e o "ganho" de 0,198 para 0,364 caía
inteiro dentro do ruído. Pior: a divisão contaminada e a divisão honesta
reportaram o MESMO número, porque as duas mediam os mesmos 35 papers — o que fez
parecer que o vazamento não importava.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.training.amostragem import amostrar_por_documento  # noqa: E402


def _agrupado(n_docs: int = 50, por_doc: int = 20) -> pl.DataFrame:
    """Imita a forma real do parquet: agrupado, cada documento em linhas seguidas."""
    return pl.DataFrame({
        "arxiv_citado": [f"doc{d:03d}" for d in range(n_docs) for _ in range(por_doc)],
        "ancora": [f"consulta {d}-{r}" for d in range(n_docs) for r in range(por_doc)],
        "positivo": [f"texto do doc {d}" for d in range(n_docs) for _ in range(por_doc)],
    })


def test_head_cobriria_poucos_documentos_e_sample_cobre_muitos():
    """O teste que impede o `head` de voltar.

    Com 50 documentos de 20 linhas, as primeiras 100 linhas são 5 papers. Um sorteio
    das mesmas 100 linhas cobre uma ordem de magnitude mais.
    """
    d = _agrupado()
    quantos_o_head_cobriria = d.head(100)["arxiv_citado"].n_unique()
    _, n_doc = amostrar_por_documento(d, 100, semente=17)

    assert quantos_o_head_cobriria == 5, "a fixture deixou de imitar o agrupamento"
    assert n_doc > 4 * quantos_o_head_cobriria, (
        f"amostragem cobriu {n_doc} documentos, quase o mesmo que o head "
        f"({quantos_o_head_cobriria}) — voltou a ser `head`?")


def test_devolve_documentos_distintos_e_nao_linhas():
    """O segundo valor dimensiona o intervalo de confiança; se contar linhas, mente."""
    d = _agrupado(n_docs=10, por_doc=30)
    amostra, n_doc = amostrar_por_documento(d, 120, semente=17)
    assert len(amostra) == 120
    assert n_doc <= 10, f"n_doc={n_doc} passou dos 10 documentos existentes — contou linhas"
    assert n_doc == amostra["arxiv_citado"].n_unique()


def test_n_maior_que_o_disponivel_devolve_tudo():
    d = _agrupado(n_docs=3, por_doc=4)
    amostra, n_doc = amostrar_por_documento(d, 9_999, semente=17)
    assert len(amostra) == 12 and n_doc == 3


def test_n_zero_devolve_tudo_para_contar_o_universo():
    """`treinar` usa n=0 só para saber quantos documentos existiam ANTES do corte."""
    d = _agrupado(n_docs=7, por_doc=5)
    amostra, n_doc = amostrar_por_documento(d, 0, semente=17)
    assert len(amostra) == 35 and n_doc == 7


def test_mesma_semente_mesma_amostra():
    """Sem isto, retomar um treino avaliaria noutro conjunto e a curva seria ruído."""
    d = _agrupado()
    a, na = amostrar_por_documento(d, 100, semente=17)
    b, nb = amostrar_por_documento(d, 100, semente=17)
    assert a.equals(b) and na == nb


def test_sementes_diferentes_amostras_diferentes():
    d = _agrupado()
    a, _ = amostrar_por_documento(d, 100, semente=17)
    b, _ = amostrar_por_documento(d, 100, semente=18)
    assert not a.equals(b)


def test_sem_a_coluna_de_documento_cai_para_linhas():
    """Não deve explodir num DataFrame sem `arxiv_citado` — só perde o n efetivo."""
    d = pl.DataFrame({"ancora": ["a", "b", "c"], "positivo": ["x", "y", "z"]})
    amostra, n = amostrar_por_documento(d, 2, semente=17)
    assert len(amostra) == 2 and n == 2


# ─── `amostrar_do_plano`: o corte do TREINO, e os 10,7× que ele custou ───────
#
# O mesmo defeito deste arquivo, no caminho do embedding, onde nunca foi
# consertado. Medido em `pares_treino.parquet` (6.564.111 linhas, 667.304
# documentos citados distintos):
#
#     n=   20.000   head ->    984 docs   sample ->  17.837 docs   18,1x
#     n=  400.000   head -> 17.844 docs   sample -> 191.300 docs   10,7x
#     n=1.500.000   head -> 67.232 docs   sample -> 390.966 docs    5,8x
#
# O run da T1a — 400 mil pares, que produziu o campeão atual do G1 — treinou com
# 17.844 documentos distintos onde o sorteio do MESMO tamanho daria 191.300.


def test_amostrar_do_plano_sorteia_e_nao_pega_o_prefixo(tmp_path):
    """⚠️ A asserção central desta seção. Ver o comentário acima.

    O corte tem de acontecer no PLANO — coletar 6,5 M pares com 1,0 GB de RAM
    livre matava o processo antes do primeiro passo, sem traceback.
    """
    from phifm.training.amostragem import amostrar_do_plano

    # 300 documentos citados, 20 linhas cada, AGRUPADOS: a patologia do parquet.
    linhas = [{"arxiv_id": f"{d}.{k}", "arxiv_citado": f"cit{d:04d}",
               "ancora": f"a{d}-{k}", "positivo": f"p{d}"}
              for d in range(300) for k in range(20)]
    arq = tmp_path / "pares.parquet"
    pl.DataFrame(linhas).write_parquet(arq)

    d, total, n_doc = amostrar_do_plano(pl.scan_parquet(arq), 600, semente=17)
    assert (d.height, total) == (600, 6000)
    # `head(600)` cobriria 30 documentos; o sorteio tem de cobrir muito mais.
    por_head = pl.read_parquet(arq).head(600)["arxiv_citado"].n_unique()
    assert por_head == 30, f"a patologia mudou: head(600) cobre {por_head}"
    assert n_doc > 250, (
        f"o sorteio cobriu {n_doc} documentos de 300 — perto dos {por_head} do "
        "`head`, então ele não está sorteando")


def test_amostrar_do_plano_e_deterministico_e_pede_o_n_exato(tmp_path):
    """Um treino que não é reproduzível não isola variável nenhuma, e um corte
    que devolve menos linhas do que pediu treina sobre outro conjunto."""
    from phifm.training.amostragem import amostrar_do_plano

    arq = tmp_path / "p.parquet"
    pl.DataFrame({"arxiv_id": [str(i) for i in range(1000)],
                  "arxiv_citado": [f"c{i // 5}" for i in range(1000)],
                  "ancora": [f"a{i}" for i in range(1000)],
                  "positivo": [f"p{i // 5}" for i in range(1000)]}
                 ).write_parquet(arq)
    a, _, _ = amostrar_do_plano(pl.scan_parquet(arq), 200, semente=17)
    b, _, _ = amostrar_do_plano(pl.scan_parquet(arq), 200, semente=17)
    assert a["arxiv_id"].to_list() == b["arxiv_id"].to_list()
    c, _, _ = amostrar_do_plano(pl.scan_parquet(arq), 200, semente=99)
    assert c["arxiv_id"].to_list() != a["arxiv_id"].to_list()
    assert a.height == 200

    # n maior que o total devolve tudo, sem levantar: é o caso "sem teto".
    d, total, _ = amostrar_do_plano(pl.scan_parquet(arq), 5000, semente=17)
    assert (d.height, total) == (1000, 1000)


def test_os_pontos_de_corte_do_treino_nao_usam_head():
    """Os quatro lugares que cortavam com `head`, e o preço de cada um.

    O de `rerank.py` já estava consertado — com a explicação escrita — e o
    conserto nunca chegou ao caminho do embedding.
    """
    from conftest import so_codigo_de

    for caminho, agulha in (
        ("src/phifm/training/embedding.py", "treino.head(self.cfg.max_pares)"),
        ("scripts/train_embedding.py", "head(a.max_pares)"),
        ("scripts/empacotar_kaggle.py", "head(a.max_pares)"),
    ):
        fonte = so_codigo_de(RAIZ / caminho)
        assert agulha not in fonte, f"voltou o `head` em {caminho}"
        assert "amostrar_" in fonte, f"{caminho} não sorteia"
