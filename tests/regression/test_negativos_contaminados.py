"""Negativo não é "veio do conjunto cs/econ/q-bio" — é "não é Física".

Regressão de 2026-08-11. A coleta do S2 buscou negativos nos conjuntos `cs`,
`econ` e `q-bio` do arXiv. Mas conjunto do arXiv é por CATEGORIA, e um paper de
`cs.LG` com cross-list em `quant-ph` está nos dois. Usá-lo como negativo ensina
o classificador a **rejeitar Física**, que é o oposto do que ele existe para
fazer.

Medido nos negativos coletados:

    cs      988.244    5,7% com cross-list de Física
    econ     16.984    5,5%
    q-bio    56.142   32,8%   ← um terço do conjunto

Total: 72.919 de 1.041.652 (7,0%).

## Por que a regra é «fora do spine» e não uma lista de prefixos

Dos 72.919 negativos com cross-list de Física, **exatamente** 72.919 estão no
spine e **zero** ficaram fora. A pertinência ao conjunto do arXiv é o rótulo
autoritativo, e usá-la dispensa manter lista de prefixos — que é onde eu erraria:
a nossa `PHYSICS_PREFIXES` não tem os arquivos legados (`adap-org`, `chao-dyn`,
`patt-sol`, `solv-int`, `acc-phys`, `atom-ph`, `chem-ph`, `plasm-ph`, `supr-con`).

Nesse caso não custou nada, porque o arXiv retroagiu cross-list atual em todos os
~5,5 mil papers legados — medido: ZERO papers com arquivo legado e sem prefixo
atual. Mas a lista continua sendo dívida à espera de um caso novo, e é por isso
que o rótulo não depende dela.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.corpus.filter.classifier import montar_binario  # noqa: E402


def montar_disco(tmp: Path) -> tuple[Path, Path]:
    """Spine e negativos sintéticos em parquet, como o código real os lê."""
    spine = tmp / "spine.parquet"
    pl.DataFrame({
        "arxiv_id": ["fis/1", "fis/2", "cs+quant/9"],
        "title": ["Entanglement entropy", "Neutron stars", "Quantum machine learning"],
        "abstract": ["a" * 40, "b" * 40, "c" * 40],
    }).write_parquet(spine)

    negs = tmp / "negativos"
    (negs / "cs").mkdir(parents=True)
    pl.DataFrame({
        # `cs+quant/9` veio do conjunto cs E está no spine: é Física.
        "arxiv_id": ["cs/1", "cs/2", "cs+quant/9"],
        "title": ["Transformer scaling", "Graph coloring", "Quantum machine learning"],
        "abstract": ["d" * 40, "e" * 40, "c" * 40],
    }).write_parquet(negs / "cs" / "000.parquet")
    return spine, negs


def test_negativo_que_esta_no_spine_e_excluido(tmp_path):
    spine, negs = montar_disco(tmp_path)
    df = montar_binario(spine, negs, max_por_classe=100)

    negativos = set(df.filter(pl.col("is_physics") == "nao_fisica")["arxiv_id"])
    assert "cs+quant/9" not in negativos, (
        "paper com cross-list de Física entrou como negativo — é assim que se "
        "ensina o classificador a rejeitar Física")
    assert negativos == {"cs/1", "cs/2"}


def test_nenhum_id_nas_duas_classes(tmp_path):
    """Um id em ambos os lados é rótulo contraditório: o modelo aprende ruído."""
    spine, negs = montar_disco(tmp_path)
    df = montar_binario(spine, negs, max_por_classe=100)
    f = set(df.filter(pl.col("is_physics") == "fisica")["arxiv_id"])
    n = set(df.filter(pl.col("is_physics") == "nao_fisica")["arxiv_id"])
    assert f & n == set()


def test_positivos_saem_do_spine(tmp_path):
    spine, negs = montar_disco(tmp_path)
    df = montar_binario(spine, negs, max_por_classe=100)
    f = set(df.filter(pl.col("is_physics") == "fisica")["arxiv_id"])
    assert f == {"fis/1", "fis/2", "cs+quant/9"}, (
        "o paper cross-listado é positivo, não descartado")


def test_a_regra_nao_depende_da_lista_de_prefixos(tmp_path):
    """Um arquivo legado que a nossa `PHYSICS_PREFIXES` não conhece.

    `adap-org` não está na lista. Se o rótulo dependesse dela, este paper viraria
    negativo. Como depende do spine, ele é positivo — que é o correto, porque o
    arXiv o serve no conjunto de Física.
    """
    spine = tmp_path / "spine.parquet"
    pl.DataFrame({
        "arxiv_id": ["adap-org/9711001"],
        "title": ["Self-organization in a lattice"],
        "abstract": ["z" * 40],
    }).write_parquet(spine)

    negs = tmp_path / "negativos"
    (negs / "q_bio").mkdir(parents=True)
    pl.DataFrame({
        "arxiv_id": ["adap-org/9711001", "qbio/1"],
        "title": ["Self-organization in a lattice", "Protein folding rates"],
        "abstract": ["z" * 40, "y" * 40],
    }).write_parquet(negs / "q_bio" / "000.parquet")

    df = montar_binario(spine, negs, max_por_classe=100)
    assert set(df.filter(pl.col("is_physics") == "nao_fisica")["arxiv_id"]) == {"qbio/1"}
    assert "adap-org/9711001" in set(df.filter(pl.col("is_physics") == "fisica")["arxiv_id"])


def test_negativos_duplicados_entre_conjuntos_contam_uma_vez(tmp_path):
    """Um paper `cs.IT`+`econ.EM` aparece nas duas coletas."""
    spine = tmp_path / "spine.parquet"
    pl.DataFrame({"arxiv_id": ["fis/1"], "title": ["T"],
                  "abstract": ["a" * 40]}).write_parquet(spine)

    negs = tmp_path / "negativos"
    for conj in ("cs", "econ"):
        (negs / conj).mkdir(parents=True)
        pl.DataFrame({"arxiv_id": ["dup/1"], "title": ["Mechanism design"],
                      "abstract": ["b" * 40]}).write_parquet(negs / conj / "000.parquet")

    df = montar_binario(spine, negs, max_por_classe=100)
    assert df.filter(pl.col("is_physics") == "nao_fisica").height == 1


def test_amostragem_e_deterministica(tmp_path):
    """Duas montagens com a mesma semente dão o mesmo conjunto.

    Sem isto o relatório de uma rodada não fala do dado da outra.
    """
    spine = tmp_path / "spine.parquet"
    pl.DataFrame({
        "arxiv_id": [f"fis/{i}" for i in range(50)],
        "title": [f"T{i}" for i in range(50)],
        "abstract": ["a" * 40] * 50,
    }).write_parquet(spine)
    negs = tmp_path / "negativos"
    (negs / "cs").mkdir(parents=True)
    pl.DataFrame({
        "arxiv_id": [f"cs/{i}" for i in range(50)],
        "title": [f"C{i}" for i in range(50)],
        "abstract": ["b" * 40] * 50,
    }).write_parquet(negs / "cs" / "000.parquet")

    a = montar_binario(spine, negs, max_por_classe=10)
    b = montar_binario(spine, negs, max_por_classe=10)
    assert a["arxiv_id"].to_list() == b["arxiv_id"].to_list()
    assert a.height == 20
