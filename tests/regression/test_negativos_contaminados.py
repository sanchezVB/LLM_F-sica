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


# ─── estratificação por domínio ──────────────────────────────────────────────

def test_negativos_sao_estratificados_por_dominio(tmp_path):
    """Amostrar uniformemente dá a proporção do arXiv, e ela é desequilibrada.

    `cs` tem 932 mil negativos utilizáveis contra 16 mil de `econ`. Uniformemente,
    `q-bio` fica com 3% do treino e `econ` com 1% — os domínios seguem quase NÃO
    VISTOS mesmo estando em disco. Medido em 2026-08-13, falso positivo:

                         cs    econ   q-bio    math   revocacao
      proporcional     2,2%    4,9%   18,9%    5,1%       0,952
      estratificado    2,8%    1,3%    4,8%    5,8%       0,944

    Pior caso de 18,9% para 5,8% por 0,8 ponto de revocação. Num FILTRO é o pior
    domínio que determina a contaminação, então o critério é o pior caso.
    """
    spine = tmp_path / "spine.parquet"
    pl.DataFrame({"arxiv_id": ["fis/1"], "title": ["T"],
                  "abstract": ["a" * 40]}).write_parquet(spine)

    negs = tmp_path / "negativos"
    # `grande` tem 200 negativos, `pequeno` tem 10. Sem estratificar, `pequeno`
    # apareceria em ~5% da amostra; estratificando, contribui com tudo o que tem.
    for dom, n in (("grande", 200), ("pequeno", 10)):
        (negs / dom).mkdir(parents=True)
        pl.DataFrame({
            "arxiv_id": [f"{dom}/{i}" for i in range(n)],
            "title": [f"{dom} paper {i}" for i in range(n)],
            "abstract": ["b" * 40] * n,
        }).write_parquet(negs / dom / "000.parquet")

    df = montar_binario(spine, negs, max_por_classe=40)
    neg = df.filter(pl.col("is_physics") == "nao_fisica")["arxiv_id"].to_list()
    g = sum(1 for x in neg if x.startswith("grande/"))
    p = sum(1 for x in neg if x.startswith("pequeno/"))
    # cota = 40 // 2 = 20; `pequeno` só tem 10.
    assert g == 20, f"o domínio grande devia respeitar a cota de 20, veio {g}"
    assert p == 10, f"o domínio pequeno devia contribuir com os 10 que tem, veio {p}"


def test_cota_e_teto_nao_exigencia(tmp_path):
    """Quem tem menos que a cota contribui com o que tem, e o resto NÃO é
    redistribuído — redistribuir recriaria o desequilíbrio que a cota corrige."""
    spine = tmp_path / "spine.parquet"
    pl.DataFrame({"arxiv_id": ["fis/1"], "title": ["T"],
                  "abstract": ["a" * 40]}).write_parquet(spine)
    negs = tmp_path / "negativos"
    for dom, n in (("cheio", 500), ("vazio", 3)):
        (negs / dom).mkdir(parents=True)
        pl.DataFrame({"arxiv_id": [f"{dom}/{i}" for i in range(n)],
                      "title": [f"t{i}" for i in range(n)],
                      "abstract": ["b" * 40] * n}).write_parquet(negs / dom / "000.parquet")

    df = montar_binario(spine, negs, max_por_classe=100)
    neg = df.filter(pl.col("is_physics") == "nao_fisica")["arxiv_id"].to_list()
    assert sum(1 for x in neg if x.startswith("cheio/")) == 50, "cota estourada"
    assert sum(1 for x in neg if x.startswith("vazio/")) == 3
    assert len(neg) == 53, "o déficit foi redistribuído — não devia"
