"""4.042 papers pré-1998 ficavam fora do treino de subárea.

`normalize/spine.py` já aplica `_LEGADO` desde 2026-08-13 — o defeito era no
ARTEFATO: a `spine.parquet` no disco foi construída antes daquela correção, e nela
os papers de arquivo legado (`chao-dyn`, `solv-int`, `patt-sol`, `mtrl-th`,
`supr-con`, …) têm `subfield = "Outro"`. E `train()` descarta "Outro".

São os papers mais antigos do acervo, justamente os de conteúdo clássico mais
distintivo, e eram 100% invisíveis ao classificador de subárea.

## Por que o conserto é no consumidor

Reconstruir a espinha cascatearia: ela é entrada dos pares de citação (6,5 M
linhas), do próprio classificador e da fatia do RedPajama — o que exigiria rebaixar
81 GB e re-derivar 22 GB de corpus. Para 0,25% da espinha, não se paga.

## O que este conserto NÃO faz, e é o mais importante

Ele **não** normaliza "Outro" em geral. Dos 72.872 papers em "Outro", só 6,2% são
legado; os outros 68.328 têm primária `math.AP`, `cs.LG`, `q-bio.PE` — outra área,
com cross-list de Física. Para esses, "Outro" é o rótulo CERTO, e dar-lhes subárea
de Física seria inventar rótulo. Minha premissa inicial da tarefa era que "Outro"
fosse uma lacuna de rotulagem; o dado mostrou que 93,8% dele é rótulo correto.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.corpus.filter.classifier import recuperar_legado  # noqa: E402


def _df(pares: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame({"primary_category": [c for c, _ in pares],
                         "subfield": [s for _, s in pares],
                         "title": ["t"] * len(pares),
                         "abstract": ["a"] * len(pares)})


def test_legado_de_fisica_sai_de_outro():
    """O caso central: `chao-dyn` é `nlin`, que é Dinâmica Não-Linear."""
    r = recuperar_legado(_df([("chao-dyn", "Outro"), ("solv-int", "Outro"),
                              ("mtrl-th", "Outro"), ("supr-con", "Outro")]))
    assert r["subfield"].to_list() == ["Dinâmica Não-Linear", "Dinâmica Não-Linear",
                                       "Matéria Condensada", "Matéria Condensada"]


def test_nao_fisica_em_outro_permanece_em_outro():
    """⚠️ 93,8% do balde "Outro" é rótulo CERTO, não lacuna.

    Papers de `math.AP`, `cs.LG` e `q-bio.PE` estão na espinha por cross-list de
    Física. Dar-lhes subárea de Física seria inventar rótulo — e o classificador
    treinaria a dizer que análise de EDPs é Matéria Condensada.
    """
    cats = [("math.AP", "Outro"), ("cs.LG", "Outro"), ("q-bio.PE", "Outro"),
            ("eess.IV", "Outro"), ("math.PR", "Outro")]
    r = recuperar_legado(_df(cats))
    assert r["subfield"].to_list() == ["Outro"] * len(cats)


def test_legado_de_matematica_permanece_em_outro():
    """`q-alg`, `alg-geom`, `dg-ga` e `funct-an` são legado de MATEMÁTICA.

    O `_LEGADO` os mapeia para `math.*`, que não tem subárea de Física. Um mapa de
    legado que subisse tudo de "Outro" daria subárea de Física a álgebra quântica.
    """
    r = recuperar_legado(_df([("q-alg", "Outro"), ("alg-geom", "Outro"),
                              ("dg-ga", "Outro"), ("funct-an", "Outro"),
                              ("cmp-lg", "Outro")]))
    assert r["subfield"].to_list() == ["Outro"] * 5


def test_subarea_ja_atribuida_nao_e_sobrescrita():
    """Só sobe de "Outro". Um paper já rotulado não pode ser reclassificado.

    Se a função sobrescrevesse, ela reescreveria a subárea de 1,5 M de papers pela
    primária — apagando o rótulo do autor, que é a única fonte autoritativa de
    subárea (é o argumento que o DOC-02 usa para nem treinar classificador de
    subárea sobre rótulo de terceiro).
    """
    r = recuperar_legado(_df([("chao-dyn", "Astrofísica e Cosmologia"),
                              ("cond-mat.str-el", "Matéria Condensada")]))
    assert r["subfield"].to_list() == ["Astrofísica e Cosmologia",
                                       "Matéria Condensada"]


def test_sem_coluna_de_categoria_e_no_op():
    """Um dataframe sem `primary_category` passa intacto, sem levantar.

    `train()` é chamado com recortes diferentes ao longo do projeto, e derrubar o
    treino por causa de uma coluna ausente numa função de conserto seria trocar um
    problema de 0,25% por um de 100%.
    """
    d = pl.DataFrame({"subfield": ["Outro", "Mecânica Quântica"], "title": ["a", "b"]})
    assert recuperar_legado(d).equals(d)


def test_train_descarta_outro_depois_de_recuperar(monkeypatch):
    """A ordem importa: recuperar ANTES de descartar "Outro".

    Invertida, a função rodaria sobre um dataframe de onde os 4.042 já saíram, e o
    conserto seria um no-op silencioso — passaria em todos os testes acima e não
    mudaria nada em produção.
    """
    import inspect

    from phifm.corpus.filter import classifier

    fonte = inspect.getsource(classifier.train)
    i_rec = fonte.index("recuperar_legado")
    i_desc = fonte.index('!= "Outro"')
    assert i_rec < i_desc, (
        "a recuperação do legado tem de vir ANTES do descarte de \"Outro\", "
        "senão é no-op silencioso")


@pytest.mark.parametrize("legado,esperado", [
    ("chao-dyn", "Dinâmica Não-Linear"),
    ("patt-sol", "Dinâmica Não-Linear"),
    ("adap-org", "Dinâmica Não-Linear"),
    ("comp-gas", "Dinâmica Não-Linear"),
    ("mtrl-th", "Matéria Condensada"),
    ("supr-con", "Matéria Condensada"),
    ("chem-ph", "Física (diversos)"),
    ("atom-ph", "Física (diversos)"),
    ("acc-phys", "Física (diversos)"),
    ("plasm-ph", "Física (diversos)"),
    ("ao-sci", "Física (diversos)"),
    ("bayes-an", "Física (diversos)"),
])
def test_cada_arquivo_legado_de_fisica_tem_destino(legado, esperado):
    """Um por um, porque um mapa parcialmente certo é o pior caso.

    Se `supr-con` fosse para "Dinâmica Não-Linear" em vez de "Matéria Condensada",
    o classificador aprenderia supercondutividade como dinâmica não-linear — e o
    número agregado de "4.042 recuperados" continuaria batendo.
    """
    r = recuperar_legado(_df([(legado, "Outro")]))
    assert r["subfield"][0] == esperado
