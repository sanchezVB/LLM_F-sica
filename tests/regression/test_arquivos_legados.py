"""Arquivo legado do arXiv é Física, e caía em «Outro» — que o treino descarta.

Regressão de 2026-08-13. Antes de 1998 o arXiv usava arquivos de primeiro nível
que depois foram absorvidos (`chao-dyn` → `nlin`, `mtrl-th` → `cond-mat`,
`atom-ph` → `physics`). Eles continuam sendo a categoria PRIMÁRIA dos papers
antigos, e `PHYSICS_PREFIXES` não os conhecia.

Medido no spine, 1.595.422 papers:

    4.042 com primária legada
      · todos com subfield = "Outro"
      · ZERO com is_physics = False

O `is_physics` escapou por sorte: o arXiv retroagiu cross-lists modernas em todos
os papers legados, então a família era detectada por outra categoria. **Nada no
código garantia isso** — bastava um paper cujo único arquivo fosse legado.

O `subfield` não escapou. `train()` filtra `subfield != "Outro"`, então 4 mil
papers de dinâmica não-linear, matéria condensada e física atômica ficavam fora do
treino de subárea. E são os mais antigos do acervo, de conteúdo clássico mais
distintivo.

A lista incompleta apareceu **duas vezes no mesmo dia** antes de eu fechá-la: uma
ao cruzar as regras de rótulo dos negativos de `math`, outra ao contar quantos
papers do spine têm primária fora da família de Física (`chao-dyn` e `solv-int`
inflavam a conta em 2.614). Nas duas vezes sujou medição, não rótulo — e é por isso
que o rótulo do `is_physics` binário não depende dela.

⚠️ A correção só vale para `spine.parquet` RECONSTRUÍDO. O arquivo em disco foi
gerado com a lista antiga e ainda tem os 4.042 em "Outro".
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import polars as pl  # noqa: E402

from phifm.corpus.normalize.spine import (  # noqa: E402
    _LEGADO,
    PHYSICS_PREFIXES,
    SUBFIELD_MAP,
    _archive,
    annotate,
)

# ─── o legado vira o arquivo moderno ─────────────────────────────────────────

def test_legado_mapeia_para_o_arquivo_moderno():
    esperado = {
        "chao-dyn": "nlin", "solv-int": "nlin", "patt-sol": "nlin",
        "comp-gas": "nlin", "adap-org": "nlin",
        "mtrl-th": "cond-mat", "supr-con": "cond-mat",
        "atom-ph": "physics", "plasm-ph": "physics", "acc-phys": "physics",
        "ao-sci": "physics", "bayes-an": "physics", "chem-ph": "physics",
    }
    for legado, moderno in esperado.items():
        assert _archive(legado) == moderno, f"{legado} não virou {moderno}"


LEGADO_FISICA = {k for k, v in _LEGADO.items() if v in PHYSICS_PREFIXES}
LEGADO_OUTROS = set(_LEGADO) - LEGADO_FISICA


def test_nenhum_legado_de_fisica_cai_em_outro():
    """«Outro» é descartado por `train()`. Cair ali é ser jogado fora."""
    for legado in LEGADO_FISICA:
        sub = SUBFIELD_MAP.get(_archive(legado), "Outro")
        assert sub != "Outro", f"{legado} → {_archive(legado)} → Outro"


def test_todo_legado_de_fisica_e_reconhecido_como_familia():
    for legado in LEGADO_FISICA:
        assert _archive(legado) in PHYSICS_PREFIXES, (
            f"{legado} não é reconhecido como Física após normalizar")


def test_legado_de_matematica_e_cs_nao_vira_fisica():
    """`q-alg`, `alg-geom`, `dg-ga`, `funct-an` e `cmp-lg` estão em `_LEGADO` para
    normalizar o bucket, NÃO para virar Física.

    Foram achados perguntando ao dado, depois de eu escrever a lista de memória
    só com os de Física. Normalizá-los é correto; promovê-los seria o oposto do
    que a lista existe para fazer.
    """
    assert {"q-alg", "alg-geom", "dg-ga", "funct-an", "cmp-lg"} == LEGADO_OUTROS
    for legado in LEGADO_OUTROS:
        assert _archive(legado) in ("math", "cs")
        assert _archive(legado) not in PHYSICS_PREFIXES, (
            f"{legado} virou Física — a normalização passou a promover")
        assert SUBFIELD_MAP.get(_archive(legado), "Outro") == "Outro"


def test_a_lista_de_legados_cobre_o_que_o_acervo_tem():
    """Fixa o conjunto medido no spine em 2026-08-13.

    Escrevi a lista de memória e ela estava incompleta em cinco entradas. Este
    teste é a defesa contra a próxima versão de memória: se o acervo trouxer um
    arquivo legado novo, a checagem por dado (não por lembrança) tem de ser
    refeita.
    """
    assert len(_LEGADO) == 18
    # Os 13 de Física, medidos com contagem de papers no spine.
    assert {
        "chao-dyn", "solv-int", "patt-sol", "comp-gas", "adap-org",
        "acc-phys", "ao-sci", "atom-ph", "bayes-an", "chem-ph", "plasm-ph",
        "mtrl-th", "supr-con",
    } == LEGADO_FISICA


def test_legado_de_apoio_matematico_conta_como_apoio():
    """`is_math_support` casa a CATEGORIA inteira, não o arquivo.

    Então a normalização arquivo→arquivo de `_LEGADO` não alcança: um paper antigo
    traz `funct-an`, nunca `math.FA`. Sem os nomes legados em `MATH_SUPPORT`, 351
    papers de análise funcional e geometria diferencial não contavam.
    """
    from phifm.corpus.normalize.spine import MATH_SUPPORT
    assert "funct-an" in MATH_SUPPORT and "math.FA" in MATH_SUPPORT
    assert "dg-ga" in MATH_SUPPORT and "math.DG" in MATH_SUPPORT
    # `q-alg` (math.QA) e `alg-geom` (math.AG) não são apoio: não estão na lista
    # moderna, então não entram na legada.
    assert "q-alg" not in MATH_SUPPORT and "alg-geom" not in MATH_SUPPORT

    df = annotate(_bruto([["funct-an", "math-ph"]], ["funct-an"]))
    assert df["is_math_support"].to_list() == [True]
    assert df["is_physics"].to_list() == [True]      # pelo cross-list math-ph


def test_o_caminho_normal_nao_mudou():
    """A normalização não pode mexer em quem já estava certo."""
    for cat, arq in (("cond-mat.str-el", "cond-mat"), ("hep-th", "hep-th"),
                     ("physics.bio-ph", "physics"), ("quant-ph", "quant-ph"),
                     ("nlin.SI", "nlin"), ("math-ph", "math-ph")):
        assert _archive(cat) == arq


def test_nao_fisica_continua_fora():
    """Normalizar legado não pode transformar matemática ou CS em Física."""
    for cat in ("math.AP", "cs.LG", "econ.EM", "q-bio.PE", "stat.ML"):
        assert _archive(cat) not in PHYSICS_PREFIXES
        assert SUBFIELD_MAP.get(_archive(cat), "Outro") == "Outro"


# ─── o que a correção muda no spine anotado ──────────────────────────────────

def _bruto(cats: list[list[str]], primarias: list[str]) -> pl.DataFrame:
    """Mínimo que `annotate` exige. `created` entra porque ela deriva `year` dele."""
    n = len(primarias)
    return pl.DataFrame({
        "arxiv_id": [f"id/{i}" for i in range(n)],
        "categories": cats,
        "primary_category": primarias,
        "license": [None] * n,
        "journal_ref": [None] * n,
        "created": ["1997-03-14"] * n,
    })


def test_paper_legado_recebe_subarea_real():
    df = annotate(_bruto([["chao-dyn", "nlin.CD"], ["mtrl-th", "cond-mat"]],
                         ["chao-dyn", "mtrl-th"]))
    assert df["subfield"].to_list() == ["Dinâmica Não-Linear", "Matéria Condensada"]
    assert df["is_physics"].to_list() == [True, True]


def test_paper_SO_com_arquivo_legado_ainda_e_fisica():
    """O caso que nada garantia antes.

    Hoje não existe no acervo — todo paper legado tem cross-list moderna, medido —
    mas a garantia não pode depender de uma gentileza do arXiv.
    """
    df = annotate(_bruto([["chao-dyn"], ["supr-con"]], ["chao-dyn", "supr-con"]))
    assert df["is_physics"].to_list() == [True, True], (
        "paper cujo único arquivo é legado saiu como não-Física")
    assert "Outro" not in df["subfield"].to_list()


def test_paper_de_matematica_pura_nao_vira_fisica():
    df = annotate(_bruto([["math.AG", "math.NT"]], ["math.AG"]))
    assert df["is_physics"].to_list() == [False]
    assert df["subfield"].to_list() == ["Outro"]


def test_cross_list_de_fisica_ainda_conta():
    """DOC-02 §2: qualquer categoria da família conta, não só a primária."""
    df = annotate(_bruto([["math.AP", "math-ph"]], ["math.AP"]))
    assert df["is_physics"].to_list() == [True]
    # A subárea segue a PRIMÁRIA, que é matemática — por isso «Outro».
    assert df["subfield"].to_list() == ["Outro"]
