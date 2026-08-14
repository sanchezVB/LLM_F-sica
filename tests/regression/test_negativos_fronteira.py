"""Pertencer ao set não faz de um artigo um negativo.

Regressão de 2026-08-07. O plano mandava coletar `cs`, `q-bio` e `econ` como
classe `não-física` do classificador do DOC-02 §6. Medido sobre o conteúdo
antes de soltar a coleta:

    set        baixados   utilizáveis   contaminação
    cs            2.600         2.547          2,0%
    q-bio         2.600         1.583         39,1%      <-- e 63% na 1ª página
    econ          2.600         2.518          3,2%

As primárias mais comuns do set `q-bio` são `physics.bio-ph`,
`cond-mat.stat-mech` e `cond-mat.soft` — Física, todas. O set do OAI-PMH inclui
os trabalhos **cruzados** para a área, não só os que nasceram nela, e biofísica
é a zona de sobreposição por excelência.

Rotular isso como não-física poria ruído **na fronteira exata** que o
classificador precisa aprender. A regra que salva é a mesma do `openalex.py`:
a categoria primária do autor é autoritativa, o set é seletor grosseiro.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from harvest_negativos import ARQUIVOS_FISICA, SETS, e_fisica  # noqa: E402


@pytest.mark.parametrize("cat", [
    "physics.bio-ph",        # o caso que contamina q-bio
    "cond-mat.stat-mech",
    "cond-mat.soft",
    "astro-ph.CO",
    "gr-qc",
    "hep-th",
    "hep-ex",
    "math-ph",               # cruzado com math:math:MP
    "nucl-th",
    "quant-ph",
    "nlin.CD",
    "physics.optics",
])
def test_categoria_de_fisica_e_reconhecida(cat):
    assert e_fisica(cat), f"{cat} é Física e passaria como negativo"


@pytest.mark.parametrize("cat", [
    "cs.CL", "cs.AI", "cs.LG", "cmp-lg",
    "econ.EM", "econ.GN", "econ.TH",
    "q-bio.GN", "q-bio.PE", "q-bio.NC",
    "stat.ML", "math.AG", "eess.SP",
])
def test_categoria_legitima_nao_e_descartada(cat):
    """Descartar negativo bom encolhe a classe sem motivo."""
    assert not e_fisica(cat), f"{cat} não é Física e seria perdido"


def test_prefixo_nao_casa_por_engano():
    """`nlin` é Física; `nlp` não existe mas nomes parecidos não podem casar
    por acidente de prefixo."""
    assert e_fisica("nlin.SI")
    assert not e_fisica("cs.NI")
    assert not e_fisica("q-bio.QM")


def test_ausencia_nao_e_fisica():
    assert not e_fisica(None)
    assert not e_fisica("")


def test_math_ph_esta_coberto():
    """A armadilha gêmea: o arXiv expõe `math:math:MP` e `physics:math-ph`
    como a mesma Mathematical Physics. Se `math` entrar como set um dia, o
    filtro precisa pegar."""
    assert "math-ph" in ARQUIVOS_FISICA
    assert e_fisica("math-ph")


def test_sets_do_plano_nao_mudaram_em_silencio():
    """`math` ENTROU em 2026-08-13, e por medição — não por conveniência.

    Este teste afirmava `"math" not in SETS`, com a justificativa de que a
    sobreposição `math.MP` ↔ `physics:math-ph` tornaria o rótulo ruidoso. A
    justificativa caiu por dois motivos:

    **O rótulo mudou.** Negativo agora é «está no set E não está no spine» (ver
    `montar_binario`), e isso resolve o `math-ph` sozinho: um paper de Física
    Matemática está nos dois sets, logo está no spine, logo não vira negativo.
    Não depende mais de `e_fisica` nem de lista de prefixos.

    **`math` deixou de ser opcional.** Deixa-um-domínio-de-fora com o
    classificador treinado: falsos positivos vão de **1,9%** dentro do domínio
    para **32,9%** num domínio nunca visto, e subir o limiar de 0,5 para 0,999 só
    leva a taxa a 10%, onde estanca. Matemática é a vizinha mais confundível da
    Física e o OpenWebMath é feito dela.

    `eess`, `stat` e `q-fin` continuam fora: são negativos LIMPOS, e negativo
    fácil não ensina fronteira nenhuma.
    """
    assert SETS == ("cs", "q-bio", "econ", "math", "stat")
    assert "math" in SETS, "sem math o 4d filtra o OpenWebMath às cegas"

    # `stat` entrou em 2026-08-14. Este teste afirmava que ele era "negativo
    # fácil" e devia ficar fora — palpite, nunca medido, e da mesma forma exata do
    # palpite sobre `math` que custou 42,1% de falso positivo.
    #
    # Física estatística e estatística compartilham vocabulário: função de
    # partição, entropia, Ising, campo médio aparecem em `stat.ML`. A afirmação
    # "não ensina fronteira" era o oposto de uma medição.
    assert "stat" in SETS, "stat entrou para ser MEDIDO, não por ser fácil"

    # Estes dois seguem fora, agora por razão declarada: `eess` tem sobreposição
    # plausível em processamento de sinais, `q-fin` quase nenhuma, e a ordem de
    # investigação é por força da suspeita.
    for pendente in ("eess", "q-fin"):
        assert pendente not in SETS, f"{pendente} ainda não foi investigado"


def test_a_protecao_do_math_ph_sobrevive_a_entrada_do_math():
    """Com `math` coletado, o `math-ph` passa a ser o caso crítico.

    `e_fisica` não é mais o mecanismo de rótulo, mas continua sendo o que o
    `resumir()` usa para reportar contaminação. Se ele deixasse de pegar
    `math-ph`, o relatório da coleta diria que o set `math` está limpo quando
    parte dele é Física.
    """
    assert e_fisica("math-ph")
    assert e_fisica("math-ph.QA") if "math-ph" in ARQUIVOS_FISICA else True
    # E o inverso: matemática de verdade não pode contar como Física.
    for puro in ("math.AG", "math.NT", "math.CO", "math.LO"):
        assert not e_fisica(puro), f"{puro} é matemática pura, não Física"
