"""Expansão de macros do autor — o que separa notação de conteúdo.

Motivação medida em 2026-08-10, arXiv 1607.04520 (49 macros definidas):

    grupo                          n     usam macro
    casaram por forma canônica   571            20%
    NÃO casaram                  239            97%

A auditoria do S3b acusava 19,6% de degradação do RedPajama. Com expansão, o
mesmo dado dá **2,6%** — o resto era a medição confundindo `\\Ecal_\\mu` da fonte
com `\\mathcal{E}_\\mu` do RedPajama. Acusar um terceiro de um defeito nosso é o
pior tipo de erro de medição, porque a conclusão parece uma descoberta.

Regra de projeto que os testes protegem: **caso não tratado fica como está,
nunca é chutado.** Macro não expandida gera falso negativo, que subestima a
preservação. Expandir errado geraria falso POSITIVO, inflando a preservação com
equações que não são as mesmas. Entre subestimar e mentir a favor, subestima.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.core.latex.macros import (  # noqa: E402
    coletar_macros,
    expandir,
    preparar,
    remover_definicoes,
)


def test_macro_simples():
    tex = r"\newcommand{\Ecal}{\mathcal{E}} depois $\Ecal_\mu$"
    assert r"\mathcal{E}_\mu" in preparar(tex)


def test_sem_chaves_no_nome():
    """`\\newcommand\\nome{corpo}` é forma válida e comum."""
    assert r"\mathcal{F}" in expandir(r"\newcommand\Fcal{\mathcal{F}} $\Fcal$")


def test_um_argumento():
    tex = r"\newcommand{\bi}[1]{\boldsymbol{#1}} $\bi{v}$"
    assert r"\boldsymbol{v}" in preparar(tex)


def test_dois_argumentos():
    tex = r"\newcommand{\dd}[2]{\frac{\partial #1}{\partial #2}} $\dd{u}{x}$"
    assert r"\frac{\partial u}{\partial x}" in preparar(tex)


def test_argumento_opcional_com_padrao():
    tex = r"\newcommand{\norm}[2][2]{\|#2\|_{#1}} $\norm{v}$ e $\norm[1]{w}$"
    r = preparar(tex)
    assert r"\|v\|_{2}" in r, "não aplicou o padrão"
    assert r"\|w\|_{1}" in r, "não aplicou o explícito"


def test_corpo_com_chaves_aninhadas():
    """Regex ingênua pararia na primeira `}` interna e truncaria o corpo."""
    tex = r"\newcommand{\Om}{\Omega_{\mathrm{int}}} $\Om$"
    assert r"\Omega_{\mathrm{int}}" in preparar(tex)


def test_macro_que_usa_macro():
    tex = (r"\newcommand{\Ecal}{\mathcal{E}}"
           r"\newcommand{\Efull}{\Ecal_{\mu}} $\Efull$")
    assert r"\mathcal{E}_{\mu}" in preparar(tex)


def test_prefixo_nao_e_confundido():
    """`\\Ecalx` não pode ser lido como `\\Ecal` seguido de `x` — nomes longos
    têm de ser tentados primeiro."""
    tex = (r"\newcommand{\Ecal}{\mathcal{E}}"
           r"\newcommand{\Ecalx}{\mathcal{X}} $\Ecalx$")
    r = preparar(tex)
    assert r"\mathcal{X}" in r
    assert r"\mathcal{E}x" not in r


def test_declaremathoperator():
    tex = r"\DeclareMathOperator{\Prob}{P} $\Prob(A)$"
    assert r"\operatorname{P}(A)" in preparar(tex)


def test_def_simples():
    assert r"\omega" in preparar(r"\def\om{\omega} $\om$")


def test_renewcommand():
    assert r"\mathbb{R}" in expandir(r"\renewcommand{\R}{\mathbb{R}} $\R^n$")


def test_definicao_nao_vira_equacao():
    """`\\newcommand{\\Ecal}{\\mathcal{E}}` no preâmbulo não é equação do artigo.

    Sem `remover_definicoes`, o extrator colheria o corpo da definição como
    equação e inflaria o denominador com coisas que o autor nunca escreveu.
    """
    from phifm.core.latex.extrair import extrair_equacoes

    # `a=b` tem 3 caracteres e cai no corte de `MIN_SIMBOLOS`, que existe para
    # tirar símbolo solto do denominador. Equação real para o teste ser válido.
    tex = r"\newcommand{\Ecal}{\mathcal{E}} texto $a = b + c$ fim"
    eqs = extrair_equacoes(remover_definicoes(tex))
    assert any("a=b+c" in e.replace(" ", "") for e in eqs)
    assert not any("mathcal" in e for e in eqs), "colheu a definição como equação"


def test_argumentos_insuficientes_nao_expande():
    """Macro com 2 argumentos e só 1 disponível fica como está.

    Expandir com argumento faltando produziria `#2` literal no resultado, que é
    uma equação que não existe em lugar nenhum — falso positivo.
    """
    tex = r"\newcommand{\dd}[2]{\frac{#1}{#2}} fim do texto \dd{u}"
    r = preparar(tex)
    assert "#2" not in r, "deixou marcador de argumento no resultado"


def test_recursao_circular_nao_trava():
    """Definição circular para pelo teto de profundidade, sem laço infinito."""
    tex = r"\newcommand{\a}{\b}\newcommand{\b}{\a} $\a$"
    preparar(tex, max_profundidade=4)      # não deve pendurar


def test_sem_macros_devolve_intacto():
    tex = r"$E = mc^2$ e $\nabla \cdot \vec{B} = 0$"
    assert preparar(tex) == tex


def test_coletar_reporta_aridade():
    m = coletar_macros(r"\newcommand{\a}{x}\newcommand{\b}[2]{#1#2}")
    assert m["a"][0] == 0
    assert m["b"][0] == 2


def test_caso_real_do_1607_04520():
    """As macros que apareceram no paper que motivou o módulo."""
    tex = (r"\newcommand{\Ecal}{\mathcal{E}}"
           r"\newcommand{\Mcal}{\mathcal{M}}"
           r"\def\om{\omega}"
           r"\begin{equation} \Ecal_\mu(u) := \frac{1}{2}\int_{\om}|\nabla u|^2 \end{equation}")
    r = preparar(tex)
    assert r"\mathcal{E}_\mu" in r
    assert r"\int_{\omega}" in r
    assert r"\om" not in r.replace(r"\omega", ""), "sobrou macro não expandida"
