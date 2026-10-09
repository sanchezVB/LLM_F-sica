"""O cortador de trechos (DOC-13 §3): a equação nunca é partida, a seção é fronteira.

`contar` aqui é contagem de palavras — o cortador não conhece tokenizer.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.retrieval.trechos import (  # noqa: E402
    blocos,
    cortar,
    cortes_em_equacao,
    expandir_atalhos,
    frases,
    limpar,
    reparar,
    secoes,
    tipo_de_secao,
)


def contar(textos: list[str]) -> list[int]:
    return [len(t.split()) for t in textos]


def _frases(n: int, palavras: int = 9, marca: str = "p") -> str:
    return " ".join(f"Frase {marca}{i} " + " ".join(["x"] * (palavras - 3)) + "." for i in range(n))


EQ_GRANDE = r"\begin{align} " + " ".join(f"a_{i} &= b_{i} \\\\" for i in range(40)) + r" \end{align}"


def test_uma_equacao_maior_que_o_limite_sai_inteira_e_marcada():
    ts = cortar(f"{_frases(3)}\n\n{EQ_GRANDE}\n\n{_frases(3, marca='q')}", contar, limite=40)
    com_eq = [t for t in ts if t.tem_equacao]
    assert len(com_eq) == 1 and com_eq[0].excede
    assert r"\begin{align}" in com_eq[0].texto and r"\end{align}" in com_eq[0].texto
    assert all(cortes_em_equacao(t.texto) == 0 for t in ts)


def test_a_equacao_vai_com_o_paragrafo_que_a_antecede_quando_os_dois_cabem():
    ts = cortar(_frases(2) + "\n" + r"\begin{equation} E = mc^2 \end{equation}", contar,
                limite=40)
    assert len(ts) == 1 and ts[0].tem_equacao and not ts[0].excede
    assert ts[0].texto.startswith("Frase p0") and ts[0].texto.endswith(r"\end{equation}")


def test_sem_caber_o_trecho_novo_abre_com_a_ultima_frase_do_anterior():
    ts = cortar(_frases(5) + "\n" + r"\begin{equation} E = mc^2 \end{equation}", contar,
                limite=40)
    assert len(ts) == 2
    assert ts[1].texto.startswith("Frase p4") and ts[1].tem_equacao
    assert "Frase p4" in ts[0].texto            # é sobreposição, não mudança de lugar


def test_a_cauda_nao_duplica_um_trecho_que_e_uma_frase_so():
    ts = cortar(_frases(1, palavras=30) + "\n\n" + _frases(1, palavras=30, marca="q"), contar,
                limite=40, sobreposicao=40)
    assert [t.texto.split()[1] for t in ts] == ["p0", "q0"]


def test_nenhum_trecho_atravessa_secao_e_a_subsecao_herda_o_tipo():
    doc = (r"\section{Introduction}" + _frases(2)
           + r"\subsection{Background}" + _frases(2, marca="b")
           + r"\section{Model}" + _frases(2, marca="m")
           + r"\section{Summary and outlook}" + _frases(2, marca="c"))
    ts = cortar(doc, contar, limite=190)
    assert [(t.tipo, t.secao) for t in ts] == [
        ("introducao", "Introduction"), ("introducao", "Introduction"),
        ("corpo", "Model"), ("conclusao", "Summary and outlook")]
    assert "b0" in ts[1].texto and "p0" not in ts[1].texto


def test_tipos_de_secao():
    assert tipo_de_secao("1. Introduction") == "introducao"
    assert tipo_de_secao("Discussion and Conclusions") == "conclusao"
    assert tipo_de_secao("Experimental setup") == "corpo"
    assert tipo_de_secao("Acknowledgments") == "fora"


def test_agradecimentos_nao_viram_trecho():
    doc = r"\section{Results}" + _frases(3) + r"\section*{Acknowledgements}" + _frases(3, marca="g")
    assert all("g0" not in t.texto for t in cortar(doc, contar))


def test_limpar_tira_comentario_citacao_rotulo_e_bibliografia():
    doc = ("Texto~\\cite{a,b} com 5\\% de erro, Eq.~\\eqref{e1}. % um comentário\n"
           "\\label{sec:x} Fim \\textit{em itálico} {\\bf negrito}.\n"
           "\\begin{thebibliography}{9} \\bibitem{a} Fulano \\end{thebibliography}")
    limpo = " ".join(limpar(doc).split())
    assert limpo == "Texto com 5\\% de erro, Eq. . Fim em itálico negrito."


def test_definicao_de_macro_que_sobrou_no_corpo_sai_inteira():
    doc = ("Antes.\n\\newcommand{ \\begin{equation}}{\\begin{equation}} \\def\\na{\\nabla}\n"
           "\\def\\div{div \\;}\nDepois de definir $\\delta$.")
    assert " ".join(limpar(doc).split()) == "Antes. Depois de definir $\\delta$."


def test_quebra_de_linha_com_argumento_nao_abre_equacao():
    assert blocos(r"linha um \\[2ex] linha dois \\[1ex] fim") == [
        (r"linha um \\[2ex] linha dois \\[1ex] fim", False)]
    assert cortes_em_equacao(r"a \\[2ex] b") == 0


def test_da_tabela_fica_a_legenda():
    doc = ("Antes.\n\\begin{table}[b] \\begin{tabular}{|l|c|} \\hline a & 1 \\\\[-2ex] b & 2 "
           "\\end{tabular} \\caption{Valores de {\\it a} e $b$ no equilíbrio.} \\end{table}\nDepois.")
    limpo = limpar(doc)
    assert "tabular" not in limpo and "hline" not in limpo
    assert "Valores de a e $b$ no equilíbrio." in limpo
    assert [b for b, _ in blocos(limpo)] == ["Antes.", "Valores de a e $b$ no equilíbrio.",
                                             "Depois."]


def test_dois_cifroes_colados_nao_sao_equacao_destacada():
    assert blocos(r"temos $a$$b$ e pronto") == [(r"temos $a$$b$ e pronto", False)]
    assert blocos("antes $$x = 1$$ depois") == [("antes", False), ("$$x = 1$$", True),
                                                ("depois", False)]
    assert cortes_em_equacao("fim de uma $$ x = 1") == 1


def test_a_referencia_removida_nao_fabrica_cifrao_duplo():
    assert "$$" not in limpar(r"a solução $\eqref{w3}$. Depois $x$ cresce.")


def test_frase_nao_e_cortada_dentro_de_matematica_em_linha():
    fs = frases(r"Seja $a. B$ o valor. Então segue.")
    assert fs == [r"Seja $a. B$ o valor.", "Então segue."]


def test_um_cifrao_solto_nao_engole_o_paragrafo():
    assert len(frases("Custa $ 5. Depois sobe. E para.")) == 3


def test_frase_maior_que_o_limite_e_cortada_em_palavras():
    ts = cortar(" ".join(["palavra"] * 100) + ".", contar, limite=40, sobreposicao=0)
    assert len(ts) == 3 and max(t.n_tokens for t in ts) <= 40
    assert sum(t.n_tokens for t in ts) == 100


def test_atalho_de_equacao_so_e_expandido_com_prova():
    par = expandir_atalhos(r"Um \be a=b \ee dois \be c=d \ee fim \beta")
    assert par.count(r"\begin{equation}") == 2 and par.count(r"\end{equation}") == 2
    assert r"\beta" in par
    # sem alternância (`\ba` é "a em negrito" aqui), nada muda
    solto = r"o vetor \ba e \ba de novo, \ea et al."
    assert expandir_atalhos(solto) == solto
    # um lado só: sobram dois `\end{eqnarray}` e há exatamente dois `\ba`
    meio = expandir_atalhos(r"x \ba a=b \end{eqnarray} y \ba c=d \end{eqnarray}")
    assert meio.count(r"\begin{eqnarray}") == 2 and r"\ba" not in meio
    # a mistura: `\be … \ee` numa equação, `\be … \end{equation}` na outra
    misto = expandir_atalhos(r"\be a \ee e \be b \end{equation} e \begin{equation} c \ee")
    assert misto.count(r"\begin{equation}") == 3 and misto.count(r"\end{equation}") == 3
    # `\ba` em negrito DENTRO de uma equação: abre depois de abrir, e não é atalho
    negrito = r"\begin{eqnarray} \ba = 1 \end{eqnarray} e \begin{eqnarray} \ba = 2 \end{eqnarray}"
    assert expandir_atalhos(negrito) == negrito


def test_atalho_expandido_vira_bloco_indivisivel():
    doc = _frases(2) + r" \be " + " ".join(["a. B"] * 30) + r" \ee " + _frases(2, marca="q")
    ts = cortar(doc, contar, limite=40)
    assert sum(cortes_em_equacao(t.texto) for t in ts) == 0
    assert any(t.excede and t.texto.count("a. B") == 30 for t in ts)


def test_resto_minusculo_nao_e_devolvido():
    doc = r"\section{A}" + _frases(3) + r"\section{B} \appendix"
    assert [t.secao for t in cortar(doc, contar, minimo=5)] == ["A"]


def test_secoes_sem_comando_nenhum():
    assert secoes("só um parágrafo") == [("", "só um parágrafo")]


def test_o_dano_de_origem_e_reparado_so_com_prova():
    # o RedPajama comeu o caractere antes de cada `%`: `\begin{align}%` → `\begin{align`
    assert reparar("\\begin{align\nE = 1 \\end{align\n") == "\\begin{align}\nE = 1 \\end{align}\n"
    assert reparar(r"\begin{aligned} x \end{aligned}") == r"\begin{aligned} x \end{aligned}"
    assert reparar(r"\begin {equation} x \end {equation}") == r"\begin{equation} x \end{equation}"
    assert reparar("são\n\\\n a = b.\n\\]\nDepois") == "são\n\\[\n a = b.\n\\]\nDepois"
    assert reparar("são\n\\[\n a = b.\n\\\nDepois") == "são\n\\[\n a = b.\n\\]\nDepois"
    solta = "uma\n\\\nbarra sem fecho"
    assert reparar(solta) == solta
