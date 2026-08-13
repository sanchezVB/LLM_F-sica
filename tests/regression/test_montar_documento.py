"""A fonte tem de ser o que o LaTeX veria, não o conteúdo do tarball.

Regressão de 2026-08-11, achada ao investigar a auditoria do S3b. A versão
anterior (`juntar_fontes`) concatenava TODOS os `.tex` do pacote de submissão.
Duas consequências, ambas inflando a contagem da fonte:

**Arquivo não incluído.** Rascunho, versão anterior, seção cortada, resposta a
referee — tudo isso viaja no tarball e o documento não inclui. Cada equação ali
aparecia como "equação que o RedPajama perdeu".

**Texto depois de `\\end{document}`.** O LaTeX para de ler ali; eu continuava.

O viés é ASSIMÉTRICO, e é isso que o torna perigoso: inflar a fonte **aumenta** a
degradação medida, empurrando na direção de gastar US$ 100–180 do bulk pago do
arXiv. Um erro que puxasse para o outro lado custaria uma decisão conservadora;
este custa dinheiro.

Medido: o paper 1607.04847 declarava 1.427 equações na fonte contra 322 no
RedPajama.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.core.latex.extrair import (  # noqa: E402
    extrair_equacoes,
    juntar_fontes,
    montar_documento,
)

PRINCIPAL = r"""
\documentclass{article}
\begin{document}
\input{secoes/intro}
\begin{equation} E = m c^2 \end{equation}
\end{document}
"""


# ─── o que o documento inclui entra ──────────────────────────────────────────

def test_segue_input():
    d = montar_documento({
        "main.tex": PRINCIPAL,
        "secoes/intro.tex": r"\begin{equation} F = m a \end{equation}",
    })
    assert d.modo == "seguido"
    assert d.principal == "main.tex"
    assert "secoes/intro.tex" in d.incluidos
    eqs = extrair_equacoes(d.texto)
    assert any("F = m a" in e for e in eqs), "a equação do incluído tem de entrar"
    assert any("E = m c^2" in e for e in eqs)


def test_input_sem_chaves():
    """TeX aceita `\\input arquivo` terminado por espaço."""
    d = montar_documento({
        "main.tex": "\\documentclass{article}\n\\begin{document}\n"
                    "\\input secoes/intro\n\\end{document}",
        "secoes/intro.tex": r"\begin{equation} F = m a \end{equation}",
    })
    assert "secoes/intro.tex" in d.incluidos


def test_inclusao_recursiva():
    d = montar_documento({
        "main.tex": "\\documentclass{a}\\begin{document}\\input{um}\\end{document}",
        "um.tex": r"\input{dois} \begin{equation} a = b \end{equation}",
        "dois.tex": r"\begin{equation} c = d \end{equation}",
    })
    assert set(d.incluidos) == {"um.tex", "dois.tex"}
    eqs = extrair_equacoes(d.texto)
    assert any("c = d" in e for e in eqs)


def test_subfile_tambem_e_seguido():
    d = montar_documento({
        "main.tex": "\\documentclass{a}\\begin{document}\\subfile{cap1}\\end{document}",
        "cap1.tex": r"\begin{equation} x = y \end{equation}",
    })
    assert "cap1.tex" in d.incluidos


# ─── o que o documento NÃO inclui fica fora ──────────────────────────────────

def test_tex_nao_incluido_e_ignorado():
    """O defeito central: rascunho no tarball inflava a fonte."""
    d = montar_documento({
        "main.tex": PRINCIPAL,
        "secoes/intro.tex": r"\begin{equation} F = m a \end{equation}",
        "rascunho_velho.tex": r"\begin{equation} \Lambda = 42 \alpha \end{equation}",
    })
    assert d.ignorados == ["rascunho_velho.tex"]
    assert all(r"\Lambda = 42" not in e for e in extrair_equacoes(d.texto)), \
        "equação de arquivo não incluído entrou na fonte"


def test_corta_depois_do_end_document():
    d = montar_documento({"main.tex": r"""
\documentclass{article}
\begin{document}
\begin{equation} E = m c^2 \end{equation}
\end{document}
\begin{equation} \Xi = \text{rascunho abandonado} \end{equation}
"""})
    eqs = extrair_equacoes(d.texto)
    assert any("E = m c^2" in e for e in eqs)
    assert all(r"\Xi" not in e for e in eqs), "texto após \\end{document} entrou"


def test_input_depois_do_end_document_nao_conta():
    """Ordem importa: expandir primeiro, cortar depois."""
    d = montar_documento({
        "main.tex": "\\documentclass{a}\\begin{document}\\input{bom}\n"
                    "\\end{document}\n\\input{ruim}",
        "bom.tex": r"\begin{equation} a = b \end{equation}",
        "ruim.tex": r"\begin{equation} \Theta = 99 \end{equation}",
    })
    eqs = extrair_equacoes(d.texto)
    assert any("a = b" in e for e in eqs)
    assert all(r"\Theta" not in e for e in eqs)


def test_input_comentado_nao_e_seguido():
    d = montar_documento({
        "main.tex": "\\documentclass{a}\\begin{document}\n"
                    "% \\input{descartado}\n\\end{document}",
        "descartado.tex": r"\begin{equation} \Omega = 7 \beta \end{equation}",
    })
    assert "descartado.tex" not in d.incluidos
    assert all(r"\Omega" not in e for e in extrair_equacoes(d.texto))


# ─── quando não há como saber, dizer que não há ──────────────────────────────

def test_sem_documentclass_cai_na_concatenacao_e_marca():
    """Sem `\\documentclass` não se sabe o que o documento inclui.

    Concatenar é o único recurso — mas o modo tem de dizer isso, porque a
    contagem passa a ser cota superior e não pode entrar na mesma média.
    """
    d = montar_documento({
        "a.tex": r"\begin{equation} x = 1 + y \end{equation}",
        "b.tex": r"\begin{equation} z = 2 + w \end{equation}",
    })
    assert d.modo == "concatenado"
    assert d.confiavel is False
    assert len(extrair_equacoes(d.texto)) == 2, "concatenação ainda colhe tudo"


def test_varios_documentclass_marca_ambiguidade():
    """Versão alternativa no pacote: escolhe, mas registra que escolheu."""
    def corpo(eq: str) -> str:
        # Sem `.format()`: o próprio LaTeX é cheio de chaves e `{article}` viraria
        # campo de formatação.
        return ("\\documentclass{article}\n\\begin{document}\n"
                + eq + "\n\\end{document}")

    d = montar_documento({
        "main.tex": corpo(r"\begin{equation} a = b + c \end{equation}"),
        "versao_antiga.tex": corpo(r"\begin{equation} d = e + f \end{equation}"),
    })
    assert d.modo == "seguido-ambiguo"
    assert d.principal == "main.tex", "devia preferir o nome provável"
    assert d.confiavel is True, "ambíguo ainda é melhor que concatenado"


def test_ciclo_nao_trava():
    d = montar_documento({
        "main.tex": "\\documentclass{a}\\begin{document}\\input{um}\\end{document}",
        "um.tex": r"\input{dois} \begin{equation} a = b \end{equation}",
        "dois.tex": r"\input{um} \begin{equation} c = d \end{equation}",
    })
    assert any("c = d" in e for e in extrair_equacoes(d.texto))


def test_input_ausente_e_registrado_nao_silenciado():
    d = montar_documento({
        "main.tex": "\\documentclass{a}\\begin{document}\\input{nao_existe}\\end{document}",
    })
    assert "nao_existe" in d.faltantes


def test_sty_ausente_nao_conta_como_faltante():
    """`\\input{revtex.sty}` ausente é normal; `.tex` ausente não é."""
    d = montar_documento({
        "main.tex": "\\documentclass{a}\\begin{document}"
                    "\\input{estilo.sty}\\end{document}",
    })
    assert d.faltantes == []


# ─── o viés que isto corrige, medido no próprio teste ────────────────────────

def test_o_vies_e_assimetrico_e_infla_a_degradacao():
    """Reproduz o cenário real: RedPajama preservou TUDO, e a fonte suja acusa perda.

    É o mecanismo exato pelo qual eu chegaria a recomendar um gasto de US$ 180.
    """
    arquivos = {
        "main.tex": "\\documentclass{a}\\begin{document}\n"
                    r"\begin{equation} E = m c^2 \end{equation}" "\n\\end{document}",
        "rascunho.tex": "\n".join(
            rf"\begin{{equation}} q_{i} = {i} \alpha \end{{equation}}" for i in range(9)),
    }
    # O RedPajama tem a única equação que o documento realmente publica.
    rp = r"\begin{equation} E = m c^2 \end{equation}"

    sujo = len(extrair_equacoes(juntar_fontes({**arquivos})))
    limpo = len(extrair_equacoes(montar_documento(arquivos).texto))
    n_rp = len(extrair_equacoes(rp))

    assert limpo == n_rp == 1
    assert sujo == 1, "juntar_fontes agora delega a montar_documento"

    # E se a montagem cair na concatenação (sem \documentclass), o modo avisa.
    sem_classe = {k: v.replace("\\documentclass{a}", "") for k, v in arquivos.items()}
    d = montar_documento(sem_classe)
    assert d.modo == "concatenado"
    assert len(extrair_equacoes(d.texto)) == 10, (
        "sem o principal a fonte infla 10x — e é por isso que o modo importa")
