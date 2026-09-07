"""O renderizador de artigo falha em silêncio de duas formas, e as duas aconteceram.

Regressão de 2026-09-07, escrita durante a primeira renderização do
`rascunho-artigo-programa-phifm.md`. Os dois defeitos abaixo são reais e o
segundo é do próprio código que existe para pegar o primeiro.

**1. A ênfase atravessava trechos de código.** A §5.4 do artigo cita a notação
mutilada do peS2o: `p^*` virou `p *`. São dois trechos de código separados, cada
um com um `*`, e o regex de itálico casava do primeiro para o segundo — gerando
`<i>` aberto dentro de um `<font>` e fechado dentro do outro. O reportlab levantou
`saw </font> instead of expected </i>`, o que ao menos é barulhento. O MESMO
casamento entre dois pares de `**` teria produzido negrito em metade de um
parágrafo, sem exceção e sem aviso.

**2. A guarda de glifo reprovava o que existia para isentar.** `EXPOENTES` é um
`str.maketrans`, cujas chaves são ORDINAIS. `'⁰' in EXPOENTES` é sempre falso,
então a guarda acusava `⁰` e `⁴` — os dois únicos caracteres que `_inline` já
converte em `<super>`. Com `--estrito` ela derrubava a renderização do documento
correto.

A terceira coisa que este arquivo tranca não é um defeito passado, é o modo de
falha que o renderizador tem por construção: bloco de código não quebra linha, e
o reportlab escreve para fora da margem **sem erro**. As tabelas de largura fixa
deste projeto vêm do `ESTADO.md` e crescem quando uma coluna é acrescentada.
"""

from __future__ import annotations

import sys
from importlib import util
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

ARTIGO = RAIZ / "docs" / "papers" / "rascunho-artigo-programa-phifm.md"


def _modulo():
    pytest.importorskip("reportlab", reason="o renderizador é o extra `relatorio`")
    spec = util.spec_from_file_location("artigo_pdf", RAIZ / "scripts" / "artigo_pdf.py")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 1. ênfase contra trecho de código ────────────────────────────────────────

def test_italico_nao_atravessa_dois_trechos_de_codigo():
    """O caso literal da §5.4, que quebrou a primeira renderização."""
    saida = _modulo()._inline("No segundo, `p^*` virou `p *` e o índice caiu")
    assert "<i>" not in saida, f"o itálico atravessou os dois trechos: {saida}"
    assert saida.count("<font") == 2, "cada trecho de código continua sendo um font"


def test_negrito_nao_atravessa_dois_trechos_de_codigo():
    """O mesmo casamento com `**`, que NÃO levantaria — sairia errado e calado."""
    saida = _modulo()._inline("compare `a**b` com `c**d` no texto")
    assert "<b>" not in saida, f"o negrito atravessou os dois trechos: {saida}"


def test_a_enfase_de_verdade_continua_funcionando():
    """A proteção não pode ter desligado a marcação — seria consertar apagando."""
    saida = _modulo()._inline("isto é **forte** e isto é *ênfase* e isto é `codigo`")
    assert "<b>forte</b>" in saida
    assert "<i>ênfase</i>" in saida
    assert 'face="Courier"' in saida


# ── 2. a guarda de glifo ─────────────────────────────────────────────────────

def test_expoente_vira_marcacao_e_nao_caixa_preta():
    """`10¹⁰` do resumo: `¹` é Latin-1 e sai, `⁰` é U+2070 e virava retângulo."""
    saida = _modulo()._inline("na ordem de 10¹⁰–10¹¹ tokens")
    assert "<super>10</super>" in saida
    assert "⁰" not in saida, "o expoente sobrou como caractere e sairia sem glifo"


def test_a_guarda_isenta_os_expoentes_que_o_inline_converte():
    """O defeito nº 2: chave de `str.maketrans` é ordinal, não caractere."""
    mod = _modulo()
    mod._conferir_glifos("energia na ordem de 10¹⁰ e 10²⁴", estrito=True)


def test_a_guarda_levanta_no_glifo_que_o_base_14_nao_tem():
    """Sem ela, o emoji de estado do ESTADO.md sai como retângulo preto no PDF."""
    mod = _modulo()
    with pytest.raises(ValueError, match="caixa preta"):
        mod._conferir_glifos("| S1 | 🟢 completo |", estrito=True)


def test_a_guarda_aceita_grego_e_acento():
    """Φ, ρ e a acentuação do português são o corpo do documento, não exceção."""
    mod = _modulo()
    mod._conferir_glifos("O ΦEmb mede ρ_xy com precisão — e a fusão não.", estrito=True)


# ── 3. bloco de código contra a margem ───────────────────────────────────────

def test_bloco_largo_demais_levanta_em_vez_de_sair_cortado():
    mod = _modulo()
    with pytest.raises(ValueError, match="não cabe na margem"):
        mod.bloco_codigo(["x" * 400], estrito=True)


def test_as_tabelas_do_artigo_cabem_na_margem():
    """Ponta a ponta no documento real: `--estrito` é o que o build usa."""
    mod = _modulo()
    largura_max = mod.LARGURA_UTIL - 10 * mod.mm
    dentro, fora = False, []
    for linha in ARTIGO.read_text(encoding="utf-8").splitlines():
        if linha.lstrip().startswith("```"):
            dentro = not dentro
            continue
        if dentro and len(linha) * mod.RAZAO_COURIER * mod.CORPO_CODIGO_MIN > largura_max:
            fora.append(linha)
    assert not fora, f"linhas que não cabem nem no corpo mínimo: {fora}"


# ── 4. o documento inteiro ───────────────────────────────────────────────────

def test_o_artigo_renderiza_inteiro(tmp_path):
    """Nenhum bloco pode sumir: um `.md` que vira PDF de 2 páginas é falha muda."""
    mod = _modulo()
    saida = mod.render(ARTIGO, tmp_path / "artigo.pdf", estrito=True)
    bytes_pdf = saida.read_bytes()
    assert bytes_pdf.startswith(b"%PDF"), "saída não é um PDF"
    paginas = bytes_pdf.count(b"/Type /Page\n") + bytes_pdf.count(b"/Type /Page ")
    assert paginas >= 15, f"o artigo encolheu para {paginas} páginas — algum bloco sumiu"


def test_bloco_desconhecido_vira_paragrafo_e_nao_desaparece():
    """Markdown não suportado precisa APARECER torto, nunca sumir limpo."""
    mod = _modulo()
    st = mod._estilos()
    flow = mod.montar("<div>marcação que ele não entende</div>", st, estrito=False)
    textos = " ".join(getattr(f, "text", "") for f in flow)
    assert "marcação que ele não entende" in textos
