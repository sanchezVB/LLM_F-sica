"""Suíte golden do canonicalizador LaTeX (DOC-03 §3).

Duas metades, e a segunda é a que protege o corpus:

  §1  variações que DEVEM colapsar   — se não colapsarem, a dedup e a
                                       descontaminação por equação não
                                       funcionam
  §2  distinções que NÃO PODEM       — cada uma é uma normalização que parece
      colapsar                         óbvia e destruiria Física real

Um falso negativo custa um documento duplicado. Um falso positivo destrói
distinção física em **todo** o corpus. Os testes da §2 existem porque essa
assimetria é enorme.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.core.latex.canonical import (  # noqa: E402
    canonicalizar,
    equivalentes,
    hash_canonico,
)


# ══════════════════════════════════════════════════════════════════════════
# §1 — DEVEM colapsar
# ══════════════════════════════════════════════════════════════════════════

COLAPSAM = [
    ("delimitador de modo",       r"$E = mc^2$",              r"\(E = mc^2\)"),
    ("modo display",              r"\[E = mc^2\]",            r"$$E = mc^2$$"),
    ("ambiente equation",         r"\begin{equation}E=mc^2\end{equation}", r"E=mc^2"),
    ("equation estrelado",        r"\begin{equation*}E=mc^2\end{equation*}", r"E=mc^2"),
    ("rótulo",                    r"E=mc^2\label{eq:einstein}", r"E=mc^2"),
    ("espaçamento fino",          r"E = m\,c^2",              r"E = mc^2"),
    ("quad",                      r"E \quad = \quad mc^2",    r"E=mc^2"),
    ("dfrac vs frac",             r"\dfrac{L}{g}",            r"\frac{L}{g}"),
    ("tfrac vs frac",             r"\tfrac{1}{2}mv^2",        r"\frac{1}{2}mv^2"),
    ("left/right",                r"\left(\frac{L}{g}\right)", r"(\frac{L}{g})"),
    ("big",                       r"\bigl(x+y\bigr)",         r"(x+y)"),
    ("chave unitária no expoente", r"x^{2}",                  r"x^2"),
    ("chave unitária no índice",  r"v_{0}",                   r"v_0"),
    ("espaço variável",           r"F  =  m   a",             r"F=ma"),
    ("to vs rightarrow",          r"x \to \infty",            r"x \rightarrow \infty"),
    ("le vs leq",                 r"v \le c",                 r"v \leq c"),
    ("cdot entre escalares",      r"F = m \cdot a",           r"F = m a"),
    ("nonumber",                  r"E=mc^2 \nonumber",        r"E=mc^2"),
]


@pytest.mark.parametrize("nome,a,b", COLAPSAM, ids=[c[0] for c in COLAPSAM])
def test_variacao_notacional_colapsa(nome, a, b):
    assert equivalentes(a, b), (
        f"{nome}: não colapsaram\n  {canonicalizar(a)!r}\n  {canonicalizar(b)!r}"
    )


def test_pendulo_em_quatro_notacoes():
    """O caso que motiva a descontaminação por equação (DOC-04 §6.2)."""
    formas = [
        r"T = 2\pi\sqrt{\frac{L}{g}}",
        r"$T = 2\pi \sqrt{ \frac{L}{g} }$",
        r"\begin{equation} T = 2\pi\sqrt{\dfrac{L}{g}} \label{eq:pend} \end{equation}",
        r"\[ T \, = \, 2\pi\sqrt{\frac{L}{g}} \]",
    ]
    hashes = {hash_canonico(f) for f in formas}
    assert len(hashes) == 1, f"{len(hashes)} formas distintas: {[canonicalizar(f) for f in formas]}"


# ══════════════════════════════════════════════════════════════════════════
# §2 — NÃO PODEM colapsar
# ══════════════════════════════════════════════════════════════════════════

class TestDistincoesPreservadas:
    """Cada teste aqui corresponde a uma linha da lista de rejeitadas do §2
    de `canonical.py`. São normalizações tentadoras que destruiriam Física."""

    def test_epsilon_e_varepsilon_sao_grandezas_diferentes(self):
        """É comum `\\epsilon` ser permissividade e `\\varepsilon` deformação,
        no mesmo paper."""
        assert not equivalentes(r"\epsilon_0", r"\varepsilon_0")

    def test_phi_e_varphi_sao_diferentes(self):
        assert not equivalentes(r"\phi", r"\varphi")

    def test_ordem_de_operandos_e_preservada(self):
        """Operadores não comutam. ÂB̂ ≠ B̂Â, e toda a Mecânica Quântica está
        nessa distinção."""
        assert not equivalentes(r"\hat{A}\hat{B}", r"\hat{B}\hat{A}")

    def test_posicao_de_indices_tensoriais_e_preservada(self):
        """T^μ_ν e T_ν^μ diferem em Relatividade Geral."""
        assert not equivalentes(r"T^{\mu}_{\nu}", r"T_{\nu}^{\mu}")

    def test_ordem_de_indices_e_preservada(self):
        """Resistividade Hall é antissimétrica: ρ_xy = −ρ_yx."""
        assert not equivalentes(r"\rho_{xy}", r"\rho_{yx}")

    def test_vetor_e_escalar_sao_diferentes(self):
        assert not equivalentes(r"\mathbf{B}", r"B")
        assert not equivalentes(r"\vec{v}", r"v")

    def test_produto_vetorial_nao_vira_escalar(self):
        """`\\times` entre vetores é produto vetorial; `\\cdot` é escalar.
        São operações que devolvem objetos de tipos diferentes."""
        assert not equivalentes(r"\mathbf{A} \times \mathbf{B}", r"\mathbf{A} \cdot \mathbf{B}")

    def test_cdot_entre_vetores_nao_e_removido(self):
        """Remover o `\\cdot` transformaria produto escalar em produto de
        vetores. Só é seguro remover quando não há marcador vetorial."""
        assert not equivalentes(r"\mathbf{A} \cdot \mathbf{B}", r"\mathbf{A}\mathbf{B}")

    def test_covariante_e_contravariante_sao_diferentes(self):
        assert not equivalentes(r"g_{\mu\nu}", r"g^{\mu\nu}")

    def test_nao_faz_algebra(self):
        """`x+1` e `1+x` são matematicamente iguais e canonicamente distintos,
        de propósito: decidir isso exige um CAS, e o CAS é `verify/symbolic`."""
        assert not equivalentes("x+1", "1+x")

    def test_sinal_e_preservado(self):
        assert not equivalentes(r"-\frac{GMm}{r}", r"\frac{GMm}{r}")

    def test_expoente_e_preservado(self):
        assert not equivalentes(r"\frac{GMm}{r^2}", r"\frac{GMm}{r^3}")


# ══════════════════════════════════════════════════════════════════════════
# Propriedades estruturais
# ══════════════════════════════════════════════════════════════════════════

class TestPropriedades:
    def test_idempotente(self):
        """Sem isso, um índice construído em duas passadas divergiria de si."""
        for e in [r"\begin{equation}\left(\dfrac{L}{g}\right)^{2}\end{equation}",
                  r"$E \, = \, mc^{2}$", r"T^{\mu}_{\nu} = 0"]:
            uma = canonicalizar(e)
            assert canonicalizar(uma) == uma, f"não idempotente: {e!r} → {uma!r}"

    def test_deterministico(self):
        e = r"\begin{align} E &= mc^2 \\ p &= mv \end{align}"
        assert len({hash_canonico(e) for _ in range(5)}) == 1

    def test_vazio_e_seguro(self):
        for e in ["", "   ", "$$", r"\begin{equation}\end{equation}"]:
            assert canonicalizar(e) == "" or canonicalizar(e).strip() == ""

    def test_nao_quebra_em_latex_malformado(self):
        for e in [r"\frac{1}{", r"$x", r"\begin{equation}x", "}{", "\\" * 50]:
            canonicalizar(e)  # não pode levantar exceção

    def test_align_multilinha_nao_cola_tokens(self):
        """`&` e `\\\\` viram separador, não desaparecem — senão `E` e `p`
        colariam num identificador inexistente."""
        c = canonicalizar(r"\begin{align} E &= mc^2 \\ p &= mv \end{align}")
        assert "2p" not in c, f"tokens colados: {c!r}"


def test_hash_e_estavel_entre_execucoes():
    """O hash entra no índice de recuperação e no manifesto de descontaminação.
    Se mudar entre versões, os dois precisam ser reconstruídos."""
    assert hash_canonico(r"T = 2\pi\sqrt{\frac{L}{g}}") == hash_canonico(
        r"\[T = 2\pi\sqrt{\dfrac{L}{g}}\]"
    )
