"""Regressão: o registro de licenças impõe o ADR-0001.

Estes testes não checam formatação — checam **decisões de projeto**. Se um
deles quebrar, ou o ADR mudou (e precisa de um novo ADR registrando a
mudança), ou alguém introduziu um defeito com consequência jurídica.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.core.licensing.registry import (  # noqa: E402
    CATALOG,
    Partition,
    resolve,
    resolve_partition,
)


class TestTresDireitos:
    """ADR-0001 §2: D2 (treinar) e D3 (redistribuir) são independentes."""

    def test_arxiv_padrao_treina_mas_nao_redistribui(self):
        """O caso que decide o tamanho do corpus. Colapsar D2 e D3 aqui
        derrubaria o corpus treinável de ~30 B para ~8 B tokens."""
        r = resolve("http://arxiv.org/licenses/nonexclusive-distrib/1.0/")
        assert r.train_ok is True
        assert r.redistributable is False
        assert r.partition is Partition.TRAIN_ONLY

    def test_cc_by_treina_e_redistribui(self):
        r = resolve("http://creativecommons.org/licenses/by/4.0/")
        assert (r.train_ok, r.redistributable, r.commercial_ok) == (True, True, True)
        assert r.partition is Partition.TRAIN_OPEN

    def test_cc0_e_o_caso_mais_livre(self):
        r = resolve("http://creativecommons.org/publicdomain/zero/1.0/")
        assert r.partition is Partition.TRAIN_OPEN
        assert r.attribution_required is False


class TestClausulaNaoComercial:
    """ADR-0001 §4: sob Q3 (pesos Apache-2.0), conteúdo NC fica fora do treino."""

    @pytest.mark.parametrize("url", [
        "http://creativecommons.org/licenses/by-nc-sa/4.0/",
        "http://creativecommons.org/licenses/by-nc-nd/4.0/",
        "https://creativecommons.org/licenses/by-nc/4.0/",
    ])
    def test_nc_nunca_treina(self, url):
        r = resolve(url)
        assert r.non_commercial is True
        assert r.train_ok is False, "NC no treino é incompatível com pesos Apache-2.0"
        assert r.partition is Partition.EVAL_ONLY

    def test_by_sa_nao_e_confundido_com_by_nc_sa(self):
        """`by-sa` e `by-nc-sa` diferem por duas letras e por uma decisão
        de projeto inteira. Ordem das regras importa."""
        assert resolve("https://creativecommons.org/licenses/by-sa/4.0/").train_ok is True
        assert resolve("https://creativecommons.org/licenses/by-nc-sa/4.0/").train_ok is False

    def test_by_nao_captura_by_nc(self):
        """O padrão de `by` precisa de fronteira, ou engoliria `by-nc-*`."""
        assert resolve("https://creativecommons.org/licenses/by-nc-nd/4.0/").non_commercial is True


class TestObrasSobCopyright:
    """ADR-0001 §5: livros sob copyright ingerem para AVALIAÇÃO, nunca treino."""

    def test_copyright_vai_para_eval_only(self):
        r = CATALOG["COPYRIGHTED"]
        assert r.train_ok is False
        assert r.partition is Partition.EVAL_ONLY


class TestPadroesConservadores:
    def test_licenca_ausente_no_arxiv_e_a_padrao(self):
        """Ausência do campo `<license>` significa licença padrão do arXiv,
        não licença aberta. Presumir o contrário seria erro caro."""
        assert resolve(None).spdx_id == CATALOG["arXiv-1.0"].spdx_id
        assert resolve("").redistributable is False

    def test_licenca_irreconhecivel_e_visivel_e_conservadora(self):
        """A3 exige que a não-resolução seja CONTÁVEL, não um nulo silencioso."""
        r = resolve("https://exemplo.invalido/licenca-nunca-vista")
        assert r.spdx_id == "NOASSERTION"
        assert r.train_ok is True and r.redistributable is False

    def test_resolve_nunca_levanta_excecao(self):
        for entrada in [None, "", "   ", "não é url", "http://", "🙂", "a" * 5000]:
            assert resolve(entrada) is not None


class TestGovernoEDominioPublico:
    def test_obra_do_governo_dos_eua_e_totalmente_livre(self):
        """17 U.S.C. §105 — NASA NTRS, NIST, OSTI. A categoria mais limpa."""
        r = CATALOG["US-PD"]
        assert r.partition is Partition.TRAIN_OPEN
        assert r.commercial_ok is True


def test_toda_licenca_do_catalogo_tem_particao_coerente():
    """Invariante: train_ok=False ⇒ EVAL_ONLY, sempre."""
    for key, r in CATALOG.items():
        if not r.train_ok:
            assert r.partition is Partition.EVAL_ONLY, f"{key} viola o invariante"
        elif r.redistributable:
            assert r.partition is Partition.TRAIN_OPEN, f"{key} viola o invariante"


def test_nc_implica_nao_comercial_e_nao_redistribuivel():
    for key, r in CATALOG.items():
        if r.non_commercial:
            assert not r.commercial_ok and not r.redistributable, f"{key} incoerente"


def test_resolve_partition_devolve_string_do_enum():
    assert resolve_partition("http://creativecommons.org/licenses/by/4.0/") == "train_open"
