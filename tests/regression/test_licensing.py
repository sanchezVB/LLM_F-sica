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


class TestVersaoETipo:
    """A versão e o tipo fazem parte da licença: o rótulo tem de ser o que está na URL.

    Até 2026-10-01 as regras olhavam só o tipo. Os DIREITOS saíam certos, e por
    isso nada acusava; o IDENTIFICADOR saía errado em 7.631 registros do índice,
    e 1.660 dedicados ao domínio público contavam como não redistribuíveis.
    """

    @pytest.mark.parametrize("url, spdx, particao", [
        ("http://creativecommons.org/licenses/by/3.0/", "CC-BY-3.0", Partition.TRAIN_OPEN),
        ("http://creativecommons.org/licenses/by-nc-sa/3.0/", "CC-BY-NC-SA-3.0", Partition.EVAL_ONLY),
        ("https://creativecommons.org/licenses/by-sa/2.5/", "CC-BY-SA-2.5", Partition.TRAIN_OPEN),
        ("https://creativecommons.org/licenses/by-nc/4.0/", "CC-BY-NC-4.0", Partition.EVAL_ONLY),
        ("http://creativecommons.org/licenses/by/4.0/", "CC-BY-4.0", Partition.TRAIN_OPEN),
    ])
    def test_o_rotulo_carrega_a_versao_da_url(self, url, spdx, particao):
        r = resolve(url)
        assert r.spdx_id == spdx
        assert r.partition is particao
        assert r.license_url.endswith(url.split("creativecommons.org")[1])

    def test_versoes_do_mesmo_tipo_tem_os_mesmos_direitos(self):
        """O ADR-0001 §2 decide por cláusula (BY, SA, NC, ND), não por versão."""
        direitos = lambda r: (r.train_ok, r.redistributable, r.commercial_ok,  # noqa: E731
                              r.attribution_required, r.share_alike, r.non_commercial)
        for tipo in ("by", "by-sa", "by-nc", "by-nc-sa", "by-nc-nd"):
            base = f"https://creativecommons.org/licenses/{tipo}"
            assert direitos(resolve(f"{base}/3.0/")) == direitos(resolve(f"{base}/4.0/"))

    def test_dedicacao_ao_dominio_publico_e_redistribuivel(self):
        """`licenses/publicdomain/` é a ferramenta anterior ao CC0, com os mesmos direitos."""
        r = resolve("http://creativecommons.org/licenses/publicdomain/")
        assert r.spdx_id == "CC-PDDC"
        assert r.partition is Partition.TRAIN_OPEN
        assert r.attribution_required is False

    @pytest.mark.parametrize("url", [
        "https://creativecommons.org/licenses/by-nd-nc/1.0/",   # a ordem da versão 1.0
        "https://creativecommons.org/licenses/by-nc-sa/9.9/",   # versão que não existe
        "https://creativecommons.org/licenses/by-nc-sa/",       # sem versão
        "https://creativecommons.org/licenses/nc-sampling+/1.0/",
    ])
    def test_nc_fora_do_catalogo_continua_fora_do_treino(self, url):
        """O caminho de fuga: `UNKNOWN` treina, e NC não pode cair nele."""
        r = resolve(url)
        assert r.train_ok is False and r.partition is Partition.EVAL_ONLY
        assert r.spdx_id == "NOASSERTION", "sem tipo e versão conhecidos, não se afirma rótulo"

    @pytest.mark.parametrize("url", [
        "https://creativecommons.org/licenses/by/",             # sem versão
        "https://creativecommons.org/licenses/by/9.9/",
        "https://creativecommons.org/licenses/by-nd/4.0/",      # ND sem NC: fora do catálogo
    ])
    def test_cc_aberta_sem_rotulo_afirmavel_nao_e_redistribuida(self, url):
        """Sem identificador não há atribuição correta, e sem atribuição não se publica."""
        r = resolve(url)
        assert r.spdx_id == "NOASSERTION"
        assert r.train_ok is True and r.redistributable is False

    def test_as_dez_licencas_do_indice_tem_identificador(self):
        """A distribuição real, medida no índice em 2026-10-01: nenhuma cai em UNKNOWN."""
        observadas = [
            "http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
            "arXiv-perpetual-nonexclusive",
            "http://creativecommons.org/licenses/by/4.0/",
            "http://creativecommons.org/licenses/by-nc-nd/4.0/",
            "http://creativecommons.org/licenses/by-nc-sa/4.0/",
            "http://creativecommons.org/publicdomain/zero/1.0/",
            "http://creativecommons.org/licenses/by-sa/4.0/",
            "http://creativecommons.org/licenses/by/3.0/",
            "http://creativecommons.org/licenses/by-nc-sa/3.0/",
            "http://creativecommons.org/licenses/publicdomain/",
        ]
        rotulos = [resolve(u).spdx_id for u in observadas]
        assert "NOASSERTION" not in rotulos
        assert len(set(rotulos)) == 9, "as duas grafias da licença padrão do arXiv são uma só"


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
