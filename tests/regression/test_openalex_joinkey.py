"""Regressão: extração da chave de junção do OpenAlex.

Sem `arxiv_id`, o grafo de citações não casa com a espinha do arXiv e todo o
valor do OpenAlex se perde. Dois defeitos reais encontrados em 2026-08-03:

  1. `ids.arxiv` NÃO EXISTE no OpenAlex. Confiar nele deu 0% de cobertura.
  2. Extrair só do DOI dá 1,5%: quando o paper é publicado, `doi` vira o DOI
     da editora, não o `10.48550/arXiv.*`. É preciso ler `locations`.

E um terceiro, silencioso: sem normalizar o sufixo de versão, `2405.12345v2`
vira chave distinta de `2405.12345` e a junção falha numa fração dos
registros sem que a contagem pareça errada.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.corpus.acquire.openalex import extract_arxiv_id, parse_work  # noqa: E402


def loc(url: str) -> dict:
    return {"landing_page_url": url, "source": {"display_name": "arXiv (Cornell University)"}}


def test_extrai_do_doi_datacite():
    assert extract_arxiv_id({"doi": "https://doi.org/10.48550/arXiv.1412.6980"}) == "1412.6980"


def test_extrai_de_locations_quando_doi_e_da_editora():
    """O caso que mais importa: paper publicado em revista."""
    w = {
        "doi": "https://doi.org/10.1103/PhysRevD.106.063007",  # DOI da APS, não do arXiv
        "locations": [
            {"landing_page_url": "https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.063007"},
            loc("http://arxiv.org/abs/2203.07905"),
        ],
    }
    assert extract_arxiv_id(w) == "2203.07905"


def test_normaliza_sufixo_de_versao():
    """`v5` precisa sair, ou a junção falha em silêncio."""
    assert extract_arxiv_id({"locations": [loc("https://arxiv.org/abs/1412.6980v5")]}) == "1412.6980"
    assert extract_arxiv_id({"doi": "10.48550/arxiv.2405.12345v2"}) == "2405.12345"


def test_aceita_identificadores_antigos_com_barra():
    """IDs pré-2007 têm a forma `hep-th/9711200`."""
    assert extract_arxiv_id({"locations": [loc("http://arxiv.org/abs/hep-th/9711200")]}) == "hep-th/9711200"


def test_retorna_none_sem_arxiv():
    assert extract_arxiv_id({"doi": "https://doi.org/10.1038/nature12373"}) is None
    assert extract_arxiv_id({}) is None
    assert extract_arxiv_id({"locations": [{"landing_page_url": None}]}) is None


def test_ids_arxiv_nao_e_usado():
    """Guarda contra a regressão nº 1: `ids.arxiv` não existe e não deve ser lido."""
    w = {"ids": {"arxiv": "9999.99999"}, "locations": [loc("https://arxiv.org/abs/2101.00001")]}
    assert extract_arxiv_id(w) == "2101.00001"


def test_parse_work_encurta_ids_e_conta_arestas():
    w = {
        "id": "https://openalex.org/W123",
        "doi": "https://doi.org/10.48550/arXiv.2301.00001",
        "title": "T",
        "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
        "open_access": {"is_oa": True, "oa_status": "green"},
        "primary_topic": {"display_name": "Quantum Optics", "field": {"display_name": "Physics and Astronomy"}},
    }
    p = parse_work(w)
    assert p["openalex_id"] == "W123"
    assert p["arxiv_id"] == "2301.00001"
    assert p["doi"] == "10.48550/arXiv.2301.00001"
    assert p["referenced_works"] == ["W1", "W2"], "IDs devem ser encurtados: 50 M arestas × 25 bytes de URL é desperdício"
    assert p["n_references"] == 2
    assert p["openalex_field"] == "Physics and Astronomy"
