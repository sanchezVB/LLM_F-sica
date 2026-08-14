"""`expected_count` e `actual_count` não estão na mesma unidade — não divida.

Regressão de 2026-08-13, achada ao investigar por que a coleta de `math` deixava
`expected_count` nulo. Os manifestos em disco:

    coletor              expected_count      actual_count
    arXiv OAI                      None         1.595.422
    OpenAlex snapshot       510.372.821         4.613.751

No snapshot, `expected` eram as obras VARRIDAS — o OpenAlex inteiro — e `actual`
são as GUARDADAS, as que casaram com o arXiv. A razão dá **0,9%**, e qualquer
leitor concluiria que a coleta perdeu 99% dos dados. Não perdeu: é a filtragem
funcionando exatamente como projetada.

Dois campos com unidades diferentes convidando à divisão é armadilha, não
documentação. A contagem de varredura passou para `query_spec`, onde ninguém a
confunde com um denominador de completude.

## E não existe verificação de completude

Vale registrar por escrito: nada no código compara `actual_count` com
`expected_count`. O campo serve só ao percentual de progresso. Para o arXiv o
sinal de fim é o do protocolo — `resumptionToken` ausente na última página — e não
uma conferência de contagem. Achar que existe conferência é pior que saber que
não existe.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.core.schema.manifest import (  # noqa: E402
    AcquisitionManifest,
    HarvestMethod,
    LicenseResolution,
    RateLimit,
)


def manifesto(**kw) -> AcquisitionManifest:
    return AcquisitionManifest(
        source_name="teste",
        harvest_method=HarvestMethod.OAI_PMH,
        endpoint="https://exemplo/oai",
        rate_limit=RateLimit(requests_per_second=1.0),
        license_resolution=LicenseResolution(method="per_record",
                                             evidence_url="https://exemplo"),
        **kw,
    )


def test_expected_count_nasce_nulo_e_nao_e_promessa():
    """Nulo é o estado normal, não erro: o arXiv não declara total."""
    m = manifesto()
    assert m.expected_count is None
    assert m.actual_count == 0


def test_nada_no_manifesto_valida_completude():
    """Fixa a ausência, para que ela seja escolha e não descuido.

    Se um dia entrar validação, este teste falha e obriga quem entrar a decidir o
    que fazer com os coletores cuja unidade não bate.
    """
    m = manifesto(expected_count=1_000, actual_count=7)
    # Nenhum validador reclama de 7 contra 1.000.
    assert m.actual_count == 7 and m.expected_count == 1_000
    d = m.model_dump()
    assert "completude" not in d and "complete_ratio" not in d


def test_snapshot_nao_usa_expected_count_para_varredura():
    """A contagem de varredura vai para `query_spec`, longe do denominador.

    Este é o teste que impede a volta do 0,9%: se alguém puser a contagem de
    varredura em `expected_count` de novo, a razão volta a mentir.
    """
    from phifm.corpus.acquire import openalex_snapshot as os_

    fonte = Path(os_.__file__).read_text(encoding="utf-8")
    assert 'm.query_spec["registros_varridos"]' in fonte
    assert "m.expected_count = sum(" not in fonte, (
        "a contagem de varredura voltou para expected_count — a razão "
        "actual/expected passa a ler 0,9% e sugerir coleta falhada")


def test_query_spec_aceita_a_contagem_de_varredura():
    m = manifesto()
    m.query_spec["registros_varridos"] = 510_372_821
    m.actual_count = 4_613_751
    # As duas convivem sem convidar à divisão, porque não são o mesmo par.
    assert m.query_spec["registros_varridos"] > m.actual_count
    assert m.expected_count is None
