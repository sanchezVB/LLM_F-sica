"""Quanto do peS2o de Física já está no índice: a normalização, as duas chaves, a conta e
o script ponta a ponta — com dados de mentira."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.corpus import pes2o_novos as pn  # noqa: E402

RESUMO = ("We measure the decay rate of the helium three nucleus in a cold trap and compare "
          "the result with the prediction of the shell model, finding agreement within two "
          "standard deviations over the whole range of temperatures that we explored here.")


@pytest.mark.parametrize("arxiv,s2", [
    ("$\\alpha$-decay of $^{3}$He in a trap", "α-decay of ³He in a trap"),
    ("The $\\lambda$ point of liquid helium", "The λ point of liquid helium"),
    ("Schrödinger cats in \\mathrm{Rb} gases", "Schrodinger Cats in Rb Gases"),
    ("$\\varepsilon$-expansion near four dimensions", "ε-expansion near four dimensions"),
    ("Spin  waves:\n a review", "Spin waves - a review"),
])
def test_a_mesma_grafia_para_latex_e_unicode(arxiv, s2):
    assert pn.chave_do_titulo(arxiv) == pn.chave_do_titulo(s2) is not None


def test_titulo_curto_nao_casa_e_resumo_curto_nao_tem_chave():
    assert pn.chave_do_titulo("Reply to comment") is None
    assert pn.chave_do_titulo("Introduction") is None
    assert pn.chave_do_resumo("short abstract") is None
    assert pn.chave_do_resumo(RESUMO) == pn.chave_do_resumo(RESUMO + " Extra sentence.")


def test_separar_o_formato_do_s2ag():
    assert pn.separar(f"A title here\n\n{RESUMO}") == ("A title here", RESUMO)
    assert pn.separar(f"  A title here \n{RESUMO}") == ("A title here", RESUMO)
    assert pn.separar(RESUMO) is None                         # sem quebra de linha
    assert pn.separar(f"\n\n{RESUMO}") is None                # título vazio
    assert pn.separar(f"{'x' * 401}\n{RESUMO}") is None       # título longo demais
    assert pn.separar("A title here\nshort") is None          # resumo curto


def _arxiv():
    a = pn.Arxiv()
    a.acrescentar("2101.00001", "$\\alpha$-decay of $^{3}$He in a trap", RESUMO)
    a.acrescentar("2101.00002", "Topological phases of cold atoms", "Other text " * 20)
    return a


def test_casar_pelo_titulo_ou_so_pelo_resumo():
    a = _arxiv()
    assert a.casar("α-decay of ³He in a trap", "anything else") == ("2101.00001", "titulo")
    assert a.casar("A completely different journal title", RESUMO) == ("2101.00001", "resumo")
    assert a.casar("Reply", RESUMO) == ("2101.00001", "resumo")
    assert a.casar("Brand new paper on superconductivity", "New " * 40) == (None, None)


def test_a_conta_registro_a_registro():
    c = pn.Contagem(_arxiv())
    novo = "Brand new paper on superconductivity\n\n" + "Fresh words about pairing " * 10
    for k, texto in enumerate([
            "α-decay of ³He in a trap\n\n" + RESUMO,          # casa pelo título
            "Journal version with new title\n\n" + RESUMO,    # casa pelo resumo
            novo, novo,                                        # novo e o repetido
            "Erratum\n\n" + "A correction of the table " * 5,  # novo, título curto
            "no newline at all"]):                             # fora do formato
        c.ver(texto, "part-00000.parquet", k)
    n = c.n
    assert (n["registros"], n["fora_do_formato"], n["no_arxiv_pelo_titulo"],
            n["no_arxiv_so_pelo_resumo"], n["novos"], n["novos_com_titulo_curto"],
            n["repetidos_entre_os_novos"]) == (6, 1, 1, 1, 2, 1, 1)
    novos = c.tirar_novos()
    assert [x["linha"] for x in novos] == [2, 4] and c.tirar_novos() == []
    assert novos[0]["sha256"] == pn.sha_do_texto(novo)
    r = c.resumo()
    assert r["no_arxiv"] == 2 and r["taxa_no_arxiv"] == 0.4 and r["plausivel"]
    assert len(c.exemplos_casados.itens) == 2 and len(c.exemplos_novos.itens) == 2


def test_taxa_implausivel_vira_aviso():
    c = pn.Contagem(pn.Arxiv())
    for k in range(30):
        c.ver(f"Some unrelated title number {k}\n\n" + f"Text {k} " * 30, "p", k)
    r = c.resumo()
    assert not r["plausivel"] and "separação" in r["aviso"]


def test_reservatorio_tem_tamanho_fixo_e_e_reprodutivel():
    a, b = pn._Reservatorio(5, 1), pn._Reservatorio(5, 1)
    for i in range(1000):
        a.ver({"i": i})
        b.ver({"i": i})
    assert len(a.itens) == 5 and a.itens == b.itens and a.vistos == 1000


def test_o_script_ponta_a_ponta(tmp_path):
    import contar_pes2o_fora_do_arxiv as script

    (tmp_path / "indice").mkdir()
    pl.DataFrame({"arxiv_id": ["2101.00001"]}).write_parquet(tmp_path / "indice/documentos.parquet")
    pl.DataFrame({"arxiv_id": ["2101.00001", "2101.00002"],
                  "title": ["$\\alpha$-decay of $^{3}$He in a trap",
                            "Topological phases of cold atoms"],
                  "abstract": [RESUMO, "Other text " * 20]}).write_parquet(tmp_path / "spine.parquet")
    pes2o = tmp_path / "pes2o"
    pes2o.mkdir()
    pl.DataFrame({"texto": ["α-decay of ³He in a trap\n\n" + RESUMO,
                            # está no arXiv, mas FORA do índice: conta como novo
                            "Topological phases of cold atoms\n\n" + "Other text " * 20,
                            "Full text paper body " * 50],
                  "url": ["s2ag/train", "s2ag/valid", "s2orc/train"], "score": [0.95] * 3}
                 ).write_parquet(pes2o / "part-00000.parquet")
    pl.DataFrame({"texto": ["Body " * 100], "url": ["s2orc/train"], "score": [0.99]}
                 ).write_parquet(pes2o / "part-00001.parquet")
    r = script.contar(pes2o, tmp_path / "spine.parquet", tmp_path / "indice",
                      tmp_path / "saida", tmp_path / "agregado.json")
    assert r["s2ag"]["registros"] == 2 and r["s2ag"]["no_arxiv"] == 1
    assert r["s2ag"]["novos"] == 1 and r["pes2o"]["partes_so_com_texto_pleno"] == 1
    novos = pl.read_parquet(tmp_path / "saida/novos.parquet")
    assert novos["linha"].to_list() == [1] and novos["parte"].to_list() == ["part-00000.parquet"]
    ag = json.loads((tmp_path / "agregado.json").read_text(encoding="utf-8"))
    assert "Topological" not in json.dumps(ag)              # o agregado vai ao git: só números
