"""Passo 2 do acervo maior: achar o id de cada resumo novo pelo hash do texto, parar quando
achou todos e retomar de onde parou — com um peS2o de mentira em arquivos locais."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.corpus import pes2o_ids as pi  # noqa: E402
from phifm.corpus import pes2o_novos as pn  # noqa: E402


def _original(tmp_path: Path) -> dict[str, Path]:
    """Três arquivos do "peS2o original": os novos estão nos dois primeiros."""
    pasta = tmp_path / "original"
    pasta.mkdir()
    conteudo = {
        "data/v2/a.parquet": [("t1\nresumo um " * 5, "101"), ("outro texto", "102")],
        "data/v2/b.parquet": [("t2\nresumo dois " * 5, "201"), ("t3\nresumo três " * 5, "202")],
        "data/v2/c.parquet": [("nada aqui", "301")],
    }
    caminhos = {}
    for nome, linhas in conteudo.items():
        arq = pasta / Path(nome).name
        pl.DataFrame({"id": [i for _, i in linhas], "text": [t for t, _ in linhas],
                      "created": ["2019-01-01"] * len(linhas),
                      "source": ["s2ag/train"] * len(linhas)}).write_parquet(arq)
        caminhos[nome] = arq
    return caminhos


def _novos(tmp_path: Path, textos: list[str]) -> Path:
    arq = tmp_path / "novos.parquet"
    pl.DataFrame({"sha256": [pn.sha_do_texto(t) for t in textos]}).write_parquet(arq)
    return arq


def test_procurar_acha_pelo_hash_e_traz_id_data_e_origem():
    df = pl.DataFrame({"id": [7, 8], "text": ["alvo", "outro"], "created": ["2020", "2021"],
                       "source": ["s2ag/train", "s2orc/train"]})
    r = pi.procurar(df, pi.alvos([pn.sha_do_texto("alvo")]))
    assert r == [{"sha256": pn.sha_do_texto("alvo"), "s2_id": "7", "data": "2020",
                  "origem": "s2ag/train"}]
    with pytest.raises(ValueError, match="nenhuma coluna"):
        pi.procurar(pl.DataFrame({"text": ["x"]}), set())


def test_para_quando_acha_todos_e_retoma(tmp_path):
    import recuperar_ids_pes2o as script

    orig = _original(tmp_path)
    novos = _novos(tmp_path, ["t1\nresumo um " * 5, "t2\nresumo dois " * 5,
                              "t3\nresumo três " * 5])
    baixados = []

    def baixar(nome):
        baixados.append(nome)
        return orig[nome]

    def ler(local):
        yield pl.read_parquet(local)

    saida, agregado = tmp_path / "ids", tmp_path / "pes2o_ids.json"
    # Um arquivo só: acha um, faltam dois.
    r = script.executar(novos, "abc123", sorted(orig), baixar, ler, saida, agregado,
                        max_arquivos=1)
    assert (r["com_id"], r["sem_id"], r["completo"]) == (1, 2, False)
    # Retoma: não baixa de novo o primeiro, e para antes do terceiro.
    r = script.executar(novos, "abc123", sorted(orig), baixar, ler, saida, agregado)
    assert baixados == ["data/v2/a.parquet", "data/v2/b.parquet"]
    assert (r["com_id"], r["sem_id"], r["completo"], r["arquivos_na_colecao"]) == (3, 0, True, 3)
    ids = pl.concat([pl.read_parquet(p) for p in sorted(saida.glob("ids-*.parquet"))])
    assert sorted(ids["s2_id"].to_list()) == ["101", "201", "202"]
    ag = json.loads(agregado.read_text(encoding="utf-8"))
    assert ag["revisao"] == "abc123" and "resumo dois" not in json.dumps(ag, ensure_ascii=False)


def test_recusa_retomar_de_outra_revisao(tmp_path):
    import recuperar_ids_pes2o as script

    orig = _original(tmp_path)
    novos = _novos(tmp_path, ["t1\nresumo um " * 5])
    saida = tmp_path / "ids"
    script.executar(novos, "rev1", sorted(orig), orig.get, lambda p: [pl.read_parquet(p)],
                    saida, tmp_path / "a.json", max_arquivos=1)
    with pytest.raises(SystemExit, match="outra|revisão"):
        script.executar(novos, "rev2", sorted(orig), orig.get, lambda p: [pl.read_parquet(p)],
                        saida, tmp_path / "a.json")
