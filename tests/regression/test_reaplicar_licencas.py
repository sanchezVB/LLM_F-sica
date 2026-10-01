"""Regressão: reaplicar o registro de licenças troca duas colunas, e só elas.

O registro passou, em 2026-10-01, a ler a versão da licença e a reconhecer a
dedicação ao domínio público da Creative Commons. A tabela mestra gravada antes
ficou com `spdx_id` e `partition` defasados, e reconstruí-la inteira não é opção:
a junção com o OpenAlex não se refaz mais (ver o cabeçalho do script).

O que estes testes guardam é o contrato do atalho: ele não pode tocar em nenhuma
outra coluna, não pode reordenar linhas, tem de ser idempotente, e tem de recusar
um plano que tire registros de `eval_only`.
"""

from __future__ import annotations

from importlib import util
from pathlib import Path

import polars as pl
import pytest

RAIZ = Path(__file__).resolve().parents[2]

# As dez licenças que existem no índice do arXiv, com o rótulo que o registro
# ANTIGO dava a cada uma. As três últimas são as que estavam erradas.
ANTIGO = [
    ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/", "LicenseRef-arXiv-perpetual-nonexclusive", "train_only"),
    ("arXiv-perpetual-nonexclusive", "LicenseRef-arXiv-perpetual-nonexclusive", "train_only"),
    ("http://creativecommons.org/licenses/by/4.0/", "CC-BY-4.0", "train_open"),
    ("http://creativecommons.org/licenses/by-nc-nd/4.0/", "CC-BY-NC-ND-4.0", "eval_only"),
    ("http://creativecommons.org/licenses/by-nc-sa/4.0/", "CC-BY-NC-SA-4.0", "eval_only"),
    ("http://creativecommons.org/publicdomain/zero/1.0/", "CC0-1.0", "train_open"),
    ("http://creativecommons.org/licenses/by-sa/4.0/", "CC-BY-SA-4.0", "train_open"),
    ("http://creativecommons.org/licenses/by/3.0/", "CC-BY-4.0", "train_open"),
    ("http://creativecommons.org/licenses/by-nc-sa/3.0/", "CC-BY-NC-SA-4.0", "eval_only"),
    ("http://creativecommons.org/licenses/publicdomain/", "NOASSERTION", "train_only"),
]


@pytest.fixture(scope="module")
def mod():
    spec = util.spec_from_file_location("reaplicar_licencas",
                                        RAIZ / "scripts" / "reaplicar_licencas.py")
    m = util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _spine(caminho: Path, linhas=ANTIGO, repeticoes: int = 3) -> pl.DataFrame:
    """Uma tabela mestra mínima, com coluna aninhada e nulos, fora de ordem."""
    reg = [
        {"arxiv_id": f"{2000 + i:04d}.{j:05d}", "license": lic,
         "authors": [f"A{i}", f"B{j}"], "abstract": None if j == 0 else f"resumo {i}-{j}",
         "spdx_id": spdx, "partition": parte, "year": 2008 + i}
        for i, (lic, spdx, parte) in enumerate(linhas) for j in range(repeticoes)
    ]
    df = pl.DataFrame(reg).sample(fraction=1.0, shuffle=True, seed=17)
    df.write_parquet(caminho)
    return df


def test_o_plano_acha_exatamente_as_tres_licencas_defasadas(mod, tmp_path):
    p = tmp_path / "spine.parquet"
    _spine(p)
    m = mod.mudancas(mod.plano(p))
    assert sorted(m["license"].to_list()) == [
        "http://creativecommons.org/licenses/by-nc-sa/3.0/",
        "http://creativecommons.org/licenses/by/3.0/",
        "http://creativecommons.org/licenses/publicdomain/",
    ]
    novo = dict(zip(m["license"], zip(m["spdx_id_novo"], m["partition_novo"], strict=True),
                    strict=True))
    assert novo["http://creativecommons.org/licenses/by/3.0/"] == ("CC-BY-3.0", "train_open")
    assert novo["http://creativecommons.org/licenses/by-nc-sa/3.0/"] == (
        "CC-BY-NC-SA-3.0", "eval_only")
    assert novo["http://creativecommons.org/licenses/publicdomain/"] == (
        "CC-PDDC", "train_open")


def test_gravar_troca_as_duas_colunas_e_mais_nada(mod, tmp_path):
    p = tmp_path / "spine.parquet"
    antes = _spine(p)
    mod.gravar(p)
    depois = pl.read_parquet(p)

    outras = [c for c in antes.columns if c not in mod.RECALCULADAS]
    assert depois.columns == antes.columns, "a ordem das colunas mudou"
    assert depois.select(outras).equals(antes.select(outras)), (
        "uma coluna que não é `spdx_id` nem `partition` mudou, ou as linhas foram "
        "reordenadas")
    assert not (tmp_path / "spine.parquet.tmp").exists()

    por_licenca = {r["license"]: (r["spdx_id"], r["partition"])
                   for r in depois.unique("license").iter_rows(named=True)}
    assert por_licenca["http://creativecommons.org/licenses/publicdomain/"] == (
        "CC-PDDC", "train_open")
    assert por_licenca["http://creativecommons.org/licenses/by/3.0/"][0] == "CC-BY-3.0"
    assert "NOASSERTION" not in depois["spdx_id"].to_list(), (
        "toda licença observada no índice do arXiv tem de ter identificador")


def test_licenca_nula_recebe_a_padrao_do_arxiv(mod, tmp_path):
    p = tmp_path / "spine.parquet"
    _spine(p, linhas=[*ANTIGO, (None, "desatualizado", "train_open")])
    mod.gravar(p)
    nulas = pl.read_parquet(p).filter(pl.col("license").is_null())
    assert nulas.height == 3
    assert set(nulas["spdx_id"]) == {"LicenseRef-arXiv-perpetual-nonexclusive"}
    assert set(nulas["partition"]) == {"train_only"}, (
        "ausência de licença no arXiv é a licença padrão, que não redistribui")


def test_e_idempotente(mod, tmp_path):
    p = tmp_path / "spine.parquet"
    _spine(p)
    mod.gravar(p)
    assert mod.mudancas(mod.plano(p)).height == 0
    primeira = pl.read_parquet(p)
    mod.gravar(p)
    assert pl.read_parquet(p).equals(primeira)


def test_recusa_tirar_registros_de_eval_only(mod, tmp_path):
    """Um registro CC-BY marcado `eval_only` não volta ao treino por este caminho.

    Pode ser um rótulo errado; pode ser uma exclusão deliberada que alguém gravou
    na coluna. O script não tem como saber, e a direção — de nunca-treina para
    treina — é a que tem consequência jurídica.
    """
    p = tmp_path / "spine.parquet"
    _spine(p, linhas=[("http://creativecommons.org/licenses/by/4.0/", "CC-BY-4.0", "eval_only")])
    with pytest.raises(SystemExit, match="RECUSADO"):
        mod.recusar_saida_de_eval_only(mod.plano(p))


def test_falha_na_conferencia_nao_troca_o_arquivo(mod, tmp_path, monkeypatch):
    """Se a conferência reprova, o original fica como estava e o temporário some."""
    p = tmp_path / "spine.parquet"
    antes = _spine(p)
    monkeypatch.setattr(mod, "_digesto", lambda c: (0, hash(str(c))))
    with pytest.raises(SystemExit, match="ABORTADO"):
        mod.gravar(p)
    assert pl.read_parquet(p).equals(antes)
    assert not (tmp_path / "spine.parquet.tmp").exists()
