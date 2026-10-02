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
    # `cast` explícito: uma coluna toda nula nasceria com tipo Null, e a tabela real
    # tem String com nulos — o esquema é uma das conferências do script.
    df = (pl.DataFrame(reg)
          .with_columns(pl.col("license", "spdx_id", "partition").cast(pl.Utf8))
          .sample(fraction=1.0, shuffle=True, seed=17))
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


@pytest.mark.parametrize("particao", ["eval_only", "excluded", "EVAL_ONLY", " eval_only"])
def test_recusa_levar_ao_treino_o_que_esta_fora_dele(mod, tmp_path, particao):
    """Um registro CC-BY marcado como fora do treino não volta a ele por este caminho.

    Pode ser um rótulo errado; pode ser uma exclusão deliberada que alguém gravou
    na coluna. O script não tem como saber, e a direção — de não-treina para
    treina — é a que tem consequência jurídica. A guarda é pelo destino: a primeira
    versão só conhecia a string `eval_only`, e `excluded` passava.
    """
    p = tmp_path / "spine.parquet"
    _spine(p, linhas=[("http://creativecommons.org/licenses/by/4.0/", "CC-BY-4.0", particao)])
    antes = p.read_bytes()
    with pytest.raises(SystemExit, match="RECUSADO"):
        mod.gravar(p)
    assert p.read_bytes() == antes
    assert not (tmp_path / "spine.parquet.tmp").exists()


def test_a_recusa_vale_pela_linha_de_comando(mod, tmp_path, monkeypatch):
    """`main --gravar` é o caminho que as pessoas usam, e era o que não tinha teste."""
    p = tmp_path / "spine.parquet"
    _spine(p, linhas=[*ANTIGO,
                      ("http://creativecommons.org/licenses/by/4.0/", "CC-BY-4.0", "eval_only")])
    antes = p.read_bytes()
    monkeypatch.setattr("sys.argv", ["reaplicar_licencas.py", "--spine", str(p), "--gravar"])
    with pytest.raises(SystemExit, match="RECUSADO"):
        mod.main()
    assert p.read_bytes() == antes, "recusou e gravou assim mesmo"


def test_rotulo_que_muda_dentro_de_eval_only_e_permitido(mod, tmp_path):
    p = tmp_path / "spine.parquet"
    _spine(p, linhas=[("http://creativecommons.org/licenses/by-nc-sa/3.0/",
                       "CC-BY-NC-SA-4.0", "eval_only")])
    mod.gravar(p)
    df = pl.read_parquet(p)
    assert set(df["spdx_id"]) == {"CC-BY-NC-SA-3.0"} and set(df["partition"]) == {"eval_only"}


def test_rotulos_nulos_sao_preenchidos_e_o_plano_os_imprime(mod, tmp_path, monkeypatch, capsys):
    """Nulo não é decisão de ninguém: conta como mudança, e não derruba a impressão."""
    p = tmp_path / "spine.parquet"
    _spine(p, linhas=[("http://creativecommons.org/licenses/by/4.0/", None, None),
                      ("http://arxiv.org/licenses/nonexclusive-distrib/1.0/",
                       "LicenseRef-arXiv-perpetual-nonexclusive", None)])
    assert mod.mudancas(mod.plano(p)).height == 2

    monkeypatch.setattr("sys.argv", ["reaplicar_licencas.py", "--spine", str(p), "--gravar"])
    assert mod.main() == 0
    saida = capsys.readouterr().out
    assert "None" in saida and "train_open" in saida and "train_only" in saida

    df = pl.read_parquet(p)
    assert df["spdx_id"].null_count() == 0 and df["partition"].null_count() == 0
    assert set(df["partition"]) == {"train_open", "train_only"}


def test_o_quadro_mostra_a_particao_que_so_existe_depois(mod, tmp_path, monkeypatch, capsys):
    p = tmp_path / "spine.parquet"
    _spine(p, linhas=[("http://creativecommons.org/licenses/publicdomain/",
                       "NOASSERTION", "train_only")])
    monkeypatch.setattr("sys.argv", ["reaplicar_licencas.py", "--spine", str(p)])
    mod.main()
    quadro = capsys.readouterr().out.split("Partições:")[1]
    assert "train_open" in quadro, "os três registros somem de train_only e não aparecem em lugar nenhum"


def test_varios_row_groups_ficam_na_mesma_ordem(mod, tmp_path):
    """A tabela real tem 18 row groups; a de 30 linhas dos outros testes tem um."""
    p = tmp_path / "spine.parquet"
    n = 40_000
    licencas = [a[0] for a in ANTIGO]
    antes = pl.DataFrame({
        "arxiv_id": [f"{i:07d}" for i in range(n)],
        "license": [licencas[(i * 7) % len(licencas)] for i in range(n)],
        "authors": [[f"A{i}"] if i % 3 else [] for i in range(n)],
        "spdx_id": ["velho"] * n, "partition": ["train_only"] * n,
        "year": [1992 + i % 35 for i in range(n)],
    }).sample(fraction=1.0, shuffle=True, seed=17)
    antes.write_parquet(p, row_group_size=1_000)
    mod.gravar(p)
    depois = pl.read_parquet(p)
    outras = [c for c in antes.columns if c not in mod.RECALCULADAS]
    assert depois.select(outras).equals(antes.select(outras))
    assert mod.mudancas(mod.plano(p)).height == 0


def test_preserva_metadados_permissoes_e_elo_simbolico(mod, tmp_path):
    """O que o esquema polars não mostra, e uma regravação ingênua perde em silêncio."""
    alvo = tmp_path / "alvo.parquet"
    _spine(alvo)
    pl.read_parquet(alvo).write_parquet(alvo, metadata={"proveniencia": "coleta de teste"})
    alvo.chmod(0o600)
    elo = tmp_path / "spine.parquet"
    elo.symlink_to(alvo)

    mod.gravar(elo)

    assert elo.is_symlink(), "o elo simbólico virou arquivo comum"
    assert mod.mudancas(mod.plano(alvo)).height == 0, "o alvo do elo ficou com os rótulos antigos"
    assert pl.read_parquet_metadata(alvo).get("proveniencia") == "coleta de teste"
    assert (alvo.stat().st_mode & 0o777) == 0o600


def test_interrupcao_durante_a_escrita_nao_deixa_temporario(mod, tmp_path, monkeypatch):
    """A escrita é a parte longa; interrompida, sobrava até 1 GB de `.tmp` em disco."""
    p = tmp_path / "spine.parquet"
    antes = _spine(p)

    def interrompe(self, caminho, **kw):
        Path(caminho).write_bytes(b"pela metade")
        raise KeyboardInterrupt

    monkeypatch.setattr(pl.LazyFrame, "sink_parquet", interrompe)
    with pytest.raises(KeyboardInterrupt):
        mod.gravar(p)
    assert not (tmp_path / "spine.parquet.tmp").exists()
    assert pl.read_parquet(p).equals(antes)


def test_falha_na_conferencia_nao_troca_o_arquivo(mod, tmp_path, monkeypatch):
    """Se a conferência reprova, o original fica como estava e o temporário some."""
    p = tmp_path / "spine.parquet"
    antes = _spine(p)
    monkeypatch.setattr(mod, "_digesto", lambda c: (0, hash(str(c))))
    with pytest.raises(SystemExit, match="ABORTADO"):
        mod.gravar(p)
    assert pl.read_parquet(p).equals(antes)
    assert not (tmp_path / "spine.parquet.tmp").exists()
