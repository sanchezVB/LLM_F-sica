"""O leitor das fatias do HuggingFace tem de aceitar o formato REAL da fonte.

Este arquivo existe por causa de um defeito que teria custado 42,7 horas de
download para produzir um diretório vazio.

`filtrar()` fazia `pl.read_parquet(local)` fixo. O peS2o v1 é distribuído em
`.json.gz`, então cada arquivo levantava; a exceção era capturada, logada como
aviso, e o resumo final imprimia com código de saída 0:

    data/v1/train-00000-of-00020.json.gz falhou: parquet: File must end with PAR1
    1/1 · 0 vistos · 0 aceitos (0,0%) · 2,9 GB

Baixou 2,9 GB e leu zero registros — e chamou isso de sucesso.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import polars as pl

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.corpus.slices.hf_filtrado import _ler_blocos, _texto_e_url  # noqa: E402


def _gz(destino: Path, n: int) -> Path:
    p = destino / "train-00000-of-00020.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"text": f"documento {i} sobre supercondutividade",
                                 "url": f"http://exemplo.org/{i}"}) + "\n")
    return p


def test_le_json_gz_que_e_o_formato_do_pes2o(tmp_path):
    """O teste que impede o `read_parquet` fixo de voltar."""
    blocos = list(_ler_blocos(_gz(tmp_path, 120), por_bloco=50))
    assert [len(b) for b in blocos] == [50, 50, 20]
    assert sum(len(b) for b in blocos) == 120, "perdeu registros entre os blocos"


def test_parquet_continua_funcionando_pelo_mesmo_caminho(tmp_path):
    """O OpenWebMath é parquet; o conserto não pode ter quebrado a outra fonte."""
    p = tmp_path / "parte.parquet"
    pl.DataFrame({"text": ["a", "b"], "url": ["u", "v"]}).write_parquet(p)
    blocos = list(_ler_blocos(p))
    assert len(blocos) == 1 and len(blocos[0]) == 2


def test_texto_e_url_saem_alinhados_de_cada_bloco(tmp_path):
    """Desalinhar texto e URL parearia um documento com a fonte de outro."""
    for bloco in _ler_blocos(_gz(tmp_path, 30), por_bloco=10):
        textos, urls = _texto_e_url(bloco)
        assert len(textos) == len(urls)
        for t, u in zip(textos, urls, strict=True):
            i = t.split()[1]
            assert u.endswith(f"/{i}"), f"texto {t!r} veio com a URL {u!r}"


def test_formato_desconhecido_levanta_em_vez_de_devolver_vazio(tmp_path):
    """Devolver zero blocos em silêncio é como o defeito original passou."""
    import pytest

    p = tmp_path / "arquivo.tar.bz2"
    p.write_bytes(b"nao importa")
    with pytest.raises(ValueError, match="formato"):
        list(_ler_blocos(p))


def test_bloco_vazio_no_fim_nao_gera_dataframe_vazio(tmp_path):
    """Um arquivo com múltiplo exato de `por_bloco` não deve emitir bloco a mais."""
    blocos = list(_ler_blocos(_gz(tmp_path, 100), por_bloco=50))
    assert [len(b) for b in blocos] == [50, 50]


def test_linhas_em_branco_sao_ignoradas(tmp_path):
    """`read_ndjson` levanta numa linha vazia, e arquivos reais as têm no fim."""
    p = tmp_path / "com-brancos.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": "um", "url": "http://a"}) + "\n")
        fh.write("\n")
        fh.write(json.dumps({"text": "dois", "url": "http://b"}) + "\n")
        fh.write("   \n")
    blocos = list(_ler_blocos(p, por_bloco=50))
    assert sum(len(b) for b in blocos) == 2


# ─── a assinatura que amarra o ordinal à lista ───────────────────────────────


def test_retomada_recusa_manifesto_de_outra_lista(tmp_path):
    """O manifesto guarda NÚMEROS de unidade, e um número só vale contra uma lista.

    Medido em 2026-08-25: o peS2o publica v1 E v2 da mesma coleção no mesmo
    repositório e o filtro pegava as duas. Ao restringir para `data/v2/`, o
    manifesto ainda dizia "unidade 1 feita" — mas a unidade 1 havia deixado de ser
    `data/v1/train-00000` e passado a ser `data/v2/train-00000`, nunca processado.
    Retomar teria pulado esse arquivo E deixado dados de v1 no destino.
    """
    import pytest

    from phifm.corpus.slices.retomada import (
        assinatura_da_lista,
        feitas,
        marcar,
    )

    velha = assinatura_da_lista(["data/v1/a.json.gz", "data/v1/b.json.gz"])
    nova = assinatura_da_lista(["data/v2/a.json.gz", "data/v2/b.json.gz"])
    assert velha != nova

    marcar(tmp_path, {1}, assinatura=velha)
    assert feitas(tmp_path, assinatura=velha) == {1}, "mesma lista tem de retomar"
    with pytest.raises(ValueError, match="OUTRA lista"):
        feitas(tmp_path, assinatura=nova)


def test_manifesto_antigo_sem_assinatura_ainda_e_aceito(tmp_path):
    """Coletas em andamento foram escritas sem o campo; quebrá-las custaria horas."""
    import json

    from phifm.corpus.slices.retomada import MANIFESTO, assinatura_da_lista, feitas

    (tmp_path / MANIFESTO).write_text(json.dumps({"unidades": [1, 2]}), encoding="utf-8")
    a = assinatura_da_lista(["x"])
    assert feitas(tmp_path, assinatura=a) == {1, 2}


def test_assinatura_depende_da_ORDEM_nao_so_do_conjunto():
    """Os ordinais vêm da posição, então trocar a ordem invalida o manifesto."""
    from phifm.corpus.slices.retomada import assinatura_da_lista

    assert assinatura_da_lista(["a", "b"]) != assinatura_da_lista(["b", "a"])


def test_uma_versao_por_execucao(tmp_path, monkeypatch):
    """`filtrar` recusa uma lista que abranja duas versões do mesmo corpus."""
    import pytest

    from phifm.corpus.slices import hf_filtrado as hf

    def falsa_listagem(sessao, ds, ext):
        return ([("data/v1/train-00000.json.gz", 1), ("data/v2/train-00000.json.gz", 1)],
                "rev123")

    monkeypatch.setattr(hf, "_arquivos", falsa_listagem)
    with pytest.raises(ValueError, match="versões do mesmo corpus"):
        hf.filtrar("allenai/peS2o", tmp_path / "out",
                   Path("models/isphysics-clf"), ext=(".json.gz",))
