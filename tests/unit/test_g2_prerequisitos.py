import json

import pytest

from phifm.eval import g2
from phifm.eval import g2_prerequisitos as pre


def test_ambiente_vazio_nao_parece_pronto(tmp_path):
    r = pre.conferir(tmp_path)
    assert not r.pronto and "pré-registro" in r.bloqueios[0]
    with pytest.raises(pre.PrerequisitosAusentes):
        pre.exigir(tmp_path)


def test_hash_mudado_bloqueia(tmp_path):
    arquivo = tmp_path / "artefato"
    arquivo.write_text("antes")
    esperado = pre.sha256_bytes(arquivo)
    arquivo.write_text("depois")
    r = pre.Requisitos()
    pre._hash(arquivo, esperado, r)
    assert not r.pronto


def test_autoteste_atual_transporta_exclusoes_e_recusa_obsoleto(tmp_path, monkeypatch):
    codigo = {"corretor": "abc"}
    monkeypatch.setattr(g2, "codigo_do_corretor", lambda raiz: codigo)
    dados = {"codigo_sha256": codigo, "conjuntos": {
        f"{b}/dev": {"itens": 20, "excluidos": 1, "ids_excluidos": ["x"]}
        for b in ("ugphysics", "olympiadbench", "scibench")}}
    arq = tmp_path / "g2_autoteste.json"
    arq.write_text(json.dumps(dados))
    r = pre.Requisitos()
    pre._autoteste(tmp_path, tmp_path, "dev", r)
    assert r.pronto and r.exclusoes["ugphysics"] == {"x"}
    codigo["corretor"] = "def"
    r = pre.Requisitos()
    pre._autoteste(tmp_path, tmp_path, "dev", r)
    assert not r.pronto


def test_caminho_fora_da_raiz_recusado(tmp_path):
    with pytest.raises(ValueError, match="sai da raiz"):
        pre._dentro(tmp_path, "../fora")
