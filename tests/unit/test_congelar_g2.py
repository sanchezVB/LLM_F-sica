"""A emenda do congelamento do G2 (`scripts/congelar_g2.py --emenda`), feita em CÓPIAS do
g2.yaml, do g2_congelamento.json e do g2.py: o motivo e a regra com aspas, barra invertida e
acento voltam iguais; número fora de ordem, emenda vazia e g2.py sem a trava não gravam nada."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

import congelar_g2 as c  # noqa: E402
from phifm.eval import g2, g2_geracao  # noqa: E402

MOTIVO = 'sondas S1/S2: "lacre" por hash, barra \\ e acentuação'
REGRA = 'condição 3: sinal = S1 (13 fichas) ou S2 (trecho ≥32 fichas); "híbrida"'


@pytest.fixture
def copias(tmp_path, monkeypatch):
    arqs = {"yaml": tmp_path / "g2.yaml", "cong": tmp_path / "g2_congelamento.json",
            "g2py": tmp_path / "g2.py"}
    shutil.copy(g2.PREREGISTRO, arqs["yaml"])
    shutil.copy(g2.CONGELAMENTO, arqs["cong"])
    shutil.copy(c.G2_PY, arqs["g2py"])
    anterior = json.loads(arqs["cong"].read_text(encoding="utf-8"))
    monkeypatch.setattr(g2, "PREREGISTRO", arqs["yaml"])
    monkeypatch.setattr(g2, "CONGELAMENTO", arqs["cong"])
    monkeypatch.setattr(g2, "PROTOCOLO_G2_SHA256", g2.sha256_arquivo(arqs["yaml"]))
    monkeypatch.setattr(c, "G2_PY", arqs["g2py"])
    hc = anterior["hash_da_configuracao"]["Qwen3-1.7B-Base"]
    monkeypatch.setattr(g2_geracao, "hash_da_configuracao", lambda *a, **k: hc)
    estado = {"codigo": dict(anterior["codigo"]), "dados": dict(anterior["dados"])}
    estado["codigo"]["src/phifm/eval/g2_novo.py"] = "a" * 64
    monkeypatch.setattr(g2, "estado_congelavel", lambda raiz=None: estado)
    proxima = max(e["n"] for e in anterior["emendas"]) + 1
    return arqs, estado, anterior, proxima


def _bytes(arqs):
    return {k: p.read_bytes() for k, p in arqs.items()}


def test_emenda_com_motivo_e_regra_dificeis(copias):
    arqs, _, anterior, proxima = copias
    assert c.emendar(proxima, MOTIVO, REGRA) == 0
    lido = yaml.safe_load(arqs["yaml"].read_text(encoding="utf-8"))
    assert lido["emendas"][-1]["motivo"] == MOTIVO and lido["emendas"][-1]["regra"] == REGRA
    assert lido["congelamento"]["sha256"] == g2.sha256_arquivo(arqs["cong"])
    cong = json.loads(arqs["cong"].read_text(encoding="utf-8"))
    assert cong["emendas"][-1]["regra"] == REGRA and cong["emendas"][-1]["n"] == proxima
    assert cong["emendas"][-1]["mudou"] == ["src/phifm/eval/g2_novo.py"]
    assert cong["emendas"][-1]["protocolo_anterior_sha256"] == g2.PROTOCOLO_G2_SHA256
    novo = g2.sha256_arquivo(arqs["yaml"])
    assert f'PROTOCOLO_G2_SHA256: str | None = "{novo}"' in arqs["g2py"].read_text(encoding="utf-8")
    assert len(cong["emendas"]) == len(anterior["emendas"]) + 1


def test_emenda_fora_de_ordem_vazia_ou_sem_trava_nao_grava_nada(copias, monkeypatch):
    arqs, estado, anterior, proxima = copias
    antes = _bytes(arqs)
    with pytest.raises(SystemExit, match="não é a seguinte"):
        c.emendar(proxima + 1, MOTIVO)
    assert _bytes(arqs) == antes
    monkeypatch.setattr(g2, "estado_congelavel",
                        lambda raiz=None: {"codigo": anterior["codigo"], "dados": anterior["dados"]})
    with pytest.raises(SystemExit, match="emenda vazia"):
        c.emendar(proxima, MOTIVO)
    assert _bytes(arqs) == antes
    monkeypatch.setattr(g2, "estado_congelavel", lambda raiz=None: estado)
    arqs["g2py"].write_text("# sem a trava\n", encoding="utf-8")
    antes = _bytes(arqs)
    with pytest.raises(SystemExit, match="trava"):
        c.emendar(proxima, MOTIVO)
    assert _bytes(arqs) == antes
