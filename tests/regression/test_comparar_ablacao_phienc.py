"""O comparador da ablação do §2.3: aplica a regra, e RECUSA em vez de subtrair.

Sem torch — os artefatos são sintéticos, no formato que `avaliar_phienc_mlm.py` grava.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
SCRIPT = RAIZ / "scripts" / "comparar_ablacao_phienc.py"


def _artefato(caminho: Path, prova: str, por_seq: list, indices: list, **proto) -> Path:
    protocolo = {"prova": prova, "amostragem": "sorteada", "contexto": 1024,
                 "fracao_mascara": 0.15, "semente": 17, "n_sequencias_pedidas": len(indices)}
    protocolo.update(proto)
    caminho.write_text(json.dumps({
        "modelo": caminho.stem, "dados": "data/processed/phienc_aval_E",
        "tokenizer_sha": "1e1b49380deb8514", "protocolo": protocolo,
        "indices_sorteados": indices, "por_sequencia": por_seq}), encoding="utf-8")
    return caminho


def _cenario(tmp: Path, *, ganho_eq: float, ganho_pr: float, ganho_prova_eq: float,
             fracao: float = 0.538, **proto_tratado) -> list[str]:
    rng = np.random.default_rng(11)
    idx = list(range(0, 1200, 2))

    def uni(acc_eq, acc_pr):
        return [[i, int(rng.binomial(40, acc_eq)), 40, int(rng.binomial(60, acc_pr)), 60]
                for i in idx]

    def eq(acc):
        return [[i, int(rng.binomial(30, acc)), 30, 0, 0] for i in idx]

    cu = _artefato(tmp / "cu.json", "uniforme", uni(0.80, 0.70), idx)
    tu = _artefato(tmp / "tu.json", "uniforme",
                   uni(0.80 + ganho_eq, 0.70 + ganho_pr), idx, **proto_tratado)
    ce = _artefato(tmp / "ce.json", "equacao", eq(0.30), idx)
    te = _artefato(tmp / "te.json", "equacao", eq(0.30 + ganho_prova_eq), idx)
    treino = tmp / "treino.json"
    treino.write_text(json.dumps({"mascaramento": {"fracao_tratada": fracao}}),
                      encoding="utf-8")
    return ["--controle-uniforme", str(cu), "--tratado-uniforme", str(tu),
            "--controle-equacao", str(ce), "--tratado-equacao", str(te),
            "--treino-tratado", str(treino), "--n-boot", "1000",
            "--out", str(tmp / "ablacao.json")]


def _rodar(args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True,
                          text=True, encoding="utf-8", errors="replace", cwd=str(RAIZ))


def test_ganho_SO_em_equacao_com_as_checagens_aprovadas_da_TRATADO_A_FRENTE(tmp_path):
    r = _rodar(_cenario(tmp_path, ganho_eq=0.06, ganho_pr=0.0, ganho_prova_eq=0.2))
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads((tmp_path / "ablacao.json").read_text(encoding="utf-8"))
    assert d["desfecho"] == "TRATADO À FRENTE"


def test_ganho_IGUAL_nas_duas_regioes_NAO_e_evidencia(tmp_path):
    """⚠️ A lição da correção da regra, ponta a ponta."""
    r = _rodar(_cenario(tmp_path, ganho_eq=0.05, ganho_pr=0.05, ganho_prova_eq=0.2))
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads((tmp_path / "ablacao.json").read_text(encoding="utf-8"))
    assert d["desfecho"] == "NÃO DECIDIDO"


def test_checagem_2_reprovada_invalida_o_run_mesmo_com_primaria_positiva(tmp_path):
    r = _rodar(_cenario(tmp_path, ganho_eq=0.06, ganho_pr=0.0, ganho_prova_eq=0.0))
    assert r.returncode == 0, r.stdout + r.stderr
    d = json.loads((tmp_path / "ablacao.json").read_text(encoding="utf-8"))
    assert d["desfecho"] == "RUN INVÁLIDO"


def test_protocolo_diferente_e_RECUSADO(tmp_path):
    r = _rodar(_cenario(tmp_path, ganho_eq=0.06, ganho_pr=0.0, ganho_prova_eq=0.2,
                        contexto=512))
    assert r.returncode != 0
    assert "contexto" in (r.stdout + r.stderr)
    assert not (tmp_path / "ablacao.json").exists()
