"""A conta do piloto dos trechos: a posição de P e o que o ΦRank deixa entre as fontes.

São as duas funções de que a tabela inteira depende; um erro nelas não daria erro nenhum,
daria uma tabela plausível.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

import piloto_trechos as pt  # noqa: E402


def _caso(n: int = 60, p: int = 40) -> tuple[dict, dict[str, float]]:
    """`n` concorrentes; P é o de índice `p`. O ΦRank põe P em 3º entre os que pontuou."""
    ids = [f"d{i}" for i in range(n)]
    d = {"arxiv_id": f"d{p}", "ids": ids}
    rr = {a: -float(i) for i, a in enumerate(ids)}
    rr[f"d{p}"] = -1.5
    return d, rr


def test_posicao_conta_os_concorrentes_e_os_de_fora():
    d, rr = _caso()
    f = np.linspace(1.0, 0.0, 60)
    fora = np.array([0.99, 0.5, 0.1])        # um de fora na frente de P
    pos, _ = pt._por_escore(d, f, fora, rr)
    assert pos == 40 + 1 + 2                 # 40 concorrentes, 2 de fora (0,99 e 0,5), e P


def test_p_alem_dos_50_nao_chega_ao_phirank():
    d, rr = _caso(p=55)
    pos, ok = pt._por_escore(d, np.linspace(1.0, 0.0, 60), np.array([]), rr)
    assert pos == 56 and ok is False


def test_p_entre_os_50_entra_e_o_phirank_decide():
    d, rr = _caso(p=40)
    pos, ok = pt._por_escore(d, np.linspace(1.0, 0.0, 60), np.array([]), rr)
    assert pos == 41 and ok is True          # o ΦRank deixa P em 3º: está entre as 6


def test_o_phirank_pode_tirar_p_das_fontes():
    d, rr = _caso(p=40)
    rr["d40"] = -30.0                        # 30 dos 50 candidatos na frente
    assert pt._por_escore(d, np.linspace(1.0, 0.0, 60), np.array([]), rr)[1] is False


def test_candidato_sem_escore_conta_contra_p_e_e_contado():
    d, rr = _caso(p=5)
    for a in ("d0", "d1", "d2", "d3", "d4", "d6", "d7"):
        del rr[a]                            # 7 dos 50 sem escore do ΦRank
    pt.SEM_ESCORE["candidatos"] = 0
    assert pt._depois_do_phirank(d, rr, np.arange(50)) is False
    assert pt.SEM_ESCORE["candidatos"] == 7


def test_os_de_fora_ocupam_vagas_entre_os_50():
    d, rr = _caso(p=48)
    f = np.linspace(1.0, 0.0, 60)
    fora = np.full(10, 0.999)                # 10 de fora entram na frente de quase todos
    pos, ok = pt._por_escore(d, f, fora, rr)
    assert pos == 59 and ok is False         # P caiu para além dos 50
