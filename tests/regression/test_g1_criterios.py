"""O portão G1 tem de ser julgado pelo que o DOC-00 §5 escreve.

Regressão de 2026-08-11. Três defeitos distintos, todos do tipo que produz um
veredito plausível e errado — o pior tipo num portão, porque ninguém desconfia.

## 1. O campeão era o PRIMEIRO dos nossos, não o melhor

`next((r for r in ok if r.nosso), None)` bastava com um modelo nosso. Com dois —
o ΦEmb sobre SciBERT (110M) e o sobre MiniLM (23M) — passou a devolver o que a
ordem do dicionário entregasse. O portão seria julgado por um modelo sorteado.

## 2. A métrica não era a do critério

O G1.1 pede **nDCG@10**; eu julgava por recall@1. Dá para acertar o veredito por
sorte e errar o critério, e num portão o critério é o resultado.

## 3. A cláusula de tamanho é relativa ao RIVAL

O G1.2 pede superar o melhor embedder geral «com ≤ 1/10 dos parâmetros». Vencer
um genérico de 23M com um modelo de 23M NÃO fecha o critério: a razão exige
rival de ≥ 230M. Tratar «bati o MiniLM» como G1.2 fechado seria satisfazer o
critério fácil e declarar o difícil.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino, Python 3.12)")

from phifm.eval.encoders import (  # noqa: E402
    ALVO_DOMINIO,
    MARGEM_G1_1,
    Resultado,
    campeao,
    veredito,
)

GTE = "thenlper/gte-large"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def R(nome, caminho, *, r1=0.2, ndcg=0.3, params=100.0, nosso=False, acertos=None):
    """Resultado sintético. `acertos` fixa as posições para o teste pareado."""
    pos = acertos if acertos is not None else [1] * int(r1 * 100) + [7] * (100 - int(r1 * 100))
    return Resultado(nome, caminho, r1, 0.0, 0.0, ndcg, params, 1.0,
                     nosso=nosso, posicoes=pos)


# ─── 1. campeão ──────────────────────────────────────────────────────────────

def test_campeao_e_o_melhor_nao_o_primeiro():
    rs = [R("nosso fraco", "models/a", r1=0.20, nosso=True),
          R("nosso forte", "models/b", r1=0.32, nosso=True)]
    assert campeao(rs).nome == "nosso forte"


def test_campeao_ignora_ordem_de_insercao():
    """Inverter a ordem não pode mudar quem é o campeão."""
    a = R("nosso fraco", "models/a", r1=0.20, nosso=True)
    b = R("nosso forte", "models/b", r1=0.32, nosso=True)
    assert campeao([a, b]).nome == campeao([b, a]).nome == "nosso forte"


def test_campeao_ignora_modelos_de_fora():
    """Um rival melhor que o nosso não vira nosso campeão."""
    rs = [R("nosso", "models/a", r1=0.20, nosso=True),
          R("rival", GTE, r1=0.90)]
    assert campeao(rs).nome == "nosso"


def test_campeao_ignora_os_que_falharam():
    ruim = R("nosso quebrado", "models/x", r1=0.99, nosso=True)
    ruim.erro = "OSError: sem config.json"
    rs = [ruim, R("nosso ok", "models/b", r1=0.10, nosso=True)]
    assert campeao(rs).nome == "nosso ok"


def test_campeao_sem_nossos_e_none():
    assert campeao([R("rival", GTE)]) is None


# ─── 2. o nDCG@10 é a métrica do G1.1 ────────────────────────────────────────

def test_g1_1_julga_por_ndcg_nao_por_recall():
    """Constrói o caso em que as duas métricas discordam.

    O nosso tem recall@1 MAIOR e nDCG@10 MENOR que o PhysBERT. Julgar por
    recall@1 aprovaria; o critério reprova.
    """
    rs = [R("nosso", "models/b", r1=0.40, ndcg=0.20, nosso=True),
          R("PhysBERT", ALVO_DOMINIO, r1=0.10, ndcg=0.50)]
    v = veredito(rs, 2000)
    assert "G1.1: NÃO PASSOU" in v, v


def test_g1_1_exige_a_margem_de_cinco_pontos():
    """Superar não basta: o limiar é +0,05 de nDCG@10."""
    rs = [R("nosso", "models/b", ndcg=0.30 + MARGEM_G1_1 / 2, nosso=True),
          R("PhysBERT", ALVO_DOMINIO, ndcg=0.30)]
    assert "G1.1: NÃO PASSOU" in veredito(rs, 2000)

    rs[0] = R("nosso", "models/b", ndcg=0.30 + MARGEM_G1_1, nosso=True)
    assert "G1.1: PASSOU" in veredito(rs, 2000)


def test_o_numero_impresso_nunca_contradiz_o_veredito():
    """Na fronteira, o relatório não pode dizer «+0.050 … NÃO PASSOU».

    Foi o que acontecia: `0.30 + 0.05` em float é 0.34999999999999998, a
    diferença saía 0.049999999999999996 e reprovava — mas o texto exibia +0.050,
    arredondado. Quem lesse veria uma contradição sem causa visível.

    O invariante: se o número EXIBIDO alcança o limiar, o veredito aprova.
    """
    import re
    for passo in range(-3, 4):
        base = 0.30
        rs = [R("nosso", "models/b", ndcg=base + MARGEM_G1_1 + passo * 1e-3, nosso=True),
              R("PhysBERT", ALVO_DOMINIO, ndcg=base)]
        v = veredito(rs, 2000)
        exibido = float(re.search(r"nDCG@10 ([+-][\d.]+) sobre o PhysBERT", v).group(1))
        aprovou = "G1.1: PASSOU" in v
        assert aprovou == (exibido >= MARGEM_G1_1), (
            f"exibiu {exibido:+.3f} e {'aprovou' if aprovou else 'reprovou'}")


def test_ndcg_de_um_relevante_e_o_inverso_do_log():
    """Com UM relevante por consulta, nDCG@10 = 1/log2(1+pos), 0 fora do top-10.

    Fixa a fórmula, que é o que torna a métrica comparável ao que o portão pede.
    """
    import torch
    pos = torch.tensor([1, 2, 10, 11])
    dcg = torch.where(pos <= 10, 1.0 / torch.log2(pos.float() + 1.0),
                      torch.zeros_like(pos, dtype=torch.float))
    assert dcg[0].item() == pytest.approx(1.0)                          # log2(2)=1
    assert dcg[1].item() == pytest.approx(1 / math.log2(3))
    assert dcg[2].item() == pytest.approx(1 / math.log2(11))
    assert dcg[3].item() == 0.0, "posição 11 não pode pontuar em nDCG@10"


# ─── 3. a cláusula de tamanho do G1.2 ────────────────────────────────────────

def test_g1_2_nao_fecha_contra_generico_do_mesmo_tamanho():
    """O caso que me enganava: vencer o MiniLM de 23M com 23M."""
    rs = [R("nosso", "models/b", ndcg=0.45, params=23, nosso=True),
          R("MiniLM-L6", MINILM, ndcg=0.32, params=23)]
    v = veredito(rs, 2000)
    assert "G1.2: PARCIAL" in v, v
    assert "1/1.0" in v
    assert "≥ 230M" in v, "devia dizer qual rival fecharia o critério"


def test_g1_2_fecha_contra_generico_dez_vezes_maior():
    rs = [R("nosso", "models/b", ndcg=0.45, params=23, nosso=True),
          R("GTE-large", GTE, ndcg=0.38, params=335)]
    v = veredito(rs, 2000)
    assert "G1.2: PASSOU" in v, v
    assert "1/14.6" in v


def test_g1_2_perder_para_o_generico_nao_e_parcial():
    rs = [R("nosso", "models/b", ndcg=0.30, params=23, nosso=True),
          R("GTE-large", GTE, ndcg=0.38, params=335)]
    assert "G1.2: NÃO PASSOU" in veredito(rs, 2000)


def test_g1_2_escolhe_o_MELHOR_generico_nao_o_mais_facil():
    """Com dois genéricos, o critério é contra o melhor deles."""
    rs = [R("nosso", "models/b", ndcg=0.35, params=23, nosso=True),
          R("MiniLM-L6", MINILM, ndcg=0.32, params=23),
          R("GTE-large", GTE, ndcg=0.38, params=335)]
    v = veredito(rs, 2000)
    assert "G1.2: NÃO PASSOU" in v, "perdeu do GTE; vencer o MiniLM não salva"
    assert "GTE-large" in v


# ─── ausência de dado não é aprovação ────────────────────────────────────────

def test_sem_physbert_o_g1_1_e_indeterminado():
    rs = [R("nosso", "models/b", ndcg=0.9, nosso=True), R("GTE-large", GTE, ndcg=0.1)]
    assert "G1.1: INDETERMINADO" in veredito(rs, 2000)


def test_sem_generico_o_g1_2_e_indeterminado():
    rs = [R("nosso", "models/b", ndcg=0.9, nosso=True),
          R("PhysBERT", ALVO_DOMINIO, ndcg=0.1)]
    assert "G1.2: INDETERMINADO" in veredito(rs, 2000)


def test_modelo_que_falhou_aparece_na_ressalva():
    quebrado = R("GTE-large", GTE)
    quebrado.erro = "OSError: conexão"
    rs = [R("nosso", "models/b", ndcg=0.9, nosso=True),
          R("PhysBERT", ALVO_DOMINIO, ndcg=0.1), quebrado]
    v = veredito(rs, 2000)
    assert "não avaliados" in v and "GTE-large" in v


def test_veredito_sempre_declara_que_o_portao_segue_aberto():
    """G1.1 e G1.2 verdes não fecham o G1: faltam G1.3, G1.4 e G1.5.

    O benchmark também é NOSSO, não um reservado e publicado. Sem esta ressalva
    o relatório convida a ler «G1 passou», que seria falso.
    """
    rs = [R("nosso", "models/b", ndcg=0.9, params=23, nosso=True),
          R("PhysBERT", ALVO_DOMINIO, ndcg=0.1),
          R("GTE-large", GTE, ndcg=0.2, params=335)]
    v = veredito(rs, 2000)
    assert "G1.1: PASSOU" in v and "G1.2: PASSOU" in v
    for esperado in ("G1.3", "G1.4", "G1.5", "benchmark PRÓPRIO"):
        assert esperado in v, f"faltou a ressalva sobre {esperado}"
