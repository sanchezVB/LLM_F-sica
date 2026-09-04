"""O laço de pré-treino, num modelo minúsculo, em CPU.

É o único arquivo do pré-treino que precisa de torch, e por isso ele pula na suíte
rápida. O que ele confere não é a qualidade do modelo — é que as quatro peças estão
costuradas do jeito certo:

  1. **a perda inicial é `ln(V)`** — é o teste de correção mais forte que existe para
     um MLM. Se o mascaramento, os alvos ou o `-100` estiverem errados, este número
     sai diferente e nenhum outro sintoma aparece;
  2. o checkpoint retoma no mesmo passo, com o plano WSD do disco;
  3. o weight decay não toca norms nem bias;
  4. os contadores do mascaramento chegam ao JSON.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

torch = pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")
pytest.importorskip("transformers")

from phifm.models.encoder.config import ConfigEnc  # noqa: E402
from phifm.training.pretrain.dados import (  # noqa: E402
    NOME_MANIFESTO,
    NOME_MARCAS,
    NOME_TOKENS,
    ConfigDados,
    Fluxo,
    marcas_de,
)
from phifm.training.pretrain.laco import (  # noqa: E402
    NOME_METRICAS,
    ConfigTreino,
    Treinador,
)
from phifm.training.pretrain.mascaramento import ConfigMascara  # noqa: E402

# Minúsculo de propósito: o teste é sobre a costura, não sobre o modelo.
VOCAB = 512
MINI = ConfigEnc(nome="mini", camadas=2, d_model=64, cabecas=4, ffn=96,
                 vocab=VOCAB, contexto=128, janela_local=64)


def _dados(tmp_path: Path, n: int = 128 * 40) -> Path:
    raiz = tmp_path / "dados"
    raiz.mkdir(parents=True, exist_ok=True)
    # ⚠️ Idempotente: no Windows nao se reescreve um arquivo que outro
    # `np.memmap` mantem aberto, e o teste de retomada cria DOIS treinadores sobre
    # o mesmo tmp_path. Reescrever daria OSError 22 em vez do que se quer testar.
    if (raiz / NOME_MANIFESTO).exists():
        return raiz
    rng = np.random.default_rng(3)
    ids = rng.integers(5, VOCAB, size=n, dtype=np.uint16)
    ide = np.full(n, -1, dtype=np.int32)
    disp = np.zeros(n, dtype=bool)
    for k, i in enumerate(range(20, n - 40, 100)):
        ide[i:i + 30] = k
        disp[i:i + 30] = True
    (raiz / NOME_TOKENS).write_bytes(ids.tobytes())
    (raiz / NOME_MARCAS).write_bytes(marcas_de(ide, disp).tobytes())
    (raiz / NOME_MANIFESTO).write_text(
        json.dumps({"tokens": int(n), "git_sha": "smoke",
                    "tokenizer": "sintetico"}), encoding="utf-8")
    return raiz


def _treinador(tmp_path: Path, total: int = 6, p_eq: float = 0.5) -> Treinador:
    fluxo = Fluxo(ConfigDados(raiz=_dados(tmp_path), contexto=MINI.contexto,
                              sequencias=2, semente=17))
    return Treinador(
        MINI, ConfigTreino(total_passos=total, passos_log=1, passos_estado=2,
                           amp=False),
        ConfigMascara(p_equacao=p_eq, semente=17), fluxo,
        dev=torch.device("cpu"))


def test_a_perda_inicial_e_ln_do_vocabulario(tmp_path):
    """⚠️ O teste de correção mais forte que existe para um MLM.

    Um modelo não treinado prevê uniforme, então a entropia cruzada é `ln(V)`. Se o
    mascaramento apagar os alvos, se o `-100` estiver no lugar errado, ou se os
    alvos vierem da entrada já mascarada, este número sai diferente — e **nenhum
    outro sintoma aparece**: o treino roda, a perda desce de onde estiver, e o
    modelo fica pior de um jeito que nenhuma métrica aponta.

    Medido no corpus de verdade com V=40.960: 10,73 contra ln(40.960)=10,62.
    """
    # `total=10` e nao 1: com um passo so, warmup + decay nao deixam plato e o
    # `plano_wsd` levanta — de proposito. Aqui so o passo 0 e executado.
    t = _treinador(tmp_path, total=10)
    perda, _, _ = t._passo(0)
    assert perda == pytest.approx(math.log(VOCAB), rel=0.06), (
        f"perda inicial {perda:.4f} contra ln({VOCAB})={math.log(VOCAB):.4f}. "
        "O MLM não está prevendo uniforme no início — algo na cadeia de "
        "mascaramento/alvos está errado.")


def test_a_perda_desce_em_alguns_passos(tmp_path):
    t = _treinador(tmp_path, total=8)
    m = t.treinar(tmp_path / "saida")
    perdas = [h["perda"] for h in m.historico]
    assert perdas[-1] < perdas[0], perdas


def test_o_weight_decay_nao_toca_norms_nem_bias(tmp_path):
    """⚠️ Decay em `LayerNorm.weight` empurra o ganho para zero — é apagar a
    normalização devagar. O treino não quebra; ele fica pior em silêncio."""
    t = _treinador(tmp_path)
    grupos = t.opt.param_groups
    assert len(grupos) == 2
    com = next(g for g in grupos if g["weight_decay"] > 0)
    sem = next(g for g in grupos if g["weight_decay"] == 0)
    assert all(p.ndim >= 2 for p in com["params"]), (
        "um tensor 1-D entrou no grupo com weight decay")
    assert any(p.ndim == 1 for p in sem["params"])


def test_o_checkpoint_retoma_no_mesmo_passo_com_o_plano_do_disco(tmp_path):
    """O plano WSD vem do checkpoint, não das frações: recalculá-lo com outro
    `total_passos` mudaria a LR de passos já dados."""
    saida = tmp_path / "saida"
    t = _treinador(tmp_path, total=4)
    t.treinar(saida)
    plano = (t.passos_warmup, t.passos_decay)

    # Uma retomada com OUTRO total: o plano do disco tem de vencer.
    t2 = _treinador(tmp_path, total=99)
    assert (t2.passos_warmup, t2.passos_decay) != plano, "o teste não testa nada"
    assert t2.retomar(saida) == 4
    assert (t2.passos_warmup, t2.passos_decay) == plano, (
        "o plano WSD foi recalculado na retomada — a LR de passos já dados mudaria")


def test_os_contadores_do_mascaramento_chegam_ao_json(tmp_path):
    """`fracao_tratada` baixa invalida a ablação do DOC-07 §2.3, e quem lê o
    resultado depois precisa do número junto da perda — não só no log."""
    saida = tmp_path / "saida"
    _treinador(tmp_path, total=4, p_eq=1.0).treinar(saida)
    d = json.loads((saida / NOME_METRICAS).read_text(encoding="utf-8"))
    assert "mascaramento" in d
    assert d["mascaramento"]["fracao_tratada"] > 0.0
    assert "invalida" in d["mascaramento"]["nota"]
    # e a taxa efetiva bate com a pedida, que é o invariante do orçamento igual
    assert d["mascaramento"]["taxa_efetiva"] == pytest.approx(0.30, abs=0.01)


def test_o_controle_da_ablacao_nao_trata_nenhum_exemplo(tmp_path):
    saida = tmp_path / "saida"
    t = _treinador(tmp_path, total=4, p_eq=0.0)
    t.treinar(saida)
    d = json.loads((saida / NOME_METRICAS).read_text(encoding="utf-8"))
    assert d["mascaramento"]["sorteados_para_tratamento"] == 0
    assert d["mascaramento"]["tokens_de_equacao_mascarados"] == 0
    # e a taxa de máscara é a MESMA do braço tratado: é o orçamento igual
    assert d["mascaramento"]["taxa_efetiva"] == pytest.approx(0.30, abs=0.01)


def test_o_json_registra_a_ressalva_da_mfu(tmp_path):
    """MFU contra FLOPS nominais é um limite superior otimista, e sem FA-2 o teto
    prático é bem menor. Um número de MFU sem essa ressalva vira alegação."""
    saida = tmp_path / "saida"
    _treinador(tmp_path, total=4).treinar(saida)
    d = json.loads((saida / NOME_METRICAS).read_text(encoding="utf-8"))
    assert "NOMINAIS" in d["ressalva_mfu"] and "FA-2" in d["ressalva_mfu"]


def test_o_resumo_conta_tokens_e_epocas(tmp_path):
    t = _treinador(tmp_path, total=10)
    r = t.resumo()
    assert r["tokens_totais"] == 10 * 2 * MINI.contexto
    assert r["parametros"] == MINI.parametros()["total"]
    assert r["epocas"] > 0


def test_config_de_treino_invalida_levanta():
    for kw in ({"total_passos": 0}, {"total_passos": 10, "acumulacao": 0}):
        with pytest.raises(ValueError, match="positivos"):
            ConfigTreino(**kw)
