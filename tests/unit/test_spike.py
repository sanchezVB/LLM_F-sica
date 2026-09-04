"""O detector de spike, e as duas armadilhas que o DOC-08 §6.1 não especifica.

O §6.1 dá o critério (μ+4σ, ou 10× a mediana da norma) e a resposta em cinco passos.
O que ele não diz é **contra qual janela** julgar e **o que fazer com o valor do
spike** — e as duas escolhas decidem se o detector funciona ou se ele apenas parece
funcionar:

  1. estatísticas da janela ANTES do passo julgado, senão um spike grande esconde a
     si mesmo;
  2. o valor do spike não entra na janela, senão ele polui os 100 passos seguintes e
     uma sequência de spikes passa inteira.

Os dois testes que fixam isso são `test_um_spike_grande_nao_esconde_a_si_mesmo` e
`test_uma_sequencia_de_spikes_nao_passa_toda`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from phifm.training.pretrain.spike import (  # noqa: E402
    ConfigSpike,
    Detector,
    lr_wsd,
    plano_wsd,
)


def _aquecer(d: Detector, n: int = 100, perda: float = 2.0,
             norma: float = 0.5) -> None:
    """Enche a janela com passos normais, com um ruído pequeno e determinístico."""
    for i in range(n):
        d.observar(i, perda + 0.01 * ((i % 7) - 3), norma + 0.001 * (i % 5))


# ─── as duas armadilhas ──────────────────────────────────────────────────────


def test_um_spike_grande_nao_esconde_a_si_mesmo():
    """⚠️ Se as estatísticas incluíssem o próprio valor, um spike 50× maior que os
    outros levantaria σ o suficiente para ficar dentro de 4σ dele mesmo.

    É o modo de falha mais traiçoeiro possível: o detector fica mais cego quanto
    maior for o problema.
    """
    d = Detector()
    _aquecer(d)
    v = d.observar(100, 100.0, 0.5)
    assert v.e_spike, "um spike de 50× a perda passou"
    assert "μ+4σ" in v.motivo


def test_uma_sequencia_de_spikes_nao_passa_toda():
    """⚠️ Se o valor do spike entrasse na janela, o limiar subiria e os spikes
    seguintes ficariam dentro dele.

    A janela é a memória do que é NORMAL, e um spike não é.
    """
    d = Detector(ConfigSpike(max_spikes=99))  # sem parar, para contar todos
    _aquecer(d)
    detectados = sum(d.observar(100 + i, 50.0, 0.5).e_spike for i in range(10))
    assert detectados == 10, (
        f"só {detectados} de 10 spikes idênticos foram detectados — o valor do "
        "spike está poluindo a janela")


# ─── o critério do §6.1 ──────────────────────────────────────────────────────


def test_perda_dentro_de_quatro_sigmas_nao_e_spike():
    d = Detector()
    _aquecer(d)
    v = d.observar(100, 2.05, 0.5)
    assert not v.e_spike


def test_norma_de_gradiente_alta_tambem_dispara():
    """O §6.1 tem DOIS sinais, e a norma pega o que a perda ainda não mostrou."""
    d = Detector()
    _aquecer(d)
    v = d.observar(100, 2.0, 50.0)  # perda normal, norma 100× a mediana
    assert v.e_spike
    assert "norma do gradiente" in v.motivo


def test_nao_julga_antes_de_ter_janela():
    """Os primeiros passos têm perda descendo rápido de propósito; julgá-los como
    spike pararia todo treino no começo."""
    d = Detector(ConfigSpike(minimo_para_julgar=30))
    for i in range(29):
        assert not d.observar(i, 10.0 - i * 0.3, 1.0).e_spike
    # com a janela cheia, o mesmo salto passa a ser julgado
    _aquecer(d, n=100)
    assert d.observar(200, 99.0, 1.0).e_spike


def test_nan_e_inf_param_na_hora():
    """Em fp16 — que é o que a T4 tem, porque ela não tem bf16 — NaN é o modo de
    falha mais comum, e não é desvio estatístico: é aritmética morta."""
    for ruim in (float("nan"), float("inf")):
        d = Detector()
        _aquecer(d)
        v = d.observar(100, ruim, 0.5)
        assert v.e_spike and "não é finito" in v.motivo
    # e sem janela nenhuma também para
    d = Detector()
    assert d.observar(0, float("nan"), 1.0).e_spike


# ─── a resposta em cinco passos ──────────────────────────────────────────────


def test_a_lr_cai_pela_metade_por_quinhentos_passos_e_volta():
    """Passo 4 do §6.1: reduzir a LR em 50% por 500 passos, depois restaurar."""
    d = Detector()
    _aquecer(d)
    assert d.fator_lr(100) == 1.0
    d.observar(100, 99.0, 0.5)
    assert d.fator_lr(100) == 0.5
    assert d.fator_lr(600) == 0.5
    assert d.fator_lr(601) == 1.0


def test_tres_spikes_em_cinco_mil_passos_exigem_humano():
    """Passo 5 do §6.1. Um projeto de uma pessoa que treina durante o sono precisa
    que essa parada seja automática."""
    d = Detector()
    _aquecer(d)
    v1 = d.observar(100, 99.0, 0.5)
    v2 = d.observar(200, 99.0, 0.5)
    v3 = d.observar(300, 99.0, 0.5)
    assert not v1.exigir_humano and not v2.exigir_humano
    assert v3.exigir_humano
    assert "sistêmico" in v3.motivo and "§6.1" in v3.motivo


def test_spikes_espacados_nao_acumulam():
    """Três spikes em 50.000 passos não são sintoma sistêmico."""
    d = Detector()
    _aquecer(d)
    for passo in (100, 10_000, 30_000):
        _aquecer(d, n=100)
        v = d.observar(passo, 99.0, 0.5)
        assert v.e_spike and not v.exigir_humano, passo


def test_o_dict_explica_o_criterio_e_as_duas_decisoes():
    """O JSON do treino é lido depois, por quem não escreveu isto."""
    d = Detector().como_dict()
    assert "μ+4σ" in d["criterio"]
    assert "esconde a si mesmo" in d["nota"]


def test_config_invalida_levanta():
    for kw, msg in (({"janela": 1}, "estatística"),
                    ({"fator_lr": 0.0}, "fator_lr"),
                    ({"fator_lr": 1.0}, "fator_lr"),
                    ({"minimo_para_julgar": 1}, "desvio padrão")):
        with pytest.raises(ValueError, match=msg):
            ConfigSpike(**kw)


# ─── o agendamento WSD ───────────────────────────────────────────────────────


def test_o_wsd_tem_as_tres_fases():
    total, pico = 10_000, 1e-3
    w, d = plano_wsd(total)
    def lr(p):
        return lr_wsd(p, total, pico=pico, passos_warmup=w, passos_decay=d)
    assert lr(0) == 0.0
    assert lr(w // 2) == pytest.approx(pico / 2, rel=0.02)
    assert lr(w) == pytest.approx(pico)
    # platô: constante em todo o meio
    assert lr(1000) == pytest.approx(pico)
    assert lr(8000) == pytest.approx(pico)
    # decay: desce
    assert lr(9500) < pico
    assert lr(9999) == 0.0


def test_o_plato_NAO_depende_do_total_e_essa_e_a_razao_de_ser_do_WSD():
    """⚠️ Este teste pegou um defeito real, e a tentação era ajustar o esperado.

    A primeira versão calculava o warmup como `total × 0,03`. Com isso, estender o
    treino de 10 mil para 200 mil passos alongava o warmup de 300 para 6.000 — e a
    LR do passo 5.000, **que já havia sido dado com o pico**, passaria a valer
    8,3e-4. O agendamento deixava de ser extensível, que é o argumento inteiro do
    DOC-08 §4 para preferir WSD a cosseno.

    Com `passos_warmup` absoluto, estender move só o começo do decay.
    """
    for total in (10_000, 50_000, 200_000):
        v = lr_wsd(5_000, total, pico=1e-3, passos_warmup=300, passos_decay=1_000)
        assert v == pytest.approx(1e-3), (total, v)


def test_estender_o_treino_nao_muda_a_lr_de_passos_ja_dados():
    """A propriedade que o §4 promete, afirmada diretamente.

    Só os passos que entraram no decay do plano CURTO mudam — e isso é o esperado:
    o decay foi cancelado porque o treino continuou.
    """
    w, d = plano_wsd(10_000)
    curto = [lr_wsd(p, 10_000, pico=1e-3, passos_warmup=w, passos_decay=d)
             for p in range(0, 9_000, 137)]
    longo = [lr_wsd(p, 50_000, pico=1e-3, passos_warmup=w, passos_decay=d)
             for p in range(0, 9_000, 137)]
    assert curto == longo


def test_o_decay_respeita_o_piso():
    w, d = plano_wsd(10_000)
    v = lr_wsd(9_999, 10_000, pico=1e-3, passos_warmup=w, passos_decay=d, piso=1e-5)
    assert v == pytest.approx(1e-5)


def test_plano_sem_plato_levanta():
    """warmup + decay >= 1 não deixa fase estável, e aí o WSD é só um cosseno mal
    feito com a desvantagem de ambos."""
    with pytest.raises(ValueError, match="fase estável"):
        plano_wsd(1000, frac_warmup=0.5, frac_decay=0.6)
    with pytest.raises(ValueError, match="fase estável"):
        lr_wsd(0, 1000, pico=1e-3, passos_warmup=600, passos_decay=500)


def test_plano_com_total_invalido_levanta():
    with pytest.raises(ValueError, match="positivo"):
        plano_wsd(0)
    with pytest.raises(ValueError, match="positivo"):
        lr_wsd(0, 0, pico=1e-3, passos_warmup=1, passos_decay=1)


def test_passos_de_fase_menores_que_um_levantam():
    """`plano_wsd` nunca devolve 0, mas alguém pode passar à mão."""
    with pytest.raises(ValueError, match="plano_wsd"):
        lr_wsd(0, 1000, pico=1e-3, passos_warmup=0, passos_decay=100)


def test_o_wsd_nunca_passa_do_pico_nem_fica_negativo():
    w, d = plano_wsd(10_000)
    for passo in range(0, 10_000, 37):
        v = lr_wsd(passo, 10_000, pico=1e-3, passos_warmup=w, passos_decay=d)
        assert 0.0 <= v <= 1e-3 + 1e-12, (passo, v)


def test_plano_wsd_da_os_tres_por_cento_do_documento():
    assert plano_wsd(100_000) == (3_000, 10_000)
