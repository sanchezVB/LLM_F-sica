"""O caminho CUDA não roda nesta máquina, então tem de ser testado sem ela.

A máquina do projeto é uma RX 7600 com DirectML. O caminho CUDA existe para o
Kaggle (30 h/semana de T4, custo zero) e para GPU alugada — e código que nunca roda
no ambiente de desenvolvimento é código que ninguém verifica até doer.

Dois pontos importam mais que os outros:

**A recusa de GradCache + AMP.** É rede de segurança contra gradiente
silenciosamente errado, e rede de segurança que nunca dispara em teste é decoração.

**A ordem de precedência.** `auto` tem de preferir CUDA, e `--dispositivo cuda` num
lugar sem CUDA tem de ERRAR em vez de cair para CPU em silêncio — um treino de 15 h
que roda centenas de vezes mais devagar por queda silenciosa é pior que um que não
começa.

Tudo aqui usa dublês para o encoder e o tokenizer: exercitar quatro linhas de
decisão não deve custar 440 MB de download e 40 s de carga.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")

import torch  # noqa: E402

from phifm.training.embedding import Config, escolher_dispositivo  # noqa: E402


class ModFalso:
    """Só o que `__init__` toca. `to()` devolve self para não exigir GPU."""

    def __init__(self, **kw):
        self.kwargs = kw
        self.config = type("C", (), {"use_cache": True})()
        self.checkpointing_ligado = False

    def to(self, _dev):
        return self

    def parameters(self):
        return [torch.nn.Parameter(torch.zeros(1))]

    def gradient_checkpointing_enable(self):
        self.checkpointing_ligado = True


@pytest.fixture
def treinador_em(monkeypatch):
    """Fábrica: constrói um `TreinadorEmb` num dispositivo simulado."""
    import phifm.training.embedding as emb

    def fabricar(dispositivo: str, cfg: Config):
        criado: dict = {}

        def from_pretrained(cls, *a, **k):
            m = ModFalso(**k)
            criado["mod"] = m
            return m

        monkeypatch.setattr(emb, "escolher_dispositivo",
                            lambda _p: torch.device(dispositivo))
        monkeypatch.setattr(emb.AutoTokenizer, "from_pretrained",
                            classmethod(lambda cls, *a, **k: object()))
        monkeypatch.setattr(emb.AutoModel, "from_pretrained",
                            classmethod(from_pretrained))
        monkeypatch.setattr(torch.amp, "GradScaler", lambda *a, **k: object())
        t = emb.TreinadorEmb(cfg)
        t.mod_falso = criado["mod"]
        return t

    return fabricar


# ─── precedência de dispositivo ──────────────────────────────────────────────


def test_cuda_pedido_e_ausente_levanta_em_vez_de_cair_para_cpu():
    """Queda silenciosa para CPU num treino de 15 h é o pior resultado.

    O DirectML tem a mesma regra (`--dispositivo dml`), pelo mesmo motivo: se o
    usuário nomeou o dispositivo, ele quer aquele. "Não achei, usei outro" é
    ausência de erro lida como sucesso.
    """
    if torch.cuda.is_available():
        pytest.skip("há CUDA nesta máquina; o teste é sobre a ausência")
    with pytest.raises(RuntimeError, match="CUDA pedido explicitamente"):
        escolher_dispositivo("cuda")


def test_cpu_pedido_e_respeitado():
    assert escolher_dispositivo("cpu").type == "cpu"


def test_auto_prefere_cuda_quando_existe(monkeypatch):
    """A precedência é CUDA > DirectML > CPU, e não a ordem em que foi escrita.

    Se `auto` escolhesse DirectML havendo CUDA, o Kaggle rodaria no caminho lento
    sem ninguém notar.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Tesla T4")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda i: type("P", (), {"total_memory": 16 * 10**9})())
    assert escolher_dispositivo("auto").type == "cuda"


# ─── AMP e atenção decididos pelo dispositivo, não por bandeira ──────────────


def test_amp_e_sdpa_ligam_no_cuda(treinador_em):
    t = treinador_em("cuda", Config(amp=True))
    assert t.amp is True
    assert t.escala is not None
    assert t.atencao == "sdpa"
    assert t.mod_falso.kwargs["attn_implementation"] == "sdpa"


def test_amp_nao_liga_fora_do_cuda_mesmo_pedido(treinador_em):
    """Medido: autocast no DirectML dá queda silenciosa no backward.

    Então a decisão é pelo DISPOSITIVO. Uma bandeira que quebra em metade dos
    dispositivos é armadilha, não opção — e `amp=True` é o padrão.
    """
    t = treinador_em("cpu", Config(amp=True))
    assert t.amp is False
    assert t.escala is None
    assert t.atencao == "eager", "`eager` é a imposição do DML/CPU"


def test_sem_amp_desliga_no_cuda(treinador_em):
    """`--sem-amp` existe para comparar contra os treinos já feitos, todos em fp32.

    Mudar precisão E dados ao mesmo tempo não isola nada — é o erro que este
    projeto cometeu ao mudar base e lote juntos, e que custou um experimento.
    """
    t = treinador_em("cuda", Config(amp=False))
    assert t.amp is False and t.escala is None
    assert t.atencao == "sdpa", "a atenção é do dispositivo, não da precisão"


# ─── a rede de segurança ─────────────────────────────────────────────────────


def test_gradcache_com_amp_e_recusado(treinador_em):
    """A recusa tem de disparar no código REAL, com a mensagem que instrui.

    O GradCache deriva a perda em relação a representações cacheadas e injeta o
    gradiente num segundo forward. Com `GradScaler` no meio, a escala da fase 2
    precisa ser desfeita antes da injeção da fase 3 — e errar isso não levanta
    exceção: produz um treino que roda até o fim e aprende outra coisa.

    ⚠️ A primeira versão deste teste replicava a condição num helper e verificava a
    fonte do `__init__` por string. Passaria para sempre, inclusive com a
    verificação original apagada — o defeito clássico de teste que reimplementa o
    que testa. Agora o `__init__` de verdade roda.
    """
    with pytest.raises(RuntimeError, match="gradiente errado em silêncio"):
        treinador_em("cuda", Config(sub_lote=64, amp=True))


def test_gradcache_sem_amp_e_permitido(treinador_em):
    """O contrapositivo. Sem ele, a recusa poderia estar bloqueando o GradCache
    inteiro e o teste acima passaria igual."""
    t = treinador_em("cuda", Config(sub_lote=64, amp=False))
    assert t.cfg.sub_lote == 64 and t.amp is False


def test_gradcache_no_directml_nao_e_afetado(treinador_em):
    """É a configuração que rodou o treino de 1,5 M pares nesta máquina.

    A recusa é sobre CUDA+AMP. Se ela pegasse o DML também, teria quebrado o
    caminho que estava em produção — e o teste que prova isso vale mais que o que
    prova a recusa.
    """
    t = treinador_em("cpu", Config(sub_lote=64, amp=True))
    assert t.cfg.sub_lote == 64 and t.amp is False
