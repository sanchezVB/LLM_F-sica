"""O laço de treino tem de RODAR, não só as peças dele.

Regressão de 2026-08-18. Dez testes cobriam a máscara e a perda com negativos
difíceis, todos passando — e o treino morreu no primeiro passo:

    TypeError: 'int' object is not iterable

A causa: no desempacotamento do lote eu chamei a lista de negativos de `n`,
sombreando o contador `n` da média da perda três linhas abaixo. `n += 1` virou
`lista += 1`.

Nenhum teste de unidade pegaria isso, porque nenhum executava o laço. Testei as
peças e não a montagem — e o custo foi carregar 1,2 GB de parquet, medir a linha de
base e morrer, duas vezes.

Este teste roda `treinar` de verdade por alguns passos, com encoder e tokenizer
dublados para não baixar 90 MB nem depender de GPU. É lento comparado aos outros
(~2 s contra milissegundos), e vale cada um deles: é o único que exercita o laço.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino)")

import polars as pl  # noqa: E402
import torch  # noqa: E402

import phifm.training.embedding as emb  # noqa: E402
from phifm.training.embedding import Config, TreinadorEmb  # noqa: E402

DIM = 8


class TokFalso:
    def __call__(self, textos, **kw):
        n = len(textos)
        return {"input_ids": torch.zeros(n, 4, dtype=torch.long),
                "attention_mask": torch.ones(n, 4, dtype=torch.long)}

    def save_pretrained(self, destino):
        Path(destino).mkdir(parents=True, exist_ok=True)
        (Path(destino) / "tokenizer.json").write_text("{}", encoding="utf-8")


class ModFalso(torch.nn.Module):
    """Encoder mínimo COM parâmetros, para o backward ter o que atualizar."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(DIM, DIM)
        self.config = type("C", (), {"use_cache": True})()

    def forward(self, input_ids=None, attention_mask=None, **kw):
        n, t = input_ids.shape
        h = self.proj(torch.ones(n, t, DIM))
        return type("S", (), {"last_hidden_state": h})()

    def gradient_checkpointing_enable(self):
        pass

    def save_pretrained(self, destino, state_dict=None):
        Path(destino).mkdir(parents=True, exist_ok=True)
        (Path(destino) / "model.safetensors").write_bytes(b"x")


def _pares(n: int = 64, com_duros: bool = False) -> pl.DataFrame:
    d = {
        "arxiv_id": [f"a{i}" for i in range(n)],
        "arxiv_citado": [f"c{i % 7}" for i in range(n)],
        "ancora": [f"ancora {i}" for i in range(n)],
        "positivo": [f"positivo {i % 7}" for i in range(n)],
    }
    if com_duros:
        d["negativos"] = [[f"neg {i}a", f"neg {i}b"] for i in range(n)]
        d["negativos_id"] = [[f"n{i}a", f"n{i}b"] for i in range(n)]
        # Metade das âncoras "cita" o negativo de alguém: força a máscara a agir
        # dentro do laço, que é onde ela nunca tinha rodado.
        d["proibidos"] = [[f"n{(i + 1) % n}a"] if i % 2 == 0 else [] for i in range(n)]
    return pl.DataFrame(d)


@pytest.fixture
def treinador(monkeypatch):
    def fabricar(cfg: Config) -> TreinadorEmb:
        monkeypatch.setattr(emb, "escolher_dispositivo",
                            lambda _p: torch.device("cpu"))
        monkeypatch.setattr(emb.AutoTokenizer, "from_pretrained",
                            classmethod(lambda cls, *a, **k: TokFalso()))
        monkeypatch.setattr(emb.AutoModel, "from_pretrained",
                            classmethod(lambda cls, *a, **k: ModFalso()))
        return TreinadorEmb(cfg)

    return fabricar


def test_laco_roda_com_negativos_dificeis(treinador, tmp_path):
    """O teste que faltava. Sem ele, o sombreamento de `n` chegou a produção.

    Roda passos de verdade: desempacota o lote, monta a máscara, codifica os três
    conjuntos, calcula a perda, faz backward e avança o otimizador.
    """
    cfg = Config(lote=8, max_pares=32, passos_aval=100, passos_estado=100,
                 n_candidatos=8, checkpointing=False, amp=False)
    t = treinador(cfg)
    m = t.treinar(_pares(32, com_duros=True), _pares(16), tmp_path / "saida")
    assert m.passo >= 3, "o laço não completou passos"
    assert t.com_duros is True


def test_laco_roda_sem_negativos(treinador, tmp_path):
    """O caminho antigo — o do campeão do G1.1 — não pode ter quebrado."""
    cfg = Config(lote=8, max_pares=32, passos_aval=100, passos_estado=100,
                 n_candidatos=8, checkpointing=False, amp=False)
    t = treinador(cfg)
    m = t.treinar(_pares(32), _pares(16), tmp_path / "saida")
    assert m.passo >= 3
    assert t.com_duros is False


def test_a_media_da_perda_e_registrada_nos_dois_caminhos(treinador, tmp_path):
    """O contador `n` da média era exatamente a variável sombreada.

    Se ele voltar a ser sobrescrito, a perda registrada fica errada ou o laço morre
    — e nos dois casos este teste cai.
    """
    cfg = Config(lote=8, max_pares=32, passos_log=2, passos_aval=100,
                 passos_estado=100, n_candidatos=8, checkpointing=False, amp=False)
    for duros in (False, True):
        t = treinador(cfg)
        m = t.treinar(_pares(32, com_duros=duros), _pares(16),
                      tmp_path / f"saida_{duros}")
        assert m.pares_por_s > 0, "vazão não medida — o bloco de log não rodou"


def test_estado_e_gravado_dentro_do_laco_com_duros(treinador, tmp_path):
    """A retomada tem de funcionar também no caminho novo.

    O `salvar_estado` está fora do bloco de avaliação (regressão de 2026-08-16), mas
    isso foi testado só no caminho sem negativos.
    """
    saida = tmp_path / "saida"
    cfg = Config(lote=8, max_pares=32, passos_aval=100, passos_estado=2,
                 n_candidatos=8, checkpointing=False, amp=False)
    t = treinador(cfg)
    t.treinar(_pares(32, com_duros=True), _pares(16), saida)
    assert (saida / "estado_treino.pt").exists()
    assert (saida / "progresso.json").exists()
