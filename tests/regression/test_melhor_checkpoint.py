"""O melhor checkpoint não pode ser sobrescrito por um pior.

Regressão de 2026-08-10. No treino do ΦEmb o pico de recall@1 foi 0,461 no passo
~38.000 e o modelo entregue no fim media 0,441: `salvar` grava sempre no mesmo
diretório, e o melhor havia sido sobrescrito. Naquele caso a diferença era ruído
(±0,031 de erro padrão com 256 candidatos), então nada de valor se perdeu — mas
por sorte, não por projeto.

O caso que estes testes protegem é o da RETOMADA: aquele treino foi relançado dez
vezes num único dia, e com `_melhor_mrr` reiniciando em -1 a primeira avaliação
de cada relançamento sobrescreveria o melhor com o que estivesse à mão.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# `torch` vive só na venv de TREINO (Python 3.12) — o `torch-directml` não
# suporta o 3.14 da venv principal, que é onde o CI e a suíte de dados rodam.
# Salta com motivo declarado em vez de quebrar; a verificação de que estes
# testes realmente PASSAM é feita rodando a suíte na venv de treino, o que está
# registrado no commit. Salto silencioso seria o mesmo defeito de sempre —
# ausência de erro lida como sucesso.
pytest.importorskip("torch", reason="requer a venv de treino (.venv-treino, Python 3.12)")


class TreinadorFalso:
    """Só as peças que `_talvez_melhor` e `retomar` usam.

    Evita carregar SciBERT e o DirectML num teste de unidade: o que se testa é a
    regra de retenção, não o encoder.
    """

    def __init__(self, tmp: Path):
        from dataclasses import dataclass, field

        from phifm.training.embedding import Config, Metricas, TreinadorEmb

        @dataclass
        class ModFalso:
            gravou_em: list = field(default_factory=list)

            def state_dict(self):
                return {"peso": 1}

            def save_pretrained(self, destino, state_dict=None):
                Path(destino).mkdir(parents=True, exist_ok=True)
                (Path(destino) / "model.safetensors").write_bytes(b"x")
                self.gravou_em.append(str(destino))

        class TokFalso:
            def save_pretrained(self, destino):
                (Path(destino) / "tokenizer.json").write_text("{}", encoding="utf-8")

        self.t = TreinadorEmb.__new__(TreinadorEmb)
        self.t.cfg = Config(n_candidatos=256)
        self.t.mod = ModFalso()
        self.t.tok = TokFalso()
        self.t._melhor_mrr = -1.0
        self.m = Metricas()
        self.saida = tmp / "phiemb"
        self.saida.mkdir(parents=True, exist_ok=True)

    def aval(self, passo: int, r1: float, r10: float, mrr: float):
        self.t._talvez_melhor(self.saida, self.m, passo, r1, r10, mrr)

    @property
    def melhor(self) -> dict | None:
        p = self.saida.parent / f"{self.saida.name}-melhor" / "melhor.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def test_primeira_avaliacao_vira_o_melhor(tmp_path):
    t = TreinadorFalso(tmp_path)
    t.aval(500, 0.30, 0.70, 0.42)
    assert t.melhor["passo"] == 500
    assert t.melhor["mrr"] == pytest.approx(0.42)


def test_pior_nao_sobrescreve(tmp_path):
    t = TreinadorFalso(tmp_path)
    t.aval(500, 0.40, 0.85, 0.55)
    t.aval(1000, 0.30, 0.70, 0.42)     # piorou
    assert t.melhor["passo"] == 500, "o pior sobrescreveu o melhor"
    assert t.melhor["mrr"] == pytest.approx(0.55)


def test_melhor_sobrescreve(tmp_path):
    t = TreinadorFalso(tmp_path)
    t.aval(500, 0.40, 0.85, 0.55)
    t.aval(1000, 0.44, 0.90, 0.60)
    assert t.melhor["passo"] == 1000
    assert t.melhor["mrr"] == pytest.approx(0.60)


def test_empate_nao_sobrescreve(tmp_path):
    """Empate mantém o mais ANTIGO: chegar ao mesmo MRR com menos passos é
    igual ou melhor, e trocar por trocar só gera escrita de 440 MB."""
    t = TreinadorFalso(tmp_path)
    t.aval(500, 0.40, 0.85, 0.55)
    t.aval(1000, 0.40, 0.85, 0.55)
    assert t.melhor["passo"] == 500


def test_criterio_e_mrr_nao_recall_1(tmp_path):
    """recall@1 é proporção binária: com 256 candidatos cada acerto vale 0,004 e
    a métrica pula em degraus. MRR usa a posição e é bem menos ruidoso — escolher
    por recall@1 seria escolher pelo maior ruído.

    Aqui o passo 1000 tem recall@1 melhor e MRR pior. O melhor deve continuar
    sendo o 500.
    """
    t = TreinadorFalso(tmp_path)
    t.aval(500, 0.40, 0.90, 0.60)
    t.aval(1000, 0.45, 0.80, 0.55)
    assert t.melhor["passo"] == 500, "escolheu por recall@1 em vez de MRR"


def test_registro_do_melhor_e_interpretavel(tmp_path):
    """Sem `n_candidatos` e `base`, o número gravado não é comparável a nada."""
    t = TreinadorFalso(tmp_path)
    t.aval(500, 0.40, 0.85, 0.55)
    d = t.melhor
    for chave in ("passo", "recall_1", "recall_10", "mrr", "n_candidatos", "base", "criterio"):
        assert chave in d, f"falta {chave} em melhor.json"


def test_retomada_preserva_o_melhor(tmp_path):
    """O caso que motivou tudo: dez relançamentos num dia.

    Um treinador NOVO, retomando, não pode começar com `_melhor_mrr = -1` — a
    primeira avaliação sobrescreveria o melhor com o que estivesse à mão.
    """
    from phifm.training.embedding import TreinadorEmb

    t1 = TreinadorFalso(tmp_path)
    t1.aval(38000, 0.461, 0.90, 0.624)     # o pico real do treino de 2026-08-09

    t2 = TreinadorFalso(tmp_path)           # simula o relançamento
    assert t2.t._melhor_mrr == -1.0, "o falso deve começar virgem"
    TreinadorEmb.retomar(t2.t, t2.saida)    # sem estado_treino.pt: só lê o melhor
    assert t2.t._melhor_mrr == pytest.approx(0.624), "não recuperou o melhor anterior"

    t2.aval(42000, 0.441, 0.895, 0.602)     # o que o treino media no fim
    assert t2.melhor["passo"] == 38000, "a retomada sobrescreveu o melhor com um pior"
    assert t2.melhor["mrr"] == pytest.approx(0.624)


def test_melhor_json_corrompido_nao_derruba_o_treino(tmp_path):
    """Arquivo ilegível vira aviso e busca reiniciada, não exceção — perder o
    histórico do melhor é ruim; abortar um treino de horas é pior."""
    from phifm.training.embedding import TreinadorEmb

    t = TreinadorFalso(tmp_path)
    d = t.saida.parent / f"{t.saida.name}-melhor"
    d.mkdir(parents=True, exist_ok=True)
    (d / "melhor.json").write_text("{ isto não é json", encoding="utf-8")

    TreinadorEmb.retomar(t.t, t.saida)      # não deve levantar
    assert t.t._melhor_mrr == -1.0
