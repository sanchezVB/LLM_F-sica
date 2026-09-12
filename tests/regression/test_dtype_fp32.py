"""Base fp16 + AMP = `Attempting to unscale FP16 gradients`, e custou DUAS vezes.

`thenlper/gte-base` guarda os pesos em fp16 e o `transformers` novo carrega no
dtype do checkpoint. O AMP exige pesos-mestres em fp32 — é ele que faz a passagem
em fp16, e o `GradScaler` que desescala os gradientes de volta. Um modelo já em
fp16 não tem para onde desescalar.

⚠️ O histórico é o ponto deste arquivo:

  * **2026-09-03, T1c** — o braço `gte` do reordenador morreu com esse erro. Foi
    diagnosticado, escrito numa nota de 10 linhas, e consertado em `rerank.py`.
  * **2026-09-11, T1f** — o treino do ΦEmb com base `gte-base` morreu aos 2 min
    com o MESMO erro. `embedding.py` nunca recebeu a correção, porque nenhuma base
    fp16 tinha passado por ele: MiniLM e SciBERT são fp32.

A lição estava paga, documentada e aplicada **num arquivo só**. Este teste cobre os
dois módulos de treino de uma vez, para a terceira vez não existir.

⚠️ Por AST: a suíte rápida roda sem torch.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]

# Os módulos que carregam uma base do Hub E treinam com AMP.
MODULOS = ("src/phifm/training/embedding.py", "src/phifm/training/rerank.py")


@pytest.mark.parametrize("rel", MODULOS)
def test_o_modelo_e_carregado_em_fp32_explicito(rel):
    """Sem `dtype=torch.float32`, uma base fp16 derruba o treino no primeiro
    passo de otimizador — depois de a sessão já ter sido alocada."""
    fonte = (RAIZ / rel).read_text(encoding="utf-8")
    chamadas = [n for n in ast.walk(ast.parse(fonte))
                if isinstance(n, ast.Call)
                and ast.unparse(n.func).endswith(".from_pretrained")
                and "Tokenizer" not in ast.unparse(n.func)]
    assert chamadas, f"{rel} não carrega modelo nenhum"
    for c in chamadas:
        dtypes = [k for k in c.keywords if k.arg == "dtype"]
        assert dtypes, (
            f"{rel}: `{ast.unparse(c.func)}` sem `dtype`. Uma base fp16 "
            "(thenlper/gte-base é uma) quebraria em "
            "`Attempting to unscale FP16 gradients`.")
        assert ast.unparse(dtypes[0].value) == "torch.float32", (
            f"{rel}: dtype é {ast.unparse(dtypes[0].value)}, e o AMP exige "
            "pesos-mestres em fp32")


@pytest.mark.parametrize("rel", MODULOS)
def test_a_razao_esta_escrita_junto(rel):
    """⚠️ `dtype=torch.float32` parece redundante — fp32 é "o padrão", e alguém
    vai querer limpar. A nota é o que impede a terceira vez."""
    fonte = (RAIZ / rel).read_text(encoding="utf-8")
    assert "unscale FP16" in fonte, (
        f"{rel} não diz POR QUE o dtype é explícito; sem isso ele vira ruído "
        "que o próximo removedor de redundância apaga")
