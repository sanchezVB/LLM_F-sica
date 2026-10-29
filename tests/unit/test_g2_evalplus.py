"""O EvalPlus do G2.3: a consulta e as paradas oficiais, a sanitização e os agregados, com
tarefas SINTÉTICAS (nada aqui executa código gerado: a execução é só no Kaggle)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_evalplus as f  # noqa: E402

pytest.importorskip("evalplus")


def test_consulta_e_paradas_como_o_codegen_oficial():
    t = {"prompt": "\n\ndef dobro(x):\n    \"\"\"O dobro.\"\"\"\n\n", "entry_point": "dobro"}
    assert f.consulta(t) == 'def dobro(x):\n    """O dobro."""\n'
    he, mb = f.paradas("humaneval"), f.paradas("mbpp")
    assert he[:3] == ["<|endoftext|>", "<|endofmask|>", "</s>"]
    assert "\ndef " in he and "\nassert " in he and "\ndef " not in mb
    assert mb[-2:] == ['\n"""', "\nassert"]


def test_solucao_sanitizada_pelo_ponto_de_entrada():
    t = {"prompt": 'def dobro(x):\n    """O dobro."""\n', "entry_point": "dobro"}
    s = f.solucao(t, "\treturn 2 * x\n\n\nprint(dobro(3))\nlixo = (")
    assert "def dobro(x):" in s and "    return 2 * x" in s
    assert "print" not in s and "lixo" not in s and "\t" not in s


def test_agregados_e_sanidade():
    resumo = {"humaneval": {"HumanEval/0": {"n": 10, "c_base": 10, "c_plus": 8},
                            "HumanEval/1": {"n": 10, "c_base": 4, "c_plus": 2}},
              "mbpp": {"Mbpp/2": {"n": 10, "c_base": 6, "c_plus": 6}}}
    r = f.agregar(resumo)
    assert r["humaneval"] == {"tarefas": 2, "n_por_tarefa": [10], "base": 70.0, "plus": 50.0}
    assert r["mbpp"]["plus"] == 60.0
    assert r["plus_542"] == round(100 * (0.8 + 0.2 + 0.6) / 3, 2)      # média por problema
    assert r["media_4"] == round((70 + 50 + 60 + 60) / 4, 2)
    assert f.sanidade_gulosa("1.7B", 56.0)["ok"] and not f.sanidade_gulosa("1.7B", 47.0)["ok"]


@pytest.mark.skipif(not (f.DADOS / f.ARQUIVOS["mbpp"]).exists(), reason="dados oficiais fora")
def test_releases_oficiais():
    assert len(f.carregar("humaneval")) == 164
    mbpp = f.carregar("mbpp")
    assert len(mbpp) == 378 and all(t["prompt"].startswith('"""\n') for t in mbpp)
