"""O EvalPlus do G2.3 (DOC-12 §3.7.4): a geração aqui, a execução só no Kaggle.

- **Dados.** Os releases oficiais (`data/raw/benchmarks/evalplus-oficial`): HumanEval+
  v0.1.10 (164) e MBPP+ v0.2.0 (378), os mesmos que o `evalplus.evaluate` lê. Os espelhos
  do Hugging Face não servem: o prompt do MBPP+ deles é outro em 18 tarefas, e eles não
  separam as entradas *base* das *plus*.
- **Consulta.** Completação direta, 0-shot, como o `evalplus.codegen`: `prompt.strip() +
  "\\n"`. As paradas são as do `evalplus.provider.utility` (EOS mais as do conjunto).
- **Amostragem.** k=10 a T=0,2 e top-p 0,95, mais 1 gulosa; até 768 fichas. A guarda de
  repetição do G2 vale aqui também.
- **Solução.** Como no `evalplus.codegen`: consulta + geração (tabs viram 4 espaços), pelo
  `evalplus.sanitize` com o ponto de entrada da tarefa.
- **Acerto.** Passar nos testes *base* e *plus* (o "+" do EvalPlus). pass@1 = média de c/n.
  O HumanEval original sai das mesmas gerações (o *base* do HumanEval+).
- **Sanidade.** A média gulosa da base fica a ±5 de 52,70 (1,7B) e 67,65 (8B). A
  referência é a do relatório do Qwen3, que chama de EvalPlus a média de HumanEval, MBPP,
  HumanEval+ e MBPP+; a sanidade usa essa mesma média.
"""
from __future__ import annotations

import importlib
import json
import statistics
import sys
import types
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
DADOS = RAIZ / "data/raw/benchmarks/evalplus-oficial"
ARQUIVOS = {"humaneval": "HumanEvalPlus-v0.1.10.jsonl", "mbpp": "MbppPlus-v0.2.0.jsonl"}
K = 10
FICHAS_MAX = 768
REFERENCIA_GULOSA = {"1.7B": 52.70, "8B": 67.65}
FOLGA_DA_REFERENCIA = 5.0


def amostragem(gulosa: bool = False) -> dict:
    from phifm.eval.g2_geracao import amostragem as base

    return base(temperatura=0.0, top_p=1.0) if gulosa else base(temperatura=0.2, top_p=0.95)


def carregar(conjunto: str, pasta: Path = DADOS) -> list[dict]:
    texto = (Path(pasta) / ARQUIVOS[conjunto]).read_text(encoding="utf-8")
    return [json.loads(x) for x in texto.split("\n") if x.strip()]


def consulta(tarefa: dict) -> str:
    return tarefa["prompt"].strip() + "\n"


def paradas(conjunto: str) -> list[str]:
    from evalplus.provider.utility import EOS, extra_eos_for_direct_completion

    return [*EOS, *extra_eos_for_direct_completion(conjunto)]


def _sanitizador():
    """O `evalplus.sanitize`. O `evalplus.data` importa o `datasets` só para o EvalPerf,
    que não entra: sem ele instalado, um módulo vazio no lugar, que recusa se chamado."""
    try:
        importlib.import_module("datasets")
    except ImportError:
        falso = types.ModuleType("datasets")

        def _recusa(*_a, **_k):
            raise RuntimeError("o G2 não usa o EvalPerf")

        falso.load_dataset = _recusa
        sys.modules["datasets"] = falso
    from evalplus.sanitize import sanitize

    return sanitize


def solucao(tarefa: dict, texto: str) -> str:
    bruta = consulta(tarefa) + texto.replace("\t", "    ")
    return _sanitizador()(bruta, entrypoint=tarefa["entry_point"])


def agregar(resumo: dict) -> dict:
    """Os agregados de um resumo do Kaggle ({conjunto: {task_id: {n, c_base, c_plus}}}),
    em pontos. `plus_542` é o que decide; `media_4` é a sanidade."""
    r: dict = {}
    taxas_plus = []
    for conjunto, tarefas in resumo.items():
        base = [t["c_base"] / t["n"] for t in tarefas.values()]
        plus = [t["c_plus"] / t["n"] for t in tarefas.values()]
        taxas_plus += plus
        r[conjunto] = {"tarefas": len(tarefas), "n_por_tarefa": sorted({t["n"] for t in tarefas.values()}),
                       "base": round(100 * statistics.mean(base), 2),
                       "plus": round(100 * statistics.mean(plus), 2)}
    r["plus_542"] = round(100 * statistics.mean(taxas_plus), 2) if taxas_plus else None
    if {"humaneval", "mbpp"} <= set(resumo):
        r["media_4"] = round(statistics.mean(
            r[c][s] for c in ("humaneval", "mbpp") for s in ("base", "plus")), 2)
    return r


def sanidade_gulosa(modelo: str, media_4: float) -> dict:
    ref = REFERENCIA_GULOSA[modelo]
    return {"referencia": ref, "folga": FOLGA_DA_REFERENCIA, "medida": media_4,
            "ok": abs(media_4 - ref) <= FOLGA_DA_REFERENCIA}
