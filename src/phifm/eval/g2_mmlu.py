"""O MMLU do G2.3 (DOC-12 §3.7.4): 5-shot com os 5 itens de dev da matéria, uma chamada por
item, e a previsão é a maior probabilidade entre as fichas " A", " B", " C" e " D".

- **O que decide:** o MMLU sem física (13.402 itens). Relatados: o MMLU de 57 matérias, que é
  também a guarda do instrumento (a base a ±3 pontos de 62,63 na 1,7B e de 76,89 na 8B), e o
  MMLU-física (astronomy, college_physics, conceptual_physics, high_school_physics: 640).
- **O formato** é o clássico do MMLU (Hendrycks et al.): o cabeçalho da matéria, as 5
  demonstrações com "Answer: X" e a pergunta terminando em "Answer:".
- **A previsão** vem dos logprobs da próxima ficha (`n_probs` do llama-server), a temperatura
  0. Letra fora do top-N conta como −∞; nenhuma das quatro no top-N é ERRO, e o item vale 0.

Nenhum texto de item sai daqui para o git; só ids, letras e agregados.
"""
from __future__ import annotations

import collections
import math
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[3]
DADOS = RAIZ / "data/raw/benchmarks/mmlu/all"
FISICA = ("astronomy", "college_physics", "conceptual_physics", "high_school_physics")
LETRAS = ("A", "B", "C", "D")
# as fichas da letra com o espaço antes, no tokenizador da Qwen3 (g2.yaml, g2_3.mmlu_sem_fisica)
FICHAS_DAS_LETRAS = {"A": 362, "B": 425, "C": 356, "D": 422}
N_PROBS = 20


def carregar(split: str) -> list[dict]:
    """Os itens de um split ("dev", "validation", "test"), com um id estável por posição."""
    import pyarrow.parquet as pq

    linhas = pq.read_table(DADOS / f"{split}-00000-of-00001.parquet").to_pylist()
    return [{"id": f"{split}/{k}", **x} for k, x in enumerate(linhas)]


def _materia(s: str) -> str:
    return s.replace("_", " ")


def formatar(item: dict, com_resposta: bool) -> str:
    texto = item["question"].strip() + "\n"
    texto += "".join(f"{letra}. {str(c).strip()}\n" for letra, c in zip(LETRAS, item["choices"], strict=True))
    texto += "Answer:"
    if com_resposta:
        texto += f" {LETRAS[item['answer']]}\n\n"
    return texto


def exemplos_por_materia(dev: list[dict]) -> dict[str, list[dict]]:
    por = collections.defaultdict(list)
    for x in dev:
        por[x["subject"]].append(x)
    return dict(por)


def montar_prompt(item: dict, exemplos: list[dict]) -> str:
    cabecalho = ("The following are multiple choice questions (with answers) about "
                 f"{_materia(item['subject'])}.\n\n")
    return cabecalho + "".join(formatar(e, True) for e in exemplos) + formatar(item, False)


def letra_prevista(top: list[dict]) -> tuple[str | None, dict[str, float]]:
    """A letra de maior logprob entre as quatro, e os logprobs (−∞ se fora do top-N), a partir
    da lista `top_logprobs` do llama-server ([{"id", "token", "logprob"}, …])."""
    por_id = {t["id"]: t["logprob"] for t in top if "id" in t}
    lp = {letra: por_id.get(f, -math.inf) for letra, f in FICHAS_DAS_LETRAS.items()}
    melhor = max(lp, key=lp.get)
    return (melhor if lp[melhor] > -math.inf else None), lp


def agregar(itens: list[dict], previstas: dict[str, str | None]) -> dict:
    """Acerto no MMLU sem física (o que decide), nas 57 matérias (a guarda), na física e por
    matéria. Item sem previsão vale 0."""
    def acerto(sub: list[dict]) -> float | None:
        if not sub:
            return None
        return round(sum(previstas.get(x["id"]) == LETRAS[x["answer"]] for x in sub) / len(sub), 4)

    sem_fisica = [x for x in itens if x["subject"] not in FISICA]
    fisica = [x for x in itens if x["subject"] in FISICA]
    por_materia = collections.defaultdict(list)
    for x in itens:
        por_materia[x["subject"]].append(x)
    return {
        "sem_fisica": {"itens": len(sem_fisica), "acerto": acerto(sem_fisica)},
        "mmlu_57": {"itens": len(itens), "acerto": acerto(itens)},
        "fisica": {"itens": len(fisica), "acerto": acerto(fisica)},
        "sem_previsao": sum(previstas.get(x["id"]) is None for x in itens),
        "por_materia": {m: {"itens": len(v), "acerto": acerto(v)}
                        for m, v in sorted(por_materia.items())},
    }
