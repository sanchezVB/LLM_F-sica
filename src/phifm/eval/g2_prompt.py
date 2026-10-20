"""Os prompts de física do G2 (DOC-12 §3.7.4): modelo base, texto corrido, sem template de
chat. A mesma montagem vale para a base e para o ΦGen.

    The following are physics problems with step-by-step solutions. Each solution ends with: The final answer is $\\boxed{...}$.

    Problem: {enunciado do exemplo}
    {linha de resposta}
    Solution: {solução oficial} The final answer is $\\boxed{gabarito}$.

    ... (os outros exemplos)

    Problem: {enunciado do item}
    {linha de resposta}
    Solution:

A linha de resposta segue o TIPO EFETIVO do item (`g2_itens.tipo_efetivo`). Os exemplos
são fixos, escolhidos uma vez por `escolher_exemplos` e gravados por id com hash; nenhum é
irmão (13-grama) de item de teste.
"""
from __future__ import annotations

import difflib
import random
from collections.abc import Callable, Iterable

from phifm.eval import g2
from phifm.eval.g2_itens import Item, tipo_efetivo

CABECALHO = ("The following are physics problems with step-by-step solutions. Each solution "
             "ends with: The final answer is $\\boxed{...}$.\n\n")
PARADAS = ["\n\nProblem:", "\nProblem:", "<|endoftext|>", "<|im_end|>"]

_UG = {
    "EX": "The answer is an expression.",
    "EQ": "The answer is an equation.",
    "IN": "The answer is an interval.",
    "MC": "Answer with the option letter(s).",
    "TF": "Answer True or False.",
}
_OLY = {"Numerical": "a numerical value", "Expression": "an expression",
        "Equation": "an equation", "Interval": "an interval"}


def linha_de_resposta(item: Item) -> str:
    if item.bench == "ugphysics":
        t = tipo_efetivo(item, 0)
        if t == "NV":
            if item.unidade:
                return (f"The answer is a number in {item.unidade}; put only the number in "
                        "\\boxed{}.")
            return "The answer is a number; put only the number in \\boxed{}."
        if len(item.gabaritos) > 1:
            return (f"The answer has {len(item.gabaritos)} equations; put all of them in one "
                    "\\boxed{}, separated by commas.")
        return _UG[t]
    if item.bench == "olympiadbench":
        partes = [f"The answer is {_OLY.get(item.tipos[0], 'an expression')}."]
        if len(item.gabaritos) > 1:
            partes.append(f"This problem has {len(item.gabaritos)} answers; put all of them in "
                          "one \\boxed{}, separated by commas.")
        if item.unidade:
            partes.append("Give the unit outside the \\boxed{}.")
        return " ".join(partes)
    if item.bench == "phybench":
        return ("Give the final answer as a single symbolic expression in the given quantities, "
                "in one \\boxed{}.")
    if item.bench == "scibench":
        u = f"The unit of the answer is {item.unidade}. " if item.unidade else ""
        return f"{u}Give only the number in \\boxed{{}}."
    raise ValueError(item.bench)


def _gabarito_do_exemplo(item: Item) -> str:
    if item.bench == "scibench":
        return item.extra.get("resposta_exibida") or item.gabaritos[0]
    return ", ".join(item.gabaritos)


def formatar_exemplo(exemplo: Item, linha: str) -> str:
    """O exemplo com a solução oficial e a frase final acrescentada (as soluções do UG não
    trazem `\\boxed`)."""
    solucao = (exemplo.solucao or "").strip()
    return (f"Problem: {exemplo.enunciado.strip()}\n{linha}\nSolution: {solucao} The final answer "
            f"is $\\boxed{{{_gabarito_do_exemplo(exemplo)}}}$.\n\n")


def montar_prompt(item: Item, exemplos: Iterable[Item]) -> str:
    """O prompt inteiro. Cada exemplo leva a linha de resposta DELE, e o item, a dele."""
    corpo = "".join(formatar_exemplo(e, linha_de_resposta(e)) for e in exemplos)
    return f"{CABECALHO}{corpo}Problem: {item.enunciado.strip()}\n{linha_de_resposta(item)}\nSolution:"


# ── a escolha dos exemplos (§3.7.4) ──────────────────────────────────────────

# por benchmark: [(tipo efetivo, precisa de unidade?)], níveis aceitos e faixa de fichas
PEDIDOS = {
    "ugphysics": ([("NV", True), ("NV", False), ("EX", None), ("EQ", None)], None, (120, 600)),
    "olympiadbench": ([("NV", True), ("NV", True), ("EX", None), ("EX", None)],
                      {"Math Derivation", "Laws Application"}, (200, 800)),
    "phybench": ([("EX", None)] * 3, {"Math Derivation"}, (200, 800)),
}
SIMILARIDADE_MAX_SCI = 0.6


def irmao_de_teste(texto: str, ngramas_de_teste: set[str], n: int = 13) -> bool:
    return bool(g2.ngramas(texto, n) & ngramas_de_teste)


def escolher_exemplos(ug_dev: list[Item], sci_dev: list[Item], textos_de_teste: list[str],
                      textos_sci_teste: dict[str, list[str]], fichas: Callable[[Item], int],
                      gradavel: Callable[[Item], bool], semente: int = g2.SEMENTE_G2
                      ) -> dict[str, list[str]]:
    """Os ids dos exemplos, por benchmark. UG, OlympiadBench e PHYBench tiram do UG-dev:
    candidatos com nível, solução na faixa de fichas, gabarito que o próprio corretor
    aprova (`gradavel`) e nenhum 13-grama em comum com item de teste; sorteio pela semente,
    matérias diferentes dentro do benchmark. SciBench: 3 `_sol` por livro, sem 10-grama em
    comum nem similaridade ≥ 0,6 com item de teste do mesmo livro."""
    teste13 = {g for t in textos_de_teste for g in g2.ngramas(t, 13)}
    rng = random.Random(semente)
    escolha: dict[str, list[str]] = {}
    for bench, (pedidos, niveis, (lo, hi)) in PEDIDOS.items():
        usados_materia: set[str] = set()
        ids: list[str] = []
        for tipo, com_unidade in pedidos:
            cands = sorted(
                (i for i in ug_dev
                 if i.id not in ids and len(i.gabaritos) == 1
                 and tipo_efetivo(i) == tipo
                 and (com_unidade is None or bool(i.unidade) == com_unidade)
                 and i.extra.get("nivel") and (niveis is None or i.extra["nivel"] in niveis)
                 and i.extra.get("materia") not in usados_materia
                 and lo <= fichas(i) <= hi),
                key=lambda i: i.id)
            rng.shuffle(cands)
            for i in cands:
                if gradavel(i) and not irmao_de_teste(f"{i.enunciado}\n{i.solucao}", teste13):
                    ids.append(i.id)
                    usados_materia.add(i.extra["materia"])
                    break
            else:
                raise RuntimeError(f"sem candidato para {bench} {tipo} (unidade={com_unidade})")
        escolha[bench] = ids
    sci: list[str] = []
    for livro in ("class", "fund", "thermo"):
        teste10 = {g for t in textos_sci_teste[livro] for g in g2.ngramas(t, 10)}
        cands = sorted((i for i in sci_dev if i.extra["livro"] == livro), key=lambda i: i.id)
        limpos = [i for i in cands
                  if not g2.ngramas(i.enunciado, 10) & teste10
                  and max((difflib.SequenceMatcher(None, i.enunciado, t).ratio()
                           for t in textos_sci_teste[livro]), default=0) < SIMILARIDADE_MAX_SCI
                  and gradavel(i)]
        rng.shuffle(limpos)
        if len(limpos) < 3:
            raise RuntimeError(f"SciBench {livro}: só {len(limpos)} _sol limpos")
        sci += [i.id for i in limpos[:3]]
    escolha["scibench"] = sci
    return escolha
