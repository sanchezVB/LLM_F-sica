"""Os itens de física do G2 e a correção de cada benchmark (DOC-12 §3.7.2 e §3.7.4).

`carregar(bench, conjunto)` devolve os itens de um conjunto da partição
(`g2_particao.json`). `corrigir(item, texto)` extrai a resposta da geração e aplica o
critério do benchmark:

| bench | critério |
|---|---|
| ugphysics | NV a 1e-2 relativo ao gabarito; EX, EQ, IN, MC e TF pelos comparadores |
| olympiadbench | por resposta: Numerical a ±`error` absoluto (sem `error`, 1e-2 relativo); Expression, Equation e Interval pelos comparadores; várias respostas em qualquer ordem |
| phybench | equivalência simbólica com variáveis positivas |
| scibench | o número a `isclose(rel_tol=0,05)` contra answer_number × 10^x da unidade |

Ler um item de TESTE aqui não é medir: gerar no teste exige
`g2.exigir_aceite_e_congelamento()`, e quem gera chama isso antes. O conteúdo dos itens
nunca vai para o git nem para log.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import sympy as sp

from phifm.eval import g2_corretor as c
from phifm.eval.g2_corretor import Veredito

RAIZ = Path(__file__).resolve().parents[3]
BENCH = RAIZ / "data/raw/benchmarks"
PARTICAO = RAIZ / "data/processed/avaliacao/g2_particao.json"

TOL_UG = 1e-2
TOL_OLY_SEM_ERRO = 1e-2
TOL_SCI = 0.05


@dataclass(frozen=True)
class Item:
    bench: str
    id: str
    enunciado: str
    gabaritos: tuple[str, ...]          # um por resposta
    tipos: tuple[str, ...]              # um por resposta
    tolerancias: tuple[float | None, ...] = ()
    unidade: str | None = None          # a que o prompt mostra
    solucao: str | None = None
    extra: dict = field(default_factory=dict, compare=False, hash=False)


# ── carregadores ─────────────────────────────────────────────────────────────


def _particao() -> dict:
    return json.loads(PARTICAO.read_text(encoding="utf-8"))


def _sem_dolar(s: str) -> str:
    return c.limpar(str(s))


def ugphysics(conjunto: str) -> list[Item]:
    ids = set(_particao()["ugphysics"][conjunto])
    itens = []
    for arq in sorted((BENCH / "ugphysics").glob("*/en.jsonl")):
        for linha in arq.read_text(encoding="utf-8").splitlines():
            d = json.loads(linha)
            i = f"{arq.parent.name}/{d['index']}"
            if i not in ids:
                continue
            gab = c.extrair_boxed(d["answers"])
            gab = gab if gab is not None else _sem_dolar(d["answers"])
            tipo = str(d["answer_type"]).strip()
            # `a=\frac{\gamma}{\eta}, b=\frac{\tau}{\eta}` com um rótulo EQ só: são duas
            # equações, corrigidas como várias respostas (correção de dev, §3.7.4)
            partes = c._partes(c.limpar(gab)) if tipo == "EQ" else [gab]
            if len(partes) < 2 or not all("=" in p for p in partes):
                partes = [gab]
            unidade = d.get("unit")
            itens.append(Item(
                "ugphysics", i, d["problem"], tuple(partes), (tipo,) * len(partes),
                (None,) * len(partes),
                unidade.strip() if isinstance(unidade, str) and unidade.strip() else None,
                d.get("solution"), {"materia": arq.parent.name, "nivel": d.get("level")}))
    return itens


def olympiadbench(conjunto: str) -> list[Item]:
    import pyarrow.parquet as pq

    sub = {"teste": "OE_TO_physics_en_COMP", "dev": "OE_TO_physics_zh_CEE"}[conjunto]
    linhas = pq.read_table(BENCH / f"olympiadbench/OlympiadBench/{sub}/{sub}.parquet").to_pylist()
    itens = []
    for x in linhas:
        partes = [p for f in x["final_answer"] for p in c._partes(_sem_dolar(f))]
        tipos = [t.strip() for t in str(x["answer_type"]).split(",")]
        if len(tipos) != len(partes):
            tipos = [tipos[0]] * len(partes)
        erros: list[float | None] = []
        for e in (str(x["error"]).split(",") if x.get("error") not in (None, "") else []):
            try:
                erros.append(float(e))
            except ValueError:
                erros.append(None)
        if len(erros) != len(partes):
            erros = (erros[:1] * len(partes)) if erros else [None] * len(partes)
        contexto = (x.get("context") or "").strip()
        enunciado = f"{contexto}\n\n{x['question']}" if contexto else x["question"]
        itens.append(Item(
            "olympiadbench", str(x["id"]), enunciado, tuple(partes), tuple(tipos), tuple(erros),
            x.get("unit") or None, (x["solution"][0] if x.get("solution") else None),
            {"multiplo": bool(x.get("is_multiple_answer")), "n_respostas": len(partes)}))
    return itens


def phybench(conjunto: str = "teste") -> list[Item]:
    if conjunto != "teste":
        raise ValueError("o PHYBench é 100% teste (§3.7.3)")
    dados = json.loads((BENCH / "phybench/PHYBench-fullques_v1.json").read_text(encoding="utf-8"))
    itens = []
    for x in dados:
        if not str(x.get("answer", "")).strip():
            continue
        gab = c.extrair_boxed(x["answer"])
        itens.append(Item("phybench", str(x["id"]), x["content"],
                          (gab if gab is not None else _sem_dolar(x["answer"]),), ("EX",), (None,),
                          None, x.get("solution"), {"tag": x.get("tag")}))
    return itens


_ESCALA = re.compile(r"[$]?\s*10\s*\^\s*[{]?\s*(-?[0-9]+)\s*[}]?\s*[$]?")


def escala_da_unidade(unidade: str) -> tuple[float, str]:
    """(10^x, a unidade sem o 10^x), como o `cal_not`/`remove_not` do SciBench."""
    m = _ESCALA.search(unidade or "")
    if not m:
        return 1.0, (unidade or "").strip()
    return 10.0 ** int(m.group(1)), _ESCALA.split(unidade)[-1].strip()


def scibench(conjunto: str) -> list[Item]:
    sufixo = {"teste": "", "dev": "_sol"}[conjunto]
    itens = []
    for livro in ("class", "fund", "thermo"):
        dados = json.loads((BENCH / f"scibench/{livro}{sufixo}.json").read_text(encoding="utf-8"))
        for x in dados:
            escala, unidade = escala_da_unidade(str(x.get("unit", "")))
            numero_oficial = str(x["answer_number"]).strip()
            valor = float(numero_oficial) * escala
            expoente = round(math.log10(escala))
            exibida = (numero_oficial if expoente == 0
                       else f"{numero_oficial} \\times 10^{{{expoente}}}")
            itens.append(Item("scibench", f"{livro}/{x['problemid']}", x["problem_text"],
                              (repr(valor),), ("NV",), (None,), unidade or None,
                              x.get("solution"), {"livro": livro, "escala": escala,
                                                  "resposta_exibida": exibida}))
    return itens


CARREGADORES: dict[str, Callable[[str], list[Item]]] = {
    "ugphysics": ugphysics, "olympiadbench": olympiadbench, "phybench": phybench,
    "scibench": scibench,
}


def carregar(bench: str, conjunto: str) -> list[Item]:
    return CARREGADORES[bench](conjunto)


# ── correção ─────────────────────────────────────────────────────────────────


def tipo_efetivo(item: Item, k: int = 0) -> str:
    """O tipo pela FORMA do gabarito, quando o rótulo do UGPhysics não bate com ela
    (correção de dev registrada no DOC-12 §3.7.4, igual nos dois braços):

    - NV cujo gabarito não é número (`\\frac{B \\pi a^2}{R}`) vira EX;
    - EQ do tipo `grandeza = número` (`\\tau = 4.5\\times10^9`) vira NV pelo lado direito;
    - EQ do tipo `grandeza = expressão` (`\\kappa(p, T) = …`) vira EX pelo lado direito:
      é a definição da grandeza, e a resposta pode trazer o lado esquerdo ou não;
    - EQ sem `=` vira EX.

    No OlympiadBench, "Numerical" com gabarito simbólico vira "Expression".
    """
    tipo = item.tipos[k]
    g = item.gabaritos[k]
    if item.bench == "olympiadbench":
        return "Expression" if tipo == "Numerical" and c.numero(g, estrito=True) is None else tipo
    if item.bench != "ugphysics":
        return tipo
    if tipo == "NV" and c.numero(g, estrito=True) is None:
        return "EX"
    if tipo == "EQ":
        lados = c.dividir_nivel_zero(c.limpar(g), "=")
        if len(lados) == 1:
            return "EX"
        if c.lado_esquerdo_e_grandeza(g):
            return "NV" if c.numero(lados[1], estrito=True) is not None else "EX"
    return tipo


def _comparador(item: Item, k: int) -> Callable[[str, str], Veredito]:
    tipo, tol = tipo_efetivo(item, k), item.tolerancias[k] if item.tolerancias else None
    # Gabarito `G = expr` numa resposta simbólica: vale o lado direito (com G conferido)
    # ou a mesma equação rearranjada (`comparar_definicao`).
    expressao = (c.comparar_definicao if c.lado_esquerdo_e_grandeza(item.gabaritos[k])
                 else c.comparar_expressao)
    if item.bench == "scibench":
        return partial(c.comparar_numero, rel=TOL_SCI, simetrico=True)
    if item.bench == "phybench":
        return expressao
    if item.bench == "olympiadbench":
        if tipo == "Numerical":
            return (partial(c.comparar_numero, abs_=tol) if tol is not None
                    else partial(c.comparar_numero, rel=TOL_OLY_SEM_ERRO))
        return {"Equation": c.comparar_equacao,
                "Interval": c.comparar_intervalo}.get(tipo, expressao)
    return {"NV": partial(c.comparar_numero, rel=TOL_UG), "EX": expressao,
            "EQ": c.comparar_equacao, "IN": c.comparar_intervalo, "MC": c.comparar_letras,
            "TF": c.comparar_vf}[tipo]


def corrigir_resposta(item: Item, resposta: str) -> Veredito:
    """A resposta já extraída, contra o gabarito, sob o teto de tempo."""
    comps = [_comparador(item, k) for k in range(len(item.gabaritos))]
    if len(item.gabaritos) == 1:
        return c.com_teto(lambda: comps[0](resposta, item.gabaritos[0]))
    return c.com_teto(lambda: c.comparar_varias(resposta, list(item.gabaritos), comps))


def corrigir(item: Item, texto: str) -> Veredito:
    resposta = c.extrair_respostas(texto, len(item.gabaritos))
    if resposta is None:
        return Veredito.SEM_RESPOSTA
    return corrigir_resposta(item, resposta)


# ── autoteste do gabarito (§3.7.3, item 1) ───────────────────────────────────


def _numero_perturbado(item: Item, k: int) -> list[str]:
    v = c.numero(item.gabaritos[k], estrito=True)
    if v is None:
        return []
    tol = item.tolerancias[k] if item.tolerancias else None
    if item.bench == "olympiadbench" and tol is not None:
        return [repr(v + 3 * tol), repr(v - 3 * tol)]
    rel = {"scibench": TOL_SCI, "olympiadbench": TOL_OLY_SEM_ERRO}.get(item.bench, TOL_UG)
    if v == 0:
        return ["1.0", "-1.0"]
    return [repr(v * (1 + 3 * rel)), repr(v * (1 - 3 * rel))]


def _simbolo_trocado(gabarito: str) -> list[str]:
    e = c.expressao(c._depois_do_igual(c.limpar(gabarito)))
    if e is None or not e.free_symbols:
        return []
    s0 = sorted(e.free_symbols, key=str)[0]
    return [sp.latex(e.xreplace({s0: sp.Symbol("zz")}))]


def perturbacoes(item: Item, k: int = 0) -> list[str]:
    """Respostas erradas construídas do próprio gabarito k, que o corretor TEM de
    reprovar: número deslocado por 3× a tolerância do item; expressão com +1 e com um
    símbolo trocado; equação com +1; intervalo com o colchete trocado; outra letra; o V/F
    oposto."""
    g, tipo = item.gabaritos[k], tipo_efetivo(item, k)
    if item.bench == "scibench" or tipo in ("NV", "Numerical"):
        return _numero_perturbado(item, k)
    if tipo in ("EX", "Expression") or item.bench == "phybench":
        # "0" pega o gabarito que o parser lê como zero (aconteceu com `v^2_2`)
        e = c.expressao(c._depois_do_igual(c.limpar(g)))
        zero = ["0"] if e is not None and e.free_symbols else []
        return [f"{g} + 1", *_simbolo_trocado(g), *zero]
    if tipo in ("EQ", "Equation"):
        return [f"{g} + 1"]
    if tipo in ("IN", "Interval"):
        s = c.limpar(g)
        if c._RELACAO.search(s):
            return [f"{g} + 1"]
        troca = {"(": "[", "[": "(", ")": "]", "]": ")"}
        return [troca[s[0]] + s[1:]] if s and s[0] in troca else []
    if tipo == "MC":
        tem = c.letras(g)
        outra = next((x for x in "ABCDEFGHIJ" if x not in tem), None)
        return [outra] if outra else []
    if tipo == "TF":
        v = c.verdade(g)
        return [] if v is None else ["No" if v else "Yes"]
    return []


def autoteste(item: Item) -> dict:
    """O gabarito em `\\boxed{}` tem de dar PASS contra si mesmo, e cada perturbação, FAIL.
    Devolve só vereditos e contagens: nenhum texto do item sai daqui."""
    proprio = corrigir(item, "\\boxed{" + ", ".join(item.gabaritos) + "}")
    falhas_pert, n_pert = 0, 0
    for k in range(len(item.gabaritos)):
        for p in perturbacoes(item, k):
            partes = list(item.gabaritos)
            partes[k] = p
            n_pert += 1
            if corrigir(item, "\\boxed{" + ", ".join(partes) + "}") is not Veredito.FAIL:
                falhas_pert += 1
    ok = proprio is Veredito.PASS and n_pert > 0 and falhas_pert == 0
    return {"id": item.id, "ok": ok, "proprio": str(proprio), "perturbacoes": n_pert,
            "perturbacoes_que_passaram": falhas_pert}
