"""A precisão de citação, medida: a fonte citada sustenta a frase? (DOC-13 §6.3, PROPOSTA de
2026-10-17; o critério G2.4 da §6.2)

O portão do assistente (`phifm.rag.assistente.verificar_citacoes`) remove citação para
fonte que não existe e marca frase sem citação. Ele NÃO confere se a fonte citada diz o
que a frase afirma — o *entailment* da §6.2, declarado como limite desde a §9.1 e nunca
medido. Este módulo mede.

## A unidade: o par (frase, fonte citada)

A resposta é cortada em frases pela MESMA regra do portão (`_FRASE`). Cada citação
válida `[n]` de uma frase vira um par com a fonte `n`. Citação colada no começo da frase
seguinte ("… 650 K. [1] Além disso …") volta para a frase anterior, que é a que ela cita.

## O juiz e o veredicto

O Qwen3-8B, temperatura 0, lê a pergunta (contexto), a frase sem as marcas [n] e a fonte
como o modelo a leu (título, ano, resumo), e diz `sustenta` · `parcial` · `nao_sustenta`.
**Só `sustenta` conta** — a meta é de precisão, e meia sustentação é uma afirmação sem
fonte para o resto.

Ele é o modelo que escreveu as respostas, e pode ser brando consigo. Por isso a guarda: o
Claude julga às cegas uma amostra de pares, com o hash no git antes de o juiz rodar nela,
e o κ entre os dois tem de passar do mínimo da §9.1.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from phifm.eval.assistente import KAPPA_MINIMO, SEMENTE
from phifm.eval.assistente_bracos import kappa
from phifm.rag.assistente import _CITACAO, _FRASE, _numeros, sem_citacoes

# ── a regra (DOC-13 §6.3) ────────────────────────────────────────────────────
REGRA_6_3_ACEITA = False
META = 0.95                       # o G2.4 da §6.2
N_VALIDACAO = 150                 # pares que o Claude julga por rodada
N_BOOTSTRAP = 10_000
MAX_RODADAS = 3                   # versões do prompt, no máximo, como no I3
VEREDICTOS = ("sustenta", "parcial", "nao_sustenta")

SISTEMA_SUSTENTA = (
    "You check ONE citation in an answer written by a physics research assistant. You get "
    "the question (for context only), ONE sentence of the answer (it may be in Portuguese), "
    "and the source that sentence cites: the title and abstract of a paper. Decide whether "
    "the source supports the sentence.\n\n"
    "- sustenta: every factual claim in the sentence is stated in the source or follows "
    "directly from it. A paraphrase, a translation, a rounded number or a standard unit "
    "conversion is fine.\n"
    "- parcial: the source supports some of the factual claims in the sentence but not all "
    "of them, or only a vaguer version of a claim.\n"
    "- nao_sustenta: the source does not state the claim, gives a different value, system "
    "or condition, contradicts it, or is about something else.\n\n"
    "Judge only against this source. Do not use your own knowledge: a claim that is true "
    "but absent from the source is not supported by it.\n"
    'Output only one JSON object: {"motivo": "<one short sentence>", '
    '"veredicto": "sustenta" | "parcial" | "nao_sustenta"}')

_CITACOES_NO_COMECO = re.compile(r"^(\s*" + _CITACAO.pattern + r")+")


def frases_citadas(texto: str, n_fontes: int) -> list[tuple[str, list[int]]]:
    """`[(frase sem as marcas, [n válidos])]`, só as frases com citação, na ordem.

    O corte é o do portão; a única diferença é devolver à frase anterior as citações que
    abrem uma frase — ninguém cita para a frente."""
    partes = [p.strip() for p in _FRASE.split(texto or "") if p.strip()]
    frases: list[list] = []
    for p in partes:
        m = _CITACOES_NO_COMECO.match(p)
        if m and frases:
            frases[-1][0] += " " + m.group(0).strip()
            p = p[m.end():].strip()
        if p:
            frases.append([p])
    saida = []
    for (f,) in frases:
        nums = [n for m in _CITACAO.finditer(f) for n in _numeros(m.group(1))]
        validos = list(dict.fromkeys(n for n in nums if 1 <= n <= n_fontes))
        if validos:
            saida.append((sem_citacoes(f).strip(), validos))
    return saida


def pares(texto: str, fontes: list[str]) -> list[dict]:
    """Os pares (frase, fonte citada) de uma resposta. `fontes`: os arxiv_id, na ordem do
    prompt."""
    return [{"frase_n": i, "frase": f, "n": n, "arxiv_id": fontes[n - 1]}
            for i, (f, nums) in enumerate(frases_citadas(texto, len(fontes))) for n in nums]


def bloco_da_fonte(titulo: str, ano, arxiv_id: str, resumo: str) -> str:
    """A fonte como o modelo a leu no prompt (`bloco_de_fontes`, sem o número)."""
    return f"{titulo} ({ano or 'n.d.'}, arXiv:{arxiv_id})\n{resumo}"


def usuario(pergunta: str, frase: str, fonte: str) -> str:
    return f"Question: {pergunta}\n\nSentence: {frase}\n\nSource:\n{fonte}"


def chave(pergunta: str, frase: str, fonte: str) -> str:
    return hashlib.sha256(f"{pergunta}\n{frase}\n{fonte}".encode()).hexdigest()[:24]


def ler_veredicto(bruto: str) -> tuple[str, str]:
    """`(veredicto, motivo)`; fora do formato, a última das três palavras; nenhuma →
    `ilegivel`, que conta como não sustentada e é relatado."""
    m = re.search(r"\{.*\}", bruto or "", re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            v = str(d.get("veredicto", "")).strip().lower().replace("ã", "a")
            if v in VEREDICTOS:
                return v, str(d.get("motivo", "")).strip()
        except json.JSONDecodeError:
            pass
    achados = re.findall(r"\b(sustenta|parcial|nao_sustenta|não_sustenta)\b", (bruto or "").lower())
    if achados:
        return achados[-1].replace("ã", "a"), (bruto or "").strip()[:300]
    return "ilegivel", (bruto or "").strip()[:300]


def julgar(modelo, pergunta: str, frase: str, fonte: str) -> tuple[str, str]:
    bruto = modelo.gerar(SISTEMA_SUSTENTA, usuario(pergunta, frase, fonte), max_tokens=200,
                         temperatura=0.0, semente=SEMENTE)
    return ler_veredicto(bruto)


def amostra_validacao(n_pares: int, rodada: int = 1, semente: int = SEMENTE) -> list[int]:
    """As posições dos `N_VALIDACAO` pares da rodada, numa permutação fixa: a rodada 2 pega
    os seguintes, disjuntos — a 1 vira desenvolvimento do prompt, como no I3."""
    if not 1 <= rodada <= MAX_RODADAS:
        raise ValueError(f"rodada {rodada}: a regra permite de 1 a {MAX_RODADAS}")
    ini = (rodada - 1) * N_VALIDACAO
    if n_pares < ini + N_VALIDACAO:
        raise ValueError(f"{n_pares} pares não bastam para a rodada {rodada}")
    perm = np.random.default_rng(semente + 6).permutation(n_pares)
    return [int(x) for x in perm[ini:ini + N_VALIDACAO]]


def validade(claude: list[str], juiz: list[str]) -> dict:
    """A guarda: κ de `sustenta` contra o resto, juiz × Claude, ≥ o mínimo da §9.1. A
    direção do desacordo vai junto: o juiz brando consigo mesmo diz `sustenta` onde o
    Claude não diz."""
    bc = ["s" if v == "sustenta" else "o" for v in claude]
    bj = ["s" if v == "sustenta" else "o" for v in juiz]
    k = kappa(bc, bj)
    return {"n": len(claude), "kappa_sustenta": round(k, 4), "minimo": KAPPA_MINIMO,
            "passa": k >= KAPPA_MINIMO,
            "juiz_sustenta_claude_nao": sum(a == "o" and b == "s" for a, b in zip(bc, bj,
                                                                                strict=True)),
            "claude_sustenta_juiz_nao": sum(a == "s" and b == "o" for a, b in zip(bc, bj,
                                                                                strict=True)),
            "precisao_pelo_claude": round(bc.count("s") / len(bc), 4),
            "precisao_pelo_juiz": round(bj.count("s") / len(bj), 4)}


def precisao(por_resposta: list[list[str]], n_bootstrap: int = N_BOOTSTRAP,
             semente: int = SEMENTE) -> tuple[float, tuple[float, float]]:
    """A fração de pares `sustenta`, com IC 95% por bootstrap das RESPOSTAS: os pares de
    uma mesma resposta não são independentes (mesmo modelo, mesmas fontes)."""
    s = np.array([sum(v == "sustenta" for v in r) for r in por_resposta], dtype=float)
    t = np.array([len(r) for r in por_resposta], dtype=float)
    if t.sum() == 0:
        raise ValueError("nenhum par")
    i = np.random.default_rng(semente + 7).integers(0, s.size, (n_bootstrap, s.size))
    reps = s[i].sum(axis=1) / np.maximum(t[i].sum(axis=1), 1)
    return (round(float(s.sum() / t.sum()), 4),
            (round(float(np.percentile(reps, 2.5)), 4), round(float(np.percentile(reps, 97.5)), 4)))


def decidir(ic: tuple[float, float], valido: bool) -> str:
    """O G2.4, com o IC inteiro de um lado da meta, como as regras da §9."""
    if not valido:
        return "juiz_invalido"
    lo, hi = ic
    return "passa" if lo >= META else "nao_passa" if hi < META else "fronteira"
