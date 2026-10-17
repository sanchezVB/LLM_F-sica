"""A medida do assistente, segunda etapa: os braços, o juiz e a regra (DOC-13 §9.1).

Os itens e a guarda I1 estão em `phifm.eval.assistente`, e os limiares da regra também:
este módulo não define limiar nenhum, só o que a regra manda fazer com eles.

## O que roda, por item

- **A · sistema** — `Assistente.responder`: consulta hipotética → busca → 6 fontes.
- **B · fonte garantida** — as fontes de A; se o artigo P não veio, P entra no lugar da
  6ª, numa posição sorteada (fixa por item). Onde P veio, B É a resposta de A — não se
  gera de novo, e ela vem marcada `igual_a_A`.
- **C · sem fontes** — a pergunta sozinha, respondida de memória.

Mesmo modelo e configuração do produto nos três (temperatura 0,2 e 0,3 na consulta),
com a semente da regra.

## O juiz

O Qwen3-8B, temperatura 0, com a pergunta, o gabarito e a resposta **sem as marcas [n]**
— com elas ele saberia que a resposta teve fontes. Ele não vê o braço nem o artigo. O
cache é pelo TEXTO julgado: B igual a A é julgada uma vez só, e com o mesmo veredicto.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from phifm.eval.assistente import (
    K_FONTES,
    KAPPA_MINIMO,
    LIMIAR_BUSCA,
    LIMIAR_GERADOR,
    MARGEM_I2,
    SEMENTE,
)
from phifm.rag.assistente import (
    SISTEMA_HYDE,
    SISTEMA_RESPOSTA,
    Fonte,
    instrucao_de_idioma,
    sem_citacoes,
    verificar_citacoes,
)

# ── a regra, na parte que não é limiar (DOC-13 §9.1) ─────────────────────────
N_I3_POR_BRACO = 50               # 50 de A + 50 de B, braço oculto
MAX_REVISAO_R = 100               # acima disto, R revisa 100 sorteadas
N_BOOTSTRAP = 10_000
VEREDICTOS = ("certo", "parcial", "errado", "absteve")
CLASSES_R = ("modelo_errou", "juiz_errou", "item_invalido")

# C é "a pergunta sozinha, respondida de memória". O prompt é o de A sem as fontes: a
# mesma licença de dizer que não sabe, nem mais nem menos — um C que se abstém por
# instrução deixaria a guarda I2 passar de graça.
SISTEMA_MEMORIA = (
    "You are a physics research assistant. Answer the question from your own knowledge. "
    "If you do not know the answer, say so plainly instead of guessing. Answer clearly and "
    "concisely.")

# O juiz da PRIMEIRA rodada. Reprovado em I3 (2026-10-04): κ 0,52 contra o Claude às
# cegas. Nos 16 casos em que ele negou um `certo`, puniu informação a mais (5), a forma —
# tradução, ordem, "alguns × 10⁻⁷" — (4) e complementos do gabarito que a pergunta não
# pede (7). Fica aqui para o cache antigo (`juiz_b13c72b07dae.jsonl`) continuar legível.
SISTEMA_JUIZ_V1 = (
    "You grade one answer given by a physics assistant, comparing it with a reference "
    "answer (the gabarito) taken from the paper that answers the question. The question, "
    "the gabarito and the answer may be in Portuguese.\n"
    "Choose exactly one verdict:\n"
    "- certo: the answer states the content of the gabarito (a paraphrase, an equivalent "
    "number or unit, or a more precise value that agrees with it is fine) and does not "
    "contradict it. Extra material is fine unless it contradicts the gabarito. If the "
    "gabarito has more than one essential element, all of them must be there.\n"
    "- parcial: the answer states only part of the essential content of the gabarito, or "
    "only a vaguer version of it, and does not contradict it.\n"
    "- errado: the answer gives a different answer or contradicts the gabarito, even if "
    "it also says something right.\n"
    "- absteve: the answer says it cannot answer (for example, that the information is not "
    "available) and does not commit to an answer.\n"
    "Compare only with the gabarito: do not use your own knowledge to decide which is right.\n"
    'Output only one JSON object: {"motivo": "<one short sentence>", '
    '"veredicto": "certo" | "parcial" | "errado" | "absteve"}')

# Versão 2, o conserto. Três mudanças, uma para cada erro da v1: (1) a pergunta decide o
# que é essencial — o NÚCLEO do gabarito —, e o juiz o escreve antes do veredicto;
# (2) uma lista explícita do que NÃO baixa o veredicto (forma, ordem, arredondamento,
# complemento ausente, informação a mais); (3) `errado` e `absteve` definidos nos casos
# de fronteira. Desenvolvida nas 100 respostas da rodada 1 de I3; validada numa amostra
# NOVA (rodada 2), julgada às cegas antes de esta versão rodar nela.
SISTEMA_JUIZ = (
    "You grade one answer given by a physics assistant against a reference answer (the "
    "gabarito), which was written from the paper that answers the question. The question, "
    "the gabarito and the answer may be in Portuguese.\n\n"
    "What counts is ONE thing: does the answer give what the QUESTION asks for, as the "
    "gabarito states it?\n\n"
    "First find the CORE of the gabarito: the part that directly answers the question (a "
    "value, a name, a direction of change, a yes or no with its reason). The gabarito often "
    "adds complements beyond what was asked: a second number, an exception, a comparison, a "
    "consequence, a clause after a dash or a semicolon or in parentheses. Complements are "
    "NOT required.\n\n"
    "Choose exactly one verdict:\n"
    "- certo: the answer states the core and does not contradict the gabarito. None of the "
    "following lowers the verdict: different wording or a paraphrase; a translation or an "
    "English term; an awkward or slightly wrong word when the meaning is clear; a different "
    "order; an equivalent or rounded number; a missing complement; extra information that "
    "is not in the gabarito.\n"
    "- parcial: the answer states only a piece of the core (one of two things the question "
    "asks for), or only a vaguer version of it ('much larger' where the core is a factor), "
    "without contradicting the gabarito.\n"
    "- errado: the answer gives a different core, contradicts the gabarito, or answers "
    "something else. An answer that first gives a wrong core and then also mentions the "
    "right one is errado.\n"
    "- absteve: the answer says it cannot answer (for example, that the sources do not "
    "contain the information) and does not commit to an answer, even if it discusses the "
    "topic around it.\n\n"
    "Compare only with the gabarito: do not use your own knowledge to decide which is right.\n"
    'Output only one JSON object: {"nucleo": "<the core of the gabarito, in a few words>", '
    '"motivo": "<one short sentence>", '
    '"veredicto": "certo" | "parcial" | "errado" | "absteve"}')


def assinatura_bracos(modelo: str, sha_itens: str) -> str:
    """O que muda as respostas: prompts, k, semente, o modelo e os itens."""
    return hashlib.sha256(
        f"{SISTEMA_HYDE}\n{SISTEMA_RESPOSTA}\n{SISTEMA_MEMORIA}\n{K_FONTES}\n{SEMENTE}\n"
        f"{modelo}\n{sha_itens}".encode()).hexdigest()


def assinatura_juiz(modelo: str) -> str:
    return hashlib.sha256(f"{SISTEMA_JUIZ}\n{SEMENTE}\n{modelo}".encode()).hexdigest()


# ── os braços ────────────────────────────────────────────────────────────────


@dataclass
class RespostaDoBraco:
    estrato: str
    ordem: int
    arxiv_id: str                 # o artigo P
    braco: str                    # A · B · C
    texto: str                    # depois do portão de citações
    fontes: list[str] = field(default_factory=list)   # na ordem do prompt; vazio em C
    posicao_p: int | None = None  # 1..k se P estava entre as fontes
    p_citada: bool = False
    citadas: list[int] = field(default_factory=list)
    removidas: list[int] = field(default_factory=list)
    n_frases_sem_fonte: int = 0
    igual_a_A: bool = False       # B em que a busca já trouxe P
    consulta: str = ""            # a consulta hipotética (A e B)
    segundos: float = 0.0
    segundos_recuperacao: float = 0.0   # só no braço A′: busca + ΦRank
    # só no braço A″ (§9.6): os trechos entregues, como [arxiv_id, posição no artigo]
    passagens: list[list] = field(default_factory=list)
    segundos_passagens: float = 0.0     # traduzir a pergunta, ler, cortar e escolher
    consulta_passagens: str = ""        # a pergunta em inglês, com que os trechos foram procurados


def posicao_sorteada(arxiv_id: str, k: int = K_FONTES) -> int:
    """Onde P entra em B, de 1 a `k`: fixa por item, da semente da regra e do id."""
    h = int(hashlib.sha256(f"{SEMENTE}:{arxiv_id}".encode()).hexdigest()[:12], 16)
    return 1 + h % k


def fontes_de_b(fontes_a: list[Fonte], p: Fonte) -> tuple[list[Fonte], bool]:
    """`(fontes, igual_a_A)`. Se P já está entre as de A, são as de A. Senão sai a última
    e P entra na posição sorteada; as outras mantêm a ordem, renumeradas."""
    if any(f.arxiv_id == p.arxiv_id for f in fontes_a):
        return fontes_a, True
    resto = fontes_a[:-1]
    pos = posicao_sorteada(p.arxiv_id, len(fontes_a))
    nova = resto[:pos - 1] + [p] + resto[pos - 1:]
    return [replace(f, numero=i) for i, f in enumerate(nova, start=1)], False


def _registro(item: dict, braco: str, r, segundos: float, igual: bool = False
              ) -> RespostaDoBraco:
    ids = [f.arxiv_id for f in r.fontes]
    pos = ids.index(item["arxiv_id"]) + 1 if item["arxiv_id"] in ids else None
    return RespostaDoBraco(item["estrato"], item["ordem"], item["arxiv_id"], braco, r.texto,
                           ids, pos, pos in r.citadas if pos else False, r.citadas,
                           r.removidas, len(r.frases_sem_fonte), igual, r.consulta,
                           round(segundos, 2))


def rodar_item(assistente, item: dict) -> list[RespostaDoBraco]:
    """Os três braços de um item. `item` tem estrato, ordem, arxiv_id, pergunta."""
    pergunta = item["pergunta"]
    t0 = time.perf_counter()
    a = assistente.responder(pergunta)
    ra = _registro(item, "A", a, time.perf_counter() - t0)

    fontes_b, igual = fontes_de_b(a.fontes, assistente.fonte(item["arxiv_id"], K_FONTES))
    if igual:
        rb = replace(ra, braco="B", igual_a_A=True, segundos=0.0)
    else:
        t0 = time.perf_counter()
        b = assistente.responder_com(pergunta, fontes_b, a.consulta)
        rb = _registro(item, "B", b, time.perf_counter() - t0)

    return [ra, rb, _sem_fontes(assistente, item)]


def _sem_fontes(assistente, item: dict) -> RespostaDoBraco:
    """O braço C: a pergunta sozinha, respondida de memória."""
    pergunta = item["pergunta"]
    t0 = time.perf_counter()
    bruto = assistente.modelo.gerar(
        SISTEMA_MEMORIA, f"Question: {pergunta}\n\n{instrucao_de_idioma(pergunta)}",
        semente=assistente.semente)
    texto, citadas, removidas, sem_fonte = verificar_citacoes(bruto, 0)
    return RespostaDoBraco(item["estrato"], item["ordem"], item["arxiv_id"], "C", texto,
                           citadas=citadas, removidas=removidas,
                           n_frases_sem_fonte=len(sem_fonte),
                           segundos=round(time.perf_counter() - t0, 2))


def ler_respostas(cache: Path) -> list[RespostaDoBraco]:
    if not cache.exists():
        return []
    return [RespostaDoBraco(**json.loads(linha))
            for linha in cache.read_text(encoding="utf-8").splitlines() if linha.strip()]


def rodar(assistente, itens: list[dict], cache: Path, limite: int | None = None,
          ao_terminar=None, por_item=None) -> list[RespostaDoBraco]:
    """Os três braços em cada item, na ordem de `itens`. Retomável: os três registros de
    um item vão ao cache JUNTOS, ao fim do item — um item nunca fica pela metade.

    `por_item(assistente, item)` troca os braços: é como o estrato "corpo" (§9.5) entra."""
    por_item = por_item or rodar_item
    feitos = {(r.estrato, r.arxiv_id) for r in ler_respostas(cache)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with cache.open("a", encoding="utf-8") as f:
        for item in itens:
            if (item["estrato"], item["arxiv_id"]) in feitos:
                continue
            if limite is not None and n >= limite:
                break
            regs = por_item(assistente, item)
            f.write("".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in regs))
            f.flush()
            n += 1
            if ao_terminar:
                ao_terminar(item, regs)
    return ler_respostas(cache)


# ── o juiz ───────────────────────────────────────────────────────────────────


def chave_do_julgamento(pergunta: str, gabarito: str, texto: str) -> str:
    return hashlib.sha256(
        f"{pergunta}\n{gabarito}\n{sem_citacoes(texto)}".encode()).hexdigest()[:24]


def ler_veredicto(bruto: str) -> tuple[str, str]:
    """`(veredicto, motivo)`. Fora do formato, a última das quatro palavras que aparecer;
    nenhuma → `ilegivel`, que não conta como acerto e é relatado."""
    m = re.search(r"\{.*\}", bruto, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            v = str(d.get("veredicto", "")).strip().lower()
            if v in VEREDICTOS:
                return v, str(d.get("motivo", "")).strip()
        except json.JSONDecodeError:
            pass
    achados = re.findall(r"\b(" + "|".join(VEREDICTOS) + r")\b", bruto.lower())
    return (achados[-1], bruto.strip()[:300]) if achados else ("ilegivel", bruto.strip()[:300])


def julgar(modelo, pergunta: str, gabarito: str, texto: str) -> tuple[str, str]:
    limpo = sem_citacoes(texto).strip()
    if not limpo:
        return "absteve", "resposta vazia"
    bruto = modelo.gerar(SISTEMA_JUIZ,
                         f"Question: {pergunta}\nGabarito: {gabarito}\nAnswer:\n{limpo}",
                         max_tokens=260, temperatura=0.0, semente=SEMENTE)
    return ler_veredicto(bruto)


def ler_julgamentos(cache: Path) -> dict[str, dict]:
    if not cache.exists():
        return {}
    saida = {}
    for linha in cache.read_text(encoding="utf-8").splitlines():
        if linha.strip():
            d = json.loads(linha)
            saida[d["chave"]] = d
    return saida


def julgar_todas(modelo, itens: list[dict], respostas: list[RespostaDoBraco], cache: Path,
                 ao_julgar=None) -> dict[str, dict]:
    """Julga cada texto distinto uma vez. Retomável, como os braços."""
    por_id = {(i["estrato"], i["arxiv_id"]): i for i in itens}
    feitos = ler_julgamentos(cache)
    with cache.open("a", encoding="utf-8") as f:
        for r in respostas:
            it = por_id[(r.estrato, r.arxiv_id)]
            chave = chave_do_julgamento(it["pergunta"], it["gabarito"], r.texto)
            if chave in feitos:
                continue
            t0 = time.perf_counter()
            v, motivo = julgar(modelo, it["pergunta"], it["gabarito"], r.texto)
            d = {"chave": chave, "veredicto": v, "motivo": motivo,
                 "segundos": round(time.perf_counter() - t0, 2)}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            f.flush()
            feitos[chave] = d
            if ao_julgar:
                ao_julgar(r, d)
    return feitos


def veredictos_por_braco(itens: list[dict], respostas: list[RespostaDoBraco],
                         julgamentos: dict[str, dict]) -> dict[str, dict[tuple, str]]:
    """{braço: {(estrato, arxiv_id): veredicto}}. Recusa se faltar julgamento."""
    por_id = {(i["estrato"], i["arxiv_id"]): i for i in itens}
    saida: dict[str, dict[tuple, str]] = {"A": {}, "B": {}, "C": {}}
    for r in respostas:
        it = por_id[(r.estrato, r.arxiv_id)]
        chave = chave_do_julgamento(it["pergunta"], it["gabarito"], r.texto)
        if chave not in julgamentos:
            raise RuntimeError(f"resposta {r.braco} de {r.arxiv_id} sem julgamento")
        saida.setdefault(r.braco, {})[(r.estrato, r.arxiv_id)] = julgamentos[chave]["veredicto"]
    return saida


# ── o braço A′ da §9.2: A com o reordenador (PROPOSTA — não roda antes do aceite) ──
BRACO_REORDENADO = "A2"
# A §9.2 toca o conjunto de teste UMA vez. ACEITA pelo dono em 2026-10-01, com o limiar de
# adoção mais exigente que o proposto (0,03 em vez de 0) — registro no DOC-13 §9.2.
REGRA_9_2_ACEITA = True
LIMIAR_ADOCAO_REORDENADO = 0.03   # o IC de acerto_A′ − acerto_A tem de ficar INTEIRO acima
TEMPO_MAX_REORDENADO_S = 60.0     # mediana por pergunta


def rodar_reordenado(assistente, itens: list[dict], respostas_a: dict[tuple, RespostaDoBraco],
                     cache: Path, limite: int | None = None, ao_terminar=None
                     ) -> list[RespostaDoBraco]:
    """O braço A′ (DOC-13 §9.2): a MESMA consulta hipotética de A, com as fontes
    reordenadas. Onde as 6 fontes saem iguais às de A, na mesma ordem, o prompt é o de A e
    a resposta de A é reaproveitada (`igual_a_A`) — não se gera de novo. Retomável."""
    feitos = {(r.estrato, r.arxiv_id) for r in ler_respostas(cache)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with cache.open("a", encoding="utf-8") as f:
        for item in itens:
            chave = (item["estrato"], item["arxiv_id"])
            if chave in feitos:
                continue
            if limite is not None and n >= limite:
                break
            ra = respostas_a[chave]
            t0 = time.perf_counter()
            fontes = assistente.recuperar(ra.consulta)
            t_rec = round(time.perf_counter() - t0, 2)
            if [x.arxiv_id for x in fontes] == ra.fontes:
                reg = replace(ra, braco=BRACO_REORDENADO, igual_a_A=True, segundos=t_rec)
            else:
                r = assistente.responder_com(item["pergunta"], fontes, ra.consulta)
                reg = _registro(item, BRACO_REORDENADO, r, time.perf_counter() - t0)
            reg = replace(reg, segundos_recuperacao=t_rec)
            f.write(json.dumps(asdict(reg), ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if ao_terminar:
                ao_terminar(item, reg)
    return ler_respostas(cache)


def comparar_reordenado(veredictos: dict[str, dict[tuple, str]],
                        respostas: list[RespostaDoBraco], estrato: str = "primario") -> dict:
    """A regra da §9.2 no estrato: acerto_A′ − acerto_A pelo juiz, pareado por item, e o
    tempo. ADOTA se o IC 95% ficar inteiro acima de `LIMIAR_ADOCAO_REORDENADO` e o tempo
    mediano couber em `TEMPO_MAX_REORDENADO_S`; fora disso, fica o sistema atual.

    O tempo de A′ é a mediana de A mais a mediana da recuperação com o ΦRank — conta a
    busca simples duas vezes, errando para o lado de reprovar (definido antes de rodar)."""
    ids = sorted(k for k in veredictos[BRACO_REORDENADO] if k[0] == estrato)
    a = np.array([veredictos["A"][k] == "certo" for k in ids], dtype=float)
    a2 = np.array([veredictos[BRACO_REORDENADO][k] == "certo" for k in ids], dtype=float)
    dif, ic = diferenca_pareada(a2, a)
    do_estrato = [r for r in respostas if r.estrato == estrato]
    t_a = float(np.median([r.segundos for r in do_estrato if r.braco == "A"]))
    t_rec = float(np.median([r.segundos_recuperacao for r in do_estrato
                             if r.braco == BRACO_REORDENADO]))
    tempo = round(t_a + t_rec, 1)
    adota = ic[0] > LIMIAR_ADOCAO_REORDENADO and tempo <= TEMPO_MAX_REORDENADO_S
    iguais = sum(1 for r in do_estrato if r.braco == BRACO_REORDENADO and r.igual_a_A)
    return {"estrato": estrato, "n_itens": len(ids), "acerto_A": round(float(a.mean()), 4),
            "acerto_A2": round(float(a2.mean()), 4), "A2_menos_A": dif, "ic95": ic,
            "limiar": LIMIAR_ADOCAO_REORDENADO,
            "ganha": int(((a2 - a) > 0).sum()), "perde": int(((a2 - a) < 0).sum()),
            "respostas_iguais_as_de_A": iguais,
            "tempo_mediano_s": tempo, "tempo_maximo_s": TEMPO_MAX_REORDENADO_S,
            "decisao": "ADOTA" if adota else "FICA O SISTEMA ATUAL"}


# ── a busca fundida com o BM25 (DOC-13 §9.9): A5 = A′ com outra busca ──────────
BRACO_BUSCA_LEXICA = "A5"
# A §9.9 toca o conjunto de teste uma vez — `scripts/medir_busca_lexical.py` recusa o teste
# enquanto isto for False. ACEITA pelo dono em 2026-10-14, como escrita.
REGRA_9_9_ACEITA = True
LIMIAR_ADOCAO_BUSCA_LEXICA = 0.03  # o IC de acerto_A5 − acerto_A′ tem de ficar INTEIRO acima
TEMPO_MAX_BUSCA_LEXICA_S = 60.0    # mediana por pergunta, pela conta da §9.2


def rodar_busca_lexica_sobre_a2(assistente, itens: list[dict],
                                respostas_a2: dict[tuple, RespostaDoBraco], cache: Path,
                                limite: int | None = None, ao_terminar=None
                                ) -> list[RespostaDoBraco]:
    """O braço A5: a MESMA consulta hipotética de A′ (do cache), a pergunta traduzida para
    inglês e a busca fundida com o BM25, com o ΦRank por cima — `assistente.recuperar`,
    o caminho do produto. Onde as 6 fontes saem iguais às de A′, na mesma ordem, o prompt
    é o de A′ e a resposta de A′ é reaproveitada (`igual_a_A`). O tempo de recuperação
    inclui a tradução. Retomável."""
    feitos = {(r.estrato, r.arxiv_id) for r in ler_respostas(cache)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with cache.open("a", encoding="utf-8") as f:
        for item in itens:
            chave = (item["estrato"], item["arxiv_id"])
            if chave in feitos:
                continue
            if limite is not None and n >= limite:
                break
            ra = respostas_a2[chave]
            t0 = time.perf_counter()
            ingles = assistente.em_ingles(item["pergunta"])
            fontes = assistente.recuperar(ra.consulta, ingles)
            t_rec = round(time.perf_counter() - t0, 2)
            if [x.arxiv_id for x in fontes] == ra.fontes:
                reg = replace(ra, braco=BRACO_BUSCA_LEXICA, igual_a_A=True, segundos=t_rec)
            else:
                r = assistente.responder_com(item["pergunta"], fontes, ra.consulta)
                reg = _registro(item, BRACO_BUSCA_LEXICA, r, time.perf_counter() - t0)
            reg = replace(reg, segundos_recuperacao=t_rec, consulta_passagens=ingles)
            f.write(json.dumps(asdict(reg), ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if ao_terminar:
                ao_terminar(item, reg)
    return ler_respostas(cache)


def comparar_busca_lexica(veredictos: dict[str, dict[tuple, str]],
                          respostas: list[RespostaDoBraco], tempo_a_s: float,
                          estrato: str = "primario") -> dict:
    """A regra da §9.9 no estrato: acerto_A5 − acerto_A′ pelo juiz, pareado por item, e o
    tempo. ADOTA se o IC 95% ficar inteiro acima de `LIMIAR_ADOCAO_BUSCA_LEXICA` e o tempo
    couber em `TEMPO_MAX_BUSCA_LEXICA_S`.

    O tempo é o da §9.2 com A5 no lugar de A′: a mediana de A de ponta a ponta
    (`tempo_a_s`) mais a mediana da recuperação de A5 — tradução, as duas buscas, a fusão
    e o ΦRank. Conta a busca simples duas vezes: erra para o lado de reprovar."""
    ids = sorted(k for k in veredictos[BRACO_BUSCA_LEXICA] if k[0] == estrato)
    a2 = np.array([veredictos[BRACO_REORDENADO][k] == "certo" for k in ids], dtype=float)
    a5 = np.array([veredictos[BRACO_BUSCA_LEXICA][k] == "certo" for k in ids], dtype=float)
    dif, ic = diferenca_pareada(a5, a2)
    de = {b: {(r.estrato, r.arxiv_id): r for r in respostas if r.braco == b and r.estrato == estrato}
          for b in (BRACO_REORDENADO, BRACO_BUSCA_LEXICA)}
    p6 = {b: np.array([de[b][k].posicao_p is not None for k in ids], dtype=float) for b in de}
    dif_p, ic_p = diferenca_pareada(p6[BRACO_BUSCA_LEXICA], p6[BRACO_REORDENADO])
    t_rec = float(np.median([r.segundos_recuperacao for r in de[BRACO_BUSCA_LEXICA].values()]))
    tempo = round(tempo_a_s + t_rec, 1)
    adota = ic[0] > LIMIAR_ADOCAO_BUSCA_LEXICA and tempo <= TEMPO_MAX_BUSCA_LEXICA_S
    return {"estrato": estrato, "n_itens": len(ids),
            "acerto_A2": round(float(a2.mean()), 4), "acerto_A5": round(float(a5.mean()), 4),
            "A5_menos_A2": dif, "ic95": ic, "limiar": LIMIAR_ADOCAO_BUSCA_LEXICA,
            "ganha": int((a5 > a2).sum()), "perde": int((a5 < a2).sum()),
            "P_nas_6_fontes": {"A2": round(float(p6[BRACO_REORDENADO].mean()), 4),
                               "A5": round(float(p6[BRACO_BUSCA_LEXICA].mean()), 4),
                               "A5_menos_A2": dif_p, "ic95": ic_p},
            "respostas_iguais_as_de_A2": sum(1 for r in de[BRACO_BUSCA_LEXICA].values()
                                             if r.igual_a_A),
            "tempo": {"mediana_A_ponta_a_ponta_s": round(tempo_a_s, 1),
                      "mediana_recuperacao_A5_s": round(t_rec, 1), "regra_s": tempo,
                      "maximo_s": TEMPO_MAX_BUSCA_LEXICA_S},
            "decisao": "ADOTA" if adota else "FICA O SISTEMA ATUAL"}


# ── a ordem pelo texto completo no modo das passagens (DOC-13 §9.11), só posição ──
# `scripts/medir_busca_corpo.py` recusa enquanto isto for False. ACEITA pelo dono em
# 2026-10-16, como escrita.
REGRA_9_11_ACEITA = True
MARGEM_NAO_PIORA_TEXTO = 0.02     # teste primário: o IC de novo − hoje, inteiro acima de −0,02
TEMPO_MAX_TEXTO_S = 3.0           # mediana do tempo a mais por pergunta


# ── o estrato "corpo" (DOC-13 §9.5): o sistema de hoje, a passagem garantida, e C ──
BRACO_PASSAGEM = "Bp"
N_I3C_POR_BRACO = 20              # 20 de A′ + 20 de Bp, braço oculto
MAX_ACERTO_SO_COM_O_RESUMO = 0.20  # guarda "fora do resumo"


def fontes_com_passagem(fontes_a: list[Fonte], p: Fonte) -> list[Fonte]:
    """As fontes do braço Bp. `p` é o artigo certo com a PASSAGEM no lugar do resumo. Se P
    já está entre as fontes, a passagem toma o lugar do resumo dele, na mesma posição;
    senão sai a última e P entra na posição sorteada — como no braço B da §9.1."""
    for i, f in enumerate(fontes_a):
        if f.arxiv_id == p.arxiv_id:
            return [*fontes_a[:i], replace(p, numero=f.numero, escore=f.escore),
                    *fontes_a[i + 1:]]
    resto = fontes_a[:-1]
    pos = posicao_sorteada(p.arxiv_id, len(fontes_a))
    nova = resto[:pos - 1] + [p] + resto[pos - 1:]
    return [replace(f, numero=i) for i, f in enumerate(nova, start=1)]


def rodar_item_corpo(assistente, item: dict, passagem: str) -> list[RespostaDoBraco]:
    """Os três braços de um item do corpo: A′ (o sistema de hoje, com o reordenador do
    `assistente`), Bp e C. A consulta hipotética é uma só, a de A′."""
    pergunta = item["pergunta"]
    t0 = time.perf_counter()
    consulta = assistente.consulta_hipotetica(pergunta)
    fontes = assistente.recuperar(consulta)
    t_rec = round(time.perf_counter() - t0, 2)
    a = assistente.responder_com(pergunta, fontes, consulta)
    ra = replace(_registro(item, BRACO_REORDENADO, a, time.perf_counter() - t0),
                 segundos_recuperacao=t_rec)

    p = replace(assistente.fonte(item["arxiv_id"], K_FONTES), resumo=passagem)
    t0 = time.perf_counter()
    b = assistente.responder_com(pergunta, fontes_com_passagem(a.fontes, p), consulta)
    rb = _registro(item, BRACO_PASSAGEM, b, time.perf_counter() - t0)
    return [ra, rb, _sem_fontes(assistente, item)]


def amostra_i3c(respostas: list[RespostaDoBraco], semente: int = SEMENTE,
                bracos: tuple[str, str] = (BRACO_REORDENADO, BRACO_PASSAGEM), salto: int = 5
                ) -> list[RespostaDoBraco]:
    """I3c: `N_I3C_POR_BRACO` respostas de cada um dos dois `bracos` (A′ e Bp na §9.5),
    de itens DIFERENTES, embaralhadas — quem julga não sabe o braço. `salto` separa os
    sorteios de medidas diferentes."""
    itens = sorted({(r.estrato, r.arxiv_id) for r in respostas})
    rng = np.random.default_rng(semente + salto)
    perm = rng.permutation(len(itens))
    de_a = {itens[j] for j in perm[:N_I3C_POR_BRACO]}
    de_b = {itens[j] for j in perm[N_I3C_POR_BRACO:2 * N_I3C_POR_BRACO]}
    amostra = [r for r in respostas
               if (r.braco == bracos[0] and (r.estrato, r.arxiv_id) in de_a)
               or (r.braco == bracos[1] and (r.estrato, r.arxiv_id) in de_b)]
    amostra.sort(key=lambda r: (r.estrato, r.arxiv_id, r.braco))
    return [amostra[j] for j in rng.permutation(len(amostra))]


def resumo_corpo(veredictos: dict[str, dict[tuple, str]], respostas: list[RespostaDoBraco]
                 ) -> dict:
    """As leituras e as guardas da §9.5, nos itens do estrato "corpo"."""
    ids = sorted(veredictos[BRACO_PASSAGEM])
    certo = {b: np.array([veredictos[b][k] == "certo" for k in ids], dtype=float)
             for b in (BRACO_REORDENADO, BRACO_PASSAGEM, "C")}
    de_a = {(r.estrato, r.arxiv_id): r for r in respostas if r.braco == BRACO_REORDENADO}
    p_veio = np.array([de_a[k].posicao_p is not None for k in ids])
    acerto = {b: proporcao(certo[b]) for b in certo}
    lacuna, ic_lacuna = diferenca_pareada(certo[BRACO_PASSAGEM], certo[BRACO_REORDENADO])
    so_resumo = float(certo[BRACO_REORDENADO][p_veio].mean()) if p_veio.any() else float("nan")
    return {
        "n_itens": len(ids),
        "acerto": {b: {"valor": v, "ic95": ic} for b, (v, ic) in acerto.items()},
        "veredictos": {b: {v: sum(1 for k in ids if veredictos[b][k] == v)
                           for v in (*VEREDICTOS, "ilegivel")} for b in certo},
        "gerador_com_a_passagem": {
            "acerto_Bp": acerto[BRACO_PASSAGEM][0], "ic95": acerto[BRACO_PASSAGEM][1],
            "limiar": LIMIAR_GERADOR,
            "desfecho": "BASTA" if acerto[BRACO_PASSAGEM][1][0] >= LIMIAR_GERADOR else "LIMITA"},
        "lacuna_do_corpo": {"Bp_menos_A2": lacuna, "ic95": ic_lacuna,
                            "ganha": int((certo[BRACO_PASSAGEM] > certo[BRACO_REORDENADO]).sum()),
                            "perde": int((certo[BRACO_PASSAGEM] < certo[BRACO_REORDENADO]).sum())},
        "p_nas_fontes_de_A2": {"itens": int(p_veio.sum()),
                               "fracao": round(float(p_veio.mean()), 4)},
        "guardas": {
            "I2c": {"acerto_C": acerto["C"][0], "acerto_Bp": acerto[BRACO_PASSAGEM][0],
                    "margem": MARGEM_I2,
                    "valida": bool(acerto["C"][0] < acerto[BRACO_PASSAGEM][0] - MARGEM_I2)},
            "fora_do_resumo": {
                "itens_com_o_resumo_de_P_nas_fontes": int(p_veio.sum()),
                "acerto_de_A2_neles": None if np.isnan(so_resumo) else round(so_resumo, 4),
                "acerto_de_Bp_neles": round(float(certo[BRACO_PASSAGEM][p_veio].mean()), 4)
                if p_veio.any() else None,
                "maximo": MAX_ACERTO_SO_COM_O_RESUMO,
                "valida": bool(np.isnan(so_resumo) or so_resumo <= MAX_ACERTO_SO_COM_O_RESUMO)}},
        "acerto_de_Bp_onde_P_nao_veio": round(float(certo[BRACO_PASSAGEM][~p_veio].mean()), 4)
        if (~p_veio).any() else None}


# ── as passagens sob demanda (DOC-13 §9.6): A′, A″ = A′ + passagens, Bp e C ──
BRACO_PASSAGENS = "A3"
# A §9.6 toca o conjunto de teste UMA vez, no eixo "não piora" — `scripts/medir_passagens.py`
# recusa o teste enquanto isto for False. ACEITA pelo dono em 2026-10-13, com a emenda: o
# braço que vai ao teste é A‴ (`BRACO_PASSAGENS_V2`), e um juiz independente confere.
REGRA_9_6_ACEITA = True
LIMIAR_GANHO_PASSAGENS = 0.15     # corpo: o IC de A″ − A′ tem de ficar INTEIRO acima
MARGEM_NAO_PIORA = 0.03           # primário: o IC de A″ − A′, inteiro acima de −0,03
TEMPO_MAX_PASSAGENS_S = 60.0      # mediana por pergunta, de ponta a ponta — o teto da §9.2


def rodar_item_passagens(assistente, passo, item: dict, passagem: str) -> list[RespostaDoBraco]:
    """Os quatro braços de um item do corpo 2. A consulta hipotética e as 6 fontes são as
    mesmas em A′, A″ e Bp: de A′ para A″ só mudam as passagens que `passo` anexa. O tempo
    de A″ é o do produto, de ponta a ponta: consulta, busca, passagens e resposta."""
    pergunta = item["pergunta"]
    t0 = time.perf_counter()
    consulta = assistente.consulta_hipotetica(pergunta)
    fontes = assistente.recuperar(consulta)
    t_rec = time.perf_counter() - t0
    t0 = time.perf_counter()
    a = assistente.responder_com(pergunta, fontes, consulta)
    ra = replace(_registro(item, BRACO_REORDENADO, a, t_rec + time.perf_counter() - t0),
                 segundos_recuperacao=round(t_rec, 2))

    t0 = time.perf_counter()
    com, escolhidos, em_ingles = passo.anexar_com_registro(pergunta, fontes)
    t_pas = time.perf_counter() - t0
    t0 = time.perf_counter()
    a3 = assistente.responder_com(pergunta, com, consulta)
    r3 = replace(_registro(item, BRACO_PASSAGENS, a3,
                           t_rec + t_pas + time.perf_counter() - t0),
                 segundos_recuperacao=round(t_rec, 2), segundos_passagens=round(t_pas, 2),
                 consulta_passagens=em_ingles,
                 passagens=[[e.arxiv_id, e.posicao] for e in escolhidos])

    p = replace(assistente.fonte(item["arxiv_id"], K_FONTES), resumo=passagem)
    t0 = time.perf_counter()
    bp = assistente.responder_com(pergunta, fontes_com_passagem(fontes, p), consulta)
    rb = _registro(item, BRACO_PASSAGEM, bp, time.perf_counter() - t0)
    return [ra, r3, rb, _sem_fontes(assistente, item)]


def rodar_passagens_sobre_a2(assistente, passo, itens: list[dict],
                             respostas_a2: dict[tuple, RespostaDoBraco], fontes_de,
                             cache: Path, limite: int | None = None, ao_terminar=None,
                             braco: str = BRACO_PASSAGENS, reaproveitar: bool = True
                             ) -> list[RespostaDoBraco]:
    """A″ onde A′ JÁ foi medido (o teste primário, §9.2): a consulta e as fontes de A′
    vêm do cache — `fontes_de(ids)` as refaz — e só as passagens são novas. Onde nenhuma
    passagem é anexada o prompt é o de A′, e a resposta de A′ é reaproveitada
    (`igual_a_A`). Retomável.

    Com um `assistente` de OUTRA instrução (A‴, a emenda da §9.6) o prompt muda mesmo sem
    passagem: aí `reaproveitar=False`, e todo item é respondido de novo. O tempo gravado é
    o do produto: a busca de A′, as passagens e a resposta."""
    feitos = {(r.estrato, r.arxiv_id) for r in ler_respostas(cache)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with cache.open("a", encoding="utf-8") as f:
        for item in itens:
            chave = (item["estrato"], item["arxiv_id"])
            if chave in feitos:
                continue
            if limite is not None and n >= limite:
                break
            ra = respostas_a2[chave]
            t0 = time.perf_counter()
            com, escolhidos, em_ingles = passo.anexar_com_registro(item["pergunta"],
                                                                   fontes_de(ra.fontes))
            t_pas = round(time.perf_counter() - t0, 2)
            if not escolhidos and reaproveitar:
                reg = replace(ra, braco=braco, igual_a_A=True, segundos=t_pas)
            else:
                t0 = time.perf_counter()
                r = assistente.responder_com(item["pergunta"], com, ra.consulta)
                dt = time.perf_counter() - t0
                reg = replace(_registro(item, braco, r, ra.segundos_recuperacao + t_pas + dt),
                              segundos_recuperacao=ra.segundos_recuperacao,
                              passagens=[[e.arxiv_id, e.posicao] for e in escolhidos])
            reg = replace(reg, segundos_passagens=t_pas, consulta_passagens=em_ingles)
            f.write(json.dumps(asdict(reg), ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if ao_terminar:
                ao_terminar(item, reg)
    return ler_respostas(cache)


BRACO_PASSAGENS_V2 = "A4"         # A″ com a instrução candidata (emenda à §9.6, §9.7)


def rodar_instrucao_sobre_a3(assistente, passo, itens: list[dict],
                             respostas_a3: dict[tuple, RespostaDoBraco], fontes_de,
                             cache: Path, limite: int | None = None, ao_terminar=None
                             ) -> list[RespostaDoBraco]:
    """O braço A‴: as fontes e as passagens de A″, do cache, respondidas por um
    `assistente` com OUTRA instrução. De A″ para cá só muda a instrução — e por isso a
    seleção refeita tem de bater, trecho a trecho, com a que A″ gravou. O tempo é o de A″
    até a resposta (consulta, busca, passagens) mais o desta resposta. Retomável."""
    feitos = {(r.estrato, r.arxiv_id) for r in ler_respostas(cache)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with cache.open("a", encoding="utf-8") as f:
        for item in itens:
            chave = (item["estrato"], item["arxiv_id"])
            if chave in feitos:
                continue
            if limite is not None and n >= limite:
                break
            r3 = respostas_a3[chave]
            com, escolhidos, _ = passo.anexar_com_registro(
                item["pergunta"], fontes_de(r3.fontes), consulta=r3.consulta_passagens)
            if [[e.arxiv_id, e.posicao] for e in escolhidos] != r3.passagens:
                raise RuntimeError(f"a seleção de passagens de {item['arxiv_id']} não é a "
                                   "que A″ gravou: o texto ou a regra mudou.")
            t0 = time.perf_counter()
            r = assistente.responder_com(item["pergunta"], com, r3.consulta)
            dt = time.perf_counter() - t0
            reg = replace(_registro(item, BRACO_PASSAGENS_V2, r,
                                    r3.segundos_recuperacao + r3.segundos_passagens + dt),
                          segundos_recuperacao=r3.segundos_recuperacao,
                          segundos_passagens=r3.segundos_passagens,
                          consulta_passagens=r3.consulta_passagens, passagens=r3.passagens)
            f.write(json.dumps(asdict(reg), ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if ao_terminar:
                ao_terminar(item, reg)
    return ler_respostas(cache)


def comparar_instrucao(veredictos: dict[str, dict[tuple, str]],
                       respostas: list[RespostaDoBraco]) -> dict:
    """A‴ ao lado de A′ e de A″ nos mesmos itens: o ganho sobre o sistema de hoje, o que a
    instrução muda sobre A″, e o tempo."""
    ids = sorted(veredictos[BRACO_PASSAGENS_V2])
    certo = {b: np.array([veredictos[b][k] == "certo" for k in ids], dtype=float)
             for b in (BRACO_REORDENADO, BRACO_PASSAGENS, BRACO_PASSAGENS_V2)}
    v, ic = proporcao(certo[BRACO_PASSAGENS_V2])
    ganho, ic_g = diferenca_pareada(certo[BRACO_PASSAGENS_V2], certo[BRACO_REORDENADO])
    dif, ic_d = diferenca_pareada(certo[BRACO_PASSAGENS_V2], certo[BRACO_PASSAGENS])
    tempo = round(float(np.median([r.segundos for r in respostas
                                   if r.braco == BRACO_PASSAGENS_V2])), 1)
    return {"n_itens": len(ids), "acerto_A4": {"valor": v, "ic95": ic},
            "veredictos_A4": {x: sum(1 for k in ids if veredictos[BRACO_PASSAGENS_V2][k] == x)
                              for x in (*VEREDICTOS, "ilegivel")},
            "ganho_no_corpo": {"A4_menos_A2": ganho, "ic95": ic_g,
                               "limiar": LIMIAR_GANHO_PASSAGENS,
                               "passa": bool(ic_g[0] > LIMIAR_GANHO_PASSAGENS)},
            "A4_menos_A3": {"valor": dif, "ic95": ic_d,
                            "ganha": int((certo[BRACO_PASSAGENS_V2]
                                          > certo[BRACO_PASSAGENS]).sum()),
                            "perde": int((certo[BRACO_PASSAGENS_V2]
                                          < certo[BRACO_PASSAGENS]).sum())},
            "tempo": {"mediana_A4_s": tempo, "maximo_s": TEMPO_MAX_PASSAGENS_S,
                      "passa": bool(tempo <= TEMPO_MAX_PASSAGENS_S)}}


def resumo_passagens(veredictos: dict[str, dict[tuple, str]], respostas: list[RespostaDoBraco]
                     ) -> dict:
    """O eixo do GANHO da §9.6, nos itens do corpo 2, com as guardas da §9.5."""
    ids = sorted(veredictos[BRACO_PASSAGENS])
    bracos = (BRACO_REORDENADO, BRACO_PASSAGENS, BRACO_PASSAGEM, "C")
    certo = {b: np.array([veredictos[b][k] == "certo" for k in ids], dtype=float)
             for b in bracos}
    de = {b: {(r.estrato, r.arxiv_id): r for r in respostas if r.braco == b} for b in bracos}
    p_veio = np.array([de[BRACO_REORDENADO][k].posicao_p is not None for k in ids])
    acerto = {b: proporcao(certo[b]) for b in bracos}
    ganho, ic = diferenca_pareada(certo[BRACO_PASSAGENS], certo[BRACO_REORDENADO])
    so_resumo = float(certo[BRACO_REORDENADO][p_veio].mean()) if p_veio.any() else float("nan")
    tempo = {b: round(float(np.median([de[b][k].segundos for k in ids])), 1)
             for b in (BRACO_REORDENADO, BRACO_PASSAGENS)}
    t_pas = round(float(np.median([de[BRACO_PASSAGENS][k].segundos_passagens for k in ids])), 1)
    return {
        "n_itens": len(ids),
        "acerto": {b: {"valor": v, "ic95": i} for b, (v, i) in acerto.items()},
        "veredictos": {b: {v: sum(1 for k in ids if veredictos[b][k] == v)
                           for v in (*VEREDICTOS, "ilegivel")} for b in bracos},
        "ganho_no_corpo": {
            "A3_menos_A2": ganho, "ic95": ic, "limiar": LIMIAR_GANHO_PASSAGENS,
            "ganha": int((certo[BRACO_PASSAGENS] > certo[BRACO_REORDENADO]).sum()),
            "perde": int((certo[BRACO_PASSAGENS] < certo[BRACO_REORDENADO]).sum()),
            "passa": bool(ic[0] > LIMIAR_GANHO_PASSAGENS)},
        "tempo": {"mediana_A2_s": tempo[BRACO_REORDENADO],
                  "mediana_A3_s": tempo[BRACO_PASSAGENS], "das_quais_passagens_s": t_pas,
                  "maximo_s": TEMPO_MAX_PASSAGENS_S,
                  "passa": bool(tempo[BRACO_PASSAGENS] <= TEMPO_MAX_PASSAGENS_S)},
        "p_nas_fontes": {"itens": int(p_veio.sum()), "fracao": round(float(p_veio.mean()), 4),
                         "acerto_de_A3_neles": round(float(
                             certo[BRACO_PASSAGENS][p_veio].mean()), 4) if p_veio.any() else None,
                         "acerto_de_A3_fora": round(float(
                             certo[BRACO_PASSAGENS][~p_veio].mean()), 4)
                         if (~p_veio).any() else None},
        "o_teto": {"acerto_Bp": acerto[BRACO_PASSAGEM][0],
                   "fracao_do_teto_colhida": round(
                       acerto[BRACO_PASSAGENS][0] / acerto[BRACO_PASSAGEM][0], 4)
                   if acerto[BRACO_PASSAGEM][0] else None},
        "guardas": {
            "I2c": {"acerto_C": acerto["C"][0], "acerto_Bp": acerto[BRACO_PASSAGEM][0],
                    "margem": MARGEM_I2,
                    "valida": bool(acerto["C"][0] < acerto[BRACO_PASSAGEM][0] - MARGEM_I2)},
            "fora_do_resumo": {
                "itens_com_o_resumo_de_P_nas_fontes": int(p_veio.sum()),
                "acerto_de_A2_neles": None if np.isnan(so_resumo) else round(so_resumo, 4),
                "maximo": MAX_ACERTO_SO_COM_O_RESUMO,
                "valida": bool(np.isnan(so_resumo) or so_resumo <= MAX_ACERTO_SO_COM_O_RESUMO)}}}


def nao_piora(veredictos: dict[str, dict[tuple, str]], respostas: list[RespostaDoBraco],
              estrato: str = "primario", braco: str = BRACO_PASSAGENS) -> dict:
    """O eixo "NÃO PIORA" da §9.6: nas perguntas cuja resposta está no resumo, o candidato
    (`braco`: A″, ou A‴ pela emenda) − A′ pelo juiz, pareado por item. Passa se o IC 95%
    ficar inteiro acima de −`MARGEM_NAO_PIORA`."""
    ids = sorted(k for k in veredictos[braco] if k[0] == estrato)
    a2 = np.array([veredictos[BRACO_REORDENADO][k] == "certo" for k in ids], dtype=float)
    a3 = np.array([veredictos[braco][k] == "certo" for k in ids], dtype=float)
    dif, ic = diferenca_pareada(a3, a2)
    do_braco = [r for r in respostas if r.braco == braco and r.estrato == estrato]
    return {"estrato": estrato, "braco": braco, "n_itens": len(ids),
            "acerto_A2": round(float(a2.mean()), 4),
            f"acerto_{braco}": round(float(a3.mean()), 4), f"{braco}_menos_A2": dif, "ic95": ic,
            "margem": MARGEM_NAO_PIORA,
            "ganha": int((a3 > a2).sum()), "perde": int((a3 < a2).sum()),
            "veredictos": {b: {v: sum(1 for k in ids if veredictos[b][k] == v)
                               for v in (*VEREDICTOS, "ilegivel")}
                           for b in (BRACO_REORDENADO, braco)},
            "respostas_iguais_as_de_A2": sum(1 for r in do_braco if r.igual_a_A),
            "com_passagem_anexada": sum(1 for r in do_braco if r.passagens),
            "passa": bool(ic[0] > -MARGEM_NAO_PIORA)}


def decidir_passagens(corpo: dict, primario: dict | None, i3: dict,
                      candidato: dict | None = None) -> dict:
    """A regra da §9.6. ADOTA só com os três eixos passando — ganho no corpo, não piorar o
    primário e tempo — e com as guardas e o juiz válidos. Sem o eixo do primário (o teste
    ainda não foi tocado), a decisão fica PENDENTE.

    `candidato` é o `comparar_instrucao` de A‴: pela emenda, o ganho e o tempo são os DELE
    (as guardas, que são de A′, Bp e C, continuam as de `corpo`)."""
    guardas = all(g["valida"] for g in corpo["guardas"].values())
    if not i3.get("passa"):
        situacao = "NÃO DECIDIDO pelo instrumento (I3: κ abaixo do mínimo)"
    elif not guardas:
        situacao = "INVÁLIDO (uma guarda do estrato não vale)"
    else:
        situacao = "válido"
    de = candidato or corpo
    eixos = {"ganho_no_corpo": de["ganho_no_corpo"]["passa"], "tempo": de["tempo"]["passa"],
             "nao_piora_o_primario": None if primario is None else primario["passa"]}
    if situacao != "válido":
        decisao = "—"
    elif primario is None:
        decisao = "PENDENTE (falta o eixo do primário)"
    else:
        decisao = "ADOTA" if all(eixos.values()) else "FICA O SISTEMA ATUAL"
    return {"candidato": BRACO_PASSAGENS_V2 if candidato else BRACO_PASSAGENS,
            "situacao": situacao, "eixos": eixos, "decisao": decisao}


# ── I3: a concordância do juiz com o dono ────────────────────────────────────


def amostra_i3(respostas: list[RespostaDoBraco], semente: int = SEMENTE, rodada: int = 1
               ) -> list[RespostaDoBraco]:
    """50 respostas de A e 50 de B, de itens DIFERENTES (onde B = A, o mesmo item daria
    o mesmo texto duas vezes), embaralhadas: quem julga não sabe o braço.

    `rodada` anda na MESMA permutação de itens: a rodada 2 pega os 100 itens seguintes,
    disjuntos dos da rodada 1 — é a amostra de validação de um juiz consertado com a 1."""
    itens = sorted({(r.estrato, r.arxiv_id) for r in respostas})
    rng = np.random.default_rng(semente + 3)
    perm = rng.permutation(len(itens))
    ini = (rodada - 1) * 2 * N_I3_POR_BRACO
    de_a = {itens[j] for j in perm[ini:ini + N_I3_POR_BRACO]}
    de_b = {itens[j] for j in perm[ini + N_I3_POR_BRACO:ini + 2 * N_I3_POR_BRACO]}
    amostra = [r for r in respostas
               if (r.braco == "A" and (r.estrato, r.arxiv_id) in de_a)
               or (r.braco == "B" and (r.estrato, r.arxiv_id) in de_b)]
    amostra.sort(key=lambda r: (r.estrato, r.arxiv_id, r.braco))
    return [amostra[j] for j in rng.permutation(len(amostra))]


def kappa(x: list[str], y: list[str]) -> float:
    """κ de Cohen entre duas listas de rótulos pareadas."""
    n = len(x)
    if n == 0 or n != len(y):
        raise ValueError("listas vazias ou de tamanhos diferentes")
    cats = sorted(set(x) | set(y))
    po = sum(a == b for a, b in zip(x, y, strict=True)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return 1.0 if pe >= 1 else (po - pe) / (1 - pe)


def apurar_i3(humano: list[str], juiz: list[str]) -> dict:
    """A regra decide com o κ da variável que decide: `certo` contra o resto — só `certo`
    conta como acerto, e confundir `parcial` com `errado` não muda número nenhum. O κ nas
    quatro categorias vai junto, relatado."""
    binario_h = ["certo" if v == "certo" else "outro" for v in humano]
    binario_j = ["certo" if v == "certo" else "outro" for v in juiz]
    k = kappa(binario_h, binario_j)
    matriz = {h: {j: sum(1 for a, b in zip(humano, juiz, strict=True) if a == h and b == j)
                  for j in (*VEREDICTOS, "ilegivel")} for h in VEREDICTOS}
    return {"n": len(humano), "kappa_certo": round(k, 4),
            "kappa_4_categorias": round(kappa(humano, juiz), 4),
            "concordancia_certo": round(sum(a == b for a, b in
                                            zip(binario_h, binario_j, strict=True)) / len(humano), 4),
            "matriz_humano_x_juiz": matriz, "minimo": KAPPA_MINIMO,
            "passa": k >= KAPPA_MINIMO}


# ── R: a revisão dos erros de B ──────────────────────────────────────────────


def amostra_r(veredictos_b: dict[tuple, str], estrato: str = "primario",
              semente: int = SEMENTE) -> list[tuple]:
    """Os itens do estrato em que B não saiu `certo`; se passarem de `MAX_REVISAO_R`,
    `MAX_REVISAO_R` sorteados. Em ordem embaralhada."""
    nao_certos = sorted(k for k, v in veredictos_b.items() if k[0] == estrato and v != "certo")
    rng = np.random.default_rng(semente + 4)
    perm = rng.permutation(len(nao_certos))
    return [nao_certos[j] for j in perm[:MAX_REVISAO_R]]


# ── a regra ──────────────────────────────────────────────────────────────────


def _ic(amostras: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(amostras, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def acerto_b_com_r(certo_b: np.ndarray, classe: list[str | None], n_bootstrap: int = N_BOOTSTRAP,
                   semente: int = SEMENTE) -> tuple[float, tuple[float, float]]:
    """acerto_B depois de R, com IC por bootstrap pareado por item.

    `certo_b[i]` é 1 se o juiz disse `certo`. `classe[i]` é a classe de R para os não
    certos revisados, None para os não revisados (quando R foi por amostra). `juiz_errou`
    conta como acerto; `item_invalido` sai do denominador. O não revisado recebe as
    proporções da amostra revisada, e a amostra revisada é reamostrada junto — a
    incerteza de R entra no IC.
    """
    certo_b = np.asarray(certo_b, dtype=float)
    n = certo_b.size
    revisado = np.array([c is not None for c in classe])
    je = np.array([c == "juiz_errou" for c in classe], dtype=float)
    inv = np.array([c == "item_invalido" for c in classe], dtype=float)
    nao_certo = certo_b == 0
    faltam = nao_certo & ~revisado
    idx_rev = np.flatnonzero(nao_certo & revisado)

    def estimar(i_itens: np.ndarray, i_rev: np.ndarray) -> float:
        p_je = je[i_rev].mean() if i_rev.size else 0.0
        p_inv = inv[i_rev].mean() if i_rev.size else 0.0
        c, f = certo_b[i_itens], faltam[i_itens]
        rv = ~f & (c == 0)
        acertos = c.sum() + je[i_itens][rv].sum() + f.sum() * p_je
        validos = i_itens.size - inv[i_itens][rv].sum() - f.sum() * p_inv
        return float(acertos / validos) if validos > 0 else float("nan")

    ponto = estimar(np.arange(n), idx_rev)
    rng = np.random.default_rng(semente)
    reps = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        i_itens = rng.integers(0, n, n)
        i_rev = idx_rev[rng.integers(0, idx_rev.size, idx_rev.size)] if idx_rev.size else idx_rev
        reps[b] = estimar(i_itens, i_rev)
    return round(ponto, 4), _ic(reps)


def diferenca_pareada(x: np.ndarray, y: np.ndarray, n_bootstrap: int = N_BOOTSTRAP,
                      semente: int = SEMENTE) -> tuple[float, tuple[float, float]]:
    """mean(x − y) e o IC por bootstrap pareado por item."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    rng = np.random.default_rng(semente + 1)
    reps = d[rng.integers(0, d.size, (n_bootstrap, d.size))].mean(axis=1)
    return round(float(d.mean()), 4), _ic(reps)


def proporcao(x: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, semente: int = SEMENTE
              ) -> tuple[float, tuple[float, float]]:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(semente + 2)
    reps = x[rng.integers(0, x.size, (n_bootstrap, x.size))].mean(axis=1)
    return round(float(x.mean()), 4), _ic(reps)


def decidir(veredictos: dict[str, dict[tuple, str]], revisao_r: dict[tuple, str],
            i3: dict, estrato: str = "primario") -> dict:
    """Aplica a regra do DOC-13 §9.1 no estrato primário. `revisao_r` é {item: classe}
    dos itens revisados em R; `i3` é o resultado de `apurar_i3`."""
    ids = sorted(k for k in veredictos["B"] if k[0] == estrato)
    certo = {b: np.array([veredictos[b][k] == "certo" for k in ids], dtype=float)
             for b in ("A", "B", "C")}
    classe = [None if veredictos["B"][k] == "certo" else revisao_r.get(k) for k in ids]

    acerto_b, ic_b = acerto_b_com_r(certo["B"], classe)
    lacuna, ic_lacuna = diferenca_pareada(certo["B"], certo["A"])
    juiz = {b: proporcao(certo[b]) for b in ("A", "B", "C")}

    if ic_b[0] >= LIMIAR_GERADOR:
        gerador = "BASTA"
    elif ic_b[1] < LIMIAR_GERADOR:
        gerador = "LIMITA"
    else:
        gerador = "NÃO DECIDIDO"
    if ic_lacuna[0] > LIMIAR_BUSCA:
        busca = "LIMITA"
    elif ic_lacuna[1] < LIMIAR_BUSCA:
        busca = "NÃO LIMITA"
    else:
        busca = "NÃO DECIDIDO"

    # A mesma regra SEM a revisão R (só o juiz): relatada ao lado, não decide. Desde
    # 2026-10-04 R é feita pelo Claude, que escreveu os itens — se a decisão do gerador
    # depender de R, isto deixa à vista.
    ic_sem_r = juiz["B"][1]
    if ic_sem_r[0] >= LIMIAR_GERADOR:
        gerador_sem_r = "BASTA"
    elif ic_sem_r[1] < LIMIAR_GERADOR:
        gerador_sem_r = "LIMITA"
    else:
        gerador_sem_r = "NÃO DECIDIDO"

    i2_valida = juiz["C"][0] < juiz["B"][0] - MARGEM_I2
    if not i3.get("passa"):
        situacao = "NÃO DECIDIDO pelo instrumento (I3: κ abaixo do mínimo)"
    elif not i2_valida:
        situacao = "INVÁLIDO (I2: as perguntas se respondem de memória)"
    else:
        situacao = "válido"
    return {
        "estrato": estrato, "n_itens": len(ids), "situacao": situacao,
        "gerador": gerador if situacao == "válido" else "—",
        "gerador_sem_R": gerador_sem_r if situacao == "válido" else "—",
        "busca": busca if situacao == "válido" else "—",
        "acerto_B_com_R": acerto_b, "ic_acerto_B_com_R": ic_b,
        "limiar_gerador": LIMIAR_GERADOR,
        "lacuna_da_busca": lacuna, "ic_lacuna_da_busca": ic_lacuna,
        "limiar_busca": LIMIAR_BUSCA,
        "acerto_pelo_juiz": {b: {"ponto": v[0], "ic": v[1]} for b, v in juiz.items()},
        "i2": {"acerto_C": juiz["C"][0], "acerto_B_menos_margem":
               round(juiz["B"][0] - MARGEM_I2, 4), "valida": i2_valida},
        "r": {"nao_certos_B": int((certo["B"] == 0).sum()), "revisados": len(revisao_r),
              "por_classe": {c: sum(1 for v in revisao_r.values() if v == c)
                             for c in CLASSES_R}},
        "i3": {k: i3.get(k) for k in ("kappa_certo", "kappa_4_categorias", "passa")},
    }


def secundarias(itens: list[dict], respostas: list[RespostaDoBraco],
                veredictos: dict[str, dict[tuple, str]]) -> dict:
    """Relatadas, não decidem: por estrato, os três braços pelo juiz, A − C, P entre as
    fontes, P citada, citações removidas, abstenções e tempo."""
    saida: dict = {}
    for estrato in ("primario", "pos_corte"):
        rs = [r for r in respostas if r.estrato == estrato]
        if not rs:
            continue
        ids = sorted({(r.estrato, r.arxiv_id) for r in rs})
        certo = {b: np.array([veredictos[b][k] == "certo" for k in ids], dtype=float)
                 for b in ("A", "B", "C")}
        por = {b: [r for r in rs if r.braco == b] for b in ("A", "B", "C")}
        a_menos_c = diferenca_pareada(certo["A"], certo["C"])
        contagem = {b: {v: sum(1 for k in ids if veredictos[b][k] == v)
                        for v in (*VEREDICTOS, "ilegivel")} for b in ("A", "B", "C")}
        saida[estrato] = {
            "n_itens": len(ids),
            "acerto_pelo_juiz": {b: proporcao(certo[b])[0] for b in certo},
            "A_menos_C": {"ponto": a_menos_c[0], "ic": a_menos_c[1]},
            "veredictos": contagem,
            "p_entre_as_fontes_A": round(np.mean([r.posicao_p is not None for r in por["A"]]), 4),
            "p_citada_quando_presente": {
                b: round(float(np.mean([r.p_citada for r in por[b] if r.posicao_p])), 4)
                for b in ("A", "B") if any(r.posicao_p for r in por[b])},
            "respostas_com_citacao_removida": {
                b: sum(1 for r in por[b] if r.removidas) for b in ("A", "B", "C")},
            "b_igual_a_a": sum(1 for r in por["B"] if r.igual_a_A),
            "segundos_medianos": {b: round(float(np.median([r.segundos for r in por[b]
                                                            if not r.igual_a_A])), 1)
                                  for b in ("A", "B", "C")
                                  if any(not r.igual_a_A for r in por[b])},
        }
    return saida
