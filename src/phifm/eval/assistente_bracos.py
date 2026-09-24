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

SISTEMA_JUIZ = (
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

    t0 = time.perf_counter()
    bruto = assistente.modelo.gerar(
        SISTEMA_MEMORIA, f"Question: {pergunta}\n\n{instrucao_de_idioma(pergunta)}",
        semente=assistente.semente)
    texto, citadas, removidas, sem_fonte = verificar_citacoes(bruto, 0)
    rc = RespostaDoBraco(item["estrato"], item["ordem"], item["arxiv_id"], "C", texto,
                         citadas=citadas, removidas=removidas,
                         n_frases_sem_fonte=len(sem_fonte),
                         segundos=round(time.perf_counter() - t0, 2))
    return [ra, rb, rc]


def ler_respostas(cache: Path) -> list[RespostaDoBraco]:
    if not cache.exists():
        return []
    return [RespostaDoBraco(**json.loads(linha))
            for linha in cache.read_text(encoding="utf-8").splitlines() if linha.strip()]


def rodar(assistente, itens: list[dict], cache: Path, limite: int | None = None,
          ao_terminar=None) -> list[RespostaDoBraco]:
    """Os três braços em cada item, na ordem de `itens`. Retomável: os três registros de
    um item vão ao cache JUNTOS, ao fim do item — um item nunca fica pela metade."""
    feitos = {(r.estrato, r.arxiv_id) for r in ler_respostas(cache)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with cache.open("a", encoding="utf-8") as f:
        for item in itens:
            if (item["estrato"], item["arxiv_id"]) in feitos:
                continue
            if limite is not None and n >= limite:
                break
            regs = rodar_item(assistente, item)
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
                         max_tokens=200, temperatura=0.0, semente=SEMENTE)
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
        saida[r.braco][(r.estrato, r.arxiv_id)] = julgamentos[chave]["veredicto"]
    return saida


# ── I3: a concordância do juiz com o dono ────────────────────────────────────


def amostra_i3(respostas: list[RespostaDoBraco], semente: int = SEMENTE
               ) -> list[RespostaDoBraco]:
    """50 respostas de A e 50 de B, de itens DIFERENTES (onde B = A, o mesmo item daria
    o mesmo texto duas vezes), embaralhadas: o dono não sabe o braço."""
    itens = sorted({(r.estrato, r.arxiv_id) for r in respostas})
    rng = np.random.default_rng(semente + 3)
    perm = rng.permutation(len(itens))
    de_a = {itens[j] for j in perm[:N_I3_POR_BRACO]}
    de_b = {itens[j] for j in perm[N_I3_POR_BRACO:2 * N_I3_POR_BRACO]}
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
