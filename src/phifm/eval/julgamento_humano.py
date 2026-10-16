"""O juiz contra físicos (DOC-13 §9.11, ACEITA em 2026-10-16).

As constantes abaixo SÃO a regra pré-registrada; o DOC-13 §9.11 é o texto dela.

## Por que

Todo resultado da §9 que passa pelo juiz automático (o Qwen3-8B, versão 2 da instrução)
sai rotulado "calibrado contra o Claude, sem revisão humana": em 2026-10-04 o dono decidiu
não julgar, e o Claude — que escreveu as perguntas e os gabaritos — ficou no lugar dele
(§9.1, desvio). Aqui, físicos de fora do projeto julgam uma amostra de respostas às cegas,
com a mesma regra de correção do juiz, e o κ diz se o juiz concorda com eles.

## A amostra — 100 respostas, em duas metades que respondem a perguntas diferentes

- **50 seladas:** sorteadas das 100 da rodada 2 do I3 (§9.1), que o Claude julgou às
  cegas com o hash no git antes de o juiz rodar. Nelas há três julgadores — os físicos, o
  juiz e o Claude —, e a pergunta é também se o Claude foi um bom substituto.
- **50 novas:** 25 de A′ (§9.2) e 25 de A5 (§9.9, o sistema de hoje), do primário do
  teste, de itens fora das duas rodadas do I3 e diferentes entre si. São as respostas das
  decisões mais recentes, que nenhum humano nem o Claude julgou.

## Às cegas

A folha de cada físico traz só pergunta, gabarito e resposta (sem as marcas [n], como o
juiz lê), em ordem própria. O id de cada item é opaco: nem o código-fonte da página diz o
braço, o veredicto do juiz ou o do Claude.
"""
from __future__ import annotations

import hashlib

import numpy as np

from phifm.eval.assistente import KAPPA_MINIMO, MS_MINIMO_POR_JULGAMENTO, julgamentos_rapidos
from phifm.eval.assistente_bracos import VEREDICTOS, _ic, apurar_i3, kappa

# ── a regra (DOC-13 §9.11) ───────────────────────────────────────────────────
# O sorteio e as folhas recusam enquanto isto for False. ACEITA pelo dono em 2026-10-16,
# como escrita.
REGRA_9_11_ACEITA = True
SEMENTE_HUMANO = 20261017
N_SELADOS = 50
N_NOVOS_POR_BRACO = 25
BRACOS_NOVOS = ("A2", "A5")
MIN_JULGADORES = 2
N_BOOTSTRAP_KAPPA = 10_000
# Mais de um quarto dos julgamentos a menos de 1 s do anterior não é leitura (§9.1, I1).
MAX_RAPIDOS = (N_SELADOS + 2 * N_NOVOS_POR_BRACO) // 4

ORIGENS = ("selado_r2", *BRACOS_NOVOS)


def id_opaco(pergunta: str, resposta: str, semente: int = SEMENTE_HUMANO) -> str:
    """O id que vai na folha: não diz nada além de "este item"."""
    return hashlib.sha256(f"{semente}\n{pergunta}\n{resposta}".encode()).hexdigest()[:12]


def sortear_selados(n_rodada_2: int, semente: int = SEMENTE_HUMANO) -> list[int]:
    """As posições, na amostra da rodada 2, das `N_SELADOS` que vão aos físicos."""
    if n_rodada_2 < N_SELADOS:
        raise ValueError(f"a rodada 2 tem {n_rodada_2} respostas; a regra pede {N_SELADOS}")
    return sorted(int(x) for x in np.random.default_rng(semente).permutation(n_rodada_2)
                  [:N_SELADOS])


def sortear_novos(chaves: list[tuple[str, str]], excluir: set[tuple[str, str]],
                  semente: int = SEMENTE_HUMANO) -> dict[str, list[tuple[str, str]]]:
    """`{braço: [(estrato, arxiv_id)]}`: itens do primário fora de `excluir` (as rodadas do
    I3), uma permutação fixa, os `N_NOVOS_POR_BRACO` primeiros para o primeiro braço e os
    seguintes para o segundo — itens diferentes, então nunca o mesmo texto duas vezes."""
    elegiveis = sorted(k for k in set(chaves) if k[0] == "primario" and k not in excluir)
    precisa = N_NOVOS_POR_BRACO * len(BRACOS_NOVOS)
    if len(elegiveis) < precisa:
        raise ValueError(f"{len(elegiveis)} itens elegíveis; a regra pede {precisa}")
    perm = np.random.default_rng(semente + 1).permutation(len(elegiveis))
    return {braco: [elegiveis[j] for j in perm[n * N_NOVOS_POR_BRACO:(n + 1) * N_NOVOS_POR_BRACO]]
            for n, braco in enumerate(BRACOS_NOVOS)}


def conferir_amostra(entradas: list[dict]) -> None:
    """A composição da regra, ids únicos, textos distintos e veredictos válidos — antes de
    selar. Levanta em qualquer desvio."""
    contagem = {o: sum(1 for e in entradas if e["origem"] == o) for o in ORIGENS}
    esperado = {"selado_r2": N_SELADOS, **dict.fromkeys(BRACOS_NOVOS, N_NOVOS_POR_BRACO)}
    if contagem != esperado or len(entradas) != sum(esperado.values()):
        raise ValueError(f"composição {contagem}, a regra pede {esperado}")
    if len({e["id"] for e in entradas}) != len(entradas):
        raise ValueError("ids repetidos na amostra")
    if len({(e["pergunta"], e["resposta"]) for e in entradas}) != len(entradas):
        raise ValueError("o mesmo texto aparece duas vezes na amostra")
    for e in entradas:
        if e["juiz"] not in (*VEREDICTOS, "ilegivel"):
            raise ValueError(f"{e['id']}: veredicto do juiz {e['juiz']!r}")
        if (e["origem"] == "selado_r2") != (e.get("claude") in VEREDICTOS):
            raise ValueError(f"{e['id']}: o veredicto do Claude existe só nas seladas")


def ordem_do_julgador(ids: list[str], julgador: str, semente: int = SEMENTE_HUMANO
                      ) -> list[str]:
    """Cada físico numa ordem própria e fixa: o cansaço não cai sempre nos mesmos itens."""
    s = int(hashlib.sha256(f"{semente}\n{julgador}".encode()).hexdigest()[:15], 16)
    return [sorted(ids)[j] for j in np.random.default_rng(s).permutation(len(ids))]


def itens_da_folha(entradas: list[dict], julgador: str) -> list[dict]:
    """O que vai na página: o id opaco e os três textos. Nada de braço, juiz ou Claude."""
    por_id = {e["id"]: e for e in entradas}
    return [{"id": i, "campos": [
        {"rotulo": "pergunta", "texto": por_id[i]["pergunta"], "classe": "pergunta"},
        {"rotulo": "gabarito", "texto": por_id[i]["gabarito"]},
        {"rotulo": "resposta do assistente", "texto": por_id[i]["resposta"]}]}
        for i in ordem_do_julgador(list(por_id), julgador)]


def ler_veredictos(baixado: dict, assinatura: str, ids: set[str]) -> dict[str, str]:
    """O JSON baixado de uma folha, conferido: desta folha, completo, nas quatro
    categorias, e lido — não clicado."""
    if baixado.get("assinatura_amostra") != assinatura:
        raise ValueError(f"veredictos de outra folha ({baixado.get('assinatura_amostra')} ≠ "
                         f"{assinatura})")
    lista = baixado["veredictos"]
    lidos = {v["id"]: v["veredicto"] for v in lista}
    if set(lidos) != ids or len(lista) != len(ids):
        raise ValueError(f"{len(lidos)} julgamentos de {len(ids)}: a folha está incompleta")
    fora = {x for x in lidos.values() if x not in VEREDICTOS}
    if fora:
        raise ValueError(f"veredictos fora das quatro categorias: {sorted(fora)}")
    rapidos = julgamentos_rapidos(lista)
    if rapidos is None or rapidos > MAX_RAPIDOS:
        raise ValueError(f"{rapidos} julgamentos a menos de {MS_MINIMO_POR_JULGAMENTO} ms do "
                         "anterior: isso não é leitura. A folha não vale.")
    return lidos


# ── a apuração ───────────────────────────────────────────────────────────────


def _certo(xs: list[str]) -> list[str]:
    return ["certo" if x == "certo" else "outro" for x in xs]


def kappa_ic(x: list[str], y: list[str], n_bootstrap: int = N_BOOTSTRAP_KAPPA,
             semente: int = SEMENTE_HUMANO) -> tuple[float, tuple[float, float]]:
    """κ em certa × resto e o IC 95% por bootstrap nos itens. Relatado: a regra decide pelo
    ponto, como no I3 — o IC mostra quão largo é decidir com 100 respostas."""
    bx = np.array([v == "certo" for v in x])
    by = np.array([v == "certo" for v in y])
    i = np.random.default_rng(semente + 2).integers(0, bx.size, (n_bootstrap, bx.size))
    xs, ys = bx[i], by[i]
    po = (xs == ys).mean(axis=1)
    px, py = xs.mean(axis=1), ys.mean(axis=1)
    pe = px * py + (1 - px) * (1 - py)
    # Reamostra com uma categoria só nos dois lados: κ = 1, como em `kappa`.
    with np.errstate(divide="ignore", invalid="ignore"):
        reps = np.where(pe >= 1, 1.0, (po - pe) / (1 - pe))
    return round(kappa(_certo(x), _certo(y)), 4), _ic(reps)


def _par(x: list[str], y: list[str]) -> dict:
    k, ic = kappa_ic(x, y)
    r = apurar_i3(x, y)
    return {"n": len(x), "kappa_certo": k, "ic95 (relatado)": ic,
            "kappa_4_categorias": r["kappa_4_categorias"],
            "concordancia_certo": r["concordancia_certo"]}


def apurar(entradas: list[dict], humanos: dict[str, dict[str, str]]) -> dict:
    """A regra: os físicos concordam entre si (guarda) e o juiz concorda com cada um deles.
    O resto é relatado."""
    if len(humanos) < MIN_JULGADORES:
        raise ValueError(f"{len(humanos)} julgador(es); a regra pede {MIN_JULGADORES}")
    ids = [e["id"] for e in entradas]
    for nome, v in humanos.items():
        if set(v) != set(ids):
            raise ValueError(f"{nome}: os veredictos não cobrem exatamente a amostra")
    juiz = [e["juiz"] for e in entradas]
    nomes = sorted(humanos)
    h = {n: [humanos[n][i] for i in ids] for n in nomes}

    entre = {f"{a} × {b}": _par(h[a], h[b])
             for k, a in enumerate(nomes) for b in nomes[k + 1:]}
    contra_juiz = {n: {**_par(h[n], juiz), "matriz_fisico_x_juiz":
                       apurar_i3(h[n], juiz)["matriz_humano_x_juiz"]} for n in nomes}
    min_entre = min(d["kappa_certo"] for d in entre.values())
    min_juiz = min(d["kappa_certo"] for d in contra_juiz.values())
    if min_entre < KAPPA_MINIMO:
        decisao = "inconclusivo"
    else:
        decisao = "passa" if min_juiz >= KAPPA_MINIMO else "nao_passa"

    sel = [k for k, e in enumerate(entradas) if e["origem"] == "selado_r2"]
    claude = [entradas[k]["claude"] for k in sel]
    por_origem = {}
    for origem in ORIGENS:
        ks = [k for k, e in enumerate(entradas) if e["origem"] == origem]
        so_juiz = [juiz[k] for k in ks]
        por_origem[origem] = {
            "n": len(ks),
            "acerto_pelo_juiz": round(sum(x == "certo" for x in so_juiz) / len(ks), 4),
            **{f"acerto_por_{n}": round(sum(h[n][k] == "certo" for k in ks) / len(ks), 4)
               for n in nomes},
            # A direção que infla o acerto: o juiz deu certo e o físico não.
            **{f"juiz_certo_{n}_nao": sum(juiz[k] == "certo" and h[n][k] != "certo"
                                         for k in ks) for n in nomes},
            **{f"{n}_certo_juiz_nao": sum(juiz[k] != "certo" and h[n][k] == "certo"
                                         for k in ks) for n in nomes},
            **{f"kappa_juiz_x_{n}": round(kappa(_certo([h[n][k] for k in ks]),
                                                _certo(so_juiz)), 4) for n in nomes}}
    return {
        "n": len(entradas), "julgadores": nomes, "minimo": KAPPA_MINIMO,
        "entre_os_fisicos (guarda)": entre,
        "juiz_x_fisico (decide)": contra_juiz,
        "decisao": decisao,
        "claude_x_fisico nas seladas (relatado)": {n: _par([h[n][k] for k in sel], claude)
                                                   for n in nomes},
        "claude_x_juiz nas seladas (relatado)": _par(claude, [juiz[k] for k in sel]),
        "por_origem (relatado)": por_origem,
    }


INTRO = """<p class=sutil>Obrigado por ajudar. São 100 respostas de um assistente de
pesquisa em Física a perguntas sobre artigos do arXiv. Cada cartão traz a <b>pergunta</b>,
o <b>gabarito</b> (escrito a partir do resumo do artigo que responde a ela) e a
<b>resposta</b> do assistente, sem as marcas de citação. A tarefa é dizer se a resposta dá o
que a pergunta pede, <b>como o gabarito diz</b>. Leva cerca de uma hora; o progresso fica
guardado no navegador, e dá para parar e voltar.</p>
<div class=aviso><b>Como julgar.</b> Primeiro ache o <b>núcleo</b> do gabarito: a parte que
responde diretamente à pergunta (um valor, um nome, o sentido de uma variação, um sim ou
não com o motivo). O gabarito muitas vezes traz complementos além do que se perguntou — um
segundo número, uma exceção, uma comparação, uma oração depois de travessão ou entre
parênteses. <b>Complementos não são exigidos.</b><br><br>
<b>Certa</b> — a resposta dá o núcleo e não contradiz o gabarito. Não baixam o veredicto:
outras palavras ou paráfrase; tradução ou termo em inglês; uma palavra estranha quando o
sentido é claro; outra ordem; número equivalente ou arredondado; complemento ausente;
informação a mais que não está no gabarito.<br>
<b>Parcial</b> — a resposta dá só um pedaço do núcleo (uma de duas coisas pedidas), ou uma
versão mais vaga dele ("muito maior" onde o núcleo é um fator), sem contradizer.<br>
<b>Errada</b> — dá outro núcleo, contradiz o gabarito ou responde outra coisa. Uma
resposta que primeiro dá o núcleo errado e depois também menciona o certo é errada.<br>
<b>Não respondeu</b> — diz que não pode responder (por exemplo, que as fontes não trazem a
informação) e não se compromete com uma resposta, mesmo que fale do assunto em volta.
<br><br>
Compare <b>só com o gabarito</b>: não use o que você sabe de Física para decidir quem está
certo. Você não sabe de qual versão do sistema veio cada resposta, nem o que o corretor
automático disse — de propósito. Por favor, não comente os itens com outra pessoa que
esteja julgando até os dois terminarem, e não repasse a folha: as perguntas são de um
conjunto de teste que não é publicado.</div>
<p class=sutil>Teclas <kbd>1</kbd>–<kbd>4</kbd> escolhem; <kbd>&larr;</kbd> volta. No fim,
<b>Baixar veredictos</b> salva um arquivo: é ele que volta para quem te mandou a folha.</p>"""
