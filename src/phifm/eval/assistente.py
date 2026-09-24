"""A medida do assistente: busca ou gerador — quem limita? (DOC-13 §9.1, ACEITA 2026-09-24)

As constantes abaixo SÃO a regra pré-registrada; o DOC-13 §9.1 é o texto dela. Mudar
um número aqui depois de ver resultado é trocar a regra, e exige registro no DOC-13.

Este módulo cobre a primeira etapa — os ITENS e a guarda I1 (a validade das perguntas,
conferida pelo dono antes de qualquer braço rodar). Os braços vêm depois de I1 passar.

## Como um item nasce

1. Um artigo P é sorteado das âncoras de validação (fora do treino do encoder, dentro do
   índice). Dois estratos DISJUNTOS: antes e depois de `CORTE`, o lançamento do Qwen3.
2. O Qwen3-8B lê só o RESUMO de P — sem o título, para não copiá-lo — e escreve uma
   pergunta em português cuja resposta é um fato do resumo, mais o gabarito.
3. Guarda de cópia: sai a pergunta com metade ou mais das palavras de conteúdo no título.

Cada tentativa, aceita ou não, fica registrada com o motivo: um conjunto que só guarda
os aceitos esconde quanto o filtro jogou fora.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

# ── a regra (DOC-13 §9.1) ────────────────────────────────────────────────────
# ⚠️ Duas sementes. A primeira versão do gerador rodou com 20260924 e eu li as 40
# perguntas: elas viraram conjunto de DESENVOLVIMENTO. Ajustar o gerador olhando para
# esses artigos e depois validá-lo (I1) nos mesmos inflaria a validade. Então o conjunto
# formal usa outra semente e EXCLUI os primeiros `N_DEV_*` artigos da ordem de dev.
SEMENTE_DEV = 20260924
SEMENTE = 20260925
N_DEV_PRIMARIO = 100
N_DEV_POS_CORTE = 20
CORTE = "2025-06-01"               # Qwen3 lançado no fim de abril de 2025
N_PRIMARIO = 500
N_POS_CORTE = 150
N_REVISAO_I1 = 40
MINIMO_VALIDAS_I1 = 32
MAX_SOBREPOSICAO_TITULO = 0.5
LIMIAR_GERADOR = 0.90
LIMIAR_BUSCA = 0.05
MARGEM_I2 = 0.10
KAPPA_MINIMO = 0.6
K_FONTES = 6

# Versão 2. A versão 1 (sha256 em `avaliacao/assistente_itens_dev_v1.json`) deu, na
# minha triagem das 40 de dev, perguntas que só fazem sentido depois de ler o artigo
# ("previsto pelo modelo", "o sistema proposto no estudo", `r_1` sem definição) e
# gabaritos que não respondem ("é grande", "é avaliado em forma fechada"). Os exemplos
# abaixo são desses dois defeitos.
SISTEMA_PERGUNTA = (
    "You are preparing an evaluation of a physics literature assistant. You receive the "
    "abstract of a physics paper; its title is withheld. Decide whether the abstract states "
    "a concrete, checkable finding, and if so write ONE question, in Brazilian Portuguese, "
    "whose answer is that finding.\n\n"
    "The question will be asked to an assistant that has NEVER seen this abstract, so it "
    "must stand on its own:\n"
    "- Name the specific system, material, object, process or model explicitly. Never say "
    "'o modelo proposto', 'o sistema estudado', 'o método apresentado', 'o formalismo', "
    "'o enfoque', 'neste trabalho', 'os autores', or anything that only makes sense after "
    "reading the paper.\n"
    "- Do not use symbols defined only in the paper (a bare r_1, H_Q or v_Delta); say in "
    "words what they are, or ask about something else.\n"
    "- Paraphrase; do not copy distinctive phrases from the abstract.\n\n"
    "The answer (gabarito) must be the specific content, in one short sentence:\n"
    "- a number with units, a sign or direction of change, a named mechanism, a named "
    "material or particle, a scaling law, or a yes/no together with its condition;\n"
    "- it must fully answer the question and be stated in the abstract.\n"
    "If the abstract only says that something is 'large', 'confirmed', 'discussed', "
    "'derived', 'obtained in closed form' or 'competitive', without giving the content, "
    "there is no checkable finding.\n\n"
    "Examples of what NOT to write, and why:\n"
    "- 'Qual é o cenário de mistura previsto pelo modelo?' (which model? the reader cannot "
    "know)\n"
    "- 'Qual é a forma das autofunções dos modos centrais?' with gabarito 'São avaliadas em "
    "forma fechada.' (the gabarito does not give the form)\n"
    "Examples of good items:\n"
    '- {"pergunta": "Qual é a massa mínima do companheiro da estrela TYC 4110-01037-1, '
    'supondo sen i = 1?", "gabarito": "Cerca de 97,7 ± 5,8 massas de Júpiter."}\n'
    '- {"pergunta": "Em filmes finos de hélio adsorvido, o que acontece com a temperatura '
    'em que surge a rigidez quando a cobertura se aproxima da cobertura crítica?", '
    '"gabarito": "Ela vai a zero kelvin."}\n\n'
    "If there is no checkable finding, output exactly: NENHUMA\n"
    "Otherwise output only one JSON object, with both fields in Brazilian Portuguese:\n"
    '{"pergunta": "...", "gabarito": "..."}')

# Versão 3: a versão 2 NÃO melhorou na triagem (~20 de 40 de novo) e copiou um dos
# exemplos para um artigo sobre outra coisa. Duas mudanças: o modelo RACIOCINA antes de
# escrever (o modo thinking do Qwen3), e um segundo passo, o CRÍTICO, relê cada item
# contra o resumo e o derruba se ele falhar em qualquer um dos três critérios de I1.
# O crítico não substitui I1 — é o mesmo modelo — só tira o lixo antes de o dono ver.
PENSAR = True
MAX_TOKENS_PENSANDO = 2500

SISTEMA_CRITICO = (
    "You audit one item of an evaluation set for a physics literature assistant. You get "
    "the abstract of a paper, a question written from it, and the reference answer "
    "(gabarito). Judge three things strictly:\n"
    "1. autonoma: could a physicist who has NOT read this abstract understand exactly what "
    "is being asked? Fail it if the question relies on 'the model', 'the system', 'the "
    "material', 'the formalism', 'the experiment' without naming it, or on symbols defined "
    "only in the paper.\n"
    "2. responde: does the gabarito give the specific content the question asks for (a "
    "value, a direction, a named mechanism, material or law)? Fail it if the gabarito only "
    "restates the question, says something is 'large', 'suppressed', 'competitive', "
    "'obtained in closed form', or answers a different question.\n"
    "3. sustentada: is the gabarito stated in the abstract (paraphrase is fine)?\n"
    "Output only one JSON object: "
    '{"autonoma": true|false, "responde": true|false, "sustentada": true|false}')

# Guarda mecânica da mesma falha: expressões que só fazem sentido para quem leu o artigo.
# Estreita de propósito — "o mecanismo proposto por Kibble" também cai, e é raro.
_DEPENDE_DO_ARTIGO = re.compile(
    r"(?i)\b(propost[oa]s?|apresentad[oa]s?|estudad[oa]s?|analisad[oa]s?|considerad[oa]s?"
    r"|investigad[oa]s?|neste (?:trabalho|estudo|artigo)|nest[ea] (?:modelo|abordagem)"
    r"|(?:no|do|pelo) (?:estudo|trabalho|artigo)|os autores"
    # "pelo modelo de Hubbard" e "segundo o formalismo de Keldysh" nomeiam a coisa: passam.
    r"|(?:pelo modelo|segundo o (?:formalismo|enfoque|modelo|método|trabalho))(?! d[eoa]s? )"
    r"|descrit[oa]s?|mencionad[oa]s?|em quest[ãa]o)\b")

# As perguntas dos exemplos do prompt. A versão 2 copiou uma delas, palavra por palavra,
# para um artigo sobre outro assunto.
_PERGUNTAS_EXEMPLO = re.findall(r'"pergunta": "([^"]+)"', SISTEMA_PERGUNTA) + [
    "Qual é o cenário de mistura previsto pelo modelo?",
    "Qual é a forma das autofunções dos modos centrais?"]


def assinatura_gerador() -> str:
    """O que muda as perguntas: os dois prompts e o modo de raciocínio. É a chave do cache
    — mudar qualquer um e reaproveitar tentativas antigas misturaria versões."""
    import hashlib

    return hashlib.sha256(
        f"{SISTEMA_PERGUNTA}\n{SISTEMA_CRITICO}\n{PENSAR}".encode()).hexdigest()


# Palavras que não carregam assunto, nas duas línguas: a pergunta é em português e o
# título em inglês, e a guarda só deve ver termos técnicos em comum.
_TEXTO_VAZIAS = """
para como qual quais quando onde entre sobre pelo pela pelos pelas numa numas este esta
isso isto esse essa estes estas esses essas seus suas sua seu mais menos muito muita
qual que com sem dos das nos nas aos uma umas uns ser sao sera foram tem pode podem
modo forma tipo caso efeito efeitos segundo atraves ainda tambem apos
with from that this these those their there where which what when into onto over under
between about using based study studies paper results result show shows shown than then
they them have been were does model models effect effects new via its also such
"""
_VAZIAS = set(_TEXTO_VAZIAS.split())


def _sem_acento(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def palavras_de_conteudo(texto: str) -> set[str]:
    """Minúsculas, sem acento, ≥ 4 caracteres, fora da lista de vazias."""
    return {p for p in re.findall(r"[a-z0-9]+", _sem_acento(texto.lower()))
            if len(p) >= 4 and p not in _VAZIAS}


def sobreposicao_com_titulo(pergunta: str, titulo: str) -> float:
    """Fração das palavras de conteúdo da PERGUNTA que estão no título."""
    q = palavras_de_conteudo(pergunta)
    return len(q & palavras_de_conteudo(titulo)) / len(q) if q else 0.0


def ler_pergunta(bruto: str) -> tuple[str, str] | None:
    """`(pergunta, gabarito)` da saída do modelo, ou None se ele disse NENHUMA / não seguiu
    o formato. Aceita o JSON cercado de texto ou de ```json — o modelo às vezes cerca."""
    if re.fullmatch(r"\W*NENHUMA\W*", bruto.strip(), re.IGNORECASE):
        return None
    m = re.search(r"\{.*\}", bruto, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    p, g = str(d.get("pergunta", "")).strip(), str(d.get("gabarito", "")).strip()
    return (p, g) if p and g else None


# ── o sorteio ────────────────────────────────────────────────────────────────


def sortear(pares_validacao: Path, spine: Path, semente: int = SEMENTE,
            excluir: frozenset[str] | set[str] = frozenset()) -> dict[str, list[str]]:
    """As âncoras de validação em ordem aleatória, por estrato: `primario` (antes do
    corte) e `pos_corte`. A ordem inteira é devolvida; o gerador anda nela até completar
    a cota, e a mesma semente dá a mesma ordem em qualquer máquina. `excluir` sai DEPOIS
    de embaralhar, para não mudar a ordem dos que ficam."""
    ancoras = pl.scan_parquet(pares_validacao).select("arxiv_id").unique()
    d = (pl.scan_parquet(spine).select("arxiv_id", "created")
         .join(ancoras, on="arxiv_id", how="semi")
         .sort("arxiv_id").collect())
    rng = np.random.default_rng(semente)
    saida = {}
    for nome, filtro in (("primario", pl.col("created") < CORTE),
                         ("pos_corte", pl.col("created") >= CORTE)):
        ids = d.filter(filtro)["arxiv_id"].to_list()
        saida[nome] = [ids[i] for i in rng.permutation(len(ids)) if ids[i] not in excluir]
    return saida


def ordem_formal(pares_validacao: Path, spine: Path) -> dict[str, list[str]]:
    """A ordem do conjunto formal: `SEMENTE`, sem os artigos que o desenvolvimento viu."""
    dev = sortear(pares_validacao, spine, SEMENTE_DEV)
    vistos = set(dev["primario"][:N_DEV_PRIMARIO]) | set(dev["pos_corte"][:N_DEV_POS_CORTE])
    return sortear(pares_validacao, spine, SEMENTE, excluir=vistos)


def cotas_de_revisao(n_revisao: int = N_REVISAO_I1) -> dict[str, int]:
    """A amostra de I1 na proporção dos estratos: 40 → 31 do primário e 9 do pós-corte."""
    primario = round(n_revisao * N_PRIMARIO / (N_PRIMARIO + N_POS_CORTE))
    return {"primario": primario, "pos_corte": n_revisao - primario}


# ── a geração ────────────────────────────────────────────────────────────────


@dataclass
class Tentativa:
    estrato: str
    ordem: int                    # posição na permutação do estrato
    arxiv_id: str
    situacao: str                 # aceita · nenhuma · formato · depende_do_artigo ·
    #                               copia_exemplo · copia_titulo · reprovada_critico
    pergunta: str = ""
    gabarito: str = ""
    sobreposicao: float = 0.0
    critica: str = ""             # os critérios em que o crítico reprovou


def criticar(modelo, resumo: str, pergunta: str, gabarito: str) -> list[str]:
    """Os critérios em que o item FALHA, segundo o crítico. Saída fora do formato conta
    como falha dos três: item que não se consegue auditar não entra."""
    bruto = modelo.gerar(SISTEMA_CRITICO,
                         f"Abstract:\n{resumo}\n\nQuestion: {pergunta}\nGabarito: {gabarito}",
                         max_tokens=MAX_TOKENS_PENSANDO if PENSAR else 100,
                         temperatura=0.0, semente=SEMENTE, pensar=PENSAR)
    m = re.search(r"\{.*\}", bruto, re.DOTALL)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return [k for k in ("autonoma", "responde", "sustentada") if d.get(k) is not True]


def tentar(modelo, estrato: str, ordem: int, arxiv_id: str, titulo: str,
           resumo: str) -> Tentativa:
    bruto = modelo.gerar(SISTEMA_PERGUNTA, resumo,
                         max_tokens=MAX_TOKENS_PENSANDO if PENSAR else 300,
                         temperatura=0.0, semente=SEMENTE, pensar=PENSAR)
    if re.fullmatch(r"\W*NENHUMA\W*", bruto.strip(), re.IGNORECASE):
        return Tentativa(estrato, ordem, arxiv_id, "nenhuma")
    lido = ler_pergunta(bruto)
    if lido is None:
        return Tentativa(estrato, ordem, arxiv_id, "formato")
    t = guardas(estrato, ordem, arxiv_id, titulo, *lido)
    if t.situacao != "aceita":
        return t
    falhas = criticar(modelo, resumo, t.pergunta, t.gabarito)
    if falhas:
        t.situacao, t.critica = "reprovada_critico", ",".join(falhas)
    return t


def guardas(estrato: str, ordem: int, arxiv_id: str, titulo: str, pergunta: str,
            gabarito: str) -> Tentativa:
    """As guardas mecânicas, iguais para qualquer autor: dependência do artigo, cópia de
    exemplo do prompt, cópia do título."""
    s = round(sobreposicao_com_titulo(pergunta, titulo), 4)

    def caiu(situacao: str) -> Tentativa:
        return Tentativa(estrato, ordem, arxiv_id, situacao, pergunta, gabarito, s)

    if _DEPENDE_DO_ARTIGO.search(pergunta):
        return caiu("depende_do_artigo")
    if any(sobreposicao_com_titulo(pergunta, e) >= 0.6 for e in _PERGUNTAS_EXEMPLO):
        return caiu("copia_exemplo")
    if s >= MAX_SOBREPOSICAO_TITULO:
        return caiu("copia_titulo")
    return caiu("aceita")


def gerar(modelo, ordem: dict[str, list[str]], spine: Path, cotas: dict[str, int],
          cache: Path, ao_aceitar=None) -> list[Tentativa]:
    """Anda em cada permutação até `cotas[estrato]` aceitas. Retomável: cada tentativa vai
    para `cache` (uma linha JSON) ao terminar, e o que já está lá não é refeito."""
    feitas: dict[tuple[str, str], Tentativa] = {}
    if cache.exists():
        for linha in cache.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                t = Tentativa(**json.loads(linha))
                feitas[(t.estrato, t.arxiv_id)] = t
    s = pl.scan_parquet(spine).select("arxiv_id", "title", "abstract")
    cache.parent.mkdir(parents=True, exist_ok=True)
    saida: list[Tentativa] = []
    with cache.open("a", encoding="utf-8") as f:
        for estrato, ids in ordem.items():
            aceitas = 0
            for i, aid in enumerate(ids):
                if aceitas >= cotas.get(estrato, 0):
                    break
                t = feitas.get((estrato, aid))
                if t is None:
                    meta = s.filter(pl.col("arxiv_id") == aid).collect().row(0, named=True)
                    resumo = " ".join((meta["abstract"] or "").split())
                    t = tentar(modelo, estrato, i, aid, meta["title"] or "", resumo)
                    f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
                    f.flush()
                saida.append(t)
                if t.situacao == "aceita":
                    aceitas += 1
                    if ao_aceitar:
                        ao_aceitar(t, aceitas)
    return saida


# ── o conjunto formal: o Claude escreve, o script confere ────────────────────
# Decisão do dono (DOC-13 §9.1): as perguntas do conjunto formal são escritas pelo Claude,
# com as regras de `SISTEMA_PERGUNTA`, lendo só o resumo. O script exporta os resumos em
# lotes, na ordem do sorteio, e importa as respostas passando pelas mesmas `guardas`.
AUTOR = "claude-opus-5-5"


def assinatura_autor() -> str:
    import hashlib

    return hashlib.sha256(f"{SISTEMA_PERGUNTA}\n{AUTOR}".encode()).hexdigest()


def ler_cache(cache: Path) -> dict[tuple[str, str], Tentativa]:
    feitas: dict[tuple[str, str], Tentativa] = {}
    if cache.exists():
        for linha in cache.read_text(encoding="utf-8").splitlines():
            if linha.strip():
                t = Tentativa(**json.loads(linha))
                feitas[(t.estrato, t.arxiv_id)] = t
    return feitas


def pendentes(ordem: dict[str, list[str]], feitas: dict, estrato: str, n: int) -> list[dict]:
    """Os próximos `n` artigos do estrato ainda sem tentativa, EM ORDEM, sem lacuna."""
    saida = []
    for i, aid in enumerate(ordem[estrato]):
        if (estrato, aid) in feitas:
            if saida:
                raise RuntimeError(f"lacuna na ordem: {aid} tem tentativa depois de pendentes")
            continue
        saida.append({"estrato": estrato, "ordem": i, "arxiv_id": aid})
        if len(saida) >= n:
            break
    return saida


def importar(respostas: list[dict], ordem: dict[str, list[str]], spine: Path,
             cache: Path) -> list[Tentativa]:
    """Confere e registra um lote escrito pelo Claude. Cada resposta é `{estrato,
    arxiv_id, pergunta, gabarito}` ou `{estrato, arxiv_id, nenhuma: true}`. Recusa o lote
    inteiro se ele não for exatamente a continuação da ordem do estrato."""
    feitas = ler_cache(cache)
    por_estrato: dict[str, list[dict]] = {}
    for r in respostas:
        por_estrato.setdefault(r["estrato"], []).append(r)
    novas: list[Tentativa] = []
    titulos = dict(pl.scan_parquet(spine).select("arxiv_id", "title")
                   .filter(pl.col("arxiv_id").is_in([r["arxiv_id"] for r in respostas]))
                   .collect().iter_rows())
    for estrato, lote in por_estrato.items():
        esperado = pendentes(ordem, feitas, estrato, len(lote))
        if [e["arxiv_id"] for e in esperado] != [r["arxiv_id"] for r in lote]:
            raise ValueError(f"o lote de {estrato} não continua a ordem do sorteio")
        for e, r in zip(esperado, lote, strict=True):
            if r.get("nenhuma"):
                novas.append(Tentativa(estrato, e["ordem"], r["arxiv_id"], "nenhuma"))
            else:
                novas.append(guardas(estrato, e["ordem"], r["arxiv_id"],
                                     titulos.get(r["arxiv_id"]) or "",
                                     r["pergunta"].strip(), r["gabarito"].strip()))
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("a", encoding="utf-8") as f:
        for t in novas:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
    return novas


def aceitas_em_ordem(ordem: dict[str, list[str]], feitas: dict,
                     cotas: dict[str, int]) -> list[Tentativa]:
    """As primeiras `cotas[estrato]` aceitas de cada estrato, andando na ordem. Recusa se
    a cota não foi alcançada: o conjunto não sai incompleto em silêncio."""
    saida: list[Tentativa] = []
    for estrato, n in cotas.items():
        achadas = []
        for aid in ordem[estrato]:
            if len(achadas) >= n:
                break
            t = feitas.get((estrato, aid))
            if t is None:
                break
            if t.situacao == "aceita":
                achadas.append(t)
        if len(achadas) < n:
            raise RuntimeError(f"{estrato}: {len(achadas)} aceitas de {n} — escreva mais lotes")
        saida += achadas
    return saida


# ── I1: a apuração da revisão do dono ────────────────────────────────────────


def apurar_i1(veredictos: dict) -> dict:
    """Conta as válidas e aplica a guarda. `veredictos` é o JSON baixado da folha."""
    lista = veredictos["veredictos"]
    validas = sum(1 for v in lista if v["veredicto"] == "valida")
    motivos: dict[str, int] = {}
    for v in lista:
        if v["veredicto"] != "valida":
            motivos[v["veredicto"]] = motivos.get(v["veredicto"], 0) + 1
    completa = len(lista) == veredictos["n_amostra"] == N_REVISAO_I1
    return {"julgadas": len(lista), "validas": validas, "invalidas_por_motivo": motivos,
            "minimo": MINIMO_VALIDAS_I1, "completa": completa,
            "passa": completa and validas >= MINIMO_VALIDAS_I1}
