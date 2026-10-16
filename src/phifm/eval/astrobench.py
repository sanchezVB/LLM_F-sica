"""Um teste de fora: o AstroBench (DOC-13 §9.10, ACEITA em 2026-10-16).

As constantes abaixo SÃO a regra pré-registrada; o DOC-13 §9.10 é o texto dela. Mudar
um número aqui depois de ver resultado é trocar a regra, e exige registro no DOC-13.

## Por que este conjunto

Tudo o que a §9 mediu usa perguntas que o próprio Qwen3-8B escreveu a partir de resumos
do nosso corpo, julgadas pelo mesmo modelo. O AstroBench (AstroMLab) foi escrito por
outros, com outro método, sobre um campo que o índice cobre (astro-ph, ~476 mil
artigos), e é de múltipla escolha: o acerto não depende de juiz nenhum.

## O gabarito não é nosso

A versão pública (`AstroMLab/Astrobench_MCQ_v1_Public`) traz a pergunta e as quatro
opções, SEM a resposta. A correção é da equipe do AstroMLab, a partir das respostas
enviadas por e-mail. Este módulo gera os arquivos de envio e lê o que voltar: acerto por
item (a regra pareada) ou só o acerto de cada arquivo (a leitura degradada, declarada).
Um arquivo com gabarito próprio (coluna `answer`) é corrigido aqui mesmo.

## Os dois braços — mesmo modelo, mesma configuração, semente fixa

- **C · sem fontes** — o enunciado e as quatro opções; uma linha, `Answer: X`.
- **A · com a busca do sistema** — a busca do produto (consulta hipotética → densa
  fundida ao BM25 → ΦRank nos 50 → 6 fontes), com o enunciado inteiro como consulta; as
  6 fontes vêm antes da pergunta, e a instrução manda usá-las quando ajudam.

A instrução de resposta NÃO é a do produto: a do produto manda responder só com as
fontes e dizer quando elas não bastam, e na múltipla escolha a abstenção é erro — a
medida misturaria "a busca ajuda" com "a política de abstenção". O que se mede é a busca,
com o gerador fixo entre os braços.
"""
from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phifm.eval.assistente import SEMENTE
from phifm.eval.assistente_bracos import diferenca_pareada, proporcao
from phifm.eval.statistics.proporcao import binomial_exata_bicaudal, wilson
from phifm.rag.assistente import Fonte, bloco_de_fontes

# ── a regra (DOC-13 §9.10) ───────────────────────────────────────────────────
# As fases que tocam os 1.000 do teste recusam enquanto isto for False; o
# desenvolvimento (50 itens disjuntos) roda sempre. ACEITA pelo dono em 2026-10-16,
# como escrita.
REGRA_9_10_ACEITA = True
CONJUNTO = "AstroMLab/Astrobench_MCQ_v1_Public"
SEMENTE_ASTROBENCH = 20261016     # a ordem dos itens; a da geração é a do produto
N_TESTE = 1000
N_DEV = 50
MIN_LEITURA = 0.98                # guarda: letra lida em ≥ 98% das respostas de cada braço
MIN_ACERTO_C = 0.50               # guarda: abaixo disto, prompt ou leitura quebrados
LIMIAR_MATERIAL = 0.03            # o de §9.2 e §9.9: relatado, não muda a leitura
MAX_TOKENS_RESPOSTA = 32
ENDERECO_DE_CORRECAO = "astromachinelearninglab@gmail.com"

COLUNAS = ("question", "A", "B", "C", "D")
LETRAS = ("A", "B", "C", "D")
COLUNAS_GABARITO = ("answer", "Answer", "correct_answer", "gabarito", "label")

SISTEMA_SEM_FONTES = (
    "You are an expert in astronomy and astrophysics. Answer the multiple-choice question "
    "below. Reply with a single line, in the form 'Answer: X', where X is A, B, C or D.")
SISTEMA_COM_FONTES = (
    "You are an expert in astronomy and astrophysics. Answer the multiple-choice question "
    "below. Before it come numbered research papers (title, year and abstract) found by a "
    "search engine for this question; they may or may not be relevant. Use them when they "
    "help, and your own knowledge otherwise. Reply with a single line, in the form "
    "'Answer: X', where X is A, B, C or D.")


@dataclass(frozen=True)
class Item:
    indice: int                   # a linha no arquivo público, a partir de 0
    id: str                       # sha256 do enunciado e das opções, 16 hex
    pergunta: str
    opcoes: tuple[str, str, str, str]
    gabarito: str | None = None


def _texto(x) -> str:
    return " ".join(str(x if x is not None else "").split())


def id_do_item(pergunta: str, opcoes: tuple[str, ...]) -> str:
    """O item por conteúdo: casa a resposta com a pergunta mesmo se a ordem das linhas mudar
    do lado de quem corrige."""
    return hashlib.sha256("\n".join((pergunta, *opcoes)).encode("utf-8")).hexdigest()[:16]


def _ler_tabela(arquivo: Path):
    import polars as pl

    sufixo = arquivo.suffix.lower()
    if sufixo == ".parquet":
        return pl.read_parquet(arquivo)
    if sufixo == ".csv":
        return pl.read_csv(arquivo, infer_schema_length=0)
    if sufixo in (".jsonl", ".ndjson"):
        return pl.read_ndjson(arquivo)
    if sufixo == ".json":
        return pl.read_json(arquivo)
    raise ValueError(f"{arquivo.name}: formato não reconhecido (parquet, csv, jsonl ou json)")


def carregar(arquivo: Path) -> list[Item]:
    """Os itens, na ordem do arquivo. Recusa arquivo sem as cinco colunas, enunciado ou
    opção vazios, e gabarito fora de A–D — um conjunto mal lido não vira medida."""
    df = _ler_tabela(Path(arquivo))
    faltam = [c for c in COLUNAS if c not in df.columns]
    if faltam:
        raise ValueError(f"{Path(arquivo).name}: faltam as colunas {faltam} "
                         f"(tem {df.columns}); o AstroBench público traz {list(COLUNAS)}")
    col_gab = next((c for c in COLUNAS_GABARITO if c in df.columns), None)
    itens, vazios = [], []
    for i, linha in enumerate(df.iter_rows(named=True)):
        pergunta = _texto(linha["question"])
        opcoes = tuple(_texto(linha[c]) for c in LETRAS)
        if not pergunta or not all(opcoes):
            vazios.append(i)
            continue
        gab = None
        if col_gab is not None:
            gab = _texto(linha[col_gab]).upper().strip("(). ")
            if gab not in LETRAS:
                raise ValueError(f"linha {i}: gabarito {linha[col_gab]!r} fora de A–D")
        itens.append(Item(i, id_do_item(pergunta, opcoes), pergunta, opcoes, gab))
    if vazios:
        raise ValueError(f"{len(vazios)} linhas com enunciado ou opção vazios (a primeira: "
                         f"{vazios[0]}): confira o arquivo antes de medir")
    return itens


def sha256_arquivo(arquivo: Path) -> str:
    h = hashlib.sha256()
    with Path(arquivo).open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def manifesto(arquivo: Path, itens: list[Item], origem: str) -> dict:
    """O que congela o conjunto: o hash do arquivo e o da ordem dos itens. Versionado; as
    fases conferem contra ele antes de rodar."""
    teste, dev = amostras(len(itens))
    return {"conjunto": CONJUNTO, "origem": origem, "arquivo": Path(arquivo).name,
            "sha256_arquivo": sha256_arquivo(arquivo), "linhas": len(itens),
            "sha256_ids": hashlib.sha256("\n".join(i.id for i in itens).encode()).hexdigest(),
            "ids_repetidos": len(itens) - len({i.id for i in itens}),
            "tem_gabarito": all(i.gabarito for i in itens),
            "semente": SEMENTE_ASTROBENCH, "n_teste": len(teste), "n_dev": len(dev)}


def conferir_manifesto(arquivo: Path, itens: list[Item], registrado: dict) -> None:
    atual = {"sha256_arquivo": sha256_arquivo(arquivo), "linhas": len(itens),
             "sha256_ids": hashlib.sha256("\n".join(i.id for i in itens).encode()).hexdigest()}
    for chave, valor in atual.items():
        if registrado.get(chave) != valor:
            raise ValueError(f"o arquivo não é o registrado no manifesto ({chave}: "
                             f"{registrado.get(chave)} ≠ {valor}): alguém trocou o conjunto "
                             "depois de congelado.")


def amostras(n_linhas: int, semente: int = SEMENTE_ASTROBENCH, n_teste: int = N_TESTE,
             n_dev: int = N_DEV) -> tuple[list[int], list[int]]:
    """`(teste, dev)`: posições numa permutação fixa — os `n_teste` primeiros e os `n_dev`
    seguintes, disjuntos. Na ordem da permutação: uma rodada parada no meio é uma amostra
    aleatória, não o começo do arquivo."""
    if n_linhas < n_teste + n_dev:
        raise ValueError(f"o conjunto tem {n_linhas} linhas; a regra pede {n_teste} + {n_dev}")
    perm = np.random.default_rng(semente).permutation(n_linhas)
    return [int(x) for x in perm[:n_teste]], [int(x) for x in perm[n_teste:n_teste + n_dev]]


# ── os braços ────────────────────────────────────────────────────────────────


def enunciado(item: Item) -> str:
    """A pergunta como o modelo a lê — e, nos dois passos da busca, como a busca a lê."""
    return f"{item.pergunta}\n\n" + "\n".join(f"{x}. {o}" for x, o in zip(LETRAS, item.opcoes,
                                                                        strict=True))


def usuario(item: Item, fontes: list[Fonte] | None = None) -> str:
    if not fontes:
        return f"Question: {enunciado(item)}"
    return f"Sources:\n\n{bloco_de_fontes(fontes)}\n\nQuestion: {enunciado(item)}"


def responder(modelo, item: Item, fontes: list[Fonte] | None = None,
              semente: int = SEMENTE) -> str:
    """Um braço, uma chamada: C sem fontes, A com elas. Temperatura e semente do produto."""
    sistema = SISTEMA_COM_FONTES if fontes else SISTEMA_SEM_FONTES
    return modelo.gerar(sistema, usuario(item, fontes), max_tokens=MAX_TOKENS_RESPOSTA,
                        semente=semente)


# A letra fica fora do (?i): "the answer is a planet" não é a opção A.
_RESPOSTA = re.compile(r"(?i:\b(?:final\s+)?answer\b(?:\s+is)?)\s*[:：=\-]?\s*[\"'(\[*]*\s*"
                       r"([ABCD])\b")
_JSON = re.compile(r'(?i:"answer")\s*:\s*"\s*\(?([ABCD])\b')
_INICIO = re.compile(r"^[\s*(\[\"']*([ABCD])(?:[.):\]*\"']|\s*$|\s+-)")
_OPCAO = re.compile(r"(?i:\b(?:option|choice)\b(?:\s+is)?)\s*[:\-]?\s*[(\[*]*([ABCD])\b")


def extrair_letra(texto: str) -> str | None:
    """A letra da resposta, ou None. Na ordem: `Answer: X`, o JSON do AstroMLab, a letra
    sozinha no começo, `option X`. Com duas letras, vale a primeira que o modelo declarou.
    None vai em branco no envio e conta como erro."""
    texto = texto or ""
    for padrao in (_RESPOSTA, _JSON, _INICIO, _OPCAO):
        m = padrao.search(texto)
        if m:
            return m.group(1)
    return None


def taxa_de_leitura(letras: list[str | None]) -> float:
    return sum(x is not None for x in letras) / len(letras) if letras else 0.0


# ── o envio ──────────────────────────────────────────────────────────────────

CABECALHO_ENVIO = ("row_index", "question_id", "answer")


def exportar(itens: list[Item], letras: list[str | None], arquivo: Path) -> str:
    """O arquivo de um braço: linha do conjunto público, id por conteúdo e a letra (em
    branco se não foi lida). Recusa se a guarda de leitura não passa. Devolve o sha256 do
    arquivo — vai para o agregado antes de a correção voltar."""
    if len(itens) != len(letras):
        raise ValueError(f"{len(itens)} itens e {len(letras)} respostas")
    lida = taxa_de_leitura(letras)
    if lida < MIN_LEITURA:
        raise ValueError(f"letra lida em {lida:.1%} das respostas (mínimo {MIN_LEITURA:.0%}): "
                         "conserte a leitura antes de enviar (§9.10)")
    arquivo = Path(arquivo)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    with arquivo.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CABECALHO_ENVIO)
        for it, letra in zip(itens, letras, strict=True):
            w.writerow((it.indice, it.id, letra or ""))
    return sha256_arquivo(arquivo)


def rascunho_de_email(n: int, sha_conjunto: str, anexos: list[str]) -> str:
    """O pedido à equipe do AstroMLab. Em inglês; o dono assina e manda."""
    lista = "\n".join(f"- {a}" for a in anexos)
    return f"""To: {ENDERECO_DE_CORRECAO}
Subject: AstroBench public MCQ set — request to score two prediction files

Dear AstroMLab team,

I am running an independent, small-scale research project on retrieval of arXiv physics
papers under tight compute: an 8B open model (Qwen3-8B, 4-bit, run locally) that answers
questions from paper abstracts found by our own retriever. To test it on questions we did
not write, I used the public AstroBench MCQ set ({CONJUNTO}, file sha256
{sha_conjunto[:16]}…).

Attached are two prediction files over the same random subset of {n} questions of the
public set (fixed seed), one per configuration:

{lista}

- "C": the model alone, no retrieved context.
- "A": the same model, with six paper abstracts retrieved by our system before the question.

Each row has the row index in the public file, a hash of the question text and options
(to guard against row-order differences), and the predicted letter (blank where no letter
could be read; please count those as wrong).

Could you score both files? If your policy allows it, per-question correctness (1/0 for
each row of each file) would let me compare the two configurations question by question,
which is the analysis I registered before running. If it does not, the accuracy of each file
is enough, and I will report the comparison as unpaired.

I will not publish per-question results or anything that could reconstruct the answer key,
and any write-up will cite the AstroMLab papers and say that the scoring was done by your
team.

Thank you for making the benchmark available.

Best regards,
[nome]
"""


# ── a correção ───────────────────────────────────────────────────────────────

_VERDADE = {"1": True, "0": False, "true": True, "false": False, "yes": True, "no": False,
            "correct": True, "incorrect": False, "wrong": False}


def _coluna(campos: list[str], nomes: tuple[str, ...], arquivo: Path) -> str:
    achada = next((c for c in campos if c.strip().lower() in nomes), None)
    if achada is None:
        raise ValueError(f"{Path(arquivo).name}: nenhuma coluna entre {nomes} (tem {campos})")
    return achada


def ler_correcao(arquivo: Path) -> dict[int, bool]:
    """O acerto por item de UM braço, como a equipe devolver: a linha do conjunto e 1/0."""
    with Path(arquivo).open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        lin = _coluna(r.fieldnames or [], ("row_index", "indice", "index", "row"), arquivo)
        ok = _coluna(r.fieldnames or [], ("correct", "correta", "is_correct", "score"), arquivo)
        saida = {}
        for d in r:
            v = d[ok].strip().lower()
            if v not in _VERDADE:
                raise ValueError(f"{Path(arquivo).name}: acerto {d[ok]!r} não é 1/0")
            saida[int(d[lin])] = _VERDADE[v]
    return saida


def ler_gabarito(arquivo: Path) -> dict[int, str]:
    """O gabarito de uma linha do conjunto, se a equipe devolver as letras."""
    with Path(arquivo).open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        lin = _coluna(r.fieldnames or [], ("row_index", "indice", "index", "row"), arquivo)
        res = _coluna(r.fieldnames or [], ("answer", "gabarito", "correct_answer", "label"),
                      arquivo)
        saida = {}
        for d in r:
            letra = d[res].strip().upper().strip("(). ")
            if letra not in LETRAS:
                raise ValueError(f"{Path(arquivo).name}: gabarito {d[res]!r} fora de A–D")
            saida[int(d[lin])] = letra
    return saida


def corrigir(indices: list[int], letras: list[str | None], gabarito: dict[int, str]
             ) -> np.ndarray:
    faltam = [i for i in indices if i not in gabarito]
    if faltam:
        raise ValueError(f"o gabarito não cobre {len(faltam)} dos itens (o primeiro: {faltam[0]})")
    return np.array([x is not None and x == gabarito[i] for i, x in
                     zip(indices, letras, strict=True)], dtype=float)


def acertos_de(indices: list[int], correcao: dict[int, bool]) -> np.ndarray:
    faltam = [i for i in indices if i not in correcao]
    if faltam:
        raise ValueError(f"a correção não cobre {len(faltam)} dos itens (o primeiro: "
                         f"{faltam[0]})")
    return np.array([correcao[i] for i in indices], dtype=float)


# ── a leitura ────────────────────────────────────────────────────────────────


def leitura(ic: tuple[float, float], acerto_c: float) -> dict:
    """A regra: o lado do IC de A − C. A guarda de sanidade vem antes de tudo."""
    if acerto_c < MIN_ACERTO_C:
        return {"leitura": "invalida", "motivo": f"acerto de C {acerto_c:.3f} < "
                f"{MIN_ACERTO_C}: prompt ou leitura quebrados, não é resultado",
                "material": False}
    lo, hi = ic
    texto = "ajuda" if lo > 0 else "atrapalha" if hi < 0 else "nao_se_distingue"
    return {"leitura": texto, "material": bool(lo > LIMIAR_MATERIAL),
            "limiar_material": LIMIAR_MATERIAL}


def apurar_por_item(certo_c: np.ndarray, certo_a: np.ndarray) -> dict:
    """A regra pareada: acerto por braço e A − C com IC por bootstrap pareado por item."""
    certo_c, certo_a = np.asarray(certo_c, dtype=float), np.asarray(certo_a, dtype=float)
    if certo_c.size != certo_a.size or certo_c.size == 0:
        raise ValueError("os dois braços têm de cobrir os mesmos itens")
    ac, ic_c = proporcao(certo_c)
    aa, ic_a = proporcao(certo_a)
    dif, ic = diferenca_pareada(certo_a, certo_c)
    ganha = int(((certo_a == 1) & (certo_c == 0)).sum())
    perde = int(((certo_a == 0) & (certo_c == 1)).sum())
    return {"pareado": True, "n": int(certo_c.size),
            "acerto_C": ac, "ic95_C": ic_c, "acerto_A": aa, "ic95_A": ic_a,
            "A_menos_C": dif, "ic95": ic, "ganha": ganha, "perde": perde,
            "mcnemar_exato_p (relatado)": round(binomial_exata_bicaudal(ganha, perde)["p"], 6),
            **leitura(ic, ac)}


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """IC 95% de p1 − p2 para proporções INDEPENDENTES, pelos intervalos de Wilson
    (Newcombe 1998, método 10). Com os braços errando nos mesmos itens — o esperado —, é
    mais largo que o pareado: a leitura degradada erra para o lado de não distinguir."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (round(d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), 4),
            round(d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2), 4))


def apurar_agregado(k_c: int, k_a: int, n: int) -> dict:
    """A leitura degradada: só o acerto de cada arquivo voltou."""
    if not (0 <= k_c <= n and 0 <= k_a <= n and n > 0):
        raise ValueError(f"contagens fora de 0..{n}: C={k_c}, A={k_a}")
    ic = newcombe(k_a, n, k_c, n)
    return {"pareado": False, "n": n, "acerto_C": round(k_c / n, 4),
            "ic95_C": tuple(round(x, 4) for x in wilson(k_c, n)),
            "acerto_A": round(k_a / n, 4), "ic95_A": tuple(round(x, 4) for x in wilson(k_a, n)),
            "A_menos_C": round((k_a - k_c) / n, 4), "ic95": ic,
            "aviso": "sem pareamento: intervalo de Newcombe para proporções independentes",
            **leitura(ic, k_c / n)}


def secundarias(letras_c: list[str | None], letras_a: list[str | None],
                fontes: list[list[str]], meta: dict[str, tuple[str | None, int | None]]
                ) -> dict:
    """Relatadas, não decidem — e todas medíveis sem o gabarito: a leitura, quanto a busca
    muda a resposta, a distribuição das letras (viés de posição) e de onde vêm as fontes."""
    ambos = [(c, a) for c, a in zip(letras_c, letras_a, strict=True)
             if c is not None and a is not None]
    cats = [meta.get(x, (None, None))[0] or "" for fs in fontes for x in fs]
    anos = [meta.get(x, (None, None))[1] for fs in fontes for x in fs]
    anos = [a for a in anos if a]
    return {
        "n": len(letras_c),
        "leitura_C": round(taxa_de_leitura(letras_c), 4),
        "leitura_A": round(taxa_de_leitura(letras_a), 4),
        "resposta_mudou (A ≠ C)": round(sum(c != a for c, a in ambos) / len(ambos), 4)
        if ambos else None,
        "letras_C": {x: letras_c.count(x) for x in (*LETRAS, None)},
        "letras_A": {x: letras_a.count(x) for x in (*LETRAS, None)},
        "fontes_astro_ph": round(sum(c.startswith("astro-ph") for c in cats) / len(cats), 4)
        if cats else None,
        "itens_com_fonte_astro_ph": round(sum(
            any((meta.get(x, (None, None))[0] or "").startswith("astro-ph") for x in fs)
            for fs in fontes) / len(fontes), 4) if fontes else None,
        "ano_mediano_das_fontes": float(np.median(anos)) if anos else None,
    }


# Os números publicados, de OUTRO protocolo (o conjunto interno de 4.425 perguntas e o
# prompt do AstroMLab): referência de ordem de grandeza, não comparação ponto a ponto.
COMPARADORES_PUBLICADOS = {
    "Llama-3.1-8B-Instruct": 0.737,
    "AstroSage-Llama-3.1-8B": 0.809,
}
