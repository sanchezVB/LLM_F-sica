"""A regra do portão G2 (DOC-00, Tier 2; DOC-12 §3.7) — PROPOSTA de 2026-10-19.

Aqui ficam as constantes da regra, as duas travas do teste, o livro de acessos, os grupos
por 13-grama, a partição do UGPhysics e a decisão dos três critérios a partir das notas por
item. O motor, o extrator e o corretor NÃO ficam aqui: eles vêm com o harness, depois do
aceite, e são desenvolvidos só no dev.

As duas travas, nesta ordem:

1. `REGRA_G2_ACEITA`: o dono aceita a regra como escrita no DOC-12 §3.7.
2. `PROTOCOLO_G2_SHA256`: o SHA-256 de `configs/eval/preregistration/g2.yaml` no
   CONGELAMENTO. O congelamento vem depois do dev (prompt, exemplos, extrator, corretor,
   envelope do IFEval e partição) e antes da primeira geração no teste. Enquanto for None,
   nada toca o teste.

As notas entram já corrigidas, uma por item e em [0, 1]: a média das k amostras (avg@k),
com FAIL, sem resposta, truncado, INCONCLUSIVE e ERROR valendo 0. Os deltas saem em pontos
(×100).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[3]
PREREGISTRO = RAIZ / "configs/eval/preregistration/g2.yaml"
LIVRO_DE_ACESSOS = RAIZ / "data/processed/avaliacao/g2_acessos.jsonl"

# ── as travas ────────────────────────────────────────────────────────────────
# PROPOSTA de 2026-10-19; ACEITA pelo dono em 2026-10-19, como escrita, sem o GPQA (o G2.2
# fica em 3 de 4). Mesmo aceita, nada toca o teste antes do congelamento abaixo.
REGRA_G2_ACEITA = True
# Preenchido no congelamento, com o hash do g2.yaml completo (partição, exemplos, envelope,
# código do extrator e do corretor, motor). Antes disso, None.
PROTOCOLO_G2_SHA256: str | None = "5d848f12d014b01586bd127c92b11d1a232d5c8a2fe5fae4c04cf2584874aed8"

SEMENTE_G2 = 20261019
N_BOOTSTRAP = 10_000

# ── G2.1: o ΦGen-CPT contra a PRÓPRIA base, no agregado de física ─────────────
BENCH_G2_1 = ("ugphysics", "olympiadbench", "phybench", "scibench")
LIMIAR_G2_1 = 10.0        # o limite de baixo do IC 95% de Δ_macro tem de ser ≥ isto

# ── G2.2: só na 8B, contra bases abertas ≤32B no mesmo protocolo ─────────────
BENCH_G2_2 = ("gpqa_fisica", "olympiadbench", "phybench", "scibench", "ugphysics")
VITORIAS_G2_2 = 3         # de 5; o GPQA não medido conta como não vencido
NIVEL_G2_2 = 0.99         # Bonferroni sobre os 5 benchmarks do charter
MARGEM_QUANTIZADO_G2_2 = 1.0  # contra comparador medido abaixo de Q8_0
COMPARADORES_G2_2 = ("Qwen3-14B-Base", "Qwen3-30B-A3B-Base", "Qwen2.5-32B")

# ── G2.3: sem regressão geral > 2 pontos (desclassificatório) ────────────────
MARGEM_G2_3 = 2.0
# MMLU: não-inferioridade demonstrada (limite de baixo ≥ −2). Código e IFEval: desclassifica
# se Δ̂ < −2 E o IC inteiro abaixo de 0.
NAO_INFERIORIDADE_G2_3 = ("mmlu_sem_fisica",)
DESCLASSIFICA_G2_3 = ("evalplus", "ifeval_envelope")

# ── o livro de acessos ao teste (K10) ────────────────────────────────────────
# `retomada`: continuar, pelos itens que faltam, uma medida interrompida (queda do Colab,
# do servidor, da sessão do Windows). A regra prevê retomada por item (§3.7.5); ela fica
# no livro, não conta como acesso novo e só vale com o MESMO hash da medida em curso.
MOTIVOS_DE_ACESSO = ("medida", "falha_operacional", "retomada")


class RegraNaoAceita(RuntimeError):
    pass


class ProtocoloNaoCongelado(RuntimeError):
    pass


class ProtocoloMudou(RuntimeError):
    pass


class AcessoRecusado(RuntimeError):
    pass


def sha256_arquivo(caminho: Path) -> str:
    """SHA-256 do arquivo com o fim de linha normalizado para LF: com `core.autocrlf`, o
    mesmo commit sai em CRLF numa cópia e em LF noutra, e o hash do congelamento não pode
    depender disso."""
    return hashlib.sha256(Path(caminho).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# O que decide um veredito do corretor: os módulos dele, os de que ele depende e as versões
# do parser. Antes o hash cobria só g2_corretor e g2_itens; uma mudança em subscritos.py ou
# symbolic.py (o PR do parse do verify) mudava a correção sem mudar o hash (sessão do
# verify, 2026-10-21).
CODIGO_DO_CORRETOR = ("src/phifm/eval/g2_corretor.py", "src/phifm/eval/g2_itens.py",
                      "src/phifm/core/latex/subscritos.py", "src/phifm/verify/symbolic.py")
PACOTES_DO_CORRETOR = ("sympy", "antlr4-python3-runtime", "mpmath")


def codigo_do_corretor(raiz: Path, extras: tuple[str, ...] = ()) -> dict[str, str | None]:
    """{caminho: SHA-256} do código do corretor (e de `extras`), mais {pacote: versão}."""
    from importlib.metadata import PackageNotFoundError, version

    hashes: dict[str, str | None] = {p: sha256_arquivo(raiz / p)
                                     for p in CODIGO_DO_CORRETOR + tuple(extras)}
    for pacote in PACOTES_DO_CORRETOR:
        try:
            hashes[pacote] = version(pacote)
        except PackageNotFoundError:
            hashes[pacote] = None
    return hashes


# Congelado junto com o g2.yaml: tudo o que decide a geração e a correção da medida. O
# g2.yaml guarda o hash de `g2_congelamento.json`, e a medida no teste recusa se algum
# arquivo daqui mudar depois. O g2.py fica de fora: a trava dele é o próprio hash do yaml.
CONGELAMENTO = RAIZ / "configs/eval/preregistration/g2_congelamento.json"
CODIGO_DA_MEDIDA = ("src/phifm/eval/g2_prompt.py", "src/phifm/eval/g2_geracao.py",
                    "src/phifm/eval/g2_correcao.py", "scripts/medir_g2.py",
                    # G2.3 (emenda 2): o MMLU, o IFEval com o código oficial, e o EvalPlus
                    "src/phifm/eval/g2_mmlu.py", "scripts/medir_g2_mmlu.py",
                    "src/phifm/eval/g2_ifeval.py", "scripts/medir_g2_ifeval.py",
                    *(f"ferramentas/ifeval-oficial/instruction_following_eval/{m}.py"
                      for m in ("evaluation_lib", "instructions", "instructions_registry",
                                "instructions_util")),
                    "src/phifm/eval/g2_evalplus.py", "scripts/medir_g2_evalplus.py",
                    "kaggle/g2_evalplus.py")
DADOS_DA_MEDIDA = ("data/processed/avaliacao/g2_particao.json",
                   "data/processed/avaliacao/g2_exemplos.json",
                   "data/processed/avaliacao/g2_autoteste.json",
                   "configs/eval/preregistration/g2_ifeval_envelope.txt",
                   # G2.3 (emenda 2)
                   "configs/eval/g2_ifeval_dev20.jsonl",
                   "data/raw/benchmarks/mmlu/all/dev-00000-of-00001.parquet",
                   "data/raw/benchmarks/mmlu/all/test-00000-of-00001.parquet",
                   "data/raw/benchmarks/ifeval/ifeval_input_data.jsonl",
                   "data/raw/benchmarks/evalplus-oficial/HumanEvalPlus-v0.1.10.jsonl",
                   "data/raw/benchmarks/evalplus-oficial/MbppPlus-v0.2.0.jsonl")
# As versões que decidem a correção e a sanitização do G2.3. Fora do `codigo_do_corretor`,
# para não mudar o hash do autoteste do corretor de física.
PACOTES_DA_MEDIDA = ("langdetect", "nltk", "absl-py", "immutabledict", "evalplus",
                     "tree-sitter", "tree-sitter-python")


def estado_congelavel(raiz: Path = RAIZ) -> dict[str, dict]:
    """O código (com as versões do parser e dos pacotes do G2.3) e os dados da medida, por
    SHA-256."""
    from importlib.metadata import PackageNotFoundError, version

    codigo = codigo_do_corretor(raiz, extras=CODIGO_DA_MEDIDA)
    for pacote in PACOTES_DA_MEDIDA:
        try:
            codigo[pacote] = version(pacote)
        except PackageNotFoundError:
            codigo[pacote] = None
    return {"codigo": codigo, "dados": {p: sha256_arquivo(raiz / p) for p in DADOS_DA_MEDIDA}}


def exigir_estado_congelado(raiz: Path = RAIZ, congelamento: Path = CONGELAMENTO,
                            sha_do_congelamento: str | None = None) -> None:
    """Levanta ProtocoloMudou se `g2_congelamento.json` não é o que o g2.yaml registra, ou
    se algum arquivo ou versão de pacote difere do que ele grava."""
    if sha_do_congelamento is not None and sha256_arquivo(congelamento) != sha_do_congelamento:
        raise ProtocoloMudou("g2_congelamento.json não é o registrado no g2.yaml")
    gravado = json.loads(congelamento.read_text(encoding="utf-8"))
    atual = estado_congelavel(raiz)
    difere = [k for grupo in ("codigo", "dados") for k, v in gravado[grupo].items()
              if atual[grupo].get(k) != v]
    if difere:
        raise ProtocoloMudou(f"mudou depois do congelamento: {', '.join(difere)}; mudança só "
                             "por emenda registrada")


def _sha_do_congelamento_no_yaml(preregistro: Path) -> str | None:
    import yaml

    conf = yaml.safe_load(preregistro.read_text(encoding="utf-8")) or {}
    return (conf.get("congelamento") or {}).get("sha256")


def exigir_aceite_e_congelamento(preregistro: Path = PREREGISTRO, aceita: bool | None = None,
                                 sha_congelado: str | None = None) -> str:
    """Levanta exceção se a regra não foi aceita, se o protocolo não foi congelado, se o
    g2.yaml mudou depois do congelamento ou, com o g2.yaml de verdade, se o código ou os
    dados da medida mudaram (`exigir_estado_congelado`). Devolve o hash conferido.
    `aceita` e `sha_congelado` só existem para os testes; o normal é ler as constantes."""
    aceita = REGRA_G2_ACEITA if aceita is None else aceita
    sha_congelado = PROTOCOLO_G2_SHA256 if sha_congelado is None else sha_congelado
    if not aceita:
        raise RegraNaoAceita("a regra do G2 (DOC-12 §3.7) é PROPOSTA: o dono ainda não "
                             "aceitou, e nenhum item de teste pode ser tocado")
    if not sha_congelado:
        raise ProtocoloNaoCongelado("a regra foi aceita, mas o protocolo não foi congelado: "
                                    "termine o dev e grave PROTOCOLO_G2_SHA256")
    atual = sha256_arquivo(preregistro)
    if atual != sha_congelado:
        raise ProtocoloMudou(f"g2.yaml mudou depois do congelamento: {atual[:12]}… ≠ "
                             f"{sha_congelado[:12]}…; mudança só por emenda registrada")
    if Path(preregistro) == PREREGISTRO:
        sha = _sha_do_congelamento_no_yaml(preregistro)
        if not sha:
            raise ProtocoloNaoCongelado("o g2.yaml não registra o g2_congelamento.json")
        exigir_estado_congelado(sha_do_congelamento=sha)
    return atual


def _chave(r: dict) -> tuple:
    # A base é re-medida quando o hash da configuração muda (a base só vale no veredito com
    # o MESMO hash do ΦGen); o ΦGen e cada comparador têm um acesso só, qualquer que seja o
    # hash. Por medida: a nota separa o G2.1 ("") do MMLU, do IFEval e do EvalPlus (emenda 2).
    return (r["tamanho"], r["braco"], r.get("nota", ""),
            r["hash_config"] if r["braco"] == "base" else None)


def motivo_de_recusa(registros: Sequence[dict], tamanho: str, braco: str, motivo: str,
                     hash_config: str, nota: str = "") -> str | None:
    """None se o acesso é permitido; senão, o porquê. Por chave: 1 `medida` e, depois dela,
    no máximo 1 `falha_operacional` (a parcial é descartada e registrada)."""
    if motivo not in MOTIVOS_DE_ACESSO:
        return f"motivo desconhecido: {motivo}"
    novo = {"tamanho": tamanho, "braco": braco, "hash_config": hash_config, "nota": nota}
    mesmos = [r for r in registros if _chave(r) == _chave(novo)]
    medidas = sum(r["motivo"] == "medida" for r in mesmos)
    repeticoes = sum(r["motivo"] == "falha_operacional" for r in mesmos)
    if motivo == "retomada":
        abertas = [r for r in mesmos if r["motivo"] in ("medida", "falha_operacional")]
        if not abertas:
            return "retomada sem uma medida aberta"
        if abertas[-1]["hash_config"] != hash_config:
            return ("retomada com outro hash de configuração: isso não é continuar a mesma "
                    "medida")
        return None
    if motivo == "medida" and medidas:
        return (f"{braco} ({tamanho}) já teve o seu acesso ao teste; outro só por emenda "
                "escrita (a reserva lacrada do UG existe para isso)")
    if motivo == "falha_operacional" and not medidas:
        return "repetição por falha operacional sem uma medida antes"
    if motivo == "falha_operacional" and repeticoes:
        return f"{braco} ({tamanho}) já usou a sua única repetição por falha operacional"
    return None


def medida_aberta(tamanho: str, braco: str, hash_config: str, nota: str = "",
                  livro: Path | None = None) -> bool:
    """Se o livro já tem uma `medida` (ou `falha_operacional`) desta chave com este hash: a
    próxima entrada é `retomada`. Decidir pelo livro, e não pela saída gravada, faz uma
    medida morta entre o registro e a 1ª amostra (o prazo do Kaggle, um desligamento)
    retomar, em vez de ser recusada como 2º acesso (emenda 3)."""
    livro = Path(livro or LIVRO_DE_ACESSOS)
    if not livro.exists():
        return False
    novo = {"tamanho": tamanho, "braco": braco, "hash_config": hash_config, "nota": nota}
    return any(_chave(r) == _chave(novo) and r["motivo"] in ("medida", "falha_operacional")
               and r["hash_config"] == hash_config
               for r in (json.loads(x) for x in livro.read_text(encoding="utf-8").splitlines()
                         if x.strip()))


def registrar_acesso(tamanho: str, braco: str, motivo: str, hash_config: str,
                     nota: str = "", livro: Path = LIVRO_DE_ACESSOS) -> dict:
    """Confere a regra, grava o acesso no livro (versionado) e só então devolve. Quem toca
    o teste chama isto ANTES da primeira geração."""
    livro = Path(livro)
    registros = ([json.loads(x) for x in livro.read_text(encoding="utf-8").splitlines()
                  if x.strip()] if livro.exists() else [])
    if (porque := motivo_de_recusa(registros, tamanho, braco, motivo, hash_config, nota)):
        raise AcessoRecusado(porque)
    r = {"quando": datetime.now(UTC).isoformat(timespec="seconds"), "tamanho": tamanho,
         "braco": braco, "motivo": motivo, "hash_config": hash_config, "nota": nota}
    livro.parent.mkdir(parents=True, exist_ok=True)
    with livro.open("a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return r


# ── grupos por 13-grama e a partição do UGPhysics ────────────────────────────


def ngramas(texto: str, n: int = 13) -> set[str]:
    """Os n-gramas de palavras (\\w+, minúsculas) do texto."""
    p = re.findall(r"\w+", texto.lower())
    return {" ".join(p[i:i + n]) for i in range(len(p) - n + 1)}


def grupos_por_ngramas(textos: Sequence[str], n: int = 13, min_itens: int = 2,
                       max_itens: int = 50) -> list[int]:
    """O grupo de cada item: componentes conexos de itens que compartilham ≥1 n-grama,
    contando só os n-gramas presentes em `min_itens` a `max_itens` itens (acima disso é
    frase-feita do benchmark). O rótulo do grupo é o menor índice dele."""
    gramas = [ngramas(t, n) for t in textos]
    df = Counter(g for s in gramas for g in s)
    pai = list(range(len(textos)))

    def raiz(x: int) -> int:
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    dono: dict[str, int] = {}
    for i, s in enumerate(gramas):
        for g in s:
            if not min_itens <= df[g] <= max_itens:
                continue
            if g in dono:
                a, b = raiz(i), raiz(dono[g])
                if a != b:
                    pai[max(a, b)] = min(a, b)
            else:
                dono[g] = i
    return [raiz(i) for i in range(len(textos))]


def particionar_por_grupos(materia: Sequence[str], elegivel: Sequence[bool],
                           grupo: Sequence[int], frac_dev: float = 0.2,
                           n_teste: int = 2000, semente: int = SEMENTE_G2
                           ) -> dict[str, list[int]]:
    """A partição do UGPhysics da §3.7: índices de DEV, TESTE e RESERVA (só elegíveis).

    Nenhum grupo é dividido. O grupo que cruza matérias vai para a matéria do seu menor
    índice. `frac_dev` dos grupos de cada matéria vai para o DEV. Dos restantes, ≈`n_teste`
    elegíveis vão para o TESTE, proporcionalmente à matéria, sorteados por grupo (alocação
    gulosa: entra o grupo que não passa do alvo da matéria + 0,5). O resto é a RESERVA
    lacrada."""
    membros: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(grupo):
        membros[g].append(i)
    por_materia: dict[str, list[list[int]]] = defaultdict(list)
    for mem in membros.values():
        por_materia[materia[min(mem)]].append(sorted(mem))
    rng = random.Random(semente)
    dev: list[list[int]] = []
    resto: dict[str, list[list[int]]] = {}
    for m in sorted(por_materia):
        gs = sorted(por_materia[m])
        rng.shuffle(gs)
        k = round(frac_dev * len(gs))
        dev += gs[:k]
        resto[m] = gs[k:]
    n_eleg = {m: sum(elegivel[i] for g in gs for i in g) for m, gs in resto.items()}
    total = sum(n_eleg.values())
    alvo = {m: n_teste * c / total for m, c in n_eleg.items()}
    teste: list[int] = []
    for m in sorted(resto):
        gs = resto[m]
        rng.shuffle(gs)
        ja = 0
        for g in gs:
            e = sum(elegivel[i] for i in g)
            if e and (ja + e <= alvo[m] + 0.5 or ja == 0):
                ja += e
                teste += [i for i in g if elegivel[i]]
    no_dev = {i for g in dev for i in g}
    no_teste = set(teste)
    so = [i for i in range(len(materia)) if elegivel[i]]
    return {"dev": [i for i in so if i in no_dev],
            "teste": sorted(no_teste),
            "reserva": [i for i in so if i not in no_dev and i not in no_teste]}


def agrupar_por_contexto(contextos: Sequence[str | None], questoes: Sequence[str],
                         bloco: int = 30, prefixo: int = 300) -> list[int]:
    """Os agrupamentos do OlympiadBench: itens com o mesmo contexto (pelos primeiros
    `prefixo` caracteres) são um grupo, porque os contextos progressivos encadeiam as
    questões. Um contexto com mais de `bloco` itens não é progressão, é um bloco de
    constantes que vale para todos: dentro dele, os itens se agrupam pelos 13-gramas da
    questão. O item sem contexto é o seu próprio grupo. O rótulo é o menor índice."""
    por_contexto: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(contextos):
        if c and str(c).strip():
            por_contexto[str(c)[:prefixo]].append(i)
    grupo = list(range(len(contextos)))
    for membros in por_contexto.values():
        if len(membros) > bloco:
            dentro = grupos_por_ngramas([questoes[i] for i in membros])
            for i, g in zip(membros, dentro, strict=True):
                grupo[i] = membros[g]
        else:
            for i in membros:
                grupo[i] = min(membros)
    return grupo


# ── a estatística ────────────────────────────────────────────────────────────


def _ic(reps: np.ndarray, nivel: float = 0.95) -> tuple[float, float]:
    a = (1 - nivel) / 2 * 100
    lo, hi = np.percentile(reps, [a, 100 - a])
    return round(float(lo), 4), round(float(hi), 4)


def reps_delta_agrupado(nota_a: np.ndarray, nota_b: np.ndarray, grupo: Sequence[int] | None,
                        rng: np.random.Generator, n_bootstrap: int = N_BOOTSTRAP
                        ) -> tuple[float, np.ndarray]:
    """100·mean(a − b) e as réplicas do bootstrap PAREADO, reamostrando GRUPOS inteiros com
    reposição (cada item é o seu grupo quando `grupo` é None)."""
    d = (np.asarray(nota_a, dtype=float) - np.asarray(nota_b, dtype=float)) * 100
    if d.shape != (d.size,) or not d.size:
        raise ValueError("as notas têm de ser vetores não vazios do mesmo tamanho")
    rotulos = np.arange(d.size) if grupo is None else np.asarray(grupo)
    _, inv = np.unique(rotulos, return_inverse=True)
    soma = np.bincount(inv, weights=d)
    conta = np.bincount(inv).astype(float)
    reps = np.empty(n_bootstrap)
    for ini in range(0, n_bootstrap, 1000):
        fim = min(ini + 1000, n_bootstrap)
        idx = rng.integers(0, soma.size, (fim - ini, soma.size))
        reps[ini:fim] = soma[idx].sum(axis=1) / conta[idx].sum(axis=1)
    return float(d.mean()), reps


def delta_macro(notas_a: dict[str, np.ndarray], notas_b: dict[str, np.ndarray],
                grupos: dict[str, Sequence[int] | None], bench: Sequence[str] = BENCH_G2_1,
                nivel: float = 0.95, n_bootstrap: int = N_BOOTSTRAP,
                semente: int = SEMENTE_G2) -> dict:
    """Δ_b por benchmark e Δ_macro = média simples dos Δ_b, com IC por bootstrap pareado,
    estratificado por benchmark e reamostrando agrupamentos dentro de cada um."""
    rng = np.random.default_rng(semente)
    por, reps = {}, []
    for b in bench:
        ponto, r = reps_delta_agrupado(notas_a[b], notas_b[b], grupos.get(b), rng, n_bootstrap)
        por[b] = {"delta": round(ponto, 4), "ic": _ic(r, nivel), "n": int(np.size(notas_a[b]))}
        reps.append(r)
    macro = np.mean(reps, axis=0)
    return {"por_benchmark": por, "delta_macro": round(float(np.mean([por[b]["delta"]
                                                                     for b in bench])), 4),
            "ic_macro": _ic(macro, nivel), "nivel": nivel}


# ── a decisão ────────────────────────────────────────────────────────────────


def decidir_g2_1(notas_phigen: dict[str, np.ndarray], notas_base: dict[str, np.ndarray],
                 grupos: dict[str, Sequence[int] | None],
                 memorizacao_nova: dict[str, np.ndarray],
                 n_bootstrap: int = N_BOOTSTRAP, semente: int = SEMENTE_G2) -> dict:
    """PASSA só com: (1) limite de baixo do IC 95% de Δ_macro ≥ +10; (2) nenhum Δ_b com o IC
    inteiro abaixo de 0; (3) sem os itens de memorização NOVA (sinal S1/S2 no ΦGen e não na
    base), o limite de baixo continua ≥ +10."""
    todos = delta_macro(notas_phigen, notas_base, grupos, n_bootstrap=n_bootstrap,
                        semente=semente)
    limpos = {}
    for nome, notas in (("phigen", notas_phigen), ("base", notas_base)):
        limpos[nome] = {b: np.asarray(notas[b])[~np.asarray(memorizacao_nova[b], dtype=bool)]
                        for b in BENCH_G2_1}
    grupos_limpos = {b: (None if grupos.get(b) is None else
                         np.asarray(grupos[b])[~np.asarray(memorizacao_nova[b], dtype=bool)])
                     for b in BENCH_G2_1}
    sem_memorizacao = delta_macro(limpos["phigen"], limpos["base"], grupos_limpos,
                                  n_bootstrap=n_bootstrap, semente=semente)
    c1 = todos["ic_macro"][0] >= LIMIAR_G2_1
    c2 = not any(v["ic"][1] < 0 for v in todos["por_benchmark"].values())
    c3 = sem_memorizacao["ic_macro"][0] >= LIMIAR_G2_1
    passa = c1 and c2 and c3
    if passa:
        leitura = "passa"
    elif todos["ic_macro"][1] < LIMIAR_G2_1:
        leitura = "demonstrado abaixo"
    else:
        leitura = "não demonstrado"
    return {"passa": passa, "leitura": leitura,
            "condicoes": {"ic_macro_acima_do_limiar": c1, "nenhum_benchmark_piora": c2,
                          "sem_memorizacao_nova": c3},
            "todos": todos, "sem_memorizacao_nova": sem_memorizacao}


def decidir_g2_3(pares: dict[str, tuple[np.ndarray, np.ndarray]],
                 n_bootstrap: int = N_BOOTSTRAP, semente: int = SEMENTE_G2) -> dict:
    """`pares[m] = (notas_phigen, notas_base)`, por item, para as três medidas. O MMLU
    passa só se o limite de baixo ≥ −2 (não-inferioridade demonstrada); código e IFEval
    desclassificam se Δ̂ < −2 E o IC 95% inteiro abaixo de 0. Uma que falhe desclassifica
    sozinha."""
    rng = np.random.default_rng(semente + 3)
    medidas, desclassificado = {}, False
    for m in (*NAO_INFERIORIDADE_G2_3, *DESCLASSIFICA_G2_3):
        a, b = pares[m]
        ponto, reps = reps_delta_agrupado(a, b, None, rng, n_bootstrap)
        lo, hi = _ic(reps)
        if m in NAO_INFERIORIDADE_G2_3:
            falha = lo < -MARGEM_G2_3
        else:
            falha = ponto < -MARGEM_G2_3 and hi < 0
        desclassificado |= falha
        medidas[m] = {"delta": round(ponto, 4), "ic": (lo, hi), "n": int(np.size(a)),
                      "falha": falha}
    return {"passa": not desclassificado, "medidas": medidas}


def decidir_g2_2(notas_phigen: dict[str, np.ndarray],
                 comparadores: dict[str, dict[str, np.ndarray]],
                 grupos: dict[str, Sequence[int] | None],
                 abaixo_de_q8: frozenset[str] = frozenset(),
                 n_bootstrap: int = N_BOOTSTRAP, semente: int = SEMENTE_G2) -> dict:
    """Só na 8B. O ΦGen vence o benchmark b se, contra CADA comparador, o IC 99% da
    diferença pareada fica inteiro acima de 0 (de +1 contra comparador abaixo de Q8_0).
    PASSA com ≥3 vitórias entre os 5 do charter; benchmark não medido não é vencido."""
    rng = np.random.default_rng(semente + 5)
    por = {}
    for b in BENCH_G2_2:
        if b not in notas_phigen or any(b not in c for c in comparadores.values()):
            por[b] = {"vence": False, "medido": False}
            continue
        contra = {}
        for nome, notas in sorted(comparadores.items()):
            ponto, reps = reps_delta_agrupado(notas_phigen[b], notas[b], grupos.get(b), rng,
                                              n_bootstrap)
            ic = _ic(reps, NIVEL_G2_2)
            piso = MARGEM_QUANTIZADO_G2_2 if nome in abaixo_de_q8 else 0.0
            contra[nome] = {"delta": round(ponto, 4), "ic99": ic, "vence": ic[0] > piso}
        por[b] = {"medido": True, "contra": contra,
                  "vence": bool(contra) and all(v["vence"] for v in contra.values())}
    vitorias = sum(v["vence"] for v in por.values())
    return {"passa": vitorias >= VITORIAS_G2_2, "vitorias": vitorias, "por_benchmark": por}
