"""As sondas de contaminação do G2 (DOC-12 §3.7.9 A): S1 e S2, gulosas e sem exemplos, nos
2.528 itens de física, na base e depois no ΦGen (§3.7.9 C).

- **S1.** O modelo recebe a primeira metade das fichas do enunciado e gera 48. Só vale para
  enunciado com metade de pelo menos 32 fichas (g2.yaml: "metade do enunciado (≥32 fichas)").
- **S2.** O modelo recebe o enunciado, uma linha em branco e os primeiros 40% das fichas da
  solução, e gera 64. Só vale para item com solução cujo resto tenha pelo menos 13 fichas
  (o SciBench do teste não tem solução: lá só cabe a S1).
- **Suspeito** (o critério pré-registrado): alguma sequência de 13 fichas da geração aparece,
  em qualquer posição, na continuação verdadeira: o resto do enunciado (S1) ou o resto da
  solução (S2). É o 13-grama do GPT-3, contado em fichas do próprio tokenizer.
- **Relatados, sem decidir:** se algum casamento cai na mesma posição (alinhado); se todo
  13-grama casado também está no prompt (cópia do prompt, não memória); o maior trecho comum
  em fichas, e quantos suspeitos têm trecho de ≥32 fichas. No dev, a S2 casa sobretudo
  pedaços de LaTeX (13 fichas de fórmula são poucos símbolos), que quem sabe continuar a
  conta escreve igual à solução: o trecho de ≥32 fichas separa melhor a memória.
- **Nulo.** O PHYBench saiu depois dos dados da base: a fração de suspeitos nele estima o
  acaso (frase-feita de enunciado e fórmula de solução).
- **O lacre** (§3.7.3: até o veredito, nada se lê por item). As listas de ids suspeitos são
  gravadas fora do git, num JSON canônico; o agregado versionado leva só os contadores e o
  SHA-256 desse JSON. No veredito, a lista é aberta e conferida contra o hash.

Tudo em fichas: o corte é feito nas fichas do tokenizer do modelo (o do ΦGen é o mesmo da
base, §3.7.5), o prompt vai ao `llama-server` como lista de fichas, e as fichas geradas
voltam pelo próprio servidor, sem retokenizar. Gulosa (T=0), sem cache de prompt (a S1 e a
S2 do mesmo item têm o mesmo começo, e o cache mudaria a conta conforme a ordem), num
servidor de um slot só, um pedido por vez, sempre na mesma ordem (`medir_g2_sondas`): assim
a medida inteira se repete ficha a ficha. Prompt e geração que não cabem no contexto do slot
não se aplicam (não acontece no dev).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Sequence
from pathlib import Path

N_FICHAS = 13
S1_GERA, S2_GERA = 48, 64
S1_MIN_METADE = 32
S2_FRACAO = 0.4
SEPARADOR_S2 = "\n\n"
TRECHO_LONGO = 32
SONDAS = ("S1", "S2")


def parametros() -> dict:
    """Os parâmetros das sondas, para o hash da configuração."""
    return {"n_fichas": N_FICHAS, "S1": {"gera": S1_GERA, "metade_minima": S1_MIN_METADE},
            "S2": {"gera": S2_GERA, "fracao_da_solucao": S2_FRACAO, "separador": SEPARADOR_S2,
                   "resto_minimo": N_FICHAS},
            "criterio": "13-grama da geração em qualquer posição da continuação verdadeira",
            "relatados": {"alinhado": "mesma posição", "copia_do_prompt": "todo 13-grama "
                          "casado também no prompt", "trecho_longo": TRECHO_LONGO},
            "teto": "prompt + geradas ≤ contexto do slot (/props), senão não se aplica"}


def pedido(codigo: dict | None = None) -> dict:
    """Gulosa e tudo o mais desligado, como nas outras medidas; sem cache de prompt. A chave
    `sondas` (os parâmetros e, se dado, o SHA-256 do código das sondas) só entra no hash da
    configuração: `gerar` a tira antes de mandar ao servidor."""
    from phifm.eval.g2_geracao import amostragem

    sondas = parametros() | ({"codigo": codigo} if codigo else {})
    return amostragem(temperatura=0.0, top_p=1.0) | {"cache_prompt": False, "sondas": sondas}


# ── os cortes ────────────────────────────────────────────────────────────────


def corte_s1(enunciado: Sequence[int]) -> tuple[list[int], list[int]] | None:
    """(prompt, continuação verdadeira) da S1, ou None se a metade tem menos de 32 fichas."""
    meio = len(enunciado) // 2
    if meio < S1_MIN_METADE:
        return None
    return list(enunciado[:meio]), list(enunciado[meio:])


def corte_s2(prefixo: Sequence[int], solucao: Sequence[int]) -> tuple[list[int], list[int]] | None:
    """(prompt, continuação verdadeira) da S2: o prefixo (enunciado e separador) e 40% da
    solução; None se não há solução ou se o resto tem menos de 13 fichas. 0,4·n nunca cai
    em x,5 para n inteiro: o `round` não tem empate."""
    k = round(S2_FRACAO * len(solucao))
    if k < 1 or len(solucao) - k < N_FICHAS:
        return None
    return list(prefixo) + list(solucao[:k]), list(solucao[k:])


# ── a comparação ─────────────────────────────────────────────────────────────


def _gramas(ids: Sequence[int], n: int = N_FICHAS) -> set[tuple[int, ...]]:
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}


def suspeito(gerado: Sequence[int], verdadeira: Sequence[int], n: int = N_FICHAS) -> bool:
    """≥1 sequência de n fichas da geração em qualquer posição da continuação verdadeira."""
    return bool(_gramas(gerado, n) & _gramas(verdadeira, n))


def alinhado(gerado: Sequence[int], verdadeira: Sequence[int], n: int = N_FICHAS) -> bool:
    """O mesmo, mas na mesma posição (gerado[i:i+n] == verdadeira[i:i+n]); só relatado."""
    return any(tuple(gerado[i:i + n]) == tuple(verdadeira[i:i + n])
               for i in range(min(len(gerado), len(verdadeira)) - n + 1))


def copia_do_prompt(gerado: Sequence[int], verdadeira: Sequence[int], prompt: Sequence[int],
                    n: int = N_FICHAS) -> bool:
    """Suspeito, mas todo 13-grama casado também está no prompt; só relatado."""
    casados = _gramas(gerado, n) & _gramas(verdadeira, n)
    return bool(casados) and casados <= _gramas(prompt, n)


def maior_trecho_comum(a: Sequence[int], b: Sequence[int]) -> int:
    """O maior trecho contíguo de fichas comum às duas sequências; só relatado."""
    melhor, linha = 0, [0] * (len(b) + 1)
    for x in a:
        nova = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            if x == y:
                nova[j] = linha[j - 1] + 1
                melhor = max(melhor, nova[j])
        linha = nova
    return melhor


# ── o servidor ───────────────────────────────────────────────────────────────


def _post(url: str, rota: str, corpo: dict, tempo_max_s: int = 600) -> dict:
    req = urllib.request.Request(url + rota, data=json.dumps(corpo).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=tempo_max_s) as r:
        return json.loads(r.read())


def contexto_do_slot(url: str) -> int:
    """O contexto de um slot, como o servidor o montou (/props). Não é contexto // slots: com
    1 slot, o servidor limita ao contexto de treino do modelo (32.768 no Qwen3)."""
    with urllib.request.urlopen(url + "/props", timeout=60) as r:
        return int(json.loads(r.read())["default_generation_settings"]["n_ctx"])


def fichas(url: str, texto: str) -> list[int]:
    return _post(url, "/tokenize", {"content": texto, "add_special": False})["tokens"]


def gerar(url: str, prompt: list[int], n_predict: int, pedido_base: dict) -> list[int]:
    corpo = {k: v for k, v in pedido_base.items() if k != "sondas"}
    corpo |= {"prompt": prompt, "n_predict": n_predict, "stream": False, "return_tokens": True}
    return list(_post(url, "/completion", corpo)["tokens"])


def sha_entrada(prompt: Sequence[int], verdadeira: Sequence[int]) -> str:
    """O SHA-256 das fichas que a sonda mostrou e das que ela esperava: base e ΦGen (e a 1,7B
    e a 8B, com o mesmo tokenizer) têm de ter sondado exatamente as mesmas."""
    return hashlib.sha256(json.dumps([list(prompt), list(verdadeira)]).encode()).hexdigest()


def sondar(url: str, enunciado: str, solucao: str | None, sonda: str, pedido_base: dict,
           teto: int | None = None) -> dict:
    """Uma sonda num item: os contadores e as fichas geradas (nada de texto). `teto` é o
    contexto do slot do servidor."""
    if sonda == "S1":
        corte, n, motivo = corte_s1(fichas(url, enunciado)), S1_GERA, "metade_curta"
    elif not (isinstance(solucao, str) and solucao.strip()):
        return {"aplicavel": False, "motivo": "sem_solucao"}
    else:
        corte = corte_s2(fichas(url, enunciado + SEPARADOR_S2), fichas(url, solucao))
        n, motivo = S2_GERA, "resto_curto"
    if corte is None:
        return {"aplicavel": False, "motivo": motivo}
    prompt, verdadeira = corte
    if teto is not None and len(prompt) + n > teto:
        return {"aplicavel": False, "motivo": "contexto", "n_prompt": len(prompt)}
    gerado = gerar(url, prompt, n, pedido_base)
    return {"aplicavel": True, "n_prompt": len(prompt), "n_verdadeira": len(verdadeira),
            "sha_entrada": sha_entrada(prompt, verdadeira),
            "n_gerado": len(gerado), "suspeito": suspeito(gerado, verdadeira),
            "alinhado": alinhado(gerado, verdadeira),
            "copia_do_prompt": copia_do_prompt(gerado, verdadeira, prompt),
            "maior_trecho": maior_trecho_comum(gerado, verdadeira), "gerado": gerado}


# ── a retomada depois de uma queda ───────────────────────────────────────────


def aparar(arquivo: Path) -> int:
    """Tira de um jsonl a cauda de uma queda de energia (bytes NUL que o NTFS deixa e a última
    linha sem fim que não é JSON) e devolve os bytes tirados. Linha completa nunca sai: a que
    só perdeu o "\\n" ganha o "\\n" de volta."""
    p = Path(arquivo)
    if not p.exists():
        return 0
    dados = p.read_bytes()
    limpo = dados.rstrip(b"\x00")
    if limpo and not limpo.endswith(b"\n"):
        ultima = limpo.rsplit(b"\n", 1)[-1]
        try:
            json.loads(ultima)
            limpo += b"\n"
        except ValueError:
            limpo = limpo[:len(limpo) - len(ultima)]
    if limpo != dados:
        p.write_bytes(limpo)
    return max(0, len(dados) - len(limpo))


# ── os agregados e o lacre ───────────────────────────────────────────────────


def _lista(ids: dict[str, dict[str, set[str]]]) -> dict[str, dict[str, list[str]]]:
    return {b: {s: sorted(x) for s, x in sorted(ss.items())} for b, ss in sorted(ids.items())}


def canonico(listas: dict) -> bytes:
    """O JSON canônico das listas lacradas: o arquivo fora do git tem exatamente estes bytes,
    e o SHA-256 deles vai ao agregado versionado."""
    return json.dumps(listas, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def agregar(registros: list[dict], nulo: str = "phybench") -> dict:
    """Por benchmark e sonda: aplicáveis (e por que não), suspeitos e a fração, e os
    relatados; por benchmark, a união S1 ∪ S2 por item, sobre todos os itens e sobre os que
    têm ≥1 sonda aplicável (os outros nunca podem ter sinal). `listas` (por item: só fora do
    git, ver o lacre) traz os ids suspeitos e os de trecho ≥32, por benchmark e sonda;
    `sha256_listas` é o hash do JSON canônico delas. `sha256_entradas` resume as fichas
    sondadas (ou o motivo de não aplicar) de cada sonda: tem de ser igual entre os braços."""
    por: dict = {}
    suspeitos: dict[str, dict[str, set[str]]] = {}
    longos: dict[str, dict[str, set[str]]] = {}
    aplicaveis: dict[str, set[str]] = {}
    todos: dict[str, set[str]] = {}
    for r in registros:
        b, s = r["bench"], r["sonda"]
        todos.setdefault(b, set()).add(r["item_id"])
        x = por.setdefault(b, {}).setdefault(s, {
            "itens": 0, "aplicaveis": 0, "suspeitos": 0, "alinhados": 0,
            "so_copia_do_prompt": 0, "trecho_longo": 0, "nao_aplicaveis": {}})
        x["itens"] += 1
        if not r.get("aplicavel"):
            m = r.get("motivo", "?")
            x["nao_aplicaveis"][m] = x["nao_aplicaveis"].get(m, 0) + 1
            continue
        x["aplicaveis"] += 1
        aplicaveis.setdefault(b, set()).add(r["item_id"])
        if not r["suspeito"]:
            continue
        x["suspeitos"] += 1
        x["alinhados"] += bool(r["alinhado"])
        x["so_copia_do_prompt"] += bool(r.get("copia_do_prompt"))
        suspeitos.setdefault(b, {}).setdefault(s, set()).add(r["item_id"])
        if r["maior_trecho"] >= TRECHO_LONGO:
            x["trecho_longo"] += 1
            longos.setdefault(b, {}).setdefault(s, set()).add(r["item_id"])
    for b, ss in por.items():
        for x in ss.values():
            x["fracao"] = round(x["suspeitos"] / x["aplicaveis"], 4) if x["aplicaveis"] else None
        uniao = set().union(*suspeitos.get(b, {}).values())
        uniao_longa = set().union(*longos.get(b, {}).values())
        itens, com_sonda = len(todos[b]), len(aplicaveis.get(b, set()))
        ss["uniao"] = {"itens": itens, "com_sonda_aplicavel": com_sonda,
                       "sem_sonda_aplicavel": itens - com_sonda, "suspeitos": len(uniao),
                       "fracao": round(len(uniao) / itens, 4) if itens else None,
                       "fracao_dos_aplicaveis": round(len(uniao) / com_sonda, 4) if com_sonda else None,
                       "trecho_longo": len(uniao_longa)}
    listas = {"suspeitos": _lista(suspeitos), "trecho_longo": _lista(longos)}
    no_nulo = por.get(nulo, {})
    entradas = sorted((r["bench"], r["item_id"], r["sonda"],
                       r.get("sha_entrada") if r.get("aplicavel") else f"nao:{r.get('motivo')}")
                      for r in registros)
    return {"por_benchmark": por, "nulo": nulo,
            "sha256_entradas": hashlib.sha256(json.dumps(entradas).encode()).hexdigest(),
            "fracao_no_nulo": no_nulo.get("uniao", {}).get("fracao_dos_aplicaveis"),
            "fracao_no_nulo_por_sonda": {s: no_nulo.get(s, {}).get("fracao") for s in SONDAS},
            "total_suspeitos": sum(len(set().union(*ss.values())) for ss in suspeitos.values()),
            "listas": listas, "sha256_listas": hashlib.sha256(canonico(listas)).hexdigest()}
