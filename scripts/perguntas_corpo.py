#!/usr/bin/env python3
"""As perguntas do estrato "corpo": a resposta está no TEXTO COMPLETO e não no resumo.

As 650 perguntas da medida (DOC-13 §9.1) saem de um fato do resumo — é o desenho dela. O
que um índice de trechos compra de verdade é responder o que o resumo NÃO diz, e para isso
não havia pergunta nenhuma (PLANO-trechos-do-texto-completo §2). Este é o conjunto que
falta: escrito pelo Claude, como o formal, lendo o resumo e uma passagem do corpo.

    P = .venv-treino\\Scripts\\python.exe scripts\\perguntas_corpo.py
    P --passagens 450          # sorteia os artigos e UMA passagem do corpo de cada (CPU)
    P --exportar -n 25         # o próximo lote: resumo + passagem, sem título
    P --importar <respostas>   # o lote escrito, pelas guardas
    P --fechar                 # as 150 primeiras aceitas, e o agregado

Disjunto do teste e do desenvolvimento da busca: anda na MESMA permutação do conjunto
formal, depois de tudo o que os dois tentaram, e só em artigos com texto completo.

As perguntas, os gabaritos e as passagens ficam em `data/processed/assistente/`, fora do
git (DOC-11 §8.1). Vai para o git só o agregado, com os hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

DIR = RAIZ / "data/processed/assistente"
AVALIACAO = RAIZ / "data/processed/avaliacao"
TEXTOS = RAIZ / "data/processed/redpajama_fisica"
INDICE = RAIZ / "data/processed/indice_busca"
PASSAGENS = DIR / "corpo_passagens.jsonl"
ITENS = DIR / "itens_corpo.json"
SEMENTE_CORPO = 20261009
N_CORPO = 150
TRECHOS_POR_PASSAGEM = 3
LIMITE_TRECHO = 160
MIN_TOKENS_PASSAGEM = 300

# As regras com que o Claude escreve. São as do conjunto formal (`SISTEMA_PERGUNTA`: a
# pergunta vale sozinha, o gabarito é o conteúdo específico) mais as três do estrato.
REGRAS_CORPO = (
    "Você recebe o RESUMO de um artigo de Física e uma PASSAGEM do corpo dele; o título "
    "fica de fora. Escreva UMA pergunta, em português, cuja resposta é um achado concreto e "
    "conferível que está na PASSAGEM.\n"
    "1. FORA DO RESUMO: nem o valor, nem a direção, nem o mecanismo da resposta podem ser "
    "tirados do resumo. Se o resumo já responde, não há item.\n"
    "2. DO PRÓPRIO ARTIGO: um resultado, um parâmetro, um detalhe do método ou da montagem "
    "DESTE artigo — não um fato que a passagem atribui a outro trabalho.\n"
    "3. NA PASSAGEM: o gabarito está escrito nela; números saem como estão lá.\n"
    "A pergunta precisa valer sozinha, para quem nunca viu o artigo: nomeia o sistema, o "
    "material ou o modelo; não usa 'o modelo proposto', 'neste trabalho', 'os autores', nem "
    "símbolo definido só no artigo. O gabarito é o conteúdo específico em uma frase curta.\n"
    "Sem achado conferível que cumpra as três (derivação intermediária, revisão de trabalhos "
    "alheios, definição de notação): NENHUMA.")

_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")


def assinatura() -> str:
    from phifm.eval import assistente as m

    return hashlib.sha256(f"{REGRAS_CORPO}\n{m.SISTEMA_PERGUNTA}\n{m.AUTOR}".encode()).hexdigest()


def _cache() -> Path:
    from phifm.eval import assistente as m

    return DIR / f"tentativas_corpo_{m.AUTOR}_{assinatura()[:12]}_{SEMENTE_CORPO}.jsonl"


def _ler(arq: Path) -> list[dict]:
    if not arq.exists():
        return []
    return [json.loads(x) for x in arq.read_text(encoding="utf-8").splitlines() if x.strip()]


def numeros(texto: str) -> set[str]:
    """Os números do texto, com vírgula decimal virando ponto. Um dígito sozinho não entra:
    "2" está em qualquer resumo."""
    return {n for n in (x.replace(",", ".") for x in _NUMERO.findall(texto)) if len(n) >= 2}


def guarda_numerica(gabarito: str, resumo: str, passagem: str) -> str:
    """`no_resumo` se TODOS os números do gabarito já estão no resumo; `fora_da_passagem`
    se algum não está na passagem; senão `ok` — ou `sem_numero`, quando a guarda não tem
    o que conferir e a única conferência é a leitura de quem escreveu."""
    ns = numeros(gabarito)
    if not ns:
        return "sem_numero"
    if not ns <= numeros(passagem):
        return "fora_da_passagem"
    return "no_resumo" if ns <= numeros(resumo) else "ok"


def ordem_corpo(pares_validacao: Path, spine: Path) -> list[str]:
    """A permutação do primário, sem o que o teste e o desenvolvimento tentaram, e só com
    quem tem texto completo."""
    import polars as pl

    from phifm.eval import assistente as m

    base = m.assinatura_autor()[:12]
    teste = m.ler_cache(DIR / f"tentativas_{m.AUTOR}_{base}_{m.SEMENTE}.jsonl")
    dev = m.ler_cache(DIR / f"tentativas_devbusca_{m.AUTOR}_{base}_{m.SEMENTE}.jsonl")
    if not teste or not dev:
        raise SystemExit("faltam os caches do teste ou do desenvolvimento: sem eles não há "
                         "como garantir que o estrato é disjunto.")
    vistos = {a for _, a in teste} | {a for _, a in dev}
    com_texto = set(pl.scan_parquet(TEXTOS / "part-*.parquet").select("arxiv_id")
                    .collect()["arxiv_id"].to_list())
    return [a for a in m.ordem_formal(pares_validacao, spine)["primario"]
            if a not in vistos and a in com_texto]


def passagem_de(arxiv_id: str, texto: str, contar) -> dict | None:
    """UMA passagem do corpo: `TRECHOS_POR_PASSAGEM` trechos seguidos da mesma seção, fora
    da introdução e da conclusão, sorteada com semente presa ao artigo."""
    import numpy as np

    from phifm.retrieval.trechos import cortar

    ts = cortar(texto, contar, limite=LIMITE_TRECHO, sobreposicao=0)
    k = TRECHOS_POR_PASSAGEM
    janelas = [i for i in range(len(ts) - k + 1)
               if all(t.tipo == "corpo" and t.secao == ts[i].secao and not t.excede
                      for t in ts[i:i + k])
               and sum(t.n_tokens for t in ts[i:i + k]) >= MIN_TOKENS_PASSAGEM]
    if not janelas:
        return None
    semente = int(hashlib.blake2b(arxiv_id.encode(), digest_size=4).hexdigest(), 16)
    i = janelas[int(np.random.default_rng([SEMENTE_CORPO, semente]).integers(len(janelas)))]
    return {"secao": ts[i].secao, "primeiro_trecho": i,
            "passagem": " ".join(t.texto for t in ts[i:i + k]),
            "n_tokens": sum(t.n_tokens for t in ts[i:i + k])}


def sortear_passagens(n: int, pares_validacao: Path, spine: Path) -> None:
    import polars as pl
    from transformers import AutoTokenizer

    feitas = _ler(PASSAGENS)
    ordem = ordem_corpo(pares_validacao, spine)
    if [x["arxiv_id"] for x in feitas] != ordem[:len(feitas)]:
        raise SystemExit(f"{PASSAGENS.name} não é um prefixo da ordem: o sorteio mudou.")
    faltam = ordem[len(feitas):n]
    if not faltam:
        print(f"{len(feitas)} passagens já sorteadas.")
        return
    modelo = json.loads((INDICE / "manifesto.json").read_text(encoding="utf-8"))["modelo"]
    tok = AutoTokenizer.from_pretrained(modelo)

    def contar(textos: list[str]) -> list[int]:
        if not textos:
            return []
        return [len(x) for x in tok(textos, add_special_tokens=False, verbose=False)["input_ids"]]

    textos = dict(pl.scan_parquet(TEXTOS / "part-*.parquet")
                  .filter(pl.col("arxiv_id").is_in(faltam)).unique("arxiv_id")
                  .collect().iter_rows())
    resumos = dict(pl.scan_parquet(spine).select("arxiv_id", "abstract")
                   .filter(pl.col("arxiv_id").is_in(faltam)).collect().iter_rows())
    with PASSAGENS.open("a", encoding="utf-8") as f:
        for i, a in enumerate(faltam, start=len(feitas)):
            p = passagem_de(a, textos[a], contar)
            d = {"ordem": i, "arxiv_id": a, "resumo": " ".join((resumos[a] or "").split())}
            d |= p if p else {"sem_passagem": True}
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    todas = _ler(PASSAGENS)
    print(f"{len(todas)} artigos · {sum(1 for x in todas if not x.get('sem_passagem'))} com "
          f"passagem → {PASSAGENS.relative_to(RAIZ)}")


def _pendentes(n: int) -> list[dict]:
    """As próximas `n` passagens sem tentativa, em ordem. Artigo sem passagem não é pendente."""
    feitas = {t["arxiv_id"] for t in _ler(_cache())}
    saida = []
    for x in _ler(PASSAGENS):
        if x.get("sem_passagem"):
            continue
        if x["arxiv_id"] in feitas:
            if saida:
                raise SystemExit(f"lacuna na ordem: {x['arxiv_id']} tem tentativa depois de pendentes")
            continue
        saida.append(x)
        if len(saida) >= n:
            break
    return saida


def exportar(n: int) -> None:
    lote = _pendentes(n)
    if not lote:
        raise SystemExit("não há passagem pendente: rode --passagens com um número maior.")
    destino = DIR / f"lote_corpo_{lote[0]['ordem']:04d}.json"
    destino.write_text(json.dumps(
        [{k: x[k] for k in ("ordem", "arxiv_id", "resumo", "secao", "passagem")} for x in lote],
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(lote)} passagens (ordem {lote[0]['ordem']}–{lote[-1]['ordem']}) → {destino}")


def importar(arq: Path, spine: Path) -> None:
    import polars as pl

    from phifm.eval import assistente as m

    respostas = json.loads(arq.read_text(encoding="utf-8"))
    esperado = _pendentes(len(respostas))
    if [x["arxiv_id"] for x in esperado] != [r["arxiv_id"] for r in respostas]:
        raise SystemExit("o lote não continua a ordem do sorteio")
    titulos = dict(pl.scan_parquet(spine).select("arxiv_id", "title")
                   .filter(pl.col("arxiv_id").is_in([r["arxiv_id"] for r in respostas]))
                   .collect().iter_rows())
    novas = []
    for x, r in zip(esperado, respostas, strict=True):
        if r.get("nenhuma"):
            novas.append({"estrato": "corpo", "ordem": x["ordem"], "arxiv_id": x["arxiv_id"],
                          "situacao": "nenhuma", "pergunta": "", "gabarito": "",
                          "sobreposicao": 0.0, "critica": ""})
            continue
        t = m.guardas("corpo", x["ordem"], x["arxiv_id"], titulos.get(x["arxiv_id"]) or "",
                      r["pergunta"].strip(), r["gabarito"].strip())
        d = {"estrato": t.estrato, "ordem": t.ordem, "arxiv_id": t.arxiv_id,
             "situacao": t.situacao, "pergunta": t.pergunta, "gabarito": t.gabarito,
             "sobreposicao": t.sobreposicao, "critica": ""}
        numerica = guarda_numerica(t.gabarito, x["resumo"], x["passagem"])
        d["critica"] = numerica
        if t.situacao == "aceita" and numerica in ("no_resumo", "fora_da_passagem"):
            d["situacao"] = numerica
        novas.append(d)
    with _cache().open("a", encoding="utf-8") as f:
        for d in novas:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(novas)} tentativas registradas · {dict(Counter(d['situacao'] for d in novas))}")
    for d in novas:
        if d["situacao"] not in ("aceita", "nenhuma"):
            print(f"   caiu ({d['situacao']}): {d['pergunta']}")
    print(f"   aceitas no total: {sum(1 for t in _ler(_cache()) if t['situacao'] == 'aceita')}")


def fechar() -> None:
    from phifm.eval import assistente as m

    tentativas = _ler(_cache())
    aceitas = [t for t in tentativas if t["situacao"] == "aceita"][:N_CORPO]
    if len(aceitas) < N_CORPO:
        raise SystemExit(f"{len(aceitas)} aceitas de {N_CORPO} — escreva mais lotes.")
    ultimo = aceitas[-1]["ordem"]
    usadas = [t for t in tentativas if t["ordem"] <= ultimo]
    passagens = {x["arxiv_id"]: x for x in _ler(PASSAGENS)}
    base = m.assinatura_autor()[:12]
    vistos = ({a for _, a in m.ler_cache(DIR / f"tentativas_{m.AUTOR}_{base}_{m.SEMENTE}.jsonl")}
              | {a for _, a in m.ler_cache(
                  DIR / f"tentativas_devbusca_{m.AUTOR}_{base}_{m.SEMENTE}.jsonl")})
    if vistos & {t["arxiv_id"] for t in aceitas}:
        raise SystemExit("o estrato não é disjunto do teste e do desenvolvimento.")
    ITENS.write_text(json.dumps(aceitas, ensure_ascii=False, indent=1), encoding="utf-8")
    sha = hashlib.sha256(json.dumps(aceitas, ensure_ascii=False).encode()).hexdigest()
    sha_p = hashlib.sha256(json.dumps(
        [passagens[t["arxiv_id"]]["passagem"] for t in aceitas], ensure_ascii=False).encode()).hexdigest()
    sem_passagem = sum(1 for x in passagens.values()
                       if x.get("sem_passagem") and x["ordem"] <= ultimo)
    agregado = {
        "etapa": "corpo", "regra": "PLANO-trechos-do-texto-completo §4 · estrato corpo",
        "autor": m.AUTOR, "semente": SEMENTE_CORPO, "cota": N_CORPO,
        "o_que_e": "perguntas cuja resposta está numa passagem do corpo do artigo e não no resumo",
        "passagem": f"{TRECHOS_POR_PASSAGEM} trechos seguidos de uma seção do corpo "
                    f"(≥ {MIN_TOKENS_PASSAGEM} tokens), sorteada por artigo",
        "disjunto_de": ["teste (650)", "desenvolvimento da busca (200)"],
        "artigos_andados": ultimo + 1, "artigos_sem_passagem": sem_passagem,
        "tentativas_por_situacao": dict(Counter(t["situacao"] for t in usadas)),
        "guarda_numerica_das_aceitas": dict(Counter(t["critica"] for t in aceitas)),
        "sobreposicao_media_com_o_titulo": round(
            sum(t["sobreposicao"] for t in aceitas) / len(aceitas), 4),
        "sem_revisao_humana": "escrito e conferido pelo Claude; o dono não julga amostras "
                              "(decisão de 2026-10-04)",
        "assinatura_regras": assinatura(), "sha256_itens": sha, "sha256_passagens": sha_p}
    arq = AVALIACAO / "assistente_itens_corpo.json"
    arq.write_text(json.dumps(agregado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(agregado, ensure_ascii=False, indent=2))
    print(f"→ {ITENS.relative_to(RAIZ)} (fora do git) · {arq.relative_to(RAIZ)}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--passagens", type=int, help="sorteia até este número de artigos")
    p.add_argument("--exportar", action="store_true")
    p.add_argument("-n", type=int, default=25)
    p.add_argument("--importar", type=Path)
    p.add_argument("--fechar", action="store_true")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    p.add_argument("--pares-validacao", type=Path,
                   default=RAIZ / "data/processed/pares/pares_validacao.parquet")
    a = p.parse_args()
    if a.passagens:
        sortear_passagens(a.passagens, a.pares_validacao, a.spine)
    elif a.exportar:
        exportar(a.n)
    elif a.importar:
        importar(a.importar, a.spine)
    elif a.fechar:
        fechar()
    else:
        p.error("--passagens, --exportar, --importar ou --fechar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
