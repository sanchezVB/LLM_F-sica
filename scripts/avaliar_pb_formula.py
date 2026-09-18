#!/usr/bin/env python3
"""PB-Formula: recuperar documentos por EQUAÇÃO, e a regra ANTES de qualquer número.

    .venv-treino\\Scripts\\python.exe scripts\\avaliar_pb_formula.py \\
        --modelos models/phiemb-do-sistema models/phiemb-gte-base-400k-melhor \\
                  models/phiemb-t2eq-modernbert-200k-melhor \\
        --out data/processed/avaliacao/pb_formula_resultado.json

O DOC-11 §6.3 diz que este benchmark "mede diretamente a capacidade que recuperação
densa costuma perder: casamento simbólico exato". Nenhum modelo foi medido nele até
agora, então a regra abaixo é pré-registrada de verdade.

## ⚠️ As duas decisões de protocolo, e por que elas não são detalhe

**1. O conjunto primário é o estrato `notacional` sem documento quase igual: 24.508
itens dos 374.739.** Medido no corpus inteiro (DOC-11 §6.3-medido): em 53,1% dos itens
todos os alvos escrevem a equação byte a byte como a consulta, e em mais 37,0% a
diferença é só espaço, `\\label`, `&` ou pontuação. Um casador de string resolve nove
de cada dez itens sem saber nada de notação, e a média sobre o benchmark inteiro
premiaria isso. E 36,3% dos itens ligam a consulta a um documento com o qual ela
divide 5 ou mais equações — nesses, o que se recupera é o documento parecido, não a
equação. Os estratos descartados são medidos e reportados AO LADO, como controle.

**2. O documento é representado pelas EQUAÇÕES dele, e não pelo texto truncado.** Os
nossos recuperadores leem 192 tokens (T1e: ler 256 ou 384 não muda o recall), e as
equações de um artigo estão no corpo — fora dessa janela. Indexar o texto truncado
mediria "o começo do artigo casa com a equação?", que não é a pergunta do §6.3. Então
cada documento entra pelo conjunto das equações de conteúdo dele, e o escore do
documento é o MÁXIMO sobre elas.

⚠️ Entram TODAS as equações de conteúdo dos documentos do pool, e não só as que o
benchmark usou para formar itens. Indexar só as compartilhadas entregaria o gabarito:
o índice seria exatamente o conjunto das respostas.

## A REGRA, escrita antes

**PRIMÁRIA: recall@10 por item no conjunto primário**, com o BM25 sobre as mesmas
equações como CONTROLE LEXICAL, e IC por bootstrap pareado por item.

  DENSO ABAIXO DO BM25  ->  a premissa do DOC-11 §6.3 se confirma: recuperação densa
  (IC exclui zero)          perde casamento simbólico, e o benchmark mede algo que o
                            sistema não tem. Justifica o estágio simbólico do DOC-13.

  EMPATE                ->  não estabelecido nesta escala; reportar como tal.
  (IC cruza zero)

  DENSO ACIMA DO BM25   ->  a premissa não vale para estes modelos; o benchmark mede
  (IC exclui zero)          algo que o denso já faz, e o §6.3 precisa ser revisto.

**SECUNDÁRIA (não derruba a primária):** os mesmos sistemas nos estratos `identica` e
`superficial`. A previsão registrada é que a vantagem do BM25 seja MAIOR neles — é o
que separa "casar string" de "reconhecer notação". Se a diferença entre estratos não
aparecer, o estrato primário não está medindo o que o nome diz.

**CHECAGEM DE MANIPULAÇÃO:** o teto do pool tem de ser 1,0 — todo item precisa de ao
menos um alvo dentro do pool, senão nenhum modelo pode acertá-lo (a armadilha que o
G1 pagou, `formula.conferir_pool`).

⚠️ **O documento da CONSULTA é ignorado POR ITEM**, e não retirado do pool. Ele contém
a própria equação, então recuperá-lo não é acerto nem erro — é o ponto de partida. E
retirá-lo do pool seria errado: medido com 2.000 itens, o documento de consulta de um
item é ALVO de outro, e tirá-lo derrubaria o teto do segundo abaixo de 1,0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402

utf8()

import polars as pl  # noqa: E402

from phifm.eval.benchmarks.pb_formula_protocolo import (  # noqa: E402
    ESTRATO_PRIMARIO,
    MAX_FORMAS_EM_COMUM,
    escore_por_documento,
    itens_primarios,
    metricas_por_item,
)
from phifm.eval.hibrido import BM25  # noqa: E402

PB = Path("data/processed/pb_formula")
SAIDA = Path("data/processed/avaliacao/pb_formula_resultado.json")
KS = (1, 10, 50)


def carregar_itens(pb: Path, estrato: str, max_comum: int, n: int, semente: int,
                   ) -> tuple[pl.DataFrame, dict]:
    itens = (pl.read_parquet(pb / "itens.parquet")
             .join(pl.read_parquet(pb / "itens_estratos.parquet"), on="forma"))
    comuns = itens["max_formas_em_comum"].fill_null(0).to_numpy()
    mascara = itens_primarios(itens["grafia"].to_numpy(), comuns, estrato, max_comum)
    primarios = itens.filter(pl.Series(mascara))
    if primarios.height < n:
        raise SystemExit(f"o estrato {estrato!r} tem {primarios.height} itens e "
                         f"foram pedidos {n}")
    rng = np.random.default_rng((semente, 0x7B1F))
    escolhidos = np.sort(rng.choice(primarios.height, n, replace=False))
    return primarios[escolhidos], {"itens_no_estrato": primarios.height,
                                   "itens_totais": itens.height}


def montar_pool(itens: pl.DataFrame, todos: np.ndarray, n_pool: int, semente: int,
                ) -> list[str]:
    """Alvos de todos os itens + documentos sorteados.

    ⚠️ O documento de consulta de um item pode ser ALVO de outro — medido: com 2.000
    itens acontece. Tirá-lo do pool quebraria o segundo item (teto < 1,0); deixá-lo
    daria ao primeiro um documento que contém a própria equação e não está no gabarito
    dele. Então ele fica no pool, e o mascaramento é POR ITEM, na hora de ordenar
    (`escores[i, consulta_de[i]] = -inf`).
    """
    alvos = {d for lista in itens["alvos"].to_list() for d in lista}
    for lista, consulta in zip(itens["alvos"].to_list(),
                               itens["documento_da_consulta"].to_list(), strict=True):
        if consulta in set(lista):
            raise SystemExit(f"o documento {consulta} é alvo do PRÓPRIO item")
    fora = alvos | set(itens["documento_da_consulta"].to_list())
    # ⚠️ ORDENADO antes de sortear, e isto não é zelo: `todos` vem de um `unique()`
    # em streaming, cuja ORDEM muda de execução para execução. Com um processo por
    # modelo, cada um montava um pool diferente — medido em 2026-09-18: 841.101,
    # 843.127, 836.422 e 831.048 equações nos quatro processos. Os sistemas ficavam
    # medidos contra distratores diferentes, e o pareado por item não valia nada.
    candidatos = np.array(sorted(d for d in todos if d not in fora), dtype=object)
    faltam = max(n_pool - len(alvos), 0)
    rng = np.random.default_rng((semente, 0x9001))
    extras = rng.choice(candidatos, min(faltam, len(candidatos)), replace=False)
    return sorted(alvos | set(extras.tolist()))


def equacoes_do_pool(pb: Path, pool: list[str]) -> pl.DataFrame:
    """Todas as equações de conteúdo dos documentos do pool — inclusive as não
    compartilhadas, senão o índice seria o gabarito."""
    return (pl.scan_parquet(str(pb / "ocorrencias" / "*.parquet"))
            .filter(pl.col("documento").is_in(pool))
            .select("documento", "latex")
            .unique()
            # ⚠️ ORDENADO: o `unique()` em streaming devolve em ordem arbitrária, e a
            # ordem entra na impressão digital do protocolo. O escore não muda com ela
            # — mas duas execuções que deveriam ser idênticas passariam a divergir na
            # assinatura, e a guarda do cache deixaria de distinguir o que importa.
            .sort("documento", "latex")
            .collect(engine="streaming"))


def pontuar_bm25(consultas: list[str], textos: list[str], doc_de_cada: np.ndarray,
                 n_docs: int) -> np.ndarray:
    indice = BM25().indexar(textos)
    saida = np.empty((len(consultas), n_docs), dtype=np.float32)
    for i, q in enumerate(consultas):
        saida[i] = escore_por_documento(indice.pontuar(q), doc_de_cada, n_docs)[0]
    return saida


def pontuar_denso(caminho: str, consultas: list[str], textos: list[str],
                  doc_de_cada: np.ndarray, n_docs: int, dispositivo: str,
                  max_tokens: int, lote: int) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    from phifm.eval.encoders import _codificar
    from phifm.training.embedding import escolher_dispositivo

    dev = escolher_dispositivo(dispositivo)
    tok = AutoTokenizer.from_pretrained(caminho)
    mod = AutoModel.from_pretrained(caminho, attn_implementation="eager").to(dev).eval()
    with torch.no_grad():
        vt = _codificar(mod, tok, textos, dev, max_tokens, lote).numpy()
        vq = _codificar(mod, tok, consultas, dev, max_tokens, lote).numpy()
    saida = np.empty((len(consultas), n_docs), dtype=np.float32)
    # Em blocos: a matriz consulta × equação inteira não cabe confortavelmente.
    for i in range(0, len(consultas), 128):
        bloco = vq[i:i + 128] @ vt.T
        saida[i:i + 128] = escore_por_documento(bloco, doc_de_cada, n_docs)
    return saida


def assinatura(a, pool: list[str] | None = None, n_equacoes: int = 0) -> dict:
    """O que define o protocolo. Um cache de outro protocolo não pode ser reusado.

    ⚠️ O POOL entra pela impressão digital, e não só pelo tamanho. Sem isso, dois
    sistemas medidos contra distratores diferentes entrariam no mesmo arquivo e o
    bootstrap pareado por item compararia coisas que não são pareadas — foi o que
    aconteceu em 2026-09-18, com um processo por modelo.
    """
    assin = {"estrato": a.estrato, "max_comum": a.max_comum, "n": a.n, "pool": a.pool,
             "semente": a.semente, "max_tokens": a.max_tokens}
    if pool is not None:
        digest = hashlib.sha256("\n".join(pool).encode("utf-8")).hexdigest()[:16]
        assin |= {"pool_sha": digest, "equacoes": n_equacoes}
    return assin


def carregar_cache(caminho: Path, assin: dict) -> dict:
    """O que já foi medido, se for do MESMO protocolo.

    ⚠️ Existe porque a primeira execução gastou 5 h e morreu sem memória de vídeo no
    TERCEIRO modelo, perdendo os dois primeiros. Cada sistema vai ao disco assim que
    termina; e cada modelo denso roda num processo próprio, porque a DirectML não
    devolve a memória do modelo anterior.
    """
    if not caminho.exists():
        return {}
    d = json.loads(caminho.read_text(encoding="utf-8"))
    if d.get("assinatura") != assin:
        raise SystemExit(
            f"{caminho} é de outro protocolo. Cache: {d.get('assinatura')}. "
            f"Agora: {assin}. Apague o cache, ou volte ao protocolo dele.")
    return {n: {k: np.asarray(v) if isinstance(v, list) else v for k, v in m.items()}
            for n, m in d["sistemas"].items()}


def gravar_cache(caminho: Path, assin: dict, sistemas: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(
        {"assinatura": assin,
         "sistemas": {n: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in m.items()} for n, m in sistemas.items()}},
        ensure_ascii=False), encoding="utf-8")


def avaliar(escores: np.ndarray, alvos_idx: list[set[int]],
            consulta_de: np.ndarray) -> dict:
    """⚠️ Mascara o documento da CONSULTA de cada item antes de ordenar: ele contém a
    própria equação, e recuperá-lo não é acerto nem erro — é o ponto de partida."""
    for i, d in enumerate(consulta_de):
        if d >= 0:
            escores[i, d] = -np.inf
    ordem = np.argsort(-escores, axis=1)
    return metricas_por_item(ordem, alvos_idx, KS)


def ler_pela_regra(nd: dict) -> tuple[str, str]:
    lo, hi = nd["ic95"]
    if hi < 0:
        return "DENSO ABAIXO DO BM25", (
            "a premissa do DOC-11 §6.3 se confirma: recuperação densa perde casamento "
            "simbólico, e o benchmark mede algo que o sistema não tem.")
    if lo > 0:
        return "DENSO ACIMA DO BM25", (
            "a premissa não vale para estes modelos: o denso já faz o que o §6.3 diz "
            "que ele perde, e a especificação precisa ser revista.")
    return "EMPATE", "não estabelecido nesta escala; fica reportado assim."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pb", type=Path, default=PB)
    ap.add_argument("--modelos", nargs="*", default=[],
                    help="sem nenhum, roda só o BM25 — é assim que se calibra o "
                         "tamanho do pool sem olhar para número de modelo")
    ap.add_argument("--n", type=int, default=2000, help="itens do conjunto primário")
    ap.add_argument("--pool", type=int, default=5000, help="documentos no pool")
    ap.add_argument("--estrato", default=ESTRATO_PRIMARIO)
    ap.add_argument("--max-comum", type=int, default=MAX_FORMAS_EM_COMUM)
    ap.add_argument("--semente", type=int, default=17)
    ap.add_argument("--dispositivo", default="dml")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--out", type=Path, default=SAIDA)
    ap.add_argument("--cache", type=Path, default=None,
                    help="por omissão, <out>.parcial.json. Cada sistema é gravado nele "
                         "assim que termina, e uma nova execução NÃO refaz o que já "
                         "está lá — ver `carregar_cache`")
    a = ap.parse_args()
    cache = a.cache or a.out.with_suffix(".parcial.json")

    itens, contagens = carregar_itens(a.pb, a.estrato, a.max_comum, a.n, a.semente)
    todos = (pl.scan_parquet(str(a.pb / "ocorrencias" / "*.parquet"))
             .select("documento").unique().collect(engine="streaming")
             ["documento"].to_numpy())
    pool = montar_pool(itens, todos, a.pool, a.semente)
    indice_de = {d: i for i, d in enumerate(pool)}
    alvos_idx = [{indice_de[d] for d in lista if d in indice_de}
                 for lista in itens["alvos"].to_list()]
    consulta_de = np.array([indice_de.get(d, -1)
                            for d in itens["documento_da_consulta"].to_list()],
                           dtype=np.int64)
    if any(not s for s in alvos_idx):
        raise SystemExit("teto do pool < 1,0: item sem alvo dentro do pool")

    eq = equacoes_do_pool(a.pb, pool)
    textos = eq["latex"].to_list()
    doc_de_cada = np.array([indice_de[d] for d in eq["documento"].to_list()],
                           dtype=np.int64)
    consultas = itens["consulta"].to_list()
    # A assinatura sai DAQUI: ela inclui a impressão digital do pool e o número de
    # equações, que é o que distingue duas execuções que só pareciam iguais.
    assin = assinatura(a, pool, len(textos))
    feitos = carregar_cache(cache, assin)
    print(f"itens {len(consultas):,} · pool {len(pool):,} documentos · "
          f"{len(textos):,} equações indexadas · pool_sha {assin['pool_sha']}",
          flush=True)
    if feitos:
        print(f"cache: {sorted(feitos)} já medidos, e não serão refeitos", flush=True)

    sistemas = dict(feitos)
    if "bm25" not in sistemas:
        t0 = time.perf_counter()
        sistemas["bm25"] = avaliar(
            pontuar_bm25(consultas, textos, doc_de_cada, len(pool)),
            alvos_idx, consulta_de)
        sistemas["bm25"]["segundos"] = round(time.perf_counter() - t0, 1)
        gravar_cache(cache, assin, sistemas)
        print(f"  bm25 · recall@10 {sistemas['bm25']['recall_10'].mean():.4f} · "
              f"{sistemas['bm25']['segundos']:.0f} s", flush=True)
    for caminho in a.modelos:
        nome = Path(caminho).name
        if nome in sistemas:
            continue
        t0 = time.perf_counter()
        sistemas[nome] = avaliar(
            pontuar_denso(caminho, consultas, textos, doc_de_cada, len(pool),
                          a.dispositivo, a.max_tokens, a.lote), alvos_idx, consulta_de)
        sistemas[nome]["segundos"] = round(time.perf_counter() - t0, 1)
        # ⚠️ Ao disco ANTES do próximo modelo: foi o terceiro que estourou a memória
        # de vídeo e levou junto as 5 h dos dois primeiros.
        gravar_cache(cache, assin, sistemas)
        print(f"  {nome} · recall@10 {sistemas[nome]['recall_10'].mean():.4f} · "
              f"{sistemas[nome]['segundos']:.0f} s", flush=True)

    def resumo(m: dict) -> dict:
        return {**{f"recall_{k}": round(float(np.asarray(m[f'recall_{k}']).mean()), 4)
                   for k in KS},
                "mrr": round(float(m["rr"].mean()), 4),
                "posto_mediano": float(np.median(m["posto"]))}

    # ⚠️ Import TARDIO: `avaliar_ablacao_secundarias` puxa o torch, e sem isto o
    # módulo inteiro deixa de ser importável na venv rápida — onde moram os testes
    # das partes puras (o do pool determinístico, por exemplo).
    from avaliar_ablacao_secundarias import bootstrap_pareado_itens

    contra_bm25 = {
        nome: bootstrap_pareado_itens(sistemas["bm25"]["recall_10"], m["recall_10"])
        for nome, m in sistemas.items() if nome != "bm25"}
    melhor = max(contra_bm25, key=lambda n: contra_bm25[n]["diferenca"]) if contra_bm25 else None
    desfecho, leitura = (ler_pela_regra(contra_bm25[melhor]) if melhor else
                         ("SÓ O CONTROLE", "nenhum modelo denso nesta execução: é a "
                          "calibração do pool, feita sem olhar número de modelo."))

    resultado = {
        "benchmark": "PB-Formula (DOC-11 §6.3)",
        "regra": "scripts/avaliar_pb_formula.py · docstring, escrita antes",
        "protocolo": {"estrato": a.estrato, "max_formas_em_comum": a.max_comum,
                      "itens": len(consultas), "pool": len(pool),
                      "equacoes_indexadas": len(textos), "semente": a.semente,
                      "max_tokens": a.max_tokens, "teto_do_pool": 1.0,
                      "escore_do_documento": "máximo sobre as equações dele",
                      **contagens},
        "sistemas": {nome: resumo(m) for nome, m in sistemas.items()},
        "recall_10_contra_bm25_bootstrap_por_item": contra_bm25,
        "melhor_denso": melhor,
        "desfecho": desfecho, "leitura": leitura,
        "custo_segundos": {n: m.get("segundos") for n, m in sistemas.items()},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print()
    print("=" * 78)
    print(f"  PB-Formula · estrato {a.estrato} · {len(consultas):,} itens · pool "
          f"{len(pool):,} · teto 1,0")
    print("=" * 78)
    print(f"  {'sistema':<34} {'r@1':>7} {'r@10':>7} {'r@50':>7} {'MRR':>7}")
    for nome, m in sistemas.items():
        r = resumo(m)
        print(f"  {nome:<34} {r['recall_1']:>7.4f} {r['recall_10']:>7.4f} "
              f"{r['recall_50']:>7.4f} {r['mrr']:>7.4f}")
    for nome, nd in contra_bm25.items():
        print(f"  recall@10 {nome} − bm25: {nd['diferenca']:+.4f} {nd['ic95']}")
    print()
    print(f"  DESFECHO ({melhor}): {desfecho}")
    print(f"  {leitura}")
    print("=" * 78)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
