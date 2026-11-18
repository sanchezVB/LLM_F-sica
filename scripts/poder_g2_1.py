#!/usr/bin/env python3
"""O poder do G2.1 recalculado com as bases MEDIDAS (DOC-12 §3.7.6: o poder foi simulado
"com premissas não medidas, que serão recalculadas com a base medida sem mudar a regra").

    .venv\\Scripts\\python.exe scripts\\poder_g2_1.py [--modelo 1.7B 8B] [--replicas 150]

O que faz, por base:
1. Recorrige as gerações de teste (CPU, o código congelado, como `corrigir_g2_teste.py`) e
   monta a nota de cada item, avg@k. As notas por item ficam FORA do git
   (`g2_saidas/teste_<modelo>_base_notas.json`, com o hash da configuração) e nunca são
   impressas: o lacre (§3.7.3) deixa o processo consumi-las, não alguém lê-las. A média por
   benchmark tem de bater com o agregado versionado.
2. Simula um ΦGen com ganho verdadeiro de Δ pontos em cada benchmark (o macro é a média,
   logo Δ_macro = Δ): a nota do item i é Binomial(k_b, p_i')/k_b, com p_i a nota medida da
   base, e a base fica como foi medida. O ganho é uniforme (p_i' = p_i + Δ/100) ou
   concentrado em 30% dos itens, sorteados (p_i' = p_i + Δ/0,3, teto 1): é assim que um
   modelo ganha de verdade, e isso alarga a variância do delta pareado. Aplica
   `g2.decidir_g2_1` como está (bootstrap pareado por agrupamento, 10.000 réplicas) e conta
   quantas vezes PASSA.
3. Quatro cenários: (a) a regra com ganho uniforme (o limite otimista); (b) a regra com o
   ganho concentrado (o que vale); (c) o ganho concentrado com marcas de memorização nova
   sorteadas ao acaso nas taxas do placebo 8B×1,7B da lista híbrida (DOC-12 item 31), que a
   condição 3 tira; (d) decodificação gulosa (k=1) com o ganho concentrado, para comparar
   com o erro-padrão declarado.

Sai só com agregados: o erro-padrão do macro, o poder por ganho e o ganho em que ele chega
a 80%. A guarda `__main__` é obrigatória: o corretor isolado sobe por spawn.
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

from phifm.eval import g2, g2_geracao, g2_itens  # noqa: E402

AVAL = RAIZ / "data/processed/avaliacao"
SAIDAS = AVAL / "g2_saidas"
GANHOS = (0.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
# as taxas de "memorização nova" do placebo 8B×1,7B com a lista híbrida (DOC-12 item 31)
MARCAS_PLACEBO = {"ugphysics": 0.056, "olympiadbench": 0.097, "phybench": 0.020, "scibench": 0.259}
FRACAO_CONCENTRADA = 0.3      # o ganho cai em 30% dos itens: o cenário de estresse da revisão
SEMENTE = g2.SEMENTE_G2 + 11


# ── as notas da base, por item ───────────────────────────────────────────────


def notas_por_item(itens, amostras: list[dict], vereditos: dict, k: dict[str, int]) -> dict:
    """{bench: [avg@k de cada item, na ordem de `itens`]}: a fração de amostras PASS, com as
    k_b amostras de cada item (falta é erro, nota 0)."""
    from phifm.eval.g2_corretor import Veredito

    passes: dict[tuple, int] = {}
    for a in amostras:
        chave = (a["bench"], a["item_id"])
        passes[chave] = passes.get(chave, 0) + (vereditos[(a["bench"], a["item_id"], a["j"])] == Veredito.PASS)
    notas: dict[str, list[float]] = {}
    for i in itens:
        notas.setdefault(i.bench, []).append(passes.get((i.bench, i.id), 0) / k[i.bench])
    return notas


def notas_da_base(modelo: str, itens, k: dict[str, int]) -> tuple[dict, str, Path]:
    """As notas por item da base, recorrigidas aqui ou lidas do arquivo fora do git."""
    import medir_g2 as m
    from phifm.eval.g2_correcao import CorretorIsolado
    from phifm.eval.g2_corretor import Veredito

    saida = SAIDAS / f"teste_{modelo}_base_veredito.jsonl"
    hashes = {d["hash_config"] for d in g2_geracao.ler_jsonl(saida)}
    if len(hashes) != 1:
        raise SystemExit(f"{saida.name} tem {len(hashes)} hashes de configuração; esperado 1")
    hc = hashes.pop()
    codigo = g2.codigo_do_corretor(RAIZ)
    cache = SAIDAS / f"teste_{modelo}_base_notas.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        if d["hash_config"] == hc and d["codigo_sha256"] == codigo:
            return d["notas"], hc, cache
    por_id = {(i.bench, i.id): i for i in itens}
    amostras = m._amostras(saida, hc, por_id)
    esperadas = sum(k[i.bench] for i in itens)
    if len(amostras) != esperadas or any(a["parada"] == "erro" for a in amostras):
        raise SystemExit(f"{modelo}: {len(amostras)} amostras de {esperadas}, ou linha 'erro'")
    t0 = time.time()
    vereditos = {}
    with CorretorIsolado() as cor:
        for a in amostras:
            if a["truncada"]:
                v = Veredito.TRUNCADO
            else:
                v = cor.corrigir(por_id[(a["bench"], a["item_id"])], a["texto"])
            vereditos[(a["bench"], a["item_id"], a["j"])] = v
    notas = notas_por_item(itens, amostras, vereditos, k)
    cache.write_text(json.dumps({"o_que_e": "notas por item (avg@k) da base no teste do G2.1; "
                                             "fora do git (lacre), só o processo as consome",
                                 "modelo": modelo, "hash_config": hc, "codigo_sha256": codigo,
                                 "segundos": round(time.time() - t0), "notas": notas}),
                     encoding="utf-8")
    return notas, hc, cache


def conferir_com_o_agregado(modelo: str, notas: dict) -> dict[str, float]:
    """A média das notas por benchmark tem de ser o `acerto` do agregado versionado."""
    agg = json.loads((AVAL / f"g2_teste_{modelo}_base_veredito.json").read_text(encoding="utf-8"))
    medias = {}
    for b, ns in notas.items():
        medias[b] = round(float(np.mean(ns)), 4)
        if abs(medias[b] - agg["por_benchmark"][b]["acerto"]) > 1e-4:
            raise SystemExit(f"{modelo} {b}: as notas dão {medias[b]}, o agregado diz "
                             f"{agg['por_benchmark'][b]['acerto']}")
    return medias


def grupos_do_teste(itens) -> dict[str, list[int] | None]:
    """Os agrupamentos do bootstrap (§3.7.6): os grupos de 13-grama no UG e os do
    OlympiadBench, pela partição; itens no PHYBench e no SciBench."""
    part = json.loads((AVAL / "g2_particao.json").read_text(encoding="utf-8"))
    grupos: dict[str, list[int] | None] = {}
    for b in g2.BENCH_G2_1:
        ids = [i.id for i in itens if i.bench == b]
        if b in part:
            rotulos = {g: n for n, g in enumerate(sorted({part[b]["grupo"][i] for i in ids}))}
            grupos[b] = [rotulos[part[b]["grupo"][i]] for i in ids]
        else:
            grupos[b] = None
    return grupos


# ── a simulação ──────────────────────────────────────────────────────────────


def simular_phigen(rng: np.random.Generator, notas_base: dict, k: dict[str, int],
                   ganho: float, fracao: float | None = None) -> dict[str, np.ndarray]:
    """As notas de um ΦGen com ganho verdadeiro de `ganho` pontos em cada benchmark: uniforme
    nos itens ou, com `fracao`, concentrado numa fração sorteada deles (p_i + Δ/fração, com
    teto 1), que é como um modelo ganha de verdade e alarga a variância do delta."""
    saida = {}
    for b, p in notas_base.items():
        p = np.asarray(p, dtype=float)
        if fracao is None:
            alvo = p + ganho / 100
        else:
            escolhidos = rng.random(len(p)) < fracao
            alvo = p + np.where(escolhidos, ganho / 100 / fracao, 0.0)
        saida[b] = rng.binomial(k[b], np.clip(alvo, 0, 1)) / k[b]
    return saida


def marcas(rng: np.random.Generator, notas_base: dict, taxas: dict[str, float] | None) -> dict:
    return {b: (rng.random(len(p)) < taxas[b]) if taxas else np.zeros(len(p), dtype=bool)
            for b, p in notas_base.items()}


def poder(notas_base: dict, grupos: dict, k: dict[str, int], ganhos=GANHOS, replicas: int = 150,
          n_bootstrap: int = g2.N_BOOTSTRAP, taxas_de_marcas: dict | None = None,
          fracao: float | None = None, semente: int = SEMENTE, aviso=None) -> dict:
    """Por ganho: a fração de réplicas em que o G2.1 PASSA, o erro-padrão do macro (a meia
    largura do IC 95% / 1,96), o Δ_macro médio que saiu, e o ganho em que o poder chega a
    80%, por interpolação."""
    rng = np.random.default_rng(semente)
    por_ganho = {}
    for ganho in ganhos:
        t0 = time.time()
        passa, ep, delta, c1, c2, c3 = [], [], [], [], [], []
        for _ in range(replicas):
            phigen = simular_phigen(rng, notas_base, k, ganho, fracao)
            d = g2.decidir_g2_1(phigen, {b: np.asarray(p) for b, p in notas_base.items()}, grupos,
                                marcas(rng, notas_base, taxas_de_marcas), n_bootstrap=n_bootstrap,
                                semente=int(rng.integers(2**31)))
            passa.append(d["passa"])
            lo, hi = d["todos"]["ic_macro"]
            ep.append((hi - lo) / 2 / 1.96)
            delta.append(d["todos"]["delta_macro"])
            c1.append(d["condicoes"]["ic_macro_acima_do_limiar"])
            c2.append(d["condicoes"]["nenhum_benchmark_piora"])
            c3.append(d["condicoes"]["sem_memorizacao_nova"])
        por_ganho[str(ganho)] = {"poder": round(float(np.mean(passa)), 4),
                                 "erro_padrao_macro": round(float(np.mean(ep)), 4),
                                 "delta_macro_medio": round(float(np.mean(delta)), 4),
                                 "condicao_1": round(float(np.mean(c1)), 4),
                                 "condicao_2": round(float(np.mean(c2)), 4),
                                 "condicao_3": round(float(np.mean(c3)), 4)}
        if aviso:
            aviso(f"ganho +{ganho:g}: poder {por_ganho[str(ganho)]['poder']:.2f}, EP "
                  f"{por_ganho[str(ganho)]['erro_padrao_macro']:.2f} · {time.time() - t0:.0f} s")
    return {"replicas": replicas, "n_bootstrap": n_bootstrap, "por_ganho": por_ganho,
            **pontos_de_80(por_ganho),
            "erro_padrao_macro": round(float(np.mean([v["erro_padrao_macro"]
                                                      for v in por_ganho.values()])), 4)}


def _onde_cruza(xs: list[float], ys: list[float], alvo: float = 0.8) -> float | None:
    for a, b, ya, yb in zip(xs[:-1], xs[1:], ys[:-1], ys[1:], strict=True):
        if ya < alvo <= yb:
            return round(a + (alvo - ya) / (yb - ya) * (b - a), 2)
    return None


def pontos_de_80(por_ganho: dict) -> dict:
    """Onde o poder chega a 80%, por interpolação: no ganho NOMINAL e no EFETIVO (o Δ_macro
    médio que saiu). No ganho concentrado, o teto de nota 1 corta parte do ganho nos itens
    fáceis, e o efetivo fica abaixo do nominal: ele é o eixo honesto."""
    gs = sorted(por_ganho, key=float)
    ys = [por_ganho[g]["poder"] for g in gs]
    efetivos = [por_ganho[g].get("delta_macro_medio") for g in gs]
    return {"ganho_para_poder_80": _onde_cruza([float(g) for g in gs], ys),
            "ganho_efetivo_para_poder_80": (_onde_cruza(efetivos, ys)
                                            if None not in efetivos else None)}


def main() -> int:
    import medir_g2 as m

    p = argparse.ArgumentParser()
    p.add_argument("--modelo", nargs="+", choices=tuple(m.MODELOS), default=["1.7B", "8B"])
    p.add_argument("--replicas", type=int, default=150)
    p.add_argument("--bootstrap", type=int, default=g2.N_BOOTSTRAP)
    a = p.parse_args()
    g2.exigir_aceite_e_congelamento()
    fora = m.excluidos("teste")
    itens = [i for b in g2.BENCH_G2_1 for i in g2_itens.carregar(b, "teste")
             if i.id not in fora.get(b, set())]
    grupos = grupos_do_teste(itens)
    k = m.K_TESTE
    t0 = time.time()
    resultado = {"o_que_e": "G2.1: o poder da regra (DOC-12 §3.7.6) recalculado com as bases "
                            "medidas, sem mudar a regra; ΦGen simulado com ganho uniforme por "
                            "benchmark sobre a nota medida de cada item da base",
                 "premissas": {"ganho": "Δ_b = Δ em cada benchmark (o macro é a média); uniforme nos "
                                        "itens, ou concentrado numa fração sorteada deles (p_i + Δ/fração, teto 1)",
                               "fracao_concentrada": FRACAO_CONCENTRADA,
                               "phigen": "Binomial(k_b, p_i')/k_b, p_i = nota medida da base",
                               "base": "fixa, como medida", "k": k,
                               "agrupamentos": {b: (len(set(g)) if g else "itens") for b, g in grupos.items()},
                               "itens": {b: sum(i.bench == b for i in itens) for b in g2.BENCH_G2_1},
                               "marcas_placebo": MARCAS_PLACEBO, "semente": SEMENTE},
                 "por_base": {}}

    def aviso(txt):
        print("   " + txt, flush=True)

    for modelo in a.modelo:
        print(f"{modelo}: corrigindo o teste (ou lendo as notas guardadas)", flush=True)
        notas, hc, cache = notas_da_base(modelo, itens, k)
        medias = conferir_com_o_agregado(modelo, notas)
        print(f"{modelo}: notas conferidas com o agregado: {medias}", flush=True)
        por = {"hash_config": hc, "acerto_por_benchmark": medias,
               "sha256_notas": hashlib.sha256(cache.read_bytes()).hexdigest(), "cenarios": {}}
        for nome, kk, taxas, fracao in (
                ("regra_ganho_uniforme", k, None, None),
                ("regra_ganho_concentrado_30", k, None, FRACAO_CONCENTRADA),
                ("regra_com_marcas_do_placebo", k, MARCAS_PLACEBO, FRACAO_CONCENTRADA),
                ("gulosa_k1", dict.fromkeys(k, 1), None, FRACAO_CONCENTRADA)):
            print(f"{modelo} · {nome}:", flush=True)
            por["cenarios"][nome] = poder(notas, grupos, kk, replicas=a.replicas,
                                          n_bootstrap=a.bootstrap, taxas_de_marcas=taxas,
                                          fracao=fracao, aviso=aviso)
        resultado["por_base"][modelo] = por
        resultado["segundos"] = round(time.time() - t0)
        (AVAL / "g2_poder_g2_1.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1) + "\n",
                                                 encoding="utf-8")
    for modelo, por in resultado["por_base"].items():
        for nome, c in por["cenarios"].items():
            print(f"{modelo} {nome}: EP {c['erro_padrao_macro']:.2f} · 80% em +{c['ganho_para_poder_80']} · "
                  f"a +10: {c['por_ganho']['10.0']['poder']:.3f} · a +12: {c['por_ganho']['12.0']['poder']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
