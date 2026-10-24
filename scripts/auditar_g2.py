#!/usr/bin/env python3
"""A auditoria cega do corretor do G2 (DOC-12 §3.7.3, item 3): o Claude julga 200 gerações
do DEV sem ver o veredito do corretor, e mede-se a concordância (alvo: ≥95% e κ ≥0,8).

    # 1. sorteia, corrige com o corretor ATUAL e grava ids, veredictos e o hash deles
    .venv\\Scripts\\python.exe scripts\\auditar_g2.py sortear
    #    (commit de g2_auditoria.json ANTES de julgar)
    # 2. o julgamento cego lê os lotes (fora do git, sem veredicto) e grava os julgamentos
    # 3. apura concordância e κ
    .venv\\Scripts\\python.exe scripts\\auditar_g2.py apurar <julgamentos.json>

O sorteio é estratificado por benchmark, entre as gerações de dev corrigidas (sem as
truncadas, sem as `erro` e sem os itens que o autoteste exclui). Os veredictos do corretor
são refeitos com o código atual, o que vai ser congelado; os das iterações valiam para o
código de então. No git ficam só ids, veredictos e hashes; o conteúdo dos lotes (enunciado,
gabarito, geração) fica em `g2_saidas/auditoria/`.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2, g2_itens, g2_prompt  # noqa: E402
from phifm.eval.g2_correcao import CorretorIsolado  # noqa: E402

AVAL = RAIZ / "data/processed/avaliacao"
SAIDAS = AVAL / "g2_saidas"
LOTES = SAIDAS / "auditoria"
DESTINO = AVAL / "g2_auditoria.json"
ROTULOS = ("it1", "it2")
COTAS = {"ugphysics": 130, "olympiadbench": 50, "scibench": 20}
POR_LOTE = 10
CLASSES = ("pass", "fail", "sem_resposta")


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _criterio(item: g2_itens.Item, k: int) -> str:
    """O critério do protocolo para a resposta k, em palavras, para o juiz."""
    tipo = g2_itens.tipo_efetivo(item, k)
    tol = item.tolerancias[k] if item.tolerancias else None
    if item.bench == "scibench":
        return "número; isclose com rel_tol = 5% (simétrico), na unidade mostrada"
    if item.bench == "olympiadbench" and tipo == "Numerical":
        return (f"número; tolerância ABSOLUTA ±{tol}" if tol is not None
                else "número; tolerância relativa 1% do gabarito")
    return {"NV": "número; tolerância relativa 1% do gabarito",
            "EX": "expressão; equivalência matemática com variáveis positivas",
            "Expression": "expressão; equivalência matemática com variáveis positivas",
            "EQ": "equação; equivalente se (L−R) da resposta ÷ (L−R) do gabarito for constante "
                  "não nula; gabarito `G = expr` com G grandeza: vale o VALOR de G",
            "Equation": "equação; idem EQ",
            "IN": "intervalo; mesmos colchetes e extremos iguais",
            "Interval": "intervalo; mesmos colchetes e extremos iguais",
            "MC": "letras; o mesmo conjunto de alternativas",
            "TF": "verdadeiro/falso; o mesmo valor"}.get(tipo, f"tipo {tipo}")


def sortear() -> int:
    fora = {k.split("/")[0]: set(v["ids_excluidos"]) for k, v in json.loads(
        (AVAL / "g2_autoteste.json").read_text(encoding="utf-8"))["conjuntos"].items()
        if k.endswith("/dev")}
    por_id = {(i.bench, i.id): i for b in COTAS for i in g2_itens.carregar(b, "dev")}
    candidatas: dict[str, list[dict]] = collections.defaultdict(list)
    for rotulo in ROTULOS:
        hc = json.loads((AVAL / f"g2_dev_1.7B_base_{rotulo}.json")
                        .read_text(encoding="utf-8"))["hash_config"]
        vistas = {}
        for x in (SAIDAS / f"dev_1.7B_base_{rotulo}.jsonl").read_text(encoding="utf-8").splitlines():
            a = json.loads(x) if x.strip() else None
            if (a and a["hash_config"] == hc and a["parada"] != "erro" and not a["truncada"]
                    and (a["bench"], a["item_id"]) in por_id
                    and a["item_id"] not in fora.get(a["bench"], set())):
                vistas[(a["bench"], a["item_id"], a["j"])] = a | {"rotulo": rotulo}
        for a in vistas.values():
            candidatas[a["bench"]].append(a)
    rng = random.Random(g2.SEMENTE_G2 + 300)
    amostra = []
    for bench, n in COTAS.items():
        lista = sorted(candidatas[bench], key=lambda a: (a["rotulo"], a["item_id"], a["j"]))
        amostra += rng.sample(lista, min(n, len(lista)))
    rng.shuffle(amostra)
    chaves = [f"{a['rotulo']}|{a['bench']}|{a['item_id']}|{a['j']}" for a in amostra]
    vereditos = {}
    with CorretorIsolado() as cor:
        for chave, a in zip(chaves, amostra, strict=True):
            vereditos[chave] = str(cor.corrigir(por_id[(a["bench"], a["item_id"])], a["texto"]))
    LOTES.mkdir(parents=True, exist_ok=True)
    for n_lote in range(0, len(amostra), POR_LOTE):
        casos = []
        for chave, a in zip(chaves[n_lote:n_lote + POR_LOTE], amostra[n_lote:n_lote + POR_LOTE],
                            strict=True):
            it = por_id[(a["bench"], a["item_id"])]
            casos.append({"chave": chave, "bench": it.bench, "enunciado": it.enunciado,
                          "linha_de_resposta": g2_prompt.linha_de_resposta(it),
                          "gabaritos": list(it.gabaritos),
                          "criterios": [_criterio(it, k) for k in range(len(it.gabaritos))],
                          "unidade": it.unidade, "geracao": a["texto"]})
        (LOTES / f"lote_{n_lote // POR_LOTE:02d}.json").write_text(
            json.dumps(casos, ensure_ascii=False, indent=1), encoding="utf-8")
    r = {"o_que_e": "auditoria cega do corretor do G2 (DOC-12 §3.7.3, item 3): a amostra, os "
                    "veredictos do corretor e o hash deles, gravados ANTES do julgamento",
         "codigo_sha256": g2.codigo_do_corretor(RAIZ),
         "cotas": COTAS, "n": len(chaves), "amostra": chaves,
         "vereditos_do_corretor": vereditos, "sha256_vereditos_do_corretor": _hash(vereditos),
         "contagem_do_corretor": dict(collections.Counter(vereditos.values()))}
    DESTINO.write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(chaves)} gerações · {len(range(0, len(amostra), POR_LOTE))} lotes em {LOTES}")
    print(f"corretor: {r['contagem_do_corretor']} · hash {r['sha256_vereditos_do_corretor'][:16]}")
    return 0


def _kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def apurar(arquivo: Path) -> int:
    r = json.loads(DESTINO.read_text(encoding="utf-8"))
    if _hash(r["vereditos_do_corretor"]) != r["sha256_vereditos_do_corretor"]:
        raise SystemExit("os veredictos do corretor mudaram depois do sorteio")
    julg = json.loads(arquivo.read_text(encoding="utf-8"))
    chaves = [k for k in r["amostra"] if k in julg]
    if len(chaves) != len(r["amostra"]):
        raise SystemExit(f"faltam julgamentos: {len(r['amostra']) - len(chaves)}")
    cor = [r["vereditos_do_corretor"][k] for k in chaves]
    aud = [julg[k] for k in chaves]
    # binário: o que conta na nota é PASS contra o resto (INCONCLUSIVE e sem resposta valem 0)
    cb, ab = ["pass" if v == "pass" else "nao" for v in cor], ["pass" if v == "pass" else "nao"
                                                                 for v in aud]
    concord = sum(x == y for x, y in zip(cb, ab, strict=True)) / len(chaves)
    kappa = _kappa(cb, ab)
    desacordos = [{"chave": k, "corretor": c, "auditoria": a}
                  for k, c, a in zip(chaves, cor, aud, strict=True)
                  if (c == "pass") != (a == "pass")]
    r["auditoria"] = {
        "sha256_julgamentos": _hash({k: julg[k] for k in chaves}),
        "julgamentos": {k: julg[k] for k in chaves},
        "contagem_da_auditoria": dict(collections.Counter(aud)),
        "concordancia_pass": round(concord, 4), "kappa_pass": round(kappa, 4),
        "concordancia_3_classes": round(sum(x == y for x, y in zip(cor, aud, strict=True))
                                        / len(chaves), 4),
        "desacordos_pass": desacordos,
        "alvo_cumprido": concord >= 0.95 and kappa >= 0.8,
    }
    DESTINO.write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"concordância {concord:.1%} · κ {kappa:.3f} · desacordos {len(desacordos)} · "
          f"alvo {'cumprido' if r['auditoria']['alvo_cumprido'] else 'NÃO cumprido'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("passo", choices=("sortear", "apurar"))
    p.add_argument("julgamentos", nargs="?", type=Path)
    a = p.parse_args()
    return sortear() if a.passo == "sortear" else apurar(a.julgamentos)


if __name__ == "__main__":
    sys.exit(main())
