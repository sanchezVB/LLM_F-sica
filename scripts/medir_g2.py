#!/usr/bin/env python3
"""A medida de física do G2 (DOC-12 §3.7): gera, corrige e agrega.

    # DEV (permitido agora): itens de dev, k=1, para afinar extrator e corretor
    .venv\\Scripts\\python.exe scripts\\medir_g2.py dev --modelo 1.7B --iteracao 1 [--ug 300]

    # TESTE: recusa sem o aceite da regra, sem o congelamento e sem o livro de acessos
    .venv\\Scripts\\python.exe scripts\\medir_g2.py teste --modelo 1.7B --braco base

Na fase de dev, cada iteração usa ~300 itens sorteados do UG-dev (com `--ug 0`, o UG-dev
inteiro), os 115 do OlympiadBench-zh e os `_sol` do SciBench que não viraram exemplo. Itens
que viraram exemplo não são medidos.

Saídas:
- as gerações, em `data/processed/avaliacao/g2_saidas/` (fora do git);
- os agregados, em `data/processed/avaliacao/g2_<fase>_<modelo>_<rótulo>.json` (versionados):
  PASS, sem resposta, truncado, INCONCLUSIVE e ERROR por benchmark, e as fichas geradas.

O teste só lê agregados até o veredito (§3.7.3, lacre). O GPQA não existe aqui (decisão do
dono, 2026-10-19).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2, g2_geracao, g2_itens, g2_prompt  # noqa: E402
from phifm.eval.g2_correcao import CorretorIsolado  # noqa: E402
from phifm.eval.g2_corretor import Veredito  # noqa: E402

AVAL = RAIZ / "data/processed/avaliacao"
SAIDAS = AVAL / "g2_saidas"
MODELOS = {"1.7B": ("Qwen3-1.7B-Base", RAIZ / "models/base/gguf/Qwen3-1.7B-Base-Q8_0.gguf"),
           "8B": ("Qwen3-8B-Base", RAIZ / "models/base/gguf/Qwen3-8B-Base-Q8_0.gguf")}
K_TESTE = {"ugphysics": 3, "olympiadbench": 8, "phybench": 16, "scibench": 8}
UG_DEV_POR_ITERACAO = 300


def _exemplos() -> dict[str, list[g2_itens.Item]]:
    escolha = json.loads((AVAL / "g2_exemplos.json").read_text(encoding="utf-8"))["exemplos"]
    por_id = {i.id: i for i in g2_itens.carregar("ugphysics", "dev")
              + g2_itens.carregar("scibench", "dev")}
    return {b: [por_id[i] for i in e["ids"]] for b, e in escolha.items()}


def _exemplos_do(item: g2_itens.Item, exemplos: dict[str, list[g2_itens.Item]]
                 ) -> list[g2_itens.Item]:
    if item.bench != "scibench":
        return exemplos[item.bench]
    return [e for e in exemplos["scibench"] if e.extra["livro"] == item.extra["livro"]]


def itens_de_dev(n_ug: int, iteracao: int, exemplos: dict[str, list[g2_itens.Item]]
                 ) -> list[g2_itens.Item]:
    usados = {e.id for es in exemplos.values() for e in es}
    ug = sorted((i for i in g2_itens.carregar("ugphysics", "dev") if i.id not in usados),
                key=lambda i: i.id)
    if n_ug:
        ug = random.Random(g2.SEMENTE_G2 + 100 + iteracao).sample(ug, min(n_ug, len(ug)))
    oly = g2_itens.carregar("olympiadbench", "dev")
    sci = [i for i in g2_itens.carregar("scibench", "dev") if i.id not in usados]
    return ug + oly + sci


def _agregar(itens: list[g2_itens.Item], amostras: list[dict], vereditos: dict) -> dict:
    por_bench: dict[str, dict] = {}
    for bench in sorted({i.bench for i in itens}):
        linhas = [a for a in amostras if a["bench"] == bench]
        cont = collections.Counter(vereditos[(a["bench"], a["item_id"], a["j"])] for a in linhas)
        n = len(linhas)
        fichas = [a["fichas"] for a in linhas]
        por_bench[bench] = {
            "itens": len({a["item_id"] for a in linhas}), "amostras": n,
            "acerto": round(cont[Veredito.PASS] / n, 4) if n else None,
            "fracoes": {str(v): round(c / n, 4) for v, c in sorted(cont.items())},
            "fichas_media": round(statistics.mean(fichas), 1) if fichas else None,
            "fichas_p95": (sorted(fichas)[int(0.95 * (len(fichas) - 1))] if fichas else None),
        }
    return por_bench


def rodar(fase: str, modelo: str, rotulo: str, itens: list[g2_itens.Item],
          exemplos: dict[str, list[g2_itens.Item]], k: dict[str, int], braco: str) -> dict:
    nome, gguf = MODELOS[modelo]
    kl = json.loads((AVAL / f"g2_kl_{modelo}.json").read_text(encoding="utf-8"))
    pedido = g2_geracao.amostragem()
    saida = SAIDAS / f"{fase}_{modelo}_{braco}_{rotulo}.jsonl"
    pedidos = [(i.bench, i.id, j, g2_prompt.montar_prompt(i, _exemplos_do(i, exemplos)))
               for i in itens for j in range(k[i.bench])]
    log = SAIDAS / f"{fase}_{modelo}_{braco}_{rotulo}.servidor.log"
    SAIDAS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with g2_geracao.ServidorG2(gguf, log=log) as srv:
        hc = g2_geracao.hash_da_configuracao(srv, kl["gguf"]["q8_0"]["sha256"],
                                             "RX 7600 (Vulkan)", pedido)
        if fase == "teste":
            g2.exigir_aceite_e_congelamento()
            # saída já começada com o MESMO hash: é a mesma medida continuando
            continua = hc in {json.loads(x).get("hash_config")
                              for x in (saida.read_text(encoding="utf-8").splitlines()
                                        if saida.exists() else []) if x.strip()}
            g2.registrar_acesso(modelo, braco, "retomada" if continua else "medida", hc)
        lote = g2_geracao.gerar_lote(srv.url, pedidos, g2_prompt.PARADAS, pedido, saida, hc)
        for feitas, a in enumerate(lote, start=1):
            if feitas % 25 == 0:
                print(f"   {feitas}/{len(pedidos)} amostras · {time.time() - t0:.0f} s · "
                      f"última: {a.bench} {a.fichas} fichas ({a.parada})", flush=True)
    por_id = {(i.bench, i.id): i for i in itens}
    amostras = [json.loads(x) for x in saida.read_text(encoding="utf-8").splitlines() if x.strip()]
    amostras = [a for a in amostras if a["hash_config"] == hc and (a["bench"], a["item_id"]) in por_id]
    # A correção roda DEPOIS da geração, com o servidor já parado (a máquina sem outra carga
    # pesada), num processo que pode ser morto (g2_correcao).
    vereditos = {}
    with CorretorIsolado() as cor:
        for a in amostras:
            if a["parada"] == "erro":
                v = Veredito.ERROR
            elif a["truncada"]:
                v = Veredito.TRUNCADO
            else:
                v = cor.corrigir(por_id[(a["bench"], a["item_id"])], a["texto"])
            vereditos[(a["bench"], a["item_id"], a["j"])] = v
        mortes = cor.mortes
    resultado = {
        "o_que_e": f"G2, fase {fase}: agregados por benchmark (DOC-12 §3.7); as gerações ficam "
                   "fora do git",
        "modelo": nome, "braco": braco, "rotulo": rotulo, "hash_config": hc,
        "codigo_sha256": {n: g2.sha256_arquivo(RAIZ / f"src/phifm/eval/{n}.py")
                          for n in ("g2_corretor", "g2_itens", "g2_prompt", "g2_geracao",
                                    "g2_correcao")},
        "correcoes_mortas_no_teto": mortes,
        "k": k, "segundos": round(time.time() - t0),
        "por_benchmark": _agregar(itens, amostras, vereditos),
    }
    if fase == "dev":       # só no dev: no teste, nada por item até o veredito (lacre)
        resultado["vereditos_por_amostra"] = {f"{b}|{i}|{j}": str(v)
                                              for (b, i, j), v in sorted(vereditos.items())}
    destino = AVAL / f"g2_{fase}_{modelo}_{braco}_{rotulo}.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
    return resultado


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("fase", choices=("dev", "teste"))
    p.add_argument("--modelo", choices=tuple(MODELOS), default="1.7B")
    p.add_argument("--braco", default="base")
    p.add_argument("--iteracao", type=int, default=1)
    p.add_argument("--ug", type=int, default=UG_DEV_POR_ITERACAO,
                   help="itens do UG-dev por iteração (0 = todos)")
    a = p.parse_args()
    exemplos = _exemplos()
    if a.fase == "dev":
        itens = itens_de_dev(a.ug, a.iteracao, exemplos)
        k = dict.fromkeys(("ugphysics", "olympiadbench", "scibench"), 1)
        rotulo = f"it{a.iteracao}" + ("_ugcompleto" if a.ug == 0 else "")
    else:
        g2.exigir_aceite_e_congelamento()          # antes de LER qualquer coisa do teste
        itens = [i for b in g2.BENCH_G2_1 for i in g2_itens.carregar(b, "teste")]
        k, rotulo = K_TESTE, "veredito"
    print(f"{a.fase}: {len(itens)} itens, k={k}")
    r = rodar(a.fase, a.modelo, rotulo, itens, exemplos, k, a.braco)
    for b, x in r["por_benchmark"].items():
        print(f"{b}: acerto {x['acerto']} · {x['fracoes']} · fichas média {x['fichas_media']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
