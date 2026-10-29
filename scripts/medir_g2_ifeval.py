#!/usr/bin/env python3
"""O IFEval do G2.3 (DOC-12 §3.7.4) no llama-server do G2, com o envelope congelado.

    # DEV: o envelope em 20 instruções escritas fora do IFEval (gulosa)
    .venv\\Scripts\\python.exe scripts\\medir_g2_ifeval.py dev --modelo 1.7B
    # TESTE: k=16 + 1 gulosa no envelope, e a gulosa crua (relatada); exige o congelamento
    .venv\\Scripts\\python.exe scripts\\medir_g2_ifeval.py teste --modelo 1.7B --braco base

As gerações ficam em `g2_saidas/` (fora do git, retomáveis); os agregados, em
`data/processed/avaliacao/g2_ifeval_<fase>_<modelo>_<braco>.json`. O que decide é o
prompt-level strict médio das 16 amostras. A guarda de repetição do G2 vale aqui também;
a resposta cortada é corrigida como está (o laço quase nunca cumpre a instrução).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2, g2_geracao, g2_ifeval  # noqa: E402

AVAL = RAIZ / "data/processed/avaliacao"
SAIDAS = AVAL / "g2_saidas"
MODELOS = {"1.7B": ("Qwen3-1.7B-Base", RAIZ / "models/base/gguf/Qwen3-1.7B-Base-Q8_0.gguf"),
           "8B": ("Qwen3-8B-Base", RAIZ / "models/base/gguf/Qwen3-8B-Base-Q8_0.gguf")}
DEV20 = RAIZ / "configs/eval/g2_ifeval_dev20.jsonl"


def _gerar(srv, pedidos, pedido, saida, hc, t0) -> None:
    lote = g2_geracao.gerar_lote(srv.url, pedidos, g2_ifeval.PARADAS, pedido, saida, hc,
                                 fichas_max=g2_ifeval.FICHAS_MAX)
    for n, a in enumerate(lote, start=1):
        if n % 100 == 0:
            print(f"   {n}/{len(pedidos)} · {time.time() - t0:.0f} s · {a.fichas} fichas "
                  f"({a.parada})", flush=True)


def rodar(fase: str, modelo: str, braco: str, itens: list[dict]) -> dict:
    nome, gguf = MODELOS[modelo]
    kl = json.loads((AVAL / f"g2_kl_{modelo}.json").read_text(encoding="utf-8"))
    por_key = {str(x["key"]): x for x in itens}
    # (rótulo, pedido, envelope, k): o que decide, a gulosa no envelope e a gulosa crua
    corridas = ([("gulosa", g2_ifeval.amostragem(True), True, 1)] if fase == "dev" else
                [("amostras", g2_ifeval.amostragem(), True, g2_ifeval.K),
                 ("gulosa", g2_ifeval.amostragem(True), True, 1),
                 ("crua", g2_ifeval.amostragem(True), False, 1)])
    SAIDAS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hashes, notas = {}, {}
    with g2_geracao.ServidorG2(gguf, log=SAIDAS / f"ifeval_{fase}_{modelo}_{braco}.servidor.log") as srv:
        for rotulo, pedido, envelope, k in corridas:
            pedido = pedido | {"ifeval": f"envelope={envelope}"}
            hc = g2_geracao.hash_da_configuracao(srv, kl["gguf"]["q8_0"]["sha256"],
                                                 "RX 7600 (Vulkan)", pedido)
            hashes[rotulo] = hc
            saida = SAIDAS / f"ifeval_{fase}_{modelo}_{braco}_{rotulo}.jsonl"
            if fase == "teste" and rotulo == "amostras":
                g2.exigir_aceite_e_congelamento()
                continua = saida.exists() and hc in {d.get("hash_config")
                                                     for d in g2_geracao.ler_jsonl(saida)}
                g2.registrar_acesso(modelo, braco, "retomada" if continua else "medida", hc,
                                    nota="ifeval")
            pedidos = [("ifeval", key, j, g2_ifeval.consulta(x["prompt"], envelope))
                       for key, x in por_key.items() for j in range(k)]
            _gerar(srv, pedidos, pedido, saida, hc, t0)
            notas[rotulo] = saida
    resultado = {"o_que_e": f"G2.3, IFEval, fase {fase} (DOC-12 §3.7.4): agregados",
                 "modelo": nome, "braco": braco, "hashes": hashes,
                 "codigo_sha256": g2.codigo_do_corretor(RAIZ, extras=(
                     "src/phifm/eval/g2_ifeval.py", "scripts/medir_g2_ifeval.py")),
                 "segundos": round(time.time() - t0)}
    for rotulo, saida in notas.items():
        amostras = {}
        for a in g2_geracao.ler_jsonl(saida):
            if a.get("hash_config") == hashes[rotulo] and a["parada"] != "erro":
                amostras[(a["item_id"], a["j"])] = a
        corr = {c: g2_ifeval.corrigir(por_key[c[0]], a["texto"]) for c, a in amostras.items()}
        por_j = {}
        for (_key, j), x in corr.items():
            por_j.setdefault(j, []).append(x)
        stricts = [g2_ifeval.agregar(v)["prompt_strict"] for v in por_j.values()]
        resultado[rotulo] = {
            **g2_ifeval.agregar(list(corr.values())),
            "strict_medio_das_amostras": round(statistics.mean(stricts), 4) if stricts else None,
            "truncadas": round(sum(a["truncada"] for a in amostras.values()) / max(1, len(amostras)), 4),
            "fichas_media": round(statistics.mean(a["fichas"] for a in amostras.values()), 1)
            if amostras else None,
            "paradas": {p: sum(a["parada"] == p for a in amostras.values())
                        for p in sorted({a["parada"] for a in amostras.values()})}}
    destino = AVAL / f"g2_ifeval_{fase}_{modelo}_{braco}.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return resultado


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("fase", choices=("dev", "teste"))
    p.add_argument("--modelo", choices=tuple(MODELOS), default="1.7B")
    p.add_argument("--braco", default="base")
    a = p.parse_args()
    if a.fase == "dev":
        itens = g2_ifeval.carregar(DEV20)
    else:
        g2.exigir_aceite_e_congelamento()          # antes de LER o teste
        itens = g2_ifeval.carregar()
    print(f"IFEval {a.fase}: {len(itens)} prompts")
    r = rodar(a.fase, a.modelo, a.braco, itens)
    for rot in ("amostras", "gulosa", "crua"):
        if rot in r:
            print(f"{rot}: {r[rot]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
