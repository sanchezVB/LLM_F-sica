#!/usr/bin/env python3
"""O MMLU do G2.3 (DOC-12 §3.7.4) no llama-server do G2: 5-shot, letra pelo logprob.

    # DEV: 50 itens da validação (sem física), para conferir com o transformers na CPU
    .venv\\Scripts\\python.exe scripts\\medir_g2_mmlu.py dev --modelo 1.7B
    # TESTE: exige o congelamento e registra o acesso no livro (nota "mmlu")
    .venv\\Scripts\\python.exe scripts\\medir_g2_mmlu.py teste --modelo 1.7B --braco base

Saídas: as previsões por item em `g2_saidas/` (fora do git, retomáveis) e os agregados em
`data/processed/avaliacao/g2_mmlu_<fase>_<modelo>_<braco>.json`. No teste, só agregados; a
guarda do instrumento (MMLU-57 da base a ±3 de 62,63 na 1,7B) é conferida e gravada.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2, g2_geracao, g2_mmlu  # noqa: E402

AVAL = RAIZ / "data/processed/avaliacao"
SAIDAS = AVAL / "g2_saidas"
MODELOS = {"1.7B": ("Qwen3-1.7B-Base", RAIZ / "models/base/gguf/Qwen3-1.7B-Base-Q8_0.gguf"),
           "8B": ("Qwen3-8B-Base", RAIZ / "models/base/gguf/Qwen3-8B-Base-Q8_0.gguf")}
GUARDA = {"1.7B": (62.63, 3.0), "8B": (76.89, 3.0)}     # MMLU-57 da base, ± pontos
N_DEV = 50


def pedido_mmlu() -> dict:
    return {**g2_geracao.amostragem(temperatura=0.0, top_p=1.0), "n_predict": 1,
            "n_probs": g2_mmlu.N_PROBS, "formato": "mmlu 5-shot classico, letra por logprob"}


def prever(url: str, prompt: str, pedido: dict) -> list[dict]:
    corpo = {k: v for k, v in pedido.items() if k != "formato"}
    corpo |= {"prompt": prompt, "cache_prompt": True, "stream": False}
    req = urllib.request.Request(url + "/completion", data=json.dumps(corpo).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    probs = d.get("completion_probabilities") or []
    return probs[0].get("top_logprobs", []) if probs else []


def rodar(fase: str, modelo: str, braco: str, itens: list[dict]) -> dict:
    nome, gguf = MODELOS[modelo]
    kl = json.loads((AVAL / f"g2_kl_{modelo}.json").read_text(encoding="utf-8"))
    exemplos = g2_mmlu.exemplos_por_materia(g2_mmlu.carregar("dev"))
    pedido = pedido_mmlu()
    saida = SAIDAS / f"mmlu_{fase}_{modelo}_{braco}.jsonl"
    SAIDAS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with g2_geracao.ServidorG2(gguf, log=SAIDAS / f"mmlu_{fase}_{modelo}_{braco}.servidor.log") as srv:
        hc = g2_geracao.hash_da_configuracao(srv, kl["gguf"]["q8_0"]["sha256"],
                                             "RX 7600 (Vulkan)", pedido)
        feitos = {d["id"]: d for d in g2_geracao.ler_jsonl(saida)
                  if d.get("hash_config") == hc} if saida.exists() else {}
        if fase == "teste":
            g2.exigir_aceite_e_congelamento()
            g2.registrar_acesso(modelo, braco, "retomada" if feitos else "medida", hc,
                                nota="mmlu")
        fila = [x for x in itens if x["id"] not in feitos]
        trava = threading.Lock()

        def um(x: dict) -> dict:
            try:
                top = prever(srv.url, g2_mmlu.montar_prompt(x, exemplos[x["subject"]]), pedido)
                letra, lp = g2_mmlu.letra_prevista(top)
            except OSError:
                letra, lp = None, {}
            r = {"id": x["id"], "letra": letra, "logprobs": lp, "hash_config": hc}
            with trava, saida.open("a", encoding="utf-8") as f:
                f.write(g2_geracao.linha_jsonl(r) + "\n")
            return r

        with ThreadPoolExecutor(4) as ex:
            for n, fut in enumerate(as_completed([ex.submit(um, x) for x in fila]), start=1):
                fut.result()
                if n % 500 == 0:
                    print(f"   {n}/{len(fila)} itens · {time.time() - t0:.0f} s", flush=True)
    previstas = {d["id"]: d["letra"] for d in g2_geracao.ler_jsonl(saida)
                 if d.get("hash_config") == hc}
    resultado = {"o_que_e": f"G2.3, MMLU, fase {fase} (DOC-12 §3.7.4): agregados",
                 "modelo": nome, "braco": braco, "hash_config": hc,
                 "codigo_sha256": g2.codigo_do_corretor(RAIZ, extras=(
                     "src/phifm/eval/g2_mmlu.py", "scripts/medir_g2_mmlu.py")),
                 "segundos": round(time.time() - t0),
                 **g2_mmlu.agregar(itens, previstas)}
    if fase == "teste":
        ref, folga = GUARDA[modelo]
        a57 = 100 * resultado["mmlu_57"]["acerto"]
        resultado["guarda_do_instrumento"] = {"referencia": ref, "folga": folga,
                                              "medido": round(a57, 2),
                                              "ok": abs(a57 - ref) <= folga}
        resultado.pop("por_materia")              # lacre: só os agregados que decidem
    else:
        resultado["previstas"] = previstas        # dev: para conferir com o transformers
    destino = AVAL / f"g2_mmlu_{fase}_{modelo}_{braco}.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return resultado


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("fase", choices=("dev", "teste"))
    p.add_argument("--modelo", choices=tuple(MODELOS), default="1.7B")
    p.add_argument("--braco", default="base")
    a = p.parse_args()
    if a.fase == "dev":
        val = [x for x in g2_mmlu.carregar("validation") if x["subject"] not in g2_mmlu.FISICA]
        itens = sorted(random.Random(g2.SEMENTE_G2 + 400).sample(val, N_DEV), key=lambda x: x["id"])
    else:
        g2.exigir_aceite_e_congelamento()          # antes de LER o teste
        itens = g2_mmlu.carregar("test")
    print(f"MMLU {a.fase}: {len(itens)} itens")
    r = rodar(a.fase, a.modelo, a.braco, itens)
    print(f"sem física {r['sem_fisica']} · 57 {r['mmlu_57']} · física {r['fisica']} · "
          f"sem previsão {r['sem_previsao']}")
    if "guarda_do_instrumento" in r:
        print(f"guarda do instrumento: {r['guarda_do_instrumento']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
