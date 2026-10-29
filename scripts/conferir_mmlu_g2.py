#!/usr/bin/env python3
"""Confere o MMLU do G2.3 (DOC-12 §3.7.6, item 3): a letra que o llama-server (Q8_0) prevê
nos itens de dev tem de bater com a do transformers na CPU (pesos originais).

    .venv-treino\\Scripts\\python.exe scripts\\conferir_mmlu_g2.py --modelo 1.7B

Lê `g2_mmlu_dev_<modelo>_base.json` (de `medir_g2_mmlu.py dev`), refaz os mesmos prompts e
compara o argmax das fichas " A"…" D". Desacordo com a diferença de logprob do transformers
abaixo de 0,05 é empate numérico (Q8_0 contra os pesos originais) e é relatado à parte.

Em float32: os pesos bf16 sobem sem perda, e o bf16 na CPU desta máquina roda num núcleo só
(a 1ª tentativa, em 2026-10-29, passou de 35 min sem terminar).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_mmlu  # noqa: E402

AVAL = RAIZ / "data/processed/avaliacao"
PESOS = {"1.7B": RAIZ / "models/base/Qwen3-1.7B-Base", "8B": RAIZ / "models/base/Qwen3-8B-Base"}
EMPATE = 0.05


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="1.7B")
    a = p.parse_args()
    dev = json.loads((AVAL / f"g2_mmlu_dev_{a.modelo}_base.json").read_text(encoding="utf-8"))
    por_id = {x["id"]: x for x in g2_mmlu.carregar("validation")}
    exemplos = g2_mmlu.exemplos_por_materia(g2_mmlu.carregar("dev"))
    tok = AutoTokenizer.from_pretrained(PESOS[a.modelo])
    for letra, ficha in g2_mmlu.FICHAS_DAS_LETRAS.items():
        assert tok.encode(f" {letra}", add_special_tokens=False) == [ficha], letra
    modelo = AutoModelForCausalLM.from_pretrained(PESOS[a.modelo], torch_dtype=torch.float32)
    modelo.eval()
    iguais, empates, diferentes = 0, [], []
    for n, (iid, letra_srv) in enumerate(sorted(dev["previstas"].items()), start=1):
        x = por_id[iid]
        ids = tok(g2_mmlu.montar_prompt(x, exemplos[x["subject"]]), return_tensors="pt").input_ids
        with torch.no_grad():
            logits = modelo(ids).logits[0, -1].float()
        lp = torch.log_softmax(logits, -1)
        notas = {L: lp[f].item() for L, f in g2_mmlu.FICHAS_DAS_LETRAS.items()}
        letra_hf = max(notas, key=notas.get)
        ordem = sorted(notas.values(), reverse=True)
        if letra_hf == letra_srv:
            iguais += 1
        elif ordem[0] - ordem[1] < EMPATE:
            empates.append(iid)
        else:
            diferentes.append(iid)
        print(f"  {n}/{len(dev['previstas'])} {iid}: servidor {letra_srv} · transformers "
              f"{letra_hf}", flush=True)
    r = {"o_que_e": "conferência do MMLU do G2.3: llama-server Q8_0 contra transformers "
                    "float32 na CPU, nos itens de dev (DOC-12 §3.7.11, item 3)",
         "itens": len(dev["previstas"]), "iguais": iguais, "empates_numericos": empates,
         "diferentes": diferentes, "ok": not diferentes}
    (AVAL / f"g2_mmlu_conferencia_{a.modelo}.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"iguais {iguais}/{r['itens']} · empates {len(empates)} · diferentes {len(diferentes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
