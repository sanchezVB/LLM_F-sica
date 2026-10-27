#!/usr/bin/env python3
"""Piloto causal de desenvolvimento, com base local e corpus auditado; nenhum download.

Não congela nem executa G2. O checkpoint deste piloto não demonstra ganho em Física.
Execute ambiente_hd.ps1 antes: base, corpus e saída devem estar no HD informado.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.corpus.auditoria_cpt import sha256, validar_manifesto


def blocos(arquivo, tok, contexto, max_blocos):
    import torch
    tokens = []
    with arquivo.open(encoding="utf-8") as f:
        for linha in f:
            if not linha.strip():
                continue
            tokens.extend(tok.encode(json.loads(linha)["text"], add_special_tokens=False))
            if tok.eos_token_id is not None:
                tokens.append(tok.eos_token_id)
            if len(tokens) >= contexto * max_blocos:
                break
    n = min(len(tokens) // contexto, max_blocos)
    if n < 1:
        raise ValueError("dados insuficientes para um bloco completo")
    return torch.tensor(tokens[:n * contexto], dtype=torch.long).reshape(n, contexto)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--manifesto", type=Path, required=True)
    p.add_argument("--saida", type=Path, required=True)
    p.add_argument("--passos", type=int, default=20)
    p.add_argument("--contexto", type=int, default=128)
    p.add_argument("--max-blocos", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--dispositivo", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--verificar", action="store_true", help="confere dados e memória, sem treinar")
    a = p.parse_args()
    if a.contexto < 2 or a.max_blocos < 1:
        p.error("contexto >= 2 e max-blocos >= 1 são obrigatórios")
    hd = os.environ.get("PHIFM_HD_ROOT")
    if not hd:
        p.error("execute ambiente_hd.ps1 com o caminho do HD antes do piloto")
    raiz_hd = Path(hd).resolve(strict=True)
    for caminho in (RAIZ, a.base, a.manifesto, a.saida):
        if not caminho.resolve().is_relative_to(raiz_hd):
            p.error(f"caminho fora do HD informado: {caminho}")
    auditoria = validar_manifesto(a.manifesto)
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from phifm.training.causal import ConfigCausal, treinar_causal
    cfg = ConfigCausal(passos=a.passos, lr=a.lr)
    conf = AutoConfig.from_pretrained(a.base, local_files_only=True, trust_remote_code=False)
    with torch.device("meta"):
        vazio = AutoModelForCausalLM.from_config(conf)
    parametros = sum(x.numel() for x in vazio.parameters())
    del vazio
    # Pesos/gradientes/momentos FP32 e margem; ativações exigem memória adicional.
    minimo = parametros * 20
    if a.dispositivo == "cuda":
        if not torch.cuda.is_available():
            p.error("CUDA indisponível nesta instalação; piloto grande exige outra máquina/stack")
        if torch.cuda.mem_get_info()[0] < minimo:
            p.error(f"memória CUDA insuficiente: mínimo estimado {minimo / 2**30:.1f} GiB + ativações")
    elif parametros > 100_000_000:
        p.error("treino completo acima de 100 M parâmetros requer CUDA neste piloto")
    print(json.dumps({"parametros": parametros, "memoria_minima_bytes": minimo,
                      "escopo": "desenvolvimento, não G2", "corpus": auditoria["schema"]}))
    if a.verificar:
        return 0
    tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True, trust_remote_code=False)
    conjuntos = {s: blocos(a.manifesto.parent / auditoria["outputs"][s]["path"],
                           tok, a.contexto, a.max_blocos) for s in ("train", "dev")}
    arquivos = sorted(f for f in a.base.rglob("*") if f.is_file())
    if not arquivos:
        p.error("base local vazia")
    if any(not f.resolve().is_relative_to(raiz_hd) for f in arquivos):
        p.error("base contém links para fora do HD")
    identidade = {"corpus_sha256": sha256(a.manifesto),
                   "base": {str(f.relative_to(a.base)): sha256(f) for f in arquivos}}
    modelo = AutoModelForCausalLM.from_pretrained(
        a.base, local_files_only=True, trust_remote_code=False).to(a.dispositivo)
    r = treinar_causal(modelo, conjuntos["train"], conjuntos["dev"], a.saida, cfg, identidade)
    print(json.dumps({k: r[k] for k in ("concluido", "passo", "perda_dev_inicial", "perda_dev_final")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
