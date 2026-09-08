#!/usr/bin/env python3
"""A sonda de estrutura tensorial rodando num encoder. Terceira medida do §11.2.

    .venv-treino\\Scripts\\python.exe scripts\\avaliar_sonda_tensorial.py \\
        --modelo "PhysBERT=thellert/physbert_cased" \\
        --modelo "variante A=models/phienc-variante-A" \\
        --out data/processed/avaliacao/sonda_tensorial.json

A especificação da sonda — os itens, a régua de dois lados e o piso de superfície
— está em `phifm.eval.sonda_tensorial`, testada na suíte rápida. Aqui só a
passagem pelo modelo: 216 textos curtos, média mascarada, cosseno.

## ⚠️ O piso de superfície sai SEMPRE, junto do resultado

Por semelhança de string o item pede o contrário do que a superfície responde, e
medido em 2026-09-08 a superfície acerta **0 de 72**. Isso dá a escala:

    ~0,0   o modelo segue a forma da string
    ~0,5   não representa nem uma coisa nem outra
    >0,5   representa estrutura, contra a superfície

Um número da sonda sem o piso ao lado convida a ler 0,45 como "quase bom" quando
é evidência de que o modelo está do lado da superfície.

## ⚠️ A mesma média mascarada do G1, e não uma nova

`media_mascarada` + `normalize` é exatamente o que `eval/encoders.py` usa. Duas
formas de agrupar tokens no mesmo projeto dariam dois espaços diferentes com o
mesmo nome, e a comparação entre a sonda e a recuperação deixaria de valer.

Aceita caminho local **ou** id do Hub, para a sonda poder ser calibrada nos
encoders que já existem antes de qualquer ΦEnc ser treinado — que foi como se
descobriu se ela tem faixa dinâmica.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from phifm.eval.sonda_tensorial import (  # noqa: E402
    Resultado,
    itens,
    piso_de_superficie,
)
from phifm.training.embedding import (  # noqa: E402
    escolher_dispositivo,
    media_mascarada,
)

log = logging.getLogger("sonda-tensorial")


def vetores(caminho: str, textos: list[str], dev, max_tokens: int,
            lote: int) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(caminho)
    mod = AutoModel.from_pretrained(
        caminho, attn_implementation="eager").to(dev).eval()
    saidas = []
    with torch.no_grad():
        for i in range(0, len(textos), lote):
            b = tok(textos[i:i + lote], padding=True, truncation=True,
                    max_length=max_tokens, return_tensors="pt")
            b = {k: v.to(dev) for k, v in b.items()}
            h = mod(**b).last_hidden_state
            saidas.append(F.normalize(
                media_mascarada(h, b["attention_mask"]).float(), dim=-1)
                .cpu().numpy())
    del mod
    return np.vstack(saidas)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", action="append", default=[],
                   metavar="ROTULO=CAMINHO_OU_ID", required=True,
                   help="caminho local ou id do Hub; repetível")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=64,
                   help="os carregadores são curtos; 64 cobre com folga")
    p.add_argument("--lote", type=int, default=16)
    p.add_argument("--dispositivo", default="auto")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")

    its = itens()
    # Ordem fixa: base, estrutural, renomeado de cada item, achatados. O
    # desachatamento abaixo depende dela.
    textos: list[str] = []
    for it in its:
        textos.extend(it.textos())
    log.info("%d itens × 3 expressões = %d textos", len(its), len(textos))

    dev = escolher_dispositivo(a.dispositivo)
    piso = piso_de_superficie()
    log.info("piso de superfície: taxa %.4f · margem %.4f",
             piso["taxa"], piso["margem_media"])

    resultados = {}
    falhas = {}
    for spec in a.modelo:
        rotulo, _, caminho = spec.partition("=")
        if not caminho:
            log.error("--modelo espera ROTULO=CAMINHO, recebi %r", spec)
            return 2
        try:
            V = vetores(caminho, textos, dev, a.max_tokens, a.lote)
        except Exception as e:  # noqa: BLE001
            # ⚠️ Um modelo que não carrega não pode derrubar os outros: a sonda
            # é uma comparação, e perder três braços por causa do quarto seria
            # perder a sessão. A falha vai no artefato, nomeada.
            log.error("%s não carregou: %s", rotulo, e)
            falhas[rotulo] = f"{type(e).__name__}: {e}"
            continue
        r = Resultado()
        for k, it in enumerate(its):
            vb, ve, vr = V[3 * k], V[3 * k + 1], V[3 * k + 2]
            r.somar(it.familia, float(vb @ vr), float(vb @ ve))
        d = r.como_dict()
        d["acertos_por_item"] = r.acertos
        d["familias_por_item"] = r.familias
        d["caminho"] = caminho
        # ⚠️ A distância até 0,5 COM SINAL, porque é ela que diz de que lado o
        # modelo está — e é o que um leitor apressado não calcula.
        d["acima_do_acaso"] = round(d["taxa"] - 0.5, 4)
        d["lado"] = ("estrutura" if d["taxa"] > 0.5 else
                     "superfície" if d["taxa"] < 0.5 else "indefinido")
        resultados[rotulo] = d
        log.info("%-28s taxa %.4f (%+.4f do acaso) · margem %+.4f · %s",
                 rotulo, d["taxa"], d["acima_do_acaso"], d["margem_media"],
                 d["lado"])

    if not resultados:
        log.error("nenhum modelo carregou; nada a comparar")
        return 1

    artefato = {
        "itens": len(its),
        "piso_de_superficie": piso,
        "protocolo": {"max_tokens": a.max_tokens,
                      "pooling": "média mascarada + L2, a mesma do G1",
                      "dispositivo": str(dev)},
        "modelos": resultados,
        "falhas": falhas,
        "como_ler": (
            "A sonda é de DOIS LADOS: o item passa quando sim(base, renomeado) > "
            "sim(base, estrutural). Por semelhança de superfície o estrutural é o "
            "mais parecido, então a superfície acerta 0 de 72 — taxa abaixo de "
            "0,5 é evidência de que o modelo segue a string, e acima de 0,5 é "
            "evidência de estrutura. O 0,5 do acaso fica ENTRE os dois regimes."),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(artefato, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print()
    print("=" * 78)
    print(f"  SONDA DE ESTRUTURA TENSORIAL · {len(its)} itens")
    print("=" * 78)
    print(f"  {'modelo':<28} {'taxa':>7} {'do acaso':>10} {'margem':>9}  lado")
    print(f"  {'(piso: só superfície)':<28} {piso['taxa']:>7.4f} "
          f"{piso['taxa'] - 0.5:>+10.4f} {piso['margem_media']:>+9.4f}  "
          "superfície")
    for rotulo, d in sorted(resultados.items(), key=lambda x: -x[1]["taxa"]):
        print(f"  {rotulo:<28} {d['taxa']:>7.4f} {d['acima_do_acaso']:>+10.4f} "
              f"{d['margem_media']:>+9.4f}  {d['lado']}")
    print()
    print("  por família:")
    fams = sorted({f for d in resultados.values() for f in d["por_familia"]})
    print(f"  {'modelo':<28} " + " ".join(f"{f[:16]:>17}" for f in fams))
    for rotulo, d in sorted(resultados.items(), key=lambda x: -x[1]["taxa"]):
        print(f"  {rotulo:<28} " + " ".join(
            f"{d['por_familia'][f]['taxa']:>17.4f}" for f in fams))
    for rotulo, erro in falhas.items():
        print(f"  ⚠️  {rotulo} não carregou: {erro}")
    print("=" * 78)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
