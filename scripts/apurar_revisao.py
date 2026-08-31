#!/usr/bin/env python3
"""Apura os veredictos da folha de revisão: taxa de falso positivo do isphysics.

    .venv\\Scripts\\python.exe scripts\\apurar_revisao.py --veredictos veredictos_revisao.json

A folha (`scripts/folha_de_revisao.py`) produz o JSON; este script o transforma no
número que decide se os 27,75 B tokens do corpus são confiáveis.

## A pergunta que isto responde, e a que não responde

**Responde:** dos documentos que o classificador aceitou com escore ≥ 0,9, que fração
não é de Física? Isto é a **precisão** do filtro, e é o que degrada o corpus.

**Não responde:** quanta Física o filtro rejeitou. Isso é o *recall*, exige uma
amostra dos REJEITADOS, e é outra medição — barata, mas não esta. Confundir as duas é
o erro mais comum ao ler um número de filtro.

## O que a comparação com 1,5–13,6% significa

Aquele intervalo veio de **resumos do arXiv**, e é o que justificou o limiar 0,9. O
corpus agora é texto pleno do peS2o: outra distribuição, outras fontes. Se a taxa
medida aqui cair dentro daquele intervalo, o limiar transfere. Se estourar por cima, o
limiar 0,9 foi calibrado no lugar errado — e a decisão seguinte é subir o limiar e
remedir, não seguir com o corpus como está.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.eval.statistics.proporcao import wilson  # noqa: E402

# O intervalo medido em resumos do arXiv, que justificou o limiar 0,9.
REFERENCIA_ARXIV = (0.015, 0.136)
# Tokens do corpus, para traduzir a taxa em volume. DOC-04, corpus de 2026-08-26.
TOKENS_CORPUS_B = 27.75


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--veredictos", type=Path, required=True)
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/avaliacao/revisao_pes2o.json"))
    a = p.parse_args()
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")

    d = json.loads(a.veredictos.read_text(encoding="utf-8"))
    linhas = d.get("veredictos") or []
    if not linhas:
        raise SystemExit(
            f"{a.veredictos} não tem veredictos. Abra a folha, julgue e clique em "
            "'Baixar veredictos'.")

    contagem = Counter(x["veredicto"] for x in linhas)
    duvidas = contagem.get("duvida", 0)
    # ⚠️ As dúvidas saem do denominador, e isso é uma escolha com consequência.
    #
    # Contá-las como "é Física" empurraria a taxa para baixo; como "não é",
    # para cima. Tirá-las mede a taxa entre os casos DECIDÍVEIS, que é o que a
    # pergunta quer — e o número de dúvidas é reportado ao lado, porque se ele for
    # grande a medição inteira é frágil e isso tem de ser visível.
    n = contagem.get("fisica", 0) + contagem.get("nao", 0)
    k = contagem.get("nao", 0)
    if not n:
        raise SystemExit("nenhum julgamento decidido — só dúvidas. Não há taxa.")

    baixo, alto = wilson(k, n)
    taxa = k / n
    alvo = d.get("alvo_pre_comprometido", 200)

    print("=" * 72)
    print("  Taxa de FALSO POSITIVO do isphysics em texto pleno do peS2o")
    print("=" * 72)
    print(f"  julgados decidíveis   : {n}" +
          (f"   ⚠️ abaixo do alvo de {alvo}" if n < alvo else ""))
    print(f"  \"não sei\"             : {duvidas}"
          f"  ({100*duvidas/(n+duvidas):.1f}% do total julgado)")
    print(f"  não são de Física     : {k}")
    print(f"  TAXA                  : {100*taxa:.1f}%")
    print(f"  Wilson 95%            : {100*baixo:.1f}% a {100*alto:.1f}%")
    print()
    r_baixo, r_alto = REFERENCIA_ARXIV
    print(f"  referência (resumos do arXiv): {100*r_baixo:.1f}% a {100*r_alto:.1f}%")
    if alto <= r_alto:
        veredito = ("o limiar 0,9 TRANSFERE — o intervalo cabe no que foi medido em "
                    "resumos")
    elif baixo > r_alto:
        veredito = ("⚠️ o limiar 0,9 NÃO transfere: a taxa em texto pleno é "
                    "SIGNIFICATIVAMENTE maior que em resumos. Subir o limiar e "
                    "remedir antes de usar o corpus")
    else:
        veredito = ("indeciso — o intervalo cruza o teto da referência. Julgar mais "
                    "documentos aperta; não dá para concluir com este n")
    print(f"  veredito: {veredito}")
    print()
    print(f"  tradução em volume: a {100*taxa:.1f}%, dos {TOKENS_CORPUS_B} B tokens "
          f"do corpus,")
    print(f"    entre {TOKENS_CORPUS_B*baixo:.2f} B e {TOKENS_CORPUS_B*alto:.2f} B "
          "não são de Física.")
    print()
    print("  ⚠️ Isto é PRECISÃO, não recall. Quanta Física o filtro jogou fora exige")
    print("     uma amostra dos REJEITADOS, e é outra medição.")
    print("=" * 72)

    # Onde o julgamento discordou do modelo, por faixa de escore: se os falsos
    # positivos se concentram perto de 0,9, subir o limiar resolve barato.
    faixas: dict[str, list[int]] = {}
    for x in linhas:
        if x["veredicto"] == "duvida":
            continue
        s = float(x["score"])
        rot = "0,90–0,95" if s < 0.95 else ("0,95–0,99" if s < 0.99 else "0,99–1,00")
        faixas.setdefault(rot, [0, 0])
        faixas[rot][0] += 1
        faixas[rot][1] += x["veredicto"] == "nao"
    print("  falsos positivos por faixa de escore:")
    for rot in sorted(faixas):
        tot, ruins = faixas[rot]
        b2, a2 = wilson(ruins, tot)
        print(f"    {rot}  n={tot:4d}  {ruins:3d} ruins  {100*ruins/tot:5.1f}%  "
              f"[{100*b2:.1f}%, {100*a2:.1f}%]")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "n_decidiveis": n, "duvidas": duvidas, "falsos_positivos": k,
        "taxa": round(taxa, 4),
        "ic95_wilson": [round(baixo, 4), round(alto, 4)],
        "referencia_resumos_arxiv": list(REFERENCIA_ARXIV),
        "veredito": veredito,
        "alvo_pre_comprometido": alvo,
        "por_faixa_de_escore": {r: {"n": v[0], "ruins": v[1]}
                                for r, v in sorted(faixas.items())},
        "assinatura_amostra": d.get("assinatura_amostra"),
        "nota": ("PRECISÃO do filtro em texto pleno do peS2o, não recall. As dúvidas "
                 "saem do denominador; a contagem delas está ao lado porque uma "
                 "fração alta de dúvidas torna a medição frágil."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
