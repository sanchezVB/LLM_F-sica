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

Aquele intervalo veio de **resumos do arXiv**, e é o que justificou o limiar 0,9. Se a
taxa medida aqui cair dentro dele, o limiar transfere. Se estourar por cima, o limiar
0,9 foi calibrado no lugar errado — e a decisão seguinte é subir o limiar e remedir,
não seguir com o corpus como está.

## ⚠️ Os dois estratos são contados separado, e combinados por TOKEN

O peS2o filtrado é 65,6% resumo por documento e 92,1% texto pleno por token. Uma taxa
única sobre a amostra inteira responderia a uma pergunta que ninguém fez: ela
dependeria de quantos de cada estrato foram sorteados, não de como o corpus é.

O que degrada o pré-treino é token contaminado, então o número que decide é a
combinação pelo peso em tokens, que vem do manifesto de `amostrar_para_revisao.py`. A
taxa por documento também sai, porque as duas juntas dizem se a contaminação está no
lixo curto ou nos papers longos — e a ação é diferente em cada caso.

## ⚠️ Isto mede o peS2o, e só ele

São **14,60 B** dos 27,75 B do corpus. Os outros 13,15 B (RedPajama-arXiv 10,54 B e
OpenWebMath 2,62 B) passaram por outros filtros e não estão nesta amostra. Multiplicar
esta taxa pelo corpus inteiro — que é o que este script fazia — atribui ao RedPajama
uma contaminação que ninguém mediu.
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
# ⚠️ Tokens da FATIA DO peS2o, não do corpus. O corpus tem 27,75 B; os outros
# 13,15 B vêm de RedPajama-arXiv e OpenWebMath, que esta amostra não tocou.
TOKENS_PES2O_B = 14.60
TOKENS_CORPUS_B = 27.75
MANIFESTO = Path("data/processed/avaliacao/revisao_pes2o_amostragem.json")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--veredictos", type=Path, required=True)
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/avaliacao/revisao_pes2o.json"))
    p.add_argument("--manifesto", type=Path, default=MANIFESTO,
                   help="manifesto de `amostrar_para_revisao.py`; traz o peso em "
                        "tokens de cada estrato")
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

    # ── por estrato, e a combinação pelo peso em tokens ──────────────────────
    por_estrato: dict[str, dict] = {}
    for x in linhas:
        est = x.get("estrato")
        if not est or x["veredicto"] == "duvida":
            continue
        e = por_estrato.setdefault(est, {"n": 0, "k": 0})
        e["n"] += 1
        e["k"] += x["veredicto"] == "nao"
    for e in por_estrato.values():
        e["taxa"] = e["k"] / e["n"]
        e["ic95_wilson"] = list(wilson(e["k"], e["n"]))

    pesos: dict[str, float] = {}
    if por_estrato and a.manifesto.exists():
        man = json.loads(a.manifesto.read_text(encoding="utf-8"))
        pesos = {est: v["peso_em_tokens"]
                 for est, v in man.get("estratos", {}).items()}
        faltando = sorted(set(por_estrato) - set(pesos))
        if faltando:
            raise SystemExit(
                f"o manifesto {a.manifesto} não tem peso para {faltando}. Sem o "
                "peso não há combinação por token, e uma média entre estratos de "
                "tamanhos diferentes não responde nada.")

    ponderado = None
    if pesos and abs(sum(pesos[e] for e in por_estrato) - 1) < 0.02:
        # ⚠️ Combinação CONSERVADORA: soma ponderada dos limites de Wilson, e não
        # a normal com Σ wₛ²·Var. A normal dá largura ZERO quando um estrato sai
        # com 0 falsos positivos — é o defeito que `proporcao.wilson` existe para
        # evitar. Esta fica mais larga que o intervalo exato, e larga demais é o
        # lado seguro de errar.
        ponderado = {
            "taxa": sum(pesos[e] * por_estrato[e]["taxa"] for e in por_estrato),
            "ic95": [sum(pesos[e] * por_estrato[e]["ic95_wilson"][i]
                         for e in por_estrato) for i in (0, 1)],
            "pesos": {e: pesos[e] for e in por_estrato},
            "metodo": "soma ponderada dos limites de Wilson por estrato; conservadora",
        }

    print("=" * 72)
    print("  Taxa de FALSO POSITIVO do isphysics na fatia do peS2o")
    print("=" * 72)
    print(f"  julgados decidíveis   : {n}" +
          (f"   ⚠️ abaixo do alvo de {alvo}" if n < alvo else ""))
    print(f"  \"não sei\"             : {duvidas}"
          f"  ({100*duvidas/(n+duvidas):.1f}% do total julgado)")
    print(f"  não são de Física     : {k}")
    # ⚠️ Estes dois são SOBRE A AMOSTRA. Com estratos de tamanhos iguais e pesos
    # de 8% e 92%, este número diz mais sobre quantos de cada um foram sorteados
    # do que sobre o corpus. O que decide está na tabela por estrato, abaixo.
    print(f"  taxa na amostra       : {100*taxa:.1f}%   ⚠️ não é a taxa do corpus")
    print(f"  Wilson 95%            : {100*baixo:.1f}% a {100*alto:.1f}%")
    if por_estrato:
        print()
        print("  por estrato — o corpus não é uma população só:")
        for est in sorted(por_estrato):
            e = por_estrato[est]
            w = pesos.get(est)
            print(f"    {est:12s} n={e['n']:4d}  {e['k']:3d} ruins  "
                  f"{100*e['taxa']:5.1f}%  "
                  f"[{100*e['ic95_wilson'][0]:.1f}%, {100*e['ic95_wilson'][1]:.1f}%]"
                  + (f"  peso em token {100*w:.1f}%" if w is not None
                     else "  (sem peso: manifesto ausente)"))
        if ponderado:
            print(f"    {'PONDERADO':12s} por token: {100*ponderado['taxa']:.1f}%  "
                  f"[{100*ponderado['ic95'][0]:.1f}%, "
                  f"{100*ponderado['ic95'][1]:.1f}%]")
        else:
            print("    ⚠️ sem combinação por token: manifesto de amostragem ausente")
            print("       ou incompleto. O número que decide é o ponderado — rode")
            print("       `scripts/amostrar_para_revisao.py`.")

    # O veredito olha o PONDERADO quando ele existe: é a taxa do corpus, e não a
    # da amostra, que depende de quantos de cada estrato foram sorteados.
    if ponderado:
        taxa_v = ponderado["taxa"]
        baixo_v, alto_v = ponderado["ic95"]
        base_v = "ponderada por token"
    else:
        baixo_v, alto_v, taxa_v = baixo, alto, taxa
        base_v = "sobre a amostra inteira, sem peso de estrato"

    print()
    r_baixo, r_alto = REFERENCIA_ARXIV
    print(f"  referência (resumos do arXiv): {100*r_baixo:.1f}% a {100*r_alto:.1f}%")
    if alto_v <= r_alto:
        veredito = ("o limiar 0,9 TRANSFERE — o intervalo cabe no que foi medido em "
                    "resumos")
    elif baixo_v > r_alto:
        veredito = ("⚠️ o limiar 0,9 NÃO transfere: a taxa em texto pleno é "
                    "SIGNIFICATIVAMENTE maior que em resumos. Subir o limiar e "
                    "remedir antes de usar o corpus")
    else:
        veredito = ("indeciso — o intervalo cruza o teto da referência. Julgar mais "
                    "documentos aperta; não dá para concluir com este n")
    print(f"  veredito ({base_v}): {veredito}")
    print()
    print(f"  tradução em volume: a {100*taxa_v:.1f}%, dos {TOKENS_PES2O_B} B tokens "
          "DO peS2o,")
    print(f"    entre {TOKENS_PES2O_B*baixo_v:.2f} B e "
          f"{TOKENS_PES2O_B*alto_v:.2f} B não são de Física.")
    print(f"  ⚠️ Só o peS2o. Os outros {TOKENS_CORPUS_B - TOKENS_PES2O_B:.2f} B do "
          f"corpus de {TOKENS_CORPUS_B} B")
    print("     (RedPajama-arXiv e OpenWebMath) passaram por outros filtros e não")
    print("     estão nesta amostra.")
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
        "por_estrato": por_estrato,
        "ponderado_por_token": ponderado,
        "base_do_veredito": base_v,
        "referencia_resumos_arxiv": list(REFERENCIA_ARXIV),
        "veredito": veredito,
        "alvo_pre_comprometido": alvo,
        "por_faixa_de_escore": {r: {"n": v[0], "ruins": v[1]}
                                for r, v in sorted(faixas.items())},
        "assinatura_amostra": d.get("assinatura_amostra"),
        "tokens_pes2o_B": TOKENS_PES2O_B,
        "tokens_corpus_B": TOKENS_CORPUS_B,
        "nota": ("PRECISÃO do filtro na fatia do peS2o, não recall, e não o corpus "
                 "inteiro: os 13,15 B de RedPajama-arXiv e OpenWebMath não estão "
                 "nesta amostra. As dúvidas saem do denominador; a contagem delas "
                 "está ao lado porque uma fração alta de dúvidas torna a medição "
                 "frágil. O número que decide é o ponderado por token — por "
                 "documento os resumos são 65,6% do peS2o, por token 7,9%."),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
