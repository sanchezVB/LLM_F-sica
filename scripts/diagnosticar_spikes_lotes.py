#!/usr/bin/env python3
"""Reproduz os lotes dos passos em que o detector de spike disparou, SEM GPU.

    .venv\\Scripts\\python.exe scripts\\diagnosticar_spikes_lotes.py \\
        --dados data/processed/phienc_dados_modernbert --p-equacao 0.6 \\
        --passos 429 1699 4489 --out data/processed/avaliacao/t2eq_cpt_spikes_lotes.json

É o que o DOC-08 §6.1 pede quando o detector para um run ("olhe os batches"), e o
laço deixa isso barato: o fluxo e a máscara são funções de `(semente, índice do
micro-passo)`, então o lote de qualquer passo sai de novo sem reexecutar nada antes.

## A pergunta

Um spike de perda é instabilidade do treino, ou é o LOTE que era mais difícil? No
braço tratado a pergunta tem resposta mecânica: um lote que esconde mais equações
inteiras tem perda média maior sem nada estar errado. O script conta, para cada passo
pedido e para uma amostra de passos sorteados, quantos alvos vieram da equação
inteira, e diz onde os passos do spike caem nessa distribuição.

Medido em 2026-09-19 no CPT tratado do ModernBERT-base: os passos 429, 1.699 e 4.489
tinham +3,7σ, +3,0σ e +4,1σ de tokens de equação inteira, e dois passavam do máximo
de 300 passos sorteados. Ver o § do spike de perda em `phifm.training.pretrain.laco`.

⚠️ Os argumentos têm de ser os do run (`contexto`, `sequencias`, `acumulacao`,
`semente`, `id_mask`), e estão no `phienc.json` dele. Outro valor reproduz outro lote,
sem erro nenhum.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from phifm.core.console import utf8 as console_utf8  # noqa: E402
from phifm.training.pretrain.dados import ConfigDados, Fluxo  # noqa: E402
from phifm.training.pretrain.mascaramento import (  # noqa: E402
    ConfigMascara,
    Contadores,
    mascarar_com_origem,
)

console_utf8()


def contar_passo(fluxo: Fluxo, cfg: ConfigMascara, passo: int, acumulacao: int,
                 id_mask: int, n_vocab: int, especiais: frozenset[int]) -> dict:
    """Os contadores do passo inteiro — todos os micro-passos, de todos os processos."""
    c = Contadores()
    de_equacao = 0
    for j in range(acumulacao):
        indice = passo * acumulacao + j
        ids, ide, disp = fluxo.lote(indice)
        rng = np.random.default_rng((cfg.semente, indice))
        for s in range(ids.shape[0]):
            _, _, origem = mascarar_com_origem(
                ids[s], ide[s], disp[s], cfg=cfg, rng=rng, id_mask=id_mask,
                n_vocab=n_vocab, ids_especiais=especiais, contadores=c)
            de_equacao += int(origem.sum())
    return {"passo": passo, "alvos": c.tokens_mascarados,
            "alvos_de_equacao_inteira": de_equacao,
            "fracao_de_equacao_inteira": de_equacao / max(c.tokens_mascarados, 1),
            "exemplos_tratados": c.tratados}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dados", type=Path, required=True)
    p.add_argument("--passos", type=int, nargs="+", required=True)
    p.add_argument("--p-equacao", type=float, required=True)
    p.add_argument("--contexto", type=int, default=1024)
    p.add_argument("--sequencias", type=int, default=4)
    p.add_argument("--acumulacao", type=int, default=16,
                   help="a do passo INTEIRO, somando os processos")
    p.add_argument("--semente", type=int, default=17)
    p.add_argument("--taxa", type=float, default=0.3)
    p.add_argument("--id-mask", type=int, default=50284)
    p.add_argument("--n-vocab", type=int, default=50368)
    p.add_argument("--especiais", type=int, nargs="+",
                   default=[50279, 50280, 50281, 50282, 50283, 50284])
    p.add_argument("--amostra", type=int, default=300)
    p.add_argument("--faixa", type=int, nargs=2, default=None,
                   help="de onde sortear a amostra; padrão: 250 até o maior passo")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    fluxo = Fluxo(ConfigDados(raiz=a.dados, contexto=a.contexto,
                              sequencias=a.sequencias, semente=a.semente))
    cfg = ConfigMascara(taxa=a.taxa, p_equacao=a.p_equacao, semente=a.semente)
    kw = dict(acumulacao=a.acumulacao, id_mask=a.id_mask, n_vocab=a.n_vocab,
              especiais=frozenset(a.especiais))

    ini, fim = a.faixa or (250, max(a.passos))
    rng = np.random.default_rng(2026)
    sorteados = sorted(int(x) for x in rng.choice(
        np.arange(ini, fim), min(a.amostra, fim - ini), replace=False))
    ref = [contar_passo(fluxo, cfg, s, **kw) for s in sorteados]
    v = np.array([r["alvos_de_equacao_inteira"] for r in ref], dtype=float)
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    print(f"referência: {len(ref)} passos sorteados em [{ini}, {fim}) · alvos de "
          f"equação inteira μ={mu:.1f} σ={sd:.1f} máx={v.max():.0f}")

    alvo = []
    for passo in a.passos:
        r = contar_passo(fluxo, cfg, passo, **kw)
        r["z"] = (r["alvos_de_equacao_inteira"] - mu) / sd if sd else None
        r["fracao_da_referencia_maior_ou_igual"] = float(
            (v >= r["alvos_de_equacao_inteira"]).mean())
        alvo.append(r)
        print(f"  passo {passo}: {r['alvos_de_equacao_inteira']} alvos de equação "
              f"inteira ({r['fracao_de_equacao_inteira']:.4f} dos alvos) · "
              f"z={r['z']:+.2f} · referência ≥ ele: "
              f"{r['fracao_da_referencia_maior_ou_igual']:.3f}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "dados": str(a.dados), "p_equacao": a.p_equacao,
            "contexto": a.contexto, "sequencias": a.sequencias,
            "acumulacao": a.acumulacao, "semente": a.semente,
            "referencia": {"n": len(ref), "faixa": [ini, fim], "media": mu,
                           "desvio": sd, "maximo": float(v.max()),
                           "p99": float(np.percentile(v, 99))},
            "passos": alvo,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
