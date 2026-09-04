#!/usr/bin/env python3
"""Pré-treina o ΦEnc. DOC-07 §2 e DOC-08 §4.

    .venv-treino\\Scripts\\python.exe scripts\\train_phienc.py \\
        --config proxy-bakeoff --total-passos 2000 --p-equacao 0.5

## As duas execuções que este script existe para permitir

**A ablação do DOC-07 §2.3**, que é a razão de o ΦEnc ser treinado do zero:

    --p-equacao 0.0    controle  (MLM padrão, nada de Física no objetivo)
    --p-equacao 0.5    tratado   (metade dos exemplos mascara uma equação inteira)

Tudo o mais idêntico — mesma semente, mesmo fluxo, mesmos hiperparâmetros. O
orçamento de máscara é igual nos dois braços por construção (ver
`phifm.training.pretrain.mascaramento`), então a comparação é sobre a ESTRUTURA do
que se mascara e não sobre quanto.

⚠️ **Leia `fracao_tratada` no `phienc.json` antes de acreditar em qualquer
resultado.** Se ela estiver baixa, o braço tratado recaiu em MLM aleatório e o
"empate" que ele reportar não diz nada sobre a hipótese. Medido no corpus:
0,883 com `--p-equacao 1.0`.

**E o bake-off do DOC-05 §11.2**, uma execução por variante de tokenizer. Ressalva
registrada em `phifm.models.encoder.config`: a 50 M a embedding domina, e o
`proxy-bakeoff` fixa o corpo por isso.

## O que este script NÃO faz

Não avalia. O DOC-05 §11.2 pede recuperação de Física, MLM em texto denso em
equações, e uma sonda de estrutura tensorial — três avaliações que não existem
ainda. Um `phienc.json` com perda baixa **não** é um veredito sobre a hipótese.
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402

from phifm.core.sistema import impedir_suspensao, liberar_suspensao  # noqa: E402
from phifm.models.encoder.config import CONFIGS, obter  # noqa: E402
from phifm.training.pretrain.dados import ConfigDados, Fluxo  # noqa: E402
from phifm.training.pretrain.laco import (  # noqa: E402
    ConfigTreino,
    Treinador,
)
from phifm.training.pretrain.mascaramento import ConfigMascara  # noqa: E402
from phifm.training.pretrain.spike import ConfigSpike  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="proxy-bakeoff", choices=sorted(CONFIGS))
    p.add_argument("--dados", type=Path, default=Path("data/processed/phienc_dados"))
    p.add_argument("--out", type=Path, default=None,
                   help="por omissão, models/phienc-<config>-eq<p_equacao>")
    p.add_argument("--total-passos", type=int, required=True)
    p.add_argument("--sequencias", type=int, default=1,
                   help="sequências por micro-passo; o que couber na VRAM")
    p.add_argument("--acumulacao", type=int, default=1,
                   help="micro-passos por passo de otimizador. Os ~2 M tokens por "
                        "passo do DOC-08 §4 vêm daqui, não de um lote gigante")
    p.add_argument("--contexto", type=int, default=None,
                   help="por omissão, o da config (8.192 no ΦEnc)")
    p.add_argument("--p-equacao", type=float, default=0.0,
                   help="0.0 é o braço de CONTROLE da ablação do DOC-07 §2.3")
    p.add_argument("--taxa-mascara", type=float, default=0.30)
    p.add_argument("--lr-pico", type=float, default=1e-3)
    p.add_argument("--semente", type=int, default=17)
    p.add_argument("--passos-log", type=int, default=50)
    p.add_argument("--passos-estado", type=int, default=500)
    p.add_argument("--sem-amp", action="store_true")
    p.add_argument("--so-resumo", action="store_true",
                   help="imprime o orçamento e sai, sem treinar")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8", errors="replace")

    cfg_enc = obter(a.config)
    fluxo = Fluxo(ConfigDados(
        raiz=a.dados, contexto=a.contexto or cfg_enc.contexto,
        sequencias=a.sequencias, semente=a.semente))
    cfg = ConfigTreino(
        total_passos=a.total_passos, acumulacao=a.acumulacao, lr_pico=a.lr_pico,
        amp=not a.sem_amp, passos_log=a.passos_log,
        passos_estado=a.passos_estado)
    cfg_mascara = ConfigMascara(taxa=a.taxa_mascara, p_equacao=a.p_equacao,
                                semente=a.semente)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        # Não é erro: o teste de fumaça roda em CPU de propósito. Mas um treino de
        # verdade em CPU levaria meses, e dizer isso alto é mais barato que
        # descobrir depois de duas horas.
        logging.warning("sem CUDA — em CPU isto serve para fumaça, não para treino")

    t = Treinador(cfg_enc, cfg, cfg_mascara, fluxo, ConfigSpike(), dev)
    r = t.resumo()
    print()
    print("=" * 74)
    print(f"  {cfg_enc.nome} · {r['parametros'] / 1e6:.1f} M parâmetros "
          f"({100 * cfg_enc.fracao_de_embedding():.1f}% embedding)")
    print(f"  {a.total_passos:,} passos × {r['tokens_por_passo']:,} tokens = "
          f"{r['tokens_totais'] / 1e9:.3f} B tokens · {r['epocas']:.2f} épocas")
    print(f"  FLOPs {r['flops']:.3e} · máscara {a.taxa_mascara:.0%} · "
          f"p_equacao {a.p_equacao}")
    print(f"  {'TRATADO' if a.p_equacao > 0 else 'CONTROLE'} da ablação do "
          f"DOC-07 §2.3")
    print("=" * 74)
    if a.so_resumo:
        return 0

    saida = a.out or Path(f"models/phienc-{a.config}-eq{a.p_equacao}")
    impedir_suspensao()
    try:
        m = t.treinar(saida)
    finally:
        liberar_suspensao()

    print()
    print(f"  passo {m.passo:,} · perda {m.perda:.4f} · {m.tokens_por_s:.0f} tok/s")
    print(f"  fração tratada {t.contadores.fracao_tratada():.3f} · "
          f"taxa efetiva {t.contadores.taxa_efetiva():.4f}")
    print(f"  rollbacks {m.rollbacks} · passos pulados pelo scaler "
          f"{m.passos_pulados_pelo_scaler}")
    print(f"  -> {saida}")
    if a.p_equacao > 0 and t.contadores.fracao_tratada() < 0.5:
        print()
        print("  ⚠️ fração tratada abaixo de 0,5: o braço tratado recaiu em MLM "
              "aleatório na maioria\n     dos exemplos, e a ablação do §2.3 não "
              "mediu o que promete. Ver `mascaramento` no JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
