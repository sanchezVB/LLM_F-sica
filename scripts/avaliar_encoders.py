#!/usr/bin/env python3
"""Portão G1 — ΦEmb contra PhysBERT e embedders gerais (DOC-00 §5, DOC-07 §3).

Roda na venv de TREINO (Python 3.12), e em CPU por padrão para não disputar VRAM
com um treino em curso.

    .venv-treino/Scripts/python.exe scripts/avaliar_encoders.py

O cache (`--cache`) guarda as posições por item de cada modelo. Uma passada a
2.000 candidatos custa ~2,5 h de CPU e o PhysBERT sozinho leva 22 min; somar UM
modelo à comparação não deve custar a soma de todos. É invalidado inteiro se o
protocolo mudar — ver `comparar` em `phifm.eval.encoders`.
"""
import argparse
import contextlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polars as pl  # noqa: E402

from phifm.core.sistema import impedir_suspensao, liberar_suspensao  # noqa: E402
from phifm.eval.encoders import (  # noqa: E402
    comparar,
    salvar,
    tabela,
    tabela_pareada,
    veredito,
)

# Nossos candidatos. O `-melhor` é o checkpoint de pico, não o último passo — no
# treino sobre MiniLM o pico deu MRR 0,477 contra 0,469 do fim.
NOSSOS = {
    "ΦEmb/SciBERT (110M)": Path("models/phiemb"),
    "ΦEmb/MiniLM (23M)": Path("models/phiemb-minilm-melhor"),
    # Lote lógico 512 (511 negativos) via GradCache, contra os 127 do acima. Mesma
    # base, mesmos 400 mil pares — a única variável é o número de negativos.
    #
    # ⚠️ O checkpoint `-melhor` foi escolhido por MRR, que NÃO é a métrica do
    # portão. O G1 usa nDCG@10, e as duas divergiram neste treino: recall@1 ficou
    # preso em 0,280 por quatro avaliações enquanto recall@10 subiu de 0,716 para
    # 0,780. É esta avaliação que resolve, medindo nDCG@10 de verdade.
    "ΦEmb/MiniLM+GC (23M, 511 neg)": Path("models/phiemb-minilm-gc-melhor"),
    # MAIS DADOS, com o lote que ganhou por medição: 1,5 M de pares contra os
    # 400 mil do candidato acima, mesma base, mesmos 127 negativos. A única
    # variável é o volume.
    #
    # Interrompido no passo 4.500 de 11.719 (38%), por platô medido: os ganhos
    # aconteceram até o passo 2.000, e de 2.500 a 4.500 — 256 mil pares, metade
    # deles inéditos para o run de 400 mil — o nDCG@10 oscilou entre 0,528 e
    # 0,542 sem tendência. MRR e recall@1 terminaram empatados com o campeão de
    # 400 mil, que usou 33% dos dados.
    #
    # O checkpoint é o do passo 4.000, o pico (nDCG@10 0,5423 entre 1.000
    # candidatos). Aqui ele é remedido nos 2.000 do protocolo, que é a única
    # medição comparável ao veredito — e a razão de esta avaliação existir: o
    # `melhor.json` do campeão de 400 mil não tem nDCG@10, porque foi gravado
    # antes da correção de critério, então MRR era o único proxy disponível.
    "ΦEmb/MiniLM 1,5M (23M, 127 neg)": Path("models/phiemb-minilm-1m5-melhor"),
}


def main() -> int:
    # ⚠️ A tabela imprime `Φ` e o console do Windows entrega cp1252. Sem isto o
    # script levanta UnicodeEncodeError na ÚLTIMA linha, DEPOIS de medir tudo —
    # medido em 2026-08-27: a avaliação do modelo da T4 terminou em 115 s e o
    # resultado morreu no `print`. Trabalho feito, saída dizendo que falhou.
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser()
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"))
    p.add_argument("--modelo", action="append", default=[], metavar="ROTULO=CAMINHO",
                   help="acrescenta um modelo nosso; repetível. Sem isto usa os padrões.")
    p.add_argument("--n", type=int, default=2000,
                   help="candidatos na avaliação; 256 não separa margens de ~0,02")
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--lote", type=int, default=16)
    p.add_argument("--dispositivo", default="cpu", choices=["cpu", "dml"])
    p.add_argument("--cache", type=Path,
                   default=Path("data/processed/avaliacao/g1_cache.json"),
                   help="posições por item, para não remedir; NÃO é o resultado")
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/avaliacao/g1_resultado.json"),
                   help="o resultado, que é versionado — o log do lançador não é")
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    val = pl.read_parquet(a.pares / "pares_validacao.parquet")
    logging.info("validação: %s pares · usando os %d primeiros", f"{val.height:,}", a.n)

    pedidos = dict(NOSSOS)
    for spec in a.modelo:
        rotulo, _, caminho = spec.partition("=")
        if not caminho:
            logging.error("--modelo espera ROTULO=CAMINHO, recebi %r", spec)
            return 2
        pedidos[rotulo] = Path(caminho)

    extras = {}
    for rotulo, caminho in pedidos.items():
        if (caminho / "config.json").exists():
            extras[rotulo] = str(caminho)
        else:
            logging.warning("%s sem config.json — %s fora da comparação", caminho, rotulo)
    if not extras:
        logging.error("nenhum modelo nosso disponível; nada a comparar")
        return 1

    # A comparação completa leva ~2,5 h de CPU. Ver o comentário em
    # train_embedding.py: a máquina dormiu e matou um treino por falta disto.
    impedir_suspensao()
    try:
        rs = comparar(val, extras, cache=a.cache, n=a.n, max_tokens=a.max_tokens,
                      lote=a.lote, dispositivo=a.dispositivo)
    finally:
        liberar_suspensao()

    print("\n" + "=" * 78)
    print(tabela(rs, a.n))
    print("=" * 78)
    print()
    print(tabela_pareada(rs))
    print()
    print("=" * 78)
    print(veredito(rs, a.n))
    salvar(rs, a.out, a.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
