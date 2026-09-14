#!/usr/bin/env python3
"""Ablação do ΦEnc em recuperação — DOC-07 §2.3 e a primeira das três do DOC-05 §11.2.

Roda na venv de TREINO (Python 3.12), em CPU por padrão.

    .venv-treino/Scripts/python.exe scripts/avaliar_phienc.py \\
        --controle models/phienc-controle --tratado models/phienc-tratado

## ⚠️ O número absoluto daqui NÃO entra na tabela do G1

O ΦEnc é um MLM sem cabeça contrastiva. Agregar a média dos estados ocultos de um
MLM é linha de base fraca em recuperação, e o próprio repositório mede a escala:
o SciBERT, que é exatamente isso, dá nDCG@10 **0,207** contra 0,370 do MiniLM
treinado PARA embedding. Esperar menos que o SciBERT aqui é o correto.

O que esta execução decide é **controle contra tratado** — os dois MLMs crus, no
mesmo pool, com a mesma semente. A fraqueza absoluta é comum aos dois e se
cancela; o que sobra é o efeito do mascaramento consciente de equações.

## Um braço só

`--modelo` mede um checkpoint isolado e **não emite veredito**, de propósito: sem
o outro braço não há nada a concluir sobre a hipótese. Serve para conferir que o
checkpoint carrega e que o tokenizer é o certo, antes de gastar a segunda metade
da cota.
"""
import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polars as pl  # noqa: E402

from phifm.eval.phienc import (  # noqa: E402
    RESSALVA,
    avaliar,
    carregar,
    comparar_bracos,
    meta_do_tokenizer,
)
from phifm.training.amostragem import SEMENTE_POOL, preparar_pool, teto_do_pool  # noqa: E402

log = logging.getLogger("avaliar_phienc")

TOKENIZER = Path("data/processed/tokenizer/variante_A.json")


def _linha(r) -> str:
    return (f"  {r.nome:<34} nDCG@10 {r.ndcg_10:.4f} · recall@1 {r.recall_1:.4f} · "
            f"recall@10 {r.recall_10:.4f} · MRR {r.mrr:.4f}")


def main(argv: list[str] | None = None) -> int:
    # O console do Windows entrega cp1252 e a saída tem `Φ`. Sem isto o script
    # levanta na ÚLTIMA linha, depois de medir tudo — já aconteceu no G1.
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--controle", type=Path, help="checkpoint com p_equacao=0")
    p.add_argument("--tratado", type=Path, help="checkpoint com p_equacao>0")
    p.add_argument("--modelo", type=Path,
                   help="mede UM checkpoint e não emite veredito; para conferência")
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER,
                   help="o MESMO com que o checkpoint foi treinado; o tamanho do "
                        "vocabulário é conferido contra a config do modelo")
    p.add_argument("--pares", type=Path, default=Path("data/processed/pares"),
                   help="diretório com `pares_validacao.parquet`")
    p.add_argument("--n", type=int, default=2000,
                   help="candidatos; 256 não separa margens de ~0,02")
    p.add_argument("--semente", type=int, default=SEMENTE_POOL,
                   help="sorteio do pool. Mudar muda o protocolo — de propósito")
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--lote", type=int, default=16)
    p.add_argument("--dispositivo", default="cpu")
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/avaliacao/phienc_recuperacao.json"))
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    if not a.modelo and not (a.controle and a.tratado):
        log.error("passe --controle E --tratado (a ablação), ou --modelo (conferência)")
        return 2

    val = pl.read_parquet(a.pares / "pares_validacao.parquet")

    # O teto sai no artefato ao lado da métrica. Um nDCG@10 de 0,08 não diz, sozinho,
    # se o modelo é fraco ou se o protocolo é — foi o defeito que invalidou o G1.
    pool = preparar_pool(val, a.n, a.semente)
    teto = teto_do_pool(pool)
    log.info("pool: %d itens · teto recall@1 %.4f · teto nDCG@10 %.4f",
             pool.height, teto["recall_1"], teto["ndcg_10"])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    comum = {"n": a.n, "max_tokens": a.max_tokens, "lote": a.lote,
             "dispositivo": a.dispositivo, "semente": a.semente}

    if a.modelo:
        r, meta = avaliar(a.modelo, a.tokenizer, val, a.modelo.name, **comum)
        _, tok, _ = carregar(a.modelo, a.tokenizer, a.dispositivo)
        print("\nUM BRAÇO — sem veredito, por desenho\n")
        print(_linha(r))
        print(f"\n  p_equacao ...... {(meta.get('mascara') or {}).get('p_equacao')}")
        print(f"  fracao_tratada . {(meta.get('mascaramento') or {}).get('fracao_tratada')}")
        print(f"  tokenizer ...... {meta_do_tokenizer(tok)}")
        d = {"tarefa": "recuperacao_por_citacao", "um_braco": r.__dict__,
             "mascaramento": meta.get("mascaramento"), "teto_do_pool": teto,
             "veredito": "sem veredito: um braço não decide a ablação",
             # A ressalva acompanha os DOIS modos. Num artefato de um braço só,
             # ela é ainda mais necessária: é o número mais fácil de copiar para
             # a linha errada de uma tabela.
             "ressalva": RESSALVA}
    else:
        d = comparar_bracos(a.controle, a.tratado, a.tokenizer, val, **comum)
        d["teto_do_pool"] = teto
        print("\nABLAÇÃO DO MASCARAMENTO CONSCIENTE DE EQUAÇÕES\n")
        for chave in ("controle", "tratado"):
            b = d["bracos"][chave]
            print(f"  {b['nome']:<34} nDCG@10 {b['ndcg_10']:.4f} · "
                  f"recall@1 {b['recall_1']:.4f} · recall@10 {b['recall_10']:.4f}")
        pa = d["pareado"]
        if not pa.get("erro"):
            print(f"\n  pareado em recall@1: {pa['ganha_a']} a {pa['ganha_b']} sobre "
                  f"{pa['discordantes']} discordantes (p={pa['p']:.4f})")
        if d["impedimentos"]:
            print("\n  IMPEDIMENTOS:")
            for x in d["impedimentos"]:
                print(f"    · {x}")
        print(f"\n  VEREDITO: {d['veredito']}")

    print(f"\n  {d['ressalva']}\n")
    a.out.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str),
                     encoding="utf-8")
    log.info("gravado em %s", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
