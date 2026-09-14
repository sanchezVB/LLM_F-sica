#!/usr/bin/env python3
"""MLM em texto denso em equações — DOC-05 §11.2, segunda das três avaliações.

Roda na venv de TREINO (Python 3.12).

    .venv-treino/Scripts/python.exe scripts/avaliar_mlm_phienc.py \\
        --controle models/phienc-controle --tratado models/phienc-tratado \\
        --dados data/processed/phienc_holdout \\
        --dados-treino data/processed/phienc_dados

## ⚠️ O conjunto de avaliação precisa ser DISJUNTO, e ele não é por padrão

O `Fluxo` do pré-treino permuta **todas** as sequências do binário: não existe
divisão de validação. Apontar `--dados` para `phienc_dados` mede perplexidade em
dado visto e não produz erro nenhum — produz um número bom.

Por isso `--dados-treino` é **obrigatório**: ele é o manifesto contra o qual a
disjunção é provada, partição por partição. Para montar o conjunto:

    python scripts/preparar_dados_phienc.py \\
        --out data/processed/phienc_holdout \\
        --excluir-de data/processed/phienc_dados/MANIFESTO_DADOS.json \\
        --max-tokens 200_000_000

## O desenho 2×2

Cada regime favorece um braço por construção — o aleatório é o objetivo do
controle, o de equação é o do tratado. Nenhum decide sozinho; ver a docstring de
`phifm.eval.mlm`. A célula que não é tautológica sai destacada: **tokens de
equação sob mascaramento ALEATÓRIO**.
"""
import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.eval.mlm import (  # noqa: E402
    RESSALVA,
    avaliar_ablacao,
    carregar_fluxo,
    conferir_disjuncao,
)
from phifm.eval.phienc import carregar_mlm, conferir_uma_variavel  # noqa: E402

log = logging.getLogger("avaliar_mlm")

TOKENIZER = Path("data/processed/tokenizer/variante_A.json")


def main(argv: list[str] | None = None) -> int:
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--controle", type=Path, required=True)
    p.add_argument("--tratado", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    p.add_argument("--dados", type=Path, required=True,
                   help="o conjunto de AVALIAÇÃO, disjunto do de treino")
    p.add_argument("--dados-treino", type=Path, required=True,
                   help="o conjunto de TREINO; serve para provar a disjunção. "
                        "Obrigatório de propósito: sem ele, avaliar em dado visto "
                        "não produziria erro nenhum")
    p.add_argument("--sequencias", type=int, default=256,
                   help="sequências sorteadas do conjunto de avaliação")
    p.add_argument("--taxa", type=float, default=0.30,
                   help="taxa de mascaramento; a MESMA do treino, senão os dois "
                        "regimes deixam de ser comparáveis ao que foi treinado")
    p.add_argument("--contexto", type=int, default=None,
                   help="padrão: o do preparo. Abaixo de ~67 tokens o orçamento de "
                        "máscara fica menor que MIN_TOKENS_TRATAMENTO e o regime de "
                        "equação recai INTEIRO em aleatório — ali `mascaras_do_regime` "
                        "levanta em vez de reportar dois regimes idênticos")
    p.add_argument("--semente", type=int, default=17)
    p.add_argument("--lote", type=int, default=1)
    p.add_argument("--dispositivo", default="cpu")
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/avaliacao/phienc_mlm.json"))
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    # Antes de carregar qualquer modelo: se os conjuntos se tocam, não há o que medir.
    disj = conferir_disjuncao(a.dados_treino, a.dados)
    log.info("disjunção provada: %d partições no treino, %d na avaliação",
             disj["particoes_treino"], disj["particoes_avaliacao"])
    if disj["tokenizer_treino"] != disj["tokenizer_avaliacao"]:
        log.warning("os dois preparos usaram tokenizers diferentes (%s contra %s) — "
                    "os ids não significam a mesma coisa nos dois binários",
                    disj["tokenizer_treino"], disj["tokenizer_avaliacao"])

    ctrl, _, meta_c = carregar_mlm(a.controle, a.tokenizer, a.dispositivo)
    trat, _, meta_t = carregar_mlm(a.tratado, a.tokenizer, a.dispositivo)

    # Mesma checagem de uma-variável da avaliação de recuperação. Um braço com
    # hiperparâmetro escorregado ainda produz perplexidade, e é ela que alguém copia.
    divergencias = conferir_uma_variavel(meta_c, meta_t)
    if divergencias:
        log.warning("os braços diferem em mais do que o tratamento: %s", divergencias)

    fluxo = carregar_fluxo(a.dados, a.contexto)
    d = avaliar_ablacao({"controle": ctrl, "tratado": trat}, fluxo,
                        sequencias=a.sequencias, taxa=a.taxa, semente=a.semente,
                        n_vocab=meta_c["modelo"]["vocab"], dispositivo=a.dispositivo,
                        lote=a.lote)
    d["disjuncao"] = disj
    d["uma_variavel"] = divergencias

    print("\nMLM EM TEXTO DENSO EM EQUAÇÕES — 2×2\n")
    print(f"  {'regime':<12} {'braço':<10} {'perda':>8} {'ppl':>9} {'acerto':>8} "
          f"{'perda eq':>9} {'perda prosa':>12}")
    for regime, celula in d["celulas"].items():
        for nome, m in celula.items():
            print(f"  {regime:<12} {nome:<10} {m['perda']:>8.4f} "
                  f"{m['perplexidade']:>9.2f} {m['acerto']:>8.4f} "
                  f"{m['perda_equacao']:>9.4f} {m['perda_prosa']:>12.4f}")
        print(f"  {'':<12} → {d['leitura_por_regime'][regime]}")

    mec = d["mecanismo"]
    print("\n  CÉLULA DE MECANISMO — tokens de equação sob mascaramento ALEATÓRIO")
    print(f"    controle {mec['controle_perda_equacao']:.4f} · "
          f"tratado {mec['tratado_perda_equacao']:.4f} · "
          f"delta {mec['delta']:+.4f} sobre {mec['tokens']:,} tokens".replace(",", "."))
    print(f"    {mec['o_que_significa']}")

    if divergencias:
        print("\n  ⚠️ UMA-VARIÁVEL:")
        for x in divergencias:
            print(f"    · {x}")

    print(f"\n  {RESSALVA}\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str),
                     encoding="utf-8")
    log.info("gravado em %s", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
