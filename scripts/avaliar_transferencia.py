#!/usr/bin/env python3
"""Quanto o `is_physics` perde num domínio negativo que nunca viu.

O F1 de 0,972 do classificador é arXiv contra arXiv — mesma distribuição. Ele vai
ser aplicado ao peS2o e ao OpenWebMath, onde a distribuição negativa é outra, e o
número de validação não responde a pergunta que importa.

Deixa-um-domínio-de-fora é a aproximação disponível sem dado novo: treina sem um
domínio inteiro e testa só nele.

    PYTHONPATH=src .venv/Scripts/python.exe scripts/avaliar_transferencia.py

Sem `--omitir`, roda todos os domínios disponíveis — cada um por vez. Com `math`
coletado, esse é o caso que decide o passo 4d, porque matemática é a vizinha mais
confundível da Física e o OpenWebMath é feito dela.
"""
import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from phifm.corpus.filter.classifier import avaliar_transferencia  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--spine", type=Path, default=Path("data/processed/spine.parquet"))
    p.add_argument("--negativos", type=Path, default=Path("data/raw/arxiv_negativos"))
    p.add_argument("--omitir", action="append", default=[],
                   help="domínio a omitir do treino; repetível. Sem isto, todos.")
    p.add_argument("--n-por-classe", type=int, default=120_000)
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/avaliacao/transferencia.json"))
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    doms = a.omitir or sorted(d.name for d in a.negativos.iterdir() if d.is_dir())
    logging.info("domínios a omitir, um por vez: %s", ", ".join(doms))

    rs = []
    for d in doms:
        rs.append(avaliar_transferencia(a.spine, a.negativos, d, a.n_por_classe))

    print("\n" + "=" * 76)
    print("Transferência de domínio — falso positivo é a taxa de negativos")
    print("aceitos como Física. É o que contamina o corpus.")
    print()
    print(f"{'omitido':10} {'FP dentro':>10} {'FP fora':>9} {'piora':>7} "
          f"{'precisão fora':>14}")
    print("-" * 76)
    for r in rs:
        print(f"{r.dominio_omitido:10} {100*r.fp_dentro:9.1f}% {100*r.fp_fora:8.1f}% "
              f"{r.degradacao:6.1f}x {r.precisao_fora:14.3f}")

    for r in rs:
        print()
        print(f"curva de limiar no domínio nunca visto ({r.dominio_omitido}) — "
              "o limiar é o botão que se tem em produção")
        print(f"  {'limiar':>7} {'precisão':>9} {'revocação':>10} {'FP':>8}")
        for t, pr, rc, fp in r.curva:
            print(f"  {t:>7.3f} {pr:9.3f} {rc:10.3f} {100*fp:7.1f}%")

    pior = max(rs, key=lambda r: r.fp_fora)
    print()
    print("=" * 76)
    print(f"PIOR CASO: omitindo {pior.dominio_omitido}, o falso positivo vai de "
          f"{100*pior.fp_dentro:.1f}% para {100*pior.fp_fora:.1f}%.")
    menor_fp = min(fp for _, _, _, fp in pior.curva)
    print(f"E o limiar não resolve: no melhor ponto da curva ainda são "
          f"{100*menor_fp:.1f}%.")
    print()
    print("Isto NÃO é a taxa esperada no peS2o — é cota pessimista para vizinhos")
    print("próximos, medida no vizinho mais difícil que temos. Sobre texto de web")
    print("não diz nada, e nenhum dado que temos diria.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps([asdict(r) for r in rs], indent=2, ensure_ascii=False),
                     encoding="utf-8")
    logging.info("resultado → %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
