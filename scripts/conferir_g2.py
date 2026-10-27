#!/usr/bin/env python3
"""Lista bloqueios do G2 sem iniciar modelo, baixar ou carregar itens do teste.

    python scripts/conferir_g2.py --fase dev [--modelo 1.7B]

O comando não congela o protocolo nem preenche hashes automaticamente. Campos que
precisam ser registrados ao fim do dev: codigo_congelado.arquivos (mapa de código e
versões), exemplos_few_shot.sha256 (JSON selecionado), motor.sha256_executavel e
motor.hash_da_configuracao. Conteúdo de benchmark nunca aparece na saída.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval.g2_prerequisitos import conferir  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raiz", type=Path, default=RAIZ)
    p.add_argument("--fase", choices=("dev", "teste"), default="dev")
    p.add_argument("--modelo", choices=("1.7B", "8B"), default="1.7B")
    p.add_argument("--braco", default="base")
    p.add_argument("--gguf", type=Path)
    p.add_argument("--kl", type=Path)
    p.add_argument("--manifesto-cpt", type=Path)
    p.add_argument("--executavel", type=Path)
    p.add_argument("--hash-config")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    r = conferir(a.raiz, fase=a.fase, modelo=a.modelo, braco=a.braco, gguf=a.gguf,
                  kl=a.kl, manifesto_cpt=a.manifesto_cpt, executavel=a.executavel,
                  hash_config=a.hash_config)
    if a.json:
        # O relatório público só contém contagens: IDs de exclusão não abrem o lacre.
        d = r.como_dict()
        d["exclusoes"] = {b: len(ids) for b, ids in r.exclusoes.items()}
        print(json.dumps(d, ensure_ascii=False, indent=2))
    else:
        print("Pré-condições conferidas." if r.pronto else "G2 bloqueado:")
        for erro in r.bloqueios:
            print(f"- {erro}")
        print(f"{len(r.conferidos)} arquivos conferidos por hash.")
    return 0 if r.pronto else 1


if __name__ == "__main__":
    sys.exit(main())
