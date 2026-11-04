#!/usr/bin/env python3
"""Corrige aqui as gerações de TESTE do G2.1 já feitas em outra máquina (a 8B, no Kaggle),
com o código congelado: as mesmas `excluidos`, `_amostras`, `corrigir_e_agregar` e `K_TESTE`
do `medir_g2.py`. Grava o mesmo agregado que o `medir_g2.py teste` gravaria (só agregados).

    .venv\\Scripts\\python.exe scripts\\corrigir_g2_teste.py --modelo 8B --braco base

Por que existe: a geração da 8B roda no Kaggle (T4×2), e a correção é só CPU. Na sessão 3
(2026-11-04), a geração terminou e o prazo pegou a correção; refazê-la lá gastaria cota de
GPU parada. A sanidade da 8B mostrou os mesmos 439 vereditos aqui e no Kaggle (DOC-12 item
23), e a 1,7B também foi corrigida nesta máquina, então os dois braços ficam no mesmo lugar.

Recusa, antes de corrigir qualquer coisa, se:
- o congelamento não confere (`exigir_aceite_e_congelamento`);
- o jsonl tem mais de um hash de configuração, alguma linha `erro`, ou amostras faltando;
- o hash não é o de uma `medida` aberta no livro de acessos (tamanho, braço, nota "").

A guarda `__main__` é obrigatória: o corretor isolado sobe por spawn.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))


def main() -> int:
    import medir_g2 as m
    from phifm.eval import g2, g2_geracao, g2_itens

    p = argparse.ArgumentParser()
    p.add_argument("--modelo", choices=tuple(m.MODELOS), required=True)
    p.add_argument("--braco", default="base")
    a = p.parse_args()
    g2.exigir_aceite_e_congelamento()
    fora = m.excluidos("teste")
    itens = [i for b in g2.BENCH_G2_1 for i in g2_itens.carregar(b, "teste")
             if i.id not in fora.get(b, set())]
    saida = m.SAIDAS / f"teste_{a.modelo}_{a.braco}_veredito.jsonl"
    hashes = {d["hash_config"] for d in g2_geracao.ler_jsonl(saida)}
    if len(hashes) != 1:
        raise SystemExit(f"o jsonl tem {len(hashes)} hashes de configuração; esperado 1")
    hc = hashes.pop()
    if not g2.medida_aberta(a.modelo, a.braco, hc):
        raise SystemExit(f"{hc[:12]}… não é o hash de uma medida aberta no livro de acessos")
    amostras = m._amostras(saida, hc, {(i.bench, i.id): i for i in itens})
    esperadas = sum(m.K_TESTE[i.bench] for i in itens)
    erros = sum(x["parada"] == "erro" for x in amostras)
    if len(amostras) != esperadas or erros:
        raise SystemExit(f"{len(amostras)} amostras de {esperadas}, {erros} com 'erro': "
                         "a geração não terminou; retome-a antes de corrigir")
    print(f"teste {a.modelo} {a.braco}: {len(amostras)} amostras, hash {hc[:12]}…; corrigindo",
          flush=True)
    r = m.corrigir_e_agregar("teste", a.modelo, a.braco, "veredito", itens, m.K_TESTE, saida,
                             hc, time.time())
    for b, x in r["por_benchmark"].items():
        print(f"{b}: acerto {x['acerto']} · {x['fracoes']} · fichas média {x['fichas_media']}")
    print(f"correções mortas no teto: {r['correcoes_mortas_no_teto']} · {r['segundos']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
