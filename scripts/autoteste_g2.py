#!/usr/bin/env python3
"""O autoteste do gabarito do G2 (DOC-12 §3.7.3, item 1): automático e sem modelo.

    .venv\\Scripts\\python.exe scripts\\autoteste_g2.py [--so dev|teste] [--processos 4]

Em cada item, o gabarito posto em `\\boxed{}` tem de dar PASS contra si mesmo, e cada
perturbação tem de dar FAIL: o número deslocado por 3× a tolerância do item, a expressão
com +1 e com um símbolo trocado, e assim por diante (`g2_itens.perturbacoes`). O item que
falha sai dos DOIS braços, por id.

Só o código lê os gabaritos. A saída traz contagens e ids, nunca texto de item. Ela vai
para `data/processed/avaliacao/g2_autoteste.json`, junto com o SHA-256 do código do
corretor: a lista de exclusão vale para aquela versão. Acima de 5% de exclusão num
benchmark de teste, conserta-se o parser no DEV. Olhar o gabarito de teste que falhou
exige o aval do dono.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2, g2_itens  # noqa: E402

SAIDA = RAIZ / "data/processed/avaliacao/g2_autoteste.json"
CONJUNTOS = {
    "dev": [("ugphysics", "dev"), ("olympiadbench", "dev"), ("scibench", "dev")],
    "teste": [("ugphysics", "teste"), ("olympiadbench", "teste"), ("phybench", "teste"),
              ("scibench", "teste")],
}
LIMITE_EXCLUSAO = 0.05


def _um(item: g2_itens.Item) -> dict:
    return g2_itens.autoteste(item) | {"tipo": "+".join(sorted(set(item.tipos)))}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--so", choices=tuple(CONJUNTOS))
    # 1 por padrão: o teto da comparação é de relógio, e com 2 processos o autoteste do
    # UG-dev trocou 10 ids de lado só por carga (sessão do verify, 2026-10-20)
    p.add_argument("--processos", type=int, default=1)
    a = p.parse_args()
    codigo = {nome: g2.sha256_arquivo(RAIZ / f"src/phifm/eval/{nome}.py")
              for nome in ("g2_corretor", "g2_itens")}
    anterior = json.loads(SAIDA.read_text(encoding="utf-8")) if SAIDA.exists() else {}
    resultado = {"o_que_e": "autoteste do gabarito do G2 (DOC-12 §3.7.3): só contagens e ids",
                 "codigo_sha256": codigo, "conjuntos": anterior.get("conjuntos", {})}
    if anterior.get("codigo_sha256") != codigo:
        resultado["conjuntos"] = {}       # a versão do corretor mudou: refaz tudo
    for grupo in ([a.so] if a.so else list(CONJUNTOS)):
        for bench, conjunto in CONJUNTOS[grupo]:
            itens = g2_itens.carregar(bench, conjunto)
            t0 = time.perf_counter()
            if a.processos > 1:
                with ProcessPoolExecutor(a.processos) as ex:
                    rs = list(ex.map(_um, itens, chunksize=4))
            else:
                rs = [_um(i) for i in itens]
            # Segundo passe, sequencial: quem falhou é refeito com a máquina só para ele, e
            # a exclusão só vale se falhar de novo (o estouro de tempo por carga não exclui)
            por_id = {i.id: i for i in itens}
            refeitos = 0
            for k, r in enumerate(rs):
                if not r["ok"]:
                    rs[k] = _um(por_id[r["id"]])
                    refeitos += 1
            falhos = [r for r in rs if not r["ok"]]
            taxa = len(falhos) / len(rs)
            por_tipo = collections.defaultdict(lambda: [0, 0])
            for r in rs:
                por_tipo[r["tipo"]][0] += 1
                por_tipo[r["tipo"]][1] += not r["ok"]
            resultado["conjuntos"][f"{bench}/{conjunto}"] = {
                "itens": len(rs), "excluidos": len(falhos), "taxa_exclusao": round(taxa, 4),
                "refeitos_no_segundo_passe": refeitos,
                "acima_do_limite": conjunto == "teste" and taxa > LIMITE_EXCLUSAO,
                "por_tipo": {t: {"itens": n, "excluidos": f} for t, (n, f) in sorted(por_tipo.items())},
                "motivos": dict(collections.Counter(
                    ("proprio:" + r["proprio"]) if r["proprio"] != "pass"
                    else ("sem_perturbacao" if not r["perturbacoes"] else "perturbacao_passou")
                    for r in falhos)),
                "ids_excluidos": sorted(r["id"] for r in falhos),
            }
            print(f"{bench}/{conjunto}: {len(rs)} itens, {len(falhos)} excluídos "
                  f"({taxa:.1%}) em {time.perf_counter() - t0:.0f} s · "
                  f"{resultado['conjuntos'][f'{bench}/{conjunto}']['motivos']}", flush=True)
    SAIDA.write_text(json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"→ {SAIDA.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
