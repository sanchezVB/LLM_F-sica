#!/usr/bin/env python3
"""A partição do G2 (DOC-12 §3.7.3). Roda UMA vez, depois do aceite da regra e ANTES de
qualquer saída de modelo:

    .venv\\Scripts\\python.exe scripts\\particionar_g2.py

O que ela faz:
- No UGPhysics-en: o recorte elegível, os grupos por 13-grama e a divisão em DEV, TESTE e
  RESERVA LACRADA. Também sorteia a subamostra fixa de ~1.000 do teste para o G2.2.
- No OlympiadBench-en: os agrupamentos que o bootstrap reamostra.

Grava `data/processed/avaliacao/g2_particao.json`, versionado, só com ids, rótulos de
grupo e contagens, nunca texto de item. O SHA-256 do arquivo vai para o `g2.yaml`. Rodar
de novo tem de dar o MESMO arquivo; se não der, o script recusa sobrescrever.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2  # noqa: E402

BENCH = RAIZ / "data/raw/benchmarks"
SAIDA = RAIZ / "data/processed/avaliacao/g2_particao.json"
TIPOS_LIMPOS = {"NV", "EX", "EQ", "MC", "TF", "IN"}


def _ugphysics() -> list[dict]:
    itens = []
    for arq in sorted((BENCH / "ugphysics").glob("*/en.jsonl")):
        materia = arq.parent.name
        for linha in arq.read_text(encoding="utf-8").splitlines():
            d = json.loads(linha)
            itens.append({"id": f"{materia}/{d['index']}", "materia": materia,
                          "problema": d["problem"], "tipo": str(d["answer_type"]).strip(),
                          "varias": bool(d["is_multiple_answer"])})
    return itens


def particionar() -> dict:
    ug = _ugphysics()
    vistos: set[str] = set()
    elegivel = []
    for x in ug:
        # resposta única, tipo limpo, e só a primeira cópia de um enunciado repetido
        elegivel.append(not x["varias"] and x["tipo"] in TIPOS_LIMPOS
                        and x["problema"] not in vistos)
        vistos.add(x["problema"])
    grupo = g2.grupos_por_ngramas([x["problema"] for x in ug])
    materia = [x["materia"] for x in ug]
    p = g2.particionar_por_grupos(materia, elegivel, grupo)

    # A subamostra do G2.2: ~1.000 do teste, por grupo e matéria, com a mesma regra.
    t = p["teste"]
    sub = g2.particionar_por_grupos([materia[i] for i in t], [True] * len(t),
                                    [grupo[i] for i in t], frac_dev=0.0, n_teste=1000,
                                    semente=g2.SEMENTE_G2 + 1)
    teste_g2_2 = sorted(t[j] for j in sub["teste"])

    oly = pq.read_table(BENCH / "olympiadbench/OlympiadBench/OE_TO_physics_en_COMP/"
                        "OE_TO_physics_en_COMP.parquet",
                        columns=["id", "context", "question"]).to_pylist()
    g_oly = g2.agrupar_por_contexto([x["context"] for x in oly], [x["question"] for x in oly])

    def ids(indices):
        return [ug[i]["id"] for i in indices]

    tam = collections.Counter(collections.Counter(grupo).values())
    return {
        "o_que_e": "partição do G2 (DOC-12 §3.7.3): só ids e rótulos de grupo; gravada antes "
                   "de qualquer saída de modelo",
        "semente": g2.SEMENTE_G2,
        "ugphysics": {
            "itens": len(ug), "elegiveis": sum(elegivel),
            "grupos": len(set(grupo)), "grupos_por_tamanho": dict(sorted(tam.items())),
            "contagens": {k: len(v) for k, v in p.items()} | {"teste_g2_2": len(teste_g2_2)},
            "teste_por_materia": dict(sorted(collections.Counter(materia[i] for i in p["teste"])
                                             .items())),
            "dev": ids(p["dev"]), "teste": ids(p["teste"]), "reserva": ids(p["reserva"]),
            "teste_g2_2": ids(teste_g2_2),
            # o rótulo do grupo (o id do menor membro) de cada item de dev e de teste
            "grupo": {ug[i]["id"]: ug[grupo[i]]["id"] for i in p["dev"] + p["teste"]},
        },
        "olympiadbench": {
            "itens": len(oly), "agrupamentos": len(set(g_oly)),
            "grupo": {str(x["id"]): str(oly[g]["id"]) for x, g in zip(oly, g_oly, strict=True)},
        },
    }


def main() -> int:
    r = particionar()
    texto = json.dumps(r, ensure_ascii=False, indent=1) + "\n"
    if SAIDA.exists() and SAIDA.read_text(encoding="utf-8").replace("\r\n", "\n") != texto:
        raise SystemExit(f"{SAIDA.name} já existe e a partição de agora é DIFERENTE: não "
                         "sobrescrevo. A partição só muda por emenda registrada.")
    SAIDA.write_text(texto, encoding="utf-8", newline="\n")
    u = r["ugphysics"]
    print(f"UGPhysics: {u['itens']} itens, {u['elegiveis']} elegíveis, {u['grupos']} grupos")
    print(f"   {u['contagens']}")
    print(f"OlympiadBench: {r['olympiadbench']['itens']} itens em "
          f"{r['olympiadbench']['agrupamentos']} agrupamentos")
    print(f"{SAIDA.relative_to(RAIZ)} · sha256 {g2.sha256_arquivo(SAIDA)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
