#!/usr/bin/env python3
"""Escolhe, uma vez, os exemplos few-shot do G2 (DOC-12 §3.7.4) e os grava por id com hash:

    .venv\\Scripts\\python.exe scripts\\exemplos_g2.py

Precisa de `data/processed/avaliacao/g2_fichas_dev.json` (scripts/contar_fichas_g2.py).
Os textos de teste entram só no filtro de irmãos (13-grama; no SciBench, 10-grama e
similaridade), e nada deles é impresso. Saída: `data/processed/avaliacao/g2_exemplos.json`,
com os ids e o SHA-256 do texto formatado dos exemplos de cada benchmark (o conteúdo fica
fora do git). Rodar de novo tem de dar o mesmo arquivo; se não der, recusa sobrescrever.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2_itens, g2_prompt  # noqa: E402

SAIDA = RAIZ / "data/processed/avaliacao/g2_exemplos.json"
BENCHS = ("ugphysics", "olympiadbench", "phybench", "scibench")


def main() -> int:
    fichas = json.loads((RAIZ / "data/processed/avaliacao/g2_fichas_dev.json")
                        .read_text(encoding="utf-8"))["fichas_da_solucao"]
    ug_dev = g2_itens.carregar("ugphysics", "dev")
    sci_dev = g2_itens.carregar("scibench", "dev")
    teste = {b: g2_itens.carregar(b, "teste") for b in BENCHS}
    textos = [i.enunciado for b in BENCHS for i in teste[b]]
    sci_teste = {livro: [i.enunciado for i in teste["scibench"] if i.extra["livro"] == livro]
                 for livro in ("class", "fund", "thermo")}
    escolha = g2_prompt.escolher_exemplos(
        ug_dev, sci_dev, textos, sci_teste, fichas=lambda i: fichas[i.id],
        gradavel=lambda i: g2_itens.autoteste(i)["ok"])
    por_id = {i.id: i for i in ug_dev + sci_dev}
    r = {"o_que_e": "exemplos few-shot do G2 (DOC-12 §3.7.4): ids de DEV e o sha256 do texto "
                    "formatado; o conteúdo fica fora do git",
         "exemplos": {}}
    for bench, ids in escolha.items():
        texto = "".join(g2_prompt.formatar_exemplo(por_id[i], g2_prompt.linha_de_resposta(por_id[i]))
                        for i in ids)
        r["exemplos"][bench] = {
            "ids": ids, "fichas_da_solucao": [fichas[i] for i in ids],
            "tipos": [g2_itens.tipo_efetivo(por_id[i]) for i in ids],
            "sha256_do_texto": hashlib.sha256(texto.encode()).hexdigest()}
    saida = json.dumps(r, ensure_ascii=False, indent=1) + "\n"
    if SAIDA.exists() and SAIDA.read_text(encoding="utf-8").replace("\r\n", "\n") != saida:
        raise SystemExit(f"{SAIDA.name} já existe e a escolha de agora é DIFERENTE: não "
                         "sobrescrevo. Os exemplos só mudam por emenda registrada.")
    SAIDA.write_text(saida, encoding="utf-8", newline="\n")
    for bench, e in r["exemplos"].items():
        print(f"{bench}: {e['ids']} · tipos {e['tipos']} · fichas {e['fichas_da_solucao']}")
    print(f"→ {SAIDA.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
