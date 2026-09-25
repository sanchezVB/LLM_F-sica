#!/usr/bin/env python3
"""Onde está o artigo certo quando a busca do assistente não o traz? (exploratório)

Não é parte da regra do DOC-13 §9.1 e não decide nada: é o diagnóstico para o passo
seguinte, se a medida disser que a busca limita. Para cada item, a posição de P na
ordem inteira do índice (1,59 M) com duas consultas:

- **hyde** — o resumo hipotético que o braço A usou (gravado nas respostas);
- **pergunta** — a pergunta crua, em português, sem o passo hipotético.

Se P costuma estar entre a 7ª e a 50ª posição, um reranqueador sobre os 50 primeiros
tem onde ganhar; se está a milhares, o problema é a representação, não a ordem final.
A coluna `pergunta` mede se o passo hipotético se paga.

    .venv-treino\\Scripts\\python.exe scripts\\diagnosticar_busca_assistente.py

Roda na CPU (~1,5 s por consulta), não na GPU. Retomável.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

DIR = RAIZ / "data/processed/assistente"
AVALIACAO = RAIZ / "data/processed/avaliacao"
CORTES = (1, 6, 10, 20, 50, 100, 1000)


def posicao(busca, consulta: str, linha_p: int) -> int:
    s = busca.pontuar(busca.embutir(consulta))
    return int((s > s[linha_p]).sum()) + 1


def main() -> int:
    sys.path.insert(0, str(RAIZ / "scripts"))
    from medir_assistente import caminhos, carregar_itens
    from phifm.eval import assistente_bracos as b
    from phifm.retrieval.indice import Busca

    itens, sha = carregar_itens()
    cache_b, _, ab, _ = caminhos(sha)
    a_por_item = {(r.estrato, r.arxiv_id): r for r in b.ler_respostas(cache_b) if r.braco == "A"}
    if len(a_por_item) < len(itens):
        raise SystemExit("o braço A não rodou em todos os itens.")
    saida = DIR / f"diagnostico_busca_{ab[:12]}.jsonl"
    feitos = {}
    if saida.exists():
        for linha in saida.read_text(encoding="utf-8").splitlines():
            d = json.loads(linha)
            feitos[(d["estrato"], d["arxiv_id"])] = d

    busca = Busca(RAIZ / "data/processed/indice_busca", dispositivo="cpu")
    linha = {a: i for i, a in enumerate(busca.docs["arxiv_id"].to_list())}
    t0, n = time.perf_counter(), 0
    with saida.open("a", encoding="utf-8") as f:
        for it in itens:
            k = (it["estrato"], it["arxiv_id"])
            if k in feitos:
                continue
            ra = a_por_item[k]
            d = {"estrato": k[0], "arxiv_id": k[1],
                 "hyde": posicao(busca, ra.consulta, linha[k[1]]),
                 "pergunta": posicao(busca, it["pergunta"], linha[k[1]]),
                 "p_em_a": ra.posicao_p is not None}
            f.write(json.dumps(d) + "\n")
            f.flush()
            feitos[k] = d
            n += 1
            if n % 50 == 0:
                print(f"{len(feitos)}/{len(itens)} · {(time.perf_counter() - t0) / n:.1f} s/item",
                      flush=True)

    resumo = {"exploratorio": True, "nao_decide": "DOC-13 §9.1",
              "assinatura_bracos": ab, "indice_docs": busca.docs.height, "por_estrato": {}}
    for estrato in ("primario", "pos_corte"):
        ds = [d for d in feitos.values() if d["estrato"] == estrato]
        # Conferência: a consulta gravada tem de reproduzir o braço A.
        divergem = sum(1 for d in ds if (d["hyde"] <= b.K_FONTES) != d["p_em_a"])
        por = {}
        for consulta in ("hyde", "pergunta"):
            pos = np.array([d[consulta] for d in ds])
            por[consulta] = {
                "recall": {f"@{c}": round(float((pos <= c).mean()), 4) for c in CORTES},
                "mediana_da_posicao": int(np.median(pos)),
                "fora_das_6": {  # onde P está quando não veio
                    "7-20": int(((pos > 6) & (pos <= 20)).sum()),
                    "21-50": int(((pos > 20) & (pos <= 50)).sum()),
                    "51-1000": int(((pos > 50) & (pos <= 1000)).sum()),
                    ">1000": int((pos > 1000).sum())}}
        resumo["por_estrato"][estrato] = {"itens": len(ds), "divergem_do_braco_A": divergem,
                                          **por}
    arq = AVALIACAO / "assistente_diagnostico_busca.json"
    arq.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumo["por_estrato"], ensure_ascii=False, indent=1))
    print(f"→ {arq.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
