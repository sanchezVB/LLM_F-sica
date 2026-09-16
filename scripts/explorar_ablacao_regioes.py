#!/usr/bin/env python3
"""EXPLORATÓRIA: a diferença das diferenças do §2.3 separada em display e inline.

    PYTHONPATH=src python scripts/explorar_ablacao_regioes.py \\
        --controle data/processed/avaliacao/t2eq_mlm_controle_uniforme.json \\
        --tratado  data/processed/avaliacao/t2eq_mlm_tratado_uniforme.json \\
        --dados    data/processed/phienc_aval_E \\
        --out      data/processed/avaliacao/t2eq_exploratoria_regioes.json

## ⚠️ Feita DEPOIS do número, e por isso não decide nada

A primária da regra (`kaggle/t2eq_tratado.py`) usa a região `BIT_MATH` — equação em
display e inline juntas — e deu −0,0040 [−0,0058; −0,0022], o negativo. O tratamento
só escolheu equações em DISPLAY. Separar as duas pergunta ONDE o efeito mora, e é
uma pergunta legítima; mas escolhida depois de ver o resultado, qualquer recorte que
saísse positivo seria um jardim de caminhos bifurcados. O artefato leva
`"exploratoria": true` e a leitura não toca o desfecho.

## Sem rodar o modelo de novo

As posições da prova uniforme são determinísticas em `(semente, sequência)`, então
o rótulo display/inline de cada token avaliado se reconstrói das marcas. A
reconstrução é CONFERIDA contra o rótulo de matemática gravado no artefato, token a
token; se divergir, o script levanta — o acerto iria para a região errada.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.core.console import utf8  # noqa: E402

utf8()

from phifm.eval.mlm_regiao import (  # noqa: E402
    diferenca_das_diferencas,
    posicoes_mascaradas,
)
from phifm.training.pretrain.dados import (  # noqa: E402
    BIT_DISPLAY,
    BIT_MATH,
    ConfigDados,
    Fluxo,
)


def rotulos_por_token(artefato: dict, dados: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(math, display, sequencia)` por token avaliado, na ordem do artefato."""
    proto = artefato["protocolo"]
    if proto.get("prova") != "uniforme":
        raise SystemExit("a exploração é sobre a prova UNIFORME, a da primária")
    fl = Fluxo(ConfigDados(raiz=dados, contexto=proto["contexto"],
                           semente=proto["semente"]))
    especiais = list(range(proto["n_ids_especiais"]))
    m, d, s = [], [], []
    for indice in artefato["indices_sorteados"]:
        a, b = indice * fl.cfg.contexto, (indice + 1) * fl.cfg.contexto
        ids = np.asarray(fl.tokens[a:b], dtype=np.int64)
        marcas = np.asarray(fl.marcas[a:b])
        pos = posicoes_mascaradas(
            ids.size, semente=proto["semente"], indice=indice,
            fracao=proto["fracao_mascara"],
            proibidas=np.flatnonzero(np.isin(ids, especiais)).astype(np.int64))
        m.extend(((marcas[pos] & BIT_MATH) != 0).astype(int).tolist())
        d.extend(((marcas[pos] & BIT_DISPLAY) != 0).astype(int).tolist())
        s.extend([indice] * int(pos.size))
    m, d, s = np.array(m), np.array(d), np.array(s)
    gravado = np.array(artefato["e_equacao_por_token"])
    if m.shape != gravado.shape or not np.array_equal(m, gravado):
        raise SystemExit(
            "a reconstrução das posições NÃO bate com o rótulo gravado no artefato: "
            "o protocolo mudou ou a fatia não é a mesma. O acerto iria para a região "
            "errada.")
    return m, d, s


def linhas(acertos: np.ndarray, regiao: np.ndarray, prosa: np.ndarray,
           seq: np.ndarray, indices: list[int]) -> list[list[int]]:
    """`[indice, acertos_reg, total_reg, acertos_prosa, total_prosa]` por sequência.

    Os tokens chegam agrupados por sequência, na ordem de `indices` (crescente) —
    somas por bloco em vez de uma máscara por sequência.
    """
    blocos, inicio = np.unique(seq, return_index=True)
    if blocos.tolist() != list(indices):
        raise SystemExit("há sequência sem token avaliado, ou fora de ordem")

    def soma(x: np.ndarray) -> np.ndarray:
        return np.add.reduceat(x.astype(np.int64), inicio)

    cols = (soma(acertos * regiao), soma(regiao), soma(acertos * prosa), soma(prosa))
    return [[int(i), *(int(c[k]) for c in cols)] for k, i in enumerate(indices)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controle", type=Path, required=True)
    ap.add_argument("--tratado", type=Path, required=True)
    ap.add_argument("--dados", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    c = json.loads(a.controle.read_text(encoding="utf-8"))
    t = json.loads(a.tratado.read_text(encoding="utf-8"))
    if c["indices_sorteados"] != t["indices_sorteados"] or \
            c["e_equacao_por_token"] != t["e_equacao_por_token"]:
        raise SystemExit("os dois braços não foram medidos nos mesmos tokens")
    m, d, s = rotulos_por_token(c, a.dados)
    idx = c["indices_sorteados"]
    ac_c, ac_t = np.array(c["acertos_por_token"]), np.array(t["acertos_por_token"])
    prosa = m == 0
    regioes = {"display": (m == 1) & (d == 1), "inline": (m == 1) & (d == 0),
               "matematica_toda": m == 1}

    resultado = {"exploratoria": True,
                 "aviso": ("Análise feita DEPOIS da primária. Não altera o desfecho da "
                           "regra; responde onde o efeito mora."),
                 "tokens": {k: int(v.sum()) for k, v in regioes.items()} | {
                     "prosa": int(prosa.sum())},
                 "regioes": {}}
    for nome, reg in regioes.items():
        resultado["regioes"][nome] = diferenca_das_diferencas(
            linhas(ac_c, reg, prosa, s, idx), linhas(ac_t, reg, prosa, s, idx),
            n_boot=a.n_boot)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                     encoding="utf-8")

    print()
    print("=" * 78)
    print("  EXPLORATÓRIA · §2.3 · diferença das diferenças por região (prova uniforme)")
    print("  ⚠️ feita depois da primária; não altera o desfecho")
    print("=" * 78)
    print(f"  {'região':<16} {'tokens':>8}  {'controle':>9} {'tratado':>9}  "
          f"{'tratado − controle':>20}")
    for nome in ("display", "inline", "matematica_toda"):
        r = resultado["regioes"][nome]
        print(f"  {nome:<16} {resultado['tokens'][nome]:>8,}  "
              f"{r['controle']['acuracia_equacao']:>9.4f} "
              f"{r['tratado']['acuracia_equacao']:>9.4f}  "
              f"{r['diferenca_das_diferencas']:>+9.5f} {r['ic95']}")
    r = resultado["regioes"]["display"]
    print(f"  {'prosa':<16} {resultado['tokens']['prosa']:>8,}  "
          f"{r['controle']['acuracia_prosa']:>9.4f} {r['tratado']['acuracia_prosa']:>9.4f}")
    print("=" * 78)
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
