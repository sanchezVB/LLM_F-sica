#!/usr/bin/env python3
"""As equações sobreviveram à extração? Medido por fatia do corpus.

    .venv\\Scripts\\python.exe scripts\\medir_equacoes_mutiladas.py

## A pergunta e por que ela decide um gasto

O DOC-07 §2.3 propõe **mascaramento consciente de equações** como a única adição
específica de Física ao objetivo de pré-treino do ΦEnc. É a razão científica de
treinar do zero em vez de ajustar um modelo existente. Sem equações no texto, essa
hipótese não pode ser testada — e o ΦEnc vira "mais um encoder de domínio", pior que
o ajuste fino que já está medido (23 M empatando com 335 M).

Em 2026-08-27 esta medição mostrou que o **peS2o de texto pleno tem as equações
removidas na extração**: 50,2% dos documentos com operador órfão, 3,2 por documento.
A conclusão registrada foi que o acesso pago ao fonte LaTeX do arXiv (S3 requester
pays) passava de conveniência a pré-requisito.

**Essa conclusão pulou uma verificação.** O corpus tem outras fatias, já baixadas, e
a do RedPajama-arXiv é construída a partir do **fonte LaTeX**. Medir antes de gastar é
regra deste projeto; eu recomendei o gasto sem medir a alternativa que estava no
disco.

## A assinatura: operador ÓRFÃO

Numa equação intacta o operador tem operando dos dois lados. Num texto de que a
equação foi arrancada, sobra o operador solto:

    intacto : "where E = mc^2 is the rest energy"
    mutilado: "where  =  is the rest energy"

O regex procura ` = `, ` < `, ` > ` sem operando alfanumérico de um dos lados. Não é
perfeito — tabelas e listas produzem falso positivo —, e é por isso que a comparação
é sempre **contra os resumos do arXiv**, que são a referência de matemática intacta
passando pelo mesmo regex. O número absoluto importa menos que a razão entre fatias.

⚠️ **Amostragem por SORTEIO, não `head()`.** A primeira versão usou `head()` no peS2o
e caiu na fase de resumos (1.203 ch/doc, sem LaTeX), concluindo que o peS2o não tinha
matemática nenhuma. O peS2o vem ordenado: resumos primeiro, texto pleno depois. É o
mesmo erro de amostrar o começo de um conjunto ordenado que custou dois resultados
falsos neste repositório.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

# ' = ' precedido ou seguido por algo que NAO e alfanumerico nem fecha-parenteses
ORFAO = re.compile(r"(?<![\w\)\]])\s[=<>]\s|\s[=<>]\s(?![\w\(\[\-\+])")
CTRL = re.compile(r"\\[a-zA-Z]+\*?")
DOLAR = re.compile(r"\$[^$\n]{2,}\$")
AMBIENTE = re.compile(r"\\begin\{(equation|align|eqnarray|gather|multline)")


@dataclass(frozen=True)
class Fonte:
    nome: str
    padrao: str
    coluna: str
    origem: str
    # Documentos curtos não têm equação em display nem quando o corpus é bom; medir
    # a mutilação neles mistura "sem equação" com "equação removida".
    minimo_chars: int = 2000


FONTES = (
    Fonte("arXiv resumos (REFERÊNCIA)", "data/processed/pares/pares_validacao.parquet",
          "positivo", "resumos da API do arXiv, matemática como o autor escreveu",
          minimo_chars=0),
    Fonte("RedPajama-arXiv", "data/processed/redpajama_fisica/part-*.parquet",
          "texto", "construído do FONTE LaTeX do arXiv"),
    Fonte("OpenWebMath", "data/processed/openwebmath_fisica/part-*.parquet",
          "texto", "páginas web de matemática, LaTeX preservado na extração"),
    Fonte("peS2o texto pleno", "data/processed/pes2o_fisica/part-*.parquet",
          "texto", "texto pleno do Semantic Scholar, extraído de PDF"),
)


def _amostrar(f: Fonte, n: int, semente: int) -> list[str]:
    partes = sorted(glob.glob(f.padrao)) or ([f.padrao] if Path(f.padrao).exists()
                                             else [])
    if not partes:
        return []
    # ⚠️ Sortear as PARTES, não pegar as primeiras: as fatias vêm ordenadas por
    # fase de coleta, e o começo não representa o conjunto.
    r = random.Random(semente)
    escolhidas = r.sample(partes, min(8, len(partes)))
    d = (pl.scan_parquet(escolhidas).select(f.coluna)
         .head(n * 4).collect(engine="streaming"))
    txts = [t for t in d[f.coluna].to_list() if t and len(t) >= f.minimo_chars]
    return r.sample(txts, min(n, len(txts))) if txts else []


def _medir(txts: list[str]) -> dict:
    m = len(txts)
    if not m:
        return {}
    return {
        "n": m,
        "chars_por_doc": round(sum(len(t) for t in txts) / m),
        "pct_com_orfao": round(100 * sum(1 for t in txts if ORFAO.search(t)) / m, 1),
        "orfaos_por_doc": round(sum(len(ORFAO.findall(t)) for t in txts) / m, 2),
        "pct_com_sequencia_latex": round(
            100 * sum(1 for t in txts if CTRL.search(t)) / m, 1),
        "pct_com_math_inline": round(
            100 * sum(1 for t in txts if DOLAR.search(t)) / m, 1),
        "pct_com_ambiente_de_equacao": round(
            100 * sum(1 for t in txts if AMBIENTE.search(t)) / m, 1),
        "sequencias_latex_por_doc": round(
            sum(len(CTRL.findall(t)) for t in txts) / m, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3000, help="documentos por fatia")
    p.add_argument("--semente", type=int, default=17)
    p.add_argument("--trechos", type=int, default=2,
                   help="trechos crus a imprimir por fatia, para inspeção humana")
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/avaliacao/equacoes_por_fatia.json"))
    a = p.parse_args()
    for fluxo in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            fluxo.reconfigure(encoding="utf-8", errors="replace")

    resultados: dict[str, dict] = {}
    amostras: dict[str, list[str]] = {}
    for f in FONTES:
        txts = _amostrar(f, a.n, a.semente)
        if not txts:
            print(f"⚠️ {f.nome}: nada em {f.padrao} — fatia ausente, e é isso que a "
                  "tabela vai mostrar")
            continue
        amostras[f.nome] = txts
        resultados[f.nome] = {**_medir(txts), "origem": f.origem}

    print()
    print("=" * 96)
    print("Integridade de equações por fatia do corpus")
    print("=" * 96)
    cab = (f"{'fatia':30s} {'n':>5s} {'ch/doc':>7s} {'órfão%':>7s} {'órf/doc':>8s} "
           f"{'LaTeX%':>7s} {'$..$%':>7s} {'ambnt%':>7s} {'seq/doc':>8s}")
    print(cab)
    print("-" * 96)
    for nome, v in resultados.items():
        print(f"{nome:30s} {v['n']:5d} {v['chars_por_doc']:7d} "
              f"{v['pct_com_orfao']:7.1f} {v['orfaos_por_doc']:8.2f} "
              f"{v['pct_com_sequencia_latex']:7.1f} {v['pct_com_math_inline']:7.1f} "
              f"{v['pct_com_ambiente_de_equacao']:7.1f} "
              f"{v['sequencias_latex_por_doc']:8.1f}")
    print("=" * 96)
    print("órfão% = documentos com operador matemático sem operando de um lado.")
    print("Comparar SEMPRE com a linha de referência: o mesmo regex sobre resumos do")
    print("arXiv, que são matemática intacta, dá o piso de falso positivo do método.")

    for nome, txts in amostras.items():
        if not a.trechos:
            break
        print()
        print(f"── {nome}: {a.trechos} trecho(s) ao redor de um operador ─────────")
        r = random.Random(a.semente + 1)
        vistos = 0
        for t in r.sample(txts, min(80, len(txts))):
            m = ORFAO.search(t) or DOLAR.search(t)
            if not m or vistos >= a.trechos:
                continue
            vistos += 1
            i = max(m.start() - 200, 0)
            print(f"  [{vistos}] {t[i:i + 380]!r}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "n_por_fatia": a.n, "semente": a.semente, "fatias": resultados,
        "metodo": ("operador matemático órfão: ' = ', ' < ', ' > ' sem operando "
                   "alfanumérico de um dos lados"),
        "ressalva": ("o regex tem falso positivo em tabelas e listas; a linha dos "
                     "resumos do arXiv mede esse piso. A razão entre fatias importa "
                     "mais que o valor absoluto"),
        "por_que": ("decide se o mascaramento consciente de equações do DOC-07 §2.3 "
                    "é testável com o corpus que já está no disco, ou se exige o "
                    "acesso pago ao fonte LaTeX do arXiv (S3 requester pays)"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
