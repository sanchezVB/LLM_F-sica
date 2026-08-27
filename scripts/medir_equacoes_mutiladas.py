"""As equacoes do peS2o foram removidas na extracao? Padrao ou documento ruim?

Assinatura procurada: operador matematico ORFAO — um ' = ', ' < ', ' > ' sem
operando alfanumerico de um dos lados. Numa equacao intacta o operador tem os dois
lados; num texto de que a equacao foi arrancada, sobra o operador solto.
"""
from __future__ import annotations

import glob
import random
import re
import sys

import polars as pl

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ' = ' precedido ou seguido por algo que NAO e alfanumerico nem fecha-parenteses
ORFAO = re.compile(r"(?<![\w\)\]])\s[=<>]\s|\s[=<>]\s(?![\w\(\[\-\+])")
CTRL = re.compile(r"\\[a-zA-Z]+\*?")

partes = sorted(glob.glob("data/processed/pes2o_fisica/part-*.parquet"))
d = (pl.scan_parquet(partes[-40:]).select("texto").head(4000)
     .collect(engine="streaming"))
txts = [t for t in d["texto"].to_list() if t and len(t) > 2000]
print(f"peS2o texto pleno: {len(txts):,} documentos com mais de 2.000 chars")

com_orfao = sum(1 for t in txts if ORFAO.search(t))
orfaos_por_doc = sum(len(ORFAO.findall(t)) for t in txts) / max(len(txts), 1)
com_latex = sum(1 for t in txts if CTRL.search(t))
print(f"  com operador ORFAO   : {100*com_orfao/len(txts):>5.1f}%")
print(f"  orfaos por documento : {orfaos_por_doc:>6.1f}")
print(f"  com sequencia LaTeX  : {100*com_latex/len(txts):>5.1f}%")

# o mesmo teste nos resumos do arXiv, que sao a referencia de matematica intacta
a = (pl.scan_parquet("data/processed/pares/pares_validacao.parquet")
     .select("positivo").head(4000).collect(engine="streaming"))
ax = [t for t in a["positivo"].to_list() if t]
print()
print(f"arXiv resumos: {len(ax):,} documentos")
print(f"  com operador ORFAO   : {100*sum(1 for t in ax if ORFAO.search(t))/len(ax):>5.1f}%")
print(f"  orfaos por documento : {sum(len(ORFAO.findall(t)) for t in ax)/len(ax):>6.1f}")

print()
print("=" * 74)
print("TRES trechos de documentos DIFERENTES do texto pleno, ao redor de um ' = ':")
r = random.Random(11)
mostrados = 0
for t in r.sample(txts, min(60, len(txts))):
    m = ORFAO.search(t)
    if not m or mostrados >= 3:
        continue
    mostrados += 1
    i = max(m.start() - 240, 0)
    print(f"\n[{mostrados}] {t[i:i + 480]!r}")
