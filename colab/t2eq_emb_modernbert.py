"""Passo 1 do caminho B no Google Colab: o ModernBERT-base como ΦEmb, sem pré-treino.

O conteúdo das células, mantido aqui como `.py` para ficar sob controle de versão e
ser testável. `scripts/gerar_notebook_colab.py` monta o `.ipynb`.

## Por que o Colab, e não o Kaggle

O braço é o mesmo que `kaggle/t2eq_emb.py` roda com `VARIANTE="modernbert"`, e está
montado lá desde 2026-09-17 esperando cota. No Colab ele não gasta a cota do Kaggle —
que fica para o caminho B, de 7 h por braço — e a placa é a mesma T4.

## ⚠️ Os hiperparâmetros são os do T1f, e um teste confere

`tests/regression/test_colab_t2eq_emb.py` lê as constantes desta célula e da do Kaggle
pela AST e exige que sejam iguais. Se divergirem, este braço deixa de ser comparável
com o controle e o tratado — e a comparação é a única razão de ele existir.

## ⚠️ O que o Colab tem que o Kaggle não tem, e o contrário

A favor: o `--out` vai para o Drive, então **o treino RETOMA se a sessão cair**
(`train_embedding.py` grava `progresso.json` e `estado_treino.pt`). No Kaggle a sessão
morta leva o trabalho junto.

Contra: a T4 gratuita não é garantida, e uma sessão pode vir só com CPU. A primeira
célula ABORTA nesse caso — treinar isto em CPU levaria dias e a descoberta viria
depois de horas.

## ⚠️ Os pares têm de ser OS MESMOS bytes

O controle e o tratado treinaram nos pares do pacote do T1a. A célula confere o blake3
dos dois arquivos contra os hashes do `MANIFESTO.json` daquele pacote, que estão
escritos abaixo. Sem isso, "os mesmos 200 mil pares" seria uma afirmação, e não um
fato — e é o que faz os três números serem comparáveis.
"""
from __future__ import annotations

# Do `data/processed/kaggle_t2eq_emb/MANIFESTO.json`, pacote `git_sha` 757f58f.
HASHES = {
    "pares_treino.parquet":
        "6d4737e7d64e05ddad8071abbc83f98e1dece345ed5e0d3b9dfd871dcf63beb7",
    "pares_validacao.parquet":
        "a188bce65066fdc19aba30580188bf56711412a552316fd88d695157678084b4",
}

PASTA_PADRAO = "/content/drive/MyDrive/phifm"

SETUP = r'''
# ─── 1. GPU, Drive, dependências e código ──────────────────────────────────
import json, os, subprocess, sys, time
from pathlib import Path

import torch

# ⚠️ ABORTA sem GPU. A T4 gratuita do Colab não é garantida, e uma sessão de CPU
# levaria dias neste treino — a descoberta viria depois de horas.
assert torch.cuda.is_available(), (
    "sem GPU. Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU. "
    "Se não houver T4 disponível, tente mais tarde: em CPU isto não termina.")
print(f"torch {torch.__version__} · {torch.cuda.get_device_name(0)} · "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from google.colab import drive
drive.mount("/content/drive")

PASTA = Path("__PASTA__")          # onde estão os pares e onde o treino vai gravar
DADOS = PASTA / "pares"
SAIDA_BASE = PASTA / "runs"
SAIDA_BASE.mkdir(parents=True, exist_ok=True)
for nome in ("pares_treino.parquet", "pares_validacao.parquet"):
    assert (DADOS / nome).exists(), (
        f"{DADOS / nome} não existe. Suba os dois parquets do pacote "
        "`data/processed/kaggle_t2eq_emb` para essa pasta do Drive.")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "polars", "blake3", "tokenizers"], check=True)

# ── O código, do commit exato ──────────────────────────────────────────────
SHA = "__SHA__"
REPO = "__REPO__"
CODIGO = Path("/content/codigo")
if not CODIGO.exists():
    subprocess.run(["git", "clone", "--quiet", f"https://github.com/{REPO}.git",
                    str(CODIGO)], check=True)
subprocess.run(["git", "-C", str(CODIGO), "fetch", "--quiet", "--depth", "1",
                "origin", SHA], check=True)
subprocess.run(["git", "-C", str(CODIGO), "checkout", "--quiet", SHA], check=True)
FONTE = CODIGO / "src"
assert (CODIGO / "scripts/train_embedding.py").exists()
print(f"código em {CODIGO} · SHA {SHA[:7]}")

# ── Os pares TÊM de ser os mesmos bytes dos outros dois braços ─────────────
from blake3 import blake3

HASHES = __HASHES__
for nome, esperado in HASHES.items():
    h = blake3()
    with open(DADOS / nome, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    obtido = h.hexdigest()
    assert obtido == esperado, (
        f"{nome}: blake3 {obtido[:16]}… e o pacote do T1a declara {esperado[:16]}…\n"
        "Estes não são os pares em que o controle e o tratado treinaram, e os três "
        "números deixariam de ser comparáveis.")
print(f"✅ {len(HASHES)} arquivos conferidos por blake3 contra o pacote do T1a")
'''

REGRA = r"""
  ── A REGRA, escrita ANTES (a mesma de kaggle/t2eq_emb.py) ────────────────

  A PERGUNTA (ADR-0003, passo 1 do caminho B): quanto um encoder GERAL de
  150 M, sem pré-treino nenhum em Física, faz nos MESMOS 200 mil pares e com os
  MESMOS hiperparâmetros dos braços do §2.3? É a barra que um pré-treino
  continuado do ModernBERT-base teria de superar.

  MEDIDA PRIMÁRIA: nDCG@10 no protocolo do G1, LOCAL, na mesma sessão que o
  tratado@200k (0,4712) e o controle@200k (0,3872); IC por bootstrap pareado
  por ITEM contra o tratado.

  Os desfechos:

    MODERNBERT À FRENTE ->  a base geral de 150 M já supera o nosso 48 M
    do tratado              tratado. O pré-treino continuado só se justifica
    (IC exclui zero)        se puder superar ESTE número; a distância entre os
                            dois vai junto, e a decisão do passo 2 é do dono.

    EMPATE                ->  um 48 M do zero, com mascaramento de equações, a
    (IC cruza zero)           0,6 B tokens, iguala um 150 M geral treinado em
                              2 T. Forte a favor do caminho B.

    TRATADO À FRENTE      ->  idem, mais forte.
    (IC exclui zero)

  ⚠️ Não é uma comparação limpa de uma variável: o ModernBERT tem 3x os
     parâmetros, outro tokenizer e 2 T tokens de pré-treino geral. É a barra
     do produto, não uma ablação.

  ⚠️ Memória: lote 128 sem GradCache, como nos outros braços. Se estourar na
     T4, a falha aparece nos primeiros passos. O plano B (`--sub-lote` com
     `--sem-amp`) muda a numérica e o tempo, e NÃO entra sem nova decisão.

  ⚠️ A COMPARAÇÃO NÃO ACONTECE AQUI. Esta célula treina um braço e para; os
     três checkpoints são medidos LOCAL, na mesma sessão, pelo protocolo do G1.
"""

TREINO = r'''
# ─── 2. Treinar ────────────────────────────────────────────────────────────
# ⚠️ Iguais aos do T1f e aos dos outros dois braços, conferidos por teste. O
# `--lote 128` define os 127 negativos do InfoNCE; mudar qualquer um mudaria a
# tarefa junto com a base.
BASE = "answerdotai/ModernBERT-base"
LOTE = 128
MAX_TOKENS = 192
SEMENTE = 17
N_CANDIDATOS = 1000
PASSOS_AVAL = 200
MAX_PARES = 200_000

REGRA = r"""__REGRA__"""
print(REGRA, flush=True)

SAIDA = SAIDA_BASE / "phiemb-t2eq-modernbert-200k"
LOG = SAIDA_BASE / "treino_t2eq_emb_modernbert.log"
# ⚠️ A saída vai para o DRIVE de propósito: `train_embedding.py` grava
# `progresso.json` e `estado_treino.pt`, então uma sessão derrubada retoma de onde
# parou. É a única vantagem real do Colab sobre o Kaggle aqui — e ela some se a
# saída ficar em `/content`, que morre com a sessão.
cmd = [sys.executable, "-u", str(CODIGO / "scripts/train_embedding.py"),
       "--pares", str(DADOS), "--out", str(SAIDA), "--base", BASE,
       "--lote", str(LOTE), "--max-tokens", str(MAX_TOKENS),
       "--max-pares", str(MAX_PARES), "--passos-aval", str(PASSOS_AVAL),
       "--n-candidatos", str(N_CANDIDATOS), "--dispositivo", "cuda",
       "--semente", str(SEMENTE)]
print("$ " + " ".join(cmd), flush=True)

ambiente = dict(os.environ, PYTHONPATH=str(FONTE), PYTHONIOENCODING="utf-8",
                TOKENIZERS_PARALLELISM="false")
t0 = time.perf_counter()
with open(LOG, "a", encoding="utf-8") as fh:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, encoding="utf-8",
                            errors="replace", env=ambiente, cwd=str(CODIGO))
    for linha in proc.stdout:
        fh.write(linha)
        fh.flush()
        print(linha, end="", flush=True)
    codigo_saida = proc.wait()
custo = round(time.perf_counter() - t0, 1)
assert codigo_saida == 0, (
    f"o treino saiu com {codigo_saida}. O log está em {LOG}, e rodar esta célula de "
    "novo RETOMA do último checkpoint no Drive.")
print(f"\ntreino: {custo/3600:.2f} h")
'''

RESUMO = r'''
# ─── 3. O que baixar, e o que ele mede ─────────────────────────────────────
melhor = SAIDA.parent / f"{SAIDA.name}-melhor"
assert melhor.exists(), (
    f"{melhor} não existe — o treino não elegeu checkpoint. Sem ele não há o que "
    "baixar.")
m = json.loads((melhor / "melhor.json").read_text(encoding="utf-8"))

resumo = {
    "experimento": "t2eq_emb", "variante": "modernbert", "onde": "colab",
    "base": BASE, "git_sha_codigo": SHA,
    "pares": {"de": "pacote kaggle_t2eq_emb (blake3 conferido)",
              "max_pares": MAX_PARES},
    "hiperparametros": {"lote": LOTE, "negativos_infonce": LOTE - 1,
                        "max_tokens": MAX_TOKENS, "semente": SEMENTE,
                        "n_candidatos_aval": N_CANDIDATOS,
                        "passos_aval": PASSOS_AVAL},
    "gpu": torch.cuda.get_device_name(0),
    "custo_segundos": custo,
    "melhor": m,
    "regra": REGRA,
    "nota": ("Pool INTERNO do treino, não o do G1. A comparação com o controle e o "
             "tratado é LOCAL, na mesma sessão, pelo protocolo do G1."),
}
(SAIDA_BASE / "t2eq_emb_modernbert.json").write_text(
    json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

print("=" * 74)
print(f"  passo 1 · ModernBERT-base · {MAX_PARES:,} pares · {custo/3600:.2f} h")
print(f"  melhor no pool interno (n={N_CANDIDATOS}): nDCG@10 {m.get('ndcg_10')} · "
      f"passo {m.get('passo')}")
print("  ⚠️ pool INTERNO do treino, não o do G1 — a comparação é local.")
print("=" * 74)
print(f"\nNo Drive, em {SAIDA_BASE}:")
print(f"  {melhor.name}/   (o checkpoint — é o que tem de vir para a máquina local)")
print(f"  t2eq_emb_modernbert.json · {LOG.name}")
'''

CELULAS = (SETUP, TREINO, RESUMO)
