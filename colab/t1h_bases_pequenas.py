"""T1h — bases PEQUENAS no lugar do MiniLM-L6, no Colab: a base pesa sem o custo de 4,4×?

O conteúdo das células, mantido aqui como `.py` para ficar sob controle de versão e
ser testável. `scripts/gerar_notebook_colab.py --celulas t1h_bases_pequenas` monta o
`.ipynb`.

## De onde vem a pergunta

A base pesa mais que o volume de pares: o GTE-base supera o MiniLM-L6 por +0,063 a 400
mil pares (T1f) e empata com o ΦEmb do sistema a 1 M (T1g, 0,6211 contra 0,6223). Ele
NÃO substituiu o sistema por um motivo só — custa **4,4×** para embutir o corpus (954 s
contra 217 s). A pergunta que sobra: uma base moderna do PORTE do MiniLM (~33 M) captura
o ganho de base sem esse custo?

## Uma variável: a base

Os cinco braços usam os MESMOS 200 mil pares (os mesmos bytes, conferidos por blake3), o
mesmo lote de 128, 192 tokens e semente 17. O MiniLM-L6 entra como REFERÊNCIA, treinado
aqui também: a 200 mil pares ele nunca foi medido, e comparar contra o número de 400 mil
mudaria o volume junto com a base.

## ⚠️ Duas bases foram feitas para usar prefixo, e aqui não usam

O `e5-small-v2` espera `query: `/`passage: ` e o `bge-small-en-v1.5` recomenda uma
instrução na consulta. O treino e a avaliação deste projeto não põem prefixo em nenhuma
base. O ajuste contrastivo adapta o modelo à entrada sem prefixo, mas o ponto de partida
fica fora do uso previsto — é uma desvantagem declarada, e a mesma para as duas.

## ⚠️ O que o Colab tem, e o que exige

As saídas vão para o Drive e o treino RETOMA se a sessão cair: rodar a célula 2 de novo
pula as bases concluídas e continua a que parou. Mas a conta é Colab Pro, não Pro+: a aba
tem de ficar aberta e o PC acordado durante as ~3 h. Cerca de 3 GB no Drive.
"""
from __future__ import annotations

# Do `data/processed/kaggle_t2eq_emb/MANIFESTO.json`, pacote `git_sha` 757f58f — os
# MESMOS bytes dos ajustes do §2.3, do passo 1, do GTE@200k e do CPT.
HASHES = {
    "pares_treino.parquet":
        "6d4737e7d64e05ddad8071abbc83f98e1dece345ed5e0d3b9dfd871dcf63beb7",
    "pares_validacao.parquet":
        "a188bce65066fdc19aba30580188bf56711412a552316fd88d695157678084b4",
}

# A referência primeiro: é a mais rápida, e sem ela nenhuma das outras se lê.
BASES = {
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "gte-small": "thenlper/gte-small",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "e5-small": "intfloat/e5-small-v2",
    "minilm-l12": "sentence-transformers/all-MiniLM-L12-v2",
}
REFERENCIA = "minilm-l6"

PASTA_PADRAO = "/content/drive/MyDrive/phifm"

SETUP = r'''
# ─── 1. GPU, Drive, dependências, código e os pares ────────────────────────
import json, os, subprocess, sys, time
from pathlib import Path

import torch

# ⚠️ ABORTA sem GPU. A T4 do Colab não é garantida, e em CPU isto levaria dias.
assert torch.cuda.is_available(), (
    "sem GPU. Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU.")
print(f"torch {torch.__version__} · {torch.cuda.get_device_name(0)} · "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from google.colab import drive
drive.mount("/content/drive")

PASTA = Path("__PASTA__")
DADOS = PASTA / "pares"
SAIDA_BASE = PASTA / "runs" / "t1h"
SAIDA_BASE.mkdir(parents=True, exist_ok=True)
for nome in ("pares_treino.parquet", "pares_validacao.parquet"):
    assert (DADOS / nome).exists(), (
        f"{DADOS / nome} não existe. Os pares do pacote `kaggle_t2eq_emb` ficam "
        "nessa pasta do Drive — são os mesmos do passo 1 e do CPT.")

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

# ── Os pares TÊM de ser os mesmos bytes dos outros ajustes ─────────────────
from blake3 import blake3

HASHES = __HASHES__
for nome, esperado in HASHES.items():
    h = blake3()
    with open(DADOS / nome, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    assert h.hexdigest() == esperado, (
        f"{nome}: blake3 {h.hexdigest()[:16]}… e o pacote declara {esperado[:16]}…\n"
        "Estes não são os pares dos outros ajustes, e os números deixariam de ser "
        "comparáveis.")
print(f"✅ pares conferidos por blake3 · saídas em {SAIDA_BASE}")
'''

REGRA = r"""
  ── A REGRA, escrita ANTES (2026-09-23) ───────────────────────────────────

  A PERGUNTA: uma base moderna do porte do MiniLM-L6 (~33 M), ajustada na mesma
  receita, captura o ganho de base que o GTE-base mostrou — sem o custo de 4,4x?

  MEDIDA PRIMÁRIA: nDCG@10 no protocolo do G1, LOCAL, os cinco na mesma sessão.
  Cada candidata contra o MiniLM-L6@200k treinado AQUI, bootstrap pareado por
  ITEM. São 4 comparações contra o mesmo controle: correção de Bonferroni, IC de
  98,75% (0,05/4).

  Por candidata:
    ACIMA  (IC exclui 0, acima)  -> a base pesa também no porte pequeno.
    EMPATE (IC cruza 0)          -> no porte pequeno a base não se separa.
    ABAIXO (IC exclui 0, abaixo) -> o MiniLM-L6 já é a melhor base pequena.

  SECUNDÁRIA, e é ela que decide o PRODUTO: o custo de embutir, medido local no
  mesmo universo, em razão ao MiniLM-L6. Uma candidata ACIMA e com custo ≤ 2,5x
  vira candidata a subir de volume (1 M de pares, como o T1g) contra o ΦEmb do
  sistema. 2,5x porque o GTE-base, a 4,4x, só empatou — e uma base de 12 camadas
  com a largura do MiniLM custa cerca de 2x por FLOPs; acima de 2,5x a vantagem de
  custo que motiva este experimento deixa de existir.

  ⚠️ O e5-small e o bge-small foram feitos para usar prefixo, e aqui não usam — a
     desvantagem é declarada e igual para os dois.
  ⚠️ Uma semente por base, 200 mil pares — metade da receita do T1a.
  ⚠️ O nDCG que este notebook imprime é do pool INTERNO do treino. A comparação é
     LOCAL, pelo protocolo do G1.
"""

TREINO = r'''
# ─── 2. Treinar as cinco bases, uma depois da outra ────────────────────────
# ⚠️ Iguais aos do T1f e de todos os ajustes de 200 mil pares, conferidos por teste.
LOTE = 128
MAX_TOKENS = 192
SEMENTE = 17
N_CANDIDATOS = 1000
PASSOS_AVAL = 200
MAX_PARES = 200_000

BASES = __BASES__

REGRA = r"""__REGRA__"""
print(REGRA, flush=True)

ambiente = dict(os.environ, PYTHONPATH=str(FONTE), PYTHONIOENCODING="utf-8",
                TOKENIZERS_PARALLELISM="false")


def concluido(saida: Path) -> bool:
    """O treino grava `phiemb.json` com `completed_at` só quando termina."""
    meta = saida / "phiemb.json"
    if not meta.exists():
        return False
    return bool(json.loads(meta.read_text(encoding="utf-8")).get("completed_at"))


custos = {}
for nome, base in BASES.items():
    saida = SAIDA_BASE / f"phiemb-t1h-{nome}-200k"
    if concluido(saida) and (saida.parent / f"{saida.name}-melhor").exists():
        print(f"\n⏭️  {nome}: já concluído em {saida.name} — pulando", flush=True)
        continue
    log = SAIDA_BASE / f"treino_t1h_{nome}.log"
    cmd = [sys.executable, "-u", str(CODIGO / "scripts/train_embedding.py"),
           "--pares", str(DADOS), "--out", str(saida), "--base", base,
           "--lote", str(LOTE), "--max-tokens", str(MAX_TOKENS),
           "--max-pares", str(MAX_PARES), "--passos-aval", str(PASSOS_AVAL),
           "--n-candidatos", str(N_CANDIDATOS), "--dispositivo", "cuda",
           "--semente", str(SEMENTE)]
    print(f"\n{'=' * 74}\n▶ {nome} · {base}\n$ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    with open(log, "a", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, encoding="utf-8",
                                errors="replace", env=ambiente, cwd=str(CODIGO))
        for linha in proc.stdout:
            fh.write(linha)
            fh.flush()
            print(linha, end="", flush=True)
        codigo_saida = proc.wait()
    custos[nome] = round(time.perf_counter() - t0, 1)
    assert codigo_saida == 0, (
        f"{nome} saiu com {codigo_saida}; log em {log}. Rodar esta célula de novo "
        "pula as bases concluídas e RETOMA esta do último checkpoint no Drive.")
    print(f"✅ {nome}: {custos[nome] / 60:.1f} min", flush=True)
'''

RESUMO = r'''
# ─── 3. O que baixar ───────────────────────────────────────────────────────
resumo = {"experimento": "t1h", "onde": "colab", "git_sha_codigo": SHA,
          "gpu": torch.cuda.get_device_name(0),
          "hiperparametros": {"lote": LOTE, "max_tokens": MAX_TOKENS,
                              "semente": SEMENTE, "max_pares": MAX_PARES,
                              "n_candidatos_aval": N_CANDIDATOS,
                              "passos_aval": PASSOS_AVAL},
          "custos_de_treino_s_desta_sessao": custos, "bases": {}, "regra": REGRA,
          "nota": ("nDCG do pool INTERNO do treino; a comparação é LOCAL, pelo "
                   "protocolo do G1, com o custo de embutir medido lá.")}
print("=" * 74)
for nome, base in BASES.items():
    melhor = SAIDA_BASE / f"phiemb-t1h-{nome}-200k-melhor"
    if not melhor.exists():
        print(f"  ❌ {nome}: sem checkpoint — rode a célula 2 de novo")
        continue
    m = json.loads((melhor / "melhor.json").read_text(encoding="utf-8"))
    resumo["bases"][nome] = {"base": base, "melhor": m}
    print(f"  {nome:<11} nDCG@10 interno {m.get('ndcg_10'):.4f} · passo {m.get('passo')}")
(SAIDA_BASE / "t1h.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
print("=" * 74)
print(f"\nNo Drive, em {SAIDA_BASE}: as cinco pastas `*-melhor` e o `t1h.json` — é o")
print("que tem de vir para D:\\LLMFísica\\models\\. Pode baixar a pasta `t1h` inteira.")
'''

CELULAS = (SETUP, TREINO, RESUMO)
