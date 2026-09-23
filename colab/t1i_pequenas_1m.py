"""T1i — `gte-small` e `bge-small` com 1 M de pares contra o ΦEmb do sistema, no Colab.

O conteúdo das células, mantido aqui como `.py` para ficar sob controle de versão e
ser testável. `scripts/gerar_notebook_colab.py --celulas t1i_pequenas_1m` monta o `.ipynb`.

## De onde vem a pergunta

O T1h mostrou, com regra escrita antes, que duas bases do porte do MiniLM capturam ~60% do
ganho de base do GTE-base a 1,75× do custo de embutir, em vez de 4,1–4,4×: `gte-small`
(+0,0417) e `bge-small` (+0,0403) sobre o MiniLM-L6, a 200 mil pares. Pela mesma regra, as
duas sobem de volume — 1 M de pares — contra o ΦEmb do sistema (MiniLM-L6 a 6 M, 0,6223).

## ⚠️ Os pares são os do T1g, e é isso que torna os três comparáveis

O T1g mediu o GTE-base a 1 M com o pacote do T1a 1,5 M limitado por `--max-pares
1_000_000`. Aqui é o MESMO pacote, a MESMA semente, o MESMO limite: as três bases veem o
mesmo milhão de pares. O arquivo de validação desse pacote é byte a byte o mesmo do pacote
de 400 mil que já está no Drive (blake3 `a188bce6…` nos dois), então só o de TREINO sobe.

## ⚠️ ~5,6 h de Colab, com a aba aberta

A ~100 pares/s (medido no T1h), 1 M de pares são ~2,8 h por base. As saídas vão para o
Drive e o treino RETOMA se a sessão cair: rodar as células 1 e 2 de novo pula a base
concluída e continua a que parou. A conta é Colab Pro: aba aberta e PC acordado.
"""
from __future__ import annotations

# Do `data/processed/kaggle_t1a15/MANIFESTO.json`, pacote `git_sha` e33561d — os pares do
# T1g. A validação é a mesma do pacote de 400 mil (`kaggle_t2eq_emb`).
HASHES = {
    "pares_treino.parquet":
        "88e2d79e2136745f4d7bed95a8ae29950d22afaaeb9b40fd28cc9142bcf85674",
    "pares_validacao.parquet":
        "a188bce65066fdc19aba30580188bf56711412a552316fd88d695157678084b4",
}

BASES = {
    "gte-small": "thenlper/gte-small",
    "bge-small": "BAAI/bge-small-en-v1.5",
}

PASTA_PADRAO = "/content/drive/MyDrive/phifm"

# O que o cabeçalho do notebook manda subir. O de validação já está no Drive.
INSTRUCAO = ("suba `data/processed/kaggle_t1a15/pares_treino.parquet` (691 MB) para "
             "`__PASTA__/pares_15m/`. A validação já está em `__PASTA__/pares/` e é o "
             "mesmo arquivo.")

SETUP = r'''
# ─── 1. GPU, Drive, dependências, código e os pares de 1,5 M ───────────────
import json, os, shutil, subprocess, sys, time
from pathlib import Path

import torch

assert torch.cuda.is_available(), (
    "sem GPU. Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU.")
print(f"torch {torch.__version__} · {torch.cuda.get_device_name(0)} · "
      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from google.colab import drive
drive.mount("/content/drive")

PASTA = Path("__PASTA__")
SAIDA_BASE = PASTA / "runs" / "t1i"
SAIDA_BASE.mkdir(parents=True, exist_ok=True)

# O treino de 1,5 M sobe para `pares_15m/`; a validação é a de `pares/`, que já está lá
# e é o MESMO arquivo. Os dois vão para o disco da sessão: ler 691 MB pelo Drive a cada
# época seria lento, e o treino abre os pares uma vez.
ORIGENS = {"pares_treino.parquet": PASTA / "pares_15m" / "pares_treino.parquet",
           "pares_validacao.parquet": PASTA / "pares" / "pares_validacao.parquet"}
for nome, origem in ORIGENS.items():
    assert origem.exists(), (
        f"{origem} não existe. Suba `data/processed/kaggle_t1a15/pares_treino.parquet` "
        "para a pasta `phifm/pares_15m/` do Drive (a validação já está em `phifm/pares/`).")
DADOS = Path("/content/pares_15m")
DADOS.mkdir(parents=True, exist_ok=True)
for nome, origem in ORIGENS.items():
    # Recopia se o tamanho não bate: uma cópia interrompida deixaria um arquivo
    # parcial que o blake3 abaixo recusaria sem que a célula conseguisse se consertar.
    alvo = DADOS / nome
    if not alvo.exists() or alvo.stat().st_size != origem.stat().st_size:
        shutil.copy2(origem, alvo)

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

# ── Os pares TÊM de ser os do T1g ──────────────────────────────────────────
from blake3 import blake3

HASHES = __HASHES__
for nome, esperado in HASHES.items():
    h = blake3()
    with open(DADOS / nome, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    assert h.hexdigest() == esperado, (
        f"{nome}: blake3 {h.hexdigest()[:16]}… e o pacote do T1a 1,5 M declara "
        f"{esperado[:16]}…\nSem os mesmos pares do T1g, as três bases a 1 M deixam de "
        "ser comparáveis.")
print(f"✅ pares do T1g conferidos por blake3 · saídas em {SAIDA_BASE}")
'''

REGRA = r"""
  ── A REGRA, escrita ANTES (2026-09-23) ───────────────────────────────────

  A PERGUNTA (produto): uma base pequena, ajustada em 1 M de pares, supera o
  ΦEmb do sistema — o MiniLM-L6 ajustado em 6 M, nDCG@10 0,6223 — a um custo de
  embutir de ~1,75x, em vez dos 4,4x do GTE-base?

  MEDIDA PRIMÁRIA: nDCG@10 no protocolo do G1, LOCAL, na mesma sessão que o
  sistema; cada candidata contra ele, bootstrap pareado por ITEM. São 2
  comparações contra o mesmo controle: Bonferroni, IC de 97,5% (0,05/2).

  Por candidata:
    ACIMA  (IC exclui 0, acima)  -> existe um encoder melhor que o do sistema, a
                                    1/6 do volume e a ~1,75x do custo. A troca
                                    vai ao dono com a margem e o custo medidos.
    EMPATE (IC cruza 0)          -> alcança o sistema e não o passa; com custo
                                    maior, o sistema fica.
    ABAIXO (IC exclui 0, abaixo) -> o volume ainda manda a 1 M.

  DIAGNÓSTICO (não decide): contra o GTE-base@1M do T1g (0,6211), nos mesmos
  pares — a base grande contra as pequenas, no mesmo volume.

  ⚠️ Não é ablação de uma variável contra o sistema: muda base E volume. É a
     pergunta do produto, declarada assim antes do número.
  ⚠️ Uma semente por base. O nDCG que este notebook imprime é do pool INTERNO.
"""

TREINO = r'''
# ─── 2. Treinar as duas bases, uma depois da outra ─────────────────────────
# ⚠️ A receita de todos os ajustes, conferida por teste; o volume é o do T1g.
LOTE = 128
MAX_TOKENS = 192
SEMENTE = 17
N_CANDIDATOS = 1000
PASSOS_AVAL = 200
MAX_PARES = 1_000_000

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
    saida = SAIDA_BASE / f"phiemb-t1i-{nome}-1m"
    if concluido(saida) and (saida.parent / f"{saida.name}-melhor").exists():
        print(f"\n⏭️  {nome}: já concluído em {saida.name} — pulando", flush=True)
        continue
    log = SAIDA_BASE / f"treino_t1i_{nome}.log"
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
        f"{nome} saiu com {codigo_saida}; log em {log}. Rodar as células 1 e 2 de novo "
        "pula a base concluída e RETOMA esta do último checkpoint no Drive.")
    print(f"✅ {nome}: {custos[nome] / 3600:.2f} h", flush=True)
'''

RESUMO = r'''
# ─── 3. O que baixar ───────────────────────────────────────────────────────
resumo = {"experimento": "t1i", "onde": "colab", "git_sha_codigo": SHA,
          "gpu": torch.cuda.get_device_name(0),
          "hiperparametros": {"lote": LOTE, "max_tokens": MAX_TOKENS,
                              "semente": SEMENTE, "max_pares": MAX_PARES,
                              "n_candidatos_aval": N_CANDIDATOS,
                              "passos_aval": PASSOS_AVAL},
          "custos_de_treino_s_desta_sessao": custos, "bases": {}, "regra": REGRA,
          "nota": ("nDCG do pool INTERNO do treino; a comparação é LOCAL, pelo "
                   "protocolo do G1, contra o ΦEmb do sistema.")}
print("=" * 74)
for nome, base in BASES.items():
    melhor = SAIDA_BASE / f"phiemb-t1i-{nome}-1m-melhor"
    if not melhor.exists():
        print(f"  ❌ {nome}: sem checkpoint — rode as células 1 e 2 de novo")
        continue
    m = json.loads((melhor / "melhor.json").read_text(encoding="utf-8"))
    resumo["bases"][nome] = {"base": base, "melhor": m}
    print(f"  {nome:<10} nDCG@10 interno {m.get('ndcg_10'):.4f} · passo {m.get('passo')}")
(SAIDA_BASE / "t1i.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
print("=" * 74)
print(f"\nNo Drive, em {SAIDA_BASE}: as duas pastas `*-melhor` e o `t1i.json` — é o que")
print("tem de vir para D:\\LLMFísica\\. Pode baixar a pasta `t1i` inteira.")
'''

CELULAS = (SETUP, TREINO, RESUMO)
