"""A SECUNDÁRIA do caminho B: os dois encoders do CPT ajustados como ΦEmb, no Colab.

O conteúdo das células, mantido aqui como `.py` para ficar sob controle de versão e
ser testável. `scripts/gerar_notebook_colab.py --celulas t2eq_cpt_emb --variante <braço>`
monta o `.ipynb` de cada braço.

## O que esta célula mede, e por que ela existe

A primária do caminho B (`kaggle/t2eq_cpt.py`) foi medida em 2026-09-22 e deu **NÃO
DECIDIDO** a 0,4 B: a diferença das diferenças de MLM ficou em −0,00039 [−0,0016;
+0,0009]. O tratamento pegou — reconstruir equação inteira escondida foi de 0,0266
para 0,1999 —, mas não transferiu para a máscara pontual.

A secundária é outra pergunta, e é a que o SISTEMA usa: o mascaramento de equações
melhora o encoder **como recuperador**, depois do ajuste contrastivo? A 48 M a
resposta foi sim, e grande: 0,3872 → 0,4712 de nDCG@10, +0,084 [+0,071; +0,097].

## ⚠️ Três números, uma régua

Os três ajustes têm de ser o MESMO treino com bases diferentes: 200 mil pares (os
mesmos bytes), lote 128, 192 tokens, semente 17. Os hashes dos pares estão abaixo e a
célula os confere, como no passo 1.

A barra é **0,5270** — o ModernBERT-base CRU ajustado assim, medido no Colab em
2026-09-17. Ela e os dois números daqui são comparáveis por construção; a comparação
acontece LOCAL, no protocolo do G1, na mesma sessão.

## ⚠️ Os PESOS vêm do Drive, e são conferidos por blake3

Os dois encoders do CPT são artefatos nossos, de 598,6 MB cada — não há de onde
baixá-los no Colab. Eles sobem para o Drive (pasta ou `.zip`), e a célula confere o
blake3 do `model.safetensors` contra o hash gravado aqui. Sem isso, "o tratado" seria
uma afirmação sobre um arquivo que ninguém conferiu — e os dois braços diferem só nos
pesos.

## ⚠️ O que o Colab tem que o Kaggle não tem

O `--out` vai para o Drive, e `train_embedding.py` grava `progresso.json` e
`estado_treino.pt`: uma sessão derrubada RETOMA. E não gasta a cota do Kaggle.

Contra: a T4 gratuita não é garantida. A primeira célula ABORTA sem GPU, porque em CPU
isto levaria dias e a descoberta viria depois de horas.
"""
from __future__ import annotations

# Do `data/processed/kaggle_t2eq_emb/MANIFESTO.json`, pacote `git_sha` 757f58f — os
# MESMOS bytes em que o controle, o tratado de 48 M e o ModernBERT-base treinaram.
HASHES = {
    "pares_treino.parquet":
        "6d4737e7d64e05ddad8071abbc83f98e1dece345ed5e0d3b9dfd871dcf63beb7",
    "pares_validacao.parquet":
        "a188bce65066fdc19aba30580188bf56711412a552316fd88d695157678084b4",
}

# blake3 do `model.safetensors` de cada braço, exportado local em 2026-09-22 do
# checkpoint do passo 6.103. É o que separa "o tratado" de "um arquivo qualquer".
HASHES_ENCODER = {
    "controle":
        "2544130457123c72fb53cf973ee71a04afa4adb7b3db5d3c57e0bc3b24662a5b",
    "tratado":
        "e2547d000211a550ca88681a25822e549346218350ea1470c90bedb6df614647",
}

PASTA_PADRAO = "/content/drive/MyDrive/phifm"

SETUP = r'''
# ─── 1. GPU, Drive, dependências, código e o ENCODER do braço ──────────────
import json, os, shutil, subprocess, sys, time, zipfile
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

VARIANTE = "__VARIANTE__"          # controle (p_equacao 0,0) ou tratado (0,6)
PASTA = Path("__PASTA__")          # onde estão os pares, os encoders e os runs
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

# ── O ENCODER do braço: pasta no Drive, ou .zip que descompacta aqui ───────
# ⚠️ Descompactar para /content e não para o Drive: ler 598 MB de pesos pelo
# FUSE do Drive a cada passo é lento, e o treino abre o modelo uma vez só.
NOME = f"phienc-cpt-{VARIANTE}"
NO_DRIVE = PASTA / "encoders" / NOME
ZIP = PASTA / "encoders" / f"{NOME}.zip"
ENCODER = Path("/content/encoders") / NOME
if NO_DRIVE.is_dir():
    ENCODER.parent.mkdir(parents=True, exist_ok=True)
    if not ENCODER.exists():
        shutil.copytree(NO_DRIVE, ENCODER)
    print(f"encoder copiado de {NO_DRIVE}")
elif ZIP.exists():
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(ENCODER.parent)
    print(f"encoder descompactado de {ZIP}")
else:
    raise AssertionError(
        f"não achei nem {NO_DRIVE} nem {ZIP}. Suba o export local de "
        f"models/{NOME} para o Drive (a pasta inteira, ou o .zip dela).")

# ── Os pesos TÊM de ser os do braço, e os pares os mesmos dos outros ───────
from blake3 import blake3

def _blake3(caminho):
    h = blake3()
    with open(caminho, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()

HASHES = __HASHES__
for nome, esperado in HASHES.items():
    obtido = _blake3(DADOS / nome)
    assert obtido == esperado, (
        f"{nome}: blake3 {obtido[:16]}… e o pacote do T1a declara {esperado[:16]}…\n"
        "Estes não são os pares em que os outros braços treinaram, e os números "
        "deixariam de ser comparáveis.")

HASHES_ENCODER = __HASHES_ENCODER__
esperado = HASHES_ENCODER[VARIANTE]
obtido = _blake3(ENCODER / "model.safetensors")
assert obtido == esperado, (
    f"os pesos de {ENCODER} têm blake3 {obtido[:16]}… e o braço {VARIANTE} é "
    f"{esperado[:16]}….\nOs dois braços diferem SÓ nos pesos: medir o arquivo "
    "errado daria o número do outro braço com o nome deste.")
prov = json.loads((ENCODER / "phienc_exportado.json").read_text(encoding="utf-8"))
print(f"✅ pares e pesos conferidos por blake3 · {NOME} do passo {prov.get('passo')} "
      f"· base {prov.get('base')}")
'''

REGRA = r"""
  ── A REGRA, escrita ANTES ────────────────────────────────────────────────

  A PERGUNTA (ADR-0003, caminho B, secundária): mascarar equações inteiras no
  pré-treino CONTINUADO melhora o encoder COMO RECUPERADOR? A primária, de
  MLM, deu NÃO DECIDIDO a 0,4 B (diferença das diferenças −0,00039 [−0,0016;
  +0,0009], medida em 2026-09-22). Esta é a medida que o sistema usa, e é onde
  o §2.3 deu positivo a 48 M: 0,3872 → 0,4712, +0,084 [+0,071; +0,097].

  MEDIDA PRIMÁRIA desta secundária: nDCG@10 no protocolo do G1, LOCAL, na mesma
  sessão, com IC por bootstrap pareado por ITEM entre os dois braços do CPT.

  Os desfechos, entre os braços (tratado_cpt − controle_cpt):

    TRATADO À FRENTE (IC exclui zero) -> o ganho de recuperação do §2.3 se
      repete a 150 M, e o mascaramento de equações entra no programa como
      objetivo de pré-treino continuado.

    EMPATE (IC cruza zero) -> o ganho de 48 M NÃO se repete na escala que
      importa. Com a primária também não decidida, o §2.3 deixa de ter
      evidência a favor fora do proxy.

    CONTROLE À FRENTE (IC exclui zero) -> o tratamento ATRAPALHA a recuperação
      a 150 M, e o objetivo sai do programa.

  E a segunda leitura, contra a BARRA de 0,5270 (o ModernBERT-base CRU ajustado
  nos mesmos 200 mil pares, medido em 2026-09-17):

    Se o MELHOR dos dois braços não superar a barra, o pré-treino continuado
    não paga o próprio custo — 0,4 B tokens e ~15 h de T4 — e o encoder do
    sistema continua sendo a base geral ajustada. Este é o número que decide o
    ADR-0003 para o produto, e a decisão é do dono.

  ⚠️ NÃO é uma ablação limpa contra a barra: o CPT viu 0,4 B tokens de Física a
     mais. Entre os DOIS BRAÇOS, sim: uma variável, `p_equacao`.

  ⚠️ Uma semente por braço, e o pool de avaliação do treino é INTERNO — a
     comparação vale pelo protocolo do G1, local, e não por este número.
"""

TREINO = r'''
# ─── 2. Treinar ────────────────────────────────────────────────────────────
# ⚠️ Iguais aos do T1f, aos dos braços de 48 M e ao do passo 1, conferidos por
# teste. O `--lote 128` define os 127 negativos do InfoNCE; mudar qualquer um
# mudaria a tarefa junto com a base.
LOTE = 128
MAX_TOKENS = 192
SEMENTE = 17
N_CANDIDATOS = 1000
PASSOS_AVAL = 200
MAX_PARES = 200_000

REGRA = r"""__REGRA__"""
print(REGRA, flush=True)

SAIDA = SAIDA_BASE / f"phiemb-cpt-{VARIANTE}-200k"
LOG = SAIDA_BASE / f"treino_t2eq_cpt_emb_{VARIANTE}.log"
# ⚠️ A saída vai para o DRIVE de propósito: uma sessão derrubada retoma de onde
# parou. É a única vantagem real do Colab sobre o Kaggle aqui — e ela some se a
# saída ficar em `/content`, que morre com a sessão.
cmd = [sys.executable, "-u", str(CODIGO / "scripts/train_embedding.py"),
       "--pares", str(DADOS), "--out", str(SAIDA), "--base", str(ENCODER),
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
    "experimento": "t2eq_cpt_emb", "variante": VARIANTE, "onde": "colab",
    "base": f"CPT phienc-cpt-{VARIANTE} (passo 6103)",
    "base_blake3": HASHES_ENCODER[VARIANTE],
    "git_sha_codigo": SHA,
    "pares": {"de": "pacote kaggle_t2eq_emb (blake3 conferido)",
              "max_pares": MAX_PARES},
    "hiperparametros": {"lote": LOTE, "negativos_infonce": LOTE - 1,
                        "max_tokens": MAX_TOKENS, "semente": SEMENTE,
                        "n_candidatos_aval": N_CANDIDATOS,
                        "passos_aval": PASSOS_AVAL},
    "gpu": torch.cuda.get_device_name(0),
    "custo_segundos": custo,
    "melhor": m,
    "barra_do_passo_1": 0.5270,
    "regra": REGRA,
    "nota": ("Pool INTERNO do treino, não o do G1. A comparação entre os braços e "
             "contra a barra é LOCAL, na mesma sessão, pelo protocolo do G1."),
}
(SAIDA_BASE / f"t2eq_cpt_emb_{VARIANTE}.json").write_text(
    json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")

print("=" * 74)
print(f"  secundária do caminho B · braço {VARIANTE} · {MAX_PARES:,} pares · "
      f"{custo/3600:.2f} h")
print(f"  melhor no pool interno (n={N_CANDIDATOS}): nDCG@10 {m.get('ndcg_10')} · "
      f"passo {m.get('passo')}")
print("  ⚠️ pool INTERNO do treino, não o do G1 — a comparação é local.")
print("=" * 74)
print(f"\nNo Drive, em {SAIDA_BASE}:")
print(f"  {melhor.name}/   (o checkpoint — é o que tem de vir para a máquina local)")
print(f"  t2eq_cpt_emb_{VARIANTE}.json · {LOG.name}")
'''

CELULAS = (SETUP, TREINO, RESUMO)
