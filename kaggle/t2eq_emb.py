"""T2eq-emb — o ganho de recuperação do §2.3 sobrevive ao ajuste contrastivo?

Este arquivo é o CONTEÚDO de uma célula do Kaggle. A mesma célula roda os TRÊS
braços; o que muda é `VARIANTE` ("controle", "tratado" ou "modernbert"), injetada
na publicação.

## ⚠️ O terceiro braço: a BARRA do caminho B (2026-09-17)

Acrescentado depois de controle e tratado rodarem (tratado à frente, nDCG@10 0,4712
contra 0,3872), e ANTES de o braço existir. O ADR-0003 aponta para um pré-treino
continuado do ModernBERT-base; antes de gastar ~22 h de T4 nele, a pergunta é quanto
o ModernBERT-base faz SEM pré-treino nenhum, nos mesmos pares e hiperparâmetros. Ele
baixa do Hub (`answerdotai/ModernBERT-base`), não do zip.

## A pergunta (ADR-0003, opção C)

A ablação do DOC-07 §2.3 deu negativo na primária e, na secundária de recuperação,
**nDCG@10 0,0171 → 0,1391** — oito vezes, e não é geometria. Mas é MLM cru agregado
por média. O recuperador do sistema é um bi-encoder AJUSTADO por pares de citação, e
o T1f mostrou que o ajuste pode apagar ou ampliar diferença de base. Se o ganho não
sobreviver ao ajuste, ele não importa para a busca.

Aqui cada braço de 48 M é ajustado como ΦEmb, com a receita do T1f. Uma variável: a
base — controle (`p_equacao` 0,0) ou tratado (`p_equacao` 0,6).

## ⚠️ 200 mil pares, e não os 400 mil da receita

Decidido em 2026-09-16 pelo dono do projeto, antes de o braço existir. Medido: o proxy
custa 2,1× o MiniLM por passo, então ~86 pares/s estimados na T4; 400 mil seriam ~78
min de treino por braço e os dois não cabem nas 2 h 13 de cota que restavam. A 200 mil
são ~39 min. Os DOIS braços usam o mesmo `--max-pares`, sorteado com a mesma semente
dos MESMOS bytes do pacote do T1a (conferidos por blake3 no empacotador), então a
comparação entre eles é limpa. Contra o MiniLM@400k ou o GTE-base@400k do T1f, não é:
o volume difere.

## ⚠️ Os hiperparâmetros são os do T1f, e um teste confere

`tests/regression/test_kaggle_t2eq_emb.py` lê as constantes das duas células pela AST
e exige lote, tokens, semente, candidatos e passos de avaliação iguais. A única
diferença declarada é `MAX_PARES`.
"""
from __future__ import annotations

CELULA = r'''
# ─── T2eq-emb / o ganho do §2.3 sobrevive ao ajuste? — cole numa célula do Kaggle ─
import hashlib, io, json, os, subprocess, sys, tarfile, time, urllib.request, zipfile
from pathlib import Path

ENTRADA = Path("/kaggle/input")
TRABALHO = Path("/kaggle/working")

VARIANTE = "__VARIANTE__"          # "controle", "tratado" ou "modernbert"
BASES = {"controle": "phienc-t2a-E", "tratado": "phienc-t2eq-E-tratado",
         "modernbert": "answerdotai/ModernBERT-base"}
assert VARIANTE in BASES, VARIANTE

# ── 1. Achar o dataset ──────────────────────────────────────────────────────
candidatos = []
for profundidade in range(0, 5):
    padrao = "/".join(["*"] * profundidade + ["MANIFESTO.json"])
    candidatos = sorted({m.parent for m in ENTRADA.glob(padrao)})
    if candidatos:
        break
assert candidatos, (
    f"nenhum MANIFESTO.json em {ENTRADA} até 4 níveis. Presentes na raiz: "
    f"{[d.name for d in ENTRADA.iterdir()]}")
DADOS = candidatos[0]
man = json.loads((DADOS / "MANIFESTO.json").read_text())
print(f"dataset: {DADOS.name} · git {man['git_sha']} · braço {VARIANTE}")
assert man["experimento"] == "t2eq_emb_controle", (
    f"o dataset anexado é do experimento {man['experimento']!r}; os dois braços "
    "têm de treinar sobre o MESMO pacote")
assert man.get("pares_iguais_ao_t1a") is True, (
    "o manifesto não atesta que os pares são os bytes do T1a")

# ── 2. Conferir os hashes ANTES de treinar ──────────────────────────────────
try:
    from blake3 import blake3 as _h
    algo = "blake3"
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "blake3"], check=False)
    try:
        from blake3 import blake3 as _h
        algo = "blake3"
    except ImportError:
        _h, algo = hashlib.sha256, "sha256"
        print("⚠️ blake3 indisponível — conferência DESATIVADA, o manifesto é blake3")

if algo == "blake3":
    for nome, esperado in man["arquivos"].items():
        caminho = DADOS / nome
        assert caminho.exists(), f"{nome} não está no dataset"
        h = _h()
        with open(caminho, "rb") as f:
            while b := f.read(1 << 22):
                h.update(b)
        assert h.hexdigest() == esperado["blake3"], (
            f"{nome}: hash difere. Upload truncado ou dataset trocado — o braço "
            "deixaria de ser comparável ao outro.")
    print(f"✅ {len(man['arquivos'])} arquivos conferidos por blake3")

# ── 3. Código do GitHub, e a base deste braço ──────────────────────────────
SHA = "__SHA__"
REPO = "__REPO__"
url = f"https://codeload.github.com/{REPO}/tar.gz/{SHA}"
print(f"baixando o código de {url}", flush=True)
with urllib.request.urlopen(url, timeout=180) as r:
    bruto = r.read()
with tarfile.open(fileobj=io.BytesIO(bruto)) as tf:
    raizes = {m.name.split("/")[0] for m in tf.getmembers()}
    assert len(raizes) == 1, f"tarball com {len(raizes)} raízes: {sorted(raizes)}"
    tf.extractall(TRABALHO / "codigo", filter="data")
CODIGO = TRABALHO / "codigo" / raizes.pop()
FONTE = CODIGO / "src"
assert (CODIGO / "scripts/train_embedding.py").exists(), (
    f"scripts/train_embedding.py não está no tarball de {SHA[:7]}")
sys.path.insert(0, str(FONTE))
print(f"código em {CODIGO} · {len(bruto)/1e3:.0f} KB · SHA {SHA[:7]}")

from phifm.core.kaggle import assinatura_do_manifesto

ASSINATURA_ESPERADA = "__ASSINATURA_DADOS__"
_obtida = assinatura_do_manifesto(man["arquivos"])
assert _obtida == ASSINATURA_ESPERADA, (
    f"o dataset anexado tem assinatura {_obtida} e esta célula foi publicada para "
    f"{ASSINATURA_ESPERADA}. O Kaggle fixou uma versão ANTIGA do dataset no anexo.")
print(f"✅ assinatura do bundle confere: {_obtida}")

MODELOS = TRABALHO / "modelos"
with zipfile.ZipFile(DADOS / "modelos.zip.bin") as z:
    z.extractall(MODELOS)
# ⚠️ O braço `modernbert` baixa a base do Hub; os outros dois a tiram do zip.
if VARIANTE == "modernbert":
    BASE = BASES[VARIANTE]
else:
    BASE = MODELOS / BASES[VARIANTE]
    assert (BASE / "config.json").exists() and (BASE / "model.safetensors").exists(), (
        f"{BASE} não tem config e pesos — o zip de modelos não é o esperado")

import torch
_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sem GPU"
print(f"torch {torch.__version__} · {_gpu}")
assert torch.cuda.is_available(), "sem GPU. Settings → Accelerator → GPU."

# ── 4. Os hiperparâmetros do T1f, e o volume declarado ──────────────────────
# ⚠️ Iguais aos do T1f, conferidos por teste. O `--lote 128` define os 127
# negativos do InfoNCE; mudar qualquer um mudaria a tarefa junto com a base.
LOTE = 128
MAX_TOKENS = 192
SEMENTE = 17
N_CANDIDATOS = 1000
PASSOS_AVAL = 200
# ⚠️ A metade da receita, decidida antes (ver o docstring do arquivo): os dois
# braços não cabiam na cota a 400 mil. O MESMO nos dois braços.
MAX_PARES = 200_000

REGRA_BRACOS = r"""
  ── A REGRA, escrita ANTES ────────────────────────────────────────────────

  A PERGUNTA (ADR-0003, opção C): o ganho de recuperação do braço tratado do
  §2.3 — nDCG@10 cru 0,0171 -> 0,1391 — sobrevive ao ajuste contrastivo?

  MEDIDA PRIMÁRIA: nDCG@10 no protocolo do G1 (pool de 2.000 desduplicado,
  `SEMENTE_POOL`), com os DOIS checkpoints ajustados medidos LOCAL na mesma
  sessão, e IC por bootstrap pareado por ITEM. Recall@1 pelo McNemar exato
  vai junto, como diagnóstico.

  Os desfechos:

    TRATADO À FRENTE  ->  o ganho sobrevive ao ajuste. ADR-0003: seguir com o
    (IC exclui zero)      pré-treino continuado do ModernBERT-base com
                          `p_equacao` 0,6.

    EMPATE            ->  o ajuste apaga a diferença a 200 mil pares. ADR-0003:
    (IC cruza zero)       o ganho do §2.3 não chega ao sistema nesta escala, e o
                          ΦEnc pausa. ⚠️ A 400 mil ou 6 M a ordem poderia mudar,
                          e isso fica escrito ao lado.

    CONTROLE À FRENTE ->  o tratamento atrapalha a base de embedding. ADR-0003:
    (IC exclui zero)      pausar o ΦEnc.

  ⚠️ Bases de 48 M a 0,6 B, uma semente por braço, 200 mil pares. O número
     absoluto NÃO se compara com o MiniLM@400k nem com o GTE-base@400k do T1f.
"""

REGRA_MODERNBERT = r"""
  ── A REGRA do braço modernbert, escrita ANTES (2026-09-17) ───────────────

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
"""

REGRA = REGRA_MODERNBERT if VARIANTE == "modernbert" else REGRA_BRACOS

print(f"""
{'=' * 74}
T2eq-emb — o ganho de recuperação do §2.3 sobrevive ao ajuste?  ·  braço {VARIANTE}

  base {BASES[VARIANTE]}
  lote {LOTE} ({LOTE - 1} negativos) · max_tokens {MAX_TOKENS} · semente {SEMENTE}
  {MAX_PARES:,} pares sorteados dos MESMOS bytes do T1a

  ⚠️ A COMPARAÇÃO NÃO ACONTECE AQUI. Esta célula treina um braço e para.
{REGRA}
{'=' * 74}
""", flush=True)

AMBIENTE = {**os.environ, "PYTHONPATH": str(FONTE)}
SAIDA = TRABALHO / f"phiemb-t2eq-{VARIANTE}-200k"


def _rodar(cmd, log_em):
    """Roda com a saída nos DOIS destinos, e LEVANTA se falhou."""
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    with open(log_em, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                encoding="utf-8", errors="replace", env=AMBIENTE)
        for linha in proc.stdout:
            fh.write(linha)
            fh.flush()
            print(linha, end="", flush=True)
        codigo = proc.wait()
    if codigo != 0:
        raise SystemExit(f"{cmd[2]} saiu com {codigo}; log em {log_em}")


# ── 5. Treinar ──────────────────────────────────────────────────────────────
t0 = time.perf_counter()
_rodar([sys.executable, "-u", CODIGO / "scripts/train_embedding.py",
        "--pares", DADOS, "--out", SAIDA, "--base", BASE,
        "--lote", LOTE, "--max-tokens", MAX_TOKENS, "--max-pares", MAX_PARES,
        "--passos-aval", PASSOS_AVAL, "--n-candidatos", N_CANDIDATOS,
        "--dispositivo", "cuda", "--semente", SEMENTE],
       TRABALHO / f"treino_t2eq_emb_{VARIANTE}.log")
custo = round(time.perf_counter() - t0, 1)

melhor = SAIDA.parent / f"{SAIDA.name}-melhor"
assert melhor.exists(), (
    f"{melhor} não existe — o treino não elegeu checkpoint. Sem ele não há o que "
    "baixar, e a sessão foi gasta.")
m = json.loads((melhor / "melhor.json").read_text(encoding="utf-8"))

(TRABALHO / f"t2eq_emb_{VARIANTE}.json").write_text(json.dumps({
    "experimento": "t2eq_emb", "variante": VARIANTE, "base": BASES[VARIANTE],
    "git_sha_dados": man["git_sha"], "git_sha_codigo": SHA,
    "hiperparametros": {"lote": LOTE, "negativos_infonce": LOTE - 1,
                        "max_tokens": MAX_TOKENS, "semente": SEMENTE,
                        "n_candidatos_aval": N_CANDIDATOS,
                        "passos_aval": PASSOS_AVAL, "max_pares": MAX_PARES},
    "custo_segundos": custo,
    "melhor": m,
    "nota": ("A comparação controle × tratado NÃO está aqui: os dois checkpoints "
             "ajustados são medidos LOCAL, na mesma sessão, pelo protocolo do G1."),
}, indent=2, ensure_ascii=False), encoding="utf-8")

print()
print("=" * 74)
print(f"  T2eq-emb · braço {VARIANTE} · {MAX_PARES:,} pares · {custo/3600:.2f} h")
print(f"  melhor no pool interno (n={N_CANDIDATOS}): nDCG@10 {m.get('ndcg_10')} · "
      f"passo {m.get('passo')}")
print("  ⚠️ pool INTERNO do treino, não o do G1 — a comparação é local.")
print(f"  -> {melhor}")
print("=" * 74)
print(f"\n⚠️ Baixe `{melhor.name}/` inteiro, `t2eq_emb_{VARIANTE}.json` e o log.")
'''
