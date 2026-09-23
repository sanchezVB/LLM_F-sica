"""T1g — o GTE-base com 1 milhão de pares supera o ΦEmb do sistema?

Este arquivo é o CONTEÚDO de uma célula do Kaggle, mantido aqui como `.py` para
ficar sob controle de versão e ser testável.

## De onde vem a pergunta

Três medidas, todas no protocolo do G1:

| | pares | nDCG@10 |
|---|---|---|
| MiniLM (o ΦEmb do sistema) | 6 M | **0,6223** |
| GTE-base (T1f) | 400 mil | 0,6094 — empata com o do sistema, p=0,908 |
| GTE-base (`t2eq_emb_gte`, 2026-09-22) | 200 mil | 0,5964 — +0,069 sobre o ModernBERT-base |

A base pesa mais que o volume: dobrar os pares do GTE de 200 para 400 mil rendeu
+0,013, e o GTE a 400 mil já alcança o MiniLM a 6 M. A pergunta que sobra é se ele
PASSA o encoder do sistema com uma fração do volume.

## ⚠️ NÃO é ablação de uma variável, e é de propósito

Muda a base E o volume (1 M contra 6 M). É a pergunta do PRODUTO — "existe um
recuperador melhor que o atual, a um custo de treino que cabe na cota?" —, e não a
da ciência. A ablação de uma variável seria GTE@1,5M contra o MiniLM@1,5M (0,5780),
e ela custaria ~9,6 h: não cabe nas ~8,4 h de cota que sobram na semana. O MiniLM@1,5M
entra como DIAGNÓSTICO, com o volume 1,5× maior contra o GTE.

## ⚠️ 1 M, e não mais, por causa do relógio

A 43,6 pares/s (medido no `t2eq_emb_gte`, na mesma T4), 1 M de pares são ~6,4 h. Com
a variação de ~8% entre máquinas do Kaggle já medida, ~7 h: dentro da sessão e da
cota. O dataset é o do T1a 1,5 M, já publicado, limitado a 1 M pelo `--max-pares` —
nada novo sobe.

## ⚠️ E o custo de serviço entra na decisão, porque é permanente

O GTE-base leva **4,4×** o tempo do MiniLM para embutir o mesmo universo (954 s contra
217 s, medido no T1f). Uma vitória aqui não é troca automática: a margem tem de pagar
4,4× de inferência para sempre. Essa decisão é do dono do projeto.
"""

CELULA = r'''
# ─── T1g / GTE-base@1M contra o ΦEmb do sistema — cole numa célula do Kaggle ─
import hashlib, io, json, os, subprocess, sys, tarfile, time, urllib.request
from pathlib import Path

ENTRADA = Path("/kaggle/input")
TRABALHO = Path("/kaggle/working")

# ── 1. Achar o dataset ──────────────────────────────────────────────────────
# Busca em PROFUNDIDADE: a imagem nova do Kaggle monta em
# `/kaggle/input/datasets/<dono>/<slug>/` e a antiga em `/kaggle/input/<slug>/`.
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
print(f"dataset: {DADOS.name} · git {man['git_sha']} · "
      f"{man['linhas_treino']:,} pares · "
      f"{man.get('documentos_distintos', 0):,} documentos citados distintos")

# ⚠️ O dataset do T1a 1,5 M, já publicado. Os pares são o mesmo sorteio (semente
# 17) que alimentou o MiniLM@1,5M do diagnóstico.
assert man["experimento"] == "t1a15", (
    f"o dataset anexado é do experimento {man['experimento']!r}. O T1g usa os pares "
    "do T1a 1,5 M, limitados a 1 M; com outro pacote o diagnóstico contra o "
    "MiniLM@1,5M perderia o sorteio em comum.")

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
            f"{nome}: hash difere do manifesto. Treinar sobre isto daria um número "
            "sobre pares que ninguém registrou.")
    print(f"✅ {len(man['arquivos'])} arquivos conferidos por blake3")

# ── 3. Código do GitHub ─────────────────────────────────────────────────────
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

import torch
_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sem GPU"
print(f"torch {torch.__version__} · {_gpu}")
assert torch.cuda.is_available(), "sem GPU. Settings → Accelerator → GPU."

# ── 4. Os hiperparâmetros do T1f, e o volume ────────────────────────────────
# ⚠️ Iguais aos do T1f e dos ajustes de 200 mil, conferidos por teste. O que muda é
# SÓ o volume: `MAX_PARES`.
BASE = "thenlper/gte-base"
LOTE = 128
MAX_TOKENS = 192
SEMENTE = 17
N_CANDIDATOS = 1000
PASSOS_AVAL = 200
MAX_PARES = 1_000_000

REGRA = r"""
  ── A REGRA, escrita ANTES (2026-09-22) ───────────────────────────────────

  A PERGUNTA (produto): o GTE-base ajustado em 1 M de pares supera o ΦEmb do
  sistema — o MiniLM ajustado em 6 M, nDCG@10 0,6223?

  MEDIDA PRIMÁRIA: nDCG@10 no protocolo do G1, LOCAL, com os DOIS checkpoints
  medidos na mesma sessão; IC por bootstrap pareado por ITEM (gte − sistema).

  Os desfechos:

    GTE À FRENTE   ->  existe um recuperador melhor que o do sistema, a 1/6 do
    (IC exclui 0)      volume de pares. ⚠️ NÃO é troca automática: o GTE custa
                       4,4x para embutir, para sempre. A margem medida vai ao
                       dono do projeto junto com esse custo, e a decisão é dele.

    EMPATE         ->  o GTE a 1 M alcança o sistema e não o passa. Com o custo
    (IC cruza 0)       de 4,4x, o sistema fica como está. A pergunta do GTE@6M
                       (~39 h de T4) continua aberta, mas sem sinal que a pague.

    SISTEMA À FRENTE -> o volume ainda manda a 1 M: o MiniLM a 6 M segura o
    (IC exclui 0)       lugar. O GTE só voltaria à mesa com muito mais pares.

  DIAGNÓSTICO (não decide): o GTE@1M contra o MiniLM@1,5M (0,5780) — as duas
  bases perto do mesmo volume, com o MiniLM tendo 1,5x mais pares.

  ⚠️ Não é ablação de uma variável: muda base E volume. É a pergunta do
     produto, e está declarada assim antes do número.
"""

print(f"""
{'=' * 74}
T1g — o GTE-base com 1 M de pares supera o ΦEmb do sistema (MiniLM@6M, 0,6223)?

  base {BASE} · {MAX_PARES:,} pares · lote {LOTE} ({LOTE - 1} negativos) ·
  max_tokens {MAX_TOKENS} · semente {SEMENTE}

  ⚠️ A COMPARAÇÃO NÃO ACONTECE AQUI. Esta célula treina e para; os dois
  checkpoints são medidos LOCAL, na mesma sessão, pelo protocolo do G1.
{REGRA}
{'=' * 74}
""", flush=True)

AMBIENTE = {**os.environ, "PYTHONPATH": str(FONTE)}
SAIDA = TRABALHO / "phiemb-gte-base-1m"


def _rodar(cmd, log_em):
    """Roda com a saída nos DOIS destinos, e LEVANTA se falhou.

    ⚠️ Mandar só para o arquivo custou 33 h de cegueira em 2026-09-07.
    """
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
       TRABALHO / "treino_t1g.log")
custo = round(time.perf_counter() - t0, 1)

melhor = SAIDA.parent / f"{SAIDA.name}-melhor"
assert melhor.exists(), (
    f"{melhor} não existe — o treino não elegeu checkpoint. Sem ele não há o que "
    "baixar, e a sessão foi gasta.")
m = json.loads((melhor / "melhor.json").read_text(encoding="utf-8"))

(TRABALHO / "t1g.json").write_text(json.dumps({
    "experimento": "t1g",
    "pergunta": "o GTE-base com 1 M de pares supera o ΦEmb do sistema?",
    "base": BASE,
    "git_sha_dados": man["git_sha"], "git_sha_codigo": SHA,
    "hiperparametros": {"lote": LOTE, "negativos_infonce": LOTE - 1,
                        "max_tokens": MAX_TOKENS, "semente": SEMENTE,
                        "n_candidatos_aval": N_CANDIDATOS,
                        "passos_aval": PASSOS_AVAL, "max_pares": MAX_PARES},
    "gpu": _gpu,
    "custo_segundos": custo,
    "melhor": m,
    "alvo": {"checkpoint": "models/phiemb-do-sistema", "ndcg_10_registrado": 0.6223,
             "pares": 6_000_000},
    "diagnostico": {"checkpoint": "models/phiemb-minilm-15m-sorteado-melhor",
                    "ndcg_10_registrado": 0.5780, "pares": 1_500_000},
    "custo_de_servico": {"gte_base_s": 954, "minilm_s": 217, "razao": 4.4},
    "regra": REGRA,
}, indent=2, ensure_ascii=False), encoding="utf-8")

print()
print("=" * 74)
print(f"  T1g · GTE-base@1M · {custo/3600:.2f} h")
print(f"  melhor no pool interno (n={N_CANDIDATOS}): nDCG@10 {m.get('ndcg_10')} · "
      f"passo {m.get('passo')}")
print("  ⚠️ pool INTERNO do treino, não o do G1 — a comparação é local.")
print(f"  -> {melhor}")
print("=" * 74)
'''
