"""ΦRank de base DIFERENTE — T1c, na cota gratuita da T4, custo zero em dinheiro.

Este arquivo é o CONTEÚDO de um notebook do Kaggle, mantido aqui como `.py` para
ficar sob controle de versão e ser testável. Ver `kaggle/t1a_phiemb.py` para o
porquê de não ser um `.ipynb`.

## A pergunta, e de onde ela vem

O T1b mediu a composição inteira e o veredito do reranqueador foi **empate com a
fusão**: nDCG@10 0,1493 contra 0,1584, McNemar p=0,118 sobre 105 discordantes. Ele
ficou fora do sistema.

O diagnóstico não foi "o reranqueador é ruim" — ele acerta 49,8% no grupo, muito
acima do 12,5% do acaso. Foi **redundância informacional**: o ΦRank parte do
`all-MiniLM-L6-v2`, a mesma base do ΦEmb, e o Spearman entre o escore dele e a
posição da fusão é −0,466 (concordância, no sinal em que posição menor é melhor).
Ele reordena reproduzindo a ordem que a fusão já tinha. Um reranqueador que
concorda com o recuperador não tem como acrescentar nada.

E há espaço para acrescentar: o recall@50 da fusão é 0,446 contra um nDCG@10 de
0,1584 — **2,8× de folga dentro dos candidatos que já foram recuperados**.

## A hipótese, registrada ANTES de medir

> Se a redundância informacional é o que anula o ΦRank, então um cross-encoder
> partindo de uma base com pré-treino DIFERENTE do ΦEmb deve bater a fusão.

Duas bases, escolhidas para separar dois mecanismos possíveis:

| variante | base | 109 M | o que ela traz que o MiniLM não tem |
|---|---|---|---|
| `gte` | `thenlper/gte-base` | sim | pré-treino de recuperação forte, dados gerais |
| `phys` | `thellert/physbert_cased` | sim | pré-treino em Física — o domínio |

## A regra de decisão, também registrada antes

Desfecho primário: **McNemar exato em k=10 contra a fusão medida na MESMA execução**.

- as duas vencem  ⇒ o mecanismo é DIVERSIDADE de base; basta não compartilhar o
  pré-treino com o recuperador
- só a `phys`     ⇒ o mecanismo é CONHECIMENTO DE DOMÍNIO, não diversidade
- só a `gte`      ⇒ o mecanismo é CAPACIDADE/força do pré-treino, não domínio
- nenhuma vence   ⇒ a redundância informacional NÃO é a restrição que manda; o
  problema está no objetivo de treino ou nos dados, e a próxima medição é outra

⚠️ **Duas variantes são dois testes.** O limiar honesto para afirmar que "uma
variante funciona" é Bonferroni, **p < 0,025**. Entre 0,025 e 0,05 o resultado é
sugestivo e pede replicação, não anúncio. Isto está escrito aqui para não ser
escolhido depois de ver os números.

## O que fica FIXO, e por quê

Só a base muda. `--max-grupos 12500 --grupos 2 --n-negativos 7 --lr 2e-5
--max-tokens 384 --semente 17` são exatamente os do `phirank-rrf-melhor`, o
controle. Este repositório já perdeu um experimento por trocar base e lote na mesma
corrida, e não saber qual dos dois explicava o resultado.

⚠️ O `lr 2e-5` NÃO foi re-ajustado para 109 M de parâmetros. É o padrão de ajuste
fino de BERT-base, então é defensável, mas é uma variável não controlada: se as duas
variantes falharem, "a taxa estava errada para este tamanho" continua sendo uma
explicação viva, e dizer isso agora é mais honesto que descobrir depois.

## O controle é reavaliado, não copiado do JSON antigo

O número de referência do ΦRank saiu de 1.000 consultas. Aqui tudo roda com 2.000,
porque 105 discordantes deram p=0,118 e o teste estava sem poder. Comparar uma
variante nova em 2.000 contra o controle em 1.000 misturaria efeito com poder — o
controle roda de novo, no mesmo protocolo.

Ele roda PRIMEIRO, e de propósito: é a avaliação mais barata (23 M contra 109 M) e
exercita o caminho inteiro. Se o encanamento estiver quebrado, isso aparece em ~12
min em vez de depois de 45 min de treino.

## O RESULTADO: domínio, não diversidade (fechado em 2026-09-03)

    variante   base                       params  acc@1  +ΦRank      Δ  disc  p(k=10)
    controle   MiniLM-L6 (= a do ΦEmb)       23M  0,498  0,1483  -,0093  229   0,1458
    gte        gte-base (geral forte)       109M  0,510  0,1530  -,0046  220   0,6371
    phys       physbert (Física)            109M  0,566  0,1666  +,0090  237   0,0062

    fusão RRF = 0,1576 nas três · 2.000 consultas · profundidade 50 · teto 0,4495

**Leitura pré-registrada: "só a `phys` ⇒ o mecanismo é CONHECIMENTO DE DOMÍNIO, não
diversidade".** É o que saiu.

O que torna isto uma afirmação de mecanismo e não uma coincidência: `gte-base` e
`physbert_cased` têm o **mesmo tamanho** (109 M), a **mesma arquitetura**
(`BertModel`), os **mesmos seis hiperparâmetros**, os **mesmos dados**, a **mesma
semente** e o **mesmo protocolo**. A única diferença entre as duas é o corpus de
pré-treino. Diversidade de base não basta; capacidade não basta.

⚠️ **As duas corridas são combináveis, e há evidência disso.** O braço `gte` rodou
numa execução separada, e o controle saiu **byte a byte idêntico** nas duas —
`sistemas`, `pareado_contra_a_fusao` e `teto` iguais campo por campo, incluindo
p=0,14584 sobre 229 discordantes. É isso que licencia comparar `gte` com `phys`.

### O contraponto que este resultado produz

O PhysBERT é um recuperador **ruim** neste benchmark: nDCG 0,2752 contra 0,4657 do
ΦEmb — perde por 0,190, e foi essa a medição do G1.1. E é a **melhor base de
reranqueador** das três. Pré-treino de domínio não fez um bi-encoder bom aqui e fez
um cross-encoder bom. Não é o que eu esperaria, e a assimetria é o achado.

## Os dois erros meus na execução de 2026-08-31, e um terceiro em 2026-09-03

**A predição do T1b se confirmou.** O reranqueador de base diferente bate a fusão com
p = 0,00625, abaixo do limiar de Bonferroni de 0,025. É a primeira vez que a
composição inteira funciona: recall@10 de 0,2890 contra 0,2675 da fusão. E o controle
replicou o empate do T1b com o dobro das consultas (p = 0,146 contra 0,118 antes),
o que é a evidência de que a diferença é da base e não do protocolo.

⚠️ **Erro 3 — o Kaggle FIXA a versão do dataset no anexo ao kernel.** Consertei o
fp16, subi versão nova do dataset (`status` = `ready`), empurrei o notebook, e ele
rodou 15 min sobre o código ANTIGO, morrendo com o mesmo erro. `kernels push` não
re-resolve para a versão mais recente. O notebook delatou na saída — `git_sha:
73088dc` contra o conserto em `68fe86e` —, e sem essa linha eu teria concluído que o
conserto do fp16 não funcionava. Agora o código vem do GitHub num SHA injetado na
publicação; os dados ficam no dataset, onde a fixação é inofensiva porque eles não
mudam.

⚠️ **Erro 1 — o `gte` morreu por um bug nosso, não dele.** `thenlper/gte-base` guarda
os pesos em **fp16**, o `transformers` novo carrega no dtype do checkpoint por
padrão, e o `GradScaler` recusa desescalar gradientes fp16:

    ValueError: Attempting to unscale FP16 gradients.

O AMP exige pesos-mestres em fp32. Consertado com `dtype=torch.float32` explícito em
`rerank.py`. Ficou invisível até aqui porque MiniLM e PhysBERT são fp32 — qualquer
base fp16 quebraria, e o custo foi um braço inteiro do experimento.

⚠️ **Erro 2 — o meu pré-registro confundia "perdeu" com "não rodou".** Com o `gte`
ausente, o script imprimiu *"o mecanismo é CONHECIMENTO DE DOMÍNIO, não
diversidade"* — uma leitura que **exige** o `gte` ter produzido um número e perdido.
A lógica olhava só `venceram`, nunca `falhas`. O pré-registro existia para impedir
exatamente esse tipo de conclusão, e a implementação dele tinha o buraco.

Agora qualquer braço ausente torna o teste de mecanismo **INCONCLUSIVO**. O que está
estabelecido é que *uma* base diferente vence; **qual propriedade da base** faz isso
— domínio ou capacidade — segue aberto até o `gte` rodar.

## Vazão medida, contra a minha estimativa

PhysBERT treinou a **35,4 exemplos/s** na T4, com 2.332 MB de VRAM. Eu havia estimado
14–24/s — **pessimista por ~2×**. As avaliações caíram na faixa estimada (13 min o
controle, ~40 min o de 109 M). E os 2,3 GB num cartão de 16 GB dizem que `--grupos 2`
subutiliza a placa — aumentar seria mais rápido e **quebraria a comparabilidade com o
controle**, então fica como está.
"""

CELULA = r'''
# ─── ΦRank de base diferente / T1c — cole isto numa célula do Kaggle ─────────
import hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path

ENTRADA = Path("/kaggle/input")
TRABALHO = Path("/kaggle/working")

# ── 1. Achar o dataset ──────────────────────────────────────────────────────
# Busca em PROFUNDIDADE: a imagem nova monta em /kaggle/input/datasets/<dono>/<slug>/
# e a antiga em /kaggle/input/<slug>/. Medido em 2026-08-26, quando o pin da imagem
# de CPU foi solto e o notebook morreu com "Encontrados: ['datasets']".
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
print(f"dataset: {DADOS} · git {man['git_sha']} · {man.get('grupos', '?')} grupos")
print("⚠️ o git do dataset é a proveniência dos DADOS. O do CÓDIGO é o SHA abaixo, "
      "e os dois podem divergir de propósito: os parquets não mudam, o código muda.")

# ── 2. Conferir os hashes ANTES de treinar ──────────────────────────────────
# Sem isto, um upload truncado produziria um número que parece comparável ao medido
# na máquina local e não é. O Kaggle não garante nada sobre o input além de existir.
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
    conferidos, ausentes = 0, []
    for nome, esperado in man["arquivos"].items():
        caminho = DADOS / nome
        if not caminho.exists():
            ausentes.append(nome)
            continue
        h = _h()
        with open(caminho, "rb") as f:
            while b := f.read(1 << 22):
                h.update(b)
        obtido = h.hexdigest()
        assert obtido == esperado["blake3"], (
            f"{nome}: hash difere. Esperado {esperado['blake3'][:12]}…, obtido "
            f"{obtido[:12]}…. Upload truncado ou dataset trocado.")
        conferidos += 1
    print(f"✅ {conferidos} arquivos conferidos por blake3")
    for nome in ausentes:
        print(f"⚠️ {nome} não está no dataset — hash NÃO conferido")

for obrigatorio in ("pares_do_recuperador_limpos.parquet", "pares_validacao.parquet"):
    assert (DADOS / obrigatorio).exists(), (
        f"{obrigatorio} não está no dataset. Confira o Input do notebook.")

# ── 3. Código e modelos ────────────────────────────────────────────────────
# `.zip.bin` porque o Kaggle DESCOMPACTA `.zip` no upload: em 2026-08-24 um
# `phifm_src.zip` chegou como o diretório `phifm_src/`, o ZipFile morreu aos 26 s e
# — pior — o hash do fonte deixou de ser conferível. Extensão que ele não reconhece
# preserva os bytes; o `zipfile` abre pelo conteúdo, não pela extensão.
def _abrir(nomes, destino, rotulo):
    z = next((DADOS / n for n in nomes if (DADOS / n).exists()), None)
    if z is not None:
        with zipfile.ZipFile(z) as f:
            f.extractall(destino)
        return destino
    extraido = next((DADOS / n.split(".zip")[0] for n in nomes
                     if (DADOS / n.split(".zip")[0]).is_dir()), None)
    if extraido is not None:
        print(f"⚠️ usando `{extraido.name}/` que o Kaggle extraiu — a integridade "
              f"do {rotulo} NÃO foi conferida. Republique com empacotar_kaggle.py.")
        return extraido
    raise SystemExit(
        f"não achei o {rotulo} no dataset. Esperava um de {nomes} em {DADOS}. "
        f"Presentes: {sorted(p.name for p in DADOS.iterdir())}")

MODELOS = _abrir(("modelos.zip.bin", "modelos.zip"), TRABALHO / "modelos", "modelos")

# ⚠️ O CÓDIGO vem do GitHub, num SHA fixo — NÃO do dataset.
#
# Medido em 2026-09-03: o Kaggle FIXA a versão do dataset no momento em que ela é
# anexada ao kernel, e `kernels push` não re-resolve para a mais recente. O conserto
# do fp16 subiu numa versão nova do dataset, `datasets status` disse `ready`, e o
# notebook rodou 15 min sobre o código ANTIGO — morrendo com o mesmo erro. Ele
# delatou na própria saída: `git_sha: 73088dc`, e o conserto estava em `68fe86e`.
#
# O notebook, ao contrário do dataset, é reempurrado a cada publicação. Então um SHA
# injetado aqui é sempre o do commit atual, e não há versão a fixar.
SHA = "__SHA__"
REPO = "__REPO__"
import io, tarfile, urllib.request
alvo = TRABALHO / "codigo"
url = f"https://codeload.github.com/{REPO}/tar.gz/{SHA}"
print(f"baixando o código de {url}", flush=True)
with urllib.request.urlopen(url, timeout=180) as r:
    bruto = r.read()
with tarfile.open(fileobj=io.BytesIO(bruto)) as tf:
    raizes = {m.name.split("/")[0] for m in tf.getmembers()}
    assert len(raizes) == 1, f"tarball com {len(raizes)} raízes: {sorted(raizes)}"
    tf.extractall(alvo, filter="data")
CODIGO = alvo / raizes.pop()
print(f"código em {CODIGO} · {len(bruto)/1e3:.0f} KB · SHA {SHA[:7]}")

# O tarball do GitHub preserva `src/`, então o pacote vive em `<raiz>/src/phifm` —
# ao contrário do zip antigo, que gravava `phifm/` na raiz.
FONTE = CODIGO / "src"
for exigido in ("src/phifm/training/rerank.py", "scripts/train_rerank.py",
                "scripts/avaliar_t1b.py"):
    assert (CODIGO / exigido).exists(), (
        f"{exigido} não está no tarball de {SHA[:7]} — o SHA está errado ou o "
        f"arquivo foi renomeado. Presentes na raiz: "
        f"{sorted(p.name for p in CODIGO.iterdir())}")
sys.path.insert(0, str(FONTE))

EMB = MODELOS / "phiemb-minilm-melhor"
CONTROLE = MODELOS / "phirank-rrf-melhor"
for d in (EMB, CONTROLE):
    assert (d / "model.safetensors").exists(), f"{d} não tem pesos"

import torch
print(f"torch {torch.__version__} · CUDA {torch.cuda.is_available()} · "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU'}")
assert torch.cuda.is_available(), (
    "sem GPU. Settings → Accelerator → GPU. Isto em CPU levaria dias.")

# ⚠️ PYTHONPATH no AMBIENTE do subprocesso. O sys.path.insert desta célula não vale
# para o filho, e o `sys.path.insert(parents[1] / "src")` que os scripts fazem não
# resolve: no repositório o pacote vive em `src/phifm/`, mas o ZIP grava `phifm/` na
# raiz. Medido em 2026-08-26: ModuleNotFoundError: No module named 'phifm'.
AMBIENTE = {**os.environ, "PYTHONPATH": str(FONTE)}

# ── 4. A REGRA DE DECISÃO, impressa antes de qualquer número ────────────────
N_CONSULTAS = 2000
PROFUNDIDADE = 50
ALFA_BONFERRONI = 0.025   # 0,05 / 2 variantes
print(f"""
{'=' * 74}
T1c — o reranqueador precisa de uma base DIFERENTE do recuperador?

hipótese : se a redundância informacional anula o ΦRank, um cross-encoder de base
           com pré-treino diferente do ΦEmb deve BATER a fusão
desfecho : McNemar exato em k=10 contra a fusão medida NESTA execução
limiar   : p < {ALFA_BONFERRONI} (Bonferroni, 2 variantes). Entre 0,025 e 0,05 é
           sugestivo e pede replicação — não é anúncio.
leitura  : as duas vencem -> diversidade de base basta
           só phys        -> o mecanismo é domínio
           só gte         -> o mecanismo é capacidade do pré-treino
           nenhuma        -> a redundância não é a restrição que manda
fixo     : max-grupos 12500 · grupos 2 · n-negativos 7 · lr 2e-5 · 384 tok · seed 17
protocolo: {N_CONSULTAS} consultas · profundidade {PROFUNDIDADE} · universo 88.807
{'=' * 74}
""", flush=True)


def _rodar(cmd, log_em):
    """Roda, tudo no log, e LEVANTA se falhou.

    ⚠️ O `check=False` de antes transformou um treino morto num notebook COMPLETE, e
    foi assim que uma execução sem NENHUM modelo treinado passou por sucesso em
    2026-08-26. E a saída vai para um ARQUIVO em /kaggle/working porque o
    `kernels output` da API devolveu log de 0 BYTES em três execuções seguidas.
    """
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    with open(log_em, "w", encoding="utf-8") as fh:
        r = subprocess.run([str(c) for c in cmd], stdout=fh,
                           stderr=subprocess.STDOUT, text=True, env=AMBIENTE)
    cauda = Path(log_em).read_text(encoding="utf-8", errors="replace")
    print(cauda[-4000:], flush=True)
    if r.returncode != 0:
        raise SystemExit(f"{cmd[2]} saiu com {r.returncode}; log em {log_em}")
    return cauda


def avaliar(nome, rank_dir):
    saida = TRABALHO / f"t1b_{nome}.json"
    _rodar([sys.executable, "-u", CODIGO / "scripts/avaliar_t1b.py",
            "--pares", DADOS, "--emb", EMB, "--rank", rank_dir,
            "--n-consultas", N_CONSULTAS, "--profundidade", PROFUNDIDADE,
            "--out", saida, "--dispositivo", "cuda"],
           TRABALHO / f"aval_{nome}.log")
    return json.loads(saida.read_text(encoding="utf-8"))


def treinar(nome, base):
    saida = TRABALHO / f"phirank-{nome}"
    _rodar([sys.executable, "-u", CODIGO / "scripts/train_rerank.py",
            "--negativos", DADOS / "pares_do_recuperador_limpos.parquet",
            "--out", saida, "--base", base,
            "--max-grupos", 12500, "--grupos", 2, "--n-negativos", 7,
            "--lr", 2e-5, "--max-tokens", 384, "--semente", 17,
            "--dispositivo", "cuda"],
           TRABALHO / f"treino_{nome}.log")
    melhor = saida.parent / f"{saida.name}-melhor"
    assert (melhor / "model.safetensors").exists(), (
        f"{melhor} não existe — o treino terminou sem gravar um melhor")
    return melhor


# ── 5. Executar: controle primeiro, que é o mais barato ─────────────────────
BASES = {"gte": "thenlper/gte-base", "phys": "thellert/physbert_cased"}

# ⚠️ Existe para COMPLETAR um braço que falhou por bug de infraestrutura, não para
# escolher braços depois de ver resultados. Em 2026-08-31 a `gte` morreu no primeiro
# passo (checkpoint fp16 contra o GradScaler) e a `phys` venceu; refazer as duas
# custaria 3 h de cota para remedir um número que já existe.
#
# A regra que mantém isto honesto: `SO` só pode conter braços SEM número gravado, e
# o `mecanismo` fica INCONCLUSIVO enquanto algum braço de `BASES` não tiver o seu. Um
# braço que rodou e perdeu NUNCA pode ser refeito por aqui.
SO: set[str] = set()
if SO:
    print(f"⚠️ rodando só {sorted(SO)} — completando braço que falhou por bug de "
          f"infraestrutura. Os demais vêm da execução anterior.")

resultados, falhas = {}, {}

resultados["minilm (controle)"] = avaliar("controle", CONTROLE)

for nome, base in ((n, b) for n, b in BASES.items() if not SO or n in SO):
    # ⚠️ Uma variante que morre não pode levar a outra com ela. Um id de modelo
    # errado ou um OOM custaria a sessão inteira, e a variante que já treinou
    # continua sendo um resultado.
    try:
        resultados[f"{nome} ({base})"] = avaliar(nome, treinar(nome, base))
    except BaseException as exc:
        falhas[nome] = f"{type(exc).__name__}: {exc}"
        print(f"❌ variante {nome} falhou: {falhas[nome]}", flush=True)

# ── 6. A tabela, e a regra aplicada ────────────────────────────────────────
def _linha(res, sistema):
    for s in res["sistemas"]:
        if s["sistema"] == sistema:
            return s
    return {}


def _p_contra_fusao(res, k=10):
    for c in res["pareado_contra_a_fusao"]:
        if c["b"] == "ΦEmb+BM25+ΦRank" and c["k"] == k:
            return c
    return {}


print(f"\n{'=' * 74}\nT1c — RESULTADO\n{'=' * 74}")
print(f"{'variante':26s} {'fusão':>8s} {'+ΦRank':>8s} {'Δ':>8s} "
      f"{'disc':>5s} {'p(k=10)':>9s}  veredito")
linhas = []
for nome, res in resultados.items():
    f10 = _linha(res, "ΦEmb+BM25 (RRF)").get("ndcg_10")
    r10 = _linha(res, "ΦEmb+BM25+ΦRank").get("ndcg_10")
    c = _p_contra_fusao(res)
    p, disc = c.get("p"), c.get("discordantes")
    # `ganha_a` é a fusão; o reranqueador só vence quando `ganha_b` > `ganha_a`.
    melhor_b = c.get("ganha_b", 0) > c.get("ganha_a", 0)
    if p is None:
        vd = "sem par"
    elif p < ALFA_BONFERRONI and melhor_b:
        vd = "VENCE a fusão"
    elif p < ALFA_BONFERRONI:
        vd = "PERDE da fusão"
    elif p < 0.05 and melhor_b:
        vd = "sugestivo (não passa Bonferroni)"
    else:
        vd = "empate"
    linhas.append({"variante": nome, "ndcg_fusao": f10, "ndcg_rank": r10,
                   "delta": None if None in (f10, r10) else round(r10 - f10, 4),
                   "discordantes": disc, "p_k10": p, "veredito": vd,
                   "teto_recall": res.get("teto_do_reranker")})
    d = "     —" if None in (f10, r10) else f"{r10 - f10:+8.4f}"
    print(f"{nome:26s} {f10 or 0:8.4f} {r10 or 0:8.4f} {d} "
          f"{disc or 0:5d} {p if p is None else round(p, 5):>9}  {vd}")

for nome, erro in falhas.items():
    print(f"{nome:26s} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>5s} {'—':>9s}  FALHOU: {erro}")

venceram = sorted(n.split()[0] for n, l in
                  zip(resultados, linhas, strict=True) if "VENCE" in l["veredito"])

# ⚠️ Uma variante que NÃO RODOU não é uma variante que PERDEU, e a primeira versão
# desta lógica confundia as duas.
#
# Medido em 2026-08-31: a `gte` morreu no primeiro passo do treino (checkpoint fp16
# contra o GradScaler), a `phys` venceu, e o script imprimiu "o mecanismo é
# CONHECIMENTO DE DOMÍNIO, não diversidade" — uma conclusão que exige a `gte` ter
# produzido um número. O pré-registro existia justamente para impedir esse tipo de
# leitura, e a implementação dele tinha o buraco.
#
# Agora: qualquer braço ausente torna o teste de mecanismo INCONCLUSIVO, e isso é
# dito antes de qualquer leitura.
ausentes = sorted(set(BASES) - set(resultados_por_base := {
    n.split()[0] for n in resultados}) - {"minilm"})
if ausentes:
    mecanismo = (f"INCONCLUSIVO sobre o mecanismo: o braço {ausentes} não produziu "
                 f"número (ver `falhas`). Venceram: {venceram or 'nenhuma'}. Uma "
                 f"variante ausente não é uma variante que perdeu, e sem ela "
                 f"'domínio' e 'diversidade' não se separam.")
elif set(venceram) >= set(BASES):
    mecanismo = "DIVERSIDADE de base basta — as duas venceram"
elif venceram == ["phys"]:
    mecanismo = "o mecanismo é CONHECIMENTO DE DOMÍNIO, não diversidade"
elif venceram == ["gte"]:
    mecanismo = "o mecanismo é CAPACIDADE do pré-treino, não domínio"
elif venceram:
    mecanismo = f"venceram: {venceram} (inclui o controle? leia com cuidado)"
else:
    mecanismo = ("NENHUMA venceu — a redundância informacional não é a restrição "
                 "que manda. O próximo lugar a olhar é o objetivo de treino ou os "
                 "dados, não a base.")
del resultados_por_base
print(f"\nveredito pré-registrado: {mecanismo}")

(TRABALHO / "t1c_resultado.json").write_text(json.dumps({
    "n_consultas": N_CONSULTAS, "profundidade": PROFUNDIDADE,
    "alfa_bonferroni": ALFA_BONFERRONI, "bases": BASES,
    "git_sha_dados": man["git_sha"], "git_sha_codigo": SHA,
    "linhas": linhas, "falhas": falhas,
    "mecanismo": mecanismo,
    "fixo": ("max-grupos 12500, grupos 2, n-negativos 7, lr 2e-5, 384 tokens, "
             "semente 17 — idênticos ao controle phirank-rrf-melhor. Só a base muda."),
    "ressalva": ("o lr 2e-5 não foi re-ajustado para 109 M de parâmetros; se as duas "
                 "variantes falharem, 'a taxa estava errada para este tamanho' segue "
                 "sendo explicação viva"),
}, indent=2, ensure_ascii=False), encoding="utf-8")

for f in sorted(TRABALHO.rglob("*.json")) + sorted(TRABALHO.glob("*.log")):
    print(f"  {f.relative_to(TRABALHO)}  {f.stat().st_size/1e3:.1f} KB")
print("\n⚠️ Baixe `t1c_resultado.json`, os `t1b_*.json` e os `.log`. Os pesos das "
      "variantes só valem a pena baixar se alguma VENCEU.")
'''


def main() -> int:
    """Imprime a célula, para copiar. Não roda nada aqui."""
    print(CELULA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
