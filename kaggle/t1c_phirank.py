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

CODIGO = _abrir(("phifm_src.zip.bin", "phifm_src.zip"), TRABALHO / "codigo", "código")
MODELOS = _abrir(("modelos.zip.bin", "modelos.zip"), TRABALHO / "modelos", "modelos")
sys.path.insert(0, str(CODIGO))

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
AMBIENTE = {**os.environ, "PYTHONPATH": str(CODIGO)}

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
resultados, falhas = {}, {}

resultados["minilm (controle)"] = avaliar("controle", CONTROLE)

for nome, base in BASES.items():
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
if set(venceram) >= set(BASES):
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
print(f"\nveredito pré-registrado: {mecanismo}")

(TRABALHO / "t1c_resultado.json").write_text(json.dumps({
    "n_consultas": N_CONSULTAS, "profundidade": PROFUNDIDADE,
    "alfa_bonferroni": ALFA_BONFERRONI, "bases": BASES,
    "git_sha": man["git_sha"], "linhas": linhas, "falhas": falhas,
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
