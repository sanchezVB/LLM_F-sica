"""ΦEmb no Kaggle — T1a do DOC-17A §8.2, em T4, custo zero.

Este arquivo é o CONTEÚDO de um notebook do Kaggle, mantido aqui como `.py` para
ficar sob controle de versão e ser testável. Para usar: crie um notebook, cole o
conteúdo numa célula, anexe o dataset e habilite a GPU.

    Notebook → Settings → Accelerator: GPU T4 x2   (usa uma só; ver abaixo)
    Notebook → Add Input → Datasets → o dataset com os 211 MB

## Por que este arquivo existe em vez de um `.ipynb`

Um `.ipynb` é JSON com saídas embutidas: o diff é ilegível, o merge é impossível e
o conteúdo executável fica misturado com o resultado da última execução. Como `.py`
ele entra na suíte de testes — e há um teste que confere que este arquivo não
reimplementa o treino, porque a tentação de "só copiar o laço aqui" é o caminho
para dois laços divergindo em silêncio.

## O que o Kaggle dá, medido pela documentação da plataforma

30 h/semana de GPU, sessões de até 9 h, 2× T4 de 16 GB. As sessões caem: o
`estado_treino.pt` do nosso treino é gravado a cada 100 passos e `retomar` volta de
onde parou, então uma queda custa ~6 min, não a corrida.

⚠️ **Uma T4, não duas.** O treino contrastivo não é paralelizado por dados aqui, e
`DataParallel` num lote contrastivo é sutilmente errado: cada réplica calcularia o
InfoNCE só sobre a sua fatia, então o número de negativos por âncora cairia de 127
para 63 sem nada avisar. É o mesmo tipo de erro que o GradCache existe para não
cometer. Usar as duas placas exige `DistributedDataParallel` com `all_gather` das
representações, que é trabalho e ainda não foi feito.

## O que muda em relação à RX 7600, e é o motivo de vir para cá

| | RX 7600 (DirectML) | T4 (CUDA) |
|---|---|---|
| atenção | `eager` obrigatório | `sdpa` — não materializa a matriz N×N |
| precisão | fp32 | fp16 com `GradScaler` |
| memória liberada em `del` | **não** | sim |
| medido | 20–26 pares/s | a medir |

As duas primeiras linhas são restrições do DirectML que eu quase tratei como
propriedades do problema. `escolher_dispositivo` já prefere CUDA e liga as duas.

## ⚠️ Retreino de 2026-09-06: os pares agora são SORTEADOS

O run original empacotou `head(400.000)` de `pares_treino.parquet`, que vem
agrupado por documento citado. Medido:

    n=  400.000   head -> 17.844 documentos citados   sorteio -> 191.300   10,7×

O modelo campeão do G1 treinou com **17.844** documentos distintos onde o sorteio
do mesmo tamanho dá **191.300** — mesma GPU, mesmo tempo. Ver `amostrar_do_plano`
em `phifm.training.amostragem`.

Duas coisas mudaram por causa disso:

* **O código vem do GitHub num SHA**, não do dataset. O Kaggle fixa a versão do
  dataset no momento em que ela é anexada ao kernel e `kernels push` não
  re-resolve — na T1c isso custou 15 min de execução sobre código antigo.
* **A assinatura do `pares_treino.parquet` é injetada nesta célula** na publicação,
  e a célula levanta se o manifesto do dataset anexado declarar outra. Um bundle
  errado falha em segundos, e não depois de 36 min de T4.
"""

CELULA = r'''
# ─── ΦEmb / T1a — cole isto numa célula do Kaggle ────────────────────────────
import hashlib, json, os, subprocess, sys
from pathlib import Path

ENTRADA = Path("/kaggle/input")
TRABALHO = Path("/kaggle/working")

# 1. Achar o dataset. Não fixamos o nome: quem cria o dataset escolhe o slug, e
#    um caminho fixo quebraria com uma renomeação sem dizer por quê.
# ⚠️ Busca em PROFUNDIDADE, não só nos filhos diretos. A imagem nova do Kaggle
#    monta o dataset em `/kaggle/input/datasets/<dono>/<slug>/`; a antiga montava
#    em `/kaggle/input/<slug>/`. Medido em 2026-08-26, depois que o pin da imagem
#    de CPU foi solto: o notebook morreu com
#
#        AssertionError: nenhum dataset com MANIFESTO.json em /kaggle/input.
#                        Encontrados: ['datasets']
#
#    Vai de raso para fundo e para no primeiro nível que casa — assim funciona nos
#    dois layouts, e um dia a mais de mudança do Kaggle não derruba de novo.
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
print(f"dataset em {DADOS}")
man = json.loads((DADOS / "MANIFESTO.json").read_text())
print(f"dataset: {DADOS.name} · git {man['git_sha']} · {man['linhas_treino']:,} pares"
      f" · {man.get('documentos_distintos', 0):,} documentos citados distintos")


# 2. ⚠️ CONFERIR OS HASHES antes de treinar.
#
#    Sem isto, um upload truncado ou um dataset trocado produziria um número que
#    parece comparável aos medidos na máquina local e não é. O Kaggle não garante
#    nada sobre o que está no input — só que existe.
#
#    blake3 não vem instalado no Kaggle; usamos o hash do manifesto quando dá, e
#    caímos para sha256 registrando a troca. Nunca pular em silêncio.
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
        print("⚠️ blake3 indisponível — conferência de hash DESATIVADA, "
              "o manifesto é blake3 e não há com o que comparar")

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
            f"{nome}: hash difere. Esperado {esperado['blake3'][:12]}…, "
            f"obtido {obtido[:12]}…. Upload truncado ou dataset trocado — "
            f"treinar sobre isto daria número incomparável.")
        conferidos += 1
    print(f"✅ {conferidos} arquivos conferidos por blake3")
    for nome in ausentes:
        print(f"⚠️ {nome} não está no dataset — hash NÃO conferido")

    # ⚠️ Os parquets não têm desculpa para faltar: são os DADOS. Sem eles não há
    # treino, e um `assert` aqui é mais barato que descobrir no meio do laço.
    for obrigatorio in ("pares_treino.parquet", "pares_validacao.parquet"):
        assert (DADOS / obrigatorio).exists(), (
            f"{obrigatorio} não está no dataset. Confira o Input do notebook.")

# 3. O CÓDIGO vem do GitHub, num SHA fixo — NÃO do dataset. O notebook NÃO
#    reimplementa o treino.
#
# Medido em 2026-09-03 na T1c: o Kaggle fixa a versão do dataset no anexo e
# `kernels push` não re-resolve, então o notebook rodou 15 min sobre o código
# ANTIGO. O notebook, ao contrário do dataset, é reempurrado a cada publicação —
# um SHA injetado aqui é sempre o do commit atual, e não há versão a fixar.
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
for exigido in ("src/phifm/training/embedding.py",
                "src/phifm/training/amostragem.py",
                "scripts/train_embedding.py"):
    assert (CODIGO / exigido).exists(), (
        f"{exigido} não está no tarball de {SHA[:7]} — o SHA está errado ou o "
        f"arquivo foi renomeado. Presentes na raiz: "
        f"{sorted(p.name for p in CODIGO.iterdir())}")
sys.path.insert(0, str(FONTE))

# 3b. ⚠️ A ASSINATURA DO BUNDLE — um `assert`, não um print.
#
# O Kaggle FIXA a versão do dataset quando ela é anexada ao kernel, e `kernels
# push` não re-resolve. Medido na T1c em 2026-09-03: uma versão nova subiu,
# `datasets status` disse `ready`, e o notebook rodou 15 min sobre o conteúdo
# ANTIGO. Só foi percebido porque a saída imprimia o `git_sha` e alguém leu —
# depender de leitura humana não é uma guarda.
#
# A conferência de blake3 do passo 2 compara os arquivos com o manifesto QUE VEIO
# NO MESMO dataset: pega upload truncado, e não bundle da versão errada, porque um
# bundle velho é internamente consistente. Só um valor de FORA distingue os dois, e
# ele é injetado aqui na publicação.
#
# Vem DEPOIS do download do código de propósito: assim a conta mora num lugar só
# (`phifm.core.kaggle`) em vez de ser reimplementada nesta célula. Custa os ~10 s
# do tarball, e não os 36 min de T4.
from phifm.core.kaggle import assinatura_do_manifesto

ASSINATURA_ESPERADA = "__ASSINATURA_DADOS__"
_obtida = assinatura_do_manifesto(man["arquivos"])
assert _obtida == ASSINATURA_ESPERADA, (
    f"o dataset anexado tem assinatura {_obtida} e esta célula foi publicada para "
    f"{ASSINATURA_ESPERADA}. O Kaggle fixou uma versão ANTIGA do dataset no anexo "
    f"— treinar aqui reproduziria justamente o run que se queria substituir. "
    f"Troque o Input do notebook pelo dataset novo. Arquivos vistos: "
    f"{sorted(man['arquivos'])}")
print(f"✅ assinatura do bundle confere: {_obtida}")

import torch
print(f"torch {torch.__version__} · CUDA {torch.cuda.is_available()} · "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU'}")
assert torch.cuda.is_available(), (
    "sem GPU. Settings → Accelerator → GPU. Rodar isto em CPU levaria dias.")

# 4. Treinar. Os argumentos são os do campeão do G1.1, mais o que a T4 permite.
#
#    `--lote 128` é o do campeão, de propósito: mudar o lote junto com o
#    dispositivo faria a comparação medir duas coisas. A T4 comporta mais, e
#    aumentar é um experimento SEPARADO.
SAIDA = TRABALHO / "phiemb-minilm-t4"
cmd = [
    sys.executable, "-u", str(CODIGO / "scripts/train_embedding.py"),
    "--pares", str(DADOS),
    "--out", str(SAIDA),
    "--base", "sentence-transformers/all-MiniLM-L6-v2",
    "--lote", "128",
    "--passos-aval", "200",
    "--n-candidatos", "1000",
    "--dispositivo", "cuda",
    # A semente do pool de avaliação e de qualquer corte. Explícita porque o
    # `-melhor` é eleito por esta métrica, e um pool que muda entre execuções
    # tornaria a eleição irreprodutível.
    "--semente", "17",
]
print(" ".join(cmd))

# ⚠️ A saída do treino vai para um ARQUIVO em /kaggle/working, não só para o stdout
# da célula. Medido em 2026-08-26: `kernels output` da API devolveu um log de 0
# BYTES em três execuções seguidas, e sem log não há como saber por que o treino
# falhou. Um arquivo em /kaggle/working é baixável mesmo quando o log da API não vem.
# ⚠️ `PYTHONPATH` no AMBIENTE do subprocesso. O `sys.path.insert` desta célula não
# vale para ele — processo filho tem o seu próprio path. Medido em 2026-08-26:
#
#     ModuleNotFoundError: No module named 'phifm'
#       em /kaggle/working/codigo/scripts/train_embedding.py
#
# Com o tarball do GitHub o pacote vive em `<raiz>/src/phifm`, então o PYTHONPATH
# é `CODIGO/src` — e não `CODIGO`, que era o certo para o zip antigo (ele gravava
# `phifm/` na raiz, porque `_zipar_fonte` usava `relative_to(raiz / "src")`).
AMBIENTE = {**os.environ, "PYTHONPATH": str(FONTE)}

TREINO_LOG = TRABALHO / "treino.log"
with open(TREINO_LOG, "w", encoding="utf-8") as fh:
    r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True,
                       env=AMBIENTE)
print(TREINO_LOG.read_text(encoding="utf-8", errors="replace")[-6000:])
print(f"código de saída: {r.returncode}")

# ⚠️ E LEVANTA se falhou. Antes era `check=False` com o código apenas impresso, e a
# justificativa era "uma sessão que cai deixa o estado para a próxima retomar" — o
# que estava errado: `/kaggle/working` persiste como saída do notebook
# independentemente de a célula levantar. O que o `check=False` fazia de verdade era
# transformar um treino morto num notebook `COMPLETE`, e foi assim que uma execução
# sem NENHUM modelo treinado passou por sucesso em 2026-08-26.
if r.returncode != 0:
    raise SystemExit(
        f"o treino saiu com {r.returncode}. As últimas linhas estão acima e o log "
        f"inteiro em {TREINO_LOG.name}, que desce junto com a saída do notebook.")

# 5. O que salvar. `/kaggle/working` persiste como output do notebook; o resto some.
for f in sorted(SAIDA.parent.rglob("*")):
    if f.is_file():
        print(f"  {f.relative_to(TRABALHO)}  {f.stat().st_size/1e6:.1f} MB")
print("\n⚠️ Baixe o diretório `-melhor` e o `phiemb.json`. O veredito do G1 roda na "
      "máquina local, sobre o protocolo de 2.000 candidatos — o número de 1.000 "
      "candidatos do log NÃO é comparável ao veredito.")
'''


def main() -> int:
    """Imprime a célula, para copiar. Não roda o treino aqui."""
    print(CELULA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
