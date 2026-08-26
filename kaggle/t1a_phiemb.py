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
"""

CELULA = r'''
# ─── ΦEmb / T1a — cole isto numa célula do Kaggle ────────────────────────────
import hashlib, json, os, subprocess, sys, zipfile
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
print(f"dataset: {DADOS.name} · git {man['git_sha']} · {man['linhas_treino']:,} pares")

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

# 3. O código vem do pacote: o notebook NÃO reimplementa o treino.
#
# ⚠️ Duas formas possíveis, e a segunda é o Kaggle mexendo no que subiu. Medido em
# 2026-08-24: um `phifm_src.zip` é DESCOMPACTADO no upload e chega como o diretório
# `phifm_src/` — o `ZipFile` morria em FileNotFoundError aos 26 s. O empacotador
# passou a gravar `.zip.bin` por isso, mas um dataset antigo ainda tem a forma
# extraída, então as duas são aceitas.
_zip = next((DADOS / n for n in ("phifm_src.zip.bin", "phifm_src.zip")
             if (DADOS / n).exists()), None)
if _zip is not None:
    with zipfile.ZipFile(_zip) as z:
        z.extractall(TRABALHO / "codigo")
    CODIGO = TRABALHO / "codigo"
elif (DADOS / "phifm_src").is_dir():
    # Não dá para conferir hash de árvore extraída contra um manifesto de arquivo,
    # e dizer isso alto é melhor que treinar em silêncio sobre código não conferido.
    print("⚠️ usando `phifm_src/` que o Kaggle extraiu. A integridade do CÓDIGO "
          "não foi conferida — só a dos parquets. Para conferir, republique o "
          "dataset com `scripts/empacotar_kaggle.py`, que agora grava .zip.bin")
    CODIGO = DADOS / "phifm_src"
else:
    raise SystemExit(
        f"não achei o código no dataset. Esperava `phifm_src.zip.bin` ou o "
        f"diretório `phifm_src/` em {DADOS}. Presentes: "
        f"{sorted(p.name for p in DADOS.iterdir())}")
sys.path.insert(0, str(CODIGO))

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
# E o `sys.path.insert(parents[1] / "src")` que o próprio script faz não resolve
# aqui: no repositório o pacote vive em `src/phifm/`, mas o ZIP grava `phifm/` na
# RAIZ (`_zipar_fonte` usa `relative_to(raiz / "src")`), então `codigo/src` não
# existe. Apontar o PYTHONPATH para `CODIGO` funciona nos dois layouts.
AMBIENTE = {**os.environ, "PYTHONPATH": str(CODIGO)}

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
