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
candidatos = [d for d in ENTRADA.iterdir() if (d / "MANIFESTO.json").exists()]
assert candidatos, (
    f"nenhum dataset com MANIFESTO.json em {ENTRADA}. "
    f"Encontrados: {[d.name for d in ENTRADA.iterdir()]}")
DADOS = candidatos[0]
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
    for nome, esperado in man["arquivos"].items():
        h = _h()
        with open(DADOS / nome, "rb") as f:
            while b := f.read(1 << 22):
                h.update(b)
        obtido = h.hexdigest()
        assert obtido == esperado["blake3"], (
            f"{nome}: hash difere. Esperado {esperado['blake3'][:12]}…, "
            f"obtido {obtido[:12]}…. Upload truncado ou dataset trocado — "
            f"treinar sobre isto daria número incomparável.")
    print(f"✅ {len(man['arquivos'])} arquivos conferidos por blake3")

# 3. O código vem do ZIP: o notebook NÃO reimplementa o treino.
with zipfile.ZipFile(DADOS / "phifm_src.zip") as z:
    z.extractall(TRABALHO / "codigo")
sys.path.insert(0, str(TRABALHO / "codigo"))

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
    sys.executable, "-u", str(TRABALHO / "codigo/scripts/train_embedding.py"),
    "--pares", str(DADOS),
    "--out", str(SAIDA),
    "--base", "sentence-transformers/all-MiniLM-L6-v2",
    "--lote", "128",
    "--passos-aval", "200",
    "--n-candidatos", "1000",
    "--dispositivo", "cuda",
]
print(" ".join(cmd))
# `check=False` e o código de saída impresso: uma sessão que cai deixa o estado em
# `SAIDA`, e a próxima execução retoma. Falha aqui não é o fim, é o próximo passo.
r = subprocess.run(cmd)
print(f"\ncódigo de saída: {r.returncode}")

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
