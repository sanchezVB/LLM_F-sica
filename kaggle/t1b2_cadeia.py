"""T1b2 — remede a CADEIA depois da troca do recuperador. T4, custo zero.

Este arquivo é o CONTEÚDO de uma célula do Kaggle, mantido aqui como `.py` para
ficar sob controle de versão e ser testável.

## Por que existe

O recuperador do sistema mudou em 2026-09-08: o `phiemb-do-sistema` (T1a 6 M
sorteado) dá nDCG@10 **0,6223** no protocolo do portão contra 0,5246 do
`phiemb-minilm-melhor` — **+0,098**. E o nDCG **0,1666** do ΦRank (p=0,0062), que
é o resultado do T1b/T1c, foi medido sobre a fusão RRF do recuperador **antigo**.
Ele está obsoleto desde a troca.

## ⚠️ Os DOIS recuperadores, na MESMA sessão

A célula roda `avaliar_t1b.py` duas vezes — com o recuperador novo e com o antigo
—, e é a decisão de desenho central deste experimento.

Comparar a medição nova contra o número de agosto pareceria mais barato e seria
inválido: entre agosto e hoje mudaram o protocolo do G1, o pool de candidatos e
quatro versões do código. A diferença medida não seria atribuível à troca do
recuperador. Duas execuções na mesma sessão, com o mesmo commit, as mesmas 2.000
consultas sorteadas com a mesma semente e o mesmo universo isolam **uma** variável.

Este projeto já pagou por medir duas coisas de uma vez — foi o que fez o G1.2
"passar" por +0,003 em agosto.

## O que esperar, e por que uma queda não seria regressão

`avaliar_t1b.py` imprime o **recall@100 do recuperador antes do nDCG**, porque ele
é o **teto** do reranker: um ΦRank perfeito sobre um recall@100 de 0,70 não passa
de 0,70. Um recuperador melhor sobe esse teto e deixa **menos** para o reranker
reordenar, então o ganho marginal do ΦRank deve **encolher**. Encolher não é
regressão — é o teto subindo. O que decidiria contra o ΦRank seria a cadeia
completa ficar abaixo da fusão sem ele.
"""

CELULA = r'''
# ─── T1b2 / cadeia remedida — cole isto numa célula do Kaggle ────────────────
import hashlib, io, json, os, subprocess, sys, tarfile, urllib.request, zipfile
from pathlib import Path

ENTRADA = Path("/kaggle/input")
TRABALHO = Path("/kaggle/working")

# ── 1. Achar o dataset ──────────────────────────────────────────────────────
# Busca em PROFUNDIDADE: a imagem nova monta em /kaggle/input/datasets/<dono>/<slug>/
# e a antiga em /kaggle/input/<slug>/. Medido em 2026-08-26, quando o pin da imagem
# foi solto e o notebook morreu com "Encontrados: ['datasets']".
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
print(f"dataset: {DADOS} · git {man['git_sha']}")

# ── 2. Conferir os hashes ANTES de medir ────────────────────────────────────
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
            f"{nome}: hash difere. Upload truncado ou dataset trocado — medir "
            f"sobre isto daria número incomparável.")
    print(f"✅ {len(man['arquivos'])} arquivos conferidos por blake3")

# ── 3. Modelos do dataset, código do GitHub ─────────────────────────────────
# `.zip.bin` porque o Kaggle DESCOMPACTA `.zip` no upload: em 2026-08-24 um
# `phifm_src.zip` chegou como diretório, o ZipFile morreu aos 26 s e o hash deixou
# de ser conferível. Extensão que ele não reconhece preserva os bytes.
_z = next((DADOS / n for n in ("modelos.zip.bin", "modelos.zip")
           if (DADOS / n).exists()), None)
if _z is None:
    raise SystemExit(
        f"não achei os modelos. Presentes: {sorted(p.name for p in DADOS.iterdir())}")
MODELOS = TRABALHO / "modelos"
with zipfile.ZipFile(_z) as f:
    f.extractall(MODELOS)

# O CÓDIGO vem do GitHub num SHA — não do dataset. O Kaggle FIXA a versão do
# dataset no anexo e `kernels push` não re-resolve: em 2026-09-03 isso fez um
# notebook rodar 15 min sobre código antigo.
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
for exigido in ("src/phifm/eval/hibrido.py", "scripts/avaliar_t1b.py"):
    assert (CODIGO / exigido).exists(), (
        f"{exigido} não está no tarball de {SHA[:7]} — SHA errado ou renomeado")
sys.path.insert(0, str(FONTE))
print(f"código em {CODIGO} · {len(bruto)/1e3:.0f} KB · SHA {SHA[:7]}")

# A assinatura do bundle, injetada na publicação — um `assert`, não um print. A
# conferência de blake3 acima compara os arquivos com o manifesto QUE VEIO NO
# MESMO dataset: pega upload truncado, e não bundle da versão errada, porque um
# bundle velho é internamente consistente.
from phifm.core.kaggle import assinatura_do_manifesto

ASSINATURA_ESPERADA = "__ASSINATURA_DADOS__"
_obtida = assinatura_do_manifesto(man["arquivos"])
assert _obtida == ASSINATURA_ESPERADA, (
    f"o dataset anexado tem assinatura {_obtida} e esta célula foi publicada para "
    f"{ASSINATURA_ESPERADA}. O Kaggle fixou uma versão ANTIGA do dataset no anexo.")
print(f"✅ assinatura do bundle confere: {_obtida}")

import torch
print(f"torch {torch.__version__} · CUDA {torch.cuda.is_available()} · "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU'}")
assert torch.cuda.is_available(), "sem GPU. Settings → Accelerator → GPU."

# ── 4. O protocolo, declarado ANTES de qualquer número ──────────────────────
N_CONSULTAS = 2000
PROFUNDIDADE = 100
RANK = MODELOS / "phirank-physbert-melhor"
# ⚠️ Os DOIS recuperadores, na MESMA sessão. Comparar a medição nova contra o
# número de agosto pareceria mais barato e seria inválido: entre agosto e hoje
# mudaram o protocolo do G1, o pool de candidatos e quatro versões do código, e a
# diferença não seria atribuível à troca do recuperador.
BRACOS = {
    "novo": MODELOS / "phiemb-do-sistema",
    "antigo": MODELOS / "phiemb-minilm-melhor",
}
for nome, d in BRACOS.items():
    assert (d / "model.safetensors").exists(), f"o braço {nome} não tem pesos em {d}"
assert (RANK / "model.safetensors").exists(), f"o ΦRank não tem pesos em {RANK}"

print(f"""
{'=' * 74}
T1b2 — a cadeia remedida depois da troca do recuperador

  hipótese: a cadeia com o recuperador NOVO entrega nDCG@10 maior que com o
            antigo, e o ganho MARGINAL do ΦRank sobre a fusão ENCOLHE, porque o
            recall@100 do recuperador é o teto do reranker.

  ⚠️ Encolher o ganho do ΦRank NÃO é regressão. O que decidiria contra ele seria
     a cadeia completa ficar ABAIXO da fusão sem reranker.

  protocolo: {N_CONSULTAS} consultas · profundidade {PROFUNDIDADE} · universo 88.807
             mesma semente, mesmo commit, mesma sessão nos dois braços
{'=' * 74}
""", flush=True)

AMBIENTE = {**os.environ, "PYTHONPATH": str(FONTE)}


def _rodar(cmd, log_em):
    """Roda com a saída nos DOIS destinos, e LEVANTA se falhou.

    ⚠️ Arquivo E stdout. Mandar só para o arquivo custou 33 h de cegueira em
    2026-09-07: nada aparecia no log da célula entre o lançamento e o fim, e uma
    execução saudável ficava indistinguível de uma travada. O arquivo continua
    porque o `kernels output` da API devolveu 0 BYTE em três execuções seguidas.
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


# ── 5. Medir os dois braços ─────────────────────────────────────────────────
resultados = {}
for nome, emb in BRACOS.items():
    saida = TRABALHO / f"t1b2_{nome}.json"
    _rodar([sys.executable, "-u", CODIGO / "scripts/avaliar_t1b.py",
            "--pares", DADOS, "--emb", emb, "--rank", RANK,
            "--n-consultas", N_CONSULTAS, "--profundidade", PROFUNDIDADE,
            "--out", saida, "--dispositivo", "cuda"],
           TRABALHO / f"aval_{nome}.log")
    resultados[nome] = json.loads(saida.read_text(encoding="utf-8"))

# ── 6. A leitura, contra a hipótese declarada acima ─────────────────────────
print("\n" + "=" * 74)
print("  T1b2 — cadeia com o recuperador NOVO contra o ANTIGO")
print("=" * 74)


# ⚠️ Imprime o JSON CRU dos dois braços, sem tentar extrair linhas por nome.
# A leitura contra a hipótese roda na máquina local, onde o esquema do
# `avaliar_t1b.py` é conhecido — um `KeyError` aqui perderia a medição inteira
# depois de a GPU já ter sido gasta.
for nome, r in resultados.items():
    print(f"\n  braço {nome}:")
    print(json.dumps(r, indent=2, ensure_ascii=False)[:3000])

(TRABALHO / "t1b2_resultado.json").write_text(json.dumps({
    "n_consultas": N_CONSULTAS, "profundidade": PROFUNDIDADE,
    "git_sha_dados": man["git_sha"], "git_sha_codigo": SHA,
    "reranqueador": RANK.name,
    "bracos": {k: str(v.name) for k, v in BRACOS.items()},
    "resultados": resultados,
    "nota": ("Os dois braços na MESMA sessão, mesmo commit, mesmas consultas. "
             "Comparar contra o numero de agosto seria invalido: entre agosto e "
             "hoje mudaram o protocolo do G1, o pool de candidatos e quatro "
             "versoes do codigo. Encolher o ganho marginal do PhiRank NAO e "
             "regressao -- o recall@100 do recuperador e o teto do reranker."),
}, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n⚠️ Baixe `t1b2_resultado.json` e os `aval_*.log`. A leitura contra a "
      "hipótese roda na máquina local.")
'''
