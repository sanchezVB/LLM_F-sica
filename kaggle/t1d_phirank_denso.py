"""T1d — o ΦRank retreinado nos negativos que ele VÊ. T4, custo zero.

Este arquivo é o CONTEÚDO de uma célula do Kaggle, mantido aqui como `.py` para
ficar sob controle de versão e ser testável.

## Por que existe

O ΦRank instalado (`phirank-physbert-melhor`) foi treinado com negativos difíceis
minerados pela **fusão RRF**. Em 2026-09-08 a regra pré-registrada do T1b2 tirou o
BM25 da composição: a cadeia é `ΦEmb → ΦRank`. **A distribuição de treino dele
deixou de ser a distribuição que ele vê**, e isso é a causa mecânica provável de o
ganho marginal dele ter encolhido de p=0,0081 para p=0,086.

Todo o argumento do `minerar_do_recuperador.py` é *treinar na distribuição do
teste*. Os negativos novos foram minerados do top-50 do ΦEmb e filtrados por
co-citação: 16.391 grupos, 41,96 negativos por grupo.

## ⚠️ Uma variável: a DISTRIBUIÇÃO dos negativos, não a quantidade deles

`--max-grupos 12500` é o mesmo do T1c, e os outros seis hiperparâmetros são
byte a byte os dele. Os negativos novos têm 16.391 grupos disponíveis e **2.102
ficam de fora de propósito**: usá-los mudaria distribuição *e* volume na mesma
rodada, e o ganho não seria atribuível.

Este projeto já perdeu um experimento por mudar duas coisas de uma vez.

## ⚠️ Os DOIS reordenadores, na MESMA sessão

A pergunta é "quanto mudou", e essa exige o braço de referência medido na mesma
sessão. Em 2026-09-08 o T1b2 mediu o recuperador antigo junto e descobriu que o
número histórico (0,1666) era **0,1685** no protocolo de hoje: comparar contra o
histórico teria reportado **sete vezes** o efeito real.

Cabe numa sessão porque `--sem-fusao` reordena **uma** vez por consulta em vez de
duas — a fusão que a chave descarta é justamente a composição que a regra já tirou.

## ⚠️ E o pareado que a regra nomeia é CALCULADO aqui

Em 2026-09-08 a regra do T1b2 pedia um confronto entre duas cadeias, o run mediu as
duas, e o teste não pôde ser calculado depois: as posições por consulta não eram
gravadas. Deu para decidir por aritmética, mas foi sorte do tamanho do efeito.

Agora o `avaliar_t1b.py` grava `posicoes` em cada sistema, e a célula roda o
`mcnemar_em` entre os dois braços. O veredito da regra sai da execução, não de uma
conta feita depois.
"""

CELULA = r'''
# ─── T1d / ΦRank nos negativos densos — cole isto numa célula do Kaggle ─────
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
            f"{nome}: hash difere. Upload truncado ou dataset trocado — treinar "
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
for exigido in ("src/phifm/eval/hibrido.py", "scripts/avaliar_t1b.py",
                "scripts/train_rerank.py"):
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

# ── 4. O protocolo e a REGRA, declarados antes de qualquer número ───────────
N_CONSULTAS = 2000
PROFUNDIDADE = 100
EMB = MODELOS / "phiemb-do-sistema"
ANTIGO = MODELOS / "phirank-physbert-melhor"
BASE = "thellert/physbert_cased"
NEGATIVOS = DADOS / "pares_do_recuperador_denso_limpos.parquet"

# ⚠️ Byte a byte os do T1c, e `--max-grupos` inclusive. Os negativos novos têm
# 16.391 grupos e 2.102 ficam de fora DE PROPÓSITO: usá-los mudaria distribuição
# E volume na mesma rodada, e o ganho não seria atribuível. Uma variável.
HIPER = ["--max-grupos", 12500, "--grupos", 2, "--n-negativos", 7,
         "--lr", 2e-5, "--max-tokens", 384, "--semente", 17]

assert (EMB / "model.safetensors").exists(), f"o ΦEmb não tem pesos em {EMB}"
assert (ANTIGO / "model.safetensors").exists(), f"o ΦRank antigo não está em {ANTIGO}"
assert NEGATIVOS.exists(), f"os negativos densos não estão em {NEGATIVOS}"

print(f"""
{'=' * 74}
T1d — o ΦRank retreinado nos negativos que ele VÊ

  hipótese: o ΦRank instalado foi treinado com negativos minerados pela FUSÃO
            RRF, que deixou de ser a composição em 2026-09-08. A distribuição
            de treino dele não é mais a que ele vê, e é a causa mecânica
            provável de o ganho marginal ter encolhido de p=0,0081 para p=0,086.
            Treinar nos negativos do top-50 do ΦEmb deve recuperar parte dele.

  UMA variável: a distribuição dos negativos. Os sete hiperparâmetros são os do
                T1c, `--max-grupos 12500` incluído — 2.102 grupos disponíveis
                ficam de fora para não mudar volume junto.

  REGRA, no pareado k=10 entre as duas cadeias, mesmas consultas, mesma sessão:

    NOVO VENCE (p<0,05)   -> instala; a hipótese da distribuição fica confirmada
    EMPATE                -> NÃO substitui. Trocar peça do sistema sem evidência
                             é risco sem retorno; a hipótese fica sem apoio
    NOVO PERDE (p<0,05)   -> hipótese REFUTADA, e é resultado negativo real:
                             negativos do recuperador PIOR eram melhores

  Segunda leitura, INDEPENDENTE do confronto acima: se NENHUMA das duas cadeias
  vencer o ΦEmb sozinho em k=10, o estágio de reordenação não paga o próprio
  custo com este recuperador — e isso vai para o registro de todo jeito.

  protocolo: {N_CONSULTAS} consultas · profundidade {PROFUNDIDADE} · universo 88.807
             `--sem-fusao`: a composição decidida, uma reordenação por consulta
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


# ── 5. Treinar o novo ───────────────────────────────────────────────────────
import time

t0 = time.perf_counter()
saida = TRABALHO / "phirank-denso"
_rodar([sys.executable, "-u", CODIGO / "scripts/train_rerank.py",
        "--negativos", NEGATIVOS, "--out", saida, "--base", BASE,
        *HIPER, "--dispositivo", "cuda"],
       TRABALHO / "treino_denso.log")
NOVO = saida.parent / f"{saida.name}-melhor"
assert (NOVO / "model.safetensors").exists(), (
    f"{NOVO} não existe — o treino terminou sem gravar um melhor")
custo_treino = round(time.perf_counter() - t0, 1)
print(f"\n✅ treino em {custo_treino:.0f} s ({custo_treino/60:.0f} min)", flush=True)

# ── 6. Medir os dois braços, na MESMA sessão ────────────────────────────────
BRACOS = {"novo": NOVO, "antigo": ANTIGO}
resultados, custos = {}, {"treino_s": custo_treino}
for nome, rank in BRACOS.items():
    t1 = time.perf_counter()
    alvo = TRABALHO / f"t1d_{nome}.json"
    _rodar([sys.executable, "-u", CODIGO / "scripts/avaliar_t1b.py",
            "--pares", DADOS, "--emb", EMB, "--rank", rank, "--sem-fusao",
            "--n-consultas", N_CONSULTAS, "--profundidade", PROFUNDIDADE,
            "--out", alvo, "--dispositivo", "cuda"],
           TRABALHO / f"aval_{nome}.log")
    resultados[nome] = json.loads(alvo.read_text(encoding="utf-8"))
    custos[f"aval_{nome}_s"] = round(time.perf_counter() - t1, 1)

# ── 7. O CONFRONTO que a regra nomeia, calculado aqui ───────────────────────
# ⚠️ Em 2026-09-08 a regra do T1b2 pedia um pareado entre duas cadeias, o run
# mediu as duas e o teste não pôde ser calculado depois — as posições por consulta
# não eram gravadas. Agora são, e o veredito sai da execução.
from phifm.eval.hibrido import mcnemar_em


def _posicoes(res, sistema):
    for s in res["sistemas"]:
        if s["sistema"] == sistema:
            return s["posicoes"]
    raise SystemExit(
        f"sistema {sistema!r} não está no artefato; presentes: "
        f"{[s['sistema'] for s in res['sistemas']]}")


CADEIA = "ΦEmb+ΦRank"
confronto = [
    mcnemar_em(_posicoes(resultados["antigo"], CADEIA),
               _posicoes(resultados["novo"], CADEIA), k,
               "ΦRank ANTIGO (negativos da fusão)",
               "ΦRank NOVO (negativos densos)")
    for k in (1, 10)]

# E a segunda leitura: cada cadeia contra o recuperador sozinho, que já sai de
# dentro de cada braço (`pareado_contra_a_fusao`, com o ΦEmb como referência).
print("\n" + "=" * 74)
print("  T1d — CONFRONTO das duas cadeias (a regra pré-registrada)")
print("=" * 74)
for c in confronto:
    if "erro" in c:
        print(f"  ⚠️ {c['erro']}")
        continue
    print(f"  top-{c['k']:<3} antigo {c['ganha_a']:>3} × {c['ganha_b']:<3} novo "
          f"· {c['veredito']}")
print()
print("  A regra: novo vence -> instala; empate -> NÃO substitui; novo perde ->")
print("           hipótese da distribuição REFUTADA (resultado negativo real).")
print("=" * 74)

# ⚠️ Imprime o JSON CRU dos dois braços, sem tentar extrair linhas por nome.
# A leitura contra a hipótese roda na máquina local, onde o esquema do
# `avaliar_t1b.py` é conhecido — um `KeyError` aqui perderia a medição inteira
# depois de a GPU já ter sido gasta. As `posicoes` saem do dump por volume.
for nome, r in resultados.items():
    print(f"\n  braço {nome}:")
    enxuto = {**r, "sistemas": [{k: v for k, v in s.items() if k != "posicoes"}
                                for s in r["sistemas"]]}
    print(json.dumps(enxuto, indent=2, ensure_ascii=False)[:2500])

(TRABALHO / "t1d_resultado.json").write_text(json.dumps({
    "n_consultas": N_CONSULTAS, "profundidade": PROFUNDIDADE,
    "git_sha_dados": man["git_sha"], "git_sha_codigo": SHA,
    "recuperador": EMB.name, "base_do_rank": BASE,
    "negativos": NEGATIVOS.name,
    "hiperparametros": [str(x) for x in HIPER],
    "bracos": {k: v.name for k, v in BRACOS.items()},
    "custo_segundos": custos,
    "confronto_das_cadeias": confronto,
    "resultados": resultados,
    "regra": ("Pareado k=10 entre as duas cadeias, mesmas consultas, mesma "
              "sessão. NOVO VENCE (p<0,05) -> instala. EMPATE -> não substitui: "
              "trocar peça do sistema sem evidência é risco sem retorno. NOVO "
              "PERDE -> hipótese da distribuição de treino REFUTADA, e é "
              "resultado negativo real. Segunda leitura independente: se nenhuma "
              "cadeia vencer o ΦEmb sozinho em k=10, o estágio de reordenação "
              "não paga o próprio custo com este recuperador."),
    "uma_variavel": ("Só a distribuição dos negativos muda. Os sete "
                     "hiperparâmetros são os do T1c, `--max-grupos 12500` "
                     "incluído: os negativos novos têm 16.391 grupos e 2.102 "
                     "ficam de fora para não mudar volume junto."),
}, indent=2, ensure_ascii=False), encoding="utf-8")
print("\n⚠️ Baixe `t1d_resultado.json`, `treino_denso.log` e os `aval_*.log`.")
'''
