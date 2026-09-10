"""T1e — a curva de PROFUNDIDADE da cadeia. Um run, três profundidades. T4.

Este arquivo é o CONTEÚDO de uma célula do Kaggle, mantido aqui como `.py` para
ficar sob controle de versão e ser testável.

## Por que existe

O diagnóstico de 2026-09-10 mediu o posto verdadeiro do alvo entre os 88.807
documentos e achou o prêmio **perto**: das 735 consultas que o recuperador perde no
top-100, a **mediana do posto é 396**, e só **10 consultas** (0,5% do total) são
inalcançáveis pelos dois métodos. Não é teto da tarefa; é o corte.

    recall@100   0,6325
    recall@200   0,7300   <- +0,098 de teto, sem treinar nada

Pela curva de volume (+0,0196 por dobra), +0,098 exigiria **cinco dobras de dado**,
~294 h de T4. A profundidade custa o dobro da reordenação e zero de treino.

Sobra uma pergunta: **o reranqueador GANHA com os candidatos a mais, ou eles só
trazem distratores?**

## ⚠️ As três profundidades saem da MESMA passagem

O escore do cross-encoder é do par (consulta, documento) e não depende de quem mais
está no conjunto. Pontuados os 200, a cadeia @100 é reordenar os 100 primeiros da
ordem densa pelos mesmos escores — zero passagens adicionais.

Medir @100 num segundo run custaria 85 min e cairia na armadilha que o T1b2
documentou: comparar sessões diferentes. Assim as três saem pareadas exatamente,
com o mesmo modelo em memória e as mesmas consultas.

## A faixa esperada, dos números que já temos

Hoje o reranqueador converte "o alvo está no conjunto" em "o alvo está no top-10" a
**46,6%** (0,2945 ÷ 0,6325). Se a taxa se mantivesse em @200, o recall@10 iria a
0,3399 — **+0,045**, o maior ganho isolado do sistema até hoje. Ela pode cair até
**13,4%** antes de @200 empatar com @100.

Mais candidatos dão mais chances *e* mais distratores. É essa a medição.
"""

CELULA = r'''
# ─── T1e / curva de profundidade — cole isto numa célula do Kaggle ──────────
import hashlib, io, json, os, subprocess, sys, tarfile, time, urllib.request, zipfile
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

# ⚠️ A chave `--sub-profundidades` é o experimento inteiro. Se o tarball for de um
# commit anterior a ela, o `avaliar_t1b.py` a recusa e a sessão morre no início —
# mas melhor no início que depois de 2h50 produzindo uma linha só.
_ajuda = subprocess.run(
    [sys.executable, str(CODIGO / "scripts/avaliar_t1b.py"), "--help"],
    capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(FONTE)})
assert "--sub-profundidades" in _ajuda.stdout, (
    f"o `avaliar_t1b.py` do SHA {SHA[:7]} não tem `--sub-profundidades`, que é o "
    "experimento inteiro. Publique de um commit que a tenha.")
print("✅ o avaliador tem --sub-profundidades")

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
PROFUNDIDADE = 200
SUB = "50,100"
EMB = MODELOS / "phiemb-do-sistema"
RANK = MODELOS / "phirank-physbert-melhor"

assert (EMB / "model.safetensors").exists(), f"o ΦEmb não tem pesos em {EMB}"
assert (RANK / "model.safetensors").exists(), f"o ΦRank não tem pesos em {RANK}"

print(f"""
{'=' * 74}
T1e — o reranqueador ganha com mais candidatos, ou só com mais distratores?

  o que motiva: o diagnóstico de 2026-09-10 mediu o posto verdadeiro do alvo
                entre 88.807 documentos. Das 735 consultas perdidas no top-100,
                a MEDIANA do posto é 396 — e só 10 consultas (0,5%) são
                inalcançáveis pelos dois métodos. O prêmio está perto.

                    recall@100  0,6325
                    recall@200  0,7300   (+0,098 de teto, sem treinar nada)

                Pela curva de volume (+0,0196 por dobra), +0,098 exigiria CINCO
                DOBRAS de dado, ~294 h de T4.

  faixa esperada: hoje o reranqueador converte "alvo no conjunto" em "alvo no
                top-10" a 46,6%. Mantida a taxa, @200 daria recall@10 0,3399
                (+0,045). Ela pode cair até 13,4% antes de @200 empatar.

  REGRA, no pareado k=10 entre @200 e @100, mesma passagem, pareamento exato:

    @200 VENCE (p<0,05)  -> a profundidade do sistema passa a 200 e o ΦRank fica
    EMPATE               -> a profundidade fica em 100. E como o T1d já mostrou
                            que o ΦRank não está estabelecido em @100, o estágio
                            de reordenação SAI do sistema
    @200 PERDE (p<0,05)  -> os distratores dominam; o @50 dirá quanto encurtar,
                            e o estágio sai também

  Segunda leitura, INDEPENDENTE: as três profundidades contra o ΦEmb sozinho. Se
  NENHUMA vencer em k=10, o estágio de reordenação não paga o próprio custo em
  profundidade alguma — e isso vale seja qual for o vencedor entre elas.

  protocolo: {N_CONSULTAS} consultas · profundidade {PROFUNDIDADE} (sub {SUB})
             universo 88.807 · `--sem-fusao`, a composição decidida no T1b2
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


# ── 5. Uma passagem, três profundidades ─────────────────────────────────────
t0 = time.perf_counter()
saida = TRABALHO / "t1e_profundidade.json"
_rodar([sys.executable, "-u", CODIGO / "scripts/avaliar_t1b.py",
        "--pares", DADOS, "--emb", EMB, "--rank", RANK, "--sem-fusao",
        "--n-consultas", N_CONSULTAS, "--profundidade", PROFUNDIDADE,
        "--sub-profundidades", SUB,
        "--out", saida, "--dispositivo", "cuda"],
       TRABALHO / "aval_profundidade.log")
custo = round(time.perf_counter() - t0, 1)
r = json.loads(saida.read_text(encoding="utf-8"))

# ── 6. A leitura, contra a regra declarada acima ────────────────────────────
print("\n" + "=" * 74)
print("  T1e — a curva de profundidade")
print("=" * 74)
for s in r["sistemas"]:
    print(f"  {s['sistema']:<22} r@1 {s['recall_1']:.4f}  r@10 {s['recall_10']:.4f}"
          f"  nDCG {s['ndcg_10']:.4f}")
print()
print("  CONFRONTO entre profundidades (a regra):")
for c in r.get("confronto_das_cadeias", []):
    if "erro" in c:
        print(f"    ⚠️ {c['erro']}")
        continue
    print(f"    k={c['k']:<3} {c['a']} {c['ganha_a']:>4} × {c['ganha_b']:<4} "
          f"{c['b']} · {c['veredito']}")
print()
print("  PAREADO contra o ΦEmb sozinho (a segunda leitura, independente):")
for x in r.get("pareado_contra_a_fusao", []):
    if "erro" in x:
        print(f"    ⚠️ {x['erro']}")
        continue
    print(f"    k={x['k']:<3} {x['b']:<22} · {x['veredito']}")
print("=" * 74)

# ⚠️ O JSON CRU também, sem as posições (volume). A leitura contra a hipótese
# roda na máquina local, onde o esquema é conhecido — um `KeyError` aqui perderia
# a medição inteira depois de a GPU já ter sido gasta.
enxuto = {**r, "sistemas": [{k: v for k, v in s.items() if k != "posicoes"}
                            for s in r["sistemas"]]}
print(json.dumps(enxuto, indent=2, ensure_ascii=False)[:3000])

(TRABALHO / "t1e_resultado.json").write_text(json.dumps({
    "n_consultas": N_CONSULTAS, "profundidade": PROFUNDIDADE,
    "sub_profundidades": SUB,
    "git_sha_dados": man["git_sha"], "git_sha_codigo": SHA,
    "recuperador": EMB.name, "reranqueador": RANK.name,
    "custo_segundos": {"total_s": custo},
    "resultado": r,
    "regra": ("Pareado k=10 entre @200 e @100, mesma passagem. @200 VENCE -> a "
              "profundidade do sistema passa a 200 e o ΦRank fica. EMPATE -> a "
              "profundidade fica em 100, e como o T1d mostrou que o ΦRank não "
              "está estabelecido em @100, o estágio SAI. @200 PERDE -> os "
              "distratores dominam e o estágio sai também. Segunda leitura "
              "INDEPENDENTE: se nenhuma profundidade vencer o ΦEmb sozinho em "
              "k=10, o estágio não paga o próprio custo em profundidade alguma."),
    "de_graca": ("As três profundidades saem da MESMA passagem: o escore do "
                 "cross-encoder é do par (consulta, documento) e não depende de "
                 "quem mais está no conjunto. Medir @100 num segundo run custaria "
                 "85 min e compararia sessões diferentes."),
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n⚠️ Baixe `t1e_resultado.json` e `aval_profundidade.log`. "
      f"Custo: {custo:.0f} s ({custo/3600:.2f} h).")
'''
