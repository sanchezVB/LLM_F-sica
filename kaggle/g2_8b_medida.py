"""G2 — a Qwen3-8B-Base no Kaggle, parte 2: as medidas (DOC-12 §3.7.5 e §3.7.11).

Este arquivo é o CONTEÚDO de uma célula do Kaggle: notebook **GPU T4 x2**, **internet
LIGADA** (instala as versões congeladas dos pacotes e baixa o binário oficial do llama.cpp),
sem segredos. Rode por "Save Version → Save & Run All": a célula para sozinha antes das 12 h
(e antes de a cota acabar) e grava o estado no Output ao longo do caminho. Entradas:
- o dataset `g2-8b` (`scripts/empacotar_g2_8b.py`), já com o `g2_kl_8B.json` da parte 1;
- o dataset `g2-8b-gguf`, feito do Output da parte 1 (o Q8_0 e os logits do F16);
- da 2ª sessão em diante: o dataset `g2-8b-estado`, feito do Output da sessão ANTERIOR (a
  versão mais nova; a célula recusa outra).

Antes de cada sessão, ajuste as três linhas abaixo:
- FASE = "sanidade" na 1ª sessão (SESSAO = 1, ~1 h 30 de cota):
  1. o GPU, o driver e o binário, escolhido por um teste de verdade nas duas T4;
  2. a KL do Q8_0 no próprio motor (CUDA) contra os logits do F16 da parte 1;
  3. a vazão (`llama-batched-bench`, os parâmetros da 1,7B): duas placas em camadas, uma
     placa, e duas placas cada uma com o seu processo; a memória do servidor do G2;
  4. a sanidade da 8B-base em 300 itens do UG-dev (§3.7.11 item 8), primeiro cortada por
     prazo e retomada, para a retomada ser testada antes de qualquer item de teste.
  Para aqui: o dono decide os cortes (§3.7.11 item 1) e libera o teste.
- FASE = "teste" nas seguintes (SESSAO = 2, 3, …): MMLU, G2.1, IFEval e a geração do
  EvalPlus, nessa ordem (a guarda do instrumento vem antes do acesso mais caro). Cada uma
  retoma do que já gravou; uma etapa nova não começa a menos de 1 h do prazo; a guarda do
  MMLU (76,89 ± 3) reprovada para tudo; a etapa só conta como feita com os contadores
  completos.
- COTA_RESTANTE_H: a cota de GPU que o Kaggle mostra antes de rodar.

Desde a sessão 1, deixe o notebook em "Settings → Environment → Pin to original
environment": a imagem, o driver e o binário entram no hash, e uma sessão em outro
ambiente para antes de medir.
"""
FASE = "sanidade"          # "sanidade" (só na sessão 1) ou "teste"
SESSAO = 1                 # 1, 2, 3, … : some 1 a cada sessão
COTA_RESTANTE_H = 30.0     # a cota de GPU que resta, como o Kaggle mostra

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
import zipfile  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from pathlib import Path  # noqa: E402

T0 = time.time()
PRAZO = T0 + 3600 * min(10.75, COTA_RESTANTE_H - 0.75)
MARGEM_ETAPA_S = 3600          # uma medida nova não começa com menos que isto até o prazo
SAIDA = Path("/kaggle/working")
RASCUNHO = Path("/kaggle/tmp/g2_8b")
RASCUNHO.mkdir(parents=True, exist_ok=True)
assert FASE in ("sanidade", "teste"), FASE
assert (FASE == "sanidade") == (SESSAO == 1), "a sanidade é a sessão 1, e o teste vem depois"
assert PRAZO - T0 > 2 * 3600, "cota de menos de 2 h 45: não comece uma sessão assim"


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while bloco := f.read(1 << 24):
            h.update(bloco)
    return h.hexdigest()


def passo(texto):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {texto}", flush=True)


def smi(*campos):
    r = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(campos)}", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return [x.strip() for x in r.stdout.strip().split("\n") if x.strip()]


def processos_na_gpu():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return [x.strip() for x in r.stdout.split("\n") if x.strip()]


def unico(nome, chave=None, valor=None):
    """O único arquivo `nome` nas entradas (com `chave == valor`, se dado); None se nenhum.
    Mais de um é ambiguidade, e a célula recusa."""
    achados = []
    for m in sorted(Path("/kaggle/input").rglob(nome)):
        if chave is None or json.loads(m.read_text(encoding="utf-8")).get(chave) == valor:
            achados.append(m)
    assert len(achados) <= 1, f"mais de um {nome} nas entradas: {achados}"
    return achados[0] if achados else None


# ── 0. o GPU, antes de tudo ──────────────────────────────────────────────────
gpus = smi("name", "driver_version", "memory.total")
passo(f"GPUs: {gpus}")
assert len(gpus) == 2 and all("T4" in g for g in gpus), "o acelerador tem de ser GPU T4 x2"

# ── 1. o pacote, conferido; o repositório extraído BYTE A BYTE ───────────────
m = unico("MANIFESTO.json", "pacote", "g2_8b")
assert m, "o dataset g2-8b não está nas entradas"
PACOTE = m.parent
MANIFESTO = json.loads(m.read_text(encoding="utf-8"))
ruins = [f for f, info in MANIFESTO["arquivos"].items()
         if not (PACOTE / f).is_file() or sha256(PACOTE / f) != info["sha256"]]
assert not ruins, f"arquivos diferentes do MANIFESTO: {ruins[:10]}"
assert MANIFESTO.get("kl_8b_pronta"), "o pacote ainda não tem o g2_kl_8B.json da parte 1"
REPO = RASCUNHO / "repo"
with zipfile.ZipFile(PACOTE / "repo.zip.bin") as z:
    for rel, h in MANIFESTO["repo"].items():
        dados = z.read(rel)
        assert hashlib.sha256(dados).hexdigest() == h, f"repo.zip.bin: {rel} não confere"
        (REPO / rel).parent.mkdir(parents=True, exist_ok=True)
        (REPO / rel).write_bytes(dados)
sys.path.insert(0, str(REPO / "src"))
from phifm.eval import g2_kaggle  # noqa: E402

passo(f"pacote conferido: {len(MANIFESTO['repo'])} arquivos do repositório")

# ── 2. o estado da sessão anterior ───────────────────────────────────────────
e = unico("ESTADO.json", "estado", "g2_8b")
if SESSAO == 1:
    assert e is None, "a sessão 1 começa sem estado: tire o g2-8b-estado das entradas"
else:
    assert e, f"a sessão {SESSAO} precisa do estado da sessão {SESSAO - 1} (g2-8b-estado)"
    g2_kaggle.restaurar_estado(e.parent, REPO, sessao_esperada=SESSAO - 1)
    passo(f"estado da sessão {SESSAO - 1} restaurado")
AV = REPO / "data/processed/avaliacao"
SAIDAS = AV / "g2_saidas"
SAIDAS.mkdir(parents=True, exist_ok=True)
LLAMA_ARQ = SAIDAS / "llama_8B.json"
BIN = json.loads((PACOTE / "binarios.json").read_text(encoding="utf-8"))


def compilado_guardado():
    """O binário compilado na sessão 1, que volta pelo estado (g2-8b-estado) e vai de novo
    para o Output desta sessão, para a seguinte também o ter. Chamado ANTES do 1º salvar():
    um estado salvo sem o binário deixaria a sessão seguinte sem ele."""
    man = unico("MANIFESTO_LLAMA.json", "commit", BIN["commit"])
    assert man, "a variante é 'compilado', mas o binário guardado não está nas entradas"
    info = json.loads(man.read_text(encoding="utf-8"))
    ruins = [n for n, h in info["arquivos"].items() if sha256(man.parent / n) != h]
    assert not ruins, f"o binário compilado guardado não confere: {ruins}"
    pasta = SAIDA / "llama_compilado"
    shutil.copytree(man.parent, pasta, dirs_exist_ok=True)
    for n in info["arquivos"]:
        (pasta / n).chmod(0o755)
    return pasta, str(pasta)


if LLAMA_ARQ.exists() and not json.loads(LLAMA_ARQ.read_text(encoding="utf-8"))["variante"].startswith("oficial"):
    compilado_guardado()
if FASE == "teste":
    assert (AV / "g2_dev_8B_base_it1.json").exists() and LLAMA_ARQ.exists(), \
        "o estado não traz a sanidade (sessão 1): o teste não começa sem ela"

resultado = {"o_que_e": "G2, a 8B no Kaggle, parte 2: as medidas (DOC-12 §3.7)", "fase": FASE,
             "sessao": SESSAO, "quando": datetime.now(UTC).isoformat(timespec="seconds"),
             "etapas": {}}


def salvar():
    """O estado no Output (zip com SHA-256) e o resumo, com as linhas da 8B no livro."""
    m = g2_kaggle.salvar_estado(REPO, SAIDA / "estado", sessao=SESSAO)
    resultado["livro_8B"] = m["livro_8B"]
    resultado["segundos"] = round(time.time() - T0)
    (SAIDA / "g2_8b_medida_resultado.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


salvar()      # o estado segue adiante mesmo que esta sessão caia cedo

# ── 3. o Q8_0 da parte 1, o mesmo da KL ──────────────────────────────────────
KL = json.loads((AV / "g2_kl_8B.json").read_text(encoding="utf-8"))
gguf = unico("Qwen3-8B-Base-Q8_0.gguf")
assert gguf, "o g2-8b-gguf (Output da parte 1) não está nas entradas"
assert sha256(gguf) == KL["gguf"]["q8_0"]["sha256"], "o Q8_0 não é o da conferência de KL"
assert KL["passa"], (f"a KL do Q8_0 contra o F16 ({KL['kl_media_nats']} nats) não passa a regra "
                     "(≤ 0,01, §3.7.5): nenhuma medida da 8B")
Q8 = REPO / "models/base/gguf/Qwen3-8B-Base-Q8_0.gguf"
Q8.parent.mkdir(parents=True, exist_ok=True)
Q8.symlink_to(gguf)
subprocess.run(["cat", str(gguf)], stdout=subprocess.DEVNULL, check=True)   # no cache do disco
passo(f"Q8_0 conferido: {KL['gguf']['q8_0']['sha256'][:12]}…")

# ── 4. as versões congeladas dos pacotes e o congelamento ────────────────────
for req, extra in (("requisitos.txt", []), ("requisitos_sem_dependencias.txt", ["--no-deps"])):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *extra, "-r", PACOTE / req],
                   check=True)
r = subprocess.run([sys.executable, "-B", "-c", "import sys; sys.path.insert(0, 'src'); "
                    "from phifm.eval import g2; print(g2.exigir_aceite_e_congelamento())"],
                   cwd=REPO, capture_output=True, text=True)
assert r.returncode == 0, f"o congelamento NÃO confere no Kaggle:\n{r.stderr[-3000:]}"
passo(f"congelamento conferido ({r.stdout.strip()[:12]}…)")

# ── 5. o llama.cpp b11159 com CUDA, escolhido por um teste de verdade ────────
# O release oficial (sha256 fixado) traz o sm_75 só em PTX (o driver compila na 1ª carga) e
# foi compilado no Ubuntu 24.04. Se ele não gerar fichas nas duas T4, a sessão 1 compila a
# tag para sm_75 (com as bibliotecas CUDA junto) e guarda o binário no Output. A variante,
# o SHA-256 do binário, o driver e a imagem da sessão 1 ficam no estado (llama_8B.json) e
# entram no rótulo da GPU, que entra no hash da configuração; uma sessão seguinte com outro
# driver, outra imagem ou outro binário PARA antes de qualquer item de teste.
ALVOS = ("llama-server", "llama-batched-bench", "llama-perplexity")
os.environ["CUDA_CACHE_MAXSIZE"] = "4294967296"
os.environ["PYTHONUNBUFFERED"] = "1"


def oficial():
    pasta = RASCUNHO / "llama_cuda"
    for nome, esperado in BIN["cuda"].items():
        tgz = RASCUNHO / nome
        if not tgz.exists():
            urllib.request.urlretrieve(BIN["base_url"] + nome, tgz)
        assert sha256(tgz) == esperado, f"{nome}: sha256 não confere"
        with tarfile.open(tgz) as t:
            t.extractall(pasta, filter="data")
    servidor = next(pasta.rglob("llama-server"))
    cudart = next(p.parent for p in pasta.rglob("libcudart.so*"))
    return servidor.parent, f"{cudart}:{servidor.parent}"


def compilar():
    """A tag b11159 para sm_75, com as .so de CUDA que o binário usa, guardada no Output."""
    fonte = RASCUNHO / "llama.cpp"
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BIN["tag"],
                    "https://github.com/ggml-org/llama.cpp", fonte], check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=fonte, capture_output=True,
                          text=True).stdout.strip()
    assert head == BIN["commit"], f"a tag {BIN['tag']} aponta para {head}, não {BIN['commit']}"
    subprocess.run(["cmake", "-B", "build", "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75",
                    "-DGGML_NATIVE=OFF", "-DLLAMA_CURL=OFF", "-DBUILD_SHARED_LIBS=OFF",
                    "-DCMAKE_BUILD_TYPE=Release"], cwd=fonte, check=True)
    subprocess.run(["cmake", "--build", "build", "-j", str(os.cpu_count()), "--target", *ALVOS],
                   cwd=fonte, check=True)
    pasta = SAIDA / "llama_compilado"
    pasta.mkdir(parents=True, exist_ok=True)
    for n in ALVOS:
        shutil.copy2(fonte / "build/bin" / n, pasta / n)
    ldd = subprocess.run(["ldd", str(pasta / "llama-server")], capture_output=True, text=True).stdout
    for linha in ldd.split("\n"):
        achado = re.search(r"(lib(?:cudart|cublas|cublasLt)\.so\S*) => (\S+)", linha)
        if achado:
            shutil.copy2(achado.group(2), pasta / achado.group(1))
    arquivos = {p.name: sha256(p) for p in sorted(pasta.iterdir()) if p.name != "MANIFESTO_LLAMA.json"}
    (pasta / "MANIFESTO_LLAMA.json").write_text(json.dumps(
        {"commit": BIN["commit"], "cuda_arch": "75", "arquivos": arquivos}, indent=1), encoding="utf-8")
    return pasta, str(pasta)


def gera_nas_duas(pasta, ld):
    """Um bench mínimo em cada T4 (código de saída e fichas/s): prova que os kernels rodam
    e já faz a compilação do PTX, que senão cairia dentro da primeira medida."""
    for gpu in ("0", "1"):
        amb = os.environ | {"LD_LIBRARY_PATH": f"{ld}:{os.environ.get('LD_LIBRARY_PATH', '')}",
                            "CUDA_VISIBLE_DEVICES": gpu}
        try:
            r = subprocess.run([pasta / "llama-batched-bench", "-m", Q8, "-c", "2048", "-ngl", "99",
                                "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-npp", "64", "-ntg",
                                "16", "-npl", "1", "--output-format", "jsonl"],
                               capture_output=True, text=True, env=amb, timeout=1800)
        except subprocess.TimeoutExpired:
            return False, f"o bench mínimo passou de 30 min na GPU {gpu}"
        linhas = [json.loads(x) for x in r.stdout.split("\n") if x.startswith("{")]
        if r.returncode != 0 or not linhas or not linhas[0].get("speed_tg"):
            return False, (r.stdout + r.stderr)[-1500:]
    return True, ""


def plataforma(variante, pasta):
    imagem = next((x.split("=", 1)[1].strip('"') for x in Path("/etc/os-release").read_text().split("\n")
                   if x.startswith("PRETTY_NAME=")), "?")
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", subprocess.run(["nvidia-smi"], capture_output=True,
                                                               text=True).stdout)
    return {"variante": variante, "commit": BIN["commit"],
            "llama_server_sha256": sha256(pasta / "llama-server"),
            "driver": smi("driver_version")[0], "cuda_do_driver": cuda.group(1) if cuda else "?",
            "imagem": imagem, "python": platform.python_version(),
            "glibc": platform.libc_ver()[1]}


if SESSAO == 1:
    PASTA, LD = oficial()
    ok, diag = gera_nas_duas(PASTA, LD)
    variante = "oficial 12.8"
    if not ok:
        passo(f"o binário oficial não gera nas duas T4; compilando a tag:\n{diag}")
        PASTA, LD = compilar()
        ok, diag = gera_nas_duas(PASTA, LD)
        variante = "compilado sm_75"
    assert ok, f"nenhum binário gera nas duas T4:\n{diag}"
    PLAT = plataforma(variante, PASTA)
    LLAMA_ARQ.write_text(json.dumps(PLAT, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
else:
    PLAT = json.loads(LLAMA_ARQ.read_text(encoding="utf-8"))
    PASTA, LD = oficial() if PLAT["variante"].startswith("oficial") else compilado_guardado()
    agora = plataforma(PLAT["variante"], PASTA)
    mudou = [k for k in ("llama_server_sha256", "driver", "imagem", "python", "glibc")
             if agora[k] != PLAT[k]]
    resultado["plataforma_agora"] = agora
    assert not mudou, (
        f"o ambiente mudou desde a sessão 1 ({mudou}): o hash da configuração seria outro no "
        "meio da medida. Nada foi medido nesta sessão. Caminhos (o dono decide): esperar o "
        "ambiente antigo (o notebook fixado em 'Pin to original environment'), ou trazer o "
        "estado (scripts/trazer_g2_8b.py), reempacotar e começar outra campanha da sessão 1, "
        f"com o hash novo (a base é re-medida).\nantes: {PLAT}\nagora: {agora}")
    ok, diag = gera_nas_duas(PASTA, LD)
    assert ok, f"o binário da sessão 1 não gera mais nas duas T4:\n{diag}"
GPU = (f"2×T4 (CUDA, llama.cpp {BIN['tag']} {PLAT['variante']}, driver {PLAT['driver']}, "
       f"{PLAT['imagem']}, Python {PLAT['python']})")
os.environ.update({"LD_LIBRARY_PATH": f"{LD}:{os.environ.get('LD_LIBRARY_PATH', '')}",
                   "G2_LLAMA_SERVER": str(PASTA / "llama-server"), "G2_GPU": GPU})
resultado["plataforma"] = PLAT
resultado["gpu_no_hash"] = GPU
salvar()
passo(f"llama.cpp: {PLAT['variante']} · rótulo no hash: {GPU}")


# ── as etapas ────────────────────────────────────────────────────────────────
LIVRO = REPO / g2_kaggle.LIVRO
ultimo = {"livro": LIVRO.stat().st_size, "quando": time.time()}


def salvar_se_preciso():
    """A cada minuto: salva já se o livro mudou (a linha do acesso chega ao Output em ~1 min)
    e, de resto, a cada 20 min (as gerações)."""
    tamanho = LIVRO.stat().st_size
    if tamanho != ultimo["livro"] or time.time() - ultimo["quando"] >= 1200:
        salvar()
        ultimo.update(livro=tamanho, quando=time.time())


def medir(rotulo, script, *args, prazo=None):
    """Uma medida congelada, como subprocesso com prazo, salvando o estado pelo caminho."""
    passo(f"{rotulo}: começa")
    st = g2_kaggle.rodar_com_prazo(
        [sys.executable, "-u", "-X", "utf8", REPO / "scripts" / script, *args],
        prazo or PRAZO, SAIDA / f"{rotulo}.log", cwd=REPO, env=dict(os.environ),
        periodico=salvar_se_preciso, intervalo_s=60)
    sobra = processos_na_gpu()
    resultado["etapas"][rotulo] = st if not sobra else f"{st}; processos na GPU: {sobra}"
    passo(f"{rotulo}: {resultado['etapas'][rotulo]}")
    salvar()
    return st


def tentar(rotulo, funcao):
    """Uma conferência da sanidade que não derruba a sessão: o erro vai para o resultado."""
    try:
        resultado[rotulo] = funcao()
    except (Exception, SystemExit) as erro:  # noqa: BLE001 — o ServidorG2 sai por SystemExit
        resultado[rotulo] = {"erro": f"{type(erro).__name__}: {erro}"[:2000]}
    salvar()
    passo(f"{rotulo}: {str(resultado[rotulo])[:300]}")


def kl_no_motor():
    """A KL do Q8_0 no CUDA (o motor da medida) contra os logits do F16 da parte 1."""
    kld = unico("kl_f16.kld")
    assert kld, "os logits do F16 (kl_f16.kld, Output da parte 1) não estão nas entradas"
    log = SAIDA / "kl_8b_cuda.log"
    with log.open("w", encoding="utf-8") as f:
        r = subprocess.run([PASTA / "llama-perplexity", "-m", Q8, "-c", "1024", "-ngl", "99",
                            "--kl-divergence-base", kld, "--kl-divergence"],
                           stdout=f, stderr=subprocess.STDOUT)
    texto = log.read_text(encoding="utf-8", errors="replace")
    kl = re.search(r"Mean\s+KLD\s*:\s*(-?[\d.]+)", texto)
    topo = re.search(r"Same top p\s*:\s*(-?[\d.]+)", texto)
    d = {"o_que_e": "KL do Q8_0 contra o F16 no motor da medida da 8B (CUDA nas T4), com os "
                    "logits do F16 da parte 1 (CPU); a regra é a mesma (≤0,01 nats)",
         "codigo": r.returncode, "kl_media_nats": float(kl.group(1)) if kl else None,
         "mesma_ficha_mais_provavel_pct": float(topo.group(1)) if topo else None,
         "gpu": GPU}
    d["passa"] = d["kl_media_nats"] is not None and d["kl_media_nats"] <= 0.01
    (AV / "g2_kl_8B_cuda.json").write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n",
                                           encoding="utf-8")
    return d


def bench(rotulo, gpus_visiveis, npl="1,4,6,8", ntg="512,2048"):
    """A vazão com os parâmetros da 1,7B (g2_vazao_1.7B.json), KV q8_0 e flash attention."""
    amb = os.environ | ({"CUDA_VISIBLE_DEVICES": gpus_visiveis} if gpus_visiveis else {})
    with (SAIDA / f"vazao_{rotulo}.jsonl").open("w", encoding="utf-8") as f, \
         (SAIDA / f"vazao_{rotulo}.log").open("w", encoding="utf-8") as erro:
        subprocess.run([PASTA / "llama-batched-bench", "-m", Q8, "-c", "49152", "-b", "2048",
                        "-ub", "512", "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
                        "-npp", "3500", "-ntg", ntg, "-npl", npl, "--output-format", "jsonl"],
                       stdout=f, stderr=erro, env=amb, check=True)
    linhas = [json.loads(x) for x in (SAIDA / f"vazao_{rotulo}.jsonl").read_text(encoding="utf-8").split("\n")
              if x.startswith("{")]
    return [{"paralelos": x["pl"], "prompt": x["pp"], "geradas_por_sequencia": x["tg"],
             "fichas_s_gerando": round(x["speed_tg"], 1),
             "fichas_s_lendo_prompt": round(x["speed_pp"], 1)} for x in linhas]


def dois_processos():
    """Duas T4, cada uma com o seu bench ao mesmo tempo (4 paralelos, 512 fichas): a vazão
    de um servidor por placa, que o código congelado ainda NÃO roda (seria emenda)."""
    procs = []
    for gpu in ("0", "1"):
        amb = os.environ | {"CUDA_VISIBLE_DEVICES": gpu}
        f = (SAIDA / f"vazao_dois_{gpu}.jsonl").open("w", encoding="utf-8")
        procs.append((f, subprocess.Popen(
            [PASTA / "llama-batched-bench", "-m", Q8, "-c", "16384", "-b", "2048", "-ub", "512",
             "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-npp", "3500",
             "-ntg", "512", "-npl", "4", "--output-format", "jsonl"],
            stdout=f, stderr=subprocess.DEVNULL, env=amb)))
    total = 0.0
    for f, p in procs:
        p.wait()
        f.close()
        linhas = [json.loads(x) for x in Path(f.name).read_text(encoding="utf-8").split("\n")
                  if x.startswith("{")]
        total += linhas[0]["speed_tg"] if linhas else 0.0
    return round(total, 1)


def vazao():
    ref = json.loads((AV / "g2_vazao_1.7B.json").read_text(encoding="utf-8"))
    medidas = {"duas_placas_camadas": bench("duas", None), "uma_placa": bench("uma", "0")}
    v4 = next(x["fichas_s_gerando"] for x in medidas["duas_placas_camadas"]
              if x["paralelos"] == 4 and x["geradas_por_sequencia"] == 512)
    v_dois = dois_processos()
    escala = ref["horas_estimadas_por_modelo"]["total"] * ref["vazao_usada_na_conta"]
    d = {"o_que_e": "ensaio de vazão do G2 na 8B (DOC-12 §3.7.11 item 1): llama-batched-bench "
                    f"b11159, Qwen3-8B-Base Q8_0, {GPU}, flash attention, KV q8_0",
         "data": datetime.now(UTC).date().isoformat(), "medidas": medidas,
         "slots_escolhidos": 4, "vazao_usada_na_conta": v4,
         "vazao_um_processo_por_placa_somada": v_dois,
         # as mesmas cargas da 1,7B (16,8 h a 178,6 fichas/s), na vazão daqui
         "horas_estimadas_total_um_servidor": round(escala / v4, 1),
         "horas_estimadas_total_um_servidor_por_placa": round(escala / v_dois, 1) if v_dois else None,
         "premissa_da_regra_fichas_s_da_1_7B": ref["premissa_da_regra_fichas_s"],
         "suposta_da_8B": None,
         "corte_disparado": None,
         "decisao": "do dono (§3.7.11 item 1): cortes de k e/ou um servidor por placa (emenda)"}
    (AV / "g2_vazao_8B.json").write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8")
    return {k: d[k] for k in ("vazao_usada_na_conta", "vazao_um_processo_por_placa_somada",
                              "horas_estimadas_total_um_servidor",
                              "horas_estimadas_total_um_servidor_por_placa")}


def memoria_e_subida():
    """O ServidorG2 com as flags do G2: o tempo de subida e a memória de cada placa com 4
    pedidos de 512 fichas ao mesmo tempo."""
    from concurrent.futures import ThreadPoolExecutor

    from phifm.eval import g2_geracao

    srv = g2_geracao.ServidorG2(Q8, log=SAIDA / "memoria_servidor.log")
    t = time.time()
    try:
        srv.iniciar(espera_s=900)       # a 1ª subida pode compilar o PTX
        subida = round(time.time() - t)
        corpo = json.dumps({"prompt": "The history of physics " * 400, "n_predict": 512,
                            "temperature": 0.0, "cache_prompt": False}).encode()

        def um(_):
            req = urllib.request.Request(srv.url + "/completion", data=corpo,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as resp:
                return json.loads(resp.read()).get("tokens_predicted")

        with ThreadPoolExecutor(4) as ex:
            futuros = [ex.submit(um, i) for i in range(4)]
            time.sleep(20)
            usada = smi("memory.used")
            geradas = [f.result() for f in futuros]
    finally:
        srv.parar()                     # também quando a subida falha: nada órfão na GPU
    return {"executavel": str(srv.executavel), "flags": srv.flags(), "subida_s": subida,
            "memoria_usada_por_placa": usada, "fichas_geradas": geradas}


if FASE == "sanidade":
    tentar("kl_no_motor", kl_no_motor)
    tentar("vazao", vazao)
    tentar("memoria", memoria_e_subida)
    args = ("dev", "--modelo", "8B", "--iteracao", "1", "--ug", "300")
    # a retomada testada antes do teste: cortada por prazo, conferida e retomada
    st = medir("sanidade_ug300_cortada", "medir_g2.py", *args, prazo=time.time() + 480)
    jsonl = SAIDAS / "dev_8B_base_it1.jsonl"
    feitas = sum(1 for x in jsonl.read_bytes().split(b"\n") if x.strip()) if jsonl.exists() else 0
    resultado["retomada"] = {"cortada": st, "amostras_antes": feitas,
                             "processos_na_gpu_depois_do_corte": processos_na_gpu()}
    medir("sanidade_ug300", "medir_g2.py", *args)
    if (AV / "g2_dev_8B_base_it1.json").exists():
        d = json.loads((AV / "g2_dev_8B_base_it1.json").read_text(encoding="utf-8"))
        resultado["sanidade_ug300"] = {"por_benchmark": d["por_benchmark"], "hash_config": d["hash_config"],
                                       "correcoes_mortas_no_teto": d["correcoes_mortas_no_teto"]}
else:
    mmlu_agg = AV / "g2_mmlu_teste_8B_base.json"
    ETAPAS = (("mmlu", "medir_g2_mmlu.py"), ("g2_1", "medir_g2.py"),
              ("ifeval", "medir_g2_ifeval.py"), ("evalplus", "medir_g2_evalplus.py"))
    for rotulo, script in ETAPAS:
        # a guarda só vale com o MMLU COMPLETO: um MMLU cortado (itens sem previsão contam
        # como erro) não reprova o instrumento; ele é refeito na tentativa seguinte
        if (SAIDAS / "etapa_mmlu_8B.ok").exists() and not json.loads(
                mmlu_agg.read_text(encoding="utf-8")).get("guarda_do_instrumento", {}).get("ok"):
            resultado["parada"] = "a guarda do instrumento do MMLU reprovou: a medida para (§3.7.4)"
            break
        marcador = SAIDAS / f"etapa_{rotulo}_8B.ok"
        if marcador.exists():
            resultado["etapas"][rotulo] = "feita antes"
            continue
        for tentativa in (1, 2):            # a 2ª refaz as amostras 'erro' da 1ª
            if PRAZO - time.time() < MARGEM_ETAPA_S:
                resultado["parada"] = f"menos de 1 h até o prazo antes de {rotulo}"
                break
            st = medir(f"{rotulo}_{tentativa}", script, "teste", "--modelo", "8B", "--braco", "base")
            if st == "prazo":
                break
            if st == "falhou":              # o EvalPlus sai com 1 se faltam gerações ('erro')
                continue
            completa, porque = g2_kaggle.conferir_etapa(rotulo, REPO)
            resultado["etapas"][f"{rotulo}_{tentativa}_conferida"] = porque
            if completa:
                marcador.write_text(datetime.now(UTC).isoformat(timespec="seconds") + "\n",
                                    encoding="utf-8")
                break
        salvar()
        if not marcador.exists():
            resultado.setdefault("parada", f"{rotulo} não terminou completa nesta sessão")
            break

salvar()
usado = sum(p.stat().st_size for p in SAIDA.rglob("*") if p.is_file())
passo(f"fim: {resultado['etapas']} · {resultado.get('parada', '')} · Output {usado / 1e9:.2f} GB")
print(f"baixe /kaggle/working/g2_8b_medida_resultado.json e a pasta estado/; para a sessão "
      f"{SESSAO + 1}, faça uma nova versão do dataset g2-8b-estado com o Output desta versão")
