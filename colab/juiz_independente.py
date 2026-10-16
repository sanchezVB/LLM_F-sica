"""O juiz independente da §9.6 do DOC-13, no Colab: outro modelo corrige as mesmas respostas.

O conteúdo das células, mantido aqui como `.py` para ficar sob controle de versão e ser
testável. `scripts/gerar_notebook_colab.py --celulas juiz_independente` monta o `.ipynb`.

## O que roda

O juiz da regra, linha por linha — `assistente_bracos.julgar`, do commit exato: o mesmo
prompt, a mesma entrada, temperatura 0, a mesma semente e o mesmo leitor de veredicto —
com OUTRO modelo: Mistral Small 3.2 (24B), GGUF Q4_K_M, no `llama-server` do llama.cpp
`b11159` (o build da máquina local), no binário oficial para Linux com CUDA.

## O que sobe e o que desce

Sobe um arquivo: `data/processed/assistente/juiz2/pacote_juiz2.json` (perguntas, gabaritos
e respostas sem as marcas [n]; sem braço e sem artigo). A primeira célula o pede se ele
não estiver no Drive, e confere o sha256 contra o versionado. Desce um arquivo:
`veredictos_juiz2.jsonl`, que `scripts/conferencia_do_juiz.py --apurar` lê.

## ⚠️ ~2,5 h de L4, com a aba aberta

~1 min para baixar o llama.cpp (730 MB, conferidos por sha256; nada é compilado, a não
ser que o binário oficial não rode na imagem do dia), ~5 min para baixar o modelo (14,3 GB,
para o disco da sessão) e ~6 s por texto. Os veredictos vão para o Drive um a um: se a
sessão cair, rodar tudo de novo continua de onde parou. A última célula desliga a sessão
sozinha.
"""
from __future__ import annotations

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# O modelo e o build do llama.cpp estão em `phifm.eval.juiz_independente`: as células os
# leem de lá, do commit exato, e não há cópia aqui para divergir.
MANIFESTO = RAIZ / "data/processed/avaliacao/juiz_independente_pacote.json"
GPU = "L4"
PASTA_PADRAO = "/content/drive/MyDrive/phifm"

# O sha256 do pacote, do manifesto versionado. Sem o manifesto — o teste ainda não
# terminou e nada foi empacotado — não há o que conferir, e o notebook não é gerado.
HASHES = ({"pacote_juiz2.json": json.loads(MANIFESTO.read_text(encoding="utf-8"))["sha256"]}
          if MANIFESTO.exists() else {})


def antes_de_gerar() -> None:
    if not HASHES:
        raise SystemExit("falta o pacote: rode `scripts/conferencia_do_juiz.py --empacotar` "
                         "(depois de o teste terminar) e versione o manifesto.")


INSTRUCAO = ("**Executar tudo** (Ctrl+F9). A primeira célula pede o arquivo "
             "`D:\\LLMFísica\\data\\processed\\assistente\\juiz2\\pacote_juiz2.json` se ele "
             "ainda não estiver em `__PASTA__/juiz2/`. São ~2,5 h com a aba aberta; no fim "
             "o navegador baixa `veredictos_juiz2.jsonl` e a sessão se desliga.")

REGRA = r"""
  ── A REGRA, escrita ANTES de o teste rodar (DOC-13 §9.6, 2026-10-13) ──────

  O juiz da regra é o Qwen3-8B, o modelo que escreve as respostas. Aqui outro
  modelo corrige os mesmos textos, com o mesmo prompt e o mesmo leitor.

  ELE VALE?  κ de `certo` contra o resto, dele contra os 180 veredictos que o
             Claude deu às cegas e selou, >= 0,6. Abaixo disso a conferência
             não vale; o prompt NÃO é ajustado para ele.

  COMO SE LÊ: a decisão é a do juiz da regra. Cada eixo que depende de juiz
             (ganho no corpo, "não piora" no teste) é refeito com este:
               mesmo desfecho    -> robusto
               desfecho diferente -> FRÁGIL, e a adoção volta ao dono.

  ⚠️ Este notebook só julga. Ele não sabe de que sistema veio cada texto, e
     não imprime acerto nenhum: a apuração é LOCAL, com os braços.
"""

SETUP = r'''
# ─── 1. GPU, Drive, o pacote, o código e o llama.cpp ───────────────────────
import hashlib, http.client, json, os, shutil, subprocess, sys, time, urllib.request
from pathlib import Path

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True)
assert gpu.returncode == 0 and gpu.stdout.strip(), (
    "sem GPU. Ambiente de execução → Alterar o tipo de ambiente de execução → L4 GPU.")
NOME_GPU, MEM_GPU = [x.strip() for x in gpu.stdout.strip().splitlines()[0].split(",")]
assert int(MEM_GPU) >= 20000, (
    f"{NOME_GPU} tem {MEM_GPU} MiB: o modelo (14,3 GB) com o contexto não cabe. "
    "Ambiente de execução → Alterar o tipo de ambiente de execução → L4 GPU.")
print(f"{NOME_GPU} · {int(MEM_GPU) / 1024:.1f} GB")

from google.colab import drive
drive.mount("/content/drive")

PASTA = Path("__PASTA__") / "juiz2"
PASTA.mkdir(parents=True, exist_ok=True)
PACOTE = PASTA / "pacote_juiz2.json"
SAIDA = PASTA / "veredictos_juiz2.jsonl"

# ── O pacote: do Drive, ou enviado agora ───────────────────────────────────
HASHES = __HASHES__
if not PACOTE.exists():
    from google.colab import files
    print("Escolha o arquivo  D:\\LLMFísica\\data\\processed\\assistente\\juiz2\\"
          "pacote_juiz2.json")
    enviado = files.upload()
    assert len(enviado) == 1, "envie um arquivo só: o pacote_juiz2.json"
    PACOTE.write_bytes(next(iter(enviado.values())))
sha_pacote = hashlib.sha256(PACOTE.read_bytes()).hexdigest()
if sha_pacote != HASHES["pacote_juiz2.json"]:
    PACOTE.unlink()                      # o errado não fica no Drive para a próxima vez
    raise AssertionError(
        f"o pacote enviado tem sha256 {sha_pacote[:16]}… e o versionado é "
        f"{HASHES['pacote_juiz2.json'][:16]}…. Rode a célula de novo com o arquivo certo.")
pacote = json.loads(PACOTE.read_text(encoding="utf-8"))
print(f"✅ pacote conferido por sha256 · {len(pacote)} textos · saída em {SAIDA}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "polars", "huggingface_hub"],
               check=True)

# ── O código, do commit exato ──────────────────────────────────────────────
SHA = "__SHA__"
REPO = "__REPO__"
CODIGO = Path("/content/codigo")
if not CODIGO.exists():
    subprocess.run(["git", "clone", "--quiet", f"https://github.com/{REPO}.git",
                    str(CODIGO)], check=True)
subprocess.run(["git", "-C", str(CODIGO), "fetch", "--quiet", "--depth", "1",
                "origin", SHA], check=True)
subprocess.run(["git", "-C", str(CODIGO), "checkout", "--quiet", SHA], check=True)
if str(CODIGO / "src") not in sys.path:
    sys.path.insert(0, str(CODIGO / "src"))
from phifm.eval import assistente_bracos as b
from phifm.eval import juiz_independente as ji
from phifm.eval.assistente import SEMENTE
print(f"código em {CODIGO} · SHA {SHA[:7]}")

# ── O llama.cpp: o binário oficial do build da máquina local, com CUDA ─────
LLAMA = ji.LLAMA_CPP
DIR_LLAMA = Path("/content/llama")
FONTE_LLAMA = Path("/content/llama.cpp")
AMBIENTE = dict(os.environ)


def localizar(raiz: Path):
    """O `llama-server` debaixo de `raiz`, com as bibliotecas dele (e as do CUDA, que vêm
    no segundo arquivo) no caminho de busca. None se não houver."""
    achados = sorted(p for p in raiz.rglob("llama-server") if p.is_file())
    if not achados:
        return None
    pastas = sorted({str(p.parent) for p in raiz.rglob("*.so*")})
    AMBIENTE["LD_LIBRARY_PATH"] = ":".join(
        [*pastas, os.environ.get("LD_LIBRARY_PATH", "")]).strip(":")
    os.chmod(achados[0], 0o755)
    return achados[0]


def servidor_roda(servidor) -> bool:
    """O binário executa E enxerga a placa pelo CUDA. Sem as bibliotecas certas ele roda
    na CPU sem reclamar — por isso a lista de dispositivos, e não só `--version`."""
    if servidor is None:
        return False
    try:
        r = subprocess.run([str(servidor), "--list-devices"], capture_output=True, text=True,
                           timeout=120, env=AMBIENTE)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and "CUDA" in r.stdout + r.stderr


SERVIDOR = localizar(DIR_LLAMA) if DIR_LLAMA.exists() else None
if not servidor_roda(SERVIDOR):
    t0 = time.perf_counter()
    shutil.rmtree(DIR_LLAMA, ignore_errors=True)
    DIR_LLAMA.mkdir(parents=True)
    for nome, esperado in ji.LLAMA_BINARIOS.items():
        alvo = Path("/content") / nome
        urllib.request.urlretrieve(
            f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA}/{nome}", alvo)
        h = hashlib.sha256()
        with open(alvo, "rb") as f:
            while bloco := f.read(1 << 24):
                h.update(bloco)
        assert h.hexdigest() == esperado, (
            f"{nome}: sha256 {h.hexdigest()[:16]}… e a release declara {esperado[:16]}….")
        subprocess.run(["tar", "-xzf", str(alvo), "-C", str(DIR_LLAMA)], check=True)
        alvo.unlink()
    SERVIDOR = localizar(DIR_LLAMA)
    print(f"llama.cpp {LLAMA} baixado e conferido em {time.perf_counter() - t0:.0f} s")

if not servidor_roda(SERVIDOR):
    # O binário oficial não roda nesta imagem (outra versão do CUDA, outra glibc): compila
    # o mesmo build aqui. ~10 a 15 min; é o caminho de reserva, e avisa que foi usado.
    t0 = time.perf_counter()
    print(f"⚠️ o binário oficial não rodou; compilando o llama.cpp {LLAMA} com CUDA...",
          flush=True)
    shutil.rmtree(FONTE_LLAMA, ignore_errors=True)
    passos = [
        ["git", "clone", "--quiet", "--depth", "1", "--branch", LLAMA,
         "https://github.com/ggml-org/llama.cpp", str(FONTE_LLAMA)],
        ["cmake", "-S", str(FONTE_LLAMA), "-B", str(FONTE_LLAMA / "build"),
         "-DGGML_CUDA=ON", "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_OPENSSL=OFF",
         "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(FONTE_LLAMA / "build"), "--config", "Release",
         "-j", str(os.cpu_count() or 4), "--target", "llama-server"]]
    with open("/content/llama_build.log", "w") as log:
        for passo in passos:
            r = subprocess.run(passo, stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                log.flush()
                print(open("/content/llama_build.log").read()[-4000:])
                raise AssertionError(f"`{' '.join(passo[:3])} …` saiu com {r.returncode}; "
                                     "o fim do log está acima.")
    AMBIENTE = dict(os.environ)
    SERVIDOR = localizar(FONTE_LLAMA / "build")
    assert servidor_roda(SERVIDOR), "o llama-server compilou e não roda"
    print(f"compilado em {(time.perf_counter() - t0) / 60:.1f} min")
VERSAO_LLAMA = subprocess.run([str(SERVIDOR), "--version"], capture_output=True, text=True,
                              env=AMBIENTE).stderr.strip() or "?"
print(f"✅ llama-server {LLAMA} pronto · {VERSAO_LLAMA.splitlines()[0] if VERSAO_LLAMA else ''}")
'''

MODELO = r'''
# ─── 2. O modelo, o servidor e um ensaio de três casos ─────────────────────
from huggingface_hub import hf_hub_download

t0 = time.perf_counter()
GGUF = Path(hf_hub_download(ji.MODELO_REPO, ji.MODELO_ARQUIVO, revision=ji.MODELO_REVISAO,
                            local_dir="/content/modelo"))
assert GGUF.stat().st_size == ji.MODELO_BYTES, (
    f"{GGUF.name} tem {GGUF.stat().st_size} bytes e o fixado na regra tem {ji.MODELO_BYTES}.")
if not Path("/content/modelo/.sha_conferido").exists():
    h = hashlib.sha256()
    with open(GGUF, "rb") as f:
        while bloco := f.read(1 << 24):
            h.update(bloco)
    assert h.hexdigest() == ji.MODELO_SHA256, (
        f"{GGUF.name}: sha256 {h.hexdigest()[:16]}… e o fixado na regra é "
        f"{ji.MODELO_SHA256[:16]}…. Não é o modelo declarado antes do teste.")
    Path("/content/modelo/.sha_conferido").write_text(h.hexdigest())
print(f"✅ {GGUF.name} conferido por sha256 · {(time.perf_counter() - t0) / 60:.1f} min")

# ⚠️ A porta é uma das LIVRES, pedida ao sistema: a 8080 é do próprio Colab, e o servidor
# sai com "couldn't bind HTTP server socket" (2026-10-13, na primeira sessão).
import socket


def porta_livre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORTA = globals().get("PORTA") or porta_livre()
URL = f"http://127.0.0.1:{PORTA}"
LOG_SERVIDOR = Path("/content/llama-server.log")
processo = globals().get("processo")


def no_ar() -> bool:
    try:
        with urllib.request.urlopen(URL + "/health", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def subir_servidor(espera_s: int = 420) -> None:
    """Sobe o `llama-server` na GPU, uma conversa por vez — como o juiz da regra."""
    global processo
    if no_ar():
        return
    if processo is not None and processo.poll() is None:
        processo.kill()
    processo = subprocess.Popen(
        [str(SERVIDOR), "-m", str(GGUF), "-ngl", "99", "-c", str(ji.CONTEXTO), "-np", "1",
         "--jinja", "--host", "127.0.0.1", "--port", str(PORTA)],
        stdout=open(LOG_SERVIDOR, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        env=AMBIENTE)
    t = time.time()
    while time.time() - t < espera_s:
        assert processo.poll() is None, (
            f"o llama-server saiu com {processo.returncode}:\n"
            + LOG_SERVIDOR.read_text(errors="replace")[-3000:])
        if no_ar():
            return
        time.sleep(3)
    raise AssertionError(f"o llama-server não ficou no ar em {espera_s} s:\n"
                         + LOG_SERVIDOR.read_text(errors="replace")[-3000:])


subir_servidor()
modelo = ji.ModeloDeConversa(URL)

# O modelo tem de estar NA GPU: sem as bibliotecas do CUDA o servidor sobe na CPU sem
# reclamar, e 24B na CPU levaria dias.
usada = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True,
                           text=True).stdout.strip().splitlines()[0])
assert usada >= 12000, (
    f"só {usada} MiB da GPU em uso: o modelo não foi para a placa. O fim do log:\n"
    + LOG_SERVIDOR.read_text(errors="replace")[-3000:])
print(f"✅ modelo na GPU · {usada / 1024:.1f} GB em uso")

# O prompt como o servidor o monta para este modelo — fica na saída, para o registro.
try:
    req = urllib.request.Request(
        URL + "/apply-template",
        data=json.dumps({"messages": [{"role": "system", "content": "<SISTEMA>"},
                                      {"role": "user", "content": "<USUARIO>"}]}).encode(),
        headers={"Content-Type": "application/json"})
    FORMATO = json.loads(urllib.request.urlopen(req, timeout=30).read()).get("prompt", "?")
except Exception as e:                                       # o registro não é o juiz
    FORMATO = f"(não obtido: {e})"
print("formato de conversa:", repr(FORMATO))

# ── Ensaio: três casos que não são do pacote, de veredicto óbvio ───────────
# Se o juiz erra um destes, o defeito é de montagem (o formato, o modelo), e duas horas
# de GPU não valem a pena. Não é ajuste de prompt: o prompt é o da regra e não muda.
# Cobra-se o que a regra usa — `certo` contra o resto. Entre `errado` e `absteve` o
# próprio juiz da regra troca (deu `errado` no terceiro caso), e isso não muda número.
ENSAIO = (("A carga do elétron é −1,602 × 10⁻¹⁹ C.", True),
          ("A carga do elétron é +3,2 × 10⁻¹⁹ C.", False),
          ("As fontes não trazem o valor da carga do elétron.", False))
for resposta, e_certo in ENSAIO:
    t = time.perf_counter()
    bruto = modelo.gerar(b.SISTEMA_JUIZ,
                         "Question: Qual é a carga do elétron?\n"
                         f"Gabarito: −1,602 × 10⁻¹⁹ C.\nAnswer:\n{resposta}",
                         max_tokens=260, temperatura=0.0, semente=SEMENTE)
    veredicto, _ = b.ler_veredicto(bruto)
    print(f"  {'certo' if e_certo else 'não certo':<9} → {veredicto:<8} "
          f"{time.perf_counter() - t:4.1f} s · {bruto[:160]!r}")
    assert "{" in bruto and veredicto in b.VEREDICTOS, (
        f"o juiz não devolveu um veredicto legível: {bruto[:300]!r}. Nada foi julgado.")
    assert (veredicto == "certo") == e_certo, (
        f"o ensaio esperava {'certo' if e_certo else 'outro que não certo'} e veio "
        f"`{veredicto}`: confira o formato acima. Nada foi julgado.")
print("✅ ensaio de 3 casos: formato e leitor funcionam")
'''

JULGAR = r'''
# ─── 3. Julgar o pacote — retomável ────────────────────────────────────────
REGRA = r"""__REGRA__"""
print(REGRA, flush=True)

ja = len(ji.ler(SAIDA))
print(f"{ja} de {len(pacote)} textos já julgados", flush=True)
t0, n, quedas = time.perf_counter(), [0], 0


def ao_julgar(e, d):
    n[0] += 1
    if n[0] % 25 == 0:
        s = (time.perf_counter() - t0) / n[0]
        falta = len(pacote) - ja - n[0]
        print(f"[{ja + n[0]}/{len(pacote)}] {s:.1f} s/texto · faltam ~{falta * s / 60:.0f} min",
              flush=True)


while True:
    try:
        feitos = ji.julgar_pacote(modelo, pacote, SAIDA, ao_julgar=ao_julgar)
        break
    except (OSError, http.client.HTTPException) as erro:   # a conexão caiu: o servidor morreu?
        quedas += 1
        assert quedas <= 5, f"o servidor caiu {quedas} vezes: {erro}"
        print(f"⚠️ {erro} — subindo o servidor de novo ({quedas})", flush=True)
        subir_servidor()
assert all(e["chave"] in feitos for e in pacote)
SEGUNDOS = round(time.perf_counter() - t0, 1)
print(f"✅ {len(pacote)} textos julgados · {SEGUNDOS / 3600:.2f} h nesta sessão")
'''

RESUMO = r'''
# ─── 4. O que desce, e a sessão desligada ──────────────────────────────────
feitos = ji.ler(SAIDA)
resumo = {"experimento": "juiz_independente", "onde": "colab", "git_sha_codigo": SHA,
          "gpu": NOME_GPU, "llama_cpp": LLAMA, "versao_llama": VERSAO_LLAMA,
          "modelo": {"repo": ji.MODELO_REPO, "arquivo": ji.MODELO_ARQUIVO,
                     "revisao": ji.MODELO_REVISAO, "sha256": ji.MODELO_SHA256},
          "formato_de_conversa": FORMATO, "contexto": ji.CONTEXTO, "semente": SEMENTE,
          "pacote_sha256": sha_pacote, "textos": len(pacote),
          "ilegiveis": sum(1 for d in feitos.values() if d["veredicto"] == "ilegivel"),
          "segundos_por_texto_mediana": sorted(d["segundos"] for d in feitos.values())[
              len(feitos) // 2],
          "veredictos_sha256": hashlib.sha256(SAIDA.read_bytes()).hexdigest(),
          "terminado_em": time.strftime("%Y-%m-%dT%H:%M:%S")}
(PASTA / "juiz2.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
print(json.dumps(resumo, indent=2, ensure_ascii=False))
print("=" * 74)
print(f"No Drive, em {PASTA}: `veredictos_juiz2.jsonl` e `juiz2.json` — é o que tem de vir")
print("para D:\\LLMFísica\\data\\processed\\assistente\\juiz2\\. O navegador baixa os dois agora.")

from google.colab import files, runtime
for arq in (SAIDA, PASTA / "juiz2.json"):
    try:
        # Do disco da sessão: o Drive é desmontado logo abaixo, e o navegador ainda baixa.
        files.download(shutil.copy2(arq, f"/content/{arq.name}"))
    except Exception as e:               # sem navegador na frente: os dois estão no Drive
        print(f"  (não baixou {arq.name}: {e})")
if processo is not None and processo.poll() is None:
    processo.terminate()
drive.flush_and_unmount()
time.sleep(90)                           # o tempo de o navegador terminar de baixar
runtime.unassign()                       # para de gastar unidades
'''

CELULAS = (SETUP, MODELO, JULGAR, RESUMO)
