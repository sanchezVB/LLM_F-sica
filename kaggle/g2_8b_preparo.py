"""G2 — a Qwen3-8B-Base no Kaggle, parte 1: o GGUF e a conferência de quantização
(DOC-12 §3.7.5).

Este arquivo é o CONTEÚDO de uma célula do Kaggle: notebook **só de CPU** (não gasta a
cota de GPU), **internet LIGADA** (baixa o modelo do Hugging Face e o binário oficial do
llama.cpp), sem segredos. A entrada é o dataset `g2-8b` (`scripts/empacotar_g2_8b.py`).
Rode por "Save Version → Save & Run All"; leva de 2 a 5 h.

Em ordem:
1. confere o pacote pelo MANIFESTO e o disco (≥ 50 GB livres no rascunho);
2. instala as versões congeladas e confere o congelamento AQUI, de graça: o que falhar no
   Python da imagem falha agora, e não na sessão de GPU;
3. baixa o llama.cpp b11159 de CPU (release oficial, sha256 fixado) e o roda já, antes do
   download grande; se a glibc da imagem não servir, compila a tag (CPU);
4. baixa a Qwen3-8B-Base na revisão fixada e confere cada arquivo contra o SHA-256 dos
   arquivos da máquina local;
5. converte F16 e Q8_0 com o `convert_hf_to_gguf.py` da tag (do pacote), nas versões de
   transformers/tokenizers/numpy da conversão local, e compara o Q8_0 com o local
   (`1ee221d0…`) arquivo inteiro e tensor a tensor. Pesos diferentes PARAM aqui;
6. mede a KL do Q8_0 contra o F16 como na 1,7B (o mesmo texto de 50 itens do UG-dev,
   c=1024): a regra é ≤ 0,01 nats;
7. grava no Output (< 20 GB): o Q8_0 e os logits do F16 (`kl_f16.kld`, para a KL no motor
   CUDA na parte 2), o `g2_kl_8B.json`, os logs e `g2_8b_preparo_resultado.json` — este
   último sempre, mesmo se algo reprovar.

Depois: o Output deste notebook é entrada da parte 2 (`scripts/publicar_g2_8b.py medida`).
Não rode este notebook de novo: a entrada segue a versão mais recente do Output.
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

SAIDA = Path("/kaggle/working")
RASCUNHO = Path("/kaggle/tmp/g2_8b")
RASCUNHO.mkdir(parents=True, exist_ok=True)
os.environ.update({"HF_HOME": str(RASCUNHO / "hf"), "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
                   "HF_HUB_DISABLE_TELEMETRY": "1"})
t0 = time.time()


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while bloco := f.read(1 << 24):
            h.update(bloco)
    return h.hexdigest()


def passo(texto):
    print(f"[{(time.time() - t0) / 60:5.1f} min] {texto}", flush=True)


def rodar(cmd, log, aceitar=(0,), **kw):
    """Roda e grava a saída em `log`; levanta se o código não estiver em `aceitar`."""
    with Path(log).open("a", encoding="utf-8") as f:
        f.write(f"== {datetime.now(UTC).isoformat(timespec='seconds')} {' '.join(map(str, cmd))}\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, **kw)
    if r.returncode not in aceitar:
        raise SystemExit(f"{Path(str(cmd[0])).name} saiu com {r.returncode}; veja {log}")
    return r.returncode


resultado = {"o_que_e": "G2, a 8B no Kaggle, parte 1: GGUF e KL do Q8_0 contra o F16 "
                        "(DOC-12 §3.7.5)",
             "quando": datetime.now(UTC).isoformat(timespec="seconds"),
             "plataforma": {"python": platform.python_version(), "sistema": platform.platform(),
                            "glibc": platform.libc_ver()[1], "cpus": os.cpu_count()}}


def gravar_resultado():
    resultado["segundos"] = round(time.time() - t0)
    (SAIDA / "g2_8b_preparo_resultado.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


try:
    # 1. o pacote e o disco
    achados = [m for m in Path("/kaggle/input").rglob("MANIFESTO.json")
               if json.loads(m.read_text(encoding="utf-8")).get("pacote") == "g2_8b"]
    assert len(achados) == 1, f"o dataset g2-8b tem de estar nas entradas uma vez: {achados}"
    PACOTE = achados[0].parent
    MANIFESTO = json.loads(achados[0].read_text(encoding="utf-8"))
    ruins = [f for f, info in MANIFESTO["arquivos"].items()
             if not (PACOTE / f).is_file() or sha256(PACOTE / f) != info["sha256"]]
    assert not ruins, f"arquivos diferentes do MANIFESTO: {ruins[:10]}"
    REPO = RASCUNHO / "repo"
    with zipfile.ZipFile(PACOTE / "repo.zip.bin") as z:
        for rel, h in MANIFESTO["repo"].items():
            dados = z.read(rel)
            assert hashlib.sha256(dados).hexdigest() == h, f"repo.zip.bin: {rel} não confere"
            (REPO / rel).parent.mkdir(parents=True, exist_ok=True)
            (REPO / rel).write_bytes(dados)
    livre = shutil.disk_usage(RASCUNHO).free / 1e9
    resultado["disco_livre_gb"] = round(livre, 1)
    assert livre >= 50, f"só {livre:.0f} GB livres no rascunho; o pico é de ~42 GB"
    MODELO = json.loads((PACOTE / "modelo.json").read_text(encoding="utf-8"))
    BIN = json.loads((PACOTE / "binarios.json").read_text(encoding="utf-8"))
    passo(f"pacote conferido; {livre:.0f} GB livres")

    # 2. as versões congeladas e o congelamento, de graça, antes da GPU
    for req, extra in (("requisitos.txt", []), ("requisitos_sem_dependencias.txt", ["--no-deps"])):
        rodar([sys.executable, "-m", "pip", "install", "-q", *extra, "-r", PACOTE / req],
              SAIDA / "pip.log")
    r = subprocess.run([sys.executable, "-B", "-c", "import sys; sys.path.insert(0, 'src'); "
                        "from phifm.eval import g2; print(g2.exigir_aceite_e_congelamento())"],
                       cwd=REPO, capture_output=True, text=True)
    resultado["congelamento_no_kaggle"] = (r.stdout.strip()[:12] if r.returncode == 0
                                           else f"FALHOU: {r.stderr[-1500:]}")
    assert r.returncode == 0, f"o congelamento NÃO confere no Kaggle:\n{r.stderr[-3000:]}"
    passo("versões congeladas instaladas; o congelamento confere")

    # 3. o llama.cpp b11159 de CPU, conferido E rodado antes do download grande
    (nome, esperado), = BIN["cpu"].items()
    tgz = RASCUNHO / nome
    urllib.request.urlretrieve(BIN["base_url"] + nome, tgz)
    assert sha256(tgz) == esperado, f"{nome}: sha256 não confere"
    with tarfile.open(tgz) as t:
        t.extractall(RASCUNHO / "llama_cpu", filter="data")
    PERPLEXITY = next((RASCUNHO / "llama_cpu").rglob("llama-perplexity"))
    variante = "oficial"
    if subprocess.run([PERPLEXITY, "--version"], capture_output=True).returncode != 0:
        fonte = RASCUNHO / "llama.cpp"
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BIN["tag"],
                        "https://github.com/ggml-org/llama.cpp", fonte], check=True)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=fonte, capture_output=True,
                              text=True).stdout.strip()
        assert head == BIN["commit"], f"a tag aponta para {head}, não {BIN['commit']}"
        subprocess.run(["cmake", "-B", "build", "-DGGML_NATIVE=OFF", "-DLLAMA_CURL=OFF",
                        "-DBUILD_SHARED_LIBS=OFF", "-DCMAKE_BUILD_TYPE=Release"], cwd=fonte, check=True)
        subprocess.run(["cmake", "--build", "build", "-j", str(os.cpu_count()), "--target",
                        "llama-perplexity"], cwd=fonte, check=True)
        PERPLEXITY = fonte / "build/bin/llama-perplexity"
        variante = "compilado (CPU)"
    resultado["llama_cpu"] = variante
    passo(f"llama.cpp {BIN['tag']} de CPU: {variante}")

    # 4. a Qwen3-8B-Base na revisão fixada, conferida arquivo a arquivo
    from huggingface_hub import snapshot_download

    HF_DIR = RASCUNHO / "Qwen3-8B-Base"
    for tentativa in range(1, 4):
        try:
            snapshot_download(MODELO["repositorio"], revision=MODELO["revisao"], local_dir=HF_DIR,
                              allow_patterns=list(MODELO["arquivos"]))
            break
        except Exception as erro:  # noqa: BLE001
            passo(f"download, tentativa {tentativa}: {erro}")
            if tentativa == 3:
                raise
            time.sleep(60 * tentativa)
    diferentes = [n for n, info in MODELO["arquivos"].items()
                  if not (HF_DIR / n).is_file() or sha256(HF_DIR / n) != info["sha256"]]
    assert not diferentes, f"o download difere dos arquivos locais: {diferentes}"
    resultado["safetensors_iguais_aos_locais"] = True
    passo(f"modelo baixado e conferido: {len(MODELO['arquivos'])} arquivos")

    # 5. F16 e Q8_0 com o conversor da tag, nas versões da conversão local
    try:
        rodar([sys.executable, "-m", "pip", "install", "-q", "-r", PACOTE / "requisitos_conversao.txt"],
              SAIDA / "pip.log")
        resultado["ambiente_da_conversao"] = "as versões da conversão local (requisitos_conversao.txt)"
    except SystemExit as erro:
        resultado["ambiente_da_conversao"] = f"o da imagem (o pip das versões locais falhou: {erro})"
    versoes = subprocess.run([sys.executable, "-c", "import torch, transformers, numpy; print("
                              "torch.__version__, transformers.__version__, numpy.__version__)"],
                             capture_output=True, text=True).stdout.split()
    CONVERSOR = REPO / "ferramentas/llama.cpp-conversor-b11159"
    ambiente = (f"convert_hf_to_gguf.py da tag b11159 ({BIN['commit'][:9]}), no Kaggle; Python "
                f"{platform.python_version()}, torch {versoes[0]}, transformers {versoes[1]}, "
                f"numpy {versoes[2]}; KL no llama-perplexity de CPU ({variante})")
    F16 = RASCUNHO / "Qwen3-8B-Base-F16.gguf"
    Q8 = SAIDA / "Qwen3-8B-Base-Q8_0.gguf"
    for tipo, destino in (("f16", F16), ("q8_0", Q8)):
        rodar([sys.executable, CONVERSOR / "convert_hf_to_gguf.py", HF_DIR, "--outtype", tipo,
               "--outfile", destino], SAIDA / "conversao_8b.log")
        passo(f"{tipo} convertido: {destino.stat().st_size / 1e9:.2f} GB")
    shutil.rmtree(HF_DIR)                      # 16 GB de rascunho livres para a KL
    q8_sha = sha256(Q8)
    sys.path.insert(0, str(REPO / "src"))
    from phifm.eval import g2_kaggle

    comparacao = g2_kaggle.comparar_mapas(g2_kaggle.mapa_gguf(Q8, CONVERSOR / "gguf-py"),
                                          MODELO["mapa_q8_0"])
    resultado["q8_0"] = {"bytes": Q8.stat().st_size, "sha256": q8_sha,
                         "igual_ao_local": q8_sha == MODELO["q8_0_local"]["sha256"], **comparacao}
    resultado["conversao"] = ambiente
    gravar_resultado()
    passo(f"Q8_0 {q8_sha[:12]}… igual ao local: {resultado['q8_0']['igual_ao_local']}; "
          f"metadados diferentes {comparacao['metadados_diferentes'][:5]}; "
          f"tensores diferentes {len(comparacao['tensores_diferentes'])}")
    assert not comparacao["tensores_diferentes"], (
        "os PESOS do Q8_0 daqui diferem do local: parar e decidir qual vale antes da KL")

    # 6. KL do Q8_0 contra o F16, como na 1,7B: c=1024, o texto de 50 itens do UG-dev
    TEXTO = REPO / "models/base/gguf/kl_dev.txt"
    KLD = SAIDA / "kl_f16.kld"                 # vai ao Output: a parte 2 mede a KL no CUDA
    LOG_KL = SAIDA / "kl_8b.log"
    rodar([PERPLEXITY, "-m", F16, "-f", TEXTO, "-c", "1024", "--kl-divergence-base", KLD],
          SAIDA / "kl_8b_f16.log")
    passo("logits do F16 gravados")
    rodar([PERPLEXITY, "-m", Q8, "-f", TEXTO, "-c", "1024", "--kl-divergence-base", KLD,
           "--kl-divergence"], LOG_KL)
    passo("KL medida")
    # o registro é o mesmo script da 1,7B (sai com 1 se a KL não passar: o JSON sai igual)
    rodar([sys.executable, REPO / "scripts/registrar_kl_g2.py", "--tamanho", "8B", "--log", LOG_KL,
           "--q8", Q8, "--f16", F16, "--conversao", ambiente], SAIDA / "registrar_kl.log",
          aceitar=(0, 1))
    kl = json.loads((REPO / "data/processed/avaliacao/g2_kl_8B.json").read_text(encoding="utf-8"))
    shutil.copy2(REPO / "data/processed/avaliacao/g2_kl_8B.json", SAIDA / "g2_kl_8B.json")
    resultado["kl"] = {c: kl[c] for c in ("kl_media_nats", "kl_erro_padrao", "passa",
                                          "mesma_ficha_mais_provavel_pct", "razao_ppl_q8_f16")}
    passo(f"KL média {kl['kl_media_nats']} nats (≤ 0,01: {'PASSA' if kl['passa'] else 'NÃO PASSA'})")
except BaseException as erro:
    resultado["erro"] = f"{type(erro).__name__}: {erro}"[:3000]
    raise
finally:
    gravar_resultado()
    usado = sum(p.stat().st_size for p in SAIDA.rglob("*") if p.is_file()) / 1e9
    print(f"Output: {usado:.2f} GB · baixe /kaggle/working/g2_8b_preparo_resultado.json e "
          "/kaggle/working/g2_kl_8B.json")
