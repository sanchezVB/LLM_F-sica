"""G2.3 — a execução do EvalPlus do G2 (DOC-12 §3.7.4) no Kaggle.

Este arquivo é o CONTEÚDO de uma célula do Kaggle: notebook de **CPU**, **internet
desligada**, sem segredos (exceção registrada ao DOC-10 §3.6). O código gerado pelos
modelos só roda aqui, nunca no Windows. O dataset de entrada é o pacote
`data/processed/kaggle_g2_evalplus/`, montado por `scripts/medir_g2_evalplus.py`.

Em ordem:
1. confere cada arquivo do pacote contra o MANIFESTO.json (sha256);
2. recusa rodar com internet;
3. sanidade: as soluções canônicas passam 100% e as quebradas falham 100%, pelo
   `evalplus.evaluate` oficial (o mesmo que avalia as amostras);
4. as amostras de dev (MBPP fora do MBPP+) contra os asserts delas;
5. as amostras de teste pelo `evalplus.evaluate`, com o tempo de referência das canônicas
   medido aqui (o `.pkl` vai para a saída, para o outro braço usar o mesmo);
6. grava `/kaggle/working/g2_evalplus_resultado.json`, só com contadores (nada de código).
"""
import glob
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types
from datetime import UTC, datetime
from pathlib import Path

SAIDA = Path("/kaggle/working")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def achar_pacote():
    for m in glob.glob("/kaggle/input/**/MANIFESTO.json", recursive=True):
        d = json.loads(Path(m).read_text(encoding="utf-8"))
        if d.get("pacote") == "g2_evalplus":
            return Path(m).parent, d
    raise SystemExit("o pacote g2_evalplus não está em /kaggle/input")


PACOTE, MANIFESTO = achar_pacote()
print("pacote:", PACOTE)

# 1. o pacote chegou inteiro
ruins = [f for f, info in MANIFESTO["arquivos"].items()
         if not (PACOTE / f).is_file() or sha256(PACOTE / f) != info["sha256"]]
assert not ruins, f"arquivos diferentes do MANIFESTO: {ruins[:10]}"


# 2. sem internet
def com_internet():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3).close()
        return True
    except OSError:
        return False


assert not com_internet(), "desligue a internet do notebook (Settings → Internet: off)"

# o avaliador oficial, do pacote; os dados oficiais pelos caminhos de override
os.environ["HUMANEVAL_OVERRIDE_PATH"] = str(PACOTE / "dados/HumanEvalPlus-v0.1.10.jsonl")
os.environ["MBPP_OVERRIDE_PATH"] = str(PACOTE / "dados/MbppPlus-v0.2.0.jsonl")
sys.path.insert(0, str(PACOTE / "codigo"))
# o `datasets` (só do EvalPerf) e o `codegen` (gera com modelo; puxa o tree-sitter) não entram
for nome in ("datasets", "evalplus.codegen"):
    falso = types.ModuleType(nome)
    falso.load_dataset = falso.run_codegen = None
    sys.modules[nome] = falso
import evalplus  # noqa: E402
from evalplus.data.utils import CACHE_DIR  # noqa: E402
from evalplus.evaluate import evaluate  # noqa: E402

assert evalplus.__version__ == MANIFESTO["evalplus"], evalplus.__version__
os.makedirs(CACHE_DIR, exist_ok=True)
for pkl in sorted((PACOTE / "gt").glob("*.pkl")) if (PACOTE / "gt").is_dir() else []:
    shutil.copy2(pkl, Path(CACHE_DIR) / pkl.name)     # o tempo de referência do outro braço
    print("tempo de referência reaproveitado:", pkl.name)

TRABALHO = SAIDA / "eval"
TRABALHO.mkdir(parents=True, exist_ok=True)


def avaliar(conjunto, linhas, nome):
    """Avalia {task_id, solution} pelo evaluate oficial; devolve {task_id: {n, c_base, c_plus}}
    e o sha256 do eval_results."""
    arquivo = TRABALHO / f"{nome}.jsonl"
    arquivo.write_text("".join(json.dumps(x, ensure_ascii=True) + "\n" for x in linhas),
                       encoding="utf-8")
    evaluate(dataset=conjunto, samples=str(arquivo))
    resultado = Path(str(arquivo).replace(".jsonl", "_eval_results.json"))
    ev = json.loads(resultado.read_text(encoding="utf-8"))["eval"]
    resumo = {}
    for tid, rs in ev.items():
        resumo[tid] = {"n": len(rs),
                       "c_base": sum(r["base_status"] == "pass" for r in rs),
                       "c_plus": sum(r["base_status"] == "pass" and r["plus_status"] == "pass"
                                     for r in rs)}
    return resumo, sha256(resultado)


def problemas(conjunto):
    nome = {"humaneval": "HumanEvalPlus-v0.1.10.jsonl", "mbpp": "MbppPlus-v0.2.0.jsonl"}[conjunto]
    texto = (PACOTE / "dados" / nome).read_text(encoding="utf-8")
    return [json.loads(x) for x in texto.split("\n") if x.strip()]


resultado = {"o_que_e": "G2.3, EvalPlus: execução no Kaggle (DOC-12 §3.7.4); só contadores",
             "quando": datetime.now(UTC).isoformat(timespec="seconds"),
             "plataforma": {"python": platform.python_version(), "sistema": platform.platform(),
                            "cpus": os.cpu_count()},
             "internet": False, "manifesto_ok": True, "evalplus": evalplus.__version__,
             "celula_sha256_esperado": MANIFESTO["celula_sha256"],
             "sanidade": {}, "corridas": {}}
t0 = time.time()

# 3. sanidade: canônicas passam, quebradas falham
for conjunto in ("humaneval", "mbpp"):
    ps = problemas(conjunto)
    boas = [{"task_id": p["task_id"], "solution": p["prompt"] + p["canonical_solution"]} for p in ps]
    quebradas = [{"task_id": x["task_id"], "solution": x["solution"] + (
        f"\n\n\ndef {p['entry_point']}(*args, **kwargs):\n    raise NotImplementedError('quebrada')\n")}
        for x, p in zip(boas, ps, strict=True)]
    r_boas, _ = avaliar(conjunto, boas, f"sanidade_canonicas_{conjunto}")
    r_ruins, _ = avaliar(conjunto, quebradas, f"sanidade_quebradas_{conjunto}")
    resultado["sanidade"][conjunto] = {
        "tarefas": len(ps),
        "canonicas_passam_base": sum(v["c_base"] for v in r_boas.values()),
        "canonicas_passam_plus": sum(v["c_plus"] for v in r_boas.values()),
        "quebradas_passam_base": sum(v["c_base"] for v in r_ruins.values()),
        "canonicas_que_falham": sorted(t for t, v in r_boas.items() if v["c_plus"] < v["n"])}
    print(conjunto, resultado["sanidade"][conjunto], f"{time.time() - t0:.0f} s", flush=True)
resultado["sanidade"]["ok"] = all(
    v["canonicas_passam_plus"] == v["tarefas"] and v["quebradas_passam_base"] == 0
    for c, v in resultado["sanidade"].items() if c != "ok")


# 4 e 5. as amostras desta rodada
def rodar_dev(linhas):
    """O dev: cada solução com os asserts da própria tarefa, num processo isolado (10 s)."""
    resumo = {}
    for x in linhas:
        with tempfile.TemporaryDirectory() as d:
            arq = Path(d) / "t.py"
            arq.write_text(x["solution"] + "\n\n" + "\n".join(x["testes"]) + "\n", encoding="utf-8")
            try:
                ok = subprocess.run([sys.executable, "-I", str(arq)], cwd=d, timeout=10,
                                    capture_output=True).returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
        v = resumo.setdefault(x["task_id"], {"n": 0, "c_base": 0, "c_plus": 0})
        v["n"] += 1
        v["c_base"] += ok
        v["c_plus"] += ok
    return resumo


for rel in MANIFESTO["avaliar"]:
    nome = Path(rel).stem                          # <fase>_<modelo>_<braco>_<rótulo>_<conjunto>
    conjunto = nome.rsplit("_", 1)[1]
    texto = (PACOTE / rel).read_text(encoding="utf-8")
    linhas = [json.loads(x) for x in texto.split("\n") if x.strip()]
    if conjunto == "mbppdev":
        resumo, sha_ev = rodar_dev(linhas), None
    else:
        resumo, sha_ev = avaliar(conjunto, linhas, nome)
    resultado["corridas"][nome] = {"conjunto": conjunto, "sha256_amostras": sha256(PACOTE / rel),
                                   "sha256_eval_results": sha_ev, "resumo": resumo}
    print(nome, "avaliado", f"{time.time() - t0:.0f} s", flush=True)

# 6. a saída: o resultado e o tempo de referência das canônicas
(SAIDA / "gt").mkdir(exist_ok=True)
for pkl in Path(CACHE_DIR).glob("*.pkl"):
    shutil.copy2(pkl, SAIDA / "gt" / pkl.name)
resultado["segundos"] = round(time.time() - t0)
(SAIDA / "g2_evalplus_resultado.json").write_text(
    json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("sanidade ok:", resultado["sanidade"]["ok"], "·", resultado["segundos"], "s")
print("baixe /kaggle/working/g2_evalplus_resultado.json")
