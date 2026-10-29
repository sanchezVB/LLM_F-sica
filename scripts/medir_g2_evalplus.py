#!/usr/bin/env python3
"""O EvalPlus do G2.3 (DOC-12 §3.7.4): gera no llama-server do G2, empacota para o Kaggle
e apura o que o Kaggle devolve.

    # DEV: 20 tarefas do MBPP fora do MBPP+ (gulosa), geradas e sanitizadas
    .venv\\Scripts\\python.exe scripts\\medir_g2_evalplus.py dev --modelo 1.7B
    # TESTE: k=10 + 1 gulosa nos 542 problemas; exige o congelamento (nota "evalplus")
    .venv\\Scripts\\python.exe scripts\\medir_g2_evalplus.py teste --modelo 1.7B --braco base
    # APURAR: lê o g2_evalplus_resultado.json do notebook e grava os agregados
    .venv\\Scripts\\python.exe scripts\\medir_g2_evalplus.py apurar --fase teste --modelo 1.7B --braco base

O código gerado NUNCA roda no Windows. Aqui só se gera, sanitiza e grava as soluções no
pacote `data/processed/kaggle_g2_evalplus/` (dataset do Kaggle); o `kaggle/g2_evalplus.py`
roda o `evalplus.evaluate` oficial num notebook de CPU sem internet. O resultado volta em
`data/processed/kaggle_g2_evalplus_retorno/g2_evalplus_resultado.json`.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import shutil
import statistics
import sys
import time
import warnings
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from phifm.eval import g2, g2_evalplus, g2_geracao  # noqa: E402

AVAL = RAIZ / "data/processed/avaliacao"
SAIDAS = AVAL / "g2_saidas"
PACOTE = RAIZ / "data/processed/kaggle_g2_evalplus"
RETORNO = RAIZ / "data/processed/kaggle_g2_evalplus_retorno"
CELULA = RAIZ / "kaggle/g2_evalplus.py"
MBPP = RAIZ / "data/raw/benchmarks/mbpp/full"
MODELOS = {"1.7B": ("Qwen3-1.7B-Base", RAIZ / "models/base/gguf/Qwen3-1.7B-Base-Q8_0.gguf"),
           "8B": ("Qwen3-8B-Base", RAIZ / "models/base/gguf/Qwen3-8B-Base-Q8_0.gguf")}
CONJUNTOS = tuple(g2_evalplus.ARQUIVOS)
# o código puro-Python que o avaliador oficial importa e o Kaggle pode não ter
VENDIDOS = ("evalplus", "wget.py", "tempdir.py", "appdirs.py", "termcolor", "multipledispatch")
N_DEV = 20


def pedido_evalplus(gulosa: bool) -> dict:
    # as paradas e o teto entram no pedido para entrarem no hash da configuração
    return g2_evalplus.amostragem(gulosa) | {"evalplus": {
        "fichas_max": g2_evalplus.FICHAS_MAX,
        "paradas": {c: g2_evalplus.paradas(c) for c in CONJUNTOS}}}


# ── dev: o MBPP fora do MBPP+ ────────────────────────────────────────────────


def sobreposicao() -> dict:
    """A sobreposição MBPP × MBPP+ por task_id (DOC-12 §3.7.11, item 9)."""
    import pyarrow.parquet as pq

    mbpp = {x["task_id"] for f in sorted(MBPP.glob("*.parquet")) for x in pq.read_table(f).to_pylist()}
    plus = {int(t["task_id"].split("/")[1]) for t in g2_evalplus.carregar("mbpp")}
    return {"mbpp": len(mbpp), "mbpp_plus": len(plus), "intersecao": len(mbpp & plus),
            "mbpp_plus_fora_do_mbpp": len(plus - mbpp), "dev_disponivel": len(mbpp - plus)}


def tarefas_dev() -> list[dict]:
    """20 tarefas do MBPP fora do MBPP+, no formato de prompt do MBPP+ (a descrição e o
    1º assert entre aspas triplas). O ponto de entrada é a função da solução que o assert
    chama."""
    import pyarrow.parquet as pq

    plus = {int(t["task_id"].split("/")[1]) for t in g2_evalplus.carregar("mbpp")}
    validas = []
    for f in sorted(MBPP.glob("*.parquet")):
        for x in pq.read_table(f).to_pylist():
            if x["task_id"] in plus:
                continue
            try:
                nomes = [n.name for n in _arvore(x["code"]).body if isinstance(n, ast.FunctionDef)]
            except SyntaxError:
                continue
            entrada = next((n for n in nomes if f"{n}(" in x["test_list"][0]), None)
            if entrada:
                validas.append({"task_id": f"MbppDev/{x['task_id']}", "entry_point": entrada,
                                "prompt": f'"""\n{x["text"]}\n{x["test_list"][0]}\n"""\n',
                                "testes": [x["test_setup_code"] or "", *x["test_list"]]})
    validas.sort(key=lambda t: int(t["task_id"].split("/")[1]))
    return sorted(random.Random(g2.SEMENTE_G2 + 500).sample(validas, N_DEV),
                  key=lambda t: int(t["task_id"].split("/")[1]))


# ── geração ──────────────────────────────────────────────────────────────────


def gerar(fase: str, modelo: str, braco: str, por_conjunto: dict[str, list[dict]],
          corridas: dict[str, tuple[bool, int]]) -> dict[str, str]:
    """Gera cada corrida (rótulo → (gulosa, k)) em todos os conjuntos. Devolve os hashes."""
    _nome, gguf = MODELOS[modelo]
    kl = json.loads((AVAL / f"g2_kl_{modelo}.json").read_text(encoding="utf-8"))
    SAIDAS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hashes = {}
    with g2_geracao.ServidorG2(gguf, log=SAIDAS / f"evalplus_{fase}_{modelo}_{braco}.servidor.log") as srv:
        for rotulo, (gulosa, k) in corridas.items():
            pedido = pedido_evalplus(gulosa)
            hc = g2_geracao.hash_da_configuracao(srv, kl["gguf"]["q8_0"]["sha256"],
                                                 "RX 7600 (Vulkan)", pedido)
            hashes[rotulo] = hc
            saida = SAIDAS / f"evalplus_{fase}_{modelo}_{braco}_{rotulo}.jsonl"
            if fase == "teste" and rotulo == "amostras":
                g2.exigir_aceite_e_congelamento()
                continua = saida.exists() and hc in {d.get("hash_config")
                                                     for d in g2_geracao.ler_jsonl(saida)}
                g2.registrar_acesso(modelo, braco, "retomada" if continua else "medida", hc,
                                    nota="evalplus")
            for conjunto, tarefas in por_conjunto.items():
                paradas = g2_evalplus.paradas("mbpp" if conjunto == "mbppdev" else conjunto)
                pedidos = [(conjunto, t["task_id"], j, g2_evalplus.consulta(t))
                           for t in tarefas for j in range(k)]
                lote = g2_geracao.gerar_lote(srv.url, pedidos, paradas, pedido, saida, hc,
                                             fichas_max=g2_evalplus.FICHAS_MAX)
                for n, a in enumerate(lote, start=1):
                    if n % 200 == 0:
                        print(f"   {rotulo} {conjunto} {n}/{len(pedidos)} · "
                              f"{time.time() - t0:.0f} s · {a.fichas} fichas ({a.parada})", flush=True)
    return hashes


def exportar(fase: str, modelo: str, braco: str, rotulo: str, hc: str, k: int,
             por_conjunto: dict[str, list[dict]]) -> dict:
    """Sanitiza as gerações e grava as soluções no pacote, um arquivo por conjunto. Só
    contadores saem daqui: nada por item."""
    saida = SAIDAS / f"evalplus_{fase}_{modelo}_{braco}_{rotulo}.jsonl"
    regs = {(d["bench"], d["item_id"], d["j"]): d for d in g2_geracao.ler_jsonl(saida)
            if d.get("hash_config") == hc and d["parada"] != "erro"}
    contadores = {}
    (PACOTE / "amostras").mkdir(parents=True, exist_ok=True)
    for conjunto, tarefas in por_conjunto.items():
        faltam = [(t["task_id"], j) for t in tarefas for j in range(k)
                  if (conjunto, t["task_id"], j) not in regs]
        if faltam:
            raise SystemExit(f"{rotulo}/{conjunto}: faltam {len(faltam)} gerações; rode de novo")
        linhas, usadas = [], []
        for t in tarefas:
            for j in range(k):
                d = regs[(conjunto, t["task_id"], j)]
                usadas.append(d)
                linha = {"task_id": t["task_id"], "solution": g2_evalplus.solucao(t, d["texto"])}
                if conjunto == "mbppdev":
                    linha["testes"] = t["testes"]
                linhas.append(linha)
        destino = PACOTE / "amostras" / f"{fase}_{modelo}_{braco}_{rotulo}_{conjunto}.jsonl"
        # ASCII puro: nenhum leitor de jsonl parte a linha num separador Unicode
        destino.write_text("".join(json.dumps(x, ensure_ascii=True) + "\n" for x in linhas),
                           encoding="utf-8")
        entradas = {t["task_id"]: t["entry_point"] for t in tarefas}
        contadores[conjunto] = {
            "amostras": len(linhas), "sha256": _sha(destino),
            "truncadas": round(sum(d["truncada"] for d in usadas) / len(usadas), 4),
            "fichas_media": round(statistics.mean(d["fichas"] for d in usadas), 1),
            "paradas": {p: sum(d["parada"] == p for d in usadas)
                        for p in sorted({d["parada"] for d in usadas})},
            "sanitizada_vazia": sum(not x["solution"].strip() for x in linhas),
            "sem_o_ponto_de_entrada": sum(f"def {entradas[x['task_id']]}(" not in x["solution"]
                                          for x in linhas),
            "sem_sintaxe": sum(not _compila(x["solution"]) for x in linhas)}
    return contadores


def _arvore(codigo: str) -> ast.Module:
    with warnings.catch_warnings():         # escapes inválidos em strings: só aviso
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(codigo)


def _compila(codigo: str) -> bool:
    try:
        _arvore(codigo)
        return True
    except (SyntaxError, ValueError):
        return False


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ── o pacote do Kaggle ───────────────────────────────────────────────────────


def notebook(celula: str) -> dict:
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": celula.splitlines(keepends=True)}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                        "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def empacotar(avaliar: list[str]) -> dict:
    """O código vendido, os dados oficiais, a célula como notebook e o MANIFESTO (sha256
    de cada arquivo, que o notebook confere antes de tudo). Sobrescreve, não apaga. O
    notebook avalia só os arquivos de `avaliar` (os desta rodada)."""
    import evalplus

    site = Path(evalplus.__file__).resolve().parents[1]
    for nome in VENDIDOS:
        origem = site / nome
        if origem.is_dir():
            shutil.copytree(origem, PACOTE / "codigo" / nome, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            (PACOTE / "codigo").mkdir(parents=True, exist_ok=True)
            shutil.copy2(origem, PACOTE / "codigo" / nome)
    (PACOTE / "dados").mkdir(parents=True, exist_ok=True)
    for arquivo in g2_evalplus.ARQUIVOS.values():
        shutil.copy2(g2_evalplus.DADOS / arquivo, PACOTE / "dados" / arquivo)
    (PACOTE / "g2_evalplus.ipynb").write_text(
        json.dumps(notebook(CELULA.read_text(encoding="utf-8")), ensure_ascii=False, indent=1),
        encoding="utf-8")
    arquivos = {p.relative_to(PACOTE).as_posix(): {"bytes": p.stat().st_size, "sha256": _sha(p)}
                for p in sorted(PACOTE.rglob("*")) if p.is_file() and p.name != "MANIFESTO.json"}
    manifesto = {"pacote": "g2_evalplus", "evalplus": evalplus.__version__,
                 "celula_sha256": _sha(CELULA), "avaliar": avaliar, "arquivos": arquivos}
    (PACOTE / "MANIFESTO.json").write_text(json.dumps(manifesto, ensure_ascii=False, indent=1) + "\n",
                                           encoding="utf-8")
    # o mesmo pacote num .zip, para subir pela web (o Kaggle descompacta mantendo as pastas)
    shutil.make_archive(str(PACOTE), "zip", root_dir=PACOTE)
    return manifesto


# ── apuração ─────────────────────────────────────────────────────────────────


def apurar(fase: str, modelo: str, braco: str) -> dict:
    """Lê o resultado do notebook e grava os agregados. Confere que o Kaggle avaliou
    exatamente as amostras daqui (sha256) e que a sanidade das canônicas passou."""
    origem = AVAL / f"g2_evalplus_{fase}_{modelo}_{braco}.json"
    r = json.loads(origem.read_text(encoding="utf-8"))
    k = json.loads((RETORNO / "g2_evalplus_resultado.json").read_text(encoding="utf-8"))
    # os contadores por tarefa (para o bootstrap pareado) ficam fora do git, com o lacre
    shutil.copy2(RETORNO / "g2_evalplus_resultado.json",
                 SAIDAS / f"evalplus_{fase}_{modelo}_{braco}_kaggle.json")
    r["kaggle"] = {c: k[c] for c in ("quando", "plataforma", "internet", "evalplus",
                                     "manifesto_ok", "sanidade") if c in k}
    for rotulo, conts in r["geracao"].items():
        resumo = {}
        for conjunto, cont in conts.items():
            nome = f"{fase}_{modelo}_{braco}_{rotulo}_{conjunto}"
            corrida = k["corridas"].get(nome)
            if corrida is None:
                raise SystemExit(f"o Kaggle não devolveu {nome}")
            if corrida["sha256_amostras"] != cont["sha256"]:
                raise SystemExit(f"{nome}: o Kaggle avaliou outras amostras (sha256 difere)")
            resumo[conjunto] = corrida["resumo"]
            r.setdefault("sha256_eval_results", {})[nome] = corrida.get("sha256_eval_results")
        if fase == "dev":
            r.setdefault("resultado", {})[rotulo] = {
                c: {"tarefas": len(v), "passam": sum(t["c_plus"] for t in v.values())}
                for c, v in resumo.items()}
        else:
            r.setdefault("resultado", {})[rotulo] = g2_evalplus.agregar(resumo)
    if fase == "teste" and braco == "base" and "gulosa" in r["resultado"]:
        r["sanidade_gulosa"] = g2_evalplus.sanidade_gulosa(modelo, r["resultado"]["gulosa"]["media_4"])
    origem.write_text(json.dumps(r, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return r


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("fase", choices=("dev", "teste", "apurar"))
    p.add_argument("--fase", dest="fase_apurada", choices=("dev", "teste"), default="teste")
    p.add_argument("--modelo", choices=tuple(MODELOS), default="1.7B")
    p.add_argument("--braco", default="base")
    a = p.parse_args()
    if a.fase == "apurar":
        r = apurar(a.fase_apurada, a.modelo, a.braco)
        print(json.dumps({c: r.get(c) for c in ("resultado", "sanidade_gulosa")}, ensure_ascii=False))
        return 0
    if a.fase == "dev":
        por_conjunto = {"mbppdev": tarefas_dev()}
        corridas = {"gulosa": (True, 1)}
    else:
        g2.exigir_aceite_e_congelamento()          # antes de LER o teste
        por_conjunto = {c: g2_evalplus.carregar(c) for c in CONJUNTOS}
        corridas = {"amostras": (False, g2_evalplus.K), "gulosa": (True, 1)}
    print(f"EvalPlus {a.fase}: " + ", ".join(f"{c} {len(v)}" for c, v in por_conjunto.items()))
    t0 = time.time()
    hashes = gerar(a.fase, a.modelo, a.braco, por_conjunto, corridas)
    geracao = {rot: exportar(a.fase, a.modelo, a.braco, rot, hashes[rot], k, por_conjunto)
               for rot, (_g, k) in corridas.items()}
    manifesto = empacotar([f"amostras/{a.fase}_{a.modelo}_{a.braco}_{rot}_{c}.jsonl"
                           for rot in corridas for c in por_conjunto])
    resultado = {"o_que_e": f"G2.3, EvalPlus, fase {a.fase} (DOC-12 §3.7.4): geração; a "
                            "execução é no Kaggle (apurar)",
                 "modelo": MODELOS[a.modelo][0], "braco": a.braco, "hashes": hashes,
                 "codigo_sha256": g2.codigo_do_corretor(RAIZ, extras=(
                     "src/phifm/eval/g2_evalplus.py", "scripts/medir_g2_evalplus.py",
                     "kaggle/g2_evalplus.py")),
                 "evalplus": manifesto["evalplus"], "segundos": round(time.time() - t0),
                 "geracao": geracao}
    if a.fase == "dev":
        resultado["sobreposicao_mbpp"] = sobreposicao()
    destino = AVAL / f"g2_evalplus_{a.fase}_{a.modelo}_{a.braco}.json"
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(geracao, ensure_ascii=False))
    print(f"pacote do Kaggle: {PACOTE} ({len(manifesto['arquivos'])} arquivos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
