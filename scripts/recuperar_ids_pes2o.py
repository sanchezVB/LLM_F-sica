#!/usr/bin/env python3
"""Passo 2 do acervo maior: o id do Semantic Scholar de cada resumo novo do peS2o.

    P = .venv-treino\\Scripts\\python.exe scripts\\recuperar_ids_pes2o.py
    P scripts\\recuperar_ids_pes2o.py --max-arquivos 1     # ensaio com um arquivo

Depois do passo 1 (`contar_pes2o_fora_do_arxiv.py`), que grava o sha256 de cada novo.
Baixa o peS2o v2 da MESMA revisão que o filtro de Física usou (`_filtragem.json`), um
arquivo por vez, procura os novos pelo hash do texto e apaga o arquivo baixado
(`phifm.corpus.pes2o_ids`). Para quando achou todos. É retomável: cada arquivo terminado
fica marcado.

Rede: até ~87 GB se precisar ler a coleção inteira (os resumos vêm antes, então deve
parar antes). Disco: um arquivo por vez (~4 GB). Memória: os alvos (~200 MB) e um bloco.

Grava `data/processed/pes2o_novos/ids/` (fora do git) e o resumo em
`data/processed/avaliacao/pes2o_ids.json` (vai ao git).

⚠️ Segura a trava de rodada pesada de `medir_passagens.py` enquanto baixa: o HD é o mesmo
das medidas. Se outra sessão estiver medindo, ele recusa e não começa.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "scripts"))

from phifm.core.console import utf8  # noqa: E402

utf8()

import polars as pl  # noqa: E402

from phifm.corpus import pes2o_ids as pi  # noqa: E402

PES2O = RAIZ / "data/processed/pes2o_fisica"
NOVOS = RAIZ / "data/processed/pes2o_novos/novos.parquet"
SAIDA = RAIZ / "data/processed/pes2o_novos/ids"
AGREGADO = RAIZ / "data/processed/avaliacao/pes2o_ids.json"
CONJUNTO = "allenai/peS2o"
PREFIXO = "data/v2/"              # o mesmo de `filtrar_hf.FONTES["pes2o"]`
EXTENSOES = (".parquet", ".json.gz", ".jsonl.gz")


def _progresso(saida: Path) -> dict:
    arq = saida / "progresso.json"
    return json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else {"feitos": []}


def _ja_achados(saida: Path) -> set[int]:
    partes = sorted(saida.glob("ids-*.parquet"))
    if not partes:
        return set()
    return pi.alvos(pl.concat([pl.read_parquet(p, columns=["sha256"]) for p in partes])["sha256"])


def executar(novos: Path, revisao: str, arquivos: list[str], baixar, ler, saida: Path,
             agregado: Path, max_arquivos: int | None = None) -> dict:
    """O laço retomável, com download e leitura injetados (o `main` passa os de
    `hf_filtrado`; os testes, arquivos locais)."""
    saida.mkdir(parents=True, exist_ok=True)
    prog = _progresso(saida)
    if prog.get("revisao") not in (None, revisao):
        raise SystemExit(f"a retomada é da revisão {prog['revisao'][:12]}, e o filtro usou "
                         f"{revisao[:12]}: apague {saida} para recomeçar.")
    todos = pi.alvos(pl.read_parquet(novos, columns=["sha256"])["sha256"])
    ja = _ja_achados(saida)
    feitos = set(prog["feitos"])
    alvo_agora = todos - ja
    t0 = time.perf_counter()
    na_colecao = len(arquivos)
    if max_arquivos is not None:
        arquivos = [a for a in arquivos if a not in feitos][:max_arquivos]

    def gravar(nome: str, achados: list[dict]) -> None:
        k = len(list(saida.glob("ids-*.parquet")))
        pl.DataFrame(achados, schema={"sha256": pl.Utf8, "s2_id": pl.Utf8, "data": pl.Utf8,
                                      "origem": pl.Utf8}).write_parquet(saida / f"ids-{k:04d}.parquet")
        feitos.add(nome)
        (saida / "progresso.json").write_text(json.dumps(
            {"revisao": revisao, "feitos": sorted(feitos)}, indent=1), encoding="utf-8")

    def ao_terminar(nome: str, achados: int, faltam: int) -> None:
        print(f"{nome}: {achados:,} achados · faltam {faltam:,} · "
              f"{(time.perf_counter() - t0) / 60:.0f} min", flush=True)

    r = pi.recuperar(alvo_agora, arquivos, baixar, ler, feitos, gravar, ao_terminar)
    achados = len(_ja_achados(saida) & todos)
    resumo = {"o_que_e": "o id do Semantic Scholar dos resumos de Física do peS2o fora do "
                         "índice (passo 2), achados pelo sha256 do texto no peS2o original",
              "conjunto": CONJUNTO, "revisao": revisao, "prefixo": PREFIXO,
              "novos": len(todos), "com_id": achados, "sem_id": len(todos) - achados,
              "fracao_com_id": round(achados / len(todos), 4) if todos else None,
              "arquivos_feitos": len(feitos), "arquivos_na_colecao": na_colecao,
              "completo": r["parou_cedo"] or achados == len(todos),
              "gravado_em": time.strftime("%Y-%m-%d %H:%M")}
    agregado.parent.mkdir(parents=True, exist_ok=True)
    agregado.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo


def main() -> int:
    import requests

    import medir_passagens as mp
    from phifm.core.env import contato_obrigatorio
    from phifm.corpus.acquire.base import user_agent
    from phifm.corpus.slices import hf_filtrado as hf

    p = argparse.ArgumentParser()
    p.add_argument("--max-arquivos", type=int, default=None)
    p.add_argument("--temp", type=Path, default=RAIZ / "data/processed/_temp_pes2o_ids")
    a = p.parse_args()
    if not NOVOS.exists():
        raise SystemExit(f"falta {NOVOS}: rode o passo 1 (contar_pes2o_fora_do_arxiv.py).")
    filtragem = PES2O / "_filtragem.json"
    revisao = json.loads(filtragem.read_text(encoding="utf-8")).get("revisao") \
        if filtragem.exists() else None
    if not revisao or revisao == "main":
        raise SystemExit(f"{filtragem} não registra a revisão do peS2o que o filtro usou: sem "
                         "ela, baixar de `main` pode trazer outro texto, e o hash não casa.")
    sessao = requests.Session()
    sessao.headers.update({"User-Agent": user_agent(contato_obrigatorio())})
    arquivos = [n for n, _ in hf._arquivos(sessao, CONJUNTO, EXTENSOES)[0]
                if n.startswith(PREFIXO)]
    if not arquivos:
        raise SystemExit(f"nenhum arquivo em {PREFIXO} na listagem do {CONJUNTO}.")
    a.temp.mkdir(parents=True, exist_ok=True)

    def baixar(nome: str) -> Path:
        local = a.temp / Path(nome).name
        for tentativa in range(hf.MAX_TENTATIVAS):
            try:
                hf._baixar(sessao, CONJUNTO, nome, local, revisao)
                return local
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(min(30 * 2 ** tentativa, 600))
        raise SystemExit(f"{nome}: rede indisponível depois de {hf.MAX_TENTATIVAS} tentativas.")

    def ler(local: Path):
        try:
            yield from hf._ler_blocos(local)
        finally:
            local.unlink(missing_ok=True)          # o bruto não fica

    with mp.trava("medir_passagens"):
        r = executar(NOVOS, revisao, arquivos, baixar, ler, SAIDA, AGREGADO, a.max_arquivos)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    print(f"→ {AGREGADO.relative_to(RAIZ)} · ids em {SAIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
