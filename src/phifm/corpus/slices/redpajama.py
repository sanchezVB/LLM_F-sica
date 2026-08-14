"""Sprint S3b · 4b — a fatia de Física do RedPajama-arXiv, em fluxo.

## Por que vale mesmo sabendo que o RedPajama degrada

A auditoria mediu **16,6% de perda de equações** [IC 12,9–20,8] e concluiu que o
bulk pago do arXiv se justifica. Isto não contradiz aquilo: são **83,4% do texto
disponível agora, de graça**, e servem de linha de base contra a qual medir o que
os US$ 100–180 comprariam. Comprar antes de ter com o que comparar seria gastar
sem poder avaliar o gasto.

## O que "em fluxo" significa aqui, e por que não é detalhe

Os 100 shards somam **81 GB**. O caminho ingênuo baixa, salva, e depois filtra —
81 GB em disco para produzir ~35 GB de Física. O caminho certo decodifica,
filtra e descarta na mesma passada: o bruto nunca aterra.

É a mesma técnica que fez o Sprint S1 caber em 1,4 GB em vez de 150, e a mesma
lição que quatro travamentos de memória neste projeto ensinaram — todos com a
mesma causa, materializar o que podia ser percorrido.

Medido em 2026-08-14: **49,7 MB/s** de banda, então o download é ~0,5 h e o
gargalo passa a ser a decodificação de JSON.

## O filtro é o spine, não um classificador

Cada registro do RedPajama traz `meta.arxiv_id`. O spine tem os 1.595.422
identificadores que o arXiv serve no conjunto `physics`, com o rótulo do próprio
autor. A pertinência é exata e autoritativa — nada de limiar, nada de
probabilidade.

⚠️ Medido na auditoria: **57% dos registros do shard não são Física.** O shard é o
arXiv inteiro, e foi por confundir os dois que a primeira medição do S3b mediu a
população errada.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import requests

from phifm.corpus.acquire.base import user_agent

log = logging.getLogger(__name__)

INDICE = (
    "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T"
    "/resolve/main/urls/arxiv.txt"
)
# Registros por arquivo de saída. 20 mil × ~25 KB de texto ≈ 500 MB por shard,
# que é o limite confortável para o parquet e para a memória durante a escrita.
FLUSH = 20_000
# Tentativas por shard quando a REDE está indisponível. Ver o aviso em `coletar`.
MAX_TENTATIVAS = 8
# Nome do manifesto de shards concluídos. Ver `_feitos`.
MANIFESTO = "_shards_feitos.json"


@dataclass
class Progresso:
    shards_lidos: int = 0
    registros_vistos: int = 0
    registros_guardados: int = 0
    bytes_lidos: int = 0
    caracteres_guardados: int = 0
    falhas: list[str] = field(default_factory=list)

    @property
    def taxa_fisica(self) -> float:
        return self.registros_guardados / max(self.registros_vistos, 1)

    def linha(self) -> str:
        return (f"{self.shards_lidos} shards · {self.registros_vistos:,} vistos · "
                f"{self.registros_guardados:,} guardados ({100*self.taxa_fisica:.1f}%) · "
                f"{self.bytes_lidos/1e9:.1f} GB lidos · "
                f"{self.caracteres_guardados/1e9:.1f} G chars")


def ids_do_spine(spine: Path) -> set[str]:
    """Os identificadores de Física, como conjunto para consulta O(1).

    1,59 M strings curtas custam ~150 MB de RAM — barato o suficiente para
    dispensar junção em disco, e a única estrutura que permite decidir registro a
    registro sem materializar o shard.
    """
    ids = set(pl.scan_parquet(spine).select("arxiv_id").collect()["arxiv_id"])
    log.info("spine: %s identificadores de Física", f"{len(ids):,}")
    return ids


def _urls(sessao: requests.Session) -> list[str]:
    r = sessao.get(INDICE, timeout=60)
    r.raise_for_status()
    return [l.strip() for l in r.text.splitlines() if l.strip()]


def _linhas_do_shard(sessao: requests.Session, url: str, p: Progresso):
    """Percorre um shard linha a linha, sem nunca ter o shard inteiro em memória.

    O resto de linha entre pedaços é o detalhe que quebra implementações
    ingênuas: um pedaço de 1 MB corta no meio de um registro, e concatenar o
    resto com o pedaço seguinte é o que mantém o JSON válido.
    """
    resto = b""
    with sessao.get(url, timeout=600, stream=True) as r:
        r.raise_for_status()
        for pedaco in r.iter_content(1 << 20):
            p.bytes_lidos += len(pedaco)
            linhas = (resto + pedaco).split(b"\n")
            resto = linhas.pop()          # possivelmente incompleta
            yield from linhas
    if resto.strip():
        yield resto


def _gravar(buffer: list[dict], destino: Path, indice: int) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"part-{indice:05d}.parquet"
    pl.DataFrame(buffer, schema={"arxiv_id": pl.Utf8, "texto": pl.Utf8}).write_parquet(
        caminho, compression="zstd")
    log.info("→ %s (%s registros, %.0f MB)", caminho.name, f"{len(buffer):,}",
             caminho.stat().st_size / 1e6)


def _feitos(destino: Path) -> set[int]:
    """Shards já concluídos, do manifesto.

    ## ⚠️ Por que não dá para inferir dos arquivos de saída

    A primeira versão testava se `part-{n-1}.parquet` existia, tratando número de
    shard como índice de parquet. **São dois contadores diferentes**: o índice do
    parquet avança a cada 20.000 registros guardados, e um shard rende ~8.400 —
    então um parquet cobre ~2,4 shards e os dois divergem desde o começo.

    Medido em 2026-08-14, no custo real: 77 shards concluídos produziram 34
    parquets. Ao retomar, o código pulou 34 shards e reprocessou do 35 em diante,
    gravando **40.000 registros duplicados** antes de eu perceber. Não corrompe o
    corpus (a dedup por `arxiv_id` resolve) mas desperdiça ~35 GB de download e
    faz o relatório contar duas vezes.

    Estado derivado de outro estado com regra de conversão implícita é a receita —
    o manifesto guarda o que importa, explicitamente.
    """
    m = destino / MANIFESTO
    if not m.exists():
        return set()
    return set(json.loads(m.read_text(encoding="utf-8")).get("shards", []))


def _marcar(destino: Path, feitos: set[int]) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    (destino / MANIFESTO).write_text(
        json.dumps({"shards": sorted(feitos)}, indent=0), encoding="utf-8")


def coletar(destino: Path, spine: Path, max_shards: int | None = None,
            contato: str | None = None) -> Progresso:
    """Filtra os shards do RedPajama-arXiv pelo spine, gravando só a Física.

    Retomável pelo que já está em disco: um shard cujo `part-*.parquet`
    correspondente exista é pulado. Grosseiro de propósito — o custo de refazer um
    shard é 0,8 GB de download, e um manifesto com cursor por linha seria
    complexidade sem retorno aqui.

    ## ⚠️ Falha de rede ESPERA; ela não consome o shard

    Medido em 2026-08-14: o DNS caiu às 01h56 e a primeira versão queimou os 23
    shards restantes em segundos — cada um registrando "falhou" e seguindo, porque
    `getaddrinfo` falha instantaneamente e não há timeout para desacelerar. O
    relatório final disse "concluído · falhas: 23" com 76 de 100 shards.

    Erro de projeto meu: tratei indisponibilidade TRANSITÓRIA como ausência
    definitiva. É o espelho do defeito que corrigi ontem no `PoliteSession`, onde
    um 404 definitivo era repetido como se fosse transitório — a mesma confusão,
    na direção oposta.
    """
    sessao = requests.Session()
    sessao.headers.update({"User-Agent": user_agent(contato or "phifm"),
                           "Accept-Encoding": "identity"})   # o shard já é texto
    ids = ids_do_spine(spine)
    urls = _urls(sessao)
    if max_shards:
        urls = urls[:max_shards]
    log.info("%d shards a percorrer (~%.0f GB)", len(urls), 0.81 * len(urls))

    p = Progresso()
    feitos = _feitos(destino)
    # O índice do parquet continua de onde os arquivos existentes param — ele é
    # contador de SAÍDA, e não tem relação com o número do shard.
    existentes = sorted(destino.glob("part-*.parquet"))
    proximo = 1 + max((int(x.stem.split("-")[1]) for x in existentes), default=-1)
    estado = {"buffer": [], "indice": proximo}
    if feitos:
        log.info("retomando: %d shards já feitos, próximo parquet é part-%05d",
                 len(feitos), proximo)
    t0 = time.perf_counter()

    def processar(url: str) -> None:
        for linha in _linhas_do_shard(sessao, url, p):
            p.registros_vistos += 1
            try:
                d = json.loads(linha)
            except json.JSONDecodeError:
                continue
            aid = (d.get("meta") or {}).get("arxiv_id")
            if not aid or aid not in ids or not d.get("text"):
                continue
            estado["buffer"].append({"arxiv_id": aid, "texto": d["text"]})
            p.registros_guardados += 1
            p.caracteres_guardados += len(d["text"])
            if len(estado["buffer"]) >= FLUSH:
                _gravar(estado["buffer"], destino, estado["indice"])
                estado["buffer"] = []
                estado["indice"] += 1

    for n, url in enumerate(urls, 1):
        if n in feitos:
            p.shards_lidos += 1
            continue

        # `concluiu` explícito em vez de `for/else`: são TRÊS desfechos — sucesso,
        # falha definitiva e desistência por rede — e o `else` do `for` só
        # distingue dois. Só o sucesso pode marcar o shard como feito.
        concluiu = False
        for tentativa in range(MAX_TENTATIVAS):
            try:
                processar(url)
                concluiu = True
                break
            except (requests.ConnectionError, requests.Timeout) as exc:
                espera = min(30 * 2 ** tentativa, 600)
                log.warning("shard %d: rede indisponível (%s) — aguardando %d s "
                            "(tentativa %d/%d)", n, type(exc).__name__, espera,
                            tentativa + 1, MAX_TENTATIVAS)
                time.sleep(espera)
            except Exception as exc:
                p.falhas.append(f"shard {n}: {type(exc).__name__}: {str(exc)[:80]}")
                log.warning("shard %d falhou definitivamente: %s", n, exc)
                break
        if concluiu:
            feitos.add(n)
            _marcar(destino, feitos)
        elif not any(f.startswith(f"shard {n}:") for f in p.falhas):
            p.falhas.append(f"shard {n}: rede indisponível após {MAX_TENTATIVAS} tentativas")
            log.error("shard %d desistido após %d tentativas", n, MAX_TENTATIVAS)
        p.shards_lidos += 1
        dt = time.perf_counter() - t0
        log.info("shard %d/%d · %s · %.1f MB/s", n, len(urls), p.linha(),
                 p.bytes_lidos / 1e6 / max(dt, 1))

    if estado["buffer"]:
        _gravar(estado["buffer"], destino, estado["indice"])
    return p
