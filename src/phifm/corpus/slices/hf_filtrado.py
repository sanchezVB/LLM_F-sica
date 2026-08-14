"""Sprint S3 · 4d — peS2o e OpenWebMath filtrados pelo classificador.

## Por que estas duas fontes precisam de classificador, e o RedPajama não

O RedPajama-arXiv traz `meta.arxiv_id`, então a Física se decide por pertinência
ao spine: exata, autoritativa, com o rótulo do próprio autor. O peS2o e o
OpenWebMath não têm identificador do arXiv — são papers científicos em geral e
texto matemático de web. Aqui a decisão é probabilística, e é onde o classificador
do S2 entra.

## O limiar é 0,9, e o motivo é medido

Deixa-um-domínio-de-fora, 2026-08-14:

| domínio omitido | FP a 0,5 | FP a 0,9 |
|---|---|---|
| cs | 9,8% | 1,5% |
| math | 35,4% | 13,6% |
| q-bio | 31,2% | 12,0% |
| stat | 2,9% | 0,4% |

A 0,5 o falso positivo num vizinho próximo não visto chega a 35%. A 0,9 cai para
~13%, ao custo de ~7 pontos de revocação. O corpus é abundante e um documento
irrelevante contamina o treino mais do que um relevante ausente o empobrece
(DOC-02 §6) — então o limiar alto é a troca certa.

## ⚠️ O que NENHUM número nosso responde

Todos os falsos positivos acima foram medidos em **resumos do arXiv**. O
OpenWebMath é **texto de web** — nota de aula, fórum, blog — e o peS2o é paper
científico completo de todas as áreas. Nenhum domínio do arXiv representa essas
distribuições, e extrapolar de um resumo de `math.AP` para um tópico de fórum é
troca de distribuição maior que qualquer uma que medimos.

Por isso este módulo **amostra a saída** e reporta a distribuição de domínios de
URL do que foi aceito. Não é um número de contaminação — é a evidência que permite
a um humano formar um. Inventar uma taxa de contaminação para texto de web a
partir do que medimos no arXiv seria o tipo de extrapolação que este projeto
registra como erro.

## Fluxo por arquivo

Baixa um arquivo, processa, apaga. Nunca há mais de um em disco. O peS2o são
307,7 GB em 240 arquivos; aterrar tudo custaria 3/4 do HD para produzir uma
fração.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import polars as pl
import requests

from phifm.corpus.acquire.base import user_agent

log = logging.getLogger(__name__)

API = "https://huggingface.co/api/datasets/{}"
RESOLVE = "https://huggingface.co/datasets/{}/resolve/main/{}"

# Ver §"o limiar é 0,9" na docstring.
LIMIAR = 0.9
FLUSH = 20_000
# Documentos guardados para inspeção humana. 400 é o que uma pessoa revisa numa
# sessão; mais que isso vira arquivo que ninguém abre.
N_AMOSTRA = 400
# Tentativas por arquivo quando a rede cai. Ver o aviso em `filtrar`.
MAX_TENTATIVAS = 8


@dataclass
class Filtragem:
    fonte: str
    arquivos_lidos: int = 0
    vistos: int = 0
    aceitos: int = 0
    bytes_lidos: int = 0
    caracteres_aceitos: int = 0
    # Distribuição de domínios de URL do que foi ACEITO. É o sinal objetivo sobre
    # o que está entrando, quando não há rótulo para conferir.
    dominios: Counter = field(default_factory=Counter)
    amostra: list[dict] = field(default_factory=list)
    falhas: list[str] = field(default_factory=list)

    @property
    def taxa(self) -> float:
        return self.aceitos / max(self.vistos, 1)


class Classificador:
    """O `is_physics` do S2, aplicado em lote."""

    def __init__(self, modelo: Path):
        with open(modelo / "model.pkl", "rb") as f:
            self.pipe = pickle.load(f)
        self.meta = json.loads((modelo / "meta.json").read_text(encoding="utf-8"))
        classes = list(self.pipe.classes_)
        self.i_fisica = classes.index("fisica")
        log.info("classificador de %s · classes %s", modelo, classes)

    def scores(self, textos: list[str]) -> list[float]:
        return self.pipe.predict_proba(textos)[:, self.i_fisica].tolist()


def _arquivos(sessao: requests.Session, ds: str, ext: tuple[str, ...]) -> list[tuple[str, int]]:
    r = sessao.get(API.format(ds) + "?blobs=true", timeout=120)
    r.raise_for_status()
    fs = [(f["rfilename"], f.get("size") or 0)
          for f in r.json().get("siblings", [])
          if f["rfilename"].endswith(ext)]
    return sorted(fs)


def _baixar(sessao: requests.Session, ds: str, nome: str, destino: Path) -> int:
    destino.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with sessao.get(RESOLVE.format(ds, nome), timeout=1800, stream=True) as r:
        r.raise_for_status()
        with open(destino, "wb") as f:
            for pedaco in r.iter_content(1 << 22):
                f.write(pedaco)
                n += len(pedaco)
    return n


def _texto_e_url(df: pl.DataFrame) -> tuple[list[str], list[str | None]]:
    """Extrai texto e URL do esquema de cada fonte, sem supor um formato só."""
    cols = set(df.columns)
    for c in ("text", "texto", "content", "abstract"):
        if c in cols:
            textos = df[c].fill_null("").to_list()
            break
    else:
        raise ValueError(f"nenhuma coluna de texto reconhecida em {sorted(cols)}")
    for c in ("url", "metadata", "source"):
        if c in cols:
            urls = df[c].to_list()
            break
    else:
        urls = [None] * len(textos)
    return textos, urls


def filtrar(
    ds: str,
    destino: Path,
    modelo: Path,
    *,
    limiar: float = LIMIAR,
    max_arquivos: int | None = None,
    ext: tuple[str, ...] = (".parquet",),
    temp: Path | None = None,
    contato: str | None = None,
) -> Filtragem:
    """Baixa, filtra e descarta arquivo por arquivo. Retomável pelo que há em disco."""
    sessao = requests.Session()
    sessao.headers.update({"User-Agent": user_agent(contato or "phifm")})
    clf = Classificador(modelo)
    temp = temp or (destino.parent / "_temp")
    arquivos = _arquivos(sessao, ds, ext)
    if max_arquivos:
        arquivos = arquivos[:max_arquivos]
    tot = sum(s for _, s in arquivos)
    log.info("%s · %d arquivos · %.1f GB", ds, len(arquivos), tot / 1e9)

    f = Filtragem(fonte=ds)
    buffer: list[dict] = []
    indice = 0
    t0 = time.perf_counter()

    for n, (nome, tam) in enumerate(arquivos, 1):
        marca = destino / f"part-{n - 1:05d}.parquet"
        if marca.exists():
            log.info("%d/%d já processado — pulando", n, len(arquivos))
            f.arquivos_lidos += 1
            indice = n
            continue
        local = temp / Path(nome).name
        try:
            # ⚠️ Rede indisponível ESPERA; não consome o arquivo. Ver o mesmo
            # aviso em `slices/redpajama.py`: uma queda de DNS às 01h56 queimou 23
            # shards em segundos, porque `getaddrinfo` falha instantaneamente e
            # não há timeout para desacelerar.
            for tentativa in range(MAX_TENTATIVAS):
                try:
                    f.bytes_lidos += _baixar(sessao, ds, nome, local)
                    break
                except (requests.ConnectionError, requests.Timeout) as exc:
                    espera = min(30 * 2 ** tentativa, 600)
                    log.warning("%s: rede indisponível (%s) — aguardando %d s "
                                "(tentativa %d/%d)", nome, type(exc).__name__,
                                espera, tentativa + 1, MAX_TENTATIVAS)
                    time.sleep(espera)
            else:
                raise RuntimeError(f"rede indisponível após {MAX_TENTATIVAS} tentativas")
            df = pl.read_parquet(local)
            textos, urls = _texto_e_url(df)
            # Fatia para não passar 20 mil textos de uma vez ao TF-IDF.
            for i in range(0, len(textos), 20_000):
                bloco = textos[i:i + 20_000]
                for texto, url, s in zip(bloco, urls[i:i + 20_000],
                                         clf.scores(bloco)):
                    f.vistos += 1
                    if s < limiar or not texto:
                        continue
                    f.aceitos += 1
                    f.caracteres_aceitos += len(texto)
                    if isinstance(url, str) and url.startswith("http"):
                        f.dominios[urlparse(url).netloc.lower()] += 1
                    buffer.append({"texto": texto, "url": url if isinstance(url, str) else None,
                                   "score": s})
                    if len(f.amostra) < N_AMOSTRA and f.vistos % 97 == 0:
                        f.amostra.append({"score": round(s, 4), "url": url
                                          if isinstance(url, str) else None,
                                          "inicio": texto[:600]})
                    if len(buffer) >= FLUSH:
                        _gravar(buffer, destino, indice)
                        buffer.clear()
                        indice += 1
        except Exception as exc:
            f.falhas.append(f"{nome}: {type(exc).__name__}: {str(exc)[:80]}")
            log.warning("%s falhou: %s", nome, exc)
        finally:
            local.unlink(missing_ok=True)     # o bruto NÃO fica
        f.arquivos_lidos += 1
        dt = time.perf_counter() - t0
        log.info("%d/%d · %s vistos · %s aceitos (%.1f%%) · %.1f GB · %.1f MB/s",
                 n, len(arquivos), f"{f.vistos:,}", f"{f.aceitos:,}",
                 100 * f.taxa, f.bytes_lidos / 1e9, f.bytes_lidos / 1e6 / max(dt, 1))

    if buffer:
        _gravar(buffer, destino, indice)
    return f


def _gravar(buffer: list[dict], destino: Path, indice: int) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"part-{indice:05d}.parquet"
    pl.DataFrame(buffer, schema={"texto": pl.Utf8, "url": pl.Utf8, "score": pl.Float64}
                 ).write_parquet(caminho, compression="zstd")
    log.info("→ %s (%s registros, %.0f MB)", caminho.name, f"{len(buffer):,}",
             caminho.stat().st_size / 1e6)
