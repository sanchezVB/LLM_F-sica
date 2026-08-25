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

import gzip
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
from phifm.corpus.slices.retomada import (
    assinatura_da_lista,
    feitas,
    marcar,
    proximo_indice,
)

log = logging.getLogger(__name__)

API = "https://huggingface.co/api/datasets/{}"
# ⚠️ `{rev}`, não `main`. Baixar de `main` é baixar de um alvo MÓVEL: o dataset
# pode ser revisado e a mesma execução do nosso código passa a produzir outro
# corpus, sem erro e sem aviso.
#
# Descoberto ao construir o manifesto raiz do G1.5 (DOC-00 §5): o critério pede o
# corpus refazível a partir de um hash, e uma fonte móvel torna isso impossível
# por construção. Tivemos sorte — o OpenWebMath está na revisão
# `fde8ef8de2300f5e778f56261843dab89f230815` desde 2023-10-17, anterior à nossa
# coleta, então `main` não mudou debaixo de nós. Sorte não é reprodutibilidade.
RESOLVE = "https://huggingface.co/datasets/{}/resolve/{}/{}"

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
    # Revisão (commit sha) da fonte no HuggingFace. Ver o comentário em `RESOLVE`:
    # sem isto o manifesto do G1.5 nomeia a fonte sem identificar o conteúdo.
    revisao: str = ""
    # Quantas unidades a FONTE publica. Sem isto, "acabou?" só era respondível
    # comparando contagens de naturezas diferentes — e a comparação errada dizia
    # "em curso" para uma coleta concluída. Ver o comentário em `filtrar_hf.py`.
    total_unidades: int = 0
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


def _arquivos(sessao: requests.Session, ds: str,
              ext: tuple[str, ...]) -> tuple[list[tuple[str, int]], str]:
    """(arquivos, revisão). A revisão é o que torna a fatia refazível.

    Devolve o `sha` do commit do dataset junto com a lista. Sem ele, a coleta e o
    manifesto descrevem "o dataset X", que é um nome, não um conteúdo.
    """
    r = sessao.get(API.format(ds) + "?blobs=true", timeout=120)
    r.raise_for_status()
    d = r.json()
    fs = [(f["rfilename"], f.get("size") or 0)
          for f in d.get("siblings", [])
          if f["rfilename"].endswith(ext)]
    rev = d.get("sha") or "main"
    return sorted(fs), rev


def _baixar(sessao: requests.Session, ds: str, nome: str, destino: Path,
            rev: str = "main") -> int:
    destino.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with sessao.get(RESOLVE.format(ds, rev, nome), timeout=1800, stream=True) as r:
        r.raise_for_status()
        with open(destino, "wb") as f:
            for pedaco in r.iter_content(1 << 22):
                f.write(pedaco)
                n += len(pedaco)
    return n


def _ler_blocos(local: Path, por_bloco: int = 50_000):
    """Itera o arquivo em DataFrames, despachando pelo formato.

    ⚠️ Era `pl.read_parquet(local)` fixo, e o peS2o v1 é `.json.gz`. Medido em
    2026-08-25 num teste de fumaça de UM arquivo:

        data/v1/train-00000-of-00020.json.gz falhou: parquet: File must end with PAR1
        1/1 · 0 vistos · 0 aceitos (0,0%) · 2,9 GB

    Baixou 2,9 GB e processou ZERO registros. E como a exceção é capturada e só
    logada como aviso, os 20 arquivos falhariam igual: as 42,7 h estimadas
    produziriam um diretório vazio com um resumo satisfeito.

    ## Por que em blocos, e não um DataFrame só

    Os arquivos do peS2o têm ~2,9 GB COMPRIMIDOS. Descomprimir inteiro para
    materializar um DataFrame passaria de 10 GB de RAM — e este projeto já perdeu
    treinos para um relatório que alocou 43 GB. O `gzip` lê em fluxo, e cada bloco
    de linhas vira um DataFrame pequeno que o chamador consome e descarta.
    """
    nome = local.name.lower()
    if nome.endswith(".parquet"):
        yield pl.read_parquet(local)
        return
    if nome.endswith((".json.gz", ".jsonl.gz", ".jsonl", ".json")):
        abrir = gzip.open if nome.endswith(".gz") else open
        with abrir(local, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
            lote: list[str] = []
            for linha in fh:
                if not linha.strip():
                    continue
                lote.append(linha)
                if len(lote) >= por_bloco:
                    yield pl.read_ndjson("".join(lote).encode("utf-8"))
                    lote = []
            if lote:
                yield pl.read_ndjson("".join(lote).encode("utf-8"))
        return
    raise ValueError(
        f"formato não reconhecido: {local.name}. Aceito: .parquet, .json.gz, "
        ".jsonl.gz, .jsonl, .json")


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
    prefixo: str | None = None,
    temp: Path | None = None,
    contato: str | None = None,
) -> Filtragem:
    """Baixa, filtra e descarta arquivo por arquivo. Retomável pelo que há em disco."""
    sessao = requests.Session()
    sessao.headers.update({"User-Agent": user_agent(contato or "phifm")})
    clf = Classificador(modelo)
    temp = temp or (destino.parent / "_temp")
    arquivos, revisao = _arquivos(sessao, ds, ext)
    log.info("%s · revisão fixada em %s", ds, revisao)

    # ⚠️ O peS2o publica v1 E v2 da MESMA coleção no mesmo repositório: 22
    # arquivos e 100,7 GB de v1, 22 arquivos e 87,1 GB de v2. Sem selecionar
    # uma, o filtro ingere os mesmos papers duas vezes — ~31 h de máquina para
    # produzir um corpus com metade duplicada.
    #
    # Duplicação em dado de pré-treino não é desperdício, é dano: o S3b deste
    # projeto mediu que o RedPajama **degrada 16,6%**.
    if prefixo:
        antes = len(arquivos)
        arquivos = [(n, s) for n, s in arquivos if n.startswith(prefixo)]
        log.info("prefixo %r: %d de %d arquivos", prefixo, len(arquivos), antes)
        if not arquivos:
            raise ValueError(
                f"nenhum arquivo começa com {prefixo!r}. Presentes: "
                f"{sorted({n.rsplit('/', 1)[0] for n, _ in _arquivos(sessao, ds, ext)[0]})}")

    # A guarda vale MESMO sem prefixo: descobrir a duplicação depois de 31 h de
    # download seria descobrir tarde.
    versoes = {n.split("/")[1] for n, _ in arquivos if n.count("/") >= 2}
    if len(versoes) > 1:
        raise ValueError(
            f"a lista abrange {len(versoes)} versões do mesmo corpus: "
            f"{sorted(versoes)}. Ingerir as duas duplicaria os documentos. "
            f"Passe `prefixo=` para escolher uma (ex.: 'data/v2/').")

    if max_arquivos:
        arquivos = arquivos[:max_arquivos]
    tot = sum(s for _, s in arquivos)
    log.info("%s · %d arquivos · %.1f GB", ds, len(arquivos), tot / 1e9)

    f = Filtragem(fonte=ds, total_unidades=len(arquivos), revisao=revisao)
    # ⚠️ Retomada por MANIFESTO. A versão anterior testava se
    # `part-{n-1}.parquet` existia, tratando número de ARQUIVO DE ENTRADA como
    # índice de SAÍDA — e no OpenWebMath um parquet cobre ~2,7 arquivos, então os
    # dois divergem no primeiro flush. Ver `slices/retomada.py`: o mesmo defeito
    # custou 40.000 duplicatas no RedPajama.
    assinatura = assinatura_da_lista([n for n, _ in arquivos])
    prontos = feitas(destino, assinatura=assinatura)
    buffer: list[dict] = []
    indice = proximo_indice(destino)
    if prontos:
        log.info("retomando: %d arquivos já feitos, próximo parquet é part-%05d",
                 len(prontos), indice)
    t0 = time.perf_counter()

    for n, (nome, _tam) in enumerate(arquivos, 1):
        if n in prontos:
            f.arquivos_lidos += 1
            continue
        local = temp / Path(nome).name
        try:
            # ⚠️ Rede indisponível ESPERA; não consome o arquivo. Ver o mesmo
            # aviso em `slices/redpajama.py`: uma queda de DNS às 01h56 queimou 23
            # shards em segundos, porque `getaddrinfo` falha instantaneamente e
            # não há timeout para desacelerar.
            for tentativa in range(MAX_TENTATIVAS):
                try:
                    f.bytes_lidos += _baixar(sessao, ds, nome, local, revisao)
                    break
                except (requests.ConnectionError, requests.Timeout) as exc:
                    espera = min(30 * 2 ** tentativa, 600)
                    log.warning("%s: rede indisponível (%s) — aguardando %d s "
                                "(tentativa %d/%d)", nome, type(exc).__name__,
                                espera, tentativa + 1, MAX_TENTATIVAS)
                    time.sleep(espera)
            else:
                raise RuntimeError(f"rede indisponível após {MAX_TENTATIVAS} tentativas")
            for df in _ler_blocos(local):
                textos, urls = _texto_e_url(df)
                # Fatia para não passar 20 mil textos de uma vez ao TF-IDF.
                for i in range(0, len(textos), 20_000):
                    bloco = textos[i:i + 20_000]
                    # `strict=True`: as tres listas saem do MESMO corte, e um
                    # desalinhamento parearia texto com a URL de outro documento —
                    # silenciosamente. E o defeito que o cache de vetores ja causou.
                    for texto, url, s in zip(bloco, urls[i:i + 20_000],
                                             clf.scores(bloco), strict=True):
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
        else:
            prontos.add(n)
            marcar(destino, prontos, assinatura=assinatura)
        finally:
            local.unlink(missing_ok=True)     # o bruto NÃO fica
        f.arquivos_lidos += 1
        dt = time.perf_counter() - t0
        log.info("%d/%d · %s vistos · %s aceitos (%.1f%%) · %.1f GB · %.1f MB/s",
                 n, len(arquivos), f"{f.vistos:,}", f"{f.aceitos:,}",
                 100 * f.taxa, f.bytes_lidos / 1e9, f.bytes_lidos / 1e6 / max(dt, 1))

    if buffer:
        _gravar(buffer, destino, indice)

    # ⚠️ Zero registros VISTOS nao e um resultado, e uma falha — e sem esta guarda
    # ela sai como sucesso. Medido em 2026-08-25: o leitor tentava parquet num
    # `.json.gz`, cada arquivo levantava, a excecao era capturada e logada como
    # aviso, e o resumo final imprimia "0 vistos · 0 aceitos (0,00%)" com codigo de
    # saida 0. As 42,7 h estimadas para o peS2o teriam produzido um diretorio vazio.
    #
    # A taxa de aceitacao pode legitimamente ser zero (limiar alto, fonte sem
    # Fisica). O que nunca e legitimo e nao ter LIDO nada.
    if f.vistos == 0 and f.arquivos_lidos:
        raise RuntimeError(
            f"{f.arquivos_lidos} arquivos processados e NENHUM registro lido. "
            f"Falhas: {f.falhas[:3]}. Um resumo de zero vistos com sucesso "
            "esconderia horas de download sem resultado.")
    return f


def _gravar(buffer: list[dict], destino: Path, indice: int) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    caminho = destino / f"part-{indice:05d}.parquet"
    pl.DataFrame(buffer, schema={"texto": pl.Utf8, "url": pl.Utf8, "score": pl.Float64}
                 ).write_parquet(caminho, compression="zstd")
    log.info("→ %s (%s registros, %.0f MB)", caminho.name, f"{len(buffer):,}",
             caminho.stat().st_size / 1e6)
