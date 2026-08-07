"""Coletor do OpenAlex pelo snapshot público — a rota que a cota não alcança.

O DOC-02 §3.1 registra que a **API** do OpenAlex passou a ser cotada em 2026:
1.000 requisições/dia grátis, e as 18.336 necessárias custam US$ 1,83 ou 18
dias de espera. O **snapshot continua livre e sem cota**, e é a rota correta.

Mesmo parser, transporte diferente: `parse_work`, `extract_arxiv_id` e `SCHEMA`
vêm de `openalex.py` sem alteração. Trocar API por snapshot não muda o que um
registro significa, e duplicar o parser criaria duas verdades sobre a chave de
junção — exatamente o defeito que custou caro em 2026-08-03.

## O que a medição de 2026-08-06 mudou em relação ao DOC-02

O documento descrevia `s3://openalex/data/works/…` em JSONL comprimido, ~330 GB.
O layout **mudou** e o número também:

| | DOC-02 (2026-08-03) | Medido (2026-08-06) |
|---|---|---|
| Caminho | `data/works/` | `data/parquet/works/` e `data/jsonl/` |
| Formato | JSONL `.gz` | **parquet** disponível |
| Tamanho | ~330 GB / 250 M obras | **725 GB / 510 M obras** |
| Partições | — | 2.446, mediana 184 MB |

O snapshot dobrou de tamanho, mas o parquet mais que compensa: sendo colunar,
lemos **só as treze colunas de que precisamos**, por faixa de bytes HTTP. O
`abstract_inverted_index` sozinho é 43% dos bytes e é justamente o que não
serve — o resumo vem da espinha do arXiv, que é autoritativa.

Medido sobre partições reais: **21% dos bytes**, ou seja **155 GB** em vez de
725 GB. O bucket devolve `Accept-Ranges: bytes`, então isso é transferência
economizada, não apenas leitura descartada.

## Tempo: o DOC-02 dizia ~2 h, e são ~5,6 h

O documento estimou 2 h dividindo 330 GB por 444 Mbps. A conta ignorava que o
snapshot dobrou e que nada disso é uma transferência sequencial única. Medido,
com a otimização de cada etapa:

| Versão | Projeção | O que dominava |
|---|---|---|
| Ingênua | 40,1 h | `to_pylist()` convertendo 400 mil linhas por partição |
| Filtro vetorizado (`_mascara_arxiv`) | 17,4 h | faixas HTTP em série |
| `prebuscar` com 6 simultâneas | 8,5 h | latência por faixa |
| **`prebuscar` com 16** | **5,6 h** | banda |

**Isso reabre uma decisão que o `ESTADO.md` tratava como fechada.** A 5,6 h e
US$ 0, o snapshot empata em tempo com a rota paga (~5 h, US$ 1,83) e ganha no
custo, então segue sendo a escolha. Mas a 40 h — o que se teria medido sem as
otimizações — a recomendação teria se invertido, e por isso os números acima
ficam registrados: quem for reavaliar precisa saber que a margem é de tempo
comparável, não de ordem de grandeza.

**Nunca há um `.parquet` inteiro em disco.** O documento previa "baixa partição
→ filtra → apaga", com pico de ~10 GB. Ler por faixa dispensa até isso: o pico
de disco é o próprio shard de saída, e o de memória é um row group (~32 mil
obras).

## Filtro

Não filtramos por campo do OpenAlex, e o motivo está na docstring de
`openalex.py`: a obra `2203.00339` do arXiv é classificada pelo OpenAlex como
"Computer Science". O recorte de Física vem da categoria atribuída pelo autor
na espinha do arXiv. Aqui o critério é só **ter origem no arXiv**, resolvido
por `extract_arxiv_id`.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from phifm.core.schema.manifest import (
    AcquisitionManifest,
    HarvestMethod,
    LicenseResolution,
    RateLimit,
    canonical_hash,
)
from phifm.corpus.acquire.base import CONTACT, PoliteSession, ResumableHarvester
from phifm.corpus.acquire.openalex import SCHEMA, extract_arxiv_id, parse_work

log = logging.getLogger(__name__)

BUCKET = "https://openalex.s3.amazonaws.com/"
MANIFESTO = BUCKET + "data/parquet/works/manifest.json"

# As treze colunas que `parse_work` consome. `locations` é caro (6,8% dos
# bytes) e obrigatório: sem ele a chave de junção com o arXiv cai de 98,5%
# para 1,5% — ver §"chave de junção" em `openalex.py`.
COLUNAS = [
    "id", "doi", "title", "publication_year", "publication_date", "type",
    "ids", "referenced_works", "cited_by_count", "open_access",
    "primary_topic", "language", "locations",
]

FLUSH_REGISTROS = 50_000
# Teto de partições entre flushes. O cursor durável só avança em flush
# confirmado (DOC-08 §7.2), então este número É o custo de uma interrupção:
# 10 partições ≈ 2,6 GB de releitura. Sem o teto, as ~0,5% de aproveitamento
# fariam o buffer levar ~25 partições para encher — 6,6 GB perdidos por queda.
FLUSH_PARTICOES = 10


# Faixas separadas por menos que isto viram uma requisição só: pagar por
# alguns KB a mais é mais barato que pagar outra ida e volta.
_JUNTAR_ATE = 1 << 20          # 1 MB
# 16 medido contra 6 em partições medianas: 11,5 s vs 17,6 s, ou 5,6 h vs 8,5 h
# no total. Acima disso o ganho murcha — a essa altura o limite é banda, não
# latência — e 16 faixas simultâneas ainda é uso comum de S3.
_MAX_PARALELO = 16


def _faixas_das_colunas(md, colunas: set[str]) -> list[tuple[int, int]]:
    """Onde vivem, em bytes, as colunas que interessam.

    O rodapé do parquet declara deslocamento e tamanho de cada pedaço de
    coluna. Saber isso de antemão é o que permite buscar tudo em paralelo em
    vez de esperar o pyarrow pedir um pedaço por vez.

    Colunas aninhadas (`locations.landing_page_url`) são folhas separadas no
    schema; a raiz antes do primeiro ponto é o nome que o chamador conhece.
    """
    brutas: list[tuple[int, int]] = []
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for c in range(rg.num_columns):
            col = rg.column(c)
            if col.path_in_schema.split(".")[0] in colunas:
                brutas.append((col.file_offset, col.file_offset + col.total_compressed_size))

    brutas.sort()
    juntadas: list[tuple[int, int]] = []
    for ini, fim in brutas:
        if juntadas and ini - juntadas[-1][1] <= _JUNTAR_ATE:
            juntadas[-1] = (juntadas[-1][0], max(juntadas[-1][1], fim))
        else:
            juntadas.append((ini, fim))
    return juntadas


class ArquivoRemoto(io.RawIOBase):
    """Arquivo de acesso aleatório sobre HTTP, para o pyarrow ler o parquet.

    O pyarrow pede `seek`/`read`; o S3 oferece `Range`. Esta classe é a ponte,
    e é o que permite ler treze colunas de uma partição de 1,3 GB sem baixar a
    partição.

    **`prebuscar()` é o que torna a rota viável.** Sem ela as faixas saem uma
    por vez, cada uma pagando latência: medido em 48 s por partição, ou ~42
    Mbps num link de 444 Mbps — o problema nunca foi banda, foi serialização.
    Com as faixas conhecidas de antemão pelo rodapé, elas vão juntas.

    **Sobre cortesia (A5).** O limitador por requisição do `PoliteSession`
    existe para APIs com cota e operador humano do outro lado. Este bucket é
    armazenamento de objetos público, sem cota, e `_MAX_PARALELO = 6` faixas
    simultâneas é uso normal de S3, não avalanche. A retentativa com recuo
    continua valendo, e os bytes continuam contados no manifesto — o que se
    dispensa é a espera fixa entre faixas da MESMA partição.
    """

    def __init__(self, url: str, http: PoliteSession):
        self.url, self.http = url, http
        self.pos = 0
        self.bytes_lidos = 0
        self._cache: list[tuple[int, int, bytes]] = []
        r = http.session.head(url, timeout=30)
        r.raise_for_status()
        self._tamanho = int(r.headers["Content-Length"])
        if r.headers.get("Accept-Ranges") != "bytes":
            raise RuntimeError(
                f"{url} não aceita Range — leitura por coluna impossível, "
                "o coletor baixaria 725 GB em vez de 145 GB"
            )

    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def tell(self) -> int: return self.pos
    def size(self) -> int: return self._tamanho

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self._tamanho + offset
        return self.pos

    # ── busca ─────────────────────────────────────────────────────────────
    def _buscar(self, ini: int, fim: int) -> bytes:
        """Uma faixa, com o recuo do `PoliteSession` mas sem a espera fixa."""
        limite = self.http.throttle.limit
        ultima: Exception | None = None
        for tentativa in range(limite.max_retries):
            try:
                r = self.http.session.get(
                    self.url, timeout=180, headers={"Range": f"bytes={ini}-{fim - 1}"}
                )
                if r.status_code in (429, 503):
                    import time
                    time.sleep(self.http.throttle.backoff(tentativa))
                    continue
                r.raise_for_status()
                # Contadores do manifesto: aproximados o suficiente sob
                # concorrência, e o que importa deles é a ordem de grandeza.
                self.http.requests_made += 1
                self.http.bytes_downloaded += len(r.content)
                self.bytes_lidos += len(r.content)
                return r.content
            except Exception as exc:
                ultima = exc
                import time
                time.sleep(self.http.throttle.backoff(tentativa))
        raise RuntimeError(f"faixa {ini}-{fim} de {self.url} falhou") from ultima

    def prebuscar(self, faixas: list[tuple[int, int]]) -> None:
        """Busca as faixas em paralelo e guarda em memória."""
        if not faixas:
            return
        with ThreadPoolExecutor(max_workers=_MAX_PARALELO, thread_name_prefix="faixa") as pool:
            dados = list(pool.map(lambda f: self._buscar(*f), faixas))
        self._cache = [(ini, fim, d) for (ini, fim), d in zip(faixas, dados)]

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self._tamanho - self.pos
        if n <= 0 or self.pos >= self._tamanho:
            return b""
        fim = min(self.pos + n, self._tamanho)

        for ini_c, fim_c, dados in self._cache:
            if ini_c <= self.pos and fim <= fim_c:
                pedaco = dados[self.pos - ini_c: fim - ini_c]
                self.pos += len(pedaco)
                return pedaco

        # Fora do que foi prebuscado: rodapé, ou faixa que a coalescência não
        # cobriu. Vai pelo caminho cortês, que inclui a espera entre requisições.
        r = self.http.get(self.url, timeout=180,
                          headers={"Range": f"bytes={self.pos}-{fim - 1}"})
        self.bytes_lidos += len(r.content)
        self.pos += len(r.content)
        return r.content


def _s3_para_https(url: str) -> str:
    return url.replace("s3://openalex/", BUCKET) if url.startswith("s3://") else url


def _mascara_arxiv(tabela: pa.Table) -> pa.Array:
    """Pré-filtro vetorizado: quais linhas PODEM ter origem no arXiv.

    **Frouxo por contrato.** Pode sobrar, nunca faltar — `extract_arxiv_id`
    decide depois, e é ele a autoridade sobre normalização de versão e sobre o
    formato antigo de identificador. Um pré-filtro que apertasse demais
    descartaria obras válidas sem deixar rastro, que é o pior modo de falha
    possível aqui: a contagem final continuaria parecendo plausível.

    Os dois caminhos são os mesmos de `extract_arxiv_id`, e `ignore_case=True`
    acompanha o `re.I` de lá — sem isso um `arXiv.org/abs/` com maiúscula
    escaparia.
    """
    doi = pc.match_substring_regex(
        pc.fill_null(tabela["doi"], ""), r"10\.48550/arxiv\.", ignore_case=True
    )
    mascara = np.asarray(doi.to_pylist(), dtype=bool)

    # `locations` é list<struct>: achatar dá as URLs, e `list_parent_indices`
    # diz de qual linha cada uma veio. É como se volta de "esta URL casa" para
    # "esta OBRA casa" sem laço em Python.
    locs = tabela["locations"].combine_chunks()
    urls = pc.fill_null(pc.list_flatten(locs).field("landing_page_url"), "")
    bate = pc.match_substring(urls, "arxiv.org/abs/", ignore_case=True)
    linhas = pc.filter(pc.list_parent_indices(locs), bate).to_numpy()
    if len(linhas):
        mascara[linhas] = True
    return pa.array(mascara)


class OpenAlexSnapshotHarvester(ResumableHarvester):
    """Percorre as partições do snapshot, retomável por partição."""

    @staticmethod
    def make_manifest() -> AcquisitionManifest:
        return AcquisitionManifest(
            source_name="openalex_snapshot",
            harvest_method=HarvestMethod.BULK_S3,
            endpoint=MANIFESTO,
            query_spec={"formato": "parquet", "colunas": COLUNAS,
                        "filtro": "origem no arXiv (extract_arxiv_id)"},
            # O bucket é público e sem cota, mas cortesia não é opcional
            # (A5). 4 req/s é folgado para faixas de ~2 MB e não parece
            # abuso a quem opera o bucket.
            rate_limit=RateLimit(requests_per_second=4.0, max_retries=8,
                                 backoff_base_s=2.0, backoff_max_s=300),
            license_resolution=LicenseResolution(
                method="source_policy",
                evidence_url="https://docs.openalex.org/additional-help/faq",
                default_spdx="CC0-1.0",
                notes="Todos os dados do OpenAlex são CC0. Redistribuição permitida.",
            ),
        )

    # ── manifesto do snapshot ─────────────────────────────────────────────
    def particoes(self) -> tuple[str, list[dict]]:
        """Devolve (data do snapshot, lista de partições)."""
        m = self.http.get(MANIFESTO, timeout=120).json()
        return m.get("date", "desconhecida"), m["files"]

    def harvest(self, max_particoes: int | None = None) -> AcquisitionManifest:
        m = self.manifest
        if m.completed_at:
            return m

        data_snapshot, arquivos = self.particoes()
        if m.expected_count is None:
            m.expected_count = sum(a["meta"]["record_count"] for a in arquivos)
            m.query_spec["snapshot_date"] = data_snapshot
            m.query_spec["particoes_totais"] = len(arquivos)
            log.info("snapshot %s · %d partições · %s obras",
                     data_snapshot, len(arquivos), f"{m.expected_count:,}")

        # ── retomada ──────────────────────────────────────────────────────
        # O cursor é a URL da última partição com flush confirmado. Guardar
        # índice seria mais simples e ERRADO: o snapshot é republicado todo
        # mês e os índices deslizam, então retomar por índice pularia ou
        # repetiria partições em silêncio.
        inicio = 0
        anterior = m.query_spec.get("snapshot_date")
        if m.resumable_cursor:
            if anterior and anterior != data_snapshot:
                raise RuntimeError(
                    f"o snapshot mudou de {anterior} para {data_snapshot} no meio da "
                    "coleta. Retomar misturaria duas versões do grafo — apagar o "
                    "manifesto e recomeçar, ou fixar a versão anterior."
                )
            urls = [a["url"] for a in arquivos]
            if m.resumable_cursor not in urls:
                raise RuntimeError(
                    f"cursor {m.resumable_cursor} não está no manifesto atual — "
                    "não é possível retomar sem risco de pular partições."
                )
            inicio = urls.index(m.resumable_cursor) + 1
            log.info("retomando da partição %d de %d", inicio, len(arquivos))

        buffer: list[dict] = []
        shard = len(list(self.out_dir.glob("part-*.parquet")))
        pendentes = 0            # registros no buffer, ainda não duráveis
        desde_flush = 0          # partições desde o último flush
        cursor_pendente = m.resumable_cursor
        vistas = lidos_bytes = 0

        for i, arq in enumerate(arquivos[inicio:], start=inicio):
            if max_particoes is not None and (i - inicio) >= max_particoes:
                log.info("limite de %d partições atingido — parando (retomável)", max_particoes)
                break

            url = _s3_para_https(arq["url"])
            try:
                achados, obras, bytes_part = self._ler_particao(url, buffer)
            except Exception as exc:
                # A5: falha registrada, nunca engolida. Uma partição ruim não
                # derruba a coleta inteira — as outras 2.445 continuam.
                self.record_failure("F-SNAPSHOT-PARTICAO", f"{url}: {exc}")
                log.warning("partição %d falhou (%s) — registrada e seguindo", i, exc)
                continue

            vistas += obras
            lidos_bytes += bytes_part
            pendentes += achados
            desde_flush += 1
            cursor_pendente = arq["url"]

            # ── ordem crítica (DOC-08 §7.2) ───────────────────────────────
            # O cursor NÃO avança antes de os dados estarem duráveis. Invertido,
            # uma queda entre o avanço e a escrita perderia registros em
            # silêncio; nesta ordem ela os duplica, e a dedup exata resolve.
            if len(buffer) >= FLUSH_REGISTROS or desde_flush >= FLUSH_PARTICOES:
                self._flush(buffer, shard)
                shard += 1
                buffer = []
                m.resumable_cursor = cursor_pendente
                m.actual_count += pendentes
                pendentes = 0
                desde_flush = 0
                self.checkpoint()
                pct = 100 * (i + 1) / len(arquivos)
                log.info("partição %d/%d (%.1f%%) · %s obras arXiv · %.1f GB lidos",
                         i + 1, len(arquivos), pct, f"{m.actual_count:,}", lidos_bytes / 1e9)

        if buffer:
            self._flush(buffer, shard)
        m.resumable_cursor = cursor_pendente
        m.actual_count += pendentes

        completou = (max_particoes is None
                     and cursor_pendente == arquivos[-1]["url"]
                     and not any(f.retryable for f in m.failures))
        if completou:
            m.resumable_cursor = None
            m.mark_complete()
            log.info("concluído: %s obras arXiv de %s vistas",
                     f"{m.actual_count:,}", f"{vistas:,}")
        self.checkpoint()
        return m

    def _ler_particao(self, url: str, buffer: list[dict]) -> tuple[int, int, int]:
        """Lê uma partição e absorve as obras com origem no arXiv.

        Devolve (achados, obras vistas, bytes lidos).

        **Duas otimizações medidas em 2026-08-06, e a segunda é a que decide se
        esta rota é viável.**

        *Máscara antes de converter.* A versão ingênua fazia `to_pylist()` na
        partição inteira e descartava 99,4% — 61,3 s dos 113,2 s de cada
        partição, ou 22 h das 40 h projetadas, gastos criando objetos Python
        para jogar fora. `_mascara_arxiv` decide em C++ vetorizado (0,3 s) e só
        as ~2.500 sobreviventes viram Python (0,6 s).

        *Leitura da partição inteira, não row group por row group.* Parece
        gastar memória e não gasta o que importa: são ~250 MB de **Arrow**,
        colunar e compacto, não de objetos Python. Em troca, o pyarrow coalesce
        as faixas adjacentes e o número de requisições cai de 130 para ~105 —
        e, com `_MAX_PARALELO`, elas saem simultâneas.
        """
        remoto = ArquivoRemoto(url, self.http)
        pf = pq.ParquetFile(pa.PythonFile(remoto, mode="r"))
        # O rodapé já está lido; agora sabemos onde estão as colunas e podemos
        # buscá-las todas de uma vez em vez de uma por vez.
        remoto.prebuscar(_faixas_das_colunas(pf.metadata, set(COLUNAS)))
        tabela = pf.read(columns=COLUNAS)
        obras = tabela.num_rows

        candidatas = tabela.filter(_mascara_arxiv(tabela)).to_pylist()

        # O pré-filtro é deliberadamente FROUXO: ele pode sobrar, nunca faltar.
        # `extract_arxiv_id` continua sendo a autoridade — é ele que normaliza
        # o sufixo de versão e distingue o formato antigo `hep-th/9711200` do
        # novo. Medido: pré-filtro 2.502, exato 2.502.
        achados = 0
        for w in candidatas:
            if extract_arxiv_id(w):
                buffer.append(parse_work(w))
                achados += 1
        return achados, obras, remoto.bytes_lidos

    def _flush(self, buffer: list[dict], idx: int) -> None:
        if not buffer:
            return
        caminho = self.out_dir / f"part-{idx:05d}.parquet"
        df = pl.DataFrame(buffer, schema=SCHEMA)
        tmp = caminho.with_suffix(".parquet.tmp")
        df.write_parquet(tmp, compression="zstd")
        tmp.replace(caminho)  # escrita atômica: nunca existe shard pela metade
        self.manifest.checksum_index[caminho.name] = canonical_hash(
            {"rows": len(df), "cols": sorted(df.columns)}
        )
        arestas = int(df["n_references"].sum())
        log.info("→ %s (%d obras, %d arestas de citação, %.1f MB)",
                 caminho.name, len(df), arestas, caminho.stat().st_size / 1e6)


def harvest_snapshot(
    out_dir: Path,
    max_particoes: int | None = None,
    contact: str = CONTACT,
) -> AcquisitionManifest:
    h = OpenAlexSnapshotHarvester.resume_or_create(
        out_dir, OpenAlexSnapshotHarvester.make_manifest, contact=contact
    )
    return h.harvest(max_particoes=max_particoes)
