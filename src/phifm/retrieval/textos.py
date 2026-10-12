"""O texto completo de cada artigo, com acesso por id — para as passagens sob demanda.

O RedPajama filtrado (`data/processed/redpajama_fisica/`) guarda 835 mil artigos em 44
arquivos, e cada arquivo é UM grupo de 20.000 linhas: ler um artigo é descomprimir 1 GB.
Medido em 2026-10-12, com o arquivo já na memória do sistema: 3,9 s por artigo. O
assistente lê até 6 por pergunta.

`construir` regrava a mesma tabela em grupos de `POR_GRUPO` linhas e grava um índice de
onde cada artigo está; `Textos` lê um artigo abrindo só o grupo dele (~2 MB).

⚠️ É uma CÓPIA derivada, do tamanho da origem (~12 GB, no HD), fora do git. Não substitui
a origem, que é a saída versionada da etapa de coleta; pode ser apagada e refeita.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

POR_GRUPO = 32
NOME_INDICE = "indice.parquet"
NOME_MANIFESTO = "manifesto.json"


def construir(origem: Path, destino: Path, por_grupo: int = POR_GRUPO, ao_gravar=None) -> dict:
    """Regrava `origem/part-*.parquet` em `destino`, em grupos pequenos, e grava o índice.
    Retomável por arquivo. Um artigo que aparece em dois arquivos fica com o primeiro —
    a mesma regra de quem lê a origem (`piloto_trechos`, `perguntas_corpo`)."""
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    destino.mkdir(parents=True, exist_ok=True)
    partes = sorted(origem.glob("part-*.parquet"))
    if not partes:
        raise SystemExit(f"{origem} não tem part-*.parquet")
    vistos: set[str] = set()
    linhas: list[tuple[str, int, int]] = []          # (arxiv_id, parte, linha)
    for n, p in enumerate(partes):
        alvo = destino / p.name
        if alvo.exists():
            ids = pq.read_table(alvo, columns=["arxiv_id"]).column(0).to_pylist()
        else:
            t = pq.read_table(p, columns=["arxiv_id", "texto"])
            ids_origem = t.column(0).to_pylist()
            fica, daqui = [], set()
            for a in ids_origem:
                fica.append(a not in vistos and a not in daqui)
                daqui.add(a)
            t = t.filter(pa.array(fica))
            ids = t.column(0).to_pylist()
            tmp = alvo.with_suffix(".tmp")
            pq.write_table(t, tmp, row_group_size=por_grupo, compression="zstd")
            tmp.rename(alvo)
            del t
        linhas += [(a, n, i) for i, a in enumerate(ids)]
        vistos.update(ids)
        if ao_gravar:
            ao_gravar(p.name, len(ids), len(vistos))
    pl.DataFrame(linhas, orient="row", schema={
        "arxiv_id": pl.String, "parte": pl.Int32, "linha": pl.Int32}).write_parquet(
            destino / NOME_INDICE)
    manifesto = {"artigos": len(linhas), "partes": [p.name for p in partes],
                 "por_grupo": por_grupo, "origem": str(origem),
                 "o_que_e": "cópia derivada da origem, em grupos pequenos, para ler um artigo "
                            "por vez; pode ser apagada e refeita",
                 "criado_em": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (destino / NOME_MANIFESTO).write_text(json.dumps(manifesto, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return manifesto


class Textos:
    """`ler(arxiv_id)` devolve o texto completo, ou None se o artigo não tem."""

    def __init__(self, pasta: Path):
        import polars as pl

        man = pasta / NOME_MANIFESTO
        if not man.exists():
            raise FileNotFoundError(
                f"{pasta} não tem {NOME_MANIFESTO}: rode scripts/indexar_textos.py (ele retoma).")
        self.manifesto = json.loads(man.read_text(encoding="utf-8"))
        self.pasta, self.por_grupo = pasta, int(self.manifesto["por_grupo"])
        self._partes = self.manifesto["partes"]
        idx = pl.read_parquet(pasta / NOME_INDICE)
        # (parte, linha) num inteiro só: 835 mil tuplas custariam o dobro da memória.
        self._onde = dict(zip(idx["arxiv_id"].to_list(),
                              (idx["parte"].cast(pl.Int64) * 1_000_000
                               + idx["linha"].cast(pl.Int64)).to_list(), strict=True))
        self._arquivos: dict[int, object] = {}

    def __contains__(self, arxiv_id: str) -> bool:
        return arxiv_id in self._onde

    def __len__(self) -> int:
        return len(self._onde)

    def ler(self, arxiv_id: str) -> str | None:
        import pyarrow.parquet as pq

        onde = self._onde.get(arxiv_id)
        if onde is None:
            return None
        parte, linha = divmod(onde, 1_000_000)
        if parte not in self._arquivos:
            self._arquivos[parte] = pq.ParquetFile(self.pasta / self._partes[parte])
        grupo = self._arquivos[parte].read_row_group(linha // self.por_grupo,
                                                     columns=["arxiv_id", "texto"])
        i = linha % self.por_grupo
        if grupo.column(0)[i].as_py() != arxiv_id:
            raise RuntimeError(f"o índice de textos aponta {arxiv_id} para a linha errada: "
                               f"{self.pasta} foi alterado depois de indexado.")
        return grupo.column(1)[i].as_py()
