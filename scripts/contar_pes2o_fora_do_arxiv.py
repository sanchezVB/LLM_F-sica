#!/usr/bin/env python3
"""Passo 1 do acervo maior: quantos resumos de Física do peS2o NÃO estão no arXiv?

    P = .venv-treino\\Scripts\\python.exe scripts\\contar_pes2o_fora_do_arxiv.py

Só CPU, sem o modelo. Lê os 1,59 M artigos do índice (título e resumo vêm do spine) e os
resumos `s2ag` de `data/processed/pes2o_fisica/`, casa pelo título ou pelo começo do resumo
(`phifm.corpus.pes2o_novos`) e conta. Estimativa: 15 a 30 min, quase tudo leitura do HD.

Grava:

- `data/processed/avaliacao/pes2o_fora_do_arxiv.json` (vai ao git): a conta;
- `data/processed/pes2o_novos/novos.parquet` (fora do git): parte, linha, sha256 do texto
  e título de cada novo — é o que o passo 2 usa para recuperar o id do Semantic Scholar;
- `data/processed/pes2o_novos/exemplos.json` (fora do git): 50 casados e 50 novos, para
  conferir à mão se a conta merece confiança antes de baixar qualquer coisa.

⚠️ Segura a trava de rodada pesada de `medir_passagens.py`: lê ~5 GB do HD e ~2 GB de
memória, e não roda junto de uma medida — de outra sessão, por exemplo.
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

from phifm.corpus import pes2o_novos as pn  # noqa: E402

AGREGADO = RAIZ / "data/processed/avaliacao/pes2o_fora_do_arxiv.json"
SAIDA = RAIZ / "data/processed/pes2o_novos"
LOTE = 100_000


def carregar_arxiv(spine: Path, indice: Path) -> pn.Arxiv:
    """Os artigos do ÍNDICE (o arXiv de Física que a busca vê), com título e resumo do
    spine. Casar contra o spine inteiro contaria como "já temos" um artigo que está no
    arXiv mas fora do índice."""
    ids = pl.read_parquet(indice / "documentos.parquet", columns=["arxiv_id"])["arxiv_id"]
    a = pn.Arxiv()
    df = (pl.scan_parquet(spine).select("arxiv_id", "title", "abstract")
          .filter(pl.col("arxiv_id").is_in(ids.implode())).collect())
    for bloco in df.iter_slices(LOTE):
        for arxiv_id, titulo, resumo in bloco.iter_rows():
            a.acrescentar(arxiv_id, titulo, resumo)
    return a


def contar(pes2o: Path, spine: Path, indice: Path, saida: Path, agregado: Path) -> dict:
    """A conta inteira, sem trava (o `main` a segura)."""
    t0 = time.perf_counter()
    partes = sorted(pes2o.glob("part-*.parquet"))
    if not partes:
        raise SystemExit(f"nenhum part-*.parquet em {pes2o}.")
    print(f"lendo o spine ({spine.name})...", flush=True)
    arxiv = carregar_arxiv(spine, indice)
    print(f"  {arxiv.artigos:,} artigos · {len(arxiv.por_titulo):,} chaves de título · "
          f"{len(arxiv.por_resumo):,} de resumo ({time.perf_counter() - t0:.0f} s)", flush=True)
    c = pn.Contagem(arxiv)
    blocos, so_texto_pleno = [], 0
    for k, arq in enumerate(partes, 1):
        url = pl.read_parquet(arq, columns=["url"])["url"]
        if not url.str.starts_with("s2ag").any():
            so_texto_pleno += 1
            continue
        df = (pl.read_parquet(arq, columns=["texto", "url"]).with_row_index("linha")
              .filter(pl.col("url").str.starts_with("s2ag")))
        for linha, texto in zip(df["linha"], df["texto"], strict=True):
            c.ver(texto, arq.name, int(linha))
        novos = c.tirar_novos()
        if novos:
            blocos.append(pl.DataFrame(novos))
        if k % 20 == 0:
            print(f"[{k}/{len(partes)}] {c.n['registros']:,} resumos · {c.n['novos']:,} novos "
                  f"· {time.perf_counter() - t0:.0f} s", flush=True)
    saida.mkdir(parents=True, exist_ok=True)
    novos = pl.concat(blocos) if blocos else pl.DataFrame(
        schema={"parte": pl.Utf8, "linha": pl.Int64, "sha256": pl.Utf8, "titulo": pl.Utf8,
                "caracteres": pl.Int64})
    novos.write_parquet(saida / "novos.parquet")
    (saida / "exemplos.json").write_text(json.dumps(
        {"casados": c.exemplos_casados.itens, "novos": c.exemplos_novos.itens},
        ensure_ascii=False, indent=1), encoding="utf-8")
    resultado = {
        "o_que_e": "quantos resumos de Física do peS2o (s2ag) já estão no arXiv, pelo título "
                   "ou pelo começo do resumo — e quantos seriam acervo novo para a busca",
        "criterio": {"min_palavras_titulo": pn.MIN_PALAVRAS_TITULO,
                     "palavras_do_resumo": pn.PALAVRAS_DO_RESUMO,
                     "max_caracteres_titulo": pn.MAX_CARACTERES_TITULO,
                     "min_caracteres_resumo": pn.MIN_CARACTERES_RESUMO},
        "indice (o arXiv de Física da busca)": {"artigos": arxiv.artigos, "chaves_de_titulo": len(arxiv.por_titulo),
                  "chaves_de_resumo": len(arxiv.por_resumo)},
        "pes2o": {"partes": len(partes), "partes_so_com_texto_pleno": so_texto_pleno},
        "s2ag": c.resumo(),
        "limite": "conta superestima os novos quando a revista troca título E resumo; "
                  "os exemplos fora do git servem para conferir à mão",
        "segundos": round(time.perf_counter() - t0), "gravado_em": time.strftime("%Y-%m-%d %H:%M")}
    agregado.parent.mkdir(parents=True, exist_ok=True)
    agregado.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pes2o", type=Path, default=RAIZ / "data/processed/pes2o_fisica")
    p.add_argument("--spine", type=Path, default=RAIZ / "data/processed/spine.parquet")
    p.add_argument("--indice", type=Path, default=RAIZ / "data/processed/indice_busca")
    a = p.parse_args()
    import medir_passagens as mp

    with mp.trava("medir_passagens"):
        r = contar(a.pes2o, a.spine, a.indice, SAIDA, AGREGADO)
    print(json.dumps(r["s2ag"], ensure_ascii=False, indent=1))
    if r["s2ag"]["aviso"]:
        print(f"⚠️ {r['s2ag']['aviso']}")
    print(f"→ {AGREGADO.relative_to(RAIZ)} · novos e exemplos em {SAIDA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
