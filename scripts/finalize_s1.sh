#!/usr/bin/env bash
# Aguarda as coletas do Sprint S1 terminarem e consolida tudo.
#
# Roda destacado, por horas. NÃO faz commit: gerar o commit exige julgamento
# sobre o que os números significam, e isso não se automatiza. Ao terminar,
# escreve data/S1_COMPLETO.json — o marcador que sinaliza que está pronto
# para revisão e publicação.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=data/finalize_s1.log
exec >>"$LOG" 2>&1
echo "=== iniciado em $(date '+%F %T') ==="

espera() {
  local nome="$1" proc="$2"
  while pgrep -f "$proc" >/dev/null; do sleep 60; done
  echo "$(date '+%T') $nome encerrou"
}

espera "arXiv"    "harvest_arxiv.py"
espera "OpenAlex" "harvest_openalex.py"

echo "$(date '+%T') consolidando espinha…"
PYTHONPATH=src .venv/bin/python scripts/build_spine.py

echo "$(date '+%T') retreinando classificador com o corpus completo…"
PYTHONPATH=src .venv/bin/python scripts/train_classifier.py || echo "classificador falhou (não bloqueante)"

echo "$(date '+%T') arquivando no Drive…"
PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
from phifm.core.io.storage import Storage
s = Storage.discover()
for src, kind, name in [
    (Path("data/raw/arxiv_metadata"), "raw", "arxiv_metadata"),
    (Path("data/raw/openalex_works"), "raw", "openalex_works"),
    (Path("data/processed"),          "processed", "processed"),
    (Path("models"),                  "checkpoints", "models"),
]:
    if src.exists():
        print("  →", s.archive(src, kind, name))
PY

echo "$(date '+%T') gerando marcador de conclusão…"
PYTHONPATH=src .venv/bin/python - <<'PY'
import json, pathlib, datetime as dt
import polars as pl

out = {"finished_at": dt.datetime.now().isoformat(timespec="seconds"), "sources": {}}
for d in ("arxiv_metadata", "openalex_works"):
    p = pathlib.Path(f"data/raw/{d}/_manifest.json")
    if p.exists():
        m = json.load(open(p))
        out["sources"][d] = {
            "records": m["actual_count"],
            "completed": bool(m.get("completed_at")),
            "failures": len(m.get("failures", [])),
            "gb_downloaded": round(m.get("bytes_downloaded", 0) / 1e9, 2),
        }

sp = pathlib.Path("data/processed/spine.parquet")
if sp.exists():
    df = pl.read_parquet(sp)
    phys = df.filter(pl.col("is_physics"))
    part = {r["partition"]: r["count"] for r in phys["partition"].value_counts().iter_rows(named=True)}
    out["spine"] = {
        "unique_records": df.height,
        "physics": phys.height,
        "peer_reviewed_pct": round(100 * phys["peer_reviewed"].mean(), 1),
        "partitions": part,
        "train_open_pct": round(100 * part.get("train_open", 0) / max(phys.height, 1), 1),
        "citation_edges": int(df["n_references"].fill_null(0).sum()),
    }

pathlib.Path("data/S1_COMPLETO.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(out, indent=2, ensure_ascii=False))
PY

echo "=== concluído em $(date '+%F %T') ==="
