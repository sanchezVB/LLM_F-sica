#!/usr/bin/env bash
# Lança a coleta do arXiv totalmente destacada do terminal.
#
# Duas proteções aprendidas em execução (2026-08-03):
#   - `caffeinate -i` impede o macOS de suspender o processo. Sem isso, 60%
#     do tempo de relógio virou pausa (medido: 1,63 h perdidas em 2,72 h).
#   - subshell + nohup + disown desacopla do grupo de processos do pai, para
#     que o encerramento do terminal (ou do agente) não leve a coleta junto.
#
# Idempotente: rodar de novo com a coleta em andamento não duplica nada, e
# rodar após interrupção retoma do último checkpoint durável.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-data/raw/arxiv_metadata}"
SET="${2:-physics}"
LOG="data/raw/harvest_arxiv.log"

if pgrep -f "harvest_arxiv.py" >/dev/null; then
  echo "já em execução (PID $(pgrep -f harvest_arxiv.py | tr '\n' ' '))"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
(
  PYTHONPATH=src nohup caffeinate -i .venv/bin/python \
    scripts/harvest_arxiv.py --out "$OUT" --set "$SET" >>"$LOG" 2>&1 &
  disown
)
sleep 3
if pgrep -f "harvest_arxiv.py" >/dev/null; then
  echo "iniciado (PID $(pgrep -f harvest_arxiv.py | head -1)) · log: $LOG"
else
  echo "FALHOU ao iniciar — ver $LOG" >&2
  exit 1
fi
