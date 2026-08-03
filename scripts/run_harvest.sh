#!/usr/bin/env bash
# Lança um coletor do Sprint S1 totalmente destacado do terminal.
#
#   ./scripts/run_harvest.sh arxiv      # espinha de metadados
#   ./scripts/run_harvest.sh openalex   # grafo de citações
#
# Duas proteções aprendidas em execução (2026-08-03):
#   - `caffeinate -i` impede o macOS de suspender o processo. Sem isso, 60%
#     do tempo de relógio virou pausa (medido: 1,63 h perdidas em 2,72 h).
#   - subshell + nohup + disown desacopla do grupo de processos do pai, para
#     que encerrar o terminal (ou o agente) não leve a coleta junto.
#
# Idempotente: detecta execução em andamento e não duplica; após interrupção,
# retoma do último checkpoint durável.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:-arxiv}"
case "$SRC" in
  arxiv)    SCRIPT="scripts/harvest_arxiv.py";    ARGS=(--out data/raw/arxiv_metadata --set physics) ;;
  openalex) SCRIPT="scripts/harvest_openalex.py"; ARGS=(--out data/raw/openalex_works) ;;
  *) echo "fonte desconhecida: $SRC (use: arxiv | openalex)" >&2; exit 2 ;;
esac

LOG="data/raw/harvest_${SRC}.log"
mkdir -p "$(dirname "$LOG")"

if pgrep -f "$(basename "$SCRIPT")" >/dev/null; then
  echo "$SRC já em execução (PID $(pgrep -f "$(basename "$SCRIPT")" | tr '\n' ' '))"
  exit 0
fi

(
  PYTHONPATH=src nohup caffeinate -i .venv/bin/python "$SCRIPT" "${ARGS[@]}" >>"$LOG" 2>&1 &
  disown
)
sleep 3
if pgrep -f "$(basename "$SCRIPT")" >/dev/null; then
  echo "$SRC iniciado (PID $(pgrep -f "$(basename "$SCRIPT")" | head -1)) · log: $LOG"
else
  echo "FALHOU ao iniciar $SRC — ver $LOG" >&2
  exit 1
fi
