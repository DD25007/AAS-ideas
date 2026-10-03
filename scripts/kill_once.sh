#!/bin/bash
# Resume test helper: SIGKILL the training process tree of <run_dir> once, in the middle of epoch <epoch>
# (as soon as the newest attempt log shows "Epoch: [<epoch>][200/400]"). The run_resumable.sh loop then restarts it.
#   scripts/kill_once.sh <run_dir> <epoch>
# Appends to <run_dir>/kill_resume.log: kill time, trigger line, killed PIDs, the newest resume file at that moment.
set -uo pipefail
RUN_DIR=${1:?usage: kill_once.sh <run_dir> <epoch>}; EPOCH=${2:?epoch}
cd "$(dirname "$0")/.."
LOG=$RUN_DIR/kill_resume.log
descendants() { local p; for p in $(pgrep -P "$1"); do echo "$p"; descendants "$p"; done; }
echo "[kill_once] $(date -Iseconds) armed: kill during epoch $EPOCH of $RUN_DIR" >> "$LOG"
while true; do
  A=$(ls -t "$RUN_DIR"/attempt_*.log 2>/dev/null | head -1)
  if [ -n "$A" ] && grep -q "Epoch: \[$EPOCH\]\[200/400\]" "$A"; then
    MAIN=$(pgrep -f "train_forkserver.py .*--logs-dir $RUN_DIR/logs" | head -1)
    [ -z "$MAIN" ] && { echo "[kill_once] $(date -Iseconds) trigger seen but no training process found" >> "$LOG"; exit 1; }
    PIDS="$MAIN $(descendants "$MAIN" | tr '\n' ' ')"
    NEWEST=$(ls -t "$RUN_DIR"/logs/resume_epoch*.pth 2>/dev/null | head -1)
    echo "[kill_once] $(date -Iseconds) KILL -9 pids: $PIDS" >> "$LOG"
    echo "[kill_once]   trigger line: $(grep "Epoch: \[$EPOCH\]\[200/400\]" "$A" | head -1)" >> "$LOG"
    echo "[kill_once]   newest resume file at kill time: $NEWEST" >> "$LOG"
    kill -9 $PIDS 2>/dev/null
    echo "[kill_once] $(date -Iseconds) done" >> "$LOG"
    exit 0
  fi
  sleep 5
done
