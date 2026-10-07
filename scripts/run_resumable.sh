#!/bin/bash
# Run train_al.py (through scripts/train_forkserver.py) on one GPU under the shared slot lock, writing resume files
# and restarting from the newest one if the process dies (crash, OOM, node kill).
#   scripts/run_resumable.sh <gpu> <run_dir> <train_al.py args...>
# Writes <run_dir>/{cmd.txt,wrapper.log,attempt_N.log,logs/}; --logs-dir and --resume-save-epochs are added here.
# env: MAX_RESUMES (default 3), RESUME_EVERY (default 10,20,30,40).
set -uo pipefail
GPU=${1:?usage: run_resumable.sh <gpu> <run_dir> <train_al.py args...>}
RUN_DIR=${2:?run_dir}
shift 2
cd "$(dirname "$0")/.."
mkdir -p "$RUN_DIR/logs"
export CUDA_VISIBLE_DEVICES=$GPU
. scripts/gpu_guard.sh                       # slot lock for this card, held until this script exits
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=${PY:-/storage/dd25007/conda_envs/hitl/bin/python}      # PY override is for tests only
MAX_RESUMES=${MAX_RESUMES:-3}
RESUME_EVERY=${RESUME_EVERY:-10,20,30,40}
echo "gpu=$GPU run_dir=$RUN_DIR args: $*" > "$RUN_DIR/cmd.txt"
WRAPPER_LOG=$RUN_DIR/wrapper.log
ATTEMPT=0
RESUME_ARG=""
echo "[wrapper] $(date -Iseconds) launch pid=$$ gpu=$GPU git=$(git rev-parse --short HEAD)" >> "$WRAPPER_LOG"
while true; do
  ATTEMPT_LOG=$RUN_DIR/attempt_${ATTEMPT}.log
  echo "[wrapper] $(date -Iseconds) starting attempt $ATTEMPT (resume_arg='$RESUME_ARG')" >> "$WRAPPER_LOG"
  # No `python | gawk` pipeline: when the trainer is killed (e.g. a 24 h limit) its orphaned DataLoader workers keep the pipe open, gawk never
  # sees EOF and the pipeline -- and so this loop -- hangs forever (seen in H1). With a process substitution the loop gets python's own status.
  $PY -u scripts/train_forkserver.py "$@" --logs-dir "$RUN_DIR/logs" --resume-save-epochs "$RESUME_EVERY" $RESUME_ARG \
      > >(gawk '{ print strftime("%Y-%m-%dT%H:%M:%S"), $0; fflush(); }' > "$ATTEMPT_LOG") 2>&1
  EXIT=$?
  sleep 5                                    # let gawk flush the tail of the log
  # Remove this attempt's leftovers (orphaned loader workers / forkserver). Only when this script is its own session leader
  # (launched with setsid), so a hand-started wrapper can never signal the user's shell.
  MY_SID=$(ps -o sid= -p $$ | tr -d ' ')
  if [ "$MY_SID" = "$$" ]; then
    for p in $(ps -eo sid,pid | awk -v s="$MY_SID" -v me=$$ '$1==s && $2!=me {print $2}'); do kill -TERM "$p" 2>/dev/null; done
  fi
  echo "[wrapper] $(date -Iseconds) attempt $ATTEMPT exited $EXIT" >> "$WRAPPER_LOG"
  if [ "$EXIT" -eq 0 ]; then
    echo "[wrapper] $(date -Iseconds) run finished successfully" >> "$WRAPPER_LOG"
    echo "RUN_DONE" > "$RUN_DIR/run.done"
    break
  fi
  if [ "$ATTEMPT" -ge "$MAX_RESUMES" ]; then
    echo "[wrapper] $(date -Iseconds) exceeded max resumes ($MAX_RESUMES); giving up" >> "$WRAPPER_LOG"
    echo "RUN_FAILED" > "$RUN_DIR/run.done"
    break
  fi
  LATEST=$(ls -t "$RUN_DIR"/logs/resume_epoch*.pth 2>/dev/null | head -1)
  ATTEMPT=$((ATTEMPT+1))
  if [ -z "$LATEST" ]; then
    echo "[wrapper] $(date -Iseconds) no resume file yet; restarting fresh from epoch 0 (attempt $ATTEMPT)" >> "$WRAPPER_LOG"
    RESUME_ARG=""
  else
    echo "[wrapper] $(date -Iseconds) resuming from $LATEST (attempt $ATTEMPT)" >> "$WRAPPER_LOG"
    RESUME_ARG="--resume $LATEST"
  fi
done
