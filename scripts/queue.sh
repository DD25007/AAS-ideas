#!/bin/bash
# Job queue: keeps one job per GPU, gates on free VRAM, adopts jobs already running, and
# re-queues anything that dies without completing (the shared-GPU faiss/CUDA OOM that cost
# nine runs overnight on 2026-08-30).
#
# Usage: queue.sh al  <OUTROOT> [ "<SAMPLER> <SEED>" ... ]     default: the full 6x4 AL grid
#        queue.sh row <OUTROOT> [ "<ROW> <SEED>" ... ]         default: all rows x seeds 0-3
#
# Both modes can run concurrently: GPU ownership is decided by the shared hitl_*_g<N>_*
# tmux session pattern in common.sh, not by per-queue prefixes.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"

MODE=${1:?usage: queue.sh <al|row> <OUTROOT> [jobs...]}
OUTROOT=${2:?usage: queue.sh <al|row> <OUTROOT> [jobs...]}
shift 2

MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
STATE="$HERE/.state/${MODE}_attempts.txt"
mkdir -p "$HERE/.state" "$OUTROOT"; touch "$STATE"

case "$MODE" in
  al)
    RUNNER="$HERE/run_grid.sh"
    DEFAULT_JOBS=(); for m in USL random GBAS LBAS CC ConSam; do
      for s in 1 2 3 4; do DEFAULT_JOBS+=("$m $s"); done; done
    # a finished AL run prints its total wall clock as the last thing it does
    is_complete() { local f="$OUTROOT/al/$1/s$2/train.log"
      [ -f "$f" ] && tr -d '\r' < "$f" | grep -q "Total running time"; }
    job_dir() { echo "$OUTROOT/al/$1/s$2"; }
    ;;
  row)
    RUNNER="$HERE/run_row.sh"
    DEFAULT_JOBS=(); for r in resnet50_baseline swin_foundation \
        resnet50_skyline_e30_zerobased resnet50_skyline_e30_offset \
        resnet50_skyline_e50_zerobased resnet50_skyline_e50_offset; do
      for s in 0 1 2 3; do DEFAULT_JOBS+=("$r $s"); done; done
    # main.py's final macro summary line
    is_complete() { grep -lq "^mean," "$OUTROOT/table1/$1/s$2"/run_*.log 2>/dev/null; }
    job_dir() { echo "$OUTROOT/table1/$1/s$2"; }
    ;;
  *) echo "queue.sh: mode must be 'al' or 'row'" >&2; exit 2 ;;
esac

if [ "$#" -gt 0 ]; then JOBS=("$@"); else JOBS=("${DEFAULT_JOBS[@]}"); fi

session_name() { echo "hitl_${MODE}_g$1_$2_s$3"; }
is_running()   { sessions | grep -qx "hitl_${MODE}_g[0-9]*_$1_s$2"; }
attempts()     { grep -cx "$1_s$2" "$STATE" 2>/dev/null || true; }

log "$MODE queue: ${#JOBS[@]} job(s), max $MAX_ATTEMPTS attempts each, out=$OUTROOT"

# Adopt anything already running so a restart of this script does not double-count attempts.
for j in "${JOBS[@]}"; do set -- $j
  if is_running "$1" "$2" && [ "$(attempts "$1" "$2")" -eq 0 ]; then
    echo "${1}_s${2}" >> "$STATE"; log "adopted running $1 seed$2"
  fi
done

while true; do
  pending=0
  for j in "${JOBS[@]}"; do set -- $j
    is_complete "$1" "$2" && continue
    is_running  "$1" "$2" && continue
    [ "$(attempts "$1" "$2")" -lt "$MAX_ATTEMPTS" ] && pending=$((pending+1))
  done
  running=$(sessions | grep -c "^hitl_${MODE}_g" || true)

  if [ "$pending" -eq 0 ] && [ "$running" -eq 0 ]; then
    incomplete=0
    for j in "${JOBS[@]}"; do set -- $j; is_complete "$1" "$2" || incomplete=$((incomplete+1)); done
    if [ "$incomplete" -eq 0 ]; then log "ALL ${#JOBS[@]} $MODE JOBS COMPLETE"
    else log "STOPPED: nothing running, $incomplete job(s) exhausted their attempts"; fi
    break
  fi

  # Fill every eligible GPU in one pass rather than one per outer tick.
  launched=0
  while :; do
    gpu=$(best_gpu); [ -z "$gpu" ] && break
    picked=""
    for j in "${JOBS[@]}"; do set -- $j
      is_complete "$1" "$2" && continue
      is_running  "$1" "$2" && continue
      [ "$(attempts "$1" "$2")" -ge "$MAX_ATTEMPTS" ] && continue
      picked="$1 $2"; break
    done
    [ -z "$picked" ] && break
    set -- $picked

    # Preserve the partial output of a failed attempt instead of overwriting it.
    d=$(job_dir "$1" "$2")
    if [ -d "$d" ]; then
      mv "$d" "${d}.failed$(date +%H%M%S)" 2>/dev/null && log "archived failed output of $1 seed$2"
    fi

    echo "${1}_s${2}" >> "$STATE"
    # `env -u LD_LIBRARY_PATH` so tmux itself does not load the conda env's libtinfo;
    # $RUNNER sources common.sh and sets LD_LIBRARY_PATH for the python process anyway.
    env -u LD_LIBRARY_PATH tmux new-session -d -s "$(session_name "$gpu" "$1" "$2")" \
        "bash $RUNNER $1 $2 $gpu $OUTROOT"
    log "launched $1 seed$2 on gpu$gpu (attempt $(attempts "$1" "$2"), $(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader) used)"
    launched=1
    sleep 5   # let tmux register the session before best_gpu looks again
  done

  [ "$launched" -eq 0 ] && log "waiting: $pending queued, $running running, no free GPU with >=${MIN_FREE_MIB}MiB"
  sleep 90
done
