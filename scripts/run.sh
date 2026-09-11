#!/bin/bash
# Single entry point for every Table 1 run.
#
#   run.sh list                        print the full final grid
#   run.sh one   <JOB> <GPU> <OUTROOT> run one job in the foreground
#   run.sh queue <OUTROOT> [JOB ...]   schedule jobs across GPUs (default: the full grid)
#   run.sh status <OUTROOT>            per-job done / running / pending
#   run.sh clean  <OUTROOT>            drop logs and .failed dirs of incomplete attempts
#
# A JOB is "<kind>:<name>:<seed>":
#   al:ConSam:1     an AL row  (samplers: USL random GBAS LBAS CC ConSam; seeds 1-4)
#   row:miewid:0    a non-AL row                                        (seeds 0-3)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MIEWID_PYTHON="${MIEWID_PYTHON:-/storage/dd25007/conda_envs/hitl-miewid/bin/python}"

# Must be chosen BEFORE common.sh is sourced -- it derives LD_LIBRARY_PATH from $PYTHON.
# miewid needs transformers 4.x, so it runs under its own env.
job_python() { case "$1" in row:miewid:*) echo "$MIEWID_PYTHON" ;; *) echo "" ;; esac; }

AL_SAMPLERS=(USL random GBAS LBAS CC ConSam)
AL_SEEDS=(1 2 3 4)
# swin_foundation IS the MegaDescriptor row -- get_model substitutes ImageNet Swin-L, so
# `megadescriptor` reaches identical weights and would duplicate it.
# megadescriptor_realweights is absent: it needs a get_model bypass that was reverted.
ROWS=(resnet50_baseline swin_foundation miewid
      resnet50_skyline_e30_zerobased resnet50_skyline_e30_offset
      resnet50_skyline_e50_zerobased resnet50_skyline_e50_offset)
ROW_SEEDS=(0 1 2 3)

all_jobs() {
  local m s r
  for m in "${AL_SAMPLERS[@]}"; do for s in "${AL_SEEDS[@]}"; do echo "al:$m:$s"; done; done
  for r in "${ROWS[@]}";        do for s in "${ROW_SEEDS[@]}"; do echo "row:$r:$s"; done; done
}

# tmux session name for a job: al:ConSam:1 -> hitl_al_ConSam_1_g<gpu>
tag()        { echo "hitl_$(echo "$1" | tr ':' '_')"; }
is_running() { sessions | grep -qE "^$(tag "$1")_g[0-9]+$"; }

job_dir() { # <JOB> <OUTROOT>
  local IFS=:; set -- $1 "$2"
  case "$1" in
    al)  echo "$4/al/$2/s$3" ;;
    row) echo "$4/table1/$2/s$3" ;;
  esac
}

# A finished AL run prints its wall clock last; main.py ends with its macro summary line.
is_complete() { # <JOB> <OUTROOT>
  local d; d=$(job_dir "$1" "$2")
  case "${1%%:*}" in
    al)  [ -f "$d/train.log" ] && tr -d '\r' < "$d/train.log" | grep -q "Total running time" ;;
    row) grep -lq "^mean," "$d"/run_*.log 2>/dev/null ;;
  esac
}

# ---------------------------------------------------------------- one job, foreground ---
cmd_one() {
  local JOB=${1:?job} GPU=${2:?gpu} OUTROOT=${3:?outroot}
  local p; p=$(job_python "$JOB"); [ -n "$p" ] && export PYTHON="$p"
  source "$HERE/common.sh"

  local IFS=: ; set -- $JOB ; local KIND=$1 NAME=$2 SEED=$3 ; unset IFS
  local OUT; OUT=$(job_dir "$JOB" "$OUTROOT"); mkdir -p "$OUT"
  cd "$REPO"; export CUDA_VISIBLE_DEVICES=$GPU

  if [ "$KIND" = al ]; then
    # Defaults already match the paper. USL means "no -s at all".
    local SARG=""; [ "$NAME" != USL ] && SARG="-s $NAME"
    exec "$PYTHON" -u train_al.py \
      -d wildlife --subset combined $SARG --seed "$SEED" \
      --data-dir "$DATAROOT" --logs-dir "$OUT" -j 6 \
      > "$OUT/train.log" 2>&1
  fi

  local ARGS; unset SKYLINE_LABEL_MODE
  case "$NAME" in
    resnet50_baseline)              ARGS="--baseline --model_name resnet50" ;;
    swin_foundation)                ARGS="--baseline --model_name swin_large_patch4_window7_224" ;;
    megadescriptor)                 ARGS="--baseline --model_name hf-hub:BVRA/MegaDescriptor-L-224" ;;
    miewid)
      if ! "$PYTHON" -c "import transformers,sys; sys.exit(0 if int(transformers.__version__.split('.')[0])<5 else 1)" 2>/dev/null; then
        echo "run.sh: miewid needs transformers 4.x; set MIEWID_PYTHON to an env that has it" >&2; exit 2
      fi
      ARGS="--baseline --model_name conservationxlabs/miewid-msv2" ;;
    resnet50_skyline_e30_zerobased) ARGS="--skyline --model_name resnet50 --epochs 30"; export SKYLINE_LABEL_MODE=zerobased ;;
    resnet50_skyline_e30_offset)    ARGS="--skyline --model_name resnet50 --epochs 30"; export SKYLINE_LABEL_MODE=offset ;;
    resnet50_skyline_e50_zerobased) ARGS="--skyline --model_name resnet50 --epochs 50"; export SKYLINE_LABEL_MODE=zerobased ;;
    resnet50_skyline_e50_offset)    ARGS="--skyline --model_name resnet50 --epochs 50"; export SKYLINE_LABEL_MODE=offset ;;
    *) echo "run.sh: unknown row '$NAME'" >&2; exit 2 ;;
  esac

  # --images_per_id 5 --metric precomputed are REQUIRED: the defaults build a different
  # evaluation gallery than the paper and than every AL run. The seed here is --sample_seed
  # (the gallery split); --seed (training) is pinned at 1.
  exec "$PYTHON" -u main.py \
    --dataset combined --dataset_path "$DATASET_PATH" --out_dir "$OUT" \
    $ARGS --seed 1 --sample_seed "$SEED" --num_workers 6 \
    --images_per_id 5 --metric precomputed \
    > "$OUT/run_$(date +%Y%m%d_%H%M%S).log" 2>&1
}

# ------------------------------------------------------------------------- scheduler ---
cmd_queue() {
  local OUTROOT=${1:?outroot}; shift
  source "$HERE/common.sh"
  local JOBS=(); if [ "$#" -gt 0 ]; then JOBS=("$@"); else mapfile -t JOBS < <(all_jobs); fi

  local MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
  local STATE="$HERE/.state/$(basename "$OUTROOT")_attempts.txt"
  mkdir -p "$HERE/.state" "$OUTROOT"; touch "$STATE"

  attempts()   { grep -cx "$1" "$STATE" 2>/dev/null || true; }

  log "queue: ${#JOBS[@]} job(s), max $MAX_ATTEMPTS attempts each, gate ${MIN_FREE_MIB}MiB, out=$OUTROOT"

  # Adopt anything already running so a restart does not double-count attempts.
  local j
  for j in "${JOBS[@]}"; do
    if is_running "$j" && [ "$(attempts "$j")" -eq 0 ]; then
      echo "$j" >> "$STATE"; log "adopted running $j"
    fi
  done

  while true; do
    local pending=0 running
    for j in "${JOBS[@]}"; do
      is_complete "$j" "$OUTROOT" && continue
      is_running "$j" && continue
      [ "$(attempts "$j")" -lt "$MAX_ATTEMPTS" ] && pending=$((pending+1))
    done
    running=$(sessions | grep -c "^hitl_" || true)

    if [ "$pending" -eq 0 ] && [ "$running" -eq 0 ]; then
      local bad=0
      for j in "${JOBS[@]}"; do is_complete "$j" "$OUTROOT" || bad=$((bad+1)); done
      if [ "$bad" -eq 0 ]; then log "ALL ${#JOBS[@]} JOBS COMPLETE"
      else log "STOPPED: nothing running, $bad job(s) exhausted their attempts"; fi
      break
    fi

    local launched=0 gpu picked d
    while :; do
      gpu=$(best_gpu); [ -z "$gpu" ] && break
      picked=""
      for j in "${JOBS[@]}"; do
        is_complete "$j" "$OUTROOT" && continue
        is_running "$j" && continue
        [ "$(attempts "$j")" -ge "$MAX_ATTEMPTS" ] && continue
        picked="$j"; break
      done
      [ -z "$picked" ] && break

      # Preserve a failed attempt's output instead of overwriting it.
      d=$(job_dir "$picked" "$OUTROOT")
      if [ -d "$d" ]; then
        mv "$d" "${d}.failed$(date +%H%M%S)" 2>/dev/null && log "archived failed output of $picked"
      fi
      echo "$picked" >> "$STATE"
      # env -u so tmux itself does not load the conda env's libtinfo; cmd_one re-sets it.
      env -u LD_LIBRARY_PATH tmux new-session -d -s "$(tag "$picked")_g${gpu}" \
          "bash '$HERE/run.sh' one '$picked' '$gpu' '$OUTROOT'"
      log "launched $picked on gpu$gpu (attempt $(attempts "$picked"), $(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader) used)"
      launched=1; sleep 5
    done

    [ "$launched" -eq 0 ] && log "waiting: $pending queued, $running running, no free GPU with >=${MIN_FREE_MIB}MiB"
    sleep 90
  done
}

cmd_status() {
  local OUTROOT=${1:?outroot}; shift
  source "$HERE/common.sh"
  local JOBS=(); if [ "$#" -gt 0 ]; then JOBS=("$@"); else mapfile -t JOBS < <(all_jobs); fi
  local done_=0 run=0 pend=0 j st
  for j in "${JOBS[@]}"; do
    if is_complete "$j" "$OUTROOT"; then st="done"; done_=$((done_+1))
    elif is_running "$j";           then st="RUNNING"; run=$((run+1))
    else st="pending"; pend=$((pend+1)); fi
    printf "  %-40s %s\n" "$j" "$st"
  done
  echo "  -- $done_ done, $run running, $pend pending, ${#JOBS[@]} total"
}

# Drop logs from attempts that never completed. Skips jobs still running.
cmd_clean() {
  local OUTROOT=${1:?outroot}
  source "$HERE/common.sh"
  local j d f n=0
  while IFS= read -r j; do
    is_running "$j" && { echo "  skip (running) $j"; continue; }
    is_complete "$j" "$OUTROOT" && continue
    d=$(job_dir "$j" "$OUTROOT"); [ -d "$d" ] || continue
    for f in "$d"/*.log "$d"/train.log; do
      [ -f "$f" ] || continue
      rm -f "$f"; echo "  removed incomplete log ${f#"$OUTROOT"/}"; n=$((n+1))
    done
  done < <(all_jobs)
  # ... and the .failedHHMMSS dirs the queue parks beside a retried job
  while IFS= read -r d; do
    rm -rf "$d"; echo "  removed ${d#"$OUTROOT"/}"; n=$((n+1))
  done < <(find "$OUTROOT" -maxdepth 3 -name '*.failed[0-9]*' -type d 2>/dev/null)
  echo "  $n item(s) removed"
}

case "${1:-}" in
  list)   all_jobs ;;
  clean)  shift; cmd_clean "$@" ;;
  status) shift; cmd_status "$@" ;;
  one)   shift; cmd_one "$@" ;;
  queue) shift; cmd_queue "$@" ;;
  *) sed -n '2,20p' "$HERE/run.sh" >&2; exit 2 ;;
esac
