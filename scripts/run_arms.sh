#!/bin/bash
# Launcher for the 4-arm x 4-split-seed ablation grid. Sibling of run.sh, which owns
# the Table 1 grid; this one varies --split-seed (0-3), not --seed.
#
#   run_arms.sh list                       print the grid in priority order
#   run_arms.sh one   <JOB> <GPU> <OUT>    run one job in the foreground
#   run_arms.sh queue <OUT> [MAXJOBS]      schedule the grid across free GPUs
#   run_arms.sh status <OUT>               per-job done / running / pending
#
# JOB is "<arm>:<split_seed>", arm in A B C D, split seed in 0 1 2 3.
#   A  ConSam                                   (control)
#   B  ConSam --consam_eps 0.0                  (U_os removed)
#   C  ConSamP1 --consam-finch-partition 1      (coarser FINCH)
#   D  SimBand --simband-lo 0.1 --simband-hi 1.0 (clustering-free null)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARMS=(C D A B)          # priority order, per the launch brief
SPLIT_SEEDS=(0 1 2 3)
MIN_FREE_MIB=${MIN_FREE_MIB:-14000}   # VRAM gate: never place a job below this
MAX_PER_GPU=${MAX_PER_GPU:-3}         # at most this many of OUR jobs per GPU
POLL=${POLL:-120}

arm_args() {
  case "$1" in
    A) echo "-s ConSam" ;;
    B) echo "-s ConSam --consam_eps 0.0" ;;
    C) echo "-s ConSamP1 --consam-finch-partition 1" ;;
    D) echo "-s SimBand --simband-lo 0.1 --simband-hi 1.0" ;;
    *) echo "run_arms.sh: unknown arm '$1'" >&2; return 2 ;;
  esac
}

_job_dir() { local a=${1%%:*} s=${1##*:}; echo "$2/arm${a}_split${s}"; }
tag() { echo "arm${1%%:*}_s${1##*:}"; }

grid() { for a in "${ARMS[@]}"; do for s in "${SPLIT_SEEDS[@]}"; do echo "$a:$s"; done; done; }

cmd_list() { grid; }

cmd_one() {
  local JOB=${1:?job} GPU=${2:?gpu} OUTROOT=${3:?outroot}
  source "$HERE/common.sh"
  local ARM=${JOB%%:*} SS=${JOB##*:}
  local OUT; OUT=$(_job_dir "$JOB" "$OUTROOT"); mkdir -p "$OUT"
  cd "$REPO"; export CUDA_VISIBLE_DEVICES=$GPU
  echo "$GPU" > "$OUT/.gpu"          # queue reads this to avoid stacking jobs
  # Everything not named here is left at its default, which is the published config:
  # 50 epochs, --eval-step 10 (5 AL cycles), eps 0.6, k1 30, k2 6, budget 0.02%/cycle.
  # --partition-seed stays at its 102 default: the brief says do not vary it here.
  exec "$PYTHON" -u train_al.py \
    -d wildlife --subset combined $(arm_args "$ARM") \
    --seed 1 --split-seed "$SS" \
    --data-dir "$DATAROOT" --logs-dir "$OUT" -j 6 \
    > "$OUT/train.log" 2>&1
}

done_p() { [ -s "$1/run_record.json" ] && grep -q '"status": "ok"' "$1/run_record.json" 2>/dev/null; }
running_p() { tmux has-session -t "$1" 2>/dev/null; }

# One line per running job of OURS, holding the GPU index it took. Repeats are
# meaningful: they are the per-GPU occupancy count. Needed because nvidia-smi does not
# reflect a new job's memory for tens of seconds, so a naive "freest GPU" query hands
# the same index to every job launched in that window.
busy_gpus() {
  local OUTROOT=$1 out=""
  for j in $(grid); do
    local d; d=$(_job_dir "$j" "$OUTROOT")
    running_p "$(tag "$j")" || continue
    [ -s "$d/.gpu" ] && out="$out $(cat "$d/.gpu")"
  done
  echo "$out"
}

# Freest GPU that passes BOTH gates: >= MIN_FREE_MIB actually free right now, and
# fewer than MAX_PER_GPU of our jobs already on it. Empty output means "wait".
free_gpu() {
  local held=" ${1:-} "
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  | awk -F', ' -v m="$MIN_FREE_MIB" '{f=$3-$2; if (f>=m) print f, $1}' | sort -rn \
  | while read -r f i; do
      local n; n=$(tr ' ' '\n' <<<"$held" | grep -cx "$i" || true)
      [ "$n" -lt "$MAX_PER_GPU" ] && { echo "$i"; break; }
    done
}

cmd_queue() {
  local OUTROOT=${1:?outroot}; shift || true
  local MAXJOBS=${1:-16}
  local jobs; jobs=$(grid)
  while :; do
    local pending=0 launched=0
    for j in $jobs; do
      local d; d=$(_job_dir "$j" "$OUTROOT")
      done_p "$d" && continue
      running_p "$(tag "$j")" && continue
      pending=$((pending+1))
      local nrun; nrun=$(tmux ls 2>/dev/null | grep -c '^arm' || true)
      [ "$nrun" -ge "$MAXJOBS" ] && continue
      local g; g=$(free_gpu "$(busy_gpus "$OUTROOT")")
      [ -z "$g" ] && continue
      mkdir -p "$d"
      env -u LD_LIBRARY_PATH tmux new-session -d -s "$(tag "$j")" \
        "bash '$HERE/run_arms.sh' one '$j' '$g' '$OUTROOT'"
      echo "$(date +%H:%M:%S) launched $j on gpu$g"
      launched=1; sleep 30
    done
    [ "$pending" -eq 0 ] && { echo "all jobs done"; break; }
    [ "$launched" -eq 0 ] && sleep "$POLL"
  done
}

cmd_status() {
  local OUTROOT=${1:?outroot}
  for j in $(grid); do
    local d; d=$(_job_dir "$j" "$OUTROOT")
    if done_p "$d"; then echo "done     $j"
    elif running_p "$(tag "$j")"; then echo "running  $j"
    elif [ -s "$d/train.log" ]; then echo "STOPPED  $j  (log exists, no ok record)"
    else echo "pending  $j"; fi
  done
}

case "${1:-}" in
  list) shift; cmd_list "$@" ;;
  one) shift; cmd_one "$@" ;;
  queue) shift; cmd_queue "$@" ;;
  status) shift; cmd_status "$@" ;;
  *) sed -n '2,14p' "$0"; exit 2 ;;
esac
