#!/bin/bash
# Item 13: does BasicTrainer's accumulation semantics explain the skyline overshoot?
#
# Paper skyline: 64.29 mAP / 49.75 mINP / 72.04 BAKS / 80.02 AUC ROC / 72.88 Top-1.
# Ours (unscaled loss, the shipped behaviour): 65.86 @30ep, 68.55 @50ep -- both overshoot on
# every closed-set metric while AUC ROC undershoots by 3-5. If the missing loss scaling is
# the cause, HITL_BASICTRAINER_ACCUM=1 halves the effective gradient and should pull the
# closed-set metrics down toward 64.29 and AUC ROC up toward 80.02.
#
# Waits for real GPU headroom so it cannot starve the AL grid.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
OUT=${1:-/scratch/dd25007/HITL-ReID/outputs_sky13}
NEED=${NEED:-40000}

wait_for_gpu() {
  while :; do
    g=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
        | awk -F', *' -v n="$NEED" '$3-$2>n{print $1; exit}')
    [ -n "$g" ] && { echo "$g"; return; }
    sleep 300
  done
}

for row in resnet50_skyline_e30_zerobased resnet50_skyline_e50_zerobased; do
  g=$(wait_for_gpu)
  log "item13: $row with BasicTrainer accumulation on gpu$g"
  HITL_BASICTRAINER_ACCUM=1 bash "$(dirname "${BASH_SOURCE[0]}")/run_row.sh" "$row" 0 "$g" "$OUT"
done
log "item13 experiment complete"
