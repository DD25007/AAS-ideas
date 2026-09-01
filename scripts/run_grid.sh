#!/bin/bash
# One AL run (a Table 1 "SpCL + <sampler>" row, one seed).
# Usage: run_grid.sh <SAMPLER|USL> <SEED> <GPU> <OUTROOT> [extra args...]
#
# SAMPLER is the trainer's -s value: random | GBAS | LBAS | CC (NIS) | ConSam (AAS).
# USL means "no -s at all", which is the SpCL (Base) row.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SAMPLER=$1; SEED=$2; GPU=$3; OUTROOT=$4; shift 4

OUT=$OUTROOT/al/$SAMPLER/s$SEED
mkdir -p "$OUT"

SARG=""
[ "$SAMPLER" != "USL" ] && SARG="-s $SAMPLER"

cd "$REPO"
export CUDA_VISIBLE_DEVICES=$GPU

# Defaults already match the paper: --epochs 50 --eval-step 10 (5 AL cycles),
# --consam_eps 0.6 --consam_k 5 --consam_sim_thresh 0.3, and the 0.02%-per-cycle
# budget hardcoded in spcl/sampling/base.py.
exec "$PYTHON" -u train_al.py \
  -d wildlife --subset combined $SARG --seed "$SEED" \
  --data-dir "$DATAROOT" --logs-dir "$OUT" -j 6 "$@" \
  > "$OUT/train.log" 2>&1
