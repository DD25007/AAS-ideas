#!/bin/bash
# Epoch-0 gate: plain ConSam, --train-seed 0, stopped after epoch 0. Its first [record] line must equal the
# reference run's on every field (see scripts/compare_epoch0_record.py).
#   scripts/gate_epoch0.sh <gpu> [logs-dir]       default logs-dir: outputs/gate_epoch0
# Takes the GPU slot lock for the card (exit 3 if it is held). Does not choose the card for you: check
# `python3 scripts/gpu_locks.py busy` and nvidia-smi first.
set -uo pipefail
GPU=${1:?usage: gate_epoch0.sh <gpu> [logs-dir]}
cd "$(dirname "$0")/.."
LOGS=${2:-outputs/gate_epoch0}
mkdir -p "$LOGS"
export CUDA_VISIBLE_DEVICES=$GPU
. scripts/gpu_guard.sh                       # slot lock for this card, released on exit
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/storage/dd25007/conda_envs/hitl/bin/python
$PY -u train_al.py -d wildlife --subset combined -s ConSam --seed 1 --split-seed 0 --canonical-pairs \
    --data-dir "$PWD/dataroot" -j 6 --deterministic --train-seed 0 --fast-rerank \
    --stop-after-epoch 0 --logs-dir "$LOGS" 2>&1 \
  | gawk '{ print strftime("%Y-%m-%dT%H:%M:%S"), $0; fflush(); }' > "$LOGS/attempt_0.log"
EXIT=${PIPESTATUS[0]}
echo "[gate] train_al.py exited $EXIT" >> "$LOGS/attempt_0.log"
exit $EXIT
