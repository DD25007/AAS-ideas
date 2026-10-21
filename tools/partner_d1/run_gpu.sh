#!/usr/bin/env bash
# usage: run_gpu.sh <gpu> <logfile> <python script> [args...]  -- holds a slot lock for the job's lifetime, checks the 16 GiB gate first.
set -u
GPU=$1; LOGF=$2; shift 2
export BECA_LOCK_DIR=/bhallukah/dd25007/gpu_locks
export CUDA_VISIBLE_DEVICES=$GPU PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /storage/dd25007/AAS-ideas
FREE=$(nvidia-smi --id=$GPU --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$FREE" -lt 16384 ]; then echo "GPU $GPU only $FREE MiB free (< 16 GiB gate)" >&2; exit 4; fi
source scripts/gpu_guard.sh
echo "gpu $GPU free ${FREE} MiB, slot $GPU_SLOT, start $(date -Is)" > "$LOGF"
/storage/dd25007/conda_envs/hitl/bin/python -u "$@" >> "$LOGF" 2>&1
echo "exit $? at $(date -Is)" >> "$LOGF"
