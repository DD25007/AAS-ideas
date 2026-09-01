#!/bin/bash
# Shared configuration and GPU-scheduling helpers for every launcher in this directory.
#
# Rescued 2026-09-01 from the ephemeral session scratchpad that produced the published
# Table 1 reproduction. The operational logic (1 job per GPU, 16 GiB VRAM gate, auto-retry,
# thread caps) is unchanged; only hardcoded paths were parameterised.

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Interpreter. Override to switch env:  PYTHON=/storage/dd25007/conda_envs/alm/bin/python ...
PYTHON="${PYTHON:-/storage/dd25007/conda_envs/hitl/bin/python}"

# The SpCL wildlife loader wants <data-dir>/wildlife/wildlifereid-10k/metadata.csv.
# $REPO/dataroot/wildlife is a symlink to the dataset root that provides exactly that.
DATAROOT="${DATAROOT:-$REPO/dataroot}"
# main.py takes the dataset root directly rather than through the wildlife/ indirection.
DATASET_PATH="${DATASET_PATH:-/storage/dd25007/dataset}"

# These two are normally supplied by `conda activate`, which the launchers deliberately do
# not do -- they invoke the interpreter directly. Derive them from $PYTHON so a run is
# self-contained rather than depending on an ambient value inherited from some shell.
#
# LD_LIBRARY_PATH is load-bearing (Todo.md item 8): conda's libfaiss.so needs CXXABI_1.3.15
# from the env's libstdc++, but importing pandas/torch first pulls in the older system
# libstdc++, and faiss then fails with "version CXXABI_1.3.15 not found". Symptom only
# appears when something else is imported before faiss, so it hides in short test scripts.
CONDA_ENV_PREFIX="$(cd "$(dirname "$(dirname "$PYTHON")")" && pwd)"
export LD_LIBRARY_PATH="$CONDA_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1

export HF_HOME="${HF_HOME:-/storage/dd25007/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/storage/dd25007/cache/torch}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# Cap the BLAS/OpenMP pools. Left unset, each process sizes its pool to all 224 cores;
# with 8 concurrent jobs the 1-minute load average hit 715 and everything crawled.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

# Required free VRAM before a job may be placed on a GPU.
# Measured 2026-08-30 across 8 concurrent jobs: 6088-7690 MiB resident (median 6100).
# faiss StandardGpuResources additionally requests a ~1536 MiB temporary buffer during
# compute_jaccard_distance -- that is the allocation that OOM'd nine times overnight.
# So real peak demand is ~9.3 GiB; 16 GiB gives ~70% headroom for a co-tenant growing
# after we launch, which is the failure mode a launch-time check cannot otherwise cover.
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

# Every launcher names its tmux session hitl_<queue>_g<gpu>_<job>, so a single pattern
# finds our jobs on a GPU regardless of which queue placed them there.
sessions() { tmux ls 2>/dev/null | cut -d: -f1; }
gpu_busy() { sessions | grep -q "^hitl_[a-z0-9]*_g${1}_"; }

# Pick the GPU with the most free memory that has no job of ours; print nothing if none
# clears MIN_FREE_MIB.
best_gpu() {
  local best="" bestfree=0 used total free g
  for g in $GPUS; do
    gpu_busy "$g" && continue
    read -r used total <<< "$(nvidia-smi -i "$g" --query-gpu=memory.used,memory.total \
        --format=csv,noheader,nounits | tr ',' ' ')"
    free=$((total - used))
    if [ "$free" -gt "$bestfree" ]; then bestfree=$free; best=$g; fi
  done
  [ -n "$best" ] && [ "$bestfree" -ge "$MIN_FREE_MIB" ] && echo "$best"
}

log() { echo "$(date +%T) $*"; }
