#!/bin/bash
# Shared config and GPU scheduling for scripts/run.sh. Sourced, never run.

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Override to switch env: PYTHON=/storage/dd25007/conda_envs/hitl-modern/bin/python
PYTHON="${PYTHON:-/storage/dd25007/conda_envs/hitl/bin/python}"

# train_al.py wants <data-dir>/wildlife/wildlifereid-10k/; main.py takes the root directly.
DATAROOT="${DATAROOT:-$REPO/dataroot}"
DATASET_PATH="${DATASET_PATH:-/storage/dd25007/dataset}"

# The launchers invoke $PYTHON directly instead of `conda activate`, so derive what activate
# would have set. LD_LIBRARY_PATH is load-bearing: conda's libfaiss needs CXXABI_1.3.15 from
# the env's libstdc++, and importing pandas/torch first pulls in the older system one.
CONDA_ENV_PREFIX="$(cd "$(dirname "$(dirname "$PYTHON")")" && pwd)"
export LD_LIBRARY_PATH="$CONDA_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1

# Required for reproducible AL runs: the AAS sampler iterates connected_components() over
# string-named nodes, and per-process string hashing changes which pairs it selects.
export PYTHONHASHSEED=0

export HF_HOME="${HF_HOME:-/storage/dd25007/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/storage/dd25007/cache/torch}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# Unset, each process sizes its BLAS pool to all 224 cores and the machine thrashes.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

# Peak demand is ~9.3 GiB/job: ~6-7.7 GiB resident plus a ~1.5 GiB faiss temporary during
# compute_jaccard_distance. Jobs hit that spike at different times, and `used` only reports
# what a co-tenant has allocated SO FAR -- so best_gpu reserves MIN_FREE_MIB per slot it is
# about to occupy, not just for the incoming job. Without that scaling three jobs each pass
# an individual 16 GiB check and then OOM together, which is what killed four runs.
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-3}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"

sessions() { tmux ls 2>/dev/null | cut -d: -f1; }
# Must stay in step with run.sh's tag(), which ends session names in _g<gpu>. When it did
# not, this counted zero for every device and MAX_JOBS_PER_GPU stopped applying.
gpu_jobs() { sessions | grep -cE "^hitl_.*_g${1}$"; }
gpu_busy() { [ "$(gpu_jobs "$1")" -ge "$MAX_JOBS_PER_GPU" ]; }

# Free VRAM a GPU must have before we may add one more job: one MIN_FREE_MIB slot for the
# incoming job plus one for each job of ours already there, which may not have peaked yet.
gpu_required_mib() { echo $(( MIN_FREE_MIB * ($(gpu_jobs "$1") + 1) )); }

# Emptiest eligible GPU; prints nothing if none has room.
best_gpu() {
  local best="" bestfree=0 used total free need g
  for g in $GPUS; do
    gpu_busy "$g" && continue
    read -r used total <<< "$(nvidia-smi -i "$g" --query-gpu=memory.used,memory.total \
        --format=csv,noheader,nounits | tr ',' ' ')"
    free=$((total - used))
    need=$(gpu_required_mib "$g")
    [ "$free" -lt "$need" ] && continue
    if [ "$free" -gt "$bestfree" ]; then bestfree=$free; best=$g; fi
  done
  [ -n "$best" ] && echo "$best"
}

log() { echo "$(date +%T) $*"; }
