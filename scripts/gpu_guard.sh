# Sourced right after `export CUDA_VISIBLE_DEVICES=<n>` by launchers that hard-code a card.
# Claims a slot lock on that card (/scratch/dd25007/gpu_locks/, shared with beca-al) for this script's
# lifetime and releases it on exit; if the card already holds a live slot the script stops with exit 3.
_GL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_locks.py"
GPU_SLOT=$(python3 "$_GL" acquire "$CUDA_VISIBLE_DEVICES" "$PWD" $$) \
  || { echo "$0: refusing to start: GPU $CUDA_VISIBLE_DEVICES holds a live slot lock (see /scratch/dd25007/gpu_locks)" >&2; exit 3; }
trap 'python3 "$_GL" release "$GPU_SLOT"' EXIT
