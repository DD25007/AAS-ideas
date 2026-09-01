#!/bin/bash
# One non-AL Table 1 row (baseline / foundation / skyline), one gallery split.
# Usage: run_row.sh <ROW> <SEED> <GPU> <OUTROOT> [extra args...]
#
# Merges the former run_seeded.sh + run_table1.sh + run_swintest.sh, which each knew a
# different subset of the rows -- that split is what scattered the same row's seeds across
# three output directories. --seed (training) is pinned at 1; SEED here is --sample_seed,
# which varies the 20% held-out gallery split, and is what the +/- in the README averages over.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ROW=$1; SEED=$2; GPU=$3; OUTROOT=$4; shift 4

unset SKYLINE_LABEL_MODE
case "$ROW" in
  resnet50_baseline)              ARGS="--baseline --model_name resnet50" ;;
  resnet50_skyline_e30_zerobased) ARGS="--skyline --model_name resnet50 --epochs 30"; export SKYLINE_LABEL_MODE=zerobased ;;
  resnet50_skyline_e30_offset)    ARGS="--skyline --model_name resnet50 --epochs 30"; export SKYLINE_LABEL_MODE=offset ;;
  resnet50_skyline_e50_zerobased) ARGS="--skyline --model_name resnet50 --epochs 50"; export SKYLINE_LABEL_MODE=zerobased ;;
  resnet50_skyline_e50_offset)    ARGS="--skyline --model_name resnet50 --epochs 50"; export SKYLINE_LABEL_MODE=offset ;;

  # The published "MegaDescriptor" foundation row. Passing the plain timm name is
  # deliberate: get_model's MegaDescriptor->Swin-L swap does not match it, so this loads
  # ImageNet Swin-L directly rather than relying on the swap to do it.
  swin_foundation)                ARGS="--baseline --model_name swin_large_patch4_window7_224" ;;

  megadescriptor)
    cat >&2 <<'MSG'
run_row.sh: the `megadescriptor` row (real MegaDescriptor weights, 53.34 mAP) cannot be
re-run as-is.

Those runs were produced while a local --baseline bypass of the get_model swap was active.
That bypass has since been reverted (correctly -- the swap is what the *published* numbers
went through; see README "The paper's MegaDescriptor row is ImageNet Swin-L" and Todo.md
item 15). With the swap live, --model_name hf-hub:BVRA/MegaDescriptor-L-224 under
--dataset combined silently loads ImageNet Swin-L, so this row would duplicate
swin_foundation while claiming to be MegaDescriptor.

Re-running it needs an explicit opt-out of the swap (a --no_model_swap flag), which does
not exist yet. The frozen results are in outputs_initial/table1/megadescriptor/.
MSG
    exit 2 ;;

  miewid)
    echo "run_row.sh: the miewid row is blocked by transformers 5.x (Todo.md item 14)." >&2
    echo "Its remote code targets transformers 4.x: 'MiewIdNet' object has no attribute" >&2
    echo "'all_tied_weights_keys'. Needs a transformers 4.x pin first." >&2
    exit 2 ;;

  *) echo "run_row.sh: unknown row '$ROW'" >&2; exit 2 ;;
esac

OUT=$OUTROOT/table1/$ROW/s$SEED
mkdir -p "$OUT"

cd "$REPO"
export CUDA_VISIBLE_DEVICES=$GPU

# --images_per_id 5 --metric precomputed are REQUIRED. main.py defaults to
# images_per_id=1, metric=cosine, which builds a different evaluation gallery than the
# paper (up to five exemplars per ID) and than every AL run.
exec "$PYTHON" -u main.py \
  --dataset combined --dataset_path "$DATASET_PATH" --out_dir "$OUT" \
  $ARGS --seed 1 --sample_seed "$SEED" --num_workers 6 \
  --images_per_id 5 --metric precomputed "$@" \
  > "$OUT/run_$(date +%Y%m%d_%H%M%S).log" 2>&1
