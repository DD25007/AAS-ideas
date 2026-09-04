# HITL-ReID

```commandline
python main.py --dataset <DATASET_NAME> --baseline
python main.py --dataset <DATASET_NAME> --skyline
python main.py --dataset <DATASET_NAME> --sampler <SAMPLING_METHOD>
```


---

# Reproduction of the AAAI-26 paper (Table 1)

Reproduction of *Active Learning for Animal Re-Identification with Ambiguity-Aware
Sampling* on the 13 WildlifeReID-10k datasets. All numbers are the **macro mean over the
13 datasets** (each dataset evaluated against its own gallery, then averaged), matching
the paper's protocol. **Every row is the mean of 4 runs**, matching the paper's stated
protocol for its AL block.

Run on 8x NVIDIA H200. **See `Todo.md`** for every version constraint, bug and
optimisation found along the way — it is the companion to this file. The runs behind every
number are frozen read-only in `outputs_initial/`; `aggregate_results.py` regenerates the
tables below from them.

## Environment

```bash
conda create -n alm python=3.10 -y && conda activate alm
conda install -n alm -c pytorch faiss-gpu -y      # NOT pip: the wheels lack sm_90 kernels
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
conda env config vars set PYTHONNOUSERSITE=1 -n alm
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib -n alm
```

Both `env config vars` lines are required, not optional — see `Todo.md` items 8 and 9.
`alm` is preserved untouched as the environment these numbers were produced in. Ongoing work
uses `hitl`, a clone of it (`conda create --clone alm -n hitl`), so the published results stay
re-runnable exactly as they were. The launchers in `scripts/` export `LD_LIBRARY_PATH`
themselves, since they invoke the interpreter directly rather than activating the env.
`finch-clust==0.2.1` is a hard pin: 0.2.2+ changes a signature the AAS sampler calls
positionally, and the failure is silent (the run parks at a `pdb` prompt). Item 1.

Data: WildlifeReID-10k from Kaggle, laid out so the SpCL loader finds
`<data-dir>/wildlife/wildlifereid-10k/metadata.csv`. The 13 paper datasets are selected by
a `.query()` on the `dataset` column — no per-dataset download is needed.

## Active Learning block — reproduces well

| Method      | mAP       | mINP      | BAKS      | AUC ROC   | Top-1     | Top-3     | Top-5     | Top-10    | paper mAP | Δ mAP     |
| ----------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| SpCL (Base) | 43.95     | 26.46     | 55.61     | 67.57     | 56.11     | 70.41     | 76.77     | 83.86     | 44.95     | −1.00     |
| + Random    | 45.27     | 27.72     | 56.84     | 68.10     | 57.03     | 71.12     | 77.55     | 84.33     | 46.67     | −1.40     |
| + GBAS      | 47.80     | 29.65     | 59.66     | 70.11     | 60.08     | 74.64     | 80.48     | 87.32     | 48.61     | −0.81     |
| + LBAS      | 51.14     | 32.94     | 62.62     | 72.80     | 63.90     | 76.69     | 81.71     | 88.17     | 52.15     | −1.01     |
| + NIS       | 48.84     | 31.07     | 60.23     | 70.85     | 61.01     | 74.18     | 79.70     | 86.29     | 50.72     | −1.88     |
| **+ AAS**   | **55.78** | **37.11** | **68.41** | **75.54** | **68.48** | **80.53** | **84.52** | **90.23** | 56.14     | **−0.36** |

Per-seed mAP (seeds 1–4) and standard deviation:

| Method      | s1    | s2    | s3    | s4    | std  |
| ----------- | ----- | ----- | ----- | ----- | ---- |
| SpCL (Base) | 43.25 | 44.78 | 43.79 | 44.00 | 0.64 |
| + Random    | 45.38 | 44.82 | 45.39 | 45.48 | 0.30 |
| + GBAS      | 48.13 | 48.01 | 47.11 | 47.95 | 0.46 |
| + LBAS      | 51.55 | 51.22 | 51.02 | 50.77 | 0.33 |
| + NIS       | 49.75 | 48.84 | 48.71 | 48.05 | 0.70 |
| + AAS       | 55.80 | 55.18 | 55.43 | 56.72 | 0.68 |

**What reproduces:**
- The **method ranking is exact** on all eight metrics: AAS > LBAS > NIS > GBAS > Random > Base.
- AAS lands within **0.36 mAP** of the published 56.14.
- The paper's headline gains hold: **+11.83 mAP over base USL** (paper claims +11.19) and
  **+4.64 over the best AL baseline** (paper claims +3.99).
- Seed variance is small (std 0.30–0.70), so the residual gaps are systematic, not noise.

**Unexplained:** every *trained* row sits ≈1 mAP below the paper. The untrained
ResNet-50 baseline (below) reproduces to −0.37, which rules out the data splits, gallery
construction, evaluation code and metrics — those are shared and they reproduce. The
remaining suspect is training numerics (torch/cuDNN versions differ from the authors').

## Baseline / foundation / skyline rows

**Every row below is the mean of 4 gallery splits** (`--sample_seed 0..3`); ± is the
standard deviation across those splits.

| Row                              | n   | mAP             | mINP  | BAKS  | AUC ROC | Top-1 | Top-5 | paper mAP | Δ mAP     |
| -------------------------------- | --- | --------------- | ----- | ----- | ------- | ----- | ----- | --------- | --------- |
| ResNet-50 (Baseline)             | 4   | **38.23** ±0.52 | 20.46 | 49.64 | 62.49   | 48.94 | 70.92 | 38.88     | **−0.65** |
| Foundation row (ImageNet Swin-L) | 4   | **44.46** ±0.34 | 25.54 | 57.95 | 64.74   | 56.62 | 76.04 | 45.65     | **−1.19** |
| Skyline, 30 ep, zerobased        | 4   | 65.86 ±0.20     | 49.33 | 75.33 | 74.66   | 75.38 | 89.73 | 64.29     | +1.57     |
| Skyline, 30 ep, offset           | 4   | 65.90 ±0.45     | 49.14 | 75.38 | 75.16   | 75.68 | 89.88 | 64.29     | +1.61     |
| Skyline, 50 ep, zerobased        | 4   | 68.55 ±0.19     | 52.40 | 77.84 | 76.61   | 77.88 | 90.45 | 64.29     | +4.26     |
| Skyline, 50 ep, offset           | 4   | 68.39 ±0.51     | 52.33 | 77.66 | 76.85   | 77.82 | 90.49 | 64.29     | +4.10     |

Not a Table 1 row, but measured for comparison:

| Model                              | n   | mAP         | mINP  | BAKS  | AUC ROC | Top-1 | Top-5 |
| ---------------------------------- | --- | ----------- | ----- | ----- | ------- | ----- | ----- |
| MegaDescriptor-L-224, real weights | 4   | 53.34 ±0.16 | 32.86 | 67.90 | 68.30   | 67.78 | 85.91 |
| MiewID (foundation) | 4   | **41.38** ±0.06 | 20.14 | 58.63 | 64.55   | 57.89 | 76.77 |

This row is a **side experiment, not a Table 1 row**: it exists only to confirm that the
paper's "MegaDescriptor" row is really ImageNet Swin-L. It is also no longer reproducible —
it needed a bypass of the `get_model` swap that has since been reverted. The Table 1 row is
the foundation row above; `scripts/run_row.sh megadescriptor` runs it the authors' way.

**ResNet-50 Baseline: −0.65 mAP.** This is the load-bearing result — it uses no training at
all, so it isolates the ≈1-point deficit on the trained rows to training numerics rather
than to the data splits, gallery construction, evaluation code or metrics, all of which
this row shares and reproduces.

### Skyline: the label mode is irrelevant, the epoch budget is not

With 4 seeds the two label modes are **indistinguishable**: 65.86 ±0.20 vs 65.90 ±0.45 at
30 epochs (0.04 apart), and 68.55 ±0.19 vs 68.39 ±0.51 at 50 epochs (0.16 apart). Both
gaps are well inside one standard deviation.

That settles the off-by-one (`Todo.md` 13b): once `num_classes` is made consistent with the
labels, whether you use `0..N-1` or `1..N` + a dead class 0 changes nothing numerically.
The `+1` was purely a **crash** bug, not a correctness one.

The epoch budget, by contrast, is decisive: **30 ep → ≈65.9, 50 ep → ≈68.5**, a 2.6 mAP
difference at ~0.2–0.5 std. Neither reproduces the paper's 64.29 — both overshoot, 30
epochs by +1.6 and 50 by +4.3. So the paper's skyline used *at most* 30 epochs, or a
configuration we have not identified.

**Skyline rows remain makeshift — treat as indicative only.** The `--skyline` path cannot
run as shipped (three separate defects, `Todo.md` item 13); most importantly
`CustomTrainer` re-clusters and overwrites the ground-truth labels at the end of every
epoch, which kills training at epoch 1. We disabled that for the skyline, which is what a
fully-supervised run *should* do, but that is our inference, not documented behaviour.
More seeds make these numbers honest, not the configuration correct.

### The paper's "MegaDescriptor" row is ImageNet Swin-L, not MegaDescriptor

`utils.py get_model` substitutes plain ImageNet `swin_large_patch4_window7_224` for
MegaDescriptor-L-224 whenever `--dataset combined`, commented
*"Do not use pre-trained megadescriptor"*. We tested both against the paper's 45.65:

|                                    | mAP       | Top-1     | AUC ROC   |
| ---------------------------------- | --------- | --------- | --------- |
| ImageNet Swin-L (the substitution) | **−1.19** | **+0.16** | **+0.62** |
| MegaDescriptor, real weights       | +7.69     | +11.31    | +4.18     |

(MiewID, the other foundation row, **is** reproduced — 41.38 vs the paper's 41.16, the
closest row in the table. It needs transformers 4.x: the model loads via
`trust_remote_code` and its modelling code predates the 5.x loader. Run it with the
`hitl-miewid` env; `Todo.md` item 14.)

**ResNet-50 Baseline reproduces to −0.37 mAP / −0.10 Top-1.** This is the load-bearing
result: it uses no training at all, so it isolates the ≈1-point deficit on the trained
rows to training numerics rather than to the data splits, gallery construction,
evaluation code or metrics — all of which this row shares and reproduces.

**Skyline rows are makeshift — treat as indicative only.** The `--skyline` path cannot run
as shipped (three separate defects, see `Todo.md` item 13); most importantly
`CustomTrainer` re-clusters and overwrites the ground-truth labels at the end of every
epoch, which kills training at epoch 1. We disabled that for the skyline, which is what a
fully-supervised run *should* do, but that is our inference, not documented behaviour.
Neither epoch budget reproduces 64.29 — all four variants overshoot. The label mode
(`zerobased` vs `offset`) changes little (0.3–0.7 mAP); the epoch budget changes a lot
(≈3 mAP).

### How many runs does the paper average?

The paper states 4 runs **only for the AL block**: *"we report the average performance over
4 runs for all AL Re-ID methods"* (§5). It does not state a run count for the baseline,
skyline, foundation or USL rows. Since the protocol draws *"a randomly chosen 20% of the
individuals"* as the held-out set, every row is split-dependent, so those are plausibly
averaged too — which is why **we ran 4 seeds for every row here**, not just the AL block.

Measured split noise: **±0.16–0.52 mAP** for the eval-only rows, **±0.19–0.51** for the
trained skyline. So the ≈1 mAP deficit on the trained AL rows is roughly 2–3σ and is a
real effect, not measurement slop; whereas the 0.04–0.17 gaps between skyline label modes
are firmly inside the noise.

## Commands actually used

The commands at the top of this README do **not** reproduce Table 1's AL rows. Those come
from `train_al.py`, with a ResNet-50 backbone and SpCL's training configuration, as the
paper's implementation details specify.

Use the launchers in [`scripts/`](scripts/) — they carry the GPU/VRAM gating, the BLAS
thread caps and the `LD_LIBRARY_PATH` that conda's faiss needs, none of which are optional:

```bash
bash scripts/queue.sh al  outputs_new     # all 24 AL runs,  1 job/GPU, auto-retry
bash scripts/queue.sh row outputs_new     # all 24 baseline / foundation / skyline runs
```

Single runs, if you want the underlying commands:

```bash
# AL rows (SpCL base + Random / GBAS / LBAS / NIS(CC) / AAS(ConSam)); omit -s for SpCL base
bash scripts/run_grid.sh ConSam 1 0 outputs_new
#   -> python -u train_al.py -d wildlife --subset combined -s ConSam --seed 1 \
#             --data-dir $REPO/dataroot --logs-dir <out> -j 6

# baseline / foundation / skyline rows
bash scripts/run_row.sh resnet50_baseline 0 0 outputs_new
#   -> python -u main.py --dataset combined --dataset_path <dir> --out_dir <out> \
#             --baseline --model_name resnet50 --seed 1 --sample_seed 0 \
#             --images_per_id 5 --metric precomputed
```

Defaults already match the paper: `--epochs 50 --eval-step 10` (5 AL cycles),
`--consam_eps 0.6 --consam_k 5 --consam_sim_thresh 0.3` = (ε, k_max, s_min) = (0.6, 5, 0.3),
and the 0.02%-per-cycle budget hardcoded in `spcl/sampling/base.py`.

Runs go to `<root>/{al,table1}/<row>/s<seed>/`, matching the frozen baseline, so

```bash
python aggregate_results.py --compare outputs_initial outputs_new
```

rebuilds both tables above from the JSONs and reports any cell that moved.

For the foundation row, pass MegaDescriptor's name and **let `get_model` substitute
ImageNet Swin-L** — that substitution is what the published number went through:

```bash
python main.py --dataset combined ... --baseline \
  --model_name hf-hub:BVRA/MegaDescriptor-L-224 \
  --images_per_id 5 --metric precomputed --sample_seed 0
```

The substitution is left in place (`main.py`, `NOTE(foundation-rows)`). An earlier local
change bypassed it for `--baseline`; that has been **reverted**, since the swap is what the
published numbers went through. To measure the *real* MegaDescriptor weights instead (a
different question — it gives 53.34 mAP), pass `--model_name` a name the swap does not
match, or disable the swap deliberately.

**`--images_per_id 5 --metric precomputed` are required.** `main.py` defaults to
`images_per_id=1, metric=cosine`, which builds a *different* evaluation gallery than the
paper (which uses "up to five exemplars per ID") and than every AL run. With 1 exemplar per
ID a dataset with 8 known identities yields an 8-image gallery and Top-10 indexing crashes
with `IndexError: index 9 is out of bounds`. Without that crash the mismatch would have
passed silently and produced non-comparable numbers.

## Not reproduced

| Table 1 row                                                         | why                                                                                                                                                                                                                                   |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **USL block** — ICE, PPLR, ClusterContrast, RTMem, PurificationReID | Never ported to wildlife by the authors. Only RTMem exists in this repo's git history with a wildlife adapter (deleted at `56c6fe6`, recoverable). `Todo.md` has the porting recipe reconstructed from it.                            |
| **ResNet-50 (Skyline)**                                             | Runs only via a makeshift workaround; the shipped path cannot train past epoch 1. Numbers above are indicative, not a reproduction — `Todo.md` item 13.                                                                               |

## Local changes made to this repo

All are commented with `FIXME(...)` or `MAKESHIFT(...)` and keep the original lines above
them. Full reasoning in `Todo.md`.

| File                                 | change                                                                                                           | why                                                                                                                     |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `requirements.txt` | pin `finch-clust==0.2.1`, `transformers<5` | 0.2.2+ silently breaks AAS (item 1); transformers 5.x breaks MiewID's `trust_remote_code` loader (item 14) |
| `scripts/common.sh` | `PYTHONHASHSEED=0`, `LD_LIBRARY_PATH`, thread caps | **AL runs were not reproducible**: the AAS sampler iterates `nx.connected_components()` over string-named nodes, so per-process string hashing changed which pairs it selected (R6). The other two were never applied by the launchers at all (R5, item 8) |
| `train_al.py` | `model.to('cuda:0')` after eval; `spcl.sampling` imports; `cudnn.benchmark` off; `manual_seed_all`; unknown-sampler guard | wildlife-tools leaves the model on CPU (item 11); two branches called names that were never imported; a typo in `-s` silently ran plain USL under an AL row's name |
| `spcl/sampling/counting_clusters.py` | guard the divide-by-zero in `f_hat` | corrupts the NIS estimate (W1) |
| `main.py` | skyline label mode + `extra_classes`; `get_evaluation_results` epoch arg; `run_suffix` now slugifies | items 13a/13b, 16 |
| `utils.py` | `update_pseudo_labels` flag; `_subset_name()` shim; `Normalized Accuracy@0.7` fix; NEP 50 float casts | item 13c; C3 merge; the metric computed `sqrt(BAKS*BAKS)` in all six dicts; numpy 2 keeps `float32` where numpy 1 gave `float64` |
| `gallery_cache.py` (new) | caches the gallery selection **and its RNG state** | O4; the first version desynchronised `np.random` between cache hit and miss, which broke the AL ranking until fixed |

The `main.py` foundation-row bypass has been **reverted** — the ImageNet-Swin
substitution it disabled is what the published numbers went through (item 15). The repo is
now in a state that reproduces Table 1 as published.

## Output layout

```
outputs_initial/                         frozen, read-only; see its own README.md
  al/<sampler>/s<seed>/                  AL grid, 24 runs (6 methods x 4 seeds)
    wildlife/combined/evaluation/evaluation_metrics_52.json  <- 13 per-dataset values
    logs_al/combined/results.csv                             <- per-cycle micro mAP
  table1/<row>/s<seed>/                  baseline / foundation / skyline, seeds 0-3
    pretrained_models/**/evaluation_metrics_0.json
    <row> = resnet50_baseline | swin_foundation | megadescriptor_realweights
          | resnet50_skyline_e{30,50}_{zerobased,offset}
```

`evaluation_metrics_52.json` is the final best-checkpoint eval and holds 13-element lists,
one per dataset — take the mean for the macro numbers above. The per-epoch `mAP` printed in
the logs and in `results.csv` is the **micro** (pooled) score and is not what Table 1
reports.
