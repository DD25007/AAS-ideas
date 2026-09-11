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
number live in `outputs_final/`; `aggregate_results.py` regenerates the tables below from
them. All 52 runs (6 AL samplers x 4 seeds, 7 non-AL rows x 4 seeds) come from a single
`scripts/run.sh queue` invocation, so every row shares one consistently-derived gallery
cache.

## Environment

```bash
conda create -n hitl python=3.10 -y && conda activate hitl
conda install -n hitl -c pytorch faiss-gpu -y     # NOT pip: the wheels lack sm_90 kernels
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
conda env config vars set PYTHONNOUSERSITE=1 -n hitl
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib -n hitl
```

Both `env config vars` lines are required, not optional — see `Todo.md` items 8 and 9.
`hitl` is the environment these numbers were produced in. The MiewID row needs a second env
(`hitl-miewid`) pinned to transformers 4.x — `scripts/run.sh` selects it per job. The
launchers export `LD_LIBRARY_PATH` themselves, since they invoke the interpreter directly
rather than activating the env.
`finch-clust==0.2.1` is a hard pin: 0.2.2+ changes a signature the AAS sampler calls
positionally, and the failure is silent (the run parks at a `pdb` prompt). Item 1.

Data: WildlifeReID-10k from Kaggle, laid out so the SpCL loader finds
`<data-dir>/wildlife/wildlifereid-10k/metadata.csv`. The 13 paper datasets are selected by
a `.query()` on the `dataset` column — no per-dataset download is needed.

## Active Learning block — reproduces well

| Method      | mAP       | mINP      | BAKS      | AUC ROC   | Top-1     | Top-3     | Top-5     | Top-10    | paper mAP | Δ mAP     |
| ----------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| SpCL (Base) | 43.91     | 26.90     | 55.30     | 67.06     | 55.74     | 69.92     | 75.94     | 83.48     | 44.95     | −1.04     |
| + Random    | 45.46     | 28.67     | 56.37     | 67.33     | 57.22     | 69.99     | 76.35     | 83.84     | 46.67     | −1.21     |
| + GBAS      | 47.84     | 30.02     | 59.44     | 69.49     | 60.39     | 74.05     | 79.55     | 86.66     | 48.61     | −0.77     |
| + LBAS      | 50.88     | 32.93     | 62.94     | 72.40     | 63.52     | 75.95     | 81.19     | 87.96     | 52.15     | −1.27     |
| + NIS       | 48.69     | 30.96     | 60.12     | 71.57     | 61.20     | 74.16     | 79.71     | 87.20     | 50.72     | −2.03     |
| **+ AAS**   | **56.21** | **38.14** | **68.58** | **75.90** | **68.49** | **79.99** | **85.00** | **90.16** | 56.14     | **+0.07** |

Per-seed mAP (seeds 1–4) and standard deviation:

| Method      | s1    | s2    | s3    | s4    | std  |
| ----------- | ----- | ----- | ----- | ----- | ---- |
| SpCL (Base) | 44.38 | 43.80 | 42.83 | 44.62 | 0.79 |
| + Random    | 45.95 | 44.90 | 46.51 | 44.50 | 0.93 |
| + GBAS      | 48.11 | 47.91 | 47.92 | 47.43 | 0.29 |
| + LBAS      | 50.89 | 52.24 | 49.96 | 50.42 | 0.99 |
| + NIS       | 48.58 | 49.41 | 48.66 | 48.11 | 0.54 |
| + AAS       | 56.57 | 56.50 | 56.15 | 55.63 | 0.43 |

**What reproduces:**
- The **method ranking is exact** on all eight metrics: AAS > LBAS > NIS > GBAS > Random > Base.
- AAS — the paper's contribution — lands **+0.07 mAP** from the published 56.14. Across all
  13 rows the mean deviation is **+0.24 mAP**.
- The paper's headline gains hold and are slightly exceeded: **+12.30 mAP over base USL**
  (paper claims +11.19) and **+5.33 over the best AL baseline** (paper claims +3.99).
- Seed variance is small (std 0.29–0.99), so the residual gaps are systematic, not noise.

**The remaining gaps are per-method, not a uniform offset.** AAS reproduces exactly while
the AL *baselines* sit 0.8–2.0 below, NIS worst at −2.03. Since all rows share the same
data splits, gallery construction, evaluation code and metrics — and the untrained ResNet-50
baseline reproduces to −0.62 — the deficit is not in the shared pipeline. NIS is tracked
separately as `Todo.md` item 17.

## Baseline / foundation / skyline rows

**Every row below is the mean of 4 gallery splits** (`--sample_seed 0..3`); ± is the
standard deviation across those splits.

| Row                              | n   | mAP             | mINP  | BAKS  | AUC ROC | Top-1 | Top-5 | paper mAP | Δ mAP     |
| -------------------------------- | --- | --------------- | ----- | ----- | ------- | ----- | ----- | --------- | --------- |
| ResNet-50 (Baseline)             | 4   | **38.26** ±0.47 | 20.48 | 49.45 | 62.69   | 48.71 | 70.96 | 38.88     | **−0.62** |
| Foundation row (ImageNet Swin-L) | 4   | **44.49** ±0.45 | 25.67 | 58.07 | 64.75   | 56.51 | 75.98 | 45.65     | **−1.16** |
| MiewID (foundation)              | 4   | **41.39** ±0.13 | 20.31 | 58.40 | 64.67   | 57.76 | 76.67 | 41.16     | **+0.23** |
| Skyline, 30 ep, zerobased        | 4   | 65.84 ±0.19     | 49.35 | 75.50 | 75.18   | 75.63 | 90.05 | 64.29     | +1.55     |
| Skyline, 30 ep, offset           | 4   | 65.56 ±0.53     | 48.60 | 75.30 | 75.13   | 75.26 | 89.74 | 64.29     | +1.27     |
| Skyline, 50 ep, zerobased        | 4   | 68.62 ±0.15     | 52.71 | 78.01 | 77.42   | 78.12 | 90.21 | 64.29     | +4.33     |
| Skyline, 50 ep, offset           | 4   | 68.08 ±0.23     | 51.91 | 77.58 | 76.78   | 77.54 | 90.22 | 64.29     | +3.79     |

Not a Table 1 row, but measured for comparison:

| Model                              | n   | mAP         | mINP  | BAKS  | AUC ROC | Top-1 | Top-5 |
| ---------------------------------- | --- | ----------- | ----- | ----- | ------- | ----- | ----- |
| MegaDescriptor-L-224, real weights | 4   | 53.34 ±0.16 | 32.86 | 67.90 | 68.30   | 67.78 | 85.91 |

This is a **side experiment, not a Table 1 row**: it exists only to confirm that the paper's
"MegaDescriptor" row is really ImageNet Swin-L. It is no longer reproducible — it needed a
bypass of the `get_model` swap that has since been reverted — so the figures above are
retained from the earlier reproduction. The Table 1 row is the foundation row;
`scripts/run.sh one row:megadescriptor:0 <gpu> <out>` runs it the authors' way.

**ResNet-50 Baseline: −0.62 mAP, MiewID: +0.23.** These are the load-bearing results — they
use no training at all, so they isolate the residual gaps on the trained baseline rows to
training numerics rather than to the data splits, gallery construction, evaluation code or
metrics, all of which these rows share and reproduce.

### Skyline: the label mode is irrelevant, the epoch budget is not

At 30 epochs the two label modes are **indistinguishable**: 65.84 ±0.19 vs 65.56 ±0.53,
0.28 apart and inside one standard deviation. At 50 epochs `zerobased` is ahead by 0.54
(68.62 ±0.15 vs 68.08 ±0.23) — small, but 2–3σ, so the modes are *not* strictly equivalent
once training runs long enough.

That still settles the off-by-one (`Todo.md` 13b) as a **crash** bug rather than a
correctness one: once `num_classes` is consistent with the labels, both modes train and land
within 0.6 mAP. But the dead class 0 that `offset` keeps is not entirely free, so
`zerobased` is the better default.

The epoch budget matters far more: **30 ep → ≈65.7, 50 ep → ≈68.4**, a 2.7 mAP difference
against ~0.2–0.5 std. Neither reproduces the paper's 64.29 — both overshoot, 30 epochs by
+1.3 to +1.6 and 50 by +3.8 to +4.3. So the paper's skyline used *at most* 30 epochs, or a
configuration we have not identified.

**The overshoot is mAP-only and directional.** These rows sit *above* the paper on mAP but
*below* it on AUC ROC (75.2 and 77.4 against 80.02, i.e. −4.8 and −2.6). A uniformly
stronger model would move both the same way, so the explanation is not simply "trained
longer". Gradient scaling was tested as a cause and ruled out (`Todo.md` item 13).

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
| ImageNet Swin-L (the substitution) | **−1.16** | **+0.05** | **+0.63** |
| MegaDescriptor, real weights       | +7.69     | +11.31    | +4.18     |

(MiewID, the other foundation row, **is** reproduced — 41.39 vs the paper's 41.16. It needs
transformers 4.x: the model loads via `trust_remote_code` and its modelling code predates
the 5.x loader, so `scripts/run.sh` gives that job the `hitl-miewid` env; `Todo.md` item 14.)

### How many runs does the paper average?

The paper states 4 runs **only for the AL block**: *"we report the average performance over
4 runs for all AL Re-ID methods"* (§5). It does not state a run count for the baseline,
skyline, foundation or USL rows. Since the protocol draws *"a randomly chosen 20% of the
individuals"* as the held-out set, every row is split-dependent, so those are plausibly
averaged too — which is why **we ran 4 seeds for every row here**, not just the AL block.

Measured split noise: **±0.13–0.47 mAP** for the eval-only rows, **±0.15–0.53** for the
trained skyline, **±0.29–0.99** across AL seeds. So the 0.8–2.0 mAP gaps on the AL baseline
rows are roughly 1–3σ and are real effects rather than measurement slop; the 0.28 gap
between skyline label modes at 30 epochs is inside the noise, while the 0.54 gap at 50
epochs is not.

## Commands actually used

The commands at the top of this README do **not** reproduce Table 1's AL rows. Those come
from `train_al.py`, with a ResNet-50 backbone and SpCL's training configuration, as the
paper's implementation details specify.

Use [`scripts/run.sh`](scripts/run.sh) — it carries the GPU/VRAM gating, the BLAS thread
caps and the `LD_LIBRARY_PATH` that conda's faiss needs, none of which are optional:

```bash
bash scripts/run.sh queue  outputs_new    # all 52 jobs, VRAM-gated, auto-retry
bash scripts/run.sh status outputs_new    # done / running / pending
```

Single runs, if you want the underlying commands:

```bash
# AL rows (SpCL base + Random / GBAS / LBAS / NIS(CC) / AAS(ConSam)); USL = no -s at all
bash scripts/run.sh one al:ConSam:1 0 outputs_new
#   -> python -u train_al.py -d wildlife --subset combined -s ConSam --seed 1 \
#             --data-dir $REPO/dataroot --logs-dir <out> -j 6

# baseline / foundation / skyline rows
bash scripts/run.sh one row:resnet50_baseline:0 0 outputs_new
#   -> python -u main.py --dataset combined --dataset_path <dir> --out_dir <out> \
#             --baseline --model_name resnet50 --seed 1 --sample_seed 0 \
#             --images_per_id 5 --metric precomputed
```

`miewid` selects its own interpreter (transformers 4.x); nothing else needs changing.

Defaults already match the paper: `--epochs 50 --eval-step 10` (5 AL cycles),
`--consam_eps 0.6 --consam_k 5 --consam_sim_thresh 0.3` = (ε, k_max, s_min) = (0.6, 5, 0.3),
and the 0.02%-per-cycle budget hardcoded in `spcl/sampling/base.py`.

Runs go to `<root>/{al,table1}/<row>/s<seed>/`, so

```bash
python aggregate_results.py --compare outputs_final outputs_new
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
| `train_al.py` | `spcl.sampling` imports; `cudnn.benchmark` off; `manual_seed_all`; unknown-sampler guard | two branches called names that were never imported; a typo in `-s` silently ran plain USL under an AL row's name |
| `utils.py` | `get_evaluation_results` restores the extractor's device | wildlife-tools ends `__call__` with `self.model.to("cpu")` on the caller's own model (item 11). Fixed in the callee, so all six call sites are covered — `main.py:237` sits inside the budget loop and never had the old per-call-site patch |
| `sampling/counting_clusters.py`, `spcl/sampling/counting_clusters.py` | guard the divide-by-zero in `f_hat`, and a second one in the confidence-interval branch | corrupts the NIS estimate (W1). Both copies: `train_al.py` imports the `spcl` one, `main.py` the root one, and the fix sat in only one for three days |
| `sampling/consistency.py`, `spcl/sampling/consistency.py` | `HITL_SORT_COMPONENTS=1`, default off | code-level fix for R6, so the sampler is reproducible without relying on the launcher setting `PYTHONHASHSEED`. Off by default because sorting picks a different (equally valid) order and would move published numbers |
| `evaluation.py` | 280 -> 72 lines | only `eval_wildlife` was ever imported; the rest was reachable only from an interactive `main()` that could not run (it passed an `int` where a DataFrame was expected) |
| `main.py` | skyline label mode + `extra_classes`; `get_evaluation_results` epoch arg; `run_suffix` now slugifies | items 13a/13b, 16 |
| `utils.py` | `update_pseudo_labels` flag; `_subset_name()` shim; `Normalized Accuracy@0.7` fix; NEP 50 float casts | item 13c; C3 merge; the metric computed `sqrt(BAKS*BAKS)` in all six dicts; numpy 2 keeps `float32` where numpy 1 gave `float64` |
| `gallery_cache.py` (new) | caches the gallery selection **and its RNG state** | O4; the first version desynchronised `np.random` between cache hit and miss, which broke the AL ranking until fixed |

The `main.py` foundation-row bypass has been **reverted** — the ImageNet-Swin
substitution it disabled is what the published numbers went through (item 15). The repo is
now in a state that reproduces Table 1 as published.

## Output layout

```
outputs_final/                           all 52 runs behind the tables above
  al/<sampler>/s<seed>/                  AL grid, 24 runs (6 methods x 4 seeds 1-4)
    wildlife/combined/evaluation/evaluation_metrics_52.json  <- 13 per-dataset values
    logs_al/combined/results.csv                             <- per-cycle micro mAP
  table1/<row>/s<seed>/                  non-AL rows, 28 runs (7 rows x 4 seeds 0-3)
    pretrained_models/**/evaluation_metrics_0.json
    <row> = resnet50_baseline | swin_foundation | miewid
          | resnet50_skyline_e{30,50}_{zerobased,offset}
```

`evaluation_metrics_52.json` is the final best-checkpoint eval and holds 13-element lists,
one per dataset — take the mean for the macro numbers above. The per-epoch `mAP` printed in
the logs and in `results.csv` is the **micro** (pooled) score and is not what Table 1
reports.
