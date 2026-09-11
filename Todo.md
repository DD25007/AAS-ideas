# Todo — Library Modernization

Running record of every version constraint this codebase currently depends on, why it
exists, and what needs patching to lift it. Goal: get the whole stack to current
libraries so future work isn't pinned to a 2024-era environment.

Environment: conda env `alm` at `/storage/dd25007/conda_envs/alm` (Python 3.10,
torch 2.5.1+cu124, numpy 1.26.4, scipy 1.13.1, faiss-gpu 1.12.0 from conda).

Status legend: **BLOCKING** = code breaks without the pin. **LATENT** = works today,
breaks on upgrade. **ENV** = environment/packaging, no code change needed.

> **Paths changed on 2026-09-01.** The vendored `SpCL/` tree was absorbed into the repo and
> deleted (section R3 at the end of this file). Findings recorded below still cite the old
> paths, deliberately — they are the record of what was found where. Translate with:
>
> | cited as | now |
> |---|---|
> | `SpCL/spcl/…` | `spcl/…` |
> | `SpCL/sampling/…` | `spcl/sampling/…` |
> | `SpCL/examples/al_spcl_train_usl.py` | `train_al.py` |
> | `SpCL/spcl/utils/faiss_utils.py` | gone — use the repo-root `faiss_utils.py` |
> | `outputs*/…` | `outputs_initial/{al,table1}/<row>/s<seed>/` |
> | the real-MegaDescriptor runs | `outputs_initial/table1/megadescriptor_realweights/` |
>
> The row name `megadescriptor` now means Table 1's foundation row run **the authors' way**
> — `--model_name hf-hub:BVRA/MegaDescriptor-L-224`, which `get_model` substitutes to
> ImageNet Swin-L (item 15). The real-weights side experiment keeps the explicit
> `_realweights` suffix so the two can never be confused.
>
> The environment is now `hitl` (originally a clone of `alm`; `alm` was deleted 2026-09-05 --
> see R4).

---

## 1. finch-clust — pinned `==0.2.1` — DOWNGRADED from BLOCKING 2026-09-03

> **The code no longer requires the pin.** `spcl/utils/constrained_clustering.py` now
> inspects `clust_rank`'s signature and binds by keyword, handling both the <=0.2.1 form
> (`use_ann_above_samples, initial_rank, distance, verbose` -> 2 values) and the 0.2.2+ form
> (`initial_rank, metric, verbose, ann_threshold` -> 4 values). Verified two ways: FINCH
> returns an **identical partition** under the installed 0.2.1 after the change, and the
> keyword branch was checked against the actual 0.2.3 source, not just the note below.
>
> **The pin stays, for results rather than compatibility.** 0.2.3 adds `use_faiss=True`, an
> `ann_threshold` default of 10,000 (this code passes 70,000) and a `random_state`
> parameter — each can change the partition, and every AAS number with it. Lifting the pin
> is now a one-line requirements change plus a parity run, not a code change.

**Symptom:** `TypeError: len() of unsized object`, swallowed by a bare `except` into
`pdb.set_trace()` at `SpCL/sampling/consistency.py:426`. AAS silently contributes zero
constraints and the run hangs at a `(Pdb)` prompt.

**Cause:** `finch-clust` 0.2.2+ changed `clust_rank`'s signature and return arity.
`SpCL/spcl/utils/constrained_clustering.py:75` calls it **positionally**:

```python
adj, orig_dist = clust_rank(data, use_ann_above_samples, initial_rank, distance, verbose)
```

| Repo intends (<=0.2.1)          | 0.2.3 binds                |
| ------------------------------- | -------------------------- |
| `mat`                           | `mat`                      |
| `use_ann_above_samples` (70000) | `initial_rank` = **70000** |
| `initial_rank` (ndarray)        | `metric`                   |
| `distance` ('cosine')           | `verbose`                  |
| `verbose`                       | `ann_threshold`            |

So `cols = initial_rank = 70000` (0-d) reaches `scipy.sparse.csr_matrix`, which calls
`len()` on it. Return arity also changed 2 -> 4
(`A, orig_dist` vs `adj, orig_dist, initial_rank, knn_index`).

**Signatures**
- `0.2.1`: `clust_rank(mat, use_ann_above_samples, initial_rank=None, distance='cosine', verbose=False)` -> `(A, orig_dist)`
- `0.2.3`: `clust_rank(mat, initial_rank=None, metric='cosine', verbose=False, ann_threshold=10000, random_state=None, *, use_faiss=True, ...)` -> `(adj, orig_dist, initial_rank, knn_index)`

**Patch to unpin** — `SpCL/spcl/utils/constrained_clustering.py:75`, switch to keyword
args and 4-value unpack:

```python
adj, orig_dist, _, _ = clust_rank(
    data,
    initial_rank=initial_rank,
    metric=distance,
    verbose=verbose,
    ann_threshold=use_ann_above_samples)
```

Also re-check the `ensure_early_exit` block below it: when `initial_rank` is supplied,
0.2.x sets `orig_dist = np.array([])`, so `orig_dist.shape[-1] > 2` must stay guarded.

Newer finch also wants `pynndescent` installed for n > `ann_threshold` (currently warns
`pynndescent is not installed`); with `use_ann_above_samples=70000` and n=20,565 the
exact path is used, so it's inert today but will matter on larger runs.

**Before removing the pin, verify:** FINCH must return a 1-D `(n,)` int label array.
Quick check used during diagnosis: 5,000 random unit vectors -> 1,258 clusters.

---

## 2/3/4/10. numpy, scipy, faiss, Python — ONE coupled upgrade, path established 2026-09-03

> These were filed as four independent items. They are not: **faiss's conda metadata is what
> actually pins numpy**, and everything else follows from it.
>
> ### The repo code no longer blocks any of them
>
> | check | result |
> |---|---|
> | numpy 2.0 removed aliases (`np.bool` etc.) | **none** — the last one, `ranking.py:11`, was fixed in R10 |
> | numpy 2.0 changed APIs (`in1d`, `row_stack`, `trapz`, `ptp`, `msort`) | **none** |
> | scipy 1.14 rewritten APIs | **none live** — every `sp.csr_matrix` use in the repo is commented out; the only real consumer is inside finch-clust |
> | finch-clust 0.2.2+ signature | handled (item 1, version-adaptive) |
>
> So items 2 and 3 are **not code problems at all**. Item 3's pin was described as
> "defensive"; it is in fact unnecessary from our side.
>
> ### What actually pins numpy: the installed faiss build
>
> ```
> faiss-gpu 1.12.0  depends:  numpy >=1.26.4,<2.0a0
>                             python >=3.10,<3.11.0a0
> ```
>
> That is a hard requirement, not a precaution. numpy 2 is unreachable while this faiss build
> is installed, regardless of what the repo code supports.
>
> ### The upgrade path (from real package metadata, not guesswork)
>
> `faiss 1.14.1` on `pkgs/main` ships a **Python 3.10** build that accepts numpy 2 —
> so item 10 (Python 3.12) is **not** on the critical path, contrary to the earlier framing:
>
> ```
> faiss 1.14.1 py310cuda129h6b6c4e1_101_cuda
>   numpy  >=1.21,<3   and   numpy >=2.0
>   python >=3.10,<3.11.0a0          <- stays on 3.10
>   cuda-version >=12.9,<13          <- the real blocker
> ```
>
> The chain is therefore: **numpy 2 -> faiss >=1.14 -> CUDA 12.9 -> a cu129 torch build**.
> The current stack is torch 2.5.1+cu124, so the upgrade is a torch/CUDA move, not a Python
> move. (The `pytorch`-channel faiss 1.14.3/1.15.0 builds *are* py3.12 + cuda13.2 — taking
> those would drag Python along, which is what made this look like a Python problem.)
>
> ### Sequencing, and one dependency to check first
>
> 1. Confirm `faiss.bruteForceKnn` still exists in 1.14 — it is deprecated in 1.12 and
>    `faiss_utils.py:88` still calls it. If it is gone, the `bfKnn` migration analysed in R10
>    (which needs the `||query||^2` term re-added per row) becomes a **prerequisite**, not an
>    optional cleanup.
> 2. Build a `hitl-modern` env: cu129 torch + faiss 1.14.1 + numpy 2 + scipy latest.
> 3. Run the eval-only parity gate. It is exact and cheap, and it exercises faiss through
>    `compute_jaccard_distance`.
> 4. Only then relax the pins in `requirements.txt`.
>
> ### 2026-09-03: a fresh env solves cleanly — numpy 2 is reachable on Python 3.10
>
> ```
> conda create -n hitl-modern python=3.10
> conda install -n hitl-modern -c pkgs/main "faiss-gpu=1.14.1" "numpy>=2"
> ```
>
> Result: **python 3.10.21, numpy 2.2.5, faiss 1.14.1** (`py310cuda129..._100_cuda`), and
> faiss GPU verified working — `GpuIndexFlatL2.search` matches a brute-force reference.
>
> **Item 10 (Python 3.12) is off the critical path, confirmed by installation** rather than
> by reading metadata.
>
> #### Correction: the OpenMP conflict was self-inflicted
>
> An earlier note here claimed the upgrade was structurally impossible because faiss needs
> `_openmp_mutex *_intel` while `libgcc` pins `*_gnu`. **That was wrong.** It came from
> explicitly pinning the `_101` build, which does require Intel OpenMP. The metapackage in
> a fresh solve picks the **`_100`** build, which declares `numpy >=1.21,<3` and runs
> happily on `_openmp_mutex 52_gnu` — the same GNU variant the current env has.
>
> What genuinely fails is upgrading the *clone*: `hitl` carries accumulated constraints
> (mkl-vs-openblas, a `python=3.10` pin) that make the solve unsatisfiable. So "fresh env,
> not in-place" is still the right conclusion — but for ordinary constraint-accumulation
> reasons, not an irreconcilable OpenMP split.
>
> #### `faiss.bruteForceKnn` survives in 1.14.1
>
> ```
> bruteForceKnn: True      bfKnn: True      cast_integer_to_idx_t_ptr: True
> ```
>
> So the `bfKnn` migration analysed in R10 stays **optional**, not a prerequisite. The
> authors' `cast_integer_to_idx_t_ptr` fix also still applies.
>
> ### RESULT 2026-09-04: numpy 2 runs, but it does **not** preserve the numbers
>
> Two runs of `resnet50_baseline` seed 0 on identical code, cache disabled, one per env:
>
> | metric | frozen | numpy 1 (control) | numpy 2 |
> |---|---|---|---|
> | mAP | 38.5061 | **38.5061** | 38.5938 (**+0.09**) |
> | BAKS | 48.2794 | **48.2794** | 48.7798 (**+0.50**) |
> | AUC ROC | 62.7212 | 62.7188 | 62.2869 (**-0.43**) |
> | Top-1 | 48.2666 | **48.2666** | 48.7168 (**+0.45**) |
> | Top-3 | 64.4814 | **64.4814** | 64.1135 (**-0.37**) |
>
> **The numpy-1 control is bit-identical to the frozen baseline** on 6 of 8 metrics, with
> the other two showing the known O2/O6 batching residual (mINP +0.0013, AUC ROC -0.0024).
> So the NEP 50 casts and the Normalized-Accuracy fix changed **no** Table 1 number — the
> control run is what makes that a measurement rather than a hope.
>
> **numpy 2 shifts every metric**, up to 0.50 (BAKS) and 0.45 (Top-1). That is inside this
> row's seed spread (+/-0.52 mAP) but far outside the 0.0024 the code changes cost, so it is
> attributable to numpy alone. NEP 50 changes float32/float64 promotion throughout — the
> gallery selection's `distance.cdist`, the jaccard matrix and the metric aggregation all
> sit on that boundary.
>
> #### Conclusion for the modernization
>
> numpy 2 is **reachable and functional** but **not numerically transparent**. Lifting the
> pin is therefore a decision, not a refactor:
>
> - **Option A — accept and re-baseline.** Re-run all 48 rows under `hitl-modern` and freeze
>   a new `outputs_initial`. Honest, but it forfeits comparability with the published Table 1
>   at the 0.5-point level.
> - **Option B — hunt the promotion sites.** Force float64 at the specific boundaries
>   (`cdist` in the gallery selection, `compute_jaccard_distance`'s V matrix, the metric
>   reductions) until numpy 2 reproduces numpy 1. Bounded work, and the eval-only gate makes
>   each candidate site testable in ~10 minutes.
> - **Option C — stay on numpy 1.** The pin costs nothing today; faiss 1.12 already requires
>   it, and nothing in the repo needs numpy 2 features.
>
> Recommendation: **C for now, B when there is a reason to move.** The upgrade currently buys
> no capability and costs the reproduction's comparability with the paper. What this exercise
> did buy is certainty: the blockers are mapped, `requirements-modern.txt` and `hitl-modern`
> exist, and the remaining work is a measured 0.5-point numerical gap rather than an unknown.
>
> ### Remaining work

> ### (superseded) Attempted 2026-09-03: the upgrade cannot be done in place
>
> Cloning `hitl` and upgrading it fails with a precise, structural conflict:
>
> ```
> faiss-1.14.1-py310cuda129h6b6c4e1_101_cuda requires _openmp_mutex * *_intel,
> but none of the providers can be installed
> ```
>
> The current env carries `_openmp_mutex 5.1 52_gnu`, and **`libgcc` hard-requires
> `_openmp_mutex * *_gnu`**. The GNU and Intel OpenMP variants are mutually exclusive, and
> `libgcc` underpins everything compiled in the env — so no incremental upgrade path exists.
>
> **The modernization must be a fresh env solved from scratch**, letting conda pick the
> Intel OpenMP variant before anything is built against the GNU one. That also means it
> cannot inherit `hitl`'s pip-installed stack (torch cu124, wildlife-tools, timm,
> transformers) — those get reinstalled against the new base, which is precisely why the
> eval-only parity gate matters afterwards.
>
> Note this also invalidates the "clone the env" approach that worked so well for `hitl`
> (R4): that was right for holding the stack *constant*, and is exactly wrong for changing
> its foundation.
>
> Also still unresolved: whether `faiss.bruteForceKnn` survives in 1.14. The `faiss-cpu`
> wheel cannot answer it — `bruteForceKnn` and `bfKnn` are both GPU-only symbols and neither
> appears there. It has to be checked against a GPU build once one installs.
>
> Steps 2-4 otherwise need the GPUs, which the AL re-run currently occupies.

**`SpCL/spcl/evaluation_metrics/ranking.py:11`**

```python
mask = np.zeros(num, dtype=np.bool)
```

`np.bool` was removed in NumPy 2.0. One-word fix: `np.bool` -> `bool`.
This is the only remaining removed-alias use in the repo (checked `np.float`,
`np.int`, `np.object`, `np.str` — all clean).

Also: the conda faiss build and this 2024-era stack expect numpy 1.x ABI, so the
faiss upgrade (item 4) has to land before numpy 2.

---

## 3. scipy — pinned `<1.14` — LATENT / ENV

scipy's sparse internals were rewritten (`_coo.py` now iterates `self.coords` and calls
`len()` on each). That rewrite is what turned the finch signature bug into a confusing
`len() of unsized object` instead of a clean `TypeError`. No direct repo use of the
changed APIs found; the pin is defensive. Retest after item 1 is patched.

---

## 4. faiss — conda-only, deprecated API — ENV + LATENT

**Must come from conda, not pip.** Both `faiss-gpu-cu12==1.12.0` and `==1.14.1.post1`
install but abort on H200 with `CUDA error 209 no kernel image is available` — those
wheels carry no `sm_90` kernels. The conda `pytorch` channel build
(`faiss-gpu-1.12.0-py3.10_ha3bacd1_0_cuda12.4.0`) works.

**Deprecated calls to patch:**
- `faiss_utils.py:84` and `SpCL/spcl/utils/faiss_utils.py:84` —
  `faiss.bruteForceKnn(...)` prints `bruteForceKnn is deprecated; call bfKnn instead`.
- `faiss_utils.py:10,16` and `SpCL/spcl/utils/faiss_utils.py:10,16` —
  `x.storage().data_ptr() + x.storage_offset() * 4` triggers
  `UserWarning: TypedStorage is deprecated`. Modern form is
  `x.untyped_storage().data_ptr() + x.storage_offset() * x.element_size()`.

GPU faiss is not optional: `StandardGpuResources`, `bruteForceKnn`,
`cast_integer_to_float_ptr`, `GpuIndexFlatL2` are all used across `utils.py`,
`faiss_utils.py` and `SpCL/spcl/utils/faiss_rerank.py`. `faiss-cpu` will not work.

---

## 6. wildlife-tools — git install required — ENV

Installed from GitHub `main` (v1.0.3), not PyPI. The latest PyPI release (0.0.9) has no
`wildlife_tools.inference.TopkClassifier`, which `main.py:20` imports. This also drags
in `gluefactory` from git, because `wildlife_tools/similarity/__init__.py` imports
`MatchLightGlue` at package level — needed even though this repo only uses
`CosineSimilarity` from that module.

Revisit once a PyPI release carries `TopkClassifier`; that would drop the gluefactory
git dependency too.

---

## 7. wildlife-datasets — missing module — ENV

Only `metrics.BAKS` / `metrics.BAUS` are actually used. `wildlife-eval.py` originally
imported `wildlife_datasets.baseline.utils`, which ships in neither the wheel nor the
sdist (it lives in a `baselines/` scripts folder in the repo). Already removed — this
repo's own `utils.py` carries adapted copies (`mean` at :1202,
`compute_predictions` at :1210).

---

## 8. libstdc++ / pandas vs conda faiss — ENV

Importing `pandas` before `faiss` aborted with:

```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
             (required by .../libfaiss.so)
```

pip `pandas` loads the **system** libstdc++, which lacks `CXXABI_1.3.15`; conda's
`libfaiss.so` needs it. Fixed for this env only:

```
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib -n alm
```

Order-dependent, so it only appears in real import chains — `import faiss` alone is
fine, which makes it easy to miss. Should disappear once faiss comes from a wheel built
against a modern libstdc++.

---

## 9. user site-packages shadowing — ENV

`~/.local/lib/python3.10/site-packages` was ordered ahead of the env's own
site-packages, shadowing `seaborn`, `joblib`, `threadpoolctl`, `typing_extensions`,
`contourpy`. Classic source of phantom version errors that look like CUDA problems.

```
conda env config vars set PYTHONNOUSERSITE=1 -n alm
```

Apply to any new env created on this box.

---

## 10. Python 3.10 — ENV

Chosen because the conda `faiss-gpu` build that works on H200 targets py3.10. Lifting
this depends on item 4.

---

## Non-version code issues found (separate from library upgrades)

- **Live `pdb.set_trace()` traps** in `SpCL/sampling/consistency.py` at lines 254, 357,
  381, 426. Each sits in an `except` block whose intended
  `return self.ml_pairs, self.cl_pairs` is commented out directly beneath. Any sampler
  exception therefore hangs an unattended run instead of degrading. Same pattern at
  `sampling/fps.py:66`, `clustering_logs.py:119`, and `utils.py:542,547,664,670`.
- **Hardcoded foreign path** `SpCL/examples/al_spcl_train_usl.py:445` wrote to
  `/home/mehar21541/SpCL/logs_al/...` -> `PermissionError` at the first eval.
  **Patched** (originals left commented) to use `--logs-dir`.
- **`--data-dir` layout**: `get_data()` appends the dataset name, so the wildlife loader
  needs `<data-dir>/wildlife/wildlifereid-10k/`. Handled with a symlink, no code change.
- **Micro vs macro mAP**: the per-epoch `mAP` in the log and `results.csv` is the
  **micro** (pooled) score. Table 1 is the **macro** mean over the 13 datasets — read
  `evaluation_metrics_<epoch+1>.json`, which stores the per-dataset lists.
- **`MinMax` / `EC` samplers** are dispatched at `al_spcl_train_usl.py:149-152` but their
  imports are commented out at lines 36-37 -> `NameError` if selected. Not paper
  baselines, so inert.
- **Stale absolute paths**: `utils.py:42` (default `--dataset_path`),
  `dataset/wildlife.py:48`, `SpCL/spcl/datasets/wildlife.py:119` (the `WII` branch),
  `wildlife-eval.py:41` (`root_features`, never assigned anywhere).
- **wandb** is called unconditionally in `main.py` (lines 136, 370, 378) with a
  hardcoded entity. Use `WANDB_MODE=offline` to run without uploading.

---

## How the authors ran the USL baselines (from git history)

Recovered from the deleted `RTMem/` tree at commit `f8e2441` (restored to
`/storage/dd25007/baselines/RTMem`). RTMem is the **only** USL baseline the authors
ported, and it shows the pattern:

`RTMem/examples/RTMem_train_usl.py` imports **from SpCL**:

```python
from spcl.datasets.wildlife import WildlifeDatasets
from spcl.utils.wildlife_utils import get_evaluation_results
```

So the recipe per baseline repo is:

1. Copy `spcl/datasets/wildlife.py` into `<repo>/<pkg>/datasets/wildlife.py` and register
   it in the dataset factory — `'wildlife': Wildlife` (`RTMem/RTMem/datasets/__init__.py:18`).
2. Add `--subset` (default `combined`) to the arg parser.
3. At eval, build `WildlifeDatasets(dataset=args.subset, dataset_path=args.data_dir,
   dataset_split_seed=0, images_per_id=5, metric='precomputed')` and call SpCL's shared
   `get_evaluation_results(...)`. The evaluation harness is **reused**, not reimplemented —
   so all methods share one protocol and one metric suite.
4. Keep `--epochs 50`; `--eval-step` controls eval cadence.
5. Run with `PYTHONPATH` including the SpCL tree, since the baseline imports `spcl.*`.

**Blocker in the restored RTMem:** it calls
`get_evaluation_results(megadescriptor_extractor, wildlife_dataset, folder)` (3 args) at
lines 332 and 360, but SpCL's current signature is
`get_evaluation_results(megadescriptor_extractor, dataset, folder, epoch)` (4 args).
Needs the `epoch` argument added before it will run.

**Still to port** (stock upstream, person-Re-ID only, no wildlife support):
`ICE`, `PPLR`, `cluster-contrast-reid`, `Purification_ReID` — all in
`/storage/dd25007/baselines/`.

---

# Performance Optimizations

All applied and verified output-identical unless noted. Eval went 15m29s -> 1m26s (10.8x);
a full AL run 6h00m -> ~4h. The per-item write-ups were dropped 2026-09-08 (they are in git
history); what matters now is why the code looks the way it does.

| id | what it was | fix, and where it lives now |
|---|---|---|
| O1 | `num_workers=0` in the AL eval path | `--workers` is threaded through to the eval loader |
| O2 | query features extracted twice per eval | closed query is sliced out of query (`utils._subset_features`) |
| O3 | BLAS/OpenMP sized pools to all 224 cores; load hit 715 | thread caps in `scripts/common.sh` |
| O4 | gallery rebuilt (MegaDescriptor over the full set) on every process start | `gallery_cache.py`. **It must save and restore RNG state** — returning early on a cache hit skipped an `np.random.choice` and desynced every later draw, which silently broke AL ranking until it was caught |
| O6 | every eval image encoded twice, closed query four times | same `_subset_features` slicing |
| O7 | BAKS/BAUS swept 103 thresholds, 101 discarded | only `-inf` and `0.7` are evaluated; `HITL_FULL_THRESHOLD_SWEEP=1` restores the curve |
| O8 | 10,305 GPU->CPU syncs per `compute_jaccard_distance` | batched transfers in `spcl/utils/faiss_rerank.py`; 1.40x, bitwise identical |

The O4 lesson generalises: **a cache that short-circuits a code path must leave the RNG
where the uncached path would have left it.** The eval-only gate that O4 passed never
exercised the sampler's RNG stream, so the bug survived its own regression test.


Found while running the 16-run grid (2026-08-27). Ordered by impact.

## R11. Full comparison against the published Table 1, 2026-09-03

`aggregate_results.py --paper` now carries the paper's Table 1 (all 8 metrics, p.39188) and
diffs every reproduced row against it. Previously only a hand-copied "paper mAP" column in
the README tracked this, and only for mAP.

> **Superseded by the final run.** The deltas quoted in this section are from the
> 2026-09-03 grid. The authoritative figures are in "Reproduction status" above: the AL rows
> all moved up by roughly 0.3-0.8 mAP in `outputs_final`, and the "~1-point deficit on every
> trained row" this section describes no longer holds -- AAS reproduces to +0.07 and the
> mean deviation is +0.24. The method-by-method observations below still stand.

### The ~1-point deficit is real, and it is confined to mAP and mINP

Frozen reproduction, AL block, delta vs paper:

| row         | mAP   | mINP  | BAKS      | AUC ROC   | Top-1     | Top-3     |
| ----------- | ----- | ----- | --------- | --------- | --------- | --------- |
| SpCL (Base) | −1.00 | −0.91 | **+0.23** | **+0.56** | −0.54     | −0.63     |
| + Random    | −1.40 | −1.30 | −0.44     | −0.23     | −1.08     | −0.84     |
| + GBAS      | −0.81 | −0.70 | −0.87     | −1.39     | −0.48     | **+0.75** |
| + LBAS      | −1.01 | −1.12 | −1.00     | −0.35     | −0.09     | **+0.05** |
| + NIS       | −1.88 | −1.66 | −0.98     | −1.45     | −0.77     | −1.40     |
| + AAS       | −0.36 | −1.06 | **+1.26** | **+0.33** | **+0.77** | **+1.45** |

mAP and mINP are down ~1 on every row; BAKS, AUC ROC and Top-k are mixed and often
*above* the paper. That is diagnostic rather than cosmetic: **mAP and mINP integrate over
the whole ranked list, while Top-k and BAKS only see its head.** Ranking the correct match
slightly deeper — but still inside the top-k — costs mAP/mINP and leaves Top-k untouched.
That is the signature of marginally less-converged embeddings, consistent with the
training-numerics explanation, and inconsistent with an error in the data splits, the
gallery protocol or the metric code (which would move all eight together).

### New: the skyline's AUC ROC is anomalous in the opposite direction

Every skyline metric **overshoots** the paper (+1.3 to +6.5) except AUC ROC, which
**undershoots by 3.2–5.5** in both the frozen and the rerun trees. A consistent sign flip
on one metric is a behavioural difference, not noise.

Mechanistically this fits item 13: our skyline disables `CustomTrainer`'s per-epoch
re-clustering and trains purely supervised on ground-truth labels. That should sharpen
closed-set discrimination (Top-1 +5.0) while degrading the known-vs-unknown separation
AUC ROC measures — exactly the pattern observed. It is further evidence the skyline
configuration is not the paper's, and it gives item 13 a concrete signal to fix against.

### Rerun (optimised code) vs paper — complete rows only

| row         | frozen delta | rerun delta |
| ----------- | ------------ | ----------- |
| SpCL (Base) | −1.00        | **−0.71**   |
| + GBAS      | −0.81        | **−0.24**   |
| + Random    | −1.40        | **−2.52**   |

Base and GBAS land *closer* to the paper than the original reproduction; Random moves
further away. With AL run-to-run nondeterminism at ~1 mAP (R6) this is within the expected
envelope, but Random remains the watch item from R6/R8.

## Where an AL run's time actually goes (profiled, USL s1, 6h00m total)

| phase                        | time   | share | status                              |
| ---------------------------- | ------ | ----- | ----------------------------------- |
| training (50 ep x 400 iters) | 3.9 h  | 65%   | GPU-bound (data wait 2%); untouched |
| jaccard + clustering, x50    | ~1.5 h | 25%   | **O8** cuts ~10.7 min               |
| threshold sweep, 6 evals     | 19 min | 5%    | **O7** cuts ~19 min                 |
| eval extraction, 6 evals     | 14 min | 4%    | **O1/O2/O4/O6** cut ~12 min         |

Cumulative eval cost per run: **15m29s -> 1m26s (10.8x)**. Training now dominates almost
everything that remains, and it is genuinely GPU-bound -- the DataLoader reports 0.034s of
data wait against 1.707s of step time (2%), so more workers or faster I/O would buy
nothing. Reducing it further means AMP, a larger batch or `torch.compile`, all of which
change numerics and would need their own parity argument.

## O5. Hardcoded `device='cuda:0'` in eval — WON'T FIX (harmless under the launchers)

> Every launcher sets `CUDA_VISIBLE_DEVICES=$GPU`, so `cuda:0` always resolves to the
> one GPU the process can see. It would only matter for a run that hand-picks a device
> without that env var, which nothing here does. Left alone rather than churn the
> numerics path for no behavioural gain.

`al_spcl_train_usl.py:427,463`. Correct under `CUDA_VISIBLE_DEVICES=N` (device 0 is the
assigned GPU), but breaks any multi-GPU or explicit-device usage. Prefer the same device
the model is already on.

---

# Cleanup

## Closed items — index

The full write-ups were removed 2026-09-08; they are in git history. Kept here as one line
each so older cross-references still resolve.

| id | what it was |
|---|---|
| 5 | `pretrained=` is deprecated in torchvision, not timm — item 5 conflated the two |
| 11 | `wildlife_tools` left the model on CPU after every eval; fixed in `get_evaluation_results`, covering all six call sites |
| 12 | two `sampling` packages shadowed each other; resolved by the `spcl/` rename |
| 14 | transformers 5.x breaks MiewID's `trust_remote_code` loader — hence `hitl-miewid` |
| 16 | `run_suffix` whitelist raised on unknown backbones; now falls back to a slug |
| C1 | dead file `dataset/person.py` |
| C2 | `faiss_utils.py` was a byte-for-byte duplicate |
| C3 | `utils.py` vs `spcl/utils/wildlife_utils.py`, merged after an AST equivalence check |
| C5 | mixed line endings |
| C6 | standalone scripts on no code path |
| C7 | `pdb.set_trace()` traps in production paths |
| W1 | divide-by-zero corrupting the NIS estimate — in **both** `counting_clusters.py` copies, plus a second occurrence in the confidence-interval branch |
| W3 | `DtypeWarning` on every metadata read; `low_memory=False`, verified identical dtypes and values |
| W4 | `pynndescent not installed` — non-issue |
| R1 | the reproduction frozen in `outputs_initial/` |
| R3 | `SpCL/` absorbed into the tree and deleted |
| R5 | `LD_LIBRARY_PATH` was missing from the launchers; now set in `scripts/common.sh` |
| R7 | cleanup pass closing C1/C2/C5/C7 |
| R8 | `aggregate_results.py --compare` had three defects that made its gate untrustworthy |
| R9 | dead-code and unused-file sweep, incl. `evaluation.py` 280 -> 72 lines |
| R10 | deprecated-API sweep: `np.bool`, `random.sample(<set>)`, `TypedStorage`. The one open piece is below |

## `faiss.bruteForceKnn` -> `bfKnn` — DO NOT migrate naively (analysed, deferred)

`faiss_utils.py:88`. Deprecated, and it prints on every call (~176 times per run), but the
two APIs are **not** interchangeable and a rename would silently corrupt clustering.

Measured on 2000x128 vs 500x128, k=20:

```
indices   bitwise identical : True
distances bitwise identical : False
D1 (bruteForceKnn) row0[:5] : 36.56, 52.58, 53.37, 54.63, 56.67
D2 (bfKnn)         row0[:5] : 138.96, 154.98, 155.77, 157.03, 159.07
```

The difference is a **constant per-row offset** (102.40 for row 0) -- `bfKnn` omits the
`||query||^2` term. Ranking is therefore unaffected, which is why the indices match, but
`compute_jaccard_distance` compares distances *across* rows to build the k-reciprocal
neighbourhoods, so dropping that term would change the pseudo-labels and every metric
downstream.

A correct migration must add `||xq||^2` back per row (and re-verify bitwise against the
legacy call). Signature for reference: `bfKnn(res, args)` with a `GpuDistanceParams`
carrying `metric, k, dims, vectors, vectorsRowMajor, numVectors, queries,
queriesRowMajor, numQueries, outDistances, outIndices`. Left as-is: it is a printed
warning with no functional impact, and this sits directly on the clustering path.


## C4. `sampling/` root vs spcl — WON'T MERGE, but they must be kept in sync

**Updated 2026-09-05 — the divergence has now cost us a real bug.** The two packages are
genuinely different code (`consistency.py` alone differs by 1,638 lines), so merging them
stays off the table. But they implement the *same samplers*, and fixes have been landing in
only one copy:

| fix                                      | `spcl/sampling/` (used by `train_al.py`) | `sampling/` (used by `main.py`) |
| ---------------------------------------- | ---------------------------------------- | ------------------------------- |
| W1 divide-by-zero in `counting_clusters` | fixed 2026-09-02                         | **missed until 2026-09-05**     |
| W1's second occurrence (`w_ci`)          | missed until 2026-09-05                  | missed until 2026-09-05         |
| R6 `HITL_SORT_COMPONENTS`                | added 2026-09-05                         | ported same day, 6 sites        |

Both are now in sync. The audit that found this is cheap and worth repeating after any
sampler change: compare the repair markers (`FIXME(...)`, `Todo.md <id>`, `HITL_*`) present
in each copy and investigate any that appear on one side only.

### Original finding (2026-09-04): they are not duplicates

> Measured with the same AST comparison that made C3 tractable, and it gives the opposite
> answer:
>
> | pair | identical / shared defs |
> |---|---|
> | `consistency.py` | **0 / 3** (1253 lines vs 427) |
> | `counting_clusters.py` | 0 / 4 |
> | `random.py` | 1 / 4 |
> | `balanced.py` | 2 / 7 |
> | `dataset/wildlife.py` vs `spcl/datasets/wildlife.py` | 2 / 4 |
>
> They have **incompatible constructors** (`(database, n, gallery_size)` vs `args=...`),
> different base classes (`WildlifeDatasets` vs `Wildlife(BaseImageDataset)`), different
> consumers (`main.py` vs `train_al.py`), and root's `ConsistencySampling` carries methods
> the spcl one lacks entirely (`get_sampling_pools`, `select2`,
> `top_n_high_entropy_outliers`).
>
> Merging is therefore a **redesign**, not a de-duplication: it would mean unifying two call
> contracts and reconciling genuinely different `select()` implementations, where the spcl
> version is the one that produced the published Table 1. Large risk, no correctness gain.
>
> **Done instead:** module headers on all four files stating what each is, who consumes it,
> the measured divergence, and that they once shadowed each other and silently killed a
> 16-run batch (item 12). The shadowing itself was fixed by the C3-era rename.
>
> **Left open deliberately:** root `sampling/` is reachable only via `main.py --sampler`, a
> workflow the reproduction never runs -- every Table 1 AL row comes from `train_al.py`. It
> is live but unexercised code. Deleting it would remove a real entry point, so that is a
> product decision rather than a cleanup.

> **The danger is gone, the duplication is not.** The SpCL copy is now `spcl/sampling/`,
> so neither package can shadow the other (item 12). Both copies still exist and still
> diverge; merging them is still open. The policy quoted below (keep `SpCL/` pristine)
> was **superseded** on 2026-09-01 — see R3.

Normalized differing lines (root vs `SpCL/sampling/`):

| file                   | root | spcl | difflines |
| ---------------------- | ---- | ---- | --------- |
| `balanced.py`          | 251  | 225  | 78        |
| `base.py`              | 82   | 99   | 27        |
| `counting_clusters.py` | 145  | 117  | 42        |
| `random.py`            | 77   | 71   | 28        |
| `consistency.py`       | 1094 | 427  | 1443      |

**Policy decision (user, 2026-08-27): `SpCL/` is a different upstream project
(yxgeee/SpCL) and its files must be kept intact.** Any extra functions, or changes to its
existing functions, are to be **overridden from our own code** rather than edited in
place. So do *not* "consolidate" by rewriting files under `SpCL/`.

This changes the shape of the fix for C2/C3/C4. Instead of merging, the direction is:

- Keep `SpCL/` as a pristine vendored upstream so it can be re-synced later.
- Put our divergences in our own modules at the repo root, importing from `spcl.*` and
  subclassing / wrapping / monkey-patching where behaviour must differ.
- Where our root copy is the richer one (e.g. `sampling/consistency.py`, 1094 lines vs
  427), that root copy is the canonical implementation and the SpCL-side copy should
  ideally be *imported from* it rather than maintained separately.

**Caveats to sort out before acting:**

1. `SpCL/` here is **not** a git submodule and has no nested `.git` — all 61 files are
   tracked directly by the parent repo. So "keep upstream intact" is currently a
   convention, not something git enforces. Making it a real submodule or a vendored
   pin (recording the upstream commit) would make the boundary enforceable.
2. This fork is **already modified** relative to upstream by the paper's authors:
   `SpCL/spcl/datasets/wildlife.py`, `SpCL/spcl/utils/wildlife_utils.py`,
   `SpCL/examples/al_spcl_train_usl.py` and all of `SpCL/sampling/` are additions that do
   not exist upstream. So the boundary has to be drawn as "upstream SpCL + the authors'
   wildlife/AL layer", not "pristine upstream".
3. There is currently **one live local modification** to that tree:
   `SpCL/examples/al_spcl_train_usl.py` (the hardcoded `/home/mehar21541/...` output path
   at line 445, which raises `PermissionError` on the first eval). Under this policy that
   should eventually move out of the SpCL file — e.g. by making the log directory a
   parameter our launcher supplies, rather than an edit in place. Left as-is for now
   because nothing can run without it.

# Exactly what the authors changed in the vendored `SpCL/`

Established by diffing the repo's `SpCL/` at HEAD (`9cd9991`) against a fresh clone of
upstream **https://github.com/yxgeee/SpCL**. Upstream has 42 files; this copy has 61.
This is the boundary referred to in C4.

## A. Files ADDED by the authors (do not exist upstream) — 4,173 lines of new code

| path                                   | lines | what it is                                                                |
| -------------------------------------- | ----- | ------------------------------------------------------------------------- |
| `spcl/utils/wildlife_utils.py`         | 1625  | metric suite + NP3 constrained clustering + `get_evaluation_results`      |
| `spcl/datasets/wildlife.py`            | 581   | `Wildlife` / `WildlifeDatasets`, the open-set gallery-from-train protocol |
| `examples/al_spcl_train_usl.py`        | 534   | the AL training loop — the entry point for every Table 1 AL row           |
| `sampling/consistency.py`              | 427   | **AAS itself** (`ConsistencySampling`)                                    |
| `spcl/evaluation.py`                   | 258   | eval helpers                                                              |
| `spcl/utils/constrained_clustering.py` | 236   | FINCH wrapper + `constrained_clustering`                                  |
| `sampling/balanced.py`                 | 225   | LBAS / GBAS baselines                                                     |
| `sampling/counting_clusters.py`        | 117   | NIS baseline (`CountingClusters`)                                         |
| `sampling/base.py`                     | 99    | `BaseSampling`, incl. the hardcoded 0.02% budget                          |
| `sampling/random.py`                   | 71    | random-pair baseline                                                      |
| `sampling/__init__.py`                 | 0     |                                                                           |
| `logs_al/`, `logs_al_100/`             | 72K   | their own prior run results (`results.csv` per dataset)                   |

**Nothing was deleted** from upstream.

## B. Files MODIFIED from upstream — only 5, and 4 are trivial

**`spcl/datasets/__init__.py`** — registers the new dataset (2 lines):
```python
+from .wildlife import Wildlife
+    'wildlife': Wildlife,
```

**`spcl/utils/faiss_utils.py`** — a faiss API rename, newer faiss dropped the old symbol:
```python
-    return faiss.cast_integer_to_long_ptr(
+    return faiss.cast_integer_to_idx_t_ptr(
```

**`spcl/utils/serialization.py`** — torch 2.6+ compatibility:
```python
-        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
+        checkpoint = torch.load(fpath, map_location=torch.device('cpu'), weights_only=False)
```

**`setup.py`** — `install_requires` emptied (deps commented out), so `pip install -e SpCL`
no longer drags in `faiss_gpu==1.6.3` etc.

**`examples/spcl_train_usl.py`** — 50 differing lines. The upstream `Evaluator` is
commented out and replaced with the wildlife evaluation path; `--subset` added;
`print-freq` 10 -> 20.

## C. Consequences for the "keep SpCL intact" policy

1. The **entire method under test lives in the added files**, not in modified upstream
   ones. AAS, NP3, the wildlife protocol and the AL loop are all category A. So keeping
   "upstream intact" is cheap — only 5 upstream files were touched, and 4 of those are
   one-or-two-line compatibility fixes that upstream would likely accept anyway.
2. A clean split is therefore realistic: pin genuine upstream SpCL as a submodule, and
   carry categories A + B as our own overlay. The `faiss_utils`/`serialization`/`setup.py`
   fixes are the only real friction, and all three are version-compat patches that belong
   in the modernization work anyway (items 4 and 5 above).
3. Note the **root-level `faiss_utils.py` already contains the authors' `idx_t` fix** — it
   is byte-identical to the SpCL copy (C2), so both carry it.

## D. Bug found while diffing: `examples/spcl_train_usl.py` cannot run

It calls the 4-parameter function with 3 arguments, at lines **162, 268 and 295**:

```python
results = get_evaluation_results(megadescriptor_extractor, wildlife_dataset, folder)
```

but `spcl/utils/wildlife_utils.py:1359` defines

```python
def get_evaluation_results(megadescriptor_extractor, dataset, folder, epoch):
```

-> `TypeError: get_evaluation_results() missing 1 required positional argument: 'epoch'`
at the first eval. This is the **same stale-signature bug as the restored RTMem script**
(which calls it with 3 args at its lines 332/360), so `get_evaluation_results` gained the
`epoch` parameter at some point and only `al_spcl_train_usl.py` was updated.

Does **not** affect the current reproduction: the "SpCL (Base)" row is produced by running
`al_spcl_train_usl.py` with no `-s` flag, which uses the correct 4-argument call. But
`spcl_train_usl.py` is dead as shipped, and any USL baseline ported by copying its eval
block will inherit the bug.

Also note `spcl_train_usl.py` rebuilds `WildlifeDatasets(...)` at every eval (lines ~157,
~264, ~291) — the O4 waste — whereas `al_spcl_train_usl.py` correctly builds it once.

---

## W2. LBAS cannot fill its budget — MEASURED 2026-09-04 (and it verifies two paper claims)

> Counted across the full 24-run grid (4 seeds x 5 cycles = 20 opportunities per method):
>
> | sampler | "pool too small" | "budget unspent" |
> |---|---|---|
> | random / GBAS / NIS | 0 | 0 |
> | **LBAS** | **20 / 20** | 0 |
> | **AAS** | 0 | **20 / 20** |
>
> **LBAS hits an insufficient triplet pool in every cycle of every seed.** Its warning
> carries no count, so the shortfall cannot be quantified from the logs -- but a sampler that
> never fills its allowance is not being compared at the same budget as random/GBAS/NIS,
> which spend theirs in full. The paper does not report this. Quantifying it needs
> instrumentation in `spcl/sampling/balanced.py`.
>
> ### AAS: the paper's efficiency claim reproduces almost exactly
>
> AAS's message *does* carry the unspent count, so its true spend is recoverable. With
> n=20565 -> 211,449,330 possible pairs, budget 42,289/cycle (0.02%), cap 211,445 (0.1%):
>
> | seed | c1 | c2 | c3 | c4 | c5 | total |
> |---|---|---|---|---|---|---|
> | s1 | 42,284 | 10,348 | 7,493 | 4,781 | 2,856 | 67,762 |
> | s2 | 42,285 | 10,952 | 7,498 | 4,012 | 2,478 | 67,225 |
> | s3 | 42,285 | 10,640 | 10,912 | 4,276 | 2,678 | 70,791 |
> | s4 | 42,285 | 10,974 | 7,064 | 4,166 | 2,802 | 67,291 |
>
> - **Mean spend 68,267 pairs = 0.032% of all pairs.** The paper claims **0.033%**. Independently
>   reproduced from our own runs, not copied from the text.
> - **32.3% of the 0.1% cap** -- i.e. AAS reaches its accuracy while declining two thirds of
>   the annotation it is allowed.
> - **Spend falls monotonically across cycles** (42k -> 11k -> 7k -> 4k -> 3k) in every seed,
>   matching the paper's Fig. 3 observation that "the budget utilized by AAS in each AL cycle
>   reduces as the training progresses" -- pseudo-labels start noisy and need more
>   supervision.
>
> So the paper's efficiency claim holds up. The like-for-like concern applies to **LBAS**,
> not to AAS: AAS under-spends by design and reports it, LBAS under-spends silently.

Hit by `combined_LBAS_s1` and `combined_LBAS_s3` (twice each — i.e. in both AL cycles
those runs reached).

```python
if len(triplets) < self.n/2:
    print("Warning! Number of triplets in the sampling pool are not sufficient for the budget.")
```

LBAS mines triplets inside DBSCAN clusters; at `n = 20,565` with a 0.02%-per-cycle budget
(~42,000 pairs) its pool is too small to spend the allowance. This is **methodological,
not a crash**: LBAS is then being compared against other samplers at a budget it never
actually consumes, so a "same budget" comparison is not like-for-like. Worth measuring the
*utilised* budget per method per cycle rather than assuming the cap — which is exactly the
statistic the paper reports for AAS (0.033% of a 0.1% cap).

## 13. `--skyline` — makeshift, and a concrete cause found 2026-09-04

**HYPOTHESIS REFUTED 2026-09-05 (skyline experiment, `outputs_sky13`).** The theory was that
`BasicTrainer` accumulates gradients without scaling the loss (`loss.backward()` rather than
`(loss / accumulation_steps).backward()`), so the effective gradient is doubled and the
skyline overshoots the paper. `HITL_BASICTRAINER_ACCUM=1` restores the scaling. If the theory
held, mAP should have fallen toward the paper's 64.29. It rose:

| row   | variant   | mAP               | mINP  | AUC ROC | Top-1 |
| ----- | --------- | ----------------- | ----- | ------- | ----- |
| e30   | baseline  | 65.77             | 48.67 | 76.33   | 75.40 |
| e30   | `ACCUM=1` | **66.44** (+0.67) | 49.69 | 75.40   | 76.11 |
| e50   | baseline  | 68.71             | 52.30 | 77.64   | 78.22 |
| e50   | `ACCUM=1` | **69.16** (+0.45) | 52.97 | 77.76   | 78.04 |
| paper | —         | 64.29             | —     | 80.02   | —     |

So the missing loss scaling is **not** the cause of the overshoot, which remains +2.15 (e30)
and +4.87 (e50). One seed each, and the shifts (+0.67 / +0.45) are close to the ~0.5 seed
noise measured on the AL rows, so the honest reading is "no effect in the predicted
direction" rather than "a real increase".

Note the overshoot is **mAP-only and directional**: the skyline is *above* the paper on mAP
by 2-5 points while sitting *below* it on AUC ROC by 2-4.6. A uniformly stronger model would
move both the same way. That asymmetry, not the gradient scale, is where the remaining
explanation lives -- it points at the evaluation protocol or the label construction rather
than the optimiser.


> ### `CustomTrainer` does not scale the loss for gradient accumulation
>
> `CustomTrainer.train_epoch` (utils.py) diverges from the `BasicTrainer.train_epoch` it
> overrides, in three ways:
>
> | | BasicTrainer (wildlife-tools) | CustomTrainer (this repo) |
> |---|---|---|
> | loss | `loss_scaled = loss / accumulation_steps` then `.backward()` | `loss.backward()` — **unscaled** |
> | step condition | `(i + 1) % accumulation_steps == 0` | `(i - 1) % accumulation_steps == 0` |
> | epoch end | steps leftover gradients | **drops them** |
>
> The first is the substantive one: with `accumulation_steps=2` (what `main.py` passes), two
> unscaled backward passes accumulate before each step, so **the effective gradient is 2x
> what the same configuration would produce under `BasicTrainer`** — equivalent to doubling
> the learning rate.
>
> That is a plausible mechanical cause for the skyline **overshooting** the paper on every
> closed-set metric (+1.6 mAP at 30 epochs, +4.3 at 50, +5.0 Top-1) while **undershooting**
> AUC ROC by 3.2–5.5. Over-training sharpens known-identity separation and degrades the
> known-vs-unknown margin AUC ROC measures — the sign pattern matches.
>
> The step-condition difference happens to be harmless at `accumulation_steps=2`
> (both fire on odd `i`), but diverges for any other value — checked for 1..4.
>
> ### Why it is still not fixed
>
> Switching the skyline to `BasicTrainer`, or adding the loss scaling, would change the
> skyline numbers — so it needs its own parity run and a decision about which behaviour the
> paper's 64.29 mAP actually corresponds to. Given the skyline already overshoots, scaling
> the loss is the more likely match, and that is now a testable hypothesis rather than a
> guess.
>
> The original blocker stands as before: `CustomTrainer` re-clusters at the end of every
> epoch and overwrites the ground-truth labels, killing supervised training at epoch 1. The
> `update_pseudo_labels=False` guard works around it.

**Priority: the Table 1 "ResNet-50 (Skyline)" row (64.29% mAP) is currently NOT
reproducible. Treat any skyline number we produce as indicative only.**

Three independent defects sit on this path. The first two are patched; the third is
worked around and still needs a proper fix.

### 13a. Stale `get_evaluation_results` signature — PATCHED

`main.py:305` called it with 3 args; `utils.py:1367` needs 4 (`..., folder, epoch`).
`TypeError` at the first eval. Fourth site of this same bug — see also
`SpCL/examples/spcl_train_usl.py:162,268,295` and
`RTMem/examples/RTMem_train_usl.py:332,360`. Patched with `-1` to match the sibling
`--baseline` call.

### 13b. Off-by-one in the skyline labels — PATCHED (selectable)

`main.py` built `pseudo_identity` as `unique_ids.index(x) + 1` -> labels `1..N`, while
`SoftmaxLoss` gets `num_classes = N` from wildlife_tools' `pd.factorize` (`0..N-1`).
The last identity indexes past the classifier ->
`nll_loss_forward: Assertion 't >= 0 && t < n_classes' failed`.

Now selectable via `SKYLINE_LABEL_MODE`:
- `zerobased` (default) -> labels `0..N-1`, `num_classes = N`
- `offset` -> labels `1..N`, `num_classes = N+1` (keeps an unused class 0)

`get_trainer(..., extra_classes=)` keeps `num_classes` consistent with the mode. Both are
being run to see which reproduces the paper; if they agree within noise, the `+1` was a
latent bug that never fired in the authors' environment.

### 13c. Per-epoch re-clustering destroys the ground-truth labels — WORKED AROUND

**This is the real blocker and the one that needs a designed fix.**

`CustomTrainer.train_epoch` (`utils.py`, end of the method) re-extracts features over the
whole dataset at the end of **every** epoch, re-runs `constrained_clustering(method="ConSam")`,
and then does:

```python
self.pseudo_labels = pseudo_labels_mapped
self.dataset.metadata['identity'] = self.pseudo_labels
```

That is correct for the **AL** path -- refreshing pseudo-labels each epoch is the entire
point. It is wrong for **`--skyline`**, which is by definition fully supervised on 100% of
the ground-truth identities. After epoch 0 the real labels are thrown away and replaced by
cluster ids containing **-1 for outliers**, so epoch 1 feeds -1 into `nll_loss`:

```
Assertion `t >= 0 && t < n_classes` failed
```

The cluster count also need not equal `num_classes`, so the upper bound can break too.
Symptom profile: epoch 0 always completes cleanly, epoch 1 dies at batch 0. One variant
instead died with `cuDNN error: CUDNN_STATUS_EXECUTION_FAILED` -- same poisoned CUDA
context, different kernel in flight.

**Diagnostic trap worth remembering:** the `pseudo-labels-seed*.npy` dumped by
`EpochCheckpoint` looks perfectly clean (`min 0, max N-1`, no negatives). It is captured
from `epoch_data["pseudo_labels"]` at the *start* of `train_epoch`, i.e. the labels used
*during* the epoch -- not the re-clustered ones written at its end. Reading that dump as
proof the labels were valid sends you looking for a GPU fault that isn't there. The
confirming evidence is the `Computing feats to update PLs:` progress bar in the log, which
is the re-clustering block running.

**Current workaround:** `CustomTrainer(..., update_pseudo_labels=True)` added, with an
early `return epoch_data` before the re-clustering block when False. `main.py`'s skyline
branch passes `update_pseudo_labels=False`. Defaults keep every existing caller unchanged,
so the AL path is untouched.

**Why this is only makeshift:** it is *not* known how the authors produced the 64.29% mAP
skyline, and this path cannot have been run in its shipped state -- it dies at epoch 1
every time. Skipping the re-clustering is the behaviour a skyline *should* have, but that
is our inference, not something the paper or code documents. Open questions:
- Did they run the skyline through a different entry point entirely?
- Is the skyline meant to use `--epochs 30` (main.py default) or 50 (the USL/AL budget)?
  Both are being run to compare.
- Does the intended skyline use `identity` directly as the label rather than the
  `pseudo_identity` indirection, bypassing `CustomTrainer` altogether?

**Proper fix:** give the skyline its own trainer path that trains on the ground-truth
labels with no pseudo-label machinery, rather than reusing the AL trainer with a flag.
The `pseudo_identity` column, the `original_index` round-trip and the re-clustering are
all AL concepts that a fully-supervised run should never touch.

---

## 15. The MegaDescriptor swap is INTENDED — do not "fix" it

`utils.py:923` substitutes plain ImageNet `swin_large_patch4_window7_224` for
MegaDescriptor-L-224 whenever `--dataset combined`, commented *"Do not use pre-trained
megadescriptor"*. This looks like a bug that makes the zero-shot foundation row impossible
to produce. **It is not.** Measured over 4 gallery splits against the paper's 45.65 mAP row:

|                                    | mAP                     | Top-1             | AUC ROC           |
| ---------------------------------- | ----------------------- | ----------------- | ----------------- |
| ImageNet Swin-L (the substitution) | 44.46 ±0.34 → **−1.19** | 56.62 → **+0.16** | 64.74 → **+0.62** |
| MegaDescriptor, real weights       | 53.34 ±0.16 → +7.69     | 67.77 → +11.31    | 68.30 → +4.18     |

Top-1 is decisive: 56.62 vs 56.46 is a 0.16 gap, against 11.31 for the real weights, with
split noise only ±0.16–0.34. **Table 1's "MegaDescriptor" row measures ImageNet Swin-L.**

**Status: REVERTED.** `main.py` briefly carried a local change disabling the substitution
for `--baseline`; it was added before this was understood and has been reverted. The call
site now reads `combined=use_combined_dataset` again, with a `NOTE(foundation-rows)`
recording the measurement above so nobody "fixes" it a second time.

Implication for the modernization work: this substitution is load-bearing for the published
numbers, so it must not be "cleaned up" into always using the real weights.

---

## 17. NIS is the largest remaining gap: −2.03 mAP

With the `counting_clusters.py` divide-by-zero fixed (W1) and the full grid re-run,
`+ NIS` lands at **48.69 mAP vs the paper's 50.72 (−2.03)**. Every other AL row is between
+0.07 and −1.27, and AAS reproduces exactly, so this is now the largest single gap in the
table. Seed std is 0.54, so the extra point is comfortably outside noise. The corrupted
`f_hat` was therefore **not** the whole story for that row.

**Narrowed 2026-09-04.** Two candidates eliminated, one concrete lead found:

- **Not a budget problem.** Across all 4 seeds x 5 cycles, NIS emits **zero** "pool too
  small" and **zero** "budget unspent" warnings -- it spends its full 42,289 pairs/cycle,
  unlike LBAS (20/20 short) and AAS (deliberately 32%). So the shortfall is in *which* pairs
  it picks, not how many.
- **`--sampling_ratio` never reaches the sampler.** `train_al.py:228` hardcodes
  `CountingClusters(ratio=1.0, args=args)`, so the CLI argument exposed in `utils.py` is
  dead on the AL path. With `ratio=1.0`, `Nv = ceil(sqrt(n/ratio)) = 206` vertices x ~206
  neighbours -- a balanced split. If the NIS paper uses an unbalanced vertex/neighbour
  allocation, this is exactly where that difference would live, and it is currently
  unreachable without editing the source.

**Checked against the paper's own description of NIS 2026-09-05.** The AAS paper says NIS
"samples vertices with estimated low degrees, followed by sampling their most similar
neighbors". Our implementation matches that:

```python
n_hat[i] = np.sum(s_ij[i])                      # estimated degree
Q = (1 / n_hat) / np.sum(1 / n_hat)             # inversely proportional to degree  ✓
sampled_vertices  = np.random.choice(N, N_v, p=Q,  replace=True)
sampled_neighbors = np.random.choice(N, N_n-1, p=q, replace=True)
```

So the sampling *distribution* is right. Two deviations remain as candidates:

1. **`replace=True` on both draws** wastes ~1% of the budget on duplicates (measured: 206
   vertex draws yield ~205 unique, 205 neighbour draws ~204). Small, and it does not explain
   a 2-point gap on its own.
2. **`ratio` is hardcoded to 1.0** (`train_al.py:228`), forcing a balanced `Nv ≈ Nn ≈ √n`
   split. Whether Perez et al. specify a balanced allocation is the open question -- and
   `--sampling_ratio` cannot currently reach the AL path to test it.

Cheapest next experiment: unhardcode `ratio` and sweep it (e.g. 0.25 / 1 / 4) on one seed.
That is a one-line change plus three runs, and it directly tests the only structural
parameter that differs.

---

# Reproduction status (final, all runs complete)

All 52 runs from one `scripts/run.sh queue` invocation into `outputs_final/`, so every row
shares a consistently-derived gallery cache.

| Table 1 row                                | reproduced    | seeds | Δ mAP                     | note                                |
| ------------------------------------------ | ------------- | ----- | ------------------------- | ----------------------------------- |
| ResNet-50 (Baseline)                       | yes           | 4     | −0.62                     | untrained; isolates the shared path |
| ResNet-50 (Skyline)                        | **makeshift** | 4     | +1.3/+1.6 (30ep), +3.8/+4.3 (50ep) | item 13; overshoots at both budgets |
| MegaDescriptor                             | yes           | 4     | −1.16                     | as ImageNet Swin-L, item 15         |
| MiewID                                     | yes           | 4     | **+0.23**                 | needs transformers 4.x              |
| SpCL (Base)                                | yes           | 4     | −1.04                     |                                     |
| + Random                                   | yes           | 4     | −1.21                     |                                     |
| + GBAS                                     | yes           | 4     | −0.77                     |                                     |
| + LBAS                                     | yes           | 4     | −1.27                     | cannot spend its budget, W2         |
| + NIS                                      | yes           | 4     | −2.03                     | item 17, still unexplained          |
| + AAS                                      | yes           | 4     | **+0.07**                 | the paper's contribution            |
| USL block (ICE/PPLR/CC/RTMem/Purification) | **no**        | —     | —                         | never ported; recipe above          |

**Method ranking reproduces exactly, and AAS lands within 0.07 mAP.** Mean deviation across
the 13 rows is **+0.24 mAP**. The remaining gaps are per-method rather than a uniform
offset: AAS and MiewID reproduce, while the AL *baselines* sit 0.8–2.0 below. Since the
untrained ResNet-50 row is at −0.62 and every row shares the same splits, gallery, eval code
and metrics, the deficit is not in the shared pipeline.

Headline gains hold and are slightly exceeded: **+12.30 mAP over base USL** (paper claims
+11.19) and **+5.33 over the best AL baseline** (paper claims +3.99).

## Highest-value next steps

1. **NIS (item 17), −2.03 mAP** — now the largest single gap in the table, and the only AL
   row worse than −1.3. The cheapest untried experiment is unhardcoding `ratio` in
   `train_al.py` and sweeping it on one seed.
2. **Skyline (item 13)** — give it a real trainer path instead of the re-clustering
   workaround. The overshoot is mAP-only while AUC ROC sits 2.6–4.8 *below* the paper;
   gradient scaling has been ruled out as the cause.
3. **`HITL_SORT_COMPONENTS` A/B (R6)** — verify the code-level fix lands inside seed noise,
   then make it the default so reproducibility no longer depends on the launcher setting
   `PYTHONHASHSEED`.
4. **USL block** — port the 5 baselines using the RTMem recipe above; the largest missing
   block, and days of work.
5. **numpy 2 migration (items 2/3/4/10)** — `hitl-modern` runs everything but shifts metrics
   up to 0.50, so it needs a re-baseline or the promotion sites pinned to float64.

---

## R2. `aggregate_results.py` — the README tables are now generated

The published tables were assembled by hand, so nothing could check them. The new script
rebuilds them from the JSONs (macro-mean over 13 datasets, then over seeds) and diffs two
output trees:

```bash
python aggregate_results.py outputs_initial
python aggregate_results.py --compare outputs_initial outputs_verify --tol 0.0
```

It reproduced every published figure except three cells that were off by 0.01 from
last-digit rounding in the hand-assembled table — **README corrected**: skyline 50 ep
zerobased mAP 68.56→68.55 (68.5527) and AUC ROC 76.60→76.61 (76.6073), MegaDescriptor Top-1
67.77→67.78 (67.7751). The Δ mAP for that skyline row moves +4.27→+4.26.

## R4. Environments — `alm` deleted 2026-09-05; `hitl` is NOT redundant

**`alm` removed** (8.8 GB freed, /storage now 7.4T free). Nothing referenced it at runtime --
no live process, no launcher, no code path; the only mentions were prose here, a comment in
`requirements.txt`, and one stale line in `scripts/common.sh` (now updated). `hitl` is a
clone of it, so anything `alm` could do, `hitl` still does.

**Current envs**

| env           | size  | role                                                   | can it replace `hitl`? |
| ------------- | ----- | ------------------------------------------------------ | ---------------------- |
| `hitl`        | 8.8 G | numpy 1.x, transformers 4.x — **the reproduction env** | —                      |
| `hitl-modern` | 6.4 G | numpy 2.2.6, scipy 1.15.3, transformers 5.16.1         | **no**, see below      |
| `hitl-miewid` | 6.3 G | transformers 4.57.6, for the MiewID row                | no (single purpose)    |
| `cattle`      | 6.0 G | unrelated project                                      | n/a                    |

**`hitl-modern` does not make `hitl` redundant — two independent blockers:**

1. **It cannot run MiewID.** Reproduced directly, not inferred — loading
   `conservationxlabs/miewid-msv3` under transformers 5.16.1 raises
   `AttributeError: 'MiewIdNet' object has no attribute 'all_tied_weights_keys'`, exactly
   item 14. That row needs `transformers<5`, which is why `hitl-miewid` exists.
2. **numpy 2 moves the metrics.** Previously measured at up to 0.50 mAP, against a numpy-1
   control that was bit-identical to the frozen baseline. `outputs_initial` and
   `outputs_r6grid` are both numpy-1 results, so switching would break comparability with
   every number published so far.

`hitl-modern` is otherwise healthy (full stack imports, CUDA available), so it stays as the
forward-looking env for the eventual numpy-2 migration (items 2/3/4/10). Deleting `hitl`
should wait until that migration is actually made and the tables are regenerated under it.


`conda create --clone alm -n hitl`, plus the two `env config vars`. (`alm` was deleted on
2026-09-05 -- see the top of this section. `hitl` is a byte-for-byte clone, so the published
numbers stay re-runnable in an environment identical to the one that produced them.) Cloning
rather than reinstalling holds the package set constant, so the refactor is the only variable
in the parity check. Library modernization (items 1–10) gets its own env later.

---

## ~~R6. AL runs do not reproduce against themselves~~ — SOLVED 2026-09-04

> **Cause: `PYTHONHASHSEED`.** The AAS sampler builds networkx graphs with *string* node
> names (`f'dbscan-{i}'`, `f'finch-{j}'`, `spcl/sampling/consistency.py:105-107,133,146`) and
> iterates `nx.connected_components()` over them. Python randomises string hashing per
> process, so set-iteration order differs between runs, flipping which pair is selected at a
> boundary. That one-pair delta propagates through NP3 and compounds over 50 epochs into the
> ~1 mAP spread.
>
> Isolated: 3 processes gave 3 different component orders with a random seed, and identical
> order with `PYTHONHASHSEED=0`.
>
> **Fix:** `export PYTHONHASHSEED=0` in `scripts/common.sh`.
>
> **Code-level fix added 2026-09-05, default off: `HITL_SORT_COMPONENTS=1`.** The env-var fix
> only holds for callers who go through `scripts/common.sh`; run `train_al.py` by hand and the
> nondeterminism is back. `spcl/sampling/consistency.py` now has `_ordered()` /
> `_ordered_components()`, which sort graph nodes by `(prefix, int(idx))` -- numerically, so
> `dbscan-2` precedes `dbscan-10` -- at the three sites where a set of string-named nodes
> becomes an ordered sequence (`consistency.py:111,114-115,154`).
>
> It is **off by default on purpose**: sorting picks a different (equally valid) component
> order than `PYTHONHASHSEED=0` does, so switching it on would move every published AL number
> by roughly the one-pair boundary delta above. Flag-off is provably untouched -- substituting
> the flag-off definitions back in and AST-comparing against `HEAD` gives an exact match, so
> the default path is the original code, not a re-implementation of it.
>
> **Still to measure (needs GPU, blocked on the 24-run grid):** an A/B of
> `HITL_SORT_COMPONENTS=1` vs `PYTHONHASHSEED=0` on one ConSam seed. Expected to differ by
> ~1 pair in 42,285 (0.002%) and land inside seed noise; if it does, sorting should become the
> default and `PYTHONHASHSEED` drops back to being a belt-and-braces measure.
>
> **Ported to `sampling/consistency.py` (the root copy) the same day.** That file is what
> `main.py` imports and it has the same defect at **six** sites -- 145, 148-149, 203 in the
> first implementation and 701, 704-707, 812 in the second. Both copies AST-verify identical
> to their pre-patch selves with the flag off.
>
> **Verified** — two sequential ConSam runs, same GPU:
>
> | | before | after |
> |---|---|---|
> | `epoch0_mlcl` (constraint pairs) | 42285 vs 42284 | **42285 = 42285** |
> | `epoch0_postNP3` | `873806e8` vs `8845804e` | **`5c7a306e` = `5c7a306e`** |
> | clusters / un-clustered | 2288/4337 vs 2288/4320 | **2292/4327 = 2292/4327** |
>
> Consequences: AL runs are now reproducible, so a re-run is finally a valid parity check for
> them. Numbers shift relative to `outputs_initial`, which ran with random hash seeds -- but
> there was never a canonical value to preserve, since the old behaviour could not reproduce
> itself. Eval-only rows are unaffected.
>
> **Follow-up (not done):** the cleaner fix is to sort the component iteration in
> `consistency.py` so the sampler is deterministic regardless of environment. The env var
> covers any other string-set iteration too, so both are worth having.

`train_al.py:121-133` sets

```python
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
cudnn.deterministic = True
...
cudnn.benchmark = True      # <- 9 lines later, defeats the above
```

`benchmark = True` re-enables nondeterministic cuDNN algorithm selection, and
`torch.cuda.manual_seed_all` is never called at all (`set_seed` in `spcl/utils/wildlife_utils.py`
does call it, but the AL entry point does not use it). So **a trained AL run is not
bit-reproducible even against itself**, and no refactor can be validated by re-running one
and diffing.

Consequences:

- The parity gate for any future change has to be the *eval-only* rows, which are pure
  inference and do reproduce exactly, plus a byte-diff of the training code. Re-running an
  AL row can only ever confirm "within seed noise" (measured std 0.30–0.70 mAP).
- Worth fixing on its own merits: drop `benchmark = True`, add `torch.cuda.manual_seed_all`,
  and seed the DataLoader workers. Expect a throughput cost from losing cuDNN autotuning,
  which is presumably why it was set.

### Measured, 2026-09-01 — the nondeterminism is present before the first gradient step

An AAS seed-1 run under the restructured tree came out 1.01 mAP above the frozen run at the
epoch-10 eval (and −1.38 AUC ROC), which looked like a refactor regression. It is not.

Clustering statistics diverge at **epoch 0**, which runs on the freshly-loaded pretrained
model before any training. Two *identical* runs of the *same* code, same `--seed 1`, on
different GPUs:

| run                        | clusters @ epoch 0 | un-clustered | R_indep    |
| -------------------------- | ------------------ | ------------ | ---------- |
| frozen (pre-restructure)   | 2300               | 4323         | 0.3684     |
| verify (post-restructure)  | 2281               | 4341         | 0.3684     |
| probe A (post-restructure) | 2287               | 4327         | 0.3684     |
| probe B (post-restructure) | 2281               | 4353         | **0.3182** |

A vs B differ by as much as frozen vs verify — all four inside a ~19-cluster band, and probe
B even selected a different `R_indep` threshold. Last-bit float differences in the first
feature extraction shift DBSCAN/jaccard cluster boundaries, and a clustering-driven method
has no mechanism to wash that out; it compounds through the pseudo-labels every epoch.

**Consequences, and they are not small:**

1. **An AL row cannot be parity-checked by re-running it.** Not at 10 epochs, not at 50, not
   with more seeds — the run does not reproduce against *itself*. For AL code the only real
   guarantee is a byte-diff of the source (the Tier 1 check in R3).
2. Any future claim that a change "did not move the AL numbers" has to rest on the byte-diff
   or on a fixed evaluation of fixed checkpoints — never on a re-run comparison.
3. The eval-only rows are unaffected: pure inference, and they reproduce **exactly**
   (verified, zero delta on all 8 metrics x 4 seeds x 2 rows).
4. It also caps what the published +/-0.30-0.70 seed std actually measures — that is seed
   variation *plus* this, not seed variation alone.

### Attempted fix, 2026-09-03 — necessary but NOT sufficient

Applied to `train_al.py`:

- `torch.cuda.manual_seed_all(args.seed)` added. `torch.manual_seed` does not cover the
  CUDA RNGs, so a "seeded" run was never fully seeded.
- `cudnn.benchmark` now defaults **off** (`HITL_CUDNN_BENCHMARK=1` restores it). Measured
  cost of losing autotuning on this workload: **-0.3%** (2.38 -> 2.37 ms/step), i.e. the
  only argument for `benchmark=True` does not exist.

**It did not achieve reproducibility.** Two identical runs, same seed, same code, still
diverge at epoch 0: **2284 vs 2294 clusters** (pre-fix: 2287 vs 2281). Keep the change --
it removes a real source at no cost -- but the dominant source is elsewhere.

### What has been ruled out (measured, not reasoned)

| candidate                 | test                                                         | verdict                                                                |
| ------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------- |
| cuDNN algorithm selection | same model + input forwarded twice                           | **repeatable** — `torch.equal` True                                    |
| faiss GPU kNN             | two searches on identical structured features, N=10300, k=30 | **deterministic** — 0/309,000 indices *and* 0/309,000 distances differ |
| different GPUs            | `nvidia-smi`                                                 | all 8 are identical H200s                                              |

Two measurement traps hit while establishing this, both worth remembering:

1. A first pass on isotropic gaussian features reported **85% of neighbours differing**.
   That is an artifact: in 2048-d, random gaussians have all pairwise distances
   concentrated, so the 30-NN are near-ties and arbitrary. Real embeddings are clustered
   and show 0%. **Never characterise a nearest-neighbour method on isotropic noise.**
2. A throwaway comparison reported `max|dD| = 1.88` — the *identical* value on two
   unrelated datasets, which is the tell for a broken comparison rather than a finding.
   A clean re-measure gave 0.

### Root-cause hunt, 2026-09-03 — one real bug found (self-inflicted), one still open

Split the search space by replicating `train_al.py`'s memory-init in a standalone probe and
running it as two separate processes on the **same** GPU:

| stage                                                               | cross-process result                                             |
| ------------------------------------------------------------------- | ---------------------------------------------------------------- |
| feature extraction (20565 x 2048, 6-worker DataLoader, real images) | **bit-identical** — sha `c6cfa00c…`                              |
| `compute_jaccard_distance`                                          | **bit-identical** — sha `e2ed164a…`                              |
| DBSCAN(eps=0.6, precomputed)                                        | **bit-identical** — 817 clusters, 2706 outliers, sha `c1f92b7b…` |

So the whole epoch-0 chain is deterministic in isolation, which forced the question of what
differs in a real run — and turned up a bug **introduced by O4**.

#### BUG (fixed): the gallery cache desynchronised the RNG

`dataset/wildlife.py` seeds at line 48, then `get_representative_gallery_from_train` runs at
line 150 and consumes `np.random.choice` draws while selecting exemplars. **The O4 cache
path returned early, skipping those draws**, and nothing re-seeds afterwards. So a cache hit
left the global RNG in a different state than a cache miss, and every downstream consumer
diverged. Demonstrated:

```
compute path : next np.random draws = [0.50272652, 0.9025452, 0.2675786, ...]
cache path   : next np.random draws = [0.5488135,  0.71518937, 0.60276338, ...]   <- fresh seed(0)
```

Fixed by snapshotting `np.random` + `random` state after the selection and restoring it on a
hit; entries without a snapshot are treated as misses. Verified: both paths now yield
`[0.50272652, 0.9025452, ...]`.

**Consequence for results already produced.** The frozen baseline predates the cache, so it
always took the compute path. The rerun AL grid ran with the buggy cache, so RNG-sensitive
methods drew a different stream. This is the most likely explanation for the **Random row's
-1.12 mAP shift** flagged in R11 — `RandomSampling` is the method most directly driven by
RNG draws, and it was the one row whose four seeds all fell outside the frozen range.
**The AL block of `outputs_rerun` should be treated as suspect and re-run.**

Process note: the first attempt at this fix silently no-opped — `str.replace()` with
anchors written against the pre-black layout, and a success message printed without
asserting the anchors matched. Caught only because the verification probe still showed
diverging draws. Assert every anchor.

#### CONFIRMED 2026-09-04: the RNG desync was the cause, and the fix restores the ranking

Re-ran only the two RNG-driven rows (Random, NIS) x 4 seeds with the RNG-transparent cache;
the buggy versions are preserved in `outputs_rerun_al_v1/`.

| row      | v1 (buggy cache) | v2 (fixed) | frozen | recovery  |
| -------- | ---------------- | ---------- | ------ | --------- |
| + Random | 44.15            | **44.97**  | 45.27  | **+0.82** |
| + NIS    | 47.92            | **48.52**  | 48.84  | **+0.60** |

And the method ordering is exact again:

```
paper  : AAS > LBAS > NIS > GBAS > Random > Base
frozen : AAS > LBAS > NIS > GBAS > Random > Base
rerun  : AAS > LBAS > NIS > GBAS > Random > Base   <- restored (was ... > NIS > Base > Random)
```

Both inversions are gone, including Random falling below Base — which had contradicted the
paper's claim that random pair sampling helps at all. The prediction was specific (the two
most RNG-driven samplers would recover, the four insensitive ones were never touched) and it
held.

A residual of -0.30 (Random) and -0.32 (NIS) against frozen remains. That is ordinary AL
run-to-run nondeterminism, which R6 measures at ~1 mAP and has **not** yet been eliminated --
see "Still open" below. It is not evidence of a further cache problem.

**Lesson.** O4 passed its own parity gate and was reported as verified. The gate is eval-only
and never exercises the RNG stream the AL samplers consume, so a cache that silently
desynchronised `np.random` sailed through it. A gate proves what it covers, not what you hope
it covers -- and the coverage gap should be stated whenever the gate is cited.

### ROOT CAUSE FOUND 2026-09-04: the AAS sampler's pair selection

Located by checksumming the *real* pipeline stage by stage (`HITL_DEBUG_CHECKSUM=1`,
`_ckpt()` in `train_al.py`) across two sequential runs on one GPU:

| stage                                  | A vs B                 |
| -------------------------------------- | ---------------------- |
| `memory_features` (20565x2048)         | identical              |
| `epoch0_rerank` (jaccard)              | identical              |
| `epoch0_labels` (DBSCAN)               | identical              |
| `R_comp`, `R_indep`, `cluster_R_indep` | identical              |
| `indep_thres`                          | identical              |
| pseudo-labels entering NP3 (`preNP3`)  | **identical**          |
| **`len(ml_pairs) + len(cl_pairs)`**    | **42285 vs 42284**     |
| labels leaving NP3 (`postNP3`)         | differs (consequence)  |
| reported clusters                      | 2288/4337 vs 2288/4320 |

**Everything upstream is bit-reproducible.** The sampler returned one constraint pair fewer
in run B, and NP3 faithfully propagated that into different cluster assignments, which then
compounds over 50 epochs into the ~1 mAP spread seen at the end.

Every earlier hypothesis was wrong, and each was disproved by measurement rather than
argument: cuDNN algorithm selection (forward pass repeatable), faiss GPU kNN (0/309,000
neighbours differ on realistic features), cross-GPU variation (all 8 are identical H200s),
concurrency (sequential same-GPU still diverges), large CPU reductions (repeatable
in-process). Useful, but they were all upstream of the actual cause.

#### Candidate mechanisms inside `ConsistencySampling.select()`

Not yet isolated. In rough order of likelihood:

1. **FINCH** — reached via `constrained_clustering(..., FINCH)`. It falls back off
   `pynndescent`, which is not installed (W4), and that fallback path has never been checked
   for determinism.
2. **networkx set iteration** — AAS builds "regions of uncertainty" from connected
   components of a graph; iteration over `set`/`dict` intermediates can reorder medoid
   selection, and a single boundary flip is exactly a one-pair difference.
3. `np.random.choice(..., p=probs, replace=False)` over a pool assembled from sets.

A one-pair delta is the signature of a single boundary flip, not of accumulated float noise.

#### Why this matters beyond reproducibility

This sits in the paper's core contribution, not in scaffolding. It also predicts something
testable and cheap: **a USL run (no `-s`) should be fully deterministic**, because with no
sampler `ml_pairs`/`cl_pairs` stay empty and
`spcl/utils/constrained_clustering.py:96` early-returns `labels` unchanged. If a USL A/B is
bit-identical, that confirms the diagnosis and localises the fix to the samplers rather than
the training loop.

#### Prediction CONFIRMED 2026-09-04

The diagnosis predicted that a USL run (no `-s`) would be fully deterministic, since with no
sampler `ml_pairs`/`cl_pairs` stay empty and `constrained_clustering` early-returns. Two
sequential USL runs on one GPU:

```
USL    A: 736 clusters, 5498 un-clustered      B: 736 clusters, 5498 un-clustered   IDENTICAL
ConSam A: 2288 clusters, 4337 un-clustered     B: 2288 clusters, 4320 un-clustered  DIVERGES
```

Same code, same GPU, sequential. **The base SpCL pipeline is deterministic; the AL samplers
are not.** That localises any fix to `spcl/sampling/`, and it means the USL baseline row is
reproducible today while the five AL rows are not.

**Status: root-caused and confirmed, NOT fixed.** The specific mechanism inside
`ConsistencySampling.select()` (FINCH fallback / networkx set iteration /
`np.random.choice` over set-derived pools) is still un-isolated, and no fix has been
applied.

Instrumentation kept: `_ckpt()` in `train_al.py`, off unless `HITL_DEBUG_CHECKSUM=1`.
It found this after three wrong hypotheses and is the right tool for finishing the job.

#### Also found while narrowing

Two different `constrained_clustering` functions with incompatible signatures --
`utils.constrained_clustering(X, method, ...)` and
`spcl.utils.constrained_clustering.constrained_clustering(labels, rerank_dist, method, ...)`.
`train_al.py` uses the second. Calling the first with the second's argument order fails with
a confusing `ValueError: truth value of an array ... is ambiguous`. Worth renaming.

### Still open

The source of the epoch-0 divergence is **not yet isolated**. Remaining candidates, in
rough order of likelihood: the memory-init pass over the real dataset via the 6-worker
DataLoader (only a synthetic single forward has been tested, not the full `Preprocessor`
path); `HybridMemory` initialisation; FINCH (which falls back off `pynndescent`, W4);
load-dependent kernel selection across concurrently-busy GPUs.

Cheapest next probe: dump a checksum of the memory-init features in two runs on the *same*
GPU. If the checksums match, the divergence is downstream in clustering; if not, it is in
the extraction path. That splits the remaining search space in one experiment.


---
