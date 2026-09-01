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
> The environment is now `hitl` (a clone of `alm`); `alm` is preserved untouched.

---

## 1. finch-clust — pinned `==0.2.1` — BLOCKING

**Symptom:** `TypeError: len() of unsized object`, swallowed by a bare `except` into
`pdb.set_trace()` at `SpCL/sampling/consistency.py:426`. AAS silently contributes zero
constraints and the run hangs at a `(Pdb)` prompt.

**Cause:** `finch-clust` 0.2.2+ changed `clust_rank`'s signature and return arity.
`SpCL/spcl/utils/constrained_clustering.py:75` calls it **positionally**:

```python
adj, orig_dist = clust_rank(data, use_ann_above_samples, initial_rank, distance, verbose)
```

| Repo intends (<=0.2.1) | 0.2.3 binds |
|---|---|
| `mat` | `mat` |
| `use_ann_above_samples` (70000) | `initial_rank` = **70000** |
| `initial_rank` (ndarray) | `metric` |
| `distance` ('cosine') | `verbose` |
| `verbose` | `ann_threshold` |

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

## 2. numpy — pinned `<2` — LATENT

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

## 5. torchvision — deprecated `pretrained=` — LATENT

`pretrained=True` has been deprecated since torchvision 0.13 and is slated for removal.

- `SpCL/spcl/models/resnet.py:32` — `ResNet.__factory[depth](pretrained=pretrained)`
- `SpCL/spcl/models/resnet_ibn.py:21`

Modern form: `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)`.
Note `weights=DEFAULT` would pull **V2** ImageNet weights and change the ResNet-50
baseline/skyline numbers — pin `IMAGENET1K_V1` to stay comparable to the paper.

(`utils.py:933` and `SpCL/spcl/utils/wildlife_utils.py:928` also pass `pretrained=True`,
but those go to **timm**'s `create_model`, where the kwarg is still current. Leave them.)

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

## Dead code: `dataset/person.py`

`dataset/person.py:8` does `from random_erasing import RandomErasing`, but:

- no `random_erasing.py` exists in the repo, and it never did (checked all of git history)
- it is not an installed package
- nothing imports `dataset/person.py`

Importing the file therefore raises `ModuleNotFoundError`. It is dead code.

**Origin:** the file is a verbatim copy of `datasets/Market1501.py` from
[CoinCheung/triplet-reid-pytorch](https://github.com/CoinCheung/triplet-reid-pytorch)
(only differences: stripped shebang/encoding header, removed `torch`/`torchvision`
imports, missing trailing newline). The missing `RandomErasing` is that repo's top-level
`random_erasing.py` — Zhong et al., "Random Erasing Data Augmentation" (arXiv:1708.04896).

**Note:** the live training path does not use this. SpCL ships its own implementation at
`SpCL/spcl/utils/data/transforms.py:52`, used at `al_spcl_train_usl.py:67` as
`T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])`. So `dataset/person.py`
can be deleted or given the missing file; it affects nothing either way.

---

# Performance Optimizations

Found while running the 16-run grid (2026-08-27). Ordered by impact.

## O1. `num_workers=0` in the AL eval path — BIGGEST WIN

`SpCL/examples/al_spcl_train_usl.py:427` and `:463`:

```python
megadescriptor_extractor = DeepFeatures(model, device='cuda:0', num_workers=0, batch_size=32)
```

`num_workers=0` means **all** image loading, JPEG decode and resize for evaluation happen
in the main process, single-threaded, while the GPU idles. Measured on H200 with 16
concurrent runs: **25-40 seconds per batch of 32** (~0.8 s/image, vs ~10-20 ms expected).
A single eval covers query (7,626) + closed query (~4,600) + gallery (8,630) = ~20,900
images, so one eval pass runs for hours, and there are 5 evals per 50-epoch run.
**Evaluation, not training, dominates wall-clock.**

This is almost certainly an oversight, not a deliberate choice: the authors' own
non-AL sibling script `SpCL/examples/spcl_train_usl.py:157,263` uses the same call with
`num_workers=8`. `main.py` uses `args.num_workers`.

**Fix:** use `num_workers=args.workers` (the script already has `-j/--workers`).

## O2. Query features extracted twice per eval

`SpCL/spcl/utils/wildlife_utils.py:1384-1386`:

```python
default_closed_query_features = megadescriptor_extractor(dataset.default_closed_query)
default_query_features        = megadescriptor_extractor(dataset.default_query)
default_gallery_features      = megadescriptor_extractor(dataset.default_gallery)
```

`default_closed_query` is a strict **subset** of `default_query` — it is
`test_split[test_split['identity'].isin(identity_test_closed)]`
(`SpCL/spcl/datasets/wildlife.py:143-144,168`). So ~4,600 of the 7,626 query images are
encoded twice on every eval.

**Fix:** extract `default_query` once and index the closed-query rows out of it by
metadata index. Cuts roughly 22% off each eval pass. Combine with O1.

## O3. BLAS/OpenMP thread oversubscription

`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` are unset, so every process
sizes its thread pools to all **224** cores. Measured with 16 concurrent runs: hottest
process had **347 threads**, individual mains burning 8-12 cores, load average **715**
while total CPU use was only ~96 of 224 cores — i.e. cores idle, scheduler thrashing.
Worst during Jaccard distance and DBSCAN-on-precomputed over the 20,565^2 matrix.

**Fix:** export `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8` in the
launcher. Note this is *not* fixed by reducing processes per GPU — 8 processes x 347
threads still thrashes.

## O4. Gallery construction re-runs MegaDescriptor on every process start

`SpCL/spcl/datasets/wildlife.py:174` loads MegaDescriptor-L-224 and runs feature
extraction over the train split in `Wildlife.__init__`, purely to pick <=5 exemplars per
ID for the gallery (`get_representative_gallery_from_train`). This costs minutes at
startup for **every run**, and the result is fully deterministic given
`(subset, dataset_split_seed, images_per_id)` — it does not depend on the model being
trained.

**Fix:** cache the selected gallery/query/train split (just the metadata row indices) to
disk keyed by those three values, and reuse. Saves ~5 min x every run launched.

Note: this is *not* re-run at each eval in `al_spcl_train_usl.py` — that script builds
`dataset` once at line 135 and passes the same object to `get_evaluation_results`. It
**is** re-run per eval in RTMem's script, which constructs a fresh `WildlifeDatasets(...)`
at lines 325 and 353.

## O5. Hardcoded `device='cuda:0'` in eval

`al_spcl_train_usl.py:427,463`. Correct under `CUDA_VISIBLE_DEVICES=N` (device 0 is the
assigned GPU), but breaks any multi-GPU or explicit-device usage. Prefer the same device
the model is already on.

---

# Cleanup

## C1. Dead file: `dataset/person.py`

Covered in detail above — dangling `from random_erasing import RandomErasing`, imported
by nothing, copied from CoinCheung/triplet-reid-pytorch. Delete it, or add the missing
`random_erasing.py`. Nothing in the training path uses it.

## C2. `faiss_utils.py` is a byte-for-byte duplicate

Root `faiss_utils.py` (107 lines) and `SpCL/spcl/utils/faiss_utils.py` (108 lines) are
identical apart from a trailing newline. Keep one and import it.

## C3. `utils.py` is ~92% duplicated by `SpCL/spcl/utils/wildlife_utils.py`

Root `utils.py` is 1,748 lines; the SpCL copy is 1,625. After normalizing line endings
only **134 lines differ**, and **17 of 19** top-level functions share names —
`FINCH`, `constrained_clustering`, `CDBSCAN`, `_CDBSCAN`, `constrained_dbscan`,
`get_evaluation_results`, `compute_baks_baus`, `compute_predictions`, `auc_roc_new_class`,
`get_model`, `get_main_args`, `load_checkpoint`, `set_seed`, `predict`, `mean`,
`unfurl`, `unify_types`.

Two copies of the NP3 constrained-clustering implementation and the whole metric suite
means a bug fixed in one silently persists in the other. Consolidate into one module.

## C4. `sampling/` is duplicated between root and SpCL

Normalized differing lines (root vs `SpCL/sampling/`):

| file | root | spcl | difflines |
|---|---|---|---|
| `balanced.py` | 251 | 225 | 78 |
| `base.py` | 82 | 99 | 27 |
| `counting_clusters.py` | 145 | 117 | 42 |
| `random.py` | 77 | 71 | 28 |
| `consistency.py` | 1094 | 427 | 1443 |

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

## C5. Mixed line endings

### What "line endings" means

A text file marks the end of each line with an invisible control character. There are two
conventions:

- **LF** (`\n`, byte `0x0A`) — Unix, Linux, macOS. One byte per line break.
- **CRLF** (`\r\n`, bytes `0x0D 0x0A`) — Windows. Two bytes per line break.

The `\r` (carriage return) is a real character sitting at the end of every line. It is
invisible in most editors, which is what makes this easy to miss.

### What's wrong here

Every root-level Python file is **CRLF**; everything under `SpCL/` is **LF**:

| CRLF (root) | LF (SpCL) |
|---|---|
| `utils.py` (1748 lines) | all of `SpCL/` |
| `main.py` (503) | |
| `dataset/person.py` (72), `dataset/wildlife.py` (298) | |
| all of `sampling/*.py` | |

### Why it matters

1. **Every cross-tree diff is unreadable.** `diff` compares byte-for-byte, so a CRLF line
   never equals its identical LF twin. Comparing `utils.py` against
   `SpCL/spcl/utils/wildlife_utils.py` naively reports **3,374 differing lines** — i.e.
   "100% different". After stripping `\r` the true figure is **134 lines**. I hit exactly
   this during the C3 survey and initially mis-measured the duplication because of it.
2. **Noisy commits.** An editor that normalizes on save rewrites all 1,748 lines, so the
   diff shows the whole file changed and the one real edit is invisible in review.
3. **It can break execution.** A shebang line ending `\r` makes the kernel look for an
   interpreter literally named `python3\r`. Not currently a problem here (these files are
   invoked as `python file.py`), but it is a live hazard for any `.sh` added later.

### Fix

Normalize everything to LF and add a `.gitattributes` so it stays that way:

```
* text=auto eol=lf
*.py text eol=lf
*.sh text eol=lf
```

Then re-normalize the working tree once (`dos2unix`, or
`git add --renormalize .`). Worth doing **before** any C2/C3/C4 refactor — otherwise every
comparison between the root and SpCL trees is misleading.

## C6. Standalone scripts not on any code path

`generate_graph.py` (98), `generate_graphs.py` (121), `generate_plots.py` (150),
`wildlife-eval.py`, `main_train_acc.py` (719) are imported by nothing.

**Plan (user, 2026-08-27): merge the three plotting scripts into a single
`visualization.py`.**

All three read result JSON and emit matplotlib figures, so they share most of their
structure:

| script | lines | has `__main__` |
|---|---|---|
| `generate_graph.py` | 98 | yes |
| `generate_graphs.py` | 121 | no |
| `generate_plots.py` | 150 | no |

Suggested shape for `visualization.py`:

- one loader for the results JSON / `results.csv` layout, shared by all plot types
- one function per figure kind (per-cycle mAP curve, per-dataset breakdown, budget-vs-mAP)
- a small `argparse` entry point selecting which figure to emit, so the two scripts that
  currently have no `__main__` (and therefore run on import) become properly callable
- keep the existing output paths so nothing downstream breaks

Do this **after** the reproduction runs finish — these scripts consume the run outputs, so
it is easier to validate the merged version against figures produced from real results.

Remaining two, not part of the merge:

- `main_train_acc.py` (719 lines) differs substantially from `main.py` (1,222 differing
  lines); it appears to be a variant that additionally tracks train accuracy. Decide
  whether it is maintained or archived.
- `wildlife-eval.py` cannot run as-is — `root_features` is never assigned anywhere in the
  repo (see item 7 above). Either wire it up or drop it.

## C7. `pdb.set_trace()` traps in production paths

Already listed above. Every one sits in an `except` whose intended return is commented
out directly beneath, so any exception hangs an unattended run instead of failing or
degrading. `SpCL/sampling/consistency.py:254,357,381,426`, `sampling/fps.py:66`,
`clustering_logs.py:119`, `utils.py:542,547,664,670`. Highest priority of the cleanups —
this is what silently hung the first AAS run.

---

# Exactly what the authors changed in the vendored `SpCL/`

Established by diffing the repo's `SpCL/` at HEAD (`9cd9991`) against a fresh clone of
upstream **https://github.com/yxgeee/SpCL**. Upstream has 42 files; this copy has 61.
This is the boundary referred to in C4.

## A. Files ADDED by the authors (do not exist upstream) — 4,173 lines of new code

| path | lines | what it is |
|---|---|---|
| `spcl/utils/wildlife_utils.py` | 1625 | metric suite + NP3 constrained clustering + `get_evaluation_results` |
| `spcl/datasets/wildlife.py` | 581 | `Wildlife` / `WildlifeDatasets`, the open-set gallery-from-train protocol |
| `examples/al_spcl_train_usl.py` | 534 | the AL training loop — the entry point for every Table 1 AL row |
| `sampling/consistency.py` | 427 | **AAS itself** (`ConsistencySampling`) |
| `spcl/evaluation.py` | 258 | eval helpers |
| `spcl/utils/constrained_clustering.py` | 236 | FINCH wrapper + `constrained_clustering` |
| `sampling/balanced.py` | 225 | LBAS / GBAS baselines |
| `sampling/counting_clusters.py` | 117 | NIS baseline (`CountingClusters`) |
| `sampling/base.py` | 99 | `BaseSampling`, incl. the hardcoded 0.02% budget |
| `sampling/random.py` | 71 | random-pair baseline |
| `sampling/__init__.py` | 0 | |
| `logs_al/`, `logs_al_100/` | 72K | their own prior run results (`results.csv` per dataset) |

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

## 11. wildlife-tools — model left on CPU after every eval — BLOCKING (PATCHED)

**Symptom:** every run dies at the *first eval boundary* (epoch 10 with `--eval-step 10`),
after the eval itself has completed and written results:

```
RuntimeError: module must have its parameters and buffers on device cuda:0
              (device_ids[0]) but found one of them on device: cpu
```

**Cause:** `wildlife_tools`' feature extractor ends its `__call__` with

```python
self.model = self.model.to("cpu")
```

(`wildlife_tools/data/cache.py:84` in 1.0.3; `wildlife_tools/features/deep.py:61` in
1.0.1). `self.model` is the **same object** as the training script's `model`, so the model
is silently left on the CPU after every evaluation, and the next epoch's forward pass
fails inside `DataParallel`.

`al_spcl_train_usl.py` already had `model = model.to('cuda:0')`, but it runs *before*
`get_evaluation_results()`, so the extractor undoes it afterwards.

**Cannot be fixed by pinning** — unlike the finch bug there is no good version:

| version | has `.to("cpu")` | has `TopkClassifier` |
|---|---|---|
| 0.0.9 (PyPI latest) | no | **no** |
| 1.0.1 (only git tag) | yes | yes |
| 1.0.3 (git main, installed) | yes | yes |

`TopkClassifier` is imported at module level by `SpCL/spcl/datasets/wildlife.py:20`, so
0.0.9 fails at import. No version satisfies both constraints.

**Applied fix** (1 functional line + `FIXME(wrapper)` comment), after the eval call at
`SpCL/examples/al_spcl_train_usl.py:438`:

```python
results = get_evaluation_results(megadescriptor_extractor, dataset, folder, epoch)
model = model.to('cuda:0')
```

**Proper fix (TODO):** wrap `DeepFeatures` so it restores the model's own device, instead
of patching every call site. Reference implementation that was validated:

```python
_orig_call = DeepFeatures.__call__

def _call_preserving_device(self, dataset):
    try:
        device = next(self.model.parameters()).device
    except StopIteration:
        device = None
    try:
        return _orig_call(self, dataset)
    finally:
        if device is not None:
            self.model = self.model.to(device)

DeepFeatures.__call__ = _call_preserving_device
```

Verified: unpatched leaves the model on `cpu`, patched leaves it on `cuda`.

**Same bug is latent elsewhere** — `SpCL/examples/spcl_train_usl.py` and the restored
`RTMem/examples/RTMem_train_usl.py` call `DeepFeatures` identically and will hit it too.
This is one more reason to fix it in a wrapper rather than per call site.

The final-eval site (`al_spcl_train_usl.py:486`) was deliberately left unpatched: nothing
uses the model after it.

---

## 12. Two `sampling` packages shadow each other — BLOCKING (PATCHED)

A demonstrated instance of C4, not a theoretical one. `sampling/` exists **both** at the
repo root and under `SpCL/`, and the classes are *not* interchangeable:

- `SpCL/sampling/*` — constructors take `args=` (what `al_spcl_train_usl.py` passes)
- `sampling/*` (repo root) — different signatures entirely

Running the trainer from the repo root instead of from `SpCL/` puts the repo root first
on `sys.path`, so `import sampling` silently binds the **wrong** package and every
sampler fails at construction:

```
TypeError: ConsistencySampling.__init__() got an unexpected keyword argument 'args'
TypeError: RandomSampling.__init__()      got an unexpected keyword argument 'args'
TypeError: GBAS.__init__()                got an unexpected keyword argument 'args'
TypeError: LBAS.__init__()                got an unexpected keyword argument 'args'
TypeError: CountingClusters.__init__()    got an unexpected keyword argument 'args'
```

USL runs are unaffected (they never construct a sampler), so a whole grid can appear to
"start fine" while every AL row is dead. Cost this incident: a 16-run batch launched and
lost before anyone noticed.

**Mitigation:** always run the trainer with `SpCL/` as the working directory, so
`sys.path[0]` is `SpCL/examples` and `SpCL/` precedes the repo root.

**Also patched:** `al_spcl_train_usl.py:36` had `from sampling.assc import MinMax,
ExploreConsolidate` **uncommented** in the working tree (upstream has it commented). That
raises `ModuleNotFoundError: No module named 'sampling.assc'`, because `assc.py` and
`fps.py` exist only in the *root* `sampling/`, never in `SpCL/sampling/`. Re-commented,
with a `FIXME(cleanup)` note. The `MinMax`/`EC` samplers it feeds are unreachable anyway
and are not paper baselines.

**Proper fix (TODO):** merge the two `sampling` packages (C4), or import the root-only
samplers explicitly from the root package by path.

---

# Runtime warnings harvested from the crashed 16-run batch

Collected from all 16 `train.log` files of the run that died at the first eval boundary
(item 11), before that folder was deleted. Counts are across the 16 logs, each covering
10 epochs + one full evaluation.

| count | warning | where |
|---|---|---|
| 176 | `bruteForceKnn is deprecated; call bfKnn instead` | `faiss_utils.py:84` |
| 16 | `UserWarning: TypedStorage is deprecated` | `SpCL/spcl/utils/faiss_utils.py:10` |
| 16 | `UserWarning: 'pretrained' is deprecated since 0.13` | torchvision `_utils.py` |
| 16 | `UserWarning: Arguments other than a weight enum ... deprecated` | torchvision `_utils.py` |
| 16 | `DtypeWarning: Columns (2) have mixed types` | `SpCL/spcl/datasets/wildlife.py:61` |
| 4 | `Warning! Number of triplets ... not sufficient for the budget` | `SpCL/sampling/balanced.py:86` |
| 3 | `UserWarning: pynndescent not installed` | `finch/finch.py` |
| 2 | `RuntimeWarning: divide by zero encountered in scalar divide` | `SpCL/sampling/counting_clusters.py:100` |

The first four are already covered by items 4 and 5. The rest are new:

## W1. `counting_clusters.py:100` — divide-by-zero corrupts the NIS estimate (REAL BUG)

Hit by `combined_CC_s2` and `combined_CC_s3`.

```python
n_bar = sum_n_bar/len(sampled_neighbors[i])
try:
    sum_cc += (1/n_bar)*(1/Q[v_i])
except:
    sum_cc = 0
f_hat = sum_cc/len(sampled_vertices)
```

Two distinct defects:

1. **The `except` never fires for this case.** `n_bar` is a numpy scalar, so `1/0.0`
   yields `inf` and a `RuntimeWarning` — it does **not** raise `ZeroDivisionError`. So
   `sum_cc` silently becomes `inf`, and `f_hat` with it. NIS's population-size estimate
   (and therefore its `K`) is then garbage for that cycle.
2. **The handler is wrong even when it does fire.** `sum_cc = 0` *resets* the accumulator,
   discarding every term summed so far, rather than skipping the offending term.

Fix: guard explicitly, e.g. `if n_bar > 0 and Q[v_i] > 0:` before accumulating, and drop
the bare `except`. Note this affects the **NIS baseline row** of Table 1, so any NIS number
produced before this is fixed should be treated as suspect.

## W2. `balanced.py:86` — LBAS cannot fill its annotation budget

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

## W3. `wildlife.py:61` — `DtypeWarning` on every metadata read

```python
self.csv = pd.read_csv(os.path.join(self.dataset_dir, 'metadata.csv'))
```

Column 2 of `metadata.csv` has mixed types, so pandas falls back to slow per-chunk type
inference and warns. Harmless today but it makes the parse slower and the dtype
non-deterministic across pandas versions. Fix: pass an explicit `dtype=` for that column,
or `low_memory=False`.

## W4. `pynndescent not installed`

Emitted by finch. Inert at the current scale: `use_ann_above_samples=70000` and
n = 20,565, so the exact path is taken. It becomes a hard failure if a future run exceeds
that threshold — newer finch raises `MemoryError` telling you to install faiss or
pynndescent. Add `pynndescent` to requirements before scaling up. See also item 1.

---

## 13. `--skyline` cannot run as shipped — MAKESHIFT WORKAROUND IN PLACE

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

## 14. transformers 5.x breaks MiewID — BLOCKING, NOT FIXED

**The MiewID foundation row of Table 1 was not reproduced.**

`utils.py:929` loads MiewID via `AutoModel.from_pretrained(..., trust_remote_code=True)`,
so the modelling code comes from the HF repo `conservationxlabs/miewid-msv2` and targets
**transformers 4.x**. This env has **5.16.1**, whose loader calls `all_tied_weights_keys`
where the remote code defines the older `_tied_weights_keys`:

```
AttributeError: 'MiewIdNet' object has no attribute 'all_tied_weights_keys'.
                Did you mean: '_tied_weights_keys'?
```

`requirements.txt` pins only `transformers>=4.30.2`, which let 5.x in — the same class of
failure as item 1 (finch).

**Not attempted, deliberately:** the AL path also imports transformers (via wildlife-tools),
and AL jobs were in flight with a watchdog that could relaunch them onto a changed version.
Now that all runs are complete this is safe to do.

**Fix:** pin the latest transformers 4.x, verify BOTH
`AutoModel.from_pretrained('conservationxlabs/miewid-msv2', trust_remote_code=True)` **and**
`import spcl.datasets.wildlife, spcl.utils.wildlife_utils` still work, then run
`main.py --dataset combined --baseline --model_name conservationxlabs/miewid-msv2
--images_per_id 5 --metric precomputed --sample_seed 0..3`.

Note the code expects **msv2**; `msv3` exists on HF but is not handled — any other name
falls through to timm's `create_model`, which cannot take an HF repo id.

---

## 15. The MegaDescriptor swap is INTENDED — do not "fix" it

`utils.py:923` substitutes plain ImageNet `swin_large_patch4_window7_224` for
MegaDescriptor-L-224 whenever `--dataset combined`, commented *"Do not use pre-trained
megadescriptor"*. This looks like a bug that makes the zero-shot foundation row impossible
to produce. **It is not.** Measured over 4 gallery splits against the paper's 45.65 mAP row:

| | mAP | Top-1 | AUC ROC |
|---|---|---|---|
| ImageNet Swin-L (the substitution) | 44.46 ±0.34 → **−1.19** | 56.62 → **+0.16** | 64.74 → **+0.62** |
| MegaDescriptor, real weights | 53.34 ±0.16 → +7.69 | 67.77 → +11.31 | 68.30 → +4.18 |

Top-1 is decisive: 56.62 vs 56.46 is a 0.16 gap, against 11.31 for the real weights, with
split noise only ±0.16–0.34. **Table 1's "MegaDescriptor" row measures ImageNet Swin-L.**

**Status: REVERTED.** `main.py` briefly carried a local change disabling the substitution
for `--baseline`; it was added before this was understood and has been reverted. The call
site now reads `combined=use_combined_dataset` again, with a `NOTE(foundation-rows)`
recording the measurement above so nobody "fixes" it a second time.

Implication for the modernization work: this substitution is load-bearing for the published
numbers, so it must not be "cleaned up" into always using the real weights.

---

## 16. `run_suffix` whitelist rejects unknown backbones

`main.py:255-266` maps `--model_name` to a short tag through a hardcoded `if/elif` chain and
`raise NotImplementedError` otherwise. Any backbone not on the list fails instantly, before
the dataset is even built — including the ImageNet Swin models `get_model` itself
substitutes. Added `swin_large_patch4_window7_224` -> `swinL_in` and
`swin_tiny_patch4_window7_224` -> `swinT_in`.

**Proper fix:** derive the tag from the model name (slugify) instead of a whitelist, or fall
back to a sanitised name rather than raising.

---

## 17. NIS remains ~1 point below the other AL rows after the W1 fix

With the `counting_clusters.py` divide-by-zero fixed (W1) and all 4 seeds re-run on the
fixed code, `+ NIS` still lands at **48.84 mAP vs the paper's 50.72 (−1.88)**, while every
other AL row sits at −0.36 to −1.40. Seed std is 0.70, so the extra ~0.5–1.0 point is
outside noise. The corrupted `f_hat` was therefore **not** the whole story for that row.

Worth checking next: whether `Q[v_i]` and the proposal distribution `q` match the NIS paper
(Perez et al., ECCV 2024), and whether `--sampling_ratio` (default 1.0) is the value the
authors used — it is exposed in `utils.py` but never mentioned in the paper.

---

# Reproduction status (final, all runs complete)

| Table 1 row | reproduced | seeds | Δ mAP | note |
|---|---|---|---|---|
| ResNet-50 (Baseline) | yes | 4 | −0.65 | isolates the training-numerics gap |
| ResNet-50 (Skyline) | **makeshift** | 4 | +1.6 (30ep) / +4.3 (50ep) | item 13; overshoots at both budgets |
| MegaDescriptor | yes | 4 | −1.19 | as ImageNet Swin-L, item 15 |
| MiewID | **no** | — | — | item 14, transformers 5.x |
| SpCL (Base) | yes | 4 | −1.00 | |
| + Random | yes | 4 | −1.40 | |
| + GBAS | yes | 4 | −0.81 | |
| + LBAS | yes | 4 | −1.01 | cannot spend its budget, W2 |
| + NIS | yes | 4 | −1.88 | item 17, still unexplained |
| + AAS | yes | 4 | **−0.36** | |
| USL block (ICE/PPLR/CC/RTMem/Purification) | **no** | — | — | never ported; recipe above |

**Method ranking reproduces exactly.** The residual ≈1 mAP deficit across all *trained*
rows, with the untrained baseline at −0.65 and split noise only ±0.2–0.5, points at
training numerics (torch/cuDNN/faiss versions) rather than data, protocol or metrics.

## Highest-value next steps

1. Pin transformers 4.x and finish the MiewID row (item 14) — small, unblocks a whole row.
2. Revert the `main.py` foundation-row bypass (item 15) before any Table 1 rerun.
3. Fix `num_workers=0` (O1) — evaluation, not training, dominated wall-clock all week.
4. Give the skyline a real trainer path (item 13c) instead of the re-clustering workaround.
5. Investigate the NIS residual (item 17).
6. Port the 5 USL baselines using the RTMem recipe above — the largest missing block.

---

# Repo restructure, 2026-09-01

Done before starting the optimisation work, so that "the metrics still match" is a
falsifiable claim rather than a hope. Three tasks: freeze the reproduction, absorb the
vendored `SpCL/` tree, and move to a fresh conda env.

## R1. The reproduction is frozen in `outputs_initial/`, read-only

23 GB of run outputs sat in four sibling directories that new runs would have written
straight back into. They are now one read-only tree with its own provenance README,
`MANIFEST.sha256` (483 metric artifacts), and `CHECKPOINTS.txt` (192 checkpoints, 18.2 GB).

Regrouped so every seed of a row lives under one parent — `al/<sampler>/s<seed>/` and
`table1/<row>/s<seed>/` — replacing the old split where seed 0 lived in `outputs_table1/`,
seeds 1–3 in `outputs_table1_seeds/`, and the two foundation-row model families in
`outputs_swintest/`. `scripts/` writes the identical layout, so comparison is a tree walk.

**Pruned 4.36 GB.** All 19 `.failed*` AL attempts were GPU-memory casualties — 6 × faiss
`StandardGpuResourcesImpl::allocMemory`, 3 × `torch.OutOfMemoryError`, 2 × the faiss
`blasStatus == CUBLAS_STATUS_SUCCESS` assertion, the rest killed mid-epoch — and every one
has a completed counterpart that reached `Total running time`. Their checkpoints and partial
metrics were superseded; `train.log` + `log.txt` are kept (7.7 MB) as the evidence trail,
with per-attempt causes in `al/_failed_logs/WHY.txt` and the full deletion inventory in
`REMOVED.txt`. `_debug_skyline` got the same treatment: no metrics, 329 MB of weights, log kept.

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

## R3. `SpCL/` absorbed into the tree and deleted — closes item 12 and C2

`SpCL/spcl/` → `spcl/`, `SpCL/sampling/` → **`spcl/sampling/`**, and
`SpCL/examples/al_spcl_train_usl.py` → `train_al.py`. Licence preserved at
`third_party/SpCL-LICENSE` with provenance in `third_party/NOTICE.md`; the authors' prior
results moved to `reference/spcl_logs_al{,_100}/`.

Deleted as unreachable from the AL entry point: the 6 unused datasets, `models/dsbn.py`, the
`resnet_ibn` backbones, the three unused examples, `setup.py`/`setup.cfg`, and the upstream
README/figures.

- **Item 12 is closed structurally.** There is no second importable `sampling` package, so
  the failure where running from the repo root silently binds the wrong samplers and every AL
  row dies with `TypeError: ... unexpected keyword argument 'args'` can no longer happen. The
  `cd SpCL` + `PYTHONPATH` requirement is gone; runs start from the repo root.
- **C2 is closed.** `spcl/utils/faiss_utils.py` was byte-identical to the repo-root copy and
  was deleted in favour of it.
- `train_al.py`'s `-d` default was `market1501`, whose module is now gone; changed to
  `wildlife`. Every invocation already passed `-d wildlife` explicitly.

**Verified as a pure move:** of 42 relocated files, 34 are byte-identical to `HEAD` and the
8 modified ones contain *only* the import/factory edits listed above. No numerics touched.

## R4. Environment: `hitl`, cloned from `alm`

`conda create --clone alm -n hitl`, plus the two `env config vars`. `alm` is untouched so the
published numbers stay re-runnable in the exact environment that produced them. Cloning
rather than reinstalling holds the package set constant, so the refactor is the only variable
in the parity check. Library modernization (items 1–10) gets its own env later.

---

## R5. `LD_LIBRARY_PATH` was never in the launchers — NEW, FIXED

A latent instance of item 8. The launch scripts invoke the interpreter directly
(`/storage/.../bin/python`) rather than activating the env, so the `conda env config vars`
carrying `LD_LIBRARY_PATH` **never applied**. Every run all week depended on the value being
inherited from some shell where the env had been activated at some point.

Reproduced on both envs, so it is not a clone artifact:

```
$ .../alm/bin/python  -c "import faiss"                 # OK
$ .../alm/bin/python  -c "import pandas; import faiss"  # ImportError: CXXABI_1.3.15 not found
$ LD_LIBRARY_PATH=.../alm/lib .../alm/bin/python -c "import pandas; import faiss"   # OK
```

Import order is what hides it: pandas/torch pull in the older system `libstdc++` first, and
conda's `libfaiss.so` then cannot find `CXXABI_1.3.15`. A bare `import faiss` succeeds, so
the fault never shows up in a quick test.

`scripts/common.sh` now derives `LD_LIBRARY_PATH` and `PYTHONNOUSERSITE` from `$PYTHON`, so
a run is self-contained regardless of the calling shell.

## R6. `cudnn.deterministic` is overridden by `cudnn.benchmark` — NEW, NOT FIXED

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
