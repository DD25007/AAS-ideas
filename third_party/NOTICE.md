# Third-party code

## SpCL — `spcl/`

The `spcl/` package is derived from **SpCL** (*Self-paced Contrastive Learning with Hybrid
Memory for Domain Adaptive Object Re-ID*, Ge et al., NeurIPS 2020).

- Upstream: https://github.com/yxgeee/SpCL
- Upstream reference point: `01a486e` (2021-01-22), the tip when this fork was compared
  against upstream on 2026-08-27
- Licence: MIT, © 2020 Yixiao Ge — full text in [`SpCL-LICENSE`](SpCL-LICENSE)

It arrived here as a vendored `SpCL/` directory rather than a submodule, already modified by
the AAAI-26 paper's authors. On 2026-09-01 that directory was absorbed into the repo as
`spcl/` and deleted, so the tree no longer carries a second parallel copy of the codebase or
requires a `cd SpCL` + `PYTHONPATH` dance to run.

### What this package actually contains

**Upstream SpCL, unmodified** — the training loop, hybrid memory, ResNet backbones, the
evaluation and ranking metrics, and the data utilities.

**Modified from upstream — 4 files, all version-compatibility one-liners:**

| file | change |
|---|---|
| `spcl/datasets/__init__.py` | registers the `wildlife` dataset |
| `faiss_utils.py` (repo root) | `cast_integer_to_long_ptr` → `cast_integer_to_idx_t_ptr`; newer faiss dropped the old symbol |
| `spcl/utils/serialization.py` | `torch.load(..., weights_only=False)` for torch 2.6+ |
| `spcl/models/__init__.py`, `spcl/datasets/__init__.py` | factories trimmed to what the wildlife path uses |

**Added by the paper's authors — ~4,200 lines, and this is where the method under test
lives.** None of it exists upstream:

| file | what it is |
|---|---|
| `spcl/utils/wildlife_utils.py` | metric suite + NP3 constrained clustering + `get_evaluation_results` |
| `spcl/datasets/wildlife.py` | the open-set gallery-from-train protocol |
| `../train_al.py` | the AL training loop — entry point for every Table 1 AL row |
| `spcl/sampling/consistency.py` | **AAS** (`ConsistencySampling`), the paper's contribution |
| `spcl/sampling/balanced.py` | LBAS / GBAS baselines |
| `spcl/sampling/counting_clusters.py` | NIS baseline |
| `spcl/sampling/random.py`, `base.py` | random baseline; `BaseSampling` and the 0.02%/cycle budget |
| `spcl/evaluation.py`, `spcl/utils/constrained_clustering.py` | eval helpers; FINCH wrapper |

`spcl/sampling/` was `SpCL/sampling/` before the absorption. It was renamed into the `spcl`
package so it can no longer be shadowed by the repo-root `sampling/` package — see Todo.md
item 12, where that shadowing silently destroyed a 16-run batch.

### Removed during the absorption

Files on no code path reachable from `train_al.py`: the `market1501`, `msmt17`, `personx`,
`veri`, `vehicleid` and `vehiclex` datasets; `models/dsbn.py` and the `resnet_ibn` backbones;
the `spcl_train_usl.py` / `spcl_train_uda.py` / `test.py` examples; `setup.py`, `setup.cfg`
and the upstream README and figures. `spcl/utils/faiss_utils.py` was byte-identical to the
repo-root `faiss_utils.py` and was dropped in favour of it.

The authors' own prior run results, formerly `SpCL/logs_al{,_100}/`, are kept at
`../reference/spcl_logs_al{,_100}/`.
