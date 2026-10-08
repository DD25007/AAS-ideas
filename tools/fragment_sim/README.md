# fragment_sim: offline simulator for "which cluster pairs should an AL query ask?"

Built in Phase X1 (2026-10-08) from the diagnostics H1/E1/D1-D4 on branch `e1-oracle-merge` (tag `al-loop-diagnostics-2026-10-08`).
Everything is CPU only and read-only on training runs. Full provenance, per-phase prereg and reports are in the archive
`/bhallukah/dd25007/archives/aas-ideas_al-loop-diagnostics_H1-E1-D1-D4_2026-10-08.tar.zst` (sha256 next to it).

## The setting
State = the epoch-95 state of `consam_long` (100-epoch ConSam run: features, 75,681 answered ML/CL pairs) and the pseudo-labels **P** of the next
re-clustering (epoch 96; 2,506 units, 20,565 images, 13 datasets). A GT identity split over m units needs m-1 merges ("needed merges": 695 for the
primary datasets CatIndividualImages 554 + MultiCamCows2024 102 + PrimFace 39; PolarBearVidID 88 is reported separately). The simulator asks a noiseless GT oracle
one medoid image pair per cluster pair, applies transitivity (ML merges clusters, CL blocks them, implied relations cost nothing) and reports merges
recovered, wasted queries (answer = different) and pairwise F1 at budgets 500/1k/2k/5k/10k/42,289.

## Files
| file | what it is |
|---|---|
| `paths.py` | locations. Small data in `data/`, `ref/` (git); big data in `$FRAGSIM_BIG` (default `/bhallukah/dd25007/AAS-ideas-out/kit`); outputs in `$FRAGSIM_WORK` (default `$FRAGSIM_BIG/work`) |
| `common.py` | loader (`load()`), pair counting (contingency / Moebius), `pair_f1`, `make_probe` (the real `ConsistencySampling.select`, text-patched to expose its locals). Copy of `headroom/d2/common.py` |
| `history.py` | `units()` (unit sizes/centroids/medoids/dominant dataset), `history()` (D3 history score H), `load_H()`. Copy of `headroom/d3/common3.py` |
| `sim_d2.py` | the D2 simulator: within-dataset / global scope, rule M (medoid, centroid order) and rule C (closest pair, min-linkage). Writes `sim_d2.json` |
| `sim_d3.py` | the D3 simulator = D2 within-dataset rule M with a pluggable candidate order (`RANKERS`). `python sim_d3.py M,R_H,R_HD` |
| `s_adaptive.py` | the D4 adaptive logistic ranker A / A-ex; mode `off` = pure centroid order (reproduces rule M). `python s_adaptive.py [off,A,A-ex]` |
| `d1/load_pool.py`, `d1/recluster.py` | D1: rebuild the pool (path, pid, dataset per image) and reproduce the epoch-96 re-clustering on CPU (rerank, DBSCAN, R_indep/R_comp, NP3) from the resume file |
| `check_reproduction.py` | the check below |
| `data/` | `pool.npz` (paths/pid/dset), `P.npz` (epoch-96 pseudo-labels), `views.npz` (V1/V2 = the two views used by `make_probe`), `labels_by_epoch.npz` (memory_labels of `consam_long` at epochs 5..95), `H.npz` (history score, epochs < 95), `needed_merges.json` |
| `ref/` | reference outputs: `d2_sim.json` (the D2 numbers all later phases are compared to), `d3_sim.json`, `d4_sim.json`, `d2_fragments.json` (per-fragment geometry) |
| `ref_metrics/` | last-epoch `evaluation_metrics_*.json` of skyline50, skyline100, consam_long, e1_oracle |
| `$FRAGSIM_BIG/` (not in git) | `resume_epoch95.pth` (453 MB) and `rerank_e95.npy` (1.7 GB; recomputable with `d1/recluster.py`) |

## Reproduction check (how D3 reproduces D2 rule M exactly)
D3 does not change the simulator, only the order in which the same 703,042 within-dataset unit pairs are tried. With the order `argsort(centroid distance, stable)`
`sim_d3.py` takes the identical code path as D2 `rule M`; `check_reproduction.py` compares recovered-per-dataset and wasted queries at all six budgets to `ref/d2_sim.json`
(307/695 = 44.2% at 5,000 queries, 4,452 wasted). It also recomputes H from the stored labels and runs the D4 simulator with the model disabled. All three must pass:
`python tools/fragment_sim/check_reproduction.py` (about 20 s, 4 GB RAM).

## Run a new ranker
In `sim_d3.py` add an entry to `RANKERS`: a function `(dc, h, iu) -> permutation` over the candidate pairs (`dc` = centroid-cosine distance, `h` = history score, `iu` = (unit A, unit B) index arrays; best pair first).
Then `python tools/fragment_sim/sim_d3.py M,<name>`; the report line gives primary-pooled recovery and wasted share at 5,000 queries; the full tables are in `$FRAGSIM_WORK/sim_d3.json`
(same schema as `ref/d3_sim.json`). A ranker that needs answers so far (adaptive) follows `s_adaptive.py` (rounds of 200 paid queries, refit, re-rank).

## What was found (for orientation; details in the archive reports)
Oracle merging of fragments gives +4.98 macro mAP (E1) but the nearest-cluster order recovers only 44% of needed merges in 5,000 queries (89% wasted), and neither the cluster-history
score (D3, 45.3% best) nor an adaptive logistic ranker (D4, 47.8% best) reaches the 54.2% bar.
