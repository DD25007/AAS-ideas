# Phase D1: 4-seed fragmentation facts and partner-discovery gate (offline)

No training was run. All numbers below were produced by the scripts in `tools/partner_d1/` (copies in `/bhallukah/dd25007/phaseD1/scripts/`); the JSON behind every table is in `/bhallukah/dd25007/phaseD1/out/`.

**Read this first.** The matcher part of the gate (D1.5 R_LF / R_LFC and their D1.6 simulations) was **not run**: the 1,000-pair timing projects 31.9 GPU-h for the specified 3x3 scoring of the 4 seeds, above the 20 GPU-h cap, so under the rule I stopped there and give a cost table and a cheaper variant (D1.5). Everything that needs no matcher was run: D1.0-D1.4, the DINOv2 control, the baseline simulation and a free oracle upper bound, D1.8.

## 0. Provenance

- git HEAD `ee820fa3b1816bff9a26edcf749cb611c446f810` (commit `ee820fa`), branch `d1-partner-discovery` (created from that HEAD before any code was written).
- `git status --short` at the end of the work (before the commit): ` M .gitignore`; `?? reports/`; `?? tools/partner_d1/`. The ` M .gitignore` entry is the unrelated uncommitted edit that was there from the start; it was not touched and is not in the commit.
- Outputs only under `/bhallukah/dd25007/phaseD1/` (writable, checked first). `train_al.py`, `spcl/`, `.gitignore` not edited. New code: `tools/partner_d1/` only; `tools/fragment_sim/sim_d3.py` was **not** modified (its simulate routine is copied into `tools/partner_d1/sim_d1.py`, validated in D1.6).
- GPU: one slot lock per job through `scripts/gpu_guard.sh` (via `tools/partner_d1/run_gpu.sh`), `BECA_LOCK_DIR=/bhallukah/dd25007/gpu_locks`, 16 GiB free-VRAM gate checked at each launch (GPUs 6, 3, 5; GPU 2 held a beca-al process when I looked at launch time and was never used). Slot time of this phase: about 1.0 GPU-h in total (validation 2.4 min, feature extraction 4 seeds 28.2 min, seed analyses 2.7 + 2.8 + 2.6 + 4.6 min, matcher timing 6.6 min, DINOv2 11.0 min; the slot time is wall-clock of the job, one card each), far below the 20 GPU-h cap.
- No weights were downloaded: DINOv2-large from the local HF cache (`HF_HUB_OFFLINE=1`), ALIKED/LightGlue weights from `/storage/dd25007/cache/torch/hub/checkpoints`.

### Headline facts

- Epoch-50 C0 models (4 seeds), re-clustered with the epoch-40 constraints: **1884 ± 30 needed merges** pooled over the 13 datasets (seeds: 1840, 1902, 1903, 1890; the kit's 100-epoch epoch-95 state: 1398).
- **85.4 ± 3.7%** of wrong within-dataset pairs are *agree-wrong* (both ConSam views repeat the pseudo-label error; pooled over datasets). Kit epoch-95 state, recomputed: 89.8%.
- One answered must-link pair merges the **whole** of both groups it touches (D1.1: cluster-cluster, cluster-singleton and singleton-singleton, 6 of 6 each, sizes add up exactly), including any other identities inside them.
- Baseline B (nearest unit by centroid cosine): rank-1 35.7 ± 1.0%, top-5 60.1 ± 1.0%, true partner inside the 20 nearest for 80.0 ± 0.5% of fragments (all datasets pooled). That last number is a ceiling for any ranker that only re-orders the 20 nearest.
- B recovers 46.5 ± 1.0% of all needed merges in 5,000 questions with 82.9 ± 0.5% of questions wasted. An oracle that puts every candidate pair sharing an identity first (free upper bound, no matcher) reaches only 64.3 ± 1.4% and still wastes 76.2 ± 0.8%: the ALIVE rule (wasted <= 75%) is not reachable by any ranker of this form in this simulator.
- Matcher (ALIKED + LightGlue on idea-1's cached keypoints) scores 4.05 image pairs/s here; 3x3 scoring of the 4 seeds is out of budget (D1.5).

## D1.0 Provenance and inputs

| seed | run dir | `checkpoint_50` epoch (stored) | `model_best` epoch (stored) | model_best == ckpt50 (every tensor) | `resume_epoch40` epoch (stored) | ML / CL in resume file | ML / CL in Phase A file | identical to Phase A (arrays and repo sha) | repo `_pairs_sha` ML / CL (first 10) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | …/runs/C0_consam_det/logs | 50 | 50 | yes | 40 | 22524 / 44834 | 22524 / 44834 | yes | 9a520d5b99 / fa64b39e3e |
| 1 | …/runs/C0_consam_det/s1/logs | 50 | 50 | yes | 40 | 22393 / 46020 | 22393 / 46020 | yes | 32624b957c / 798ee3d455 |
| 2 | …/runs/C0_consam_det/s2/logs | 50 | 50 | yes | 40 | 22224 / 43905 | 22224 / 43905 | yes | bedd9ec866 / dbcb85623b |
| 3 | …/runs/C0_consam_det/s3/logs | 50 | 50 | yes | 40 | 22133 / 45225 | 22133 / 45225 | yes | 2fc5d6209f / 931a36f310 |

- Every seed has an epoch-50 `checkpoint_50.pth.tar` (the `epoch` stored inside is 50) and `model_best.pth.tar` is tensor-for-tensor identical to it, so the **last-epoch (epoch-50) model is used for all four seeds; there is no epoch mismatch to flag**. Seed 0 is the original C0 s0 run (`archive/idea-1/runs/C0_consam_det/logs`; `outputs/P2b_full_gate` is bit-identical per CLAUDE.md and was not needed). The `epoch` stored in `resume_epoch40.pth` is 40, the last AL cycle (epochs 0, 10, 20, 30, 40), so its ML/CL are the final constraint sets.
- Phase A constraint files `/bhallukah/dd25007/phaseA/constraints/C0_s{0..3}_e40.npz` were extracted from these same resume files: the arrays are element-for-element equal (same order, int32 vs int64 only) and the sha computed with the repo's own `spcl.utils.run_record._pairs_sha` is identical for ML and CL of all four seeds. (A plain byte hash differs only because of the dtype; that is not a difference in content.) Phase A additionally recorded that the epoch-40 sha matches the logged `[record]` for all four seeds.
- The 'epoch-50 model' pool features are a fresh inference with the clustering transform (`train_al.get_test_loader`, Resize 256x128 bicubic, no flip), **not** the momentum memory bank the run itself clusters from, so the unit counts differ somewhat from the training log's own last statistics (seed 0: 2,809 units here against 2,988 in the P2b log at epoch 49).

## D1.1 Does one answered pair merge two whole clusters?

Code (`spcl/utils/constrained_clustering.py`, `constrained_clustering(labels, rerank_dist, method, ml_pairs, cl_pairs)`), as `train_al.py:704-711` calls it with `method="~DBSCAN"` on `constrained_pseudo_labels`:

| endpoints of the ML pair | what the code does | lines |
|---|---|---|
| two different clusters | `labels[labels == cluster_b] = cluster_a`: **every** image carrying `b`'s label is relabelled, so the two clusters become one | 129-131 read the two labels, 140-141 the merge |
| a cluster and a singleton | same branch: in `train_al.py` a singleton is *not* `-1`. DBSCAN noise gets fresh ids (`train_al.py:594-605`) and reliability-filter rejects get `len(cluster_R_indep) + outliers` (`train_al.py:683-694`), so `labels` never contains `-1` on this path. The singleton's one-image label is merged into the cluster (or, if `a` is the singleton, the whole cluster is relabelled to the singleton's id: same partition) | 140-141 |
| two singletons | same branch; the result is a 2-image cluster. It survives, because the clean-up that sends clusters smaller than `min_samples` back to `-1` runs only for `method == "DBSCAN"` and the call uses `"~DBSCAN"` | 140-141; clean-up 319-323 skipped |

- The `-1` branches (lines 132-139: both noise / one noise) and the initial `labels == -1` renumbering (122-125) are dead code on the training path (`method != "DBSCAN"`, no `-1` labels). That the 5,436 singletons of `labels_pre` are real labels and not `-1` is confirmed below.
- The merge is not final: after all ML merges the cannot-link repair (167-272) splits any merged cluster that contains a CL pair along its ML-connected components (183-234; `[cc-diag]` of `b0_det_A` epoch 0: 198 of 4,453 clusters needed repair). A cluster without a CL pair inside stays merged.

Empirical check on the `b0_det_A` epoch-0 state (real `labels_pre`, 20,565 images, 6,659 labels of which 5,436 are singletons and 732 clusters have >= 2 images; real `epoch0_rerank.npy`). One synthetic ML pair between two images of the **same GT identity**, no other constraint, through the repo function, 6 random draws per kind (`d11_merge_probe.py`, `out/d11_merge_probe.json`):

| kind | draws | sizes before → after (size_a + size_b → merged size) | all images of both groups end in one label | merged size == size_a + size_b | no outside image joined |
|---|---|---|---|---|---|
| cluster-cluster | 6 | 28+8→36; 46+70→116; 214+160→374; 36+27→63; 52+8→60; 20+14→34 | 6/6 | 6/6 | 6/6 |
| cluster-singleton | 6 | 54+1→55; 12+1→13; 60+1→61; 1+289→290; 1+128→129; 23+1→24 | 6/6 | 6/6 | 6/6 |
| singleton-singleton | 6 | 1+1→2; 1+1→2; 1+1→2; 1+1→2; 1+1→2; 1+1→2 | 6/6 | 6/6 | 6/6 |

- All 18 draws merged **all** images of both groups and nothing else. In the cluster-cluster draws the two clusters held 2+1, 12+7, 22+20, 2+2, 5+1, 1+1 distinct GT identities respectively: one correct 'same' answer also fuses every other identity inside the two clusters (a false merge it cannot detect). Conversely a correct answer between two *pure* fragments of one identity repairs the whole fragment at once, which is the mechanism behind 'needed merges' being counted per unit pair.

## D1.2 Label reconstruction

`final_labels(features, ml, cl, indep_thres=None, rerank_dist=None)` in `tools/partner_d1/common.py`. **What is imported and what is mirrored:** `constrained_clustering` (from `spcl.utils.constrained_clustering`) and the rerank (`infra.fast_rerank.compute_jaccard_distance_fast`, exactly what `--fast-rerank` selects through `_fast.pick`) are imported, as are `get_data`, `create_model`, `get_test_loader` (feature extraction) from `train_al`. The statements between the rerank and the `constrained_clustering` call (3 x DBSCAN, `generate_pseudo_labels`, R_comp / R_indep, reliability filter; `train_al.py:461-711`) are **inline in `main_worker`, not a function**, so they cannot be imported; they are mirrored statement for statement, each block cites its source lines, and the exact-equality validation below is what stands in for 'imported'. `indep_thres` is the epoch-0 value carried in the resume file (`train_al.py:553-561` computes it at epoch 0 only); with `None` it is derived as epoch 0 does.

| validation target | result |
|---|---|
| (a) `b0_det_A` epoch 0 (a real `train_al.py` run, no `--fast-rerank`): `epoch0_features.npy` + cycle-1 ML/CL → rerank on GPU with the fast-rerank function | rerank `array_equal` to the saved `epoch0_rerank.npy`: **True** (max abs diff 0.0); preNP3 labels `array_equal` to `epoch0_labels_pre.npy`: **True**; **post-NP3 labels `array_equal` to `epoch0_labels_post.npy`: True** (0 differing entries); derived `indep_thres` 0.7328386 (= 1 − 0.2671614, the logged 'R_indep threshold') |
| (b) kit epoch-95 state (`resume_epoch95.pth` features + 75,681 answers) vs `tools/fragment_sim/data/P.npz`, using the kit's saved CPU rerank | **`array_equal`: True** (0 differing entries) |
| (b') same state, rerank recomputed on GPU by the fast-rerank function | `array_equal`: False — **2 of 20,565 labels differ** (ARI 0.999982); the GPU and the kit's CPU rerank matrices are not identical (max abs diff 0.0234), which is the kit's own documented 'CPU port, identical up to float rounding' |

Verdict: the target 'reproduce a saved label array exactly' is met by (a), the real-run state, end to end including the GPU rerank, and by (b) given the same rerank. (b') shows the only residual: CPU-vs-GPU rerank rounding in the kit's P, 2 labels. No STOP was needed.

## D1.3 Fragmentation on the C0 pipeline, 4 seeds

State per seed: epoch-50 model → pool features (20,565 images, clustering transform) → `final_labels` with that seed's epoch-40 constraints. Definitions (kit: `headroom/d1`, `headroom/d2`): **unit** = a label of the final pseudo-labels (singletons included); a unit's dataset = its majority dataset; **fragment** = for each GT identity split over m units of its dataset, every unit except the one holding most of its images; **needed merges** = sum over identities of (units holding it − 1) (= number of fragments); **false split** = a GT-same pair of images in different units (singleton included); **false merge** = a same-unit pair with different GT identity; pair counts are within-dataset. **Agree-wrong** = a wrong pair (false split or false merge, queried or not) that was **not queried** and on which both ConSam views (V1 constrained DBSCAN, V2 constrained FINCH, from the real `ConsistencySampling.select` run with `n = 0`) agree with the pseudo-label's wrong relation; its share is taken over *all* wrong pairs including the queried ones (as in D1: 55,960 / 63,611). Units per seed: s0 2809 (1056 singletons), s1 2904 (1120 singletons), s2 2889 (1095 singletons), s3 2904 (1101 singletons); kit epoch-95 state 2506 (822).

**Recomputation check.** The kit's epoch-95 state, run through the same code: needed merges Cat 554, MultiCamCows2024 102, PrimFace 39, PolarBearVidID 88 (the kit's numbers), 1,398 fragments in total (= the 1,398 rows of `ref/d2_fragments.json`); the recomputed V1/V2 equal the shipped `data/views.npz` exactly; the primary-datasets wrong pairs are 63,611 of which agree-wrong 55,960 (88.0%) and disagree-unqueried 1,519, identical to the archived D1 report.

| dataset | needed merges, seeds 0-3 | needed merges, mean ± sd | false-split pairs | false-merge pairs | agree-wrong share of wrong pairs | kit epoch-95 (needed / FS / FM / agree-wrong) |
|---|---|---|---|---|---|---|
| AmvrakikosTurtles | 6, 5, 5, 5 | 5.2 ± 0.5 | 10 ± 1 | 1 ± 2 | 100.0 ± 0.0 % | 6 / 10 / 0 / 90.0 % |
| CatIndividualImages | 625, 645, 647, 661 | 644.5 ± 14.8 | 34342 ± 1421 | 7282 ± 1892 | 91.8 ± 1.4 % | 554 / 32637 / 2382 / 96.8 % |
| Chicks4FreeID | 60, 61, 81, 64 | 66.5 ± 9.8 | 1506 ± 189 | 197 ± 59 | 93.7 ± 2.3 % | 34 / 946 / 142 / 92.4 % |
| CowDataset | 38, 35, 57, 34 | 41.0 ± 10.8 | 12558 ± 2770 | 117 ± 46 | 59.7 ± 20.3 % | 18 / 6469 / 118 / 83.6 % |
| DogFaceNet | 450, 478, 464, 472 | 466.0 ± 12.1 | 2842 ± 98 | 1222 ± 179 | 78.1 ± 5.2 % | 356 / 1998 / 937 / 84.8 % |
| MPDD | 69, 65, 58, 51 | 60.8 ± 7.9 | 388 ± 51 | 215 ± 48 | 69.3 ± 9.8 % | 35 / 207 / 110 / 83.3 % |
| MultiCamCows2024 | 183, 182, 197, 200 | 190.5 ± 9.3 | 59109 ± 6297 | 982 ± 200 | 81.6 ± 4.8 % | 102 / 22339 / 4212 / 75.8 % |
| PolarBearVidID | 107, 109, 121, 103 | 110.0 ± 7.7 | 26094 ± 1387 | 402 ± 123 | 96.2 ± 0.6 % | 88 / 22909 / 285 / 96.4 % |
| PrimFace | 70, 65, 53, 63 | 62.8 ± 7.1 | 3137 ± 863 | 269 ± 141 | 82.1 ± 12.4 % | 39 / 1270 / 771 / 95.3 % |
| ReunionTurtles | 18, 20, 13, 19 | 17.5 ± 3.1 | 31 ± 5 | 1 ± 2 | 96.4 ± 7.1 % | 16 / 28 / 3 / 100.0 % |
| SeaStarReID2023 | 166, 181, 166, 174 | 171.8 ± 7.2 | 3552 ± 207 | 238 ± 55 | 86.4 ± 2.7 % | 117 / 2451 / 136 / 95.5 % |
| SouthernProvinceTurtles | 40, 50, 34, 36 | 40.0 ± 7.1 | 422 ± 143 | 37 ± 27 | 91.4 ± 5.1 % | 25 / 255 / 5 / 90.4 % |
| ZakynthosTurtles | 8, 6, 7, 8 | 7.2 ± 1.0 | 13 ± 1 | 0 ± 0 | 94.6 ± 10.7 % | 8 / 14 / 0 / 100.0 % |
| PRIMARY (Cat+MCC+PrimFace) | 878, 892, 897, 924 | 897.8 ± 19.3 | 96588 ± 4520 | 8533 ± 2093 | 85.6 ± 3.6 % | 695 / 56246 / 7365 / 88.0 % |
| ALL 13 | 1840, 1902, 1903, 1890 | 1883.8 ± 29.8 | 144003 ± 6632 | 10963 ± 2062 | 85.4 ± 3.7 % | 1398 / 91533 / 9101 / 89.8 % |

Notes: mean ± sd with ddof = 1 over the 4 seeds. Pair counts are exact (contingency tables). The datasets with few images (turtles, Amvrakikos 21 images, Zakynthos 21) have needed-merge counts of 5-40; their agree-wrong shares rest on a handful of wrong pairs and can be 0/100% by construction.
- False splits outnumber false merges 13.4 ± 2.1:1 in pairs pooled over the 13 datasets (kit epoch-95: 10.1:1). The 4-seed epoch-50 states are more fragmented than the kit's 100-epoch state: 1884 ± 30 vs 1398 needed merges.

## D1.4 Partner-rank baseline B

For every fragment (one row per dataset / identity / non-largest unit, as in `ref/d2_fragments.json`; rows without a partner unit in the same dataset are left out: kit 1 of 1,398, seeds 4, 1, 3, 2 of 1840, 1902, 1903, 1890), the nearest true partner is ranked among **all other units of the same dataset** by cosine distance of unit centroids (C0 epoch-50 features). Rank-1 = the nearest unit is a partner; top-5 = a partner is within the 5 nearest; 'in 20' = a partner is within the 20 nearest (the candidate list of D1.5). Query = the fragment's unit centroid (the kit's step 1 used the fragment's own sub-centroid; identical for pure units).

| dataset | fragments n (seeds 0-3) | rank-1 % | top-5 % | median rank (all same-dataset units) | partner in 20 nearest % | kit epoch-95: rank-1 / top-5 / median |
|---|---|---|---|---|---|---|
| AmvrakikosTurtles | 6, 5, 5, 5 | 33.3 ± 18.9 | 70.8 ± 20.6 | 3.1 ± 2.1 | 100.0 ± 0.0 | 16.7 / 66.7 / 3 |
| CatIndividualImages **(Cat)** | 625, 645, 647, 661 | 32.0 ± 1.7 | 56.2 ± 1.3 | 3.8 ± 0.5 | 76.3 ± 0.9 | 25.5 / 48.9 / 6 |
| Chicks4FreeID | 60, 61, 81, 64 | 48.7 ± 4.4 | 87.5 ± 1.4 | 1.8 ± 0.5 | 97.5 ± 2.1 | 44.1 / 70.6 / 2 |
| CowDataset | 38, 35, 57, 34 | 74.9 ± 6.3 | 89.2 ± 7.1 | 1.0 ± 0.0 | 96.7 ± 1.8 | 61.1 / 88.9 / 1 |
| DogFaceNet | 448, 478, 463, 470 | 21.8 ± 0.9 | 43.0 ± 0.7 | 8.0 ± 0.8 | 66.7 ± 2.9 | 9.3 / 21.4 / 22 |
| MPDD | 69, 65, 56, 51 | 45.7 ± 6.7 | 66.2 ± 2.0 | 2.0 ± 0.8 | 86.0 ± 3.1 | 17.1 / 48.6 / 6 |
| MultiCamCows2024 | 182, 181, 197, 200 | 53.4 ± 4.7 | 79.9 ± 2.7 | 1.2 ± 0.5 | 95.4 ± 0.3 | 52.9 / 78.4 / 1 |
| PolarBearVidID | 107, 109, 121, 103 | 64.4 ± 3.7 | 90.6 ± 2.8 | 1.0 ± 0.0 | 98.9 ± 0.8 | 55.7 / 89.8 / 1 |
| PrimFace | 70, 65, 53, 63 | 38.2 ± 6.0 | 70.8 ± 5.4 | 2.0 ± 0.0 | 94.6 ± 2.7 | 30.8 / 74.4 / 3 |
| ReunionTurtles | 18, 20, 13, 19 | 7.8 ± 10.0 | 46.5 ± 5.5 | 5.9 ± 0.9 | 85.7 ± 6.3 | 18.8 / 50.0 / 6 |
| SeaStarReID2023 | 166, 181, 166, 174 | 30.8 ± 1.3 | 53.8 ± 1.6 | 4.2 ± 0.5 | 77.2 ± 1.9 | 13.7 / 40.2 / 8 |
| SouthernProvinceTurtles | 39, 50, 34, 36 | 45.5 ± 14.2 | 73.6 ± 6.0 | 2.2 ± 1.0 | 96.3 ± 2.5 | 24.0 / 52.0 / 3 |
| ZakynthosTurtles | 8, 6, 7, 8 | 16.5 ± 13.0 | 57.6 ± 14.2 | 5.8 ± 1.0 | 100.0 ± 0.0 | 25.0 / 75.0 / 4 |
| PRIMARY (Cat+MCC+PrimFace) | 877, 891, 897, 924 | 37.1 ± 1.6 | 62.3 ± 1.3 | 2.8 ± 0.5 | 81.6 ± 0.6 | 29.8 / 54.7 / 4 |
| ALL 13 | 1836, 1901, 1900, 1888 | 35.7 ± 1.0 | 60.1 ± 1.0 | 3.0 ± 0.0 | 80.0 ± 0.5 | 25.0 / 48.0 / 6 |

Per-seed values behind the Cat row and the pooled rows are in `out/d13_s{0..3}.json` (`partner_rank`). Ties of exactly equal centroid distance (a few identical-feature images in Cat) are broken by unit index here; the Gate-1 table in D1.5 counts ties against the ranker, which moves 3 dataset-seed cells by 1-2 fragments (Cat s0, s3).

## D1.5 Matcher ranker (Gate 1)

### 1. Which matcher, and is it on disk

- idea-1's closing sources (tag `idea-1-final`, `idea-1/lf_b/lf_b_report.md`) compared four variants on ConSam's cycle-1 questions: SuperPoint+LightGlue 9.61%, DISK+LightGlue 9.16%, ALIKED production features (matches with score >= 0.9) 9.16%, ALIKED no-upscaling 8.65%, ALIKED default match count 8.77% (share of held-out pairs answered at >= 97% precision; none reached the 10% line; the report notes all are within one point). idea-1's production feature set (the Phase-B 'production' config and the pool-wide cache `mv/cache/aliked_native.h5`, `lg_common.build('aliked')`) is **ALIKED + LightGlue**, the default named in the task. SuperPoint+LightGlue is nominally 0.45 points above ALIKED production (9.61 vs 9.16, one deterministic run per variant, no significance claim in the report), but its features are cached only for the 13,357 images that ConSam's cycle-1 pairs touch (`lf_b/cache/feats_superpoint`, per the LF-B report), not for the pool. **ALIKED + LightGlue is used.**
- On disk: LightGlue ALIKED weights `/storage/dd25007/cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth`, `aliked-n16.pth`; idea-1's cached native-resolution ALIKED keypoints/descriptors (<= 2048 per image) for **all 20,565 pool images**, `/bhallukah/dd25007/HITL-ReID-idea1-cache/mv/cache/aliked_native.h5` (12.9 GB). The index of the cache matches the pool row order: image sizes stored in the cache equal the file sizes of the pool images for 60 of 60 random images (`d15_match.py` asserts it). So no extraction cost.
- 'Inliers' = the number of LightGlue matches of the pair at the matcher's default threshold (`n_matches`, idea-1's own score; `mean_score` also kept). No geometric verification (RANSAC) was part of idea-1's score and none is added.

### 2-3. Candidates and timing

Candidate unit pairs = (fragment unit, one of its 20 nearest same-dataset units by centroid cosine), de-duplicated; representatives per unit = medoid + the 2 images nearest to the medoid (all images if the unit is smaller than 3); 3 x 3 image pairs per unit pair (`d15_candidates.py`). **Timing first:** 1,000 image pairs drawn at random from seed 0's 114,680, after 20 warm-up pairs, unbatched, features read from RAM, one H200 (GPU 3, shared with other tenants, slot-locked): **4.05 pairs/s** (247 s for 1,000; mean 62 matches/pair, median 22). idea-1's own production run of the same matching measured 5.53 pairs/s (mv2_match, 19.7 h for 392,032 pairs).

| seed | fragment units | unit pairs (= medoid-only image pairs) | 3x3 image pairs | GPU-h at the measured 4.05 pairs/s | GPU-h at idea-1's 5.53 pairs/s | GPU-h, medoid-only at 4.05 pairs/s |
|---|---|---|---|---|---|---|
| s0 | 1687 | 26,399 | 114,680 | 7.86 | 5.76 | 1.81 |
| s1 | 1748 | 27,240 | 114,908 | 7.88 | 5.77 | 1.87 |
| s2 | 1744 | 27,176 | 117,324 | 8.04 | 5.89 | 1.86 |
| s3 | 1765 | 27,405 | 117,805 | 8.07 | 5.92 | 1.88 |
| **4 seeds** |  | 108,220 | **464,717** | **31.9** | 23.3 | **7.4** |

**Cost stop.** The specified 3 x 3 scoring projects **31.9 GPU-h** for the 4 seeds (23.3 GPU-h even at idea-1's faster rate), above the 20 GPU-h cap of this phase (about 1.0 GPU-h is already spent). Per the instruction I stopped before scoring: **R_LF and R_LFC were not run**, and neither were their D1.6 simulations. Cheaper variant I propose (not run): **medoid-only** (one image pair per unit pair, 108,220 pairs, **7.4 GPU-h** for the 4 seeds at the measured rate, 3.8x cheaper); a middle option is 2 x 2 (medoid + nearest image). Both keep the specified candidate lists, logistic fit on the seed's own answered pairs, and decision rules unchanged. Needs your go-ahead (rule 3 only applies above 20 GPU-h, but this changes the preregistered scoring design).

### 4-5. Rank-1 / top-5 inside the 20 nearest, per dataset

R_LF / R_LFC: **not run** (cost stop above). R_DINO (optional control; frozen DINOv2-large CLS embedding, 224 x 224, unit centroid cosine, re-ranking the same 20 candidates; weights from the local HF cache; ~11 min of one card). Ties are counted against the ranker; `n` is seed 0's fragment count (other seeds differ by a few). Fragments whose partner is not among the 20 nearest are misses for every ranker (ceiling = 'in 20').

| dataset | n | B rank-1 % | R_DINO rank-1 % | R_DINO − B per seed (pp) | B top-5 % | R_DINO top-5 % | partner in 20 (ceiling) % | Gate 1 for R_DINO (control) |
|---|---|---|---|---|---|---|---|---|
| AmvrakikosTurtles | 6 | 33.3 ± 18.9 | 14.2 ± 9.6 | -16.7, -40.0, +0.0, -20.0 | 70.8 ± 20.6 | 71.7 ± 10.0 | 100.0 ± 0.0 | fail |
| CatIndividualImages | 625 | 31.9 ± 1.7 | 49.4 ± 2.7 | +13.9, +18.9, +16.8, +20.1 | 56.2 ± 1.3 | 66.7 ± 1.6 | 76.3 ± 0.9 | fail |
| Chicks4FreeID | 60 | 48.7 ± 4.4 | 58.7 ± 1.6 | +13.3, +11.5, +8.6, +6.2 | 87.5 ± 1.4 | 83.7 ± 2.3 | 97.5 ± 2.1 | fail |
| CowDataset | 38 | 74.9 ± 6.3 | 72.6 ± 6.6 | +5.3, -8.6, +0.0, -5.9 | 89.2 ± 7.1 | 88.7 ± 2.9 | 96.7 ± 1.8 | fail |
| DogFaceNet | 448 | 21.8 ± 0.9 | 48.6 ± 3.5 | +22.1, +26.8, +29.2, +28.9 | 43.0 ± 0.7 | 61.7 ± 3.7 | 66.7 ± 2.9 | fail |
| MPDD | 69 | 45.7 ± 6.7 | 68.6 ± 3.3 | +18.8, +23.1, +32.1, +17.6 | 66.2 ± 2.0 | 81.9 ± 2.5 | 86.0 ± 3.1 | PASS |
| MultiCamCows2024 | 182 | 53.4 ± 4.7 | 29.1 ± 5.0 | -15.9, -28.2, -32.5, -20.5 | 79.9 ± 2.7 | 65.0 ± 0.7 | 95.4 ± 0.3 | fail |
| PolarBearVidID | 107 | 64.4 ± 3.7 | 68.4 ± 4.2 | +6.5, -1.8, +1.7, +9.7 | 90.6 ± 2.8 | 94.6 ± 1.2 | 98.9 ± 0.8 | fail |
| PrimFace | 70 | 38.2 ± 6.0 | 62.0 ± 3.2 | +15.7, +26.2, +26.4, +27.0 | 70.8 ± 5.4 | 85.6 ± 3.3 | 94.6 ± 2.7 | PASS |
| ReunionTurtles | 18 | 7.8 ± 10.0 | 24.8 ± 7.5 | +22.2, +20.0, +15.4, +10.5 | 46.5 ± 5.5 | 57.6 ± 7.7 | 85.7 ± 6.3 | fail |
| SeaStarReID2023 | 166 | 30.8 ± 1.3 | 34.7 ± 3.7 | -1.2, +7.2, +6.0, +3.4 | 53.8 ± 1.6 | 49.1 ± 1.7 | 77.2 ± 1.9 | fail |
| SouthernProvinceTurtles | 39 | 45.5 ± 14.2 | 37.4 ± 12.4 | -5.1, -10.0, -11.8, -5.6 | 73.6 ± 6.0 | 67.5 ± 7.6 | 96.3 ± 2.5 | fail |
| ZakynthosTurtles | 8 | 16.5 ± 13.0 | 24.4 ± 7.8 | +0.0, +33.3, -14.3, +12.5 | 57.6 ± 14.2 | 48.2 ± 10.8 | 100.0 ± 0.0 | fail |
| PRIMARY (Cat+MCC+PrimFace) | 877 | 37.0 ± 1.6 | 46.0 ± 1.9 | +7.9, +9.9, +6.6, +11.8 | 62.3 ± 1.3 | 67.7 ± 1.3 | 81.6 ± 0.6 | fail |
| ALL 13 | 1836 | 35.7 ± 1.0 | 48.2 ± 1.9 | +10.7, +12.9, +12.0, +14.4 | 60.1 ± 1.0 | 67.4 ± 1.3 | 80.0 ± 0.5 | fail |

Gate 1 as preregistered (4-seed mean rank-1 >= 50% **and** >= mean B + 15 pp **and** every seed above its own B) applied mechanically to the control; the row for R_LF / R_LFC is 'not run'.

## D1.6 Merge-recovery simulation

### 1. Model-disabled check (kit)

`tools/fragment_sim/check_reproduction.py` on the epoch-95 state (FRAGSIM_WORK redirected to `out/kit_work/`): {'H_matches': True, 'd3_ruleM': True, 'd4_off': True} — H recomputed equals shipped H, `sim_d3.py` rule M reproduces D2 within/M exactly at all 6 budgets, and the D4 simulator with the model disabled (`s_adaptive.py off`) reproduces it too (`model disabled reproduces D2 within/M exactly … True`). **Passed; no STOP.**
The simulator used below is a state-parametrised copy of `sim_d3.simulate` (`sim_d1.py`; `sim_d3.py` itself runs on import against the kit state and cannot be called for other states). Check of the copy: on the kit state with order B it reproduces `ref/d2_sim.json` within/M exactly (recovered per dataset, wasted and free merges at all 6 budgets): **True** (307/695 primary, 557/1398 all, 4452 wasted at 5,000).

### 2. Simulation on each seed's state

Same protocol as the kit: a noiseless GT oracle answers one medoid image pair per unit pair, within-dataset pairs only, transitivity on the existing answers, a 'same' answer merges the two units. Orders: **B** = centroid cosine (rule M); **ORACLE** = free upper bound for any ranker that only re-orders the 20-nearest candidate pairs (candidate pairs whose two units share a GT identity first, B order inside, then all other pairs in B order); **R_DINO** = candidate pairs by DINOv2 centroid cosine, then the rest in B order; **R_DINO_g** = all within-dataset pairs by DINOv2 centroid cosine. R_LF / R_LFC: not run. 'Recovered' pooled over **all** needed merges (the preregistered denominator); 'primary' = Cat + MultiCamCows2024 + PrimFace (the kit's headline). Wasted = answers 'different' / questions asked. For every ranker and seed the candidate list (784k-867k within-dataset unit pairs) was not exhausted before 42,289 questions (`exhausted_at` is null in all 16 runs).

**At 5,000 questions:**

| ranker | recovery % of all needed merges, seeds 0-3 | mean ± sd | primary-3 recovery % mean ± sd | wasted % of questions, mean ± sd |
|---|---|---|---|---|
| B | 45.4, 46.0, 47.7, 47.0 | 46.5 ± 1.0 | 47.8 ± 0.9 | 82.9 ± 0.5 |
| ORACLE | 62.5, 64.3, 64.5, 65.8 | 64.3 ± 1.4 | 63.4 ± 1.2 | 76.2 ± 0.8 |
| R_DINO | 58.9, 60.5, 60.3, 61.3 | 60.2 ± 1.0 | 61.1 ± 1.4 | 77.7 ± 0.6 |
| R_DINO_g | 36.8, 37.7, 35.5, 37.1 | 36.8 ± 1.0 | 39.9 ± 1.7 | 86.5 ± 0.4 |

**Across budgets** (recovery % of all needed merges / wasted % of the questions asked, mean ± sd over 4 seeds; the candidate list was not exhausted at any budget):

| ranker | 500 | 1,000 | 2,000 | 5,000 | 10,000 | 42,289 |
|---|---|---|---|---|---|---|
| B | 19.8 ± 0.6 / 28.3 ± 2.3 | 28.4 ± 0.5 / 48.0 ± 1.5 | 36.2 ± 0.6 / 66.8 ± 0.8 | 46.5 ± 1.0 / 82.9 ± 0.5 | 55.3 ± 1.6 / 89.8 ± 0.4 | 70.4 ± 1.0 / 96.9 ± 0.1 |
| ORACLE | 24.8 ± 0.6 / 9.5 ± 1.2 | 46.8 ± 0.8 / 13.8 ± 0.8 | 64.3 ± 1.4 / 40.5 ± 1.9 | 64.3 ± 1.4 / 76.2 ± 0.8 | 64.4 ± 1.3 / 88.1 ± 0.4 | 70.9 ± 1.1 / 96.9 ± 0.1 |
| R_DINO | 14.8 ± 0.4 / 47.2 ± 1.5 | 23.0 ± 0.5 / 58.2 ± 0.6 | 37.3 ± 0.5 / 65.8 ± 0.4 | 60.2 ± 1.0 / 77.7 ± 0.6 | 64.1 ± 1.3 / 88.1 ± 0.4 | 70.8 ± 1.1 / 96.9 ± 0.1 |
| R_DINO_g | 9.9 ± 0.6 / 65.0 ± 2.4 | 14.3 ± 0.5 / 74.5 ± 0.9 | 20.7 ± 0.3 / 81.3 ± 0.3 | 36.8 ± 1.0 / 86.5 ± 0.4 | 54.3 ± 1.0 / 90.0 ± 0.2 | 77.0 ± 1.2 / 96.6 ± 0.1 |

**Per dataset at 5,000 questions** (recovered / needed merges, %; mean ± sd over seeds):

| dataset | needed merges (mean) | B | ORACLE | R_DINO | R_DINO_g |
|---|---|---|---|---|---|
| AmvrakikosTurtles | 5.2 ± 0.5 | 45.8 ± 31.3 | 95.0 ± 10.0 | 95.0 ± 10.0 | 90.0 ± 20.0 |
| CatIndividualImages | 644.5 ± 14.8 | 39.4 ± 0.7 | 55.2 ± 2.1 | 53.5 ± 2.2 | 40.2 ± 2.0 |
| Chicks4FreeID | 66.5 ± 9.8 | 70.8 ± 2.2 | 81.6 ± 3.1 | 69.9 ± 0.7 | 41.3 ± 2.9 |
| CowDataset | 41.0 ± 10.8 | 90.2 ± 4.1 | 92.0 ± 4.2 | 92.0 ± 4.2 | 84.7 ± 3.3 |
| DogFaceNet | 466.0 ± 12.1 | 28.5 ± 1.7 | 50.4 ± 3.3 | 42.8 ± 2.2 | 8.2 ± 1.5 |
| MPDD | 60.8 ± 7.9 | 49.0 ± 8.8 | 64.8 ± 4.4 | 55.0 ± 4.5 | 14.0 ± 5.8 |
| MultiCamCows2024 | 190.5 ± 9.3 | 70.6 ± 3.8 | 84.4 ± 0.6 | 79.5 ± 1.7 | 25.0 ± 1.8 |
| PolarBearVidID | 110.0 ± 7.7 | 85.1 ± 3.8 | 92.8 ± 1.0 | 90.1 ± 2.0 | 70.8 ± 2.2 |
| PrimFace | 62.8 ± 7.1 | 64.9 ± 7.1 | 84.3 ± 7.5 | 84.3 ± 7.5 | 82.2 ± 6.2 |
| ReunionTurtles | 17.5 ± 3.1 | 25.3 ± 15.4 | 83.0 ± 5.2 | 83.0 ± 5.2 | 94.0 ± 4.7 |
| SeaStarReID2023 | 171.8 ± 7.2 | 41.7 ± 1.8 | 65.1 ± 2.0 | 65.1 ± 2.0 | 62.0 ± 2.8 |
| SouthernProvinceTurtles | 40.0 ± 7.1 | 63.7 ± 8.6 | 88.4 ± 6.2 | 78.4 ± 4.3 | 34.5 ± 9.2 |
| ZakynthosTurtles | 7.2 ± 1.0 | 33.8 ± 14.8 | 86.6 ± 11.7 | 86.6 ± 11.7 | 86.6 ± 11.7 |
| PRIMARY (Cat+MCC+PrimFace) | 897.8 ± 19.3 | 47.8 ± 0.9 | 63.4 ± 1.2 | 61.1 ± 1.4 | 39.9 ± 1.7 |
| ALL 13 | 1883.8 ± 29.8 | 46.5 ± 1.0 | 64.3 ± 1.4 | 60.2 ± 1.0 | 36.8 ± 1.0 |

## D1.7 Pre-registered decision rules (applied mechanically)

The rules are applied in the decision table at the end of this report (one row per rule and ranker); notes:

- **Cat fails Gate 1 for the R_DINO control**: mean rank-1 49.4% < 50% (mean gain over B +17.5 pp; per seed +13.9, +18.9, +16.8, +20.1). Datasets that pass Gate 1 for R_DINO: MPDD, PrimFace; they hold 6.6 ± 0.8% of the needed merges (mean ± sd over seeds; Cat alone holds 34.2 ± 0.5%). For R_LF / R_LFC this share cannot be given because they were not run.
- These verdicts are for the **control and the oracle bound only**. The preregistered verdict on the matcher (Gate 1 and the recovery rule for R_LF / R_LFC) is **open**, not DEAD / WEAK / ALIVE.
- What the oracle row says about the rule itself: a ranker that puts every identity-sharing candidate pair first still has wasted 76.2 ± 0.8% at 5,000 questions, above the 75% line, so 'wasted <= 75%' is not reachable by re-ranking the 20-nearest lists under this protocol (medoid query, fall-back to centroid order). The oracle's recovery is flat at about 64.3 ± 1.4% from 2,000 questions on, and its wasted share is already 40.5 ± 1.9% at 2,000 (table above). I did **not** isolate the cause (fall-back queries after the identity-sharing candidates run out versus medoid pairs of impure units that answer 'different' although the units share an identity). The recovery threshold B + 15 pp needs a ranker that captures 84% of the B → oracle headroom (+17.8 pp); R_DINO captures 77%.
- **Even an ALIVE verdict would only earn an end-to-end 4-seed run; offline recovery has never predicted mAP in this project.**
- No p-values or Hedges g are given: nothing here is a between-arm mAP contrast; the rules are thresholds on 4-seed means.

## D1.8 Cost table for Phase D2 (no runs)

Measured wall-clock per 50-epoch run (1 card each, so GPU-h = wall-clock h). C0 values are read from the run logs (first to last timestamp, `Total running time`); E1 and skyline text logs are not on disk (their run directories hold only evaluation files), so their wall-clocks are the watcher figures printed in the archived reports `headroom/e1/REPORT.md` and `headroom/h1/REPORT.md` (first launch to last log write, includes the gap before the resume).

| run | wall-clock = GPU-h (1 card) | source / conditions |
|---|---|---|
| C0 s0 clean pass (`P2b_full_gate`, killed in epoch 14 and resumed) | 8.8 h incl. 3.5 redone epochs (8.3 h one clean pass) | P2b `attempt_0.log` 2026-10-03T23:14:54 → `attempt_1.log` 2026-10-04T08:01:22 (one card, `P2b_full_gate`); 8.3 h clean-pass figure from the archived H0 findings |
| C0 s1 / s2 / s3 | 16.1 h / 15.8 h / 15.5 h | archived idea-1 `attempt_*.log`: `Total running time` 16:08:26, 15:45:58, 15:27:11; three runs concurrent on a heavily loaded node |
| E1 oracle merge (`e1_oracle`, seed 0, resumed 1x) | 14.33 h | archived E1 REPORT (GPU 5; the E1 cost-guard measurement saw load average 59) |
| `--full-supervision` skyline 50 epochs (`skyline50`, seed 0, no resume) | 18.03 h | archived H1 REPORT (the H1 operator note reports load average ~126 at launch); the H0 estimate for an unloaded card was ~7.4 h (6.8-8.3), an estimate, not a measurement |

| Phase D2 item | runs | GPU-h at the measured (loaded) speed | GPU-h at the unloaded reference (estimate) |
|---|---|---|---|
| E1 oracle merge, seeds 1-3 | 3 x 1 card | 3 x 14.33 = **43.0** | 3 x 8.3 = 24.9 (E1 is plain ConSam + a label merge; same compute as a C0 pass) |
| like-for-like skyline (`--full-supervision`, 50 epochs), seeds 1-3 | 3 x 1 card | 3 x 18.03 = **54.1** | 3 x 7.4 = 22.2 (H0 estimate) |
| **total** | 6 runs | **97.1** | 47.1 |

Both totals are above the 20 GPU-h line, so CLAUDE.md rule 3 (cost table, wait for approval) applies to D2. Caveats: seed 1-3 runs of E1 and skyline have not been run, so these are seed-0 costs times 3; H0 findings list the determinism of the `--full-supervision` path as untested.

GPU status at report time (2026-10-22 01:12; free memory moves; locks at `/bhallukah/dd25007/gpu_locks`, `gpu_locks.py busy` printed `(nothing)`):

| GPU | free GiB | ≥ 16 GiB gate | note | slot lock |
|---|---|---|---|---|
| 0 | 16.7 | yes |  | no live slot |
| 1 | 18.5 | yes |  | no live slot |
| 2 | 26.5 | yes | beca-al process on this card → excluded | no live slot |
| 3 | 24.4 | yes |  | no live slot |
| 4 | 20.9 | yes |  | no live slot |
| 5 | 18.9 | yes |  | no live slot |
| 6 | 23.2 | yes |  | no live slot |
| 7 | 5.1 | no | beca-al process on this card → excluded | no live slot |

## Decision table

| rule | threshold | value | verdict |
|---|---|---|---|
| Gate 1, CatIndividualImages: R_LF / R_LFC | mean rank-1 >= 50%, >= B + 15 pp, every seed > own B | B 31.9% | **NOT RUN** (cost stop, D1.5) |
| Gate 1, CatIndividualImages: R_DINO (control only) | same | rank-1 49.4% (B 31.9%, Δ per seed +13.9, +18.9, +16.8, +20.1) | fail |
| Gate 1, MultiCamCows2024: R_LF / R_LFC | mean rank-1 >= 50%, >= B + 15 pp, every seed > own B | B 53.4% | **NOT RUN** (cost stop, D1.5) |
| Gate 1, MultiCamCows2024: R_DINO (control only) | same | rank-1 29.1% (B 53.4%, Δ per seed -15.9, -28.2, -32.5, -20.5) | fail |
| Gate 1, PrimFace: R_LF / R_LFC | mean rank-1 >= 50%, >= B + 15 pp, every seed > own B | B 38.2% | **NOT RUN** (cost stop, D1.5) |
| Gate 1, PrimFace: R_DINO (control only) | same | rank-1 62.0% (B 38.2%, Δ per seed +15.7, +26.2, +26.4, +27.0) | PASS |
| Gate 1, PolarBearVidID: R_LF / R_LFC | mean rank-1 >= 50%, >= B + 15 pp, every seed > own B | B 64.4% | **NOT RUN** (cost stop, D1.5) |
| Gate 1, PolarBearVidID: R_DINO (control only) | same | rank-1 68.4% (B 64.4%, Δ per seed +6.5, -1.8, +1.7, +9.7) | fail |
| Gate 1, SeaStarReID2023: R_LF / R_LFC | mean rank-1 >= 50%, >= B + 15 pp, every seed > own B | B 30.8% | **NOT RUN** (cost stop, D1.5) |
| Gate 1, SeaStarReID2023: R_DINO (control only) | same | rank-1 34.7% (B 30.8%, Δ per seed -1.2, +7.2, +6.0, +3.4) | fail |
| Recovery @5,000, R_LF / R_LFC | DEAD < B+10 pp (or < 3 of 4 seeds ≥ B+10); ALIVE ≥ B+15 pp and wasted ≤ 75% | B_rec 46.5 ± 1.0% | **NOT RUN** (cost stop, D1.5) |
| Recovery @5,000, R_DINO (control) | same | mean 60.2% = B +13.7 pp; seeds ≥ B+10 pp: 4/4; wasted 77.7% | WEAK |
| Recovery @5,000, R_DINO_g (control) | same | mean 36.8% = B -9.7 pp; seeds ≥ B+10 pp: 0/4; wasted 86.5% | DEAD |
| Recovery @5,000, ORACLE (upper bound, not a method) | same | mean 64.3% = B +17.8 pp; seeds ≥ B+10 pp: 4/4; wasted 76.2% | WEAK |

## Not measured, and why

- **R_LF and R_LFC (matcher rankers), their Gate-1 rank-1 / top-5 and their D1.6 recovery: not measured.** 3x3 scoring of the 4 seeds projects above the 20 GPU-h cap (D1.5 cost table) and the instruction was to stop and propose a cheaper variant; the cheaper variant was not run without your go-ahead. So the matcher's Gate 1 / DEAD-WEAK-ALIVE verdicts are open. Consequently also not fitted: the logistic model of R_LFC on the seed's own answered pairs.
- **'Inliers'** are LightGlue match counts (idea-1's score); no RANSAC-verified inlier count exists or was computed.
- **R_DINO is a control with a simple design choice** (DINOv2-large CLS, 224 x 224 resize, unit-centroid cosine, no fine-tuning). Other DINOv2 sizes, patch-token pooling or crops were not tried; a stronger or weaker control is possible.
- **The oracle row is a bound for rankers that only re-order the 20-nearest lists with the kit's medoid query**; it is not a method and says nothing about other question protocols (e.g. querying the closest image pair instead of medoids, rule C of D2, was not re-run).
- **Features are a fresh inference with the epoch-50 weights, not the training run's momentum memory bank**; the training log's own cluster statistics (e.g. seed 0 epoch 49: 1,805 clusters / 1,183 un-clustered) differ from the states analysed here (seed 0: 1,753 / 1,056). The states are 'what a re-clustering with the final model and the final answers would produce', not the labels the last training epoch used. Seeds 1-3 logs were not compared for this reason.
- **The re-clustering block of `train_al.py` is mirrored, not imported** (inline code); equality was proven on one real run (b0_det_A epoch 0, `array_equal`) and on the kit state with the kit's rerank; it was not re-proven on a C0 seed's epoch-50 state because no saved label array exists for it.
- **D1.1 is one state, one pair at a time, with no CL constraints** (epoch 0 of `b0_det_A`); interaction with CL repair on later cycles was not measured, only read from the code.
- **Partner rank uses unit centroids**; the kit's D2 step 1 used the fragment's own sub-centroid (identical for pure units). Min-linkage ranks on the rerank matrix (the kit's second distance) were not recomputed.
- **Oracle definition**: two units "share an identity" if any GT identity occurs in both; cross-dataset units (always wrong; 59 in the kit state) are counted in no per-dataset table.
- **E1 and skyline wall-clocks** come from archived watcher reports, not from re-reading their text logs (not on disk); seeds 1-3 of E1/skyline do not exist yet, so the D2 totals are seed-0 cost x 3. Unloaded-card figures are estimates (H0 findings), not measurements.
- **No mAP, p-value or Hedges g** is reported: nothing in this phase is an arm-vs-arm mAP comparison. n = 4 seeds, sd with ddof = 1, rules applied to means; a verdict on 4 seeds is a screen, not a test.
- **Free-GPU table** is a snapshot; foreign jobs change it minute to minute.

## Artifacts under `/bhallukah/dd25007/phaseD1/`

- `scripts/` — copies of all code of this phase (`tools/partner_d1/*.py`, `run_gpu.sh`) and the driver `scripts_driver.sh` (also at the top level).
- `logs/` — every run log (`d12_validate.log`, `d13_features.log`, `d13_analyze_s{0..3}.log`, `d13_kit95.log`, `d15_timing_s0.log`, `d15_dino.log`, `d16_s{0..3}.log`, `d16_check_reproduction.log`, `untar*.log`).
- `out/` — `d10_provenance.json`, `d10_repo_sha.json`, `d11_merge_probe.json`, `d12_validate.json`, `feat_s{0..3}.npy` (epoch-50 pool features), `P_s{0..3}.npy` (final labels per seed), `views_*.npz` (V1/V2), `d13_*.json` (fragmentation + partner ranks), `cand_*.npz` and `d15_candidate_counts_*.json` (matcher candidates), `d15_timing_s0.json`, `dino_feat.npy`, `d16_s{0..3}.json` (simulations), `d16_copy_check.json`, `d17_decisions.json`, `kit_work/` (kit reproduction-check outputs).
- `arch_x/` — the few report/prereg files extracted from the diagnostics archive to read the D1/D2 definitions (read-only reference).
