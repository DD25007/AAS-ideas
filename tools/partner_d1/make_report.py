"""Builds reports/phaseD1_partner.md.  Every number comes from $OUT/*.json (written by the d1x scripts); prose is static text."""
import os, sys, json, re, datetime, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tables import *
REPO = "/storage/dd25007/AAS-ideas"
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True, cwd=REPO).stdout.strip()

D3 = {s: d13(f"s{s}") for s in SEEDS}; K95 = d13("kit95"); SM = {s: d16(f"s{s}") for s in SEEDS}
DS = sorted(D3[0]["frag"]); P0 = J("d10_provenance"); PS = J("d10_repo_sha"); V = J("d12_validate"); PR = J("d11_merge_probe")
CC = J("d16_copy_check"); TM = J("d15_timing_s0")
CNT = {}
for f in os.listdir(OUT):
    if f.startswith("d15_candidate_counts_"): CNT.update(json.load(open(f"{OUT}/{f}")))
SELECT = lambda sel, fr, k: sum(fr[x][k] for x in sel)
out = []
def w(s=""): out.append(s)
def pct(x, f="{:.1f}"): return f.format(100 * x)

# ------------------------------------------------------------------ provenance
w("# Phase D1: 4-seed fragmentation facts and partner-discovery gate (offline)\n")
w("No training was run. All numbers below were produced by the scripts in `tools/partner_d1/` (copies in `/bhallukah/dd25007/phaseD1/scripts/`); the JSON behind every table is in `/bhallukah/dd25007/phaseD1/out/`.\n")
w("**Read this first.** The matcher part of the gate (D1.5 R_LF / R_LFC and their D1.6 simulations) was **not run**: the 1,000-pair timing projects "
  f"{sum(CNT[t]['n_image_pairs_3x3'] for t in ['s0','s1','s2','s3']) / TM['pairs_per_s'] / 3600:.1f} GPU-h for the specified 3x3 scoring of the 4 seeds, above the 20 GPU-h cap, so under the rule I stopped there and give a cost table and a cheaper variant (D1.5). "
  "Everything that needs no matcher was run: D1.0-D1.4, the DINOv2 control, the baseline simulation and a free oracle upper bound, D1.8.\n")
w("## 0. Provenance\n")
w(f"- git HEAD `{sh('git rev-parse HEAD')}` (commit `ee820fa`), branch `d1-partner-discovery` (created from that HEAD before any code was written).")
w("- `git status --short` at the end of the work (before the commit): " + "; ".join("`" + l + "`" for l in subprocess.run("git status --short", shell=True, capture_output=True, text=True, cwd=REPO).stdout.splitlines()) + ". The ` M .gitignore` entry is the unrelated uncommitted edit that was there from the start; it was not touched and is not in the commit.")
w("- Outputs only under `/bhallukah/dd25007/phaseD1/` (writable, checked first). `train_al.py`, `spcl/`, `.gitignore` not edited. New code: `tools/partner_d1/` only; `tools/fragment_sim/sim_d3.py` was **not** modified (its simulate routine is copied into `tools/partner_d1/sim_d1.py`, validated in D1.6).")
w("- GPU: one slot lock per job through `scripts/gpu_guard.sh` (via `tools/partner_d1/run_gpu.sh`), `BECA_LOCK_DIR=/bhallukah/dd25007/gpu_locks`, 16 GiB free-VRAM gate checked at each launch (GPUs 6, 3, 5; GPU 2 held a beca-al process when I looked at launch time and was never used). Slot time of this phase: " +
  "about 1.0 GPU-h in total (validation 2.4 min, feature extraction 4 seeds 28.2 min, seed analyses 2.7 + 2.8 + 2.6 + 4.6 min, matcher timing 6.6 min, DINOv2 11.0 min; the slot time is wall-clock of the job, one card each), far below the 20 GPU-h cap.")
w("- No weights were downloaded: DINOv2-large from the local HF cache (`HF_HUB_OFFLINE=1`), ALIKED/LightGlue weights from `/storage/dd25007/cache/torch/hub/checkpoints`.\n")

w("### Headline facts\n")
need_all = [SELECT(DS, D3[s]["frag"], "needed_merges") for s in SEEDS]
aw_all = [SELECT(DS, D3[s]["frag"], "c_agree_wrong") / SELECT(DS, D3[s]["frag"], "wrong_pairs") for s in SEEDS]
b1 = [SM[s]["gate"]["B"]["ALL"]["top1"] for s in SEEDS]; b5 = [SM[s]["gate"]["B"]["ALL"]["top5"] for s in SEEDS]; c20 = [SM[s]["gate"]["B"]["ALL"]["in_top20"] for s in SEEDS]
rec = lambda k, key="recovery_all": [SM[s]["summary_5000"][k][key] for s in SEEDS]
w(f"- Epoch-50 C0 models (4 seeds), re-clustered with the epoch-40 constraints: **{ms(need_all, '{:.0f}')} needed merges** pooled over the 13 datasets (seeds: {', '.join(str(x) for x in need_all)}; the kit's 100-epoch epoch-95 state: {SELECT(DS, K95['frag'], 'needed_merges')}).")
w(f"- **{ms(aw_all, '{:.1f}', 100)}%** of wrong within-dataset pairs are *agree-wrong* (both ConSam views repeat the pseudo-label error; pooled over datasets). Kit epoch-95 state, recomputed: {pct(SELECT(DS, K95['frag'], 'c_agree_wrong') / SELECT(DS, K95['frag'], 'wrong_pairs'))}%.")
w("- One answered must-link pair merges the **whole** of both groups it touches (D1.1: cluster-cluster, cluster-singleton and singleton-singleton, 6 of 6 each, sizes add up exactly), including any other identities inside them.")
w(f"- Baseline B (nearest unit by centroid cosine): rank-1 {ms(b1, '{:.1f}', 100)}%, top-5 {ms(b5, '{:.1f}', 100)}%, true partner inside the 20 nearest for {ms(c20, '{:.1f}', 100)}% of fragments (all datasets pooled). That last number is a ceiling for any ranker that only re-orders the 20 nearest.")
w(f"- B recovers {ms(rec('B'), '{:.1f}', 100)}% of all needed merges in 5,000 questions with {ms(rec('B', 'wasted_share'), '{:.1f}', 100)}% of questions wasted. An oracle that puts every candidate pair sharing an identity first (free upper bound, no matcher) reaches only {ms(rec('ORACLE'), '{:.1f}', 100)}% and still wastes {ms(rec('ORACLE', 'wasted_share'), '{:.1f}', 100)}%: the ALIVE rule (wasted <= 75%) is not reachable by any ranker of this form in this simulator.")
w("- Matcher (ALIKED + LightGlue on idea-1's cached keypoints) scores 4.05 image pairs/s here; 3x3 scoring of the 4 seeds is out of budget (D1.5).\n")

# ------------------------------------------------------------------ D1.0
w("## D1.0 Provenance and inputs\n")
rows = []
for s in SEEDS:
    p = P0[str(s)]
    rows.append([s, p["dir"].replace("/bhallukah/dd25007/archive/idea-1/runs/", "…/runs/"), p["checkpoint_50.pth.tar"]["epoch"], p["model_best.pth.tar"]["epoch"], "yes" if p["model_best_equals_ckpt50"] else "NO",
                 p["resume_epoch40"]["epoch"], f"{p['resume_epoch40']['n_ml']} / {p['resume_epoch40']['n_cl']}", p["phaseA"]["n_ml"].__str__() + " / " + str(p["phaseA"]["n_cl"]),
                 "yes" if p["same_arrays_exact"] and PS[str(s)]["equal"] else "NO", f"{PS[str(s)]['resume_ml'][:10]} / {PS[str(s)]['resume_cl'][:10]}"])
w(md(["seed", "run dir", "`checkpoint_50` epoch (stored)", "`model_best` epoch (stored)", "model_best == ckpt50 (every tensor)", "`resume_epoch40` epoch (stored)", "ML / CL in resume file", "ML / CL in Phase A file", "identical to Phase A (arrays and repo sha)", "repo `_pairs_sha` ML / CL (first 10)"], rows))
w("- Every seed has an epoch-50 `checkpoint_50.pth.tar` (the `epoch` stored inside is 50) and `model_best.pth.tar` is tensor-for-tensor identical to it, so the **last-epoch (epoch-50) model is used for all four seeds; there is no epoch mismatch to flag**. Seed 0 is the original C0 s0 run (`archive/idea-1/runs/C0_consam_det/logs`; `outputs/P2b_full_gate` is bit-identical per CLAUDE.md and was not needed). The `epoch` stored in `resume_epoch40.pth` is 40, the last AL cycle (epochs 0, 10, 20, 30, 40), so its ML/CL are the final constraint sets.")
w("- Phase A constraint files `/bhallukah/dd25007/phaseA/constraints/C0_s{0..3}_e40.npz` were extracted from these same resume files: the arrays are element-for-element equal (same order, int32 vs int64 only) and the sha computed with the repo's own `spcl.utils.run_record._pairs_sha` is identical for ML and CL of all four seeds. (A plain byte hash differs only because of the dtype; that is not a difference in content.) Phase A additionally recorded that the epoch-40 sha matches the logged `[record]` for all four seeds.")
w("- The 'epoch-50 model' pool features are a fresh inference with the clustering transform (`train_al.get_test_loader`, Resize 256x128 bicubic, no flip), **not** the momentum memory bank the run itself clusters from, so the unit counts differ somewhat from the training log's own last statistics (seed 0: 2,809 units here against 2,988 in the P2b log at epoch 49).\n")

# ------------------------------------------------------------------ D1.1
w("## D1.1 Does one answered pair merge two whole clusters?\n")
w("Code (`spcl/utils/constrained_clustering.py`, `constrained_clustering(labels, rerank_dist, method, ml_pairs, cl_pairs)`), as `train_al.py:704-711` calls it with `method=\"~DBSCAN\"` on `constrained_pseudo_labels`:\n")
w("""| endpoints of the ML pair | what the code does | lines |
|---|---|---|
| two different clusters | `labels[labels == cluster_b] = cluster_a`: **every** image carrying `b`'s label is relabelled, so the two clusters become one | 129-131 read the two labels, 140-141 the merge |
| a cluster and a singleton | same branch: in `train_al.py` a singleton is *not* `-1`. DBSCAN noise gets fresh ids (`train_al.py:594-605`) and reliability-filter rejects get `len(cluster_R_indep) + outliers` (`train_al.py:683-694`), so `labels` never contains `-1` on this path. The singleton's one-image label is merged into the cluster (or, if `a` is the singleton, the whole cluster is relabelled to the singleton's id: same partition) | 140-141 |
| two singletons | same branch; the result is a 2-image cluster. It survives, because the clean-up that sends clusters smaller than `min_samples` back to `-1` runs only for `method == "DBSCAN"` and the call uses `"~DBSCAN"` | 140-141; clean-up 319-323 skipped |
""")
w("- The `-1` branches (lines 132-139: both noise / one noise) and the initial `labels == -1` renumbering (122-125) are dead code on the training path (`method != \"DBSCAN\"`, no `-1` labels). That the 5,436 singletons of `labels_pre` are real labels and not `-1` is confirmed below.")
w("- The merge is not final: after all ML merges the cannot-link repair (167-272) splits any merged cluster that contains a CL pair along its ML-connected components (183-234; `[cc-diag]` of `b0_det_A` epoch 0: 198 of 4,453 clusters needed repair). A cluster without a CL pair inside stays merged.\n")
w("Empirical check on the `b0_det_A` epoch-0 state (real `labels_pre`, 20,565 images, 6,659 labels of which 5,436 are singletons and 732 clusters have >= 2 images; real `epoch0_rerank.npy`). One synthetic ML pair between two images of the **same GT identity**, no other constraint, through the repo function, 6 random draws per kind (`d11_merge_probe.py`, `out/d11_merge_probe.json`):\n")
rows = []
for k in ("cluster-cluster", "cluster-singleton", "singleton-singleton"):
    r = [x for x in PR if x["kind"] == k]
    rows.append([k, len(r), "; ".join(f"{x['size_a']}+{x['size_b']}→{x['size_after']}" for x in r), f"{sum(x['merged_all_images'] for x in r)}/{len(r)}", f"{sum(x['size_after'] == x['sum_sizes'] for x in r)}/{len(r)}", f"{sum(x['nothing_else_joined'] for x in r)}/{len(r)}"])
w(md(["kind", "draws", "sizes before → after (size_a + size_b → merged size)", "all images of both groups end in one label", "merged size == size_a + size_b", "no outside image joined"], rows))
imp = [(x['n_gt_ids_in_a'], x['n_gt_ids_in_b']) for x in PR if x["kind"] == "cluster-cluster"]
w(f"- All 18 draws merged **all** images of both groups and nothing else. In the cluster-cluster draws the two clusters held {', '.join(f'{a}+{b}' for a, b in imp)} distinct GT identities respectively: one correct 'same' answer also fuses every other identity inside the two clusters (a false merge it cannot detect). Conversely a correct answer between two *pure* fragments of one identity repairs the whole fragment at once, which is the mechanism behind 'needed merges' being counted per unit pair.\n")

# ------------------------------------------------------------------ D1.2
w("## D1.2 Label reconstruction\n")
w("`final_labels(features, ml, cl, indep_thres=None, rerank_dist=None)` in `tools/partner_d1/common.py`. **What is imported and what is mirrored:** `constrained_clustering` (from `spcl.utils.constrained_clustering`) and the rerank (`infra.fast_rerank.compute_jaccard_distance_fast`, exactly what `--fast-rerank` selects through `_fast.pick`) are imported, as are `get_data`, `create_model`, `get_test_loader` (feature extraction) from `train_al`. The statements between the rerank and the `constrained_clustering` call (3 x DBSCAN, `generate_pseudo_labels`, R_comp / R_indep, reliability filter; `train_al.py:461-711`) are **inline in `main_worker`, not a function**, so they cannot be imported; they are mirrored statement for statement, each block cites its source lines, and the exact-equality validation below is what stands in for 'imported'. `indep_thres` is the epoch-0 value carried in the resume file (`train_al.py:553-561` computes it at epoch 0 only); with `None` it is derived as epoch 0 does.\n")
w(md(["validation target", "result"], [
    ["(a) `b0_det_A` epoch 0 (a real `train_al.py` run, no `--fast-rerank`): `epoch0_features.npy` + cycle-1 ML/CL → rerank on GPU with the fast-rerank function", f"rerank `array_equal` to the saved `epoch0_rerank.npy`: **{V['a_rerank_array_equal']}** (max abs diff {V['a_rerank_max_abs_diff']}); preNP3 labels `array_equal` to `epoch0_labels_pre.npy`: **{V['a_pre_array_equal']}**; **post-NP3 labels `array_equal` to `epoch0_labels_post.npy`: {V['a_post_array_equal']}** ({V['a_post_n_diff']} differing entries); derived `indep_thres` {V['a_indep_thres_derived']:.7f} (= 1 − {V['a_indep_thres_1minus']:.7f}, the logged 'R_indep threshold')"],
    ["(b) kit epoch-95 state (`resume_epoch95.pth` features + 75,681 answers) vs `tools/fragment_sim/data/P.npz`, using the kit's saved CPU rerank", f"**`array_equal`: {V['b_with_kit_cpu_rerank_equal_P']}** ({V['b_with_kit_cpu_rerank_n_diff']} differing entries)"],
    ["(b') same state, rerank recomputed on GPU by the fast-rerank function", f"`array_equal`: {V['b_gpu_fast_rerank_equal_P']} — **{V['b_gpu_fast_rerank_n_diff']} of 20,565 labels differ** (ARI {V['b_gpu_vs_P_ARI']:.6f}); the GPU and the kit's CPU rerank matrices are not identical (max abs diff {V['b_gpu_vs_kit_rerank_max_abs_diff']:.4f}), which is the kit's own documented 'CPU port, identical up to float rounding'"],
]))
w("Verdict: the target 'reproduce a saved label array exactly' is met by (a), the real-run state, end to end including the GPU rerank, and by (b) given the same rerank. (b') shows the only residual: CPU-vs-GPU rerank rounding in the kit's P, 2 labels. No STOP was needed.\n")

# ------------------------------------------------------------------ D1.3
w("## D1.3 Fragmentation on the C0 pipeline, 4 seeds\n")
w("State per seed: epoch-50 model → pool features (20,565 images, clustering transform) → `final_labels` with that seed's epoch-40 constraints. Definitions (kit: `headroom/d1`, `headroom/d2`): **unit** = a label of the final pseudo-labels (singletons included); a unit's dataset = its majority dataset; **fragment** = for each GT identity split over m units of its dataset, every unit except the one holding most of its images; **needed merges** = sum over identities of (units holding it − 1) (= number of fragments); **false split** = a GT-same pair of images in different units (singleton included); **false merge** = a same-unit pair with different GT identity; pair counts are within-dataset. **Agree-wrong** = a wrong pair (false split or false merge, queried or not) that was **not queried** and on which both ConSam views (V1 constrained DBSCAN, V2 constrained FINCH, from the real `ConsistencySampling.select` run with `n = 0`) agree with the pseudo-label's wrong relation; its share is taken over *all* wrong pairs including the queried ones (as in D1: 55,960 / 63,611). Units per seed: " +
  ", ".join(f"s{s} {D3[s]['glob']['n_units']} ({D3[s]['glob']['n_singletons']} singletons)" for s in SEEDS) + f"; kit epoch-95 state {K95['glob']['n_units']} ({K95['glob']['n_singletons']}).\n")
w("**Recomputation check.** The kit's epoch-95 state, run through the same code: needed merges Cat 554, MultiCamCows2024 102, PrimFace 39, PolarBearVidID 88 (the kit's numbers), 1,398 fragments in total (= the 1,398 rows of `ref/d2_fragments.json`); the recomputed V1/V2 equal the shipped `data/views.npz` exactly; the primary-datasets wrong pairs are 63,611 of which agree-wrong 55,960 (88.0%) and disagree-unqueried 1,519, identical to the archived D1 report.\n")
def tab_D13(key):
    rows = []
    for d in DS + ["PRIMARY (Cat+MCC+PrimFace)", "ALL 13"]:
        sel = [d] if d in DS else (PRIMARY if d.startswith("PRIMARY") else DS)
        def v(src): return {"needed": SELECT(sel, src["frag"], "needed_merges"), "fs": SELECT(sel, src["frag"], "false_split_pairs"), "fm": SELECT(sel, src["frag"], "false_merge_pairs"),
                            "aw": SELECT(sel, src["frag"], "c_agree_wrong") / max(1, SELECT(sel, src["frag"], "wrong_pairs"))}
        per = [v(D3[s]) for s in SEEDS]; kk = v(K95)
        rows.append([d, ", ".join(str(p["needed"]) for p in per), ms([p["needed"] for p in per], "{:.1f}"), ms([p["fs"] for p in per], "{:.0f}"), ms([p["fm"] for p in per], "{:.0f}"), ms([p["aw"] for p in per], "{:.1f}", 100) + " %",
                     f"{kk['needed']} / {kk['fs']} / {kk['fm']} / {pct(kk['aw'])} %"])
    return md(["dataset", "needed merges, seeds 0-3", "needed merges, mean ± sd", "false-split pairs", "false-merge pairs", "agree-wrong share of wrong pairs", "kit epoch-95 (needed / FS / FM / agree-wrong)"], rows)
w(tab_D13(None))
w("Notes: mean ± sd with ddof = 1 over the 4 seeds. Pair counts are exact (contingency tables). The datasets with few images (turtles, Amvrakikos 21 images, Zakynthos 21) have needed-merge counts of 5-40; their agree-wrong shares rest on a handful of wrong pairs and can be 0/100% by construction.")
nm_split = [(sum(D3[s]['frag'][d]['false_split_pairs'] for d in DS), sum(D3[s]['frag'][d]['false_merge_pairs'] for d in DS)) for s in SEEDS]
w(f"- False splits outnumber false merges {ms([a / b for a, b in nm_split], '{:.1f}')}:1 in pairs pooled over the 13 datasets (kit epoch-95: {SELECT(DS, K95['frag'], 'false_split_pairs') / SELECT(DS, K95['frag'], 'false_merge_pairs'):.1f}:1). The 4-seed epoch-50 states are more fragmented than the kit's 100-epoch state: {ms(need_all, '{:.0f}')} vs {SELECT(DS, K95['frag'], 'needed_merges')} needed merges.\n")

# ------------------------------------------------------------------ D1.4
w("## D1.4 Partner-rank baseline B\n")
w("For every fragment (one row per dataset / identity / non-largest unit, as in `ref/d2_fragments.json`; rows without a partner unit in the same dataset are left out: kit 1 of 1,398, seeds " + ", ".join(str(sum(D3[s]['n_fragments'].values()) - D3[s]['n_fragment_rows_with_partner']) for s in SEEDS) + f" of {', '.join(str(sum(D3[s]['n_fragments'].values())) for s in SEEDS)}), the nearest true partner is ranked among **all other units of the same dataset** by cosine distance of unit centroids (C0 epoch-50 features). Rank-1 = the nearest unit is a partner; top-5 = a partner is within the 5 nearest; 'in 20' = a partner is within the 20 nearest (the candidate list of D1.5). Query = the fragment's unit centroid (the kit's step 1 used the fragment's own sub-centroid; identical for pure units).\n")
rows = []
for d in DS + ["PRIMARY", "ALL"]:
    t = [D3[s]["partner_rank"][d] for s in SEEDS]
    label = {"PRIMARY": "PRIMARY (Cat+MCC+PrimFace)", "ALL": "ALL 13"}.get(d, d)
    kk = K95["partner_rank"][d]
    rows.append([label + (" **(Cat)**" if d == "CatIndividualImages" else ""), ", ".join(str(x["n"]) for x in t), ms([x["top1"] for x in t], "{:.1f}", 100), ms([x["top5"] for x in t], "{:.1f}", 100),
                 ms([x["median_rank"] for x in t], "{:.1f}"), ms([x["top20"] for x in t], "{:.1f}", 100), f"{100*kk['top1']:.1f} / {100*kk['top5']:.1f} / {kk['median_rank']:.0f}" if kk else "-"])
w(md(["dataset", "fragments n (seeds 0-3)", "rank-1 %", "top-5 %", "median rank (all same-dataset units)", "partner in 20 nearest %", "kit epoch-95: rank-1 / top-5 / median"], rows))
w("Per-seed values behind the Cat row and the pooled rows are in `out/d13_s{0..3}.json` (`partner_rank`). Ties of exactly equal centroid distance (a few identical-feature images in Cat) are broken by unit index here; the Gate-1 table in D1.5 counts ties against the ranker, which moves 3 dataset-seed cells by 1-2 fragments (Cat s0, s3).\n")

# ------------------------------------------------------------------ D1.5
w("## D1.5 Matcher ranker (Gate 1)\n")
w("### 1. Which matcher, and is it on disk\n")
w("- idea-1's closing sources (tag `idea-1-final`, `idea-1/lf_b/lf_b_report.md`) compared four variants on ConSam's cycle-1 questions: SuperPoint+LightGlue 9.61%, DISK+LightGlue 9.16%, ALIKED production features (matches with score >= 0.9) 9.16%, ALIKED no-upscaling 8.65%, ALIKED default match count 8.77% (share of held-out pairs answered at >= 97% precision; none reached the 10% line; the report notes all are within one point). idea-1's production feature set (the Phase-B 'production' config and the pool-wide cache `mv/cache/aliked_native.h5`, `lg_common.build('aliked')`) is **ALIKED + LightGlue**, the default named in the task. SuperPoint+LightGlue is nominally 0.45 points above ALIKED production (9.61 vs 9.16, one deterministic run per variant, no significance claim in the report), but its features are cached only for the 13,357 images that ConSam's cycle-1 pairs touch (`lf_b/cache/feats_superpoint`, per the LF-B report), not for the pool. **ALIKED + LightGlue is used.**")
w("- On disk: LightGlue ALIKED weights `/storage/dd25007/cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth`, `aliked-n16.pth`; idea-1's cached native-resolution ALIKED keypoints/descriptors (<= 2048 per image) for **all 20,565 pool images**, `/bhallukah/dd25007/HITL-ReID-idea1-cache/mv/cache/aliked_native.h5` (12.9 GB). The index of the cache matches the pool row order: image sizes stored in the cache equal the file sizes of the pool images for 60 of 60 random images (`d15_match.py` asserts it). So no extraction cost.")
w("- 'Inliers' = the number of LightGlue matches of the pair at the matcher's default threshold (`n_matches`, idea-1's own score; `mean_score` also kept). No geometric verification (RANSAC) was part of idea-1's score and none is added.\n")
w("### 2-3. Candidates and timing\n")
w("Candidate unit pairs = (fragment unit, one of its 20 nearest same-dataset units by centroid cosine), de-duplicated; representatives per unit = medoid + the 2 images nearest to the medoid (all images if the unit is smaller than 3); 3 x 3 image pairs per unit pair (`d15_candidates.py`). **Timing first:** 1,000 image pairs drawn at random from seed 0's 114,680, after 20 warm-up pairs, unbatched, features read from RAM, one H200 (GPU 3, shared with other tenants, slot-locked): " +
  f"**{TM['pairs_per_s']:.2f} pairs/s** ({TM['seconds']:.0f} s for 1,000; mean {TM['mean_matches']:.0f} matches/pair, median {TM['median_matches']:.0f}). idea-1's own production run of the same matching measured 5.53 pairs/s (mv2_match, 19.7 h for 392,032 pairs).\n")
rate = TM["pairs_per_s"]; rows = []
for t in ["s0", "s1", "s2", "s3"]:
    c = CNT[t]; rows.append([t, c["n_fragment_units"], f"{c['n_unit_pairs']:,}", f"{c['n_image_pairs_3x3']:,}", f"{c['n_image_pairs_3x3'] / rate / 3600:.2f}", f"{c['n_image_pairs_3x3'] / 5.53 / 3600:.2f}", f"{c['n_image_pairs_medoid_only'] / rate / 3600:.2f}"])
tot3 = sum(CNT[t]["n_image_pairs_3x3"] for t in CNT if t.startswith("s")); totm = sum(CNT[t]["n_image_pairs_medoid_only"] for t in CNT if t.startswith("s"))
rows.append(["**4 seeds**", "", f"{totm:,}", f"**{tot3:,}**", f"**{tot3 / rate / 3600:.1f}**", f"{tot3 / 5.53 / 3600:.1f}", f"**{totm / rate / 3600:.1f}**"])
w(md(["seed", "fragment units", "unit pairs (= medoid-only image pairs)", "3x3 image pairs", "GPU-h at the measured 4.05 pairs/s", "GPU-h at idea-1's 5.53 pairs/s", "GPU-h, medoid-only at 4.05 pairs/s"], rows))
w(f"**Cost stop.** The specified 3 x 3 scoring projects **{tot3 / rate / 3600:.1f} GPU-h** for the 4 seeds ({tot3 / 5.53 / 3600:.1f} GPU-h even at idea-1's faster rate), above the 20 GPU-h cap of this phase (about 1.0 GPU-h is already spent). Per the instruction I stopped before scoring: **R_LF and R_LFC were not run**, and neither were their D1.6 simulations. "
  f"Cheaper variant I propose (not run): **medoid-only** (one image pair per unit pair, {totm:,} pairs, **{totm / rate / 3600:.1f} GPU-h** for the 4 seeds at the measured rate, 3.8x cheaper); a middle option is 2 x 2 (medoid + nearest image). Both keep the specified candidate lists, logistic fit on the seed's own answered pairs, and decision rules unchanged. Needs your go-ahead (rule 3 only applies above 20 GPU-h, but this changes the preregistered scoring design).\n")
w("### 4-5. Rank-1 / top-5 inside the 20 nearest, per dataset\n")
w("R_LF / R_LFC: **not run** (cost stop above). R_DINO (optional control; frozen DINOv2-large CLS embedding, 224 x 224, unit centroid cosine, re-ranking the same 20 candidates; weights from the local HF cache; ~11 min of one card). Ties are counted against the ranker; `n` is seed 0's fragment count (other seeds differ by a few). Fragments whose partner is not among the 20 nearest are misses for every ranker (ceiling = 'in 20').\n")
rows = []
Gate = {}
for d in DS + ["PRIMARY", "ALL"]:
    B = [SM[s]["gate"]["B"][d] for s in SEEDS]; R = [SM[s]["gate"]["R_DINO"][d] for s in SEEDS]
    dl = [100 * (r["top1"] - b["top1"]) for r, b in zip(R, B)]
    m_r1, m_b1 = np.mean([r["top1"] for r in R]), np.mean([b["top1"] for b in B])
    ok = (m_r1 >= 0.5) and (m_r1 >= m_b1 + 0.15) and all(x > 0 for x in dl)
    Gate[d] = dict(pass_=bool(ok), mean_r1=m_r1, mean_b1=m_b1, deltas=dl, n=B[0]["n"])
    label = {"PRIMARY": "PRIMARY (Cat+MCC+PrimFace)", "ALL": "ALL 13"}.get(d, d)
    rows.append([label, B[0]["n"], ms([b["top1"] for b in B], "{:.1f}", 100), ms([r["top1"] for r in R], "{:.1f}", 100), ", ".join(f"{x:+.1f}" for x in dl), ms([b["top5"] for b in B], "{:.1f}", 100), ms([r["top5"] for r in R], "{:.1f}", 100), ms([b["in_top20"] for b in B], "{:.1f}", 100), "PASS" if ok else "fail"])
w(md(["dataset", "n", "B rank-1 %", "R_DINO rank-1 %", "R_DINO − B per seed (pp)", "B top-5 %", "R_DINO top-5 %", "partner in 20 (ceiling) %", "Gate 1 for R_DINO (control)"], rows))
w("Gate 1 as preregistered (4-seed mean rank-1 >= 50% **and** >= mean B + 15 pp **and** every seed above its own B) applied mechanically to the control; the row for R_LF / R_LFC is 'not run'.\n")

# ------------------------------------------------------------------ D1.6
w("## D1.6 Merge-recovery simulation\n")
w("### 1. Model-disabled check (kit)\n")
chk = open("/bhallukah/dd25007/phaseD1/logs/d16_check_reproduction.log").read()
w("`tools/fragment_sim/check_reproduction.py` on the epoch-95 state (FRAGSIM_WORK redirected to `out/kit_work/`): " + re.search(r"RESULT (\{.*\})", chk).group(1) + " — H recomputed equals shipped H, `sim_d3.py` rule M reproduces D2 within/M exactly at all 6 budgets, and the D4 simulator with the model disabled (`s_adaptive.py off`) reproduces it too (`model disabled reproduces D2 within/M exactly … True`). **Passed; no STOP.**")
w(f"The simulator used below is a state-parametrised copy of `sim_d3.simulate` (`sim_d1.py`; `sim_d3.py` itself runs on import against the kit state and cannot be called for other states). Check of the copy: on the kit state with order B it reproduces `ref/d2_sim.json` within/M exactly (recovered per dataset, wasted and free merges at all 6 budgets): **{CC['copy_reproduces_D2_within_M']}** ({CC['summary_5000']['recovered_primary']}/{CC['summary_5000']['needed_primary']} primary, {CC['summary_5000']['recovered_all']}/{CC['summary_5000']['needed_all']} all, {CC['summary_5000']['wasted']} wasted at 5,000).\n")
w("### 2. Simulation on each seed's state\n")
w("Same protocol as the kit: a noiseless GT oracle answers one medoid image pair per unit pair, within-dataset pairs only, transitivity on the existing answers, a 'same' answer merges the two units. Orders: **B** = centroid cosine (rule M); **ORACLE** = free upper bound for any ranker that only re-orders the 20-nearest candidate pairs (candidate pairs whose two units share a GT identity first, B order inside, then all other pairs in B order); **R_DINO** = candidate pairs by DINOv2 centroid cosine, then the rest in B order; **R_DINO_g** = all within-dataset pairs by DINOv2 centroid cosine. R_LF / R_LFC: not run. 'Recovered' pooled over **all** needed merges (the preregistered denominator); 'primary' = Cat + MultiCamCows2024 + PrimFace (the kit's headline). Wasted = answers 'different' / questions asked. For every ranker and seed the candidate list (784k-867k within-dataset unit pairs) was not exhausted before 42,289 questions (`exhausted_at` is null in all 16 runs).\n")
kinds = ["B", "ORACLE", "R_DINO", "R_DINO_g"]
rows = []
for k in kinds:
    sm = [SM[s]["summary_5000"][k] for s in SEEDS]
    rows.append([k, ", ".join(pct(x["recovery_all"]) for x in sm), ms([x["recovery_all"] for x in sm], "{:.1f}", 100), ms([x["recovery_primary"] for x in sm], "{:.1f}", 100), ms([x["wasted_share"] for x in sm], "{:.1f}", 100)])
w("**At 5,000 questions:**\n")
w(md(["ranker", "recovery % of all needed merges, seeds 0-3", "mean ± sd", "primary-3 recovery % mean ± sd", "wasted % of questions, mean ± sd"], rows))
w("**Across budgets** (recovery % of all needed merges / wasted % of the questions asked, mean ± sd over 4 seeds; the candidate list was not exhausted at any budget):\n")
rows = []
for k in kinds:
    r = [k]
    for b in ["500", "1000", "2000", "5000", "10000", "42289"]:
        rr = []; ww = []
        for s in SEEDS:
            sn = SM[s]["sim"][k]["snapshots"][b]; rr.append(sum(sn["recovered"].values()) / sum(SM[s]["need"].values())); ww.append(sn["wasted"] / sn["queries"])
        r.append(f"{ms(rr, '{:.1f}', 100)} / {ms(ww, '{:.1f}', 100)}")
    rows.append(r)
w(md(["ranker", "500", "1,000", "2,000", "5,000", "10,000", "42,289"], rows))
w("**Per dataset at 5,000 questions** (recovered / needed merges, %; mean ± sd over seeds):\n")
rows = []
for d in DS + ["PRIMARY", "ALL"]:
    r = [{"PRIMARY": "PRIMARY (Cat+MCC+PrimFace)", "ALL": "ALL 13"}.get(d, d), ms([sum(SM[s]["need"][x] for x in (PRIMARY if d == "PRIMARY" else DS if d == "ALL" else [d])) for s in SEEDS], "{:.1f}")]
    for k in kinds:
        v = []
        for s in SEEDS:
            pdv = SM[s]["summary_5000"][k]["per_dataset"]; sel = PRIMARY if d == "PRIMARY" else DS if d == "ALL" else [d]
            nn = sum(pdv[x]["needed"] for x in sel); v.append(sum(pdv[x]["recovered"] for x in sel) / nn if nn else float("nan"))
        r.append(ms(v, "{:.1f}", 100))
    rows.append(r)
w(md(["dataset", "needed merges (mean)"] + kinds, rows))

# ------------------------------------------------------------------ D1.7
w("## D1.7 Pre-registered decision rules (applied mechanically)\n")
Bs = [SM[s]["summary_5000"]["B"]["recovery_all"] for s in SEEDS]
def verdict(k):
    rs = [SM[s]["summary_5000"][k]["recovery_all"] for s in SEEDS]; ws = [SM[s]["summary_5000"][k]["wasted_share"] for s in SEEDS]
    d10 = sum(1 for r, b in zip(rs, Bs) if r >= b + 0.10); md_ = np.mean(rs) - np.mean(Bs)
    if np.mean(rs) < np.mean(Bs) + 0.10 or d10 < 3: v = "DEAD"
    elif np.mean(rs) >= np.mean(Bs) + 0.15 and np.mean(ws) <= 0.75: v = "ALIVE"
    else: v = "WEAK"
    return v, np.mean(rs), md_, d10, np.mean(ws)
rows = []
for dname in ["CatIndividualImages", "MultiCamCows2024", "PrimFace", "PolarBearVidID", "SeaStarReID2023"]:
    g = Gate[dname]
    rows.append([f"Gate 1, {dname}: R_LF / R_LFC", "mean rank-1 >= 50%, >= B + 15 pp, every seed > own B", f"B {100 * g['mean_b1']:.1f}%", "**NOT RUN** (cost stop, D1.5)"])
    rows.append([f"Gate 1, {dname}: R_DINO (control only)", "same", f"rank-1 {100 * g['mean_r1']:.1f}% (B {100 * g['mean_b1']:.1f}%, Δ per seed " + ", ".join(f"{x:+.1f}" for x in g["deltas"]) + ")", "PASS" if g["pass_"] else "fail"])
passing = [d for d in DS if Gate[d]["pass_"]]
share_pass = [sum(SM[s]["need"][d] for d in passing) / sum(SM[s]["need"].values()) for s in SEEDS]
rows.append(["Recovery @5,000, R_LF / R_LFC", "DEAD < B+10 pp (or < 3 of 4 seeds ≥ B+10); ALIVE ≥ B+15 pp and wasted ≤ 75%", f"B_rec {ms(Bs, '{:.1f}', 100)}%", "**NOT RUN** (cost stop, D1.5)"])
for k in ["R_DINO", "R_DINO_g", "ORACLE"]:
    v, m, dlt, n10, wst = verdict(k)
    rows.append([f"Recovery @5,000, {k}" + (" (control)" if k.startswith("R_D") else " (upper bound, not a method)"), "same", f"mean {100 * m:.1f}% = B {dlt * 100:+.1f} pp; seeds ≥ B+10 pp: {n10}/4; wasted {100 * wst:.1f}%", v])
dec_rows = rows
w("The rules are applied in the decision table at the end of this report (one row per rule and ranker); notes:\n")
gc = Gate["CatIndividualImages"]
fails = []
if gc["mean_r1"] < 0.5: fails.append(f"mean rank-1 {100 * gc['mean_r1']:.1f}% < 50%")
if gc["mean_r1"] < gc["mean_b1"] + 0.15: fails.append(f"mean rank-1 not >= B + 15 pp (B {100 * gc['mean_b1']:.1f}%)")
if not all(x > 0 for x in gc["deltas"]): fails.append("not every seed above its own B")
w(("- **Cat " + ("fails" if not gc["pass_"] else "passes") + " Gate 1 for the R_DINO control**" + (": " + "; ".join(fails) + f" (mean gain over B {np.mean(gc['deltas']):+.1f} pp; per seed " + ", ".join(f"{x:+.1f}" for x in gc["deltas"]) + ")" if fails else "") + ". "
   + f"Datasets that pass Gate 1 for R_DINO: {', '.join(passing) if passing else 'none'}; they hold {ms(share_pass, '{:.1f}', 100)}% of the needed merges (mean ± sd over seeds; Cat alone holds {ms([SM[s]['need']['CatIndividualImages'] / sum(SM[s]['need'].values()) for s in SEEDS], '{:.1f}', 100)}%). For R_LF / R_LFC this share cannot be given because they were not run."))
w("- These verdicts are for the **control and the oracle bound only**. The preregistered verdict on the matcher (Gate 1 and the recovery rule for R_LF / R_LFC) is **open**, not DEAD / WEAK / ALIVE.")
gap = np.mean(rec('ORACLE')) - np.mean(Bs)
w("- What the oracle row says about the rule itself: a ranker that puts every identity-sharing candidate pair first still has wasted " + ms(rec('ORACLE', 'wasted_share'), '{:.1f}', 100) + "% at 5,000 questions, above the 75% line, so 'wasted <= 75%' is not reachable by re-ranking the 20-nearest lists under this protocol (medoid query, fall-back to centroid order). The oracle's recovery is flat at about " + ms([sum(SM[s]['sim']['ORACLE']['snapshots']['2000']['recovered'].values()) / sum(SM[s]['need'].values()) for s in SEEDS], '{:.1f}', 100) + "% from 2,000 questions on, and its wasted share is already " + ms([SM[s]['sim']['ORACLE']['snapshots']['2000']['wasted'] / 2000 for s in SEEDS], '{:.1f}', 100) + "% at 2,000 (table above). I did **not** isolate the cause (fall-back queries after the identity-sharing candidates run out versus medoid pairs of impure units that answer 'different' although the units share an identity). The recovery threshold B + 15 pp needs a ranker that captures " + f"{100 * 0.15 / gap:.0f}" + "% of the B → oracle headroom (" + f"{100 * gap:+.1f}" + " pp); R_DINO captures " + f"{100 * (np.mean(rec('R_DINO')) - np.mean(Bs)) / gap:.0f}" + "%.")
w("- **Even an ALIVE verdict would only earn an end-to-end 4-seed run; offline recovery has never predicted mAP in this project.**")
w("- No p-values or Hedges g are given: nothing here is a between-arm mAP contrast; the rules are thresholds on 4-seed means.\n")

# ------------------------------------------------------------------ D1.8
w("## D1.8 Cost table for Phase D2 (no runs)\n")
w("Measured wall-clock per 50-epoch run (1 card each, so GPU-h = wall-clock h). C0 values are read from the run logs (first to last timestamp, `Total running time`); E1 and skyline text logs are not on disk (their run directories hold only evaluation files), so their wall-clocks are the watcher figures printed in the archived reports `headroom/e1/REPORT.md` and `headroom/h1/REPORT.md` (first launch to last log write, includes the gap before the resume).\n")
w(md(["run", "wall-clock = GPU-h (1 card)", "source / conditions"], [
    ["C0 s0 clean pass (`P2b_full_gate`, killed in epoch 14 and resumed)", "8.8 h incl. 3.5 redone epochs (8.3 h one clean pass)", "P2b `attempt_0.log` 2026-10-03T23:14:54 → `attempt_1.log` 2026-10-04T08:01:22 (one card, `P2b_full_gate`); 8.3 h clean-pass figure from the archived H0 findings"],
    ["C0 s1 / s2 / s3", "16.1 h / 15.8 h / 15.5 h", "archived idea-1 `attempt_*.log`: `Total running time` 16:08:26, 15:45:58, 15:27:11; three runs concurrent on a heavily loaded node"],
    ["E1 oracle merge (`e1_oracle`, seed 0, resumed 1x)", "14.33 h", "archived E1 REPORT (GPU 5; the E1 cost-guard measurement saw load average 59)"],
    ["`--full-supervision` skyline 50 epochs (`skyline50`, seed 0, no resume)", "18.03 h", "archived H1 REPORT (the H1 operator note reports load average ~126 at launch); the H0 estimate for an unloaded card was ~7.4 h (6.8-8.3), an estimate, not a measurement"],
]))
e1, sk = 14.33, 18.03
w(md(["Phase D2 item", "runs", "GPU-h at the measured (loaded) speed", "GPU-h at the unloaded reference (estimate)"], [
    ["E1 oracle merge, seeds 1-3", "3 x 1 card", f"3 x {e1} = **{3 * e1:.1f}**", "3 x 8.3 = 24.9 (E1 is plain ConSam + a label merge; same compute as a C0 pass)"],
    ["like-for-like skyline (`--full-supervision`, 50 epochs), seeds 1-3", "3 x 1 card", f"3 x {sk} = **{3 * sk:.1f}**", "3 x 7.4 = 22.2 (H0 estimate)"],
    ["**total**", "6 runs", f"**{3 * e1 + 3 * sk:.1f}**", "47.1"],
]))
w("Both totals are above the 20 GPU-h line, so CLAUDE.md rule 3 (cost table, wait for approval) applies to D2. Caveats: seed 1-3 runs of E1 and skyline have not been run, so these are seed-0 costs times 3; H0 findings list the determinism of the `--full-supervision` path as untested.\n")
gst = subprocess.run("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits", shell=True, capture_output=True, text=True).stdout.strip().splitlines()
apps = subprocess.run("nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip().splitlines()
buses = {l.split(",")[0].strip(): l.split(",")[1].strip() for l in subprocess.run("nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip().splitlines() and [] or []}
beca_gpus = set()
bus2idx = {l.split(",")[1].strip(): l.split(",")[0].strip() for l in subprocess.run("nvidia-smi --query-gpu=index,gpu_bus_id --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip().splitlines()}
for l in apps:
    bus, pid = [x.strip() for x in l.split(",")]
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
        if "beca-al" in cwd: beca_gpus.add(bus2idx.get(bus))
    except OSError: pass
live = sh("BECA_LOCK_DIR=/bhallukah/dd25007/gpu_locks python3 scripts/gpu_locks.py busy")
rows = []
for l in gst:
    i, fr = [x.strip() for x in l.split(",")]
    rows.append([i, f"{int(fr) / 1024:.1f}", "yes" if int(fr) >= 16384 else "no", "beca-al process on this card → excluded" if i in beca_gpus else "", "live slot" if i in live.split() else "no live slot"])
w(f"GPU status at report time ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}; free memory moves; locks at `/bhallukah/dd25007/gpu_locks`, `gpu_locks.py busy` printed `{live or '(nothing)'}`):\n")
w(md(["GPU", "free GiB", "≥ 16 GiB gate", "note", "slot lock"], rows))

# ------------------------------------------------------------------ decision table / not measured / artifacts
w("## Decision table\n")
w(md(["rule", "threshold", "value", "verdict"], dec_rows))
w("## Not measured, and why\n")
w("""- **R_LF and R_LFC (matcher rankers), their Gate-1 rank-1 / top-5 and their D1.6 recovery: not measured.** 3x3 scoring of the 4 seeds projects above the 20 GPU-h cap (D1.5 cost table) and the instruction was to stop and propose a cheaper variant; the cheaper variant was not run without your go-ahead. So the matcher's Gate 1 / DEAD-WEAK-ALIVE verdicts are open. Consequently also not fitted: the logistic model of R_LFC on the seed's own answered pairs.
- **'Inliers'** are LightGlue match counts (idea-1's score); no RANSAC-verified inlier count exists or was computed.
- **R_DINO is a control with a simple design choice** (DINOv2-large CLS, 224 x 224 resize, unit-centroid cosine, no fine-tuning). Other DINOv2 sizes, patch-token pooling or crops were not tried; a stronger or weaker control is possible.
- **The oracle row is a bound for rankers that only re-order the 20-nearest lists with the kit's medoid query**; it is not a method and says nothing about other question protocols (e.g. querying the closest image pair instead of medoids, rule C of D2, was not re-run).
- **Features are a fresh inference with the epoch-50 weights, not the training run's momentum memory bank**; the training log's own cluster statistics (e.g. seed 0 epoch 49: 1,805 clusters / 1,183 un-clustered) differ from the states analysed here (seed 0: 1,753 / 1,056). The states are 'what a re-clustering with the final model and the final answers would produce', not the labels the last training epoch used. Seeds 1-3 logs were not compared for this reason.
- **The re-clustering block of `train_al.py` is mirrored, not imported** (inline code); equality was proven on one real run (b0_det_A epoch 0, `array_equal`) and on the kit state with the kit's rerank; it was not re-proven on a C0 seed's epoch-50 state because no saved label array exists for it.
- **D1.1 is one state, one pair at a time, with no CL constraints** (epoch 0 of `b0_det_A`); interaction with CL repair on later cycles was not measured, only read from the code.
- **Partner rank uses unit centroids**; the kit's D2 step 1 used the fragment's own sub-centroid (identical for pure units). Min-linkage ranks on the rerank matrix (the kit's second distance) were not recomputed.
- **Oracle definition**: two units \"share an identity\" if any GT identity occurs in both; cross-dataset units (always wrong; 59 in the kit state) are counted in no per-dataset table.
- **E1 and skyline wall-clocks** come from archived watcher reports, not from re-reading their text logs (not on disk); seeds 1-3 of E1/skyline do not exist yet, so the D2 totals are seed-0 cost x 3. Unloaded-card figures are estimates (H0 findings), not measurements.
- **No mAP, p-value or Hedges g** is reported: nothing in this phase is an arm-vs-arm mAP comparison. n = 4 seeds, sd with ddof = 1, rules applied to means; a verdict on 4 seeds is a screen, not a test.
- **Free-GPU table** is a snapshot; foreign jobs change it minute to minute.
""")
w("## Artifacts under `/bhallukah/dd25007/phaseD1/`\n")
w("""- `scripts/` — copies of all code of this phase (`tools/partner_d1/*.py`, `run_gpu.sh`) and the driver `scripts_driver.sh` (also at the top level).
- `logs/` — every run log (`d12_validate.log`, `d13_features.log`, `d13_analyze_s{0..3}.log`, `d13_kit95.log`, `d15_timing_s0.log`, `d15_dino.log`, `d16_s{0..3}.log`, `d16_check_reproduction.log`, `untar*.log`).
- `out/` — `d10_provenance.json`, `d10_repo_sha.json`, `d11_merge_probe.json`, `d12_validate.json`, `feat_s{0..3}.npy` (epoch-50 pool features), `P_s{0..3}.npy` (final labels per seed), `views_*.npz` (V1/V2), `d13_*.json` (fragmentation + partner ranks), `cand_*.npz` and `d15_candidate_counts_*.json` (matcher candidates), `d15_timing_s0.json`, `dino_feat.npy`, `d16_s{0..3}.json` (simulations), `d16_copy_check.json`, `d17_decisions.json`, `kit_work/` (kit reproduction-check outputs).
- `arch_x/` — the few report/prereg files extracted from the diagnostics archive to read the D1/D2 definitions (read-only reference).
""")
text = "\n".join(out)
open(os.path.join(REPO, "reports", "phaseD1_partner.md"), "w").write(text)
json.dump(dict(gate={d: dict(pass_=g["pass_"], mean_r1=g["mean_r1"], mean_b1=g["mean_b1"], deltas=g["deltas"]) for d, g in Gate.items()}, verdicts={k: verdict(k)[0] for k in ["R_DINO", "R_DINO_g", "ORACLE"]}), open(f"{OUT}/d17_decisions.json", "w"), indent=1)
print("written", len(text))
