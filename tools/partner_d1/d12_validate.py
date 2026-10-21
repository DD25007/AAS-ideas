"""D1.2: final_labels must reproduce saved label arrays exactly (np.array_equal).
 (a) b0_det_A epoch 0 (a real train_al.py run): epoch0_features.npy + cycle1 ml/cl  vs epoch0_labels_post.npy (+ preNP3, rerank).
 (b) kit epoch-95 state: resume_epoch95 memory_features + ml/cl vs tools/fragment_sim/data/P.npz (kit port computed on CPU rerank)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
B = "/bhallukah/dd25007/HITL-ReID-idea2-outputs/b0_det_A/state"
res = {}
# ---- (a)
feat = torch.from_numpy(np.load(f"{B}/epoch0_features.npy")).float()
ml = [tuple(map(int, p)) for p in np.load(f"{B}/cycle1_ml.npy")]; cl = [tuple(map(int, p)) for p in np.load(f"{B}/cycle1_cl.npy")]
log("b0_det_A features", tuple(feat.shape), "ml", len(ml), "cl", len(cl), "norms", float(feat.norm(dim=1).min()), float(feat.norm(dim=1).max()))
saved_rr = np.load(f"{B}/epoch0_rerank.npy"); saved_pre = np.load(f"{B}/epoch0_labels_pre.npy"); saved_post = np.load(f"{B}/epoch0_labels_post.npy")
post, aux = final_labels(feat.cuda(), ml, cl, indep_thres=None, return_all=True)
rr = aux["rerank"]
res["a_rerank_array_equal"] = bool(np.array_equal(rr, saved_rr)); res["a_rerank_max_abs_diff"] = float(np.abs(rr - saved_rr).max())
res["a_pre_array_equal"] = bool(np.array_equal(aux["pre"], saved_pre))
res["a_post_array_equal"] = bool(np.array_equal(post, saved_post)); res["a_post_n_diff"] = int((post != saved_post).sum())
res["a_indep_thres_derived"] = aux["indep_thres"]; res["a_indep_thres_1minus"] = 1 - aux["indep_thres"]
log("(a)", json.dumps(res))
if not res["a_post_array_equal"]:
    # diagnose: same with the saved rerank
    post2 = final_labels(feat, ml, cl, indep_thres=None, rerank_dist=saved_rr)
    res["a_post_with_saved_rerank_equal"] = bool(np.array_equal(post2, saved_post)); log("(a) with saved rerank:", res["a_post_with_saved_rerank_equal"])
json.dump(res, open(f"{OUT}/d12_validate.json", "w"), indent=1)
# ---- (b)
st = torch.load("/bhallukah/dd25007/AAS-ideas-out/kit/resume_epoch95.pth", map_location="cpu", weights_only=False)
f95 = st["memory_features"].float(); ml95 = [tuple(map(int, p)) for p in st["ml_pairs"]]; cl95 = [tuple(map(int, p)) for p in st["cl_pairs"]]
P = np.load(f"{KIT_DATA}/P.npz")["P"]; thr = float(st["indep_thres"])
log("(b) epoch", int(st["epoch"]), "ml", len(ml95), "cl", len(cl95), "indep_thres", thr)
kit_rr = np.load("/bhallukah/dd25007/AAS-ideas-out/kit/rerank_e95.npy")
pb_saved = final_labels(f95, ml95, cl95, indep_thres=thr, rerank_dist=kit_rr)
res["b_with_kit_cpu_rerank_equal_P"] = bool(np.array_equal(pb_saved, P)); res["b_with_kit_cpu_rerank_n_diff"] = int((pb_saved != P).sum()); log("(b) kit rerank:", res["b_with_kit_cpu_rerank_equal_P"], res["b_with_kit_cpu_rerank_n_diff"])
json.dump(res, open(f"{OUT}/d12_validate.json", "w"), indent=1)
pb, aux = final_labels(f95.cuda(), ml95, cl95, indep_thres=thr, return_all=True)
res["b_gpu_fast_rerank_equal_P"] = bool(np.array_equal(pb, P)); res["b_gpu_fast_rerank_n_diff"] = int((pb != P).sum())
res["b_gpu_vs_kit_rerank_max_abs_diff"] = float(np.abs(aux["rerank"] - kit_rr).max()); res["b_gpu_vs_kit_rerank_array_equal"] = bool(np.array_equal(aux["rerank"], kit_rr))
from sklearn.metrics import adjusted_rand_score
res["b_gpu_vs_P_ARI"] = float(adjusted_rand_score(P, pb))
log("(b) gpu fast rerank:", json.dumps({k: v for k, v in res.items() if k.startswith("b_")}))
json.dump(res, open(f"{OUT}/d12_validate.json", "w"), indent=1)
