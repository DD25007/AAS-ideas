"""D1.1 empirical: one synthetic ML pair through constrained_clustering on the b0_det_A epoch-0 labels_pre (CPU only)."""
import os, sys, json, io, contextlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
B = "/bhallukah/dd25007/HITL-ReID-idea2-outputs/b0_det_A/state"
pre = np.load(f"{B}/epoch0_labels_pre.npy").astype(np.int64); rr = np.load(f"{B}/epoch0_rerank.npy", mmap_mode="r")
G = pool_meta()["G"]; dset = pool_meta()["dset"]; N = len(pre)
sz = np.bincount(pre); size_of = sz[pre]
log("labels_pre: images", N, "labels", len(sz), "singletons", int((sz == 1).sum()), "clusters(size>1)", int((sz > 1).sum()), "images in size>1 clusters", int((size_of > 1).sum()))
rng = np.random.default_rng(0)
def members(lbl): return np.where(pre == lbl)[0]
def run(a, b):
    with contextlib.redirect_stdout(io.StringIO()):
        out = constrained_clustering(pre.copy(), rr, "~DBSCAN", min_samples=4, ml_pairs=[(a, b)], cl_pairs=[])
    return np.asarray(out)
def pick(kind):
    """random pair (a, b) with G[a]==G[b], different pre-labels, of the requested kind"""
    while True:
        a = int(rng.integers(N)); same = np.where((G == G[a]) & (pre != pre[a]))[0]
        if len(same) == 0: continue
        b = int(rng.choice(same)); sa, sb = size_of[a], size_of[b]
        ok = {"cluster-cluster": sa > 1 and sb > 1, "cluster-singleton": (sa > 1) != (sb > 1), "singleton-singleton": sa == 1 and sb == 1}[kind]
        if ok: return a, b
rows = []
for kind in ("cluster-cluster", "cluster-singleton", "singleton-singleton"):
    for _ in range(6):
        a, b = pick(kind); ma, mb = members(pre[a]), members(pre[b]); out = run(a, b)
        union = np.union1d(ma, mb); merged_all = bool(len(set(out[union])) == 1)
        newsize = int((out == out[a]).sum()); others_changed = int((np.setdiff1d(np.arange(N), union)[:0]).size)  # placeholder, replaced below
        rest = np.setdiff1d(np.arange(N), union)
        # every image outside the two groups keeps the same grouping (partition of `rest` unchanged and disjoint from the merged label)
        untouched = bool(out[a] not in set(out[rest].tolist()))
        gts_a, gts_b = len(set(G[ma])), len(set(G[mb]))
        rows.append(dict(kind=kind, a=a, b=b, size_a=int(len(ma)), size_b=int(len(mb)), size_after=newsize, sum_sizes=int(len(ma) + len(mb)),
                         merged_all_images=merged_all, nothing_else_joined=untouched, n_gt_ids_in_a=gts_a, n_gt_ids_in_b=gts_b,
                         dset_a=str(dset[a]), gt=str(G[a])))
        log(json.dumps(rows[-1]))
json.dump(rows, open(f"{OUT}/d11_merge_probe.json", "w"), indent=1)
for k in ("cluster-cluster", "cluster-singleton", "singleton-singleton"):
    r = [x for x in rows if x["kind"] == k]
    log(k, "n", len(r), "all merged:", all(x["merged_all_images"] for x in r), "size_after==sum:", all(x["size_after"] == x["sum_sizes"] for x in r), "nothing else joined:", all(x["nothing_else_joined"] for x in r))
