"""Phase D1: fragmentation facts + partner-rank baseline, with the kit's definitions
(tools/fragment_sim: units = labels of the final pseudo-labels P, a unit's dataset = its majority dataset; per GT identity the largest unit is
the 'main' fragment, every other unit holding the identity is a fragment (headroom/d2/s1_geometry.py); needed merges = sum_g (units holding g - 1);
wrong pairs and the classes (a) queried / (b) disagree-unqueried / (c) agree-wrong / (d) other = headroom/d1/analyze.py with the views V1/V2 of the real
ConsistencySampling.select).  CPU only."""
import os, sys, json, itertools, collections, inspect, textwrap, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import spcl.utils.constrained_clustering as _cc
import spcl.sampling.consistency as cons
from spcl.sampling.consistency import ConsistencySampling

PRIMARY = ["CatIndividualImages", "MultiCamCows2024", "PrimFace"]
KEY5 = ["CatIndividualImages", "MultiCamCows2024", "PrimFace", "PolarBearVidID", "SeaStarReID2023"]
NAMES = ["G", "P", "V1", "V2"]


def c2(x): return x * (x - 1) // 2
def n_same(cols, idx):
    if len(cols) == 0: return c2(len(idx))
    key = np.stack([c[idx] for c in cols], 1)
    _, cnt = np.unique(key, axis=0, return_counts=True)
    return int(c2(cnt.astype(np.int64)).sum())
def patterns(parts, idx, names=NAMES):
    """exact pair counts by the set of partitions calling the pair 'same' (Moebius inversion; headroom/d1/analyze.py)"""
    nS = {}
    for r in range(len(names) + 1):
        for sub in itertools.combinations(names, r): nS[sub] = n_same([parts[x] for x in sub], idx)
    out = {}
    for r in range(len(names) + 1):
        for T in itertools.combinations(names, r):
            tot = 0
            for r2 in range(r, len(names) + 1):
                for S in itertools.combinations(names, r2):
                    if set(T) <= set(S): tot += (-1) ** (r2 - r) * nS[S]
            out[T] = tot
    return out


def make_views(feat, rerank, ml, cl, G, A):
    """V1 (constrained DBSCAN) and V2 (constrained FINCH) = labels_dbscan / labels_finch of the REAL ConsistencySampling.select, text-patched only to
    stash its locals at the return and to draw nothing (n = 0).  Same patch as tools/fragment_sim/common.make_probe / headroom/d1/analyze.py."""
    _cc.CANONICAL_PAIRS = True; _cc.NP3_PER_CLUSTER_SKIP = False
    src = textwrap.dedent(inspect.getsource(ConsistencySampling.select))
    old = "        return self.ml_pairs, self.cl_pairs\n    except KeyboardInterrupt:"
    assert src.count(old) == 1
    src = src.replace(old, "        self._dbg = {k: v for k, v in locals().items() if k in ('labels_dbscan','labels_finch')}\n" + old, 1)
    ns = dict(vars(cons)); exec(src, ns)
    class Probe(ConsistencySampling): pass
    Probe.select = ns["select"]
    args = argparse.Namespace(dataset="wildlife", consam_sim_thresh=A["consam_sim_thresh"], consam_k=5, consam_eps=A["consam_eps"])
    S = Probe(thresh=A["eps"], min_samples=4, metric="precomputed", num_gt_ids_gallery=0, args=args)
    S.set_gt_labels([int(x) for x in G]); S.set_constraints(list(ml), list(cl)); S.n = 0
    n0 = (len(S.ml_pairs), len(S.cl_pairs))
    S.select(features=torch.from_numpy(np.ascontiguousarray(feat)).float(), rerank_dist=rerank)
    assert n0 == (len(S.ml_pairs), len(S.cl_pairs))
    return np.asarray(S._dbg["labels_dbscan"]), np.asarray(S._dbg["labels_finch"])


def units_of(P, feat, dset):
    unit = np.unique(P, return_inverse=True)[1]; U = int(unit.max()) + 1
    cnt = np.bincount(unit); order = np.argsort(unit, kind="stable"); st = np.r_[0, np.cumsum(cnt)[:-1]]
    members = [order[st[u]:st[u] + cnt[u]] for u in range(U)]
    udom = np.array([collections.Counter(dset[m]).most_common(1)[0][0] for m in members])
    cent = np.stack([feat[m].mean(0) for m in members]); cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    medoid = np.array([members[u][np.argmax(feat[members[u]] @ cent[u])] for u in range(U)])
    return dict(unit=unit, U=U, cnt=cnt, members=members, udom=udom, cent=cent, medoid=medoid)


def fragment_rows(u, G, dset, ds_names):
    """one row per (dataset, identity g, non-largest unit holding g) = headroom/d2/s1_geometry.py; partners = the other units of the dataset holding g"""
    unit, udom = u["unit"], u["udom"]; rows = []
    for ds in ds_names:
        idx_d = np.where(dset == ds)[0]; units_d = np.where(udom == ds)[0]; inD = np.zeros(u["U"], bool); inD[units_d] = True
        for g in np.unique(G[idx_d]):
            gi = idx_d[G[idx_d] == g]; us, cnt = np.unique(unit[gi], return_counts=True)
            if len(us) < 2: continue
            big = us[np.lexsort((us, -cnt))[0]]
            for uu, c in zip(us, cnt):
                if uu == big: continue
                partners = [int(x) for x in us if x != uu and inD[x]]
                rows.append(dict(dataset=str(ds), g=int(g), unit=int(uu), frag_size=int(c), partners=partners))
    return rows


def fragmentation(P, G, dset, feat, ml, cl, V1, V2, ds_names):
    """D1.3 numbers per dataset"""
    N = len(P); unit = np.unique(P, return_inverse=True)[1]
    sz = np.bincount(unit); U = len(sz)
    parts = {"G": G, "P": P, "V1": V1, "V2": V2}
    uq = collections.defaultdict(set)                          # unique queried pairs a != b (headroom/d1/analyze.py)
    for a, b in ml:
        if a != b: uq[(min(a, b), max(a, b))].add("ML")
    for a, b in cl: uq[(min(a, b), max(a, b))].add("CL")
    def pat_of(a, b): return tuple(n for n in NAMES if parts[n][a] == parts[n][b])
    q_by = {d: collections.Counter() for d in ds_names}
    for (a, b) in uq:
        if dset[a] == dset[b]: q_by[dset[a]][pat_of(a, b)] += 1
    # needed merges: sum_g (distinct units holding g - 1), per dataset
    key = np.unique(G.astype(np.int64) * U + unit); gg = key // U; m_g = np.bincount(gg, minlength=G.max() + 1)
    out = {}
    for ds in ds_names:
        idx = np.where(dset == ds)[0]
        pt = patterns(parts, idx)
        TP = sum(v for k, v in pt.items() if "G" in k and "P" in k)
        Pp = sum(v for k, v in pt.items() if "P" in k); Gp = sum(v for k, v in pt.items() if "G" in k)
        cls = collections.Counter(); wrong = 0
        for k, v in pt.items():
            g_, p_ = "G" in k, "P" in k
            if g_ == p_: continue
            qn = q_by[ds][k]; assert v - qn >= 0
            wrong += v
            v1, v2 = "V1" in k, "V2" in k
            cls["a_queried"] += qn
            if v1 != v2: cls["b_disagree_unqueried"] += v - qn
            elif v1 == p_: cls["c_agree_wrong"] += v - qn
            else: cls["d_other"] += v - qn
        needed = int(sum(m_g[g] - 1 for g in np.unique(G[idx])))
        out[str(ds)] = dict(n_images=int(len(idx)), n_ids=int(len(np.unique(G[idx]))), needed_merges=needed,
                            false_split_pairs=int(Gp - TP), false_merge_pairs=int(Pp - TP), wrong_pairs=int(wrong), same_gt_pairs=int(Gp), same_P_pairs=int(Pp),
                            precision=TP / Pp if Pp else float("nan"), recall=TP / Gp if Gp else float("nan"),
                            a_queried=int(cls["a_queried"]), b_disagree_unqueried=int(cls["b_disagree_unqueried"]), c_agree_wrong=int(cls["c_agree_wrong"]), d_other=int(cls["d_other"]),
                            agree_wrong_share=cls["c_agree_wrong"] / wrong if wrong else float("nan"))
    return out, dict(n_units=int(U), n_singletons=int((sz == 1).sum()), n_clusters_gt1=int((sz > 1).sum()))


def partner_ranks(u, rows, K=20):
    """D1.4 baseline B: for every fragment row with a partner in its dataset, the rank (1-based, among ALL other same-dataset units) of the nearest true partner by centroid
    cosine distance of unit centroids; plus the K nearest same-dataset units of the fragment unit (the candidate list the matcher rankers re-rank)."""
    cent, udom = u["cent"], u["udom"]; res = []
    for r in rows:
        if not r["partners"]: continue
        uu = r["unit"]; cand = np.where((udom == r["dataset"]) & (np.arange(u["U"]) != uu))[0]
        d = 1 - cent[cand] @ cent[uu]; order = np.argsort(d, kind="stable"); cs = cand[order]
        pos = {int(c): i + 1 for i, c in enumerate(cs)}
        rank = min(pos[p] for p in r["partners"] if p in pos)
        res.append(dict(dataset=r["dataset"], g=r["g"], unit=uu, rank=int(rank), n_cand=int(len(cand)), topk=[int(x) for x in cs[:K]]))
    return res


def share_table(res, ds_names, ks=(1, 5, 20)):
    out = {}
    for ds in list(ds_names) + ["ALL", "PRIMARY"]:
        rr = [r for r in res if (ds == "ALL") or (ds == "PRIMARY" and r["dataset"] in PRIMARY) or r["dataset"] == ds]
        if not rr: out[ds] = None; continue
        ranks = np.array([r["rank"] for r in rr])
        out[ds] = dict(n=len(rr), median_rank=float(np.median(ranks)), **{f"top{k}": float((ranks <= k).mean()) for k in ks})
    return out
