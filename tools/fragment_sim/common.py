"""Fragment-sim kit: loading + helpers (copy of headroom/d2/common.py; only the paths changed). CPU only, read-only on inputs.
Inputs: the epoch-95 state of consam_long ($FRAGSIM_BIG/resume_epoch95.pth), pool/P/views (data/), rerank ($FRAGSIM_BIG/rerank_e95.npy)."""
import os, sys, json, time, itertools, inspect, textwrap, argparse, collections
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
from paths import REPO, DATA, REF, BIG, WORK, KIT
D1 = DATA; D2 = WORK
SCR = DATA; SCR2 = WORK
sys.path.insert(0, REPO); os.chdir(REPO)
PRIMARY = ["CatIndividualImages", "MultiCamCows2024", "PrimFace"]; SECONDARY = ["PolarBearVidID"]; HI = PRIMARY + SECONDARY
RESUME = os.path.join(BIG, "resume_epoch95.pth")
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

def load():
    st = torch.load(RESUME, map_location="cpu", weights_only=False)
    d = {"epoch": int(st["epoch"]), "feat": st["memory_features"].float().numpy(), "A": st["args"],
         "ml": [tuple(map(int, p)) for p in st["ml_pairs"]], "cl": [tuple(map(int, p)) for p in st["cl_pairs"]]}
    pool = np.load(f"{D1}/pool.npz"); d["pid"], d["dset"] = pool["pid"], pool["dset"]
    d["G"] = np.unique(d["pid"], return_inverse=True)[1]
    d["P"] = np.load(f"{D1}/P.npz")["P"]
    v = np.load(f"{SCR}/views.npz"); d["V1"], d["V2"] = v["V1"], v["V2"]
    d["rerank"] = np.load(f"{BIG}/rerank_e{d['epoch']}.npy")
    d["ds_names"] = sorted(set(d["dset"].tolist()))
    d["N"] = len(d["P"])
    return d

def c2(x): return x * (x - 1) // 2

def n_same(cols, idx):
    if len(cols) == 0: return c2(len(idx))
    key = np.stack([c[idx] for c in cols], 1)
    _, cnt = np.unique(key, axis=0, return_counts=True)
    return int(c2(cnt.astype(np.int64)).sum())

def patterns(parts, idx):
    """exact pair counts by the set of partitions (names) that call the pair 'same'. parts: dict name -> labels (Moebius inversion)."""
    names = list(parts); nS = {}
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

def pair_f1(labels, G, idx):
    """pairwise precision/recall/F1 of `labels` vs GT within idx (contingency, no pair enumeration)."""
    lab, g = labels[idx], G[idx]
    tp = n_same([labels, G], idx); pp = n_same([labels], idx); gp = n_same([G], idx)
    pr = tp / pp if pp else float("nan"); rc = tp / gp if gp else float("nan")
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else float("nan"))

def make_probe(d, override=None):
    """The REAL ConsistencySampling.select (consistency.py:51-445), text-patched only to (i) stash its locals at the return, (ii) optionally
    replace the two views by `override` = (labels_a, labels_b, clusters_a, clusters_b) right after the FINCH block (consistency.py:97)."""
    import spcl.utils.constrained_clustering as cc
    import spcl.sampling.consistency as cons
    from spcl.sampling.consistency import ConsistencySampling
    cc.CANONICAL_PAIRS = True; cc.NP3_PER_CLUSTER_SKIP = False
    src = textwrap.dedent(inspect.getsource(ConsistencySampling.select))
    old = "        return self.ml_pairs, self.cl_pairs\n    except KeyboardInterrupt:"
    assert src.count(old) == 1
    src = src.replace(old, "        self._dbg = {k: v for k, v in locals().items() if k in ('labels_dbscan','clusters_dbscan','labels_finch','clusters_finch','pairs','cross_pairs','all_pairs','all_probs')}\n" + old, 1)
    hook = "        clusters_finch = labels_finch\n"
    assert src.count(hook) == 1
    src = src.replace(hook, hook + "        if getattr(self, '_ov', None) is not None:\n            labels_dbscan, labels_finch, clusters_dbscan, clusters_finch = self._ov\n", 1)
    ns = dict(vars(cons)); exec(src, ns)
    class Probe(ConsistencySampling): pass
    Probe.select = ns["select"]
    A = d["A"]
    args = argparse.Namespace(dataset="wildlife", consam_sim_thresh=A["consam_sim_thresh"], consam_k=5, consam_eps=A["consam_eps"])
    S = Probe(thresh=A["eps"], min_samples=4, metric="precomputed", num_gt_ids_gallery=0, args=args)
    S.set_gt_labels([int(x) for x in d["G"]]); S.set_constraints(list(d["ml"]), list(d["cl"])); S.n = 0
    S._ov = override
    return S

def pool_pairs(d, S):
    """run select (draws nothing) and return the set of unique unordered non-self candidate pairs of the sampler's pool"""
    n0 = (len(S.ml_pairs), len(S.cl_pairs))
    S.select(features=torch.from_numpy(d["feat"]), rerank_dist=d["rerank"])
    assert n0 == (len(S.ml_pairs), len(S.cl_pairs))
    D = S._dbg; allp = np.asarray(D["all_pairs"])
    return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in allp if a != b}, D
