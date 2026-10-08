"""Fragment-sim kit: units, centroids, medoids and the history score H (copy of headroom/d3/common3.py; only the label source changed:
the 19 saved memory_labels of consam_long live in data/labels_by_epoch.npz instead of the 450 MB resume files)."""
import os, sys, glob, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *            # noqa  (load, pair_f1, log, PRIMARY, SECONDARY, HI, ...)
LABELS = np.load(f"{DATA}/labels_by_epoch.npz")

def resume_epochs():
    return sorted(int(k[1:]) for k in LABELS.files)

def load_H():
    """the D3 history score (epochs < 95), shipped in data/H.npz; history(unit, U, [5,...,90]) recomputes it exactly"""
    return np.load(f"{DATA}/H.npz")["H"].astype(np.float64)

def units(d):
    P, G, dset, feat = d["P"], d["G"], d["dset"], d["feat"]
    unit = np.unique(P, return_inverse=True)[1]; U = int(unit.max()) + 1
    cnt = np.bincount(unit); order = np.argsort(unit, kind="stable"); st = np.r_[0, np.cumsum(cnt)[:-1]]
    members = [order[st[u]:st[u] + cnt[u]] for u in range(U)]
    udom = np.array([collections.Counter(dset[m]).most_common(1)[0][0] for m in members])
    cent = np.stack([feat[m].mean(0) for m in members]); cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    medoid = np.array([members[u][np.argmax(feat[members[u]] @ cent[u])] for u in range(U)])
    return dict(unit=unit, U=U, cnt=cnt, order=order, st=st, members=members, udom=udom, cent=cent, medoid=medoid, Dc=(1 - cent @ cent.T))

def history(unit, U, epochs):
    """H[A,B] = mean_t sum_l nA_t(l) nB_t(l) / (|A||B|);  NE[A,B] = #epochs with sum_l nA_t(l) nB_t(l) > 0."""
    import scipy.sparse as sp
    size = np.bincount(unit, minlength=U).astype(np.float64)
    Hs = np.zeros((U, U)); NE = np.zeros((U, U), dtype=np.int16)
    for t in epochs:
        lab = np.unique(LABELS[f"e{t}"], return_inverse=True)[1]
        M = sp.csr_matrix((np.ones(len(lab)), (unit, lab)), shape=(U, lab.max() + 1))
        C = (M @ M.T).toarray(); Hs += C / np.outer(size, size); NE += (C > 0)
    return Hs / len(epochs), NE
