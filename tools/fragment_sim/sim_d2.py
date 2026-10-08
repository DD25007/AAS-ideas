"""[kit copy of headroom/d2/s2_sim.py; writes $FRAGSIM_WORK/sim_d2.json] D2 step 2: nearest-cluster baseline simulation with a noiseless GT oracle.

State: P (units) + ALL existing answered pairs. Two layers of knowledge:
  image layer  : union-find over images from the existing ML answers (+ new ones) and CL blocks between image components, giving
                 'relation already implied' (same -> free merge, different -> skip);
  cluster layer: union-find over P units = the clustering after applying the merges (what NP3 would do with an ML answer); a 'different'
                 answer blocks that pair of merged clusters from being queried again.
Candidates: cluster pairs in increasing distance order. Two query-pair rules (both run):
  M (medoids)       order = centroid cosine distance; query = (medoid(A), medoid(B))
  C (closest pair)  order = min-linkage on rerank distance; query = the closest image pair between A and B
Scopes: within-dataset (units of the same dominant dataset only) and global (all unit pairs, no dataset information, as AAS).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
d = load(); P, G, dset, feat, R, N = d["P"], d["G"], d["dset"], d["feat"], d["rerank"], d["N"]
unit = np.unique(P, return_inverse=True)[1]; U = unit.max() + 1
cnt = np.bincount(unit); order = np.argsort(unit, kind="stable"); st = np.r_[0, np.cumsum(cnt)[:-1]]
members = [order[st[u]:st[u] + cnt[u]] for u in range(U)]
udom = np.array([collections.Counter(dset[m]).most_common(1)[0][0] for m in members])
cent = np.stack([feat[m].mean(0) for m in members]); cent /= np.linalg.norm(cent, axis=1, keepdims=True)
medoid = np.array([members[u][np.argmax(feat[members[u]] @ cent[u])] for u in range(U)])
Dc = 1 - cent @ cent.T                                                        # centroid cosine distance between units
ML = np.empty((U, U), dtype=np.float32)                                       # min-linkage on rerank between units
for u in range(U):
    r = R[members[u]].min(0); ML[u] = np.minimum.reduceat(r[order], st)
log("distance matrices", U, "units")
ds_names = d["ds_names"]; dsi = {x: np.where(dset == x)[0] for x in ds_names}
BUD = [500, 1000, 2000, 5000, 10000, 42289]

# ---- needed merges: per identity m_g - 1 (units containing g); 'recovered' = reduction of #components containing g
gkey_ds = {g: dset[np.where(G == g)[0][0]] for g in np.unique(G)}
def frag_counts(comp):                                                        # comp: component id per image -> {g: #distinct components}
    key = np.unique(G.astype(np.int64) * (comp.max() + 1) + comp)
    gg = key // (comp.max() + 1); return np.bincount(gg, minlength=G.max() + 1)
m0 = frag_counts(unit)
need = {x: int(sum(m0[g] - 1 for g in np.unique(G[dsi[x]]))) for x in ds_names}
log("needed merges", {x: need[x] for x in HI}, "primary pooled", sum(need[x] for x in PRIMARY))

class UF:
    def __init__(s, n): s.p = np.arange(n)
    def find(s, x):
        p = s.p
        while p[x] != x: p[x] = p[p[x]]; x = p[x]
        return x
def simulate(scope, rule):
    dist = Dc if rule == "M" else ML
    iu = np.triu_indices(U, 1)
    if scope == "within":
        keep = udom[iu[0]] == udom[iu[1]]; iu = (iu[0][keep], iu[1][keep])
    o = np.argsort(dist[iu], kind="stable"); A_, B_ = iu[0][o], iu[1][o]
    # image layer
    iuf = UF(N); blocks = collections.defaultdict(set)
    for a, b in d["ml"]:
        if a != b:
            ra, rb = iuf.find(a), iuf.find(b)
            if ra != rb: iuf.p[rb] = ra
    for a, b in d["cl"]:
        ra, rb = iuf.find(a), iuf.find(b)
        if ra != rb: blocks[ra].add(rb); blocks[rb].add(ra)
    cuf = UF(U); cblocks = collections.defaultdict(set)
    q = wasted = free = skipped = 0; snaps = {}; exhausted = None
    def snapshot(nq):
        comp = np.array([cuf.find(x) for x in range(U)])[unit]
        m = frag_counts(comp)
        rec = {x: int(sum(m0[g] - m[g] for g in np.unique(G[dsi[x]]))) for x in ds_names}
        f1 = {x: pair_f1(comp, G, dsi[x]) for x in ds_names}
        snaps[nq] = {"queries": nq, "wasted": wasted, "free_merges": free, "recovered": rec, "f1": {x: list(v) for x, v in f1.items()}}
    pending = list(BUD)
    for A, B in zip(A_.tolist(), B_.tolist()):
        ca, cb = cuf.find(A), cuf.find(B)
        if ca == cb: continue
        if cb in cblocks[ca]: skipped += 1; continue
        if rule == "M": a, b = int(medoid[A]), int(medoid[B])
        else:
            sub = R[np.ix_(members[A], members[B])]; i, j = np.unravel_index(np.argmin(sub), sub.shape); a, b = int(members[A][i]), int(members[B][j])
        ra, rb = iuf.find(a), iuf.find(b)
        if ra == rb:                                                          # relation already implied: same -> free merge
            cuf.p[cb] = ca; free += 1
            for z in cblocks[cb]: cblocks[z].discard(cb); cblocks[z].add(ca); cblocks[ca].add(z)
            cblocks.pop(cb, None); continue
        if rb in blocks[ra]: skipped += 1; continue                           # already implied different
        q += 1
        if G[a] == G[b]:
            iuf.p[rb] = ra; blocks[ra] |= blocks.pop(rb, set())
            for z in blocks[ra]: blocks[z].discard(rb); blocks[z].add(ra)
            cuf.p[cb] = ca
            for z in cblocks[cb]: cblocks[z].discard(cb); cblocks[z].add(ca); cblocks[ca].add(z)
            cblocks.pop(cb, None)
        else:
            wasted += 1; blocks[ra].add(rb); blocks[rb].add(ra); cblocks[ca].add(cb); cblocks[cb].add(ca)
        while pending and q >= pending[0]:
            snapshot(pending.pop(0))
        if not pending: break
    if pending:
        exhausted = q
        for b_ in pending: snapshot(b_)                                       # candidate list exhausted: final state reported at the remaining budgets
        for b_ in BUD:
            if b_ in snaps and b_ > q: snaps[b_]["queries"] = q
    return {"scope": scope, "rule": rule, "candidate_pairs": int(len(A_)), "exhausted_at_queries": exhausted, "skipped_implied_different": skipped, "snapshots": snaps}
out = {"needed": need, "primary_needed": int(sum(need[x] for x in PRIMARY)), "runs": []}
for scope in ("within", "global"):
    for rule in ("M", "C"):
        r = simulate(scope, rule); out["runs"].append(r)
        s5 = r["snapshots"][5000]
        log(f"{scope:6s} {rule}: candidates {r['candidate_pairs']:>8d} | @5000 q recovered primary {sum(s5['recovered'][x] for x in PRIMARY)}/{out['primary_needed']} "
            f"= {100*sum(s5['recovered'][x] for x in PRIMARY)/out['primary_needed']:.1f}%  wasted {s5['wasted']}  exhausted_at {r['exhausted_at_queries']}")
json.dump(out, open(f"{WORK}/sim_d2.json", "w"))
log("done")
