"""D1.6 simulator: a state-parametrised copy of tools/fragment_sim/sim_d3.py::simulate (identical query/transitivity/bookkeeping code; only the globals that
sim_d3 loads from the kit's epoch-95 state are arguments here, and the candidate order is an input).  Validated against ref/d2_sim.json rule M on the kit state
(d16_sim.py --check).  A noiseless GT oracle answers one medoid image pair per cluster pair."""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frag import *

BUD = [500, 1000, 2000, 5000, 10000, 42289]

class UF:
    def __init__(s, n): s.p = np.arange(n)
    def find(s, x):
        p = s.p
        while p[x] != x: p[x] = p[p[x]]; x = p[x]
        return x

def make_state(P, G, dset, feat, ml, cl):
    u = units_of(P, feat, dset); unit = u["unit"]; ds_names = sorted(set(dset.tolist())); dsi = {x: np.where(dset == x)[0] for x in ds_names}
    def frag_counts(comp):
        key = np.unique(G.astype(np.int64) * (comp.max() + 1) + comp); gg = key // (comp.max() + 1); return np.bincount(gg, minlength=G.max() + 1)
    m0 = frag_counts(unit)
    need = {x: int(sum(m0[g] - 1 for g in np.unique(G[dsi[x]]))) for x in ds_names}
    return dict(u=u, unit=unit, U=u["U"], N=len(P), G=G, dset=dset, ml=ml, cl=cl, ds_names=ds_names, dsi=dsi, frag_counts=frag_counts, m0=m0, need=need,
                Dc=1 - u["cent"] @ u["cent"].T)

def within_pairs(S):
    iu = np.triu_indices(S["U"], 1); keep = S["u"]["udom"][iu[0]] == S["u"]["udom"][iu[1]]; return iu[0][keep], iu[1][keep]

def order_B(S):
    iu = within_pairs(S); o = np.argsort(S["Dc"][iu], kind="stable"); return iu[0][o], iu[1][o]

def simulate(S, A_, B_):
    U, N, G, unit, medoid = S["U"], S["N"], S["G"], S["unit"], S["u"]["medoid"]; ds_names, dsi, m0, frag_counts = S["ds_names"], S["dsi"], S["m0"], S["frag_counts"]
    iuf = UF(N); blocks = collections.defaultdict(set)
    for a, b in S["ml"]:
        if a != b:
            ra, rb = iuf.find(a), iuf.find(b)
            if ra != rb: iuf.p[rb] = ra
    for a, b in S["cl"]:
        ra, rb = iuf.find(a), iuf.find(b)
        if ra != rb: blocks[ra].add(rb); blocks[rb].add(ra)
    cuf = UF(U); cblocks = collections.defaultdict(set)
    q = wasted = free = skipped = 0; snaps = {}; exhausted = None
    def snapshot(nq):
        comp = np.array([cuf.find(x) for x in range(U)])[unit]; m = frag_counts(comp)
        rec = {x: int(sum(m0[g] - m[g] for g in np.unique(G[dsi[x]]))) for x in ds_names}
        snaps[nq] = {"queries": nq, "wasted": wasted, "free_merges": free, "recovered": rec}
    pending = list(BUD)
    for A, B in zip(A_.tolist(), B_.tolist()):
        ca, cb = cuf.find(A), cuf.find(B)
        if ca == cb: continue
        if cb in cblocks[ca]: skipped += 1; continue
        a, b = int(medoid[A]), int(medoid[B]); ra, rb = iuf.find(a), iuf.find(b)
        if ra == rb:
            cuf.p[cb] = ca; free += 1
            for z in cblocks[cb]: cblocks[z].discard(cb); cblocks[z].add(ca); cblocks[ca].add(z)
            cblocks.pop(cb, None); continue
        if rb in blocks[ra]: skipped += 1; continue
        q += 1
        if G[a] == G[b]:
            iuf.p[rb] = ra; blocks[ra] |= blocks.pop(rb, set())
            for z in blocks[ra]: blocks[z].discard(rb); blocks[z].add(ra)
            cuf.p[cb] = ca
            for z in cblocks[cb]: cblocks[z].discard(cb); cblocks[z].add(ca); cblocks[ca].add(z)
            cblocks.pop(cb, None)
        else:
            wasted += 1; blocks[ra].add(rb); blocks[rb].add(ra); cblocks[ca].add(cb); cblocks[cb].add(ca)
        while pending and q >= pending[0]: snapshot(pending.pop(0))
        if not pending: break
    if pending:
        exhausted = q
        for b_ in pending: snapshot(b_)
        for b_ in BUD:
            if b_ in snaps and b_ > q: snaps[b_]["queries"] = q
    return {"candidate_pairs": int(len(A_)), "exhausted_at_queries": exhausted, "skipped_implied_different": skipped, "snapshots": snaps}

def summarize(S, run, budget=5000):
    s = run["snapshots"][budget]; need = S["need"]; tot = sum(need.values()); pri = sum(need[x] for x in PRIMARY)
    rec_all = sum(s["recovered"].values()); rec_pri = sum(s["recovered"][x] for x in PRIMARY)
    return dict(budget=budget, queries=s["queries"], wasted=s["wasted"], wasted_share=s["wasted"] / s["queries"] if s["queries"] else float("nan"),
                recovered_all=rec_all, needed_all=tot, recovery_all=rec_all / tot, recovered_primary=rec_pri, needed_primary=pri, recovery_primary=rec_pri / pri,
                per_dataset={x: dict(recovered=s["recovered"][x], needed=need[x]) for x in S["ds_names"]})
