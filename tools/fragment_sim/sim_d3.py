"""[kit copy of headroom/d3/s2_sim.py] D3 step 3: the D2 simulator (within-dataset, medoid query, transitivity, noiseless GT), candidate order replaced by:
   M = D2 rule M (centroid cosine; must reproduce D2 sim.json), R_H = (-H, centroid cosine), R_HD = centroid cosine - H.  Same code path as headroom/d2/s2_sim.py."""
from history import *
d = load(); P, G, dset, feat, R, N = d["P"], d["G"], d["dset"], d["feat"], d["rerank"], d["N"]
u = units(d); unit, U, members, udom, medoid, Dc = u["unit"], u["U"], u["members"], u["udom"], u["medoid"], u["Dc"]
ds_names = d["ds_names"]; dsi = {x: np.where(dset == x)[0] for x in ds_names}
BUD = [500, 1000, 2000, 5000, 10000, 42289]
def frag_counts(comp):
    key = np.unique(G.astype(np.int64) * (comp.max() + 1) + comp); gg = key // (comp.max() + 1); return np.bincount(gg, minlength=G.max() + 1)
m0 = frag_counts(unit); need = {x: int(sum(m0[g] - 1 for g in np.unique(G[dsi[x]]))) for x in ds_names}
log("needed merges", {x: need[x] for x in HI}, "primary", sum(need[x] for x in PRIMARY))
class UF:
    def __init__(s, n): s.p = np.arange(n)
    def find(s, x):
        p = s.p
        while p[x] != x: p[x] = p[p[x]]; x = p[x]
        return x
# ---- ranker registry. A ranker maps (dc, h, iu) -> a permutation of the within-dataset candidate pairs (best first):
#      dc = centroid-cosine distance of each pair, h = D3 history score H of each pair, iu = (A, B) unit-index arrays. Add yours here.
RANKERS = {
    "M":    lambda dc, h, iu: np.argsort(dc, kind="stable"),          # D2 rule M (centroid order)
    "R_H":  lambda dc, h, iu: np.lexsort((dc, -h)),                   # D3 R_H
    "R_HD": lambda dc, h, iu: np.argsort(dc - h, kind="stable"),      # D3 R_HD
}
def simulate(Hm, ranker):
    iu = np.triu_indices(U, 1); keep = udom[iu[0]] == udom[iu[1]]; iu = (iu[0][keep], iu[1][keep])
    dc = Dc[iu]
    o = RANKERS[ranker](dc, Hm[iu], iu)
    A_, B_ = iu[0][o], iu[1][o]
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
        comp = np.array([cuf.find(x) for x in range(U)])[unit]; m = frag_counts(comp)
        rec = {x: int(sum(m0[g] - m[g] for g in np.unique(G[dsi[x]]))) for x in ds_names}
        snaps[nq] = {"queries": nq, "wasted": wasted, "free_merges": free, "recovered": rec, "f1": {x: list(pair_f1(comp, G, dsi[x])) for x in ds_names}}
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
    return {"ranker": ranker, "candidate_pairs": int(len(A_)), "exhausted_at_queries": exhausted, "skipped_implied_different": skipped, "snapshots": snaps}
H = load_H()
out = {"needed": need, "primary_needed": int(sum(need[x] for x in PRIMARY)), "runs": []}
pn = out["primary_needed"]
names = (sys.argv[1] if len(sys.argv) > 1 else "M,R_H,R_HD").split(",")          # usage: python sim_d3.py [M,R_H,R_HD,<your ranker>]
for rk in names:
    r = simulate(H, rk); r["H_variant"] = "main"; out["runs"].append(r); s5 = r["snapshots"][5000]; rec = sum(s5["recovered"][x] for x in PRIMARY)
    log(f"{rk:5s}: candidates {r['candidate_pairs']} | @5000q recovered primary {rec}/{pn} = {100*rec/pn:.1f}%  wasted {s5['wasted']}/{s5['queries']} = {100*s5['wasted']/s5['queries']:.1f}%  exhausted_at {r['exhausted_at_queries']}")
json.dump(out, open(f"{WORK}/sim_d3.json", "w"))
d2 = json.load(open(f"{REF}/d2_sim.json")); w = [r for r in d2["runs"] if r["scope"] == "within" and r["rule"] == "M"][0]
mine = [r for r in out["runs"] if r["ranker"] == "M"]
if not mine: sys.exit(0)
mine = mine[0]
same = all(mine["snapshots"][str(b)]["recovered"] == w["snapshots"][str(b)]["recovered"] and mine["snapshots"][str(b)]["wasted"] == w["snapshots"][str(b)]["wasted"] for b in BUD) if isinstance(list(mine["snapshots"])[0], str) else all(mine["snapshots"][b]["recovered"] == w["snapshots"][str(b)]["recovered"] and mine["snapshots"][b]["wasted"] == w["snapshots"][str(b)]["wasted"] for b in BUD)
log("rule M reproduces D2 within/M exactly (recovered per dataset + wasted at all budgets):", same)
json.dump({"m_reproduces_d2": bool(same)}, open(f"{WORK}/m_check_d3.json", "w"))
