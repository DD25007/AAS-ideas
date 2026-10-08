"""[kit copy of headroom/d4/s_adaptive.py] D4: adaptive ranker A / A-ex on the D3 simulator (within-dataset, medoid query, transitivity, noiseless GT; same state handling as headroom/d2/s2_sim.py and
headroom/d3/s2_sim.py). Implementation choices not fixed by the prereg (all label-free or symmetric encodings): continuous features standardized with a scaler fit on the
fixed candidate pool (all within-dataset unit pairs); size features encoded as (min, max) of log unit sizes; min-linkage rerank symmetrized by min(ML, ML.T);
labels = the GT answer of the medoid query (1 = same identity). 'Round' = 200 PAID queries (free/implied pairs are not answers)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from history import *
from sklearn.linear_model import LogisticRegression
D4 = WORK
d = load(); P, G, dset, feat, R, N = d["P"], d["G"], d["dset"], d["feat"], d["rerank"], d["N"]
u = units(d); unit, U, members, udom, medoid, Dc, cnt, order, st = u["unit"], u["U"], u["members"], u["udom"], u["medoid"], u["Dc"], u["cnt"], u["order"], u["st"]
ds_names = d["ds_names"]; dsi = {x: np.where(dset == x)[0] for x in ds_names}
BUD = [500, 1000, 2000, 5000, 10000, 42289]; ROUND = 200
H = load_H()
# ---- min-linkage rerank between units (as D2 s2_sim)
ML = np.empty((U, U), dtype=np.float32)
for a in range(U):
    r = R[members[a]].min(0); ML[a] = np.minimum.reduceat(r[order], st)
ML = np.minimum(ML, ML.T); log("ML built")
# ---- candidate pairs (within dataset), in centroid order (stable) = D2 rule M order
iu = np.triu_indices(U, 1); keep = udom[iu[0]] == udom[iu[1]]; iu = (iu[0][keep], iu[1][keep])
o = np.argsort(Dc[iu], kind="stable"); PA, PB = iu[0][o], iu[1][o]; NP = len(PA)       # PA,PB: pairs sorted by centroid order; index = centroid rank
log("candidate pairs", NP)
ls = np.log(cnt.astype(np.float64)); lmin = np.minimum(ls[PA], ls[PB]); lmax = np.maximum(ls[PA], ls[PB])
cont = np.stack([1 - Dc[PA, PB], ML[PA, PB], lmin, lmax, H[PA, PB]], 1)
cont = (cont - cont.mean(0)) / cont.std(0)
dsid = {x: i for i, x in enumerate(ds_names)}; oh = np.zeros((NP, len(ds_names))); oh[np.arange(NP), [dsid[x] for x in udom[PA]]] = 1
X = np.hstack([cont, oh]).astype(np.float32); FEAT = ["centroid_cosine", "minlink_rerank", "log_size_min", "log_size_max", "H"] + [f"ds={x}" for x in ds_names]

def frag_counts(comp):
    key = np.unique(G.astype(np.int64) * (comp.max() + 1) + comp); gg = key // (comp.max() + 1); return np.bincount(gg, minlength=G.max() + 1)
m0 = frag_counts(unit); need = {x: int(sum(m0[g] - 1 for g in np.unique(G[dsi[x]]))) for x in ds_names}
class UF:
    def __init__(s, n): s.p = np.arange(n)
    def find(s, x):
        p = s.p
        while p[x] != x: p[x] = p[p[x]]; x = p[x]
        return x

def simulate(mode):
    """mode: 'off' (pure centroid order, must equal D2 rule M), 'A', 'A-ex'."""
    iuf = UF(N); blocks = collections.defaultdict(set)
    for a, b in d["ml"]:
        if a != b:
            ra, rb = iuf.find(a), iuf.find(b)
            if ra != rb: iuf.p[rb] = ra
    for a, b in d["cl"]:
        ra, rb = iuf.find(a), iuf.find(b)
        if ra != rb: blocks[ra].add(rb); blocks[rb].add(ra)
    cuf = UF(U); cblocks = collections.defaultdict(set)
    S = dict(q=0, wasted=0, free=0, skipped=0); snaps = {}; pending = list(BUD)
    done = np.zeros(NP, dtype=bool)                           # processed pairs (asked, free, or implied-different / same-cluster): permanently out
    ans_idx, ans_y = [], []
    def snapshot(nq):
        comp = np.array([cuf.find(x) for x in range(U)])[unit]; m = frag_counts(comp)
        snaps[nq] = {"queries": nq, "wasted": S["wasted"], "free_merges": S["free"], "recovered": {x: int(sum(m0[g] - m[g] for g in np.unique(G[dsi[x]]))) for x in ds_names},
                     "f1": {x: list(pair_f1(comp, G, dsi[x])) for x in ds_names}}
    def process(i):
        """returns None (not paid) or 1/0 (paid, answer same/different). Mirrors the D2 loop body."""
        A, B = int(PA[i]), int(PB[i]); done[i] = True
        ca, cb = cuf.find(A), cuf.find(B)
        if ca == cb: return None
        if cb in cblocks[ca]: S["skipped"] += 1; return None
        a, b = int(medoid[A]), int(medoid[B]); ra, rb = iuf.find(a), iuf.find(b)
        if ra == rb:
            cuf.p[cb] = ca; S["free"] += 1
            for z in cblocks[cb]: cblocks[z].discard(cb); cblocks[z].add(ca); cblocks[ca].add(z)
            cblocks.pop(cb, None); return None
        if rb in blocks[ra]: S["skipped"] += 1; return None
        S["q"] += 1
        if G[a] == G[b]:
            iuf.p[rb] = ra; blocks[ra] |= blocks.pop(rb, set())
            for z in blocks[ra]: blocks[z].discard(rb); blocks[z].add(ra)
            cuf.p[cb] = ca
            for z in cblocks[cb]: cblocks[z].discard(cb); cblocks[z].add(ca); cblocks[ca].add(z)
            cblocks.pop(cb, None); y = 1
        else:
            S["wasted"] += 1; blocks[ra].add(rb); blocks[rb].add(ra); cblocks[ca].add(cb); cblocks[cb].add(ca); y = 0
        ans_idx.append(i); ans_y.append(y)
        while pending and S["q"] >= pending[0]: snapshot(pending.pop(0))
        return y
    def run_ranking(idx_iter, n_paid):
        """walk an index sequence, processing pairs not yet done, until n_paid paid queries made (or budget exhausted); returns (paid, same)"""
        paid = same = 0
        for i in idx_iter:
            if not pending: break
            if done[i]: continue
            y = process(i)
            if y is not None: paid += 1; same += y
            if paid >= n_paid: break
        return paid, same
    rounds = []; model = None; ptr = 0                                    # ptr: scan pointer into the centroid-order list for exploration / pure rule
    def centroid_iter():
        nonlocal ptr
        while ptr < NP:
            if not done[ptr]: yield ptr
            ptr += 1
    r = 0
    while pending:
        if r == 0 or mode == "off":
            paid, same = run_ranking(centroid_iter(), ROUND); rounds.append({"round": r, "paid": paid, "same": same, "model_paid": 0, "model_same": 0, "explore_paid": paid, "explore_same": same, "source": "centroid"})
        else:
            ai = np.array(ans_idx); ay = np.array(ans_y)
            if ay.min() == ay.max(): model = None
            else:
                model = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs").fit(X[ai], ay)
            if model is None:
                paid, same = run_ranking(centroid_iter(), ROUND); rounds.append({"round": r, "paid": paid, "same": same, "model_paid": 0, "model_same": 0, "explore_paid": paid, "explore_same": same, "source": "centroid(no model)"})
            else:
                n_ex = int(round(0.2 * ROUND)) if mode == "A-ex" else 0; n_mod = ROUND - n_ex
                sc = model.decision_function(X).astype(np.float64); sc[done] = -np.inf
                ordr = np.argsort(-sc, kind="stable")                      # ties -> centroid order
                mp, ms = run_ranking(ordr, n_mod)
                ep, es = (run_ranking(centroid_iter(), n_ex) if n_ex else (0, 0))
                rounds.append({"round": r, "paid": mp + ep, "same": ms + es, "model_paid": mp, "model_same": ms, "explore_paid": ep, "explore_same": es, "source": "model" + ("+explore" if n_ex else "")})
        r += 1
        if r > 400: break
    final_model = None
    if mode != "off" and len(set(ans_y)) == 2 and len(ans_y) > ROUND:
        # the model used for the LAST round = fit on all answers except that round's; refit on all answers for reporting is also given
        k = len(ans_y) - rounds[-1]["paid"]
        m_last = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=2000).fit(X[np.array(ans_idx[:k])], np.array(ans_y[:k]))
        m_all = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", max_iter=2000).fit(X[np.array(ans_idx)], np.array(ans_y))
        final_model = {"features": FEAT, "coef_used_in_last_round": m_last.coef_[0].tolist(), "intercept_last": float(m_last.intercept_[0]), "n_train_last": k,
                       "coef_all_answers": m_all.coef_[0].tolist(), "intercept_all": float(m_all.intercept_[0]), "n_train_all": len(ans_y), "positive_rate_all": float(np.mean(ans_y))}
    return {"mode": mode, "snapshots": snaps, "rounds": rounds, "stats": S, "final_model": final_model}

out = {"needed": need, "primary_needed": int(sum(need[x] for x in PRIMARY)), "features": FEAT, "runs": []}
pn = out["primary_needed"]
for mode in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["off", "A", "A-ex"]):          # usage: python s_adaptive.py [off,A,A-ex]
    t = time.time(); r = simulate(mode); out["runs"].append(r); s5 = r["snapshots"].get(5000)
    rec = sum(s5["recovered"][x] for x in PRIMARY)
    log(f"{mode:5s}: @5000q recovered primary {rec}/{pn} = {100*rec/pn:.1f}% wasted {s5['wasted']}/{s5['queries']} = {100*s5['wasted']/s5['queries']:.1f}%  rounds {len(r['rounds'])}  ({time.time()-t:.0f}s)")
json.dump(out, open(f"{D4}/sim_d4.json", "w"))
d2 = json.load(open(f"{REF}/d2_sim.json")); w = [x for x in d2["runs"] if x["scope"] == "within" and x["rule"] == "M"][0]
off = [r for r in out["runs"] if r["mode"] == "off"]
if not off: sys.exit(0)
off = off[0]
same = all(off["snapshots"][b]["recovered"] == w["snapshots"][str(b)]["recovered"] and off["snapshots"][b]["wasted"] == w["snapshots"][str(b)]["wasted"] and off["snapshots"][b]["free_merges"] == w["snapshots"][str(b)]["free_merges"] for b in BUD)
log("model disabled reproduces D2 within/M exactly (recovered, wasted, free merges at all 6 budgets):", same)
json.dump({"off_reproduces_d2_ruleM": bool(same)}, open(f"{D4}/m_check_d4.json", "w"))
