"""D1.5 (R_DINO control, Gate-1 rank-1/top-5 inside the 20 nearest) + D1.6 (merge-recovery simulation) for one C0 seed.  CPU only.
Orders over within-dataset unit pairs (best first):
  B        = centroid cosine (C0 features), the kit's rule M order
  ORACLE   = (free upper bound for any ranker restricted to the 20-nearest candidate lists) candidate pairs whose two units share a GT identity first (B order inside), then everything else in B order
  R_DINO   = candidate pairs (u, v in the 20 nearest of a fragment unit u) by DINOv2 unit-centroid cosine, then the non-candidate pairs in B order
  R_DINO_g = control, global: ALL within-dataset unit pairs by DINOv2 unit-centroid cosine
usage: d16_sim.py <seed>"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_d1 import *
s = int(sys.argv[1]); tag = f"s{s}"
meta = pool_meta(); G, dset = meta["G"], meta["dset"]; ds_names = sorted(set(dset.tolist()))
ml, cl = load_constraints(s)
P = np.load(f"{OUT}/P_{tag}.npy"); feat = np.load(f"{OUT}/feat_{tag}.npy")
S = make_state(P, G, dset, feat, ml, cl); u = S["u"]; U = S["U"]
dino = np.load(f"{OUT}/dino_feat.npy")
dc = np.stack([dino[m].mean(0) for m in u["members"]]); dc /= np.linalg.norm(dc, axis=1, keepdims=True)
Dd = 1 - dc @ dc.T
R = json.load(open(f"{OUT}/d13_{tag}.json"))["ranks"]
# ---------------- Gate-1 ranks inside the 20 nearest
rows_all = fragment_rows(u, G, dset, ds_names); part = {(r["dataset"], r["g"], r["unit"]): r["partners"] for r in rows_all}
def gate(rank_fn):
    out = []
    for r in R:
        partners = part[(r["dataset"], r["g"], r["unit"])]; cand = r["topk"]; hit = [c for c in cand if c in partners]
        if not hit: out.append(dict(dataset=r["dataset"], rank=10**6)); continue
        sc = {c: rank_fn(r["unit"], c) for c in cand}; best = min(sc[c] for c in hit)       # smaller = better
        rank = 1 + sum(1 for c in cand if c not in partners and sc[c] <= best)               # ties count against the ranker
        out.append(dict(dataset=r["dataset"], rank=rank))
    return out
gB = gate(lambda a, c: S["Dc"][a, c]); gD = gate(lambda a, c: Dd[a, c])
def table(g):
    o = {}
    for ds in ds_names + ["ALL", "PRIMARY"]:
        rr = [x for x in g if ds == "ALL" or (ds == "PRIMARY" and x["dataset"] in PRIMARY) or x["dataset"] == ds]
        if rr: o[ds] = dict(n=len(rr), top1=float(np.mean([x["rank"] <= 1 for x in rr])), top5=float(np.mean([x["rank"] <= 5 for x in rr])), in_top20=float(np.mean([x["rank"] < 10**6 for x in rr])))
    return o
gate_tab = {"B": table(gB), "R_DINO": table(gD)}
# ---------------- candidate set and orders
cand = set()
for r in R:
    for v in r["topk"]: cand.add((min(r["unit"], v), max(r["unit"], v)))
iu = within_pairs(S); keyc = np.array([(a, b) in cand for a, b in zip(*iu)])
same_gt = {}
ident_sets = [set(G[m].tolist()) for m in u["members"]]
share = np.array([bool(ident_sets[a] & ident_sets[b]) for a, b in zip(*iu)])
oB = np.argsort(S["Dc"][iu], kind="stable")
def order(kind):
    if kind == "B": o = oB
    elif kind == "ORACLE":
        k = (keyc & share)[oB]; o = np.concatenate([oB[k], oB[~k]])
    elif kind == "R_DINO":
        k = keyc[oB]; cc = oB[k]; cc = cc[np.argsort(Dd[iu][cc], kind="stable")]; o = np.concatenate([cc, oB[~k]])
    elif kind == "R_DINO_g": o = np.argsort(Dd[iu], kind="stable")
    return iu[0][o], iu[1][o]
res = dict(tag=tag, n_units=U, n_candidate_unit_pairs=len(cand), n_within_pairs=int(len(iu[0])), need=S["need"], gate=gate_tab, sim={}, summary_5000={})
for kind in ("B", "ORACLE", "R_DINO", "R_DINO_g"):
    A_, B_ = order(kind); run = simulate(S, A_, B_); res["sim"][kind] = run; res["summary_5000"][kind] = summarize(S, run, 5000)
    sm = res["summary_5000"][kind]; log(tag, f"{kind:9s} @5000: recovery all {sm['recovered_all']}/{sm['needed_all']} = {100*sm['recovery_all']:.1f}%  primary {100*sm['recovery_primary']:.1f}%  wasted {sm['wasted']}/{sm['queries']} = {100*sm['wasted_share']:.1f}%  exhausted_at {run['exhausted_at_queries']}")
json.dump(res, open(f"{OUT}/d16_{tag}.json", "w"))
log(tag, "gate B", json.dumps(gate_tab["B"]["PRIMARY"]), "R_DINO", json.dumps(gate_tab["R_DINO"]["PRIMARY"]))
