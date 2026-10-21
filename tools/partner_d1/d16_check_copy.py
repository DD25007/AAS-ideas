"""D1.6 pre-check of my state-parametrised simulator copy: on the kit epoch-95 state, order B must reproduce ref/d2_sim.json within/M exactly (recovered per dataset + wasted at all budgets)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_d1 import *
meta = pool_meta(); G, dset = meta["G"], meta["dset"]
st = torch.load("/bhallukah/dd25007/AAS-ideas-out/kit/resume_epoch95.pth", map_location="cpu", weights_only=False)
ml = [tuple(map(int, p)) for p in st["ml_pairs"]]; cl = [tuple(map(int, p)) for p in st["cl_pairs"]]
S = make_state(np.load(f"{KIT_DATA}/P.npz")["P"], G, dset, st["memory_features"].float().numpy(), ml, cl)
A_, B_ = order_B(S); run = simulate(S, A_, B_)
ref = json.load(open(f"{REPO}/tools/fragment_sim/ref/d2_sim.json")); w = [r for r in ref["runs"] if r["scope"] == "within" and r["rule"] == "M"][0]
same = all(run["snapshots"][b]["recovered"] == w["snapshots"][str(b)]["recovered"] and run["snapshots"][b]["wasted"] == w["snapshots"][str(b)]["wasted"] and run["snapshots"][b]["free_merges"] == w["snapshots"][str(b)]["free_merges"] for b in BUD)
s5 = summarize(S, run)
log("copy reproduces D2 within/M exactly at all 6 budgets:", same, "| @5000 primary", s5["recovered_primary"], "/", s5["needed_primary"], "all", s5["recovered_all"], "/", s5["needed_all"], "wasted", s5["wasted"], "/", s5["queries"])
json.dump({"copy_reproduces_D2_within_M": bool(same), "summary_5000": s5}, open(f"{OUT}/d16_copy_check.json", "w"), indent=1)
