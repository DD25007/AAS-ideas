"""Reproduction check, run from the kit: (1) H recomputed from data/labels_by_epoch.npz equals the shipped data/H.npz; (2) the D3 simulator with the D2 order
(ranker M) reproduces D2 within-dataset rule M exactly (recovered per dataset, wasted queries, at all 6 budgets) against ref/d2_sim.json;
(3) the D4 simulator with the model disabled ('off') reproduces it too. Exit code 0 only if all pass.  Usage: python check_reproduction.py [--skip-d4]"""
import sys, os, subprocess, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from history import *
d = load(); u = units(d); EP = [t for t in resume_epochs() if t < 95]
Hn, _ = history(u["unit"], u["U"], EP); Hs = load_H()
ok1 = bool(np.allclose(Hn, Hs, atol=1e-6)); log("H from labels == shipped H.npz (atol 1e-6):", ok1, "| max abs diff", float(np.abs(Hn - Hs).max()), "| epochs", EP)
res = {"H_matches": ok1}
for name, cmd, f in (("d3_ruleM", [sys.executable, f"{HERE}/sim_d3.py", "M"], "m_check_d3.json"), ("d4_off", [sys.executable, f"{HERE}/s_adaptive.py", "off"], "m_check_d4.json")):
    if name == "d4_off" and "--skip-d4" in sys.argv: continue
    p = subprocess.run(cmd, capture_output=True, text=True); print(p.stdout[-600:], p.stderr[-300:])
    j = json.load(open(f"{WORK}/{f}")); res[name] = list(j.values())[0]
log("RESULT", res)
sys.exit(0 if all(res.values()) else 1)
