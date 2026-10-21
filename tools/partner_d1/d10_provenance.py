"""D1.0: locate each C0 seed's final weights / resume_epoch40 / Phase A constraints; read stored epochs; compare ML/CL sha with Phase A."""
import os, sys, json, hashlib, subprocess
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
RUNS = "/bhallukah/dd25007/archive/idea-1/runs/C0_consam_det"
LOGS = {0: f"{RUNS}/logs", 1: f"{RUNS}/s1/logs", 2: f"{RUNS}/s2/logs", 3: f"{RUNS}/s3/logs"}
CON = "/bhallukah/dd25007/phaseA/constraints"
OUT = "/bhallukah/dd25007/phaseD1/out"
def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
res = {}
for s, L in LOGS.items():
    r = {"dir": L}
    for name in ("checkpoint_50.pth.tar", "model_best.pth.tar", "checkpoint_40.pth.tar"):
        ck = torch.load(f"{L}/{name}", map_location="cpu", weights_only=False)
        r[name] = {"epoch": ck.get("epoch"), "best_mAP": ck.get("best_mAP"), "keys": sorted(ck.keys())}
        if name == "checkpoint_50.pth.tar": sd50 = ck["state_dict"]
        if name == "model_best.pth.tar":
            r["model_best_equals_ckpt50"] = all(torch.equal(sd50[k], ck["state_dict"][k]) for k in sd50)
    st = torch.load(f"{L}/resume_epoch40.pth", map_location="cpu", weights_only=False)
    ml = np.array([tuple(map(int, p)) for p in st["ml_pairs"]]); cl = np.array([tuple(map(int, p)) for p in st["cl_pairs"]])
    r["resume_epoch40"] = {"epoch": int(st["epoch"]), "n_ml": len(ml), "n_cl": len(cl), "indep_thres": float(st["indep_thres"]),
                           "ml_sha": sha(ml), "cl_sha": sha(cl), "keys": sorted(st.keys()), "mem_shape": list(st["memory_features"].shape)}
    c = np.load(f"{CON}/C0_s{s}_e40.npz"); r["phaseA_file"] = f"{CON}/C0_s{s}_e40.npz"; r["phaseA_keys"] = list(c.files)
    k_ml = [k for k in c.files if "ml" in k.lower()][0]; k_cl = [k for k in c.files if "cl" in k.lower()][0]
    pml, pcl = c[k_ml], c[k_cl]
    r["phaseA"] = {"n_ml": len(pml), "n_cl": len(pcl), "ml_sha": sha(pml), "cl_sha": sha(pcl), "ml_key": k_ml, "cl_key": k_cl, "ml_dtype": str(pml.dtype)}
    # same order? and same set?
    r["same_arrays_exact"] = bool(pml.shape == ml.shape and (pml == ml).all() and pcl.shape == cl.shape and (pcl == cl).all())
    r["same_sets"] = bool(set(map(tuple, pml.tolist())) == set(map(tuple, ml.tolist())) and set(map(tuple, pcl.tolist())) == set(map(tuple, cl.tolist())))
    res[s] = r; print(s, "epochs", {k: r[k]["epoch"] for k in ("checkpoint_50.pth.tar","model_best.pth.tar","checkpoint_40.pth.tar")}, "resume", r["resume_epoch40"]["epoch"], flush=True)
json.dump(res, open(f"{OUT}/d10_provenance.json", "w"), indent=1)

# ---- sha with the repo's own definition (spcl.utils.run_record._pairs_sha), on the resume lists and on the Phase A arrays
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from spcl.utils.run_record import _pairs_sha
sha_rows = {}
for s, L in LOGS.items():
    st = torch.load(f"{L}/resume_epoch40.pth", map_location="cpu", weights_only=False)
    c = np.load(f"{CON}/C0_s{s}_e40.npz")
    a = (_pairs_sha(st["ml_pairs"]), _pairs_sha(st["cl_pairs"]))
    b = (_pairs_sha([tuple(map(int, p)) for p in c["ml"]]), _pairs_sha([tuple(map(int, p)) for p in c["cl"]]))
    sha_rows[s] = {"resume_ml": a[0], "resume_cl": a[1], "phaseA_ml": b[0], "phaseA_cl": b[1], "equal": a == b}
    print("repo-sha seed", s, sha_rows[s], flush=True)
json.dump(sha_rows, open(f"{OUT}/d10_repo_sha.json", "w"), indent=1)
