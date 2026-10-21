"""D1.5 matcher scoring: ALIKED (idea-1's cached native-resolution keypoints, 2048 max) + LightGlue (aliked weights), score = number of LightGlue matches and mean match score
per image pair (idea-1's own statistic; lg_common.match / mv2_match.py).  Candidates from d15_candidates.py (3x3 representatives per unit pair).
  --mode time --tag s0 --n 1000   : score a random sample of N image pairs, report pairs/s  (no cache written)
  --mode full --tag s0            : score every image pair of that tag, resumable (.partial.npz every 5000 pairs)"""
import os, sys, json, time, argparse
os.environ.setdefault("TORCH_HOME", "/storage/dd25007/cache/torch"); os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import h5py
H5 = "/bhallukah/dd25007/HITL-ReID-idea1-cache/mv/cache/aliked_native.h5"

class Store:
    def __init__(self, path):
        f = h5py.File(path, "r"); assert (f["starts"][:] >= 0).all()
        t = time.time(); self.kp = f["keypoints"][:]; self.desc = f["descriptors"][:]; self.sc = f["keypoint_scores"][:]
        self.starts, self.counts, self.size = f["starts"][:], f["counts"][:], f["image_size"][:]; f.close()
        log("h5 loaded to RAM in", round(time.time() - t, 1), "s", self.kp.shape)
    def get(self, i, dev="cuda"):
        s, c = int(self.starts[i]), int(self.counts[i])
        return {"keypoints": torch.from_numpy(self.kp[s:s + c]).unsqueeze(0).to(dev), "descriptors": torch.from_numpy(self.desc[s:s + c]).unsqueeze(0).to(dev),
                "keypoint_scores": torch.from_numpy(self.sc[s:s + c]).unsqueeze(0).to(dev), "image_size": torch.from_numpy(self.size[i:i + 1]).to(dev)}

def image_pairs(tag):
    z = np.load(f"{OUT}/cand_{tag}.npz"); pairs, reps = z["pairs"], z["reps"]; ip = []; up = []
    for k, (a, b) in enumerate(pairs):
        for ra in reps[a]:
            if ra < 0: continue
            for rb in reps[b]:
                if rb < 0: continue
                ip.append((int(ra), int(rb))); up.append(k)
    return np.array(ip, dtype=np.int64), np.array(up, dtype=np.int64)

def check_index_mapping(store, n=60):
    from PIL import Image
    paths = np.load(f"{KIT_DATA}/pool.npz")["paths"]; rng = np.random.default_rng(1); bad = 0
    for i in rng.choice(len(paths), n, replace=False):
        w, h = Image.open(paths[i]).size
        if not (abs(store.size[i][0] - w) < 1 and abs(store.size[i][1] - h) < 1): bad += 1
    return bad, n

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    from lightglue import LightGlue
    from lightglue.utils import rbd
    store = Store(H5); bad, n = check_index_mapping(store); log(f"h5 index->pool path check: {bad} mismatches of {n} (image_size vs file size)")
    assert bad == 0
    matcher = LightGlue(features="aliked").eval().cuda()
    ip, up = image_pairs(a.tag); log(a.tag, "image pairs", len(ip), "unit pairs", int(up.max()) + 1)
    def score(i, j):
        f0, f1 = store.get(i), store.get(j)
        with torch.no_grad(): m = rbd(matcher({"image0": f0, "image1": f1}))
        mt = m["matches"]; sc = m.get("scores")
        return int(mt.shape[0]), (float(sc.mean()) if sc is not None and sc.numel() else 0.0)
    if a.mode == "time":
        rng = np.random.default_rng(0); sel = rng.choice(len(ip), min(a.n, len(ip)), replace=False)
        for k in sel[:20]: score(*ip[k])                       # warm-up (not timed)
        torch.cuda.synchronize(); t = time.time(); res = []
        for k in sel: res.append(score(*ip[k]))
        torch.cuda.synchronize(); dt = time.time() - t
        nm = np.array([r[0] for r in res])
        rep = dict(tag=a.tag, n_pairs_timed=len(sel), seconds=dt, pairs_per_s=len(sel) / dt, mean_matches=float(nm.mean()), median_matches=float(np.median(nm)),
                   total_image_pairs_this_tag=int(len(ip)), projected_h_this_tag=len(ip) / (len(sel) / dt) / 3600)
        log(json.dumps(rep)); json.dump(rep, open(f"{OUT}/d15_timing_{a.tag}.json", "w"), indent=1); return
    part = f"{OUT}/match_{a.tag}.partial.npz"
    if os.path.exists(part): z = np.load(part); nmat, msc = z["n_matches"].copy(), z["mean_score"].copy()
    else: nmat, msc = np.full(len(ip), -1, np.int32), np.full(len(ip), np.nan, np.float32)
    t0 = time.time(); done = 0
    for k in range(len(ip)):
        if nmat[k] >= 0: continue
        nmat[k], msc[k] = score(*ip[k]); done += 1
        if done % 5000 == 0:
            np.savez(part + ".tmp.npz", n_matches=nmat, mean_score=msc); os.replace(part + ".tmp.npz", part)
            log(f"{k + 1}/{len(ip)} {done / (time.time() - t0):.2f} pairs/s")
    np.savez(f"{OUT}/match_{a.tag}.npz", n_matches=nmat, mean_score=msc, image_pairs=ip, unit_pair_idx=up); log("done", a.tag, f"{(time.time() - t0) / 3600:.2f} h")

if __name__ == "__main__": main()
