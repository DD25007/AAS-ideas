"""D1.5 candidates: for each seed, the unit pairs (fragment unit u, one of its 20 nearest same-dataset units by centroid cosine) and the 3x3 image pairs
(representatives per unit = medoid + the 2 images nearest to it, fewer if the unit is smaller).  CPU only.  Writes $OUT/cand_<tag>.npz and prints the counts."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frag import *

def reps_of(u, feat, n_extra=2):
    reps = []
    for m, med in zip(u["members"], u["medoid"]):
        if len(m) == 1: reps.append([int(m[0])]); continue
        sims = feat[m] @ feat[med]; order = np.argsort(-sims, kind="stable")
        r = [int(med)] + [int(m[i]) for i in order if int(m[i]) != int(med)][:n_extra]
        reps.append(r)
    return reps

def build(tag, P, feat, dset, ranks_json, K=20):
    u = units_of(P, feat, dset); reps = reps_of(u, feat)
    rk = json.load(open(ranks_json))["ranks"]
    pairs = set(); funits = set()
    for r in rk:
        funits.add(r["unit"])
        for v in r["topk"][:K]: pairs.add((min(r["unit"], v), max(r["unit"], v)))
    pairs = np.array(sorted(pairs), dtype=np.int64)
    n_img = sum(len(reps[a]) * len(reps[b]) for a, b in pairs)
    return u, reps, pairs, funits, n_img

if __name__ == "__main__":
    meta = pool_meta(); dset = meta["dset"]; out = {}
    for tag in sys.argv[1:]:
        if tag.startswith("s"): P = np.load(f"{OUT}/P_{tag}.npy"); feat = np.load(f"{OUT}/feat_{tag}.npy")
        else: P = np.load(f"{KIT_DATA}/P.npz")["P"]; feat = torch.load("/bhallukah/dd25007/AAS-ideas-out/kit/resume_epoch95.pth", map_location="cpu", weights_only=False)["memory_features"].float().numpy()
        u, reps, pairs, funits, n_img = build(tag, P, feat, dset, f"{OUT}/d13_{tag}.json")
        n_med = len(pairs)
        # unique images that need keypoints (cached for all 20,565)
        out[tag] = dict(n_fragment_units=len(funits), n_unit_pairs=int(n_med), n_image_pairs_3x3=int(n_img), n_image_pairs_medoid_only=int(n_med))
        np.savez(f"{OUT}/cand_{tag}.npz", pairs=pairs, reps=np.array([np.array(r + [-1] * (3 - len(r))) for r in reps]))
        log(tag, json.dumps(out[tag]))
    json.dump(out, open(f"{OUT}/d15_candidate_counts_{'_'.join(sys.argv[1:])}.json", "w"), indent=1)
