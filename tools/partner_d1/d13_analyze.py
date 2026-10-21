"""D1.3 / D1.4 for one state.  usage: d13_analyze.py seed S   (epoch-50 features of C0 seed S, epoch-40 constraints, final_labels)
                                   d13_analyze.py kit95      (the kit's own epoch-95 100-epoch state, recomputed)"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frag import *
which = sys.argv[1]; s = int(sys.argv[2]) if which == "seed" else None
meta = pool_meta(); G, dset = meta["G"], meta["dset"]; ds_names = sorted(set(dset.tolist()))
if which == "seed":
    st = torch.load(f"{LOGS[s]}/resume_epoch40.pth", map_location="cpu", weights_only=False)
    A, thr = st["args"], float(st["indep_thres"]); ml, cl = load_constraints(s)
    assert ml == [tuple(map(int, p)) for p in st["ml_pairs"]] and cl == [tuple(map(int, p)) for p in st["cl_pairs"]]
    feat = np.load(f"{OUT}/feat_s{s}.npy"); tag = f"s{s}"
    P, aux = final_labels(torch.from_numpy(feat).cuda(), ml, cl, indep_thres=thr, return_all=True); rr = aux["rerank"]
    np.save(f"{OUT}/P_{tag}.npy", P)
else:
    st = torch.load("/bhallukah/dd25007/AAS-ideas-out/kit/resume_epoch95.pth", map_location="cpu", weights_only=False)
    A = st["args"]; ml = [tuple(map(int, p)) for p in st["ml_pairs"]]; cl = [tuple(map(int, p)) for p in st["cl_pairs"]]
    feat = st["memory_features"].float().numpy(); P = np.load(f"{KIT_DATA}/P.npz")["P"]; rr = np.load("/bhallukah/dd25007/AAS-ideas-out/kit/rerank_e95.npy"); tag = "kit95"
log(tag, "feat", feat.shape, "P units", len(np.unique(P)), "ml", len(ml), "cl", len(cl))
V1, V2 = make_views(feat, rr, ml, cl, G, A); del rr
if which == "kit95":
    v = np.load(f"{KIT_DATA}/views.npz"); log("recomputed views == shipped kit views.npz:", bool(np.array_equal(v["V1"], V1) and np.array_equal(v["V2"], V2)))
np.savez(f"{OUT}/views_{tag}.npz", V1=V1, V2=V2)
fr, glob_ = fragmentation(P, G, dset, feat, ml, cl, V1, V2, ds_names)
u = units_of(P, feat, dset); rows = fragment_rows(u, G, dset, ds_names)
ranks = partner_ranks(u, rows, K=20)
tab = share_table(ranks, ds_names)
nfrag = {d: sum(1 for r in rows if r["dataset"] == d) for d in ds_names}
res = dict(tag=tag, glob=glob_, frag=fr, n_fragments=nfrag, n_fragment_rows_with_partner=len(ranks), partner_rank=tab,
           n_fragment_units=len({r["unit"] for r in rows}), ranks=[{k: v for k, v in r.items()} for r in ranks])
json.dump(res, open(f"{OUT}/d13_{tag}.json", "w"))
log(tag, "needed merges (all)", sum(v["needed_merges"] for v in fr.values()), "fragments", sum(nfrag.values()), "units", glob_)
for d in ds_names: log(f"{d:26s} needed {fr[d]['needed_merges']:4d} frags {nfrag[d]:4d} FS {fr[d]['false_split_pairs']:9d} FM {fr[d]['false_merge_pairs']:8d} agree-wrong {fr[d]['agree_wrong_share']:.3f} | B top1 {tab[d]['top1'] if tab[d] else None} top5 {tab[d]['top5'] if tab[d] else None}")
log("PRIMARY B", tab["PRIMARY"], "ALL", tab["ALL"])
