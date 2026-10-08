"""D1 step 2: reproduce the final re-clustering exactly as train_al.py does at the start of the epoch after the newest
resume file (here epoch 96, from resume_epoch95.pth), CPU only. Each block cites the train_al.py lines it mirrors.
Intermediates go to SCRATCH (not the repo); the pseudo-labels P go to $FRAGSIM_WORK/P.npz (shipped copy: data/P.npz).
"""
import os, sys, json, time, collections
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch, faiss
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import REPO, BIG, WORK
SCR = BIG                       # rerank cache: $FRAGSIM_BIG/rerank_e95.npy (recomputed on CPU if absent)
sys.path.insert(0, REPO); os.chdir(REPO)
import spcl.utils.constrained_clustering as cc
from spcl.utils.faiss_rerank import k_reciprocal_neigh
import torch.nn.functional as F

RESUME = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BIG, "resume_epoch95.pth")
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

st = torch.load(RESUME, map_location="cpu", weights_only=False)
A = st["args"]
feat = st["memory_features"].float().contiguous()                   # memory bank after epoch st["epoch"]
ml_pairs = [tuple(map(int, p)) for p in st["ml_pairs"]]; cl_pairs = [tuple(map(int, p)) for p in st["cl_pairs"]]
indep_thres = st["indep_thres"]
log("resume", RESUME, "epoch", st["epoch"], "N", feat.shape[0], "ML", len(ml_pairs), "CL", len(cl_pairs), "indep_thres", indep_thres)
assert A["canonical_pairs"] and not A["np3_per_cluster_skip"] and A["eps"] == 0.6 and A["eps_gap"] == 0.02 and A["k1"] == 30 and A["k2"] == 6

# --- train_al.py:414-416  rerank_dist = rerank_fn(features, k1, k2); rerank_fn = infra.fast_rerank (bit-identical to
# spcl/utils/faiss_rerank.compute_jaccard_distance). Same algorithm here, CPU: faiss CPU exact L2 search instead of the GPU search,
# torch CPU matmul instead of CUDA. Identical up to float rounding; validated below against the training log.
def rerank_cpu(target_features, k1=30, k2=6):
    N = target_features.size(0)
    index = faiss.IndexFlatL2(target_features.size(-1))
    x = target_features.numpy(); index.add(x)
    _, initial_rank = index.search(x, k1)                            # (N, k1) incl. self, as faiss_rerank.py search_option!=0
    nn_k1 = [k_reciprocal_neigh(initial_rank, i, k1) for i in range(N)]
    nn_k1_half = [k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))) for i in range(N)]
    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):                                               # faiss_rerank.py:90-110
        kri = nn_k1[i]; kre = kri
        for cand in kri:
            c_idx = nn_k1_half[cand]
            if len(np.intersect1d(c_idx, kri)) > 2 / 3 * len(c_idx):
                kre = np.append(kre, c_idx)
        kre = np.unique(kre)
        dist = 2 - 2 * torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[kre].t())
        V[i, kre] = F.softmax(-dist, dim=1).view(-1).numpy().astype(np.float32)
    del nn_k1, nn_k1_half
    if k2 != 1:                                                      # faiss_rerank.py:128-132
        V_qe = np.zeros_like(V)
        for i in range(N):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe; del V_qe
    invIndex = [np.where(V[:, i] != 0)[0] for i in range(N)]
    jd = np.zeros((N, N), dtype=np.float32)
    for i in range(N):                                               # faiss_rerank.py:138-152
        temp_min = np.zeros((1, N), dtype=np.float32)
        nz = np.where(V[i, :] != 0)[0]
        imgs = [invIndex[ind] for ind in nz]
        for j in range(len(nz)):
            temp_min[0, imgs[j]] = temp_min[0, imgs[j]] + np.minimum(V[i, nz[j]], V[imgs[j], nz[j]])
        jd[i] = 1 - temp_min / (2 - temp_min)
    jd[jd < 0] = 0.0
    return jd

rp = os.path.join(SCR, f"rerank_e{st['epoch']}.npy")
if os.path.exists(rp):
    rerank_dist = np.load(rp); log("rerank loaded from cache")
else:
    rerank_dist = rerank_cpu(feat, A["k1"], A["k2"]); np.save(rp, rerank_dist); log("rerank computed", rerank_dist.shape, rerank_dist.dtype)

# --- train_al.py:458-491  DBSCAN at eps, eps-gap, eps+gap on the precomputed distance
eps, gap = A["eps"], 0.02
pl = DBSCAN(eps=eps, min_samples=4, metric="precomputed").fit_predict(rerank_dist)
pl_t = DBSCAN(eps=eps - gap, min_samples=4, metric="precomputed").fit_predict(rerank_dist)
pl_l = DBSCAN(eps=eps + gap, min_samples=4, metric="precomputed").fit_predict(rerank_dist)
nid = lambda x: len(set(x)) - (1 if -1 in x else 0)
num_ids, num_t, num_l = nid(pl), nid(pl_t), nid(pl_l)
log("DBSCAN raw clusters", num_ids, "noise", int((pl == -1).sum()))

def gen(cluster_id, num):                                           # train_al.py:594-605
    labels, outliers = [], 0
    for id_ in cluster_id:
        if id_ != -1: labels.append(id_)
        else: labels.append(num + outliers); outliers += 1
    return torch.Tensor(labels).long()
pseudo_labels, pl_tight, pl_loose = gen(pl, num_ids), gen(pl_t, num_t), gen(pl_l, num_l)   # train_al.py:607-613

N = pseudo_labels.size(0)                                           # train_al.py:615-642
ls = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
lt = pl_tight.expand(N, N).eq(pl_tight.expand(N, N).t()).float()
ll = pl_loose.expand(N, N).eq(pl_loose.expand(N, N).t()).float()
R_comp = 1 - torch.min(ls, lt).sum(-1) / torch.max(ls, lt).sum(-1)
R_indep = 1 - torch.min(ls, ll).sum(-1) / torch.max(ls, ll).sum(-1)
del ls, lt, ll
cR_comp, cR_indep = collections.defaultdict(list), collections.defaultdict(list)   # train_al.py:644-660
for comp, indep, label in zip(R_comp, R_indep, pseudo_labels):
    cR_comp[label.item()].append(comp.item()); cR_indep[label.item()].append(indep.item())
cR_comp = [min(cR_comp[i]) for i in sorted(cR_comp.keys())]
cR_indep = [min(cR_indep[i]) for i in sorted(cR_indep.keys())]

constrained_pseudo_labels, outl = [], 0                              # train_al.py:675-694
for i, label in enumerate(pseudo_labels):
    if (cR_indep[label.item()] <= indep_thres) and (R_comp[i].item() <= cR_comp[label.item()]):
        constrained_pseudo_labels.append(label.item())
    else:
        pseudo_labels[i] = len(cR_indep) + outl
        constrained_pseudo_labels.append(pseudo_labels[i])
        outl += 1
pre_np3 = np.array(constrained_pseudo_labels)

cc.CANONICAL_PAIRS = True; cc.NP3_PER_CLUSTER_SKIP = False           # train_al.py:247-253 (--canonical-pairs, no --np3-per-cluster-skip)
P_arr = cc.constrained_clustering(np.array(constrained_pseudo_labels), rerank_dist, "~DBSCAN", min_samples=4,
                                  ml_pairs=ml_pairs, cl_pairs=cl_pairs)          # train_al.py:704-711
pseudo_labels = torch.tensor(np.asarray(P_arr)).long()               # train_al.py:713-716 (pseudo_labels[i] = label)

cnt = collections.Counter(pseudo_labels.tolist()); sizes = np.array(list(cnt.values()))   # train_al.py:719-730
n_clusters, n_single = int((sizes > 1).sum()), int((sizes == 1).sum())
log(f"==> Statistics for epoch {st['epoch'] + 1}: {n_clusters} clusters, {n_single} un-clustered instances, "
    f"R_indep threshold is {1 - indep_thres}")
P = pseudo_labels.numpy().astype(np.int64)
np.savez(os.path.join(WORK, "P.npz"), P=P, pre_np3=pre_np3, raw_dbscan=pl)
json.dump({"epoch_of_features": int(st["epoch"]), "epoch_relabelled": int(st["epoch"]) + 1, "n_clusters": n_clusters,
           "n_unclustered": n_single, "raw_dbscan_clusters": num_ids, "raw_noise": int((pl == -1).sum())},
          open(os.path.join(WORK, "recluster_stats.json"), "w"))
# reproduction checks
saved = st["memory_labels"].numpy()
from sklearn.metrics import adjusted_rand_score
log("ARI(P, saved memory_labels of epoch", st["epoch"], ") =", round(adjusted_rand_score(saved, P), 6),
    "(saved labels = relabelling at the START of epoch", st["epoch"], ", from epoch", st["epoch"] - 1, "features: expected high, not 1)")
