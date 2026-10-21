"""Phase D1 shared code.

final_labels(features, ml, cl, ...) = the training pseudo-labels of one re-clustering step of train_al.py (main_worker, lines 413-713).
What train_al.py exposes as importable functions is imported (get_data, get_test_loader, create_model, constrained_clustering,
infra.fast_rerank.compute_jaccard_distance_fast = what `--fast-rerank` selects through _fast.pick).  The block between the rerank and the
constrained_clustering call is inline in main_worker (not a function), so it is mirrored statement for statement, each block citing its lines;
tools/partner_d1/d12_validate.py proves the mirror with array_equal against saved labels.
"""
import os, sys, json, time, collections
REPO = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np, torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
import spcl.utils.constrained_clustering as _cc
from spcl.utils.constrained_clustering import constrained_clustering
from infra.fast_rerank import compute_jaccard_distance_fast

OUT = "/bhallukah/dd25007/phaseD1/out"
RUNS = "/bhallukah/dd25007/archive/idea-1/runs/C0_consam_det"
LOGS = {0: f"{RUNS}/logs", 1: f"{RUNS}/s1/logs", 2: f"{RUNS}/s2/logs", 3: f"{RUNS}/s3/logs"}
CON = "/bhallukah/dd25007/phaseA/constraints"
KIT_DATA = os.path.join(REPO, "tools", "fragment_sim", "data")
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:8.1f}s]", *a, flush=True)


def load_constraints(seed, epoch=40):
    c = np.load(f"{CON}/C0_s{seed}_e{epoch}.npz")
    return [tuple(map(int, p)) for p in c["ml"]], [tuple(map(int, p)) for p in c["cl"]]   # list order = the sampler's order (as in resume file)


def final_labels(features, ml, cl, indep_thres=None, rerank_dist=None, k1=30, k2=6, eps=0.6, eps_gap=0.02, return_all=False):
    """features: (N, D) L2-normalised torch tensor (cuda unless rerank_dist is given); ml/cl: lists of (a, b) in sampler order.
    indep_thres: the epoch-0 reliability threshold carried in the resume file (train_al.py:553-561 computes it at epoch 0 only);
    None = compute it from this clustering exactly as the epoch-0 pass does.  Returns the post-NP3 labels (np.int64)."""
    _cc.CANONICAL_PAIRS = True; _cc.NP3_PER_CLUSTER_SKIP = False                   # train_al.py:247-253 with --canonical-pairs, no --np3-per-cluster-skip
    if rerank_dist is None:                                                        # train_al.py:414-416 (rerank_fn = _fast.pick -> fast)
        rerank_dist = compute_jaccard_distance_fast(features.clone(), k1=k1, k2=k2)
    eps_tight, eps_loose = eps - eps_gap, eps + eps_gap                            # train_al.py:461-476
    pseudo_labels = DBSCAN(eps=eps, min_samples=4, metric="precomputed").fit_predict(rerank_dist)            # :480-484
    pseudo_labels_tight = DBSCAN(eps=eps_tight, min_samples=4, metric="precomputed").fit_predict(rerank_dist)
    pseudo_labels_loose = DBSCAN(eps=eps_loose, min_samples=4, metric="precomputed").fit_predict(rerank_dist)
    num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)                                    # :486-492
    num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
    num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)

    def generate_pseudo_labels(cluster_id, num):                                   # :594-605 (noise points get fresh ids)
        labels, outliers = [], 0
        for id in cluster_id:
            if id != -1: labels.append(id)
            else: labels.append(num + outliers); outliers += 1
        return torch.Tensor(labels).long()
    pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)                                           # :607-613
    pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
    pseudo_labels_loose = generate_pseudo_labels(pseudo_labels_loose, num_ids_loose)

    N = pseudo_labels.size(0)                                                      # :615-642 R_comp / R_indep
    label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
    label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
    label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()
    R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(label_sim, label_sim_tight).sum(-1)
    R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(label_sim, label_sim_loose).sum(-1)
    del label_sim, label_sim_tight, label_sim_loose
    cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)           # :644-660
    cluster_img_num = collections.defaultdict(int)
    for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
        cluster_R_comp[label.item()].append(comp.item()); cluster_R_indep[label.item()].append(indep.item()); cluster_img_num[label.item()] += 1
    cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
    cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
    if indep_thres is None:                                                        # :553-561, epoch 0 only
        noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if cluster_img_num[num] > 1]
        indep_thres = np.sort(noins)[min(len(noins) - 1, np.round(len(noins) * 0.9).astype("int"))]
    constrained_pseudo_labels, outliers = [], 0                                    # :675-694 reliability filter
    for i, label in enumerate(pseudo_labels):
        indep_score = cluster_R_indep[label.item()]; comp_score = R_comp[i]
        if (indep_score <= indep_thres) and (comp_score.item() <= cluster_R_comp[label.item()]):
            constrained_pseudo_labels.append(label.item())
        else:
            pseudo_labels[i] = len(cluster_R_indep) + outliers
            constrained_pseudo_labels.append(pseudo_labels[i]); outliers += 1
    pre = np.array(constrained_pseudo_labels)
    post = constrained_clustering(np.array(constrained_pseudo_labels), rerank_dist, "~DBSCAN", min_samples=4, ml_pairs=ml, cl_pairs=cl)   # :704-711
    post = np.asarray(post).astype(np.int64)
    if return_all:
        return post, dict(pre=pre, dbscan_raw=None, indep_thres=float(indep_thres), rerank=rerank_dist)
    return post


def pool_meta():
    p = np.load(f"{KIT_DATA}/pool.npz")
    pid, dset = p["pid"], p["dset"]
    return dict(paths=p["paths"] if "paths" in p.files else None, pid=pid, dset=dset, G=np.unique(pid, return_inverse=True)[1])
