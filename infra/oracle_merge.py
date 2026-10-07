"""ORACLE-MERGE (experiment E1, headroom/e1/prereg.md). Default OFF (--oracle-merge); with the flag off `apply` returns its inputs untouched.

After every re-clustering (after NP3, before `memory.labels` is set) each pseudo-label *unit* (a distinct label; an outlier is a unit of one
image) is assigned the ground-truth identity that holds the majority of its images, and all units that share a majority identity are merged
into one label. A unit whose top two identity counts are tied is left unchanged. Nothing is ever split (an impure unit stays one unit), and
nothing is added: the merged label is the smallest label among its units, so the label range never grows. An identity belongs to exactly one
dataset, so merging by identity is merging within the dataset.

This is an upper-bound probe that uses GT; it is not a method.
"""
import collections
import json
import os

import numpy as np


def _dataset_of(path):
    # .../wildlifereid-10k/images/<Dataset>/...   (the pid is '<Dataset>_<n>', checked equal for all pool images in headroom/d1)
    return path.split("/wildlifereid-10k/images/")[1].split("/")[0]


def oracle_merge(labels, gt, dataset):
    """labels: (N,) int array; gt: (N,) identity per image; dataset: (N,) dataset name per image.
    -> (new_labels (N,) int64 copy, stats dict)."""
    labels = np.asarray(labels).astype(np.int64)
    gt = np.asarray(gt); dataset = np.asarray(dataset)
    units, uinv = np.unique(labels, return_inverse=True)
    gids, ginv = np.unique(gt, return_inverse=True)
    n_g = len(gids)
    key, cnt = np.unique(uinv.astype(np.int64) * n_g + ginv, return_counts=True)
    k_unit, k_gid = key // n_g, key % n_g
    best = {}                                                   # unit -> (top count, gid, second count)
    for u, g, c in zip(k_unit.tolist(), k_gid.tolist(), cnt.tolist()):
        b = best.get(u)
        if b is None: best[u] = [c, g, 0]
        elif c > b[0]: best[u] = [c, g, b[0]]
        elif c > b[2]: b[2] = c
    groups = collections.defaultdict(list)                      # majority gid -> units
    ties = 0
    for u in range(len(units)):
        top, g, second = best[u]
        if top == second: ties += 1; continue                   # tied: leave the unit unchanged
        groups[g].append(u)
    new_of_unit = units.copy()
    absorbed_by_ds = collections.Counter(); merged_groups = 0
    first_img_of_gid = {}
    for i, g in enumerate(ginv.tolist()):
        first_img_of_gid.setdefault(g, i)
    for g, us in groups.items():
        if len(us) < 2: continue
        merged_groups += 1
        target = int(units[us].min())
        for u in us: new_of_unit[u] = target
        absorbed_by_ds[str(dataset[first_img_of_gid[g]])] += len(us) - 1
    new = new_of_unit[uinv]
    stats = {"units_before": int(len(units)), "units_after": int(len(np.unique(new))), "merge_operations": int(sum(absorbed_by_ds.values())),
             "groups_merged": int(merged_groups), "tied_units_unchanged": int(ties), "images_relabelled": int((new != labels).sum()),
             "merge_operations_by_dataset": dict(sorted(absorbed_by_ds.items()))}
    assert stats["units_before"] - stats["units_after"] == stats["merge_operations"]
    return new, stats


def apply(args, epoch, pseudo_labels, pseudo_labeled_dataset, train_sorted):
    """The call site in train_al.py. Flag off: returns the two input objects themselves, untouched.
    Flag on: returns (new pseudo_labels LongTensor, new pseudo_labeled_dataset list) and logs / appends the merge statistics."""
    if not getattr(args, "oracle_merge", False):
        return pseudo_labels, pseudo_labeled_dataset
    import torch
    labels = pseudo_labels.cpu().numpy() if hasattr(pseudo_labels, "cpu") else np.asarray(pseudo_labels)
    gt = [pid for _, pid, _ in train_sorted]
    ds = [_dataset_of(p) for p, _, _ in train_sorted]
    new, st = oracle_merge(labels, gt, ds)
    st["epoch"] = int(epoch)
    print("[oracle-merge]", json.dumps(st), flush=True)
    if getattr(args, "logs_dir", None):
        with open(os.path.join(args.logs_dir, "oracle_merge.jsonl"), "a") as f:
            f.write(json.dumps(st) + "\n")
    pl = torch.as_tensor(new, dtype=torch.long)
    pld = [(fname, int(l), cid) for (fname, _, cid), l in zip(train_sorted, new)]
    return pl, pld
