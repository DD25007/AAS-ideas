"""FINCH with a selectable partition index.

`spcl/utils/constrained_clustering.FINCH` returns immediately after the first
`get_clust`/`get_merge`, i.e. partition 0 -- the finest one. On the 13-dataset
combined pool that is ~5,400 clusters for ~1,500 identities, roughly 6.6x finer than
DBSCAN at the same eps, which is what makes the AAS disagreement signal saturate.

This module runs upstream's refinement loop so a coarser partition can be selected.
`partition=0` reproduces the existing wrapper exactly (asserted in
`investigations/diag_aas_mixture_draw.py`); nothing here changes the original.
"""

import numpy as np


def finch_partitions(data, initial_rank=None, distance="cosine",
                     ensure_early_exit=True, verbose=False,
                     use_ann_above_samples=70000):
    """-> (c, num_clust) where c is N x P (column p = partition p) and num_clust[p]
    is that partition's cluster count.

    Mirrors upstream finch.finch.FINCH; the repo wrapper is the p=0 prefix of it.
    """
    from finch.finch import clust_rank, get_clust, get_merge

    data = data.astype(np.float32)
    min_sim = None

    # Same version-tolerant call the repo wrapper uses (constrained_clustering.py:66-81):
    # 0.2.2+ renamed the kwargs and returns 4 values, so bind by keyword against the
    # installed signature rather than positionally.
    import inspect as _inspect

    _params = _inspect.signature(clust_rank).parameters
    if "ann_threshold" in _params:  # 0.2.2+
        _out = clust_rank(data, initial_rank=initial_rank, metric=distance,
                          verbose=verbose, ann_threshold=use_ann_above_samples)
    else:  # <= 0.2.1
        _out = clust_rank(data, use_ann_above_samples=use_ann_above_samples,
                          initial_rank=initial_rank, distance=distance, verbose=verbose)
    adj, orig_dist = _out[0], _out[1]

    initial_rank = None
    group, num_clust_curr = get_clust(adj, [], min_sim)
    c, mat = get_merge([], group, data)

    if ensure_early_exit and orig_dist.shape[-1] > 2:
        min_sim = np.max(orig_dist * adj.toarray())

    c = np.asarray(c).reshape(-1, 1)
    num_clust = [num_clust_curr]
    c_, exit_clust = c[:, 0], 2

    while exit_clust > 1:
        _out = (clust_rank(mat, initial_rank=None, metric=distance, verbose=verbose,
                           ann_threshold=use_ann_above_samples)
                if "ann_threshold" in _params else
                clust_rank(mat, use_ann_above_samples=use_ann_above_samples,
                           initial_rank=None, distance=distance, verbose=verbose))
        adj, orig_dist = _out[0], _out[1]
        u, num_clust_curr = get_clust(adj, orig_dist, min_sim)
        c_, mat = get_merge(c_, u, data)
        num_clust.append(num_clust_curr)
        c = np.column_stack((c, c_))
        exit_clust = num_clust[-2] - num_clust_curr
        if num_clust_curr == 1 or exit_clust < 1:
            num_clust = num_clust[:-1]
            c = c[:, :-1]
            break
    return c, num_clust


def finch_at(data, partition=0, **kw):
    """FINCH labels at `partition`, clamped to the deepest one actually computed.

    partition=0 is byte-identical to spcl.utils.constrained_clustering.FINCH.
    """
    c, num_clust = finch_partitions(data, **kw)
    p = min(partition, c.shape[1] - 1)
    return np.asarray(c[:, p]).ravel(), num_clust, p
