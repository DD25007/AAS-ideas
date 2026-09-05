"""Ranking metrics for the wildlife re-ID evaluation.

Only `eval_wildlife` is live -- it is called from `utils.py` (three sites) with the
gallery/query splits that `dataset/wildlife.py` builds. The interactive `main()` and its
`split_protocol` / `get_shuffled_indices` / `generate_report` helpers were removed
2026-09-05: nothing imported them, `main()` passed an int where a DataFrame was expected so
it could not run, and its `protocol_type == 2` branch read a hardcoded path
(`../../../cnnarch/...`) that does not exist in this repo. See Todo.md R9.
"""

import numpy as np

# borrowed from fastreid: https://github.com/JDAI-CV/fast-reid
def eval_wildlife(distmat, q_pids, g_pids, max_rank):
    """Evaluation for wildlife dataset."""
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.0  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]

        # compute cmc curve
        matches = (g_pids[order] == q_pid).astype(np.int32)
        raw_cmc = (
            matches.copy()
        )  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP
