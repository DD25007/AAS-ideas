"""Track F: a faster k-reciprocal rerank that must be BIT-IDENTICAL to the original.

`spcl/utils/faiss_rerank.py` is untouched; this is a separate implementation selected by
`train_al.py --fast-rerank` (default OFF). The profile in `reports/phase2.md` §9 (git tag `beca-al-archive`) measured one
epoch-0 rerank at 10,951 s, of which:

  V loop flush/scatter   6,420 s (59 %)
  V_qe                   3,520 s (32 %)
  everything else          995 s  (9 %)  -- the arithmetic itself is ~2 %

Two changes, both chosen because they cannot alter a single value:

1. **Hoist the per-row index transfer out of the V loop.** The original indexes a CUDA tensor
   with a NumPy array once per row (`target_features[k_reciprocal_expansion_index]`,
   faiss_rerank.py:108-110). Copying a pageable host array is synchronous with respect to the
   host, so each of the 20,565 rows drains the CUDA queue; on a card shared with other users
   each of those waits for a scheduling slot. Here the expansion indices are computed first —
   by the same code, giving the same arrays — concatenated, and moved to the device ONCE. The
   per-row gather and matmul are then exactly the same operations on exactly the same values.

2. **Block V_qe.** `V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)` becomes
   `V_qe[b0:b1] = V[initial_rank[b0:b1, :k2]].mean(axis=1)`. Verified bit-identical on random
   float32 of the same shapes at both block sizes (`reports/phase2/vqe_blocking_probe.txt` at tag `beca-al-archive`);
   the real gate is the matrix comparison in Track F.

Everything else — the faiss search, `k_reciprocal_neigh`, the expansion rule, the 512-row
flush, the inverse index and the jaccard loop — is copied unchanged from the original at
commit 290c992.
"""

import time

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from faiss_utils import (
    index_init_cpu,
    index_init_gpu,
    search_index_pytorch,
    search_raw_array_pytorch,
)
from spcl.utils.faiss_rerank import k_reciprocal_neigh

BLOCK = 512


def compute_jaccard_distance_fast(
    target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False
):
    end = time.time()
    if print_flag:
        print("Computing jaccard distance... [fast-rerank]")

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    # copied verbatim from spcl/utils/faiss_rerank.py:43-68 -- the default search_option=0
    # path is search_raw_array_pytorch, NOT an index search, and it decides initial_rank
    if search_option == 0:
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(
            res, target_features, target_features, k1
        )
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 1:
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 2:
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    # --- change 1: expansion indices first, one host->device copy for all of them ---
    expansions = []
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if len(
                np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )
        expansions.append(np.unique(k_reciprocal_expansion_index))

    offsets = np.cumsum([0] + [len(e) for e in expansions])
    flat = torch.as_tensor(np.concatenate(expansions), dtype=torch.long, device=target_features.device)

    V = np.zeros((N, N), dtype=mat_type)
    _pending = []

    def _flush():
        if not _pending:
            return
        big = torch.cat([s for _, _, s in _pending]).cpu().numpy()
        off = 0
        for _i, _idx, _s in _pending:
            n = _s.numel()
            V[_i, _idx] = big[off : off + n].astype(mat_type)
            off += n
        _pending.clear()

    for i in range(N):
        idx_gpu = flat[offsets[i] : offsets[i + 1]]
        dist = 2 - 2 * torch.mm(
            target_features[i].unsqueeze(0).contiguous(),
            target_features[idx_gpu].t(),
        )
        _pending.append((i, expansions[i], F.softmax(-dist, dim=1).view(-1)))
        if len(_pending) >= 512:
            _flush()
    _flush()
    del nn_k1, nn_k1_half, flat

    # --- change 2: blocked V_qe ---
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for b0 in range(0, N, BLOCK):
            b1 = min(b0 + BLOCK, N)
            V_qe[b0:b1] = V[initial_rank[b0:b1, :k2]].mean(axis=1)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    del invIndex, V
    pos_bool = jaccard_dist < 0
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time() - end))

    return jaccard_dist


def pick(args, original):
    """The rerank function this run should use: the original unless --fast-rerank."""
    if getattr(args, "fast_rerank", False):
        print("[fast-rerank] using infra/fast_rerank.compute_jaccard_distance_fast", flush=True)
        return compute_jaccard_distance_fast
    return original
