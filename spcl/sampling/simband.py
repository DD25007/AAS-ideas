"""Similarity-band sampler: the null model for AAS.

No clustering, no committee, no regions of uncertainty. Given the same k-reciprocal
Jaccard matrix every other sampler sees, take `self.n` pairs uniformly at random from
those whose similarity `1 - dist` lies in a band `[lo, hi]`.

The band is a calibration knob, not a discovery: it is chosen at cycle 1 so the
resulting ML fraction matches AAS's, which isolates "does AAS's machinery select
better pairs than a similarity threshold at the same difficulty?" from "does AAS
just operate at a different point on the easy/hard curve?". `--simband-lo/--simband-hi`
expose it so the calibration stays visible rather than baked in.

Constraints reach NP3 through `BaseSampling.annotate`, identical to every other
sampler.
"""

import numpy as np

from .base import BaseSampling


def band_counts(rerank_dist, lo, hi, chunk=512):
    """(n_pairs_in_band, per_row_counts) over the strict upper triangle."""
    n = rerank_dist.shape[0]
    per_row = np.zeros(n, dtype=np.int64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sim = 1.0 - rerank_dist[s:e]
        # strict upper triangle: keep j > i only
        cols = np.arange(n)[None, :]
        rows = np.arange(s, e)[:, None]
        m = (cols > rows) & (sim >= lo) & (sim <= hi)
        per_row[s:e] = m.sum(axis=1)
    return int(per_row.sum()), per_row


def sample_ordinals(total, k, rng):
    """k distinct ordinals from [0, total) without materialising a permutation.

    np.random.choice(replace=False) permutes all `total` entries, which is hopeless
    for the ~2e8 pairs in this pool. Rejection sampling into a set is O(k) when
    k << total, which it always is here (k = 0.02% of the pool).
    """
    if k >= total:
        return np.arange(total, dtype=np.int64)
    picked = set()
    while len(picked) < k:
        need = k - len(picked)
        picked.update(rng.integers(0, total, size=int(need * 1.2) + 16).tolist())
    arr = np.fromiter(picked, dtype=np.int64, count=len(picked))
    # The over-draw above usually leaves more than k. Trimming with arr[:k] would be
    # WRONG: CPython iterates a set of ints in hash order, which for ints is close to
    # value order, so the slice is biased toward small ordinals -- and ordinals map to
    # rows in order, so that biases the sample toward low row indices. Measured as an
    # 8-sigma gap between the band's true positive rate and the achieved ML fraction.
    # Choose the k uniformly instead.
    if len(arr) > k:
        arr = rng.choice(arr, size=k, replace=False)
    return np.sort(arr)


class SimilarityBandSampling(BaseSampling):
    def __init__(self, lo=None, hi=None, args=None):
        super().__init__(args=args)
        self.lo = lo
        self.hi = hi
        self.stats = {}

    def select(self, features=None, rerank_dist=None, **kwargs):
        assert rerank_dist is not None
        lo = self.lo if self.lo is not None else getattr(self.args, "simband_lo", 0.0)
        hi = self.hi if self.hi is not None else getattr(self.args, "simband_hi", 1.0)
        n = rerank_dist.shape[0]

        total, per_row = band_counts(rerank_dist, lo, hi)
        if total == 0:
            raise RuntimeError(
                f"similarity band [{lo}, {hi}] contains no pairs; widen it"
            )

        # np.random keeps this on the same global stream every other sampler uses.
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
        ordinals = sample_ordinals(total, self.n, rng)

        # Walk rows once, converting global ordinals into (i, j).
        starts = np.concatenate([[0], np.cumsum(per_row)])
        pairs = []
        row_of = np.searchsorted(starts, ordinals, side="right") - 1
        for r in np.unique(row_of):
            local = ordinals[row_of == r] - starts[r]
            sim = 1.0 - rerank_dist[r]
            cand = np.where((np.arange(n) > r) & (sim >= lo) & (sim <= hi))[0]
            pairs.extend((int(r), int(c)) for c in cand[local])

        selected_pairs = set()
        for a, b in pairs:
            fs = frozenset((a, b))
            if fs in self.all_selected_pairs:
                continue
            selected_pairs.add(fs)
        self.all_selected_pairs.update(selected_pairs)
        self.annotate(selected_pairs)

        gt = np.asarray(self.gt_labels)
        arr = np.array([tuple(p) for p in selected_pairs]) if selected_pairs else np.zeros((0, 2), int)
        tp = int((gt[arr[:, 0]] == gt[arr[:, 1]]).sum()) if len(arr) else 0
        self.stats = dict(lo=lo, hi=hi, band_pairs=total, requested=self.n,
                          returned=len(selected_pairs), ml=tp,
                          cl=len(selected_pairs) - tp,
                          ml_fraction=(tp / len(selected_pairs)) if selected_pairs else 0.0)
        print(f"[simband] band=[{lo:.3f},{hi:.3f}] pool={total:,} "
              f"returned={len(selected_pairs):,} ML={tp:,} "
              f"ML_frac={self.stats['ml_fraction']:.4f}")
        return self.ml_pairs, self.cl_pairs
