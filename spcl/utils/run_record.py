"""Per-run experiment record: everything the AL grid needs that isn't already on disk.

`utils.get_evaluation_results` already writes `evaluation_metrics_{epoch+1}.json`
(mAP, mINP, BAKS, ROC AUC, Top-1/3/5/10) into the run folder. This module adds what
that file does not carry:

  * per-cycle ML / CL counts, ML fraction, pairs consumed, cumulative consumed,
    and the "Budget not completely utilized" shortfall
  * per-cycle cluster counts from each view, region count, and % images covered
  * the source dataset of both images of every queried pair
  * wall-clock and the exact args line

Written as JSON so a crashed run still leaves everything up to its last cycle.
"""

import json
import os
import sys
import time
from collections import Counter


class RunRecord:
    def __init__(self, folder, args, n_train, pair_dataset=None):
        self.path = os.path.join(folder, "run_record.json")
        self.t0 = time.monotonic()
        self.pair_dataset = pair_dataset  # index -> dataset name
        self.n_train = int(n_train)
        self.total_edges = self.n_train * (self.n_train - 1) // 2
        self.d = {
            "argv": " ".join(sys.argv),
            "args": {k: _plain(v) for k, v in vars(args).items()},
            "n_train": self.n_train,
            "total_edges": self.total_edges,
            "cycles": [],
            "status": "running",
        }
        self._prev_consumed = 0
        self.flush()

    def cycle(self, epoch, sampler, ml_pairs, cl_pairs):
        ml, cl = len(ml_pairs), len(cl_pairs)
        consumed = ml + cl
        delta = consumed - self._prev_consumed
        self._prev_consumed = consumed
        g = lambda a: _plain(getattr(sampler, a, None))
        rec = {
            "epoch": int(epoch),
            "cycle": len(self.d["cycles"]) + 1,
            "ml_total": ml,
            "cl_total": cl,
            "consumed_total": consumed,
            "consumed_this_cycle": delta,
            "ml_fraction_total": (ml / consumed) if consumed else None,
            "budget_per_cycle": _plain(getattr(sampler, "n", None)),
            "shortfall": g("rec_shortfall"),
            "n_dbscan_raw": g("rec_n_dbscan_raw"),
            "n_dbscan_aas": g("rec_n_dbscan_aas"),
            "n_finch": g("rec_n_finch"),
            "n_regions": g("rec_n_regions"),
            "images_covered": g("rec_cov"),
            "pct_images_covered": (
                100.0 * g("rec_cov") / self.n_train
                if g("rec_cov") is not None else None
            ),
            "consumed_pct_of_all_pairs": 100.0 * consumed / self.total_edges,
        }
        # Per-dataset allocation of the pairs queried so far. Counting both endpoints
        # is what "budget landed in dataset X" means; a cross-dataset pair contributes
        # to both, so the counts sum to 2 x pairs.
        if self.pair_dataset is not None:
            c = Counter()
            cross = 0
            for fs in getattr(sampler, "all_selected_pairs", ()):
                t = tuple(fs)
                a = self.pair_dataset[t[0]]
                b = self.pair_dataset[t[-1]]
                c[a] += 1
                c[b] += 1
                if a != b:
                    cross += 1
            rec["dataset_endpoints"] = dict(sorted(c.items()))
            rec["cross_dataset_pairs"] = cross
        self.d["cycles"].append(rec)
        self.flush()
        return rec

    def finish(self, status="ok", note=""):
        self.d["status"] = status
        self.d["note"] = note
        self.d["wall_clock_s"] = round(time.monotonic() - self.t0, 1)
        if self.d["cycles"]:
            last = self.d["cycles"][-1]
            self.d["final_consumed"] = last["consumed_total"]
            self.d["final_consumed_pct_of_all_pairs"] = last["consumed_pct_of_all_pairs"]
            self.d["final_ml_fraction"] = last["ml_fraction_total"]
        self.flush()

    def flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.d, fh, indent=1, default=_plain)
        os.replace(tmp, self.path)  # atomic: a killed run never leaves a torn file


def _plain(v):
    """numpy scalars are not JSON-serialisable under NEP 50 (see utils.py:1104)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    for attr in ("item",):
        if hasattr(v, attr):
            try:
                return v.item()
            except Exception:
                pass
    return str(v)


def dataset_of_train_index(dataset):
    """index into sorted(dataset.train) -> source dataset name, via metadata.csv."""
    import os.path as osp
    lookup = dict(
        zip(
            (osp.join(dataset.img_dir, p) for p in dataset.train_split["path"]),
            dataset.train_split["dataset"],
        )
    )
    return [lookup.get(p, "?") for p, _, _ in sorted(dataset.train)]
