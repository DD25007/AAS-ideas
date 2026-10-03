#!/usr/bin/env python3
"""Compare two finished train_al.py runs for bit-identity.

    compare_runs.py <ref_run_dir> <new_run_dir> [--no-weights]

A run dir holds `attempt*.log` (timestamped stdout, one per attempt) and `logs/` (--logs-dir). Checked, all exact:

  1 records      every `[record]` line (one per AL cycle), all fields incl. ml_sha256 / cl_sha256
  2 eval history logs/logs_al/*/results.csv (mAP at the eval epochs), as bytes
  3 final metrics logs/wildlife/*/evaluation/{evaluation_metrics,micro_metrics}_*.json: JSON equality, the
                 13 per-dataset values, macro (mean of the 13) and micro mAP
  4 per-epoch    the "Statistics for epoch E" line (cluster counts, R_indep threshold) and every `Epoch: [E][i/n]`
                 loss text; lines repeated by a resumed attempt must agree with each other too
  5 weights      checkpoint_*.pth.tar and model_best.pth.tar (state_dict tensors, torch.equal) and, per
                 resume_epoch*.pth, model / optimizer / lr_scheduler / memory bank / pair sets / best_mAP /
                 indep_thres (RNG states are compared and reported separately, not part of the verdict)

Prints the first divergence (lowest epoch, then field) and exits 0 iff nothing differs.
"""
import csv
import glob
import json
import os
import re
import sys

EVAL_NAME = re.compile(r"_(\d+)\.json$")

# The 19 fields of an AL-cycle [record] line. Runs from newer code may carry extra fields (e.g. the Farthest
# counters and a wall-clock one); those are listed but not part of the verdict.
RECORD_FIELDS = ["epoch", "cycle", "ml_total", "cl_total", "consumed_total", "consumed_this_cycle", "ml_fraction_total",
                 "budget_per_cycle", "shortfall", "n_dbscan_raw", "n_dbscan_aas", "n_finch", "n_regions",
                 "images_covered", "pct_images_covered", "consumed_pct_of_all_pairs", "ml_sha256", "cl_sha256",
                 "cross_dataset_pairs"]


def attempt_logs(d):
    fs = glob.glob(os.path.join(d, "attempt*.log"))
    return sorted(fs, key=lambda p: (os.path.getmtime(p), p))


def lines(d):
    for p in attempt_logs(d):
        for ln in open(p, errors="replace"):
            yield ln.rstrip("\n")


def parse_records(d):
    out = {}
    for ln in lines(d):
        m = re.search(r"\[record\] (\{.*\})\s*$", ln)
        if m:
            r = json.loads(m.group(1))
            out.setdefault(r["epoch"], []).append(r)
    return out


def parse_stats(d):
    out = {}
    for ln in lines(d):
        m = re.search(r"==> Statistics for epoch (\d+): (.*)$", ln)
        if m:
            out.setdefault(int(m.group(1)), []).append(m.group(2).strip())
    return out


LOSS = re.compile(r"Epoch: \[(\d+)\]\[(\d+)/(\d+)\]\s+Time \S+ \([^)]*\)\s+Data \S+ \([^)]*\)\s+(Loss .*?)\s*$")


def parse_loss(d):
    out = {}
    for ln in lines(d):
        m = LOSS.search(ln)
        if m:
            out.setdefault((int(m.group(1)), int(m.group(2))), []).append(m.group(4))
    return out


class Report:
    def __init__(self):
        self.div = []   # (epoch or -1, category, detail)
        self.ok = []

    def check(self, cat, cond, detail="", epoch=-1):
        if cond:
            self.ok.append((cat, detail))
        else:
            self.div.append((epoch, cat, detail))
        print(("PASS " if cond else "FAIL ") + f"[{cat}] {detail}")


def dedupe(name, d, rep, label):
    """{key: value} with repeated values (resumed attempts) required to agree."""
    out = {}
    for k, vs in d.items():
        ok = all(v == vs[0] for v in vs)
        if not ok:
            ep = k if isinstance(k, int) else k[0]
            rep.check(name, False, f"{label}: {k} printed {len(vs)}x with different values", ep)
        out[k] = vs[0]
    return out


def deep_diff(a, b, path=""):
    """First path where two nested structures differ (tensors by torch.equal), else None."""
    import numpy as np
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return path + " (type)"
        if a.dtype != b.dtype or a.shape != b.shape:
            return path + f" (dtype/shape {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)})"
        return None if torch.equal(a, b) else path + " (values)"
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return None if (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype
                        and np.array_equal(a, b)) else path + " (ndarray)"
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return path + f" (keys {sorted(map(str, set(a) ^ set(b)))[:5]})"
        for k in a:
            r = deep_diff(a[k], b[k], f"{path}/{k}")
            if r:
                return r
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return path + f" (len {len(a)} vs {len(b)})"
        for i, (x, y) in enumerate(zip(a, b)):
            r = deep_diff(x, y, f"{path}[{i}]")
            if r:
                return r
        return None
    return None if a == b else path + f" ({str(a)[:40]} vs {str(b)[:40]})"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    weights = "--no-weights" not in sys.argv
    ref, new = args
    rep = Report()
    print(f"ref: {ref}  attempts {len(attempt_logs(ref))}\nnew: {new}  attempts {len(attempt_logs(new))}\n")

    # 1 records
    R, N = dedupe("records", parse_records(ref), rep, "ref"), dedupe("records", parse_records(new), rep, "new")
    rep.check("records", sorted(R) == sorted(N), f"cycle epochs ref {sorted(R)} new {sorted(N)}")
    for e in sorted(set(R) & set(N)):
        bad = [k for k in RECORD_FIELDS if R[e].get(k, "<missing>") != N[e].get(k, "<missing>")]
        extra = sorted((set(R[e]) | set(N[e])) - set(RECORD_FIELDS))
        rep.check("records", not bad, f"cycle at epoch {e}: {len(RECORD_FIELDS)} fields, differing: {bad}"
                  + (f" (extra fields not compared: {extra})" if extra else ""), e)

    # 2 eval history
    def csvs(d):
        return {os.path.relpath(p, d): open(p, "rb").read() for p in glob.glob(os.path.join(d, "logs/logs_al/*/results.csv"))}
    cr, cn = csvs(ref), csvs(new)
    rep.check("eval-history", cr == cn and len(cr) > 0, "results.csv bytes equal: " + "; ".join(
        f"{k}: " + ", ".join(f"{r[1]}:{float(r[2]):.6f}" for r in csv.reader(v.decode().splitlines())) for k, v in cn.items()))

    # 3 final metrics
    def evals(d):
        out = {}
        for p in glob.glob(os.path.join(d, "logs/wildlife/*/evaluation/*.json")):
            out[os.path.basename(p)] = (open(p, "rb").read(), json.load(open(p)))
        return out
    er, en = evals(ref), evals(new)
    rep.check("final-metrics", sorted(er) == sorted(en) and len(er) > 0, f"{len(en)} metric files, same names")
    for name in sorted(set(er) & set(en)):
        m = EVAL_NAME.search(name)
        ep = int(m.group(1)) - 1 if m else -1
        rep.check("final-metrics", er[name][1] == en[name][1] and er[name][0] == en[name][0], name, ep)
    for tag in ("50", "52"):
        f = f"evaluation_metrics_{tag}.json"
        g = f"micro_metrics_{tag}.json"
        if f in en and g in en:
            per = en[f][1]["mAP"]
            mac = sum(per) / len(per)
            print(f"      file *_{tag}: macro mAP {mac:.6f} (n={len(per)})  micro mAP {en[g][1]['mAP']:.6f}")
            print("      per-dataset mAP: " + " ".join(f"{x:.4f}" for x in per))
            if f in er:
                rper = er[f][1]["mAP"]
                print(f"      ref macro {sum(rper)/len(rper):.6f}  ref micro {er[g][1]['mAP']:.6f}")

    # 4 per-epoch statistics and losses
    SR, SN = dedupe("per-epoch", parse_stats(ref), rep, "ref stats"), dedupe("per-epoch", parse_stats(new), rep, "new stats")
    rep.check("per-epoch", sorted(SR) == sorted(SN), f"cluster-statistics lines for {len(SN)} epochs (ref {len(SR)})")
    for e in sorted(set(SR) & set(SN)):
        if SR[e] != SN[e]:
            rep.check("per-epoch", False, f"epoch {e} statistics: {SR[e]} vs {SN[e]}", e)
            break
    else:
        rep.check("per-epoch", True, "all cluster-statistics lines identical")
    LR, LN = dedupe("per-epoch", parse_loss(ref), rep, "ref loss"), dedupe("per-epoch", parse_loss(new), rep, "new loss")
    rep.check("per-epoch", sorted(LR) == sorted(LN), f"loss lines {len(LN)} (ref {len(LR)})")
    for k in sorted(set(LR) & set(LN)):
        if LR[k] != LN[k]:
            rep.check("per-epoch", False, f"loss at epoch {k[0]} iter {k[1]}: {LR[k]} vs {LN[k]}", k[0])
            break
    else:
        rep.check("per-epoch", True, "all printed loss values identical")

    # 5 weights
    if weights:
        import torch
        for pat in ("checkpoint_*.pth.tar", "model_best.pth.tar"):
            for p in sorted(glob.glob(os.path.join(new, "logs", pat))):
                q = os.path.join(ref, "logs", os.path.basename(p))
                m = re.search(r"checkpoint_(\d+)", p)
                ep = int(m.group(1)) - 1 if m else -1
                if not os.path.exists(q):
                    rep.check("weights", False, f"{os.path.basename(p)} missing in ref", ep)
                    continue
                a = torch.load(p, map_location="cpu", weights_only=False)
                b = torch.load(q, map_location="cpu", weights_only=False)
                n_t = len(a["state_dict"])
                d = deep_diff(a["state_dict"], b["state_dict"], "state_dict")
                d2 = deep_diff({k: v for k, v in a.items() if k != "state_dict"}, {k: v for k, v in b.items() if k != "state_dict"})
                rep.check("weights", d is None and d2 is None,
                          f"{os.path.basename(p)}: {n_t} tensors" + ("" if d is None and d2 is None else f", first difference {d or d2}"), ep)
                del a, b
        for p in sorted(glob.glob(os.path.join(new, "logs", "resume_epoch*.pth"))):
            q = os.path.join(ref, "logs", os.path.basename(p))
            ep = int(re.search(r"resume_epoch(\d+)", p).group(1))
            if not os.path.exists(q):
                rep.check("weights", False, f"{os.path.basename(p)} missing in ref", ep)
                continue
            a = torch.load(p, map_location="cpu", weights_only=False)
            b = torch.load(q, map_location="cpu", weights_only=False)
            keys = ["model", "optimizer", "lr_scheduler", "memory_features", "memory_labels", "best_mAP", "indep_thres",
                    "ml_pairs", "cl_pairs", "sampler_ml", "sampler_cl", "epoch"]
            d = deep_diff({k: a[k] for k in keys}, {k: b[k] for k in keys})
            rng = deep_diff({k: a[k] for k in a if k.startswith("rng_")}, {k: b[k] for k in b if k.startswith("rng_")})
            rep.check("weights", d is None, f"{os.path.basename(p)}: model+optimizer+scheduler+memory bank+pairs"
                      + ("" if d is None else f", first difference {d}"), ep)
            print(f"      {os.path.basename(p)}: RNG states {'identical' if rng is None else 'DIFFER at ' + rng} (informational)")
            del a, b

    print()
    if rep.div:
        rep.div.sort(key=lambda x: (x[0] if x[0] >= 0 else 10**9, x[1]))
        e, c, d = rep.div[0]
        print(f"RESULT: DIFFERENT. {len(rep.div)} failing checks; FIRST DIVERGENCE: epoch {e if e >= 0 else 'n/a'} [{c}] {d}")
        sys.exit(1)
    print(f"RESULT: IDENTICAL. {len(rep.ok)} checks passed, 0 failing.")


if __name__ == "__main__":
    main()
