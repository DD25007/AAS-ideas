#!/usr/bin/env python3
"""Rebuild the README's Table 1 numbers from a run-output tree, and diff two trees.

The published tables were assembled by hand, which makes "the metrics still match" an
unfalsifiable claim. This turns it into one command:

    python aggregate_results.py outputs_initial
    python aggregate_results.py --compare outputs_initial outputs_verify --tol 0.0

Expects the layout described in outputs_initial/README.md:

    <root>/al/<sampler>/s<seed>/wildlife/combined/evaluation/evaluation_metrics_52.json
    <root>/table1/<row>/s<seed>/pretrained_models/**/evaluation_metrics_0.json

Each JSON holds one 13-element list per metric, one entry per dataset. The macro number is
the mean over those 13, then over seeds -- matching the paper's protocol. The per-epoch mAP
in the logs is the *micro* (pooled) score and is deliberately not used here.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys

# Display order, and how each maps into the JSON. Top-N are already percentages; the rest
# are stored as fractions and scale by 100.
METRICS = [
    ("mAP", "mAP", 100.0),
    ("mINP", "mINP", 100.0),
    ("BAKS", "BAKS", 100.0),
    ("AUC ROC", "ROC AUC", 100.0),
    ("Top-1", "Top-1 Accuracy", 1.0),
    ("Top-3", "Top-3 Accuracy", 1.0),
    ("Top-5", "Top-5 Accuracy", 1.0),
    ("Top-10", "Top-10 Accuracy", 1.0),
]

# sampler directory -> the row label used in README.md
AL_ROWS = [
    ("USL", "SpCL (Base)"),
    ("random", "+ Random"),
    ("GBAS", "+ GBAS"),
    ("LBAS", "+ LBAS"),
    ("CC", "+ NIS"),
    ("ConSam", "+ AAS"),
]

# Published Table 1 of the AAAI-26 paper (Sani, Khurana, Anand -- "Active Learning for
# Animal Re-Identification with Ambiguity-Aware Sampling", p.39188). All 13-dataset macro
# averages, in percent. The AL block is the paper's mean over 4 runs.
PAPER = {
    ("al", "USL"): {
        "mAP": 44.95,
        "mINP": 27.37,
        "BAKS": 55.38,
        "AUC ROC": 67.01,
        "Top-1": 56.65,
        "Top-3": 71.04,
        "Top-5": 76.35,
        "Top-10": 84.32,
    },
    ("al", "random"): {
        "mAP": 46.67,
        "mINP": 29.02,
        "BAKS": 57.28,
        "AUC ROC": 68.33,
        "Top-1": 58.11,
        "Top-3": 71.96,
        "Top-5": 78.35,
        "Top-10": 85.03,
    },
    ("al", "GBAS"): {
        "mAP": 48.61,
        "mINP": 30.35,
        "BAKS": 60.53,
        "AUC ROC": 71.50,
        "Top-1": 60.56,
        "Top-3": 73.89,
        "Top-5": 80.06,
        "Top-10": 87.41,
    },
    ("al", "LBAS"): {
        "mAP": 52.15,
        "mINP": 34.06,
        "BAKS": 63.62,
        "AUC ROC": 73.15,
        "Top-1": 63.99,
        "Top-3": 76.64,
        "Top-5": 82.44,
        "Top-10": 88.18,
    },
    ("al", "CC"): {
        "mAP": 50.72,
        "mINP": 32.73,
        "BAKS": 61.21,
        "AUC ROC": 72.30,
        "Top-1": 61.78,
        "Top-3": 75.58,
        "Top-5": 80.44,
        "Top-10": 87.41,
    },
    ("al", "ConSam"): {
        "mAP": 56.14,
        "mINP": 38.17,
        "BAKS": 67.15,
        "AUC ROC": 75.21,
        "Top-1": 67.71,
        "Top-3": 79.08,
        "Top-5": 85.04,
        "Top-10": 91.18,
    },
    ("table1", "resnet50_baseline"): {
        "mAP": 38.88,
        "mINP": 21.34,
        "BAKS": 48.62,
        "AUC ROC": 60.87,
        "Top-1": 48.37,
        "Top-3": 65.21,
        "Top-5": 71.01,
        "Top-10": 81.05,
    },
    # the paper's "MegaDescriptor" foundation row -- reproduced as ImageNet Swin-L (item 15)
    ("table1", "swin_foundation"): {
        "mAP": 45.65,
        "mINP": 26.57,
        "BAKS": 55.57,
        "AUC ROC": 64.12,
        "Top-1": 56.46,
        "Top-3": 70.83,
        "Top-5": 78.14,
        "Top-10": 85.48,
    },
    # the paper reports a single skyline; our four variants all target it
    ("table1", "resnet50_skyline_e30_zerobased"): {
        "mAP": 64.29,
        "mINP": 49.75,
        "BAKS": 72.04,
        "AUC ROC": 80.02,
        "Top-1": 72.88,
        "Top-3": 81.06,
        "Top-5": 85.38,
        "Top-10": 91.27,
    },
}
for _k in (
    "resnet50_skyline_e30_offset",
    "resnet50_skyline_e50_zerobased",
    "resnet50_skyline_e50_offset",
):
    PAPER[("table1", _k)] = PAPER[("table1", "resnet50_skyline_e30_zerobased")]


TABLE1_ROWS = [
    ("resnet50_baseline", "ResNet-50 (Baseline)"),
    ("swin_foundation", "Foundation row (ImageNet Swin-L)"),
    ("resnet50_skyline_e30_zerobased", "Skyline, 30 ep, zerobased"),
    ("resnet50_skyline_e30_offset", "Skyline, 30 ep, offset"),
    ("resnet50_skyline_e50_zerobased", "Skyline, 50 ep, zerobased"),
    ("resnet50_skyline_e50_offset", "Skyline, 50 ep, offset"),
    ("megadescriptor", "MegaDescriptor row, authors' path (= ImageNet Swin-L)"),
    (
        "megadescriptor_realweights",
        "MegaDescriptor-L-224, real weights (side experiment)",
    ),
]


def macro(path):
    """Macro-average one run's evaluation JSON: mean over the 13 per-dataset values."""
    with open(path) as fh:
        raw = json.load(fh)
    out = {}
    for label, key, scale in METRICS:
        vals = raw.get(key)
        if not vals:
            continue
        out[label] = statistics.mean(vals) * scale
    return out




PARTIAL = set()


def find_runs(root):
    """-> {('al'|'table1', row_dir): {seed: metrics}}"""
    runs = {}

    for sampler, _ in AL_ROWS:
        for d in sorted(glob.glob(os.path.join(root, "al", sampler, "s*"))):
            m = re.fullmatch(r"s(\d+)", os.path.basename(d))
            if not m:
                continue  # e.g. an archived s3.failed131739 attempt
            seed = int(m.group(1))
            f = os.path.join(
                d, "wildlife", "combined", "evaluation", "evaluation_metrics_52.json"
            )
            if not os.path.exists(f):
                # a shortened smoke run stops at an earlier cycle
                cands = sorted(
                    glob.glob(
                        os.path.join(
                            d,
                            "wildlife",
                            "combined",
                            "evaluation",
                            "evaluation_metrics_*.json",
                        )
                    ),
                    key=lambda p: int(re.search(r"_(\d+)\.json$", p).group(1)),
                )
                if not cands:
                    continue
                f = cands[-1]
                # No final eval: still training, or a shortened smoke run. Falling back to
                # the newest cycle silently mixes a mid-training number into a row mean
                # that reads as final -- warn loudly and tag the row.
                ep = int(re.search(r"_(\d+)\.json$", f).group(1))
                print(
                    f"warning: al/{sampler}/s{seed} has no final eval (52); using epoch "
                    f"{ep} -- row is INCOMPLETE, not comparable to a finished run",
                    file=sys.stderr,
                )
                PARTIAL.add((sampler, seed, ep))
            m = macro(f)
            # Track the cycle this came from: a shortened smoke run stops early, and
            # comparing its epoch-10 eval against a full run's epoch-52 eval is
            # meaningless (it read as an 11-point "regression" on 2026-09-01).
            m["_epoch"] = int(re.search(r"_(\d+)\.json$", f).group(1))
            runs.setdefault(("al", sampler), {})[seed] = m

    for row, _ in TABLE1_ROWS:
        for d in sorted(glob.glob(os.path.join(root, "table1", row, "s*"))):
            m = re.fullmatch(r"s(\d+)", os.path.basename(d))
            if not m:
                continue  # e.g. an archived s3.failed131739 attempt
            seed = int(m.group(1))
            hits = glob.glob(
                os.path.join(d, "**", "evaluation_metrics_*.json"), recursive=True
            )
            if not hits:
                continue
            if len(hits) > 1:
                print(
                    f"warning: {d} has {len(hits)} eval files, using {hits[0]}",
                    file=sys.stderr,
                )
            runs.setdefault(("table1", row), {})[seed] = macro(hits[0])

    return runs


def summarise(runs):
    """-> {(kind, row): {metric: (mean, std, n, {seed: value})}}"""
    out = {}
    for key, per_seed in runs.items():
        agg = {}
        for label, _, _ in METRICS:
            vals = [
                per_seed[s][label] for s in sorted(per_seed) if label in per_seed[s]
            ]
            if not vals:
                continue
            agg[label] = (
                statistics.mean(vals),
                statistics.stdev(vals) if len(vals) > 1 else 0.0,
                len(vals),
                {
                    s: per_seed[s][label]
                    for s in sorted(per_seed)
                    if label in per_seed[s]
                },
            )
        out[key] = agg
    return out


def table(summary, kind, rows, title, show_std):
    present = [(d, label) for d, label in rows if (kind, d) in summary]
    if not present:
        return ""
    cols = [m[0] for m in METRICS]
    lines = [f"### {title}", ""]
    lines.append("| Method | n | " + " | ".join(cols) + " |")
    lines.append("|---|---|" + "---|" * len(cols))
    for d, label in present:
        agg = summary[(kind, d)]
        n = max((v[2] for v in agg.values()), default=0)
        incomplete = sorted(s for (m, s, _) in PARTIAL if m == d)
        if incomplete:
            label = f"{label}  [INCOMPLETE: seeds {incomplete} mid-training]"
        cells = []
        for c in cols:
            if c not in agg:
                cells.append("--")
                continue
            mean, std, _, _ = agg[c]
            cells.append(
                f"{mean:.2f} ±{std:.2f}" if show_std and std else f"{mean:.2f}"
            )
        lines.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def per_seed_table(summary, kind, rows, title, metric="mAP"):
    present = [(d, label) for d, label in rows if (kind, d) in summary]
    if not present:
        return ""
    seeds = sorted(
        {
            s
            for d, _ in present
            for s in summary[(kind, d)].get(metric, (0, 0, 0, {}))[3]
        }
    )
    lines = [f"### {title} — per-seed {metric}", ""]
    lines.append("| Method | " + " | ".join(f"s{s}" for s in seeds) + " | std |")
    lines.append("|---|" + "---|" * (len(seeds) + 1))
    for d, label in present:
        agg = summary[(kind, d)].get(metric)
        if not agg:
            continue
        _, std, _, by_seed = agg
        cells = [f"{by_seed[s]:.2f}" if s in by_seed else "--" for s in seeds]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {std:.2f} |")
    return "\n".join(lines) + "\n"


def render(root):
    summary = summarise(find_runs(root))
    parts = [f"## {root}\n"]
    parts.append(table(summary, "al", AL_ROWS, "Active Learning block", show_std=False))
    parts.append(per_seed_table(summary, "al", AL_ROWS, "Active Learning block"))
    parts.append(
        table(
            summary,
            "table1",
            TABLE1_ROWS,
            "Baseline / foundation / skyline",
            show_std=True,
        )
    )
    return summary, "\n".join(p for p in parts if p)


def compare(a_root, b_root, tol):
    """Diff two trees seed-for-seed.

    Comparing row means straight across is wrong whenever the two trees hold different
    seed sets: a 4-seed baseline against a 1-seed spot check differs by seed spread, not
    by any code change, and that reads as a failure. (It bit exactly that way on
    2026-09-01 — a 1.36 "regression" that was four seeds vs one.) So restrict every row
    to the seeds present on BOTH sides, and say so when they differ.
    """
    a_runs, b_runs = find_runs(a_root), find_runs(b_root)
    shared = sorted(set(a_runs) & set(b_runs))
    only_a = sorted(set(a_runs) - set(b_runs))
    only_b = sorted(set(b_runs) - set(a_runs))

    worst = 0.0
    rows, partial, skipped = [], [], []
    for key in shared:
        seeds = sorted(set(a_runs[key]) & set(b_runs[key]))
        if not seeds:
            partial.append((key, "no seed in common"))
            continue
        if set(a_runs[key]) != set(b_runs[key]):
            partial.append(
                (
                    key,
                    f"comparing on seeds {seeds} "
                    f"(baseline has {sorted(a_runs[key])}, "
                    f"candidate has {sorted(b_runs[key])})",
                )
            )
        mism = [
            (s, a_runs[key][s].get("_epoch"), b_runs[key][s].get("_epoch"))
            for s in seeds
            if a_runs[key][s].get("_epoch") != b_runs[key][s].get("_epoch")
        ]
        if mism:
            for s, ae, be in mism:
                skipped.append(
                    (key, f"seed {s} read at epoch {ae} vs {be} — not comparable")
                )
            seeds = [
                s
                for s in seeds
                if a_runs[key][s].get("_epoch") == b_runs[key][s].get("_epoch")
            ]
            if not seeds:
                continue
        for label, _, _ in METRICS:
            av = [a_runs[key][s][label] for s in seeds if label in a_runs[key][s]]
            bv = [b_runs[key][s][label] for s in seeds if label in b_runs[key][s]]
            if not av or not bv or len(av) != len(bv):
                continue
            am, bm = statistics.mean(av), statistics.mean(bv)
            d = bm - am
            worst = max(worst, abs(d))
            if abs(d) > tol:
                rows.append((f"{key[0]}/{key[1]}", label, am, bm, d))

    print(f"comparing {a_root} -> {b_root}   (tol={tol})")
    print(f"  rows in both: {len(shared)}", end="")
    if only_a:
        print(f" | only in {a_root}: {', '.join('/'.join(k) for k in only_a)}", end="")
    if only_b:
        print(f" | only in {b_root}: {', '.join('/'.join(k) for k in only_b)}", end="")
    print()

    if skipped:
        print("  SKIPPED — different training length, not comparable:")
        for key, msg in skipped:
            print(f"    {key[0]}/{key[1]}: {msg}")

    if partial:
        print("  seed sets differ — restricted to the intersection:")
        for key, msg in partial:
            print(f"    {key[0]}/{key[1]}: {msg}")

    if not shared:
        print("\nNOTHING TO COMPARE — no row appears in both trees.")
        return 2

    if rows:
        print(f"\n{'row':<24} {'metric':<9} {a_root:>12} {b_root:>12} {'delta':>10}")
        print("-" * 72)
        for r, m, av, bv, d in rows:
            print(f"{r:<24} {m:<9} {av:>12.4f} {bv:>12.4f} {d:>+10.4f}")
        print(
            f"\nFAIL: {len(rows)} cell(s) exceed tol={tol}; largest |delta| = {worst:.4f}"
        )
        return 1

    print(
        f"\nPASS: all {len(shared)} shared rows within tol={tol} "
        f"(largest |delta| = {worst:.4f})"
    )
    return 0


def vs_paper(root):
    """Every reproduced row against the paper's published Table 1, all metrics."""
    summary = summarise(find_runs(root))
    order = [("al", m, lab) for m, lab in AL_ROWS] + [
        ("table1", r, lab) for r, lab in TABLE1_ROWS
    ]
    cols = [m[0] for m in METRICS]
    print(f"\n## {root}  vs  published Table 1 (AAAI-26)\n")
    print("| Row | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    deltas = []
    for kind, key, label in order:
        agg, ref = summary.get((kind, key)), PAPER.get((kind, key))
        if not agg or not ref:
            continue
        tag = "  [INCOMPLETE]" if any(m == key for (m, _, _) in PARTIAL) else ""
        cells = []
        for c in cols:
            if c not in agg or c not in ref:
                cells.append("--")
                continue
            d = agg[c][0] - ref[c]
            cells.append(f"{agg[c][0]:.2f} ({d:+.2f})")
            if c == "mAP" and not tag:
                deltas.append(d)
        print(f"| {label}{tag} | " + " | ".join(cells) + " |")
    print("\nEach cell: reproduced (delta vs paper). Negative = below the paper.")
    if deltas:
        print(
            f"mAP delta over complete rows: mean {sum(deltas)/len(deltas):+.2f}, "
            f"range {min(deltas):+.2f} .. {max(deltas):+.2f}"
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "root",
        nargs="?",
        default="outputs_initial",
        help="output tree to summarise (default: outputs_initial)",
    )
    p.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="diff two trees and exit non-zero if any cell moves beyond --tol",
    )
    p.add_argument(
        "--paper",
        action="store_true",
        help="compare against the paper's published Table 1",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=0.0,
        help="allowed absolute delta per cell (default 0.0 = exact)",
    )
    args = p.parse_args()

    if args.paper:
        vs_paper(args.root)
        return

    if args.compare:
        sys.exit(compare(args.compare[0], args.compare[1], args.tol))

    _, text = render(args.root)
    print(text)


if __name__ == "__main__":
    main()
