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

TABLE1_ROWS = [
    ("resnet50_baseline", "ResNet-50 (Baseline)"),
    ("swin_foundation", "Foundation row (ImageNet Swin-L)"),
    ("resnet50_skyline_e30_zerobased", "Skyline, 30 ep, zerobased"),
    ("resnet50_skyline_e30_offset", "Skyline, 30 ep, offset"),
    ("resnet50_skyline_e50_zerobased", "Skyline, 50 ep, zerobased"),
    ("resnet50_skyline_e50_offset", "Skyline, 50 ep, offset"),
    ("megadescriptor",             "MegaDescriptor row, authors' path (= ImageNet Swin-L)"),
    ("megadescriptor_realweights", "MegaDescriptor-L-224, real weights (side experiment)"),
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


def log_summary(run_dir):
    """The 'mean,...' line main.py writes at the end of a row run, as a cross-check."""
    for log in sorted(glob.glob(os.path.join(run_dir, "*.log"))):
        with open(log, errors="replace") as fh:
            for line in fh:
                if line.startswith("mean,"):
                    return [float(x) for x in line.strip().split(",")[1:] if x]
    return None


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
            runs.setdefault(("al", sampler), {})[seed] = macro(f)

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
    a, _ = render(a_root)
    b, _ = render(b_root)
    shared = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))

    worst = 0.0
    rows = []
    for key in shared:
        for label, _, _ in METRICS:
            if label not in a[key] or label not in b[key]:
                continue
            av, bv = a[key][label][0], b[key][label][0]
            d = bv - av
            worst = max(worst, abs(d))
            if abs(d) > tol:
                rows.append((f"{key[0]}/{key[1]}", label, av, bv, d))

    print(f"comparing {a_root} -> {b_root}   (tol={tol})")
    print(f"  rows in both: {len(shared)}", end="")
    if only_a:
        print(f" | only in {a_root}: {', '.join('/'.join(k) for k in only_a)}", end="")
    if only_b:
        print(f" | only in {b_root}: {', '.join('/'.join(k) for k in only_b)}", end="")
    print()

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
        "--tol",
        type=float,
        default=0.0,
        help="allowed absolute delta per cell (default 0.0 = exact)",
    )
    args = p.parse_args()

    if args.compare:
        sys.exit(compare(args.compare[0], args.compare[1], args.tol))

    _, text = render(args.root)
    print(text)


if __name__ == "__main__":
    main()
