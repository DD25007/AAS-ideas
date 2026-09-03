#!/usr/bin/env python3
"""Figures for the WildlifeReID-10k reproduction.

Replaces generate_graph.py / generate_graphs.py / generate_plots.py (Todo.md C6). Those
three shared most of their structure, two of them ran their plotting at import time, and
all three read a `logs_*.json` / `mean_accuracy_*.json` schema that only `main.py --sampler`
emits -- a workflow this reproduction never used, so none of them could plot anything in
the current output tree. This module reads the schema the runs actually produce and reuses
the loaders in aggregate_results.py so the figures and the README tables cannot drift.

    python visualization.py --figure all --root outputs_initial --out figures/

Every figure writes a companion .csv with the numbers behind it. That is not decorative:
three of the six series colours sit below 3:1 contrast on the light surface, and the
palette rule is that such a palette ships with a table view.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate_results import AL_ROWS, METRICS, TABLE1_ROWS, macro, summarise, find_runs

# Categorical slots 1-6, assigned to methods in fixed order and never cycled: the colour
# follows the method, so adding or dropping a method never repaints the others.
SERIES = {
    "USL": "#2a78d6",  # blue
    "random": "#eb6834",  # orange
    "GBAS": "#1baf7a",  # aqua
    "LBAS": "#eda100",  # yellow
    "CC": "#e87ba4",  # magenta
    "ConSam": "#008300",  # green
}
LABEL = dict(AL_ROWS)

INK, INK_MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"


def _style(ax):
    """Recessive axes and grid; the data should be the only assertive thing on the page."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(INK_MUTED)


def _save(fig, out, name, rows, header):
    os.makedirs(out, exist_ok=True)
    png, csv = os.path.join(out, name + ".png"), os.path.join(out, name + ".csv")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    with open(csv, "w") as fh:
        fh.write(",".join(header) + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    print(f"  wrote {png}\n        {csv}")


def al_curves(root, metric="mAP"):
    """-> {method: {epoch: mean-over-seeds}}; the AL learning curve per method."""
    key, scale = next((k, s) for lab, k, s in METRICS if lab == metric)
    out = {}
    for method, _ in AL_ROWS:
        per_epoch = {}
        for d in sorted(glob.glob(os.path.join(root, "al", method, "s*"))):
            if not re.fullmatch(r"s\d+", os.path.basename(d)):
                continue
            for f in glob.glob(
                os.path.join(
                    d, "wildlife", "combined", "evaluation", "evaluation_metrics_*.json"
                )
            ):
                ep = int(re.search(r"_(\d+)\.json$", f).group(1))
                with open(f) as fh:
                    vals = json.load(fh).get(key)
                if vals:
                    per_epoch.setdefault(ep, []).append(statistics.mean(vals) * scale)
        if per_epoch:
            out[method] = {e: statistics.mean(v) for e, v in sorted(per_epoch.items())}
    return out


def fig_al_curve(root, out, metric="mAP"):
    """Change-over-budget with six identities -> multi-line, one line per method."""
    data = al_curves(root, metric)
    if not data:
        print("  no AL runs found")
        return
    # 52 is the final best-checkpoint eval, not a cycle; keep the 5 AL cycles on the axis.
    cycles = [10, 20, 30, 40, 50]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    _style(ax)
    rows = []
    for method, _ in AL_ROWS:
        if method not in data:
            continue
        xs = [c for c in cycles if c in data[method]]
        ys = [data[method][c] for c in xs]
        ax.plot(
            xs,
            ys,
            color=SERIES[method],
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=LABEL[method],
        )
        rows += [(LABEL[method], x, f"{y:.2f}") for x, y in zip(xs, ys)]
    ax.set_xticks(cycles)
    ax.set_xlabel("epoch  (one AL cycle every 10)", color=INK_MUTED, fontsize=10)
    ax.set_ylabel(
        f"{metric}  (macro over 13 datasets, %)", color=INK_MUTED, fontsize=10
    )
    ax.set_title(
        f"Active-learning {metric} by cycle — mean of 4 seeds",
        color=INK,
        fontsize=12,
        loc="left",
        pad=12,
    )
    leg = ax.legend(
        frameon=False,
        fontsize=9,
        loc="upper left",
        ncol=2,
        handlelength=1.6,
        columnspacing=1.4,
        borderaxespad=0.2,
    )
    for t in leg.get_texts():
        t.set_color(INK)
    _save(
        fig,
        out,
        f"al_curve_{metric.replace(' ', '_').lower()}",
        rows,
        ["method", "epoch", metric],
    )


def fig_method_bars(root, out, metric="mAP"):
    """Final-cycle magnitude ranking -> horizontal bars, sorted, value-labelled."""
    summary = summarise(find_runs(root))
    rows = []
    for method, label in AL_ROWS:
        agg = summary.get(("al", method), {}).get(metric)
        if agg:
            rows.append((method, label, agg[0], agg[1]))
    if not rows:
        print("  no AL runs found")
        return
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.5))
    _style(ax)
    ax.grid(axis="y", visible=False)
    ys = range(len(rows))
    ax.barh(
        list(ys),
        [r[2] for r in rows],
        height=0.46,
        color=[SERIES[r[0]] for r in rows],
        xerr=[r[3] for r in rows],
        error_kw=dict(ecolor=INK_MUTED, lw=1, capsize=3),
    )
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[1] for r in rows])
    for y, r in zip(ys, rows):
        ax.text(
            r[2] + max(r[3], 0) + 0.6,
            y,
            f"{r[2]:.2f}",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(0, max(r[2] for r in rows) * 1.18)
    ax.set_xlabel(
        f"{metric}  (macro over 13 datasets, %)", color=INK_MUTED, fontsize=10
    )
    ax.set_title(
        f"Final {metric} by sampler — mean of 4 seeds, ± std",
        color=INK,
        fontsize=12,
        loc="left",
        pad=12,
    )
    _save(
        fig,
        out,
        f"al_final_{metric.replace(' ', '_').lower()}",
        [(r[1], f"{r[2]:.2f}", f"{r[3]:.2f}") for r in rows],
        ["method", metric, "std"],
    )


def fig_table1(root, out, metric="mAP"):
    """The non-AL rows: baseline / foundation / skyline, as a magnitude ranking."""
    summary = summarise(find_runs(root))
    rows = []
    for row, label in TABLE1_ROWS:
        agg = summary.get(("table1", row), {}).get(metric)
        if agg:
            rows.append((label, agg[0], agg[1]))
    if not rows:
        print("  no table1 runs found")
        return
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.5))
    _style(ax)
    ax.grid(axis="y", visible=False)
    ys = range(len(rows))
    # One measure, one entity kind -> a single hue carries it; the labels give identity.
    ax.barh(
        list(ys),
        [r[1] for r in rows],
        height=0.46,
        color="#2a78d6",
        xerr=[r[2] for r in rows],
        error_kw=dict(ecolor=INK_MUTED, lw=1, capsize=3),
    )
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    for y, r in zip(ys, rows):
        ax.text(
            r[1] + max(r[2], 0) + 0.6,
            y,
            f"{r[1]:.2f}",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(0, max(r[1] for r in rows) * 1.18)
    ax.set_xlabel(
        f"{metric}  (macro over 13 datasets, %)", color=INK_MUTED, fontsize=10
    )
    ax.set_title(
        f"Baseline / foundation / skyline {metric} — mean of 4 gallery splits, ± std",
        color=INK,
        fontsize=12,
        loc="left",
        pad=12,
    )
    _save(
        fig,
        out,
        f"table1_{metric.replace(' ', '_').lower()}",
        [(r[0], f"{r[1]:.2f}", f"{r[2]:.2f}") for r in rows],
        ["row", metric, "std"],
    )


FIGURES = {"curve": fig_al_curve, "bars": fig_method_bars, "table1": fig_table1}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--figure", default="all", choices=list(FIGURES) + ["all"])
    p.add_argument("--root", default="outputs_initial")
    p.add_argument("--out", default="figures")
    p.add_argument("--metric", default="mAP", choices=[m[0] for m in METRICS])
    a = p.parse_args()
    for name in (FIGURES if a.figure == "all" else [a.figure]):
        print(f"{name}:")
        FIGURES[name](a.root, a.out, a.metric)


if __name__ == "__main__":
    main()
