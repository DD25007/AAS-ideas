#!/usr/bin/env python3
"""Regenerates headroom/e1/REPORT.md from whatever exists (stdlib + numpy). `python headroom/e1/make_report.py`"""
import datetime as dt, glob, hashlib, json, os, re, subprocess, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.realpath(os.path.join(HERE, "..", ".."))
RUN = os.path.join(REPO, "headroom", "runs", "e1_oracle")
C0 = os.path.join(REPO, "outputs", "P2b_full_gate"); SKY = os.path.join(REPO, "headroom", "runs", "skyline")
EPOCHS = 50
DATASETS = ["AmvrakikosTurtles", "CatIndividualImages", "Chicks4FreeID", "CowDataset", "DogFaceNet", "MPDD", "MultiCamCows2024", "PolarBearVidID",
            "PrimFace", "ReunionTurtles", "SeaStarReID2023", "SouthernProvinceTurtles", "ZakynthosTurtles"]
HI = ["CatIndividualImages", "MultiCamCows2024", "PrimFace", "PolarBearVidID"]
git = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()
sh = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rd = lambda p, d="": open(p, errors="replace").read() if os.path.exists(p) else d
def ev(run, tag):
    base = next((os.path.join(run, s) for s in ("logs/wildlife/combined/evaluation", "wildlife/combined/evaluation") if os.path.isdir(os.path.join(run, s))), None)
    if not base: return None, None
    f, g = os.path.join(base, f"evaluation_metrics_{tag}.json"), os.path.join(base, f"micro_metrics_{tag}.json")
    try:
        m = json.load(open(f))["mAP"]; mi = json.load(open(g))["mAP"] if os.path.exists(g) else None
        return (m, mi) if isinstance(m, list) and len(m) == 13 else (None, None)
    except (OSError, json.JSONDecodeError): return None, None
mac = lambda m: 100 * sum(m) / len(m)
f2 = lambda x: "—" if x is None else f"{x:.2f}"
def alive():
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    rx = re.compile(r"headroom/runs/e1_oracle(/| |$)")
    return any(rx.search(l) and any(k in l for k in ("run_resumable", "train_forkserver", "train_al", "resume_arm")) for l in out.splitlines() if "make_report" not in l)
L = []; w = L.append
w("# E1 report: oracle-merge upper bound\n")
w(f"- Generated {dt.datetime.now().astimezone().isoformat(timespec='seconds')} by `headroom/e1/make_report.py` (regenerates from whatever exists).")
w(f"- Repo HEAD now `{git('rev-parse', 'HEAD')}` on branch `{git('branch', '--show-current')}`; tracked-tree dirty: **{str(bool(git('status', '--porcelain', '--untracked-files=no'))).lower()}**. Commits of this experiment: "
  + "; ".join(git("log", "--oneline", "-3").splitlines()))
pre = rd(os.path.join(HERE, "prereg.sha256.txt")).split()
w(f"- prereg `headroom/e1/prereg.md` sha256 `{sh(os.path.join(HERE, 'prereg.md'))}` (recorded `{pre[0] if pre else '?'}`, {'**VERIFIED**' if pre and sh(os.path.join(HERE, 'prereg.md')) == pre[0] else '**MISMATCH**'}; written {rd(os.path.join(HERE, 'prereg.timestamp.txt')).strip()}, before any code change).")
w(f"- Run dir `headroom/runs/e1_oracle`: git_head at launch `{rd(os.path.join(RUN, 'git_head.txt')).strip() or '—'}`, git_dirty `{rd(os.path.join(RUN, 'git_dirty.txt')).strip() or '—'}`; "
  f"command: `{rd(os.path.join(RUN, 'cmd.txt')).strip()[:600]}`\n")

# ---- A0 and status
m_c0, mi_c0 = ev(C0, EPOCHS); A0 = mac(m_c0) if m_c0 else None
m_or, mi_or = ev(RUN, EPOCHS); m_sk, mi_sk = ev(SKY, EPOCHS)
w("## 1. A0 and status\n")
w(f"- **A0 = {A0:.4f}** = macro mAP, last epoch (`evaluation_metrics_50.json`), of `{C0}` (C0 train-seed 0, `--canonical-pairs --deterministic --split-seed 0 --fast-rerank`; bit-identical to the original C0 s0 per CLAUDE.md). Its best-checkpoint file gives the same value, so no selection effect.")
wl = rd(os.path.join(RUN, "wrapper.log")); wm = rd(os.path.join(RUN, "wrapper_manual.log")); wat = rd(os.path.join(HERE, "watcher.log"))
events = [l for l in wat.splitlines() if "EVENT" in l or "WARNING" in l]
t0 = re.search(r"\[wrapper\] (\S+)", wl)
t0 = dt.datetime.fromisoformat(t0.group(1)) if t0 else None
fs = glob.glob(os.path.join(RUN, "attempt_*.log")); tend = dt.datetime.fromtimestamp(max(os.path.getmtime(x) for x in fs)).astimezone() if fs else None
running = alive(); finished = m_or is not None and "Total running time" in "".join(rd(f) for f in fs)
stopped = os.path.exists(os.path.join(RUN, "STOPPED"))
wall = ((dt.datetime.now().astimezone() if running else tend) - t0).total_seconds() / 3600 if t0 and (running or tend) else None
status = "finished" if finished else ("STOPPED (GPU-hour cap)" if stopped else ("running" if running else "NOT RUNNING and unfinished"))
nres = sum(1 for l in events if "EVENT resume=" in l)
_g = re.search(r"gpu=(\d+)", rd(os.path.join(RUN, "cmd.txt")))
w(f"- Run status: **{status}**; resumes by the watcher: {nres}; GPU: {_g.group(1) if _g else '?'}; "
  f"start {t0.isoformat(timespec='seconds') if t0 else '—'}; last log write {tend.isoformat(timespec='seconds') if tend else '—'}; wall {f2(wall)} h; **GPU-h used (1 card) {f2(wall)}** of the 40 cap.")
for l in events: w(f"  - `{l[:240]}`")
sp = rd(os.path.join(HERE, "speed_probe.json"))
if sp:
    j = json.loads(sp); w(f"- Cost guard (step 3, before launch): 50 real training iterations on {j['device']} took {j['warmup_10_iters_s'] + j['timed_40_iters_s']:.1f}s "
                          f"({j['s_per_iter_timed']:.2f} s/iter timed, {j['s_per_iter_all50']:.2f} s/iter incl. warm-up; load average {j['loadavg'][0]:.0f}); projection in `headroom/e1/cost_guard.md`.")
w("")

# ---- 2 per-dataset
w("## 2. Per-dataset last-epoch mAP (x100)\n")
w("| dataset | oracle-merge arm | C0 s0 (A0) | skyline50 (GT labels, H1) | oracle - C0 |\n|---|---|---|---|---|")
for k, d in enumerate(DATASETS):
    a = 100 * m_or[k] if m_or else None; b = 100 * m_c0[k]; c = 100 * m_sk[k] if m_sk else None
    w(f"| {'**' + d + '**' if d in HI else d} | {f2(a)} | {b:.2f} | {f2(c)} | {f2(a - b) if a is not None else '—'} |")
w(f"| **macro (mean of 13)** | **{f2(mac(m_or) if m_or else None)}** | **{A0:.4f}** | **{f2(mac(m_sk) if m_sk else None)}** | **{f2(mac(m_or) - A0) if m_or else '—'}** |\n")

# ---- 3 curve
w("## 3. Macro mAP (micro in brackets) at every eval epoch\n")
w("| after epoch (eval tag) | oracle-merge arm | C0 s0 | skyline50 |\n|---|---|---|---|")
for t in (10, 20, 30, 40, 50):
    cells = []
    for run in (RUN, C0, SKY):
        m, mi = ev(run, t); cells.append(f"{mac(m):.2f} ({100 * mi:.2f})" if m and mi is not None else (f"{mac(m):.2f}" if m else "—"))
    w(f"| {t - 1} (tag {t}) | " + " | ".join(cells) + " |")
w("")

# ---- 4 oracle merges
w("## 4. Oracle merges per re-clustering\n")
rows = {}
for l in rd(os.path.join(RUN, "logs", "oracle_merge.jsonl")).splitlines():
    try: r = json.loads(l); rows[r["epoch"]] = r                    # the last record per epoch (an epoch redone after a resume overwrites)
    except (json.JSONDecodeError, KeyError): pass
if rows:
    ep = sorted(rows); tot = sum(rows[e]["merge_operations"] for e in ep)
    w(f"`merge operations` = units absorbed = units_before - units_after, i.e. the number of unit-into-unit merges done in that re-clustering (images relabelled and tied units alongside). "
      f"{len(ep)} of {EPOCHS} re-clusterings logged; **total merge operations {tot:,}** (mean {tot / len(ep):.0f} per re-clustering).\n")
    w("**Per dataset, summed over all logged re-clusterings (mean per re-clustering in brackets)**\n")
    w("| dataset | total merge operations | mean per re-clustering |\n|---|---|---|")
    for d in DATASETS:
        s = sum(rows[e]["merge_operations_by_dataset"].get(d, 0) for e in ep)
        w(f"| {'**' + d + '**' if d in HI else d} | {s:,} | {s / len(ep):.1f} |")
    w("\n**Per re-clustering (epoch: units before -> after, merge operations, images relabelled, tied units unchanged)**\n")
    w("| epoch | units before | units after | merge operations | images relabelled | tied units | " + " | ".join(d[:6] for d in DATASETS) + " |\n|" + "---|" * (6 + len(DATASETS)))
    for e in ep:
        r = rows[e]; w(f"| {e} | {r['units_before']} | {r['units_after']} | {r['merge_operations']} | {r['images_relabelled']} | {r['tied_units_unchanged']} | "
                       + " | ".join(str(r["merge_operations_by_dataset"].get(d, 0)) for d in DATASETS) + " |")
else:
    w("No `oracle_merge.jsonl` yet.")
w("")

# ---- 5 anomalies
w("## 5. Anomalies and checks\n")
txt = "\n".join(rd(f) for f in sorted(glob.glob(os.path.join(RUN, "attempt_*.log"))))
args_line = re.search(r"Args:Namespace\((.*?)\)\n", txt, re.S); a = args_line.group(1) if args_line else ""
ok = all(x in a for x in ("oracle_merge=True", "canonical_pairs=True", "deterministic=True", "train_seed=0", "split_seed=0", f"epochs={EPOCHS},", "sampler='ConSam'"))
w(f"- Logged args match the planned arm (oracle_merge, canonical_pairs, deterministic, train_seed 0, split_seed 0, 50 epochs, ConSam): **{'yes' if ok else 'NO / not logged yet'}**.")
_nf = len(re.findall(r"Loss\s+(?:-?nan|-?inf)", txt, re.I)); _oom = len(re.findall("out of memory", txt, re.I))
w(f"- Non-finite loss lines: {_nf}; tracebacks: {txt.count('Traceback')}; OOM lines: {_oom}.")
nor = len(re.findall(r"\[oracle-merge\]", txt))
w(f"- `[oracle-merge]` lines in the log: {nor} (expected one per re-clustering, {EPOCHS}; more if epochs were redone after a resume).")
cmp = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "compare_epoch0_record.py"), os.path.join(C0, "attempt_0.log"), os.path.join(RUN, "attempt_0.log")], capture_output=True, text=True) if os.path.exists(os.path.join(RUN, "attempt_0.log")) else None
if cmp is not None:
    w(f"- Epoch-0 `[record]` vs C0 s0 (the AL selection at epoch 0 happens before any oracle merge, so it must be identical; `compare_epoch0_record.py` exit {cmp.returncode}):\n\n```\n{(cmp.stdout + cmp.stderr).strip()[-1500:]}\n```")
nl = [l for l in txt.splitlines() if "Final Cluster Purity" in l]
if nl: w(f"- Last `Final Cluster Purity` line (computed on the oracle-merged labels): `{nl[-1].split(' ', 1)[-1].strip()[:200]}`")
w("- Bit-identity of this arm is not claimed: the oracle path is new, and the baseline arm A0 was produced by an earlier commit (`9bc6c0d`); flag-off equality was shown on a saved state (`headroom/e1/test_oracle_merge.py`).\n")

# ---- 6 verdict
w("## 6. The prereg, verbatim, and the verdict\n")
w("```\n" + rd(os.path.join(HERE, "prereg.md")).strip() + "\n```\n")
if m_or is None:
    w(f"**No verdict: the oracle arm has no last-epoch evaluation (status: {status}).** G is undefined; the rule is not applied to partial results.")
else:
    G = mac(m_or) - A0
    w(f"- oracle arm macro mAP (last epoch) = **{mac(m_or):.4f}**; A0 = **{A0:.4f}**; **G = {G:+.4f}**.")
    if G < 2: v = "G < 2 -> fragment-merging direction dead (even perfect merging cannot pay)."
    elif G >= 4: v = "G >= 4 -> fragmentation is the main lever; next = narrow literature check on efficient partner discovery."
    else: v = "2 <= G < 4 -> partial; report and stop."
    w(f"\n### Verdict: **{v}**\n")
    w("n = 1 screen (deterministic, paired with A0); no further interpretation beyond the rule is applied here.")
open(os.path.join(HERE, "REPORT.md"), "w").write("\n".join(L) + "\n")
print("wrote", os.path.join(HERE, "REPORT.md"))
