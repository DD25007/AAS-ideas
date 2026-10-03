#!/usr/bin/env python3
"""Compare the first [record] line of two run logs field by field (exact equality).

    compare_epoch0_record.py <reference.log> <new.log>

Also prints the epoch-0 cluster-count statistics line of each log (secondary evidence; not part of the verdict).
Exit 0 iff the key sets are equal and every value is identical.
"""
import json
import re
import sys


def first_record(path):
    for line in open(path, errors="replace"):
        m = re.search(r"\[record\] (\{.*\})\s*$", line)
        if m:
            return json.loads(m.group(1))
    raise SystemExit(f"{path}: no [record] line")


def stats_line(path):
    for line in open(path, errors="replace"):
        if "==> Statistics for epoch 0:" in line:
            return line.split("] ", 1)[-1].strip() if line.startswith("[") else line[line.index("==>"):].strip()
    return None


ref, new = first_record(sys.argv[1]), first_record(sys.argv[2])
keys = list(ref) + [k for k in new if k not in ref]
bad = 0
print(f"{'field':32s} {'equal':6s} reference | new")
for k in keys:
    a, b = ref.get(k, "<missing>"), new.get(k, "<missing>")
    eq = (a == b)
    bad += (not eq)
    show = lambda v: (v if not isinstance(v, str) or len(v) < 20 else v[:16] + "...")
    print(f"{k:32s} {'yes' if eq else 'NO':6s} {show(a)} | {show(b)}")
print(f"fields compared: {len(keys)}; differing: {bad}")
sa, sb = stats_line(sys.argv[1]), stats_line(sys.argv[2])
print("epoch-0 statistics, reference:", sa)
print("epoch-0 statistics, new      :", sb, "(same)" if sa == sb else "(DIFFERENT)")
sys.exit(1 if bad else 0)
