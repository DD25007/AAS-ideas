"""Table builders for reports/phaseD1_partner.md: every number is read from $OUT/*.json written by the d1x scripts."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
OUT = "/bhallukah/dd25007/phaseD1/out"
SEEDS = [0, 1, 2, 3]
PRIMARY = ["CatIndividualImages", "MultiCamCows2024", "PrimFace"]
def J(name): return json.load(open(f"{OUT}/{name}.json"))
def ms(x, f="{:.1f}", scale=1.0):
    x = np.array(x, float) * scale
    return (f + " ± " + f).format(x.mean(), x.std(ddof=1)) if len(x) > 1 else f.format(x.mean())
def md(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    return s + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n"
def d13(tag): return J(f"d13_{tag}")
def d16(tag): return J(f"d16_{tag}")
