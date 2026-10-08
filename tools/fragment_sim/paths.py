"""Fragment-sim kit paths. Small data (git): ./data, ./ref. Big data (not in git): $FRAGSIM_BIG (resume_epoch95.pth, rerank_e95.npy).
New outputs go to $FRAGSIM_WORK (default $FRAGSIM_BIG/work), never over the shipped data/ref files."""
import os
KIT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.realpath(os.path.join(KIT, "..", ".."))
DATA = os.path.join(KIT, "data"); REF = os.path.join(KIT, "ref")
BIG = os.environ.get("FRAGSIM_BIG", "/bhallukah/dd25007/AAS-ideas-out/kit")
WORK = os.environ.get("FRAGSIM_WORK", os.path.join(BIG, "work"))
os.makedirs(WORK, exist_ok=True)
