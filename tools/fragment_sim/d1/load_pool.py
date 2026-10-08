"""D1 step 1 (part): the pool exactly as train_al.py builds it (get_data -> datasets.create, sorted(dataset.train)), CPU only.
Writes $FRAGSIM_WORK/pool.npz (the shipped data/pool.npz has the same arrays, compressed): path, gt pid, dataset (per pool image, row order = memory-bank row order)."""
import os, sys, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/storage/dd25007/cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/storage/dd25007/cache/torch")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import REPO, WORK, DATA
sys.path.insert(0, REPO); os.chdir(REPO)
from spcl import datasets                         # same call as train_al.get_data (train_al.py:62-76)
ds = datasets.create("wildlife", os.path.join(REPO, "dataroot", "wildlife"), subset="combined",
                     dataset_split_seed=0, partition_seed=102)
train = sorted(ds.train)                          # train_al.py:349 / 599: the row order of memory.features
paths = np.array([p for p, _, _ in train]); pid = np.array([q for _, q, _ in train])
dset = np.array([p.split("/wildlifereid-10k/images/")[1].split("/")[0] for p in paths])
assert all(q.startswith(d + "_") for q, d in zip(pid, dset)), "pid prefix != dataset folder"
np.savez(os.path.join(WORK, "pool.npz"), paths=paths, pid=pid, dset=dset)
print("pool", len(train), "ids", len(set(pid)), "datasets", len(set(dset)))
for d in sorted(set(dset)): print(f"  {d:26s} images {np.sum(dset==d):5d}  ids {len(set(pid[dset==d])):4d}")
