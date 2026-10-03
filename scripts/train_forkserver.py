#!/usr/bin/env python3
"""Run train_al.py with DataLoader workers started via 'forkserver' instead of the
default 'fork'. Usage (cwd = the worktree that holds train_al.py):
    python -u scripts/train_forkserver.py <train_al.py args...>

Why: on this node the parent process has ~122 threads (numpy/MKL/torch/CUDA pools)
when it forks the 6 DataLoader workers for "Initialize instance features in the
hybrid memory"; intermittently one forked worker inherits a lock held by another
parent thread and never runs (single thread, 0 CPU-seconds, futex_wait) while the
parent waits on it forever -- seen in Phase B2's I4 run and 4 times in a row on
C2's relaunches. forkserver workers are forked from a clean, single-threaded helper
process (itself exec'd fresh), so that failure mode cannot occur.

Semantics are unchanged: train_al.py is executed verbatim as __main__, the loaders'
seeded generator and worker_init_fn seeding do not depend on the start method, so
per-sample augmentation streams are the same as under fork.
"""
import multiprocessing as mp
import os
import runpy
import sys

def _start_memlog(path, period=20):
    """Passive, opt-in (RUN_MEMLOG=<file>): every `period` s write this process's peak torch.cuda.max_memory_reserved /
    max_memory_allocated (MiB) and peak host RSS (ru_maxrss) to <file> as JSON.  Read-only on torch's allocator stats;
    torch is not imported here (only looked up once train_al has imported it), no RNG is touched."""
    import atexit
    import json
    import resource
    import threading
    import time
    peak = {"torch_max_reserved_mib": 0.0, "torch_max_allocated_mib": 0.0, "rss_main_peak_mib": 0.0}

    def snap():
        t = sys.modules.get("torch")
        try:
            if t is not None and t.cuda.is_initialized():
                peak["torch_max_reserved_mib"] = max(peak["torch_max_reserved_mib"], t.cuda.max_memory_reserved() / 2**20)
                peak["torch_max_allocated_mib"] = max(peak["torch_max_allocated_mib"], t.cuda.max_memory_allocated() / 2**20)
        except Exception:
            pass
        peak["rss_main_peak_mib"] = max(peak["rss_main_peak_mib"], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        peak["t"] = time.time()
        tmp = path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(peak, f)
            os.replace(tmp, path)
        except OSError:
            pass

    def loop():
        while True:
            time.sleep(period)
            snap()

    threading.Thread(target=loop, daemon=True, name="memlog").start()
    atexit.register(snap)


if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    if os.environ.get("RUN_MEMLOG"):
        _start_memlog(os.environ["RUN_MEMLOG"])
    script = "train_al.py"
    sys.path.insert(0, os.getcwd())  # what `python train_al.py` puts on sys.path[0]
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
