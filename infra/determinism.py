"""Determinism and instrumentation for train_al.py, all default-OFF.

  --deterministic      cudnn.deterministic=True, cudnn.benchmark=False,
                       torch.use_deterministic_algorithms(True, warn_only=True),
                       a seeded DataLoader generator + worker_init_fn for the train loader.
                       Requires CUBLAS_WORKSPACE_CONFIG in the environment before Python
                       starts (export CUBLAS_WORKSPACE_CONFIG=:4096:8).
  --stage-log          one line per pipeline stage: wall clock since the previous stage and
                       since start, torch.cuda.max_memory_reserved/allocated.
  --stop-after-epoch N after epoch N's training step, save model state_dict + memory bank
                       to <logs_dir>/stop_epoch{N}.pth and exit.
  --train-seed S       re-seed python random, numpy, torch (CPU + CUDA) and the train-loader
                       generator with S right before the FIRST training step, i.e. after
                       Wildlife.__init__, epoch-0 clustering and cycle-1 selection. The split
                       and the whole epoch-0 path stay identical across train seeds; repeats
                       differ only in training.
  --loss-trace         a forward hook on the memory module records every iteration's loss
                       at full precision -> <logs_dir>/loss_trace_epoch{e}.npy (float64).
                       The trainer only prints 3 decimals every --print-freq iterations.

With every flag off, `loader_kwargs()` returns {} and `stage()` returns immediately, so
train_al.py's default path executes exactly the original calls.
"""

import atexit
import json
import os
import random
import re
import sys
import time
import warnings

import numpy as np
import torch

_STATE = {"deterministic": False, "generator": None, "stage": False, "t0": None, "t_prev": None,
          "warnings": {}, "logs_dir": None, "losses": None, "train_seeded": False}

_DET_WARN = re.compile(r"deterministic", re.IGNORECASE)
CUBLAS_OK = (":4096:8", ":16:8")


def _worker_init_fn(worker_id):
    # DataLoader has already seeded this worker's torch RNG with base_seed + worker_id,
    # where base_seed comes from the loader's generator. Derive numpy/random from it.
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _record_warning(message, category, filename, lineno, file=None, line=None):
    msg = str(message)
    if _DET_WARN.search(msg):
        n = _STATE["warnings"].get(msg, 0)
        _STATE["warnings"][msg] = n + 1
        if n == 0:
            print(f"[det-warn] {category.__name__} {filename}:{lineno}: {msg}", flush=True)
        return
    _STATE["_orig_showwarning"](message, category, filename, lineno, file, line)


def _dump_warnings():
    if _STATE["logs_dir"] is None:
        return
    path = os.path.join(_STATE["logs_dir"], "det_warnings.json")
    with open(path, "w") as fh:
        json.dump({"n_unique": len(_STATE["warnings"]), "counts": _STATE["warnings"]}, fh, indent=1)


def enable(args):
    """Apply --deterministic / --stage-log. Call once, early in main_worker."""
    _STATE["logs_dir"] = args.logs_dir
    if getattr(args, "stage_log", False):
        _STATE["stage"] = True
        _STATE["t0"] = _STATE["t_prev"] = time.monotonic()
    if not getattr(args, "deterministic", False):
        return
    cfg = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if cfg not in CUBLAS_OK:
        raise RuntimeError(
            f"--deterministic needs CUBLAS_WORKSPACE_CONFIG in {CUBLAS_OK} before Python starts; "
            f"got {cfg!r}. Export CUBLAS_WORKSPACE_CONFIG=:4096:8 first."
        )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # count every occurrence, not just the first per call site
    warnings.filterwarnings("always", message=".*deterministic.*")
    _STATE["_orig_showwarning"] = warnings.showwarning
    warnings.showwarning = _record_warning
    atexit.register(_dump_warnings)
    g = torch.Generator()
    g.manual_seed(0)
    _STATE["generator"] = g
    _STATE["deterministic"] = True
    print(
        "[deterministic] cudnn.deterministic=True cudnn.benchmark=False "
        f"use_deterministic_algorithms(True, warn_only=True) CUBLAS_WORKSPACE_CONFIG={cfg} "
        "train-loader generator seed=0 + worker_init_fn",
        flush=True,
    )


def loader_kwargs():
    """Extra DataLoader kwargs for the train loader: {} unless --deterministic."""
    if not _STATE["deterministic"]:
        return {}
    return {"worker_init_fn": _worker_init_fn, "generator": _STATE["generator"]}


def stage(name):
    """Print a timing/memory line when --stage-log is on; no-op otherwise."""
    if not _STATE["stage"]:
        return
    now = time.monotonic()
    line = {"stage": name, "dt_s": round(now - _STATE["t_prev"], 2), "t_s": round(now - _STATE["t0"], 2)}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        line["max_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        line["max_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    _STATE["t_prev"] = now
    msg = "[stage] " + json.dumps(line)
    # The stage log is provenance, not training state: a transient fsync failure
    # (spcl.utils.logging.Logger.flush() os.fsync's the tee'd log.txt; /scratch had a real
    # one-off I/O error that killed an otherwise-healthy Phase 5a run) must not kill a run
    # that is otherwise fine. print()'s own write() already lands the line before flush()'s
    # fsync can raise, so on failure this warns once and moves on -- no retry, which would
    # just duplicate the line already written.
    try:
        print(msg, flush=True)
    except OSError as e:
        sys.stderr.write(f"[stage] WARNING: log fsync failed for {name!r} ({e}); "
                          "line was written but may not be durable yet, run continues\n")


def train_seed_before_first_step(args):
    """--train-seed: re-seed every RNG once, just before the first train loader is built."""
    seed = getattr(args, "train_seed", None)
    if seed is None or _STATE["train_seeded"]:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if _STATE["generator"] is not None:
        _STATE["generator"].manual_seed(seed)
    _STATE["train_seeded"] = True
    print(f"[train-seed] re-seeded random/numpy/torch/cuda"
          f"{' + train-loader generator' if _STATE['generator'] is not None else ''} with {seed} "
          "before the first training step", flush=True)


def generator_state():
    """Train-loader generator state, or None when --deterministic is off (for D7 resume)."""
    g = _STATE["generator"]
    return None if g is None else g.get_state()


def restore_generator_state(state):
    if state is not None and _STATE["generator"] is not None:
        _STATE["generator"].set_state(state)


def train_seeded():
    return _STATE["train_seeded"]


def set_train_seeded(flag):
    _STATE["train_seeded"] = bool(flag)


def trace_losses(args, memory):
    """--loss-trace: record each forward's loss value; no-op otherwise."""
    if not getattr(args, "loss_trace", False):
        return
    _STATE["losses"] = []
    memory.register_forward_hook(lambda m, inp, out: _STATE["losses"].append(float(out.detach().double())))


def after_train(args, epoch, model, memory):
    """Called right after trainer.train(): stage line, loss trace, --stop-after-epoch."""
    stage(f"epoch{epoch}_train")
    if _STATE["losses"] is not None:
        np.save(os.path.join(args.logs_dir, f"loss_trace_epoch{epoch}.npy"), np.array(_STATE["losses"], dtype=np.float64))
        _STATE["losses"] = []
    if args.stop_after_epoch is None or epoch != args.stop_after_epoch:
        return
    path = os.path.join(args.logs_dir, f"stop_epoch{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "memory_features": memory.features.detach().cpu(),
            "memory_labels": memory.labels.detach().cpu(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
        },
        path,
    )
    stage(f"epoch{epoch}_stop_saved")
    print(f"[stop-after-epoch] saved {path}; exiting", flush=True)
    sys.stdout.flush()
    if _STATE["deterministic"]:
        _dump_warnings()
    sys.exit(0)
