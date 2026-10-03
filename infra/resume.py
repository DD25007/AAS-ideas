"""Resume for train_al.py: save everything an epoch boundary needs, restore it exactly.

Default OFF. Two flags:
  --resume-save-epochs LIST   after finishing each listed epoch (training, evaluation and the
                              LR step), write <logs_dir>/resume_epoch{e}.pth
  --resume PATH               start from such a file: the next epoch is its epoch + 1
  --exit-after-epoch N        exit at the END of epoch N (after evaluation, the LR step and any
                              resume save), as opposed to --stop-after-epoch, which exits
                              immediately after the training step

What a resume file holds (what the stock checkpoint lacks):
  model state_dict, optimizer, LR scheduler
  memory bank features and labels
  sampler ML/CL pair sets (all_selected_pairs is rebuilt from them by set_constraints)
  best_mAP, indep_thres, the cumulative ML/CL lists train_al holds
  every RNG: python, numpy, torch CPU, torch CUDA (all devices), and the --deterministic
  train-loader generator
  the --train-seed "already reseeded" flag and the state saver's counters

Restoring is refused when the run's own key arguments differ from the saved ones, so a resume
cannot silently continue a different configuration.
"""

import os
import random

import numpy as np
import torch

from infra import determinism as _det
from infra import state_saver as _save

KEY_ARGS = ("dataset", "subset", "sampler", "seed", "split_seed", "partition_seed", "epochs",
            "iters", "eval_step", "batch_size", "num_instances", "height", "width", "lr",
            "step_size", "eps", "eps_gap", "k1", "k2", "canonical_pairs", "deterministic",
            "train_seed", "al_cycle_cap", "al_cycle_caps", "al_offset", "consam_eps",
            "consam_sim_thresh", "fast_rerank",
            # a resume must not silently continue under a different NP3 behaviour
            "np3_per_cluster_skip")


def save_epochs(args):
    spec = getattr(args, "resume_save_epochs", None) or ""
    return {int(x) for x in str(spec).split(",") if x.strip() != ""}


def save(args, epoch, model, memory, optimizer, lr_scheduler, sampler, best_mAP, indep_thres,
         ml_pairs, cl_pairs):
    """Write a resume file after epoch `epoch` is completely finished; no-op unless asked."""
    if epoch not in save_epochs(args):
        return None
    path = os.path.join(args.logs_dir, f"resume_epoch{epoch}.pth")
    state = {
        "format": "aas-ideas/resume/v1",
        "epoch": int(epoch),
        "args": {k: getattr(args, k, None) for k in KEY_ARGS},
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "memory_features": memory.features.detach().cpu(),
        "memory_labels": memory.labels.detach().cpu(),
        "best_mAP": float(best_mAP),
        "indep_thres": float(indep_thres) if indep_thres is not None else None,
        "ml_pairs": [(int(i), int(j)) for i, j in ml_pairs],
        "cl_pairs": [(int(i), int(j)) for i, j in cl_pairs],
        "sampler_ml": [(int(i), int(j)) for i, j in getattr(sampler, "ml_pairs", [])] if sampler else [],
        "sampler_cl": [(int(i), int(j)) for i, j in getattr(sampler, "cl_pairs", [])] if sampler else [],
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
        "rng_loader_generator": _det.generator_state(),
        "det_train_seeded": _det.train_seeded(),
        "save_state": _save.state_for_resume(),
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    print(f"[resume] wrote {path} (end of epoch {epoch})", flush=True)
    return path


def maybe_load(args, model, memory, optimizer, lr_scheduler, sampler):
    """-> (start_epoch, best_mAP, indep_thres, ml_pairs, cl_pairs). (0, 0, None, [], []) when off."""
    path = getattr(args, "resume", None)
    if not path:
        return 0, 0, None, [], []
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format") != "aas-ideas/resume/v1":
        raise RuntimeError(f"{path}: not a resume file written by infra/resume.py")
    mismatch = {k: (v, getattr(args, k, None)) for k, v in state["args"].items()
                if getattr(args, k, None) != v}
    if mismatch:
        raise RuntimeError(f"--resume refused: these arguments differ from the saved run: {mismatch}")

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    memory.features = state["memory_features"].cuda()
    memory.labels = state["memory_labels"].cuda()
    if sampler is not None:
        sampler.set_constraints(list(map(tuple, state["sampler_ml"])), list(map(tuple, state["sampler_cl"])))
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"])
    torch.cuda.set_rng_state_all(state["rng_cuda"])
    _det.restore_generator_state(state["rng_loader_generator"])
    _det.set_train_seeded(state["det_train_seeded"])
    _save.restore_state(state.get("save_state"))
    start = int(state["epoch"]) + 1
    print(f"[resume] loaded {path}: resuming at epoch {start}, best_mAP={state['best_mAP']:.4f}, "
          f"indep_thres={state['indep_thres']}, "
          f"{len(state['ml_pairs'])} ML / {len(state['cl_pairs'])} CL constraints", flush=True)
    return (start, state["best_mAP"], state["indep_thres"],
            list(map(tuple, state["ml_pairs"])), list(map(tuple, state["cl_pairs"])))


def exit_after(args, epoch):
    n = getattr(args, "exit_after_epoch", None)
    if n is not None and epoch == n:
        print(f"[exit-after-epoch] finished epoch {epoch}; exiting", flush=True)
        raise SystemExit(0)
