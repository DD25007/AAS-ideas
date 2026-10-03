"""D6 state saver for train_al.py: pseudo-labels, AL pairs/answers, and the sampler's own views.

Default OFF. With `--save-state DIR` every artefact Phases 2, 4 and 6 need is written to DIR:

  epoch0_features.npy            memory bank features after initialisation (float32)
  epoch{e}_rerank.npy            k-reciprocal Jaccard matrix (only epochs in --save-rerank-epochs)
  epoch{e}_dbscan_raw.npy        DBSCAN(eps) output, before outlier relabelling (int32, -1 = outlier)
  epoch{e}_labels_pre.npy        labels entering constrained_clustering (= USL labels; int32)
  epoch{e}_labels_post.npy       labels leaving it, i.e. what the memory bank is given (int32)
  cycle{c}_pairs.npy             pairs asked THIS cycle: (i, j, answer), answer 1 = must-link
  cycle{c}_ml.npy / _cl.npy      cumulative ML / CL pair sets after this cycle
  cycle{c}_meta.json             epoch, counts, budget, shortfall, sha256 of each pair set
  cc_call{n:03d}_{method}_epoch{e}.npz   every constrained_clustering call: labels in/out

The last one is how the sampler's internal views are captured without touching sampler code:
`constrained_clustering` is wrapped in place of every module attribute that references it, so
ConSam's DBSCAN view (method "DBSCAN"), its FINCH view ("FINCH") and train_al's own refinement
call ("~DBSCAN") are all recorded. The wrapper only copies arrays; it calls the original.
"""

import hashlib
import json
import os
import sys

import numpy as np

_S = {"dir": None, "rerank_epochs": (), "calls": 0, "epoch": None, "cycles": 0, "wrapped": []}


def _p(name):
    return os.path.join(_S["dir"], name)


def _save(name, arr, dtype=None):
    a = np.asarray(arr if not hasattr(arr, "detach") else arr.detach().cpu().numpy())
    if dtype is not None:
        a = a.astype(dtype, copy=False)
    np.save(_p(name), a)
    return a


def _sha_pairs(pairs):
    canon = sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in pairs)
    return hashlib.sha256(json.dumps(canon).encode()).hexdigest()


def _wrap_constrained_clustering():
    """Replace every module-level reference to constrained_clustering with a recorder."""
    import spcl.utils.constrained_clustering as cc

    original = cc.constrained_clustering

    def recorder(labels, rerank_dist, method, *a, **kw):
        idx = _S["calls"]
        _S["calls"] += 1
        lab_in = np.asarray(labels).copy()
        out = original(labels, rerank_dist, method, *a, **kw)
        np.savez_compressed(
            _p(f"cc_call{idx:03d}_{str(method).strip('~')}_epoch{_S['epoch']}.npz"),
            labels_in=lab_in.astype(np.int32),
            labels_out=np.asarray(out).astype(np.int32),
            method=str(method),
            n_ml=len(kw.get("ml_pairs", a[1] if len(a) > 1 else []) or []),
            n_cl=len(kw.get("cl_pairs", a[2] if len(a) > 2 else []) or []),
        )
        return out

    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            if getattr(mod, "constrained_clustering", None) is original:
                setattr(mod, "constrained_clustering", recorder)
                _S["wrapped"].append(mod.__name__)
        except Exception:  # noqa: BLE001 -- some modules raise on getattr
            continue


def enable(args):
    """Turn the saver on when --save-state is given. Returns True when active."""
    d = getattr(args, "save_state", None)
    if not d:
        return False
    os.makedirs(d, exist_ok=True)
    _S["dir"] = d
    spec = getattr(args, "save_rerank_epochs", "0") or ""
    _S["rerank_epochs"] = tuple(int(x) for x in spec.split(",") if x.strip() != "")
    _wrap_constrained_clustering()
    print(f"[save-state] {d} | rerank epochs {_S['rerank_epochs']} | "
          f"constrained_clustering wrapped in {_S['wrapped']}", flush=True)
    return True


def on() -> bool:
    return _S["dir"] is not None


def set_epoch(epoch):
    if on():
        _S["epoch"] = int(epoch)


def features(feats):
    if on():
        _save("epoch0_features.npy", feats, np.float32)


def rerank(epoch, mat):
    if on() and int(epoch) in _S["rerank_epochs"]:
        _save(f"epoch{int(epoch)}_rerank.npy", mat, np.float32)


def dbscan_raw(epoch, labels):
    if on():
        _save(f"epoch{int(epoch)}_dbscan_raw.npy", labels, np.int32)


def labels_pre(epoch, labels):
    if on():
        _save(f"epoch{int(epoch)}_labels_pre.npy", labels, np.int32)


def labels_post(epoch, labels):
    if on():
        _save(f"epoch{int(epoch)}_labels_post.npy", labels, np.int32)


def cycle(epoch, sampler, ml_pairs, cl_pairs):
    """Cumulative ML/CL sets, the pairs added this cycle with their answers, and a meta file."""
    if not on():
        return
    _S["cycles"] += 1
    c = _S["cycles"]
    ml = np.array([[int(i), int(j)] for i, j in ml_pairs], dtype=np.int64).reshape(-1, 2)
    cl = np.array([[int(i), int(j)] for i, j in cl_pairs], dtype=np.int64).reshape(-1, 2)
    np.save(_p(f"cycle{c}_ml.npy"), ml)
    np.save(_p(f"cycle{c}_cl.npy"), cl)
    prev_ml, prev_cl = _S.get("prev_ml", 0), _S.get("prev_cl", 0)
    new = np.concatenate([
        np.column_stack([ml[prev_ml:], np.ones(len(ml) - prev_ml, dtype=np.int64)]),
        np.column_stack([cl[prev_cl:], np.zeros(len(cl) - prev_cl, dtype=np.int64)]),
    ]) if (len(ml) > prev_ml or len(cl) > prev_cl) else np.empty((0, 3), dtype=np.int64)
    np.save(_p(f"cycle{c}_pairs.npy"), new)
    _S["prev_ml"], _S["prev_cl"] = len(ml), len(cl)
    meta = {
        "cycle": c, "epoch": int(epoch),
        "ml_total": len(ml), "cl_total": len(cl), "added_this_cycle": int(len(new)),
        "budget_per_cycle": int(getattr(sampler, "n", 0) or 0),
        "shortfall": int(getattr(sampler, "rec_shortfall", 0) or 0),
        "ml_sha256": _sha_pairs(ml), "cl_sha256": _sha_pairs(cl),
        "self_pairs_ml": int(sum(1 for i, j in ml if i == j)),
    }
    with open(_p(f"cycle{c}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[save-state] cycle {c}: {json.dumps(meta)}", flush=True)


def state_for_resume():
    return {"calls": _S["calls"], "cycles": _S["cycles"],
            "prev_ml": _S.get("prev_ml", 0), "prev_cl": _S.get("prev_cl", 0)}


def restore_state(d):
    if on() and d:
        _S.update({k: d[k] for k in ("calls", "cycles", "prev_ml", "prev_cl") if k in d})
