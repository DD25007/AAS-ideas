"""Disk cache for the representative-gallery selection (Todo.md O4).

`get_representative_gallery_from_train` runs MegaDescriptor over the whole train split
(~29k images, measured 4m35s-5m38s) on every process start, purely to pick <=K exemplars
per identity. Profiled on an eval-only run it is 68% of all remaining feature-extraction
time once O2/O6 land.

The selection does not depend on the model being trained -- it uses a *fixed*
MegaDescriptor-L-224 -- so for a given (subset, split seed, images-per-id, metric) it is a
pure function of the metadata. Caching the chosen row positions therefore skips the whole
pass without changing what is selected.

What is cached is deliberately small and checkable: the ordered list of positional indices
into `default_train.metadata`. Everything else is rebuilt from it, so a stale or corrupt
entry cannot silently substitute different images -- the key covers every input, and the
caller re-derives the frames.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import random as _random

CACHE_DISABLED = os.environ.get("HITL_NO_GALLERY_CACHE", "").lower() in (
    "1",
    "true",
    "yes",
)


def _cache_dir() -> str:
    d = os.environ.get(
        "HITL_GALLERY_CACHE",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "gallery"),
    )
    os.makedirs(d, exist_ok=True)
    return d


def build_key(
    *,
    subset,
    split_seed,
    images_per_id,
    metric,
    model_name,
    train_metadata,
    labeled_identities,
):
    """Hash every input the selection depends on.

    `train_metadata` paths and `labeled_identities` are included so that a changed dataset
    or a changed labelled/unlabelled split misses the cache rather than reusing a
    selection made for different data.
    """
    h = hashlib.sha256()
    # The selection runs MegaDescriptor and numpy float ops, so a library change can
    # legitimately change it. Without these in the key, an env upgrade would silently be
    # served a selection computed under the old stack -- which would mask exactly the
    # difference a parity run is trying to measure.
    import numpy as _np
    import torch as _torch

    for part in (
        str(subset),
        str(split_seed),
        str(images_per_id),
        str(metric),
        str(model_name),
        "np" + _np.__version__,
        "torch" + _torch.__version__,
    ):
        h.update(part.encode())
        h.update(b"\x00")
    paths = train_metadata["path"].astype(str).tolist()
    h.update(str(len(paths)).encode())
    h.update(hashlib.sha256("\n".join(paths).encode()).digest())
    ids = sorted(str(i) for i in labeled_identities)
    h.update(hashlib.sha256("\n".join(ids).encode()).digest())
    return h.hexdigest()[:32]


def _rng_state():
    """Snapshot the RNGs the selection consumes."""
    import numpy as np

    return base64.b64encode(
        pickle.dumps({"numpy": np.random.get_state(), "python": _random.getstate()})
    ).decode()


def _restore_rng(blob):
    import numpy as np

    st = pickle.loads(base64.b64decode(blob))
    np.random.set_state(st["numpy"])
    _random.setstate(st["python"])


def load(key):
    """-> ordered list of positional indices, or None on a miss.

    Also restores the RNG state to where the original selection left it. The compute path
    consumes `np.random.choice` draws; a hit that just returned early would leave the
    global RNG somewhere else entirely, and every downstream consumer (notably
    `RandomSampling`) would diverge. Without this the cache is not transparent.
    """
    if CACHE_DISABLED:
        return None
    path = os.path.join(_cache_dir(), key + ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            blob = json.load(fh)
        if blob.get("key") != key or "rng_state" not in blob:
            return None  # entries without an RNG snapshot are treated as misses
        _restore_rng(blob["rng_state"])
        return blob["gallery_positions"]
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        ValueError,
    ):
        return None  # a damaged entry is a miss, never a wrong answer


def save(key, positions, note=""):
    if CACHE_DISABLED:
        return
    path = os.path.join(_cache_dir(), key + ".json")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(
                {
                    "key": key,
                    "note": note,
                    "gallery_positions": [int(i) for i in positions],
                    "rng_state": _rng_state(),
                },
                fh,
            )
        os.replace(tmp, path)  # atomic: concurrent runs cannot read a half-file
    except OSError:
        pass  # caching is an optimisation, never a hard failure
