#!/usr/bin/env python3
"""gpu_locks.py -- this project's side of the shared GPU slot locks (/scratch/dd25007/gpu_locks/).

The same protocol as beca-al's launch.sh (acquire_gpu), so the two projects never share a card
by accident:

    $BECA_LOCK_DIR/gpu<N>/slot<M>/{pid,run_dir}      pid = a live process for the job's lifetime

A slot is "live" while its pid answers kill -0; a slot whose pid is missing or dead is stale and
is removed (launch.sh does exactly the same) -- except a pid-less slot younger than 30 s, which is
a launcher mid-setup and counts as held.  A card is refused when it already holds
>= $BECA_MAX_PER_GPU (default 1) live slots -- whoever took them.

Library:  live(gpu) -> [slot dirs]; busy(gpus) -> {gpu: n_live}; acquire(gpu, run_dir, pid) -> slot | None;
          set_pid(slot, pid); release(slot).
CLI (for the bash launchers):
    gpu_locks.py busy                          # space-separated indices of cards with a live slot
    gpu_locks.py acquire <gpu> <run_dir> <pid> # prints the slot path; exit 3 = card held (pid: a process that outlives the CLI)
    gpu_locks.py setpid <slot> <pid>
    gpu_locks.py release <slot>
"""
import os
import shutil
import sys
import time

LOCK_DIR = os.environ.get("BECA_LOCK_DIR", "/scratch/dd25007/gpu_locks")
MAX_PER_GPU = int(os.environ.get("BECA_MAX_PER_GPU", "1"))
PENDING_S = 30   # a slot with no pid yet is a launcher between its mkdir and its pid write (launch.sh does that in two steps)
N_GPUS = 8


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True  # exists, owned by someone else


def _read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return ""


def live(gpu):
    """Live slot dirs on this card; stale ones (pid missing/dead) are removed, as launch.sh does."""
    gdir = f"{LOCK_DIR}/gpu{gpu}"
    out = []
    if not os.path.isdir(gdir):
        return out
    for name in sorted(os.listdir(gdir)):
        s = f"{gdir}/{name}"
        if not (name.startswith("slot") and os.path.isdir(s)):
            continue
        pid = _read(f"{s}/pid")
        if pid and _alive(pid):
            out.append(s)
        elif not pid and time.time() - os.stat(s).st_mtime < PENDING_S:
            out.append(s)   # being set up right now: counts as held, must not be swept as stale
        else:
            shutil.rmtree(s, ignore_errors=True)
    return out


def busy(gpus=range(N_GPUS)):
    """{gpu: number of live slots} for the cards that are at or over the per-GPU cap."""
    res = {}
    for g in gpus:
        n = len(live(g))
        if n >= MAX_PER_GPU:
            res[g] = n
    return res


def acquire(gpu, run_dir, pid=None):
    """Take a slot on `gpu` (atomic mkdir), pid defaulting to this process.  None if the card is held."""
    if len(live(gpu)) >= MAX_PER_GPU:
        return None
    gdir = f"{LOCK_DIR}/gpu{gpu}"
    os.makedirs(gdir, exist_ok=True)
    s = 0
    while True:
        slot = f"{gdir}/slot{s}"
        try:
            os.mkdir(slot)
            break
        except FileExistsError:
            s += 1
    open(f"{slot}/pid", "w").write(f"{pid or os.getpid()}\n")
    open(f"{slot}/run_dir", "w").write(f"{run_dir}\n")
    # a concurrent launcher's stale-scan can, in the microseconds after mkdir, see a slot with no
    # pid yet and remove it; confirm ours survived, else give up rather than pretend to hold it
    if _read(f"{slot}/pid") != str(pid or os.getpid()):
        return None
    return slot


def set_pid(slot, pid):
    open(f"{slot}/pid", "w").write(f"{pid}\n")


def release(slot):
    if slot:
        shutil.rmtree(slot, ignore_errors=True)


def _main(argv):
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd == "busy":
        print(" ".join(str(g) for g in sorted(busy())))
    elif cmd == "acquire" and len(argv) == 5:   # pid is mandatory: this CLI exits at once, so its own pid would go stale
        slot = acquire(int(argv[2]), argv[3], argv[4])
        if slot is None:
            print(f"gpu {argv[2]} is held (>= {MAX_PER_GPU} live slot(s) in {LOCK_DIR})", file=sys.stderr)
            return 3
        print(slot)
    elif cmd == "setpid" and len(argv) == 4:
        set_pid(argv[2], argv[3])
    elif cmd == "release" and len(argv) == 3:
        release(argv[2])
    else:
        print(__doc__, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
