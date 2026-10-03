#!/usr/bin/env python3
"""test_gpu_locks.py -- the slot-lock library refuses a held card, frees a released one and sweeps a stale one.

Runs against a throw-away lock dir (BECA_LOCK_DIR), never the shared /scratch/dd25007/gpu_locks.
    python3 scripts/test_gpu_locks.py
"""
import os
import subprocess
import sys
import tempfile

d = tempfile.mkdtemp(prefix="gpu_locks_test_", dir=os.environ.get("TEST_TMP", "/scratch/dd25007"))
os.environ["BECA_LOCK_DIR"] = d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpu_locks  # noqa: E402

CLI = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_locks.py")]
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name)


holder = subprocess.Popen(["sleep", "60"])
try:
    s = gpu_locks.acquire(3, "/x/run", holder.pid)
    check("acquire takes a free card", s is not None)
    check("busy() lists the held card", list(gpu_locks.busy()) == [3])
    check("a second acquire on the held card is refused", gpu_locks.acquire(3, "/x/run2", holder.pid) is None)
    check("the CLI refuses it with exit 3",
          subprocess.run(CLI + ["acquire", "3", "/x/run3", str(holder.pid)], env=dict(os.environ), capture_output=True).returncode == 3)
    check("another card is still free", gpu_locks.acquire(4, "/x/run4", holder.pid) is not None)
    gpu_locks.release(s)
    check("release frees the card", 3 not in gpu_locks.busy())
    s2 = gpu_locks.acquire(5, "/x/run5", holder.pid)
    holder.kill(); holder.wait()
    check("a slot whose pid died is swept as stale", 5 not in gpu_locks.busy())
finally:
    holder.kill()
    import shutil
    shutil.rmtree(d, ignore_errors=True)
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
