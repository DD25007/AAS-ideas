# Launchers

Everything runs through one script.

| file | role |
|---|---|
| `run.sh` | the only entry point — list / status / one / queue / clean |
| `common.sh` | shared config: interpreter, paths, thread caps, GPU gate. Sourced, never run |
| `outputs.sh` | manage output trees that live on `/scratch` behind a symlink |

## Running the full Table 1

```bash
bash scripts/run.sh queue /scratch/dd25007/HITL-ReID/outputs_final
```

52 jobs: 6 AL samplers x seeds 1-4, plus 7 non-AL rows x seeds 0-3. It places one job per
GPU, only on devices with `MIN_FREE_MIB` free, retries anything that dies, and adopts jobs
already running so it is safe to restart.

```bash
bash scripts/run.sh list                      # the job list
bash scripts/run.sh status  <OUTROOT>         # done / running / pending
bash scripts/run.sh clean   <OUTROOT>         # drop incomplete logs and .failed dirs
bash scripts/run.sh one al:ConSam:1 0 <OUT>   # one job, foreground, on GPU 0
```

A job is `<kind>:<name>:<seed>`:

- `al:<sampler>:<seed>` — `USL random GBAS LBAS CC ConSam`, seeds 1-4.
  `USL` is the SpCL (Base) row and passes no `-s` at all.
- `row:<row>:<seed>` — seeds 0-3. Here the seed is `--sample_seed`, which varies the 20%
  held-out gallery split; `--seed` (training) is pinned at 1. The README's +/- averages
  over exactly this.

## Knobs

```bash
MIN_FREE_MIB=20000 GPUS="0 1 2 3" bash scripts/run.sh queue <OUTROOT>
MAX_JOBS_PER_GPU=2 bash scripts/run.sh queue <OUTROOT>
PYTHON=/storage/dd25007/conda_envs/hitl-modern/bin/python bash scripts/run.sh one ...
```

The VRAM gate is not cosmetic. Measured demand is ~9.3 GiB per job (6.1-7.7 GiB resident
plus a ~1.5 GiB faiss `StandardGpuResources` temporary during `compute_jaccard_distance`).
That temporary is what OOM-killed nine runs overnight on 2026-08-30, and four more when
`MAX_JOBS_PER_GPU=3` was tried — concurrent jobs hit their faiss spike together, so
per-job headroom is not additive. The 16 GiB default leaves ~70% headroom for a co-tenant
that grows *after* placement, which a launch-time check cannot otherwise cover.

## Two rows that are not what they look like

`miewid` runs under its own interpreter (`MIEWID_PYTHON`, default `hitl-miewid`) because
the model loads via `trust_remote_code` and its modelling code predates the transformers 5
loader (Todo.md item 14). `run.sh` selects it per job — nothing else needs to change.

`swin_foundation` *is* the MegaDescriptor row. `get_model` substitutes plain ImageNet
Swin-L for `MegaDescriptor-L-224` whenever `--dataset combined`, so naming the model
directly reaches identical weights by a shorter path. That substitution is the authors'
and is intended (Todo.md item 15). `megadescriptor_realweights` is deliberately absent
from the grid: it needs a `--baseline` bypass of that swap which has since been reverted,
so it cannot be re-run.

## If you rename a tmux session

`run.sh` names sessions `hitl_<kind>_<name>_<seed>_g<gpu>` and `common.sh`'s `gpu_jobs()`
matches on that `_g<gpu>` suffix. The two must stay in step: when they did not, `gpu_jobs()`
counted zero for every device and `MAX_JOBS_PER_GPU` silently stopped applying — three jobs
landed on one GPU before it was noticed.
