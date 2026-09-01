# Launcher scripts

Everything that produced the Table 1 reproduction in [`../README.md`](../README.md).

Rescued on 2026-09-01 from an ephemeral session scratchpad under `/tmp` — they were never
under version control, so the exact flags, thread caps and VRAM gate behind every published
number were one `/tmp` cleanup away from being lost.

## Layout

| script | what it does |
|---|---|
| `common.sh` | shared config + GPU scheduling helpers; sourced by everything else |
| `run_grid.sh` | one AL run — `run_grid.sh <SAMPLER\|USL> <SEED> <GPU> <OUTROOT>` |
| `run_row.sh` | one baseline / foundation / skyline row — `run_row.sh <ROW> <SEED> <GPU> <OUTROOT>` |
| `queue.sh` | 1-job-per-GPU queue with VRAM gate + auto-retry — `queue.sh <al\|row> <OUTROOT> [jobs...]` |
| `clean_logs.sh` | delete logs from runs that never reached the `mean,` summary |

## One-time setup

`dataroot/` is a machine-specific symlink holding absolute paths, so it is gitignored and
must be recreated on each machine:

```bash
mkdir -p dataroot && ln -sfn /storage/dd25007/dataset dataroot/wildlife
```

That gives the SpCL wildlife loader the `<data-dir>/wildlife/wildlifereid-10k/metadata.csv`
path it expects. Point it wherever WildlifeReID-10k actually lives, or override `DATAROOT`.

## Running the full reproduction

```bash
tmux new-session -d -s q_al  "bash scripts/queue.sh al  outputs_new"   # 24 AL runs,  ~12h each
tmux new-session -d -s q_row "bash scripts/queue.sh row outputs_new"   # 24 row runs, ~15min-4h
```

Both queues can run at once — GPU ownership is arbitrated through the shared
`hitl_*_g<N>_*` tmux session naming in `common.sh`, so neither will place a second job on a
GPU the other is already using.

Single job, no queue:

```bash
bash scripts/run_grid.sh ConSam 1 0 outputs_new          # AAS, seed 1, GPU 0
bash scripts/run_row.sh  resnet50_baseline 0 0 outputs_new
```

## Output layout

Both runners write `<OUTROOT>/{al,table1}/<row>/s<seed>/`, matching the frozen baseline in
`outputs_initial/`, so `aggregate_results.py --compare outputs_initial outputs_new` is a
straight tree walk.

## Configuration

Every knob in `common.sh` is overridable from the environment:

```bash
PYTHON=/storage/dd25007/conda_envs/alm/bin/python  bash scripts/queue.sh row outputs_alm
GPUS="0 1 2 3"  MIN_FREE_MIB=20000  bash scripts/queue.sh al outputs_new
```

- `PYTHON` — defaults to the `hitl` env. Point it at `alm` to re-run under the original
  reproduction environment.
- `DATAROOT` — `$REPO/dataroot`, whose `wildlife` symlink gives the SpCL loader the
  `<data-dir>/wildlife/wildlifereid-10k/metadata.csv` path it expects.
- `DATASET_PATH` — the dataset root `main.py` takes directly.
- `MIN_FREE_MIB` — 16000. Peak real demand is ~9.3 GiB (6.1 GiB resident + a ~1.5 GiB faiss
  `StandardGpuResources` temporary during `compute_jaccard_distance`); the headroom covers a
  co-tenant that grows *after* we launch, which a launch-time check cannot otherwise catch.
- `OMP_NUM_THREADS` and friends — 8. Left unset, each process sizes its BLAS pool to all 224
  cores; with 8 concurrent jobs the load average hit 715.

## Two rows that cannot currently be re-run

`run_row.sh` refuses both rather than silently writing wrong numbers under a right-looking
directory name:

- **`megadescriptor`** — needs an opt-out of the `get_model` MegaDescriptor→Swin-L swap that
  no longer exists (see `../README.md` and Todo.md item 15). Without it the row would
  duplicate `swin_foundation`.
- **`miewid`** — blocked on transformers 5.x (Todo.md item 14).
