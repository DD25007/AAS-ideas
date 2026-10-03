# CLAUDE.md — AAS-ideas (`/storage/dd25007/AAS-ideas`, branch `ideas`)

Fresh working repo for new ideas on top of AAS (ConSam). Base = `5fab110` (the code of the 56.21 reproduction) plus shared
infrastructure only (determinism, resume, `--train-seed`, `--canonical-pairs`, `--np3-per-cluster-skip`, AL cycle caps,
fast rerank, `[cc-diag]` log lines, GPU slot locks). No idea-specific code lives here.
**Anything about idea-1 (local features, Farthest, purification, BECA-L): git tag `idea-1-final` (branches `archive/*`).**
The old checkout `/storage/dd25007/HITL-ReID` is not part of this work; do not edit it or `/storage/dd25007/beca-al*`.

## AAS reference
- **56.19 ± 0.25 macro mAP** (13 datasets): C0 = plain ConSam, `--deterministic --canonical-pairs --fast-rerank`,
  `--train-seed 0..3` (55.90–56.51), 67,314.5 human pairs on average (67,315 rounded; seed values 67,358 / 68,413 / 66,129 / 67,358).
- 56.21 (sd 0.43): the non-deterministic 4-repeat figure in README.md (`outputs_final`). Judge new ideas against C0, not against it.

## Working rules
1. **Preregister before any run**: write the prereg (hypothesis, arms, seeds, decision rule), `sha256sum` it into a `.sha256.txt`, only then launch.
2. **Judge on mAP** (macro, final epoch). **n = 4 seeds** (`--train-seed 0..3`), exact permutation test; state p and Hedges g, not just means.
3. **Cost table first** for anything above 20 GPU-h (runs × hours × cards); wait for approval.
4. **VRAM gate**: start only on a card with ≥ 16 GiB free (`nvidia-smi`) and no live slot in `/scratch/dd25007/gpu_locks`;
   hold a slot for the job's lifetime (`scripts/gpu_guard.sh`, `python3 scripts/gpu_locks.py busy`). Cards are shared with other projects.
5. Findings, caveats and corrections go in markdown reports; errors are struck through, never silently replaced.

## How to run
- Env: `/storage/dd25007/conda_envs/hitl/bin/python`; export `PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` (`--deterministic` refuses to start without the latter).
- Data: `dataroot/wildlife` -> `/storage/dd25007/dataset` (symlink, gitignored). Outputs: `outputs/` -> `/scratch/dd25007/AAS-ideas-out/` (gitignored); put `--logs-dir` under it.
- Plain AAS: `train_al.py -d wildlife --subset combined -s ConSam --seed 1 --split-seed 0 --canonical-pairs --data-dir $PWD/dataroot -j 6 --deterministic --train-seed S --fast-rerank --logs-dir outputs/<run>`.
- Gate (epoch 0 must reproduce C0 seed 0's first `[record]`, all 19 fields): `scripts/gate_epoch0.sh <gpu>` then `scripts/compare_epoch0_record.py <C0 log> <new log>`.
- Smoke test after any code change: `python train_al.py --help` (CPU), `python scripts/test_gpu_locks.py`.
- Infra lives in `infra/` (determinism, resume, state_saver, fast_rerank); every flag is a no-op unless given.
