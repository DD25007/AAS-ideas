"""Samplers for `main.py`'s AL workflow.

NOT a duplicate of `spcl/sampling/` -- the two are different implementations with
incompatible contracts, and merging them would be a redesign, not a de-duplication:

| | this package | `spcl/sampling/` |
|---|---|---|
| constructor | `Sampler(database, n, gallery_size)` | `Sampler(args=...)` |
| consumer | `main.py --sampler` | `train_al.py -s` |
| `ConsistencySampling` | 1253 lines, extra `get_sampling_pools` / `select2` / `top_n_high_entropy_outliers` | 427 lines |

An AST comparison finds **0 of 3** shared definitions identical in `consistency.py`, and
1 of 4 in `random.py`. **`spcl/sampling/` is what produced the published Table 1 AL rows.**

These used to shadow each other -- both were importable as `sampling` -- which silently
killed a 16-run batch (Todo.md item 12). The SpCL copy was moved under the `spcl` package
to make that impossible.
"""
