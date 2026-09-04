"""Samplers for `train_al.py` -- the implementation behind the published Table 1 AL rows.

NOT a duplicate of the repo-root `sampling/` package. Different constructor contract
(`args=...` here, `(database, n, gallery_size)` there), different consumer, and materially
different algorithms -- `consistency.py` shares no identical definition with its root
counterpart. See the header of `sampling/__init__.py` for the full comparison.
"""
