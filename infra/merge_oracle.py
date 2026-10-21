"""MERGE-ORACLE arm (Phase D2.1, reports/phaseD2_1_gate.md). Default OFF; every function is a no-op unless its flag is given.

  --merge-queries oracle20   at the AL cycles listed in --merge-cycles (cycle index k = epoch al_offset + k * eval_step), AFTER ConSam's answers of that
                             cycle are in and the labels are re-clustered, ask --merge-q-per-cycle extra questions about pairs of pseudo-label units:
                               units      = distinct labels of the current pseudo-labels (singletons included), centroids / medoids on the memory bank;
                               candidates = for every unit its 20 nearest units of the same (majority) dataset by centroid cosine;
                               order      = D1.6 ORACLE: candidate pairs whose two units share a GT identity first (centroid order inside), then all
                                            other same-dataset unit pairs in centroid order. GT is used ONLY to order; every question, 'different'
                                            answers included, is a human pair;
                               question   = (medoid image of A, medoid image of B); pairs whose answer the existing ML/CL already imply
                                            (transitive closure) are skipped / merged for free and cost nothing.
                             The answers are appended to the sampler's ML / CL lists (so ConSam's later cycles see them), then the caller re-runs
                             constrained_clustering.
  --consam-cycle0-cut K      after ConSam's first cycle, drop the LAST K pairs it produced, in its production order (equal-budget arm).
  --human-pair-cap N         hard cap on all human pairs (ConSam + merge): the ConSam budget of a cycle is min(its budget, N - consumed - merge
                             questions still to come).

The query loop is the loop of tools/fragment_sim/sim_d3.py::simulate (D1.6), with units from the live pseudo-labels; GT enters only through the
order and the oracle answers. Nothing here uses a random number generator.
"""
import collections
import json

import numpy as np


# ------------------------------------------------------------------ flags
def merge_epochs(args):
    if getattr(args, "merge_queries", "off") == "off":
        return set()
    ks = [int(x) for x in str(args.merge_cycles).split(",") if x.strip() != ""]
    return {int(args.al_offset) + k * int(args.eval_step) for k in ks}


def active(args, epoch):
    return epoch in merge_epochs(args)


def any_flag(args):
    return (getattr(args, "merge_queries", "off") != "off" or getattr(args, "consam_cycle0_cut", 0)
            or getattr(args, "human_pair_cap", None) is not None)


def check_args(args, parser):
    if not any_flag(args):
        return
    if args.sampler != "ConSam":
        parser.error("--merge-queries / --consam-cycle0-cut / --human-pair-cap need -s ConSam")
    if args.al_cycle_cap is not None or args.al_cycle_caps is not None:
        parser.error("--merge-queries / --human-pair-cap are exclusive with --al-cycle-cap(s)")
    if args.full_supervision or args.oracle_merge:
        parser.error("--merge-queries is exclusive with --full-supervision and --oracle-merge")
    if args.merge_queries not in ("off", "oracle20"):
        parser.error("--merge-queries must be off or oracle20")
    if args.merge_queries != "off":
        eps = merge_epochs(args)
        if not eps or any(e < 0 or e >= args.epochs for e in eps):
            parser.error("--merge-cycles gives epochs outside [0, --epochs)")
        if args.al_offset + 0 * args.eval_step in eps and args.consam_cycle0_cut == 0:
            pass  # a merge step at cycle 0 is allowed by the code; the pre-registered design uses cycles 1-4 only


# ------------------------------------------------------------------ budget
def reserve_after(args, epoch):
    """merge questions still to be asked at this epoch (after ConSam's select) and at later cycles"""
    return int(args.merge_q_per_cycle) * sum(1 for e in merge_epochs(args) if e >= epoch)


def limit_budget(args, sampler, epoch, consumed):
    """--human-pair-cap: plan this cycle's ConSam draw with min(budget, cap - consumed - reserved merge questions). Returns a log dict (or None if off)."""
    cap = getattr(args, "human_pair_cap", None)
    if cap is None:
        return None
    if not hasattr(sampler, "_cap_base_n"):
        sampler._cap_base_n = sampler.n
    reserve = reserve_after(args, epoch)
    allowed = max(0, int(cap) - int(consumed) - reserve)
    sampler.n = min(sampler._cap_base_n, allowed)
    d = {"epoch": int(epoch), "cap": int(cap), "consumed": int(consumed), "reserved_merge_questions": reserve, "allowed": allowed,
         "sampler_budget": int(sampler.n), "base_budget": int(sampler._cap_base_n), "cap_binds": bool(allowed < sampler._cap_base_n)}
    print("[human-pair-cap]", json.dumps(d), flush=True)
    return d


def cut_last(sampler, k):
    """Drop the last k pairs ConSam's most recent select() produced, in production order, from the sampler's ML / CL lists and selected set.
    Only valid right after a first cycle (the lists then hold exactly that call's pairs)."""
    order = sampler.last_selection_order
    if k <= 0:
        return None
    assert len(order) == len(sampler.ml_pairs) + len(sampler.cl_pairs), "cut_last is only valid on the first cycle"
    assert k <= len(order)
    tail = order[len(order) - k:]
    n_ml = sum(1 for _, m in tail if m)
    n_cl = k - n_ml
    for fs, _ in tail:
        sampler.all_selected_pairs.discard(fs)
    if n_ml:
        del sampler.ml_pairs[len(sampler.ml_pairs) - n_ml:]
    if n_cl:
        del sampler.cl_pairs[len(sampler.cl_pairs) - n_cl:]
    del sampler.last_selection_order[len(order) - k:]
    d = {"cut": int(k), "cut_ml": int(n_ml), "cut_cl": int(n_cl), "ml_left": len(sampler.ml_pairs), "cl_left": len(sampler.cl_pairs),
         "consumed_left": len(sampler.ml_pairs) + len(sampler.cl_pairs)}
    print("[consam-cycle0-cut]", json.dumps(d), flush=True)
    return d


# ------------------------------------------------------------------ merge questions
class _UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x


def unit_geometry(labels, feats, ds):
    """units of `labels`: members, centroids (unit-normalised), medoids, majority dataset index.  feats: (N, D) float32."""
    unit = np.unique(labels, return_inverse=True)[1]
    U = int(unit.max()) + 1
    cnt = np.bincount(unit)
    order = np.argsort(unit, kind="stable")
    st = np.r_[0, np.cumsum(cnt)[:-1]]
    members = [order[st[u]:st[u] + cnt[u]] for u in range(U)]
    udom = np.array([collections.Counter(ds[m].tolist()).most_common(1)[0][0] for m in members])
    cent = np.stack([feats[m].mean(0) for m in members])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    medoid = np.array([members[u][np.argmax(feats[members[u]] @ cent[u])] for u in range(U)])
    return dict(unit=unit, U=U, members=members, udom=udom, cent=cent, medoid=medoid)


def ask(labels, feats, ml_pairs, cl_pairs, gt, ds, q, k_nn=20):
    """-> (answers [(a, b, same)], stats).  labels (N,) int, feats (N, D) float32, gt (N,) int GT identity, ds (N,) int dataset index."""
    g = unit_geometry(labels, feats, ds)
    U, unit, medoid, udom, members = g["U"], g["unit"], g["medoid"], g["udom"], g["members"]
    Dc = 1 - g["cent"] @ g["cent"].T
    iu = np.triu_indices(U, 1)
    keep = udom[iu[0]] == udom[iu[1]]
    iu = (iu[0][keep], iu[1][keep])
    oB = np.argsort(Dc[iu], kind="stable")                               # centroid order over all same-dataset unit pairs
    # candidates: for every unit its k_nn nearest same-dataset units
    cand = set()
    for u in range(U):
        same = np.where((udom == udom[u]) & (np.arange(U) != u))[0]
        if len(same) == 0:
            continue
        nn = same[np.argsort(Dc[u, same], kind="stable")[:k_nn]]
        for v in nn.tolist():
            cand.add((min(u, v), max(u, v)))
    keyc = np.fromiter(((a, b) in cand for a, b in zip(iu[0].tolist(), iu[1].tolist())), dtype=bool, count=len(iu[0]))
    ident = [set(gt[m].tolist()) for m in members]
    share = np.zeros(len(iu[0]), dtype=bool)
    for i in np.where(keyc)[0].tolist():
        share[i] = bool(ident[iu[0][i]] & ident[iu[1][i]])
    first = (keyc & share)[oB]
    o = np.concatenate([oB[first], oB[~first]])                          # D1.6 ORACLE order
    A_, B_ = iu[0][o].tolist(), iu[1][o].tolist()
    n_first = int(first.sum())
    # image layer from the existing answers, cluster layer over units (loop of sim_d3.simulate)
    N = len(labels)
    iuf = _UF(N)
    blocks = collections.defaultdict(set)
    for a, b in ml_pairs:
        if a != b:
            ra, rb = iuf.find(a), iuf.find(b)
            if ra != rb:
                iuf.p[rb] = ra
    for a, b in cl_pairs:
        ra, rb = iuf.find(a), iuf.find(b)
        if ra != rb:
            blocks[ra].add(rb)
            blocks[rb].add(ra)
    cuf = _UF(U)
    cblocks = collections.defaultdict(set)
    answers = []
    asked_from_first = free = skipped = 0
    for idx, (A, B) in enumerate(zip(A_, B_)):
        if len(answers) >= q:
            break
        ca, cb = cuf.find(A), cuf.find(B)
        if ca == cb:
            continue
        if cb in cblocks[ca]:
            skipped += 1
            continue
        a, b = int(medoid[A]), int(medoid[B])
        ra, rb = iuf.find(a), iuf.find(b)
        if ra == rb:                                                     # already implied 'same': free merge, no question
            cuf.p[cb] = ca
            free += 1
            for z in cblocks[cb]:
                cblocks[z].discard(cb)
                cblocks[z].add(ca)
                cblocks[ca].add(z)
            cblocks.pop(cb, None)
            continue
        if rb in blocks[ra]:                                             # already implied 'different'
            skipped += 1
            continue
        same = bool(gt[a] == gt[b])
        answers.append((a, b, same))
        if idx < n_first:
            asked_from_first += 1
        if same:
            iuf.p[rb] = ra
            blocks[ra] |= blocks.pop(rb, set())
            for z in blocks[ra]:
                blocks[z].discard(rb)
                blocks[z].add(ra)
            cuf.p[cb] = ca
            for z in cblocks[cb]:
                cblocks[z].discard(cb)
                cblocks[z].add(ca)
                cblocks[ca].add(z)
            cblocks.pop(cb, None)
        else:
            blocks[ra].add(rb)
            blocks[rb].add(ra)
            cblocks[ca].add(cb)
            cblocks[cb].add(ca)
    n_same = sum(1 for _, _, s in answers if s)
    by_ds = collections.defaultdict(lambda: [0, 0])
    for a, b, s in answers:
        by_ds[int(ds[a])][0 if s else 1] += 1
    stats = {"units_before": int(U), "candidate_pairs": len(cand), "candidate_pairs_sharing_identity": n_first,
             "asked": len(answers), "same": n_same, "different": len(answers) - n_same, "asked_from_sharing_candidates": asked_from_first,
             "free_merges_not_asked": free, "skipped_implied_different": skipped,
             "by_dataset_index": {str(k): {"same": v[0], "different": v[1]} for k, v in sorted(by_ds.items())}}
    return answers, stats


def add_answers(sampler, answers):
    """append the answers to the sampler's persistent lists (ConSam's own lists, so later cycles see them)"""
    for a, b, same in answers:
        (sampler.ml_pairs if same else sampler.cl_pairs).append((int(a), int(b)))
        sampler.all_selected_pairs.add(frozenset((int(a), int(b))))


def update_record(record, epoch, sampler, ml_pairs, cl_pairs, mstats, cap_log):
    """Fold the merge questions into this cycle's run_record entry (the one record.cycle() just appended) and print the final [record] line of the
    cycle, with the sha of the ML / CL sets that include the merge answers."""
    import copy
    from spcl.utils.run_record import _pairs_sha
    rec = record.d["cycles"][-1]
    assert rec["epoch"] == int(epoch)
    ml, cl = len(ml_pairs), len(cl_pairs)
    consumed = ml + cl
    consam_consumed = rec["consumed_total"]
    rec.update({"ml_total": ml, "cl_total": cl, "consumed_total": consumed, "consumed_this_cycle": consumed - (consam_consumed - rec["consumed_this_cycle"]),
                "ml_fraction_total": (ml / consumed) if consumed else None, "consumed_pct_of_all_pairs": 100.0 * consumed / record.total_edges,
                "ml_sha256": _pairs_sha(ml_pairs), "cl_sha256": _pairs_sha(cl_pairs), "stage": "after_merge",
                "consam_consumed_total_before_merge": consam_consumed, "merge": mstats, "human_pair_cap": cap_log})
    record._prev_consumed = consumed
    record.flush()
    print("[record]", json.dumps({k: v for k, v in rec.items() if k != "dataset_endpoints"}, default=str), flush=True)
