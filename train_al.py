from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import random
import numpy as np
import sys
import collections
import json
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from spcl import datasets
from spcl import models
from spcl.models.hm import HybridMemory
from spcl.trainers import SpCLTrainer_USL
from spcl.evaluators import Evaluator, extract_features
from spcl.utils.constrained_clustering import constrained_clustering
import spcl.utils.constrained_clustering as _cc_module  # [canon-pairs]
from spcl.utils.data import IterLoader
from spcl.utils.data import transforms as T
from spcl.utils.data.sampler import RandomMultipleGallerySampler
from spcl.utils.data.preprocessor import Preprocessor
from spcl.utils.logging import Logger
from spcl.utils.serialization import load_checkpoint, save_checkpoint
from spcl.utils.faiss_rerank import compute_jaccard_distance

# [record] per-run experiment record (see spcl/utils/run_record.py)
from spcl.utils.run_record import RunRecord, dataset_of_train_index

from spcl.sampling.random import RandomSampling

# The AL samplers live in spcl.sampling so the repo-root `sampling` package cannot shadow them.
from spcl.sampling.balanced import LBAS, GBAS
from spcl.sampling.consistency import ConsistencySampling

# [new-arms] ablation samplers; neither touches the originals
from spcl.sampling.consistency_p1 import ConsistencySamplingP1
from spcl.sampling.simband import SimilarityBandSampling
from spcl.sampling.counting_clusters import CountingClusters

from utils import get_evaluation_results

# --deterministic, --stage-log, --stop-after-epoch, --train-seed, --loss-trace, --save-state,
# --resume. Every entry point is a no-op when its flag is off (see infra/).
from infra import determinism as _det
from infra import resume as _resume
from infra import state_saver as _save

from wildlife_tools.features import DeepFeatures


def get_data(name, data_dir, subset="combined", split_seed=0, partition_seed=102):
    root = osp.join(data_dir, name)
    # [--split-seed] Plumb the data-split seed through. It was previously dropped here,
    # so Wildlife fell back to its default of 0 on every run and --seed never reached
    # the split. Default 0 keeps existing behaviour byte-identical.
    # [--partition-seed] 102 = the labelled/unlabelled identity partition every
    # published run used; exposed so it can be varied deliberately, not by accident.
    dataset = datasets.create(
        name,
        root,
        subset=subset,
        dataset_split_seed=split_seed,
        partition_seed=partition_seed,
    )
    return dataset


def get_train_loader(
    args,
    dataset,
    height,
    width,
    batch_size,
    workers,
    num_instances,
    iters,
    trainset=None,
):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose(
        [
            T.Resize((height, width), interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            normalizer,
            T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
        ]
    )

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(
            Preprocessor(
                train_set, root=dataset.images_dir, transform=train_transformer
            ),
            batch_size=batch_size,
            num_workers=workers,
            sampler=sampler,
            shuffle=not rmgs_flag,
            pin_memory=True,
            drop_last=True,
            **_det.loader_kwargs(),  # {} unless --deterministic
        ),
        length=iters,
    )

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose(
        [T.Resize((height, width), interpolation=3), T.ToTensor(), normalizer]
    )

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        pin_memory=True,
    )

    return test_loader


def _ckpt(tag, arr):
    """Checksum a pipeline stage when HITL_DEBUG_CHECKSUM=1 (R6 bisect).

    Two identical runs still diverge at epoch 0 (2279 vs 2290 clusters) even sequentially
    on one GPU, while every stage tested in isolation is bit-identical. This prints a hash
    at each stage of the *real* pipeline so the first point of divergence can be located
    instead of reconstructed.
    """
    if os.environ.get("HITL_DEBUG_CHECKSUM", "").lower() not in ("1", "true", "yes"):
        return
    import hashlib
    import numpy as _np
    import torch as _torch

    a = (
        arr.detach().cpu().numpy()
        if isinstance(arr, _torch.Tensor)
        else _np.asarray(arr)
    )
    a = _np.ascontiguousarray(a)
    print(
        f"CKSUM {tag}: shape={a.shape} sha={hashlib.sha256(a.tobytes()).hexdigest()[:20]} "
        f"sum={float(a.sum()):.10g}",
        flush=True,
    )


def create_model(args):
    model = models.create(
        args.arch,
        num_features=args.features,
        norm=True,
        dropout=args.dropout,
        num_classes=0,
    )
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()
    # [al-schedule] reject schedules that would silently run fewer or no AL cycles
    if args.al_cycle_cap is not None and args.al_cycle_cap <= 0:
        parser.error("--al-cycle-cap must be a positive integer")
    if args.al_cycle_caps is not None:  # flag-isolated: only runs when the flag is given
        if args.al_cycle_cap is not None:
            parser.error("--al-cycle-caps and --al-cycle-cap are mutually exclusive")
        try:
            _caps = [int(x) for x in str(args.al_cycle_caps).split(",")]
        except ValueError:
            parser.error("--al-cycle-caps must be a comma list of integers")
        if any(c <= 0 for c in _caps):
            parser.error("--al-cycle-caps entries must be positive integers")
        _n_cycles = max(0, -(-(args.epochs - args.al_offset) // args.eval_step))
        if len(_caps) != _n_cycles:
            parser.error(
                f"--al-cycle-caps needs exactly one cap per AL cycle: {_n_cycles} for "
                f"--epochs {args.epochs} --al-offset {args.al_offset} --eval-step "
                f"{args.eval_step}, got {len(_caps)}")
    if not 0 <= args.al_offset < args.epochs:
        parser.error("--al-offset must satisfy 0 <= offset < --epochs")
    if args.stop_after_epoch is not None and not 0 <= args.stop_after_epoch < args.epochs:
        parser.error("--stop-after-epoch must satisfy 0 <= N < --epochs")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # R6: torch.manual_seed does not cover the CUDA RNGs used by dropout and any
        # GPU-side sampling; without this a "seeded" run was never fully seeded.
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    # Must stay above `cudnn.deterministic`: benchmark mode re-enables nondeterministic
    # algorithm selection and silently defeats it. Costs -0.3% on this workload.
    # HITL_CUDNN_BENCHMARK=1 restores autotuning.
    cudnn.benchmark = os.environ.get("HITL_CUDNN_BENCHMARK", "").lower() in (
        "1",
        "true",
        "yes",
    )

    sys.stdout = Logger(osp.join(args.logs_dir, "log.txt"))
    print("==========\nArgs:{}\n==========".format(args))
    _det.enable(args)  # no-op unless --deterministic / --stage-log
    _save.enable(args)  # no-op unless --save-state
    _det.stage("start")
    # [canon-pairs] process-wide switch, so the samplers' internal calls see it too
    _cc_module.CANONICAL_PAIRS = bool(args.canonical_pairs)
    if args.canonical_pairs:
        print("[canon-pairs] constrained_clustering canonicalises ML/CL pair sets")

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    # [--split-seed] forward the split seed (see get_data)
    dataset = get_data(
        args.dataset, args.data_dir, args.subset, args.split_seed, args.partition_seed
    )
    test_loader = get_test_loader(
        dataset, args.height, args.width, args.batch_size, args.workers
    )
    _det.stage("dataset_loaded")

    print(dataset.train[:20])

    # Active Learning
    if args.sampler == "random":
        sampler = RandomSampling(args=args)
    elif args.sampler == "GBAS":
        sampler = GBAS(
            thresh=args.eps,
            min_samples=4,
            metric="precomputed",
            sample_outliers=False,
            args=args,
        )
    elif args.sampler == "LBAS":
        sampler = LBAS(
            thresh=args.eps,
            min_samples=4,
            metric="precomputed",
            sample_outliers=False,
            args=args,
        )
    elif args.sampler == "CC":
        sampler = CountingClusters(ratio=1.0, args=args)
    elif args.sampler == "ConSam":
        sampler = ConsistencySampling(
            thresh=args.eps,
            min_samples=4,
            metric="precomputed",
            num_gt_ids_gallery=np.unique(np.array(dataset.gallery)[:, 1]).shape[0],
            args=args,
        )
    # [new-arms] AAS reading FINCH at a coarser partition
    elif args.sampler == "ConSamP1":
        sampler = ConsistencySamplingP1(
            thresh=args.eps,
            min_samples=4,
            metric="precomputed",
            num_gt_ids_gallery=np.unique(np.array(dataset.gallery)[:, 1]).shape[0],
            args=args,
        )
    # [new-arms] clustering-free similarity-band null model
    elif args.sampler == "SimBand":
        sampler = SimilarityBandSampling(args=args)
    elif args.sampler in (None, "None"):
        sampler = None
    else:
        raise ValueError(
            f"unknown --sampler {args.sampler!r}. Expected one of: "
            "random, GBAS, LBAS, CC (NIS), ConSam (AAS), ConSamP1, SimBand -- or omit "
            "-s entirely for the "
            "SpCL USL baseline. Falling through to sampler=None would silently run plain "
            "USL and write it into a directory named for an AL method."
        )

    gt_labels = None

    for cycle in range(1):
        start_epoch = best_mAP = 0

        # Create model
        model = create_model(args)

        # Create hybrid memory
        memory = HybridMemory(
            model.module.num_features,
            len(dataset.train),
            temp=args.temp,
            momentum=args.momentum,
        ).cuda()

        # Initialize target-domain instance features
        print("==> Initialize instance features in the hybrid memory")
        cluster_loader = get_test_loader(
            dataset,
            args.height,
            args.width,
            args.batch_size,
            args.workers,
            testset=sorted(dataset.train),
        )
        features, _ = extract_features(model, cluster_loader, print_freq=50)
        features = torch.cat(
            [features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0
        )
        memory.features = F.normalize(features, dim=1).cuda()
        _ckpt("memory_features", memory.features)
        _det.stage("epoch0_features")
        _save.features(memory.features)
        if gt_labels is None and sampler is not None:
            gt_labels = [pid for _, pid, _ in sorted(dataset.train)]
            sampler.set_gt_labels(gt_labels)
            # [al-schedule] set_gt_labels sizes the per-cycle budget; cap it here so the
            # sampler draws at most this many pairs and reports shortfall against it
            if args.al_cycle_cap is not None:
                sampler.n = min(sampler.n, args.al_cycle_cap)
                print(f"[al-cycle-cap] per-cycle budget = {sampler.n}")
        # [record] start the run record once the pool is fixed
        record = RunRecord(
            args.logs_dir,
            args,
            len(dataset.train),
            pair_dataset=dataset_of_train_index(dataset),
        )
        del cluster_loader, features

        # Evaluator
        evaluator = Evaluator(model)

        # Optimizer
        params = [
            {"params": [value]}
            for _, value in model.named_parameters()
            if value.requires_grad
        ]
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=0.1
        )

        # Trainer
        trainer = SpCLTrainer_USL(model, memory)
        _det.trace_losses(args, memory)  # no-op unless --loss-trace

        # [al-schedule] With --al-offset > 0 the epochs before the first cycle impose no
        # constraints (constrained_clustering returns labels unchanged for empty lists).
        # With the default offset of 0 these are overwritten at epoch 0 before any use.
        ml_pairs, cl_pairs = [], []

        # Resume. Without --resume this returns (0, 0, None, [], []), so the loop below
        # starts at epoch 0 and nothing else changes.
        start_epoch, best_mAP, indep_thres, ml_pairs, cl_pairs = _resume.maybe_load(
            args, model, memory, optimizer, lr_scheduler, sampler
        )

        for epoch in range(start_epoch, args.epochs):
            _save.set_epoch(epoch)
            if args.dataset == "wildlife":
                model.train()

            if not args.full_supervision:
                # Calculate distance
                print(
                    "==> Create pseudo labels for unlabeled data with self-paced policy"
                )
                features = memory.features.clone()
                rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
                _ckpt(f"epoch{epoch}_rerank", rerank_dist)
                _det.stage(f"epoch{epoch}_rerank")
                _save.rerank(epoch, rerank_dist)

                # [al-schedule] offset 0 reduces to the original `epoch % eval_step == 0`
                if (
                    sampler is not None
                    and epoch >= args.al_offset
                    and (epoch - args.al_offset) % args.eval_step == 0
                ):
                    if args.al_cycle_caps is not None:  # per-cycle budget cap
                        if not hasattr(sampler, "_base_n"):
                            sampler._base_n = sampler.n  # the sampler's own budget, before any cap
                        _cap_k = (epoch - args.al_offset) // args.eval_step
                        sampler.n = min(sampler._base_n,
                                        [int(x) for x in args.al_cycle_caps.split(",")][_cap_k])
                        print(f"[al-cycle-caps] cycle {_cap_k + 1} (epoch {epoch}): "
                              f"per-cycle budget = {sampler.n}", flush=True)
                    if args.sampler == "None":
                        ml_pairs, cl_pairs = [], []
                    else:
                        ml_pairs, cl_pairs = sampler.select(
                            features=features,
                            rerank_dist=rerank_dist,
                            gt_labels=gt_labels,
                        )
                    print("#Constraints:", len(ml_pairs) + len(cl_pairs))
                    print("#ML Constraints:", len(ml_pairs))
                    print("#CL Constraints:", len(cl_pairs))
                    # [record] persist this AL cycle
                    _r = record.cycle(epoch, sampler, ml_pairs, cl_pairs)
                    print(
                        "[record]",
                        json.dumps(
                            {k: v for k, v in _r.items() if k != "dataset_endpoints"}
                        ),
                    )
                    _det.stage(f"epoch{epoch}_al_select")
                    _save.cycle(epoch, sampler, ml_pairs, cl_pairs)

                # `or epoch == start_epoch` fires only on a resumed run, where the
                # cluster objects of epoch 0 are gone; start_epoch is 0 otherwise.
                if epoch == 0 or epoch == start_epoch:
                    # DBSCAN cluster
                    eps = args.eps
                    eps_tight = eps - args.eps_gap
                    eps_loose = eps + args.eps_gap
                    print(
                        "Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}".format(
                            eps, eps_tight, eps_loose
                        )
                    )
                    cluster = DBSCAN(eps=eps, min_samples=4, metric="precomputed")
                    cluster_tight = DBSCAN(
                        eps=eps_tight, min_samples=4, metric="precomputed"
                    )
                    cluster_loose = DBSCAN(
                        eps=eps_loose, min_samples=4, metric="precomputed"
                    )

                del features

                # select & cluster images as training set of this epochs
                pseudo_labels = cluster.fit_predict(rerank_dist)
                _ckpt(f"epoch{epoch}_labels", pseudo_labels)
                _save.dbscan_raw(epoch, pseudo_labels)
                pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
                pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)
                _det.stage(f"epoch{epoch}_dbscan")
                num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
                num_ids_tight = len(set(pseudo_labels_tight)) - (
                    1 if -1 in pseudo_labels_tight else 0
                )
                num_ids_loose = len(set(pseudo_labels_loose)) - (
                    1 if -1 in pseudo_labels_loose else 0
                )

                if sampler is not None:
                    purity_labels = pseudo_labels.copy()
                    purity_labels[purity_labels == -1] = np.arange(
                        purity_labels.max() + 1,
                        purity_labels.max() + 1 + sum(purity_labels == -1),
                    )
                    # compute purity:
                    cluster_dict = {}
                    impurity_ratio = []
                    for i, label in enumerate(purity_labels):
                        if label.item() not in cluster_dict:
                            cluster_dict[label.item()] = []
                        cluster_dict[label.item()].append(gt_labels[i])
                    cluster_purity = 0
                    pure_clusters = 0
                    for label, ids in cluster_dict.items():
                        if len(ids) > 0:
                            most_common = collections.Counter(ids).most_common(1)[0][1]
                            cluster_purity += most_common / len(ids)
                            if most_common == len(ids):
                                pure_clusters += 1
                            else:
                                impurity_ratio.append(
                                    (len(ids) - most_common) / len(ids)
                                )
                    impurity_ratio = (
                        np.mean(impurity_ratio) if len(impurity_ratio) > 0 else 0
                    )
                    cluster_purity /= len(cluster_dict)
                    print(purity_labels[:20])
                    print(
                        f"Cluster Purity: {cluster_purity:.4f} | \tPure Clusters: {pure_clusters}/{len(cluster_dict)} = {pure_clusters/len(cluster_dict):.4f} | \tImpurity Ratio: {impurity_ratio:.4f}"
                    )

                    purity_labels = pseudo_labels_tight.copy()
                    purity_labels[purity_labels == -1] = np.arange(
                        purity_labels.max() + 1,
                        purity_labels.max() + 1 + sum(purity_labels == -1),
                    )
                    # compute purity:
                    cluster_dict = {}
                    impurity_ratio = []
                    for i, label in enumerate(purity_labels):
                        if label.item() not in cluster_dict:
                            cluster_dict[label.item()] = []
                        cluster_dict[label.item()].append(gt_labels[i])
                    cluster_purity = 0
                    pure_clusters = 0
                    for label, ids in cluster_dict.items():
                        if len(ids) > 0:
                            most_common = collections.Counter(ids).most_common(1)[0][1]
                            cluster_purity += most_common / len(ids)
                            if most_common == len(ids):
                                pure_clusters += 1
                            else:
                                impurity_ratio.append(
                                    (len(ids) - most_common) / len(ids)
                                )
                    impurity_ratio = (
                        np.mean(impurity_ratio) if len(impurity_ratio) > 0 else 0
                    )
                    cluster_purity /= len(cluster_dict)
                    print(purity_labels[:20])
                    print(
                        f"Cluster Purity: {cluster_purity:.4f} | \tPure Clusters: {pure_clusters}/{len(cluster_dict)} = {pure_clusters/len(cluster_dict):.4f} | \tImpurity Ratio: {impurity_ratio:.4f}"
                    )

                    purity_labels = pseudo_labels_loose.copy()
                    purity_labels[purity_labels == -1] = np.arange(
                        purity_labels.max() + 1,
                        purity_labels.max() + 1 + sum(purity_labels == -1),
                    )
                    # compute purity:
                    cluster_dict = {}
                    impurity_ratio = []
                    for i, label in enumerate(purity_labels):
                        if label.item() not in cluster_dict:
                            cluster_dict[label.item()] = []
                        cluster_dict[label.item()].append(gt_labels[i])
                    cluster_purity = 0
                    pure_clusters = 0
                    for label, ids in cluster_dict.items():
                        if len(ids) > 0:
                            most_common = collections.Counter(ids).most_common(1)[0][1]
                            cluster_purity += most_common / len(ids)
                            if most_common == len(ids):
                                pure_clusters += 1
                            else:
                                impurity_ratio.append(
                                    (len(ids) - most_common) / len(ids)
                                )
                    impurity_ratio = (
                        np.mean(impurity_ratio) if len(impurity_ratio) > 0 else 0
                    )
                    cluster_purity /= len(cluster_dict)
                    print(purity_labels[:20])
                    print(
                        f"Cluster Purity: {cluster_purity:.4f} | \tPure Clusters: {pure_clusters}/{len(cluster_dict)} = {pure_clusters/len(cluster_dict):.4f} | \tImpurity Ratio: {impurity_ratio:.4f}"
                    )

                # generate new dataset and calculate cluster centers
                def generate_pseudo_labels(cluster_id, num):
                    labels = []
                    outliers = 0
                    for i, ((fname, _, cid), id) in enumerate(
                        zip(sorted(dataset.train), cluster_id)
                    ):
                        if id != -1:
                            labels.append(id)
                        else:
                            labels.append(num + outliers)
                            outliers += 1
                    return torch.Tensor(labels).long()

                pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
                pseudo_labels_tight = generate_pseudo_labels(
                    pseudo_labels_tight, num_ids_tight
                )
                pseudo_labels_loose = generate_pseudo_labels(
                    pseudo_labels_loose, num_ids_loose
                )

                # compute R_indep and R_comp
                N = pseudo_labels.size(0)
                label_sim = (
                    pseudo_labels.expand(N, N)
                    .eq(pseudo_labels.expand(N, N).t())
                    .float()
                )
                label_sim_tight = (
                    pseudo_labels_tight.expand(N, N)
                    .eq(pseudo_labels_tight.expand(N, N).t())
                    .float()
                )
                label_sim_loose = (
                    pseudo_labels_loose.expand(N, N)
                    .eq(pseudo_labels_loose.expand(N, N).t())
                    .float()
                )

                R_comp = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(
                    label_sim, label_sim_tight
                ).sum(-1)
                R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(
                    label_sim, label_sim_loose
                ).sum(-1)
                _ckpt(f"epoch{epoch}_Rcomp", R_comp)
                _ckpt(f"epoch{epoch}_Rindep", R_indep)
                assert (R_comp.min() >= 0) and (R_comp.max() <= 1)
                assert (R_indep.min() >= 0) and (R_indep.max() <= 1)

                cluster_R_comp, cluster_R_indep = collections.defaultdict(
                    list
                ), collections.defaultdict(list)
                cluster_img_num = collections.defaultdict(int)
                for i, (comp, indep, label) in enumerate(
                    zip(R_comp, R_indep, pseudo_labels)
                ):
                    cluster_R_comp[label.item()].append(comp.item())
                    cluster_R_indep[label.item()].append(indep.item())
                    cluster_img_num[label.item()] += 1

                cluster_R_comp = [
                    min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())
                ]
                cluster_R_indep = [
                    min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())
                ]
                _ckpt(f"epoch{epoch}_clusterRindep", np.array(cluster_R_indep))
                cluster_R_indep_noins = [
                    iou
                    for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys()))
                    if cluster_img_num[num] > 1
                ]
                if epoch == 0:
                    indep_thres = np.sort(cluster_R_indep_noins)[
                        min(
                            len(cluster_R_indep_noins) - 1,
                            np.round(len(cluster_R_indep_noins) * 0.9).astype("int"),
                        )
                    ]

                pseudo_labeled_dataset = []
                outliers = 0
                constrained_pseudo_labels = []
                for i, ((fname, _, cid), label) in enumerate(
                    zip(sorted(dataset.train), pseudo_labels)
                ):
                    indep_score = cluster_R_indep[label.item()]
                    comp_score = R_comp[i]
                    if (indep_score <= indep_thres) and (
                        comp_score.item() <= cluster_R_comp[label.item()]
                    ):
                        pseudo_labeled_dataset.append((fname, label.item(), cid))
                        constrained_pseudo_labels.append(label.item())
                    else:
                        pseudo_labeled_dataset.append(
                            (fname, len(cluster_R_indep) + outliers, cid)
                        )
                        pseudo_labels[i] = len(cluster_R_indep) + outliers
                        constrained_pseudo_labels.append(pseudo_labels[i])
                        outliers += 1

                # Impose ML and CL constraints on the final pseudo labels
                _ckpt(f"epoch{epoch}_preNP3", np.array(constrained_pseudo_labels))
                _save.labels_pre(epoch, np.array(constrained_pseudo_labels))
                if sampler is not None:
                    # ml_pairs/cl_pairs only exist on the sampler path
                    _ckpt(
                        f"epoch{epoch}_mlcl", np.array([len(ml_pairs), len(cl_pairs)])
                    )
                    constrained_pseudo_labels = constrained_clustering(
                        np.array(constrained_pseudo_labels),
                        rerank_dist,
                        "~DBSCAN",
                        min_samples=4,
                        ml_pairs=ml_pairs,
                        cl_pairs=cl_pairs,
                    )
                    _ckpt(f"epoch{epoch}_postNP3", np.array(constrained_pseudo_labels))
                    for i, label in enumerate(constrained_pseudo_labels):
                        fname, _, cid = pseudo_labeled_dataset[i]
                        pseudo_labeled_dataset[i] = fname, label, cid
                        pseudo_labels[i] = label

                # statistics of clusters and un-clustered instances
                index2label = collections.defaultdict(int)
                for label in pseudo_labels:
                    index2label[label.item()] += 1
                index2label = np.fromiter(index2label.values(), dtype=float)
                print(
                    "==> Statistics for epoch {}: {} clusters, {} un-clustered instances, R_indep threshold is {}".format(
                        epoch,
                        (index2label > 1).sum(),
                        (index2label == 1).sum(),
                        1 - indep_thres,
                    )
                )

                memory.labels = pseudo_labels.cuda()
                _det.stage(f"epoch{epoch}_labels")
                _save.labels_post(epoch, pseudo_labels)

                if sampler is not None:
                    purity_labels = pseudo_labels.cpu().numpy().copy()
                    purity_labels[purity_labels == -1] = np.arange(
                        purity_labels.max() + 1,
                        purity_labels.max() + 1 + sum(purity_labels == -1),
                    )
                    # compute purity:
                    cluster_dict = {}
                    impurity_ratio = []
                    for i, label in enumerate(purity_labels):
                        if label.item() not in cluster_dict:
                            cluster_dict[label.item()] = []
                        cluster_dict[label.item()].append(gt_labels[i])
                    cluster_purity = 0
                    pure_clusters = 0
                    for label, ids in cluster_dict.items():
                        if len(ids) > 0:
                            most_common = collections.Counter(ids).most_common(1)[0][1]
                            cluster_purity += most_common / len(ids)
                            if most_common == len(ids):
                                pure_clusters += 1
                            else:
                                impurity_ratio.append(
                                    (len(ids) - most_common) / len(ids)
                                )
                    impurity_ratio = (
                        np.mean(impurity_ratio) if len(impurity_ratio) > 0 else 0
                    )
                    cluster_purity /= len(cluster_dict)
                    print(purity_labels[:20])
                    print(
                        f"Final Cluster Purity: {cluster_purity:.4f} | \tPure Clusters: {pure_clusters}/{len(cluster_dict)} = {pure_clusters/len(cluster_dict):.4f} | \tImpurity Ratio: {impurity_ratio:.4f}"
                    )

            else:
                unique_ids = list(np.unique(np.array(dataset.train)[:, 1]))
                memory.labels = torch.tensor(
                    [unique_ids.index(pid) + 1 for _, pid, _ in sorted(dataset.train)]
                ).cuda()
                pseudo_labels = memory.labels.clone()
                pseudo_labeled_dataset = []
                for i, ((fname, _, cid), label) in enumerate(
                    zip(sorted(dataset.train), pseudo_labels)
                ):
                    pseudo_labeled_dataset.append((fname, label.item(), cid))

            # no-op unless --train-seed; fires once, before the first train loader
            _det.train_seed_before_first_step(args)
            train_loader = get_train_loader(
                args,
                dataset,
                args.height,
                args.width,
                args.batch_size,
                args.workers,
                args.num_instances,
                iters,
                trainset=pseudo_labeled_dataset,
            )

            train_loader.new_epoch()

            trainer.train(
                epoch,
                train_loader,
                optimizer,
                print_freq=args.print_freq,
                train_iters=len(train_loader),
            )
            # stage line, loss trace, --stop-after-epoch; no-op by default
            _det.after_train(args, epoch, model, memory)

            if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
                if args.dataset == "wildlife":
                    model.eval()
                    # num_workers must not be 0: eval decodes ~20,900 JPEGs and would
                    # run single-threaded while the GPU idles. shuffle=False, so this
                    # changes throughput only, not batch order.
                    megadescriptor_extractor = DeepFeatures(
                        model, device="cuda:0", num_workers=args.workers, batch_size=32
                    )
                    model = model.to("cuda:0")
                    folder = osp.join(
                        args.logs_dir, args.dataset, args.subset, "evaluation"
                    )
                    os.makedirs(folder, exist_ok=True)

                    results = get_evaluation_results(
                        megadescriptor_extractor, dataset, folder, epoch
                    )
                    # Redundant since get_evaluation_results restores the device itself,
                    # but harmless: wildlife_tools' DeepFeatures.__call__ ends with
                    # self.model.to("cpu") on this same object.
                    model = model.to("cuda:0")
                    mAP = results["mAP"]
                else:
                    mAP = evaluator.evaluate(
                        test_loader, dataset.query, dataset.gallery, cmc_flag=False
                    )
                is_best = mAP > best_mAP
                best_mAP = max(mAP, best_mAP)
                save_checkpoint(
                    {
                        "state_dict": model.state_dict(),
                        "epoch": epoch + 1,
                        "best_mAP": best_mAP,
                    },
                    is_best,
                    fpath=osp.join(args.logs_dir, f"checkpoint_{epoch+1}.pth.tar"),
                )

                import csv

                # os.makedirs(f'/home/mehar21541/SpCL/logs_al/{args.subset}', exist_ok=True)
                # with open(f'/home/mehar21541/SpCL/logs_al/{args.subset}/results.csv', 'a', newline='') as file:
                results_dir = osp.join(args.logs_dir, "logs_al", args.subset)
                os.makedirs(results_dir, exist_ok=True)
                with open(
                    osp.join(results_dir, "results.csv"), "a", newline=""
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [args.sampler if sampler is not None else "USL", epoch, mAP]
                    )

                print(
                    "\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n".format(
                        epoch, mAP, best_mAP, " *" if is_best else ""
                    )
                )
                _det.stage(f"epoch{epoch}_eval")

            lr_scheduler.step()

            # Both no-ops unless --resume-save-epochs / --exit-after-epoch
            _resume.save(args, epoch, model, memory, optimizer, lr_scheduler, sampler,
                         best_mAP, indep_thres, ml_pairs, cl_pairs)
            _resume.exit_after(args, epoch)

        print("==> Test with the best model:")
        checkpoint = load_checkpoint(osp.join(args.logs_dir, "model_best.pth.tar"))
        model.load_state_dict(checkpoint["state_dict"])
        if args.dataset == "wildlife":
            model.eval()
            # O1: see the comment at the per-cycle eval above.
            megadescriptor_extractor = DeepFeatures(
                model, device="cuda:0", num_workers=args.workers, batch_size=32
            )
            model = model.to("cuda:0")
            folder = osp.join(args.logs_dir, args.dataset, args.subset, "evaluation")
            os.makedirs(folder, exist_ok=True)

            results = get_evaluation_results(
                megadescriptor_extractor, dataset, folder, args.epochs + 1
            )
            mAP = results["mAP"]
        else:
            evaluator.evaluate(
                test_loader, dataset.query, dataset.gallery, cmc_flag=True
            )

    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))
    # [record] close out the run record
    try:
        record.finish("ok")
    except NameError:
        pass  # USL path builds no record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Self-paced contrastive learning on unsupervised re-ID"
    )
    # data
    parser.add_argument(
        "-d", "--dataset", type=str, default="wildlife", choices=datasets.names()
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--height", type=int, default=256, help="input height")
    parser.add_argument("--width", type=int, default=128, help="input width")
    parser.add_argument(
        "--num-instances",
        type=int,
        default=4,
        help="each minibatch consist of "
        "(batch_size // num_instances) identities, and "
        "each identity has num_instances instances, "
        "default: 0 (NOT USE)",
    )
    parser.add_argument(
        "--subset", type=str, default="combined", help="wildlife dataset to use"
    )
    # cluster
    parser.add_argument(
        "--eps", type=float, default=0.6, help="max neighbor distance for DBSCAN"
    )
    parser.add_argument(
        "--eps-gap",
        type=float,
        default=0.02,
        help="multi-scale criterion for measuring cluster reliability",
    )
    parser.add_argument(
        "--k1", type=int, default=30, help="hyperparameter for jaccard distance"
    )
    parser.add_argument(
        "--k2", type=int, default=6, help="hyperparameter for jaccard distance"
    )
    # model
    parser.add_argument(
        "-a", "--arch", type=str, default="resnet50", choices=models.names()
    )
    parser.add_argument("--features", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.2,
        help="update momentum for the hybrid memory",
    )
    # optimizer
    parser.add_argument("--lr", type=float, default=0.00035, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--step-size", type=int, default=20)
    # training configs
    parser.add_argument("--seed", type=int, default=1)
    # [--split-seed] Selects the data split (representative-gallery draw). Separate from
    # --seed, which is overwritten by Wildlife.__init__'s set_seed() before it reaches
    # anything. Default 0 = the split every published run used.
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="dataset_split_seed for the train/gallery split (default 0)",
    )
    # [--partition-seed] Pins the labelled/unlabelled identity partition
    # (spcl/datasets/wildlife.py:480). Default 102 = every published run.
    parser.add_argument(
        "--partition-seed",
        type=int,
        default=102,
        help="seed for the labelled/unlabelled identity partition (default 102)",
    )
    parser.add_argument("--print-freq", type=int, default=200)
    parser.add_argument("--eval-step", type=int, default=10)
    parser.add_argument(
        "--temp",
        type=float,
        default=0.05,
        help="temperature for scaling contrastive loss",
    )
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument(
        "--data-dir", type=str, metavar="PATH", default=osp.join(working_dir, "data")
    )
    parser.add_argument(
        "--logs-dir", type=str, metavar="PATH", default=osp.join(working_dir, "logs")
    )

    # Active Learning
    parser.add_argument("-s", "--sampler", type=str, default=None)
    parser.add_argument(
        "--full-supervision",
        action="store_true",
        help="use full supervision for training",
    )
    parser.add_argument(
        "--consam_eps",
        type=float,
        default=0.6,
        help="max neighbor distance for Consistency Sampling",
    )
    parser.add_argument(
        "--consam_sim_thresh",
        type=float,
        default=0.3,
        help="similarity threshold for Consistency Sampling",
    )
    parser.add_argument(
        "--consam_k",
        type=int,
        default=5,
        help="number of neighbor medoids for Consistency Sampling",
    )
    # [canon-pairs] default False reproduces the published (bug-present) behaviour
    parser.add_argument(
        "--canonical-pairs",
        action="store_true",
        default=False,
        help="canonicalise ML/CL pairs to sorted tuples inside constrained_clustering, "
        "so pairs stored (hi, lo) are no longer dropped by the cannot-link repair",
    )
    # [al-schedule] AL schedule knobs. All defaults reproduce the published schedule exactly.
    parser.add_argument(
        "--al-cycle-cap",
        type=int,
        default=None,
        help="max pairs queried per AL cycle: the sampler plans with "
        "min(budget, cap). Default: unlimited",
    )
    parser.add_argument(
        "--al-cycle-caps",
        type=str,
        default=None,
        help="per-cycle version of --al-cycle-cap: a comma list with exactly one cap "
        "per AL cycle (e.g. 10270,5794,2968,2042,1636 for --epochs 50 --eval-step 10). "
        "Cycle k plans with min(budget, caps[k]). Mutually exclusive with --al-cycle-cap. "
        "Default OFF: nothing changes.",
    )
    parser.add_argument(
        "--al-offset",
        type=int,
        default=0,
        help="epoch of the first AL cycle; later cycles follow every --eval-step "
        "epochs (default 0)",
    )
    # Determinism and instrumentation; all default OFF (infra/determinism.py)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="cudnn.deterministic, no benchmark, use_deterministic_algorithms(warn_only), "
        "seeded train-loader generator + worker_init_fn; needs CUBLAS_WORKSPACE_CONFIG",
    )
    parser.add_argument(
        "--stage-log",
        action="store_true",
        default=False,
        help="print wall clock and CUDA max memory at each pipeline stage",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="re-seed all RNGs (and the --deterministic loader generator) right before the "
        "first training step; split and epoch-0 path unchanged (default: off)",
    )
    parser.add_argument(
        "--loss-trace",
        action="store_true",
        default=False,
        help="save every iteration's loss at full precision to <logs-dir>/loss_trace_epoch<e>.npy",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help="after this epoch's training step, save model + memory bank to "
        "<logs-dir>/stop_epoch<N>.pth and exit (default: run all epochs)",
    )
    parser.add_argument(
        "--save-state",
        type=str,
        default=None,
        metavar="DIR",
        help="save pseudo-labels, AL pairs/answers and the sampler's own views to DIR",
    )
    parser.add_argument(
        "--save-rerank-epochs",
        type=str,
        default="0",
        help="which epochs' rerank matrices --save-state writes (comma-separated; ~1.6 GB each)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="resume from a file written by --resume-save-epochs",
    )
    parser.add_argument(
        "--resume-save-epochs",
        type=str,
        default=None,
        help="write <logs-dir>/resume_epoch<e>.pth after each of these epochs (comma-separated)",
    )
    parser.add_argument(
        "--exit-after-epoch",
        type=int,
        default=None,
        help="exit at the END of this epoch (after evaluation, LR step and resume save)",
    )
    # [new-arms] flags for the two ablation samplers
    parser.add_argument(
        "--consam-finch-partition",
        type=int,
        default=1,
        help="FINCH partition for -s ConSamP1 (0 = the original sampler's behaviour)",
    )
    parser.add_argument(
        "--simband-lo",
        type=float,
        default=0.0,
        help="lower similarity bound for -s SimBand (calibrated, see report)",
    )
    parser.add_argument(
        "--simband-hi",
        type=float,
        default=1.0,
        help="upper similarity bound for -s SimBand",
    )
    main()
