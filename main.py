from sampling.assc import MinMax, ExploreConsolidate
from sampling.counting_clusters import CountingClusters
from sampling.fps import FPS
from utils import *
import os
import re
import numpy as np
import json
import pandas as pd
from itertools import chain

from wildlife_tools.features import DeepFeatures
from wildlife_tools.train import ArcFaceLoss, SoftmaxLoss, TripletLoss
from wildlife_tools.train.callbacks import EpochCallbacks
from torch.optim import Adam

from dataset.wildlife import WildlifeDatasets
from sampling.balanced import LBAS, GBAS
from sampling.consistency import ConsistencySampling
from sampling.random import RandomSampling, RobustRandomSampling
import wandb


def run_one_cycle(
    args,
    sampler,
    dataset,
    megadescriptor_model,
    cycle,
    thresh=0.05,
    min_samples=3,
    metric="cosine",
    folder=f"pretrained_models/",
    checkpoint_folder=None,
    run_suffix="",
):
    logs = {}
    metadata = {}
    counts_per_gt_id = {}
    counts_per_pseudo_id = {}

    if not args.eval_only:  # Active Learning Training Cycle
        ml_pairs, cl_pairs = [], []
        database = dataset.get_test_loader()

        if checkpoint_folder is not None:
            loaded = load_checkpoint(megadescriptor_model, checkpoint_folder)
            if not loaded:
                print(
                    f"[Warning] Checkpoint not found. Using pre-trained {args.backbone}!"
                )

        megadescriptor_extractor = DeepFeatures(
            megadescriptor_model,
            device=args.device,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        database_features = megadescriptor_extractor(database)

        # Initialize the memory bank with pretrained features instead of randomized
        # sampler.set_memory_bank(database_features.features)

        # Do not sample for 0-th cycle (unsupervised training)
        if cycle > 0:
            ml_pairs, cl_pairs = sampler.select(features=database_features.features)
            metadata = {
                "ml_pairs": ml_pairs,
                "cl_pairs": cl_pairs,
            }

        # Update cluster labels using must-link and cannot-link constraints
        if args.sampler in ["MinMax", "EC", "CC"]:
            print(f"Estimated K using {args.sampler} Method:", int(sampler.K))
            logs.update({"Estimated K": sampler.K})

        print("Starting clustering...")
        pseudo_identity = constrained_clustering(
            X=database_features.features,
            method="ConSam",
            eps=thresh,
            min_samples=min_samples,
            metric=metric,
            ml_pairs=ml_pairs,
            cl_pairs=cl_pairs,
        )

        # CustomWildlifeDataset has implementation for loading the index of the images for updating the memory bank
        database_subset = CustomWildlifeDataset(
            metadata=database.metadata,
            root=dataset.image_root,
            transform=dataset.train_transform,
        )
        database_subset.metadata.loc[:, "original_index"] = (
            database_subset.metadata.index
        )
        database_subset.metadata.loc[:, "pseudo_identity"] = pseudo_identity
        # database_subset.metadata = database_subset.metadata[database_subset.metadata["pseudo_identity"] >= 0].reset_index(drop=True)

        print("Number of +ve Annotations:", len(ml_pairs))
        print("Number of -ve Annotations:", len(cl_pairs))
        print("Number of training samples:", database_subset.metadata.shape[0])
        print(
            "Number of pseudo-identities:",
            database_subset.metadata["pseudo_identity"].unique().shape[0],
        )
        print(
            "Number of gt-identities:",
            database_subset.metadata["identity"].unique().shape[0],
        )

        logs.update(
            {
                "Number of +ve Annotations": len(ml_pairs),
                "Number of -ve Annotations": len(cl_pairs),
                "Number of training samples": database_subset.metadata.shape[0],
                "Number of pseudo-identities": database_subset.metadata[
                    "pseudo_identity"
                ]
                .unique()
                .shape[0],
                "Number of gt-identities:": database_subset.metadata["identity"]
                .unique()
                .shape[0],
            }
        )

        pure = 0
        impure = 0
        impurity_ratio = []
        for _, group in database_subset.metadata.groupby("pseudo_identity"):
            counts = group.identity.value_counts().tolist()
            if len(counts) == 1:
                pure += 1
            else:
                impure += 1
                impurity_ratio.append(sum(counts[1:]) / sum(counts))
        print(f"{pure}/{impure}: {pure/(pure + impure)}")
        print(np.nanmean(impurity_ratio), "+/-", np.nanstd(impurity_ratio))
        # print(f"{pure},{impure},{np.nanmean(impurity_ratio)},{np.nanstd(impurity_ratio)},{pure/(pure + impure)}"); exit()
        logs.update({"Purity Ratio": pure / (pure + impure)})
        logs.update({"Impurity Ratio Mean": float(np.nanmean(impurity_ratio))})
        logs.update({"Impurity Ratio Std. Dev.": float(np.nanstd(impurity_ratio))})
        logs.update({"Pure": pure})
        logs.update({"Impure": impure})

        if not args.no_sweep:
            beta1, beta2, beta3 = (
                args.consam_beta1,
                args.consam_beta2,
                args.consam_beta3,
            )
            # normalize to probabilities
            beta_sum = beta1 + beta2 + beta3
            beta1 = beta1 / beta_sum
            beta2 = beta2 / beta_sum
            beta3 = beta3 / beta_sum

            config_dict = vars(args)
            config_dict.update(
                {
                    "beta1": beta1,
                    "beta2": beta2,
                    "beta3": beta3,
                }
            )
            wandb.init(
                entity="meharkhurana03",
                project="sweep-al-reid",
                name=f"d.{args.dataset}:{args.sampler}-{run_suffix} ({round(args.thresh, 2)}, [{round(args.consam_beta1, 2)},{round(args.consam_beta2, 2)},{round(args.consam_beta3, 2)}], {round(args.consam_eps, 2)}, {args.sample_seed}, {args.images_per_id})",
                tags=["wild10k-settings", "adaptive"],
                config=config_dict,
            )

            wandb.log(logs)

            exit()

        counts_per_gt_id = database_subset.metadata["identity"].value_counts().to_dict()
        counts_per_pseudo_id = (
            database_subset.metadata["pseudo_identity"].value_counts().to_dict()
        )

        database_subset.metadata.loc[:, "identity"] = database_subset.metadata.loc[
            :, "pseudo_identity"
        ]
        database_subset.labels, database_subset.labels_map = pd.factorize(
            database_subset.metadata.identity.values
        )

        # centroids = np.zeros((database_subset.metadata.identity.unique().shape[0], database_features.features.shape[1]))
        # for i, group in database_subset.metadata.groupby("identity"):
        #     centroids[i] = sampler.memory_bank[group.index.values].mean(axis=0)

        trainer, objective = get_trainer(
            megadescriptor_model=megadescriptor_model,
            train_database=database_subset,
            loss_criterion=args.loss_criterion,
            folder=folder,
            sampler=None,
            ml_pairs=ml_pairs,
            cl_pairs=cl_pairs,
            eps=thresh,
            min_samples=min_samples,
            metric=metric,
            centroids=None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            size=(224, 224),
        )

        train_logs = trainer.train()
        torch.save(objective.state_dict(), os.path.join(folder, f"objective.pth"))
        logs.update(
            {
                "counts_of_pseudo_ids": database_subset.metadata["identity"]
                .value_counts()
                .to_dict()
            }
        )
        logs.update(train_logs)
        # print("Counts of pseudo-ids:", database_subset.metadata["identity"].value_counts().to_dict())
    load_checkpoint(megadescriptor_model, folder)
    megadescriptor_model.eval()

    # Redefine database and query with test transformations
    megadescriptor_extractor = DeepFeatures(
        megadescriptor_model,
        device=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    results = get_evaluation_results(megadescriptor_extractor, dataset, folder)
    logs.update(results)
    return logs, metadata, counts_per_gt_id, counts_per_pseudo_id


def get_trainer(
    megadescriptor_model,
    train_database,
    loss_criterion="softmax",
    folder=None,
    sampler=None,
    ml_pairs=[],
    cl_pairs=[],
    eps=0.05,
    min_samples=5,
    metric="cosine",
    centroids=None,
    epochs=10,
    batch_size=64,
    num_workers=8,
    device="cuda",
    seed=0,
    size=(224, 224),
    extra_classes=0,
    update_pseudo_labels=True,
):
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, size[0], size[1])
        embedding_size = megadescriptor_model(dummy_input).shape[1]

    # FIXME(off-by-one): `extra_classes` exists only to support the skyline label modes --
    # see the SKYLINE_LABEL_MODE block below. num_classes here comes from
    # wildlife_tools' pd.factorize (0..N-1), but CustomTrainer.train_epoch (utils.py:1101)
    # overrides the batch labels with the `pseudo_identity` column, which main.py builds
    # with a +1. Set extra_classes=1 when keeping that offset, else the last identity
    # indexes past the classifier and nll_loss trips
    # `Assertion t >= 0 && t < n_classes failed`.
    n_classes = train_database.num_classes + extra_classes
    if loss_criterion == "softmax":
        objective = SoftmaxLoss(num_classes=n_classes, embedding_size=embedding_size)
    elif loss_criterion == "arcface":
        objective = ArcFaceLoss(
            num_classes=n_classes, embedding_size=embedding_size, margin=0.5, scale=64
        )
    elif loss_criterion == "triplet":
        objective = TripletLoss(margin=1.0, distance="cosine", mining="hard")
    else:
        raise NotImplementedError
    # if centroids is None:
    #     # objective = ClusterLoss(centroids=torch.rand((train_database.num_classes, embedding_size)), embedding_size=embedding_size)
    #     objective = SoftmaxLoss(num_classes=train_database.num_classes, embedding_size=embedding_size)
    # else:
    #     # objective = ClusterLoss(centroids=centroids, embedding_size=embedding_size)
    #     objective = SoftmaxLoss(num_classes=train_database.num_classes, embedding_size=embedding_size)

    # Optimizer and scheduler configuration
    params = chain(megadescriptor_model.parameters(), objective.parameters())
    optimizer = Adam(params=params, lr=1e-4)
    min_lr = 1e-6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr
    )

    trainer = CustomTrainer(
        dataset=train_database,
        model=megadescriptor_model,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=min(
            batch_size, train_database.metadata.shape[0] - 1
        ),  # because drop_last is true by default
        accumulation_steps=2,
        num_workers=num_workers,
        epochs=epochs,
        device=device,
        epoch_callback=EpochCallbacks(
            [EpochCheckpoint(folder=folder, save_step=1, seed=seed, verbose=True)]
        ),
        # sampler=sampler,
        ml_pairs=ml_pairs,
        cl_pairs=cl_pairs,
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        # MAKESHIFT(skyline): False disables the per-epoch re-clustering -- see utils.py
        update_pseudo_labels=update_pseudo_labels,
        # momentum=args.momentum
    )

    return trainer, objective


def main(args):
    dataset = WildlifeDatasets(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        dataset_split_seed=args.sample_seed,
        images_per_id=args.images_per_id,
        metric=args.metric,
    )
    database = dataset.get_train_loader()

    # Short tag for the output directory. The known backbones keep their historical tags
    # so existing result paths stay valid; anything else is slugified rather than rejected.
    # This used to be a whitelist ending in `raise NotImplementedError`, which failed before
    # the dataset was even built -- including for the ImageNet Swin models `get_model` itself
    # substitutes (Todo.md item 16).
    _KNOWN_SUFFIX = {
        "hf-hub:BVRA/MegaDescriptor-T-224": "mdt",
        "hf-hub:BVRA/MegaDescriptor-L-224": "mdl",
        "resnet50": "r50",
        "resnet18": "r18",
        "conservationxlabs/miewid-msv2": "miewid",
        "swin_large_patch4_window7_224": "swinL_in",
        "swin_tiny_patch4_window7_224": "swinT_in",
    }
    run_suffix = _KNOWN_SUFFIX.get(args.model_name)
    if run_suffix is None:
        run_suffix = re.sub(
            r"[^A-Za-z0-9._-]+", "_", args.model_name.split("/")[-1]
        ).strip("_")
        print(
            f"[run_suffix] unknown backbone {args.model_name!r} -> tag {run_suffix!r}"
        )

    use_combined_dataset = args.dataset == "combined"
    # NOTE(foundation-rows): get_model(combined=True) swaps MegaDescriptor-L-224 for a plain
    # ImageNet swin_large_patch4_window7_224 ("Do not use pre-trained megadescriptor",
    # utils.py:923). This looks like a bug that makes the zero-shot foundation rows of
    # Table 1 impossible to produce -- it is NOT. Measured over 4 gallery splits against
    # the paper's 45.65 mAP / 56.46 Top-1 foundation row:
    #     ImageNet Swin-L (this swap) : 44.46 mAP (-1.19), 56.62 Top-1 (+0.16)
    #     MegaDescriptor real weights : 53.34 mAP (+7.69), 67.77 Top-1 (+11.31)
    # with split noise of only +/-0.16-0.34 mAP. Table 1's "MegaDescriptor" row therefore
    # measures ImageNet Swin-L, and the substitution is what the published numbers went
    # through. LEAVE IT IN PLACE. A previous local change bypassed it for --baseline; that
    # was reverted once the above was measured. See README and Todo.md item 15.
    megadescriptor_model = get_model(
        args.model_name,
        eval=args.eval_only,
        combined=use_combined_dataset,
        drop_rate=0.2,
    )
    if args.skyline:
        # Skyline pretraining
        database = CustomWildlifeDataset(
            metadata=database.metadata,
            root=dataset.image_root,
            transform=dataset.train_transform,
        )
        database.metadata.loc[:, "original_index"] = database.metadata.index
        unique_ids = list(database.metadata.identity.unique())
        print(unique_ids)
        # FIXME(off-by-one): the original `+1` makes pseudo_identity run 1..N while
        # SoftmaxLoss gets num_classes=N from pd.factorize (0..N-1), so the last identity
        # indexes past the classifier and training dies ~150 batches in with
        #   nll_loss_forward: Assertion `t >= 0 && t < n_classes` failed.
        # The offset looks intended to reserve 0 for "unassigned", which matters in the AL
        # path where constrained_clustering emits -1 for outliers, but the skyline is fully
        # supervised so every image has a real identity and the offset only shifts labels.
        # Two ways to make it consistent, selected by SKYLINE_LABEL_MODE:
        #   zerobased (default) -> labels 0..N-1, num_classes = N      (smaller change)
        #   offset              -> labels 1..N,   num_classes = N + 1  (keeps a dead class 0)
        # database.metadata.loc[:, "pseudo_identity"] = [unique_ids.index(x)+1 for x in database.metadata.identity.values]
        _label_mode = os.environ.get("SKYLINE_LABEL_MODE", "zerobased")
        _offset = 1 if _label_mode == "offset" else 0
        print(
            f"[skyline] SKYLINE_LABEL_MODE={_label_mode} -> label offset {_offset}, "
            f"num_classes = {len(unique_ids)} + {_offset}"
        )
        database.metadata.loc[:, "pseudo_identity"] = [
            unique_ids.index(x) + _offset for x in database.metadata.identity.values
        ]
        print(database.metadata.pseudo_identity)
        trainer, objective = get_trainer(
            megadescriptor_model,
            database,
            loss_criterion=args.loss_criterion,
            folder=f"{args.out_dir}/pretrained_models/skyline/{args.dataset}/{run_suffix}",
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            size=(224, 224),
            extra_classes=_offset,
            # MAKESHIFT(skyline): keep the ground-truth labels
            # for all epochs; see utils.py train_epoch guard.
            update_pseudo_labels=False,
        )

        if args.eval_only:
            loaded = load_checkpoint(
                megadescriptor_model,
                f"{args.out_dir}/pretrained_models/skyline/{args.dataset}/{run_suffix}",
            )
            assert loaded is True, "Checkpoint not found!"
            print("Pretrained model loaded. Starting evaluation!")
        else:
            trainer.train()

            # save the objective head
            torch.save(
                objective.state_dict(),
                os.path.join(
                    f"{args.out_dir}/pretrained_models/skyline/{args.dataset}/{run_suffix}",
                    f"objective.pth",
                ),
            )

            # Load best saved model for evaluation
            load_checkpoint(
                megadescriptor_model,
                f"{args.out_dir}/pretrained_models/skyline/{args.dataset}/{run_suffix}",
            )
            megadescriptor_model.eval()

        # Redefine database and query with test transformations
        megadescriptor_extractor = DeepFeatures(
            megadescriptor_model,
            device=args.device,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )

        # Evaluation on the test set
        folder = os.path.join(
            args.out_dir, "pretrained_models", args.dataset, run_suffix, "skyline"
        )
        # FIXME(stale-signature): called with 3 args but utils.py:1367 defines
        #   get_evaluation_results(megadescriptor_extractor, dataset, folder, epoch)
        # -> TypeError: missing 1 required positional argument: 'epoch'. The --skyline path
        # cannot run as shipped. This is the THIRD site with the same stale call:
        # SpCL/examples/spcl_train_usl.py:162,268,295 and RTMem/examples/RTMem_train_usl.py:332,360
        # have it too -- `epoch` was added to the signature and only some callers updated.
        # Using -1 to match the sibling --baseline call at the bottom of this block.
        # results = get_evaluation_results(megadescriptor_extractor, dataset, folder)
        results = get_evaluation_results(megadescriptor_extractor, dataset, folder, -1)

    elif args.baseline:
        megadescriptor_extractor = DeepFeatures(
            megadescriptor_model,
            device=args.device,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        print("Pretrained model loaded. Starting evaluation!")

        folder = os.path.join(
            args.out_dir, "pretrained_models", args.dataset, run_suffix, "baseline"
        )
        results = get_evaluation_results(megadescriptor_extractor, dataset, folder, -1)
    else:
        n = database.metadata.identity.unique().shape[0]
        if args.sampler != "CC":
            folder = os.path.join(
                args.out_dir,
                f"pretrained_models",
                args.dataset,
                run_suffix,
                args.sampler,
            )
        else:
            folder = os.path.join(
                args.out_dir,
                f"pretrained_models",
                args.dataset,
                run_suffix,
                f"{args.sampler}_{args.sampling_ratio}",
            )

        os.makedirs(folder, exist_ok=True)
        sample_outliers = False

        # get representative gallery as dataframe
        gallery_df = dataset.default_gallery.metadata

        ml_pairs, cl_pairs = [], []  # these are dataset-wise indices
        # populate must-link and cannot-link pairs
        # print("Populating constraints...")
        # for i, group in tqdm(gallery_df.groupby("identity")):
        #     # other identity samples
        #     other_samples = gallery_df[gallery_df["identity"] != group.identity.values[0]]

        #     if len(group) > 1:
        #         ml_pairs += list(combinations(group.index, 2))

        #     if len(other_samples) > 0:
        #         # link with all other samples
        #         cl_pairs += list(product(group.index, other_samples.index))

        beta1, beta2, beta3 = args.consam_beta1, args.consam_beta2, args.consam_beta3
        # normalize to probabilities
        beta_sum = beta1 + beta2 + beta3
        beta1 = beta1 / beta_sum
        beta2 = beta2 / beta_sum
        beta3 = beta3 / beta_sum

        if args.sampler == "random":
            sampler = RandomSampling(database, n=n, gallery_size=args.gallery_size)
        elif args.sampler == "robust_random":
            sampler = RobustRandomSampling(
                database,
                n,
                thresh=args.thresh,
                gallery_size=args.gallery_size,
                min_samples=args.min_samples,
                metric=args.metric,
            )
        elif args.sampler == "FPS":
            sampler = FPS(
                database, n, gallery_size=args.gallery_size, metric=args.metric
            )
        elif args.sampler == "GBAS":
            sampler = GBAS(
                database,
                n,
                thresh=args.thresh,
                gallery_size=args.gallery_size,
                min_samples=args.min_samples,
                metric=args.metric,
                sample_outliers=sample_outliers,
            )
        elif args.sampler == "LBAS":
            sampler = LBAS(
                database,
                n,
                thresh=args.thresh,
                gallery_size=args.gallery_size,
                min_samples=args.min_samples,
                metric=args.metric,
                sample_outliers=sample_outliers,
            )
        elif args.sampler == "MinMax":
            sampler = MinMax(database, n, gallery_size=args.gallery_size)
        elif args.sampler == "EC":
            sampler = ExploreConsolidate(database, n, gallery_size=args.gallery_size)
        elif args.sampler == "CC":
            sampler = CountingClusters(
                database,
                n,
                gallery_size=args.gallery_size,
                metric=args.metric,
                ratio=args.sampling_ratio,
            )
        elif args.sampler == "ConSam":
            sampler = ConsistencySampling(
                database,
                n,
                betas=[beta1, beta2, beta3],
                eps=args.consam_eps,
                thresh=args.thresh,
                gallery_size=args.gallery_size,
                min_samples=args.min_samples,
                metric=args.metric,
                num_gt_ids_gallery=gallery_df["identity"].nunique(),
                adaptive=args.adaptive,
                lambda_=args.lambda_,
            )
        else:
            raise NotImplementedError

        sampler.set_constraints(ml_pairs, cl_pairs)

        all_logs = {}
        all_metadata = {}
        all_switch_logs = {}
        all_gt_id_counts = {}
        all_sampling_pools = {}

        if args.no_sweep:
            import wandb

            if not args.sampler == "CC":
                wandb.init(
                    entity="meharkhurana03",
                    project=args.dataset,
                    name=f"d.{args.sampler}-{run_suffix} ({args.thresh}, [{round(args.consam_beta1, 2)},{round(args.consam_beta2, 2)},{round(args.consam_beta3, 2)}], {args.consam_eps}, {args.sample_seed}, {args.images_per_id})",
                    tags=["wild10k-settings", "adaptive"],
                    config=vars(args),
                )
            else:
                wandb.init(
                    entity="meharkhurana03",
                    project=args.dataset,
                    name=f"d.{args.sampler}[{args.sampling_ratio}]-{run_suffix} ({args.thresh}, [{round(args.consam_beta1, 2)},{round(args.consam_beta2, 2)},{round(args.consam_beta3, 2)}], {args.consam_eps}, {args.sample_seed}, {args.images_per_id})",
                    tags=["wild10k-settings", "adaptive"],
                    config=vars(args),
                )
            wandb.define_metric("epoch")
            wandb.define_metric("train/loss_epoch_avg", step_metric="epoch")
            wandb.define_metric("train/acc_epoch_avg", step_metric="epoch")

            wandb.define_metric("budget")

        for budget in range(0, args.budget + 1):
            logs, metadata, counts_per_gt_id, counts_per_pseudo_id = run_one_cycle(
                args,
                sampler,
                dataset,
                megadescriptor_model,
                cycle=budget,
                thresh=args.thresh,
                min_samples=args.min_samples,
                metric=args.metric,
                folder=os.path.join(folder, f"{budget}n", str(args.thresh)),
                checkpoint_folder=os.path.join(
                    folder, f"{budget-1}n", str(args.thresh)
                ),
                run_suffix=run_suffix,
            )

            if args.sampler == "ConSam":
                beta1 = sampler.beta1
                beta2 = sampler.beta2
                beta3 = sampler.beta3
                consam_eps = sampler.eps
                mod_s = sampler.mod_s
                num_inliers = sampler.num_inliers
                num_outliers = sampler.num_outliers
                num_clusters_cdbscan = sampler.num_clusters_cdbscan
                count_inlier_inlier = sampler.count_inlier_inlier
                count_mixed_pairs = sampler.count_mixed_pairs
                count_outlier_outlier = sampler.count_outlier_outlier

                wandb.log(
                    {
                        "beta1_ada": beta1,
                        "beta2_ada": beta2,
                        "beta3_ada": beta3,
                        "consam_eps_ada": consam_eps,
                        "mod_s": mod_s,
                        "num_inliers": num_inliers,
                        "num_outliers": num_outliers,
                        "num_clusters_cdbscan": num_clusters_cdbscan,
                        "count_inlier_inlier": count_inlier_inlier,
                        "count_mixed_pairs": count_mixed_pairs,
                        "count_outlier_outlier": count_outlier_outlier,
                        "budget": budget,
                    }
                )

            if dataset.default_gallery is not None:
                if not args.eval_only:
                    wandb.log(
                        {
                            "Purity Ratio": logs["Purity Ratio"],
                            "Imurity Ratio Mean": logs["Impurity Ratio Mean"],
                            "Number of +ve Annotations": logs[
                                "Number of +ve Annotations"
                            ],
                            "Number of -ve Annotations": logs[
                                "Number of -ve Annotations"
                            ],
                            "Number of training samples": logs[
                                "Number of training samples"
                            ],
                            "Number of pseudo-identities": logs[
                                "Number of pseudo-identities"
                            ],
                            "Number of gt-identities:": logs[
                                "Number of gt-identities:"
                            ],
                            "budget": budget,
                        }
                    )

                    for i in range(args.epochs):
                        wandb.log(
                            {
                                "train/loss_epoch_avg": logs["train_loss_epoch_avg"][i],
                                "train/acc_epoch_avg": logs["train_acc_epoch_avg"][i],
                                "epochs": i + 1 + (args.epochs * (budget - 1)),
                            }
                        )
                if args.sampler == "CC":
                    wandb.log({"mod_s": logs["Estimated K"], "budget": budget})

                wandb.log(
                    {
                        "Metrics/Top-1 Accuracy": logs["Top-1 Accuracy"],
                        "Metrics/Top-3 Accuracy": logs["Top-3 Accuracy"],
                        "Metrics/Top-5 Accuracy": logs["Top-5 Accuracy"],
                        "Metrics/Top-10 Accuracy": logs["Top-10 Accuracy"],
                        "Metrics/mAP": logs["mAP"],
                        "Metrics/mINP": logs["mINP"],
                        "Metrics/BAKS": logs["BAKS"],
                        "Metrics/BAUS@0.7": logs["BAUS@0.7"],
                        "Metrics/Normalized Accuracy@0.7": logs[
                            "Normalized Accuracy@0.7"
                        ],
                        "Metrics/ROC AUC": logs["ROC AUC"],
                        "budget": budget,
                    }
                )
                print(
                    f"[{budget}/{args.budget}] Top-1 Accuracy: {logs['Top-1 Accuracy']}%"
                )
                print(
                    f"[{budget}/{args.budget}] Top-3 Accuracy: {logs['Top-3 Accuracy']}%"
                )
                print(
                    f"[{budget}/{args.budget}] Top-5 Accuracy: {logs['Top-5 Accuracy']}%"
                )
                print(
                    f"[{budget}/{args.budget}] Top-10 Accuracy: {logs['Top-10 Accuracy']}%"
                )
                print(f"[{budget}/{args.budget}] mAP: {logs['mAP']}")
                print(f"[{budget}/{args.budget}] mINP: {logs['mINP']}")
                print(f"[{budget}/{args.budget}] BAKS: {logs['BAKS']}")
                print(f"[{budget}/{args.budget}] BAUS@0.7: {logs['BAUS@0.7']}")
                print(
                    f"[{budget}/{args.budget}] Normalized Accuracy@0.7: {logs['Normalized Accuracy@0.7']}"
                )
                print(f"[{budget}/{args.budget}] ROC AUC: {logs['ROC AUC']}")

            all_logs[budget] = logs
            all_metadata[budget] = metadata
            all_gt_id_counts[budget] = {
                "gt_id": counts_per_gt_id,
                "pseudo_id": counts_per_pseudo_id,
            }
            if args.sampler in ["LBAS", "ConSam"]:
                all_switch_logs[budget] = sampler.switch_counts
                sampler.switch_counts = 0
            if args.sampler == "ConSam":
                all_sampling_pools[budget] = sampler.get_sampling_pools()

        with open(
            os.path.join(
                folder,
                f"logs_{args.thresh}_{args.seed}_{args.sample_seed}_{args.min_samples}_{args.num_splits}_{args.images_per_id}.json",
            ),
            "w",
        ) as fp:
            json.dump(all_logs, fp)
        with open(
            os.path.join(
                folder,
                f"gt_id_counts_{args.thresh}_{args.seed}_{args.sample_seed}_{args.min_samples}_{args.num_splits}_{args.images_per_id}.json",
            ),
            "w",
        ) as fp:
            json.dump(all_gt_id_counts, fp)
        with open(
            os.path.join(
                folder,
                f"switch_logs_{args.thresh}_{args.seed}_{args.sample_seed}_{args.min_samples}_{args.num_splits}_{args.images_per_id}.json",
            ),
            "w",
        ) as fp:
            json.dump(all_switch_logs, fp)
        with open(
            os.path.join(
                folder,
                f"sampling_pools_{args.thresh}_{args.seed}_{args.sample_seed}_{args.min_samples}_{args.num_splits}_{args.images_per_id}.json",
            ),
            "w",
        ) as fp:
            json.dump(all_sampling_pools, fp)
        if len(metadata.keys()) > 0:
            import pickle

            with open(
                os.path.join(
                    folder,
                    f"metadata_{args.thresh}_{args.seed}_{args.sample_seed}_{args.min_samples}_{args.num_splits}_{args.images_per_id}.pickle",
                ),
                "wb",
            ) as fp:
                pickle.dump(all_metadata, fp)


if __name__ == "__main__":
    args = get_main_args()
    if "resnet" in args.model_name:
        args.backbone = "resnet"
    set_seed(args.seed)
    main(args)
