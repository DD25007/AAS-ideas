from sampling.assc import MinMax, ExploreConsolidate
from sampling.counting_clusters import CountingClusters
from sampling.fps import FPS
from utils import *
import evaluation
import os
import numpy as np
import json
import pandas as pd
from itertools import chain

from wildlife_tools.features import DeepFeatures
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.train import SoftmaxLoss, TripletLoss, BasicTrainer
from wildlife_tools.train.callbacks import EpochCallbacks
from wildlife_tools.inference import KnnClassifier
from torch.optim import SGD

from dataset.wildlife import WildlifeDatasets
from sampling.balanced import LBAS, GBAS
from sampling.consistency import ConsistencySampling
from sampling.random import RandomSampling, RobustRandomSampling


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
):
    logs = {}
    metadata = {}
    counts_per_gt_id = {}
    counts_per_pseudo_id = {}

    # Active Learning Training Cycle
    if not args.eval_only:
        ml_pairs, cl_pairs = [], []
        # Load the training dataset with test transformations
        database = dataset.get_test_loader()

        if checkpoint_folder is not None:
            loaded = load_checkpoint(megadescriptor_model, checkpoint_folder)
            if not loaded:
                print(
                    "[Warning] Checkpoint not found. Using pre-trained MegaDescriptor!"
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

        # Do not sample for 0-th cycle (train using only pseudo labels)
        if cycle > 0:
            ml_pairs, cl_pairs = sampler.select(database_features.features)
            metadata = {
                "ml_pairs": ml_pairs,
                "cl_pairs": cl_pairs,
            }

        # Update cluster labels using must-link and cannot-link constraints
        if args.sampler in ["MinMax", "EC", "CC"]:
            #     from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
            print(f"Estimated K using {args.sampler} Method:", int(sampler.K))
            logs.update({"Estimated K": sampler.K})
        #     clusterer = PCKMeans(n_clusters=int(sampler.K))
        #     clusterer.fit(database_features.features, ml=ml_pairs, cl=cl_pairs)
        #     pseudo_identity = clusterer.labels_
        # else:
        pseudo_identity = constrained_clustering(
            X=database_features.features,
            method="DBSCAN",
            eps=thresh,
            min_samples=min_samples,
            metric=metric,
            ml_pairs=ml_pairs,
            cl_pairs=cl_pairs,
            sampler=args.sampler,
        )

        # CustomWildlifeDataset has implementation for loading the index of the images for updating the memory bank
        database_subset = WildlifeDataset(
            metadata=database.metadata,
            root=dataset.image_root,
            transform=dataset.train_transform,
        )
        database_subset.metadata.loc[:, "original_index"] = (
            database_subset.metadata.index
        )
        database_subset.metadata.loc[:, "pseudo_identity"] = pseudo_identity
        database_subset.metadata = database_subset.metadata[
            database_subset.metadata["pseudo_identity"] >= 0
        ].reset_index(drop=True)

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
        logs.update({"Purity Ratio": pure / (pure + impure)})
        logs.update({"Impurity Ratio Mean": np.nanmean(impurity_ratio)})
        logs.update({"Impurity Ratio Std. Dev.": np.nanstd(impurity_ratio)})
        logs.update({"Pure": pure})
        logs.update({"Impure": impure})

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
            val_database=None,
            folder=folder,
            sampler=None,
            centroids=None,
            custom_trainer=False,
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

    if dataset.closed_query is not None:
        # Closed set evaluation
        closed_gallery = dataset.get_test_loader()  # train set is used as gallery
        closed_gallery_features = megadescriptor_extractor(closed_gallery)
        closed_query = dataset.closed_query
        closed_query_features = megadescriptor_extractor(closed_query)

        closed_similarity = dataset.matcher(
            query=closed_query_features, database=closed_gallery_features
        )
        closed_cmc, closed_all_AP, closed_all_INP = evaluation.eval_wildlife(
            1 - closed_similarity,
            closed_query.metadata.identity.values,
            closed_gallery.metadata.identity.values,
            max_rank=10,
        )
        closed_mAP = np.mean(closed_all_AP)
        closed_mINP = np.mean(closed_all_INP)
        closed_top1_acc = closed_cmc[0] * 100
        closed_top3_acc = closed_cmc[2] * 100
        closed_top5_acc = closed_cmc[4] * 100
        closed_top10_acc = closed_cmc[-1] * 100

        logs.update(
            {
                "Closed": {
                    "Top-1 Accuracy": closed_top1_acc,
                    "Top-3 Accuracy": closed_top3_acc,
                    "Top-5 Accuracy": closed_top5_acc,
                    "Top-10 Accuracy": closed_top10_acc,
                    "AP": closed_mAP,
                    "INP": closed_mINP,
                }
            }
        )

        print(
            f"[Closed] Top-1 Accuracy: {closed_top1_acc}% +/- {np.std(closed_top1_acc)}%"
        )
        print(
            f"[Closed] Top-3 Accuracy: {closed_top3_acc}% +/- {np.std(closed_top3_acc)}%"
        )
        print(
            f"[Closed] Top-5 Accuracy: {closed_top5_acc}% +/- {np.std(closed_top5_acc)}%"
        )
        print(
            f"[Closed] Top-10 Accuracy: {closed_top10_acc}% +/- {np.std(closed_top10_acc)}%"
        )
        print(f"[Closed] AP: {closed_mAP} +/- {np.std(closed_mAP)}")
        print(f"[Closed] INP: {closed_mINP} +/- {np.std(closed_mINP)}")

    # Disjoint set evaluation
    # if args.dataset != "SeaStarReID2023":
    # Ks = [1, 2, 5]
    # else:
    Ks = [1, 5]
    top1_accuracies_by_K = {}
    top3_accuracies_by_K = {}
    top5_accuracies_by_K = {}
    top10_accuracies_by_K = {}
    knn_accs_by_K = {}
    predictions_by_K = {}
    scores_by_K = {}
    AP_by_K = {}
    INP_by_K = {}
    for K in Ks:
        top1_accuracies_by_K[K] = []
        top3_accuracies_by_K[K] = []
        top5_accuracies_by_K[K] = []
        top10_accuracies_by_K[K] = []
        knn_accs_by_K[K] = []
        predictions_by_K[K] = []
        scores_by_K[K] = []
        AP_by_K[K] = []
        INP_by_K[K] = []
        for test_gallery, test_query in zip(
            dataset.test_galleries_by_K[K], dataset.test_queries_by_K[K]
        ):
            gallery_features = megadescriptor_extractor(test_gallery)
            query_features = megadescriptor_extractor(test_query)

            similarity = dataset.matcher(
                query=query_features, database=gallery_features
            )

            test_labels = test_gallery.metadata["identity"].astype(str).values
            knn_classifier = KnnClassifier(
                k=K, database_labels=test_labels, return_scores=True
            )
            preds, scores = knn_classifier(similarity)
            predictions_by_K[K].append(preds.tolist())
            scores_by_K[K].append(scores.tolist())
            # preds_top3 = top3_classifier(similarity)
            # preds_top5 = top5_classifier(similarity)

            top_1_acc = sum(preds == test_query.labels_string) / len(preds)
            knn_accs_by_K[K].append((top_1_acc * 100).tolist())
            # top_3_acc = np.mean(np.any(preds_top3 == test_query.labels_string[:, None], axis=1))
            # top_5_acc = np.mean(np.any(preds_top5 == test_query.labels_string[:, None], axis=1))
            # top1_accuracies.append(top_1_acc * 100)
            # top3_accuracies.append(top_3_acc * 100)
            # top5_accuracies.append(top_5_acc * 100)
            # AP.append(average_precision_score(test_query.metadata.identity, similarity))

            cmc, all_AP, all_INP = evaluation.eval_wildlife(
                1 - similarity,
                test_query.metadata.identity.values,
                test_gallery.metadata.identity.values,
                max_rank=10,
            )

            mAP = np.mean(all_AP)
            mINP = np.mean(all_INP)

            top1_accuracies_by_K[K].append(cmc[0] * 100)
            top3_accuracies_by_K[K].append(cmc[2] * 100)
            top5_accuracies_by_K[K].append(cmc[4] * 100)
            top10_accuracies_by_K[K].append(cmc[-1] * 100)
            AP_by_K[K].append(mAP)
            INP_by_K[K].append(mINP)

        logs.update(
            {
                K: {
                    "Mean": {
                        "Top-1 Accuracy": (np.mean(top1_accuracies_by_K[K])),
                        "Top-3 Accuracy": (np.mean(top3_accuracies_by_K[K])),
                        "Top-5 Accuracy": (np.mean(top5_accuracies_by_K[K])),
                        "Top-10 Accuracy": (np.mean(top10_accuracies_by_K[K])),
                        "AP": (np.mean(AP_by_K[K])),
                        "INP": (np.mean(INP_by_K[K])),
                    },
                    "Std. Dev.": {
                        "Top-1 Accuracy": (np.std(top1_accuracies_by_K[K])),
                        "Top-3 Accuracy": (np.std(top3_accuracies_by_K[K])),
                        "Top-5 Accuracy": (np.std(top5_accuracies_by_K[K])),
                        "Top-10 Accuracy": (np.std(top10_accuracies_by_K[K])),
                        "AP": (np.std(AP_by_K[K])),
                        "INP": (np.std(INP_by_K[K])),
                    },
                    "KNN": {
                        "scores": scores_by_K[K],
                        "predictions": predictions_by_K[K],
                        "Top-1 Accuracy Mean": (np.mean(knn_accs_by_K[K])),
                        "Top-1 Accuracy Std. Dev.": (np.std(knn_accs_by_K[K])),
                        "Top-1 Accuracy": knn_accs_by_K[K],
                    },
                    "All": {
                        s: {
                            "Top-1 Accuracy": (top1_acc, 2),
                            "Top-3 Accuracy": (top3_acc, 2),
                            "Top-5 Accuracy": (top5_acc, 2),
                            "Top-10 Accuracy": (top10_acc, 2),
                            "AP": (ap, 2),
                            "INP": (inp, 2),
                        }
                        for s, (
                            top1_acc,
                            top3_acc,
                            top5_acc,
                            top10_acc,
                            ap,
                            inp,
                        ) in enumerate(
                            zip(
                                top1_accuracies_by_K[K],
                                top3_accuracies_by_K[K],
                                top5_accuracies_by_K[K],
                                top10_accuracies_by_K[K],
                                AP_by_K[K],
                                INP_by_K[K],
                            )
                        )
                    },
                }
            }
        )
    # print(predictions_by_K)
    # print(scores_by_K)
    return logs, metadata, counts_per_gt_id, counts_per_pseudo_id


def get_trainer(
    megadescriptor_model,
    train_database,
    val_database,
    folder,
    sampler=None,
    centroids=None,
    custom_trainer=False,
    epochs=10,
    batch_size=64,
    num_workers=8,
    device="cuda",
    seed=0,
    size=(224, 224),
):
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, size[0], size[1])
        embedding_size = megadescriptor_model(dummy_input).shape[1]

    loss_type = "softmax"
    # objective = ArcFaceLoss(num_classes=train_database.num_classes, embedding_size=embedding_size, margin=0.5, scale=64)
    if loss_type == "softmax":
        objective = SoftmaxLoss(
            num_classes=train_database.num_classes, embedding_size=embedding_size
        )
    elif loss_type == "triplet":
        objective = TripletLoss(margin=0.2, distance="cosine", mining="semihard")
    # if centroids is None:
    #     # objective = ClusterLoss(centroids=torch.rand((train_database.num_classes, embedding_size)), embedding_size=embedding_size)
    #     objective = SoftmaxLoss(num_classes=train_database.num_classes, embedding_size=embedding_size)
    # else:
    #     # objective = ClusterLoss(centroids=centroids, embedding_size=embedding_size)
    #     objective = SoftmaxLoss(num_classes=train_database.num_classes, embedding_size=embedding_size)

    # Optimizer and scheduler configuration
    params = chain(megadescriptor_model.parameters(), objective.parameters())
    optimizer = SGD(params=params, lr=0.001, momentum=0.9)
    min_lr = optimizer.defaults.get("lr") * 1e-3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=min_lr
    )

    # Use the custom trainer. It contains implementation for updating memory bank and centroids after each batch
    if custom_trainer:
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
                [
                    EpochCheckpoint(
                        val_database=val_database,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        folder=folder,
                        save_step=1,
                        seed=seed,
                        verbose=True,
                    )
                ]
            ),
            sampler=sampler,
            momentum=args.momentum,
        )
    else:
        trainer = BasicTrainer(
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
                [
                    EpochCheckpoint(
                        val_database=val_database,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        folder=folder,
                        save_step=1,
                        seed=seed,
                        verbose=True,
                    )
                ]
            ),
        )

    return trainer, objective


def main(args):
    dataset = WildlifeDatasets(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        seed=args.seed,
        num_splits=args.num_splits,
        images_per_id=args.images_per_id,
    )
    database, train_database, val_database = dataset.get_train_loader()

    if args.skyline:
        megadescriptor_model = get_model(args.model_name, eval=args.eval_only)
        trainer = get_trainer(
            megadescriptor_model,
            database,
            None,
            folder=f"{args.out_dir}/pretrained_models/skyline/{args.dataset}/{args.backbone}",
            custom_trainer=False,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            size=(224, 224),
        )

        if args.eval_only:
            loaded = load_checkpoint(
                megadescriptor_model,
                f"{args.out_dir}/pretrained_models/skyline/{args.dataset}/{args.backbone}",
            )
            assert loaded is True, "Checkpoint not found!"
            print("Pretrained model loaded. Starting evaluation!")
        else:
            trainer.train()

            # Load best saved model for evaluation
            load_checkpoint(
                megadescriptor_model,
                f"{args.out_dir}/pretrained_models/skyline/{args.dataset}/{args.backbone}",
            )
            megadescriptor_model.eval()

        # Redefine database and query with test transformations
        megadescriptor_extractor = DeepFeatures(
            megadescriptor_model,
            device=args.device,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        top1_accuracies_by_K = {}
        top3_accuracies_by_K = {}
        top5_accuracies_by_K = {}
        top10_accuracies_by_K = {}
        AP_by_K = {}
        INP_by_K = {}

        # if args.dataset != "SeaStarReID2023":
        #     Ks = [1, 2, 5]
        # else:
        Ks = [1, 5]
        for K in Ks:
            top1_accuracies_by_K[K] = []
            top3_accuracies_by_K[K] = []
            top5_accuracies_by_K[K] = []
            top10_accuracies_by_K[K] = []
            AP_by_K[K] = []
            INP_by_K[K] = []
            for test_gallery, test_query in zip(
                dataset.test_galleries_by_K[K], dataset.test_queries_by_K[K]
            ):
                gallery_features = megadescriptor_extractor(test_gallery)
                query_features = megadescriptor_extractor(test_query)

                similarity = dataset.matcher(
                    query=query_features, database=gallery_features
                )
                # preds = top1_classifier(similarity)
                # preds_top3 = top3_classifier(similarity)
                # preds_top5 = top5_classifier(similarity)
                # print(len(preds.unique()), len(test_query.labels_string.unique()))

                # top_1_acc = sum(preds == test_query.labels_string) / len(preds)
                # top_3_acc = np.mean(np.any(preds_top3 == test_query.labels_string[:, None], axis=1))
                # top_5_acc = np.mean(np.any(preds_top5 == test_query.labels_string[:, None], axis=1))
                # top1_accuracies.append(top_1_acc * 100)
                # top3_accuracies.append(top_3_acc * 100)
                # top5_accuracies.append(top_5_acc * 100)
                # AP.append(average_precision_score(test_query.metadata.identity, similarity))

                cmc, all_AP, all_INP = evaluation.eval_wildlife(
                    1 - similarity,
                    test_query.metadata.identity.values,
                    test_gallery.metadata.identity.values,
                    max_rank=10,
                )

                mAP = np.mean(all_AP)
                mINP = np.mean(all_INP)

                top1_accuracies_by_K[K].append(cmc[0] * 100)
                top3_accuracies_by_K[K].append(cmc[2] * 100)
                top5_accuracies_by_K[K].append(cmc[4] * 100)
                top10_accuracies_by_K[K].append(cmc[-1] * 100)
                AP_by_K[K].append(mAP)
                INP_by_K[K].append(mINP)
                # print(test_query.metadata.identity)
                # AP_by_K[K].append(average_precision_score(test_query.metadata.identity, similarity))

            print(
                f"[Skyline, K={K}] Top-1 Accuracy: { (np.mean(top1_accuracies_by_K[K]))}% +/- { (np.std(top1_accuracies_by_K[K]))}%"
            )
            print(
                f"[Skyline, K={K}] Top-3 Accuracy: { (np.mean(top3_accuracies_by_K[K]))}% +/- { (np.std(top3_accuracies_by_K[K]))}%"
            )
            print(
                f"[Skyline, K={K}] Top-5 Accuracy: { (np.mean(top5_accuracies_by_K[K]))}% +/- { (np.std(top5_accuracies_by_K[K]))}%"
            )
            print(
                f"[Skyline, K={K}] Top-10 Accuracy: { (np.mean(top10_accuracies_by_K[K]))}% +/- { (np.std(top10_accuracies_by_K[K]))}%"
            )
            print(
                f"[Skyline, K={K}] AP: { (np.mean(AP_by_K[K]))} +/- { (np.std(AP_by_K[K]))}"
            )
            print(
                f"[Skyline, K={K}] INP: { (np.mean(INP_by_K[K]))} +/- { (np.std(INP_by_K[K]))}"
            )
            folder = os.path.join(
                args.out_dir,
                "pretrained_models",
                args.dataset,
                args.backbone,
                "skyline",
            )
            os.makedirs(folder, exist_ok=True)
            with open(
                os.path.join(
                    folder,
                    f"mean_accuracy_{args.seed}_{args.min_samples}_{args.num_splits}_{K}.json",
                ),
                "w",
            ) as fp:
                json.dump(
                    {
                        "Mean": {
                            "Top-1 Accuracy": (np.mean(top1_accuracies_by_K[K])),
                            "Top-3 Accuracy": (np.mean(top3_accuracies_by_K[K])),
                            "Top-5 Accuracy": (np.mean(top5_accuracies_by_K[K])),
                            "Top-10 Accuracy": (np.mean(top10_accuracies_by_K[K])),
                            "AP": (np.mean(AP_by_K[K])),
                            "INP": (np.mean(INP_by_K[K])),
                        },
                        "Std. Dev.": {
                            "Top-1 Accuracy": (np.std(top1_accuracies_by_K[K])),
                            "Top-3 Accuracy": (np.std(top3_accuracies_by_K[K])),
                            "Top-5 Accuracy": (np.std(top5_accuracies_by_K[K])),
                            "Top-10 Accuracy": (np.std(top10_accuracies_by_K[K])),
                            "AP": (np.std(AP_by_K[K])),
                            "INP": (np.std(INP_by_K[K])),
                        },
                    },
                    fp,
                )
            with open(
                os.path.join(
                    folder,
                    f"all_accuracy_{args.seed}_{args.min_samples}_{args.num_splits}_{K}.json",
                ),
                "w",
            ) as fp:
                json.dump(
                    {
                        s: {
                            "Top-1 Accuracy": (top1_acc, 2),
                            "Top-3 Accuracy": (top3_acc, 2),
                            "Top-5 Accuracy": (top5_acc, 2),
                            "Top-10 Accuracy": (top10_acc, 2),
                        }
                        for s, (top1_acc, top3_acc, top5_acc, top10_acc) in enumerate(
                            zip(
                                top1_accuracies_by_K[K],
                                top3_accuracies_by_K[K],
                                top5_accuracies_by_K[K],
                                top10_accuracies_by_K[K],
                            )
                        )
                    },
                    fp,
                )
            with open(
                os.path.join(
                    folder,
                    f"all_AP_INP_{args.seed}_{args.min_samples}_{args.num_splits}_{K}.json",
                ),
                "w",
            ) as fp:
                json.dump(
                    {
                        s: {
                            "AP": (ap, 2),
                            "INP": (inp, 2),
                        }
                        for s, (ap, inp) in enumerate(zip(AP_by_K[K], INP_by_K[K]))
                    },
                    fp,
                )

    elif args.baseline:
        megadescriptor_model = get_model(args.model_name)
        megadescriptor_extractor = DeepFeatures(
            megadescriptor_model,
            device=args.device,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )

        top1_accuracies_by_K = {}
        top3_accuracies_by_K = {}
        top5_accuracies_by_K = {}
        top10_accuracies_by_K = {}
        AP_by_K = {}
        INP_by_K = {}
        # if args.dataset != "SeaStarReID2023":
        #     Ks = [1, 2, 5]
        # else:

        Ks = [1, 5]
        for K in Ks:
            top1_accuracies_by_K[K] = []
            top3_accuracies_by_K[K] = []
            top5_accuracies_by_K[K] = []
            top10_accuracies_by_K[K] = []
            AP_by_K[K] = []
            INP_by_K[K] = []

            for test_gallery, test_query in zip(
                dataset.test_galleries_by_K[K], dataset.test_queries_by_K[K]
            ):
                print(f"Gallery size: {test_gallery.metadata.shape[0]}")
                print(f"Query size: {test_query.metadata.shape[0]}")
                gallery_features = megadescriptor_extractor(test_gallery)
                query_features = megadescriptor_extractor(test_query)

                similarity = dataset.matcher(
                    query=query_features, database=gallery_features
                )
                # preds = top1_classifier(similarity)
                # preds_top3 = top3_classifier(similarity)
                # preds_top5 = top5_classifier(similarity)

                # top_1_acc = sum(preds == test_query.labels_string) / len(preds)
                # top_3_acc = np.mean(np.any(preds_top3 == test_query.labels_string[:, None], axis=1))
                # top_5_acc = np.mean(np.any(preds_top5 == test_query.labels_string[:, None], axis=1))
                # top1_accuracies_by_K[K].append(top_1_acc * 100)
                # top3_accuracies_by_K[K].append(top_3_acc * 100)
                # top5_accuracies_by_K[K].append(top_5_acc * 100)

                # unique_ids = np.unique(test_query.metadata.query('label == "test_query"').identity.values)
                # y_true = np.zeros(similarity.shape[0])
                # for i, id in enumerate(unique_ids):
                #     y_true[test_query.metadata.identity == id] = i
                # print(y_true, similarity.shape)

                # AP_by_K[K].append(average_precision_score(y_true, similarity))

                cmc, all_AP, all_INP = evaluation.eval_wildlife(
                    1 - similarity,
                    test_query.metadata.identity.values,
                    test_gallery.metadata.identity.values,
                    max_rank=10,
                )

                mAP = np.mean(all_AP)
                mINP = np.mean(all_INP)

                top1_accuracies_by_K[K].append(cmc[0] * 100)
                top3_accuracies_by_K[K].append(cmc[2] * 100)
                top5_accuracies_by_K[K].append(cmc[4] * 100)
                top10_accuracies_by_K[K].append(cmc[-1] * 100)
                AP_by_K[K].append(mAP)
                INP_by_K[K].append(mINP)
                # print(test_query.metadata.identity)
                # AP_by_K[K].append(average_precision_score(test_query.metadata.identity, similarity))

            print(
                f"[Baseline, K={K}] Top-1 Accuracy: {(np.mean(top1_accuracies_by_K[K]))}% +/- {(np.std(top1_accuracies_by_K[K]))}%"
            )
            print(
                f"[Baseline, K={K}] Top-3 Accuracy: {(np.mean(top3_accuracies_by_K[K]))}% +/- {(np.std(top3_accuracies_by_K[K]))}%"
            )
            print(
                f"[Baseline, K={K}] Top-5 Accuracy: {(np.mean(top5_accuracies_by_K[K]))}% +/- {(np.std(top5_accuracies_by_K[K]))}%"
            )
            print(
                f"[Baseline, K={K}] Top-10 Accuracy: {(np.mean(top10_accuracies_by_K[K]))}% +/- {(np.std(top10_accuracies_by_K[K]))}%"
            )
            print(
                f"[Baseline, K={K}] AP: {(np.mean(AP_by_K[K]))} +/- {(np.std(AP_by_K[K]))}"
            )
            print(
                f"[Baseline, K={K}] INP: {(np.mean(INP_by_K[K]))} +/- {(np.std(INP_by_K[K]))}"
            )
            folder = os.path.join(
                args.out_dir,
                "pretrained_models",
                args.dataset,
                args.backbone,
                "baseline",
            )
            os.makedirs(folder, exist_ok=True)
            with open(
                os.path.join(
                    folder,
                    f"mean_accuracy_{args.seed}_{args.min_samples}_{args.num_splits}_{K}.json",
                ),
                "w",
            ) as fp:
                json.dump(
                    {
                        "Mean": {
                            "Top-1 Accuracy": (np.mean(top1_accuracies_by_K[K])),
                            "Top-3 Accuracy": (np.mean(top3_accuracies_by_K[K])),
                            "Top-5 Accuracy": (np.mean(top5_accuracies_by_K[K])),
                            "Top-10 Accuracy": (np.mean(top10_accuracies_by_K[K])),
                            "AP": (np.mean(AP_by_K[K])),
                            "INP": (np.mean(INP_by_K[K])),
                        },
                        "Std. Dev.": {
                            "Top-1 Accuracy": (np.std(top1_accuracies_by_K[K])),
                            "Top-3 Accuracy": (np.std(top3_accuracies_by_K[K])),
                            "Top-5 Accuracy": (np.std(top5_accuracies_by_K[K])),
                            "Top-10 Accuracy": (np.std(top10_accuracies_by_K[K])),
                            "AP": (np.std(AP_by_K[K])),
                            "INP": (np.std(INP_by_K[K])),
                        },
                    },
                    fp,
                )
            with open(
                os.path.join(
                    folder,
                    f"all_accuracy_{args.seed}_{args.min_samples}_{args.num_splits}_{K}.json",
                ),
                "w",
            ) as fp:
                json.dump(
                    {
                        s: {
                            "Top-1 Accuracy": (top1_acc, 2),
                            "Top-3 Accuracy": (top3_acc, 2),
                            "Top-5 Accuracy": (top5_acc, 2),
                            "Top-10 Accuracy": (top10_acc, 2),
                        }
                        for s, (top1_acc, top3_acc, top5_acc, top10_acc) in enumerate(
                            zip(
                                top1_accuracies_by_K[K],
                                top3_accuracies_by_K[K],
                                top5_accuracies_by_K[K],
                                top10_accuracies_by_K[K],
                            )
                        )
                    },
                    fp,
                )
            with open(
                os.path.join(
                    folder,
                    f"all_AP_INP_{args.seed}_{args.min_samples}_{args.num_splits}_{K}.json",
                ),
                "w",
            ) as fp:
                json.dump(
                    {
                        s: {
                            "AP": (ap, 2),
                            "INP": (inp, 2),
                        }
                        for s, (ap, inp) in enumerate(zip(AP_by_K[K], INP_by_K[K]))
                    },
                    fp,
                )
    else:
        megadescriptor_model = get_model(args.model_name)
        n = database.metadata.identity.unique().shape[0]
        # n = 50
        if args.sampler != "CC":
            folder = os.path.join(
                args.out_dir,
                f"pretrained_models",
                args.dataset,
                args.backbone,
                args.sampler,
            )
        else:
            sampler_string = f"{args.sampler}_{args.sampling_ratio}"
            folder = os.path.join(
                args.out_dir,
                f"pretrained_models",
                args.dataset,
                args.backbone,
                sampler_string,
            )
        sample_outliers = False

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
                prior_id=args.prior_id,
                eps=args.consam_eps,
                thresh=args.thresh,
                gallery_size=args.gallery_size,
                min_samples=args.min_samples,
                metric=args.metric,
            )
        else:
            raise NotImplementedError

        all_logs = {}
        all_metadata = {}
        all_switch_logs = {}
        all_gt_id_counts = {}
        all_sampling_pools = {}
        # budgets = [19, 36-19, 61-39, 100-61, 135-100]
        import wandb

        run_suffix = "md" if args.backbone == "megadescriptor" else "r50"
        if not args.sampler == "CC":
            wandb.init(
                entity="depanshu-sani",
                project=args.dataset,
                name=f"{args.sampler}-{run_suffix} ({args.thresh}, {args.prior_id}, {args.consam_eps})",
            )
        else:
            wandb.init(
                entity="depanshu-sani",
                project=args.dataset,
                name=f"{args.sampler}[{args.sampling_ratio}]-{run_suffix} ({args.thresh}, {args.prior_id}, {args.consam_eps})",
            )

        for budget in range(1, args.budget + 1):
            # sampler.n = budgets[budget-1]
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
            )
            # if args.dataset != "SeaStarReID2023":
            #     Ks = [1, 2, 5]
            # else:
            Ks = [1, 5]

            if not args.eval_only:
                wandb.log(
                    {
                        "Purity Ratio": logs["Purity Ratio"],
                        "Imurity Ratio Mean": logs["Impurity Ratio Mean"],
                        "Number of +ve Annotations": logs["Number of +ve Annotations"],
                        "Number of -ve Annotations": logs["Number of -ve Annotations"],
                        "Number of training samples": logs[
                            "Number of training samples"
                        ],
                        "Number of pseudo-identities": logs[
                            "Number of pseudo-identities"
                        ],
                        "Number of gt-identities:": logs["Number of gt-identities:"],
                    },
                    step=budget,
                )

                for i in range(args.epochs):
                    wandb.log(
                        {
                            "train_loss_epoch_avg": logs["train_loss_epoch_avg"][i],
                            "train_acc_epoch_avg": logs["train_acc_epoch_avg"][i],
                        },
                        step=(budget * args.epochs - 1) + i,
                    )
            wandb.log(
                {
                    "Closed/Accuracy/Top-1": logs["Closed"]["Top-1 Accuracy"],
                    "Closed/Accuracy/Top-3": logs["Closed"]["Top-3 Accuracy"],
                    "Closed/Accuracy/Top-5": logs["Closed"]["Top-5 Accuracy"],
                    "Closed/Accuracy/Top-10": logs["Closed"]["Top-10 Accuracy"],
                    "Closed/mAP": logs["Closed"]["AP"],
                    "Closed/mINP": logs["Closed"]["INP"],
                },
                step=budget,
            )
            for K in Ks:
                wandb.log(
                    {
                        f"K-{K}/Accuracy/Top-1": logs[K]["Mean"]["Top-1 Accuracy"],
                        f"K-{K}/Accuracy/Top-3": logs[K]["Mean"]["Top-3 Accuracy"],
                        f"K-{K}/Accuracy/Top-5": logs[K]["Mean"]["Top-5 Accuracy"],
                        f"K-{K}/Accuracy/Top-10": logs[K]["Mean"]["Top-10 Accuracy"],
                        f"K-{K}/mAP": logs[K]["Mean"]["AP"],
                        f"K-{K}/mINP": logs[K]["Mean"]["INP"],
                        f"K-{K}/KNN-top1acc": logs[K]["KNN"]["Top-1 Accuracy Mean"],
                    },
                    step=budget,
                )
                print(
                    f"[{budget}/{args.budget}] Top-1 Accuracy: {logs[K]['Mean']['Top-1 Accuracy']}% +/- {logs[K]['Std. Dev.']['Top-1 Accuracy']}%"
                )
                print(
                    f"[{budget}/{args.budget}] Top-3 Accuracy: {logs[K]['Mean']['Top-3 Accuracy']}% +/- {logs[K]['Std. Dev.']['Top-3 Accuracy']}%"
                )
                print(
                    f"[{budget}/{args.budget}] Top-5 Accuracy: {logs[K]['Mean']['Top-5 Accuracy']}% +/- {logs[K]['Std. Dev.']['Top-5 Accuracy']}%"
                )
                print(
                    f"[{budget}/{args.budget}] Top-10 Accuracy: {logs[K]['Mean']['Top-10 Accuracy']}% +/- {logs[K]['Std. Dev.']['Top-10 Accuracy']}%"
                )
                print(
                    f"[{budget}/{args.budget}] AP: {logs[K]['Mean']['AP']} +/- {logs[K]['Std. Dev.']['AP']}"
                )
                print(
                    f"[{budget}/{args.budget}] INP: {logs[K]['Mean']['INP']} +/- {logs[K]['Std. Dev.']['INP']}"
                )
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
                f"logs_{args.thresh}_{args.seed}_{args.min_samples}_{args.num_splits}_{args.images_per_id}.json",
            ),
            "w",
        ) as fp:
            json.dump(all_logs, fp)
        with open(
            os.path.join(
                folder,
                f"gt_id_counts_{args.thresh}_{args.seed}_{args.min_samples}_{args.num_splits}_{args.images_per_id}.json",
            ),
            "w",
        ) as fp:
            json.dump(all_gt_id_counts, fp)
        with open(
            os.path.join(
                folder,
                f"switch_logs_{args.thresh}_{args.seed}_{args.min_samples}_{args.num_splits}_{args.images_per_id}.json",
            ),
            "w",
        ) as fp:
            json.dump(all_switch_logs, fp)
        with open(
            os.path.join(
                folder,
                f"sampling_pools_{args.thresh}_{args.seed}_{args.min_samples}_{args.num_splits}_{args.images_per_id}.json",
            ),
            "w",
        ) as fp:
            json.dump(all_sampling_pools, fp)
        if len(metadata.keys()) > 0:
            import pickle

            with open(
                os.path.join(
                    folder,
                    f"metadata_{args.thresh}_{args.seed}_{args.min_samples}_{args.num_splits}_{args.images_per_id}.pickle",
                ),
                "wb",
            ) as fp:
                pickle.dump(all_metadata, fp)


if __name__ == "__main__":
    args = get_main_args()
    set_seed(args.seed)
    main(args)
