import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from collections import defaultdict
import os


def compute_and_log_pseudo_id_stats(
    database_subset, database_features, logs, cycle, save_root="diagnostics"
):
    metadata = database_subset.metadata
    features = (
        database_features.features
    )  # This must be added before calling this function
    pseudo_id_stats = defaultdict(dict)
    pseudo_ids = metadata["pseudo_identity"].unique()
    gt_ids = metadata["identity"].unique()

    global_cosine_dists = []
    nearest_other_cluster_dists = []
    per_gt_similarities = defaultdict(list)

    idx_map = {i: idx for idx, i in enumerate(metadata.index)}
    feature_matrix = features

    # Create directories
    pseudo_dir = os.path.join(save_root, f"cycle_{cycle}", "pseudo_ids")
    gt_dir = os.path.join(save_root, f"cycle_{cycle}", "gt_ids")
    os.makedirs(pseudo_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for pseudo_id in pseudo_ids:
        idxs = metadata.index[metadata["pseudo_identity"] == pseudo_id].tolist()
        if len(idxs) < 2:
            continue

        sub_feats = feature_matrix[[idx_map[idx] for idx in idxs]]
        sim_matrix = cosine_similarity(sub_feats)
        triu_inds = np.triu_indices_from(sim_matrix, k=1)
        similarities = sim_matrix[triu_inds]

        gts = metadata.loc[idxs, "identity"].values
        same_mask = np.equal.outer(gts, gts)[triu_inds]
        diff_mask = ~same_mask

        fig, ax = plt.subplots()
        sns.histplot(
            similarities[same_mask],
            color="green",
            label="Same GT",
            kde=True,
            ax=ax,
            stat="density",
        )
        sns.histplot(
            similarities[diff_mask],
            color="red",
            label="Diff GT",
            kde=True,
            ax=ax,
            stat="density",
        )
        ax.legend()
        ax.set_title(f"Cosine similarity in Pseudo-ID {pseudo_id}")
        ax.set_xlim(-1, 1)
        fig.savefig(os.path.join(pseudo_dir, f"pseudo_id_{pseudo_id}.png"))
        plt.close(fig)

        # purity:
        pseudo_id_stats[pseudo_id]["identity"] = metadata.loc[idxs, "identity"].mode()[
            0
        ]
        pseudo_id_stats[pseudo_id]["n_samples"] = len(idxs)
        pseudo_id_stats[pseudo_id]["purity"] = (
            metadata.loc[idxs, "identity"] == pseudo_id_stats[pseudo_id]["identity"]
        ).mean()
        global_cosine_dists.extend(similarities.tolist())

        other_idxs = metadata.index[metadata["pseudo_identity"] != pseudo_id].tolist()
        if other_idxs:
            other_feats = feature_matrix[[idx_map[idx] for idx in other_idxs]]
            for f, gt in zip(sub_feats, gts):
                sims = cosine_similarity([f], other_feats).squeeze()
                best_idx = np.argmax(sims)
                best_sim = sims[best_idx]
                nearest_other_cluster_dists.append(
                    (
                        best_sim,
                        int(metadata.loc[other_idxs[best_idx], "identity"] == gt),
                    )
                )

            for gt in np.unique(gts):
                same_gt_other_pseudo_idxs = metadata.index[
                    (metadata["pseudo_identity"] != pseudo_id)
                    & (metadata["identity"] == gt)
                ].tolist()
                same_gt_other_feats = feature_matrix[
                    [idx_map[i] for i in same_gt_other_pseudo_idxs]
                ]

                curr_pseudo_idxs = metadata.index[
                    (metadata["pseudo_identity"] == pseudo_id)
                    & (metadata["identity"] == gt)
                ].tolist()
                curr_pseudo_feats = feature_matrix[
                    [idx_map[i] for i in curr_pseudo_idxs]
                ]

                # cross between current pseudo-id and other pseudo-ids with same GT
                sim_matrix = cosine_similarity(curr_pseudo_feats, same_gt_other_feats)
                similarities = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
                per_gt_similarities[pseudo_id].append((gt, similarities.tolist()))

                fig, ax = plt.subplots()
                sns.histplot(similarities, ax=ax, color="blue")
                ax.set_title(
                    f"Cosine similarity for Pseudo-ID {pseudo_id} with GT {gt} (inter-cluster)"
                )
                ax.set_xlim(-1, 1)
                fig.savefig(
                    os.path.join(pseudo_dir, f"pseudo_id_{pseudo_id}_gt_{gt}.png")
                )
                plt.close(fig)

    gt_stats = defaultdict(dict)
    for gt_id in gt_ids:
        gt_idxs = metadata.index[metadata["identity"] == gt_id].tolist()
        if len(gt_idxs) < 2:
            continue

        gt_feats = feature_matrix[[idx_map[i] for i in gt_idxs]]
        sim_matrix = cosine_similarity(gt_feats)
        similarities = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

        fig, ax = plt.subplots()
        sns.histplot(similarities, ax=ax, color="blue")
        ax.set_title(f"Cosine similarity for GT-ID {gt_id} (inter-cluster)")
        ax.set_xlim(-1, 1)
        fig.savefig(os.path.join(gt_dir, f"gt_id_{gt_id}.png"))
        plt.close(fig)

        assigned_pseudo_ids = metadata.loc[gt_idxs, "pseudo_identity"].unique()
        gt_stats[gt_id]["n_samples"] = len(gt_idxs)
        gt_stats[gt_id]["n_pseudo_ids"] = len(assigned_pseudo_ids)
        gt_stats[gt_id]["cosine_sims"] = similarities.tolist()

    logs["Global Pseudo-ID Sizes"] = [
        pseudo_id_stats[k]["n_samples"] for k in pseudo_id_stats
    ]

    counts = metadata["pseudo_identity"].value_counts()
    ys = counts.values
    xs = np.arange(len(ys))
    plt.figure(figsize=(10, 6))
    print(ys, xs)
    import pdb

    pdb.set_trace()
    plt.hist(xs, weights=ys, bins=len(xs), alpha=0.7, color="blue", edgecolor="black")
    plt.xticks(xs, counts.index, rotation=90)
    plt.xlabel("Pseudo-ID")
    plt.ylabel("Number of Samples")
    plt.title("Distribution of Samples per Pseudo-ID")
    plt.savefig(os.path.join(save_root, f"cycle_{cycle}", "pseudo_id_distribution.png"))
    plt.close(fig)

    logs["Global Pseudo-ID Purities"] = [
        pseudo_id_stats[k]["purity"] for k in pseudo_id_stats
    ]

    # plot with same order as Pseudo-ID distribution
    ys = [pseudo_id_stats[k]["purity"] for k in counts.index]
    xs = np.arange(len(ys))
    plt.figure(figsize=(10, 6))
    plt.bar(xs, ys, alpha=0.7, color="blue", edgecolor="black")
    plt.xticks(xs, counts.index, rotation=90)
    plt.xlabel("Pseudo-ID")
    plt.ylabel("Purity")
    plt.title("Purity of Pseudo-IDs")
    plt.savefig(os.path.join(save_root, f"cycle_{cycle}", "pseudo_id_purity.png"))
    plt.close(fig)

    logs["Nearest Other Cluster Cosine"] = [s for s, _ in nearest_other_cluster_dists]

    # plot nearest other cluster cosine similarities with same order as Pseudo-ID distribution
    nearest_other_cluster_dists = np.array(nearest_other_cluster_dists)
    nearest_other_cluster_dists = nearest_other_cluster_dists[
        metadata["pseudo_identity"].value_counts().index
    ]
    ys = nearest_other_cluster_dists[:, 0]
    xs = np.arange(len(ys))
    plt.figure(figsize=(10, 6))
    plt.bar(xs, ys, alpha=0.7, color="blue", edgecolor="black")
    plt.xticks(xs, counts.index, rotation=90)
    plt.xlabel("Pseudo-ID")
    plt.ylabel("Nearest Other Cluster Cosine Similarity")
    plt.title("Nearest Other Cluster Cosine Similarity per Pseudo-ID")
    plt.savefig(
        os.path.join(save_root, f"cycle_{cycle}", "nearest_other_cluster_cosine.png")
    )
    plt.close(fig)

    # logs["Nearest Other Cluster SameGT"] = [same for _, same in nearest_other_cluster_dists]

    # # plot nearest other cluster same GT with same order as Pseudo-ID distribution
    # ys = nearest_other_cluster_dists[:, 1]
    # xs = np.arange(len(ys))
    # plt.figure(figsize=(10, 6))
    # plt.bar(xs, ys, alpha=0.7, color='blue', edgecolor='black')
    # plt.xticks(xs, counts.index, rotation=90)
    # plt.xlabel("Pseudo-ID")
    # plt.ylabel("Nearest Other Cluster Same GT")
    # plt.title("Nearest Other Cluster Same GT per Pseudo-ID")
    # plt.savefig(os.path.join(save_root, f"cycle_{cycle}", "nearest_other_cluster_same_gt.png"))
    # plt.close(fig)

    logs["GT ID Sizes"] = [gt_stats[k]["n_samples"] for k in gt_stats]

    counts = metadata["identity"].value_counts()
    ys = counts.values
    xs = np.arange(len(ys))
    plt.figure(figsize=(10, 6))
    plt.hist(xs, weights=ys, bins=len(xs), alpha=0.7, color="blue", edgecolor="black")
    plt.xticks(xs, counts.index, rotation=90)
    plt.xlabel("GT-ID")
    plt.ylabel("Number of Samples")
    plt.title("Distribution of Samples per GT-ID")
    plt.savefig(os.path.join(save_root, f"cycle_{cycle}", "gt_id_distribution.png"))
    plt.close(fig)

    logs["GT ID Num Pseudo-IDs"] = [gt_stats[k]["n_pseudo_ids"] for k in gt_stats]

    # plot with same order as GT ID distribution
    ys = [gt_stats[k]["n_pseudo_ids"] for k in counts.index]
    xs = np.arange(len(ys))
    plt.figure(figsize=(10, 6))
    plt.bar(xs, ys, alpha=0.7, color="blue", edgecolor="black")
    plt.xticks(xs, counts.index, rotation=90)
    plt.xlabel("GT-ID")
    plt.ylabel("Number of Pseudo-IDs")
    plt.title("Number of Pseudo-IDs per GT-ID")
    plt.savefig(os.path.join(save_root, f"cycle_{cycle}", "gt_id_num_pseudo_ids.png"))
    plt.close(fig)

    logs["GT Cosine Similarities"] = [v["cosine_sims"] for k, v in gt_stats.items()]

    return logs, pseudo_id_stats, gt_stats
