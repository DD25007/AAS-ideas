import argparse
import os
import numpy as np
import itertools
from timm import create_model
import networkx as nx

import random
from sklearn import metrics

from wildlife_tools.data import WildlifeDataset

import torch
import torch.nn as nn
import torch.nn.functional as F

from wildlife_tools.train import BasicTrainer

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from tqdm import tqdm

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from typing import Optional, List, Tuple, Callable, Union
from wildlife_datasets import metrics
from wildlife_tools.similarity import CosineSimilarity
from collections import defaultdict


def get_main_args():
    """Command line arguments."""

    parser = argparse.ArgumentParser("HITL ReIdentification")
    parser.add_argument("--dataset", type=str, default="SeaStarReID2023")
    # parser.add_argument("--dataset_path", type=str, default="/mnt/nas/users/depanshus/WildlifeDatasets/")
    parser.add_argument("--dataset_path", type=str, default="/storage/dd25007/dataset/")
    parser.add_argument("--out_dir", type=str, default="outputs/")
    parser.add_argument(
        "--model_name", type=str, default="hf-hub:BVRA/MegaDescriptor-L-224"
    )
    parser.add_argument("--backbone", type=str, default="megadescriptor")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--skyline", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--sampler", default="random")
    parser.add_argument("--min_samples", type=int, default=5, help="minPts for CDBSCAN")
    parser.add_argument("--thresh", type=float, default=0.1, help="eps for CDBSCAN")
    parser.add_argument(
        "--sampling_ratio",
        type=float,
        default=1.0,
        help="ratio of neighbors vs vertices (CC method: Perez et al., ECCV 2024)",
    )
    # parser.add_argument("--consam_betas", type=str, default="0.5,0.35,0.15", help="comma-separated list of the 3 betas in ConSam, e.g.: 0.5,0.35,0.15")
    parser.add_argument(
        "--consam_beta1", type=float, default=0.5, help="beta1 in ConSam"
    )
    parser.add_argument(
        "--consam_beta2", type=float, default=0.35, help="beta2 in ConSam"
    )
    parser.add_argument(
        "--consam_beta3", type=float, default=0.15, help="beta3 in ConSam"
    )
    parser.add_argument("--consam_eps", type=float, default=0.05)
    parser.add_argument(
        "--eval_over_all_ids",
        action="store_true",
        help="evaluate over all ids in the dataset",
    )
    parser.add_argument(
        "--no_sweep", action="store_true", help="use for sweep on purity ratio"
    )
    parser.add_argument(
        "--run_val",
        action="store_true",
        help="run validation at each step during training",
    )
    parser.add_argument(
        "--adaptive", action="store_true", help="use adaptive ConSam sampling"
    )
    parser.add_argument(
        "--lambda_", type=float, default=3.0, help="lambda for adaptive ConSam sampling"
    )
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument(
        "--loss_criterion",
        type=str,
        default="softmax",
        choices=["arcface", "softmax", "triplet"],
    )
    parser.add_argument("--gallery_size", type=int, default=2)
    parser.add_argument(
        "--num_splits",
        type=int,
        default=5,
        help="number of query/gallery splits for evaluation",
    )
    parser.add_argument(
        "--images_per_id",
        type=int,
        default=1,
        help="number of images per id in gallery (evaluation)",
    )
    parser.add_argument("--sample_seed", type=int, default=0, help="seed for sampling")
    parser.add_argument("--momentum", type=float, default=0.2)
    args = parser.parse_args()
    return args


def set_seed(seed):
    """
    Set seed values for reproducibility
    :param seed: The seed value
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def constrained_clustering(
    X,
    method,
    eps=0.2,
    min_samples=5,
    metric="cosine",
    ml_pairs=[],
    cl_pairs=[],
    sanity_check=False,
):

    if metric == "precomputed":
        X = torch.tensor(X, dtype=torch.float32)
        X = F.normalize(X, dim=1).cuda()
        rerank_dist = compute_jaccard_distance(X)
        if method == "FINCH":
            np.fill_diagonal(rerank_dist, 1e12)
            initial_rank = np.argmin(rerank_dist, axis=1)
    else:
        rerank_dist = X
        initial_rank = None
    if method == "DBSCAN":
        from sklearn.cluster import DBSCAN

        rerank_dist = (
            rerank_dist.cpu().numpy()
            if isinstance(rerank_dist, torch.Tensor)
            else rerank_dist
        )
        labels = (
            DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            .fit(rerank_dist)
            .labels_
        )
    elif method == "FINCH":
        X = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        labels = FINCH(X, initial_rank=initial_rank)
    elif method == "ConSam":
        # Get Density-based (DBSCAN) clusters
        # "clusters_dbscan" contains information about outliers while
        # "labels_dbscan" represent each outlier as a singleton cluster
        clusters_dbscan = constrained_clustering(
            X=X,
            method="DBSCAN",
            eps=eps,
            min_samples=min_samples,
            metric=metric,
        )
        labels_dbscan = clusters_dbscan.copy()
        labels_dbscan[labels_dbscan == -1] = np.arange(
            labels_dbscan.max() + 1, labels_dbscan.max() + 1 + sum(labels_dbscan == -1)
        )

        # Get Nearest Neighbor-based (FINCH) clusters
        labels_finch = constrained_clustering(
            X=X,
            method="FINCH",
            metric=metric,
        )

        def get_pairwise_relationships(labels):
            n_points = len(labels)
            # Group points by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(labels):
                clusters[label].append(i)
            ml_pairs = set()
            # Generate ML pairs within each cluster
            for cluster_id in clusters:
                # Use itertools.combinations for efficient pair generation
                for p1, p2 in itertools.combinations(clusters[cluster_id], 2):
                    # Store in canonical form (e.g., sorted tuple or frozenset)
                    ml_pairs.add(tuple(sorted({p1, p2})))
            # Generate all possible pairs
            all_possible_pairs = set()
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    all_possible_pairs.add(tuple(sorted({i, j})))
            # CL pairs are all possible pairs minus ML pairs
            cl_pairs = all_possible_pairs - ml_pairs
            return ml_pairs, cl_pairs

        ml1, cl1 = get_pairwise_relationships(labels_dbscan)
        ml2, cl2 = get_pairwise_relationships(labels_finch)

        weak_ml_pairs = ml1.intersection(ml2)
        weak_cl_pairs = cl1.intersection(cl2)
        inconsistent_pairs = (cl1.union(cl2) - cl1.intersection(cl2)).union(
            ml1.union(ml2) - ml1.intersection(ml2)
        )
        weak_cl_pairs = weak_cl_pairs.union(inconsistent_pairs)
        labels = constrained_clustering(
            X=X,
            method="FINCH",
            metric=metric,
            ml_pairs=weak_ml_pairs,
            cl_pairs=weak_cl_pairs,
        )
        if len(ml_pairs) == 0 and len(cl_pairs) == 0:
            return labels
    else:
        raise NotImplementedError("Clustering method not implemented")

    # Step 1: Merge clusters under Must-Link constraints
    cluster_id = labels.max() + 1
    for a, b in ml_pairs:
        cluster_a = labels[a]
        cluster_b = labels[b]
        if cluster_a == -1 and cluster_b == -1:
            labels[a] = cluster_id
            labels[b] = cluster_id
            cluster_id += 1
        elif cluster_a == -1:
            labels[a] = cluster_b
        elif cluster_b == -1:
            labels[b] = cluster_a
        elif cluster_a != cluster_b:
            labels[labels == cluster_b] = cluster_a

    unique_clusters = np.unique(labels)
    cl_pairs_set = set(cl_pairs)
    ml_pairs_set = set(ml_pairs)

    for c_id in unique_clusters:
        # Ensure cannot-link constraints are not violated in any cluster
        if c_id == -1:
            continue
        clust_points = np.where(labels == c_id)[0]
        clust_point_pairs = list(itertools.product(clust_points, clust_points))
        clust_cl_pairs = set(clust_point_pairs).intersection(cl_pairs_set)
        if len(clust_cl_pairs) > 0:
            # cannot-link constraint violated
            clust_ml_pairs = set(clust_point_pairs).intersection(ml_pairs_set)
            # Step 1: ML graph (G_ML)
            ml_graph = nx.Graph()
            ml_graph.add_nodes_from(clust_points)
            ml_graph.add_edges_from(clust_ml_pairs)
            # Step 2: Find ML components
            ml_components = list(nx.connected_components(ml_graph))
            comp_map = {
                pt: idx for idx, comp in enumerate(ml_components) for pt in comp
            }
            # Step 3: Build conflict graph (G_CL)
            conflict_graph = nx.Graph()
            conflict_graph.add_nodes_from(range(len(ml_components)))
            for a, b in clust_cl_pairs:
                ca, cb = comp_map[a], comp_map[b]
                if ca != cb:
                    conflict_graph.add_edge(ca, cb)

            # *************************************************
            # Step 4: Get connected components sorted by size descending
            conflict_graph_comps = list(nx.connected_components(conflict_graph))
            comp_to_chromatic = []
            for comp in conflict_graph_comps:
                subgraph = conflict_graph.subgraph(comp)
                coloring = nx.coloring.greedy_color(subgraph, strategy="largest_first")
                chromatic_number = max(coloring.values()) + 1
                comp_to_chromatic.append((comp, chromatic_number))

            # Sort by chromatic number (descending), break ties by size
            conflict_graph_comps_sorted = sorted(
                comp_to_chromatic, key=lambda x: (x[1], len(x[0])), reverse=True
            )
            conflict_graph_comps_sorted = [
                comp for comp, _ in conflict_graph_comps_sorted
            ]

            # Prepare mapping from component idx to member points
            # Initialize new mapping of cluster_id to list of points
            cluster_id_to_points = {}

            # Process largest component first
            largest_component = conflict_graph_comps_sorted[0]
            if len(largest_component) > 1:
                # Subgraph for largest_component
                subG = conflict_graph.subgraph(largest_component).copy()
                coloring = nx.coloring.greedy_color(subG, strategy="largest_first")
                # Assign new cluster IDs
                for comp_idx, color in coloring.items():
                    sub_id = cluster_id
                    pts = list(ml_components[comp_idx])
                    cluster_id_to_points[sub_id] = pts
                    labels[pts] = sub_id
                    cluster_id += 1
            else:
                # No CL violations! Labels are already correct...
                break

            # Process remaining components via assignment
            for comp_set in conflict_graph_comps_sorted[1:]:
                from sklearn.metrics import pairwise_distances
                from scipy.optimize import linear_sum_assignment

                # Build cost matrix: comps in comp_set x existing subclusters
                existing_ids = list(cluster_id_to_points.keys())
                m = len(comp_set)
                n = len(existing_ids)
                cost = np.zeros((m, n))
                for i, comp_idx in enumerate(comp_set):
                    pts_i = list(ml_components[comp_idx])
                    # distances from each comp point to each subcluster point
                    for j, sub_id in enumerate(existing_ids):
                        pts_j = cluster_id_to_points[sub_id]
                        # compute min pairwise distance
                        if metric == "precomputed":
                            sub_dist = rerank_dist[np.ix_(pts_i, pts_j)]
                            cost[i, j] = sub_dist.min()
                        else:
                            dist_matrix = pairwise_distances(
                                X[pts_i], X[pts_j], metric=metric
                            )
                            cost[i, j] = dist_matrix.min()

                # Solve assignment
                row_ind, col_ind = linear_sum_assignment(cost)
                for i, j in zip(row_ind, col_ind):
                    comp_idx = list(comp_set)[i]
                    sub_id = existing_ids[j]
                    for pt in ml_components[comp_idx]:
                        labels[pt] = sub_id
                        cluster_id_to_points[sub_id].append(pt)

    if method == "DBSCAN":
        unique, counts = np.unique(labels, return_counts=True)
        singleton_labels = unique[counts < min_samples]
        mask = np.isin(labels, singleton_labels)
        labels[mask] = -1

    # Step 4: Renumber cluster labels to be consecutive
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    label_map = {old: new for new, old in enumerate(unique_labels)}
    for old, new in label_map.items():
        labels[labels == old] = new

    if sanity_check:
        # Sanity check for constraints
        for a, b in ml_pairs:
            if labels[a] != labels[b]:
                print(f"{a}, {b}, {labels[[a, b]]}")
            assert (
                labels[a] == labels[b]
            ), "must-link constraint violated. Implementation error"
        for a, b in cl_pairs:
            if labels[a] == labels[b] and labels[a] != -1:
                print(f"{a}, {b}, {labels[[a, b]]}")
            assert (
                labels[a] != labels[b] or labels[a] == -1
            ), "cannot-link constraint violated. Implementation error"
    return labels


def CDBSCAN(
    X, eps, min_samples, metric="cosine", ml_pairs=[], cl_pairs=[], sanity_check=False
):
    n_samples = X.shape[0]
    labels = np.full(n_samples, -1)  # -1 for noise
    cluster_id = 0

    cl_set = set(cl_pairs)

    # Step 1: Get NNs instead of building a KD-Tree
    nbrs = NearestNeighbors(radius=eps, metric=metric).fit(X)

    # Step 2: Local clustering under Cannot-Link constraints
    visited = np.zeros(n_samples, dtype=bool)
    leaf_nodes = nbrs.radius_neighbors(X, return_distance=False)
    for i in tqdm(range(n_samples)):
        if visited[i] or labels[i] != -1:
            continue
        neighbors = leaf_nodes[i]
        visited[i] = True
        if len(neighbors) < min_samples:
            labels[i] = -1  # Noise
        else:
            has_cl = any(
                (a != b and ((a, b) in cl_set or (b, a) in cl_set))
                for a, b in itertools.combinations(neighbors, 2)
            )
            if has_cl:
                for p in neighbors:
                    labels[p] = cluster_id
                    cluster_id += 1
            else:
                labels[neighbors] = cluster_id
                cluster_id += 1

    # Step 3a: Merge clusters under Must-Link constraints to form core local clusters
    for a, b in ml_pairs:
        cluster_a = labels[a]
        cluster_b = labels[b]
        if cluster_a == -1 and cluster_b == -1:
            labels[a] = cluster_id
            labels[b] = cluster_id
            cluster_id += 1
            continue
        elif cluster_a == -1:
            labels[a] = cluster_b
            cluster_a = cluster_b
        elif cluster_b == -1:
            labels[b] = cluster_a
        elif cluster_a != cluster_b:
            labels[labels == cluster_b] = cluster_a
        # Ensure cannot-link constraints are met while merging. If not, form singleton clusters for cl_pairs
        clust_points = np.where(labels == cluster_a)[0]
        clust_point_pairs = list(itertools.product(clust_points, clust_points))
        clust_cl_pairs = set(clust_point_pairs).intersection(set(cl_pairs))
        if len(clust_cl_pairs) > 0:
            clust_ml_pairs = set(clust_point_pairs).intersection(set(ml_pairs))
            # Step 1: ML graph
            ml_graph = nx.Graph()
            ml_graph.add_nodes_from(clust_points)
            ml_graph.add_edges_from(clust_ml_pairs)
            # Step 2: Find ML components
            ml_components = list(nx.connected_components(ml_graph))
            comp_map = {
                pt: idx for idx, comp in enumerate(ml_components) for pt in comp
            }
            # Step 3: Build conflict graph (over components)
            conflict_graph = nx.Graph()
            conflict_graph.add_nodes_from(range(len(ml_components)))
            for a, b in clust_cl_pairs:
                ca, cb = comp_map[a], comp_map[b]
                if ca != cb:
                    conflict_graph.add_edge(ca, cb)
            coloring = nx.coloring.greedy_color(
                conflict_graph, strategy="largest_first"
            )
            label_map = {}
            for pt in clust_points:
                if label_map.get(coloring[comp_map[pt]]) is None:
                    label_map[coloring[comp_map[pt]]] = cluster_id
                    cluster_id += 1
                labels[pt] = label_map[coloring[comp_map[pt]]]

    # Step 3b: Merge remaining clusters unless blocked by Cannot-Link
    unique, counts = np.unique(labels, return_counts=True)
    local_clusters = unique[counts == 1]
    core_local_clusters = np.setdiff1d(unique[counts > 1], [-1])

    for ci in core_local_clusters:
        for cj in local_clusters:
            points_i = np.where(labels == ci)[0]
            points_j = np.where(labels == cj)[0]
            if len(np.intersect1d(leaf_nodes[points_j][0], points_i)) == 0:
                continue
            conflict = any(
                (a, b) in cl_set or (b, a) in cl_set
                for a, b in itertools.product(points_i, points_j)
            )

            if not conflict:
                labels[labels == cj] = ci
                local_clusters = np.delete(
                    local_clusters, np.argwhere(local_clusters == cj)
                )

    unique, counts = np.unique(labels, return_counts=True)
    singleton_labels = unique[counts == 1]
    mask = np.isin(labels, singleton_labels)
    labels[mask] = -1

    # Step 4: Renumber cluster labels to be consecutive
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    label_map = {old: new for new, old in enumerate(unique_labels)}
    for old, new in label_map.items():
        labels[labels == old] = new

    if sanity_check:
        # Sanity check for constraints
        for a, b in ml_pairs:
            if labels[a] != labels[b]:
                print(f"{a}, {b}, {labels[[a,b]]}")
            assert (
                labels[a] == labels[b]
            ), "must-link constraint violated. Implementation error"
        for a, b in cl_pairs:
            if labels[a] == labels[b] and labels[a] != -1:
                print(f"{a}, {b}, {labels[[a, b]]}")
            assert (
                labels[a] != labels[b] or labels[a] == -1
            ), "cannot-link constraint violated. Implementation error"
    return labels


# https://link.springer.com/content/pdf/10.1007/978-3-540-72530-5_25.pdf (Proposed in)
# https://github.com/johnnylu305/Constrained-DBSCAN-Clustering/blob/main/constrainedDBSCAN.m (Implementation from)
# Problem: Cannot-link constraints might be violated while merging in step 3a.




# Problem mentioned in the comment


def FINCH(
    data,
    initial_rank=None,
    req_clust=None,
    distance="cosine",
    ensure_early_exit=True,
    verbose=True,
    use_ann_above_samples=70000,
):
    """FINCH clustering algorithm.
    :param data: Input matrix with features in rows.
    :param initial_rank: Nx1 first integer neighbor indices (optional).
    :param req_clust: Set output number of clusters (optional). Not recommended.
    :param distance: One of ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] Recommended 'cosine'.
    :param ensure_early_exit: [Optional flag] may help in large, high dim datasets, ensure purity of merges and helps early exit
    :param verbose: Print verbose output.
    :param use_ann_above_samples: Above this data size (number of samples) approximate nearest neighbors will be used to speed up neighbor
        discovery. For large scale data where exact distances are not feasible to compute, set this. [default = 70000]
    :return:
            c: NxP matrix where P is the partition. Cluster label for every partition.
            num_clust: Number of clusters.
            req_c: Labels of required clusters (Nx1). Only set if `req_clust` is not None.

    The code implements the FINCH algorithm described in our CVPR 2019 paper
        Sarfraz et al. "Efficient Parameter-free Clustering Using First Neighbor Relations", CVPR2019
         https://arxiv.org/abs/1902.11266
    For academic purpose only. The code or its re-implementation should not be used for commercial use.
    Please contact the author below for licensing information.
    Copyright
    M. Saquib Sarfraz (saquibsarfraz@gmail.com)
    """

    from finch.finch import clust_rank, get_clust, get_merge

    # Cast input data to float32
    data = data.astype(np.float32)

    min_sim = None
    adj, orig_dist = clust_rank(
        data, use_ann_above_samples, initial_rank, distance, verbose
    )
    initial_rank = None
    group, num_clust = get_clust(adj, [], min_sim)
    c, mat = get_merge([], group, data)

    if ensure_early_exit:
        if orig_dist.shape[-1] > 2:
            min_sim = np.max(orig_dist * adj.toarray())

    return c




class EpochCheckpoint:
    """Save trainer checkpoint after epoch (used with wildlife-tools)."""

    def __init__(self, folder=".", save_step=1, seed=0, verbose=False):
        self.folder = folder
        self.save_step = save_step
        self.seed = seed
        self.verbose = verbose
        self.pseudo_labels_array = None

    def __call__(self, trainer, **kwargs):
        os.makedirs(self.folder, exist_ok=True)
        if self.verbose:
            print(
                f"[{trainer.epoch - 1}/{trainer.epochs}] "
                f"Train Loss: {np.round(kwargs['epoch_data']['train_loss_epoch_avg'], 2)} \t |"
                f" Train Acc: {np.round(kwargs['epoch_data']['train_acc_epoch_avg']*100, 2)} \t |"
            )
            # save pseudo-labels
            pseudo_labels = kwargs["epoch_data"]["pseudo_labels"]
            # convert map to np.array in order of sorted original indices
            sorted_indices = sorted(pseudo_labels.keys())
            pseudo_labels_array = np.array([pseudo_labels[i] for i in sorted_indices])
            if self.pseudo_labels_array is None:
                self.pseudo_labels_array = pseudo_labels_array
            else:
                # append new pseudo-labels to existing array
                self.pseudo_labels_array = np.vstack(
                    (self.pseudo_labels_array, pseudo_labels_array)
                )
            np.save(
                os.path.join(self.folder, f"pseudo-labels-seed{self.seed}.npy"),
                self.pseudo_labels_array,
            )

        if trainer.epoch % self.save_step == 0:
            trainer.save(
                folder=self.folder, file_name=f"best-checkpoint-seed{self.seed}.pth"
            )




def get_model(model_name, eval=True, combined=False, drop_rate=0.2):
    if combined:  # Do not use pre-trained megadescriptor
        if model_name == "hf-hub:BVRA/MegaDescriptor-L-224":
            model_name = "swin_large_patch4_window7_224"
        elif model_name == "hf-hub:BVRA/MegaDescriptor-T-224":
            model_name = "swin_tiny_patch4_window7_224"

    if model_name == "conservationxlabs/miewid-msv2":
        from transformers import AutoModel

        megadescriptor_model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True
        )
    else:
        megadescriptor_model = create_model(
            model_name, num_classes=0, pretrained=True, drop_rate=drop_rate
        )

    if eval:
        megadescriptor_model.eval()
    else:
        megadescriptor_model.train()
    return megadescriptor_model


def load_checkpoint(megadescriptor_model, folder):
    if os.path.exists(folder):
        checkpoint_file = os.listdir(folder)
        if len(checkpoint_file) > 0:
            checkpoint_file = [f for f in checkpoint_file if f.startswith("best")][0]
        else:
            return False
    else:
        return False
    megadescriptor_model.load_state_dict(
        torch.load(os.path.join(folder, checkpoint_file), weights_only=False)["model"]
    )
    return True


class ClusterLoss(nn.Module):
    """
    CE with single dense layer classification head.

    Args:
        num_classes (int): Number of classes.
        embedding_size (int): Size of the input embeddings.
    """

    def __init__(self, centroids, embedding_size, tau=0.07):
        super().__init__()
        self.tau = tau
        self.criterion = nn.CrossEntropyLoss()
        num_clusters = centroids.shape[0]
        self.linear = nn.Linear(embedding_size, num_clusters)
        with torch.no_grad():
            self.linear.weight = nn.Parameter(torch.Tensor(centroids))
            self.linear.weight.requires_grad_(False)
            self.linear.bias = nn.Parameter(
                torch.zeros(
                    [
                        num_clusters,
                    ]
                )
            )
            self.linear.bias.requires_grad_(False)

    def forward(self, features, pseudo_labels):
        return self.criterion(self.linear(features / self.tau), pseudo_labels)

    def reset_weights(self, centroid, cluster_idx):
        self.linear.weight[cluster_idx].copy_(centroid)




def unfurl(list_of_dicts):
    """
    Unfurl a list of dictionaries into a single dictionary.
    """
    unfurled = {}
    for d in list_of_dicts:
        for k, v in d.items():
            if k not in unfurled:
                unfurled[k] = []
            unfurled[k].append(v)
    return unfurled


_BASICTRAINER_ACCUM = os.environ.get(
    "HITL_BASICTRAINER_ACCUM", ""
).lower() in ("1", "true", "yes")


class CustomTrainer(BasicTrainer):
    def __init__(
        self,
        dataset,
        model,
        objective,
        optimizer,
        epochs,
        scheduler=None,
        device="cuda",
        batch_size=128,
        num_workers=1,
        accumulation_steps=1,
        epoch_callback=None,
        ml_pairs=[],
        cl_pairs=[],
        eps=0.05,
        min_samples=5,
        metric="cosine",
        # MAKESHIFT(skyline): re-clustering at the end of each epoch is the AL
        # behaviour and is what the --skyline path must NOT do. See the guard in
        # train_epoch below. Default True keeps every existing caller unchanged.
        update_pseudo_labels=True,
        # sampler=None,
        # momentum=0.2,
    ):
        self.update_pseudo_labels = update_pseudo_labels
        super().__init__(
            dataset,
            model,
            objective,
            optimizer,
            epochs,
            scheduler,
            device,
            batch_size,
            num_workers,
            accumulation_steps,
            epoch_callback,
        )
        self.ml_pairs = ml_pairs
        self.cl_pairs = cl_pairs
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def train(self):
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

        self.train_logs = []
        # iterate over dataset.metadata and create map from original index to pseudo label
        # self.pseudo_labels = np.full(len(self.dataset), -1, dtype=int)
        self.pseudo_labels = {}
        for idx, row in self.dataset.metadata.iterrows():
            original_index = row["original_index"]
            identity = row["pseudo_identity"]
            self.pseudo_labels[original_index] = identity

        for e in range(self.epochs):
            epoch_data = self.train_epoch(loader)
            self.train_logs.append(epoch_data)
            self.epoch += 1

            if self.epoch_callback:
                self.epoch_callback(trainer=self, epoch_data=epoch_data)

        self.train_logs = unfurl(self.train_logs)
        return self.train_logs

    def train_epoch(self, loader):
        epoch_data = {
            "train_loss_epoch_avg": 0,
            "train_acc": [],
            "epoch": self.epoch,
            "pseudo_labels": self.pseudo_labels,
        }
        model = self.model.train()
        losses = []
        last_i = -1
        for i, batch in enumerate(
            tqdm(loader, desc=f"Epoch {self.epoch}: ", mininterval=1, ncols=100)
        ):
            # x, y = batch
            idx, x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            # update y using pseudo labels
            y_dtype = y.dtype
            y = [self.pseudo_labels[i_] for i_ in idx.cpu().numpy()]
            y = torch.tensor(y, dtype=y_dtype, device=self.device)

            out = model(x)
            loss = self.objective(out, y)
            # HITL_BASICTRAINER_ACCUM=1 restores BasicTrainer's accumulation semantics:
            # scale the loss, step on (i+1) % N, and flush leftover gradients at epoch end.
            # Off by default. Tested as a cause of the skyline overshoot -- it is not one.
            if _BASICTRAINER_ACCUM:
                (loss / self.accumulation_steps).backward()
                step_now = (i + 1) % self.accumulation_steps == 0
            else:
                loss.backward()
                step_now = (i - 1) % self.accumulation_steps == 0
            if step_now:
                self.optimizer.step()
                self.optimizer.zero_grad()
            last_i = i

            losses.append(loss.detach().cpu())

            epoch_data["train_acc"].append(
                [0.0]
            )  # Placeholder for accuracy, as we are not computing it for now
            model.train()

        if self.scheduler:
            self.scheduler.step()

        epoch_data["train_loss_epoch_avg"] = np.mean(losses).astype(float)
        epoch_data["train_acc_epoch_avg"] = np.mean(epoch_data["train_acc"]).astype(
            float
        )

        # MAKESHIFT(skyline): The block below re-clusters the whole dataset at the end of
        # EVERY epoch and overwrites self.pseudo_labels (and dataset.metadata['identity'])
        # with ConSam/DBSCAN cluster ids -- which contain -1 for outliers. That is correct
        # for the AL path, where refreshing pseudo-labels each epoch is the entire point.
        # It is wrong for --skyline, which is fully supervised: after epoch 0 the real
        # labels are discarded and epoch 1 feeds -1 into nll_loss.
        #
        # CAUTION: skipping this is a WORKAROUND, not a validated reproduction of the
        # authors' skyline -- this path cannot have been run as shipped. Treat skyline
        # numbers as indicative only.
        if _BASICTRAINER_ACCUM and last_i >= 0 and (last_i + 1) % self.accumulation_steps:
            self.optimizer.step()   # flush leftover gradients, as BasicTrainer does
            self.optimizer.zero_grad()

        if not self.update_pseudo_labels:
            return epoch_data

        # at the end of each epoch, update pseudo_labels using trained model
        self.model.eval()
        with torch.no_grad():
            all_features = []
            all_labels = []
            all_indices = []
            for batch in tqdm(
                loader, desc="Computing feats to update PLs: ", mininterval=1, ncols=100
            ):
                idx, x, y = batch
                all_indices.append(idx.cpu())
                x = x.to(self.device)
                features = self.model(x)
                all_features.append(features.cpu())
                all_labels.append(y.cpu())

            all_features = torch.cat(all_features, dim=0).to(self.device)
            all_labels = torch.cat(all_labels, dim=0)
            all_indices = torch.cat(all_indices, dim=0).numpy()

            # clustering
            pseudo_identity = constrained_clustering(
                X=all_features.cpu().numpy(),
                method="ConSam",
                eps=self.eps,
                min_samples=self.min_samples,
                metric=self.metric,
                ml_pairs=self.ml_pairs,
                cl_pairs=self.cl_pairs,
            )

            # Update pseudo labels
            pseudo_labels = pseudo_identity.copy()
            # map pseudo_labels according to the original indices
            pseudo_labels_mapped = {}
            for idx, label in zip(all_indices, pseudo_labels):
                pseudo_labels_mapped[idx] = label
            self.pseudo_labels = pseudo_labels_mapped
            self.dataset.metadata["identity"] = self.pseudo_labels

        return epoch_data


class CustomWildlifeDataset(WildlifeDataset):
    def __init__(
        self,
        metadata,
        root=None,
        transform=None,
        img_load="full",
        col_path="path",
        col_label="identity",
        load_label=True,
    ):
        super().__init__(
            metadata,
            root,
            transform,
            img_load,
            col_path,
            col_label,
            load_label,
        )

    def __getitem__(self, idx):
        if self.load_label:
            img, label = super().__getitem__(idx)
            return self.metadata.iloc[idx]["original_index"], img, label
        img = super().__getitem__(idx)
        return self.metadata.iloc[idx]["original_index"], img


def mean(x, idx=None):
    if idx is None:
        return np.mean(list(x.values()))
    else:
        return np.mean([x[i] for i in idx])


def compute_predictions(
    features_query: np.ndarray,
    features_database: np.ndarray,
    ignore: Optional[List[List[int]]] = None,
    matcher: Callable = CosineSimilarity,
    k: int = 4,
    return_score: bool = False,
):
    """Computes a closest match in the database for each vector in the query set.

    Args:
        features_query (np.ndarray): Query features of size n_query*n_feature.
        features_database (np.ndarray): Database features of size n_database*n_feature
        ignore (Optional[List[List[int]]], optional): `ignore[i]` is a list of indices
            in the database ignores for i-th query.
        matcher (Callable, optional): function computing similarity.
        k (int, optional): Returned number of predictions.
        return_score (bool, optional): Whether the similalarity is returned.

    Returns:
        Vector of size (n_query,) and array of size (n_query,k). The latter are indices
            in the database for the closest matches (with ignored `ignore` indices).
            If `return_score`, it also returns an array of size (n_query,k) of scores.
    """

    # Create batch chunks
    n_query = len(features_query)
    # If ignore is not provided, initialize as empty
    if ignore is None:
        ignore = [[] for _ in range(n_query)]

    idx_true = np.array(range(n_query))
    idx_pred = np.zeros((n_query, k), dtype=np.int32)
    scores = np.zeros((n_query, k))
    # Compute the cosine similarity between the query and the database

    similarity = matcher()
    similarity = similarity(features_query, features_database)
    # Set -infinity for ignored indices
    for i in range(len(ignore)):
        similarity[i, ignore[i]] = -np.inf
    # Find the closest matches (k highest values)
    idx_pred = (-similarity).argsort(axis=-1)[:, :k]
    if return_score:
        scores = np.take_along_axis(similarity, idx_pred, axis=-1)
        return idx_true, idx_pred, scores
    else:
        return idx_true, idx_pred






def unify_types(
    y_true: List,
    y_pred: List,
    new_class: Optional[Union[int, str]] = None,
) -> Tuple[List, List, Union[int, str]]:
    """Unifies label types.

    If `new_class` is string and labels integers (or the other way round),
    it converts them to integers to allow to us numpy arrays.
    If it is `None`, it only checks for compatibility of `y_true` and `y_pred`.

    Args:
        y_true (List): List of true labels.
        y_pred (List): List of predictions.
        new_class (Optional[Union[int, str]]): Name of the new class.

    Returns:
        Unified label types.
    """

    if new_class is not None:
        y_all = list(set(y_true).union(set(y_pred)) - set([new_class]))
    else:
        y_all = list(set(y_true).union(set(y_pred)))
    is_integer = pd.api.types.is_integer_dtype(pd.Series(y_all))
    is_string = pd.api.types.is_string_dtype(pd.Series(y_all))
    if not is_integer and not is_string:
        raise (Exception("Labels have mixed types. Convert all to int or str."))
    if is_string and isinstance(new_class, int):
        encoder = {new_class: new_class}
        for i, y in enumerate(y_all):
            encoder[y] = new_class + 1 + i
        y_true = [encoder[y] for y in y_true]
        y_pred = [encoder[y] for y in y_pred]
    if is_integer and isinstance(new_class, str):
        encoder = {new_class: np.min(y_all) - 1}
        for i, y in enumerate(y_all):
            encoder[y] = y
        y_true = [encoder[y] for y in y_true]
        y_pred = [encoder[y] for y in y_pred]
        new_class = encoder[new_class]
    return y_true, y_pred, new_class


def auc_roc_new_class(
    y_true: List,
    y_score: List,
    new_class: Union[int, str],
) -> float:
    """Computes the area under ROC curve for detecting new individuals.

    High score suggests known individual.
    Low score suggests new (unknown) individual.
    It converts the problem to binary (is it new individual or not)
    and computes AUC for it.

    Args:
        y_true (List): List of true labels.
        y_score (List): List of scores.
        new_class (Union[int, str]): Name of the new class.

    Returns:
        Computed arena under ROC curve.
    """

    y_true, _, new_class = unify_types(y_true, [], new_class)
    a = np.array(y_true) == new_class
    b = -np.array(y_score)
    return roc_auc_score(a, b)


def _subset_features(feature_ds, positions, expect_metadata=None):
    """Slice an already-extracted FeatureDataset instead of re-encoding the images.

    O2/O6: every image in the evaluation was being encoded two to four times. The global
    block encodes closed_query + query + gallery, then the per-dataset loop re-encodes
    each of those again as `dataset == <name>` row subsets, and closed_query is itself a
    row subset of query. Since wildlife_tools builds its loader with shuffle=False and
    `WildlifeDataset.__init__` resets the metadata index, feature row i corresponds to
    metadata row i, so every one of those subsets is a positional slice of a global
    extraction.

    `expect_metadata` is checked rather than trusted: if the slice ever stops matching
    the frame the old code would have encoded, this raises instead of quietly changing
    a published number.
    """
    from wildlife_tools.data import FeatureDataset

    positions = np.asarray(positions, dtype=int)
    if expect_metadata is not None:
        got = feature_ds.metadata.iloc[positions].reset_index(drop=True)
        exp = expect_metadata.reset_index(drop=True)
        for col in ("path", "identity"):
            if col in exp.columns and not got[col].equals(exp[col]):
                raise AssertionError(
                    f"_subset_features: positional slice disagrees on '{col}' "
                    f"({len(got)} rows vs {len(exp)})"
                )
    return FeatureDataset(
        features=feature_ds.features[positions],
        metadata=feature_ds.metadata.iloc[positions],
    )


def _json_default(o):
    """Make numpy scalars/arrays JSON-serialisable.

    Under numpy 1 most of these were `np.float64`, which subclasses Python `float`, so
    `json.dump` accepted them silently. NEP 50 in numpy 2 keeps `float32` as `float32`
    (no longer a `float` subclass) and `json.dump` raises
    "Object of type float32 is not JSON serializable". Converting here keeps the written
    values identical while removing the dependency on that promotion accident.
    """
    import numpy as _np

    if isinstance(o, _np.generic):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _subset_name(dataset):
    """Name of the wildlife subset, for either dataset class.

    The repo-root `WildlifeDatasets` sets `self.dataset`; the SpCL `Wildlife` sets
    `self.subset`. Neither sets both, so this is unambiguous -- and it is the single
    semantic difference that kept these two modules forked (Todo.md C3).
    """
    for attr in ("dataset", "subset"):
        v = getattr(dataset, attr, None)
        if v is not None:
            return v
    raise AttributeError("dataset object exposes neither .dataset nor .subset")


def get_evaluation_results(megadescriptor_extractor, dataset, folder, epoch):
    """Evaluate, then put the extractor's model back on the device it started on.

    wildlife_tools' DeepFeatures.__call__ ends with `self.model.to("cpu")` on the caller's
    own model, stranding it on the CPU after every eval. Handling it here covers all call
    sites; device placement changes no computed value.
    """
    _model = getattr(megadescriptor_extractor, "model", None)
    try:
        _device = next(_model.parameters()).device if _model is not None else None
    except StopIteration:
        _device = None
    try:
        return _get_evaluation_results(
            megadescriptor_extractor, dataset, folder, epoch
        )
    finally:
        if _device is not None:
            megadescriptor_extractor.model = _model.to(_device)


def _get_evaluation_results(megadescriptor_extractor, dataset, folder, epoch):
    import evaluation
    import copy
    import json
    from wildlife_datasets.metrics import BAKS, BAUS

    results = {
        "Top-1 Accuracy": [],
        "Top-3 Accuracy": [],
        "Top-5 Accuracy": [],
        "Top-10 Accuracy": [],
        "mAP": [],
        "mINP": [],
        "BAKS": [],
        "BAUS@0.7": [],
        "Normalized Accuracy@0.7": [],
        "ROC AUC": [],
    }
    step = 0.01
    # Only baks[-inf], baks[0.7] and baus[0.7] are ever read back, so sweeping all 103
    # thresholds computed and discarded 101 of them. HITL_FULL_THRESHOLD_SWEEP=1 restores
    # the full curve for plotting.
    if os.environ.get("HITL_FULL_THRESHOLD_SWEEP", "").lower() in ("1", "true", "yes"):
        ts = [-np.inf] + list(np.round(np.arange(0, 1 + step / 10, step), 2)) + [np.inf]
    else:
        ts = [-np.inf, 0.7]
    new_individual = "new_individual"

    # run on each dataset
    if _subset_name(dataset) != "WII":
        # also get micro-averaged results
        default_query_features = megadescriptor_extractor(dataset.default_query)
        default_gallery_features = megadescriptor_extractor(dataset.default_gallery)
        # O2: closed query is a row subset of query, so slice it out rather than
        # re-encoding ~4,600 images (9% of eval extraction time).
        _closed_pos = dataset.default_query.metadata.index[
            dataset.default_query.metadata["identity"].isin(
                dataset.default_closed_query.metadata["identity"].unique()
            )
        ].values
        default_closed_query_features = _subset_features(
            default_query_features, _closed_pos, dataset.default_closed_query.metadata
        )

        idx_train = dataset.default_gallery.metadata.index.values
        idx_test = dataset.default_query.metadata.index.values
        similarity = dataset.matcher(
            query=default_closed_query_features, database=default_gallery_features
        )
        cmc, all_AP, all_INP = evaluation.eval_wildlife(
            1 - similarity,
            dataset.default_closed_query.metadata.identity.values,
            dataset.default_gallery.metadata.identity.values,
            max_rank=10,
        )
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        baks = {}
        baus = {}

        idx_true, idx_pred, scores = compute_predictions(
            default_query_features, default_gallery_features, return_score=True
        )
        idx_true = idx_test[idx_true]
        idx_pred = idx_train[idx_pred]
        idx_pred = idx_pred[:, 0]
        scores = scores[:, 0]

        y_true = dataset.default_query.metadata["identity"].iloc[idx_true]
        y_true_w_new = np.array(
            [
                (
                    y
                    if y in dataset.default_gallery.metadata["identity"].values
                    else new_individual
                )
                for y in y_true
            ]
        )
        y_pred_closed = dataset.default_gallery.metadata["identity"].iloc[idx_pred]
        identity_test_only = list(
            set(dataset.default_query.metadata["identity"].iloc[idx_test])
            - set(dataset.default_gallery.metadata["identity"].iloc[idx_train])
        )
        for t in tqdm(ts):
            y_pred = copy.copy(y_pred_closed)
            # y_pred = y_pred.astype(str)
            # y_true = y_true.astype(str)
            with np.errstate(invalid="ignore"):
                y_pred[scores < t] = new_individual
            baks[t] = BAKS(y_true, y_pred, identity_test_only)
            baus[t] = BAUS(y_true, y_pred, identity_test_only, new_individual)
        roc_score = auc_roc_new_class(y_true_w_new, scores, new_individual)
        print("Top 1 accuracy:", (float(cmc[0]) * 100))
        print("Top 3 accuracy:", (float(cmc[2]) * 100))
        print("Top 5 accuracy:", (float(cmc[4]) * 100))
        print("Top 10 accuracy:", (float(cmc[9]) * 100))
        print("mAP:", mAP)
        print("mINP:", mINP)
        print(f"BAKS = {np.mean(baks[-np.inf])}")
        print(f"BAUS@0.7 = {np.mean(baus[0.7])}")
        print(
            f"Normalized accuracy@0.7 = {np.sqrt(np.mean(baks[0.7]) * np.mean(baus[0.7]))}"
        )
        print(f"ROC AUC score = {roc_score}")

        print(
            f"{_subset_name(dataset)},{cmc[0] * 100:.2f},{cmc[2] * 100:.2f},{cmc[4] * 100:.2f},{cmc[9] * 100:.2f},{mAP:.4f},{mINP:.4f},{np.mean(baks[-np.inf]):.4f},{np.mean(baus[0.7]):.4f},{np.sqrt(np.mean(baks[0.7]) * np.mean(baus[0.7])):.4f},{roc_score:.4f}"
        )

        micro_results = {
            "Top-1 Accuracy": (float(cmc[0]) * 100),
            "Top-3 Accuracy": (float(cmc[2]) * 100),
            "Top-5 Accuracy": (float(cmc[4]) * 100),
            "Top-10 Accuracy": (float(cmc[9]) * 100),
            "mAP": mAP,
            "mINP": mINP,
            "BAKS": np.mean(baks[-np.inf]),
            "BAUS@0.7": np.mean(baus[0.7]),
            "Normalized Accuracy@0.7": np.sqrt(np.mean(baks[0.7]) * np.mean(baus[0.7])),
            "ROC AUC": roc_score,
        }
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, f"micro_metrics_{epoch+1}.json"), "w") as fp:
            json.dump(micro_results, fp, default=_json_default)

        # keep the global extractions alive for the per-dataset loop below (O6)
        _global_query_features = default_query_features
        _global_gallery_features = default_gallery_features
        _global_closed_features = default_closed_query_features
        _global_closed_meta = dataset.default_closed_query.metadata

        for dataset_name in dataset.csv["dataset"].unique():
            filtered_closed_query = dataset.default_closed_query.metadata.query(
                "dataset == @dataset_name"
            )
            dataset_filtered_closed_query = WildlifeDataset(
                metadata=filtered_closed_query,
                root=dataset.image_root,
                transform=dataset.test_transform,
            )
            _cq_pos = _global_closed_meta.index[
                _global_closed_meta["dataset"] == dataset_name
            ].values
            default_closed_query_features = _subset_features(
                _global_closed_features, _cq_pos, filtered_closed_query
            )

            filtered_query = dataset.default_query.metadata.query(
                "dataset == @dataset_name"
            )
            dataset_filtered_query = WildlifeDataset(
                metadata=filtered_query,
                root=dataset.image_root,
                transform=dataset.test_transform,
            )
            default_query_features = _subset_features(
                _global_query_features, filtered_query.index.values, filtered_query
            )

            filtered_gallery = dataset.default_gallery.metadata.query(
                "dataset == @dataset_name"
            )
            dataset_filtered_gallery = WildlifeDataset(
                metadata=filtered_gallery,
                root=dataset.image_root,
                transform=dataset.test_transform,
            )
            default_gallery_features = _subset_features(
                _global_gallery_features,
                filtered_gallery.index.values,
                filtered_gallery,
            )

            idx_train = dataset_filtered_gallery.metadata.index.values
            idx_test = dataset_filtered_query.metadata.index.values
            similarity = dataset.matcher(
                query=default_closed_query_features, database=default_gallery_features
            )

            cmc, all_AP, all_INP = evaluation.eval_wildlife(
                1 - similarity,
                dataset_filtered_closed_query.metadata.identity.values,
                dataset_filtered_gallery.metadata.identity.values,
                max_rank=10,
            )

            mAP = np.mean(all_AP)
            mINP = np.mean(all_INP)

            baks = {}  # Balanced Accuracy on Known Samples
            baus = {}  # Balanced Accuracy on Unknown Samples

            idx_true, idx_pred, scores = compute_predictions(
                default_query_features, default_gallery_features, return_score=True
            )
            idx_true = idx_test[idx_true]
            idx_pred = idx_train[idx_pred]
            idx_pred = idx_pred[:, 0]
            scores = scores[:, 0]

            y_true = dataset_filtered_query.metadata["identity"].iloc[idx_true]
            y_true_w_new = np.array(
                [
                    (
                        y
                        if y in dataset_filtered_gallery.metadata["identity"].values
                        else new_individual
                    )
                    for y in y_true
                ]
            )
            y_pred_closed = dataset_filtered_gallery.metadata["identity"].iloc[idx_pred]

            identity_test_only = list(
                set(dataset_filtered_query.metadata["identity"].iloc[idx_test])
                - set(dataset_filtered_gallery.metadata["identity"].iloc[idx_train])
            )

            for t in tqdm(ts):
                y_pred = copy.copy(y_pred_closed)
                with np.errstate(invalid="ignore"):
                    y_pred[scores < t] = new_individual
                baks[t] = BAKS(y_true, y_pred, identity_test_only)
                baus[t] = BAUS(y_true, y_pred, identity_test_only, new_individual)

            roc_score = auc_roc_new_class(y_true_w_new, scores, new_individual)

            print(
                f"{dataset_name},{cmc[0] * 100:.2f},{cmc[2] * 100:.2f},{cmc[4] * 100:.2f},{cmc[9] * 100:.2f},{mAP:.4f},{mINP:.4f},{np.mean(baks[-np.inf]):.4f},{np.mean(baus[0.7]):.4f},{np.sqrt(np.mean(baks[0.7]) * np.mean(baus[0.7])):.4f},{roc_score:.4f}"
            )

            subset_results = {
                "Top-1 Accuracy": (float(cmc[0]) * 100),
                "Top-3 Accuracy": (float(cmc[2]) * 100),
                "Top-5 Accuracy": (float(cmc[4]) * 100),
                "Top-10 Accuracy": (float(cmc[9]) * 100),
                "mAP": mAP,
                "mINP": mINP,
                "BAKS": np.mean(baks[-np.inf]),
                "BAUS@0.7": np.mean(baus[0.7]),
                "Normalized Accuracy@0.7": np.sqrt(
                    np.mean(baks[0.7]) * np.mean(baus[0.7])
                ),
                "ROC AUC": roc_score,
            }

            for key, value in subset_results.items():
                results[key].append(value)

        print(
            f"mean,{np.mean(results['Top-1 Accuracy']):.2f},{np.mean(results['Top-3 Accuracy']):.2f},{np.mean(results['Top-5 Accuracy']):.2f},{np.mean(results['Top-10 Accuracy']):.2f},{np.mean(results['mAP']):.4f},{np.mean(results['mINP']):.4f},{np.mean(results['BAKS']):.4f},{np.mean(results['BAUS@0.7']):.4f},{np.sqrt(np.mean(results['BAKS']) * np.mean(results['BAUS@0.7'])):.4f},{np.mean(results['ROC AUC']):.4f}"
        )
        print(
            f"std,{np.std(results['Top-1 Accuracy']):.2f},{np.std(results['Top-3 Accuracy']):.2f},{np.std(results['Top-5 Accuracy']):.2f},{np.std(results['Top-10 Accuracy']):.2f},{np.std(results['mAP']):.4f},{np.std(results['mINP']):.4f},{np.std(results['BAKS']):.4f},{np.std(results['BAUS@0.7']):.4f},{np.sqrt(np.std(results['BAKS']) * np.std(results['BAUS@0.7'])):.4f},{np.std(results['ROC AUC']):.4f}"
        )

    else:
        step = 0.01
        # See above: only two thresholds are read back.
        if os.environ.get("HITL_FULL_THRESHOLD_SWEEP", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            ts = (
                [-np.inf]
                + list(np.round(np.arange(0, 1 + step / 10, step), 2))
                + [np.inf]
            )
        else:
            ts = [-np.inf, 0.7]
        new_individual = "new_individual"
        new_individual = -1

        print("Extracting features for default closed query...")
        # filter dataset
        default_closed_query_features = megadescriptor_extractor(
            dataset.default_closed_query
        )
        print("Extracting features for default query...")
        default_query_features = megadescriptor_extractor(dataset.default_query)
        print("Extracting features for default gallery...")
        default_gallery_features = megadescriptor_extractor(dataset.default_gallery)

        idx_train = dataset.default_gallery.metadata.index.values
        idx_test = dataset.default_query.metadata.index.values
        similarity = dataset.matcher(
            query=default_closed_query_features, database=default_gallery_features
        )

        cmc, all_AP, all_INP = evaluation.eval_wildlife(
            1 - similarity,
            dataset.default_closed_query.metadata.identity.values,
            dataset.default_gallery.metadata.identity.values,
            max_rank=10,
        )
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        baks = {}
        baus = {}

        idx_true, idx_pred, scores = compute_predictions(
            default_query_features, default_gallery_features, return_score=True
        )
        idx_true = idx_test[idx_true]
        idx_pred = idx_train[idx_pred]
        idx_pred = idx_pred[:, 0]
        scores = scores[:, 0]

        y_true = dataset.default_query.metadata["identity"].iloc[idx_true]
        y_true_w_new = np.array(
            [
                (
                    y
                    if y in dataset.default_gallery.metadata["identity"].values
                    else new_individual
                )
                for y in y_true
            ]
        )
        y_pred_closed = dataset.default_gallery.metadata["identity"].iloc[idx_pred]
        identity_test_only = list(
            set(dataset.default_query.metadata["identity"].iloc[idx_test])
            - set(dataset.default_gallery.metadata["identity"].iloc[idx_train])
        )
        for t in tqdm(ts):
            y_pred = copy.copy(y_pred_closed)
            # y_pred = y_pred.astype(str)
            # y_true = y_true.astype(str)
            with np.errstate(invalid="ignore"):
                y_pred[scores < t] = new_individual
            baks[t] = BAKS(y_true, y_pred, identity_test_only)
            baus[t] = BAUS(y_true, y_pred, identity_test_only, new_individual)
        roc_score = auc_roc_new_class(y_true_w_new, scores, new_individual)
        print("Top 1 accuracy:", (float(cmc[0]) * 100))
        print("Top 3 accuracy:", (float(cmc[2]) * 100))
        print("Top 5 accuracy:", (float(cmc[4]) * 100))
        print("Top 10 accuracy:", (float(cmc[9]) * 100))
        print("mAP:", mAP)
        print("mINP:", mINP)
        print(f"BAKS = {np.mean(baks[-np.inf])}")
        print(f"BAUS@0.7 = {np.mean(baus[0.7])}")
        print(
            f"Normalized accuracy@0.7 = {np.sqrt(np.mean(baks[0.7]) * np.mean(baus[0.7]))}"
        )
        print(f"ROC AUC score = {roc_score}")

        print(
            f"{_subset_name(dataset)},{cmc[0] * 100:.2f},{cmc[2] * 100:.2f},{cmc[4] * 100:.2f},{cmc[9] * 100:.2f},{mAP:.4f},{mINP:.4f},{np.mean(baks[-np.inf]):.4f},{np.mean(baus[0.7]):.4f},{np.sqrt(np.mean(baks[0.7]) * np.mean(baus[0.7])):.4f},{roc_score:.4f}"
        )

        results = {
            "Top-1 Accuracy": (float(cmc[0]) * 100),
            "Top-3 Accuracy": (float(cmc[2]) * 100),
            "Top-5 Accuracy": (float(cmc[4]) * 100),
            "Top-10 Accuracy": (float(cmc[9]) * 100),
            "mAP": mAP,
            "mINP": mINP,
            "BAKS": np.mean(baks[-np.inf]),
            "BAUS@0.7": np.mean(baus[0.7]),
            "Normalized Accuracy@0.7": np.sqrt(np.mean(baks[0.7]) * np.mean(baus[0.7])),
            "ROC AUC": roc_score,
        }

        micro_results = results

    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"evaluation_metrics_{epoch+1}.json"), "w") as fp:
        json.dump(results, fp, default=_json_default)
    return micro_results


""" Borrowed from Zhong et al. (https://github.com/yxgeee/SpCL) """
import os
import time
import numpy as np
from scipy.spatial.distance import cdist
import faiss

import torch
import torch.nn.functional as F

from faiss_utils import (
    search_index_pytorch,
    search_raw_array_pytorch,
    index_init_gpu,
    index_init_cpu,
)


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, : k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def compute_jaccard_distance(
    target_features, k1=20, k2=6, print_flag=True, search_option=0, use_float16=False
):
    end = time.time()
    if print_flag:
        print("Computing jaccard distance...")

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if search_option == 0:
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(
            res, target_features, target_features, k1
        )
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 1:
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 2:
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if len(
                np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(
            k_reciprocal_expansion_index
        )  ## element-wise unique
        dist = 2 - 2 * torch.mm(
            target_features[i].unsqueeze(0).contiguous(),
            target_features[k_reciprocal_expansion_index].t(),
        )
        if use_float16:
            V[i, k_reciprocal_expansion_index] = (
                F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
            )
        else:
            V[i, k_reciprocal_expansion_index] = (
                F.softmax(-dist, dim=1).view(-1).cpu().numpy()
            )

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = jaccard_dist < 0
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time() - end))

    return jaccard_dist
