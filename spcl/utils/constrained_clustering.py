from __future__ import print_function, absolute_import
import numpy as np

import itertools
import networkx as nx


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


def constrained_clustering(
    labels,
    rerank_dist,
    method,
    min_samples=5,
    metric="precomputed",
    ml_pairs=[],
    cl_pairs=[],
    sanity_check=False,
):
    if len(ml_pairs) == 0 and len(cl_pairs) == 0:
        return labels

    if method == "DBSCAN":
        labels[labels == -1] = np.arange(
            labels.max() + 1, labels.max() + 1 + sum(labels == -1)
        )

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
        clust_point_pairs = list(itertools.combinations(clust_points, 2))
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
                # import pdb;
                # pdb.set_trace()
            assert (
                labels[a] == labels[b]
            ), "must-link constraint violated. Implementation error"
        for a, b in cl_pairs:
            if labels[a] == labels[b] and labels[a] != -1:
                print(f"{a}, {b}, {labels[[a, b]]}")
                # import pdb;
                # pdb.set_trace()
            assert (
                labels[a] != labels[b] or labels[a] == -1
            ), "cannot-link constraint violated. Implementation error"
    return labels
