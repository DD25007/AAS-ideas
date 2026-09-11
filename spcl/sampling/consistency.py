from .base import BaseSampling
import numpy as np
from spcl.utils.constrained_clustering import constrained_clustering
import networkx as nx
import itertools
from collections import defaultdict
from sklearn.cluster import DBSCAN
from spcl.utils.constrained_clustering import FINCH
import os

# This sampler turns sets of *string*-named graph nodes into ordered lists, and Python
# randomises string hashing per process, so which coarse cluster the annotation budget
# runs out on varies run to run. The launcher pins PYTHONHASHSEED=0; setting
# HITL_SORT_COMPONENTS=1 fixes the order in code instead. Off by default: sorting picks
# a different (equally valid) order and would move published numbers.
_SORT_COMPONENTS = os.environ.get("HITL_SORT_COMPONENTS", "0") == "1"


def _node_key(node):
    """Sort key for 'dbscan-12' / 'finch-3' that orders numerically, not lexically."""
    prefix, _, idx = node.rpartition("-")
    return (prefix, int(idx))


def _ordered(nodes):
    """Sorted list when the fix is on; otherwise the original unordered set."""
    return sorted(nodes, key=_node_key) if _SORT_COMPONENTS else set(nodes)


def _ordered_components(graph):
    comps = nx.connected_components(graph)
    if not _SORT_COMPONENTS:
        return comps
    return sorted((sorted(c, key=_node_key) for c in comps), key=lambda c: _node_key(c[0]))


def compute_var(F_vectors):
    covariance_matrix = np.cov(F_vectors, rowvar=False)
    variance_of_points = np.trace(covariance_matrix)
    return variance_of_points


class ConsistencySampling(BaseSampling):
    def __init__(self, thresh=0.05, min_samples=3, metric="cosine", num_gt_ids_gallery=None, args=None):
        super().__init__(args=args)
        self.thresh = thresh
        self.min_samples = min_samples
        self.metric = metric
        self.num_gt_ids_gallery = num_gt_ids_gallery

    def select(self, features=None, rerank_dist=None, **kwargs):
        # [record] reset per-cycle instrumentation
        self.rec_n_dbscan_raw = self.rec_n_dbscan_aas = None
        self.rec_n_finch = self.rec_n_regions = None
        self.rec_cov = None
        self.rec_shortfall = 0
        sim_thresh = self.args.consam_sim_thresh
        k_medoids = self.args.consam_k
        try:
            assert features is not None and rerank_dist is not None
            features = features.to('cpu').numpy()
            n_samples = features.shape[0]

            # Get Density-based (DBSCAN) clusters
            # "clusters_dbscan" contains information about outliers while
            # "labels_dbscan" represent each outlier as a singleton cluster
            cluster = DBSCAN(eps=self.thresh, min_samples=4, metric='precomputed', n_jobs=-1)
            labels = cluster.fit_predict(rerank_dist)
            clusters_dbscan = constrained_clustering(
                labels=labels,
                rerank_dist=rerank_dist,
                method="DBSCAN",
                min_samples=self.min_samples,
                metric=self.metric,
                ml_pairs=self.ml_pairs,
                cl_pairs=self.cl_pairs,
            )
            labels_dbscan = clusters_dbscan.copy()
            labels_dbscan[labels_dbscan == -1] = np.arange(labels_dbscan.max() + 1, labels_dbscan.max() + 1 + sum(labels_dbscan == -1))
            # [record] pure instrumentation: counts only, no behaviour change
            self.rec_n_dbscan_raw = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
            self.rec_n_dbscan_aas = int(len(np.unique(labels_dbscan)))

            # Get Nearest Neighbor-based (FINCH) clusters
            rerank_dist_cp = rerank_dist.copy()
            np.fill_diagonal(rerank_dist_cp, 1e12)
            initial_rank = np.argmin(rerank_dist_cp, axis=1)
            labels = FINCH(features, initial_rank=initial_rank)
            labels_finch = constrained_clustering(
                labels=labels,
                rerank_dist=rerank_dist,
                method="FINCH",
                metric=self.metric,
                ml_pairs=self.ml_pairs,
                cl_pairs=self.cl_pairs,
            )
            clusters_finch = labels_finch
            # [record]
            self.rec_n_finch = int(len(np.unique(labels_finch)))


            pairwise_distances = rerank_dist

            cluster_label_maps = {
                "dbscan": {c_id: set() for c_id in np.unique(labels_dbscan)},
                "finch": {c_id: set() for c_id in np.unique(labels_finch)}
            }

            for i in range(n_samples):
                cluster_label_maps["dbscan"][labels_dbscan[i]].add(i)
                cluster_label_maps["finch"][labels_finch[i]].add(i)

            # Compute IoU between both clustering methods.
            # Clusters with IoU as 0 or 1 are the most certain, i.e., the clusters are exactly the same
            cluster_iou = np.zeros((len(cluster_label_maps["dbscan"]), len(cluster_label_maps["finch"])))
            for i, c1 in enumerate(cluster_label_maps["dbscan"]):
                for j, c2 in enumerate(cluster_label_maps["finch"]):
                    cluster_iou[i, j] = len(cluster_label_maps["dbscan"][c1].intersection(cluster_label_maps["finch"][c2])) / \
                                        len(cluster_label_maps["dbscan"][c1].union(cluster_label_maps["finch"][c2]))

            # Create a bipartite graph between DBSCAN and FINCH clusters. The same clusters (IoU = 0 or 1) are ignored.
            inconsistency_graph = nx.Graph()
            for i in range(cluster_iou.shape[0]):  # Loop over DBSCAN clusters
                for j in range(cluster_iou.shape[1]):  # Loop over FINCH clusters
                    iou = cluster_iou[i, j]
                    if iou == 0 or iou == 1:
                        continue
                    inconsistency_graph.add_node(f'dbscan-{i}', bipartite=0)
                    inconsistency_graph.add_node(f'finch-{j}', bipartite=1)
                    inconsistency_graph.add_edge(f'dbscan-{i}', f'finch-{j}', iou=iou)

            S = []  # List of coarse clusters S_k's
            inconsistent_dbscan_labels, inconsistent_finch_labels = [], []
            for i, c in enumerate(_ordered_components(inconsistency_graph)):
                S_k = inconsistency_graph.subgraph(c).copy()
                S.append(nx.create_empty_copy(S_k))
                dbscan_nodes = _ordered(
                    n for n, d in S_k.nodes(data=True) if d["bipartite"] == 0
                )
                finch_nodes = _ordered(set(S_k) - set(dbscan_nodes))
                inconsistent_dbscan_labels.append(list(int(dn.split('-')[1]) for dn in dbscan_nodes))
                inconsistent_finch_labels.append(list(int(fn.split('-')[1]) for fn in finch_nodes))

            self.rec_n_regions = len(S)  # [record]
            sampling_pool = defaultdict(list)
            for k, S_k in enumerate(S):
                sampling_pool["S_k"].append(set())
                # Add bridges between DBSCAN clusters
                for label1 in inconsistent_dbscan_labels[k]:
                    row_indices = list(cluster_label_maps["dbscan"][label1])
                    sampling_pool["S_k"][k].update(set(row_indices))
                    for label2 in inconsistent_dbscan_labels[k]:
                        if label1 == label2:
                            continue
                        col_indices = list(cluster_label_maps["dbscan"][label2])
                        sub_matrix = pairwise_distances[np.ix_(row_indices, col_indices)]
                        min_idx = np.unravel_index(np.argmin(sub_matrix), sub_matrix.shape)
                        min_row, min_col = row_indices[min_idx[0]], col_indices[min_idx[1]]
                        S_k.add_edge(f'dbscan-{label1}', f'dbscan-{label2}',
                                    weight=pairwise_distances[min_row, min_col],
                                    points=[min_row, min_col])
                # Add bridges between FINCH clusters
                for label1 in inconsistent_finch_labels[k]:
                    for label2 in inconsistent_finch_labels[k]:
                        if label1 == label2:
                            continue
                        row_indices = list(cluster_label_maps["finch"][label1])
                        col_indices = list(cluster_label_maps["finch"][label2])
                        sub_matrix = pairwise_distances[np.ix_(row_indices, col_indices)]
                        min_idx = np.unravel_index(np.argmin(sub_matrix), sub_matrix.shape)
                        min_row, min_col = row_indices[min_idx[0]], col_indices[min_idx[1]]
                        S_k.add_edge(f'finch-{label1}', f'finch-{label2}',
                                    weight=pairwise_distances[min_row, min_col],
                                    points=[min(min_row, min_col), max(min_row, min_col)])

                dbscan_positive_relations = []
                finch_positive_relations = []
                potential_queries = None
                # The following loop iterates twice. Once for DBSCAN and once for FINCH.
                for c in _ordered_components(S_k):
                    G = S_k.subgraph(c).copy()
                    if 'dbscan' in list(G.nodes)[0]:
                        for node in G.nodes:
                            dbscan_positive_relations.extend(
                                list(
                                    itertools.product(
                                        cluster_label_maps["dbscan"][int(node.split('-')[1])],
                                        cluster_label_maps["dbscan"][int(node.split('-')[1])])
                                )
                            )
                    if 'finch' in list(G.nodes)[0]:
                        for node in G.nodes:
                            finch_positive_relations.extend(
                                list(
                                    itertools.product(
                                        cluster_label_maps["finch"][int(node.split('-')[1])],
                                        cluster_label_maps["finch"][int(node.split('-')[1])])
                                )
                            )
                    if len(G.nodes) == 1:
                        continue
                    if potential_queries is None:
                        potential_queries = np.array(list(nx.get_edge_attributes(G, "points").values()))
                    else:
                        potential_queries = np.append(potential_queries, np.array(list(nx.get_edge_attributes(G, "points").values())), 0)

                dbscan_positive_relations = set(dbscan_positive_relations)
                finch_positive_relations = set(finch_positive_relations)

                # Code for Active Sampling
                try:
                    # \tilde{I}_k: inconsistency
                    inconsistency = np.sort(np.array(list(dbscan_positive_relations.union(finch_positive_relations) - dbscan_positive_relations.intersection(finch_positive_relations))), axis=1)
                    # \mathcal{P}_k: potential_queries
                    # I_k: pair
                    pair = np.intersect1d(np.sort(potential_queries).view([('', potential_queries.dtype)] * 2), inconsistency.view([('', inconsistency.dtype)] * 2))
                    pair = pair.view(potential_queries.dtype).reshape(-1, 2)
                except:
                    continue

                # Append the non-redundant, most informative inconsistent pairs to the sampling pool
                sampling_pool["pairs"].append(pair)

                # Get the labels of both points in each pair from DBSCAN
                labels_a_dbscan = clusters_dbscan[pair[:, 0]]
                labels_a_finch = clusters_finch[pair[:, 0]]
                labels_b_dbscan = clusters_dbscan[pair[:, 1]]
                labels_b_finch = clusters_finch[pair[:, 1]]

                # Check if the points are inliers (True) or outliers (False)
                is_inlier_a = np.logical_and(labels_a_dbscan >= 0, labels_a_finch >= 0)
                is_inlier_b = np.logical_and(labels_b_dbscan >= 0, labels_b_finch >= 0)

                # Check if the pairs satisfy the beta_i relationships
                inlier_inlier = np.logical_and(is_inlier_a, is_inlier_b)  # \beta_1
                mixed_pairs = ~inlier_inlier  # \beta_2

                # computing I_k^{beta_i}
                pair_ii = pair[inlier_inlier]
                pair_mixed = pair[mixed_pairs]

                sampling_pool["inlier_inlier_masks"].append(inlier_inlier)
                sampling_pool["pairs_ii"].append(pair_ii)
                sampling_pool["mixed_pairs_masks"].append(mixed_pairs)
                sampling_pool["pairs_mixed"].append(pair_mixed)

                # Compute the numerator part of  P(X = S_k | β = β_i)
                numerator = np.zeros(len(pair))
                if len(pair_ii) > 0:
                    numerator[inlier_inlier] = compute_var(features[np.unique(pair_ii)]) * len(pair_ii)
                if len(pair_mixed) > 0:
                    numerator[mixed_pairs] = compute_var(features[np.unique(pair_mixed)]) * len(pair_mixed)
                sampling_pool["numerators"].append(numerator)

                priors = np.zeros(len(pair))
                priors[inlier_inlier] = -1  # get masks first, then assign beta values
                priors[mixed_pairs] = 1

                sampling_pool["priors"].append(priors)

                # Compute the conditional probability of sampling a pair given its super-cluster S_k and relationship beta_i
                # P(Y = (a, b) ∈ I | X = S_k, β = β_i)
                prob = np.zeros(len(pair))
                if len(pair_ii) > 0:
                    prob[inlier_inlier] = 1 - pairwise_distances[pair_ii[:, 0], pair_ii[:, 1]]
                    prob[inlier_inlier][prob[inlier_inlier] < sim_thresh] = 0
                    prob[inlier_inlier] = prob[inlier_inlier] / prob[inlier_inlier].sum() if prob[inlier_inlier].sum() > 0 else 0
                    # prob[inlier_inlier] = log_softmax(prob[inlier_inlier])
                if len(pair_mixed) > 0:
                    prob[mixed_pairs] = 1 - pairwise_distances[pair_mixed[:, 0], pair_mixed[:, 1]]
                    prob[mixed_pairs][prob[mixed_pairs] < sim_thresh] = 0
                    prob[mixed_pairs] = prob[mixed_pairs] / prob[mixed_pairs].sum() if prob[mixed_pairs].sum() > 0 else 0
                    # prob[mixed_pairs] = log_softmax(prob[mixed_pairs])

                if np.any(np.isnan(prob)):
                    raise FloatingPointError(
                        "NaN in AAS pair probabilities; sampling would be corrupt"
                    )
                sampling_pool["probs"].append(prob)

            # Stack all the non-redundant, most-informative inconsistent pairs from all S_k's to form the sampling pool
            pairs = np.vstack(sampling_pool["pairs"])

            # Compute the denominator part of  P(X = S_k | β = β_i)
            if len(np.concatenate(sampling_pool["pairs_ii"])) > 0:
                denominator_ii = compute_var(features[np.unique(np.concatenate(sampling_pool["pairs_ii"]))]) * len(np.concatenate(sampling_pool["pairs_ii"]))
            else:
                denominator_ii = 1
            if len(np.concatenate(sampling_pool["pairs_mixed"])) > 0:
                denominator_mixed = compute_var(features[np.unique(np.concatenate(sampling_pool["pairs_mixed"]))]) * len(np.concatenate(sampling_pool["pairs_mixed"]))
            else:
                denominator_mixed = 1

            # Find pairs of non-redundant, most informative points from coarse super-clusters

            inter_cluster_sampling_pool = {
                "pairs": [],
                "sim": []
            }
            n = len(sampling_pool["S_k"])
            medoids = {}
            for k_idx, S_k in enumerate(sampling_pool["S_k"]):
                row_indices = list(S_k)
                sub_matrix_for_medoid = pairwise_distances[np.ix_(row_indices, row_indices)]
                medoid_local_idx = np.argmin(np.sum(sub_matrix_for_medoid, axis=1))
                medoids[k_idx] = row_indices[medoid_local_idx]

            # [record] images covered by regions of uncertainty
            self.rec_cov = len(set().union(*sampling_pool["S_k"])) if sampling_pool["S_k"] else 0
            for i in range(n):
                distances_to_other_medoids = []
                medoid_i_global = medoids[i]
                for j in range(n):
                    if i == j:
                        continue
                    medoid_j_global = medoids[j]
                    distances_to_other_medoids.append(pairwise_distances[medoid_i_global, medoid_j_global])
                k_medoids = 5
                closest_k = np.argsort(distances_to_other_medoids)[:k_medoids]
                for j in range(k_medoids):
                    medoid_j_global = medoids[closest_k[j]]
                    sim = 1 - pairwise_distances[medoid_i_global, medoid_j_global]
                    if sim < sim_thresh:
                        continue
                    inter_cluster_sampling_pool["pairs"].append([medoid_i_global, medoid_j_global])
                    inter_cluster_sampling_pool["sim"].append(1 - pairwise_distances[medoid_i_global, medoid_j_global])

            #  P(Y = (a, b) ∈ \mathcal{P})
            inter_cluster_sampling_pool["sim"] = np.array(inter_cluster_sampling_pool["sim"])
            p_cross = inter_cluster_sampling_pool["sim"] / inter_cluster_sampling_pool["sim"].sum()
            cross_pairs = np.array(inter_cluster_sampling_pool["pairs"])

            count_inlier_inlier = np.sum(np.concatenate(sampling_pool["inlier_inlier_masks"]))
            count_mixed_pairs = np.sum(np.concatenate(sampling_pool["mixed_pairs_masks"]))
            # count_outlier_outlier = np.sum(np.concatenate(sampling_pool["outlier_outlier_masks"]))
            self.count_inlier_inlier = count_inlier_inlier
            self.count_mixed_pairs = count_mixed_pairs
            # self.count_outlier_outlier = count_outlier_outlier

            beta1 = count_inlier_inlier / (count_inlier_inlier + count_mixed_pairs)
            beta2 = count_mixed_pairs / (count_inlier_inlier + count_mixed_pairs)
            # beta3 = count_outlier_outlier / (count_inlier_inlier + count_outlier_outlier + count_mixed_pairs)
            self.beta1 = beta1
            self.beta2 = beta2
            # self.beta3 = 0  # beta3


            for i in range(len(sampling_pool["pairs"])):
                # assign priors
                prior_ii_mask = sampling_pool["inlier_inlier_masks"][i]
                prior_mixed_mask = sampling_pool["mixed_pairs_masks"][i]
                # prior_oo_mask = sampling_pool["outlier_outlier_masks"][i]
                sampling_pool["priors"][i][prior_ii_mask] = self.beta1
                sampling_pool["priors"][i][prior_mixed_mask] = self.beta2
                # sampling_pool["priors"][i][prior_oo_mask] = beta3

            p_y = []
            for i in range(len(sampling_pool["pairs"])):
                # np.log(priors) + probs + np.log(numerators/denominators)
                inlier_inlier = sampling_pool["inlier_inlier_masks"][i]
                mixed_pairs = sampling_pool["mixed_pairs_masks"][i]
                # outlier_outlier = sampling_pool["outlier_outlier_masks"][i]

                denominator = np.ones(len(sampling_pool["pairs"][i]))
                denominator[inlier_inlier] = denominator_ii
                denominator[mixed_pairs] = denominator_mixed
                # denominator[outlier_outlier] = denominator_oo

                try:
                    p_y.extend(
                        sampling_pool["priors"][i] * sampling_pool["numerators"][i] / denominator * sampling_pool["probs"][i]
                    )
                except Exception as e:
                    raise RuntimeError(f"AAS: failed computing p_y: {e}") from e
                

                # p_y.extend((np.log(sampling_pool["priors"][i]) + sampling_pool["probs"][i] + np.log(sampling_pool["numerators"][i] / denominator)).tolist())

            p_y = np.array(p_y)
            # eps = len(p_cross) / (len(p_cross) + len(np.where(p_y > 0)[0]))
            eps = self.args.consam_eps
            all_pairs = np.vstack([pairs, cross_pairs])
            all_probs = np.hstack([(1 - eps) * p_y, eps * p_cross])
            all_probs /= all_probs.sum()  # remove rounding off errors


            # count_cross = len(p_cross)
            # count_same = len(non_zero_p_y)
            # eps = count_cross / (count_cross + count_same)
            print("eps, beta1, beta2:", eps, self.beta1, self.beta2)
            # count_cross = np.ceil(self.n * eps)
            # count_same = min(self.n - count_cross, len(non_zero_p_y))

            try:
                idx_same = np.random.choice(len(all_pairs), min(self.n, len(np.where(all_probs > 0)[0])), p=all_probs, replace=False)
            except Exception as e:
                raise RuntimeError(f"AAS: failed sampling pairs: {e}") from e
            random_selected_pairs = all_pairs[idx_same]
            # random_selected_pairs = np.vstack([random_selected_pairs, cross_pairs[np.random.choice(range(cross_pairs.shape[0]), int(count_cross), p=p_cross, replace=False)]])

            selected_pairs = set()
            for pair in random_selected_pairs:
                if frozenset((pair[0], pair[1])) in self.all_selected_pairs:
                    print("continue...")
                    continue
                selected_pairs.add(frozenset((pair[0], pair[1])))
                self.all_selected_pairs.add(frozenset((pair[0], pair[1])))
                valid = self.gt_labels[pair[0]] == self.gt_labels[pair[1]]
                if valid:
                    self.ml_pairs.append((pair[0], pair[1]))
                else:
                    self.cl_pairs.append((pair[0], pair[1]))
            pos = 0
            neg = 0
            for i, pair in enumerate(pairs):
                valid = self.gt_labels[pair[0]] == self.gt_labels[pair[1]]
                if valid:
                    pos += 1
                else:
                    neg += 1
            print("pairs pos, neg, pos/(pos+neg):", pos, neg, pos/(pos+neg))
            pos = 0
            neg = 0
            for i, pair in enumerate(cross_pairs):
                valid = self.gt_labels[pair[0]] == self.gt_labels[pair[1]]
                if valid:
                    pos += 1
                else:
                    neg += 1
            print("cross_pairs pos, neg, pos/(pos+neg):", pos, neg, pos/(pos+neg))

            if len(selected_pairs) < self.n:
                n = self.n - len(selected_pairs)
                print(f"Budget not completely utilized! {n}")
                self.rec_shortfall = int(n)  # [record]

            return self.ml_pairs, self.cl_pairs
        except KeyboardInterrupt:
            print("KeyboardInterrupt caught. Exiting...")
            return self.ml_pairs, self.cl_pairs
        except Exception as e:
            # Previously dropped into pdb, which hangs an unattended run while AAS
            # silently contributes zero constraints -- exactly how the finch-clust
            # version mismatch stayed invisible for a whole grid (Todo.md item 1).
            raise RuntimeError(f"AAS sampling failed: {e}") from e
