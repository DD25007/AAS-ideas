from sampling.base import BaseSampling
import numpy as np
from utils import constrained_clustering
from sklearn import metrics
import networkx as nx
import itertools
from scipy.stats import entropy
from scipy.special import log_softmax, softmax
from sklearn.metrics import silhouette_score
from collections import defaultdict


class ConsistencySampling(BaseSampling):
    def __init__(self, database, n, betas, gallery_size=2, thresh=0.05, min_samples=3, eps=0.05, metric="cosine", num_gt_ids_gallery=0, adaptive=True, lambda_=3):
        super().__init__(database, n, gallery_size)
        self.thresh = thresh
        self.min_samples = min_samples
        self.metric = metric
        self.betas = betas
        self.eps = eps
        self.sampling_pool = {}
        self.switch_counts = 0
        self.num_gt_ids_gallery = num_gt_ids_gallery
        self.adaptive = adaptive
        self.lambda_ = lambda_

    def select2(self, features=None, **kwargs):
        assert features is not None
        n_samples = features.shape[0]
        selected_pairs = set()

        # Get Density-based (DBSCAN) clusters
        # "clusters_dbscan" contains information about outliers while
        # "labels_dbscan" represent each outlier as a singleton cluster
        clusters_dbscan = constrained_clustering(
            X=features,
            method="DBSCAN",
            eps=self.thresh,
            min_samples=self.min_samples,
            metric=self.metric,
            ml_pairs=self.ml_pairs,
            cl_pairs=self.cl_pairs,
        )
        labels_dbscan = clusters_dbscan.copy()
        labels_dbscan[labels_dbscan == -1] = np.arange(labels_dbscan.max() + 1, labels_dbscan.max() + 1 + sum(labels_dbscan == -1))

        # Get Nearest Neighbor-based (FINCH) clusters
        labels_finch = constrained_clustering(
            X=features,
            method="FINCH",
            metric=self.metric,
            ml_pairs=self.ml_pairs,
            cl_pairs=self.cl_pairs,
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

        consistent_positive_pairs = ml1.intersection(ml2)
        consistent_negative_pairs = cl1.intersection(cl2)
        inconsistent_pairs = (cl1.union(cl2) - cl1.intersection(cl2)).union(ml1.union(ml2) - ml1.intersection(ml2))

        cluster_label_maps = {
            "dbscan": {c_id: set() for c_id in np.unique(labels_dbscan)},
            "finch": {c_id: set() for c_id in np.unique(labels_finch)}
        }

        for i in range(n_samples):
            cluster_label_maps["dbscan"][labels_dbscan[i]].add(i)
            cluster_label_maps["finch"][labels_finch[i]].add(i)

        pairwise_distances = metrics.pairwise.pairwise_distances(features, features, metric=self.metric)

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
        for i, c in enumerate(nx.connected_components(inconsistency_graph)):
            S_k = inconsistency_graph.subgraph(c).copy()
            S.append(nx.create_empty_copy(S_k))
            dbscan_nodes = {n for n, d in S_k.nodes(data=True) if d["bipartite"] == 0}
            finch_nodes = set(S_k) - dbscan_nodes
            inconsistent_dbscan_labels.append(list(int(dn.split('-')[1]) for dn in dbscan_nodes))
            inconsistent_finch_labels.append(list(int(fn.split('-')[1]) for fn in finch_nodes))

        sampling_pool = defaultdict(list)

        # all_positive_relations = {
        #     "dbscan": [],
        #     "finch": [],
        # }
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
            for c in nx.connected_components(S_k):
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

            # all_positive_relations["dbscan"].extend(dbscan_positive_relations)
            # all_positive_relations["finch"].extend(finch_positive_relations)
            dbscan_positive_relations = set(dbscan_positive_relations)
            finch_positive_relations = set(finch_positive_relations)

            # Code for Active Sampling
            try:
                # \tilde{I}_k: inconsistency
                inconsistency = np.sort(np.array(list(dbscan_positive_relations.union(finch_positive_relations) - dbscan_positive_relations.intersection(finch_positive_relations))), axis=1)
                # \mathcal{P}_k: potential_queries
                # potential_queries = list(itertools.product(np.unique(potential_queries), np.unique(potential_queries)))
                # potential_queries = np.array(potential_queries)
                # I_k: pair
                pair = np.intersect1d(potential_queries.view([('', potential_queries.dtype)] * 2), inconsistency.view([('', inconsistency.dtype)] * 2))
                pair = pair.view(potential_queries.dtype).reshape(-1, 2)
            except:
                pair = []

            if len(pair) == 0:
                # Set I_k = \mathcal{P}_k
                potential_queries = list(itertools.combinations(np.unique(potential_queries), 2))
                pair = np.array(potential_queries)

            # Append the non-redundant, most informative inconsistent pairs to the sampling pool
            sampling_pool["pairs"].append(pair)

            # Get the labels of both points in each pair from DBSCAN
            labels_a = clusters_dbscan[pair[:, 0]]
            labels_b = clusters_dbscan[pair[:, 1]]

            # Check if the points are inliers (True) or outliers (False)
            is_inlier_a = labels_a >= 0
            is_inlier_b = labels_b >= 0

            # Check if the pairs satisfy the beta_i relationships
            inlier_inlier = np.logical_and(is_inlier_a, is_inlier_b)  # \beta_1
            mixed_pairs = np.logical_xor(is_inlier_a, is_inlier_b)  # \beta_2
            outlier_outlier = np.logical_and(~is_inlier_a, ~is_inlier_b)  # \beta_3

            # computing I_k^{beta_i}
            pair_ii = pair[inlier_inlier]
            pair_mixed = pair[mixed_pairs]
            pair_oo = pair[outlier_outlier]

            sampling_pool["inlier_inlier_masks"].append(inlier_inlier)
            sampling_pool["pairs_ii"].append(pair_ii)
            sampling_pool["mixed_pairs_masks"].append(mixed_pairs)
            sampling_pool["pairs_mixed"].append(pair_mixed)
            sampling_pool["outlier_outlier_masks"].append(outlier_outlier)
            sampling_pool["pairs_oo"].append(pair_oo)

            # Compute the numerator part of  P(X = S_k | β = β_i)
            numerator = np.zeros(len(pair))
            try:
                numerator[inlier_inlier] = features[np.unique(pair_ii)].var() * len(pair_ii)
            except:
                pass
            try:
                numerator[mixed_pairs] = features[np.unique(pair_mixed)].var() * len(pair_mixed)
            except:
                pass
            try:
                numerator[outlier_outlier] = features[np.unique(pair_oo)].var() * len(pair_oo)
            except:
                pass
            sampling_pool["numerators"].append(numerator)

            priors = np.zeros(len(pair))
            priors[inlier_inlier] = -1  # get masks first, then assign beta values
            priors[mixed_pairs] = 1
            priors[outlier_outlier] = 0

            sampling_pool["priors"].append(priors)

            # Compute the conditional probability of sampling a pair given its super-cluster S_k and relationship beta_i
            # P(Y = (a, b) ∈ I | X = S_k, β = β_i)
            prob = np.zeros(len(pair))
            try:
                prob[inlier_inlier] = 1 - pairwise_distances[pair_ii[:, 0], pair_ii[:, 1]]
                prob[inlier_inlier] = log_softmax(prob[inlier_inlier])
            except:
                pass
            try:
                prob[outlier_outlier] = 1 - pairwise_distances[pair_oo[:, 0], pair_oo[:, 1]]
                prob[outlier_outlier] = log_softmax(prob[outlier_outlier])
            except:
                pass
            try:
                prob[mixed_pairs] = 1 - pairwise_distances[pair_mixed[:, 0], pair_mixed[:, 1]]
                prob[mixed_pairs] = log_softmax(prob[mixed_pairs])
            except:
                pass

            sampling_pool["probs"].append(prob)

        # Stack all the non-redundant, most-informative inconsistent pairs from all S_k's to form the sampling pool
        pairs = np.vstack(sampling_pool["pairs"])

        # Compute the denominator part of  P(X = S_k | β = β_i)
        try:
            denominator_ii = features[np.unique(np.concatenate(sampling_pool["pairs_ii"]))].var() * len(np.concatenate(sampling_pool["pairs_ii"]))
        except:
            denominator_ii = 1
        try:
            denominator_mixed = features[np.unique(np.concatenate(sampling_pool["pairs_mixed"]))].var() * len(np.concatenate(sampling_pool["pairs_mixed"]))
        except:
            denominator_mixed = 1
        try:
            denominator_oo = features[np.unique(np.concatenate(sampling_pool["pairs_oo"]))].var() * len(np.concatenate(sampling_pool["pairs_oo"]))
        except:
            denominator_oo = 1

        # Find pairs of non-redundant, most informative points from coarse super-clusters
        cross_pairs = []
        cross_pairs_sim = []
        for i, S_k1 in enumerate(sampling_pool["S_k"]):
            for j, S_k2 in enumerate(sampling_pool["S_k"]):
                if S_k1 == S_k2:
                    # Same super-cluster
                    continue
                row_indices = list(S_k1)  # Points in super-cluster k1
                col_indices = list(S_k2)  # Points in super-cluster k2
                sub_matrix = pairwise_distances[np.ix_(row_indices, col_indices)]
                min_idx = np.unravel_index(np.argmin(sub_matrix), sub_matrix.shape)
                min_row, min_col = row_indices[min_idx[0]], col_indices[min_idx[1]]
                cross_pairs.append([min_row, min_col])
                cross_pairs_sim.append(1 - pairwise_distances[min_row, min_col])

        #  P(Y = (a, b) ∈ \mathcal{P})
        p_cross = softmax(cross_pairs_sim)
        cross_pairs = np.vstack(cross_pairs)

        count_inlier_inlier = np.sum(np.concatenate(sampling_pool["inlier_inlier_masks"]))
        count_mixed_pairs = np.sum(np.concatenate(sampling_pool["mixed_pairs_masks"]))
        count_outlier_outlier = np.sum(np.concatenate(sampling_pool["outlier_outlier_masks"]))
        self.count_inlier_inlier = count_inlier_inlier
        self.count_mixed_pairs = count_mixed_pairs
        self.count_outlier_outlier = count_outlier_outlier

        beta1 = count_inlier_inlier / (count_inlier_inlier + count_outlier_outlier + count_mixed_pairs)
        beta2 = count_mixed_pairs / (count_inlier_inlier + count_outlier_outlier + count_mixed_pairs)
        beta3 = count_outlier_outlier / (count_inlier_inlier + count_outlier_outlier + count_mixed_pairs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3

        mod_s = nx.number_connected_components(inconsistency_graph)
        eps_hat = (mod_s - self.num_gt_ids_gallery) / max(mod_s, self.num_gt_ids_gallery)

        if self.adaptive:
            glc_k = 12
            glc_d = 0.75
            glc_v = -2.5
            self.eps = (1 + np.exp(-glc_k * (eps_hat - glc_d))) ** (1 / glc_v)

        for i in range(len(sampling_pool["pairs"])):
            # assign priors
            prior_ii_mask = sampling_pool["inlier_inlier_masks"][i]
            prior_mixed_mask = sampling_pool["mixed_pairs_masks"][i]
            prior_oo_mask = sampling_pool["outlier_outlier_masks"][i]
            sampling_pool["priors"][i][prior_ii_mask] = beta1
            sampling_pool["priors"][i][prior_mixed_mask] = beta2
            sampling_pool["priors"][i][prior_oo_mask] = beta3

        p_y = []
        for i in range(len(sampling_pool["pairs"])):
            # np.log(priors) + probs + np.log(numerators/denominators)
            inlier_inlier = sampling_pool["inlier_inlier_masks"][i]
            mixed_pairs = sampling_pool["mixed_pairs_masks"][i]
            outlier_outlier = sampling_pool["outlier_outlier_masks"][i]

            denominator = np.ones(len(sampling_pool["pairs"][i]))
            denominator[inlier_inlier] = denominator_ii
            denominator[mixed_pairs] = denominator_mixed
            denominator[outlier_outlier] = denominator_oo

            p_y.extend((np.log(sampling_pool["priors"][i]) + sampling_pool["probs"][i] + np.log(
                sampling_pool["numerators"][i] / denominator)).tolist())

        p_y = np.exp(p_y)  # To remove rounding off errors due to overflow
        p_y = p_y / p_y.sum()

        count_cross = np.ceil(self.n * self.eps)
        count_same = min(self.n - count_cross, len(p_y))
        count_cross = min(self.n - count_same, len(p_cross))

        idx_same = np.random.choice(range(pairs.shape[0]), int(count_same), p=p_y, replace=False)
        random_selected_pairs = pairs[idx_same]
        random_selected_pairs = np.vstack([random_selected_pairs, cross_pairs[np.random.choice(range(cross_pairs.shape[0]), int(count_cross), p=p_cross, replace=False)]])

        for pair in random_selected_pairs:
            if frozenset((pair[0], pair[1])) in self.all_selected_pairs:
                print("continue...")
                continue
            selected_pairs.add(frozenset((pair[0], pair[1])))
            self.all_selected_pairs.add(frozenset((pair[0], pair[1])))
            valid = self.database.metadata.iloc[pair[0]]["identity"] == self.database.metadata.iloc[pair[1]]["identity"]
            if valid:
                self.ml_pairs.append((pair[0], pair[1]))
            else:
                self.cl_pairs.append((pair[0], pair[1]))

        if len(selected_pairs) < self.n:
            n = self.n - len(selected_pairs)
            print(f"Budget not completely utilized! {n}")

        # all_negative_relations = {
        #     "dbscan": set(itertools.product(np.arange(labels_dbscan.shape[0]), np.arange(labels_dbscan.shape[0]))) - set(all_positive_relations["dbscan"]),
        #     "finch": set(itertools.product(np.arange(labels_finch.shape[0]), np.arange(labels_finch.shape[0]))) - set(all_positive_relations["finch"]),
        # }

        # weak_ml_pairs = set(all_positive_relations["finch"]).intersection(set(all_positive_relations["dbscan"]))
        # weak_cl_pairs = all_negative_relations["dbscan"].intersection(all_negative_relations["finch"])

        pseudo_labels = constrained_clustering(
            X=features,
            method="DBSCAN",
            eps=self.thresh,
            min_samples=self.min_samples,
            metric=self.metric,
            ml_pairs=self.ml_pairs,
            cl_pairs=self.cl_pairs,
        )
        pseudo_labels[np.unique(pairs)] = -1
        # import pdb; pdb.set_trace()
        return self.ml_pairs, self.cl_pairs, pseudo_labels

    def select(self, features=None, **kwargs):
        assert features is not None
        n_samples = features.shape[0]
        selected_pairs = set()

        # Get Density-based (DBSCAN) and Nearest Neighbor-based (FINCH) clusters
        clusters_dbscan = constrained_clustering(
            X=features,
            method="DBSCAN",
            eps=self.thresh,
            min_samples=self.min_samples,
            metric=self.metric,
            ml_pairs=self.ml_pairs,
            cl_pairs=self.cl_pairs,
        )  # named clusters because it contains outliers that are represented with -1

        change_counts = 0


        database_subset = self.database
        database_subset.metadata.loc[:, "pseudo_identity"] = clusters_dbscan

        # named labels because it considers each outlier as a singleton cluster
        labels_dbscan = clusters_dbscan.copy()
        num_outliers = sum(labels_dbscan == -1)
        num_inliers = n_samples - num_outliers
        num_clusters_cdbscan = len(np.unique(labels_dbscan)) - 1  # -1 for outliers
        labels_dbscan[labels_dbscan == -1] = np.arange(labels_dbscan.max() + 1, labels_dbscan.max() + 1 + sum(labels_dbscan == -1))

        sil_dbscan = silhouette_score(features, labels_dbscan, metric=self.metric)
        sil_dbscan_hat = 1 - sil_dbscan

        clusters_finch = constrained_clustering(
            X=features,
            method="FINCH",
            metric=self.metric,
            ml_pairs=self.ml_pairs,
            cl_pairs=self.cl_pairs,
        )
        labels_finch = clusters_finch.copy()
        labels_finch[labels_finch == -1] = np.arange(labels_finch.max() + 1, labels_finch.max() + 1 + sum(labels_finch == -1))

        sil_finch = silhouette_score(features, labels_finch, metric=self.metric)
        sil_finch_hat = 1 - sil_finch
        beta1_hat = (sil_finch_hat + sil_dbscan_hat)/2

        cluster_to_samples_map_dbscan = {c_id: set() for c_id in np.unique(labels_dbscan)}
        cluster_to_samples_map_finch = {c_id: set() for c_id in np.unique(labels_finch)}

        for i in range(n_samples):
            cluster_to_samples_map_dbscan[labels_dbscan[i]].add(i)
            cluster_to_samples_map_finch[labels_finch[i]].add(i)

        features_pairwise_distances = metrics.pairwise.pairwise_distances(features, features, metric=self.metric)

        # Compute IoU between both clustering methods.
        # Clusters with IoU as 0 or 1 are the most certain, i.e., the clusters are exactly the same
        cluster_iou = np.zeros((len(cluster_to_samples_map_dbscan), len(cluster_to_samples_map_finch)))
        size_dbscan = [len(cluster_to_samples_map_dbscan[i]) for i in range(len(cluster_to_samples_map_dbscan))]
        size_finch = [len(cluster_to_samples_map_finch[i]) for i in range(len(cluster_to_samples_map_finch))]
        for i, c1 in enumerate(cluster_to_samples_map_dbscan):
            for j, c2 in enumerate(cluster_to_samples_map_finch):
                cluster_iou[i, j] = len(cluster_to_samples_map_dbscan[c1].intersection(cluster_to_samples_map_finch[c2])) / len(cluster_to_samples_map_dbscan[c1].union(cluster_to_samples_map_finch[c2]))

        # Create a bipartite graph between DBSCAN and FINCH clusters. The same clusters (IoU = 0 or 1) are ignored.
        bipartite_inconsistency_graph = nx.Graph()
        bipartite_inconsistency_graph_attributes = []
        # all_outliers = []
        for i in range(cluster_iou.shape[0]):  # Loop over DBSCAN clusters
            for j in range(cluster_iou.shape[1]):  # Loop over FINCH clusters
                iou = cluster_iou[i, j]
                if iou == 0 or iou == 1:
                    # if iou == 1 and size_dbscan[i] == 1:
                    #     all_outliers.append(list(cluster_to_samples_map_dbscan[i])[0])
                    continue
                certainty = abs(0.5 - iou)
                total_size = size_dbscan[i] + size_finch[j]
                bipartite_inconsistency_graph.add_node(f'dbscan-{i}', bipartite=0, size=size_dbscan[i])
                bipartite_inconsistency_graph.add_node(f'finch-{j}', bipartite=1, size=size_finch[j])
                bipartite_inconsistency_graph.add_edge(f'dbscan-{i}', f'finch-{j}', iou=iou, certainty=certainty, total_size=total_size)
                bipartite_inconsistency_graph_attributes.append((i, j, iou, certainty, total_size))

        # Sort based on certainty (i.e. highly uncertain samples are ranked higher)
        bipartite_inconsistency_graph_attributes.sort(key=lambda x: (x[3], -x[4]))

        # prior_ii, prior_mixed, prior_oo = self.betas
        # beta1_hat = max(0.1, self.num_gt_ids_gallery/num_clusters_cdbscan)
        beta2_hat = num_inliers / (num_inliers + num_outliers)
        beta3_hat = num_outliers / (num_inliers + num_outliers)

        beta_sum = beta1_hat + beta2_hat + beta3_hat
        prior_ii = beta1_hat / beta_sum
        prior_mixed = beta2_hat / beta_sum
        prior_oo = beta3_hat / beta_sum

        self.beta1 = prior_ii
        self.beta2 = prior_mixed
        self.beta3 = prior_oo

        mod_s = nx.number_connected_components(bipartite_inconsistency_graph)
        eps_hat = (mod_s - self.num_gt_ids_gallery) / max(mod_s, self.num_gt_ids_gallery)

        if self.adaptive:
            # self.eps = (eps_hat + 1) / 2
            # self.eps = max(self.eps, 0.05)  # Ensure eps is not too small
            # self.eps = np.exp(self.lambda_ * (eps_hat - 1))
            glc_k = 12
            glc_d = 0.75
            glc_v = -2.5
            self.eps = (1 + np.exp(-glc_k * (eps_hat - glc_d)))**(1/glc_v)

        self.mod_s = mod_s
        self.num_inliers = num_inliers
        self.num_outliers = num_outliers
        self.num_clusters_cdbscan = num_clusters_cdbscan

        # The bipartite graph has multiple disconnected sub-graphs.
        # Sub-graphs that are not connected have no inter-sub-graph confusions.
        # Define the sampling budget for each sub-graph.
        connected_components = []
        dbscan_clusters_in_connected_components, finch_clusters_in_connected_components = [], []
        cluster_id_to_connected_components_map = {}
        for i, c in enumerate(nx.connected_components(bipartite_inconsistency_graph)):
            connected_subgraph = bipartite_inconsistency_graph.subgraph(c).copy()
            connected_components.append(nx.create_empty_copy(connected_subgraph))
            dbscan_nodes = {n for n, d in connected_subgraph.nodes(data=True) if d["bipartite"] == 0}
            finch_nodes = set(connected_subgraph) - dbscan_nodes
            dbscan_clusters_in_connected_components.append(list(int(dn.split('-')[1]) for dn in dbscan_nodes))
            finch_clusters_in_connected_components.append(list(int(fn.split('-')[1]) for fn in finch_nodes))
            for d in dbscan_clusters_in_connected_components[-1]:
                cluster_id_to_connected_components_map[d] = i
        connected_components_map_to_sampling_budget = np.array([len(dbscan_clusters_in_connected_components[i]) + len(finch_clusters_in_connected_components[i]) for i in range(len(dbscan_clusters_in_connected_components))])
        connected_components_map_to_sampling_budget = connected_components_map_to_sampling_budget * self.n // connected_components_map_to_sampling_budget.sum()
        connected_components_map_to_sampling_budget = np.where(connected_components_map_to_sampling_budget == 0, 1, connected_components_map_to_sampling_budget)

        # Traverse over each sub-graph (connected component) for sampling.
        traversed = set()

        sampling_pool = {"ccs": [],
            "pairs": [], "probs": [], "size": [], "probs_ii": [], "probs_oo": [], "probs_mixed": [], "priors": [], "numerators": [], "pairs_ii": [], "pairs_oo": [], "pairs_mixed": [], "inlier_inlier": [], "outlier_outlier": [], "mixed_pairs": []}
        for i, (dbscan_id, finch_id, iou, certainty, total_size) in enumerate(bipartite_inconsistency_graph_attributes):
            cc_id = cluster_id_to_connected_components_map[dbscan_id]
            if cc_id in traversed:
                continue
            # print("Connected Cluster ID:", cc_id)
            traversed.add(cc_id)

            sampling_pool["ccs"].append(set())


            # Add temporary smallest edges across clusters belonging to one method.
            # Later these edges are pruned to get Minimum Spanning Tree
            for c_id1 in dbscan_clusters_in_connected_components[cc_id]:
                row_indices = list(cluster_to_samples_map_dbscan[c_id1])
                sampling_pool["ccs"][-1].update(set(row_indices)) # add cluster points to "ccs"
                
                for c_id2 in dbscan_clusters_in_connected_components[cc_id]:
                    if c_id1 == c_id2:
                        continue
                    col_indices = list(cluster_to_samples_map_dbscan[c_id2])
                    sub_matrix = features_pairwise_distances[np.ix_(row_indices, col_indices)]
                    min_idx = np.unravel_index(np.argmin(sub_matrix), sub_matrix.shape)
                    min_row, min_col = row_indices[min_idx[0]], col_indices[min_idx[1]]
                    connected_components[cc_id].add_edge(f'dbscan-{c_id1}', f'dbscan-{c_id2}',
                                                         weight=features_pairwise_distances[min_row, min_col],
                                                         points=[min_row, min_col])
            for c_id1 in finch_clusters_in_connected_components[cc_id]:
                for c_id2 in finch_clusters_in_connected_components[cc_id]:
                    if c_id1 == c_id2:
                        continue
                    row_indices = list(cluster_to_samples_map_finch[c_id1])
                    col_indices = list(cluster_to_samples_map_finch[c_id2])
                    sub_matrix = features_pairwise_distances[np.ix_(row_indices, col_indices)]
                    min_idx = np.unravel_index(np.argmin(sub_matrix), sub_matrix.shape)
                    min_row, min_col = row_indices[min_idx[0]], col_indices[min_idx[1]]
                    connected_components[cc_id].add_edge(f'finch-{c_id1}', f'finch-{c_id2}',
                                                         weight=features_pairwise_distances[min_row, min_col],
                                                         points=[min(min_row, min_col), max(min_row, min_col)])

            # Find the MST and corresponding valid pair-wise relationships
            # mst = nx.minimum_spanning_tree(connected_components[cc_id])
            mst = connected_components[cc_id].copy()
            dbscan_relations = []
            finch_relations = []
            potential_queries = None
            for c in nx.connected_components(mst):
                G = mst.subgraph(c).copy()
                if 'finch' in list(G.nodes)[0]:
                    for node in G.nodes:
                        finch_relations.extend(list(itertools.product(cluster_to_samples_map_finch[int(node.split('-')[1])], cluster_to_samples_map_finch[int(node.split('-')[1])])))
                        # for neighbor in G[node]:
                        #     finch_relations.extend(list(itertools.product(cluster_to_samples_map_finch[int(neighbor.split('-')[1])], cluster_to_samples_map_finch[int(neighbor.split('-')[1])])))
                        #     finch_relations.extend(list(itertools.product(cluster_to_samples_map_finch[int(node.split('-')[1])], cluster_to_samples_map_finch[int(neighbor.split('-')[1])])))
                        #     finch_relations.extend(list(itertools.product(cluster_to_samples_map_finch[int(neighbor.split('-')[1])], cluster_to_samples_map_finch[int(node.split('-')[1])])))
                if 'dbscan' in list(G.nodes)[0]:
                    for node in G.nodes:
                        dbscan_relations.extend(list(itertools.product(cluster_to_samples_map_dbscan[int(node.split('-')[1])], cluster_to_samples_map_dbscan[int(node.split('-')[1])])))
                        # for neighbor in G[node]:
                        #     dbscan_relations.extend(list(itertools.product(cluster_to_samples_map_dbscan[int(neighbor.split('-')[1])], cluster_to_samples_map_dbscan[int(neighbor.split('-')[1])])))
                        #     dbscan_relations.extend(list(itertools.product(cluster_to_samples_map_dbscan[int(node.split('-')[1])], cluster_to_samples_map_dbscan[int(neighbor.split('-')[1])])))
                        #     dbscan_relations.extend(list(itertools.product(cluster_to_samples_map_dbscan[int(neighbor.split('-')[1])], cluster_to_samples_map_dbscan[int(node.split('-')[1])])))
                if len(G.nodes) == 1:
                    continue
                if potential_queries is None:
                    potential_queries = np.array(list(nx.get_edge_attributes(G, "points").values()))
                else:
                    potential_queries = np.append(potential_queries, np.array(list(nx.get_edge_attributes(G, "points").values())), 0)

            dbscan_relations = set(dbscan_relations)
            finch_relations = set(finch_relations)

            # Find inconsistent pair-wise relationships across both clustering methods
            try:
                inconsistency = np.sort(np.array(list(dbscan_relations.union(finch_relations) - dbscan_relations.intersection(finch_relations))), axis=1)
                potential_queries = list(itertools.product(np.unique(potential_queries), np.unique(potential_queries)))
                potential_queries = np.array(potential_queries)
                pair = np.intersect1d(potential_queries.view([('', potential_queries.dtype)] * 2), inconsistency.view([('', inconsistency.dtype)] * 2))
                pair = pair.view(potential_queries.dtype).reshape(-1, 2)
            except:
                pair = []

            # if inconsistency is not None:
            #     # potential_queries = list(itertools.product(np.unique(potential_queries), np.unique(potential_queries)))
            #     # potential_queries = np.array(potential_queries)
            #     # pair = np.intersect1d(potential_queries.view([('', potential_queries.dtype)] * 2), inconsistency.view([('', inconsistency.dtype)] * 2))
            #     # pair = pair.view(potential_queries.dtype).reshape(-1, 2)
            #     pair = inconsistency
            # else:
            if len(pair) == 0:
                potential_queries = list(itertools.combinations(np.unique(potential_queries), 2))
                pair = np.array(potential_queries)
            sampling_pool["pairs"].append(pair) # here, pair is the set of non-redundant inconsistent pairs in this cc. That is, I_k
            
            # Get the labels of both points in each pair
            labels_a = clusters_dbscan[pair[:, 0]]
            labels_b = clusters_dbscan[pair[:, 1]]

            # Boolean masks
            is_inlier_a = labels_a >= 0 # of len n_pairs
            is_inlier_b = labels_b >= 0

            # Types of pairs
            inlier_inlier = np.logical_and(is_inlier_a, is_inlier_b)
            outlier_outlier = np.logical_and(~is_inlier_a, ~is_inlier_b)
            mixed_pairs = np.logical_xor(is_inlier_a, is_inlier_b)

            sampling_pool["inlier_inlier"].append(inlier_inlier)
            sampling_pool["outlier_outlier"].append(outlier_outlier)
            sampling_pool["mixed_pairs"].append(mixed_pairs)

            # computing I_k^beta_i
            pair_ii = pair[inlier_inlier]
            pair_oo = pair[outlier_outlier]
            pair_mixed = pair[mixed_pairs]

            priors = np.zeros(len(pair))
            priors[inlier_inlier] = -1 # get masks first, then assign beta values
            priors[outlier_outlier] = 0
            priors[mixed_pairs] = 1
            
            sampling_pool["priors"].append(priors)

            # TODO:
            # sig_2_k_ii = np.array(features[np.unique(pair_ii)]).var()
            # c_k_ii = len(probs_ii)
            numerator = np.zeros(len(pair))
            try:
                numerator[inlier_inlier] = features[np.unique(pair_ii)].var() * len(pair_ii)
                # numerator[inlier_inlier] = len(pair_ii)
            except:
                pass
            try:
                numerator[outlier_outlier] = features[np.unique(pair_oo)].var() * len(pair_oo)
                # numerator[outlier_outlier] = len(pair_oo)
            except:
                pass
            try:
                numerator[mixed_pairs] = features[np.unique(pair_mixed)].var() * len(pair_mixed)
                # numerator[mixed_pairs] = len(pair_mixed)
            except:
                pass

            sampling_pool["numerators"].append(numerator)


            sampling_pool["pairs_ii"].append(pair_ii)
            sampling_pool["pairs_oo"].append(pair_oo)
            sampling_pool["pairs_mixed"].append(pair_mixed)


            prob = np.zeros(len(pair))
            try:
                prob[inlier_inlier] = 1 - features_pairwise_distances[pair_ii[:, 0], pair_ii[:, 1]]
                prob[inlier_inlier] = log_softmax(prob[inlier_inlier])
            except:
                pass
            try:
                prob[outlier_outlier] = 1 - features_pairwise_distances[pair_oo[:, 0], pair_oo[:, 1]]
                prob[outlier_outlier] = log_softmax(prob[outlier_outlier])
            except:
                pass
            try:
                prob[mixed_pairs] = 1 - features_pairwise_distances[pair_mixed[:, 0], pair_mixed[:, 1]]
                prob[mixed_pairs] = log_softmax(prob[mixed_pairs])
            except:
                pass

            sampling_pool["probs"].append(prob)

            # # Counts
            # count_inlier_inlier = np.sum(inlier_inlier)  # Helps resolve under-segmentation (also over-segmentation)
            # count_outlier_outlier = np.sum(outlier_outlier)
            # count_mixed = np.sum(mixed_pairs)  # Helps resolve over-segmentation
            # count_valid_pairs = count_inlier_inlier + 0.7 * count_mixed + 0.3 * count_outlier_outlier
            # # valid_pairs = ~np.all(clusters_dbscan[pair] < 0, axis=1)  # Ignore pairs that have both the points outliers
            # # count_valid_pairs = np.sum(valid_pairs)
            # sampling_pool["size"].append(count_valid_pairs)

            #     continue
            #     pair = np.array(list(set(potential_queries.view([('', potential_queries.dtype)] * 2).squeeze().tolist()))).reshape(-1, 2)
            continue
            # counts = {}
            n_selections_per_cc = 0
            for j in np.argsort(features_pairwise_distances[pair[:, 0], pair[:, 1]]):
                if n_selections_per_cc == connected_components_map_to_sampling_budget[cc_id]:
                    break
                if frozenset((pair[j, 0], pair[j, 1])) in self.all_selected_pairs:
                    continue
                selected_pairs.add(frozenset((pair[j, 0], pair[j, 1])))
                self.all_selected_pairs.add(frozenset((pair[j, 0], pair[j, 1])))
                n_selections_per_cc += 1
                valid = self.database.metadata.iloc[pair[j, 0]]["identity"] == self.database.metadata.iloc[pair[j, 1]]["identity"]
                if valid:
                    self.ml_pairs.append((pair[j, 0], pair[j, 1]))
                else:
                    self.cl_pairs.append((pair[j, 0], pair[j, 1]))
                if len(selected_pairs) == self.n:
                    print("[Warning] Budget exhausted without traversing all the connected components!")
                    print(f"Iterated {i} of {len(bipartite_inconsistency_graph_attributes)}")
                    return self.ml_pairs, self.cl_pairs

        # Compute the probabilities for each pair
        pairs = np.vstack(sampling_pool["pairs"])
        
        try:
            denominator_ii = features[np.unique(np.concatenate(sampling_pool["pairs_ii"]))].var() * len(np.concatenate(sampling_pool["pairs_ii"]))
            # denominator_ii = len(np.concatenate(sampling_pool["pairs_ii"]))
        except:
            denominator_ii = 1
        try:
            denominator_oo = features[np.unique(np.concatenate(sampling_pool["pairs_oo"]))].var() * len(np.concatenate(sampling_pool["pairs_oo"]))
            # denominator_oo = len(np.concatenate(sampling_pool["pairs_oo"]))
        except:
            denominator_oo = 1
        try:
            denominator_mixed = features[np.unique(np.concatenate(sampling_pool["pairs_mixed"]))].var() * len(np.concatenate(sampling_pool["pairs_mixed"]))
            # denominator_mixed = len(np.concatenate(sampling_pool["pairs_mixed"]))
        except:
            denominator_mixed = 1

        
        cross_pairs = [] # nearest point to sk; or random point from sk, random pt from sk'
        cross_pairs_sim = []
        for i, cc_set1 in enumerate(sampling_pool["ccs"]):
            for j, cc_set2 in enumerate(sampling_pool["ccs"]):
                # print(i, j)
                if cc_set1 == cc_set2:
                    continue
                
                # row_indices = list(cluster_to_samples_map_dbscan[cc_id1])
                # col_indices = list(cluster_to_samples_map_dbscan[cc_id2])
                row_indices = list(cc_set1)
                col_indices = list(cc_set2)
                # print(row_indices, col_indices)
                sub_matrix = features_pairwise_distances[np.ix_(row_indices, col_indices)]
                min_idx = np.unravel_index(np.argmin(sub_matrix), sub_matrix.shape)
                min_row, min_col = row_indices[min_idx[0]], col_indices[min_idx[1]]
                cross_pairs.append([min_row, min_col])
                cross_pairs_sim.append(1-features_pairwise_distances[min_row, min_col])
                # connected_components[cc_id].add_edge(f'dbscan-{cc_id1}', f'dbscan-{cc_id2}',
                                                        # weight=features_pairwise_distances[min_row, min_col],
                                                        # points=[min_row, min_col])

        p_cross = softmax(cross_pairs_sim)
        cross_pairs = np.vstack(cross_pairs)

        if not self.adaptive:
            beta1, beta2, beta3 = self.betas

        # compute priors and assign priors
        # counts for inlier_inlier, outlier_outlier, mixed_pairs
        count_inlier_inlier = np.sum(np.concatenate(sampling_pool["inlier_inlier"]))
        count_outlier_outlier = np.sum(np.concatenate(sampling_pool["outlier_outlier"]))
        count_mixed_pairs = np.sum(np.concatenate(sampling_pool["mixed_pairs"]))
        self.count_inlier_inlier = count_inlier_inlier
        self.count_outlier_outlier = count_outlier_outlier
        self.count_mixed_pairs = count_mixed_pairs
        
        if self.adaptive:
            beta1 = count_inlier_inlier / (count_inlier_inlier + count_outlier_outlier + count_mixed_pairs)
            beta2 = count_mixed_pairs / (count_inlier_inlier + count_outlier_outlier + count_mixed_pairs)   
            beta3 = count_outlier_outlier / (count_inlier_inlier + count_outlier_outlier + count_mixed_pairs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        
        print("Beta values:", beta1, beta2, beta3)
        
        # import pdb; pdb.set_trace()
        
        for i in range(len(sampling_pool["pairs"])):
            # assign priors
            prior_ii_mask = sampling_pool["inlier_inlier"][i]
            prior_mixed_mask = sampling_pool["mixed_pairs"][i]
            prior_oo_mask = sampling_pool["outlier_outlier"][i]
            sampling_pool["priors"][i][prior_ii_mask] = beta1
            sampling_pool["priors"][i][prior_mixed_mask] = beta2
            sampling_pool["priors"][i][prior_oo_mask] = beta3
    


        p_y = []
        for i in range(len(sampling_pool["pairs"])):
            # np.log(priors) + probs + np.log(numerators/denominators)
            inlier_inlier = sampling_pool["inlier_inlier"][i]
            outlier_outlier = sampling_pool["outlier_outlier"][i]
            mixed_pairs = sampling_pool["mixed_pairs"][i]

            denominator = np.ones(len(sampling_pool["pairs"][i]))
            denominator[inlier_inlier] = denominator_ii
            denominator[outlier_outlier] = denominator_oo
            denominator[mixed_pairs] = denominator_mixed

            p_y.extend((np.log(sampling_pool["priors"][i]) + sampling_pool["probs"][i] + np.log(sampling_pool["numerators"][i] / denominator)).tolist())

        p_y = np.exp(p_y)  # To remove rounding off errors due to overflow
        p_y = p_y / p_y.sum()
        
        

        count_cross = np.ceil(self.n * self.eps)
        count_same = min(self.n - count_cross, len(p_y))
        count_cross = min(self.n - count_same, len(p_cross))

        print("Pair counts:", count_same, count_cross)
        random_selected_pairs = pairs[np.random.choice(range(pairs.shape[0]), int(count_same), p=p_y, replace=False)]
        random_selected_pairs = np.vstack([random_selected_pairs, cross_pairs[np.random.choice(range(cross_pairs.shape[0]), int(count_cross), p=p_cross, replace=False)]])
    
        for pair in random_selected_pairs:
            if frozenset((pair[0], pair[1])) in self.all_selected_pairs:
                continue
            selected_pairs.add(frozenset((pair[0], pair[1])))
            self.all_selected_pairs.add(frozenset((pair[0], pair[1])))
            valid = self.database.metadata.iloc[pair[0]]["identity"] == self.database.metadata.iloc[pair[1]]["identity"]
            if valid:
                # check if the pseudo identities were same
                if database_subset.metadata.iloc[pair[0]]["pseudo_identity"] != database_subset.metadata.iloc[pair[1]]["pseudo_identity"]:
                    change_counts += 1
                self.ml_pairs.append((pair[0], pair[1]))
            else:
                # check if the pseudo identities were same
                if database_subset.metadata.iloc[pair[0]]["pseudo_identity"] == database_subset.metadata.iloc[pair[1]]["pseudo_identity"]:
                    change_counts += 1
                self.cl_pairs.append((pair[0], pair[1]))
        
        print(f"Number of changed pseudo identities: {change_counts}")
        self.switch_counts = change_counts

        if len(selected_pairs) == self.n:
            return self.ml_pairs, self.cl_pairs
        n = self.n - len(selected_pairs)
        print(f"Budget not completely utilized! {n}")
        
        return self.ml_pairs, self.cl_pairs

        if self.n - len(selected_pairs) > 0:
            from sklearn.metrics import pairwise_distances
            # Assume features_pairwise_distances is (N, N) and all_outliers is a 1D array
            all_indices = np.arange(features_pairwise_distances.shape[0])
            all_inliers = np.setdiff1d(all_indices, all_outliers)
            all_inliers = all_inliers[clusters_dbscan[all_inliers] != -1]

            dist_subset = features_pairwise_distances[np.ix_(all_outliers, all_outliers)]
            outlier_to_inlier_distances = features_pairwise_distances[np.ix_(all_outliers, all_inliers)]

            # Set diagonal to infinity to ignore self-distances
            np.fill_diagonal(dist_subset, np.inf)

            # Find closest other outlier for each
            nearest_indices = np.argmin(dist_subset, axis=1)

            # Convert local indices back to global indices
            closest_outlier_pairs = list({frozenset((i, all_outliers[j])) for i, j in zip(all_outliers, nearest_indices)})

            def phi(p1, p2, features):
                xi, xj = p1
                xi_, xj_ = p2
                return min(np.linalg.norm(features[xi] - features[xi_]) + np.linalg.norm(features[xj] - features[xj_]),
                           np.linalg.norm(features[xi] - features[xj_]) + np.linalg.norm(features[xj] - features[xi_]))

            dist_phi = pairwise_distances(closest_outlier_pairs, metric=phi, features=features)

            # Farthest Point Sampling (FPS) from https://minibatchai.com/2021/08/07/FPS.html
            pairs_left = np.arange(len(closest_outlier_pairs))
            sample_inds = np.zeros(len(closest_outlier_pairs), dtype='int')
            dists = np.ones_like(pairs_left) * float('inf')
            index_pairs = [tuple(sorted((all_outliers.index(i), all_outliers.index(j)))) for i, j in map(tuple, closest_outlier_pairs)]
            pair_distances = np.array([dist_subset[i, j] for i, j in index_pairs])
            selected = np.argmin(pair_distances)
            sample_inds[0] = pairs_left[selected]
            pairs_left = np.delete(pairs_left, selected)

            # Iteratively select closest_outlier_pairs for a maximum of n_samples
            pair = frozenset((list(closest_outlier_pairs[sample_inds[0]])[0], list(closest_outlier_pairs[sample_inds[0]])[1]))
            selected_pairs.add(pair)
            pair = tuple(pair)
            valid = self.database.metadata.iloc[pair[0]]["identity"] == self.database.metadata.iloc[pair[1]]["identity"]
            if valid:
                self.ml_pairs.append((pair[0], pair[1]))
                inlier_i = all_inliers[features_pairwise_distances[pair[0], all_inliers].argmin()]
                inlier_j = all_inliers[features_pairwise_distances[pair[1], all_inliers].argmin()]
                closest_inlier = inlier_i
                selected_pairs.add(frozenset((pair[0], closest_inlier)))
                valid = self.database.metadata.iloc[pair[0]]["identity"] == self.database.metadata.iloc[closest_inlier]["identity"]
                if valid:
                    self.ml_pairs.append((pair[0], closest_inlier))
                else:
                    self.cl_pairs.append((pair[0], closest_inlier))
                if clusters_dbscan[inlier_i] != clusters_dbscan[inlier_j]:
                    closest_inlier = inlier_j
                    selected_pairs.add(frozenset((pair[1], closest_inlier)))
                    valid = self.database.metadata.iloc[pair[1]]["identity"] == self.database.metadata.iloc[closest_inlier]["identity"]
                    if valid:
                        self.ml_pairs.append((pair[1], closest_inlier))
                    else:
                        self.cl_pairs.append((pair[1], closest_inlier))
            else:
                self.cl_pairs.append((pair[0], pair[1]))

            i = 1
            while len(selected_pairs) < self.n and len(pairs_left) > 0:
                # Find the distance to the last added point in selected
                # and all the others
                last_added = sample_inds[i - 1]

                dist_to_last_added_point = dist_phi[last_added, pairs_left]

                # If closer, updated distances
                dists[pairs_left] = np.minimum(dist_to_last_added_point, dists[pairs_left])

                # We want to pick the one that has the largest nearest neighbour
                # distance to the sampled closest_outlier_pairs
                selected = np.argmax(dists[pairs_left])
                sample_inds[i] = pairs_left[selected]

                pair = frozenset(
                    (list(closest_outlier_pairs[sample_inds[i]])[0], list(closest_outlier_pairs[sample_inds[i]])[1])
                )
                pair = tuple(pair)
                valid = self.database.metadata.iloc[pair[0]]["identity"] == self.database.metadata.iloc[pair[1]]["identity"]
                if valid:
                    self.ml_pairs.append((pair[0], pair[1]))
                    inlier_i = all_inliers[features_pairwise_distances[pair[0], all_inliers].argmin()]
                    inlier_j = all_inliers[features_pairwise_distances[pair[1], all_inliers].argmin()]
                    closest_inlier = inlier_i
                    selected_pairs.add(frozenset((pair[0], closest_inlier)))
                    valid = self.database.metadata.iloc[pair[0]]["identity"] == self.database.metadata.iloc[closest_inlier]["identity"]
                    if valid:
                        self.ml_pairs.append((pair[0], closest_inlier))
                    else:
                        self.cl_pairs.append((pair[0], closest_inlier))
                    if clusters_dbscan[inlier_i] != clusters_dbscan[inlier_j]:
                        closest_inlier = inlier_j
                        selected_pairs.add(frozenset((pair[1], closest_inlier)))
                        valid = self.database.metadata.iloc[pair[1]]["identity"] == self.database.metadata.iloc[closest_inlier]["identity"]
                        if valid:
                            self.ml_pairs.append((pair[1], closest_inlier))
                        else:
                            self.cl_pairs.append((pair[1], closest_inlier))
                else:
                    self.cl_pairs.append((pair[0], pair[1]))

                selected_pairs.add(pair)
                # Update pairs_left
                pairs_left = np.delete(pairs_left, selected)
                i += 1

            # Trial: 1 (outlier pairs with smallest pair-wise distances)
            # pairs = list(itertools.combinations(all_outliers, 2))
            # pair_distances = [(i, j, features_pairwise_distances[i, j]) for (i, j) in pairs]
            # sorted_pairs = sorted(pair_distances, key=lambda x: x[2])

            # Trial: 2 (outlier pairs with smallest pair-wise distances that are farthest from inliers)
            # all_indices = np.arange(features_pairwise_distances.shape[0])
            # all_inliers = np.setdiff1d(all_indices, all_outliers)
            #
            # # Compute min distance to outside for each index
            # min_dist_to_inliers = {
            #     i: np.min(features_pairwise_distances[i, all_inliers]) for i in all_outliers
            # }
            #
            # # Generate all pairs and compute score
            # pair_scores = []
            # for i, j in itertools.combinations(all_outliers, 2):
            #     dist_from_closest_inlier = min(min_dist_to_inliers[i], min_dist_to_inliers[j])
            #     dist_between_outliers = features_pairwise_distances[i, j]
            #
            #     # You can use a composite score — here we subtract closeness from farness
            #     score = dist_from_closest_inlier - dist_between_outliers  # Higher score is better
            #     pair_scores.append((i, j, score))
            #
            # # Sort by descending score (maximize distance from outsiders while keeping pairs close)
            # sorted_pairs = sorted(pair_scores, key=lambda x: -x[2])
            #
            # for (i, j, _) in sorted_pairs:
            #     pair = frozenset((i, j))
            #     if pair in self.all_selected_pairs:
            #         continue
            #     selected_pairs.add(pair)
            #     self.all_selected_pairs.add(pair)
            #     valid = self.database.metadata.iloc[i]["identity"] == self.database.metadata.iloc[j]["identity"]
            #     if valid:
            #         self.ml_pairs.append((i, j))
            #     else:
            #         self.cl_pairs.append((i, j))
            #     if len(selected_pairs) == self.n:
            #         return self.ml_pairs, self.cl_pairs

            if self.n - len(selected_pairs) > 0:
                print(f"[Warning] Budget not completely utilized. {self.n - len(selected_pairs)} budget still remaining!")

        self.sampling_pool = sampling_pool

        return self.ml_pairs, self.cl_pairs
        # ml_pairs_ob = np.array(self.ml_pairs, dtype='object')
        # cl_pairs_ob = np.array(self.cl_pairs, dtype='object')
        # return ml_pairs_ob, cl_pairs_ob

    def get_sampling_pools(self):
        return self.sampling_pool

    def top_n_high_entropy_outliers(self, distance_matrix, clusters_dbscan, n):
        outlier_points = np.where(clusters_dbscan == -1)[0]
        inlier_points = np.setdiff1d(np.arange(len(clusters_dbscan)), outlier_points)

        cluster_ids = np.unique(clusters_dbscan[clusters_dbscan != -1])

        entropies = []
        for i in outlier_points:
            # Get distances from outlier to all inliers
            distances = distance_matrix[i, inlier_points]

            # Get the cluster assignments for those inliers
            inlier_clusters = clusters_dbscan[inlier_points]

            # Get the nearest inlier from each cluster
            min_dist_per_cluster = {}
            for cid in cluster_ids:
                cluster_mask = inlier_clusters == cid
                if np.any(cluster_mask):
                    cluster_distances = distances[cluster_mask]
                    min_dist_per_cluster[cid] = np.min(cluster_distances)

            # Convert distances into similarity scores (lower dist = higher score)
            similarities = np.exp(-np.array(list(min_dist_per_cluster.values())))
            prob_dist = similarities / similarities.sum()

            # Entropy over this probability distribution
            ent = entropy(prob_dist)
            entropies.append((i, ent))

        # Sort by entropy, descending
        entropies.sort(key=lambda x: -x[1])
        top_outliers = [i for i, _ in entropies[:n]]

        return top_outliers
