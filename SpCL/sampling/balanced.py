from sampling.base import BaseSampling
import numpy as np
from spcl.utils.constrained_clustering import constrained_clustering
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN


class LBAS(BaseSampling):
    def __init__(self, thresh=0.05, min_samples=3, metric="cosine", sample_outliers=False, args=None):
        super().__init__(args=args)
        self.thresh = thresh
        self.min_samples = min_samples
        self.metric = metric
        self.sample_outliers = sample_outliers
        self.switch_count = 0

    def select(self, features=None, rerank_dist=None, **kwargs):
        assert features is not None and rerank_dist is not None
        features = features.to('cpu').numpy()
        cluster = DBSCAN(eps=self.thresh, min_samples=4, metric='precomputed', n_jobs=-1)
        labels = cluster.fit_predict(rerank_dist)
        clusters = constrained_clustering(
            labels=labels,
            rerank_dist=rerank_dist,
            method="DBSCAN",
            min_samples=self.min_samples,
            metric=self.metric,
            ml_pairs=self.ml_pairs,
            cl_pairs=self.cl_pairs,
        )

        triplets = []
        unique_clusters = set(clusters) - {-1}  # Ignore noise points

        # Compute cosine distance matrix
        dist_matrix = rerank_dist

        for cluster in unique_clusters:
            cluster_indices = np.where(clusters == cluster)[0]  # Indices of points in the cluster

            for i in cluster_indices:
                pos_indices = cluster_indices[cluster_indices != i]
                # Find farthest positive example within the same cluster
                cluster_distances = dist_matrix[i, pos_indices]
                farthest_idx = pos_indices[np.argmax(cluster_distances)]

                # Find closest negative example from another cluster
                if self.sample_outliers:
                    non_cluster_indices = np.where(clusters != cluster)[0]
                else:
                    non_cluster_indices = np.where((clusters != cluster) & (clusters != -1))[0]
                nearest_neg_idx = non_cluster_indices[np.argmin(dist_matrix[i, non_cluster_indices])]

                triplets.append((i, farthest_idx, nearest_neg_idx))

        triplets = np.array(triplets)
        p_i = dist_matrix[triplets[:, 0], triplets[:, 2]] / (dist_matrix[triplets[:, 0], triplets[:, 1]] + dist_matrix[triplets[:, 0], triplets[:, 2]])
        epsilon = 1e-7
        u_i = -p_i * np.log(p_i + epsilon) - (1 - p_i) * np.log(1 - p_i + epsilon)

        def phi(t1, t2, features):
            xi, xj, xk = t1
            xi_, xj_, xk_ = t2
            ri = features[int(xk)] + features[int(xj)] - 2*features[int(xi)]
            ri_ = features[int(xk_)] + features[int(xj_)] - 2 * features[int(xi_)]
            return np.linalg.norm(features[int(xi)] - features[int(xi_)]) + np.linalg.norm(ri - ri_)

        dist_phi = pairwise_distances(triplets, metric=phi, features=features)

        # Farthest Point Sampling (FPS) from https://minibatchai.com/2021/08/07/FPS.html
        triplets_left = np.arange(len(triplets))
        sample_inds = np.zeros(len(triplets), dtype='int')
        dists = np.ones_like(triplets_left) * float('inf')
        selected = np.argmax(u_i)
        sample_inds[0] = triplets_left[selected]
        triplets_left = np.delete(triplets_left, selected)

        # Iteratively select triplets for a maximum of n_samples
        selected_pairs = {
            frozenset((triplets[sample_inds[0], 0], triplets[sample_inds[0], 1])),
            frozenset((triplets[sample_inds[0], 0], triplets[sample_inds[0], 2]))
        }
        i = 1
        if len(triplets) < self.n/2:
            print("Warning! Number of triplets in the sampling pool are not sufficient for the budget.")
        while len(selected_pairs) < self.n and len(triplets_left) > 0:
            # Find the distance to the last added point in selected
            # and all the others
            last_added = sample_inds[i - 1]

            dist_to_last_added_point = dist_phi[last_added, triplets_left]

            # If closer, updated distances
            dists[triplets_left] = np.minimum(dist_to_last_added_point, dists[triplets_left])

            # We want to pick the one that has the largest nearest neighbour
            # distance to the sampled triplets
            selected = np.argmax(dists[triplets_left])
            sample_inds[i] = triplets_left[selected]

            # Update triplets_left
            triplets_left = np.delete(triplets_left, selected)
            if frozenset((triplets[sample_inds[i], 0], triplets[sample_inds[i], 1])) not in self.all_selected_pairs:
                selected_pairs.add(frozenset((triplets[sample_inds[i], 0], triplets[sample_inds[i], 1])))
                if len(selected_pairs) == self.n:
                    break
            if frozenset((triplets[sample_inds[i], 0], triplets[sample_inds[i], 2])) not in self.all_selected_pairs:
                selected_pairs.add(frozenset((triplets[sample_inds[i], 0], triplets[sample_inds[i], 2])))
            i += 1

        self.all_selected_pairs.update(selected_pairs)

        self.annotate(selected_pairs)

        return self.ml_pairs,  self.cl_pairs


class GBAS(BaseSampling):
    def __init__(self, thresh=0.05, min_samples=3, metric="cosine", sample_outliers=False, args=None):
        super().__init__(args=args)
        self.thresh = thresh
        self.min_samples = min_samples
        self.metric = metric
        self.sample_outliers = sample_outliers

    def select(self, features=None, rerank_dist=None, **kwargs):
        assert rerank_dist is not None
        cluster = DBSCAN(eps=self.thresh, min_samples=4, metric='precomputed', n_jobs=-1)
        labels = cluster.fit_predict(rerank_dist)
        clusters = constrained_clustering(
            labels=labels,
            rerank_dist=rerank_dist,
            method="DBSCAN",
            min_samples=self.min_samples,
            metric=self.metric,
            ml_pairs=self.ml_pairs,
            cl_pairs=self.cl_pairs,
        )

        # Compute cosine distance matrix
        dist_matrix = rerank_dist

        mcd = self.compute_mcd(dist_matrix, clusters, self.sample_outliers)

        selected_pairs = self.get_n_closest_to_mcd(dist_matrix, mcd, clusters, self.sample_outliers)

        self.all_selected_pairs.update(selected_pairs)

        self.annotate(selected_pairs)

        return self.ml_pairs,  self.cl_pairs

    def compute_mcd(self, distance_matrix, labels, sample_outliers=False):
        N = distance_matrix.shape[0]
        assert distance_matrix.shape == (N, N)

        # Upper triangle indices without diagonal (to avoid duplicate/self-pairs)
        triu_indices = np.triu_indices(N, k=1)

        # Get pairwise distances and pair labels
        pair_distances = distance_matrix[triu_indices]
        label_i = labels[triu_indices[0]]
        label_j = labels[triu_indices[1]]

        if not sample_outliers:
            non_outlier_mask = (label_i != -1) & (label_j != -1)
            pair_distances = pair_distances[non_outlier_mask]
            label_i = label_i[non_outlier_mask]
            label_j = label_j[non_outlier_mask]

        # Classify pairs as positive (1) or negative (0)
        pair_types = (label_i == label_j).astype(int)

        # Sort pairs by distance
        sorted_indices = np.argsort(pair_distances)
        sorted_distances = pair_distances[sorted_indices]
        sorted_labels = pair_types[sorted_indices]

        # Cumulative counts of positives and negatives
        cum_positives = np.cumsum(sorted_labels)
        cum_negatives = np.cumsum(1 - sorted_labels)

        # Find index where the difference is minimized
        diff = np.abs(cum_positives - cum_negatives)
        min_diff_idx = np.argmin(diff)

        # MCD is the distance at this index
        mcd = sorted_distances[min_diff_idx]
        return mcd

    def get_n_closest_to_mcd(self, distance_matrix, mcd, labels, sample_outliers=False):
        N = distance_matrix.shape[0]

        # Get upper triangular indices (excluding self-pairs)
        triu_indices = np.triu_indices(N, k=1)

        # Extract distances and corresponding pair indices
        pair_distances = distance_matrix[triu_indices]
        pair_indices = list(zip(triu_indices[0], triu_indices[1]))

        if not sample_outliers:
            label_i = labels[triu_indices[0]]
            label_j = labels[triu_indices[1]]
            non_outlier_mask = (label_i != -1) & (label_j != -1)
            pair_distances = pair_distances[non_outlier_mask]
            pair_indices = [pair_indices[i] for i in np.where(non_outlier_mask)[0]]

        # Compute how far each distance is from the MCD
        mcd_diff = np.abs(pair_distances - mcd)

        # Combine distances and pairs
        all_info = list(zip(mcd_diff, pair_indices))

        # Sort by how close to MCD
        all_info.sort(key=lambda x: x[0])

        # Fetch corresponding pairs
        selected = set()
        for _, (i, j) in all_info:
            if frozenset((i, j)) not in self.all_selected_pairs:
                selected.add(frozenset((i, j)))
                if len(selected) == self.n:
                    break
        return selected
