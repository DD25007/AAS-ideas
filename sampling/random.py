from sampling.base import BaseSampling
from itertools import combinations
import random
import numpy as np
from utils import constrained_clustering


class RandomSampling(BaseSampling):
    def __init__(self, database, n, gallery_size=2):
        super().__init__(database, n, gallery_size)
        all_combinations = list(combinations(database.metadata.index, gallery_size))
        self.all_pairs = set(frozenset(pair) for pair in all_combinations)

    def select(self, features=None, **kwargs):
        available_pairs = self.all_pairs - self.all_selected_pairs
        if len(available_pairs) < self.n:
            raise ValueError("Not enough available pairs to sample.")
        print(f"Selecting {self.n} pairs.")
        selected_pairs = set([frozenset(pair) for pair in random.sample(available_pairs, self.n)])
        self.all_selected_pairs.update(selected_pairs)

        for pair in selected_pairs:
            pair = tuple(pair)
            valid = self.database.metadata.iloc[pair[0]]["identity"] == self.database.metadata.iloc[pair[1]]["identity"]
            if valid:
                self.ml_pairs.append((pair[0], pair[1]))
            else:
                self.cl_pairs.append((pair[0], pair[1]))

        return self.ml_pairs, self.cl_pairs

        # ml_pairs_ob = np.array(self.ml_pairs, dtype='object')
        # cl_pairs_ob = np.array(self.cl_pairs, dtype='object')
        # return ml_pairs_ob, cl_pairs_ob


class RobustRandomSampling(BaseSampling):
    def __init__(self, database, n, gallery_size=2, thresh=0.05, min_samples=3, metric="cosine"):
        super().__init__(database, n, gallery_size)
        self.thresh = thresh
        self.min_samples = min_samples
        self.metric = metric

    def select(self, features=None, **kwargs):
        assert features is not None
        clusters = constrained_clustering(
            X=features,
            method="DBSCAN",
            eps=self.thresh,
            min_samples=self.min_samples,
            metric=self.metric,
            ml_pairs=self.ml_pairs,
            cl_pairs=self.cl_pairs)

        database_subset = self.database
        database_subset.metadata.loc[:, "pseudo_identity"] = clusters
        all_inlier_combinations = list(combinations(database_subset.metadata[database_subset.metadata["pseudo_identity"] >= 0].index, self.gallery_size))
        all_inlier_pairs = set(frozenset(pair) for pair in all_inlier_combinations)

        available_pairs = all_inlier_pairs - self.all_selected_pairs
        selected_pairs = set([frozenset(pair) for pair in random.sample(available_pairs, self.n)])
        self.all_selected_pairs.update(selected_pairs)

        for pair in selected_pairs:
            pair = tuple(pair)
            valid = database_subset.metadata.iloc[pair[0]]["identity"] == database_subset.metadata.iloc[pair[1]]["identity"]
            if valid:
                self.ml_pairs.append((pair[0], pair[1]))
            else:
                self.cl_pairs.append((pair[0], pair[1]))

        return self.ml_pairs, self.cl_pairs

        # ml_pairs_ob = np.array(self.ml_pairs, dtype='object')
        # cl_pairs_ob = np.array(self.cl_pairs, dtype='object')
        # return ml_pairs_ob, cl_pairs_ob

