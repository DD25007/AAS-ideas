from sampling.base import BaseSampling
from active_semi_clustering.active.pairwise_constraints import ExampleOracle as ExampleOracle_, \
    ExploreConsolidate as ExploreConsolidate_, \
    MinMax as MinMax_
import numpy as np
from scipy.spatial.distance import pdist, squareform
import math

from utils import constrained_clustering


class MinMax(BaseSampling):
    def __init__(self, database, n, gallery_size=2):
        super().__init__(database, n, gallery_size)
        self.K = 0

    def select(self, features=None, **kwargs):
        assert features is not None
        self.K = np.unique(constrained_clustering(features, method="FINCH", ml_pairs=self.ml_pairs, cl_pairs=self.cl_pairs)).shape[0]
        oracle = ExampleOracle_(self.database.metadata.identity.values, max_queries_cnt=self.n)
        active_learner = MinMax_(n_clusters=self.K)
        active_learner.fit(features, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_
        self.ml_pairs.extend(pairwise_constraints[0])
        self.cl_pairs.extend(pairwise_constraints[1])
        return self.ml_pairs, self.cl_pairs


class ExploreConsolidate(BaseSampling):
    def __init__(self, database, n, gallery_size=2):
        super().__init__(database, n, gallery_size)
        self.K = 0

    def select(self, features=None, **kwargs):
        assert features is not None
        self.K = np.unique(constrained_clustering(features, method="FINCH", ml_pairs=self.ml_pairs, cl_pairs=self.cl_pairs)).shape[0]
        oracle = ExampleOracle_(self.database.metadata.identity.values, max_queries_cnt=self.n)
        active_learner = ExploreConsolidate_(n_clusters=self.K)
        active_learner.fit(features, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_
        self.ml_pairs.extend(pairwise_constraints[0])
        self.cl_pairs.extend(pairwise_constraints[1])
        return self.ml_pairs, self.cl_pairs
