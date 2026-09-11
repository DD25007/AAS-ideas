from .base import BaseSampling
import numpy as np
import math
import time


class CountingClusters(BaseSampling):
    def __init__(self, ratio=1.0, args=None):
        super().__init__(args=args)
        self.K = 0
        self.ratio = ratio

    def select(self, features=None, rerank_dist=None, **kwargs):
        assert rerank_dist is not None
        identities = np.array(self.gt_labels)
        gt_s_ij = (identities[:, None] == identities[None, :]).astype(np.uint8)

        dists = rerank_dist
        s_ij = 1 - dists
        np.fill_diagonal(s_ij, 1.0)  # add 1s to the matrix diagonal
        s_ij_sm = s_ij

        n_hat = []
        OBSERVATIONS = len(gt_s_ij)
        for i in range(OBSERVATIONS):
            n_hat.append(np.sum(s_ij_sm[i]))

        Nv = int(np.ceil(math.sqrt(self.n / self.ratio)))
        Nn = int(np.ceil(self.n / Nv))
        time_nested_is = time.time()
        self.K, conf_inter = self.nested_is(
            gt_s_ij, s_ij_sm, Nv, Nn, ci=True, n_hat=n_hat
        )
        print("nested_is took:", time.time() - time_nested_is, "seconds")
        return self.ml_pairs, self.cl_pairs

    def nested_is(self, gt_s_ij, s_ij, N_v, N_n, n_hat=None, ci=False):
        """
        Function to estimate total number of categories in a dataset using nested importance sampling.

        input:
            - gt_s_ij ([lists]): list of lists with ground truth s_ij values. s_ij is 1 if elements belong to same
                                 category, 0 otherwise. Size: (N,N)
            - s_ij ([lists]):    list of lists with s_ij values. s_ij is the similarity metric between
                                 each of the N elements in the dataset. Size: (N,N).
                                 It is produced between each sample features using 1 -dist, where dist can be cosine,
                                 euclidean, or any distance.
            - N_v (int):         Number of sampled vertices
            - N_n (int):         Number of sampled neighbors per vertex
            - n_hat (list):      List with n_hat values (can be calculated only once to save time)
            - ci (bool):         True to return confidence intervals

        output:
            - f_hat (float):     total category count estimate.
            - ci (float):        95% confidence intervals
        """
        OBSERVATIONS = len(gt_s_ij)

        if not n_hat:  # have to be computed only once (saves time)
            # Compute \hat{n}_i
            n_hat = []
            for i in range(OBSERVATIONS):
                n_hat.append(np.sum(s_ij[i]))

        # Proposal distribution to sample a set of n vertices v_i
        Q = (1 / np.array(n_hat)) / np.sum((1 / np.array(n_hat)))
        sampled_vertices = list(
            np.random.choice(list(range(OBSERVATIONS)), N_v, p=Q, replace=True)
        )

        sampled_neighbors = []  # sampled neighbors per sampled vertex
        q_all = []
        for v_i in sampled_vertices:
            q = (s_ij[v_i]) / np.sum(
                (s_ij[v_i])
            )  # proposal distribution to sample a set of neighbors for each sampled vertex
            q_all.append(q)
            sampled_neighbors.append(
                [v_i]
                + list(
                    np.random.choice(
                        list(range(len(s_ij[v_i]))), N_n - 1, p=q, replace=True
                    )
                )
            )

        selected_pairs = set()
        # Estimate \bar{n}
        sum_cc = 0
        over_budget = False
        for i, v_i in enumerate(sampled_vertices):
            if over_budget:
                break
            sum_n_bar = 0
            for j, v_j in enumerate(sampled_neighbors[i]):
                if len(selected_pairs) >= self.n:
                    over_budget = True
                    break
                if frozenset((v_i, v_j)) in self.all_selected_pairs:
                    continue
                self.all_selected_pairs.add(frozenset((v_i, v_j)))
                selected_pairs.add(frozenset((v_i, v_j)))
                if gt_s_ij[v_i][v_j] == 1:
                    if v_i != v_j:
                        self.ml_pairs.append((v_i, v_j))
                else:
                    self.cl_pairs.append((v_i, v_j))
                sum_n_bar += gt_s_ij[v_i][v_j] / q_all[i][v_j]
            n_bar = sum_n_bar / len(sampled_neighbors[i])
            if n_bar > 0 and Q[v_i] > 0:
                sum_cc += (1 / n_bar) * (1 / Q[v_i])
        f_hat = sum_cc / len(sampled_vertices)

        if not ci:
            return f_hat
        else:
            # Estimate variance and calculate confidence intervals
            w_ci = 0
            for i, v_i in enumerate(sampled_vertices):
                sum_n_bar = 0
                for v_j in sampled_neighbors[i]:
                    sum_n_bar += gt_s_ij[v_i][v_j] / q_all[i][v_j]
                n_bar = sum_n_bar / len(sampled_neighbors[i])
                # W1 (second occurrence): same divide-by-zero as the f_hat loop. n_bar is a
                # numpy scalar, so n_bar == 0 gives inf + RuntimeWarning rather than raising,
                # and var_hat/ci become inf. Skip the term instead, matching the guard above.
                if n_bar > 0 and Q[v_i] > 0:
                    w_ci += ((1 / n_bar) * (1 / Q[v_i]) - f_hat) ** 2
            var_hat = w_ci / N_v  # estimated variance
            ci = 1.96 * (np.sqrt(var_hat / N_v))  # 95% confidence intervals
            return f_hat, ci
