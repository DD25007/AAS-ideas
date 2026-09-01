from sampling.base import BaseSampling
import numpy as np
from sklearn.metrics import pairwise_distances
from itertools import combinations


class FPS(BaseSampling):
    def __init__(self, database, n, gallery_size=2, metric="cosine"):
        super().__init__(database, n, gallery_size)
        all_combinations = list(combinations(database.metadata.index, gallery_size))
        self.all_pairs = set(frozenset(pair) for pair in all_combinations)
        self.metric = metric

    def select(self, features=None, **kwargs):
        assert features is not None
        available_pairs = np.random.choice(
            list(self.all_pairs - self.all_selected_pairs), self.n
        )
        if len(available_pairs) < self.n:
            raise ValueError("Not enough available pairs to sample.")
        print(f"Selecting {self.n} pairs.")

        def phi(p1, p2, features):
            xi, xj = p1
            xi_, xj_ = p2
            return min(
                np.linalg.norm(features[xi] - features[xi_])
                + np.linalg.norm(features[xj] - features[xj_]),
                np.linalg.norm(features[xi] - features[xj_])
                + np.linalg.norm(features[xj] - features[xi_]),
            )

        dist_phi = pairwise_distances(available_pairs, metric=phi, features=features)

        # Farthest Point Sampling (FPS) from https://minibatchai.com/2021/08/07/FPS.html
        pairs_left = np.arange(len(available_pairs))
        sample_inds = np.zeros(self.n, dtype="int")
        dists = np.ones_like(pairs_left) * float("inf")
        selected = 0
        sample_inds[0] = pairs_left[selected]
        pairs_left = np.delete(pairs_left, selected)

        # Iteratively select available_pairs for a maximum of n_samples
        selected_pairs = {
            frozenset(
                (
                    list(available_pairs[sample_inds[0]])[0],
                    list(available_pairs[sample_inds[0]])[1],
                )
            )
        }
        i = 1
        if len(available_pairs) < self.n / 2:
            print(
                "Warning! Number of available_pairs in the sampling pool are not sufficient for the budget."
            )
        while len(selected_pairs) < self.n and len(pairs_left) > 0:
            # Find the distance to the last added point in selected
            # and all the others
            last_added = sample_inds[i - 1]

            dist_to_last_added_point = dist_phi[last_added, pairs_left]

            # If closer, updated distances
            dists[pairs_left] = np.minimum(dist_to_last_added_point, dists[pairs_left])

            # We want to pick the one that has the largest nearest neighbour
            # distance to the sampled available_pairs
            selected = np.argmax(dists[pairs_left])
            sample_inds[i] = pairs_left[selected]

            if (
                frozenset(
                    (
                        list(available_pairs[sample_inds[i]])[0],
                        list(available_pairs[sample_inds[i]])[1],
                    )
                )
                not in self.all_selected_pairs
            ):
                selected_pairs.add(
                    frozenset(
                        (
                            list(available_pairs[sample_inds[i]])[0],
                            list(available_pairs[sample_inds[i]])[1],
                        )
                    )
                )
                if len(selected_pairs) == self.n:
                    break
            else:
                import pdb

                pdb.set_trace()
            # Update pairs_left
            pairs_left = np.delete(pairs_left, selected)
            i += 1
        self.all_selected_pairs.update(selected_pairs)

        for pair in selected_pairs:
            pair = tuple(pair)
            valid = (
                self.database.metadata.iloc[pair[0]]["identity"]
                == self.database.metadata.iloc[pair[1]]["identity"]
            )
            if valid:
                self.ml_pairs.append((pair[0], pair[1]))
            else:
                self.cl_pairs.append((pair[0], pair[1]))

        return self.ml_pairs, self.cl_pairs
