import numpy as np
import torch


class BaseSampling:
    def __init__(self, args=None):
        self.all_selected_pairs = set()
        self.ml_pairs = []
        self.cl_pairs = []
        self.gt_labels = None
        self.n = 0
        self.args = args

    def set_gt_labels(self, gt_labels):
        self.gt_labels = gt_labels
        n_samples = len(gt_labels)
        if self.args.dataset == "wildlife":
            total_edges = (n_samples * (n_samples - 1)) // 2
            self.n = int(total_edges * 0.02/100)
        else:
            self.n = np.unique(gt_labels).shape[0]

    def select(self, features=None, **kwargs):
        # TODO: Implemeted in the inherited classes
        pass

    def annotate(self, selected_pairs):
        for pair in selected_pairs:
            pair = tuple(pair)
            valid = self.gt_labels[pair[0]] == self.gt_labels[pair[1]]
            if valid:
                self.ml_pairs.append((pair[0], pair[1]))
            else:
                self.cl_pairs.append((pair[0], pair[1]))

    def get_selected_pairs(self):
        return self.all_selected_pairs

    def set_constraints(self, ml_pairs, cl_pairs):
        self.ml_pairs = ml_pairs
        self.cl_pairs = cl_pairs
        self.all_selected_pairs = set(frozenset(pair) for pair in ml_pairs + cl_pairs)
    
    def compute_selected_gt_identity_counts(self):
        counts_by_id = {}
        for pair in self.all_selected_pairs:
            pair = tuple(pair)
            if len(pair) == 1:
                pair0id = self.database.metadata.iloc[pair[0]]["identity"]
                if pair0id not in counts_by_id:
                    counts_by_id[pair0id] = 0
                counts_by_id[pair0id] += 1
            else:
                pair0id = self.database.metadata.iloc[pair[0]]["identity"]
                pair1id = self.database.metadata.iloc[pair[1]]["identity"]
                if pair0id not in counts_by_id:
                    counts_by_id[pair0id] = 0
                if pair1id not in counts_by_id:
                    counts_by_id[pair1id] = 0
                counts_by_id[pair0id] += 1
                counts_by_id[pair1id] += 1

            
        return counts_by_id

    def compute_selected_pseudo_identity_counts(self):
        counts_by_id = {}

        try:
            for pair in self.all_selected_pairs:
                pair = tuple(pair)
                if len(pair) == 1:
                    pair0id = self.database.metadata.iloc[pair[0]]["pseudo_identity"]
                    if pair0id not in counts_by_id:
                        counts_by_id[pair0id] = 0
                    counts_by_id[pair0id] += 1
                else:
                    pair0id = self.database.metadata.iloc[pair[0]]["pseudo_identity"]
                    pair1id = self.database.metadata.iloc[pair[1]]["pseudo_identity"]
                    if pair0id not in counts_by_id:
                        counts_by_id[pair0id] = 0
                    if pair1id not in counts_by_id:
                        counts_by_id[pair1id] = 0
                    counts_by_id[pair0id] += 1
                    counts_by_id[pair1id] += 1
        except KeyError:
            print("KeyError: 'pseudo_identity' not found in metadata. Returning empty counts.")
            return {}

            
        return counts_by_id
    

    def set_memory_bank(self, features):
        features = torch.Tensor(features)
        self.memory_bank = features / torch.linalg.norm(features, dim=1)[:, None]

    def batch_memory_bank_update(self, batch_features, batch_indices, momentum=0.2):
        self.memory_bank[batch_indices] = momentum * self.memory_bank[batch_indices] + (1 - momentum) * batch_features.detach().cpu().numpy()
