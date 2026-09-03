from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import gallery_cache

import random
import os.path as osp
import os
import pandas as pd

from ..utils.data import BaseImageDataset
from collections import defaultdict

import os
import numpy as np
import pandas as pd
from torchvision import transforms as T

from wildlife_tools.data import WildlifeDataset
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier, TopkClassifier
from wildlife_tools.features import DeepFeatures, ClipFeatures
from wildlife_datasets import analysis, datasets, splits
from scipy.spatial import distance

# byte-identical to the repo-root evaluation.py, so use the single copy (Todo.md C2)
import evaluation
from spcl.utils.faiss_rerank import compute_jaccard_distance
from spcl.utils.wildlife_utils import get_model, set_seed


class Wildlife(BaseImageDataset):
    """
    WildlifeReID-10k
    """
    dataset_dir = 'wildlifereid-10k'

    def __init__(self, root, verbose=True, subset='combined', relabel=False, dataset_split_seed=0, images_per_id=5, metric='precomputed', **kwargs):
        super(Wildlife, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_split_seed = dataset_split_seed
        self.images_per_id = images_per_id
        self.metric = metric
        self.subset = subset
        self.train_transform = T.Compose([
            T.Resize(size=(224, 224)),
            T.RandomRotation(180),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.test_transform = T.Compose([
            T.Resize(size=(224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.matcher = CosineSimilarity()
        set_seed(self.dataset_split_seed)
        # self.img_dir = osp.join(self.dataset_dir, 'images')
        self.img_dir = self.dataset_dir
        if subset != "WII":
            # low_memory=False silences the DtypeWarning from the mixed-type `date`
            # column; the resulting frame is .equals() to the chunked one (Todo.md W3).
            self.csv =  pd.read_csv(os.path.join(self.dataset_dir, 'metadata.csv'), low_memory=False)
            self.check_before_run()

        train, query, gallery = self.process_split(subset=subset, relabel=relabel)
        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print('=> Wildlife loaded')
            self.print_dataset_statistics(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        # if not osp.exists(self.split_dir):
        #     raise RuntimeError('"{}" is not available'.format(self.split_dir))
        # if not osp.exists(self.train_list):
        #     raise RuntimeError('"{}" is not available'.format(self.train_list))
        # if self.test_size not in [800, 1600, 2400]:
        #     raise RuntimeError('"{}" is not available'.format(self.test_size))
        # if not osp.exists(self.test_list):
        #     raise RuntimeError('"{}" is not available'.format(self.test_list))

    def get_pid2label(self, pids):
        pid_container = set(pids)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label

    def parse_img_pids(self, nl_pairs, pid2label=None):
        # il_pair is the pairs of img name and label
        output = []
        for info in nl_pairs:
            name = info[0]
            pid = info[1]
            if pid2label is not None:
                pid = pid2label[pid]
            camid = 0  # don't have camid information use 0 for all
            img_path = osp.join(self.img_dir, name)
            output.append((img_path, pid, camid))
        return output

    def process_split(self, subset='combined', relabel=False):
        if subset != "WII":
            # read train paths
            self.image_root = self.dataset_dir
            if subset != "combined":
                self.csv = self.csv.query(f'dataset == "{subset}"')
            else: # datasets from ['AmvrakikosTurtles', 'CatIndividualImages', 'Chicks4FreeID','CowDataset','DogFaceNet','MPDD','MultiCamCows2024','PolarBearVidID','PrimFace','ReunionTurtles','SeaStarReID2023','SouthernProvinceTurtles','ZakynthosTurtles']
                self.csv = self.csv.query('dataset in ["AmvrakikosTurtles", "CatIndividualImages", "Chicks4FreeID", "CowDataset", "DogFaceNet", "MPDD", "MultiCamCows2024", "PolarBearVidID", "PrimFace", "ReunionTurtles", "SeaStarReID2023", "SouthernProvinceTurtles", "ZakynthosTurtles"]')
            self.train_split = self.csv.query('split == "train"')
            self.test_split = self.csv.query('split == "test"')
        else:
            dataset_path = '/mnt/nas/users/jeetm/p_project_01/cnnarch/WII_dataset_densenet121_usam/data'
            self.image_root = os.path.join(dataset_path, 'cropped_images')
            self.dataset_dir = dataset_path
            self.img_dir = self.image_root
            self.train_split = pd.read_csv(os.path.join(dataset_path, 'annotations', 'train.csv')).rename(columns={'Entity': 'identity', 'Image_Name': 'path'})
            self.csv = self.train_split.copy()  # For WII, we use the train split as the metadata
            self.test_split = pd.read_csv(os.path.join(dataset_path, 'annotations', 'test.csv')).rename(columns={'Entity': 'identity', 'Image_Name': 'path'})

        identity_train = np.unique(self.train_split['identity'].to_numpy())  # set of identities in the train set
        identity_test = np.unique(self.test_split['identity'].to_numpy())  # set of identities in the test set

        # Split the set of identities in the train set into known (labeled) and unknown (unlabeled) sets
        if subset != "WII":
            self.identity_train_labeled, self.identity_train_unlabeled = self.split_train_into_gallery_and_unknown(identity_train)
            print(f"Number of identities in the train set: {len(identity_train)}")
            print(f"Number of known identities in the train set: {len(self.identity_train_labeled)}")
            print(f"Number of unknown identities in the train set: {len(self.identity_train_unlabeled)}")
            print(f"Number of identities in the test set: {len(identity_test)}")

            identity_test_only = list(set(identity_test) - set(identity_train))  # Unseen IDs w.r.t. the gallery
            identity_test_closed = list(set(identity_test).intersection(set(self.identity_train_labeled)))  # IDs observed in the gallery
            print(f"Number of unknown identities only in the test set: {len(identity_test_only)}")
            print(f"Number of known identities in the test set: {len(identity_test_closed)}")

            test_identity_mask = self.test_split['identity'].isin(identity_test_closed)
            self.test_split_closed = self.test_split[test_identity_mask]
        else:
            self.identity_test_closed, self.identity_test_open = self.split_train_into_gallery_and_unknown(identity_test)
            print(f"Number of identities in the test set: {len(identity_test)}")
            print(f"Number of closed identities (in both gallery and query sets): {len(self.identity_test_closed)}")
            print(f"Number of open identities (only in query set): {len(self.identity_test_open)}")

            test_identity_mask = self.test_split['identity'].isin(self.identity_test_closed)
            self.test_split_closed = self.test_split[test_identity_mask]


        self.default_train = WildlifeDataset(
            metadata=self.train_split,
            root=self.image_root,
            transform=self.test_transform,
        )


        self.default_query = WildlifeDataset(
            metadata=self.test_split,
            root=self.image_root,
            transform=self.test_transform,
        )

        self.default_closed_query = WildlifeDataset(
            metadata=self.test_split_closed,
            root=self.image_root,
            transform=self.test_transform,
        )

        megadescriptor_model = get_model("hf-hub:BVRA/MegaDescriptor-L-224", eval=True)
        if subset != "WII":
            self.get_representative_gallery_from_train(model=megadescriptor_model)
        else:
            self.get_representative_gallery_from_test(model=megadescriptor_model)
        
        # default_train.metadata was updated when creating the representative gallery
        self.train_split = self.default_train.metadata
        self.test_split = self.default_query.metadata
        self.test_split_closed = self.default_closed_query.metadata

        train_pid_dict = defaultdict(list)
        for _, row in self.train_split.iterrows():
            name = row['path']
            pid = row['identity']
            train_pid_dict[pid].append([name, pid])
        train_pids = list(train_pid_dict.keys())
        num_train_pids = len(train_pids)

        test_pid_dict = defaultdict(list)
        for _, row in self.test_split.iterrows():
            name = row['path']
            pid = row['identity']
            test_pid_dict[pid].append([name, pid])
        test_pids = list(test_pid_dict.keys())
        num_test_pids = len(test_pids)

        gallery_pid_dict = defaultdict(list)
        for _, row in self.default_gallery.metadata.iterrows():
            name = row['path']
            pid = row['identity']
            gallery_pid_dict[pid].append([name, pid])
        gallery_pids = list(gallery_pid_dict.keys())
        num_gallery_pids = len(gallery_pids)

        train_data = []
        query_data = []
        gallery_data = []

        # for train ids, all images are used in the train set.
        for pid in train_pids:
            imginfo = train_pid_dict[pid]  # imginfo include image name and id
            train_data.extend(imginfo)

        # for each test id, random choose one image for gallery
        # and the other ones for query.
        # if theres only one image for a test id, then
        # it is used for query.
        for pid in test_pids:
            imginfo = test_pid_dict[pid]
            query_data.extend(imginfo)
        if relabel:
            train_pid2label = self.get_pid2label(train_pids)
        else:
            train_pid2label = None
        # for key, value in train_pid2label.items():
        #     print('{key}:{value}'.format(key=key, value=value))

        for pid in gallery_pids:
            imginfo = gallery_pid_dict[pid]
            gallery_data.extend(imginfo)

        train = self.parse_img_pids(train_data, train_pid2label)
        query = self.parse_img_pids(query_data)
        gallery = self.parse_img_pids(gallery_data)
        return train, query, gallery
    
    def get_representative_gallery_from_train(self, model, device="cuda", num_workers=8, batch_size=32):
        # Split the training set into labeled and unlabeled sets.
        # The labeled set is used as the representative gallery set for re-identification
        representative_gallery = []
        K = self.images_per_id

        # O4: selection uses a FIXED MegaDescriptor, not the model being trained, so it is
        # a pure function of the metadata -- cache the chosen rows and skip the ~29k-image
        # forward pass. See gallery_cache.py.
        _cache_key = gallery_cache.build_key(
            subset=getattr(self, "subset", getattr(self, "dataset", None)),
            split_seed=self.dataset_split_seed,
            images_per_id=K,
            metric=self.metric,
            model_name="hf-hub:BVRA/MegaDescriptor-L-224",
            train_metadata=self.default_train.metadata,
            labeled_identities=self.identity_train_labeled,
        )
        _cached = gallery_cache.load(_cache_key)
        if _cached is not None:
            representative_gallery = self.default_train.metadata.loc[_cached]
            self.default_train.metadata = self.default_train.metadata.drop(_cached)
            self.default_gallery = WildlifeDataset(
                metadata=representative_gallery,
                root=self.image_root,
                transform=self.test_transform,
            )
            print(f"Gallery selection loaded from cache ({len(_cached)} images).")
            return representative_gallery

        feature_extractor = DeepFeatures(model, device=device, num_workers=num_workers, batch_size=batch_size)
        train_features = np.array(feature_extractor(self.default_train).features)

        df = self.default_train.metadata.copy()
        # filter using self.identity_train_labeled
        df = df[df['identity'].isin(self.identity_train_labeled)]

        for identity, group in df.groupby('identity'):
            # Select the corresponding images and features for each identity
            indices = group.index.tolist()

            identity_features = [train_features[idx] for idx in indices]
            centroid = np.mean(identity_features, axis=0)

            # sort identity samples by cosine distance to centroid
            if self.metric == "precomputed":
                distances = distance.cdist(identity_features, [centroid], metric='jaccard').flatten()
            else:
                distances = distance.cdist(identity_features, [centroid], metric=self.metric).flatten()
            # n_identity_features = len(identity_features)
            # X = np.stack([identity_features, centroid], axis=0)
            # distances = compute_jaccard_distance(X)
            # idxs = np.ix_(np.arange(n_identity_features), range(n_identity_features - 1, 2*n_identity_features))
            # assert len(distances) == 2* n_identity_features
            # distances = distances[idxs].flatten()
            sorted_indices = np.argsort(distances)

            # distance_matrix = 

            if len(sorted_indices) <= K:
                representative_gallery.append(group.iloc[sorted_indices])
                to_remove = group.index[sorted_indices]
                self.default_train.metadata = self.default_train.metadata.drop(to_remove)
                continue
            # central points are the closest 50% points from the centroid of the cluster
            central_points = sorted_indices[:int(np.ceil(len(sorted_indices) * 0.5))]

            if len(central_points) <= K:
                representative_gallery.append(group.iloc[sorted_indices[:K]])
                to_remove = group.index[sorted_indices[:K]]
                self.default_train.metadata = self.default_train.metadata.drop(to_remove)
                continue

            # sample K points from central_points
            sampled_indices = np.random.choice(sorted_indices[:len(central_points)], K, replace=False)
            # get the sampled images
            representative_gallery.append(group.iloc[sampled_indices])
            
            to_remove = group.index[sampled_indices]
            # remove indices from the train set
            self.default_train.metadata = self.default_train.metadata.drop(to_remove)

        representative_gallery = pd.concat(representative_gallery)
        gallery_cache.save(_cache_key, representative_gallery.index.values)

        self.default_gallery = WildlifeDataset(
            metadata=representative_gallery,
            root=self.image_root,
            transform=self.test_transform,
        )

        return representative_gallery

    def get_representative_gallery_from_test(self, model, device="cuda", num_workers=8, batch_size=32):
        # Split the test set into labeled and unlabeled sets.
        # The labeled set is used as the representative gallery set for re-identification
        representative_gallery = []
        K = self.images_per_id

        feature_extractor = DeepFeatures(model, device=device, num_workers=num_workers, batch_size=batch_size)
        test_features = np.array(feature_extractor(self.default_query).features)

        # df = self.default_query.metadata.copy()
        df = self.test_split.copy()
        # filter using self.identity_train_labeled
        df = df[df['identity'].isin(self.identity_test_closed)]

        for identity, group in df.groupby('identity'):
            # Select the corresponding images and features for each identity
            indices = group.index.tolist()

            identity_features = [test_features[idx] for idx in indices]
            centroid = np.mean(identity_features, axis=0)

            # sort identity samples by cosine distance to centroid
            if self.metric == "precomputed":
                distances = distance.cdist(identity_features, [centroid], metric='jaccard').flatten()
            else:
                distances = distance.cdist(identity_features, [centroid], metric=self.metric).flatten()
            sorted_indices = np.argsort(distances)

            if len(sorted_indices) <= K:
                representative_gallery.append(group.iloc[sorted_indices])
                to_remove = group.index[sorted_indices]
                self.test_split = self.test_split.drop(to_remove)
                self.test_split_closed = self.test_split_closed.drop(to_remove)
                continue
            # central points are the closest 50% points from the centroid of the cluster
            central_points = sorted_indices[:int(np.ceil(len(sorted_indices) * 0.5))]

            if len(central_points) <= K:
                representative_gallery.append(group.iloc[sorted_indices[:K]])
                to_remove = group.index[sorted_indices[:K]]
                self.test_split = self.test_split.drop(to_remove)
                self.test_split_closed = self.test_split_closed.drop(to_remove)
                continue

            # sample K points from central_points
            sampled_indices = np.random.choice(sorted_indices[:len(central_points)], K, replace=False)
            # get the sampled images
            representative_gallery.append(group.iloc[sampled_indices])
            to_remove = group.index[sampled_indices]
            # remove indices from the test set
            self.test_split = self.test_split.drop(to_remove)
            self.test_split_closed = self.test_split_closed.drop(to_remove)

        representative_gallery = pd.concat(representative_gallery)

        self.default_gallery = WildlifeDataset(
            metadata=representative_gallery,
            root=self.image_root,
            transform=self.test_transform,
        )

        return representative_gallery

    def split_train_into_gallery_and_unknown(self, identities):
        # Fix the seed to freeze the splits
        set_seed(102)
        np.random.shuffle(identities)
        num_labeled = int(len(identities) * 0.8)
        # Reset the seed as per the user-specific seed for AL
        set_seed(self.dataset_split_seed)
        return identities[:num_labeled], identities[num_labeled:]







# NOTE: a second `WildlifeDatasets` class lived here and was removed on 2026-09-01.
# It was dead: `main.py` uses the repo-root `dataset.wildlife.WildlifeDatasets`, and
# `train_al.py`'s import of this one was dropped as unused. Recover from git if needed.
