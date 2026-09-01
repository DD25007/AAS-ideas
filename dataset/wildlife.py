import os
import numpy as np
import pandas as pd
from torchvision import transforms as T

from wildlife_tools.data import WildlifeDataset
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.features import DeepFeatures
from scipy.spatial import distance


from utils import get_model, set_seed


class WildlifeDatasets:

    def __init__(
        self,
        dataset,
        dataset_path,
        dataset_split_seed=0,
        images_per_id=1,
        metric="cosine",  # 'euclidean', 'precomputed', 'cosine'
    ):
        self.dataset = dataset
        self.metric = metric
        self.dataset_split_seed = dataset_split_seed
        self.images_per_id = images_per_id
        self.train_transform = T.Compose(
            [
                T.Resize(size=(224, 224)),
                T.RandomRotation(180),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.test_transform = T.Compose(
            [
                T.Resize(size=(224, 224)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.matcher = CosineSimilarity()
        set_seed(self.dataset_split_seed)

        if dataset == "WII":
            dataset_path = "/mnt/nas/users/jeetm/p_project_01/cnnarch/WII_dataset_densenet121_usam/data"
            self.image_root = os.path.join(dataset_path, "cropped_images")
            self.train_split = pd.read_csv(
                os.path.join(dataset_path, "annotations", "train.csv")
            ).rename(columns={"Entity": "identity", "Image_Name": "path"})
            self.test_split = pd.read_csv(
                os.path.join(dataset_path, "annotations", "test.csv")
            ).rename(columns={"Entity": "identity", "Image_Name": "path"})
            self.csv = self.train_split.copy()
        else:
            self.csv = pd.read_csv(
                os.path.join(dataset_path, "wildlifereid-10k", "metadata.csv")
            )
            if dataset != "combined":
                self.csv = self.csv.query(f'dataset == "{dataset}"')
            else:
                self.csv = self.csv.query(
                    'dataset in ["AmvrakikosTurtles", "CatIndividualImages", "Chicks4FreeID", "CowDataset", "DogFaceNet", "MPDD", "MultiCamCows2024", "PolarBearVidID", "PrimFace", "ReunionTurtles", "SeaStarReID2023", "SouthernProvinceTurtles", "ZakynthosTurtles"]'
                )
            self.image_root = os.path.join(dataset_path, "wildlifereid-10k")
            self.train_split = self.csv.query('split == "train"')
            self.test_split = self.csv.query('split == "test"')

        identity_train = np.unique(
            self.train_split["identity"].to_numpy()
        )  # set of identities in the train set
        identity_test = np.unique(
            self.test_split["identity"].to_numpy()
        )  # set of identities in the test set

        # Split the set of identities in the train set into known (labeled) and unknown (unlabeled) sets
        if dataset != "WII":
            self.identity_train_labeled, self.identity_train_unlabeled = (
                self.split_train_into_gallery_and_unknown(identity_train)
            )
            print(f"Number of identities in the train set: {len(identity_train)}")
            print(
                f"Number of known identities in the train set: {len(self.identity_train_labeled)}"
            )
            print(
                f"Number of unknown identities in the train set: {len(self.identity_train_unlabeled)}"
            )
            print(f"Number of identities in the test set: {len(identity_test)}")

            identity_test_only = list(
                set(identity_test) - set(identity_train)
            )  # Unseen IDs w.r.t. the gallery
            identity_test_closed = list(
                set(identity_test).intersection(set(self.identity_train_labeled))
            )  # IDs observed in the gallery
            print(
                f"Number of unknown identities only in the test set: {len(identity_test_only)}"
            )
            print(
                f"Number of known identities in the test set: {len(identity_test_closed)}"
            )

            test_identity_mask = self.test_split["identity"].isin(identity_test_closed)
            self.test_split_closed = self.test_split[test_identity_mask]
        else:
            self.identity_test_closed, self.identity_test_open = (
                self.split_train_into_gallery_and_unknown(identity_test)
            )
            print(f"Number of identities in the test set: {len(identity_test)}")
            print(
                f"Number of closed identities (in both gallery and query sets): {len(self.identity_test_closed)}"
            )
            print(
                f"Number of open identities (only in query set): {len(self.identity_test_open)}"
            )

            test_identity_mask = self.test_split["identity"].isin(
                self.identity_test_closed
            )
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
        if dataset != "WII":
            self.get_representative_gallery_from_train(model=megadescriptor_model)
        else:
            self.get_representative_gallery_from_test(model=megadescriptor_model)

    def split_train_into_gallery_and_unknown(self, identities):
        # Fix the seed to freeze the splits
        set_seed(102)
        np.random.shuffle(identities)
        num_labeled = int(len(identities) * 0.8)
        # Reset the seed as per the user-specific seed for AL
        set_seed(self.dataset_split_seed)
        return identities[:num_labeled], identities[num_labeled:]

    def get_representative_gallery_from_train(
        self, model, device="cuda", num_workers=8, batch_size=32
    ):
        # Split the training set into labeled and unlabeled sets.
        # The labeled set is used as the representative gallery set for re-identification
        representative_gallery = []
        K = self.images_per_id

        feature_extractor = DeepFeatures(
            model, device=device, num_workers=num_workers, batch_size=batch_size
        )
        train_features = np.array(feature_extractor(self.default_train).features)

        df = self.default_train.metadata.copy()
        # filter using self.identity_train_labeled
        df = df[df["identity"].isin(self.identity_train_labeled)]

        for identity, group in df.groupby("identity"):
            # Select the corresponding images and features for each identity
            indices = group.index.tolist()

            identity_features = [train_features[idx] for idx in indices]
            centroid = np.mean(identity_features, axis=0)

            # sort identity samples by cosine distance to centroid
            if self.metric == "precomputed":
                distances = distance.cdist(
                    identity_features, [centroid], metric="jaccard"
                ).flatten()
            else:
                distances = distance.cdist(
                    identity_features, [centroid], metric=self.metric
                ).flatten()
            # import pdb; pdb.set_trace()
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
                self.default_train.metadata = self.default_train.metadata.drop(
                    to_remove
                )
                continue
            # central points are the closest 50% points from the centroid of the cluster
            central_points = sorted_indices[: int(np.ceil(len(sorted_indices) * 0.5))]

            if len(central_points) <= K:
                representative_gallery.append(group.iloc[sorted_indices[:K]])
                to_remove = group.index[sorted_indices[:K]]
                self.default_train.metadata = self.default_train.metadata.drop(
                    to_remove
                )
                continue

            # sample K points from central_points
            sampled_indices = np.random.choice(
                sorted_indices[: len(central_points)], K, replace=False
            )
            # get the sampled images
            representative_gallery.append(group.iloc[sampled_indices])

            to_remove = group.index[sampled_indices]
            # remove indices from the train set
            self.default_train.metadata = self.default_train.metadata.drop(to_remove)

        representative_gallery = pd.concat(representative_gallery)

        self.default_gallery = WildlifeDataset(
            metadata=representative_gallery,
            root=self.image_root,
            transform=self.test_transform,
        )

        return representative_gallery

    def get_representative_gallery_from_test(
        self, model, device="cuda", num_workers=8, batch_size=32
    ):
        # Split the test set into labeled and unlabeled sets.
        # The labeled set is used as the representative gallery set for re-identification
        representative_gallery = []
        K = self.images_per_id

        feature_extractor = DeepFeatures(
            model, device=device, num_workers=num_workers, batch_size=batch_size
        )
        test_features = np.array(feature_extractor(self.default_query).features)

        # df = self.default_query.metadata.copy()
        df = self.test_split.copy()
        # filter using self.identity_train_labeled
        df = df[df["identity"].isin(self.identity_test_closed)]

        for identity, group in df.groupby("identity"):
            # Select the corresponding images and features for each identity
            indices = group.index.tolist()

            identity_features = [test_features[idx] for idx in indices]
            centroid = np.mean(identity_features, axis=0)

            # sort identity samples by cosine distance to centroid
            if self.metric == "precomputed":
                distances = distance.cdist(
                    identity_features, [centroid], metric="jaccard"
                ).flatten()
            else:
                distances = distance.cdist(
                    identity_features, [centroid], metric=self.metric
                ).flatten()
            sorted_indices = np.argsort(distances)

            if len(sorted_indices) <= K:
                representative_gallery.append(group.iloc[sorted_indices])
                to_remove = group.index[sorted_indices]
                self.test_split = self.test_split.drop(to_remove)
                self.test_split_closed = self.test_split_closed.drop(to_remove)
                continue
            # central points are the closest 50% points from the centroid of the cluster
            central_points = sorted_indices[: int(np.ceil(len(sorted_indices) * 0.5))]

            if len(central_points) <= K:
                representative_gallery.append(group.iloc[sorted_indices[:K]])
                to_remove = group.index[sorted_indices[:K]]
                self.test_split = self.test_split.drop(to_remove)
                self.test_split_closed = self.test_split_closed.drop(to_remove)
                continue

            # sample K points from central_points
            sampled_indices = np.random.choice(
                sorted_indices[: len(central_points)], K, replace=False
            )
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

    def get_train_loader(self):
        database = WildlifeDataset(
            metadata=self.train_split,
            root=self.image_root,
            transform=self.train_transform,
        )

        # splitter = splits.ClosedSetSplit(0.8)
        # idx_train, idx_val = splitter.split(database.metadata)[0]
        #
        # train_database = WildlifeDataset(
        #     metadata=database.metadata.loc[idx_train],
        #     root=self.image_root,
        #     transform=self.train_transform
        # )
        # val_database = WildlifeDataset(
        #     metadata=database.metadata.loc[idx_val],
        #     root=self.image_root,
        #     transform=self.test_transform
        # )

        return database

    def get_test_loader(self):
        database = WildlifeDataset(
            metadata=self.train_split,
            root=self.image_root,
            transform=self.test_transform,
        )
        # splitter = splits.ClosedSetSplit(0.8)
        # idx_train, idx_val = splitter.split(database.metadata)[0]
        #
        # train_database = WildlifeDataset(
        #     metadata=database.metadata.loc[idx_train],
        #     root=self.image_root,
        #     transform=self.train_transform
        # )
        # val_database = WildlifeDataset(
        #     metadata=database.metadata.loc[idx_val],
        #     root=self.image_root,
        #     transform=self.test_transform
        # )

        return database
