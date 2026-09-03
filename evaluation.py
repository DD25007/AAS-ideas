import pandas as pd
import random
import numpy as np


def generate_report(metadata, gallery, query, split_type):
    # Total number of samples in the dataset.
    total_samples = len(metadata)

    # Ensure that any row not labeled 'gal' or 'que' is set to 'unassigned'
    metadata.loc[~metadata["label"].isin(["test_gallery", "test_query"]), "label"] = (
        "unassigned"
    )

    # Count samples for each split.
    gallery_samples = metadata[metadata["label"] == "test_gallery"].shape[0]
    query_samples = metadata[metadata["label"] == "test_query"].shape[0]
    unassigned_samples = metadata[metadata["label"] == "unassigned"].shape[0]
    total_samples = metadata.shape[0]

    # Count the number of unique classes in each split.
    gallery_classes = metadata[metadata["label"] == "test_gallery"][
        "identity"
    ].nunique()
    query_classes = metadata[metadata["label"] == "test_query"]["identity"].nunique()
    unassigned_classes = metadata[metadata["label"] == "unassigned"][
        "identity"
    ].nunique()
    total_classes = metadata["identity"].nunique()

    # Determine classes that appear only in gallery, only in query, or in both (joint).
    gallery_ids = set(metadata[metadata["label"] == "test_gallery"]["identity"])
    query_ids = set(metadata[metadata["label"] == "test_query"]["identity"])
    joint_ids = gallery_ids.intersection(query_ids)
    gallery_only_ids = gallery_ids - query_ids
    query_only_ids = query_ids - gallery_ids

    # Count samples for gallery-only and query-only (if any exist).
    gallery_only_samples = metadata[metadata["identity"].isin(gallery_only_ids)].shape[
        0
    ]
    query_only_samples = metadata[metadata["identity"].isin(query_only_ids)].shape[0]

    # Build a formatted report string.
    report = (
        f"Split: {split_type}\n"
        f"Samples: gallery/query/unassigned/total = {gallery_samples}/{query_samples}/{unassigned_samples}/{total_samples}\n"
        f"Classes: gallery/query/unassigned/total = {gallery_classes}/{query_classes}/{unassigned_classes}/{total_classes}\n"
        f"Samples: gallery only/query only        = {gallery_only_samples}/{query_only_samples}\n"
        f"Classes: gallery only/query only/joint  = {len(gallery_only_ids)}/{len(query_only_ids)}/{len(joint_ids)}\n\n"
    )
    print(report)


# def circular_offset(indices, num_splits):
#     """
#     Circularly offsets the indices by a specified number of splits.
#     This function is used to ensure that the indices are evenly distributed across the splits.
#     """
#     # Create an empty list to hold the new indices
#     new_indices_list = []
#     new_indices = []

#     # Loop through each index and apply the circular offset
#     for s in range(num_splits):
#         for i in range(len(indices)):
#             new_index = (i + s) % len(indices)
#             new_indices.append(indices[new_index])
#         new_indices_list.append(new_indices)
#         new_indices = []
#     return new_indices_list


def get_shuffled_indices(indices, num_splits, seed):
    seeds = [i for i in range(num_splits)]
    indices_list = []
    for s in seeds:
        np.random.seed(s)
        np.random.shuffle(indices)
        indices_list.append(indices.copy())
    np.random.seed(seed)

    # print(indices_list)
    return indices_list


def split_protocol(metadata, protocol_type=1, image_per_id=1, num_splits=5, seed=0):
    if protocol_type == 1:
        # Assign 'gal' to the first sample of each Entity group and 'que' to the others.
        # metadata = pd.read_csv('../../../cnnarch/WII_dataset_densenet121_usam/data/annotations/test.csv')
        gallery_indices = []
        query_indices = []
        queries = []
        galleries = []
        split_indices_dict = {}

        # Loop through each identity group in the DataFrame
        for identity, group in metadata.groupby("identity"):
            indices = group.index.tolist()
            # np.random.shuffle(indices)
            # split_indices_list = circular_offset(indices, num_splits)
            split_indices_list = get_shuffled_indices(indices, num_splits, seed)
            split_indices_dict[identity] = split_indices_list

        for i in range(num_splits):
            # Select the first image for gallery and the rest for query.
            # If not enough images for a proper split (gallery + query), add all images to gallery.

            for identity, group in metadata.groupby("identity"):
                if len(group) < image_per_id + 1:
                    # gallery_indices.extend(group.index.tolist())
                    continue
                else:
                    # Otherwise, select the first images_per_id for gallery and the rest for query.
                    gallery_indices.extend(
                        split_indices_dict[identity][i][:image_per_id]
                    )
                    query_indices.extend(split_indices_dict[identity][i][image_per_id:])

            metadata["label"] = "label"
            metadata.loc[query_indices, "label"] = "test_query"
            metadata.loc[gallery_indices, "label"] = "test_gallery"

            # Return the gallery and query DataFrames.
            gallery = metadata.loc[gallery_indices]
            query = metadata.loc[query_indices]
            galleries.append(gallery)
            queries.append(query)

            generate_report(metadata, gallery, query, "Closed Set")

            query_indices = []
            gallery_indices = []

        return galleries, queries

    elif protocol_type == 2:
        metadata = pd.read_csv(
            "../../../cnnarch/WII_dataset_densenet121_usam/data/annotations/test.csv"
        )
        extra_percent = int(
            input("Specify extra percentage of ids w.r.t OpenSetSplit (e.g - 20) : ")
        )
        # Get all identity labels
        all_ids = list(metadata["Entity"].unique())
        random.shuffle(all_ids)

        # Decide closed-set (gallery) identities.
        # We want extra open-set identities to be ~ extra_percent of the gallery identities.
        # Let x be the number of closed-set identities; then extra target = round(x * (extra_percent/100)).
        # One simple solution is to set:
        if extra_percent > 0:
            closed_set_count = int(len(all_ids) / (1 + extra_percent / 100))
        else:
            closed_set_count = len(all_ids)
        closed_set_ids = all_ids[:closed_set_count]

        # The remaining identities are candidates for extra open-set
        remaining_ids = all_ids[closed_set_count:]
        open_set_count_target = round(len(closed_set_ids) * (extra_percent / 100))
        if open_set_count_target > len(remaining_ids):
            open_set_ids = remaining_ids
        else:
            open_set_ids = random.sample(tuple(remaining_ids), open_set_count_target)

        # Initialize lists of indices for gallery and query.
        gallery_indices = []
        query_indices = []

        # Process each closed-set identity.
        for entity in closed_set_ids:
            group = metadata[metadata["Entity"] == entity]
            # If there are not enough images, assign all to gallery.
            if len(group) < image_per_id + 1:
                # gallery_indices.extend(group.index.tolist())
                continue
            else:
                gallery_indices.extend(group.index.tolist()[:image_per_id])
                # The rest become query images.
                query_indices.extend(group.index.tolist()[image_per_id:])

        # Process each extra open-set identity: assign all images to query.
        for entity in open_set_ids:
            group = metadata[metadata["Entity"] == entity]
            query_indices.extend(group.index.tolist())

        # Mark rows: default is "unassigned", then update gallery and query.
        metadata["label"] = "unassigned"
        metadata.loc[gallery_indices, "label"] = "gal"
        metadata.loc[query_indices, "label"] = "que"

        # Return the gallery and query DataFrames.
        gallery = metadata[metadata["label"] == "gal"]
        query = metadata[metadata["label"] == "que"]
        generate_report(metadata, gallery, query, "Open Set")
        return gallery, query


# borrowed from fastreid: https://github.com/JDAI-CV/fast-reid
def eval_wildlife(distmat, q_pids, g_pids, max_rank):
    """Evaluation for wildlife dataset."""
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    indices = np.argsort(distmat, axis=1)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.0  # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]

        # compute cmc curve
        matches = (g_pids[order] == q_pid).astype(np.int32)
        raw_cmc = (
            matches.copy()
        )  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        pos_idx = np.where(raw_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q

    return all_cmc, all_AP, all_INP


def main():
    try:
        # Prompt user for input
        user_input = (
            int(
                input(
                    "Choose the protocol type : 1 - ClosedSetSplit || 2 - OpenSetSplit : "
                )
            )
            or None
        )
        if user_input is not None:
            gallery, query = split_protocol(user_input)

    except ValueError:
        print("Invalid input. Please enter integers separated by spaces.")


if __name__ == "__main__":
    main()
