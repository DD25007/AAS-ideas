import os
import copy
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
from wildlife_datasets.datasets import WildlifeReID10k
from wildlife_datasets.splits import analyze_split
from wildlife_datasets.metrics import BAKS, BAUS
# from wildlife_datasets.baseline.utils import compute_predictions, mean [REMOVED, since the utils.py has the compute_predictions, mean function]
from utils import *


# root_data = '/raid/home/anands/projects-mehar/wildlifereid-10k'
root_data = '/storage/dd25007/dataset/wildlifereid-10k/'
dataset = WildlifeReID10k(root_data, check_files=False)
metadata = dataset.metadata
metadata


idx_train = np.where(metadata['split'] == 'train')[0]
idx_test = np.where(metadata['split'] == 'test')[0]
analyze_split(metadata, idx_train, idx_test)

identity = metadata['identity'].to_numpy()
identity_train = identity[metadata['split'] == 'train']
identity_test = identity[metadata['split'] == 'test']
identity_test_only = list(set(identity_test) - set(identity_train))
len(identity_test_only)

step = 0.01
ts = [-np.inf] + list(np.round(np.arange(0, 1+step/10, step), 2)) + [np.inf]
new_individual = 'new_individual'


baks = {t: {} for t in ts}
baus = {t: {} for t in ts}
result = {}
for name, metadata_dataset in tqdm(dataset.metadata.groupby('dataset')):
    features = np.load(f'{root_features}/features_{name}.npy')
    
    idx_train = np.where(metadata_dataset['split'] == 'train')[0]
    idx_test = np.where(metadata_dataset['split'] == 'test')[0]

    idx_true, idx_pred, similarity = compute_predictions(features[idx_test], features[idx_train], return_score=True)
    idx_true = idx_test[idx_true]
    idx_pred = idx_train[idx_pred]
    idx_pred = idx_pred[:,0]
    similarity = similarity[:,0]

    y_true = metadata_dataset['identity'].iloc[idx_true]
    y_pred_closed = metadata_dataset['identity'].iloc[idx_pred]

    identity_test_only = list(set(metadata_dataset['identity'].iloc[idx_test]) - set(metadata_dataset['identity'].iloc[idx_train]))

    for t in ts:
        y_pred = copy.copy(y_pred_closed)
        with np.errstate(invalid='ignore'):
            y_pred[similarity < t] = new_individual
        baks[t][name] = BAKS(y_true, y_pred, identity_test_only)
        baus[t][name] = BAUS(y_true, y_pred, identity_test_only, new_individual)