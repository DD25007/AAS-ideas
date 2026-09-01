from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os
import random
import numpy as np
import sys
import collections
import copy
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from spcl import datasets
from spcl import models
from spcl.models.hm import HybridMemory
from spcl.trainers import SpCLTrainer_USL
from spcl.evaluators import Evaluator, extract_features
from spcl.utils.constrained_clustering import constrained_clustering
from spcl.utils.data import IterLoader
from spcl.utils.data import transforms as T
from spcl.utils.data.sampler import RandomMultipleGallerySampler
from spcl.utils.data.preprocessor import Preprocessor
from spcl.utils.logging import Logger
from spcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from spcl.utils.faiss_rerank import compute_jaccard_distance

from sampling.random import RandomSampling, RobustRandomSampling
# FIXME(cleanup): re-commented. `sampling` here resolves to SpCL/sampling/, which has
# no assc.py or fps.py -- those live only in the repo-root sampling/ package. Leaving
# this import active raises ModuleNotFoundError: No module named 'sampling.assc'.
# The MinMax/EC samplers dispatched at ~line 150 are unreachable as a result; they are
# not paper baselines. Proper fix is to merge the two `sampling` packages, or import
# these explicitly from the repo-root one. See Todo.md item C4.
# from sampling.assc import MinMax, ExploreConsolidate
# from sampling.fps import FPS
from sampling.balanced import LBAS, GBAS
from sampling.consistency import ConsistencySampling
from sampling.counting_clusters import CountingClusters

from spcl.datasets.wildlife import WildlifeDatasets
from spcl.utils.wildlife_utils import get_evaluation_results

from wildlife_tools.features import DeepFeatures, ClipFeatures




def get_data(name, data_dir, subset='combined'):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root, subset=subset)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir, args.subset)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    print(dataset.train[:20])

    # Active Learning
    if args.sampler == "random":
        sampler = RandomSampling(args=args)
    elif args.sampler == "GBAS":
        sampler = GBAS(thresh=args.eps, min_samples=4,
                       metric="precomputed", sample_outliers=False, args=args)
    elif args.sampler == "LBAS":
        sampler = LBAS(thresh=args.eps, min_samples=4,
                       metric="precomputed", sample_outliers=False, args=args)
    elif args.sampler == "MinMax":
        sampler = MinMax(args=args)
    elif args.sampler == "EC":
        sampler = ExploreConsolidate(args=args)
    elif args.sampler == "CC":
        sampler = CountingClusters(ratio=1.0, args=args)
    elif args.sampler == "ConSam":
        sampler = ConsistencySampling(thresh=args.eps, min_samples=4, metric="precomputed",
                                      num_gt_ids_gallery=np.unique(np.array(dataset.gallery)[:, 1]).shape[0], args=args)
    else:
        sampler = None

    gt_labels = None

    for cycle in range(1):
        start_epoch = best_mAP = 0

        # Create model
        model = create_model(args)

        # Create hybrid memory
        memory = HybridMemory(model.module.num_features, len(dataset.train),
                                temp=args.temp, momentum=args.momentum).cuda()

        # Initialize target-domain instance features
        print("==> Initialize instance features in the hybrid memory")
        cluster_loader = get_test_loader(dataset, args.height, args.width,
                                         args.batch_size, args.workers, testset=sorted(dataset.train))
        features, _ = extract_features(model, cluster_loader, print_freq=50)
        features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        memory.features = F.normalize(features, dim=1).cuda()
        if gt_labels is None and sampler is not None:
            gt_labels = [pid for _, pid, _ in sorted(dataset.train)]
            sampler.set_gt_labels(gt_labels)
        del cluster_loader, features

        # Evaluator
        evaluator = Evaluator(model)

        # Optimizer
        params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

        # Trainer
        trainer = SpCLTrainer_USL(model, memory)

        for epoch in range(args.epochs):
            if args.dataset == 'wildlife':
                model.train()
            
            if not args.full_supervision:
                # Calculate distance
                print('==> Create pseudo labels for unlabeled data with self-paced policy')
                features = memory.features.clone()
                rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
                
                if epoch % (args.eval_step) == 0 and sampler is not None:
                    if args.sampler == "None":
                        ml_pairs, cl_pairs = [], []
                    else:
                        ml_pairs, cl_pairs = sampler.select(features=features, rerank_dist=rerank_dist, gt_labels=gt_labels)
                    print("#Constraints:", len(ml_pairs) + len(cl_pairs))
                    print("#ML Constraints:", len(ml_pairs))
                    print("#CL Constraints:", len(cl_pairs))

                if epoch == 0:
                    # DBSCAN cluster
                    eps = args.eps
                    eps_tight = eps-args.eps_gap
                    eps_loose = eps+args.eps_gap
                    print('Clustering criterion: eps: {:.3f}, eps_tight: {:.3f}, eps_loose: {:.3f}'.format(eps, eps_tight, eps_loose))
                    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed')
                    cluster_tight = DBSCAN(eps=eps_tight, min_samples=4, metric='precomputed')
                    cluster_loose = DBSCAN(eps=eps_loose, min_samples=4, metric='precomputed')

                del features

                # select & cluster images as training set of this epochs
                pseudo_labels = cluster.fit_predict(rerank_dist)
                pseudo_labels_tight = cluster_tight.fit_predict(rerank_dist)
                pseudo_labels_loose = cluster_loose.fit_predict(rerank_dist)
                num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
                num_ids_tight = len(set(pseudo_labels_tight)) - (1 if -1 in pseudo_labels_tight else 0)
                num_ids_loose = len(set(pseudo_labels_loose)) - (1 if -1 in pseudo_labels_loose else 0)

                if sampler is not None:
                    purity_labels = pseudo_labels.copy()
                    purity_labels[purity_labels == -1] = np.arange(purity_labels.max() + 1, purity_labels.max() + 1 + sum(purity_labels == -1))
                    # compute purity:
                    cluster_dict = {}
                    impurity_ratio = []
                    for i, label in enumerate(purity_labels):
                        if label.item() not in cluster_dict:
                            cluster_dict[label.item()] = []
                        cluster_dict[label.item()].append(gt_labels[i])
                    cluster_purity = 0
                    pure_clusters = 0
                    for label, ids in cluster_dict.items():
                        if len(ids) > 0:
                            most_common = collections.Counter(ids).most_common(1)[0][1]
                            cluster_purity += most_common / len(ids)
                            if most_common == len(ids):
                                pure_clusters += 1
                            else:
                                impurity_ratio.append((len(ids) - most_common) / len(ids))
                    impurity_ratio = np.mean(impurity_ratio) if len(impurity_ratio) > 0 else 0
                    cluster_purity /= len(cluster_dict)
                    print(purity_labels[:20])
                    print(f'Cluster Purity: {cluster_purity:.4f} | \tPure Clusters: {pure_clusters}/{len(cluster_dict)} = {pure_clusters/len(cluster_dict):.4f} | \tImpurity Ratio: {impurity_ratio:.4f}')

                    purity_labels = pseudo_labels_tight.copy()
                    purity_labels[purity_labels == -1] = np.arange(purity_labels.max() + 1, purity_labels.max() + 1 + sum(purity_labels == -1))
                    # compute purity:
                    cluster_dict = {}
                    impurity_ratio = []
                    for i, label in enumerate(purity_labels):
                        if label.item() not in cluster_dict:
                            cluster_dict[label.item()] = []
                        cluster_dict[label.item()].append(gt_labels[i])
                    cluster_purity = 0
                    pure_clusters = 0
                    for label, ids in cluster_dict.items():
                        if len(ids) > 0:
                            most_common = collections.Counter(ids).most_common(1)[0][1]
                            cluster_purity += most_common / len(ids)
                            if most_common == len(ids):
                                pure_clusters += 1
                            else:
                                impurity_ratio.append((len(ids) - most_common) / len(ids))
                    impurity_ratio = np.mean(impurity_ratio) if len(impurity_ratio) > 0 else 0
                    cluster_purity /= len(cluster_dict)
                    print(purity_labels[:20])
                    print(f'Cluster Purity: {cluster_purity:.4f} | \tPure Clusters: {pure_clusters}/{len(cluster_dict)} = {pure_clusters/len(cluster_dict):.4f} | \tImpurity Ratio: {impurity_ratio:.4f}')

                    purity_labels = pseudo_labels_loose.copy()
                    purity_labels[purity_labels == -1] = np.arange(purity_labels.max() + 1, purity_labels.max() + 1 + sum(purity_labels == -1))
                    # compute purity:
                    cluster_dict = {}
                    impurity_ratio = []
                    for i, label in enumerate(purity_labels):
                        if label.item() not in cluster_dict:
                            cluster_dict[label.item()] = []
                        cluster_dict[label.item()].append(gt_labels[i])
                    cluster_purity = 0
                    pure_clusters = 0
                    for label, ids in cluster_dict.items():
                        if len(ids) > 0:
                            most_common = collections.Counter(ids).most_common(1)[0][1]
                            cluster_purity += most_common / len(ids)
                            if most_common == len(ids):
                                pure_clusters += 1
                            else:
                                impurity_ratio.append((len(ids) - most_common) / len(ids))
                    impurity_ratio = np.mean(impurity_ratio) if len(impurity_ratio) > 0 else 0
                    cluster_purity /= len(cluster_dict)
                    print(purity_labels[:20])
                    print(f'Cluster Purity: {cluster_purity:.4f} | \tPure Clusters: {pure_clusters}/{len(cluster_dict)} = {pure_clusters/len(cluster_dict):.4f} | \tImpurity Ratio: {impurity_ratio:.4f}')

                # generate new dataset and calculate cluster centers
                def generate_pseudo_labels(cluster_id, num):
                    labels = []
                    outliers = 0
                    for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset.train), cluster_id)):
                        if id!=-1:
                            labels.append(id)
                        else:
                            labels.append(num+outliers)
                            outliers += 1
                    return torch.Tensor(labels).long()

                pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)
                pseudo_labels_tight = generate_pseudo_labels(pseudo_labels_tight, num_ids_tight)
                pseudo_labels_loose = generate_pseudo_labels(pseudo_labels_loose, num_ids_loose)

                # compute R_indep and R_comp
                N = pseudo_labels.size(0)
                label_sim = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
                label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
                label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()

                R_comp = 1-torch.min(label_sim, label_sim_tight).sum(-1)/torch.max(label_sim, label_sim_tight).sum(-1)
                R_indep = 1-torch.min(label_sim, label_sim_loose).sum(-1)/torch.max(label_sim, label_sim_loose).sum(-1)
                assert((R_comp.min()>=0) and (R_comp.max()<=1))
                assert((R_indep.min()>=0) and (R_indep.max()<=1))

                cluster_R_comp, cluster_R_indep = collections.defaultdict(list), collections.defaultdict(list)
                cluster_img_num = collections.defaultdict(int)
                for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
                    cluster_R_comp[label.item()].append(comp.item())
                    cluster_R_indep[label.item()].append(indep.item())
                    cluster_img_num[label.item()]+=1

                cluster_R_comp = [min(cluster_R_comp[i]) for i in sorted(cluster_R_comp.keys())]
                cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
                cluster_R_indep_noins = [iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys())) if cluster_img_num[num]>1]
                if (epoch==0):
                    indep_thres = np.sort(cluster_R_indep_noins)[min(len(cluster_R_indep_noins)-1,np.round(len(cluster_R_indep_noins)*0.9).astype('int'))]

                pseudo_labeled_dataset = []
                outliers = 0
                constrained_pseudo_labels = []
                for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
                    indep_score = cluster_R_indep[label.item()]
                    comp_score = R_comp[i]
                    if ((indep_score<=indep_thres) and (comp_score.item()<=cluster_R_comp[label.item()])):
                        pseudo_labeled_dataset.append((fname,label.item(),cid))
                        constrained_pseudo_labels.append(label.item())
                    else:
                        pseudo_labeled_dataset.append((fname,len(cluster_R_indep)+outliers,cid))
                        pseudo_labels[i] = len(cluster_R_indep)+outliers
                        constrained_pseudo_labels.append(pseudo_labels[i])
                        outliers+=1

                # Impose ML and CL constraints on the final pseudo labels
                if sampler is not None:
                    constrained_pseudo_labels = constrained_clustering(np.array(constrained_pseudo_labels), rerank_dist, "~DBSCAN",
                                                                    min_samples=4, ml_pairs=ml_pairs, cl_pairs=cl_pairs)
                    for i, label in enumerate(constrained_pseudo_labels):
                        fname, _, cid = pseudo_labeled_dataset[i]
                        pseudo_labeled_dataset[i] = fname, label, cid
                        pseudo_labels[i] = label

                # statistics of clusters and un-clustered instances
                index2label = collections.defaultdict(int)
                for label in pseudo_labels:
                    index2label[label.item()]+=1
                index2label = np.fromiter(index2label.values(), dtype=float)
                print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances, R_indep threshold is {}'
                            .format(epoch, (index2label>1).sum(), (index2label==1).sum(), 1-indep_thres))

                memory.labels = pseudo_labels.cuda()

                if sampler is not None:
                    purity_labels = pseudo_labels.cpu().numpy().copy()
                    purity_labels[purity_labels == -1] = np.arange(purity_labels.max() + 1, purity_labels.max() + 1 + sum(purity_labels == -1))
                    # compute purity:
                    cluster_dict = {}
                    impurity_ratio = []
                    for i, label in enumerate(purity_labels):
                        if label.item() not in cluster_dict:
                            cluster_dict[label.item()] = []
                        cluster_dict[label.item()].append(gt_labels[i])
                    cluster_purity = 0
                    pure_clusters = 0
                    for label, ids in cluster_dict.items():
                        if len(ids) > 0:
                            most_common = collections.Counter(ids).most_common(1)[0][1]
                            cluster_purity += most_common / len(ids)
                            if most_common == len(ids):
                                pure_clusters += 1
                            else:
                                impurity_ratio.append((len(ids) - most_common) / len(ids))
                    impurity_ratio = np.mean(impurity_ratio) if len(impurity_ratio) > 0 else 0
                    cluster_purity /= len(cluster_dict)
                    print(purity_labels[:20])
                    print(f'Final Cluster Purity: {cluster_purity:.4f} | \tPure Clusters: {pure_clusters}/{len(cluster_dict)} = {pure_clusters/len(cluster_dict):.4f} | \tImpurity Ratio: {impurity_ratio:.4f}')
                
            else:
                unique_ids = list(np.unique(np.array(dataset.train)[:, 1]))
                memory.labels = torch.tensor([unique_ids.index(pid)+1 for _, pid, _ in sorted(dataset.train)]).cuda()
                pseudo_labels = memory.labels.clone()
                pseudo_labeled_dataset = []
                for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
                    pseudo_labeled_dataset.append((fname,label.item(),cid))

            train_loader = get_train_loader(args, dataset, args.height, args.width,
                                                args.batch_size, args.workers, args.num_instances, iters,
                                                trainset=pseudo_labeled_dataset)

            train_loader.new_epoch()

            trainer.train(epoch, train_loader, optimizer,
                        print_freq=args.print_freq, train_iters=len(train_loader))

            if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
                if args.dataset == 'wildlife':
                    model.eval()
                    megadescriptor_extractor = DeepFeatures(model, device='cuda:0', num_workers=0, batch_size=32)
                    model = model.to('cuda:0')
                    folder = osp.join(args.logs_dir, args.dataset, args.subset, 'evaluation')
                    os.makedirs(folder, exist_ok=True)

                    results = get_evaluation_results(megadescriptor_extractor, dataset, folder, epoch)
                    # FIXME(wrapper): wildlife_tools' DeepFeatures.__call__ ends with
                    # `self.model = self.model.to("cpu")` (wildlife_tools/data/cache.py:84),
                    # and self.model IS this same object -- so the model is left on the CPU
                    # after every eval and the next epoch dies in DataParallel with
                    #   "module must have its parameters and buffers on device cuda:0 ...
                    #    but found one of them on device: cpu".
                    # The `model.to('cuda:0')` above runs BEFORE the extractor, so it does
                    # not help. No wildlife_tools version avoids this (0.0.9 lacks the
                    # .to("cpu") but also lacks TopkClassifier, imported at wildlife.py:20).
                    # Proper fix: wrap DeepFeatures so it restores the model's device
                    # itself, instead of patching every call site. See Todo.md.
                    model = model.to('cuda:0')
                    mAP = results['mAP']
                else:
                    mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
                is_best = (mAP>best_mAP)
                best_mAP = max(mAP, best_mAP)
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': best_mAP,
                }, is_best, fpath=osp.join(args.logs_dir, f'checkpoint_{epoch+1}.pth.tar'))

                import csv
                # os.makedirs(f'/home/mehar21541/SpCL/logs_al/{args.subset}', exist_ok=True)
                # with open(f'/home/mehar21541/SpCL/logs_al/{args.subset}/results.csv', 'a', newline='') as file:
                results_dir = osp.join(args.logs_dir, 'logs_al', args.subset)
                os.makedirs(results_dir, exist_ok=True)
                with open(osp.join(results_dir, 'results.csv'), 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([args.sampler if sampler is not None else 'USL', epoch, mAP])

                print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                      format(epoch, mAP, best_mAP, ' *' if is_best else ''))

            lr_scheduler.step()

        print ('==> Test with the best model:')
        checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        if args.dataset == 'wildlife':
            model.eval()
            megadescriptor_extractor = DeepFeatures(model, device='cuda:0', num_workers=0, batch_size=32)
            model = model.to('cuda:0')
            folder = osp.join(args.logs_dir, args.dataset, args.subset, 'evaluation')
            os.makedirs(folder, exist_ok=True)

            results = get_evaluation_results(megadescriptor_extractor, dataset, folder, args.epochs+1)
            mAP = results['mAP']
        else:
            evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--subset', type=str, default='combined', help="wildlife dataset to use")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    # Active Learning
    parser.add_argument('-s', '--sampler', type=str, default=None)
    parser.add_argument('--full-supervision', action='store_true',
                        help="use full supervision for training")
    parser.add_argument('--consam_eps', type=float, default=0.6,
                        help="max neighbor distance for Consistency Sampling")
    parser.add_argument('--consam_sim_thresh', type=float, default=0.3,
                        help="similarity threshold for Consistency Sampling")
    parser.add_argument('--consam_k', type=int, default=5,
                        help="number of neighbor medoids for Consistency Sampling")
    main()
