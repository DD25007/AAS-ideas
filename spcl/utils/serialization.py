from __future__ import print_function, absolute_import
import json
import os
import os.path as osp
import shutil

import torch
from torch.nn import Parameter







def save_checkpoint(state, is_best, fpath="checkpoint.pth.tar"):
    os.makedirs(osp.dirname(fpath), exist_ok=True)
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), "model_best.pth.tar"))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        # checkpoint = torch.load(fpath)
        checkpoint = torch.load(
            fpath, map_location=torch.device("cpu"), weights_only=False
        )
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


