import math
import numpy as np
import torch
import torch.nn as nn
from .utils import *
from torch.distributions import Normal, Independent
from torch import distributed as dist

# function credit to https://github.com/facebookresearch/barlowtwins/blob/main/main.py
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class DecorrelateLossClass(nn.Module):

    def __init__(self, reject_threshold=1, ddp=False):
        super(DecorrelateLossClass, self).__init__()
        self.eps = 1e-8
        self.reject_threshold = reject_threshold
        self.ddp = ddp

    def forward(self, x, y):
        _, C = x.shape
        if self.ddp:
            # if DDP
            # first gather all x and labels from the world
            x = torch.cat(GatherLayer.apply(x), dim=0)
            y = global_gather(y)

        loss = 0.0
        uniq_l, uniq_c = y.unique(return_counts=True)
        n_count = 0
        for i, label in enumerate(uniq_l):
            if uniq_c[i] <= self.reject_threshold:
                continue
            x_label = x[y==label, :]
            x_label = x_label - x_label.mean(dim=0, keepdim=True)
            x_label = x_label / torch.sqrt(self.eps + x_label.var(dim=0, keepdim=True))

            N = x_label.shape[0]
            corr_mat = torch.matmul(x_label.t(), x_label)

            # Notice that here the implementation is a little bit different
            # from the paper as we extract only the off-diagonal terms for regularization.
            # Mathematically, these two are the same thing since diagonal terms are all constant 1.
            # However, we find that this implementation is more numerically stable.
            loss += (off_diagonal(corr_mat).pow(2)).mean()

            n_count += N

        if n_count == 0:
            # there is no effective class to compute correlation matrix
            return 0
        else:
            loss = loss / n_count
            return loss
