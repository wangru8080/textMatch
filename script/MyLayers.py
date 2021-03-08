from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn 
from torch.nn import functional as F 
import numpy as np 
from collections import OrderedDict

class EuclideanLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-5, keepdim=False):
        super(EuclideanLayer, self).__init__()

        self.dim = dim
        self.eps = eps 
        self.keepdim = keepdim

    def forward(self, x1, x2):
        return torch.sqrt(torch.sum(torch.square(x1 - x2), axis=self.dim, keepdim=self.keepdim) + self.eps)

class CosineLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-5, keepdim=False):
        super(CosineLayer, self).__init__()

        self.dim = dim
        self.eps = eps 
        self.keepdim = keepdim

    def forward(self, x1, x2):
        norm1 = torch.sqrt(torch.sum(torch.square(x1), dim=self.dim, keepdim=self.keepdim) + self.eps)
        norm2 = torch.sqrt(torch.sum(torch.square(x2), dim=self.dim, keepdim=self.keepdim) + self.eps)
        dot_products = torch.sum(x1 * x2, dim=self.dim, keepdim=self.keepdim)
        return dot_products / (norm1 * norm2)
