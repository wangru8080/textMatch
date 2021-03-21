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
  
class SpatialDropout1D(nn.Module):
    def __init__(self, drop_prob=0.2):
        super(SpatialDropout1D, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        output = x.clone()
        if not self.training or self.drop_prob == 0:
            return x
        else:
            noise = self._make_noise(x)
            if self.drop_prob == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.drop_prob).div_(1 - self.drop_prob)
            noise = noise.expand_as(x)
            output.mul_(noise)
        return output

    def _make_noise(self, x):
        return x.new().resize_(x.size(0), *repeat(1, x.dim() - 2), x.size(2))

class SpatialDropout(nn.Module):
    def __init__(self, drop_prob=0.2):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.dropout2d(x, self.drop_prob, training=self.training)
        x = x.permute(0, 2, 1)
        return x
