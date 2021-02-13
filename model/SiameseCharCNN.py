from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn 
from torch.nn import functional as F 
import numpy as np 
from collections import OrderedDict
from script.MyLayers import EuclideanLayer
torch.manual_seed(2021)

class SiameseCharCNN(nn.Module):
    def __init__(self, args):
        super(SiameseCharCNN, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(self.args.vocab_size, self.args.embedding_size)
        self.cnn_list = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv1d', nn.Conv1d(self.args.embedding_size, self.args.filters, kernel_size)),
                ('relu', nn.ReLU(inplace=True)),
                ('max_pool', nn.MaxPool1d(self.args.max_len - kernel_size + 1))
            ]))
            for kernel_size in self.args.kernel_size_list   
        ])
        self.gap = nn.AdaptiveAvgPool1d(1) 
        self.cos_sim_layer = nn.CosineSimilarity(dim=-1)
        self.euclidean_layer = EuclideanLayer(dim=-1)

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear((self.args.filters*2+2)*len(self.args.kernel_size_list), 32)),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(32, self.args.num_class)),
            ('softmax', nn.Softmax(dim=1))
        ]))
    
    def forward(self, x1, x2):
        x1_embedd = self.embedding(x1)
        x2_embedd = self.embedding(x2)
        absSub_list = []
        mul_list = []
        cossim_list = []
        euclidean_list = []
        
        for cnn in self.cnn_list:
            x1_cnn_feature = cnn(x1_embedd.permute(0, 2, 1)) # [batch, filters, 1]
            x2_cnn_feature = cnn(x2_embedd.permute(0, 2, 1)) # [batch, filters, 1]

            sub = torch.abs(x1_cnn_feature - x2_cnn_feature).squeeze() # [batch, filters]
            absSub_list.append(sub)

            mul = torch.multiply(x1_cnn_feature, x2_cnn_feature).squeeze() # [batch, filters]
            mul_list.append(sub)

            cos_sim = self.cos_sim_layer(x1_cnn_feature.permute(0, 2, 1), x2_cnn_feature.permute(0, 2, 1)) # [batch, 1]
            cossim_list.append(cos_sim)

            euclidean = self.euclidean_layer(x1_cnn_feature.permute(0, 2, 1), x2_cnn_feature.permute(0, 2, 1)) # [batch, 1]
            euclidean_list.append(euclidean)
        
        absSub_list = torch.cat(absSub_list, axis=-1)
        mul_list = torch.cat(mul_list, axis=-1)
        cossim_list = torch.cat(cossim_list, axis=-1)
        euclidean_list = torch.cat(euclidean_list, axis=-1)

        out = torch.cat([absSub_list, mul_list, cossim_list, euclidean_list], axis=-1)
        out = self.fc(out)
        return out
