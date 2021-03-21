#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn 
from torch.nn import functional as F 
import numpy as np 
from collections import OrderedDict

'''pair sentence
'''
class SiameseCNN(nn.Module):
    def __init__(self, args, is_pretrain=False, embeddings=None):
        super(SiameseCNN, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(self.args.vocab_size, self.args.embedding_size) # [batch, seq_len, embedding_size]
        if is_pretrain:
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=True)

        self.spatial_dropout = SpatialDropout(0.5)

        self.cnn_list = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv1d', nn.Conv1d(self.args.embedding_size, self.args.filters, kernel_size)),
                ('relu', nn.ReLU(inplace=True)),
                ('max_pool', nn.MaxPool1d(self.args.max_len - kernel_size + 1)),
                ('SpatialDropout', SpatialDropout(0.5))
            ]))
            for kernel_size in self.args.kernel_size_list   
        ])
        
        self.cos_sim_layer = CosineLayer(dim=-1)
        self.euclidean_layer = EuclideanLayer(dim=-1)

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear((self.args.filters*2+2)*len(self.args.kernel_size_list), 64)),
            ('relu1', nn.ReLU(inplace=True))
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(64, 32)),
            ('relu2', nn.ReLU(inplace=True))
            ('dropout2', nn.Dropout(0.5)),
            ('fc3', nn.Linear(32, self.args.num_class))
        ]))
    
    def forward(self, x1, x2):
        x1_embedd = self.embedding(x1)
        x1_embedd = self.spatial_dropout(x1_embedd)
        x2_embedd = self.embedding(x2)
        x2_embedd = self.spatial_dropout(x2_embedd)

        absSub_list = []
        mul_list = []
        cossim_list = []
        euclidean_list = []
        
        for cnn in self.cnn_list:
            x1_cnn_feature = cnn(x1_embedd.permute(0, 2, 1)) # [batch, filters, 1]
            x2_cnn_feature = cnn(x2_embedd.permute(0, 2, 1)) # [batch, filters, 1]

            sub = torch.abs(x1_cnn_feature - x2_cnn_feature).squeeze(-1) # [batch, filters]
            absSub_list.append(sub)

            mul = torch.multiply(x1_cnn_feature, x2_cnn_feature).squeeze(-1) # [batch, filters]
            mul_list.append(sub)

            cos_sim = self.cos_sim_layer(x1_cnn_feature.permute(0, 2, 1), x2_cnn_feature.permute(0, 2, 1)) # [batch, 1]
            cossim_list.append(cos_sim)

            euclidean = self.euclidean_layer(x1_cnn_feature.permute(0, 2, 1), x2_cnn_feature.permute(0, 2, 1)) # [batch, 1]
            euclidean_list.append(euclidean)
        
        absSub_list = torch.cat(absSub_list, dim=-1)
        mul_list = torch.cat(mul_list, dim=-1)
        cossim_list = torch.cat(cossim_list, dim=-1)
        euclidean_list = torch.cat(euclidean_list, dim=-1)

        out = torch.cat([absSub_list, mul_list, cossim_list, euclidean_list], dim=-1)

        logit = self.fc(out)
        prob = F.softmax(logit, dim=1)
        return logit, prob

'''single sentence
'''
class TextCNN(nn.Module):
    def __init__(self, args, is_pretrain=False, embeddings=None):
        super(TextCNN, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(self.args.vocab_size, self.args.embedding_size)
        if is_pretrain:
            self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=True)

        self.spatial_dropout = SpatialDropout(0.5)

        self.cnn_list = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv1d', nn.Conv1d(self.args.embedding_size, self.args.filters, kernel_size)),
                ('relu', nn.ReLU(inplace=True)),
                ('max_pool', nn.MaxPool1d(self.args.max_len - kernel_size + 1)),
                ('SpatialDropout', SpatialDropout(0.5))
            ]))
            for kernel_size in self.args.kernel_size_list   
        ])

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.args.filters, 64)),
            ('relu1', nn.ReLU(inplace=True))
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(64, 32)),
            ('dropout2', nn.Dropout(0.5)),
            ('relu2', nn.ReLU(inplace=True))
            ('fc3', nn.Linear(32, self.args.num_class))
        ]))
    
    def forward(self, x):
        x_embedd = self.embedding(x)
        x_embedd = self.spatial_dropout(x_embedd)

        pool_list = []
        for cnn in self.cnn_list:
            x_cnn_feature = cnn(x_embedd.permute(0, 2, 1))
            pool_list.append(x_cnn_feature)
        
        out = torch.cat(pool_list, dim=-1) # [batch, filters, max_pool_len*n]
        out = self.gap(out).squeeze(-1) # [batch, filters]
        out = self.dropout(out)
        logit = self.fc(out)
        prob = F.softmax(logit, dim=1)
        return logit, prob
