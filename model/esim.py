#!/usr/bin/env python
# -*- encoding: utf-8 -*-

rom __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn 
from torch.nn import functional as F 
import numpy as np 
from collections import OrderedDict
torch.manual_seed(2021)

class ESIM(nn.Module):
    def __init__(self, args):
        super(ESIM, self).__init__()
        self.args = args

        self.embedding = nn.Embedding(self.args.vocab_size, self.args.embedding_size) # [batch, seq_len, embedding_size]
        self.bi_lstm1 = nn.LSTM(self.args.embedding_size, 128, batch_first=True, bidirectional=True) # [batch, seq_len, 128]
        self.bi_lstm2 = nn.LSTM(128*8, 128, batch_first=True, bidirectional=True)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(128*8, 32)),
            ('dropout1', nn.Dropout(0.5)),
            ('fc2', nn.Linear(32, self.args.num_class)),
            ('softmax', nn.Softmax(dim=1))
        ]))

    def soft_align_attention(self, x1, x2):
        attention = torch.matmul(x1, x2.permute(0, 2, 1))
        x1_att = F.softmax(attention, dim=-1)
        x2_att = F.softmax(attention.permute(0, 2, 1), dim=-1)
        x1_align = torch.matmul(x1_att, x2)
        x2_align = torch.matmul(x2_att, x1)
        return x1_align, x2_align
    
    def submul(self, x1, x2):
        sub = x1 - x2
        mul = x1 * x2
        return torch.cat([sub, mul], dim=-1) 
    
    def apply_multiple(self, x):
        rep1 = self.gap(x.permute(0, 2, 1)).squeeze(-1)
        rep2 = self.gmp(x.permute(0, 2, 1)).squeeze(-1)
        return torch.cat([rep1, rep2], dim=-1)
    
    def forward(self, x1, x2):
        x1_embedd = self.embedding(x1)
        x2_embedd = self.embedding(x2)

        # Encoder
        x1_encoder, _ = self.bi_lstm1(x1_embedd)
        x2_encoder, _ = self.bi_lstm1(x2_embedd)

        # Attention
        x1_align, x2_align = self.soft_align_attention(x1_encoder, x2_encoder)

        # Compose
        x1_submul = self.submul(x1_encoder, x1_align)
        x2_submul = self.submul(x2_encoder, x2_align)

        x1_combined = torch.cat([x1_encoder, x1_align, x1_submul], dim=-1)
        x2_combined = torch.cat([x2_encoder, x2_align, x2_submul], dim=-1)

        x1_compose, _ = self.bi_lstm2(x1_combined)
        x2_compose, _ = self.bi_lstm2(x2_combined)

        # Aggregate
        x1_rep = self.apply_multiple(x1_compose)
        x2_rep = self.apply_multiple(x2_compose)

        out = torch.cat([x1_rep, x2_rep], dim=-1)
        out = self.fc(out)
        return out
