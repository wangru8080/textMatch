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
from pytorch_pretrained_bert import BertModel
torch.manual_seed(2021)

class BertOrigin(nn.Module):
    def __init__(self, args):
        super(BertOrigin, self).__init__()
        self.args = args

        self.bert = BertModel.from_pretrained(args.bert_path) # bert-base-chinese
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(768, 2)),
            ('dropout1', nn.Dropout(0.5)),
            ('softmax', nn.Softmax(dim=1))
        ]))
    
    def forward(self, input_ids, segment_ids, input_mask):
        # all_encoder_layers 
        # output_all_encoded_layers=False shape:[batch_size, seq_len, bert_dim=768]
        # output_all_encoded_layers=True list_len=12, all_encoder_layers[0] shape:[batch_size, seq_len, bert_dim=768]
        # pooled_output shape [batch_size, bert_dim=768]
        all_encoder_layers, pooled_output = self.bert(input_ids, segment_ids, input_mask)
        print(pooled_output)

        out = self.fc(pooled_output)
        return out
