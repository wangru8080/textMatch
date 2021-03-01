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
from transformers import AlbertModel

class AlbertOrigin(nn.Module):
    def __init__(self, args, trained=True):
        super(AlbertOrigin, self).__init__()
        self.args = args

        self.albert = AlbertModel.from_pretrained(args.pretrained_model_path)
        for name, param in self.albert.named_parameters():
            param.requires_grad = trained
        
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(768, self.args.num_class))
        ]))
    
    def forward(self, input_ids, segment_ids, input_mask):
        albert_out = self.albert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        last_hidden_state = albert_out[0]
        pooled_output = albert_out[1]

        out = self.dropout(pooled_output)
        out = self.fc(out)

        return out

class AlbertOriginLarge(nn.Module):
    def __init__(self, args, trained=True):
        super(AlbertOriginLarge, self).__init__()
        self.args = args

        self.albert = AlbertModel.from_pretrained(args.pretrained_model_path)
        for name, param in self.albert.named_parameters():
            param.requires_grad = trained
        
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(1024, self.args.num_class))
        ]))
    
    def forward(self, input_ids, segment_ids, input_mask):
        albert_out = self.albert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        last_hidden_state = albert_out[0]
        pooled_output = albert_out[1]

        out = self.dropout(pooled_output)
        out = self.fc(out)

        return out

class AlbertOriginXLarge(nn.Module):
    def __init__(self, args, trained=True):
        super(AlbertOriginXLarge, self).__init__()
        self.args = args

        self.albert = AlbertModel.from_pretrained(args.pretrained_model_path)
        for name, param in self.albert.named_parameters():
            param.requires_grad = trained
        
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2048, self.args.num_class))
        ]))
    
    def forward(self, input_ids, segment_ids, input_mask):
        albert_out = self.albert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        last_hidden_state = albert_out[0]
        pooled_output = albert_out[1]

        out = self.dropout(pooled_output)
        out = self.fc(out)

        return out
