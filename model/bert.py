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
from transformers import BertModel, BertPreTrainedModel

class BertOrigin(nn.Module):
    def __init__(self, args):
        super(BertOrigin, self).__init__()
        self.args = args

        self.bert = BertModel.from_pretrained(args.pretrained_model_path) # bert-base-chinese
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(768, 2))
        ]))
    
    def forward(self, input_ids, segment_ids, input_mask):
        # all_encoder_layers 
        # output_all_encoded_layers=False shape:[batch_size, seq_len, bert_dim=768]
        # output_all_encoded_layers=True list_len=12, all_encoder_layers[0] shape:[batch_size, seq_len, bert_dim=768]
        # pooled_output shape [batch_size, bert_dim=768]
        # from pytorch_pretrained_bert import BertModel
        # all_encoder_layers, pooled_output = self.bert(input_ids, segment_ids, input_mask)
        
        # last_encoder_layers shape:[batch_size, seq_len, bert_dim=768] output all word embedding
        # all_encoder_layers output_hidden_states=True list_len=12, all_encoder_layers[0] shape:[batch_size, seq_len, bert_dim=768]
        # pooled_output shape [batch_size, bert_dim=768]
        bert_out = self.bert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, output_hidden_states=True)
        last_encoder_layers = bert_out[0]
        pooled_output = bert_out[1]
        embedding_output = bert_out[2][0]
        all_encoder_layers = bert_out[2][1:]

        out = self.fc(pooled_output)
        return out

class BertCNN(nn.Module):
    def __init__(self, args, trained=True):
        super(BertCNN, self).__init__()
        self.args = args

        self.bert = BertModel.from_pretrained(args.pretrained_model_path)
        for name, param in self.bert.named_parameters():
            param.requires_grad = trained

        self.cnn_list = nn.ModuleList([
            nn.Sequential(OrderedDict([
                ('conv1d', nn.Conv1d(768, self.args.filters, kernel_size)),
                ('relu', nn.ReLU(inplace=True)),
                ('max_pool', nn.MaxPool1d(self.args.max_len - kernel_size + 1))
            ]))
            for kernel_size in self.args.kernel_size_list   
        ])

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(len(self.args.kernel_size_list) * self.args.filters, 2))
        ]))
    
    def forward(self, input_ids, segment_ids, input_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
        last_encoder_layers = bert_out[0] # [batch, max_len, bert_dim]
        pooled_output = bert_out[1]

        pool_list = []
        for cnn in self.cnn_list:
            cnn_feature = cnn(last_encoder_layers.permute(0, 2, 1)).squeeze(2) # [batch, filters]
            pool_list.append(cnn_feature)
        
        out = torch.cat(pool_list, dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

 '''
class SiameseBert(BertPreTrainedModel):
    def __init__(self, config, num_labels=2, add_pooling_layer=True):
        super(SiameseBert, self).__init__(config)
    
        self.bert = BertModel(config, add_pooling_layer=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size * 3, num_labels)

        self.init_weights()
    
    def forward(
        self,
        input_ids1, 
        input_ids2, 
        attention_mask1=None,
        attention_mask2=None,
        token_type_ids1=None,
        token_type_ids2=None
    ):
        encode1 = self.bert(
            input_ids1,
            attention_mask=attention_mask1,
            token_type_ids=token_type_ids1
        )

        encode2 = self.bert(
            input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids2
        )

        pooled_output1 = encode1[1]
        pooled_output2 = encode2[1]

        out1 = self.dropout(pooled_output1)
        out2 = self.dropout(pooled_output2)

        add = out1 + out2
        abssub = torch.abs(out1 - out2)
        mul = out1 * out2

        out = torch.cat([add, abssub, mul], dim=1)

        out = self.classifier(out)

        return out 

'''
