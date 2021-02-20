#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf 
import torch
from codecs import open
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import BertTokenizer

class BertTokenData(Dataset):
    def __init__(self, args):
        self.args = args
        
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_path)
        self.tokenizer.padding_side='right'

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_len):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_len:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def padding(self, input_id, input_mask, segment_id):

        if len(input_id) < self.args.max_len:
            pad_input_id = input_id + [0] * (self.args.max_len - len(input_id))
            pad_input_mask = input_mask + [0] * (self.args.max_len - len(input_mask))
            pad_segment_id = segment_id + [0] * (self.args.max_len - len(segment_id))
        else:
            pad_input_id = input_id[:self.args.max_len]
            pad_input_mask = input_mask[:self.args.max_len]
            pad_segment_id = segment_id[:self.args.max_len]

        return pad_input_id, pad_input_mask, pad_segment_id

    def sentence_to_idx(self, text_a, text_b):
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)

        self._truncate_seq_pair(tokens_a, tokens_b, self.args.max_len - 3)

        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

        input_ids, input_mask, segment_ids = self.padding(input_ids, input_mask, segment_ids)
        return input_ids, input_mask, segment_ids


class BertSimTrainData(BertTokenData):
    def __init__(self, args):
        super(BertSimTrainData, self).__init__(args)
        self.args = args

        self.files = [line.strip() for line in open(self.args.train_data_file, encoding=self.args.train_encoding, errors='ignore').readlines()]

    def __getitem__(self, index):
        parts = self.files[index].split('\t')
        query1 = parts[0]
        query2 = parts[1]
        label = int(parts[2])
        if self.args.segtype == 'char':
            query1 = ' '.join(list(str(query1)))
            query2 = ' '.join(list(str(query2)))
        
        # input_ids, input_mask, segment_ids = self.sentence_to_idx(query1, query2)
        token_dict = self.tokenizer.encode_plus(query1, query2, max_length=self.args.max_len, truncation=True, padding='max_length')

        input_ids = token_dict['input_ids']
        input_mask = token_dict['attention_mask']
        segment_ids = token_dict['token_type_ids']

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return input_ids, segment_ids, input_mask, label

    def __len__(self):
        return len(self.files)


class BertSimEvalData(BertTokenData):
    def __init__(self, args):
        super(BertSimEvalData, self).__init__(args)
        self.args = args

        self.files = [line.strip() for line in open(self.args.eval_data_file, encoding=self.args.eval_encoding, errors='ignore').readlines()]
    
    def __getitem__(self, index):
        parts = self.files[index].split('\t')
        query1 = parts[0]
        query2 = parts[1]
        label = int(parts[2])
        if self.args.segtype == 'char':
            query1 = ' '.join(list(str(query1)))
            query2 = ' '.join(list(str(query2)))
        
        token_dict = self.tokenizer.encode_plus(query1, query2, max_length=self.args.max_len, truncation=True, padding='max_length')

        input_ids = token_dict['input_ids']
        input_mask = token_dict['attention_mask']
        segment_ids = token_dict['token_type_ids']

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        return input_ids, segment_ids, input_mask, label

    def __len__(self):
        return len(self.files)
