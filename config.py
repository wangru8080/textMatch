#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import pickle
import argparse
import ast
import torch 

def Config():
    parser = argparse.ArgumentParser(description='Process Similarity Sentence.')

    parser.add_argument('--is_print_conf', 
                        type=ast.literal_eval, 
                        default=False, 
                        required=False, 
                        help='whether to print conf info')

    # data file conf
    parser.add_argument('--train_data_file', 
                        type=str, 
                        default='/search/odin/wangru/textMatch/data/dev.tsv', 
                        required=False, 
                        help='train data file')
    parser.add_argument('--eval_data_file', 
                        type=str, 
                        default='/search/odin/wangru/textMatch/data/case.tsv', 
                        required=False, 
                        help='eval data file')
    parser.add_argument('--predict_data_file', 
                        type=str, 
                        default='/search/odin/wangru/textMatch/data/case.tsv', 
                        required=False, 
                        help='predict data file')
    parser.add_argument('--train_encoding', 
                        type=str, 
                        default='utf-8', 
                        required=False, 
                        help='train data file encoding')
    parser.add_argument('--eval_encoding', 
                        type=str, 
                        default='utf-8', 
                        required=False, 
                        help='eval data file encoding')
    parser.add_argument('--predict_encoding', 
                        type=str, 
                        default='utf-8', 
                        required=False, 
                        help='predict data file encoding')

    # gpu conf
    parser.add_argument('--gpuid', 
                        type=str, 
                        default='0', 
                        required=False, 
                        help='choose gpu ids')
    parser.add_argument('--n_gpu', 
                        type=int, 
                        default=1, 
                        required=False, 
                        help='Total number of gpu')
    parser.add_argument('--use_gpu', 
                        type=ast.literal_eval, 
                        default=torch.cuda.is_available(), 
                        required=False, 
                        help='whether to use gpu')
    
    # data conf
    parser.add_argument('--num_workers', 
                        type=int, 
                        default=10, 
                        required=False, 
                        help='how many subprocesses to use for data loading. '
                        '``0`` means that the data will be loaded in the main process')
    parser.add_argument('--per_gpu_batch_size', 
                        type=int, 
                        default=128, 
                        required=False,
                        help='how many samples per batch to load')
    parser.add_argument('--shuffle', 
                        type=ast.literal_eval, 
                        default=True, 
                        required=False,
                        help='set to ``True`` to have the data reshuffled at every epoch')
    parser.add_argument('--segtype',
                        type=str,
                        default='char',
                        required=False,
                        help='text segment type. choose word or char')

    # model conf
    parser.add_argument('--bert_path', 
                        type=str, 
                        default='/search/odin/wangru/textMatch/bert-base-chinese/', 
                        required=False, 
                        help='bert model&vocab path')
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=2e-5, 
                        required=False, 
                        help='The initial learning rate for Adam.')
    parser.add_argument('--max_len', 
                        type=int, 
                        default=25, 
                        required=False, 
                        help='The maximum total input sequence length after WordPiece tokenization. \n'
                             'Sequences longer than this will be truncated, and sequences shorter \n'
                             'than this will be padded.')
    parser.add_argument('--num_class', 
                        type=int, 
                        default=2, 
                        required=False, 
                        help='num class')
    parser.add_argument('--num_train_epochs', 
                        type=int, 
                        default=3, 
                        required=False, 
                        help='Total number of training epochs to perform.')
    parser.add_argument('--warmup_proportion',
                        default=0.1,
                        type=float,
                        required=False, 
                        help='Proportion of training to perform linear learning rate warmup for. '
                        'E.g., 0.1 = 10%% of training.')
    parser.add_argument('--gradient_accumulation_steps', 
                        type=int, 
                        default=1, 
                        required=False, 
                        help='Number of updates steps to accumulate before performing a backward/update pass.')

    parser.add_argument('--seed', 
                        type=int, 
                        default=2021, 
                        required=False, 
                        help='random seed for initialization')

    args = parser.parse_args()

    args.n_gpu = len(args.gpuid.split(','))
    if args.gpuid == '-1':
        args.use_gpu = False
        args.n_gpu = 0
    if args.use_gpu == False:
        args.gpuid = '-1'
        args.n_gpu = 0

    if args.is_print_conf:
        for name, value in vars(args).items():
            print('%s: %s' % (name, value))

    return args
