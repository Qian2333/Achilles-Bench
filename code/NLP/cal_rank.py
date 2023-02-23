# _*_ coding:utf-8 _*_

import torch
from torch.utils.data import DataLoader, Dataset
import os
import re
from random import sample
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from lib.dataset.aclimdb import ImdbDataset
from lib.dataset.glue import get_glue
from lib.util.toolbag import cal_para, get_gradient_tensor, setup_seed, get_gradient_tr
from lib.model.BCF import get_model
from tqdm import tqdm
import time
import argparse


DOUBLE_SENTENCES_SET = ['mrpc', 'rte', 'mnli', 'qnli', 'stsb', 'qqp', 'wnli']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', default='sst2', type=str,
                        help='the datasets')
    parser.add_argument('-md', '--mode', default='grad', type=str,
                        help='the datasets')
    parser.add_argument('-tr', '--attacked_model', default='bert', type=str,
                        help='the datasets')
    parser.add_argument('-t', '--train', default='t', type=str,
                        help='train before sample')
    parser.add_argument('-sd', '--seed', default=1, type=int,
                        help='seed for random')
    parser.add_argument('-ts', '--num_split', default=0, type=int,
                        help='seed for random')
    parser.add_argument('-id', '--block_id', default=0, type=int,
                        help='seed for random')
    args = parser.parse_args()

    setup_seed(args.seed)

    trainDatas, validata, _ = get_glue(name=args.s)
    print(len(trainDatas))

    begin_index = int(len(trainDatas) * args.block_id / args.num_split)
    end_index = int(len(trainDatas) * (args.block_id + 1) / args.num_split)
    print('range', begin_index, end_index)
    from lib.dataset.glue import MyDataset
    train_set = MyDataset(trainDatas.sentences[begin_index:end_index],
                          trainDatas.labels[begin_index:end_index])
    train_loader = trainDatas.train_loader(train_set, batch=1, shuffle=False)
    test_loader = validata.train_loader(batch=64)
    # train_loader = torch.utils.data.DataLoader(trainDatas[begin_index:end_index],
    #                                            batch_size=batchsize, shuffle=False)
    # print(trainDatas[0:2])
    # exit(0)
    print('training...')
    criterion = nn.MSELoss() if args.s == 'stsb' else nn.CrossEntropyLoss()

    if args.s in DOUBLE_SENTENCES_SET:
        net = get_model(args, mode='double_sentences', class_num=trainDatas.num_classes)
    else:
        net = get_model(args, class_num=trainDatas.num_classes)
    net = net.cuda()
    train_loader1 = trainDatas.train_loader(batch=64)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=2e-5)
    if args.train == 't':
        for i, data in enumerate(train_loader1):
            inputs, labels = data
            if args.s == 'stsb':
                labels = labels.to(torch.float32)
            labels = labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(i, len(train_loader1))

    train_str = '_train' if args.train == 't' else ''
    # rate_str = str(args.rate).split('.')
    # rate_str = rate_str[0] + rate_str[1]
    file_head = 'gd' if args.mode == 'grad' else 'ls'
    file_tr = 'logs/record1/' + file_head + '_' + args.s + train_str + '_loss_' + args.attacked_model + '.txt'
    optimizer = optim.SGD(net.parameters(), lr=1e-5)  # 

    bg = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        labels = labels.cuda()
        if args.s == 'stsb':
            labels = labels.to(torch.float32)
        outputs = net(inputs)
        if args.mode == 'loss':
            tr = criterion(outputs, labels)
        elif args.mode == 'grad':
            tr = get_gradient_norm(net, optimizer, criterion(outputs, labels))
        f2 = open(file_tr, 'a')
        f2.write(str(i + begin_index) + ' ' + str(float(tr)) + '\n')
        f2.close()
        if i % 10 == 0:
            print(time.time() - bg)
            # bg = time.time()


if __name__ == '__main__':
    main()


"""


"""