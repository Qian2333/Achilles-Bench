# _*_ coding:utf-8 _*_
# 利用深度学习做情感分析，基于Imdb 的50000个电影评论数据进行；

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
    parser.add_argument('-tr', '--attacked_model', default='bert', type=str,
                        help='the datasets')
    parser.add_argument('-r', '--rate', default=0.25, type=float,
                        help='the rate of sample parameters')
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
    print('training...(约1 hour(CPU))')
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
    file_tr = 'logs/record1/gd_' + args.s + train_str + '_loss_' + args.attacked_model + '.txt'
    optimizer = optim.SGD(net.parameters(), lr=1e-5)  # 首先定义优化器，这里用的AdamW，lr是学习率，因为bert用的就是这个

    tot_para = cal_para(net)
    bg = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        labels = labels.cuda()
        if args.s == 'stsb':
            labels = labels.to(torch.float32)
        outputs = net(inputs)
        tr = get_gradient_tr(net, optimizer, criterion(outputs, labels), tot=tot_para, rate=args.rate)
        f2 = open(file_tr, 'a')
        f2.write(str(i + begin_index) + ' ' + str(float(tr)) + '\n')
        f2.close()
        if i % 10 == 0:
            print(time.time() - bg)
            # bg = time.time()


if __name__ == '__main__':
    main()


"""

srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 -ts 3 -id 0 -tr gpt2 > ntk_gpt_sst0.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 -ts 3 -id 1 -tr gpt2 > ntk_gpt_sst1.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 -ts 3 -id 2 -tr gpt2 > ntk_gpt_sst2.log 2>&1 &

srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 -ts 3 -id 0 -tr transfomer > ntk_tf_sst0.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 -ts 3 -id 1 -tr transfomer > ntk_tf_sst1.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 -ts 3 -id 2 -tr transfomer > ntk_tf_sst2.log 2>&1 &


srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s mrpc -r 0.1 > ntk_cal_mrpc1.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s mrpc -r 0.5 > ntk_cal_mrpc5.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s mrpc -r 1 > ntk_cal_mrpc10.log 2>&1 &

srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s rte -r 0.1 > ntk_cal_rte1.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s rte -r 0.5 > ntk_cal_rte5.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s rte -r 1 > ntk_cal_rte10.log 2>&1 &

srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 -r 0.1 > ntk_cal_sst1.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 -r 0.5 > ntk_cal_sst5.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 -r 1 > ntk_cal_sst10.log 2>&1 &

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s mrpc -r 0.1 > ntk_cal_mrpc1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s mrpc -r 0.5 > ntk_cal_mrpc5.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s mrpc -r 1 > ntk_cal_mrpc10.log 2>&1 &

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s rte -r 0.1 > ntk_cal_rte1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s rte -r 0.5 > ntk_cal_rte5.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s rte -r 1 > ntk_cal_rte10.log 2>&1 &

srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 > ntk_cal_sst1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 > ntk_cal_sst5.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 > ntk_cal_sst10.log 2>&1 &

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s wnli -ts 3 -id 0 > ntk_cal_wnli0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s wnli -ts 3 -id 1 > ntk_cal_wnli1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s wnli -ts 3 -id 2 > ntk_cal_wnli2.log 2>&1 &

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s rte -ts 3 -id 0 > ntk_cal_rte0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s rte -ts 3 -id 1 > ntk_cal_rte1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s rte -ts 3 -id 2 > ntk_cal_rte2.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s mrpc -ts 3 -id 0 > ntk_cal_mrpc0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s mrpc -ts 3 -id 1 > ntk_cal_mrpc1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s mrpc -ts 3 -id 2 > ntk_cal_mrpc2.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 -ts 3 -id 0 > ntk_cal_sst0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 -ts 3 -id 1 > ntk_cal_sst1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s sst2 -ts 3 -id 2 > ntk_cal_sst2.log 2>&1 &

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s cola -ts 3 -id 0 > ntk_cal_cola0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s cola -ts 3 -id 1 > ntk_cal_cola1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s cola -ts 3 -id 2 > ntk_cal_cola2.log 2>&1 &

srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s cola -ts 3 -id 0 -t no > ntk_cal_cola0.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s cola -ts 3 -id 1 -t no > ntk_cal_cola1.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s cola -ts 3 -id 2 -t no > ntk_cal_cola2.log 2>&1 &

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s qnli -ts 5 -id 0 > ntk_cal_qnli0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s qnli -ts 5 -id 1 > ntk_cal_qnli1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s qnli -ts 5 -id 2 > ntk_cal_qnli2.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s qnli -ts 5 -id 3 > ntk_cal_qnli3.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s qnli -ts 5 -id 4 > ntk_cal_qnli4.log 2>&1 &

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s mnli -ts 5 -id 1 > ntk_cal_mnli1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s mnli -ts 5 -id 2 > ntk_cal_mnli2.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s mnli -ts 5 -id 3 > ntk_cal_mnli3.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s mnli -ts 5 -id 0 > ntk_cal_mnli0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s mnli -ts 5 -id 4 > ntk_cal_mnli4.log 2>&1 &

srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s qqp -ts 5 -id 0 > ntk_cal_qqp0.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s qqp -ts 5 -id 1 > ntk_cal_qqp1.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s qqp -ts 5 -id 2 > ntk_cal_qqp2.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s qqp -ts 5 -id 3 > ntk_cal_qqp3.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank.py -s qqp -ts 5 -id 4 > ntk_cal_qqp4.log 2>&1 &

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s qqp -ts 5 -id 0 > ntk_cal_qqp0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s qqp -ts 5 -id 1 > ntk_cal_qqp1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s qqp -ts 5 -id 2 > ntk_cal_qqp2.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s qqp -ts 5 -id 3 > ntk_cal_qqp3.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s qqp -ts 5 -id 4 > ntk_cal_qqp4.log 2>&1 &


srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s stsb -ts 5 -id 0 > ntk_cal_stsb0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s stsb -ts 5 -id 1 > ntk_cal_stsb1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s stsb -ts 5 -id 2 > ntk_cal_stsb2.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s stsb -ts 5 -id 3 > ntk_cal_stsb3.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank.py -s stsb -ts 5 -id 4 > ntk_cal_stsb4.log 2>&1 &



"""