import torch
from torch.utils.data import DataLoader, Dataset
import os
import re
from random import sample
import numpy as np
from lib.util.get_model import get_model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.dataset.get_data import get_dataset
from lib.model.cifarnet import Net as ffn
from lib.model.cifarnet import ImgNet as Iffn
from lib.model.resnext_nbn import ResNeXt29_2x64d as resnext
from lib.model.vgg import VGG
from lib.util.mytoolbag import cal_para, get_gradient_tensor, setup_seed, get_gradient_norm
from tqdm import tqdm
import time
import argparse


criterion = nn.CrossEntropyLoss()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', default='cifar10', type=str,
                        help='the datasets')
    parser.add_argument('-md', '--mode', default='grad', type=str,
                        help='the datasets')
    parser.add_argument('-t', '--train', default='t', type=str,
                        help='train before sample')
    parser.add_argument('-m', '--model', default='ffn', type=str,
                        help='models for ranking')
    parser.add_argument('-sd', '--seed', default=1, type=int,
                        help='seed for random')
    parser.add_argument('-ts', '--num_split', default=1, type=int,
                        help='seed for random')
    parser.add_argument('-id', '--block_id', default=0, type=int,
                        help='seed for random')
    parser.add_argument('-en', '--epoch_num', default=0, type=int,
                        help='seed for random')
    parser.add_argument('-d', '--device', default='gpu', type=str,
                        help='gpu or cpu?')
    args = parser.parse_args()

    setup_seed(args.seed)

    if args.model == 'ViT' or args.model == 'EfficientNet':
        config = {'norm': True, 'augment': False, 'size': 224}
        train_data, test_data = get_dataset(name=args.s, config=config)
    else:
        train_data, test_data = get_dataset(name=args.s)
    print(len(train_data))

    if args.s != 'imagenet':
        begin_index = int(len(train_data) * args.block_id / args.num_split)
        end_index = int(len(train_data) * (args.block_id + 1) / args.num_split)
        print('range', begin_index, end_index)
        from lib.dataset.get_data import MyDataset
        train_set = MyDataset(train_data.images[begin_index:end_index],
                              train_data.labels[begin_index:end_index])
        train_loader = train_data.train_loader(train_set, batch=1, shuffle=False)
        test_loader = test_data.train_loader(train_set, batch=100, shuffle=False)
    else:
        begin_index = int(len(train_data) * args.block_id / args.num_split)
        end_index = int(len(train_data) * (args.block_id + 1) / args.num_split)
        train_loader = train_data.train_loader(batch=1, shuffle=False)
        test_loader = test_data.train_loader(batch=256, shuffle=False)
    if args.model == 'ViT':
        test_loader = test_data.train_loader(batch=32)
    print('training...(约1 hour(CPU))')

    net = get_model(args.model, dataset=args.s)
    if args.device == 'gpu':
        net = net.cuda()
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    if args.model == 'ViT':
        optimizer = optim.Adam(net.parameters(), lr=1e-4)
    epoch_num = 5
    if args.epoch_num != 0:
        epoch_num = args.epoch_num
    elif args.mode == 'grad':
        epoch_num = 5
    if args.model == 'ViT':
        epoch_num = 1
    if args.s == 'imagenet':
        epoch_num = 1
    if args.train == 't' and args.s == 'imagenet':
        from lib.util.imagenet_track import train_imagenet_step
        batch_size = 256
        one_step_num = batch_size * 200
        train_imagenet_step(train_data, test_data, net, optimizer, criterion, one_step_num, 1, batch_size)
    elif args.train == 't':
        train_loader1 = train_data.train_loader(batch=256)
        if args.model == 'ViT':
            train_loader1 = train_data.train_loader(batch=32)
        for epoch in range(epoch_num):
            net.train()
            train_acc, train_loss = 0, 0
            # pbar = tqdm(total=len(train_loader1))
            for i, data in enumerate(train_loader1):
                inputs, labels = data
                if args.device == 'gpu':
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                predicted = torch.max(outputs, 1)[1].data.cpu().numpy()

                acc_now = (predicted == labels.data.cpu().numpy()).sum()
                train_acc += acc_now
                train_loss += float(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(i, len(train_loader1))
                # pbar.update(1)
                # pbar.set_description("acc now {}".format(acc_now))
                # pbar.refresh()

            # pbar = tqdm(total=len(test_loader))
            test_acc, test_loss = 0, 0
            net.eval()
            for i, data in enumerate(test_loader):
                inputs, labels = data
                if args.device == 'gpu':
                    inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                test_loss += float(criterion(outputs, labels))
                predicted = torch.max(outputs, 1)[1].data.cpu().numpy()
                test_acc += (predicted == labels.data.cpu().numpy()).sum()
                print(i, len(test_loader))
                # pbar.update(1)
            print('epoch : %d  ' % epoch, end='')
            print('train acc : %.1f ' % round(train_acc / len(train_loader1.dataset) * 100, 2), end='')
            print('test acc : %.1f ' % round(test_acc / len(test_loader.dataset) * 100, 2))

    train_str = '_train' if args.train == 't' else ''
    file_head = 'gd' if args.mode == 'grad' else 'ls'
    file_tr = 'logs/record1/' + file_head + '_' + args.s + train_str + '_loss_' + args.model + '.txt'

    optimizer = optim.SGD(net.parameters(), lr=0.1)
    bg = time.time()
    net.train()
    # pbar = tqdm(total=len(train_loader) // 10)
    for i, (inputs, labels) in enumerate(train_loader):
        if args.device == 'gpu':
            inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        tr = 0
        if args.mode == 'loss':
            tr = criterion(outputs, labels)
        elif args.mode == 'grad':
            tr = get_gradient_norm(net, optimizer, criterion(outputs, labels))

        f2 = open(file_tr, 'a')
        f2.write(str(i + begin_index) + ' ' + str(float(tr)) + '\n')
        f2.close()
        if i % 10 == 0:
            print(time.time() - bg)
            # pbar.update(1)


if __name__ == '__main__':
    main()


"""

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u cal_rank_ave.py -s mrpc -ts 3 -id 2 > ntk_cal_mrpc2.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u cal_rank_ave.py -s sst2 -ts 3 -id 0 > ntk_cal_sst0.log 2>&1 &

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m vgg16 -ts 5 -id 0 > nls_cal_cifar10_vgg0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m vgg16 -ts 5 -id 1 > nls_cal_cifar10_vgg1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m vgg16 -ts 5 -id 2 > nls_cal_cifar10_vgg2.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m vgg16 -ts 5 -id 3 > nls_cal_cifar10_vgg3.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m vgg16 -ts 5 -id 4 > nls_cal_cifar10_vgg4.log 2>&1 &

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m ffn -ts 5 -id 0 > nls_cal_cifar10_ffn0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m ffn -ts 5 -id 1 > nls_cal_cifar10_ffn1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m ffn -ts 5 -id 2 > nls_cal_cifar10_ffn2.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m ffn -ts 5 -id 3 > nls_cal_cifar10_ffn3.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m ffn -ts 5 -id 4 > nls_cal_cifar10_ffn4.log 2>&1 &

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m resnext -ts 5 -id 0 > nls_cal_cifar10_resnext0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m resnext -ts 5 -id 1 > nls_cal_cifar10_resnext1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m resnext -ts 5 -id 2 > nls_cal_cifar10_resnext2.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m resnext -ts 5 -id 3 > nls_cal_cifar10_resnext3.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m resnext -ts 5 -id 4 > nls_cal_cifar10_resnext4.log 2>&1 &

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m ffn -ts 5 -id 0 > nls_cal_cifar10_ffn0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m ffn -ts 5 -id 1 > nls_cal_cifar10_ffn1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m ffn -ts 5 -id 2 > nls_cal_cifar10_ffn2.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m ffn -ts 5 -id 3 > nls_cal_cifar10_ffn3.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -m ffn -ts 5 -id 4 > nls_cal_cifar10_ffn4.log 2>&1 &

srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 0 > nls_cal_cifar100_ffn0.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 1 > nls_cal_cifar100_ffn1.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 2 > nls_cal_cifar100_ffn2.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 3 > nls_cal_cifar100_ffn3.log 2>&1 &
srun -p NLP --quotatype=spot --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 4 > nls_cal_cifar100_ffn4.log 2>&1 &

srun -p NLP --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 0 > nls_cal_cifar100_ffn0.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 1 > nls_cal_cifar100_ffn1.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 2 > nls_cal_cifar100_ffn2.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 3 > nls_cal_cifar100_ffn3.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 4 > nls_cal_cifar100_ffn4.log 2>&1 &

srun -p NLP --gres=gpu:1--cpus-per-task=16  -N1 python -u get_rank.py -s cifar100 -s imagenet -m ffn -ts 1 > nls_cal_img_ffn0.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 1 > nls_cal_cifar100_ffn1.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 2 > nls_cal_cifar100_ffn2.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 3 > nls_cal_cifar100_ffn3.log 2>&1 &
srun -p NLP --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -m ffn -ts 5 -id 4 > nls_cal_cifar100_ffn4.log 2>&1 &

srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -m resnet18 -ts 5 -id 0 > nls_cal_cifar10_res0.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -m resnet18 -ts 5 -id 1 > nls_cal_cifar10_res1.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -m resnet18 -ts 5 -id 2 > nls_cal_cifar10_res2.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -m resnet18 -ts 5 -id 3 > nls_cal_cifar10_res3.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -m resnet18 -ts 5 -id 4 > nls_cal_cifar10_res4.log 2>&1 &

srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -m ViT -ts 5 -id 0 > nls_cal_cifar10_vit0.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -m ViT -ts 5 -id 1 > nls_cal_cifar10_vit1.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -m ViT -ts 5 -id 2 > nls_cal_cifar10_vit2.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -m ViT -ts 5 -id 3 > nls_cal_cifar10_vit3.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -m ViT -ts 5 -id 4 > nls_cal_cifar10_vit4.log 2>&1 &

srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m ffn -ts 5 -id 0 > nls_cal_cifar10_ffn0.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m ffn -ts 5 -id 1 > nls_cal_cifar10_ffn1.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m ffn -ts 5 -id 2 > nls_cal_cifar10_ffn2.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m ffn -ts 5 -id 3 > nls_cal_cifar10_ffn3.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m ffn -ts 5 -id 4 > nls_cal_cifar10_ffn4.log 2>&1 &

srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m ViT -ts 5 -id 0 > nls_cal_cifar10_ffn0.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m ViT -ts 5 -id 1 > nls_cal_cifar10_ffn1.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m ViT -ts 5 -id 2 > nls_cal_cifar10_ffn2.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m ViT -ts 5 -id 3 > nls_cal_cifar10_ffn3.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m ViT -ts 5 -id 4 > nls_cal_cifar10_ffn4.log 2>&1 &

srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m resnet18 -ts 5 -id 0 > nls_cal_cifar10_ffn0.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m resnet18 -ts 5 -id 1 > nls_cal_cifar10_ffn1.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m resnet18 -ts 5 -id 2 > nls_cal_cifar10_ffn2.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m resnet18 -ts 5 -id 3 > nls_cal_cifar10_ffn3.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -m resnet18 -ts 5 -id 4 > nls_cal_cifar10_ffn4.log 2>&1 &

srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -s cifar100 -m ffn -ts 5 -id 0 > nls_cal_cifar100_ffn0.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -s cifar100 -m ffn -ts 5 -id 1 > nls_cal_cifar100_ffn1.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -s cifar100 -m ffn -ts 5 -id 2 > nls_cal_cifar100_ffn2.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -s cifar100 -m ffn -ts 5 -id 3 > nls_cal_cifar100_ffn3.log 2>&1 &
srun -p NLP --quotatype=auto --gres=gpu:1 -N1 python -u get_rank.py -md loss -s cifar100 -m ffn -ts 5 -id 4 > nls_cal_cifar100_ffn4.log 2>&1 &



srun -p NLP --quotatype=spot --cpus-per-task=16 --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -s imagenet
srun -p NLP --quotatype=auto --cpus-per-task=24 --gres=gpu:1 -N1 python -u get_rank.py -s cifar100 -s imagenet -md loss



"""