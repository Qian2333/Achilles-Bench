import torch
from lib.util.logger import Logger, str_pad
import tqdm
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import random
from lib.util.mytoolbag import cal_para, get_gradient_tensor, setup_seed
import torch.optim as optim
import time
import argparse
from lib.util.get_model import get_model
from lib.dataset.get_data import get_dataset


criterion = nn.CrossEntropyLoss()


def train_net(train_loader, net, optimizer, test_loader, rd=50, scheduler=None, logger=None, args=None):
    best_test_acc = 0
    best_train_acc = 0
    epoch = 0
    for i in range(rd):
        begin_time = time.time()
        epoch += 1
        train_acc, train_loss = 0, 0
        net.train()
        # pbad = tqdm.tqdm(total=len(train_loader))
        for iinbatch, data in enumerate(train_loader):
            inputs, labels = data
            if args.device == 'gpu':
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            predicted = torch.max(outputs, 1)[1].data.cpu().numpy()

            train_acc += (predicted == labels.data.cpu().numpy()).sum()
            train_loss += float(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # pbad.update(1)
            best_train_acc = max(best_train_acc, train_acc)

        if scheduler:
            scheduler.step()
        if args.dataset == 'imagenet':
            if i % 5 != 4 and i < 25:
                print('epoch : %d  ' % epoch, end='')
                print('train acc : %.1f ' % round(train_acc / len(train_loader.dataset) * 100, 2), end='')
                print(time.time() - begin_time)
                if logger is not None:
                    logger.epoch_log1(epoch, train_acc / len(train_loader.dataset) * 100,
                                      train_loss / len(train_loader))
                continue

        test_acc, test_loss = 0, 0
        net.eval()
        for data in test_loader:
            inputs, labels = data
            if args.device == 'gpu':
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            test_loss += float(criterion(outputs, labels))
            predicted = torch.max(outputs, 1)[1].data.cpu().numpy()
            test_acc += (predicted == labels.data.cpu().numpy()).sum()
        best_test_acc = max(best_test_acc, test_acc)
        print('epoch : %d  ' % epoch, end='')
        print('train acc : %.1f ' % round(train_acc / len(train_loader.dataset) * 100, 2), end='')
        print('test acc : %.1f ' % round(test_acc / len(test_loader.dataset) * 100, 2), end='')
        print(time.time() - begin_time)
        if logger is not None:
            logger.epoch_log2(epoch, train_acc / len(train_loader.dataset) * 100, train_loss / len(train_loader),
                              test_acc / len(test_loader.dataset) * 100, test_loss / len(test_loader))
    if logger is not None:
        logger.epoch_log2('end', best_train_acc / len(train_loader.dataset) * 100, 0 / len(train_loader),
                          best_test_acc / len(test_loader.dataset) * 100, 0 / len(test_loader))
    print(best_test_acc)
    return best_train_acc, best_test_acc


robust_bench = ['Rebuffi2021Fixing_70_16_cutmix_extra', 'Gowal2021Improving_70_16_ddpm_100m',
                'Rade2021Helper_extra', 'Rebuffi2021Fixing_70_16_cutmix_ddpm',
                'Gowal2020Uncovering_extra', 'Debenedetti2022Light_XCiT-L12']


def round1(i, selected_set, test_loader=None, rd=50, args=None, logger=None, data=None):
    setup_seed(args.seed_begin + i)
    net = get_model(args.attacked_model, dataset=args.dataset)
    if args.device == 'gpu':
        net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)
    if args.attacked_model == 'ViT':
        optimizer = optim.Adam(net.parameters(), lr=0.00005, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.LinearLR(optimizer)
        rd = 10
        if args.mode == 'ood_tr' or args.mode == 'ood_rnd':
            rd = 5
    elif args.attacked_model == 'EfficientNetV2':
        optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.005)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, epochs=20, steps_per_epoch=5000 // 32)
        rd = 15
    elif args.attacked_model == 'vgg16':
        optimizer = optim.Adam(net.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)
    # if args.attacked_model == 'densenet':
    #     optimizer = optim.Adam(net.parameters(), lr=1e-4)
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.5)
    if args.dataset == 'imagenet':
        rd = 30
        lr = 0.1
        if args.attacked_model == 'ffn':
            lr = 0.01
        if args.attacked_model == 'vgg16':
            lr = 0.01
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    if args.training_set != 'none':
        from lib.util.train_net import train_net_pro
        return train_net_pro(selected_set, net, optimizer, test_loader, rd=rd, scheduler=scheduler, logger=logger,
                             args=args, aug=args.training_set)
    return train_net(selected_set, net, optimizer, test_loader, rd=rd, scheduler=scheduler, logger=logger, args=args)


pathes = {
    'c10_ls_ffn_trained': './logs/record1/ls_cifar10_train_loss_ffn.txt',
    'c10_gd_ffn_trained': './logs/record1/gd_cifar10_train_loss_ffn.txt',
    'c100_ls_ffn_trained': './logs/record1/ls_cifar100_train_loss_ffn.txt',
    'c100_ffn_gdn_trained': './logs/record1/gd_cifar100_train_loss_ffn.txt',
    'c10_resnet18_gd': './logs/record1/gd_cifar10_train_loss_resnet18.txt',
    'c10_ViT_gd': './logs/record1/gd_cifar10_train_loss_ViT.txt',
    'c10_resnet18_ls': './logs/record1/ls_cifar10_train_loss_resnet18.txt',
    'c10_ViT_ls': './logs/record1/ls_cifar10_train_loss_ViT.txt',
    'img_ffn_trained_gdn': './logs/record1/gd_imagenet_train_loss_ffn.txt',
    'img_ls_ffn_trained': './logs/record1/ls_imagenet_train_loss_ffn.txt',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--dataset', default='cifar10', type=str,
                        help='the datasets')
    parser.add_argument('-m', '--mode', default='rnd', type=str,
                        help='the mode of random select and attack')
    parser.add_argument('-b', default=500, type=int,
                        help="the size of a single label data in selected dataset")
    parser.add_argument('-r', default=5, type=int, help="the round of test")
    parser.add_argument('-sd', '--seed_begin', default=0, type=int, help="seed begin")
    parser.add_argument('-rev', '--reverse', default=1, type=int, help='select the highest or lowest')
    parser.add_argument('-md', '--attacked_model', default='ffn', type=str, help='the model used to test')
    parser.add_argument('-tr', '--attacking_model', default='c10_ls_ffn_trained', type=str,
                        help='the model used to rank')
    parser.add_argument('-ts', '--training_set', default='none', type=str,
                        help='cutmix or not')
    parser.add_argument('-d', '--device', default='gpu', type=str,
                        help='gpu or cpu?')
    args = parser.parse_args()
    path1 = pathes[args.attacking_model]
    print(args)
    print(path1)

    batchsize = 32
    if args.attacked_model == 'ViT':
        config = {'norm': True, 'augment': False, 'size': 224}
        train_data, test_data = get_dataset(name=args.dataset, config=config)
    elif args.attacked_model == 'EfficientNetV2':
        config = {'norm': True, 'augment': False, 'size': 384}
        train_data, test_data = get_dataset(name=args.dataset, config=config)
    else:
        train_data, test_data = get_dataset(name=args.dataset)
    logger1 = Logger(name=args.dataset + '-' + args.attacked_model + '-' + args.mode + '-' + str(args.seed_begin) + '-'
                          + args.attacking_model + '-' + str(args.b) + '-' + str(args.reverse))
    logger2 = Logger(name='train_' + args.dataset + '_result', tim=False)
    print('data loaded')
    train_acc_list, test_acc_list = [], []
    if args.dataset == 'imagenet':
        args.r = 3
        print('test data process')
        batchsize = 256
        test_data.get_ready()
    test_loader = test_data.train_loader(batch=batchsize)
    selected_set = None
    if args.mode == 'tr':
        selected_set = train_data.get_attack_set(size=args.b, reverse=args.reverse, path=path1, batch=batchsize)
    for i in range(args.r):
        setup_seed(args.seed_begin * args.r + i)
        if args.mode == 'fullsize':
            selected_set = train_data.train_loader(batch=32)
            train_acc, test_acc = round1(i, selected_set, rd=100, test_loader=test_loader, args=args, logger=logger1,
                                         data=train_data)
            print('train acc:', train_acc, 'test acc:', test_acc)
            exit(0)
        elif args.mode == 'ood_rnd':
            selected_set, test_loader = train_data.get_ood_random(batch=batchsize)
        elif args.mode == 'ood_tr':
            selected_set, test_loader = train_data.get_ood_attack(batch=batchsize, path=path1)
        elif args.mode == 'rnd':
            selected_set = train_data.get_random_set(size=args.b, batch=batchsize)
        print('train data size', len(selected_set.dataset))
        print('test size: ', len(test_data.set))
        rd_def = 50
        train_acc, test_acc = round1(i, selected_set, rd=rd_def, test_loader=test_loader,
                                     args=args, logger=logger1, data=train_data)
        test_acc /= len(test_loader.dataset) / 100
        train_acc /= len(selected_set.dataset) / 100
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('test acc: ', round(sum(test_acc_list) / len(test_acc_list), 2), round(np.std(test_acc_list), 2),
              ' | worst acc:', round(np.min(test_acc_list), 2),
              ' | train acc: ', round(np.mean(train_acc_list), 2), round(np.std(train_acc_list), 2))
    logger2.info(
        'm-' + str_pad(args.mode, 5) +
        '|aug-' + str_pad(args.training_set, 6) +
        '|md-' + str_pad(args.attacked_model, 8) +
        '|tr-' + str_pad(args.attacking_model, 16) +
        '|b-' + str_pad(str(args.b), 5) +
        ' |test acc: ' + str_pad(str(round(np.mean(test_acc_list), 2)), 6) +
        '+' + str_pad(str(round(np.std(test_acc_list), 3)), 6) +
        ' |worst acc: ' + str_pad(str(round(np.min(test_acc_list), 2)), 6) +
        ' |train acc: ' + str_pad(str(round(np.mean(train_acc_list), 2)), 6) +
        '+' + str_pad(str(round(np.std(train_acc_list), 3)), 6)
    )
    logger2.info('----------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
