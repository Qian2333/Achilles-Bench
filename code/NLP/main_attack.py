import torch
from lib.util.logger import Logger, str_pad
import tqdm
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import random
from lib.util.toolbag import cal_para, get_gradient_tensor, multi_tensor_gra
from lib.dataset.glue import get_glue
import torch.optim as optim
from scipy.stats import spearmanr, pearsonr
from lib.util.toolbag import setup_seed
from lib.model.BCF import get_model
import time
import argparse


DOUBLE_SENTENCES_SET = ['mrpc', 'rte', 'mnli', 'qnli', 'stsb', 'qqp', 'wnli', 'mnli_mismatched']


def cal_metric(predict, label, is_regression):
    predict = torch.squeeze(predict).cpu() if is_regression else torch.max(predict, 1)[1].data.cpu().numpy()
    if is_regression:
        return ((predict - label) ** 2).sum().item()
    return (predict == label).sum().item()


def train_net(train_loader, net, optimizer, testloader, rd=50, scheduler=None, logger=None,
              args=None, criterion=None):
    best_test_acc = 0
    best_train_acc = 0
    epoch = 0
    is_regression = 0
    if args.dataset == 'stsb':
        is_regression = 1
        best_p1t = -1
        best_p2t = 0
        best_p1 = -1
        best_p2 = 0
        best_r1t = 0
        best_r2t = 0
        best_r1 = 0
        best_r2 = 0
    for i in range(rd):
        begin_time = time.time()
        epoch += 1
        train_acc, train_loss = 0, 0
        net.train()
        if is_regression:
            pre_labels = torch.zeros([len(train_loader.dataset)], dtype=torch.float32)
            real_labels = torch.zeros([len(train_loader.dataset)], dtype=torch.float32)
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if args.dataset == 'stsb':
                labels = labels.to(torch.float32)
            labels = labels.cuda()
            outputs = net(inputs)
            if is_regression:
                pre_labels[
                    (i * labels.shape[0]):min((i + 1) * labels.shape[0], pre_labels.shape[0])
                ] = torch.squeeze(outputs.cpu().detach())
                real_labels[
                    (i * labels.shape[0]):min((i + 1) * labels.shape[0], pre_labels.shape[0])
                ] = labels.cpu()
            else:
                metric = cal_metric(outputs, labels.data.cpu().numpy(), is_regression)
                train_acc += metric
            loss = criterion(outputs.squeeze(), labels.squeeze())
            train_loss += float(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # pbad.update(1)
        if is_regression:
            p1t, r1t = pearsonr(pre_labels.numpy(), real_labels.numpy())
            p2t, r2t = spearmanr(pre_labels.numpy(), real_labels.numpy())
            if p1t > best_p1t:
                best_p1t = p1t
                best_r1t = r1t
                best_p2t = p2t
                best_r2t = r2t
        else:
            best_train_acc = max(best_train_acc, train_acc)

        test_acc, test_loss = 0, 0
        net.eval()
        if is_regression:
            pre_labels = torch.zeros([len(testloader.dataset)], dtype=torch.float32)
            real_labels = torch.zeros([len(testloader.dataset)], dtype=torch.float32)
        for i, data in enumerate(testloader):
            inputs, labels = data
            if args.dataset == 'stsb':
                labels = labels.to(torch.float32)
            labels = labels.cuda()
            outputs = net(inputs)

            if is_regression:
                pre_labels[
                    (i * labels.shape[0]):min((i + 1) * labels.shape[0], pre_labels.shape[0])
                ] = torch.squeeze(outputs.cpu().detach())
                real_labels[
                    (i * labels.shape[0]):min((i + 1) * labels.shape[0], pre_labels.shape[0])
                ] = labels.cpu()
            else:
                metric = cal_metric(outputs, labels.data.cpu().numpy(), is_regression)
                test_acc += metric
            test_loss += float(criterion(outputs, labels))

        if is_regression:
            p1, r1 = pearsonr(pre_labels.numpy(), real_labels.numpy())
            p2, r2 = spearmanr(pre_labels.numpy(), real_labels.numpy())
            if p1 > best_p1:
                best_p1 = p1
                best_r1 = r1
                best_p2 = p2
                best_r2 = r2
            print('epoch : %d  ' % epoch)
            print('train pearson : %.1f ' % round(p1t * 100, 2), ' r : %.1f ' % round(r1t * 100, 2),
                  'train spearman : %.1f ' % round(p2t * 100, 2), ' r : %.1f ' % round(r2t * 100, 2))
            print('test pearson : %.1f ' % round(p1 * 100, 2), ' r : %.1f ' % round(r1 * 100, 2),
                  'test spearman : %.1f ' % round(p2 * 100, 2), ' r : %.1f ' % round(r2 * 100, 2))
        else:
            best_test_acc = max(best_test_acc, test_acc)
            print('epoch : %d  ' % epoch, end='')
            print('train acc : %.1f ' % round(train_acc / len(train_loader.dataset) * 100, 2), end='')
            print('test acc : %.1f ' % round(test_acc / len(testloader.dataset) * 100, 2), end='')
        print(time.time() - begin_time)
        if logger is not None:
            if is_regression:
                logger.epoch_log2(epoch, p1t * 100, r1t * 100,
                                p1 * 100, r1 * 100)
                logger.epoch_log2(epoch, p2t * 100, r2t * 100,
                                p2 * 100, r2 * 100)
            else:
                logger.epoch_log2(epoch, train_acc / len(train_loader.dataset) * 100, train_loss / len(train_loader),
                                test_acc / len(testloader.dataset) * 100, test_loss / len(testloader))
        if scheduler:
            scheduler.step()
    if logger is not None:
        if is_regression:
            logger.epoch_log2('end', best_p1t * 100, best_r1t * 100,
                            best_p1 * 100, best_r1 * 100)
            logger.epoch_log2(epoch, best_p2t * 100, best_r2t * 100,
                            best_p2 * 100, best_r2 * 100)
        else:
            logger.epoch_log2('end', best_train_acc / len(train_loader.dataset) * 100, 0 / len(train_loader),
                            best_test_acc / len(testloader.dataset) * 100, 0 / len(testloader))
    if is_regression:
        return best_p1t * 100, best_p2t * 100, best_p1 * 100, best_p2 * 100
    print(best_test_acc)
    return best_train_acc, best_test_acc


def round1(i, selected_set, test_data=None, rd=50, args=None, logger=None, data=None):
    setup_seed(i)
    if args.dataset in DOUBLE_SENTENCES_SET:
        net = get_model(args, mode='double_sentences', class_num=data.num_classes)
    else:
        net = get_model(args, class_num=data.num_classes)
    net = net.cuda()
    # if args.dataset in DOUBLE_SENTENCES_SET:
    #     net = BertClassificationModel(mode='double_sentences', class_num=data.num_classes,
    #                                   model_name=args.attacked_model).cuda()
    # else:
    #     net = BertClassificationModel(class_num=data.num_classes,
    #                                   model_name=args.attacked_model).cuda()
    optimizer = optim.Adam(net.parameters(), lr=2e-5)
    scheduler = optim.lr_scheduler.LinearLR(optimizer)
    # if args.attacked_model == 'gpt2':
    #     optimizer = optim.Adam(net.parameters(), lr=5e-4)
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15], gamma=0.5)
    if args.dataset == 'stsb':
        return train_net(selected_set, net, optimizer, test_data, rd=rd, logger=logger,
                         scheduler=scheduler, args=args, criterion=nn.MSELoss())
    return train_net(selected_set, net, optimizer, test_data, rd=rd, logger=logger,
                     scheduler=scheduler, args=args, criterion=nn.CrossEntropyLoss())


pathes = {
    'sst2_bert_gd_t': './logs/record1/gd_sst2_train_loss_bbu.txt',
    'sst2_transformer_gd_t': './logs/record1/gd_sst2_train_loss_transfomer.txt',
    'sst2_gpt_gd_t': './logs/record1/gd_sst2_train_loss_gpt2.txt',
    'cola_bert_gd_t': './logs/record1/gd_cola_train_loss_bbu.txt',
    'mnli_bert_gd_t': './logs/record1/gd_mnli_train_loss_bbu.txt',
    'mrpc_bert_gd_t': './logs/record1/gd_mrpc_train_loss_bbu.txt',
    'rte_bert_gd_t': './logs/record1/gd_rte_train_loss_bbu.txt',
    'wnli_bert_gd_t': './logs/record1/gd_wnli_train_loss_bbu.txt',
    'qnli_bert_gd_t': './logs/record1/gd_qnli_train_loss_bbu.txt',
    'qqp_bert_gd_t': './logs/record1/gd_qqp_train_loss_bbu.txt',
    'stsb_bert_gd_t': './logs/record1/gd_stsb_train_loss_bbu.txt',
    'sst2_ls_bert_t': './logs/record1/ls_sst2_train_loss_bert.txt',
    'cola_ls_bert_t': './logs/record1/ls_cola_train_loss_bert.txt',
    'mnli_ls_bert_t': './logs/record1/ls_mnli_train_loss_bert.txt',
    'wnli_ls_bert_t': './logs/record1/ls_wnli_train_loss_bert.txt',
    'qqp_ls_bert_t': './logs/record1/ls_qqp_train_loss_bert.txt',
    'rte_ls_bert_t': './logs/record1/ls_rte_train_loss_bert.txt',
    'mrpc_ls_bert_t': './logs/record1/ls_mrpc_train_loss_bert.txt',
    'qnli_ls_bert_t': './logs/record1/ls_qnli_train_loss_bert.txt',
    'sst2_bert025_trained_loss': './logs/record1/tr_sst2_train_loss_distilbert-base-uncased025.txt',
    'mrpc_bert025_trained_loss': './logs/record1/tr_mrpc_train_loss_bert-base-uncased025.txt',
    'rte_bert025_trained_loss': './logs/record1/tr_rte_train_loss_bert-base-uncased025.txt',
    'rte_bertave_trained_loss': './logs/record1/tr_rte_train_loss_bbu_avetr.txt',
    'mrpc_trained_loss_bertave': './logs/record1/tr_mrpc_train_loss_bbu_avetr.txt',
    'sst2_trained_loss_bertave': './logs/record1/tr_sst2_train_loss_bbu_avetr.txt',
    'cola_trained_loss_bertave': './logs/record1/tr_cola_train_loss_bbu_avetr.txt',
    'mnli_trained_loss_bertave': './logs/record1/tr_mnli_train_loss_bbu_avetr.txt',
    'qnli_trained_loss_bertave': './logs/record1/tr_qnli_train_loss_bbu_avetr.txt',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--dataset', default='sst2', type=str,
                        help='the datasets')
    parser.add_argument('-m', '--mode', default='rnd', type=str,
                        help='the mode of random select and attack')
    parser.add_argument('-b', default=500, type=int,
                        help="the size of a single label data in selected dataset")
    parser.add_argument('-r', '--test_round', default=5, type=int, help="the round of test")
    parser.add_argument('-rev', '--reverse', default=1, type=int, help='highest or lowest')
    parser.add_argument('-md', '--attacked_model', default='bert-base-uncased',
                        type=str, help='the model used to test')
    parser.add_argument('-tr', '--attacking_model', default='sst2_bert_gd_t',
                        type=str, help='the model used to rank')
    args = parser.parse_args()
    # if args.dataset == 'mnli_mismatched':
    #     args.attacking_model = args.dataset[:4] + '_ls_bert_t'
    # else:
    #     args.attacking_model = args.dataset + '_ls_bert_t'
    path1 = pathes[args.attacking_model]
    print(args)
    print(path1)

    train_data, test_data, _ = get_glue(name=args.dataset)
    logger1 = Logger(name='1train-' + args.dataset + '-' + args.attacked_model + '-' + args.mode +
                          '-' + str(args.b) + '-' + args.attacking_model + '-' + str(args.reverse))
    logger2 = Logger(name='train_' + args.dataset + '_result', tim=False)
    train_acc_list, test_acc_list = [], []
    is_regression = 0
    if args.dataset == 'stsb':
        is_regression = 1
        p1t_list, p2t_list, p1_list, p2_list = [], [], [], []
    test_loader = test_data.train_loader(batch=100)
    for i in range(args.test_round):
        if args.mode == 'tr':
            selected_set = train_data.get_attack_set(size=args.b, reverse=args.reverse, path=path1)
        elif args.mode == 'fullsize':
            selected_set = train_data.train_loader(batch=32)
            train_acc, test_acc = round1(i, selected_set, rd=3, test_data=test_data, args=args, logger=logger1, data=train_data)
            print(train_acc, test_acc)
            exit(0)
        # elif args.mode == 'kmeans':
        #     if args.dataset in DOUBLE_SENTENCES_SET:
        #         net = BertClassificationModel(mode='double_sentences', class_num=data.num_classes).cuda()
        #     else:
        #         net = BertClassificationModel(class_num=data.num_classes).cuda()
        #     optimizer = optim.AdamW(net.parameters(), lr=1e-5)
        #     train_net(data.train_loader(batch=32), net, optimizer, test_data, rd=4)
        #     selected_set = data.get_attack_kmean(net, size=args.b)
        elif args.mode == 'ood_rnd':
            selected_set, test_loader = train_data.get_ood_random(batch=32)
        elif args.mode == 'ood_tr':
            selected_set, test_loader = train_data.get_ood_attack(batch=32, path=path1)
        else:
            selected_set = train_data.get_random_set(size=args.b)
        print('test size: ', len(test_loader.dataset))
        print('train data size', len(selected_set.dataset))
        rd_def = 50
        if args.b > 1000:
            rd_def = 10
        elif args.b > 100:
            rd_def = 20
        if args.mode == 'ood_rnd' or args.mode == 'ood_tr':
            if len(selected_set.dataset) > 10000:
                rd_def = 5
            else:
                rd_def = 100
        if is_regression:
            p1t, p2t, p1, p2 = round1(i, selected_set, rd=rd_def, test_data=test_loader,
                                      args=args, logger=logger1, data=train_data)
            p1t_list.append(p1t), p2t_list.append((p2t)), p1_list.append(p1), p2_list.append(p2)
            print('pearson: ', round(np.mean(p1_list), 2), round(np.std(p1_list), 2),
                  ' | worst:', round(np.min(p1_list), 2),
                  ' | train: ', round(np.mean(p1t_list), 2), round(np.std(p1t_list), 2))
            print('spearman: ', round(np.mean(p2_list), 2), round(np.std(p2_list), 2),
                  ' | worst:', round(np.min(p2_list), 2),
                  ' | train: ', round(np.mean(p2t_list), 2), round(np.std(p2t_list), 2))
            continue
        train_acc, test_acc = round1(i, selected_set, rd=rd_def, test_data=test_loader,
                                     args=args, logger=logger1, data=train_data)
        test_acc /= len(test_loader.dataset) / 100
        train_acc /= len(selected_set.dataset) / 100
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('test acc: ', round(np.mean(test_acc_list), 2), round(np.std(test_acc_list), 2),
              ' | worst acc:', round(np.min(test_acc_list), 2),
              ' | train acc: ', round(np.mean(train_acc_list), 2), round(np.std(train_acc_list), 2))
    if is_regression:
        logger2.info(
            'm-' + str_pad(args.mode, 5) +
            '|md-' + str_pad(args.attacked_model, 8) +
            '|tr-' + str_pad(args.attacking_model, 8) +
            '|b-' + str_pad(str(args.b), 5) + '\n' +
            'pearson: ' + str_pad(str(round(np.mean(p1_list), 2)), 6) +
            '+' + str_pad(str(round(np.std(p1_list), 3)), 6) +
            ' |worst acc: ' + str_pad(str(round(np.min(p1_list), 2)), 6) +
            ' |train acc: ' + str_pad(str(round(np.mean(p1t_list), 2)), 6) +
            '+' + str_pad(str(round(np.std(p1t_list), 3)), 6) + '\n' +
            'spearman: ' + str_pad(str(round(np.mean(p2_list), 2)), 6) +
            '+' + str_pad(str(round(np.std(p2_list), 3)), 6) +
            ' |worst acc: ' + str_pad(str(round(np.min(p2_list), 2)), 6) +
            ' |train acc: ' + str_pad(str(round(np.mean(p2t_list), 2)), 6) +
            '+' + str_pad(str(round(np.std(p2t_list), 3)), 6)
        )
        return
    logger2.info(
        'm-' + str_pad(args.mode, 5) +
        '|md-' + str_pad(args.attacked_model, 10) +
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

"""


"""