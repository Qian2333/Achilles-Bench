import torch
import time
import tqdm
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import random
from lib.util.mytoolbag import cal_para, get_gradient_tensor, multi_tensor_gra
from lib.dataset.mydata import CifarData
from torch.utils.data import DataLoader
from lib.model.cifarnet import Net
import torch.optim as optim
from lib.util.mytoolbag import setup_seed


criterion = nn.CrossEntropyLoss().cuda()


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class Pic:
    def __init__(self, _id, ntk):
        self.id = _id
        self.ntk = ntk


def mix_up_f(inp, lab, rnk=None, lam=None, method=None, idd=1):
    if not lam:
        lam = np.random.beta(1, 1)
    batch_size = inp.size()[0]
    index = torch.randperm(batch_size).cuda()
    lis = []
    for i in range(rnk.size()[0]):
        lis.append(Pic(i, rnk[i]))
    lis.sort(key=lambda pic: pic.ntk)
    lis2 = [0 for i in range(batch_size)]
    for i in range(rnk.size()[0]):
        lis2[lis[i].id] = i
    # l1 = (batch_size // 2) + idd
    l1 = (batch_size // 2)
    if method == 'near':
        l1 = 1
    elif method == 'near_rk':
        l1 = idd
    for i in range(rnk.size()[0]):
        if method == 'near_r':
            l1 = random.randint(1, 50)
        index[i] = i
        if random.randint(1, 10) != 1:
            index[i] = lis[(lis2[i] + l1) % batch_size].id
            while lab[index[i]] == lab[i]:
                index[i] = lis[(lis2[index[i]] + 1) % batch_size].id
        else:
            index[i] = lis[(lis2[i] + l1) % batch_size].id
            while lab[index[i]] != lab[i]:
                index[i] = lis[(lis2[index[i]] + 1) % batch_size].id
    if lam < 0.5:
        lam = 1 - lam
    mixed_x = lam * inp + (1 - lam) * inp[index, :]
    y_a, y_b = lab, lab[index]

    return mixed_x, y_a, y_b, lam


def rand_box(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(inp, lab, lam=None):
    if not lam:
        lam = np.random.beta(1, 1)
    batch_size = inp.size()[0]
    index = torch.randperm(batch_size)

    # mixed_x = lam * inp + (1 - lam) * inp[index, :]
    y_a, y_b = lab, lab[index]
    bbx1, bby1, bbx2, bby2 = rand_box(inp.size(), lam)
    inp[:, :, bbx1:bbx2, bby1:bby2] = inp[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inp.size()[-1] * inp.size()[-2]))  # 按照box面积计算lam

    return inp, y_a, y_b, lam


def mix_up(inp, lab, lam=None):
    if not lam:
        lam = np.random.beta(1, 1)
    batch_size = inp.size()[0]
    index = torch.randperm(batch_size)

    mixed_x = lam * inp + (1 - lam) * inp[index, :]
    y_a, y_b = lab, lab[index]

    return mixed_x, y_a, y_b, lam


def train_net_pro(train_loader, net, optimizer, test_loader, rd=50, scheduler=None, logger=None, args=None, aug=None):
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
            inputs, labels = inputs.cuda(), labels.cuda()
            if aug == 'mixup':
                inputs, labels_a, labels_b, lam = mix_up(inputs, labels)
                outputs = net(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam=lam)
            elif aug == 'cutmix':
                inputs, labels_a, labels_b, lam = cutmix(inputs, labels)
                outputs = net(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam=lam)
            else:
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


def mix_up234(inp, lab, rnk=None, lam=None, method=None, idd=1, test_long=False):
    if not lam:
        lam = np.random.beta(1, 1)
    batch_size = inp.size()[0]
    index = torch.randperm(batch_size).cuda()
    # print(index)
    # input()
    if test_long:
        long = 0
        for i in range(batch_size):
            long += abs(rnk[i] - rnk[index[i]])
        return long / batch_size

    mixed_x = lam * inp + (1 - lam) * inp[index, :]
    y_a, y_b = lab, lab[index]

    return mixed_x, y_a, y_b, lam


def train_net1342143241(train_loader, net, optimizer, testloader, mixup=None, epoch1=100, scheduler=None):
    accl = 0
    epoch = 0
    print(mixup)
    for i in range(epoch1):
        epoch += 1
        bg = time.time()
        net.train()
        # acc2 = 0
        for _, data in enumerate(train_loader):
            inputs, labels, rnk = data
            if mixup == 'None':
                inputs, l_a, l_b, lam = mix_up(inputs, labels)
            elif mixup == 'ori':
                inputs, l_a, l_b, lam = inputs, labels, labels, 0
            elif mixup == 'far':
                inputs, l_a, l_b, lam = mix_up_f(inputs, labels, rnk, method='far', idd=i)
            elif mixup == 'near_rk':
                inputs, l_a, l_b, lam = mix_up_f(inputs, labels, rnk, method='near_rk', idd=i)
            elif mixup == 'near_r':
                inputs, l_a, l_b, lam = mix_up_f(inputs, labels, rnk, method='near_r', idd=i)
            elif mixup == 'near':
                inputs, l_a, l_b, lam = mix_up_f(inputs, labels, rnk, method='near')
            elif mixup == 'nfnf':
                if i < epoch1 * 0.9:
                    inputs, l_a, l_b, lam = mix_up_f(inputs, labels, rnk, method='far', idd=i)
                else:
                    inputs, l_a, l_b, lam = mix_up_f(inputs, labels, rnk, method='near', idd=i)
            elif mixup == 'fnfn':
                if i < epoch1 * 0.4:
                    inputs, l_a, l_b, lam = mix_up_f(inputs, labels, rnk, method='near', idd=i)
                else:
                    inputs, l_a, l_b, lam = mix_up_f(inputs, labels, rnk, method='far', idd=i)
            # print(labels)
            inputs, l_a, l_b = Variable(inputs).cuda(), Variable(l_a).cuda(), Variable(l_b).cuda()
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, l_a, l_b, lam=lam)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # acc2 += (torch.max(outputs, 1)[1].data.cpu().numpy() == labels.data.cpu().numpy()).sum() / 500
        # print('train acc : %d  ' % acc2, end='')
        # rate = abs((lost[-min(10, epoch)] - loss) / loss)break
        # if epoch1 - epoch < 20:
        acc = 0
        net.eval()
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(Variable(images))
            predicted = torch.max(outputs, 1)[1].data.cpu().numpy()
            acc += (predicted == labels.data.cpu().numpy()).sum() / 100
        accl = max(accl, acc)
        print('epoch : %d  ' % epoch, end='')
        print('acc : %.1f ' % acc, end='')
        print(time.time() - bg)
        # # print('loss: ', loss)
        # if epoch > 20 and sum(accl[-10:]) <= sum(accl[-20:-10]) + 0.1:
        #     print('')
        #     break
        # lost.append(loss)
        if scheduler:
            scheduler.step()
    print(accl)
    return accl




def mix_up_f11(inp, lab, rnk=None, lam=None, method=None, idd=1):
    if not lam:
        lam = np.random.beta(1, 1)
    batch_size = inp.size()[0]
    index = torch.randperm(batch_size).cuda()
    lis = []
    for i in range(rnk.size()[0]):
        lis.append(Pic(i, rnk[i]))
    lis.sort(key=lambda pic: pic.ntk)
    lis2 = [0 for i in range(batch_size)]
    for i in range(rnk.size()[0]):
        lis2[lis[i].id] = i
    # l1 = (batch_size // 2) + idd
    l1 = (batch_size // 2)
    if method == 'near':
        l1 = 1
    elif method == 'near_rk':
        l1 = idd
    for i in range(rnk.size()[0]):
        if method == 'near_r':
            l1 = random.randint(1, 50)
        index[i] = i
        if random.randint(1, 10) != 1:
            index[i] = lis[(lis2[i] + l1) % batch_size].id
            while lab[index[i]] == lab[i]:
                index[i] = lis[(lis2[index[i]] + 1) % batch_size].id
        else:
            index[i] = lis[(lis2[i] + l1) % batch_size].id
            while lab[index[i]] != lab[i]:
                index[i] = lis[(lis2[index[i]] + 1) % batch_size].id
    long = 0
    for i in range(batch_size):
        long += abs(rnk[i] - rnk[index[i]])
    return long / batch_size


def train_net1(train_loader, net, optimizer, testloader, mixup=None, epoch1=100):
    accl = []
    epoch = 0
    same = 0
    diff = 0
    for i in range(1):
        epoch += 1
        loss = 0
        long1 = []
        for _, data in enumerate(train_loader):
            inputs, labels, rnk = data
            long = 0
            if mixup == 'None':
                # inputs, l_a, l_b, lam = mix_up(inputs, labels, rnk=rnk, test_long=True)
                long = mix_up(inputs, labels, rnk=rnk, test_long=True)

            elif mixup == 'far':
                # inputs, l_a, l_b, lam = mix_up_f(inputs, labels, rnk, method='far')
                long =  mix_up_f11(inputs, labels, rnk, method='far')
            elif mixup == 'near_rk':
                # inputs, l_a, l_b, lam = mix_up_f(inputs, labels, rnk, method='near_rk', idd=i)
                long =  mix_up_f11(inputs, labels, rnk, method='near_rk', idd=i)
            elif mixup == 'near_r':
                # inputs, l_a, l_b, lam = mix_up_f(inputs, labels, rnk, method='near_r', idd=i)
                long =  mix_up_f11(inputs, labels, rnk, method='near_r', idd=i)
            else:
                # inputs, l_a, l_b, lam = mix_up_f(inputs, labels, rnk, method='near')
                long =  mix_up_f11(inputs, labels, rnk, method='near')
            # print(labels)
            long1.append(long)
            print(long, end=' ')
            print(sum(long1) / len(long1))


            # 比例
            # smm = sum(l_a == l_b)
            # same += smm
            # diff += len(l_a) - smm
            # print(same, diff, float(same) / (float(diff) + float(same)))


        # rate = abs((lost[-min(10, epoch)] - loss) / loss)break
        # # print('loss: ', loss)
        # if epoch > 20 and sum(accl[-10:]) <= sum(accl[-20:-10]) + 0.1:
        #     print('')
        #     break
        # lost.append(loss)
    return same, diff