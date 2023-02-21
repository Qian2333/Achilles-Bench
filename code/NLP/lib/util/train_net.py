import torch
import tqdm
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import random
import torch.optim as optim
from lib.util.toolbag import setup_seed
import time
from lib.dataset.aclimdb import ImdbDataset
from lib.dataset.nlidata import get_sst2
from lib.dataset.wnli import get_wnli
from lib.model.BCF import BertClassificationModel
import argparse


criterion = nn.CrossEntropyLoss()


def train_net(train_loader, net, optimizer, testloader, rd=50, scheduler=None, logger=None):
    accl = 0
    acctrain = 0
    epoch = 0
    for i in range(rd):
        bg = time.time()
        epoch += 1
        train_acc, train_loss, test_loss = 0, 0, 0
        net.train()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            predicted = torch.max(outputs, 1)[1].data.cpu().numpy()
            train_acc += (predicted == labels.data.cpu().numpy()).sum()
            train_loss += float(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = 0
        net.eval()
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            test_loss += float(criterion(outputs, labels))
            predicted = torch.max(outputs, 1)[1].data.cpu().numpy()
            acc += (predicted == labels.data.cpu().numpy()).sum()
        accl = max(accl, acc)
        print('epoch : %d  ' % epoch, end='')
        print('acc : %.1f ' % acc, end='')
        print(time.time() - bg)
        if logger:
            logger.epoch_log2(epoch, train_acc / len(train_loader.dataset) * 100, train_loss / len(train_loader),
                              acc / len(testloader.dataset) * 100, test_loss / len(testloader))
        acctrain = max(acctrain, train_acc)
        if scheduler:
            scheduler.step()
    logger.epoch_log2('end', acctrain / len(train_loader.dataset) * 100, 0 / len(train_loader),
                      accl / len(testloader.dataset) * 100, 0 / len(testloader))
    print(accl)
    return acctrain, accl
