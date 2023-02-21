import csv
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import random
from datasets import load_dataset
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import os
import torchvision
import torchvision.transforms as transforms
import numpy as np


class DataIO(object):
    def __init__(self, name='cifar10', config=None):
        self.train_image, self.train_label,\
            self.test_image, self.test_label = self.read_train_dev_test(name, config)
        self.num_classes = self.get_classes()

    def get_classes(self):
        classes = []
        for item in self.train_label:
            if item in classes:
                continue
            else:
                classes.append(item)
        return len(classes)

    def read_train_dev_test(self, nam, config):
        if nam == 'imagenet':
            train_image = []
            train_label = []
            with open('/mnt/cache/share/images/meta/train.txt') as f:
                while 1:
                    s = f.readline()
                    if not s:
                        break
                    s = s.split(' ')
                    train_image.append('/mnt/cache/share/images/train/' + s[0])
                    train_label.append(int(s[1]))
            test_image = []
            test_label = []
            with open('/mnt/cache/share/images/meta/val.txt') as f:
                while 1:
                    s = f.readline()
                    if not s:
                        break
                    s = s.split(' ')
                    test_image.append('/mnt/cache/share/images/val/' + s[0])
                    test_label.append(int(s[1]))
            return train_image, train_label, test_image, test_label
        if nam == 'cifar10' or nam == 'cifar100':
            if config['norm']:
                train_transform = transforms.Compose([
                    transforms.Resize(config['size']),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                test_transform = transforms.Compose(
                    [transforms.ToTensor(),
                     transforms.Resize(config['size']),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            else:
                train_transform = transforms.Compose([
                    transforms.ToTensor()])
                test_transform = transforms.Compose(
                    [transforms.ToTensor()])
            if config['augment']:
                train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])
            if nam == 'cifar10':
                train_set = torchvision.datasets.CIFAR10(
                    root='./data', train=True, download=True, transform=train_transform)
                test_set = torchvision.datasets.CIFAR10(
                    root='./data', train=False, download=True, transform=test_transform)
            else:
                train_set = torchvision.datasets.CIFAR100(
                    root='./data', train=True, download=True, transform=train_transform)
                test_set = torchvision.datasets.CIFAR100(
                    root='./data', train=False, download=True, transform=test_transform)
            train_image, train_label, test_image, test_label = [], [], [], []
            # print(len(train_set))
            # p_bar = tqdm(total=len(train_set))
            for item in train_set:
                image, label = item
                train_image.append(image)
                train_label.append(label)
            #     p_bar.update(1)
            # print('111')
            for item in test_set:
                image, label = item
                test_image.append(image)
                test_label.append(label)
            # print('loaded')
            # exit(0)
            return train_image, train_label, test_image, test_label


class ImageDataset(Dataset):
    def __init__(self, images, labels, num_classes=2, name='cifar'):
        self.images = images
        self.labels = labels
        self.set = MyDataset(self.images, self.labels)

        self.labeled_set = None
        self.name = name
        self.rank_list = None
        self.num_classes = num_classes

    def __getitem__(self, item):
        return self.images[item], self.labels[item]

    def get_labeled_set(self):
        images = [[] for i in range(self.num_classes)]
        for i in range(len(self.images)):
            images[self.labels[i]].append(self.images[i])
        self.labeled_set = images

    def get_id_set(self):
        idx, inputs, outputs = [], [], []
        for i, data in enumerate(self.set):
            idx.append(i)
            _input, _output = data
            inputs.append(_input)
            outputs.append(_output)
        return MyDataset3(idx, inputs, outputs)

    def __len__(self):
        return len(self.labels)

    def train_loader(self, data_set=None, batch=1, shuffle=True):
        if not data_set:
            return DataLoader(self.set, batch_size=batch, shuffle=shuffle)
        return DataLoader(data_set, batch_size=batch, shuffle=shuffle)

    def get_rank(self, path='./logs/record1/tr_sst_distilbert-base-uncased.txt', siz=None):
        f = open(path)
        rank_list = [[] for i in range(self.num_classes)]
        count = 0
        while 1:
            s = f.readline()
            if not s:
                break
            s = s.split()
            if len(s) > 1:
                id = int(s[0])
                norm = float(s[1])
                label = self.labels[id]
                rank_list[label].append(Item(id, norm))
            else:
                norm = float(s[0])
                label = self.labels[count]
                rank_list[label].append(Item(count, norm))
                count += 1
        for i in range(self.num_classes):
            rank_list[i].sort(key=lambda item: item.value)
        self.rank_list = rank_list

    def get_attack_set(self, size=10, reverse=1, path='./logs/record1/tr_sst_distilbert-base-uncased.txt', batch=32):
        self.get_rank(path=path)
        # self.get_class()
        images = []
        labels = []
        for i in range(self.num_classes):
            rank = self.rank_list[i]   # the rank of sentence with label i
            if reverse:
                rank.reverse()
            for j in range(size):
                image, label = self.set[rank[j].id]
                images.append(image)
                labels.append(label)

        return self.train_loader(data_set=MyDataset(images, labels), batch=batch)

    def get_ood_attack(self, rate=0.2, path='./logs/record1/tr_sst_distilbert-base-uncased.txt', reverse=1, batch=32):
        self.get_rank(path=path)
        train_images, train_labels = [], []
        test_images, test_labels = [], []
        for i in range(self.num_classes):
            rank = self.rank_list[i]   # the rank of sentence with label i
            if reverse:
                rank.reverse()
            for j in range(int(len(rank) * rate)):
                image, label = self.set[rank[j].id]
                test_images.append(image)
                test_labels.append(label)
            for j in range(int(len(rank) * rate), len(rank)):
                image, label = self.set[rank[j].id]
                train_images.append(image)
                train_labels.append(label)
        train_loader = self.train_loader(data_set=MyDataset(train_images, train_labels), batch=batch)
        test_loader = self.train_loader(data_set=MyDataset(test_images, test_labels), batch=batch)
        return train_loader, test_loader

    def get_random_set(self, size=500, top=None, batch=32):
        if not self.labeled_set:
            self.get_labeled_set()
        images, labels = [], []
        for i in range(self.num_classes):
            random.shuffle(self.labeled_set[i])
            for j in range(size):
                images.append(self.labeled_set[i][j])
                labels.append(i)
        return self.train_loader(data_set=MyDataset(images, labels), batch=batch)

    def get_ood_random(self, rate=0.2, batch=32):
        if not self.labeled_set:
            self.get_labeled_set()
        train_images, train_labels = [], []
        test_images, test_labels = [], []
        for i in range(self.num_classes):
            random.shuffle(self.labeled_set[i])
            for j in range(int(len(self.labeled_set[i]) * rate)):
                test_images.append(self.labeled_set[i][j])
                test_labels.append(i)
            for j in range(int(len(self.labeled_set[i]) * rate), len(self.labeled_set[i])):
                train_images.append(self.labeled_set[i][j])
                train_labels.append(i)
        train_loader = self.train_loader(data_set=MyDataset(train_images, train_labels), batch=batch)
        test_loader = self.train_loader(data_set=MyDataset(test_images, test_labels), batch=batch)
        return train_loader, test_loader

    def get_attack_kmean(self, model, size=500):
        self.get_labeled_set()
        sentences, labels = [], []
        for i in range(self.num_classes):
            items = []
            siz = size * len(self.labeled_set[i]) // len(self.labels)
            print(len(self.labeled_set[i]))
            for j, data in enumerate(self.labeled_set[i]):
                input_ids = torch.unsqueeze(data['input_ids'], 0)
                attention_mask = torch.unsqueeze(data['attention_mask'], 0)
                feature = model(input_ids, attention_mask, get_feature=True)
                items.append(Item(j, feature))
                print(feature.shape)
            dicts, centers = k_means(items, len(self.labels) // len(self.labeled_set[i]))
            for i in dicts.keys():
                print(len(dicts[i]))
            exit(0)

    def get_attack_set_ori(self, size=10, l=0, r=500, path='./logs/record1/tr_sst_distilbert-base-uncased.txt', batch=32):
        if not r:
            r = size
        # if not self.rank_list:
        self.get_rank(path=path)
        if not self.labeled_set:
            self.get_labeled_set()
        # self.get_class()
        images = []
        labels = []
        for i in range(self.num_classes):
            rank = self.rank_list[i]   # the rank of sentence with label i
            first_index = int(l * len(self.labeled_set[i]) / len(self.images))
            last_index = int(r * len(self.labeled_set[i]) / len(self.images))
            # random.shuffle(rank[first_index:last_index])
            # print(first_index, last_index, len(self.labeled_set[i]))
            for j in range(last_index - first_index):
                image, label = self.set[rank[first_index + j].id]
                images.append(image)
                labels.append(label)
        return self.train_loader(data_set=MyDataset(images, labels), batch=batch)


def get_dataset(name='cifar10', config=None):
    if not config:
        config = {'norm': True, 'augment': False, 'size': 32}
    dataset = DataIO(name=name, config=config)
    if name == 'imagenet':
        from lib.dataset.get_imagenet import ImagenetSet
        train_set = ImagenetSet(dataset.train_image, dataset.train_label, num_classes=dataset.num_classes)
        test_set = ImagenetSet(dataset.test_image, dataset.test_label, num_classes=dataset.num_classes)
        return train_set, test_set
    train_set = ImageDataset(dataset.train_image, dataset.train_label, num_classes=dataset.num_classes)
    test_set = ImageDataset(dataset.test_image, dataset.test_label, num_classes=dataset.num_classes)
    return train_set, test_set


class MyDataset(Dataset):
    def __init__(self, photo: list, label: list):
        self.pho = photo
        self.label = label

    def __getitem__(self, item):
        return self.pho[item], self.label[item]

    def __len__(self):
        return len(self.pho)


class MyDataset3(Dataset):
    def __init__(self, ids: list, photo: list, label: list):
        self.ids = ids
        self.pho = photo
        self.label = label

    def __getitem__(self, item):
        return self.ids[item], self.pho[item], self.label[item]

    def __len__(self):
        return len(self.ids)


def pil_loader(path: str):
    pic = Image.open(path)
    return pic.convert('RGB')


class Indata(Dataset):
    def __init__(self, photo: list, label: list, transform=None):
        self.pho = photo
        self.label = label
        self.transform = transform

    def __getitem__(self, item):
        pic = pil_loader(self.pho[item])
        if self.transform is not None:
            pic = self.transform(pic)
        return pic, self.label[item]

    def __len__(self):
        return len(self.pho)


class Item:
    def __init__(self, _id, value):
        self.id = _id
        self.value = value


def cal_distance(node: Item, centor: Item):
    return np.sqrt(np.sum(np.square(node.value - centor.value)))


def random_center(data, k):
    data = list(data)
    return random.sample(data, k)


def get_cluster(data, centor):
    cluster_dict = dict()
    k = len(centor)
    for node in data:
        cluster_class = -1
        min_distance = float('inf')
        for i in range(k):
            dist = cal_distance(node, centor[i])
            if dist < min_distance:
                cluster_class = i
                min_distance = dist
        if cluster_class not in cluster_dict.keys():
            cluster_dict[cluster_class] = []
        cluster_dict[cluster_class].append(node)
    return cluster_dict


def get_center(cluster_dict, k):
    new_center = []
    for i in range(k):
        centor = np.mean(cluster_dict[i], axis=0)
        new_center.append(centor)
    return new_center


def cal_varience(cluster_dict, centor):
    vsum = 0
    for i in range(len(centor)):
        cluster = cluster_dict[i]
        for j in cluster:
            vsum += cal_distance(j, centor[i])
    return vsum


def k_means(data, k):
    center = random_center(data, k)
    cluster_dict = get_cluster(data, center)
    new_varience = cal_varience(cluster_dict, center)
    old_varience = 1
    print(new_varience, old_varience)
    exit(0)
    while abs(old_varience - new_varience) > 0.1:
        centor = get_center(cluster_dict, k)
        cluster_dict = get_cluster(data, centor)
        old_varience = new_varience
        new_varience = cal_varience(cluster_dict, centor)
    return cluster_dict, center
