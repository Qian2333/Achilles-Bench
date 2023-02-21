import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import time
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from PIL import Image


class Pic:
    def __init__(self, _id, ntk):
        self.id = _id
        self.ntk = ntk


def pil_loader(path: str):
    pic = Image.open(path)
    return pic.convert('RGB')


class Indata(data.Dataset):
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


def get_sig(path):
    pic = pil_loader(path)
    return pic


def get_sig1(path):
    pic = pil_loader(path)
    pic = pic.resize((256, 256), Image.BILINEAR)
    # pic = former(pic)
    return pic


former = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, 4),
    # transforms.RandomResizedCrop(size),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

former1 = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32, 4),
    # transforms.RandomResizedCrop(size),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def read_imgdata(path, transform=None):
    print('loading pic')
    bg = time.time()
    P = Pool(processes=cpu_count())
    picss = []
    pics = list(tqdm(P.imap(func=get_sig, iterable=path), total=len(path)))
    pbar = tqdm(total=len(path))
    for pic in pics:
        picss.append(torch.tensor(transform(pic)))
        pbar.update(1)
    print(time.time() - bg)
    return picss


def read_imgdata2(path, transform=None):
    print('loading pic')
    bg = time.time()
    P = Pool(processes=cpu_count())
    picss = []
    pbar = tqdm(total=len(path))
    for i in range(0, len(path), 5000):
        pics = P.map(func=get_sig, iterable=path[i:min(i + 5000, len(path))])
        for pic in pics:
            picss.append(torch.tensor(former(pic), dtype=torch.float16))
            pbar.update(1)
    print(time.time() - bg)
    return picss


def get_imgdata2(path, lab, transform=None):
    pics = []
    pbar = tqdm(total=len(path))
    for pt in path:
        pic = pil_loader(pt)
        pic = former(pic)
        pics.append(pic)
        pbar.update(1)
    return MyDataset(pics, lab)


def get_imgdata(path, lab, transform=None):
    # if len(path) > 2e5 and len(path) < 1e6:
    #     return get_imgdata2(path, lab)
    pics = read_imgdata(path)
    return MyDataset(pics, lab)


class ImagenetSet(Dataset):
    def __init__(self, images, labels, num_classes=2, size=224):
        self.images = images
        self.labels = labels
        self.transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        if size != 224:
            self.transformer = transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.set = Indata(self.images, self.labels, transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                          ]))
        self.labeled_set = None
        self.rank_list = None
        self.num_classes = num_classes

    def __getitem__(self, item):
        return self.images[item], self.labels[item]

    def __len__(self):
        return len(self.labels)

    def get_ready(self):
        self.images = read_imgdata(self.images, self.transformer)
        self.set = MyDataset(self.images, self.labels)

    def train_loader(self, data_set=None, batch=1, shuffle=True):
        if not data_set:
            return DataLoader(self.set, batch_size=batch, shuffle=shuffle)
        return DataLoader(data_set, batch_size=batch, shuffle=shuffle)

    def get_labeled_set(self):
        images = [[] for i in range(self.num_classes)]
        for i in range(len(self.images)):
            images[self.labels[i]].append(self.images[i])
        self.labeled_set = images

    def get_rank(self, path='./logs/record1/gd_imagenet_train_loss_ffn.txt', siz=None):
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
        images = []
        labels = []
        for i in range(self.num_classes):
            rank = self.rank_list[i]   # the rank of sentence with label i
            if reverse:
                rank.reverse()
            for j in range(size):
                image, label = self.images[rank[j].id], self.labels[rank[j].id]
                images.append(image)
                labels.append(label)
        images = read_imgdata(images, transform=self.transformer)
        return self.train_loader(data_set=MyDataset(images, labels), batch=batch)

    def get_random_set(self, size=500, top=None, batch=32):
        if not self.labeled_set:
            self.get_labeled_set()
        images, labels = [], []
        for i in range(self.num_classes):
            random.shuffle(self.labeled_set[i])
            for j in range(size):
                images.append(self.labeled_set[i][j])
                labels.append(i)
        images = read_imgdata(images, transform=self.transformer)
        return self.train_loader(data_set=MyDataset(images, labels), batch=batch)


if __name__ == '__main__':
    data = ImagenetSet(1, 1)


class MyDataset(data.Dataset):
    def __init__(self, photo: list, label: list):
        self.pho = photo
        self.label = label

    def __getitem__(self, item):
        return self.pho[item], self.label[item]

    def __len__(self):
        return len(self.pho)


class Item:
    def __init__(self, _id, value):
        self.id = _id
        self.value = value


class MyDataset3(data.Dataset):
    def __init__(self, photo: list, label: list, tr: list):
        self.pho = photo
        self.label = label
        self.tr = tr

    def __getitem__(self, item):
        return self.pho[item], self.label[item], self.tr[item]

    def __len__(self):
        return len(self.pho)
