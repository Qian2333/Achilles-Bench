import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import torch
import time
import random


def random_shuffle(lis):
    random.seed(int(time.time()))
    for i in range(len(lis)):
        j = random.randint(0, len(lis) - 1)
        lis[i], lis[j] = lis[j], lis[i]
    return lis


class MyDataset(data.Dataset):
    def __init__(self, photo: list, label: list):
        self.pho = photo
        self.label = label

    def __getitem__(self, item):
        return self.pho[item], self.label[item]

    def __len__(self):
        return len(self.pho)

class MyDataset3(data.Dataset):
    def __init__(self, photo: list, label: list, tr: list):
        self.pho = photo
        self.label = label
        self.tr = tr

    def __getitem__(self, item):
        return self.pho[item], self.label[item], self.tr[item]

    def __len__(self):
        return len(self.pho)


class Pic:
    def __init__(self, _id, ntk):
        self.id = _id
        self.ntk = ntk


class Item:
    def __init__(self, _id, ntk):
        self.id = _id
        self.ntk = ntk


class MnistData:
    def __init__(self, path='./data'):
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.CenterCrop(32),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])

        self.train_set = torchvision.datasets.MNIST(
            root=path, train=True, download=False, transform=transform)
        self.test_set = torchvision.datasets.MNIST(
            root=path, train=False, download=False, transform=transform)
        self.div_class = None
        self.ntk = None
        self.tr = None
        self.tr_nt = None
        self.train_pic = None
        self.train_lab = None
        self.tot_rnk = None

    def train_loader(self, data_set=None, batch=1, shuffle=True):
        if not data_set:
            return DataLoader(self.train_set, batch_size=batch, shuffle=shuffle)
        return DataLoader(data_set, batch_size=batch, shuffle=shuffle)

    def train_mix(self, batch=1, shuffle=True, size=None):
        if not self.train_pic:
            self.get_lab_pic()
        if not self.tot_rnk:
            self.get_rnk()
        if not size:
            return DataLoader(MyDataset3(self.train_pic, self.train_lab, self.tot_rnk),
                              batch_size=batch, shuffle=shuffle)
        else:
            c = list(zip(self.train_pic, self.train_lab, self.tot_rnk))
            random.shuffle(c)
            self.train_pic, self.train_lab, self.tot_rnk = zip(*c)
            return DataLoader(MyDataset3(self.train_pic[:size], self.train_lab[:size], self.tot_rnk[:size]),
                              batch_size=batch, shuffle=shuffle)

    def get_single(self, data_set=None, item=0):
        if not data_set:
            pic, lab = self.train_set[item]
            return MyDataset([pic], [lab])

    def get_rnd_data(self, size=5):
        random.seed(time.time())
        if not self.div_class:
            self.get_class()
        ids = set()
        pic = []
        lab = []
        for i in range(10):
            for j in range(size):
                tmp = random.randint(0, len(self.div_class[i]) - 1)
                while tmp in ids:
                    tmp = random.randint(0, len(self.div_class[i]) - 1)
                ids.add(tmp)
                pic.append(self.div_class[i][tmp])
                lab.append(i)
        return MyDataset(pic, lab), ids

    def get_class(self):
        self.div_class = [[] for i in range(10)]
        for item in self.train_set:
            pic, lab = item
            self.div_class[lab].append(pic)

    def get_ntk(self, path='./logs/record1/ntk_cifar1.txt'):
        f = open(path)
        prl = [[] for i in range(10)]
        con = 0
        while 1:
            s = f.readline()
            if not s:
                break
            pic, lab = self.train_set[con]
            prl[lab].append(Pic(con, float(s)))
            con += 1
        for i in range(10):
            prl[i].sort(key=lambda pic: pic.ntk)
        self.ntk = prl

    def get_sorted_testdata(self, l=0, r=100, path='./logs/record1/tr_cifar_test_1.txt'):
        f = open(path)
        prl = []
        con = 0
        while 1:
            s = f.readline()
            if not s:
                break
            prl.append(Pic(con, float(s)))
            con += 1
        inn, lan = [], []
        prl.sort(key=lambda pic: pic.ntk)
        for i in range(l, r):
            pic, lab = self.test_set[prl[i].id]
            inn.append(pic)
            lan.append(lab)
        return MyDataset(inn, lan)

    def get_rnk(self, path='./logs/record1/tr_cifar1.txt'):
        f = open(path, 'r')
        lis = []
        while 1:
            s = f.readline()
            if not s:
                break
            lis.append(float(s))
        lis2 = []
        for i in range(len(lis)):
            lis2.append(Pic(i, lis[i]))
        lis2.sort(key=lambda pic: pic.ntk)
        lis3 = [0 for i in range(len(lis))]
        for i in range(len(lis)):
            lis3[lis2[i].id] = i
            self.tot_rnk = lis3

    def get_tr_nt(self, path='./logs/record1/tr_cifar3.txt'):
        f = open(path)
        prl = []
        con = 0
        while 1:
            s = f.readline()
            if not s:
                break
            prl.append(Pic(con, float(s)))
            con += 1
        prl.sort(key=lambda pic: pic.ntk)
        self.tr_nt = prl

    def get_tr(self, path='./logs/record1/tr_cifar_denesnet.txt'):
        f = open(path)
        prl = [[] for i in range(10)]
        con = 0
        while 1:
            s = f.readline()
            if not s:
                break
            pic, lab = self.train_set[con]
            prl[lab].append(Pic(con, float(s)))
            con += 1
        for i in range(10):
            prl[i].sort(key=lambda pic: pic.ntk)
        self.tr = prl

    def get_tr_suf(self, size=10, l=0, r=5000, path='./logs/record1/tr_cifar_denesnet.txt', loader=True):
        if not r:
            r = size
        # if not self.tr:
        self.get_tr(path=path)
        # self.get_class()
        inp = []
        lab = []
        ids = []
        for i in range(10):
            pt = self.tr[i]
            random.shuffle(pt[l:r])
            for j in range(size):
                pic, lan = self.train_set[pt[l + j].id]
                inp.append(pic)
                lab.append(lan)
                ids.append(pt[l + j].id)
        # ids.sort()
        # for i in range(len(ids) - 1):
        #     if ids[i] == ids[i + 1]:
        #         print('worng')
        #         exit(0)
        # print(len(ids))
        # exit(0)
        if loader:
            return self.train_loader(data_set=MyDataset(inp, lab), batch=256)
        else:
            return MyDataset(inp, lab)

    def get_rnd_suf(self, size=10, loader=True):
        if not self.div_class:
            self.get_class()
        inp = []
        lab = []
        for i in range(10):
            random.shuffle(self.div_class[i])
            for j in range(size):
                inp.append(self.div_class[i][j])
                lab.append(i)
        # return self.train_loader(data_set=MyDataset(inp, lab), batch=100), inp
        if loader:
            return self.train_loader(data_set=MyDataset(inp, lab), batch=128)
        else:
            return MyDataset(inp, lab)

    def get_dataset_noty(self, lam='tr', l=0, r=0, siz=10, seed=0):
        self.get_tr_nt()
        self.get_ntk()
        import time
        print(time.time())
        print(seed)
        random.seed(seed)
        inn, lan, ids = [], [], []
        tpk = r
        trsum = 0
        prl = self.tr_nt
        if lam == 'ntk':
            prl = self.ntk
        if r == -1:
            tpk = len(prl)
        print(l, r)
        for i in range(10):
            print(prl[i].id, end=' ')
        print()
        random.shuffle(prl[l:r])
        for i in range(10):
            print(prl[i].id, end=' ')
        print()
        for j in range(l, l + siz):
            trsum += prl[j].ntk
            tmp = prl[j].id
            ppc, lbb = self.train_set[tmp]
            ids.append(tmp)
            inn.append(ppc)
            lan.append(lbb)
        return MyDataset(inn, lan), ids, trsum

    def get_sta(self, l=0, r=100):
        if not self.tr_nt:
            self.get_tr_nt()
        cou = [0 for i in range(10)]
        for i in range(l, r):
            pic, lab = self.train_set[self.tr_nt[i].id]
            cou[lab] += 1
        return cou

    def get_dataset(self, lam='tr', l=0, r=0, t=0, siz=10):
        if not self.tr:
            self.get_tr()
        if not self.ntk:
            self.get_ntk()
        import time
        random.seed(time.time())
        inn, lan, ids = [], [], []
        tpk = r
        trsum = 0
        prl = self.tr
        if lam == 'ntk':
            prl = self.ntk
        for j in range(siz):
            if r == -1:
                tpk = len(prl[t])
            tmp = random.randint(l, tpk - 1)
            while prl[t][tmp].id in ids:
                tmp = random.randint(l, tpk - 1)
            trsum += prl[t][-tmp].ntk
            tmp = prl[t][-tmp].id
            ppc, lbb = self.train_set[tmp]
            ids.append(tmp)
            inn.append(ppc)
            lan.append(lbb)
        return MyDataset(inn, lan), ids, trsum

    def get_lab_pic(self):
        self.train_pic = []
        self.train_lab = []
        for pic, lab in self.train_set:
            self.train_pic.append(pic)
            self.train_lab.append(lab)


class CifarData:
    def __init__(self, path='./data', norm=True, size=32, aug=False):
        """有关Cifar数据的一切操作，path是读取路径"""
        if norm:
            if size == 32:
                train_transform = transforms.Compose([
                            # transforms.RandomHorizontalFlip(),
                            # transforms.RandomCrop(32, padding=4),
                            # transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                test_transform = transforms.Compose(
                            [transforms.ToTensor(),
                            # transforms.Resize(224),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            else:
                train_transform = transforms.Compose([
                            # transforms.RandomHorizontalFlip(),
                            # transforms.RandomCrop(32, padding=4),
                            transforms.Resize(size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
                test_transform = transforms.Compose(
                            [transforms.ToTensor(),
                            transforms.Resize(size),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            train_transform = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        # transforms.RandomCrop(32, 4),
                        # transforms.Resize(224),
                        transforms.ToTensor()])
            test_transform = transforms.Compose(
                        [transforms.ToTensor()])
        if aug:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        self.train_set = torchvision.datasets.CIFAR10(
            root=path, train=True, download=False, transform=train_transform)
        self.test_set = torchvision.datasets.CIFAR10(
            root=path, train=False, download=False, transform=test_transform)
        self.div_class = None
        self.ntk = None
        self.tr = None
        self.tr_nt = None
        self.train_pic = None
        self.train_lab = None
        self.tot_rnk = None

    def train_loader(self, data_set=None, batch=1, shuffle=True):
        if not data_set:
            return DataLoader(self.train_set, batch_size=batch, shuffle=shuffle)
        return DataLoader(data_set, batch_size=batch, shuffle=shuffle)

    def train_mix(self, batch=1, shuffle=True, size=None):
        if not self.train_pic:
            self.get_lab_pic()
        if not self.tot_rnk:
            self.get_rnk()
        if not size:
            return DataLoader(MyDataset3(self.train_pic, self.train_lab, self.tot_rnk),
                              batch_size=batch, shuffle=shuffle)
        else:
            c = list(zip(self.train_pic, self.train_lab, self.tot_rnk))
            random.shuffle(c)
            self.train_pic, self.train_lab, self.tot_rnk = zip(*c)
            return DataLoader(MyDataset3(self.train_pic[:size], self.train_lab[:size], self.tot_rnk[:size]),
                              batch_size=batch, shuffle=shuffle)

    def get_single(self, data_set=None, item=0):
        if not data_set:
            pic, lab = self.train_set[item]
            return MyDataset([pic], [lab])

    def get_rnd_data(self, size=5):
        random.seed(time.time())
        if not self.div_class:
            self.get_class()
        ids = set()
        pic = []
        lab = []
        for i in range(10):
            for j in range(size):
                tmp = random.randint(0, len(self.div_class[i]) - 1)
                while tmp in ids:
                    tmp = random.randint(0, len(self.div_class[i]) - 1)
                ids.add(tmp)
                pic.append(self.div_class[i][tmp])
                lab.append(i)
        return MyDataset(pic, lab), ids

    def get_class(self):
        self.div_class = [[] for i in range(10)]
        for item in self.train_set:
            pic, lab = item
            self.div_class[lab].append(pic)

    def get_ntk(self, path='./logs/record1/ntk_cifar1.txt'):
        f = open(path)
        prl = [[] for i in range(10)]
        con = 0
        while 1:
            s = f.readline()
            if not s:
                break
            pic, lab = self.train_set[con]
            prl[lab].append(Pic(con, float(s)))
            con += 1
        for i in range(10):
            prl[i].sort(key=lambda pic: pic.ntk)
        self.ntk = prl

    def get_sorted_testdata(self, l=0, r=100, path='./logs/record1/tr_cifar_test_1.txt'):
        f = open(path)
        prl = []
        con = 0
        while 1:
            s = f.readline()
            if not s:
                break
            prl.append(Pic(con, float(s)))
            con += 1
        inn, lan = [], []
        prl.sort(key=lambda pic: pic.ntk)
        for i in range(l, r):
            pic, lab = self.test_set[prl[i].id]
            inn.append(pic)
            lan.append(lab)
        return MyDataset(inn, lan)

    def get_rnk(self, path='./logs/record1/tr_cifar1.txt'):
        f = open(path, 'r')
        lis = []
        while 1:
            s = f.readline()
            if not s:
                break
            lis.append(float(s))
        lis2 = []
        for i in range(len(lis)):
            lis2.append(Pic(i, lis[i]))
        lis2.sort(key=lambda pic: pic.ntk)
        lis3 = [0 for i in range(len(lis))]
        for i in range(len(lis)):
            lis3[lis2[i].id] = i
            self.tot_rnk = lis3

    def get_tr_nt(self, path='./logs/record1/tr_cifar3.txt'):
        f = open(path)
        prl = []
        con = 0
        while 1:
            s = f.readline()
            if not s:
                break
            prl.append(Pic(con, float(s)))
            con += 1
        prl.sort(key=lambda pic: pic.ntk)
        self.tr_nt = prl

    def get_tr(self, path='./logs/record1/tr_sst_distilbert-base-uncased.txt', siz=None):
        f = open(path)
        rank_list = [[] for i in range(10)]
        count = 0
        while 1:
            s = f.readline()
            if not s:
                break
            s = s.split()
            if len(s) > 1:
                id = int(s[0])
                norm = float(s[1])
                pic, label = self.train_set[id]
                rank_list[label].append(Item(id, norm))
            else:
                norm = float(s[0])
                pic, label = self.train_set[count]
                rank_list[label].append(Item(count, norm))
                count += 1
        for i in range(10):
            rank_list[i].sort(key=lambda item: item.ntk)
        self.tr = rank_list

    def get_tr_suf_show(self, size=10, l=0, r=5000, path='./logs/record1/tr_cifar_denesnet.txt', loader=True, batch=128):
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)

        if not r:
            r = size
        # if not self.tr:
        self.get_tr(path=path)
        # self.get_class()
        inp = []
        lab = []
        ids = []
        for i in range(10):
            pt = self.tr[i]
            # pt[l:r] = random_shuffle(pt[l:r])
            for j in range(size):
                pic, lan = train_set[pt[l + j].id]
                inp.append(pic)
                lab.append(lan)
                ids.append(pt[l + j].id)
        # ids.sort()
        # for i in range(len(ids) - 1):
        #     if ids[i] == ids[i + 1]:
        #         print('worng')
        #         exit(0)
        # print(len(ids))
        # exit(0)
        if loader:
            return self.train_loader(data_set=MyDataset(inp, lab), batch=batch)
        else:
            return MyDataset(inp, lab)

    def get_class1(self, train_set):
        div_class = [[] for i in range(10)]
        for item in train_set:
            pic, lab = item
            div_class[lab].append(pic)
        return div_class

    def get_rnd_suf_show(self, size=10, loader=True, batch=128):
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
        div_class = self.get_class1(train_set)

        inp, lab = [], []
        for i in range(10):
            div_class[i] = random_shuffle(div_class[i])
            for j in range(size):
                inp.append(div_class[i][j])
                lab.append(i)
        # return self.train_loader(data_set=MyDataset(inp, lab), batch=100), inp
        if loader:
            return self.train_loader(data_set=MyDataset(inp, lab), batch=batch)
        else:
            return MyDataset(inp, lab)

    def get_test_show(self, size=10, loader=True, batch=128):
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=train_transform)

        inp, lab = [], []
        for pic, lan in train_set:
            inp.append(pic)
        return

    def get_tr_suf_tt(self, size=10, l=0, r=5000, path='./logs/record1/tr_cifar_denesnet.txt', loader=True, batch=100):
        if not r:
            r = size
        # if not self.tr:
        self.get_tr(path=path)
        # self.get_class()
        inp, inp1 = [], []
        lab, lab1 = [], []
        ids = []
        for i in range(10):
            pt = self.tr[i]
            pt[l:r] = random_shuffle(pt[l:r])
            for j in range(size):
                pic, lan = self.train_set[pt[l + j].id]
                inp.append(pic)
                lab.append(lan)
                ids.append(pt[l + j].id)
            for j in range(0, l):
                pic, lan = self.train_set[pt[j].id]
                inp1.append(pic)
                lab1.append(lan)
            for j in range(r, len(pt)):
                pic, lan = self.train_set[pt[j].id]
                inp1.append(pic)
                lab1.append(lan)
                # ids.append(pt[l + j].id)
        # ids.sort()
        # for i in range(len(ids) - 1):
        #     if ids[i] == ids[i + 1]:
        #         print('worng')
        #         exit(0)
        # print(len(ids))
        # exit(0)
        if loader:
            return self.train_loader(data_set=MyDataset(inp, lab), batch=batch), \
                   self.train_loader(data_set=MyDataset(inp1, lab1), batch=batch)
        else:
            return MyDataset(inp, lab), MyDataset(inp1, lab1)


    def get_rnd_suf_tt(self, size=10, loader=True, batch=128):
        if not self.div_class:
            self.get_class()
        inp, inp1 = [], []
        lab, lab1 = [], []
        for i in range(10):
            self.div_class[i] = random_shuffle(self.div_class[i])
            for j in range(size):
                inp.append(self.div_class[i][j])
                lab.append(i)
            for j in range(size, len(self.div_class)):
                inp1.append(self.div_class[i][j])
                lab1.append(i)
        # return self.train_loader(data_set=MyDataset(inp, lab), batch=100), inp
        if loader:
            return self.train_loader(data_set=MyDataset(inp, lab), batch=batch), \
                   self.train_loader(data_set=MyDataset(inp1, lab1), batch=batch)
        else:
            return MyDataset(inp, lab), MyDataset(inp1, lab1)

    def get_tr_suf(self, size=10, l=0, r=5000, path='./logs/record1/tr_cifar_denesnet.txt', loader=True, batch=100):
        if not r:
            r = size
        # if not self.tr:
        self.get_tr(path=path)
        # self.get_class()
        inp = []
        lab = []
        ids = []
        for i in range(10):
            pt = self.tr[i]
            pt[l:r] = random_shuffle(pt[l:r])
            for j in range(size):
                pic, lan = self.train_set[pt[l + j].id]
                inp.append(pic)
                lab.append(lan)
                ids.append(pt[l + j].id)
        # ids.sort()
        # for i in range(len(ids) - 1):
        #     if ids[i] == ids[i + 1]:
        #         print('worng')
        #         exit(0)
        # print(len(ids))
        # exit(0)
        if loader:
            return self.train_loader(data_set=MyDataset(inp, lab), batch=batch)
        else:
            return MyDataset(inp, lab)


    def get_rnd_suf(self, size=10, loader=True, batch=128):
        if not self.div_class:
            self.get_class()
        inp, lab = [], []
        for i in range(10):
            self.div_class[i] = random_shuffle(self.div_class[i])
            for j in range(size):
                inp.append(self.div_class[i][j])
                lab.append(i)
        # return self.train_loader(data_set=MyDataset(inp, lab), batch=100), inp
        if loader:
            return self.train_loader(data_set=MyDataset(inp, lab), batch=batch)
        else:
            return MyDataset(inp, lab)

    def get_tr_suf_d(self, size=10, l=0, r=5000, path='./logs/record1/tr_cifar_denesnet.txt'):
        if not r:
            r = size
        # if not self.tr:
        self.get_tr(path=path)
        # self.get_class()
        inp = []
        lab = []
        ids, ids2 = [], []
        for i in range(10):
            pt = self.tr[i]
            pt[l:r] = random_shuffle(pt[l:r])
            for j in range(size):
                ids.append(pt[l + j].id)
            for j in range(0, l):
                ids2.append(pt[j].id)
            for j in range(size, 5000 - l):
                ids2.append(pt[l + j].id)
        return ids, ids2

    def get_rnd_suf_d(self, size=10):
        pt = self.get_class_id()
        inp, inp1 = [], []
        for i in range(10):
            pt[i] = random_shuffle(pt[i])
            for j in range(size):
                inp.append(pt[i][j])
            for j in range(size, 5000):
                inp1.append(pt[i][j])
        return inp, inp1

    def get_class_id(self):
        ids = [[] for _ in range(10)]
        for i in range(50000):
            _, lab = self.train_set[i]
            ids[lab].append(i)
        return ids

    def get_dataset_noty(self, lam='tr', l=0, r=0, siz=10, seed=0):
        self.get_tr_nt()
        self.get_ntk()
        import time
        print(time.time())
        print(seed)
        random.seed(seed)
        inn, lan, ids = [], [], []
        tpk = r
        trsum = 0
        prl = self.tr_nt
        if lam == 'ntk':
            prl = self.ntk
        if r == -1:
            tpk = len(prl)
        print(l, r)
        for i in range(10):
            print(prl[i].id, end=' ')
        print()
        random.shuffle(prl[l:r])
        for i in range(10):
            print(prl[i].id, end=' ')
        print()
        for j in range(l, l + siz):
            trsum += prl[j].ntk
            tmp = prl[j].id
            ppc, lbb = self.train_set[tmp]
            ids.append(tmp)
            inn.append(ppc)
            lan.append(lbb)
        return MyDataset(inn, lan), ids, trsum

    def get_sta(self, l=0, r=100):
        if not self.tr_nt:
            self.get_tr_nt()
        cou = [0 for i in range(10)]
        for i in range(l, r):
            pic, lab = self.train_set[self.tr_nt[i].id]
            cou[lab] += 1
        return cou

    def get_dataset(self, lam='tr', l=0, r=0, t=0, siz=10):
        if not self.tr:
            self.get_tr()
        if not self.ntk:
            self.get_ntk()
        import time
        random.seed(time.time())
        inn, lan, ids = [], [], []
        tpk = r
        trsum = 0
        prl = self.tr
        if lam == 'ntk':
            prl = self.ntk
        for j in range(siz):
            if r == -1:
                tpk = len(prl[t])
            tmp = random.randint(l, tpk - 1)
            while prl[t][tmp].id in ids:
                tmp = random.randint(l, tpk - 1)
            trsum += prl[t][-tmp].ntk
            tmp = prl[t][-tmp].id
            ppc, lbb = self.train_set[tmp]
            ids.append(tmp)
            inn.append(ppc)
            lan.append(lbb)
        return MyDataset(inn, lan), ids, trsum

    def get_lab_pic(self):
        self.train_pic = []
        self.train_lab = []
        for pic, lab in self.train_set:
            self.train_pic.append(pic)
            self.train_lab.append(lab)


class Cifar100:
    def __init__(self, path='./data', size=32, pro=None):
        """有关Cifar数据的一切操作，path是读取路径"""
        if size == 224:
            # print(1)
            train_transform = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        # transforms.RandomCrop(32, 4),
                        transforms.Resize(224),
                        # transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            test_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Resize(224),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        elif size == 384:
            train_transform = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        # transforms.RandomCrop(32, 4),
                        transforms.Resize(384),
                        # transforms.CenterCrop(384),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            test_transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Resize(384),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        else:
            train_transform = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        # transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            test_transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        if pro:
            train_transform = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        # transforms.RandomCrop(32, 4),
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        # transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            test_transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.train_set = torchvision.datasets.CIFAR100(
            root=path, train=True, download=True, transform=train_transform)
        # for pic, lab in self.train_set:
        #     print(pic.shape)
        #     exit(0)
        self.test_set = torchvision.datasets.CIFAR100(
            root=path, train=False, download=True, transform=test_transform)
        self.div_class = None
        self.ntk = None
        self.tr = None
        self.tr_nt = None
        self.train_pic = None
        self.train_lab = None
        self.tot_rnk = None

    def train_loader(self, data_set=None, batch=1, shuffle=True):
        if not data_set:
            return DataLoader(self.train_set, batch_size=batch, shuffle=shuffle)
        return DataLoader(data_set, batch_size=batch, shuffle=shuffle)

    def train_mix(self, batch=1, shuffle=True):
        if not self.train_pic:
            self.get_lab_pic()
        if not self.tot_rnk:
            self.get_rnk()
        return DataLoader(MyDataset3(self.train_pic, self.train_lab, self.tot_rnk), batch_size=batch, shuffle=shuffle)

    def get_single(self, data_set=None, item=0):
        if not data_set:
            pic, lab = self.train_set[item]
            return MyDataset([pic], [lab])

    def get_rnd_data(self, size=5):
        random.seed(time.time())
        if not self.div_class:
            self.get_class()
        ids = set()
        pic = []
        lab = []
        for i in range(100):
            for j in range(size):
                tmp = random.randint(0, len(self.div_class[i]) - 1)
                while tmp in ids:
                    tmp = random.randint(0, len(self.div_class[i]) - 1)
                ids.add(tmp)
                pic.append(self.div_class[i][tmp])
                lab.append(i)
        return MyDataset(pic, lab), ids

    def get_class(self):
        self.div_class = [[] for i in range(100)]
        pbar = tqdm.tqdm(total=50000)
        for item in self.train_set:
            pic, lab = item
            self.div_class[lab].append(pic)
            pbar.update(1)
        print()

    def get_ntk(self, path='./logs/record1/ntk_cifar1.txt'):
        f = open(path)
        prl = [[] for i in range(10)]
        con = 0
        while 1:
            s = f.readline()
            if not s:
                break
            pic, lab = self.train_set[con]
            prl[lab].append(Pic(con, float(s)))
            con += 1
        for i in range(100):
            prl[i].sort(key=lambda pic: pic.ntk)
        self.ntk = prl

    def get_rnk(self, path='./logs/record1/tr_cifar1.txt'):
        f = open(path, 'r')
        lis = []
        while 1:
            s = f.readline()
            if not s:
                break
            lis.append(float(s))
        lis2 = []
        for i in range(len(lis)):
            lis2.append(Pic(i, lis[i]))
        lis2.sort(key=lambda pic: pic.ntk)
        lis3 = [0 for i in range(len(lis))]
        for i in range(len(lis)):
            lis3[lis2[i].id] = i
            self.tot_rnk = lis3

    def get_tr_nt(self, path='./logs/record1/tr_cifar1.txt'):
        f = open(path)
        prl = []
        con = 0
        while 1:
            s = f.readline()
            if not s:
                break
            prl.append(Pic(con, float(s)))
            con += 1
        prl.sort(key=lambda pic: pic.ntk)
        self.tr_nt = prl

    def get_tr(self, path='./logs/record1/tr_cifar1.txt'):
        f = open(path)
        prl = [[] for i in range(100)]
        # print('loading tr')

        pbar = tqdm.tqdm(total=50000)
        con = 0
        while 1:
            s = f.readline()
            if not s:
                break
            pic, lab = self.train_set[con]
            prl[lab].append(Pic(con, float(s)))
            con += 1
            pbar.update(1)
            # print(con)
        print()
        for i in range(100):
            prl[i].sort(key=lambda pic: pic.ntk)
        self.tr = prl

    def get_tr_suf(self, size=10, l=0, r=500, path='./logs/record1/tr_cifar_denesnet.txt', batch=128, loader=True):
        if not r:
            r = size
        # if not self.tr:
        self.get_tr(path=path)
        # self.get_class()
        # print('loading pic')
        inp = []
        lab = []
        ids = []
        for i in range(100):
            pt = self.tr[i]
            random.shuffle(pt[l:r])
            for j in range(size):
                pic, lan = self.train_set[pt[l + j].id]
                inp.append(pic)
                lab.append(lan)
                ids.append(pt[l + j].id)
        if loader:
            return self.train_loader(data_set=MyDataset(inp, lab), batch=batch)
        else:
            return MyDataset(inp, lab)

    def get_rnd_suf(self, size=10, batch=128):
        if not self.div_class:
            # print('getting class')
            self.get_class()
        inp = []
        lab = []
        for i in range(100):
            print(i)
            random.shuffle(self.div_class[i])
            for j in range(size):
                inp.append(self.div_class[i][j])
                lab.append(i)
        return self.train_loader(data_set=MyDataset(inp, lab), batch=batch)

    def get_dataset_noty(self, lam='tr', l=0, r=0, siz=10):
        if not self.tr:
            self.get_tr_nt()
        if not self.ntk:
            self.get_ntk()
        import time
        random.seed(time.time())
        inn, lan, ids = [], [], []
        tpk = r
        trsum = 0
        prl = self.tr_nt
        if lam == 'ntk':
            prl = self.ntk
        for j in range(siz):
            if r == -1:
                tpk = len(prl)
            tmp = random.randint(l, tpk - 1)
            while prl[tmp].id in ids:
                tmp = random.randint(0, tpk - 1)
            trsum += prl[-tmp].ntk
            tmp = prl[-tmp].id
            ppc, lbb = self.train_set[tmp]
            ids.append(tmp)
            inn.append(ppc)
            lan.append(lbb)
        return MyDataset(inn, lan), ids, trsum

    def get_sta(self, l=0, r=100):
        if not self.tr_nt:
            self.get_tr_nt()
        cou = [0 for i in range(10)]
        for i in range(l, r):
            pic, lab = self.train_set[self.tr_nt[i].id]
            cou[lab] += 1
        return cou

    def get_dataset(self, lam='tr', l=0, r=0, t=0, siz=10):
        if not self.tr:
            self.get_tr()
        if not self.ntk:
            self.get_ntk()
        import time
        random.seed(time.time())
        inn, lan, ids = [], [], []
        tpk = r
        trsum = 0
        prl = self.tr
        if lam == 'ntk':
            prl = self.ntk
        for j in range(siz):
            if r == -1:
                tpk = len(prl[t])
            tmp = random.randint(l,
                                 - 1)
            while prl[t][tmp].id in ids:
                tmp = random.randint(0, tpk - 1)
            trsum += prl[t][-tmp].ntk
            tmp = prl[t][-tmp].id
            ppc, lbb = self.train_set[tmp]
            ids.append(tmp)
            inn.append(ppc)
            lan.append(lbb)
        return MyDataset(inn, lan), ids, trsum

    def get_lab_pic(self):
        self.train_pic = []
        self.train_lab = []
        for pic, lab in self.train_set:
            self.train_pic.append(pic)
            self.train_lab.append(lab)

if __name__ == '__main__':
    pass
