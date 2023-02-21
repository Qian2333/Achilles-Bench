import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
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
    # pic = pic.resize((256, 256), Image.BILINEAR)
    # pic = former(pic)
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


def read_imgdata(path):
    print('loading pic')
    bg = time.time()
    P = Pool(processes=cpu_count())
    picss = []
    pics = list(tqdm(P.map(func=get_sig, iterable=path)))
    pbar = tqdm(total=len(path))
    for pic in pics:
        # torch.tensor().detach()
        # picss.append(former(pic))
        picss.append(torch.tensor(former(pic), dtype=torch.float16))
        pbar.update(1)
    print(time.time() - bg)
    return picss


def read_imgdata2(path):
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
    pics = read_imgdata2(path)
    return MyDataset(pics, lab)


class ImageNet(data.Dataset):
    def __init__(self, size=224, load_test=True):
        """有关Cifar数据的一切操作，path是读取路径"""
        self.train_transform = transforms.Compose([
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop(32, 4),
                    # transforms.RandomResizedCrop(size),
                     transforms.Resize(256),
                     transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.test_transform = transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(size),
                         transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.train_pic = []
        self.train_lab = []
        with open('/mnt/cache/share/images/meta/train.txt') as f:
            while 1:
                s = f.readline()
                if not s:
                    break
                s = s.split(' ')
                self.train_pic.append('/mnt/cache/share/images/train/' + s[0])
                self.train_lab.append(int(s[1]))
        self.test_pic = []
        self.test_lab = []
        with open('/mnt/cache/share/images/meta/val.txt') as f:
            while 1:
                s = f.readline()
                if not s:
                    break
                s = s.split(' ')
                self.test_pic.append('/mnt/cache/share/images/val/' + s[0])
                self.test_lab.append(int(s[1]))
        self.train_set = Indata(self.train_pic, self.train_lab, transform=self.train_transform)
        self.test_set = None
        if load_test:
            self.get_test()
        self.tr = None
        self.n1 = 1000
        self.lab_n = [0 for _ in range(1000)]
        self.get_lab_n()
        self.div_class = None

    def get_tr_top_lis(self, size=10, lis=None, path='./logs/record1/tr_cifar_denesnet.txt', batch=256, loader=True):
        # if not self.tr:
        self.get_tr(path=path)
        print('get tr!')
        # self.get_class()
        inp = []
        lab = []
        con = 0
        for i in lis:
            pt = self.tr[i]
            for j in range(1, size + 1):
                pic = self.train_pic[pt[-j].id]
                inp.append(pic)
                lab.append(con)
                # ids.append(pt[-j].id)
            con += 1
        if loader:
            return self.train_loader(data_set=get_imgdata(inp, lab, transform=self.train_transform), batch=batch)
        else:
            return get_imgdata(inp, lab, transform=self.train_transform)

    def get_rnd_suf_lis(self, size=10, lis=None, batch=256, loader=True):
        # print('getting random')
        if not self.div_class:
            self.get_class()
        # print('div_class loaded')
        inp = []
        lab = []
        con = 0
        for i in lis:
            random.shuffle(self.div_class[i])
            for j in range(size):
                inp.append(self.div_class[i][j])
                lab.append(con)
            con += 1
        # print('random loaded')
        # return self.train_loader(data_set=MyDataset(inp, lab), batch=100), inp
        if loader:
            return self.train_loader(data_set=get_imgdata(inp, lab, transform=self.train_transform), batch=batch)
        else:
            return get_imgdata(inp, lab, transform=self.train_transform)

    def get_test_lis(self, mp, lis=None, batch=256):
        # print('getting random')
        # print('div_class loaded')
        inp = []
        lab = []
        for i in range(len(self.train_pic)):
            if self.train_lab[i] not in lis:
                continue
            inp.append(self.train_pic[i])
            lab.append(mp[self.train_lab[i]])
        # print('random loaded')
        # return self.train_loader(data_set=MyDataset(inp, lab), batch=100), inp
        return self.train_loader(data_set=get_imgdata(inp, lab, transform=self.train_transform), batch=batch)

    def get_test(self):
        self.test_set = get_imgdata(self.test_pic, self.test_lab)

    def total_train(self):
        return get_imgdata(self.train_pic, self.train_lab)

    def get_lab_n(self):
        for item in self.train_lab:
            self.lab_n[item] += 1

    def train_loader(self, data_set=None, batch=1, shuffle=True):
        if not data_set:
            return DataLoader(self.train_set, batch_size=batch, num_workers=8, shuffle=shuffle)
        return DataLoader(data_set, batch_size=batch, num_workers=8, shuffle=shuffle)

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
        self.div_class = [[] for _ in range(self.n1)]
        for i in range(len(self.train_pic)):
            pic, lab = self.train_pic[i], self.train_lab[i]
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
        print('getting tr')
        f = open(path)
        prl = [[] for i in range(self.n1)]
        con = 0
        # pbar = tqdm.tqdm(total=len(self.train_pic))
        while 1:
            s = f.readline()
            if not s:
                break
            lab = self.train_lab[con]
            prl[lab].append(Pic(con, float(s)))
            con += 1
            # pbar.update(1)

        # print()
        print('ranking')
        # print()
        # pbar = tqdm.tqdm(total=self.n1)
        for i in range(self.n1):
            prl[i].sort(key=lambda pic: pic.ntk)
            # pbar.update(1)
        self.tr = prl

    def get_tr_top(self, size=10, path='./logs/record1/tr_cifar_denesnet.txt', batch=256, loader=True):
        # if not self.tr:
        self.get_tr(path=path)
        print('get tr!')
        # self.get_class()
        inp = []
        lab = []
        ids = []
        for i in range(self.n1):
            pt = self.tr[i]
            for j in range(1, size + 1):
                pic, lan = self.train_pic[pt[-j].id], self.train_lab[pt[-j].id]
                inp.append(pic)
                lab.append(lan)
                ids.append(pt[-j].id)
        if loader:
            return self.train_loader(data_set=get_imgdata(inp, lab, transform=self.train_transform), batch=batch)
        else:
            return get_imgdata(inp, lab, transform=self.train_transform)

    def get_tr_suf(self, size=10, l=0, r=5000, path='./logs/record1/tr_cifar_denesnet.txt'):
        if not r:
            r = size
        # if not self.tr:
        self.get_tr(path=path)
        # self.get_class()
        inp = []
        lab = []
        ids = []
        for i in range(self.n1):
            pt = self.tr[i]
            random.shuffle(pt[l:r])
            for j in range(size):
                pic, lan = self.train_pic[pt[l + j].id], self.train_lab[pt[l + j].id]
                inp.append(pic)
                lab.append(lan)
                ids.append(pt[l + j].id)
        return self.train_loader(data_set=get_imgdata(inp, lab, transform=self.train_transform), batch=256)

    def get_rnd_suf(self, size=10, batch=256):
        # print('getting random')
        if not self.div_class:
            self.get_class()
        # print('div_class loaded')
        inp = []
        lab = []
        for i in range(self.n1):
            random.shuffle(self.div_class[i])
            for j in range(size):
                inp.append(self.div_class[i][j])
                lab.append(i)
        # print('random loaded')
        # return self.train_loader(data_set=MyDataset(inp, lab), batch=100), inp
        return self.train_loader(data_set=get_imgdata(inp, lab, transform=self.train_transform), batch=batch)

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




if __name__ == '__main__':
    data = ImageNet()


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
