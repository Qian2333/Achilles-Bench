import torch
import numpy as np
import random


def cal_para(net: torch.nn.Module, rat=1):
    params = list(net.parameters())
    ans = 0
    for i in params:
        k = 1
        for j in i.size():
            k *= j
        ans += int(k * rat)
    return ans


def cal_para_rat(net: torch.nn.Module, rat=1):
    import copy
    params = list(net.parameters())
    cou = 0
    for par in params:
        if par.grad is None:
            continue
        tmp = copy.deepcopy(par.grad.view(-1))
        cou += int(tmp.shape[0] * rat)
    return cou


def read_tr(path: str, lim=None):
    tr = []
    con = 0
    with open(path, 'r') as f:
        while 1:
            con += 1
            if lim and con > lim:
                break
            s = f.readline()
            if not s:
                break
            tr.append(float(s))
    return tr


def r_mean(a, rnd=5):
    return round(np.mean(a), rnd)


def r_std(a, rnd=5):
    return round(np.std(a), rnd)


def get_gradient_tensor_rat(net: torch.nn.Module, opt, y, tot=0, rat=1):
    import copy
    if tot == 0:
        tot = cal_para(net)
    ans = torch.tensor(np.zeros(int(tot * rat))).cuda()
    opt.zero_grad()
    y.backward(retain_graph=True)
    params = list(net.parameters())
    cou = 0
    for par in params:
        if par.grad is None:
            continue
        tmp = copy.deepcopy(par.grad.view(-1))
        ans[cou:(cou + int(tmp.shape[0] * rat))] = tmp[:int(tmp.shape[0] * rat)]
        cou += int(tmp.shape[0] * rat)
    return ans


def get_gradient_tensor(net: torch.nn.Module, opt, y, tot=0):
    import copy
    if tot == 0:
        tot = cal_para(net)
    ans = torch.tensor(np.zeros(tot))
    opt.zero_grad()
    y.backward(retain_graph=True)
    params = list(net.parameters())
    cou = 0
    for par in params:
        if par.grad is None:
            continue
        tmp = copy.deepcopy(par.grad.view(-1))
        ans[cou:(cou + tmp.shape[0])] = tmp
        cou += tmp.shape[0]
    return ans.cuda()


def get_gradient_norm(net: torch.nn.Module, opt, y):
    import copy
    ans = torch.tensor(0.)
    opt.zero_grad()
    y.backward(retain_graph=True)
    params = list(net.parameters())
    for par in params:
        if par.grad is None:
            continue
        tmp = copy.deepcopy(par.grad.view(-1))
        ans += tmp.dot(tmp).cpu()
    return ans


def multi_tensor_gra(l1, l2):
    ans = torch.tensor([1.])
    for i in range(len(l1)):
        ans += torch.sum(l1[i] * l2[i])
    return ans


def get_my_dataset_rnd(set1, rating):
    from lib.dataset.myset import MyDataset
    import random
    datas = []
    photos = []
    labels = []
    for data in set1:
        datas.append(data)
    random.shuffle(datas)
    for i in range(int(len(datas) * rating)):
        ph, la = datas[i]
        photos.append(ph)
        labels.append(la)
    return MyDataset(photos, labels)


def get_my_dataset(set1, rate, threshold):
    from lib.dataset.myset import MyDataset
    photos = []
    labels = []
    for i, data in enumerate(set1):
        if rate[i] > threshold:
            photo, label = data
            photos.append(photo)
            labels.append(label)
    return MyDataset(photos, labels)


def process_txt_log(path, threshold):
    with open(path, 'r') as f:
        inp = f.readlines()
        rate = []
        mp = {}
        for i, st in enumerate(inp):
            st1 = st.split()
            ind = float(st1[1])
            rate.append(ind)
            mp[i] = ind
        rate = sorted(rate)
        return mp, rate[int(threshold * len(rate))]


def show_img(img):
    from PIL import Image
    pil_img = Image.fromarray(np.uint8(img * 256))
    pil_img.show()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
