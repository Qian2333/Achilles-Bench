import torch
import numpy as np
import random
import torch.nn.init as init


def init_model(net):
    for par in list(net.parameters()):
        init.normal(par)
    return net


def cal_para(net: torch.nn.Module, rate=1, emb=1, op=0):
    if op:
        tot = 0
        for par in net.pre_fc.parameters():
            tmp = par.view(-1)
            # ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
            tot += int(tmp.shape[0] * rate)
        for par in net.fc.parameters():
            tmp = par.view(-1)
            # ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
            tot += int(tmp.shape[0] * rate)
        return tot
    params = list(net.parameters())
    ans = 0
    ctt = 0
    for i in params:
        if ctt == 0 and emb:
            ctt = 1
            continue
        if not i.requires_grad:
            continue
        tmp = i.view(-1)
        ans += int(tmp.shape[0] * rate)
    return ans


def get_gradient_tensor_ori(net: torch.nn.Module, opt, y, tot=0, rate=1, op=0):
    import copy
    opt.zero_grad()
    y.backward(retain_graph=True)
    if op == 1:
        if tot == 0:
            for par in net.pre_fc.parameters():
                tmp = copy.deepcopy(par.grad.view(-1))
                # ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
                tot += int(tmp.shape[0] * rate)
            for par in net.fc.parameters():
                tmp = copy.deepcopy(par.grad.view(-1))
                # ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
                tot += int(tmp.shape[0] * rate)
        params = list(net.parameters())
        cou, ctt = 0, 1
        # print(net.pre_fc.parameters().grad)
        ans = torch.tensor(np.zeros(tot))
        for par in net.pre_fc.parameters():
            tmp = copy.deepcopy(par.grad.view(-1))
            ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
            cou += int(tmp.shape[0] * rate)
        for par in net.fc.parameters():
            tmp = copy.deepcopy(par.grad.view(-1))
            ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
            cou += int(tmp.shape[0] * rate)
        return ans.cuda()
    # print(ans.shape)
    params = list(net.parameters())
    cou, ctt = 0, 1
    if tot == 0:
        for par in params:
            if par.grad is None:
                continue
            tmp = copy.deepcopy(par.grad.view(-1))
            # ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
            tot += int(tmp.shape[0] * rate)
    ans = torch.tensor(np.zeros(tot))
    # print(tot)
    # exit(0)
    for par in params:
        if ctt:
            ctt = 0
            continue
        # print(par.size(), end=' ')
        if par.grad is None:
            continue
        tmp = copy.deepcopy(par.grad.view(-1))
        if int(tmp.shape[0] * rate) == 0:
            continue
        # print(tmp.shape, int(tmp.shape[0] * rate), cou, ans[cou:(cou + int(tmp.shape[0] * rate))].shape)
        ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
        cou += int(tmp.shape[0] * rate)
    # exit(0)
    # print(sum(ans))
    # exit(0)
    return ans.cuda()


def read_ranking(path):
    f = open(path)
    count = 0
    ranking = []
    items = []
    while 1:
        s = f.readline()
        if not s:
            break
        s = s.split()
        if len(s) > 1:
            id = int(s[0])
            norm = float(s[1])
        else:
            id = count
            norm = float(s[0])
            count += 1
        items.append(Item(id, norm))
    items.sort(key=lambda item: item.id)
    for item in items:
        ranking.append(item.value)
    return ranking


class Item:
    def __init__(self, _id, value):
        self.id = _id
        self.value = value


def get_gradient_tr(net: torch.nn.Module, opt, y, args=None):
    import copy
    opt.zero_grad()
    # y.backward(retain_graph=True)
    y.backward()
    params = list(net.parameters())
    cou, ctt = 0, 1
    ans = torch.tensor(0.).cuda()
    # print(tot)
    # exit(0)
    for par in params:
        if ctt:
            ctt = 0
            continue
        # print(par.size(), end=' ')
        if par.grad is None:
            continue
        tmp = copy.deepcopy(par.grad.view(-1))
        ans += tmp.dot(tmp)
    # exit(0)
    # print(sum(ans))
    # exit(0)
    return ans.cuda()


def get_gradient_tensor(net: torch.nn.Module, opt, y, tot=0, rate=1, op=0):
    import copy
    opt.zero_grad()
    # y.backward(retain_graph=True)
    y.backward()
    if op == 1:
        if tot == 0:
            for par in net.pre_fc.parameters():
                tmp = copy.deepcopy(par.grad.view(-1))
                # ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
                tot += int(tmp.shape[0] * rate)
            for par in net.fc.parameters():
                tmp = copy.deepcopy(par.grad.view(-1))
                # ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
                tot += int(tmp.shape[0] * rate)
        params = list(net.parameters())
        cou, ctt = 0, 1
        # print(net.pre_fc.parameters().grad)
        ans = torch.tensor(np.zeros(tot))
        for par in net.pre_fc.parameters():
            tmp = copy.deepcopy(par.grad.view(-1))
            ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
            cou += int(tmp.shape[0] * rate)
        for par in net.fc.parameters():
            tmp = copy.deepcopy(par.grad.view(-1))
            ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
            cou += int(tmp.shape[0] * rate)
        return ans.cuda()
    # print(ans.shape)
    params = list(net.parameters())
    cou, ctt = 0, 1
    if tot == 0:
        for par in params:
            if par.grad is None:
                continue
            tmp = copy.deepcopy(par.grad.view(-1))
            # ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
            tot += int(tmp.shape[0] * rate)
    ans = torch.tensor(np.zeros(tot))
    # print(tot)
    # exit(0)
    for par in params:
        if ctt:
            ctt = 0
            continue
        # print(par.size(), end=' ')
        if par.grad is None:
            continue
        tmp = copy.deepcopy(par.grad.view(-1))
        if int(tmp.shape[0] * rate) == 0:
            continue
        # print(tmp.shape, int(tmp.shape[0] * rate), cou, ans[cou:(cou + int(tmp.shape[0] * rate))].shape)
        ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
        cou += int(tmp.shape[0] * rate)
    # exit(0)
    # print(sum(ans))
    # exit(0)
    return ans.cuda()


def multi_tensor_gra(l1, l2):
    ans = torch.tensor([1.])
    for i in range(len(l1)):
        ans += torch.sum(l1[i] * l2[i])
    return ans


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


def cal_para_num(net: torch.nn.Module, rate=1, num=0):
    params = list(net.named_parameters())
    ans = 0
    ctt = 0
    for names, par in params:
        name = names.split('.')
        if len(name) < 4 or name[3] != str(num):
            continue
        if ctt == 0:
            ctt = 1
            continue
        if not par.requires_grad:
            continue
        tmp = par.view(-1)
        ans += int(tmp.shape[0] * rate)
    return ans


def get_gradient_tensor_num(net: torch.nn.Module, opt, y, tot=0, rate=1, num=11):
    import copy
    opt.zero_grad()
    y.backward(retain_graph=True)
    # print(ans.shape)
    params = list(net.named_parameters())
    cou, ctt = 0, 0
    if tot == 0:
        for names, par in params:
            name = names.split('.')
            if len(name) < 4 or name[3] != str(num):
                continue
            if par.grad is None:
                continue
            tmp = copy.deepcopy(par.grad.view(-1))
            # ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
            tot += int(tmp.shape[0] * rate)
    ans = torch.tensor(np.zeros(tot))
    # print(tot)
    # print(tot)
    # exit(0)
    for names, par in params:
        name = names.split('.')
        if len(name) < 4 or name[3] != str(num):
            continue
        if ctt == 0:
            ctt = 1
            continue
        # print(par.size(), end=' ')
        if par.grad is None:
            continue
        tmp = copy.deepcopy(par.grad.view(-1))
        if int(tmp.shape[0] * rate) == 0:
            continue
        # print(tmp.shape, int(tmp.shape[0] * rate), cou, ans[cou:(cou + int(tmp.shape[0] * rate))].shape)
        ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
        cou += int(tmp.shape[0] * rate)
    # exit(0)
    # print(sum(ans))
    # exit(0)
    return ans.cuda()


def cal_para_att(net: torch.nn.Module, rate=1, nam='attention'):
    params = list(net.named_parameters())
    ans = 0
    ctt = 0
    for names, par in params:
        name = names.split('.')
        if len(name) < 5 or name[4] == str(nam):
            continue
        if ctt == 0:
            ctt = 1
            continue
        if not par.requires_grad:
            continue
        tmp = par.view(-1)
        ans += int(tmp.shape[0] * rate)
    return ans


def get_gradient_tensor_att(net: torch.nn.Module, opt, y, tot=0, rate=1, nam='attention'):
    import copy
    opt.zero_grad()
    y.backward(retain_graph=True)
    # print(ans.shape)
    params = list(net.named_parameters())
    cou, ctt = 0, 0
    ans = torch.tensor(np.zeros(tot))
    # print(tot)
    # print(tot)
    # exit(0)
    for names, par in params:
        name = names.split('.')
        if len(name) < 5 or name[4] == nam:
            continue
        if ctt == 0:
            ctt = 1
            continue
        # print(par.size(), end=' ')
        if par.grad is None:
            continue
        tmp = copy.deepcopy(par.grad.view(-1))
        if int(tmp.shape[0] * rate) == 0:
            continue
        # print(tmp.shape, int(tmp.shape[0] * rate), cou, ans[cou:(cou + int(tmp.shape[0] * rate))].shape)
        ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
        cou += int(tmp.shape[0] * rate)
    # exit(0)
    # print(sum(ans))
    # exit(0)
    return ans.cuda()


def cal_para_att2(net: torch.nn.Module, rate=1, nam='attention'):
    params = list(net.named_parameters())
    ans = 0
    ctt = 0
    for names, par in params:
        name = names.split('.')
        if len(name) > 5 and name[4] == str(nam):
            continue
        if ctt == 0:
            ctt = 1
            continue
        if not par.requires_grad:
            continue
        tmp = par.view(-1)
        ans += int(tmp.shape[0] * rate)
    return ans


def get_gradient_tensor_att2(net: torch.nn.Module, opt, y, tot=0, rate=1, nam='attention'):
    import copy
    opt.zero_grad()
    y.backward(retain_graph=True)
    # print(ans.shape)
    params = list(net.named_parameters())
    cou, ctt = 0, 0
    ans = torch.tensor(np.zeros(tot))
    # print(tot)
    # print(tot)
    # exit(0)
    for names, par in params:
        name = names.split('.')
        if len(name) > 5 and name[4] == str(nam):
            continue
        if ctt == 0:
            ctt = 1
            continue
        # print(par.size(), end=' ')
        if par.grad is None:
            continue
        tmp = copy.deepcopy(par.grad.view(-1))
        if int(tmp.shape[0] * rate) == 0:
            continue
        # print(tmp.shape, int(tmp.shape[0] * rate), cou, ans[cou:(cou + int(tmp.shape[0] * rate))].shape)
        ans[cou:(cou + int(tmp.shape[0] * rate))] = tmp[:int(tmp.shape[0] * rate)]
        cou += int(tmp.shape[0] * rate)
    # exit(0)
    # print(sum(ans))
    # exit(0)
    return ans.cuda()
