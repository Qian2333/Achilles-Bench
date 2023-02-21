from lib.dataset.get_imagenet import ImagenetSet
from PIL import Image
from multiprocessing import Pool, cpu_count
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
from lib.dataset.get_data import MyDataset
import time


def pil_loader(path: str):
    pic = Image.open(path)
    return pic.convert('RGB')


def get_sig(path):
    pic = pil_loader(path)
    # pic = pic.resize((256, 256), Image.BILINEAR)
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


def read_imgdata(path):
    print('loading images')
    bg = time.time()
    P = Pool(processes=cpu_count())
    picss = []
    pics = list(tqdm(P.imap(func=get_sig, iterable=path), total=len(path)))
    print('loaded images')
    print('transform images')
    pbar = tqdm(total=len(path))
    for pic in pics:
        # torch.tensor().detach()
        # picss.append(former(pic))
        picss.append(torch.tensor(former(pic)))
        pbar.update(1)
    print(time.time() - bg)
    return picss


def train_imagenet_step(train_data: ImagenetSet, test_data: ImagenetSet,
                        net, optimizer, criterion, one_step_num, epoch_num, batch_size):
    for epoch in range(epoch_num):
        net.train()
        pbar = tqdm(total=(len(train_data.images) // batch_size))
        train_acc, train_loss = 0, 0
        for i in range(0, len(train_data.images), one_step_num):
            train_set_images = train_data.images[i:(i + one_step_num)]
            train_set_labels = train_data.labels[i:(i + one_step_num)]
            train_set_images = read_imgdata(train_set_images)
            train_set = MyDataset(train_set_images, train_set_labels)
            train_loader = train_data.train_loader(train_set, batch=batch_size, shuffle=True)
            for items in train_loader:
                inputs, labels = items
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                predicted = torch.max(outputs, 1)[1].data.cpu().numpy()

                acc_now = (predicted == labels.data.cpu().numpy()).sum()
                train_acc += acc_now
                train_loss += float(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)
                pbar.set_description("right samples in {} batch {}".format(batch_size, acc_now))
                pbar.refresh()

        net.eval()
        pbar = tqdm(total=(len(test_data.images) // batch_size))
        test_acc, test_loss, items_count = 0, 0, 0
        for i in range(0, len(test_data.images), one_step_num):
            test_set_images = test_data.images[i:(i + one_step_num)]
            test_set_labels = test_data.labels[i:(i + one_step_num)]
            test_set_images = read_imgdata(test_set_images)
            test_set = MyDataset(test_set_images, test_set_labels)
            test_loader = test_data.train_loader(test_set, batch=batch_size, shuffle=False)
            for items in test_loader:
                inputs, labels = items
                items_count += batch_size
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = net(inputs)
                test_loss += float(criterion(outputs, labels))
                predicted = torch.max(outputs, 1)[1].data.cpu().numpy()
                test_acc += (predicted == labels.data.cpu().numpy()).sum()
                pbar.update(1)
                pbar.set_description("acc now {}".format(round(test_acc / items_count, 2)))
                pbar.refresh()
        print('epoch : %d  ' % epoch, end='')
        print('train acc : %.1f ' % round(train_acc / len(train_data.images) * 100, 2), end='')
        print('test acc : %.1f ' % round(test_acc / len(test_data.images) * 100, 2))

