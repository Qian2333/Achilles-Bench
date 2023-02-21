import time
import datetime
import os


def get_time():
    now_time = datetime.datetime.now()
    now_time = str(now_time)[8:19]
    s = now_time.split()
    tt = s[1].split(':')
    ttt = None
    for it in tt:
        if ttt:
            ttt = ttt + it
        else:
            ttt = it
    now_time = s[0] + '-' + ttt
    return now_time


def get_time_full():
    now_time = datetime.datetime.now()
    return str(now_time)[:19]

def str_pad(string1, length):
    if len(string1) >= length:
        return string1[:length]
    while len(string1) < length:
        string1 = string1 + ' '
    return string1

class Logger:
    def __init__(self, path='./logs/', name=None, tim=True):
        self.file_path = path
        self.name = get_time() + '.log'
        if name and tim:
            self.name = name + '-' + get_time() + '.log'
        elif name and not tim:
            self.name = name + '.log'
        if tim:
            path = path + 'run/'
        self.file_name = path + self.name

    def info(self, string):
        with open(self.file_name, 'a') as f:
            f.write(string + '                  ' + get_time_full() + '\n')

    def info1(self, string):
        with open(self.file_name, 'a') as f:
            f.write(string + '\n')

    def epoch_log1(self, epoch, loss, acc):
        epoch = str(epoch)
        while len(epoch) < 4:
            epoch = epoch + ' '
        self.info(str(epoch) + ' | acc: ' + str(round(float(acc), 2)) + ' |  loss: ' + str(float(loss)))

    def epoch_log2(self, epoch, train_acc, train_loss, test_acc, test_loss):
        epoch = str(epoch)
        while len(epoch) < 4:
            epoch = epoch + ' '
        self.info(epoch +
                  ' |train acc: ' + str_pad(str(round(float(train_acc), 2)), 6) +
                  ' |test acc: ' + str_pad(str(round(float(test_acc), 2)), 6) +
                  ' |train loss: ' + str_pad(str(float(train_loss)), 6) +
                  ' |test loss: ' + str_pad(str(float(test_loss)), 6))
