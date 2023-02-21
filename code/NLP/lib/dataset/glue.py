import csv
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import random
from datasets import load_dataset
import torch.nn.functional as F
from lib.model.BCF import Tokenizer, BertClassificationModel
from tqdm import tqdm
import os
import numpy as np


class DataIOGLUE(object):
    def __init__(self, name='sst2'):
        self.name = name
        self.train_word, self.train_label, \
        self.dev_word, self.dev_label, \
        self.test_word, \
        self.test_label = self.read_train_dev_test(name)
        self.num_classes = self.get_classes()

    def get_classes(self):
        if self.name == 'stsb':
            # for i in range(len(self.train_label)):
            #     self.train_label[i] = float(self.train_label[i])
            # for i in range(len(self.dev_label)):
            #     self.train_label[i] = float(self.dev_label[i])
            # for i in range(len(self.test_label)):
            #     self.train_label[i] = float(self.test_label[i])
            return 1

        classes = []
        for item in self.train_label:
            if item in classes:
                continue
            else:
                classes.append(item)
        return len(classes)

    def read_train_dev_test(self, nam):
        if nam == 'mnli':
            dataset = load_dataset('glue', 'mnli')
            train_word, train_label = self.get_data_from_dataset(dataset, 'train'), dataset['train']['label']
            dev_word, dev_label = self.get_data_from_dataset(dataset, 'validation_matched'), \
                                  dataset['validation_matched']['label']
            test_word, test_label = self.get_data_from_dataset(dataset, 'test_matched'), \
                                    dataset['test_matched']['label']
            return train_word, train_label, dev_word, dev_label, test_word, test_label
        elif nam == 'mnli_mismatched':
            dataset = load_dataset('glue', 'mnli')
            train_word, train_label = self.get_data_from_dataset(dataset, 'train'), dataset['train']['label']
            dev_word, dev_label = self.get_data_from_dataset(dataset, 'validation_mismatched'), \
                                  dataset['validation_mismatched']['label']
            test_word, test_label = self.get_data_from_dataset(dataset, 'test_mismatched'), \
                                    dataset['test_mismatched']['label']
            return train_word, train_label, dev_word, dev_label, test_word, test_label
        dataset = load_dataset('glue', nam)
        train_word, train_label = self.get_data_from_dataset(dataset, 'train'), dataset['train']['label']
        dev_word, dev_label = self.get_data_from_dataset(dataset, 'validation'), dataset['validation']['label']
        test_word, test_label = self.get_data_from_dataset(dataset, 'test'), dataset['test']['label']
        return train_word, train_label, dev_word, dev_label, test_word, test_label

    def get_data_from_dataset(self, dataset, name):
        sentences = []
        sentences_list = []
        for key in dataset[name].features.keys():
            sentences_list.append(dataset[name][key])
        for i in range(len(sentences_list[0])):
            sentences_pair = []
            for j in range(len(sentences_list) - 2):
                sentences_pair.append(sentences_list[j][i])
            sentences.append(sentences_pair)
        return sentences


class GLUEDataset_ori(Dataset):
    def __init__(self, sentences, labels, num_classes=2):
        super(GLUEDataset_ori, self).__init__()
        self.sentences = sentences
        self.labels = labels
        self.set = MyDataset(self.sentences, self.labels)
        self.labeled_set = None

        self.rank_list = None
        self.num_classes = num_classes

    def __getitem__(self, item):
        return self.sentences[item], self.labels[item]

    def get_labeled_set(self):
        if self.num_classes == 1:
            self.labeled_set = [[]]
            for i in range(len(self.sentences)):
                self.labeled_set[0].append([self.sentences[i], self.labels[i]])
            return
        sentences = [[] for i in range(self.num_classes)]
        for i in range(len(self.sentences)):
            sentences[self.labels[i]].append(self.sentences[i])
        self.labeled_set = sentences

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
        st = set()
        while 1:
            s = f.readline()
            if not s:
                break
            s = s.split()
            if len(s) > 1:
                id = int(s[0])
                norm = float(s[1])
                if id in st:
                    continue
                sentences, label = self.sentences[id], self.labels[id]
                if self.num_classes == 1:
                    rank_list[0].append(Item(id, norm))
                    continue
                rank_list[label].append(Item(id, norm))
                st.add(id)
            else:
                norm = float(s[0])
                sentences, label = self.sentences[count], self.labels[count]
                rank_list[label].append(Item(count, norm))
                count += 1
        for i in range(self.num_classes):
            rank_list[i].sort(key=lambda item: item.value)
        # print(len(rank_list[0]))
        self.rank_list = rank_list

    def get_attack_set(self, size=10, reverse=1, path='./logs/record1/tr_sst_distilbert-base-uncased.txt'):
        self.get_rank(path=path)
        sentences = []
        labels = []
        for i in range(self.num_classes):
            rank = self.rank_list[i]   # the rank of sentence with label i
            if reverse:
                rank.reverse()
            for j in range(size):
                sentence, label = self.set[rank[j].id]
                sentences.append(sentence)
                labels.append(label)
        return self.train_loader(data_set=MyDataset(sentences, labels), batch=32)

    def get_ood_attack(self, rate=0.2, path='./logs/record1/tr_sst_distilbert-base-uncased.txt', reverse=1, batch=32):
        self.get_rank(path=path)
        train_sentences, train_labels = [], []
        test_sentences, test_labels = [], []
        for i in range(self.num_classes):
            rank = self.rank_list[i]   # the rank of sentence with label i
            if reverse:
                rank.reverse()
            for j in range(int(len(rank) * rate)):
                image, label = self.set[rank[j].id]
                test_sentences.append(image)
                test_labels.append(label)
            for j in range(int(len(rank) * rate), len(rank)):
                image, label = self.set[rank[j].id]
                train_sentences.append(image)
                train_labels.append(label)
        train_loader = self.train_loader(data_set=MyDataset(train_sentences, train_labels), batch=batch)
        test_loader = self.train_loader(data_set=MyDataset(test_sentences, test_labels), batch=batch)
        return train_loader, test_loader

    def get_random_set(self, size=50):
        if not self.labeled_set:
            self.get_labeled_set()
        sentences, labels = [], []
        if self.num_classes == 1:
            random.shuffle(self.labeled_set[0])
            for j in range(size):
                sentences.append(self.labeled_set[0][j][0])
                labels.append(self.labeled_set[0][j][1])
            return self.train_loader(data_set=MyDataset(sentences, labels), batch=32)
        for i in range(self.num_classes):
            random.shuffle(self.labeled_set[i])
            for j in range(size):
                sentences.append(self.labeled_set[i][j])
                labels.append(i)
        return self.train_loader(data_set=MyDataset(sentences, labels), batch=32)

    def get_ood_random(self, rate=0.2, batch=32):
        if not self.labeled_set:
            self.get_labeled_set()
        train_sentences, train_labels = [], []
        test_sentences, test_labels = [], []
        if self.num_classes == 1:
            random.shuffle(self.labeled_set[0])
            for j in range(int(len(self.labeled_set[0]) * rate)):
                test_sentences.append(self.labeled_set[0][j][0])
                test_labels.append(self.labeled_set[0][j][1])
            print(len(self.labeled_set[0]) - int(len(self.labeled_set[0]) * rate))
            for j in range(int(len(self.labeled_set[0]) * rate), len(self.labeled_set[0])):
                train_sentences.append(self.labeled_set[0][j][0])
                train_labels.append(self.labeled_set[0][j][1])
            train_loader = self.train_loader(data_set=MyDataset(train_sentences, train_labels), batch=batch)
            test_loader = self.train_loader(data_set=MyDataset(test_sentences, test_labels), batch=batch)
            return train_loader, test_loader
        for i in range(self.num_classes):
            random.shuffle(self.labeled_set[i])
            for j in range(int(len(self.labeled_set[i]) * rate)):
                test_sentences.append(self.labeled_set[i][j])
                test_labels.append(i)
            for j in range(int(len(self.labeled_set[i]) * rate), len(self.labeled_set[i])):
                train_sentences.append(self.labeled_set[i][j])
                train_labels.append(i)
        train_loader = self.train_loader(data_set=MyDataset(train_sentences, train_labels), batch=batch)
        test_loader = self.train_loader(data_set=MyDataset(test_sentences, test_labels), batch=batch)
        return train_loader, test_loader


    def get_attack_kmean(self, model: BertClassificationModel, size=500):
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

    def get_attack_set_ori(self, size=10, l=0, r=500, path='./logs/record1/tr_sst_distilbert-base-uncased.txt'):
        if not r:
            r = size
        # if not self.rank_list:
        self.get_rank(path=path)
        if not self.labeled_set:
            self.get_labeled_set()
        # self.get_class()
        sentences = []
        labels = []
        for i in range(self.num_classes):
            rank = self.rank_list[i]   # the rank of sentence with label i
            first_index = int(l * len(self.labeled_set[i]) / len(self.sentences))
            last_index = int(r * len(self.labeled_set[i]) / len(self.sentences))
            # random.shuffle(rank[first_index:last_index])
            # print(first_index, last_index, len(self.labeled_set[i]))
            for j in range(last_index - first_index):
                sentence, label = self.set[rank[first_index + j].id]
                sentences.append(sentence)
                labels.append(label)
        return self.train_loader(data_set=MyDataset(sentences, labels), batch=32)


class GLUEDataset(Dataset):
    def __init__(self, sentences, labels, num_classes=2, tokenizer=Tokenizer):
        # self.origin_sentences = sentences
        self.sentences = []
        # self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.get_token(tokenizer, sentences)
        self.set = MyDataset(self.sentences, self.labels)
        self.labeled_set = None
        # self.set = None

        self.rank_list = None
        self.num_classes = num_classes

    def __getitem__(self, item):
        return self.sentences[item], self.labels[item]

    # def get_sentences(self):
    #     self.get_token(self.tokenizer, self.origin_sentences)
    #     self.set = MyDataset(self.sentences, self.labels)

    def get_token(self, tokenizer, sentences):
        for sentence in sentences:
            if len(sentence) == 2:
                tokenizer = tokenizer(mode='double_sentences')
            else:
                tokenizer = tokenizer()
            break
        sentences_without_padding = []
        max_length = 0
        # with tqdm(total=len(sentences)) as p_bar:
        for sentence in sentences:
            sentence = tokenizer(sentence)
            input_ids = torch.tensor(sentence['input_ids'])
            max_length = max(max_length, max(input_ids.shape))
            if input_ids.shape[0] > 1:
                input_ids = torch.cat([input_ids[0], input_ids[1][1:]], dim=0)
            input_ids = torch.squeeze(input_ids)
            attention_mask = torch.tensor(sentence['attention_mask'])
            if attention_mask.shape[0] > 1:
                attention_mask = torch.cat([attention_mask[0], attention_mask[1][1:]], dim=0)
            attention_mask = torch.squeeze(attention_mask)
            sentences_without_padding.append({'input_ids': input_ids,
                                   'attention_mask': attention_mask})
            # p_bar.update(1)
        # print(max_length)
        for item in sentences_without_padding:

            input_ids = item['input_ids']
            input_ids = F.pad(input_ids, [0, max_length - max(input_ids.shape)], "constant", 0)
            attention_mask = item['attention_mask']
            attention_mask = F.pad(attention_mask, [0, max_length - max(attention_mask.shape)], "constant", 0)
            self.sentences.append({'input_ids': input_ids, 'attention_mask': attention_mask})

    def get_labeled_set(self):
        sentences = [[] for i in range(self.num_classes)]
        for i in range(len(self.sentences)):
            sentences[self.labels[i]].append(self.sentences[i])
        self.labeled_set = sentences

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
                sentences, label = self.sentences[id], self.labels[id]
                rank_list[label].append(Item(id, norm))
            else:
                norm = float(s[0])
                sentences, label = self.sentences[count], self.labels[count]
                rank_list[label].append(Item(count, norm))
                count += 1
        for i in range(self.num_classes):
            rank_list[i].sort(key=lambda item: item.value)
        # print(len(rank_list[0]))
        self.rank_list = rank_list

    def get_attack_set(self, size=10, l=0, r=500, path='./logs/record1/tr_sst_distilbert-base-uncased.txt'):
        if not r:
            r = size
        # if not self.rank_list:
        self.get_rank(path=path)
        if not self.labeled_set:
            self.get_labeled_set()
        # self.get_class()
        sentences = []
        labels = []
        for i in range(self.num_classes):
            rank = self.rank_list[i]   # the rank of sentence with label i
            first_index = int(l * len(self.labeled_set[i]) / len(self.sentences))
            last_index = int(r * len(self.labeled_set[i]) / len(self.sentences))
            # random.shuffle(rank[first_index:last_index])
            # print(first_index, last_index, len(self.labeled_set[i]))
            for j in range(last_index - first_index):
                sentence, label = self.set[rank[first_index + j].id]
                sentences.append(sentence)
                labels.append(label)
        return self.train_loader(data_set=MyDataset(sentences, labels), batch=32)

    def get_random_set(self, size=500, top=None):
        if not self.labeled_set:
            self.get_labeled_set()
        sentences, labels = [], []
        for i in range(self.num_classes):
            if top is not None:
                random.shuffle(self.labeled_set[i][:top])
            else:
                random.shuffle(self.labeled_set[i])
            for j in range(int(size * len(self.labeled_set[i]) / len(self.sentences))):
                sentences.append(self.labeled_set[i][j])
                labels.append(i)
        return self.train_loader(data_set=MyDataset(sentences, labels), batch=32)

    def get_attack_kmean(self, model: BertClassificationModel, size=500):
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


def get_glue_ori(name='sst2'):
    dataset = DataIOGLUE(name=name)
    train_set = GLUEDataset(dataset.train_word, dataset.train_label, num_classes=dataset.num_classes)
    dev_set = GLUEDataset(dataset.dev_word, dataset.dev_label, num_classes=dataset.num_classes)
    test_set = GLUEDataset(dataset.test_word, dataset.test_label, num_classes=dataset.num_classes)
    return train_set, dev_set, test_set


def get_glue(name='sst2'):
    dataset = DataIOGLUE(name=name)
    train_set = GLUEDataset_ori(dataset.train_word, dataset.train_label, num_classes=dataset.num_classes)
    dev_set = GLUEDataset_ori(dataset.dev_word, dataset.dev_label, num_classes=dataset.num_classes)
    test_set = GLUEDataset_ori(dataset.test_word, dataset.test_label, num_classes=dataset.num_classes)
    return train_set, dev_set, test_set


class MyDataset(Dataset):
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
