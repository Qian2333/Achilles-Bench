import torch
from torch.utils.data import DataLoader, Dataset
import os
import re
from random import sample
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertForSequenceClassification, BertTokenizer, DistilBertTokenizer, \
    DistilBertForSequenceClassification, DistilBertModel, BertModel
from lib.dataset.aclimdb import ImdbDataset
from tqdm import tqdm


class Tokenizer(nn.Module):
    def __init__(self, mode='sig'):
        super(Tokenizer, self).__init__()

        # model_name = 'distilbert-base-uncased'
        model_name = 'bert-base-uncased'
        self.mode = mode

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

    def forward(self, batch_sentences):  # [batch_size,1]
        if self.mode == 'double_sentences':
            qs, st = batch_sentences
            inp = []
            for item in zip(qs, st):
                it1, it2 = item
                inp.append([it1, it2])
            # print(inp)
            # exit(0)
            batch_sentences = inp
        sentences_tokenizer = self.tokenizer(batch_sentences,
                                             truncation=True,
                                             padding=True,
                                             max_length=256,
                                             add_special_tokens=True)
        return sentences_tokenizer


class BertClassificationModel_ori(nn.Module):
    def __init__(self, hidden_size=768, class_num=2, mode='sig'):
        super(BertClassificationModel_ori, self).__init__()

        self.class_num = class_num
        # model_name = 'distilbert-base-uncased'
        model_name = 'bert-base-uncased'
        self.mode = mode

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_name)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        # for p in self.bert.parameters():  # 冻结bert参数
        #     p.requires_grad = False
        self.pre_fc = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, class_num)

    # def forward(self, input_ids, attention_mask, get_feature=False):  # [batch_size,1]
    def forward(self, batch_sentences):  # [batch_size,1]
        if self.mode == 'double_sentences':
            qs, st = batch_sentences
            inp = []
            for item in zip(qs, st):
                it1, it2 = item
                inp.append([it1, it2])
            # print(inp)
            # exit(0)
            batch_sentences = inp
        sentences_tokenizer = self.tokenizer(batch_sentences,
                                             truncation=True,
                                             padding=True,
                                             max_length=256,
                                             add_special_tokens=True)
        input_ids = torch.tensor(sentences_tokenizer['input_ids']).cuda()
        # print(input_ids[1])
        # # for i in range(len(batch_sentences)):
        # sentences_tokenizer = self.tokenizer(batch_sentences[1],
        #                                      truncation=True,
        #                                      padding=True,
        #                                      max_length=256,
        #                                      add_special_tokens=True)
        # input_ids = torch.tensor(sentences_tokenizer['input_ids']).cuda()
        # print(input_ids)
        # print(torch.cat([input_ids[0], input_ids[1][1:]], dim=0))
        # exit(0)
        # print(input_ids.shape)
        # exit(0)
        attention_mask = torch.tensor(sentences_tokenizer['attention_mask']).cuda()
        # print(input_ids.shape, attention_mask.shape)
        bert_out = self.bert(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda())

        last_hidden_state = bert_out[0]  # [batch_size, sequence_length, hidden_s
        bert_cls_hidden_state = last_hidden_state[:, 0]  
        fc_out = self.pre_fc(bert_cls_hidden_state)
        fc_out = nn.ReLU()(fc_out)
        fc_out = self.dropout(fc_out)
        fc_out = self.fc(fc_out)  
        return fc_out


class BertClassificationModel(nn.Module):
    def __init__(self, hidden_size=768, class_num=2, mode='sig'):
        super(BertClassificationModel, self).__init__()

        self.class_num = class_num
        # model_name = 'distilbert-base-uncased'
        model_name = 'bert-base-uncased'
        self.mode = mode

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=model_name)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        # for p in self.bert.parameters(): 
        #     p.requires_grad = False
        self.pre_fc = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, class_num)

    def forward(self, input_ids, attention_mask, get_feature=False):  # [batch_size,1]
        bert_out = self.bert(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda())

        last_hidden_state = bert_out[0]  # [batch_size, sequence_length, hidden_size]
        bert_cls_hidden_state = last_hidden_state[:, 0]  
        if get_feature:
            return bert_cls_hidden_state
        fc_out = self.pre_fc(bert_cls_hidden_state)
        fc_out = nn.ReLU()(fc_out)
        fc_out = self.dropout(fc_out)
        fc_out = self.fc(fc_out)  
        return fc_out
