import torch
from torch.utils.data import DataLoader, Dataset
import os
import re
from random import sample
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer, GPT2Tokenizer, GPT2Model, GPT2ForSequenceClassification, \
    BertForSequenceClassification, T5Tokenizer, T5Model, RobertaForSequenceClassification, RobertaTokenizer, \
    T5ForConditionalGeneration, BertConfig, T5EncoderModel, GPT2Config, RobertaConfig
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


class BertClassificationModel(nn.Module):
    def __init__(self, hidden_size=768, class_num=2, mode='sig', model_name='bert'):
        """
        """
        super(BertClassificationModel, self).__init__()

        self.class_num = class_num
        self.mode = mode
        self.model_name = model_name
        if model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')
            self.bert = BertForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path='bert-base-uncased')
            self.bert.classifier = nn.Linear(hidden_size, class_num)
        elif model_name == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2')
            self.bert = GPT2ForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path='gpt2')
            self.bert.score = nn.Linear(hidden_size, class_num)
        elif model_name == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path='roberta-base')
            self.bert = RobertaForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path='roberta-base')
            self.bert.classifier.out_proj = nn.Linear(hidden_size, class_num)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncased')
            config = BertConfig()
            self.bert = BertForSequenceClassification(config)
            self.bert.classifier = nn.Linear(hidden_size, class_num)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.bert.resize_token_embeddings(len(self.tokenizer))
            self.bert.config.pad_token_id = self.bert.config.eos_token_id

    def forward(self, batch_sentences):  # [batch_size,1]
        if self.mode == 'double_sentences':
            qs, st = batch_sentences
            inp = []
            for item in zip(qs, st):
                it1, it2 = item
                inp.append([it1, it2])
            batch_sentences = inp
        else:
            [batch_sentences] = batch_sentences
            inp = []
            for (item) in batch_sentences:
                inp.append([item, None])
            batch_sentences = inp

        sentences_tokenizer = self.tokenizer(batch_sentences,
                                             truncation=True,
                                             padding=True,
                                             max_length=256,
                                             add_special_tokens=True)
        input_ids = torch.tensor(sentences_tokenizer['input_ids'])
        attention_mask = torch.tensor(sentences_tokenizer['attention_mask'])
        input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return bert_out.logits


class T5ForSequenceClassification(nn.Module):
    def __init__(self, hidden_size=768, class_num=2, mode='sig'):
        super(T5ForSequenceClassification, self).__init__()

        self.class_num = class_num
        self.mode = mode
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path='t5-base')
        self.bert = T5EncoderModel.from_pretrained(pretrained_model_name_or_path='t5-base')
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.dropout = nn.Dropout(p=0.1, inplace=False)
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
            batch_sentences = inp
        else:
            [batch_sentences] = batch_sentences
            inp = []
            for (item) in batch_sentences:
                inp.append([item, None])
            batch_sentences = inp

        sentences_tokenizer = self.tokenizer(batch_sentences,
                                             truncation=True,
                                             padding=True,
                                             max_length=256,
                                             add_special_tokens=True)
        input_ids = torch.tensor(sentences_tokenizer['input_ids']).cuda()
        attention_mask = torch.tensor(sentences_tokenizer['attention_mask']).cuda()
        bert_out = self.bert(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda())
        last_hidden_state = bert_out[0]  # [batch_size, sequence_length, hidden_size] 
        bert_cls_hidden_state = last_hidden_state[:, 0]  
        fc_out = self.pre_fc(bert_cls_hidden_state)
        fc_out = nn.ReLU()(fc_out)
        fc_out = self.dropout(fc_out)
        fc_out = self.fc(fc_out)  
        return fc_out


def get_model(args, mode='sig', class_num=2):
    if args.attacked_model == 't5':
        return T5ForSequenceClassification(class_num=class_num, mode=mode)

    return BertClassificationModel(class_num=class_num, mode=mode, model_name=args.attacked_model)



class BertClassificationModel_ori(nn.Module):
    def __init__(self, hidden_size=768, class_num=2, mode='sig'):
        super(BertClassificationModel_ori, self).__init__()

        self.class_num = class_num
        # model_name = 'distilbert-base-uncased'
        model_name = 'bert-base-uncased'
        self.mode = mode

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name)
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
