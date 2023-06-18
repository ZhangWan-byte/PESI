import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets import *
from utils import *
from .common import *


class BiMLP(nn.Module):
    def __init__(self, embed_size=64, hidden=128, num_layers=2, dropout=0.1, use_pretrain=False):
        super(BiMLP, self).__init__()

        self.use_pretrain = use_pretrain

        if self.use_pretrain:
            embed_size = 320
            hidden = 64
        else:
            self.embedding = nn.Embedding(len(vocab), embed_size)
        
        if self.use_pretrain:
            self.MLP_para = nn.Sequential(nn.Linear(embed_size, embed_size*2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                        nn.Linear(embed_size*2, embed_size), nn.LeakyReLU())
            self.MLP_epi = nn.Sequential(nn.Linear(embed_size, embed_size*2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                        nn.Linear(embed_size*2, embed_size), nn.LeakyReLU())
        else:
            self.MLP_para = nn.Sequential(nn.Linear(embed_size, embed_size), nn.LeakyReLU())
            self.MLP_epi = nn.Sequential(nn.Linear(embed_size, embed_size), nn.LeakyReLU())
        
        if self.use_pretrain:
            self.output_layer = nn.Sequential(nn.Linear(embed_size*2, embed_size), nn.LeakyReLU(), nn.Dropout(dropout), \
                                              nn.Linear(embed_size, 1), nn.Sigmoid())
        else:
            self.output_layer = nn.Sequential(nn.Linear(embed_size, embed_size*2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                              nn.Linear(embed_size*2, embed_size), nn.LeakyReLU(), nn.Dropout(dropout), \
                                              nn.Linear(embed_size, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        if self.use_pretrain:
            para, epi = get_embedding(para, epi)
            # (batch, embed_size)

            para = self.MLP_para(para)
            epi = self.MLP_epi(epi)

            x = torch.cat([para, epi], dim=1)
            x = self.output_layer(x)
            
            return x

        else:
            para, epi = torch.Tensor([to_onehot(i) for i in para]).int().cuda(), torch.Tensor([to_onehot(i) for i in epi]).int().cuda()

            # paratope
            para = self.embedding(para)
            # (batch, para_seq_length, embed_size)
            para = self.MLP_para(para)
            # (batch, para_seq_length, embed_size)
            para = torch.mean(para, dim=1)
            # (batch, 1, embed_size)
            para = para.squeeze(1)
            # (batch, embed_size)
            
            # epitope
            epi = self.embedding(epi)
            # (batch, epi_seq_length, embed_size)
            epi = self.MLP_epi(epi)
            # (batch, epi_seq_length, embed_size)
            epi = torch.mean(epi, dim=1)
            # (batch, 1, embed_size)
            epi = epi.squeeze(1)
            # (batch, embed_size)

            x = para * epi
            
            x = self.output_layer(x)
            
            return x


class BiLSTM_demo(nn.Module):
    def __init__(self, embed_size=64, hidden=128, num_layers=2, dropout=0.1, use_pretrain=False):
        super(BiLSTM_demo, self).__init__()

        self.use_pretrain = use_pretrain

        if self.use_pretrain:
            embed_size = 320
            hidden = 64
        else:
            self.embedding = nn.Embedding(len(vocab), embed_size)
        
        proj_size = int(hidden/num_layers) if num_layers>1 else int(hidden/2)
        self.LSTM_para = nn.LSTM(input_size=embed_size, hidden_size=hidden, num_layers=num_layers, bidirectional=True, proj_size=proj_size)

        self.LSTM_epi = nn.LSTM(input_size=embed_size, hidden_size=hidden, num_layers=num_layers, bidirectional=True, proj_size=proj_size)
        
        if self.use_pretrain:
            pass
        else:
            self.MLP_para = nn.Sequential(nn.Linear(hidden, hidden//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                        nn.Linear(hidden//2, 1), nn.LeakyReLU())
            self.MLP_epi = nn.Sequential(nn.Linear(hidden, hidden//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                        nn.Linear(hidden//2, 1), nn.LeakyReLU())
        
        if self.use_pretrain:
            self.output_layer = nn.Sequential(nn.Linear(hidden, hidden//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                              nn.Linear(hidden//2, 1), nn.Sigmoid())
        else:
            self.output_layer = nn.Sequential(nn.Linear(hidden, hidden//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                              nn.Linear(hidden//2, 1), nn.Sigmoid())

    def forward(self, para, epi):
        
        if self.use_pretrain:
            para, epi = get_embedding(para, epi)
            # print(para.shape, epi.shape)

            # (batch, embed_size)
            para, _ = self.LSTM_para(para)
            # print(para.shape)
            # (batch, embed_size)

            # (batch, embed_size)
            epi, _ = self.LSTM_epi(epi)
            # print(epi.shape)
            # (batch, embed_size)

            x = para * epi
            # print(x.shape)
            x = self.output_layer(x)
            
            return x

        else:
            para = list(map(replace_pad, para))
            epi = list(map(replace_pad, epi))
            para, epi = torch.Tensor([to_onehot(i) for i in para]).int().cuda(), torch.Tensor([to_onehot(i) for i in epi]).int().cuda()

            # paratope
            para = self.embedding(para)
            # (batch, para_seq_length, hidden)
            para, _ = self.LSTM_para(para)
            # (batch, para_seq_length, hidden)
            para = torch.mean(para, dim=1)
            # (batch, 1, hidden)
            para = para.squeeze(1)
            # (batch, hidden)
            
            # epitope
            epi = self.embedding(epi)
            # (batch, epi_seq_length, hidden)
            epi, _ = self.LSTM_epi(epi)        
            # (batch, epi_seq_length, hidden)
            epi = torch.mean(epi, dim=1)
            # (batch, 1, hidden)
            epi = epi.squeeze(1)
            # (batch, hidden)

            x = para * epi
            
            x = self.output_layer(x)
            
            return x
        

class BiLSTMEncoder(nn.Module):
    def __init__(self, embed_size=64, hidden=128, num_layers=2):
        super(BiLSTMEncoder, self).__init__()

        self.embed_size = embed_size
        self.hidden = hidden
        self.num_layers = num_layers

        self.embedding = nn.Embedding(len(vocab), embed_size)

        proj_size = int(hidden/num_layers) if num_layers>1 else int(hidden/2)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden, 
                               num_layers=num_layers, bidirectional=True, proj_size=proj_size)


    def forward(self, x):
        x = torch.Tensor([to_onehot(i) for i in x]).int().cuda()

        x = self.embedding(x)
        # (batch, len, hidden)
        x, _ = self.encoder(x)
        # (batch, len, hidden)

        return x


class BiLSTM(nn.Module):
    def __init__(self, embed_size=64, hidden=128, num_layers=2, dropout=0.1, use_pretrain=False):
        super(BiLSTM, self).__init__()

        self.use_pretrain = use_pretrain

        if self.use_pretrain:
            embed_size = 320
            hidden = 64
        else:
            self.embedding = nn.Embedding(len(vocab), embed_size)
        
        proj_size = int(hidden/num_layers) if num_layers>1 else int(hidden/2)
        self.LSTM_para = nn.LSTM(input_size=embed_size, hidden_size=hidden, num_layers=num_layers, bidirectional=True, proj_size=proj_size)

        self.LSTM_epi = nn.LSTM(input_size=embed_size, hidden_size=hidden, num_layers=num_layers, bidirectional=True, proj_size=proj_size)
        
        if self.use_pretrain:
            pass
        else:
            self.MLP_para = nn.Sequential(nn.Linear(hidden, hidden//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                        nn.Linear(hidden//2, 1), nn.LeakyReLU())
            self.MLP_epi = nn.Sequential(nn.Linear(hidden, hidden//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                        nn.Linear(hidden//2, 1), nn.LeakyReLU())
        
        if self.use_pretrain:
            self.output_layer = nn.Sequential(nn.Linear(hidden, hidden//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                              nn.Linear(hidden//2, 1), nn.Sigmoid())
        else:
            self.output_layer = nn.Sequential(nn.Linear(hidden, hidden//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                              nn.Linear(hidden//2, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        if self.use_pretrain:
            para, epi = get_embedding(para, epi)
            # print(para.shape, epi.shape)

            # (batch, embed_size)
            para, _ = self.LSTM_para(para)
            # print(para.shape)
            # (batch, embed_size)

            # (batch, embed_size)
            epi, _ = self.LSTM_epi(epi)
            # print(epi.shape)
            # (batch, embed_size)

            x = para * epi
            # print(x.shape)
            x = self.output_layer(x)
            
            return x

        else:
            para, epi = torch.Tensor([to_onehot(i) for i in para]).int().cuda(), torch.Tensor([to_onehot(i) for i in epi]).int().cuda()

            # paratope
            para = self.embedding(para)
            # (batch, para_seq_length, hidden)
            para, _ = self.LSTM_para(para)
            # (batch, para_seq_length, hidden)
            para = torch.mean(para, dim=1)
            # (batch, 1, hidden)
            para = para.squeeze(1)
            # (batch, hidden)
            
            # epitope
            epi = self.embedding(epi)
            # (batch, epi_seq_length, hidden)
            epi, _ = self.LSTM_epi(epi)        
            # (batch, epi_seq_length, hidden)
            epi = torch.mean(epi, dim=1)
            # (batch, 1, hidden)
            epi = epi.squeeze(1)
            # (batch, hidden)

            x = para * epi
            
            x = self.output_layer(x)
            
            return x