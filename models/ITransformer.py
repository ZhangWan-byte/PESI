import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets import *
from utils import *
from .common import *


class IntTransEncoder(nn.Module):
    def __init__(self, embed_size=64, num_encoder_layers=2, nhead=2, dropout=0.1):
        super(IntTransEncoder, self).__init__()
        
        self.embedding = nn.Embedding(len(vocab), embed_size)

        self.pos_enc = PositionalEncoding(d_model=embed_size, dropout=dropout)
        layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=layer, num_layers=num_encoder_layers)
    
    def forward(self, x):
        
        x = torch.Tensor([to_onehot(i) for i in x]).int().cuda()

        x = self.embedding(x)
        # (batch, len, embed_size)
        x = self.encoder(x)        
        # (batch, len, embed_size)
        
        return x


class InteractTransformer(nn.Module):
    def __init__(self, embed_size=64, num_encoder_layers=2, nhead=2, dropout=0.1, use_coattn=False):
        super(InteractTransformer, self).__init__()

        self.use_coattn = use_coattn
        
        self.embedding = nn.Embedding(len(vocab), embed_size)

        self.pos_para = PositionalEncoding(d_model=embed_size, dropout=dropout)
        para_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, batch_first=True)
        self.transformer_para = nn.TransformerEncoder(encoder_layer=para_layer, num_layers=num_encoder_layers)

        self.pos_epi = PositionalEncoding(d_model=embed_size, dropout=dropout)
        epi_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, batch_first=True)
        self.transformer_epi = nn.TransformerEncoder(encoder_layer=epi_layer, num_layers=num_encoder_layers)

        if self.use_coattn==True:
            self.co_attn1 = CoAttention(embed_size=embed_size, output_size=embed_size)
            self.co_attn2 = CoAttention(embed_size=embed_size, output_size=embed_size)

        self.output_layer = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                          nn.Linear(embed_size//2, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        para, epi = torch.Tensor([to_onehot(i) for i in para]).int().cuda(), torch.Tensor([to_onehot(i) for i in epi]).int().cuda()

        # embedding
        para = self.embedding(para)
        # (batch, para_seq_length, embed_size)
        epi = self.embedding(epi)
        # (batch, epi_seq_length, embed_size)

        # transformer
        para = self.transformer_para(para)        
        # (batch, para_seq_length, embed_size)
        epi = self.transformer_epi(epi)        
        # (batch, epi_seq_length, embed_size)

        # co_attn
        if self.use_coattn:
            para, epi = self.co_attn2(para, epi)

        para = torch.mean(para, dim=1)
        # (batch, embed_size)
        epi = torch.mean(epi, dim=1)
        # (batch, embed_size)

        x = para * epi
        
        x = self.output_layer(x)
        
        return x


class InteractTransformerLSTM(nn.Module):
    def __init__(self, embed_size=64, hidden=64, num_encoder_layers=2, num_lstm_layers=2, nhead=2, dropout=0.1, use_coattn=False):
        super(InteractTransformerLSTM, self).__init__()

        self.use_coattn = use_coattn
        
        # embedding
        self.embedding = nn.Embedding(len(vocab), embed_size)

        # transformer
        self.pos_para = PositionalEncoding(d_model=embed_size, dropout=dropout)
        para_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, batch_first=True)
        self.transformer_para = nn.TransformerEncoder(encoder_layer=para_layer, num_layers=num_encoder_layers)

        self.pos_epi = PositionalEncoding(d_model=embed_size, dropout=dropout)
        epi_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, batch_first=True)
        self.transformer_epi = nn.TransformerEncoder(encoder_layer=epi_layer, num_layers=num_encoder_layers)

        # co-attention
        if self.use_coattn==True:
            self.co_attn = CoAttention(embed_size=embed_size, output_size=embed_size)

        # lstm
        proj_size = int(embed_size/num_lstm_layers) if num_lstm_layers>1 else int(embed_size/2)
        self.LSTM_para = nn.LSTM(input_size=embed_size, hidden_size=hidden, num_layers=num_lstm_layers, bidirectional=True, proj_size=proj_size)
        self.LSTM_epi = nn.LSTM(input_size=embed_size, hidden_size=hidden, num_layers=num_lstm_layers, bidirectional=True, proj_size=proj_size)
        
        # output
        self.output_layer = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                          nn.Linear(embed_size//2, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        para, epi = torch.Tensor([to_onehot(i) for i in para]).int().cuda(), torch.Tensor([to_onehot(i) for i in epi]).int().cuda()

        # embedding
        para = self.embedding(para)                     # (batch, para_seq_length, embed_size)
        epi = self.embedding(epi)                       # (batch, epi_seq_length, embed_size)

        # transformer
        para = self.transformer_para(para)              # (batch, para_seq_length, embed_size)        
        epi = self.transformer_epi(epi)                 # (batch, epi_seq_length, embed_size)

        # co_attn
        if self.use_coattn:
            para, epi = self.co_attn(para, epi)

        # lstm
        para, _ = self.LSTM_para(para)                  # (batch, para_seq_length, embed_size)
        epi, _ = self.LSTM_epi(epi)                     # (batch, epi_seq_length, embed_size)

        # para/epi representations
        para = torch.mean(para, dim=1)                  # (batch, embed_size)
        epi = torch.mean(epi, dim=1)                    # (batch, embed_size)
        x = para * epi

        # output
        x = self.output_layer(x)
        
        return x


class InteractTransformer_share(nn.Module):
    def __init__(self, embed_size=64, para_seq_length=128, epi_seq_length=400, hidden=128, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1, dim_feedforward=1024):
        super(InteractTransformer_share, self).__init__()
        
        self.embedding = nn.Embedding(len(vocab), embed_size)

        self.transformer = nn.Transformer(d_model=embed_size, nhead=2, \
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, \
            dim_feedforward=dim_feedforward, dropout=dropout)

        self.MLP = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                      nn.Linear(embed_size//2, 1), nn.LeakyReLU())
        
        self.output_layer = nn.Sequential(nn.Linear(para_seq_length, para_seq_length//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                          nn.Linear(para_seq_length//2, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        # paratope
        para = self.embedding(para)
        # (batch, para_seq_length, embed_size)
        # para = para.permute(0,2,1)
        # para = self.Linear_para(para)
        # para = para.permute(0,2,1)
        # (batch, hidden, embed_size)
        para = self.transformer(para, para)        
        # (batch, hidden, embed_size)
        para = self.MLP(para)
        # (batch, hidden, 1)
        para = para.squeeze(2)
        # (batch, hidden)
        
        # epitope
        epi = self.embedding(epi)
        # (batch, epi_seq_length, embed_size)
        # epi = epi.permute(0,2,1)
        # epi = self.Linear_epi(epi)
        # epi = epi.permute(0,2,1)
        # (batch, hidden, embed_size)
        epi = self.transformer(epi, epi)        
        # (batch, hidden, embed_size)
        epi = self.MLP(epi)
        # (batch, hidden, 1)
        epi = epi.squeeze(2)
        # (batch, hidden)

        x = para * epi
        
        x = self.output_layer(x)
        
        return x


class BiInteractTransformer(nn.Module):
    def __init__(self, embed_size=64, para_seq_length=128, epi_seq_length=400, hidden=128, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1, dim_feedforward=1024):
        super(BiInteractTransformer, self).__init__()
        
        self.embedding = nn.Embedding(len(vocab), embed_size)

        # self.Linear_para = nn.Sequential(nn.Linear(para_seq_length, hidden), nn.LeakyReLU(), nn.Dropout(0.1), \
        #                                  nn.Linear(hidden, hidden), nn.LeakyReLU())
        # self.Linear_epi = nn.Sequential(nn.Linear(epi_seq_length, hidden), nn.LeakyReLU(), nn.Dropout(0.1), \
        #                                 nn.Linear(hidden, hidden), nn.LeakyReLU())

        self.transformer_para = nn.Transformer(d_model=embed_size, nhead=2, \
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, \
            dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_epi = nn.Transformer(d_model=embed_size, nhead=2, \
            num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, \
            dim_feedforward=dim_feedforward, dropout=dropout)

        self.MLP_para = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                      nn.Linear(embed_size//2, 1), nn.LeakyReLU())
        self.MLP_epi = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                     nn.Linear(embed_size//2, 1), nn.LeakyReLU())
        
        self.output_layer = nn.Sequential(nn.Linear(para_seq_length, para_seq_length//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                          nn.Linear(para_seq_length//2, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        # embedding
        para = self.embedding(para)
        epi = self.embedding(epi)

        para0 = torch.clone(para)
        epi0 = torch.clone(epi)

        # para
        # (batch, para_seq_length, embed_size)
        para = self.transformer_para(src=epi0, tgt=para0)        
        # (batch, hidden, embed_size)
        para = self.MLP_para(para)
        # (batch, hidden, 1)
        para = para.squeeze(2)
        # (batch, hidden)

        # epi        
        # (batch, epi_seq_length, embed_size)
        epi = self.transformer_epi(src=para0, tgt=epi0)        
        # (batch, hidden, embed_size)
        epi = self.MLP_epi(epi)
        # (batch, hidden, 1)
        epi = epi.squeeze(2)
        # (batch, hidden)

        x = para * epi
        
        x = self.output_layer(x)
        
        return x


class InteractCoattn_noTransformer(nn.Module):
    def __init__(self, embed_size=64, seq_length=128, dropout=0.1, dim_feedforward=1024):
        super(InteractCoattn_noTransformer, self).__init__()
        
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.transformer_para = nn.Transformer(d_model=embed_size, nhead=4, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_epi = nn.Transformer(d_model=embed_size, nhead=4, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=dim_feedforward, dropout=dropout)
                
        self.co_attn = CoAttention(embed_size=embed_size, output_size=embed_size)

        self.MLP_para = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                 nn.Linear(embed_size//2, 1))
        self.MLP_epi = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                 nn.Linear(embed_size//2, 1))
        
        self.output_layer = nn.Sequential(nn.Linear(seq_length, seq_length//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                          nn.Linear(seq_length//2, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        # paratope
        para = self.embedding(para)
        # (batch, seq_length, embed_size)

        # epitope
        epi = self.embedding(epi)
        # (batch, seq_length, embed_size)


        # co-attention
        # print(para.shape, epi.shape)
        para, epi = self.co_attn(para, epi)


        # paratope
        para = self.transformer_para(para, para)
        # (batch, seq_length, embed_size)
        para = self.MLP_para(para)
        # (batch, seq_length, 1)
        para = para.squeeze(2)
        # (batch, seq_length)


        # epitope
        epi = self.transformer_epi(epi, epi)        
        # (batch, seq_length, embed_size)
        epi = self.MLP_epi(epi)
        # (batch, seq_length, 1)
        epi = epi.squeeze(2)
        # (batch, seq_length)

        x = para * epi
        
        x = self.output_layer(x)
        
        return x


if __name__ == "__main__":
    k_iter = 0

    train_dataset = SeqDataset(data_path="../data/SARS-SAbDab_Shaun/CoV-AbDab_extract.csv", seq_length=128, \
                    kfold=10, holdout_fold=k_iter, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)

    test_dataset = SeqDataset(data_path="../data/SARS-SAbDab_Shaun/CoV-AbDab_extract.csv", seq_length=128, \
                    kfold=10, holdout_fold=k_iter, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = InteractTransformer(embed_size=64, seq_length=128)

    print(model)

    for i, (para, epi, label) in enumerate(train_loader):
        pred = model(para, epi)
        break

    print(para.shape, epi.shape, label.shape, pred.shape)