import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets import *
from utils import *
from .common import *


class FTransformer(nn.Module):
    def __init__(self, embed_size=64, hidden=128, num_layers=2, dropout=0.1, k4kmer=3, use_pretrain=False, use_coattn=False, seq_encoder_type="transformer", num_heads=4):
        super(FTransformer, self).__init__()

        self.use_pretrain = use_pretrain
        self.use_coattn = use_coattn
        self.seq_encoder_type = seq_encoder_type
        self.k4kmer = k4kmer
        self.embed_size = embed_size

        # embedding
        if self.use_pretrain:
            embed_size = 320
            hidden = 64
        else:
            self.embedding = nn.Embedding(len(vocab), embed_size)
        
        # sequence encoder
        if self.use_pretrain:
            pass
        else:
            self.seq_encoder_para = SequenceEncoder(seq_encoder_type=self.seq_encoder_type, 
                                                    num_layers=num_layers, 
                                                    embed_size=embed_size, 
                                                    hidden=hidden, 
                                                    dropout=dropout, 
                                                    nhead=num_heads)

            self.seq_encoder_epi = SequenceEncoder(seq_encoder_type=self.seq_encoder_type, 
                                                   num_layers=num_layers, 
                                                   embed_size=embed_size, 
                                                   hidden=hidden, 
                                                   dropout=dropout, 
                                                   nhead=num_heads)

        # co-attention
        if use_coattn==True:
            self.co_attn = CoAttention(embed_size=embed_size, output_size=embed_size)

        # output head
        if self.use_pretrain:
            self.output_layer = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                              nn.Linear(embed_size//2, 1), nn.Sigmoid())
        else:
            self.output_layer = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                              nn.Linear(embed_size//2, 1), nn.Sigmoid())

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
            # 0. kmer embedding
            para_seq_length = para.size(1)
            para = self.embedding(para)                     # (batch, para_seq_length, embed_size)
            # 1. kmer sequential features
            para = kmer_embed(para, self.k4kmer)            # (batch*(para_seq_length-k+1), k, embed_size)
            if self.seq_encoder_type=="transformer":
                para = self.seq_encoder_para(para)          # (batch*(para_seq_length-k+1), k, embed_size)
            elif self.seq_encoder_type=="lstm":
                para, _ = self.seq_encoder_para(para)       # (batch*(para_seq_length-k+1), k, embed_size)
            para = torch.mean(para, dim=1)                  # (batch*(para_seq_length-k+1), 1, embed_size)
            para = para.reshape(-1, para_seq_length-self.k4kmer+1, self.embed_size)
                                                            # (batch, para_seq_length-k+1, embed_size)
            # # 2. paratope set interaction
            # para = self.para_enc(para)
            # para = self.para_dec(para)                      # (batch, para_seq_length-k+1, embed_size)

            # epitope
            # 0. kmer embedding
            epi_seq_length = epi.size(1)
            epi = self.embedding(epi)                       # (batch, epi_seq_length, embed_size)
            # 1. kmer sequential features
            epi = kmer_embed(epi, self.k4kmer)              # (batch*(epi_seq_length-k+1), k, embed_size)
            if self.seq_encoder_type=="transformer":
                epi = self.seq_encoder_epi(epi)             # (batch*(epi_seq_length-k+1), k, embed_size)
            elif self.seq_encoder_type=="lstm":
                epi, _ = self.seq_encoder_epi(epi)          # (batch*(epi_seq_length-k+1), k, embed_size)
            epi = torch.mean(epi, dim=1)                    # (batch*(epi_seq_length-k+1), 1, embed_size)
            epi = epi.reshape(-1, epi_seq_length-self.k4kmer+1, self.embed_size)
                                                            # (batch, epi_seq_length-k+1, embed_size)
            # # 2. epitope set interaction
            # epi = self.epi_enc(epi)
            # epi = self.epi_dec(epi)                         # (batch, epi_seq_length-k+1, embed_size)

            # co-attn
            if self.use_coattn==True:
                para, epi = self.co_attn(para, epi)         # (batch, epi_seq_length-k+1, embed_size)


            # average for paratope/epitope representation
            para = torch.mean(para, dim=1)          # (batch, embed_size)    
            epi = torch.mean(epi, dim=1)            # (batch, embed_size)

            # output
            x = para * epi                          # (batch, embed_size)
            x = self.output_layer(x)
            
            return x