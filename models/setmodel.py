import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import *
from utils import *
from .common import *


# Set Transformer Modules
# https://github.com/juho-lee/set_transformer
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetEncoder(nn.Module):
    def __init__(self, embed_size, num_outputs, dim_output, 
                 num_inds=6, hidden=128, num_heads=4, ln=False, dropout=0.1):
        super(SetEncoder, self).__init__()

        self.embedding = nn.Embedding(len(vocab), embed_size)

        self.encoder = nn.Sequential(
                ISAB(embed_size, hidden, num_heads, num_inds, ln=ln),
                ISAB(hidden, hidden, num_heads, num_inds, ln=ln))
        self.decoder = nn.Sequential(
                PMA(hidden, num_heads, num_outputs, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                nn.Linear(hidden, dim_output))

    def forward(self, x):

        x = torch.Tensor([to_onehot(i) for i in x]).int().cuda()
        
        x = self.embedding(x)                                   # (batch, num_inds, embed_size)

        x = self.encoder(x)                                     # (batch, num_inds, hidden)

        x = self.decoder(x)                                     # (batch, num_inds, dim_output)

        return x



class SetTransformer(nn.Module):
    def __init__(self, 
                 dim_input, 
                 num_outputs, 
                 dim_output, 
                 num_inds=32, 
                 dim_hidden=128, 
                 num_heads=4, 
                 ln=False, 
                 dropout=0.1, 
                 use_coattn=False, 
                 share=False, 
                 use_BSS=False, 
                 use_CLIP=False, 
                 use_cosCLS=False):
        super(SetTransformer, self).__init__()

        self.use_coattn = use_coattn
        self.use_BSS = use_BSS
        self.use_CLIP = use_CLIP
        self.use_cosCLS = use_cosCLS
        
        self.embedding = nn.Embedding(len(vocab), dim_input)
        
        self.share = share
        if self.share==True:
            self.enc = nn.Sequential(
                    ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                    ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
            
            self.dec = nn.Sequential(
                    PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                    SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                    SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                    nn.Linear(dim_hidden, dim_output))
        else:
            self.para_enc = nn.Sequential(
                    ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                    ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
            
            self.para_dec = nn.Sequential(
                    PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                    SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                    SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                    nn.Linear(dim_hidden, dim_hidden))

            self.epi_enc = nn.Sequential(
                    ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                    ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
            
            self.epi_dec = nn.Sequential(
                    PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                    SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                    SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                    nn.Linear(dim_hidden, dim_hidden))

        if self.use_coattn==True:
            self.co_attn = CoAttention(embed_size=dim_hidden, output_size=dim_hidden)

        if self.use_CLIP==True:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.use_cosCLS==True:
            self.clf = distLinear(indim=dim_hidden, outdim=2)

        if not self.use_CLIP and not self.use_cosCLS:
            self.output_layer = nn.Sequential(nn.Linear(dim_hidden, dim_hidden//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                              nn.Linear(dim_hidden//2, 1), nn.Sigmoid())


    def forward(self, para, epi):
        # embedding
        para, epi = torch.Tensor([to_onehot(i) for i in para]).int().cuda(), torch.Tensor([to_onehot(i) for i in epi]).int().cuda()
        para = self.embedding(para)
        epi = self.embedding(epi)
        # (batch, seq_len, embed_size) / (batch, num_inds, dim_input)

        if self.share==True:
            # encoder
            para = self.enc(para)
            epi = self.enc(epi)
            # (batch, seq_len, hidden) / (batch, num_inds, dim_hidden)

            if self.use_coattn==True:
                para, epi = self.co_attn(para, epi)

            # decoder
            para = self.dec(para)
            epi = self.dec(epi)
            # (batch, seq_len, embed_size) / (batch, num_inds, dim_output)
        else:
            # encoder
            para = self.para_enc(para)
            epi = self.epi_enc(epi)
            # (batch, seq_len, hidden) / (batch, num_inds, dim_hidden)

            if self.use_coattn==True:
                para, epi = self.co_attn(para, epi)

            # decoder
            para = self.para_dec(para)
            epi = self.epi_dec(epi)
            # (batch, seq_len, embed_size) / (batch, num_inds, dim_output)


        # if self.use_coattn==True:
        #     para, epi = self.co_attn(para, epi)

        # sentence representation
        para = torch.mean(para, dim=1)
        epi = torch.mean(epi, dim=1)
        # (batch, 1, embed_size) / (batch, 1, dim_output)
        para = para.squeeze(1)
        epi = epi.squeeze(1)
        # (batch, embed_size) / (batch, dim_output)

        if self.use_BSS:
            BSS = 0
            u1, s1, v1 = torch.svd(para.t())
            ll1 = s1.size(0)
            BSS += torch.pow(s1[ll1-1], 2)

            u2, s2, v2 = torch.svd(epi.t())
            ll2 = s2.size(0)
            BSS += torch.pow(s2[ll2-1], 2)

        # output
        if self.use_CLIP:
            para = para / para.norm(dim=1, keepdim=True)
            epi = epi / epi.norm(dim=1, keepdim=True)

            # print(self.logit_scale.shape, para.shape, epi.shape)

            logits_per_para = self.logit_scale.exp() * para @ epi.t()
            logits_per_epi = logits_per_para.t()

            return logits_per_para, logits_per_epi

        elif self.use_cosCLS:
            x = para * epi
            x = self.clf(x)
            
            return x

        else:
            x = para * epi
            x = self.output_layer(x)

            if self.use_BSS:
                return x, BSS
            else:
                return x
        






# below are trials

class SetCoAttnTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False, dropout=0.1):
        super(SetCoAttnTransformer, self).__init__()
        
        self.embedding = nn.Embedding(len(vocab), dim_input)
        
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        
        self.co_attn = CoAttention(embed_size=dim_hidden, output_size=dim_hidden)
        
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

        # self.MLP_para = nn.Sequential(nn.Linear(dim_output, dim_output//2), nn.LeakyReLU(), nn.Dropout(0.1), \
        #                          nn.Linear(dim_output//2, 1))
        # self.MLP_epi = nn.Sequential(nn.Linear(dim_output, dim_output//2), nn.LeakyReLU(), nn.Dropout(0.1), \
        #                          nn.Linear(dim_output//2, 1))
        self.MLP = nn.Sequential(nn.Linear(dim_output, dim_output//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                 nn.Linear(dim_output//2, 1))

        self.output_layer = nn.Sequential(nn.Linear(num_inds, num_inds//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                          nn.Linear(num_inds//2, 1), nn.Sigmoid())


    def forward(self, para, epi):
        # embedding
        para = self.embedding(para)
        epi = self.embedding(epi)
        # (batch, seq_len, embed_size) / (batch, num_inds, dim_input)

        # encoder
        para = self.enc(para)
        epi = self.enc(epi)
        # (batch, seq_len, hidden) / (batch, num_inds, dim_hidden)

        # co-attention
        para, epi = self.co_attn(para, epi)
        # (batch, seq_len, hidden) / (batch, num_inds, dim_hidden)

        # decoder
        para = self.dec(para)
        epi = self.dec(epi)
        # (batch, seq_len, embed_size) / (batch, num_inds, dim_output)

        # # co-attention
        # para, epi = self.co_attn(para, epi)
        # # (batch, seq_len, embed_size) / (batch, num_inds, dim_output)

        # MLP
        para = self.MLP(para)
        epi = self.MLP(epi)
        # (batch, seq_len, 1) / (batch, num_inds, 1)
        para = para.squeeze(2)
        epi = epi.squeeze(2)
        # (batch, seq_len) / (batch, num_inds)

        # output
        x = para * epi
        x = self.output_layer(x)

        return x


class AlternateCoattnModel(nn.Module):
    def __init__(self, embed_size=64, seq_length=128, num_alternates=3, dropout=0.1):
        super(AlternateCoattnModel, self).__init__()
        
        self.embedding = nn.Embedding(len(vocab), embed_size)

        self.mha_modules_para = nn.ModuleList([])
        self.mha_modules_epi = nn.ModuleList([])
        self.coattn_modules = nn.ModuleList([])

        self.num_alternates = num_alternates
        for i in range(self.num_alternates):
            self.mha_modules_para.append(nn.MultiheadAttention(embed_dim=embed_size, num_heads=4))
            self.mha_modules_epi.append(nn.MultiheadAttention(embed_dim=embed_size, num_heads=4))
            self.coattn_modules.append(CoAttention(embed_size=embed_size, output_size=embed_size))

        self.MLP_para = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                 nn.Linear(embed_size//2, 1))
        self.MLP_epi = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                 nn.Linear(embed_size//2, 1))
        self.output_layer = nn.Sequential(nn.Linear(seq_length*2, seq_length//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                          nn.Linear(seq_length//2, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        # embedding
        para = self.embedding(para)
        epi = self.embedding(epi)
        # (batch, seq_length, embed_size)

        for i in range(self.num_alternates):
            para, _ = self.mha_modules_para[i](para, para, para)
            epi, _ = self.mha_modules_epi[i](epi, epi, epi)

            para, epi = self.coattn_modules[i](para, epi)

        # (batch, seq_length, embed_size)
        para = self.MLP_para(para)
        para = para.squeeze(2)
        # (batch, seq_length)

        # (batch, seq_length, embed_size)
        epi = self.MLP_epi(epi)
        epi = epi.squeeze(2)
        # (batch, seq_length)

        # x = para * epi
        # # (batch, seq_length)
        x = torch.concat([para, epi], dim=1)
        x = self.output_layer(x)
        
        return x


class SetModel(nn.Module):
    def __init__(self, embed_size=64, hidden=128, num_layers=2, dropout=0.1, k4kmer=3, use_pretrain=False, use_coattn=False, seq_encoder_type="transformer", num_heads=4, num_inds=6, num_outputs=6, ln=False):
        super(SetModel, self).__init__()

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
        
        # set interaction
        self.para_enc = nn.Sequential(
                ISAB(embed_size, hidden, num_heads, num_inds, ln=ln),
                ISAB(hidden, hidden, num_heads, num_inds, ln=ln))
        
        self.para_dec = nn.Sequential(
                PMA(hidden, num_heads, num_outputs, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                nn.Linear(hidden, embed_size))

        self.epi_enc = nn.Sequential(
                ISAB(embed_size, hidden, num_heads, num_inds, ln=ln),
                ISAB(hidden, hidden, num_heads, num_inds, ln=ln))
        
        self.epi_dec = nn.Sequential(
                PMA(hidden, num_heads, num_outputs, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                nn.Linear(hidden, embed_size))

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

            
            # 0. kmer embedding
            para_seq_length = para.size(1)
            para = self.embedding(para)                     # (batch, para_seq_length, embed_size)
            epi_seq_length = epi.size(1)
            epi = self.embedding(epi)                       # (batch, epi_seq_length, embed_size)

            # co-attn
            if self.use_coattn==True:
                # para1, epi1 = self.co_attn(para, epi)
                # para, epi = para+para1, epi+epi1
                para, epi = self.co_attn(para, epi)

            # paratope
            # 1. kmer sequential features
            para = kmer_embed(para, self.k4kmer)            # (batch*(para_seq_length-k+1), k, embed_size)
            if self.seq_encoder_type=="transformer":
                para = self.seq_encoder_para(para)          # (batch*(para_seq_length-k+1), k, embed_size)
            elif self.seq_encoder_type=="lstm":
                para, _ = self.seq_encoder_para(para)       # (batch*(para_seq_length-k+1), k, embed_size)
            para = torch.mean(para, dim=1)                  # (batch*(para_seq_length-k+1), 1, embed_size)
            para = para.reshape(-1, para_seq_length-self.k4kmer+1, self.embed_size)
                                                            # (batch, para_seq_length-k+1, embed_size)
            # 2. paratope set interaction
            para = self.para_enc(para)                      # (batch, para_seq_length-k+1, hidden)
            para = self.para_dec(para)                      # (batch, num_ind, embed_size)


            # epitope
            # 1. kmer sequential features
            epi = kmer_embed(epi, self.k4kmer)              # (batch*(epi_seq_length-k+1), k, embed_size)
            if self.seq_encoder_type=="transformer":
                epi = self.seq_encoder_epi(epi)             # (batch*(epi_seq_length-k+1), k, embed_size)
            elif self.seq_encoder_type=="lstm":
                epi, _ = self.seq_encoder_epi(epi)          # (batch*(epi_seq_length-k+1), k, embed_size)
            epi = torch.mean(epi, dim=1)                    # (batch*(epi_seq_length-k+1), 1, embed_size)
            epi = epi.reshape(-1, epi_seq_length-self.k4kmer+1, self.embed_size)
                                                            # (batch, epi_seq_length-k+1, embed_size)
            # 2. epitope set interaction
            epi = self.epi_enc(epi)                         # (batch, epi_seq_length-k+1, hidden)
            epi = self.epi_dec(epi)                         # (batch, num_ind, embed_size)


            # # co-attn
            # if self.use_coattn==True:
            #     para1, epi1 = self.co_attn(para, epi)
            #     # para, epi = para+para1, epi+epi1


            # average for paratope/epitope representation
            para = torch.mean(para, dim=1)          # (batch, embed_size)    
            epi = torch.mean(epi, dim=1)            # (batch, embed_size)

            # output
            x = para * epi                          # (batch, embed_size)
            x = self.output_layer(x)
            
            return x


class SetModel_ablation(nn.Module):
    def __init__(self, embed_size=64, hidden=128, num_layers=2, dropout=0.1, k4kmer=3, use_pretrain=False, use_kmer_embed=True, use_seq_encoder=False, use_coattn=False, seq_encoder_type="transformer", num_heads=4, num_inds=6, num_outputs=6, ln=False):
        super(SetModel_ablation, self).__init__()

        self.use_pretrain = use_pretrain
        self.use_coattn = use_coattn
        self.use_kmer_embed=True
        self.use_seq_encoder=False
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
            if self.use_seq_encoder:
                self.seq_encoder = SequenceEncoder(seq_encoder_type=self.seq_encoder_type, 
                                                    num_layers=num_layers, 
                                                    embed_size=embed_size, 
                                                    hidden=hidden, 
                                                    dropout=dropout, 
                                                    nhead=num_heads)
                self.seq_encoder = SequenceEncoder(seq_encoder_type=self.seq_encoder_type, 
                                                    num_layers=num_layers, 
                                                    embed_size=embed_size, 
                                                    hidden=hidden, 
                                                    dropout=dropout, 
                                                    nhead=num_heads)
        
        # set interaction
        self.para_enc = nn.Sequential(
                ISAB(embed_size, hidden, num_heads, num_inds, ln=ln),
                ISAB(hidden, hidden, num_heads, num_inds, ln=ln))
        
        self.para_dec = nn.Sequential(
                PMA(hidden, num_heads, num_outputs, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                nn.Linear(hidden, embed_size))

        self.epi_enc = nn.Sequential(
                ISAB(embed_size, hidden, num_heads, num_inds, ln=ln),
                ISAB(hidden, hidden, num_heads, num_inds, ln=ln))
        
        self.epi_dec = nn.Sequential(
                PMA(hidden, num_heads, num_outputs, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                nn.Linear(hidden, embed_size))

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
            para = self.embedding(para)             # (batch, para_seq_length, embed_size)
            # 1. kmer sequential features
            if self.use_kmer_embed:
                para = kmer_embed_mean(para, self.k4kmer)    # (batch, para_seq_length-k+1, embed_size)
            if self.use_seq_encoder:
                if self.seq_encoder_type=="transformer":
                    para = self.seq_encoder(para)       # (batch, para_seq_length-k+1, embed_size)
                elif self.seq_encoder_type=="lstm":
                    para, _ = self.seq_encoder(para)    # (batch, para_seq_length-k+1, embed_size)
            para = torch.mean(para, dim=1)          # (batch*(para_seq_length-k+1), 1, embed_size)
            para = para.reshape(-1, para_seq_length-self.k4kmer+1, self.embed_size)
                                                    # (batch, para_seq_length-k+1, embed_size)
            # 2. paratope set interaction
            para = self.para_enc(para)
            para = self.para_dec(para)              # (batch, para_seq_length-k+1, embed_size)


            # epitope
            # 0. kmer embedding
            epi_seq_length = epi.size(1)
            epi = self.embedding(epi)               # (batch, epi_seq_length, embed_size)
            # 1. kmer sequential features
            if self.use_kmer_embed:
                epi = kmer_embed_mean(epi, self.k4kmer)    # (batch, epi_seq_length-k+1, embed_size)
            if self.use_seq_encoder:
                if self.seq_encoder_type=="transformer":
                    epi = self.seq_encoder(epi)       # (batch, epi_seq_length-k+1, embed_size)
                elif self.seq_encoder_type=="lstm":
                    epi, _ = self.seq_encoder(epi)    # (batch, epi_seq_length-k+1, embed_size)
            epi = torch.mean(epi, dim=1)            # (batch*(epi_seq_length-k+1), 1, embed_size)
            epi = epi.reshape(-1, epi_seq_length-self.k4kmer+1, self.embed_size)
                                                    # (batch, epi_seq_length-k+1, embed_size)
            # 2. epitope set interaction
            epi = self.epi_enc(epi)
            epi = self.epi_dec(epi)                 # (batch, epi_seq_length-k+1, embed_size)


            # co-attn
            if self.use_coattn==True:
                para, epi = self.co_attn(para, epi)


            # average for paratope/epitope representation
            para = torch.mean(para, dim=1)          # (batch, 1, embed_size)
            para = para.squeeze(1)                  # (batch, embed_size)
            
            epi = torch.mean(epi, dim=1)            # (batch, 1, embed_size)
            epi = epi.squeeze(1)                    # (batch, embed_size)

            # output
            x = para * epi                          # (batch, embed_size)
            x = self.output_layer(x)
            
            return x