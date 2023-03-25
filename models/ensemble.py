import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import *
from utils import *
from .common import *
from .masonscnn import *
from .ITransformer import *
from .setmodel import *



class EnsembleModel(nn.Module):
    def __init__(self, 
                 embed_size=64, 
                 hidden=128, 
                 max_len=100, 
                 num_encoder_layers=2, 
                 num_heads=2, 
                 num_inds=6, 
                 num_outputs=6, 
                 ln=False, 
                 dropout=0.1, 
                 use_coattn=False):
        super(EnsembleModel, self).__init__()

        self.max_len = max_len
        self.use_coattn = use_coattn
        
        self.embedding = nn.Embedding(len(vocab), embed_size)


        # Frame feature - CNN
        self.Frame_para = CNNmodule(in_channel=27, kernel_width=len(vocab), l=self.max_len, out_channels=embed_size)
        self.Frame_epi = CNNmodule(in_channel=27, kernel_width=len(vocab), l=self.max_len, out_channels=embed_size)

        # Sequence feature - Transformer
        self.pos_para = PositionalEncoding(d_model=embed_size, dropout=dropout)        
        para_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
        self.transformer_para = nn.TransformerEncoder(encoder_layer=para_layer, num_layers=num_encoder_layers)
        self.Seq_para = nn.Sequential(self.pos_para, self.transformer_para)

        self.pos_epi = PositionalEncoding(d_model=embed_size, dropout=dropout)
        epi_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, batch_first=True)
        self.transformer_epi = nn.TransformerEncoder(encoder_layer=epi_layer, num_layers=num_encoder_layers)
        self.Seq_epi = nn.Sequential(self.pos_epi, self.transformer_epi)

        # Set feature - SetTransformer
        # self.para_enc = nn.Sequential(
        #         ISAB(embed_size, hidden, num_heads, num_inds, ln=ln),
        #         ISAB(hidden, hidden, num_heads, num_inds, ln=ln))
        
        # self.para_dec = nn.Sequential(
        #         PMA(hidden, num_heads, num_outputs, ln=ln),
        #         SAB(hidden, hidden, num_heads, ln=ln),
        #         SAB(hidden, hidden, num_heads, ln=ln),
        #         nn.Linear(hidden, embed_size))

        # self.Set_para = nn.Sequential(self.para_enc, self.para_dec)

        self.Set_para = nn.Sequential(
                ISAB(embed_size, hidden, num_heads, num_inds, ln=ln),
                ISAB(hidden, hidden, num_heads, num_inds, ln=ln), 
                PMA(hidden, num_heads, num_outputs, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                nn.Linear(hidden, embed_size)
        )

        # self.epi_enc = nn.Sequential(
        #         ISAB(embed_size, hidden, num_heads, num_inds, ln=ln),
        #         ISAB(hidden, hidden, num_heads, num_inds, ln=ln))
        
        # self.epi_dec = nn.Sequential(
        #         PMA(hidden, num_heads, num_outputs, ln=ln),
        #         SAB(hidden, hidden, num_heads, ln=ln),
        #         SAB(hidden, hidden, num_heads, ln=ln),
        #         nn.Linear(hidden, embed_size))

        # self.Set_epi = nn.Sequential(self.epi_enc, self.epi_dec)

        self.Set_epi = nn.Sequential(
                ISAB(embed_size, hidden, num_heads, num_inds, ln=ln),
                ISAB(hidden, hidden, num_heads, num_inds, ln=ln), 
                PMA(hidden, num_heads, num_outputs, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                nn.Linear(hidden, embed_size)
        )

        # co-attn

        if self.use_coattn==True:
            self.co_attn_seq = CoAttention(embed_size=embed_size, output_size=embed_size)
            self.co_attn_set = CoAttention(embed_size=embed_size, output_size=embed_size)

        self.output_layer = nn.Sequential(nn.Linear(embed_size*3, embed_size*2), nn.ReLU(), nn.Dropout(dropout), \
                                          nn.Linear(embed_size*2, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        frame_para, frame_epi = copy.copy(para), copy.copy(epi)

        # embedding
        para = torch.Tensor([to_onehot(i) for i in para]).int().cuda()
        epi = torch.Tensor([to_onehot(i) for i in epi]).int().cuda()
        para = self.embedding(para)                 # (batch, para_seq_length, embed_size)
        epi = self.embedding(epi)                   # (batch, epi_seq_length, embed_size)

        frame_para = [seq_pad_clip(i, target_length=self.max_len) for i in frame_para]
        frame_epi = [seq_pad_clip(i, target_length=self.max_len) for i in frame_epi]
        frame_para = torch.Tensor([to_onehot(i, mode=1) for i in frame_para]).float().cuda()        
        frame_epi = torch.Tensor([to_onehot(i, mode=1) for i in frame_epi]).float().cuda()

        batch_size = frame_para.size()[0]

        # Para
        frame_para = self.Frame_para(frame_para)
        seq_para = self.Seq_para(para)        
        set_para = self.Set_para(para)

        # epi
        frame_epi = self.Frame_epi(frame_epi)
        seq_epi = self.Seq_epi(epi)
        set_epi = self.Set_epi(epi)

        # co-attn
        if self.use_coattn==True:
            seq_para, seq_epi = self.co_attn_seq(seq_para, seq_epi)
            set_para, set_epi = self.co_attn_set(set_para, set_epi)

        # unify dim
        frame_para = frame_para.view(batch_size, -1)    # (batch, embed_size//2)
        frame_epi = frame_epi.view(batch_size, -1)      # (batch, embed_size//2)
        seq_para = torch.mean(seq_para, dim=1)          # (batch, embed_size)
        set_para = torch.mean(set_para, dim=1)          # (batch, embed_size)
        seq_epi = torch.mean(seq_epi, dim=1)            # (batch, embed_size)
        set_epi = torch.mean(set_epi, dim=1)            # (batch, embed_size)

        # concat
        frame_feats = torch.cat([frame_para, frame_epi], dim=1)     # (batch, embed_size)
        seq_feats = seq_para * seq_epi                              # (batch, embed_size)
        set_feats = set_para * set_epi                              # (batch, embed_size)
        
        # output
        x = torch.cat([frame_feats, seq_feats, set_feats], dim=1)
        
        x = self.output_layer(x)
        
        return x


class PESI(nn.Module):
    def __init__(self, 
                 embed_size=64, 
                 hidden=128, 
                 max_len=100,  
                 num_heads=2, 
                 num_inds=6, 
                 num_outputs=6, 
                 ln=False, 
                 dropout=0.1, 
                 use_coattn=False):
        super(PESI, self).__init__()

        self.max_len = max_len
        self.use_coattn = use_coattn
        
        self.embedding = nn.Embedding(len(vocab), embed_size)


        # Frame feature - CNN
        self.Frame_para = CNNmodule(in_channel=27, kernel_width=len(vocab), l=self.max_len, out_channels=64, out_size=hidden)
        self.Frame_epi = CNNmodule(in_channel=27, kernel_width=len(vocab), l=self.max_len, out_channels=64, out_size=hidden)

        # Set feature - SetTransformer
        # self.para_enc = nn.Sequential(
        #         ISAB(embed_size, hidden, num_heads, num_inds, ln=ln),
        #         ISAB(hidden, hidden, num_heads, num_inds, ln=ln))
        
        # self.para_dec = nn.Sequential(
        #         PMA(hidden, num_heads, num_outputs, ln=ln),
        #         SAB(hidden, hidden, num_heads, ln=ln),
        #         SAB(hidden, hidden, num_heads, ln=ln),
        #         nn.Linear(hidden, hidden))

        # self.Set_para = nn.Sequential(self.para_enc, self.para_dec)

        self.Set_para = nn.Sequential(
                ISAB(embed_size, hidden, num_heads, num_inds, ln=ln),
                ISAB(hidden, hidden, num_heads, num_inds, ln=ln), 
                PMA(hidden, num_heads, num_outputs, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                nn.Linear(hidden, hidden)
        )

        # self.epi_enc = nn.Sequential(
        #         ISAB(embed_size, hidden, num_heads, num_inds, ln=ln),
        #         ISAB(hidden, hidden, num_heads, num_inds, ln=ln))
        
        # self.epi_dec = nn.Sequential(
        #         PMA(hidden, num_heads, num_outputs, ln=ln),
        #         SAB(hidden, hidden, num_heads, ln=ln),
        #         SAB(hidden, hidden, num_heads, ln=ln),
        #         nn.Linear(hidden, hidden))

        # self.Set_epi = nn.Sequential(self.epi_enc, self.epi_dec)

        self.Set_epi = nn.Sequential(
                ISAB(embed_size, hidden, num_heads, num_inds, ln=ln),
                ISAB(hidden, hidden, num_heads, num_inds, ln=ln), 
                PMA(hidden, num_heads, num_outputs, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                SAB(hidden, hidden, num_heads, ln=ln),
                nn.Linear(hidden, hidden)
        )

        # co-attn
        if self.use_coattn==True:
        #     self.co_attn_frame = CoAttention(embed_size=hidden, output_size=hidden)
            self.co_attn_set = CoAttention(embed_size=hidden, output_size=hidden)

        # merge
        self.cross_scale_merge = nn.Parameter(torch.ones(1))
        self.frame_linear = nn.Linear(hidden*2, hidden)
        self.set_linear = nn.Linear(hidden*2, hidden)

        # output
        # self.output_layer = nn.Sequential(nn.Linear(embed_size*2, embed_size*2), nn.ReLU(), nn.Dropout(dropout), \
        #                                   nn.Linear(embed_size*2, 1), nn.Sigmoid())
        self.output_layer = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), \
                                          nn.Linear(hidden, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        frame_para, frame_epi = copy.copy(para), copy.copy(epi)

        # embedding
        para = torch.Tensor([to_onehot(i) for i in para]).int().cuda()
        epi = torch.Tensor([to_onehot(i) for i in epi]).int().cuda()
        para = self.embedding(para)                 # (batch, para_seq_length, embed_size)
        epi = self.embedding(epi)                   # (batch, epi_seq_length, embed_size)

        frame_para = [seq_pad_clip(i, target_length=self.max_len) for i in frame_para]
        frame_epi = [seq_pad_clip(i, target_length=self.max_len) for i in frame_epi]
        frame_para = torch.Tensor([to_onehot(i, mode=1) for i in frame_para]).float().cuda()        
        frame_epi = torch.Tensor([to_onehot(i, mode=1) for i in frame_epi]).float().cuda()

        batch_size = frame_para.size()[0]

        # print(frame_para.shape, frame_epi.shape)

        # Para
        frame_para = self.Frame_para(frame_para)       # (batch, hidden)
        set_para = self.Set_para(para)                 # (batch, num_inds, embed_size)
        # print(frame_para.shape, set_para.shape)

        # epi
        frame_epi = self.Frame_epi(frame_epi)           # (batch, hidden)
        set_epi = self.Set_epi(epi)                     # (batch, num_inds, embed_size)
        # print(frame_epi.shape, set_epi.shape)

        # co-attn
        if self.use_coattn==True:
            set_para, set_epi = self.co_attn_set(set_para, set_epi)

        # unify dim
        frame_para = frame_para.view(batch_size, -1)    # (batch, hidden)
        frame_epi = frame_epi.view(batch_size, -1)      # (batch, hidden)
        set_para = torch.mean(set_para, dim=1)          # (batch, embed_size)
        set_epi = torch.mean(set_epi, dim=1)            # (batch, embed_size)
        # print(frame_para.shape, frame_epi.shape, set_para.shape, set_epi.shape)

        # concat
        frame_feats = torch.cat([frame_para, frame_epi], dim=1)     # (batch, hidden*2)
        frame_feats = self.frame_linear(frame_feats)                # (batch, hidden)
        set_feats = torch.cat([set_para, set_epi], dim=1)           # (batch, hidden*2)
        set_feats = self.frame_linear(set_feats)                    # (batch, hidden)
        
        # fusion
        # print(frame_feats.shape, set_feats.shape)
        x = frame_feats + set_feats + (frame_feats * set_feats) * self.cross_scale_merge
        
        # x = torch.cat([frame_feats, set_feats], dim=1)
        # print(frame_feats.shape, set_feats.shape)
        # print(x.shape)

        # prediction
        x = self.output_layer(x)
        
        return x