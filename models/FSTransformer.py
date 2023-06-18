import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets import *
from utils import *
from .common import *


def frame_slice(x, frame_size=6):
    """
    Args:
        x: (B, L, E)
        frame_size: size of frames
    Returns:
        frames: (num_frames*B, frame_size, C)
    """
    B, L, E = x.shape
    x = x.view(B, L // frame_size, frame_size, E)
    frames = x.contiguous().view(-1, frame_size, E)

    return frames


class FSAttention(nn.Module):
    def __init__(self, embed_size=64, nhead=2, dropout=0.1):
        super(FSAttention, self).__init__()

        self.embed_size = embed_size
        self.nhead = nhead
        self.dropout = dropout

    def forward(self, x):
        x = frame_slice(para)
        # (batch*6, para_seq_length//6, embed_size)



class FSTransformer(nn.Module):
    def __init__(self, embed_size=64, num_encoder_layers=2, nhead=2, dropout=0.1, use_coattn=False):
        super(FSTransformer, self).__init__()

        self.use_coattn = use_coattn
        
        self.embedding = nn.Embedding(len(vocab), embed_size)

        self.pos_para = PositionalEncoding(d_model=embed_size, dropout=dropout)
        para_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, batch_first=True)
        self.transformer_para = nn.TransformerEncoder(encoder_layer=para_layer, num_layers=num_encoder_layers)

        self.pos_epi = PositionalEncoding(d_model=embed_size, dropout=dropout)
        epi_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, batch_first=True)
        self.transformer_epi = nn.TransformerEncoder(encoder_layer=epi_layer, num_layers=num_encoder_layers)

        if self.use_coattn==True:
            self.co_attn = CoAttention(embed_size=embed_size, output_size=embed_size)

        self.output_layer = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(dropout), \
                                          nn.Linear(embed_size//2, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        para, epi = torch.Tensor([to_onehot(i) for i in para]).int().cuda(), torch.Tensor([to_onehot(i) for i in epi]).int().cuda()

        # embedding
        para = self.embedding(para)
        # (batch, para_seq_length, embed_size)
        epi = self.embedding(epi)
        # (batch, epi_seq_length, embed_size)

        # co_attn
        if self.use_coattn:
            para, epi = self.co_attn(para, epi)

        # transformer
        para = self.transformer_para(para)        
        # (batch, para_seq_length, embed_size)
        para = self.transformer_para(para)
        # (batch, para_seq_length, embed_size)
        para = torch.mean(para, dim=1)
        # (batch, embed_size)
        epi = self.transformer_epi(epi)        
        # (batch, epi_seq_length, embed_size)
        epi = torch.mean(epi, dim=1)
        # (batch, embed_size)

        x = para * epi
        
        x = self.output_layer(x)
        
        return x