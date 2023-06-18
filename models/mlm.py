import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets import *
from utils import *
from .common import *
from .setmodel import *


class PESILM(nn.Module):
    """
    PESI Language Model
    Masked Language Model
    """

    def __init__(self, 
                 vocab_size=len(vocab), 
                 dim_input=32, 
                 dim_hidden=64, 
                 num_outputs=32, 
                 dim_output=32, 
                 num_inds=6, 
                 num_heads=4, 
                 ln=True, 
                 dropout=0.5):
        """
        :param pesi: PESI model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dim_input)
        
        self.enc = nn.ModuleList([
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)])
        
        self.co_attn = CoAttention(embed_size=dim_hidden, output_size=dim_hidden, return_attn=True)
        
        self.dec = nn.ModuleList([
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_hidden)])

        self.mask_lm = MaskedLanguageModel(dim_hidden, vocab_size)

    def forward(self, x):

        if self.training:
            c_mask = x.ne(0).type(torch.float)
            mask = x.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            c_mask, mask = None, None
        
        # embedding
        x = self.embedding(x)
        # print("within PESILM!!!!!!!", mask.shape, c_mask.shape)
        # encoder
        for enc_layer in self.enc:
            x = enc_layer(x, mask=mask, query_mask=c_mask)
        
        # co-attention
        x, _, attn_list, _ = self.co_attn(x, x)

        # decoder
        for dec_layer in self.dec:
            if isinstance(dec_layer, nn.Linear):
                x = dec_layer(x)
            else:
                x = dec_layer(x, mask=mask, query_mask=c_mask)

        # x, attn_list = self.pesi(x)

        return self.mask_lm(x), attn_list


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))