import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from datasets import *
from utils import *
from .common import *


class PESILM(nn.Module):
    """
    PESI Language Model
    Masked Language Model
    """

    def __init__(self, pesi, vocab_size=len(vocab)):
        """
        :param pesi: PESI model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.pesi = pesi
        self.mask_lm = MaskedLanguageModel(self.pesi.hidden, vocab_size)
        self.init_model()

    def forward(self, x, pos):
        x, attn_list = self.pesi(x, pos)
        return self.mask_lm(x), attn_list

    def init_model(self):
        un_init = ['bert.embed.weight', 'bert.pos_emb.weight']
        for n, p in self.named_parameters():
            if n not in un_init and p.dim() > 1:
                nn.init.xavier_uniform_(p)


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