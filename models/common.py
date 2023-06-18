import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

import esm

from datasets import *
from utils import *


def replace_pad(seq):
    l = len(seq)
    seq = seq.replace("/", "")
    seq += "#"*(l - len(seq))

    return seq


# predict word/token representation
def get_representation(esm2, sequence, alphabet):
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(sequence)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on GPU)
    with torch.no_grad():
        results = esm2(batch_tokens.cuda(), repr_layers=[6], return_contacts=True)
    token_representations = results["representations"][6]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

    return sequence_representations


# predict paratope/epitope embedding [batch_size, seq_len, embed_size]
def get_embedding(paratope, epitope, use_subseq=False):
    esm2, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm2.cuda()
    esm2.eval()

    paratope = [("paratope", i[0].replace("#", "<pad>").replace("*", "<unk>").replace("/", "<mask>")) for i in paratope]
    epitope = [("epitope", i[1].replace("#", "<pad>").replace("*", "<unk>").replace("/", "<mask>")) for i in epitope]

    para_embed = get_representation(esm2, paratope, alphabet)
    epi_embed = get_representation(esm2, epitope, alphabet)

    para_embed = torch.vstack(para_embed)
    epi_embed = torch.vstack(epi_embed)

    return para_embed, epi_embed


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = torch.permute(x, (1, 0, 2))
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return torch.permute(x, (1, 0, 2))


class CoAttention(nn.Module):
    def __init__(self, embed_size, output_size, dropout=None):
        super(CoAttention, self).__init__()

        self.dropout = dropout
        self.embed_size = embed_size
        self.linear_a = nn.Linear(embed_size, output_size)
        self.linear_b = nn.Linear(embed_size, output_size)
        self.W = nn.Linear(output_size, output_size)
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, input_a, input_b):
        orig_a = input_a
        orig_b = input_b
        seq_len = orig_a.size()[1]

        input_a = input_a.view(-1, input_a.size()[1], input_a.size()[2])
        input_b = input_b.view(-1, input_b.size()[1], input_b.size()[2])
        
        a_len = input_a.size()[1]
        b_len = input_b.size()[1]
        input_dim = input_a.size()[2]
        max_len = max(a_len, b_len)

        input_a = self.linear_a(input_a)
        input_a = nn.ReLU()(input_a)
        input_b = self.linear_b(input_b)
        input_b = nn.ReLU()(input_b)

        # print("input_a ", input_a.shape)

        dim = input_a.size()[2]

        _b = input_b.permute(0, 2, 1)
        zz = self.W(input_a)
        z = torch.matmul(zz, _b)

        # print("_b ", _b.shape)
        # print("zz ", zz.shape)
        # print("z ", z.shape)

        att_row = torch.mean(z, 1)
        att_col = torch.mean(z, 2)

        # print("att_row, att_col", att_row.shape, att_col.shape)

        a = orig_a * att_row.unsqueeze(2)
        b = orig_b * att_col.unsqueeze(2)

        # print(att_row.unsqueeze(2).shape)
        # print(a.shape, b.shape)

        return a, b


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax, for to reproduce the result of CUB with ResNet10, use 4. see the issue#31 in the github 
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores
    

class TowerBaseModel(nn.Module):
    def __init__(self, 
                 embed_size, 
                 hidden, 
                 encoder, 
                 use_two_towers=False, 
                 mid_coattn=False, 
                 use_coattn=False, 
                 fusion=0, 
                 dropout=0.1):
        super(TowerBaseModel, self).__init__()

        self.embed_size = embed_size
        self.use_two_towers = use_two_towers
        self.mid_coattn = mid_coattn
        self.use_coattn = use_coattn
        self.fusion = fusion                             # 0 - concat ; 1 - dot ; 2 - add dot
        self.dropout = dropout

        if self.use_two_towers==True:
            self.encoder_para, self.encoder_epi = encoder
            # self.encoder_para.train()
            # self.encoder_epi.train()
        else:
            self.encoder = encoder
            # self.encoder.train()

        if self.use_coattn==True:
            if self.mid_coattn==True:
                self.coattn = CoAttention(embed_size=hidden, output_size=hidden)
            else:
                self.coattn = CoAttention(embed_size=embed_size, output_size=embed_size)

        if self.fusion==2:
            self.scale_coef = nn.Parameter(torch.ones(1))

        if self.fusion==0:
            num_neuron = 2*embed_size
        else:
            num_neuron = embed_size
            
        self.output_layer = nn.Sequential(nn.Linear(num_neuron, num_neuron), nn.ReLU(), nn.Dropout(self.dropout), 
                                          nn.Linear(num_neuron, 1), nn.Sigmoid())

    def forward(self, para, epi):

        if self.mid_coattn==True:
            if self.use_two_towers==True:
                para = torch.Tensor([to_onehot(i) for i in para]).int().cuda()
                epi = torch.Tensor([to_onehot(i) for i in epi]).int().cuda()

                para = self.encoder.embedding(para)
                epi = self.encoder.embedding(epi)

                para = self.encoder_para.encoder(para)
                epi = self.encoder_epi.encoder(epi)

                if self.use_coattn==True:
                    para, epi = self.coattn(para, epi)

                para = self.encoder_para.decoder(para)
                epi = self.encoder_epi.decoder(epi)
            else:
                para = torch.Tensor([to_onehot(i) for i in para]).int().cuda()
                epi = torch.Tensor([to_onehot(i) for i in epi]).int().cuda()

                para = self.encoder.embedding(para)
                epi = self.encoder.embedding(epi)

                para = self.encoder.encoder(para)
                epi = self.encoder.encoder(epi)

                if self.use_coattn==True:
                    para, epi = self.coattn(para, epi)

                para = self.encoder.decoder(para)
                epi = self.encoder.decoder(epi)
                
        else:
            if self.use_two_towers==True:
                para = self.encoder_para(para)
                epi = self.encoder_epi(epi)
            else:
                para = self.encoder(para)
                epi = self.encoder(epi)

            # (batch, len, embed_size)

            if self.use_coattn==True:
                para, epi = self.coattn(para, epi)

        # (batch, len, embed_size)

        para = torch.nn.functional.normalize(para, p=2, dim=1)
        epi = torch.nn.functional.normalize(epi, p=2, dim=1)

        # (batch, len, embed_size)

        if len(para.shape)==3:
            para = torch.mean(para, dim=1)
            epi = torch.mean(epi, dim=1)

        # (batch, embed_size)

        if self.fusion==0:
            x = torch.cat([para, epi], dim=1)
        elif self.fusion==1:
            x = para * epi
        elif self.fusion==2:
            x = para + epi + (para * epi) * self.scale_coef

        x = self.output_layer(x)

        return x


# ngram representation
def kmer(seq, k=3):
    ngram = [seq[i:i+k] for i in range(len(seq)-k+1)]
    return ngram

# average contiguous k embedding as token representation
# length of token is k
# (batch, seq_len, hidden) -> (batch, seq_len-k+1, hidden)
def kmer_embed_mean(seqs, k=3):
    ngram_li = []
    for seq in seqs:
        ngram = [torch.mean(seq[i:i+k, :], dim=0) for i in range(len(seq)-k+1)]
        ngram_li.append(torch.vstack(ngram))

    return torch.stack(ngram_li)

# reshape by slicing original sequence with kmer
# (batch, seq_len, hidden) -> (batch*(seq_len-k+1), k, hidden)
def kmer_embed(seqs, k=3):
    ngram_li = []
    for seq in seqs:
        ngram = [seq[i:i+k, :] for i in range(len(seq)-k+1)]
        ngram_li.append(torch.stack(ngram, dim=0))
    
    return torch.vstack(ngram_li)


class SequenceEncoder(nn.Module):
    def __init__(self, seq_encoder_type="lstm", num_layers=2, embed_size=64, hidden=512, dropout=0, nhead=4):
        super(SequenceEncoder, self).__init__()

        self.dropout = dropout
        self.seq_encoder_type = seq_encoder_type

        # transformer
        if self.seq_encoder_type=="transformer":
            self.pos_encoder = PositionalEncoding(d_model=embed_size, dropout=self.dropout)
            encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=nhead, batch_first=True)
            self.seq_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        # lstm
        else:
            self.seq_encoder = nn.LSTM(input_size=embed_size, 
                                       hidden_size=hidden, 
                                       num_layers=1, 
                                       bidirectional=True, 
                                       proj_size=embed_size//2, 
                                       batch_first=True)

    def forward(self, x):
        if self.seq_encoder_type=="transformer":
            x = self.pos_encoder(x)
        
        x = self.seq_encoder(x)

        return x
        