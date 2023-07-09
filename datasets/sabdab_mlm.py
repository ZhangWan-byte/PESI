import torch
import torch.nn as nn
import torch.optim as optim

import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import *
from .utils_func import *


class SAbDabDataset_MLM(torch.utils.data.Dataset):
    def __init__(self, corpus_path, vocab, train_test='train'):
        self.corpus_path = corpus_path
        self.vocab = vocab
        self.train_test = train_test

        self.lines = pickle.load(open(corpus_path, "rb"))
        self.lines = get_pair(data=self.lines, 
                              epi_seq_length=72, 
                              seq_clip_mode=1, 
                              neg_sample_mode=0, 
                              num_neg=1, 
                              K=48, 
                              use_cache=False, 
                              use_pair=False, 
                              only_epitope=True)
        self.lines = [i[1].strip("#") for i in self.lines]

        random.seed(42)
        random.shuffle(self.lines)
        
        if self.train_test=='train':
            self.lines = self.lines[:int(0.9*len(self.lines))]
        if self.train_test=='test':
            self.lines = self.lines[int(0.9*len(self.lines)):]
        self.len_lines = len(self.lines)

    def __len__(self):
        return self.len_lines

    def __getitem__(self, item):

        t = self.lines[item]
        t1_random, t1_label = self.random_word(t)

        # + = start, - = end, * = mask, # = pad, / = sep
        mlm_input = [self.vocab['+']] + t1_random + [self.vocab['-']]
        mlm_label = [self.vocab['#']] + t1_label + [self.vocab['#']]

        return mlm_input, mlm_label


    def random_word(self, sentence):
        tokens = sentence.split()
        tokens_len = [len(token) for token in tokens]
        chars = [char for char in sentence]
        output_label = []

        for i, char in enumerate(chars):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    chars[i] = vocab['*']                                   # self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    chars[i] = random.randrange(len(vocab))                 # self.vocab.vocab_size

                # 10% randomly change token to current token
                else:
                    chars[i] = vocab[char]                                  # self.vocab.char2index(char)

                output_label.append(vocab[char])                            # self.vocab.char2index(char)

            else:
                chars[i] = vocab[char]                                      # self.vocab.char2index(char)
                output_label.append(0)

        return chars, output_label
