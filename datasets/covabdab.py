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


# CoV-AbDab
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_path="../../MSAI_Project/codes/data/sequence_pairs.json", 
                 kfold=10, 
                 holdout_fold=0, 
                 is_train_test_full="train", 
                 use_pair=False, 
                 balance_samples=False, 
                 balance_ratio=1, 
                 pretrain_mode='normal', 
                 use_part='pretrain'):
        
        self.is_train_test_full = is_train_test_full
        self.pretrain_mode = pretrain_mode
        self.balance_samples = balance_samples
        self.use_part = use_part

        self.data_df = pd.read_csv(data_path)

        # use part of dataset as validation for pre-training
        # rest of data for k-fold
        if self.use_part == 'pretrain':
            self.data_df = self.data_df.sample(frac=0.1, random_state=42)
        elif self.use_part == 'finetune':
            self.part = self.data_df.sample(frac=0.1, random_state=42)
            self.data_df = self.data_df.drop(self.part.index)
        else:
            pass
        # print(self.data_df.shape)
        # balance samples to a certain ratio, e.g., pos:neg=1:1
        if self.balance_samples:
            self.balance(ratio=balance_ratio, num_index=self.data_df.shape[0])

        # CLIP only uses positive pairs
        if self.pretrain_mode == 'CLIP':
            self.data_df = self.data_df[self.data_df["Class"]==1]
        
        
        self.data = self.data_df.sample(frac=1, random_state=42)

        self.label = torch.Tensor(self.data["Class"].values)
        self.data = pd.concat([self.data_df["Paratope"], \
                               self.data_df["Epitope"]], axis=1)

        # split to train and test, if needed
        if self.is_train_test_full=="train" or self.is_train_test_full=="test":
            self.data_folds = []
            self.label_folds = []
            for k in range(kfold):
                data_tmp = self.data[k*int(0.1*self.data.shape[0]):(k+1)*int(0.1*self.data.shape[0])]
                label_tmp = self.label[k*int(0.1*self.label.shape[0]):(k+1)*int(0.1*self.label.shape[0])]
                self.data_folds.append(data_tmp)
                self.label_folds.append(label_tmp)
                
    #         print(self.data_folds[0])

            self.test_data = self.data_folds.pop(holdout_fold)
            self.test_label = self.label_folds.pop(holdout_fold)
            self.train_data = pd.concat(self.data_folds)
            self.train_label = torch.hstack(self.label_folds)

        # re-organise to pair data, if needed        
        if self.pretrain_mode=='pair':
            self.pair_data = []

            for i in range(len(self.data)):
                if self.label[i]==1:
                    paratope = self.data.iloc[i][0]
                    antigen_pos = self.data.iloc[i][1]

                    j = random.randint(0, len(self.data)-1)
                    antigen_neg = self.data.iloc[j][1]
                    while seq_sim(antigen_neg, antigen_pos)>=0.5:
                        j = random.randint(0, len(self.data)-1)
                        antigen_neg = self.data.iloc[j][1]
                    
                    self.pair_data.append((paratope, antigen_pos, antigen_neg))
        
        # save dataset
        # self.data_df.to_csv(os.path.join(save_path, "cov-dabdab_df.csv"))

    def balance(self, ratio, num_index):
        """
        ratio: ratio of neg:pos
        """
        num_pos = len(self.data_df[self.data_df["Class"]==1])
        num_neg = len(self.data_df[self.data_df["Class"]==0])
        num_neg_sample = num_pos*ratio - num_neg

        # sample negative para-epi pairs and add into dataset
        append_samples = []
        self.data_df = self.data_df.sample(frac=1, random_state=42)
        for i in range(self.data_df.shape[0]):
            if self.data_df.iloc[i]["Class"]==1:
                paratope = self.data_df.iloc[i]["Paratope"]
                antigen_pos = self.data_df.iloc[i]["Epitope"]

                t = 0
                while t<num_neg_sample:
                    j = random.randint(0, len(self.data_df)-1)
                    antigen_neg = self.data_df.iloc[j]["Epitope"]
                    while seq_sim(antigen_neg, antigen_pos)>=0.5:
                        j = random.randint(0, len(self.data_df)-1)
                        antigen_neg = self.data_df.iloc[j][1]
                        
                    append_samples.append((paratope, antigen_neg))
                    t += 1

        self.df_append = pd.DataFrame({'Index': [num_index+i for i in range(len(append_samples))], 
                                       'AB_name': ['neg samples']*len(append_samples), 
                                       'Class': [0]*len(append_samples), 
                                       'Paratope': [i[0] for i in append_samples],
                                       'Epitope': [i[1] for i in append_samples]})
        self.data_df = self.data_df.append(self.df_append, ignore_index=True)
        

    def __len__(self):
        if self.pretrain_mode=='normal' or self.pretrain_mode=='CLIP':
            if self.is_train_test_full=="train":
                return self.train_data.shape[0]
            elif self.is_train_test_full=="test":
                return self.test_data.shape[0]
            else:
                return self.data.shape[0]
        # elif self.pretrain_mode=='CLIP':
        #     pass
        elif self.pretrain_mode=='pair':
            return len(self.pair_data)
        else:
            print("wrong pretrain_mode in seqdataset")
            exit()
    
    def __getitem__(self, idx):
        if self.pretrain_mode=='normal' or self.pretrain_mode=='CLIP':
            if self.is_train_test_full=="train":
                return self.train_data.iloc[idx][0], self.train_data.iloc[idx][1], self.train_label[idx]
            elif self.is_train_test_full=="test":
                return self.test_data.iloc[idx][0], self.test_data.iloc[idx][1], self.test_label[idx]
            else:
                return self.data.iloc[idx][0], self.data.iloc[idx][1], self.label[idx]
        # elif self.pretrain_mode=='CLIP':
        #     pass
        elif self.pretrain_mode=='pair':
            return self.pair_data[idx][0], self.pair_data[idx][1], self.pair_data[idx][2]
        else:
            print("wrong pretrain_mode in seqdataset")
            exit()