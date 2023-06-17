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
from utils_func import *


# SAbDab
class SAbDabDataset(torch.utils.data.Dataset):
    def __init__(
            self, \
            data, \
            epi_seq_length=800, \
            seq_clip_mode=1, \
            neg_sample_mode=1, \
            kfold=10, \
            holdout_fold=0, \
            is_train_test_full="train", \
            is_shuffle=False, \
            folds_path=None, \
            save_path=None, \
            K=48, \
            data_augment=False, \
            augment_ratio=0.5, \
            use_cache=False, \
            use_pair=False, \
            num_neg=1, \
            pretrain_mode='normal'
        ):

        self.pretrain_mode = pretrain_mode
        use_pair = True if self.pretrain_mode=='pair' else False

        # load folds if existing else preprocessing
        if folds_path==None:
            print("folds_path none, preprocessing...")
            self.pair_data = get_pair(data=data, epi_seq_length=epi_seq_length, \
                seq_clip_mode=seq_clip_mode, neg_sample_mode=neg_sample_mode, num_neg=num_neg, K=K, use_pair=use_pair, \
                use_cache=use_cache)
            if save_path!=None:
                pickle.dump(self.pair_data, open(save_path, "wb"))
            else:
                print("save to ./data/processed_data_clip{}_neg{}_usepair{}_num_neg{}.pkl".format(seq_clip_mode, neg_sample_mode, use_pair, num_neg))
                pickle.dump(self.pair_data, open("./data/processed_data_clip{}_neg{}_usepair{}_num_neg{}.pkl".format(seq_clip_mode, neg_sample_mode, use_pair, num_neg), "wb"))
        else:
            print("loading preprocessed data from {}".format(folds_path))
            self.pair_data = pickle.load(open(folds_path, "rb"))
            

        if is_shuffle==True:
            random.shuffle(self.pair_data)

        self.is_train_test_full = is_train_test_full
        self.use_pair = True if self.pretrain_mode=='pair' else False

        if use_pair==False:
            self.label = torch.Tensor([pair[-1] for pair in self.pair_data])
            self.data = [(pair[0], pair[1]) for pair in self.pair_data]


        if self.is_train_test_full=="train" or self.is_train_test_full=="test":

            # train data
            self.data_folds = []
            self.label_folds = []
            for k in range(kfold):
                data_tmp = self.data[k*int((1/kfold)*len(self.data)):(k+1)*int((1/kfold)*len(self.data))]
                label_tmp = self.label[k*int((1/kfold)*len(self.label)):(k+1)*int((1/kfold)*len(self.label))]
                self.data_folds.append(data_tmp)
                self.label_folds.append(label_tmp)

            # test data
            self.test_data = self.data_folds.pop(holdout_fold)
            self.test_label = self.label_folds.pop(holdout_fold)
            self.train_data = []
            for i in range(len(self.data_folds)):
                for j in range(len(self.data_folds[i])):
                    self.train_data.append(self.data_folds[i][j])
            self.train_label = torch.hstack(self.label_folds)

            # data augmentation
            if data_augment==True:
                print(int(augment_ratio*len(self.train_data)))
                tmp = self.train_data[:int(augment_ratio*len(self.train_data))]
                tmp = [(entry[0], get_random_sequence(length=epi_seq_length)) for entry in tmp]
                # print(len(tmp), tmp)
                # print(self.data.shape, self.data)
                self.train_data = self.train_data + tmp
                self.train_label = torch.hstack([self.train_label, torch.Tensor([0]*int(augment_ratio*len(self.train_data)))])
            
    def __len__(self):
        if self.pretrain_mode=='normal' or self.pretrain_mode=='CLIP':
            if self.is_train_test_full=="train":
                return len(self.train_data)
            elif self.is_train_test_full=="test":
                return len(self.test_data)
            else:
                return len(self.data)
        elif self.pretrain_mode=='pair':
            return len(self.pair_data)
        else:
            print("wrong pretrain_mode in sabdabdataset")
            exit()
    
    def __getitem__(self, idx):
        if self.pretrain_mode=='normal':
            if self.is_train_test_full=="train":
                return self.train_data[idx][0], self.train_data[idx][1], self.train_label[idx]
            elif self.is_train_test_full=="test":
                return self.test_data[idx][0], self.test_data[idx][1], self.test_label[idx]
            else:
                return self.data[idx][0], self.data[idx][1], self.label[idx]
        elif self.pretrain_mode=='pair':
            return self.pair_data[idx][0], self.pair_data[idx][1], self.pair_data[idx][2]
        else:
            print("wrong pretrain_mode in sabdabdataset")
            exit()


if __name__=="__main__":
    # SAbDabDataset
    # data = pickle.load(open("../../MSAI_Project/codes/data/data.json", "rb"))
    # dataset = SAbDabDataset(data)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # t=0
    # for i, (para, epi, label) in enumerate(dataloader):
    #     print(i)
    #     print("para", para)
    #     print("epi", epi)
    #     print("label", label)
    #     t += 1

    #     if t==1:
    #         break

    # SeqDataset
    data = pickle.load(open("../../MSAI_Project/codes/data/data.json", "rb"))
    dataset = SeqDataset(data_path="../data/SARS-SAbDab_Shaun/CoV-AbDab_extract.csv", seq_length=128, \
                    kfold=10, holdout_fold=0, is_train=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    t=0
    for i, (para, epi, label) in enumerate(dataloader):
        print(i)
        print("para", para)
        print("epi", epi)
        print("label", label)
        t += 1

        if t==1:
            break