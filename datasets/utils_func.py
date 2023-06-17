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


def get_random_sequence(length=48):
    candidates = "".join([k for k in vocab.keys()])
    candidates = candidates[:-3]
    antigen_neg = "".join(random.choices(candidates, k=length))

    return antigen_neg


def get_pair(data, epi_seq_length=800, seq_clip_mode=1, neg_sample_mode=1, num_neg=1, K=48, use_cache=False, use_pair=False):
    
    """process original data to format in pairs

    :param data: original data
        ['pdb', 'Hchain', 'Lchain', 'Achain', 'Hseq', 'Lseq', 'Aseq', 'L1', 'L2', 'L3', 'H1', 'H2', 'H3', 'Hpos', 'Lpos', 'Apos']
        "Apos": [N, CA, C, O]
    :param epi_seq_length: epitope sequence length, defaults to 800
    :param seq_clip_mode: padding antigen seq if shorter than L else 0 - random sampling / 1 - k nearest amino acids, defaults to 1
    :param neg_sample_mode: 0-random sampling from dataset / 1 - random sequence / 2 - choose from BLAST, defaults to 1
    :return: [(paratope, antigen_pos, 1), (paratope, antigen_neg, 0), ...]
    :return: [(paratope, antigen_pos, antigen_neg)]
    """
    
    pair_data = []

    # seq_clip_mode
    # 0 - random sample amino acids
    if seq_clip_mode==0:
        pass
    # 1 - k nearest amino acids
    elif seq_clip_mode==1:
        if use_cache==False:
            data = get_knearest_epi(data, K=K)
            pickle.dump(data, open("./data/tmp_knnepi.pkl", "wb"))
        else:
            print("loading ./data/tmp_knnepi.pkl as data containing knn epitope")
            data = pickle.load(open("./data/tmp_knnepi.pkl", "rb"))
    else:
        print("Not Implemented seq_clip_mode number!")


    print("Start getting pair data...")
    print("seq_clip_mode: {}\tneg_sample_mode: {}\tuse_pair: {}\t".format(seq_clip_mode, neg_sample_mode, use_pair))
    for i in tqdm(range(len(data))):

        # paratope
        # paratope = data[i]["Hseq"][0] + "/" + data[i]["Lseq"][0]
        paratope = "/".join([data[i]["H1"], data[i]["H2"], data[i]["H3"], data[i]["L1"], data[i]["L2"], data[i]["L3"]])

        # epitope - positive sample
        if seq_clip_mode==0:
            antigen_pos = "/".join(data[i]["Aseq"])
            antigen_pos = seq_pad_clip(seq=antigen_pos, target_length=epi_seq_length)
        elif seq_clip_mode==1:
            antigen_pos = data[i]["epitope"]
            antigen_pos = seq_pad_clip(seq=antigen_pos, target_length=epi_seq_length)
        else:
            print("Not Implemented seq_clip_mode!")

        # epitope - negative sample
        antigen_negs = []
        # 0 - random sample amino acids
        if seq_clip_mode==0:
            # 0 - sample from all epitope seqs
            if neg_sample_mode==0:
                
                for _ in range(num_neg):
                    j = random.randint(0, len(data)-1)
                    antigen_neg = "/".join(data[j]["Aseq"])
                    
                    # re-sample if sim score >= 0.9
                    while seq_sim(antigen_neg, antigen_pos)>=0.5:
                        j = random.randint(0, len(data)-1)
                        antigen_neg = "/".join(data[j]["Aseq"])

                    antigen_neg = seq_pad_clip(seq=antigen_neg, target_length=epi_seq_length)
                    antigen_negs.append(antigen_neg)
            # 1 - random sequence
            elif neg_sample_mode==1:
                
                for _ in range(num_neg):
                    # candidates = "".join([k for k in vocab.keys()])
                    # antigen_neg = "".join(random.choices(candidates, k=epi_seq_length))
                    antigen_neg = get_random_sequence(length=epi_seq_length)
                    antigen_neg = seq_pad_clip(seq=antigen_neg, target_length=epi_seq_length)
                    antigen_negs.append(antigen_neg)
            # 2 - BLAST
            else:
                print("Not Implemented BLAST!")
                pass
        # 1 - k nearest amino acids
        elif seq_clip_mode==1:
            # 0 - sample from all epitope seqs
            if neg_sample_mode==0:
                for _ in range(num_neg):
                    j = random.randint(0, len(data)-1)
                    antigen_neg = data[j]["epitope"]
                    
                    while seq_sim(antigen_neg, antigen_pos)>=0.5:
                        j = random.randint(0, len(data)-1)
                        antigen_neg = data[j]["epitope"]

                    antigen_neg = seq_pad_clip(seq=antigen_neg, target_length=epi_seq_length)
                    antigen_negs.append(antigen_neg)
            # 1 - random sequence
            elif neg_sample_mode==1:
                for _ in range(num_neg):
                    candidates = "".join([k for k in vocab.keys()])
                    antigen_neg = "".join(random.choices(candidates, k=epi_seq_length))
                    antigen_negs.append(antigen_neg)
            # 2 - BLAST
            else:
                print("Not Implemented BLAST!")
                pass

        # append to pair_data after removing redundant samples
        # redundancy - 1. >=90%sim for paratope; 2. >=90%sim for epitope; 3. same_label

        if use_pair==False:
            redundant_pos = False
            redundant_negs = [False]*num_neg
            for i in range(len(pair_data)):
                if seq_sim(pair_data[i][0], paratope)>=0.9 and seq_sim(pair_data[i][1], antigen_pos)>=0.9 and pair_data[i][2]==1:
                    redundant_pos = True
                    break
                t = 0
                for antigen_neg in antigen_negs:
                    if seq_sim(pair_data[i][0], paratope)>=0.9 and seq_sim(pair_data[i][1], antigen_neg)>=0.9 and pair_data[i][2]==0:
                        redundant_negs[t] = True
                    t +=1
            if redundant_pos==False:
                pair_data.append((paratope, antigen_pos, 1))
            for t in range(len(redundant_negs)):
                if redundant_negs[t]==False:
                    pair_data.append((paratope, antigen_negs[t], 0))
        else:
            redundant = [False]*num_neg
            for i in range(len(pair_data)):
                t = 0
                for antigen_neg in antigen_negs:
                    if seq_sim(pair_data[i][0], paratope)>=0.9 and \
                       seq_sim(pair_data[i][1], antigen_pos)>=0.9 and \
                       seq_sim(pair_data[i][2], antigen_neg)>=0.9:
                            redundant[t] = True
                    t += 1

            for t in range(len(redundant)):
                if redundant[t]==False:
                    pair_data.append((paratope, antigen_pos, antigen_negs[t]))
        
    return pair_data


def my_pad_sequence(seqs):
    max_len = max(list(map(lambda x:len(x), seqs)))
    
    seqs = list(map(lambda x:"+"+x.strip("#")+"#"*(max_len-len(x.strip("#")))+"-", seqs))
    
    return seqs

def augment_fn(seq):
    # left-right flipping
    if random.random()<=0.5:
        return seq[::-1]
    else:
        return seq


def collate_fn(batch, mode=0, use_augment=False):

    paras = [b[0] for b in batch]
    epis = [b[1] for b in batch]

    if use_augment==True:
        paras = list(map(augment_fn, paras))
        epis = list(map(augment_fn, epis))

    # +ABCD-###
    if mode==0:
        labels = torch.hstack([b[2] for b in batch])
        max_len = max(max(list(map(lambda x:len(x), paras))), max(list(map(lambda x:len(x), epis))))

        paras = ["+"+i.strip("#")+"-"+"#"*(max_len-len(i.strip("#"))) for i in paras]
        epis = ["+"+i.strip("#")+"-"+"#"*(max_len-len(i.strip("#"))) for i in epis]

        new_batch = [paras, epis, labels]

        return new_batch
    
    # padding for six CDRs
    if mode==1:
        paras = [(p.split("/"), max(list(map(lambda x:len(x), p.split("/"))))) for p in paras]
        paras = list(map(my_pad_sequence, paras))
        labels = [b[2] for b in batch]
        new_batch = [paras, epis, labels]

        return new_batch
    
def pair_collate_fn(batch, mode=0):

    paras = [b[0] for b in batch]
    epis_pos = [b[1] for b in batch]
    epis_neg = [b[2] for b in batch]


    # +ABCD-###
    if mode==0:
        max_len = max(max(list(map(lambda x:len(x), paras))), 
                      max(list(map(lambda x:len(x), epis_pos))), 
                      max(list(map(lambda x:len(x), epis_neg))))

        paras = ["+"+i.strip("#")+"-"+"#"*(max_len-len(i.strip("#"))) for i in paras]
        epis_pos = ["+"+i.strip("#")+"-"+"#"*(max_len-len(i.strip("#"))) for i in epis_pos]
        epis_neg = ["+"+i.strip("#")+"-"+"#"*(max_len-len(i.strip("#"))) for i in epis_neg]

        new_batch = [paras, epis_pos, epis_neg]

        return new_batch


def my_collate_fn1(batch):
    return collate_fn(batch, mode=0, use_augment=False)

def my_collate_fn2(batch):
    return collate_fn(batch, mode=0, use_augment=True)