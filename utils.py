import os
import json
import copy
import heapq
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from Bio import Align


def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(seed=3407)


vocab = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'O': 12,
    'P': 13,
    'Q': 14,
    'R': 15,
    'S': 16,
    'T': 17,
    'U': 18,
    'V': 19,
    'W': 20,
    'Y': 21,
    '*': 22,    # UNK / MASK
    '#': 23,    # PAD
    '+': 24,    # BEGIN
    '-': 25,    # END
    '/': 26,    # SEP
}


def toOneHot(seq, seq_length=128):
    li = []

    for i in range(len(seq)):
        li.append(vocab[seq[i]])

    if len(li)>seq_length:
        li = li[:seq_length]
    if len(li)<seq_length:
        pad = [vocab["<MASK>"]]*(seq_length-len(li))
        
    return np.array((li+pad))

def to_onehot(seq, mode=0):
    li = []

    for i in range(len(seq)):
        if mode==0:
            try:
                li.append(vocab[seq[i]])
            except:
                li.append(vocab["*"])
        else:
            feat = np.zeros(len(vocab))
            try:
                feat[vocab[seq[i]]] = 1
            except:
                feat[vocab["*"]] = 1
            li.append(feat)
    
    return np.array(li)


def seq_sim(target, query):

    try:
        aligner = Align.PairwiseAligner()
        aligner = Align.PairwiseAligner(match_score=1.0)

        score = aligner.score(target, query)
        score = score / max(len(target), len(query))
        return score
    except:
        print("Error: {} {}".format(target, query))    
    


def get_span(seq, query):
    start = seq.find(query)
    end = start + len(query)
    
    return start, end

def get_subseq(seq, start, end):
    return seq[start:end]


def get_knearest_epi(data, mode=0, K=48, threshold=10):

    """only reserve k nearest amino acids as epitope

    :return: [data_entry]
    """

    # get k nearest (K = 48)
    if mode==0:
        # for i in range(len(data)):
        #     # maintain a heap with k amino acids
        #     epitope = []
        #     Apos = np.hstack(data[i]["Apos"])
        #     Aseq = "".join(data[i]["Aseq"])
        #     for Aidx in range(len(Aseq)):
        #         # traverse heavy/light chain to find nearest distance
        #         nearest_dist = np.inf
        #         for Hidx in range(len(data[i]["Hpos"])):
        #             cur_dist = np.sqrt(np.sum((Apos[Aidx][0] - data[i]["Hpos"][Hidx][0]) ** 2))
        #             nearest_dist = np.min([cur_dist, nearest_dist])
        #         for Lidx in range(len(data[i]["Lpos"])):
        #             cur_dist = np.sqrt(np.sum((Apos[Aidx][0] - data[i]["Lpos"][Lidx][0]) ** 2))
        #             nearest_dist = np.min([cur_dist, nearest_dist])

        #         epitope.append((nearest_dist, Aidx))

        #     epitope_heap = heapq.nsmallest(K, epitope, key=lambda x:x[0])
        #     epitope_index = sorted([i[1] for i in epitope_heap])

        #     data[i]["epitope"] = "".join([Aseq[i] for i in epitope_index])

        print("get k nearest AAs as epitope...")
        for i in tqdm(range(len(data))):
            epitope = []
            
            # get CDR positions
            cdr_pos = []
            
            Heavy = np.hstack([data[i]["Hseq"][k] for k in data[i]["Hseq"].keys()])
            Hpos = [i['pos'][0] for i in Heavy]
            Hseq = "".join([i['abbr'] for i in Heavy])
            for cdr in ["H1", "H2", "H3"]:
                start, end = get_span(Hseq, data[i][cdr])
                if start==-1:
                    print("{} not found in data {}th".format(cdr, i))
                    cdr_pos.extend(get_subseq(Hpos, 0, len(Hpos)))
                    break
                else:
                    cdr_pos.extend(get_subseq(Hpos, start, end))
                
            Light = np.hstack([data[i]["Lseq"][k] for k in data[i]["Lseq"].keys()])
            Lpos = [i['pos'][0] for i in Light]
            Lseq = "".join([i['abbr'] for i in Light])
            for cdr in ["L1", "L2", "L3"]:
                start, end = get_span(Lseq, data[i][cdr])
                if start==-1:
                    print("{} not found in data {}th".format(cdr, i))
                    cdr_pos.extend(get_subseq(Lpos, 0, len(Hpos)))
                    break
                else:
                    cdr_pos.extend(get_subseq(Lpos, start, end))
            
            # get antigen position and sequence
            Antigen = np.hstack([data[i]["Aseq"][k] for k in data[i]["Aseq"].keys()])    
            Apos = [i['pos'][0] for i in Antigen]
            Aseq = "".join([i['abbr'] for i in Antigen])
            if len(Apos)!=len(Aseq):
        #         break
                # print(i)
                pass
            
            if len(Antigen)<=K:
                data[i]["epitope"] = copy.copy(Aseq)
            else:
                # traverse antigen chain
                for Aidx in range(len(Apos)):
                    # traverse cdr positions to find nearest distance
                    nearest_dist = np.inf
                    for idx in range(len(cdr_pos)):
                        cur_dist = np.sqrt(np.sum((Apos[Aidx] - cdr_pos[idx]) ** 2))
                        nearest_dist = np.min([cur_dist, nearest_dist])

                    epitope.append((nearest_dist, Aidx))
                
                # heap sort
                epitope_heap = heapq.nsmallest(K, epitope, key=lambda x:x[0])
                epitope_index = sorted([i[1] for i in epitope_heap])

                data[i]["epitope"] = "".join([Aseq[i] for i in epitope_index])

    # get within threshold (10 Anstrom)
    if mode==1:
        pass

    return data


def seq_pad_clip(seq, target_length=800):
    """clip sequence to target length

    :param seq: seq
    :param target_length: target length, defaults to 800
    :return: clipped sequence
    """
    
    # padding if smaller
    if len(seq) <= target_length:
        subseq = seq + "#" * (target_length - len(seq))
        return subseq
    # sampling otherwise
    else:
        seq = [(i, seq[i]) for i in range(len(seq))]
        subseq = random.sample(seq, target_length)
        subseq = sorted(subseq, key=lambda x:x[0])
        subseq = "".join([subseq[i][1] for i in range(len(subseq))])
        
        return subseq

def load_model_op_configs(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data