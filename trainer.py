import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import *
from utils import *
from models import *

class Trainer():
    def __init__(self, config) -> None:
        self.config = config
        self.dataset = config.dataset                       # cov-abdab or sabdab
        self.data_path = config.data_path
        
        self.initialisation()

    def initialisation(self):
        
        # loading data
        if self.dataset == "cov-abdab":
            # self.data = pd.read_csv("../SARS-SAbDab_Shaun/CoV-AbDab_extract.csv")
            self.data = pd.read_csv(self.data_path)
            print("Dataset {}".format(self.dataset))
            print("data shape: {}".format(self.data.shape))
        elif self.dataset == "sabdab":
            # self.data = pickle.load(open("./data/data_list.pkl", "rb"))
            self.data = pickle.load(open(self.data_path, "rb"))
            # must: delete samples with CDR containing "..."
            data1 = []
            for i in range(len(self.data)):
                if ("." in self.data[i]["H1"]) or ("." in self.data[i]["H2"]) or ("." in self.data[i]["H3"]) or \
                    ("." in self.data[i]["L1"]) or ("." in self.data[i]["L2"]) or ("." in self.data[i]["L3"]):
                    pass
                else:
                    data1.append(self.data[i])
                    
            self.data = copy.copy(data1)
            del data1
            print("Dataset {}".format(self.dataset))
            print("type and length of data".format(type(self.data), len(self.data)))
        else:
            print("dataset unimplemented")

        
        # dataset
        if self.dataset == "cov-abdab":
            pass
        elif self.dataset == "sabdab":
            pass
        else:
            pass


        # model



        # training configurations

        if self.config.kfold==0:
            pass                           
        
        