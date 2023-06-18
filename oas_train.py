import os
import sys
import copy
import pickle
import random
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Bio import Align

from dataset import *
from utils import *
from models import *
from cov_train import *
from pre_train import *


def prepare_mlm(config):
    config["model"] = PESILM(vocab_size=len(vocab), 
                             dim_input=32, 
                             dim_hidden=64, 
                             num_outputs=32, 
                             dim_output=32, 
                             num_inds=6, 
                             num_heads=4, 
                             ln=True, 
                             dropout=0.5).cuda()
    
    return config


def oas_train(config, result_path):

    # model name
    config = prepare_mlm(config=config)

    print("training {} on OAS starting...")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset = OASDataset(corpus_path=config["oas_path"], vocab=vocab, train_test='train')

    test_dataset = OASDataset(corpus_path=config["oas_path"], vocab=vocab, train_test='test')

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=config["batch_size"], 
                                                shuffle=False, 
                                                collate_fn=collate_mlm)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=config["batch_size"], 
                                                shuffle=False, 
                                                collate_fn=collate_mlm)

    print("model parameters: ", sum(p.numel() for p in config["model"].parameters() if p.requires_grad))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(config["model"].parameters(), lr=config["lr"])
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6, last_epoch=-1)
    if config["use_lr_schedule"]:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-6, last_epoch=-1)

    loss_buf = []
    val_loss_buf = []
    best_train_loss = float("inf")
    best_val_loss = float("inf")


    for epoch in range(config["epochs"]):

        print("Epoch {}".format(epoch))

        loss_tmp = []
        
        for i, data in enumerate(tqdm(train_loader)):

            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(device) for key, value in data.items()}

            # 1. forward masked_lm model
            mask_lm_output, attn_list = config["model"].forward(data["mlm_input"], data["input_position"])

            # 2. NLLLoss of predicting masked token word
            optimizer.zero_grad()
            loss = criterion(mask_lm_output.transpose(1, 2), data["mlm_label"])

            # 3. backward and optimization only in train
            loss.backward()
            torch.nn.utils.clip_grad_norm_(config["model"].parameters(), config["clip_norm"])
            optimizer.step()

            # loss
            running_loss += loss.item()
            avg_loss = running_loss / (i + 1)


            loss_tmp.append(loss.item())
            
        loss_buf.append(np.mean(loss_tmp))
            
        if config["use_lr_schedule"]:
            scheduler.step()
        print("lr: ", optimizer.param_groups[0]['lr'])
        # print("train loss {:.4f}\n".format(np.mean(loss_buf)))


        # evaluate
        with torch.no_grad():

            config["model"].eval()

            preds = []
            labels = []
            val_loss_tmp = []
            for i, data in enumerate(tqdm(train_loader)):

                data = {key: value.to(device) for key, value in data.items()}
                mask_lm_output, attn_list = config["model"].forward(data["mlm_input"], data["input_position"])
                val_loss = criterion(mask_lm_output.transpose(1, 2), data["mlm_label"])

                preds.append((mask_lm_output.detach().cpu(), attn_list.detach().cpu()))
                labels.append(data["mlm_label"])
                val_loss_tmp.append(val_loss.item())

            # preds = torch.hstack(preds).view(-1)
            # labels = torch.hstack(labels).view(-1)

            print("Epoch {}: \n Train Loss\t{:.4f} \n Val Loss\t{:.4f} \n".format(epoch, np.mean(loss_tmp), np.mean(val_loss_tmp)))

            # save best loss
            if np.mean(val_loss_tmp)<best_val_loss:
                best_val_loss = np.mean(val_loss_tmp)
                torch.save(config["model"], os.path.join(result_path, "model_best.pth"))

        torch.cuda.empty_cache()

        print("Train loss: {}\tVal loss: {}".format(loss_buf[-1], val_loss_buf[-1]))

        config["model"].train()


    torch.save(config["model"], os.path.join(result_path, "model.pth"))
    np.save(os.path.join(result_path, "loss_buf.npy"), np.array(loss_buf))
    np.save(os.path.join(result_path, "val_loss_buf.npy"), np.array(val_loss_buf))


if __name__=='__main__':

    set_seed(seed=3407)

    config = {
        # data type
        "data_path": "./OAS/oas_data.pkl",      # data path for general antibody-antigen dataset

        # learning params
        "batch_size": 64,                       # batch size
        "use_lr_schedule": False,               # lr scheduler
        "epochs": 1000,                         # number of epochs
        "lr": 1e-4,                             # learning rate
        "clip_norm": 1,                         # gradient clipping threshold
    }

    current_time = time.strftime('%m%d%H%M%S', time.localtime())
    config["current_time"] = current_time

    result_path = "./results/OAS/{}".format(config["current_time"])
    os.makedirs(result_path, exist_ok=True)

    print(config)
    pickle.dump(config, open(os.path.join(result_path, "config"), "wb"))

    # training
    oas_train(config=config, result_path=result_path)