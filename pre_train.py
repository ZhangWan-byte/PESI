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


def load_data(data_path):
    data = pickle.load(open(data_path, "rb"))

    # must: delete samples with CDR containing "..."
    data1 = []
    for i in range(len(data)):
        if ("." in data[i]["H1"]) or ("." in data[i]["H2"]) or ("." in data[i]["H3"]) or ("." in data[i]["L1"]) or ("." in data[i]["L2"]) or ("." in data[i]["L3"]):
            pass
        else:
            data1.append(data[i])
            
    # del data
    # data = data1
    # del data1
    # type(data), len(data)

    return data1


def prepare_lstm(config):

    config["model"] = BiLSTM(embed_size=32, 
                             hidden=64, 
                             num_layers=1, 
                             dropout=0.5, 
                             use_pretrain=False).cuda()
    config["epochs"] = 100
    config["lr"] = 6e-5

    return config


def prepare_textcnn(config):
    config["model"] = TextCNN(amino_ft_dim=len(vocab), 
                              max_antibody_len=100, 
                              max_virus_len=100, 
                              h_dim=512, 
                              dropout=0.1).cuda()
    config["epochs"] = 100
    config["lr"] = 1e-4

    return config

def prepare_masonscnn(config):
    config["model"] = MasonsCNN(amino_ft_dim=len(vocab), 
                                max_antibody_len=100, 
                                max_virus_len=100, 
                                h_dim=512, 
                                dropout=0.1).cuda()
    config["epochs"] = 300
    config["lr"] = 1e-4
    config["l2_coef"] = 5e-4

    return config

def prepare_ag_fast_parapred(config):
    config["model"] = AgFastParapred(ft_dim=len(vocab), 
                                     max_antibody_len=100, 
                                     max_virus_len=100, 
                                     h_dim=512, 
                                     position_coding=True).cuda()
    config["epochs"] = 100
    config["lr"] = 1e-4

    return config

def prepare_pipr(config):
    config["model"] = PIPR(protein_ft_one_hot_dim=len(vocab)).cuda()
    
    config["epochs"] = 300
    config["lr"] = 1e-4

    return config

def prepare_resppi(config):
    config["model"] = ResPPI(amino_ft_dim=len(vocab), 
                             max_antibody_len=100, 
                             max_virus_len=100, 
                             h_dim=512, 
                             dropout=0.1).cuda()
    config["epochs"] = 300
    config["lr"] = 1e-4

    return config

def prepare_deepaai(config):
    pass

def prepare_pesi(config):
    config["model"] = SetTransformer(dim_input=32, 
                                     num_outputs=32, 
                                     dim_output=32, 
                                     dim_hidden=64, 
                                     num_inds=6, 
                                     num_heads=4, 
                                     ln=True, 
                                     dropout=0.5, 
                                     use_coattn=True, 
                                     share=False, 
                                     use_BSS=False, 
                                     use_CLIP=config["use_CLIP"], 
                                     use_CosCLF=config["use_CosCLF"]).cuda()
    
    config["epochs"] = 2000
    config["lr"] = 6e-5
    config["l2_coef"] = 1e-3

    return config


def pre_train(config):

    # model name
    if config["model_name"]=="lstm":
        config = prepare_lstm(config)
    elif config["model_name"]=="textcnn":
        config = prepare_textcnn(config)
    elif config["model_name"]=="masonscnn":
        config = prepare_masonscnn(config)
    elif config["model_name"]=="ag_fast_parapred":
        config = prepare_ag_fast_parapred(config)
    elif config["model_name"]=="pipr":
        config = prepare_pipr(config)
    elif config["model_name"]=="resppi":
        config = prepare_resppi(config)
    elif config["model_name"]=="deepaai":
        config = prepare_deepaai(config)
    elif config["model_name"]=="pesi":
        config = prepare_pesi(config)
    else:
        print("wrong model_name")
        exit()

    if config["pretrain_mode"]=="pair":
        config["model_name"] += "_pair"
    elif config["pretrain_mode"]=="CLIP":
        config["model_name"] += "_CLIP"
    else:
        print("normal mode for pretraining")

    if config["use_L2"]==True:
        config["model_name"] += "_L2"

    print("training {} on SAbDab-full".format(config["model_name"]))
    
    os.makedirs("./results/SAbDab/full/{}/{}/".format(config["data_type"], config["model_name"]), exist_ok=True)

    data = load_data(config["data_path"])

    use_pair = True if config["pretrain_mode"]=="pair" else False

    train_dataset = SAbDabDataset(data=data, 
                                    epi_seq_length=config["epi_len"], 
                                    seq_clip_mode=config["seq_clip_mode"], 
                                    neg_sample_mode=config["neg_sample_mode"], 
                                    is_train_test_full="full", 
                                    is_shuffle=True, 
                                    folds_path=config["folds_path"], 
                                    save_path=None, 
                                    K=48, 
                                    data_augment=False, 
                                    use_cache=config["use_cache"], 
                                    use_pair=use_pair, 
                                    num_neg=config["num_neg"])
    test_dataset = SeqDataset(data_path=config["test_data_path"], 
                              is_train_test_full="full", 
                              use_pair=use_pair, 
                              pretrain_mode=config["pretrain_mode"], 
                              use_part=config["use_part"])

    func = pair_collate_fn if use_pair else collate_fn
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=config["batch_size"], 
                                                shuffle=False, 
                                                collate_fn=func)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=config["batch_size"], 
                                                shuffle=False, 
                                                collate_fn=func)


    print("model parameters: ", sum(p.numel() for p in config["model"].parameters() if p.requires_grad))

    criterion = nn.BCELoss() if use_pair==False else None
    optimizer = optim.Adam(config["model"].parameters(), lr=config["lr"])
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6, last_epoch=-1)

    loss_buf = []
    val_loss_buf = []
    val_acc_buf = []
    val_f1_buf = []
    val_auc_buf = []
    val_gmean_buf = []
    val_mcc_buf = []
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_val_auc = 0.0
    best_val_gmean = 0.0
    best_val_mcc = 0

    for epoch in range(config["epochs"]):

        print("Epoch {}".format(epoch))

        loss_tmp = []
        
        if config["pretrain_mode"]=="pair":
            for i, (para, epi_pos, epi_neg) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()

                y_pred_anc = config["model"](para)
                y_pred_pos = config["model"](epi_pos)
                y_pred_neg = config["model"](epi_neg)
                
                if len(y_pred_anc.shape)==3:
                    y_pred_anc = torch.nn.functional.normalize(torch.mean(y_pred_anc, dim=1), p=2, dim=1)
                    y_pred_pos = torch.nn.functional.normalize(torch.mean(y_pred_pos, dim=1), p=2, dim=1)
                    y_pred_neg = torch.nn.functional.normalize(torch.mean(y_pred_neg, dim=1), p=2, dim=1)
                
                elif len(y_pred_anc.shape)==2:
                    y_pred_anc = torch.nn.functional.normalize(y_pred_anc, p=2, dim=1)
                    y_pred_pos = torch.nn.functional.normalize(y_pred_pos, p=2, dim=1)
                    y_pred_neg = torch.nn.functional.normalize(y_pred_neg, p=2, dim=1)
                
                loss = - (torch.dist(y_pred_anc, y_pred_pos, 2) - torch.dist(y_pred_anc, y_pred_neg, 2)).sigmoid().log().sum()

                if config["use_L2"] == True:
                    param_l2_loss = 0
                    for name, param in config["model"].named_parameters():
                        if 'bias' not in name:
                            param_l2_loss += torch.norm(param, p=2)
                    param_l2_loss = config["l2_coef"] * param_l2_loss
                    loss += param_l2_loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(config["model"].parameters(), config["clip_norm"])

                optimizer.step()

                loss_tmp.append(loss.item())

            loss_buf.append(np.mean(loss_tmp))
        
        else:
            for i, (para, epi, label) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()

                if config["pretrain_mode"]=="normal":
                    pred = config["model"](para, epi)
                    loss = criterion(pred.view(-1), label.view(-1).cuda())
                
                if config["pretrain_mode"]=="CLIP":
                    logits_per_para, logits_per_epi = config["model"](para, epi)
                    labels = torch.arange(logits_per_para.shape[0], device=torch.device("cuda"), dtype=torch.long)
                    loss = (
                        F.cross_entropy(logits_per_para, labels) +
                        F.cross_entropy(logits_per_epi, labels)
                    ) / 2

                if config["use_L2"] == True:
                    param_l2_loss = 0
                    for name, param in config["model"].named_parameters():
                        if 'bias' not in name:
                            param_l2_loss += torch.norm(param, p=2)
                    param_l2_loss = config["l2_coef"] * param_l2_loss
                    loss += param_l2_loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(config["model"].parameters(), config["clip_norm"])

                optimizer.step()

                loss_tmp.append(loss.item())
                
            loss_buf.append(np.mean(loss_tmp))
            
    #     scheduler.step()
        print("lr: ", optimizer.param_groups[0]['lr'])
        # print("train loss {:.4f}\n".format(np.mean(loss_buf)))


        # evaluate
        if config["pretrain_mode"]=="normal":
        
            with torch.no_grad():

                config["model"].eval()

                preds = []
                labels = []
                val_loss_tmp = []
                for i, (para, epi, label) in enumerate(tqdm(test_loader)):

                    pred = config["model"](para, epi)
                    val_loss = criterion(pred.view(-1), label.view(-1).cuda())

                    preds.append(pred.detach().cpu().view(-1))
                    labels.append(label.view(-1))
                    val_loss_tmp.append(val_loss.item())

                preds = torch.hstack(preds).view(-1)
                labels = torch.hstack(labels).view(-1)

                # acc = accuracy_score(y_true=labels, y_pred=torch.round(preds))
                # f1 = f1_score(y_true=labels, y_pred=torch.round(preds))
                # auc = roc_auc_score(y_true=labels, y_score=preds)
                acc, f1, auc, gmean, mcc = evaluate_metrics(pred_proba=preds, label=labels)

                val_acc_buf.append(acc)
                val_f1_buf.append(f1)
                val_auc_buf.append(auc)
                val_gmean_buf.append(gmean)
                val_mcc_buf.append(mcc)
                val_loss_buf.append(np.mean(val_loss_tmp))

                print("Epoch {}: \n Train Loss\t{:.4f} \n Val Loss\t{:.4f} \n Val Acc\t{:.4f} \n Val F1\t\t{:.4f} \n Val AUC\t{:.4f}".format(epoch, np.mean(loss_buf), np.mean(val_loss_buf), acc, f1, auc))

                # save best loss
                if np.mean(val_loss_tmp)<best_val_loss:
                    best_val_loss = np.mean(val_loss_tmp)
                    torch.save(config["model"], "./results/SAbDab/full/{}/{}/model_best.pth".format(config["data_type"], config["model_name"]))
                    np.save("./results/SAbDab/full/{}/{}/val_acc_best.npy".format(config["data_type"], config["model_name"]), acc)
                    np.save("./results/SAbDab/full/{}/{}/val_f1_best.npy".format(config["data_type"], config["model_name"]), f1)
                    np.save("./results/SAbDab/full/{}/{}/val_auc_best.npy".format(config["data_type"], config["model_name"]), auc)
                    np.save("./results/SAbDab/full/{}/{}/val_gmean_best.npy".format(config["data_type"], config["model_name"]), gmean)
                    np.save("./results/SAbDab/full/{}/{}/val_mcc_best.npy".format(config["data_type"], config["model_name"]), mcc)

                # save best f1
                if f1 > best_val_f1:
                    torch.save(config["model"], "./results/SAbDab/full/{}/{}/model_bestf1.pth".format(config["data_type"], config["model_name"]))
                    np.save("./results/SAbDab/full/{}/{}/val_acc_bestf1.npy".format(config["data_type"], config["model_name"]), acc)
                    np.save("./results/SAbDab/full/{}/{}/val_f1_bestf1.npy".format(config["data_type"], config["model_name"]), f1)
                    np.save("./results/SAbDab/full/{}/{}/val_auc_bestf1.npy".format(config["data_type"], config["model_name"]), auc)
                    np.save("./results/SAbDab/full/{}/{}/val_gmean_bestf1.npy".format(config["data_type"], config["model_name"]), gmean)
                    np.save("./results/SAbDab/full/{}/{}/val_mcc_bestf1.npy".format(config["data_type"], config["model_name"]), mcc)

                # save best auc
                if auc > best_val_auc:
                    torch.save(config["model"], "./results/SAbDab/full/{}/{}/model_bestauc.pth".format(config["data_type"], config["model_name"]))
                    np.save("./results/SAbDab/full/{}/{}/val_acc_bestauc.npy".format(config["data_type"], config["model_name"]), acc)
                    np.save("./results/SAbDab/full/{}/{}/val_f1_bestauc.npy".format(config["data_type"], config["model_name"]), f1)
                    np.save("./results/SAbDab/full/{}/{}/val_auc_bestauc.npy".format(config["data_type"], config["model_name"]), auc)
                    np.save("./results/SAbDab/full/{}/{}/val_gmean_bestauc.npy".format(config["data_type"], config["model_name"]), gmean)
                    np.save("./results/SAbDab/full/{}/{}/val_mcc_bestauc.npy".format(config["data_type"], config["model_name"]), mcc)

                # save best gmean
                if gmean > best_val_gmean:
                    torch.save(config["model"], "./results/SAbDab/full/{}/{}/model_bestgmean.pth".format(config["data_type"], config["model_name"]))
                    np.save("./results/SAbDab/full/{}/{}/val_acc_bestgmean.npy".format(config["data_type"], config["model_name"]), acc)
                    np.save("./results/SAbDab/full/{}/{}/val_f1_bestgmean.npy".format(config["data_type"], config["model_name"]), f1)
                    np.save("./results/SAbDab/full/{}/{}/val_auc_bestgmean.npy".format(config["data_type"], config["model_name"]), auc)
                    np.save("./results/SAbDab/full/{}/{}/val_gmean_bestgmean.npy".format(config["data_type"], config["model_name"]), gmean)
                    np.save("./results/SAbDab/full/{}/{}/val_mcc_bestgmean.npy".format(config["data_type"], config["model_name"]), mcc)

                # save best mcc
                if mcc > best_val_mcc:
                    torch.save(config["model"], "./results/SAbDab/full/{}/{}/model_bestmcc.pth".format(config["data_type"], config["model_name"]))
                    np.save("./results/SAbDab/full/{}/{}/val_acc_bestmcc.npy".format(config["data_type"], config["model_name"]), acc)
                    np.save("./results/SAbDab/full/{}/{}/val_f1_bestmcc.npy".format(config["data_type"], config["model_name"]), f1)
                    np.save("./results/SAbDab/full/{}/{}/val_auc_bestmcc.npy".format(config["data_type"], config["model_name"]), auc)
                    np.save("./results/SAbDab/full/{}/{}/val_gmean_bestmcc.npy".format(config["data_type"], config["model_name"]), gmean)
                    np.save("./results/SAbDab/full/{}/{}/val_mcc_bestmcc.npy".format(config["data_type"], config["model_name"]), mcc)

        elif config["pretrain_mode"]=="CLIP":
            with torch.no_grad():

                config["model"].eval()

                preds = []
                labels = []
                val_loss_tmp = []
                for i, (para, epi, label) in enumerate(tqdm(test_loader)):

                    logits_per_para, logits_per_epi = config["model"](para, epi)
                    labels = torch.arange(logits_per_para.shape[0], device=torch.device("cuda"), dtype=torch.long)
                    val_loss = (
                        F.cross_entropy(logits_per_para, labels) +
                        F.cross_entropy(logits_per_epi, labels)
                    ) / 2

                    val_loss_tmp.append(val_loss.item())
                
                val_loss_buf.append(np.mean(val_loss_tmp))
                # save best loss
                if np.mean(val_loss_tmp)<best_val_loss:
                    best_val_loss = np.mean(val_loss_tmp)
                    torch.save(config["model"], "./results/SAbDab/full/{}/{}/model_best.pth".format(config["data_type"], config["model_name"]))

        elif config["pretrain_mode"]=="pair":
#             if np.mean(loss_tmp)<best_train_loss:
#                 best_train_loss = np.mean(loss_tmp)
#                 torch.save(model, "./results/SAbDab/full/{}/{}/model_best.pth".format(data_type, model_name))
            with torch.no_grad():

                config["model"].eval()

                preds = []
                labels = []
                val_loss_tmp = []
                for i, (para1, epi_pos1, epi_neg1) in enumerate(tqdm(test_loader)):

                    y_pred_anc1 = config["model"](para1)
                    y_pred_pos1 = config["model"](epi_pos1)
                    y_pred_neg1 = config["model"](epi_neg1)
                    
                    if len(y_pred_anc1.shape)==3:
                        y_pred_anc1 = torch.nn.functional.normalize(torch.mean(y_pred_anc1, dim=1), p=2, dim=1)
                        y_pred_pos1 = torch.nn.functional.normalize(torch.mean(y_pred_pos1, dim=1), p=2, dim=1)
                        y_pred_neg1 = torch.nn.functional.normalize(torch.mean(y_pred_neg1, dim=1), p=2, dim=1)

                    elif len(y_pred_anc1.shape)==2:
                        y_pred_anc1 = torch.nn.functional.normalize(y_pred_anc1, p=2, dim=1)
                        y_pred_pos1 = torch.nn.functional.normalize(y_pred_pos1, p=2, dim=1)
                        y_pred_neg1 = torch.nn.functional.normalize(y_pred_neg1, p=2, dim=1)

                    val_loss = - (torch.dist(y_pred_anc1, y_pred_pos1, 2) - torch.dist(y_pred_anc1, y_pred_neg1, 2)).sigmoid().log().sum()

                    if config["use_L2"] == True:
                        param_l2_loss1 = 0
                        for name, param in config["model"].named_parameters():
                            if 'bias' not in name:
                                param_l2_loss1 += torch.norm(param, p=2)
                        param_l2_loss1 = config["l2_coef"] * param_l2_loss1
                        val_loss += param_l2_loss1

                    val_loss_tmp.append(val_loss.item())

                val_loss_buf.append(np.mean(val_loss_tmp))
                print("Epoch {}: \n Train Loss\t{:.4f} \n Val Loss\t{:.4f}\n".format(epoch, np.mean(loss_buf), np.mean(val_loss_buf)))

                if np.mean(val_loss_tmp)<best_val_loss:
                    best_val_loss = np.mean(val_loss_tmp)
                    torch.save(config["model"], "./results/SAbDab/full/{}/{}/model_best.pth".format(config["data_type"], config["model_name"]))
        else:
            print("Wrong")
            exit()


        torch.cuda.empty_cache()

        print("Train loss: {}\tVal loss: {}".format(loss_buf[-1], val_loss_buf[-1]))

        config["model"].train()



    torch.save(config["model"], "./results/SAbDab/full/{}/{}/model.pth".format(config["data_type"], config["model_name"]))
    np.save("./results/SAbDab/full/{}/{}/loss_buf.npy".format(config["data_type"], config["model_name"]), np.array(loss_buf))
    np.save("./results/SAbDab/full/{}/{}/val_loss_buf.npy".format(config["data_type"], config["model_name"]), np.array(val_loss_buf))
    if config["pretrain_mode"]=="normal":
        
        np.save("./results/SAbDab/full/{}/{}/val_acc_buf.npy".format(config["data_type"], config["model_name"]), np.array(val_acc_buf))
        np.save("./results/SAbDab/full/{}/{}/val_f1_buf.npy".format(config["data_type"], config["model_name"]), np.array(val_f1_buf))
        np.save("./results/SAbDab/full/{}/{}/val_auc_buf.npy".format(config["data_type"], config["model_name"]), np.array(val_auc_buf))
        np.save("./results/SAbDab/full/{}/{}/val_gmean_buf.npy".format(config["data_type"], config["model_name"]), np.array(val_gmean_buf))
        np.save("./results/SAbDab/full/{}/{}/val_mcc_buf.npy".format(config["data_type"], config["model_name"]), np.array(val_mcc_buf))


    #     break
    
    # res = evaluate(model_name=config["model_name"], kfold=config["kfold"])

    # return res


if __name__=='__main__':

    set_seed(seed=3407)
    # set_seed(seed=42)

    # # model_name = "masonscnn"
    # # model_name = "lstm"
    # # model_name = "textcnn"
    # # model_name = "ag_fast_parapred"
    # # model_name = "pipr"
    # # model_name = "resppi"
    # model_name = "pesi"

    model_name = sys.argv[1]

    config = {
        # data type
        "clip_norm": 1, 
        "seq_clip_mode": 1,                     # how to choose epitope: 0 - random AA sequence as epitope; 1 - k-nearest AA as epitope
        "neg_sample_mode": 0,                   # how to generate negative sample: 0 - random sample with dissimilarity rate 90% 1 - random sequence;
        "data_type": "seq1_neg0", 
        "data_path": "../Transformer4Ab/data/data_list.pkl",    
                                                # data path for general antibody-antigen dataset
        "test_data_path": "../SARS-SAbDab_Shaun/CoV-AbDab_extract.csv", 
                                                # data path for SARS-CoV-2 antibody-antigen dataset
        "use_cache": False,                      # whether using cached pair data
        

        # pre-training params
        "pretrain_mode": "normal",                # pre-training mode: CLIP/pair/normal
        "num_neg": 4,                           # number of negative samples per positive pair if pretrain_mode=="pair"
        "use_part": True,                       # whether use part of cov-abdab as validation for model selection

        # regularisation
        "use_L2": False,                        # whether using L2 regularisation for pre-training
        "use_BSS": False,                       # Batch Spectral Shrinkage regularisation

        # learning params
        "batch_size": 1024,                       # batch size
        "epi_len": 72,                          # max length of epitope


        # model_params
        "model_name": model_name,               # type of models
        "use_CosCLF": True                      # whether to use linear classifier on top of Set.Trans.
    }

    if config["pretrain_mode"]=="pair":
        config["folds_path"] = "../Transformer4Ab/data/processed_data_clip1_neg0_pair.pkl"
        config["use_pair"] = True if config["pretrain_mode"]=="pair" else False
    else:
        config["folds_path"] = "../Transformer4Ab/data/processed_data_clip1_neg0.pkl"
        if config["pretrain_mode"]=="CLIP":
            config["use_CLIP"] = True
        else:
            config["use_CLIP"] = False


    print(config)

    # training
    pre_train(config=config)

    # print("Results dump to: ")
    # print("./results/SAbDab/full/{}/{}/result.pkl".format(config["data_type"], config["model_name"]))
    # pickle.dump(result, open("./results/SAbDab/full/{}/{}/result_{}.pkl".format(config["data_type"], config["model_name"]), "wb"))


    

    