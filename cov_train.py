import os
import sys
import time
import json
import shutil
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from metrics import *
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from datasets import *
from utils import *
from models import *


def prepare_lstm(config):

    # if config["use_fine_tune"]==True:
    #     config["model_name"] += "_ft"

    #     if config["use_pair"]==True:
    #         config["model_name"] += "_pairPreTrain"

    if config["model_name"]=="lstm":
        config["model"] = BiLSTM(embed_size=32, 
                                 hidden=64, 
                                 num_layers=1, 
                                 dropout=0.5, 
                                 use_pretrain=False).cuda()

        config["epochs"] = 300
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4

    elif config["model_name"]=="lstm_ft":
        config["model"] = torch.load("./results/SAbDab/full/seq1_neg0/lstm/model_best.pth")
        config["model"].train()
        
        if config["fix_FE"]==True:
            for name, param in config["model"].LSTM_para.named_parameters():
                param.requires_grad = False
            for name, param in config["model"].LSTM_epi.named_parameters():
                param.requires_grad = False
        
        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="lstm_ft_pairPreTrain":

        encoder = torch.load("./results/SAbDab/full/seq1_neg0/lstm_encoder/model_best.pth")
        encoder.train()
        config["model"] = TowerBaseModel(embed_size=64, hidden=128, encoder=encoder, 
                                         use_two_towers=False, use_coattn=False, fusion=1).cuda()
        
        if config["fix_FE"]==True:
            for name, param in config["model"].encoder.named_parameters():
                param.requires_grad = False
        
        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4

    else:
        print("Error Model Name")
        print(config["model_name"])
        exit()

    return config


def prepare_textcnn(config):
    # if config["use_fine_tune"]==True:
    #     config["model_name"] += "_ft"

    #     if config["use_pair"]==True:
    #         config["model_name"] += "_pairPreTrain"

    if config["model_name"]=="textcnn":
        config["model"] = TextCNN(amino_ft_dim=len(vocab), 
                                 max_antibody_len=100, 
                                 max_virus_len=100, 
                                 h_dim=512, 
                                 dropout=0.1).cuda()
        config["epochs"] = 100
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="textcnn_ft":
        config["model"] = torch.load("./results/SAbDab/full/seq1_neg0/textcnn/model_best.pth")

        if config["fix_FE"]==True:
            for name, param in model.text_inception.named_parameters():
                param.requires_grad = False
            for name, param in model.text_inception2.named_parameters():
                param.requires_grad = False


        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="textcnn_ft_pairPreTrain":
        
        encoder = torch.load("./results/SAbDab/full/seq1_neg0/textcnn_encoder/model_best.pth")
        config["model"] = TowerBaseModel(embed_size=32, hidden=128, encoder=encoder, 
                                         use_two_towers=False, use_coattn=False, fusion=0).cuda()
        
        if config["fix_FE"]==True:
            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
        
        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4

    else:
        print("Error Model Name")
        exit()

    return config


def prepare_masonscnn(config):

    # if config["use_fine_tune"]==True:
    #     config["model_name"] += "_ft"

    #     if config["use_pair"]==True:
    #         config["model_name"] += "_pairPreTrain"

    if config["model_name"]=="masonscnn":
        config["model"] = MasonsCNN(amino_ft_dim=len(vocab), 
                                    max_antibody_len=100, 
                                    max_virus_len=100, 
                                    h_dim=512, 
                                    dropout=0.1).cuda()
        config["epochs"] = 100
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="masonscnn_ft":
        config["model"] = torch.load("./results/SAbDab/full/seq1_neg0/masonscnn/model_best.pth")

        if config["fix_FE"]==True:
            for name, param in model.cnnmodule.named_parameters():
                param.requires_grad = False
            for name, param in model.cnnmodule2.named_parameters():
                param.requires_grad = False


        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="masonscnn_ft_pairPreTrain":
        
        encoder = torch.load("./results/SAbDab/full/seq1_neg0/masonscnn_encoder/model_best.pth")
        config["model"] = TowerBaseModel(embed_size=32, hidden=128, encoder=encoder, 
                                         use_two_towers=False, use_coattn=False, fusion=0).cuda()
        
        if config["fix_FE"]==True:
            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
        
        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4

    else:
        print("Error Model Name")
        exit()

    return config


def prepare_ag_fast_parapred(config):

    # if config["use_fine_tune"]==True:
    #     config["model_name"] += "_ft"

    #     if config["use_pair"]==True:
    #         config["model_name"] += "_pairPreTrain"

    if config["model_name"]=="ag_fast_parapred":
        config["model"] = AgFastParapred(ft_dim=len(vocab), 
                                         max_antibody_len=100, 
                                         max_virus_len=100, 
                                         h_dim=512, 
                                         position_coding=True).cuda()
        config["epochs"] = 100
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="ag_fast_parapred_ft":
        config["model"] = torch.load("./results/SAbDab/full/seq1_neg0/ag_fast_parapred/model_best.pth")

        if config["fix_FE"]==True:
            # for name, param in model.cnnmodule.named_parameters():
            #     param.requires_grad = False
            # for name, param in model.cnnmodule2.named_parameters():
            #     param.requires_grad = False
            print("not implemented")
            exit()


        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="ag_fast_parapred_ft_pairPreTrain":
        
        encoder = torch.load("./results/SAbDab/full/seq1_neg0/ag_fast_parapred_encoder/model_best.pth")
        config["model"] = TowerBaseModel(embed_size=32, hidden=128, encoder=encoder, 
                                         use_two_towers=False, use_coattn=False, fusion=0).cuda()
        
        if config["fix_FE"]==True:
            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
        
        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4

    else:
        print("Error Model Name")
        exit()

    return config

def prepare_pipr(config):
    # if config["use_fine_tune"]==True:
    #     config["model_name"] += "_ft"

    #     if config["use_pair"]==True:
    #         config["model_name"] += "_pairPreTrain"

    if config["model_name"]=="pipr":
        config["model"] = PIPR(protein_ft_one_hot_dim=len(vocab)).cuda()

        config["epochs"] = 100
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="pipr_ft":
        config["model"] = torch.load("./results/SAbDab/full/seq1_neg0/pipr/model_best.pth")

        if config["fix_FE"]==True:
            # for name, param in model.cnnmodule.named_parameters():
            #     param.requires_grad = False
            # for name, param in model.cnnmodule2.named_parameters():
            #     param.requires_grad = False
            print("not implemented")
            exit()


        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4

    else:
        print("Error Model Name")
        exit()

    return config

def prepare_resppi(config):
    # if config["use_fine_tune"]==True:
    #     config["model_name"] += "_ft"

    #     if config["use_pair"]==True:
    #         config["model_name"] += "_pairPreTrain"

    if config["model_name"]=="resppi":
        config["model"] = ResPPI(amino_ft_dim=len(vocab), 
                                    max_antibody_len=100, 
                                    max_virus_len=100, 
                                    h_dim=512, 
                                    dropout=0.1).cuda()
        config["epochs"] = 100
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="resppi_ft":
        config["model"] = torch.load("./results/SAbDab/full/seq1_neg0/resppi/model_best.pth")

        if config["fix_FE"]==True:
            # for name, param in model.cnnmodule.named_parameters():
            #     param.requires_grad = False
            # for name, param in model.cnnmodule2.named_parameters():
            #     param.requires_grad = False
            print("not implemented")
            exit()


        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="resppi_ft_pairPreTrain":
        
        encoder = torch.load("./results/SAbDab/full/seq1_neg0/resppi_encoder/model_best.pth")
        config["model"] = TowerBaseModel(embed_size=32, hidden=128, encoder=encoder, 
                                         use_two_towers=False, use_coattn=False, fusion=0).cuda()
        
        if config["fix_FE"]==True:
            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
        
        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4

    else:
        print("Error Model Name")
        exit()

    return config

def prepare_deepaai(config):
    pass

def prepare_settransformer(config):
    print(config["model_name"])

    if config["model_name"]=="settransformer":
        config["model"] = SetTransformer(dim_input=32, 
                                        num_outputs=32, 
                                        dim_output=32, 
                                        dim_hidden=64, 
                                        num_inds=6, 
                                        num_heads=4, 
                                        ln=True, 
                                        dropout=0.5, 
                                        use_coattn=False, 
                                        share=False, 
                                        use_BSS=False, 
                                        use_CLIP=False, 
                                        use_CosCLF=False).cuda()
    elif config["model_name"]=="settransformer_ft":
        config["model"] = torch.load("./results/SAbDab/{}/model_best.pth".format(config["best_model_path"]))
        config["model"].train()
    else:
        pass
    
    return config

def prepare_pesi(config):
    print(config["model_name"])
    # if config["use_fine_tune"]==True:
    #     config["model_name"] += "_ft"

    #     if config["use_pair"]==True:
    #         config["model_name"] += "_pairPreTrain"
    
    if config["model_name"]=="SetTransformer":
        config["model"] = SetTransformer(dim_input=32, 
                            num_outputs=32, 
                            dim_output=32, 
                            dim_hidden=64, 
                            num_inds=6, 
                            num_heads=4, 
                            ln=True, 
                            dropout=0.5, 
                            use_coattn=False, 
                            share=False).cuda()
        
        epochs = 500
        lr = 1e-4
        l2_coef = 5e-4
        
    elif config["model_name"]=="SetTransformer_ft":

        config["model"] = torch.load(config["model_path"])
        print("load pre-trained model from {}".format(config["model_path"]))
        config["model"].train()

        if config["fix_FE"]==True:
            for name, param in config["model"].para_enc.named_parameters():
                param.requires_grad = False
            for name, param in config["model"].para_dec.named_parameters():
                param.requires_grad = False
            for name, param in config["model"].epi_enc.named_parameters():
                param.requires_grad = False
            for name, param in config["model"].epi_dec.named_parameters():
                param.requires_grad = False

        epochs = 500
        lr = 1e-4
        l2_coef = 5e-4
        
        
    # elif config["model_name"]=="SetCoAttnTransformer":
    elif config["model_name"]=="pesi":
        config["model"] = SetTransformer(dim_input=32, 
                                         num_outputs=128, 
                                         dim_output=32, 
                                         dim_hidden=64, 
                                         num_inds=128, 
                                         num_heads=4, 
                                         ln=True, 
                                         dropout=0.5, 
                                         use_coattn=True, 
                                         share=False, 
                                         use_BSS=config["use_BSS"]).cuda()

        # # pesi small
        # print("pesi_small!!!!!")
        # config["model"] = SetTransformer(dim_input=32, 
        #                                  num_outputs=32, 
        #                                  dim_output=32, 
        #                                  dim_hidden=32, 
        #                                  num_inds=6, 
        #                                  num_heads=4, 
        #                                  ln=True, 
        #                                  dropout=0.5, 
        #                                  use_coattn=True, 
        #                                  share=False, 
        #                                  use_BSS=config["use_BSS"]).cuda()
        
        # config["epochs"] = 200
        # config["lr"] = 6e-5
        # config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="pesi_ft":
        if config["use_BSS"]==False:
            if config["model_path"]!="":
                config["model"] = torch.load(config["model_path"])
                print("load pre-trained model from {}".format(config["model_path"]))
                config["model"].train()

                if config["fix_FE"]==True:
                    for name, param in config["model"].para_enc.named_parameters():
                        param.requires_grad = False
                    for name, param in config["model"].para_dec.named_parameters():
                        param.requires_grad = False
                    for name, param in config["model"].epi_enc.named_parameters():
                        param.requires_grad = False
                    for name, param in config["model"].epi_dec.named_parameters():
                        param.requires_grad = False
                
                config["epochs"] = 200
                config["lr"] = 6e-5
                config["l2_coef"] = 5e-4
            else:
                ckpt_para = torch.load(config["oas_pretrain"])
                if "sabdab_train" in config.keys():
                    ckpt_epi = torch.load(config["sabdab_pretrain"])
                else:
                    ckpt_epi = ckpt_para

                config["model"] = SetTransformer(dim_input=32, 
                                                num_outputs=128, 
                                                dim_output=32, 
                                                dim_hidden=64, 
                                                num_inds=128, 
                                                num_heads=4, 
                                                ln=True, 
                                                dropout=0.5, 
                                                use_coattn=True, 
                                                share=False, 
                                                use_BSS=False, 
                                                use_CLIP=False, 
                                                use_CosCLF=False).cuda()

                config["model"].embedding.load_state_dict(ckpt_para.embedding.state_dict())
                config["model"].para_enc.load_state_dict(ckpt_para.enc.state_dict())
                config["model"].para_dec.load_state_dict(ckpt_para.dec.state_dict())
                config["model"].epi_enc.load_state_dict(ckpt_epi.enc.state_dict())
                config["model"].epi_dec.load_state_dict(ckpt_epi.dec.state_dict())
                config["model"].co_attn.load_state_dict(ckpt_para.co_attn.state_dict())
                config["model"].train()

                config["epochs"] = 200
                config["lr"] = 6e-5
                config["l2_coef"] = 5e-4
            # model = SetTransformer(dim_input=32, 
            #                     num_outputs=32, 
            #                     dim_output=32, 
            #                     dim_hidden=64, 
            #                     num_inds=6, 
            #                     num_heads=4, 
            #                     ln=True, 
            #                     dropout=0.5, 
            #                     use_coattn=False, 
            #                     share=False, 
            #                     use_BSS=False).cuda()
        
            # pt_model = torch.load("./results/SAbDab/full/seq1_neg0/SetCoAttnTransformer/model_best.pth")
        
            # model.para_enc = pt_model.para_enc
            # model.para_dec = pt_model.para_dec
            # model.epi_enc = pt_model.epi_enc
            # model.epi_dec = pt_model.epi_dec
            # model.train()
        
            # epochs = 500
            # lr = 6e-5
            # l2_coef = 5e-4

        elif config["use_BSS"]==True:
            # print(config["model_name"], config["use_BSS"])
            config["model"] = SetTransformer(dim_input=32, 
                                            num_outputs=32, 
                                            dim_output=32, 
                                            dim_hidden=128, 
                                            num_inds=6, 
                                            num_heads=4, 
                                            ln=True, 
                                            dropout=0.5, 
                                            use_coattn=False, 
                                            share=False, 
                                            use_BSS=True).cuda()
                                            
            # load pre-trained weights
            # pt_model = torch.load("./results/SAbDab/full/seq1_neg0/SetCoAttnTransformer/model_best.pth")
            pt_model = torch.load("./results/SAbDab/full/seq1_neg0/pesi/model_best.pth")

            config["model"].embedding = pt_model.embedding
            
            config["model"].para_enc = pt_model.para_enc
            config["model"].epi_enc = pt_model.epi_enc
            
            config["model"].co_attn = pt_model.co_attn
            
            config["model"].para_dec = pt_model.para_dec
            config["model"].epi_dec = pt_model.epi_dec
            
            config["model"].output_layer = pt_model.output_layer
            
            config["model"].train()
        
            # params
            config["epochs"] = 200
            config["lr"] = 6e-5
            config["l2_coef"] = 5e-4
        else:
            print("wrong use_BSS!")
            quit()

    elif config["model_name"]=="pesi_CLIP":
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
                                     use_CLIP=False).cuda()
        
        pretrained_model = torch.load("./results/SAbDab/full/seq1_neg0/pesi_CLIP/model_best.pth")
        
        config["model"].embedding.load_state_dict(pretrained_model.embedding.state_dict())
        config["model"].para_enc.load_state_dict(pretrained_model.para_enc.state_dict())
        config["model"].para_dec.load_state_dict(pretrained_model.para_dec.state_dict())
        config["model"].epi_enc.load_state_dict(pretrained_model.epi_enc.state_dict())
        config["model"].epi_dec.load_state_dict(pretrained_model.epi_dec.state_dict())
        config["model"].co_attn.load_state_dict(pretrained_model.co_attn.state_dict())
        config["model"].output_layer.load_state_dict(pretrained_model.output_layer.state_dict())
        config["model"].train()

        # # params
        # config["epochs"] = 1000
        # config["lr"] = 6e-5
        # config["l2_coef"] = 5e-4  
        
    elif config["model_name"]=="pesi_CosCLF":
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
                                        use_CLIP=False, 
                                        use_CosCLF=True).cuda()
        
        pretrained_model = torch.load("./results/SAbDab/full/seq1_neg0/pesi/model_best.pth")
        
        config["model"].embedding.load_state_dict(pretrained_model.embedding.state_dict())
        config["model"].para_enc.load_state_dict(pretrained_model.para_enc.state_dict())
        config["model"].para_dec.load_state_dict(pretrained_model.para_dec.state_dict())
        config["model"].epi_enc.load_state_dict(pretrained_model.epi_enc.state_dict())
        config["model"].epi_dec.load_state_dict(pretrained_model.epi_dec.state_dict())
        config["model"].co_attn.load_state_dict(pretrained_model.co_attn.state_dict())
        config["model"].train()

        # params
        config["epochs"] = 1000
        config["lr"] = 6e-5
        config["l2_coef"] = 5e-4  

    elif config["model_name"]=="SetCoAttnTransformer_ft_pairPreTrain":
        
        encoder = torch.load("./results/SAbDab/full/seq1_neg0/SetTransformer_encoder/model_best.pth")
        encoder.train()
        model = TowerBaseModel(embed_size=32, hidden=128, encoder=encoder, use_two_towers=False, mid_coattn=True, use_coattn=True, fusion=1).cuda()
        
        if config["fix_FE"]==True:
            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
        
        epochs = 1500
        lr = 1e-4 #6e-5
        l2_coef = 3e-4 #5e-4

    else:
        print("Error Model Name")
        exit()

    return config


def cov_train(config, result_path):

    if config["use_fine_tune"]!='None':
        if config["use_fine_tune"]=='normal' and "ft" not in config["model_name"]:
            config["model_name"] += "_ft"
        
        if config["use_fine_tune"]=='CLIP' and "CLIP" not in config["model_name"]:
            config["model_name"] += "_CLIP"

        if config["use_fine_tune"]=='CosCLF' and 'CosCLF' not in config["model_name"]:
            config["model_name"] += "_CosCLF"

        if config["use_pair"]==True:
            config["model_name"] += "_pairPreTrain"

    # print("make folder ./results/CoV-AbDab/{}/".format(config["model_name"]))
    # os.makedirs("./results/CoV-AbDab/{}/".format(config["model_name"]), exist_ok=True)

    print(config)
    pickle.dump(config, open(os.path.join(result_path, "config_file"), "wb"))

    print("model name: {}\tuse_fine_tune: {}".format(config["model_name"], config["use_fine_tune"]))

    kfold_labels = []
    kfold_preds = []

    for k_iter in range(config["kfold"]):
        
        print("=========================================================")
        print("fold {} as val set".format(k_iter))

        # model name
        if config["model_name"][:4]=="lstm" or config["model_name"]=="lstm":
            config = prepare_lstm(config)
        elif config["model_name"][:7]=="textcnn" or config["model_name"]=="textcnn":
            config = prepare_textcnn(config)
        elif config["model_name"][:9]=="masonscnn" or config["model_name"]=="masonscnn":
            config = prepare_masonscnn(config)
        elif config["model_name"][:16]=="ag_fast_parapred" or config["model_name"]=="ag_fast_parapred":
            config = prepare_ag_fast_parapred(config)
        elif config["model_name"][:4]=="pipr" or config["model_name"]=="pipr":
            config = prepare_pipr(config)
        elif config["model_name"][:6]=="resppi" or config["model_name"]=="resppi":
            config = prepare_resppi(config)
        elif config["model_name"][:7]=="deepaai" or config["model_name"]=="deepaai":
            config = prepare_deepaai(config)
        elif config["model_name"][:4]=="pesi" or config["model_name"]=="pesi":
            config = prepare_pesi(config)
        elif config["model_name"][:14]=="settransformer" or config["model_name"]=="settransformer":
            config = prepare_settransformer(config)
        else:
            print("wrong model name")
            print(config["model_name"])
            exit()
        
        train_dataset = SeqDataset(data_path=config["data_path"], 
                                   kfold=config["kfold"], 
                                   holdout_fold=k_iter, 
                                   is_train_test_full="train", 
                                   use_pair=config["use_pair"], 
                                   balance_samples=config["balance_samples"], 
                                   balance_ratio=config["balance_ratio"],  
                                   use_part=config["use_part"])
        collate_fn_train = my_collate_fn2 if config["use_aug"]==True else collate_fn
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=config["batch_size"], 
                                                   shuffle=False, 
                                                   collate_fn=collate_fn_train)

        test_dataset = SeqDataset(data_path=config["data_path"], 
                                  kfold=config["kfold"], 
                                  holdout_fold=k_iter, 
                                  is_train_test_full="test", 
                                  use_pair=config["use_pair"], 
                                  balance_samples=config["balance_samples"], 
                                   balance_ratio=config["balance_ratio"],  
                                  use_part=config["use_part"])
        collate_fn_test = my_collate_fn1 if config["use_aug"]==True else collate_fn
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size=1, 
                                                  shuffle=False, 
                                                  collate_fn=collate_fn_test)

        print("model_name: {}".format(config["model_name"]))

        print("model parameters: ", sum(p.numel() for p in config["model"].parameters() if p.requires_grad))
        
        criterion = nn.BCELoss()
        if config["diff_lr"]==False:
            optimizer = optim.Adam(config["model"].parameters(), lr=config["lr"])#, weight_decay=wd)
        else:
            my_list = ["embedding", "para_enc", "para_dec", "epi_enc", "epi_dec", "co_attn"]
            base_params = list(filter(lambda kv: kv[0].split(".")[0] in my_list, config["model"].named_parameters()))
            classifier_params = list(filter(lambda kv: kv[0].split(".")[0] not in my_list, config["model"].named_parameters()))
            optimizer = optim.Adam([
                {'params': [temp[1] for temp in base_params], 'lr': 1e-6}, 
                {'params': [temp[1] for temp in classifier_params]}
            ], lr=config["lr"])
        
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6, last_epoch=-1)

        loss_buf = []
        val_loss_buf = []
        val_acc_buf = []
        val_f1_buf = []
        val_auc_buf = []
        val_gmean_buf = []
        val_mcc_buf = []
        best_val_loss = float("inf")
        
        for epoch in range(config["epochs"]):

            config["model"].train()

            loss_tmp = []
            for i, (para, epi, label) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()

                if config["use_BSS"]==False:
                    pred = config["model"](para, epi)
                elif config["use_BSS"]==True:
                    pred, BSS = config["model"](para, epi)
                else:
                    pass
                
                # print(pred.shape, label.shape)

                loss = criterion(pred.view(-1), label.view(-1).cuda())
                
                if config["use_reg"]==False:
                    param_l2_loss = 0
                    for name, param in config["model"].named_parameters():
                        if 'bias' not in name:
                            param_l2_loss += torch.norm(param, p=2)
                    param_l2_loss = config["l2_coef"] * param_l2_loss
                    loss += param_l2_loss
                elif config["use_reg"]==True:
                    param_l1_loss = 0
                    for name, param in config["model"].named_parameters():
                        if 'bias' not in name:
                            param_l1_loss += torch.norm(param, p=1)
                    param_l1_loss = config["l1_coef"] * param_l1_loss
                    loss += param_l1_loss
                else:
                    print("wrong use_reg! only 0 or 1!")
                    exit()
                
                if config["use_BSS"]==True:
                    loss += 0.001*BSS

                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(config["model"].parameters(), config["clip_norm"])

                optimizer.step()

                loss_tmp.append(loss.item())
            
            loss_buf.append(np.mean(loss_tmp))

        #     scheduler.step()
            print("lr: ", optimizer.param_groups[0]['lr'])

            with torch.no_grad():

                config["model"].eval()

                preds = []
                labels = []
                val_loss_tmp = []
                for i, (para, epi, label) in enumerate(test_loader):
                    if config["use_BSS"]==False:
                        pred = config["model"](para, epi)
                    elif config["use_BSS"]==True:
                        pred, BSS = config["model"](para, epi)
                    else:
                        pass
                    
                    val_loss = criterion(pred.view(-1), label.view(-1).cuda())
                    
                    if config["use_BSS"]==True:
                        val_loss += 0.001*BSS
                    
                    preds.append(pred.detach().cpu().view(-1))
                    labels.append(label.view(-1))
                    val_loss_tmp.append(val_loss.item())
                
                preds = torch.stack(preds, axis=1).view(-1)
                labels = torch.stack(labels, axis=1).view(-1)

                acc, f1, auc, gmean, mcc = evaluate_metrics(pred_proba=preds, label=labels)

                val_acc_buf.append(acc)
                val_f1_buf.append(f1)
                val_auc_buf.append(auc)
                val_gmean_buf.append(gmean)
                val_mcc_buf.append(mcc)
                val_loss_buf.append(np.mean(val_loss_tmp))

                print("Epoch {}: \n Train Loss\t{:.4f} \n Val Loss\t{:.4f} \n Val Acc\t{:.4f} \n Val F1\t\t{:.4f} \n Val AUC\t{:.4f} \n Val GMean\t{:.4f} \n Val MCC\t{:.4f}".format(epoch, np.mean(loss_buf), np.mean(val_loss_buf), acc, f1, auc, gmean, mcc))
                
                if np.mean(val_loss_tmp)<best_val_loss:
                    best_val_loss = np.mean(val_loss_tmp)
                    torch.save(config["model"], "{}/model_{}_best.pth".format(result_path, k_iter))
                    np.save("{}/val_acc_{}_best.npy".format(result_path, k_iter), acc)
                    np.save("{}/val_f1_{}_best.npy".format(result_path, k_iter), f1)
                    np.save("{}/val_auc_{}_best.npy".format(result_path, k_iter), auc)
                    np.save("{}/val_gmean_{}_best.npy".format(result_path, k_iter), gmean)
                    np.save("{}/val_mcc_{}_best.npy".format(result_path, k_iter), mcc)

            config["model"].train()
        
        torch.save(config["model"], "{}/model_{}.pth".format(result_path, k_iter))
        np.save("{}/loss_buf_{}.npy".format(result_path, k_iter), np.array(loss_buf))
        np.save("{}/val_loss_buf_{}.npy".format(result_path, k_iter), np.array(val_loss_buf))
        np.save("{}/val_acc_buf_{}.npy".format(result_path, k_iter), np.array(val_acc_buf))
        np.save("{}/val_f1_buf_{}.npy".format(result_path, k_iter), np.array(val_f1_buf))
        np.save("{}/val_auc_buf_{}.npy".format(result_path, k_iter), np.array(val_auc_buf))
        np.save("{}/val_gmean_buf_{}.npy".format(result_path, k_iter), np.array(val_gmean_buf))
        np.save("{}/val_mcc_buf_{}.npy".format(result_path, k_iter), np.array(val_mcc_buf))
        
        
        kfold_labels.append(labels)
        kfold_preds.append(preds)
        
    #     break

    res = evaluate(model_name=config["model_name"], kfold=config["kfold"], result_path=result_path)

    return res


if __name__=='__main__':

    set_seed(seed=3407)
    # set_seed(seed=42)

    # model_name = "masonscnn"
    # model_name = "lstm"
    # model_name = "textcnn"
    # model_name = "ag_fast_parapred"
    # model_name = "pipr"
    # model_name = "resppi"
    # model_name = "pesi"

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='config file path')
    args = parser.parse_args()
    print(args.config)

    with open(args.config) as json_file:
        config = json.load(json_file)
    
    # alter int to boolean
    for k in config:
        if config[k]==1:
            config[k]=True
        if config[k]==0:
            config[k]=False

    # print(config)

    # training
    current_time = time.strftime('%m-%d-%H-%M-%S', time.localtime())
    result_path = "./results/CoV-AbDab/{}_{}".format(config["model_name"], current_time)
    os.makedirs(result_path)

    shutil.copy(args.config, result_path)

    for i in range(config["ntimes"]):
        print("Run {} times of {}fold".format(config["ntimes"], config["kfold"]))
        result = cov_train(config=config, result_path=result_path)
        print("Results dump to: ")
        print("{}/result_{}.pkl".format(result_path, i))
        pickle.dump(result, open("{}/result_{}.pkl".format(result_path, i), "wb"))
        # with open("{}/result_{}.pkl".format(result_path, i), "wb") as f:
        #     pickle.dump(result, f)