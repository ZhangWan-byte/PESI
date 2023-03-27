import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics as metricser
from imblearn.metrics import geometric_mean_score


def evaluate_metrics(pred_proba, label):
    '''
    :param pred:
    :param label:
    :return:
    '''
    predicts = np.around(pred_proba)

    acc = metricser.accuracy_score(y_true=label, y_pred=predicts)
    f1 = metricser.f1_score(y_true=label, y_pred=predicts)
    auc = metricser.roc_auc_score(y_true=label, y_score=pred_proba)
    gmean = geometric_mean_score(y_true=label, y_pred=predicts)
    mcc = mcc_score(pred_proba, label)

    return acc, f1, auc, gmean, mcc


def mcc_score(pred_proba, label):
    trans_pred = np.ones(pred_proba.shape)
    trans_label = np.ones(label.shape)
    trans_pred[pred_proba < 0.5] = -1
    trans_label[label != 1] = -1
    mcc = metricser.matthews_corrcoef(trans_label, trans_pred)
    return mcc


def evaluate(model_name, kfold, result_path):
    
    print("Performance of {}".format(model_name))

    val_acc_mean = []
    val_f1_mean = []
    val_auc_mean = []
    val_gmean_mean = []
    val_mcc_mean = []

    for i in range(kfold):
        
        val_acc_i = np.load("{}/val_acc_{}_best.npy".format(result_path, i))
        val_acc_mean.append(val_acc_i)
        
        val_f1_i = np.load("{}/val_f1_{}_best.npy".format(result_path, i))
        val_f1_mean.append(val_f1_i)
        
        val_auc_i = np.load("{}/val_auc_{}_best.npy".format(result_path, i))
        val_auc_mean.append(val_auc_i)
        
        val_gmean_i = np.load("{}/val_gmean_{}_best.npy".format(result_path, i))
        val_gmean_mean.append(val_gmean_i)
        
        val_mcc_i = np.load("{}/val_mcc_{}_best.npy".format(result_path, i))
        val_mcc_mean.append(val_mcc_i)
        
    print("model: {}".format(model_name))
    print("val acc mean: ", np.mean(val_acc_mean))
    print("val f1 mean: ", np.mean(val_f1_mean))
    print("val auc mean: ", np.mean(val_auc_mean))
    print("val gmean mean: ", np.mean(val_gmean_mean))
    print("val mcc mean: ", np.mean(val_mcc_mean))

    return model_name, val_acc_mean, val_f1_mean, val_auc_mean, val_gmean_mean, val_mcc_mean