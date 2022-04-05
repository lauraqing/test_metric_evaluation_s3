import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def calculate_AUC(model_results):
    '''
    :param model_results: model_results=[TP, FP, TN, FN, pred, pred_prob, test_data, test_flag]
    :return: auc
    '''
    print(">>> Calculating AUC value:")
    true_positive, false_positive, true_negative, false_negative = model_results[0], model_results[1], model_results[2], model_results[3]
    pred, pred_prob, test_data, test_flag = model_results[4], model_results[5], model_results[6], model_results[7]
    # fpr = false_positive / (false_positive + true_negative)
    # tpr = true_positive / (true_positive + false_positive)

    # fpr, tpr, thresholds = metrics.roc_curve(test_flag, pred, pos_label=1)
    # auc = metrics.auc(fpr, tpr)

    #multi-case
    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}
    auc = {}
    n_class = 3

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(test_flag, pred_prob[:, i], pos_label=i)
        auc[i] = metrics.auc(fpr[i], tpr[i])

    # print("AUC is : ", auc)
    print("AUC-class-1 is : ", auc[0])
    print("AUC-class-2 is : ", auc[1])
    print("AUC-class-3 is : ", auc[2])

    return auc