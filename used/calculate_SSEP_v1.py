import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve

def calculate_SSEP(model_results):
    '''
    :param model_results: model_results=[TP, FP, TN, FN, pred, pred_prob, test_data, test_flag]
    :return: SSEP
    '''
    true_positive, false_positive, true_negative, false_negative = model_results[0], model_results[1], model_results[2], \
                                                                   model_results[3]
    pred, pred_prob, test_data, test_flag = model_results[4], model_results[5], model_results[6], model_results[7]
    fpr, tpr, thresholds = metrics.roc_curve(test_flag, pred, pos_label=1)

    print(">>> Calculte and plot specificity vs. Sensitivity: >>>")
    # total = (true_positive + false_positive + true_negative + false_negative)
    # total_positive = true_positive + false_negative
    # total_negative = true_negative + false_positive
    # specifity = true_negative/total_negative #specifity = 1 - FPR
    # tpr = true_positive/total_positive       #sensitivity

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}
    n_class = 3

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(test_flag, pred_prob[:, i], pos_label=i)
    # plotting SSEP
    plt.plot(1 - fpr[0], tpr[0], linestyle='--', color='orange', label='Class 0 vs Rest')
    plt.plot(1 - fpr[1], tpr[1], linestyle='--', color='green', label='Class 1 vs Rest')
    plt.plot(1 - fpr[2], tpr[2], linestyle='--', color='blue', label='Class 2 vs Rest')
    plt.title('Multiclass SSEP Test curve')
    plt.xlabel('1 - False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    # plt.show() #
    plt.savefig('Multiclass SSEP', dpi=300)

# # v02: unit-test-2 3/30/2022 QW:  compiled. 5:15AM
# json_tarFile = r'systemConfiguration_2022-03-29_15_52_25.240380.json' #ratio=0.7 test
# model_type = 'LogisticRegression'
# from stage2.model_training import model_training
# model_results = model_training(json_tarFile, model_type) #[TP, FP, TN, FN, pred, pred_prob, test_data, test_flag]
# # print("Model Results are as:", model_results[4]) #[99, 0, 0, 6, pred, pred_prob, test_data, test_flag]
# # print("Model Results are as:", model_results[5])
# # print("Model Results are as:", model_results[6])
# # print("Model Results are as:", model_results[7])
#
# calculate_SSEP(model_results) #AUC is :  0.5961700336700336

