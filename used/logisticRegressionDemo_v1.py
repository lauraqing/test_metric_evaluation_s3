from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn import datasets

def logisticRegressionDemo(ratio):

    print(">>> Applied sklearn LogisticRegression to process Iris data samples (segRation=", ratio, "):")
    ##Type2_LogisticRegressionFromSklearnLib
    data = datasets.load_iris()
    X = data['data']
    Y = data['target']
    train_data, test_data, train_flag, test_flag = train_test_split(X, Y, test_size=ratio, random_state=42)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(train_data, train_flag)
    pred = clf.predict(test_data)
    pred_prob = clf.predict_proba(test_data)
    # print("Prediction data:", pred) #[1 0 2 1 1 0 1 2 1 1 2 0 0 0 0
    # print("Prediciton probability matrix:", pred_prob) # [8.39395247e-01 1.60440790e-01 1.63963183e-04]
    print(">>> Saved model results & Exited.")
    #starting test point #1:
    model_results_tmp = [pred, pred_prob, test_data, test_flag]

    #model_results_TF_counter:
    pred, pred_prob, test_data, test_flag = model_results_tmp[0], model_results_tmp[1], model_results_tmp[2], model_results_tmp[3]
    print(">>> Loaded prediction results with size 1x", len(pred), "; Sample Data:", pred[0]) # 1x105: [1 0 2 1 1 0 1 ...
    print(">>> Loaded test_flag results with size 1x", len(test_flag), "; Sample Data:", test_flag[0]) # 105:[1 0 2 1 1 0 1 ..
    print(">>> Loaded pred_prob results with size 3x", len(pred_prob), "; Sample Data:", pred_prob[0]) # 3x105: [[0.01535705 0.65225435 0.3323886 ]
    print(">>> Loaded test_data results with size 4x", len(test_data), "; Sample Data:", test_data[0]) # 4x105: [[6.1 2.8 4.7 1.2]

    # calculate TP, FP, TN, FN using pred, pred_prob, test_data, test_flag
    # initial result metric counters and res_metric list
    TP_counter, FP_counter, TN_counter, FN_counter, model_results = 0, 0, 0, 0, []
    # Positive_counter2, OtherCase_counter0_1 = 0, 0
    # TP_rate, FP_rate, TN_rate, FN_rate = 0.0, 0.0, 0.0, 0.0
    # we want to find flower #1 & #2; not philociphay not #0:
    for i in range(len(pred)):
         if test_flag[i] is not 0: #target
            if pred[i] == test_flag[i]:
                TP_counter += 1
                continue
            else:
                FP_counter += 1
                continue

    for i in range(len(pred)):

        if test_flag[i] == 0: #non-target
            if pred[i] == test_flag[i]:
                TN_counter += 1
                continue
            else:
                FN_counter += 1
                continue

    print("True Positive cases are ", TP_counter)
    print("True Negative cases are ", TN_counter)
    print("False Positive cases are ", FP_counter)
    print("False Negative cases are ", FN_counter)
    model_results = [TP_counter, FP_counter, TN_counter, FN_counter, pred, pred_prob, test_data, test_flag] #TP, FP, TN, FN

    return model_results

# # #v0.6: fixed the issue that logic loop didn't work with 3-cases; QW 3/29/2022 7pm
# # Accucracy = 96.8 %
# # Precision = 95.55555555555556 %
# # Recall = 100.0 %
# # F1 Score = 97.72727272727273 %
# ratio = 0.6
# model_results = logisticRegressionDemo(ratio)
# TP, FP, TN, FN = model_results[0], model_results[1], model_results[2], model_results[3]
# print(model_results) #[86, 0, 0, 4]
# total = TP+FP+TN+FN
# Accucracy = (TP+TN)/total * 100
# Precision = (TP)/(TP+FP) * 100
# Recall = (TP)/(TP+FN) * 100
# print("Accucracy =", Accucracy, "%")
# print("Precision =", Precision, "%")
# print("Recall =", Recall, "%")
# print("F1 Score =", 2/(1/Recall + 1/Precision), "%")