from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from stage3.results_counter import results_counter

def logisticRegressionDemo(threshold, segmentation_ratio):
    '''
    :param threshold: float
    :param segmentation_ratio: float
    :return: model_results = [TP, FP, TN, FN, pred, pred_prob, test_data, test_flag]
    '''

    print(">>> Applied sklearn LogisticRegression to process Iris data samples (segsegmentation_ration=", segmentation_ratio, "):")
    ##Type2_LogisticRegressionFromSklearnLib
    data = datasets.load_iris()
    X = data['data']
    Y = data['target']
    train_data, test_data, train_flag, test_flag = train_test_split(X, Y, test_size=segmentation_ratio, random_state=42)
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

    [TP_counter, FP_counter, TN_counter, FN_counter] = results_counter(pred, test_flag, threshold)
    model_results = [TP_counter, FP_counter, TN_counter, FN_counter, pred, pred_prob, test_data, test_flag] #TP, FP, TN, FN

    return model_results