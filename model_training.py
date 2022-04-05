import json
from referencesCodeLearning.KNN import KNN
from stage3.logisticRegressionDemo import logisticRegressionDemo

def model_training(json_tarFile, model_type, segmentation_ratio):
    '''
    :param json_tarFile: metric_input.json
    :param model_type: machine learning model
    :param bootstrap_parameters: [num_bootstraps, threshold, alpha_confidence_interval]
    :return: training results to the next level and save important results inside metric_input.json
    '''

    print(">>> Loading system configuration json file: ")
    f = open(json_tarFile)
    data_jsonIn = json.load(f)

    if model_type == 'KNN':
        # loading src data from input json file:
        train_data, test_data, train_flag, test_flag = [[]], [[]], [], [[]]  # initial variables
        train_data = data_jsonIn["train_data"]  # 100 x 4
        test_data = data_jsonIn["test_data"]  # 50 x 4
        train_flag = data_jsonIn["train_flag"]  # 50 x 1
        test_flag = data_jsonIn["test_flag"]  # 50 x 1
        threshold = data_jsonIn["threshold"]
        # # tested loading passed
        # print("Train data size:", len(train_data), "x", len(train_data[0]))
        # print("Train flag size:", len(train_flag), "x", len(train_flag[0]))
        # print("Test data size:", len(test_data), "x", len(test_data[0]))
        # print("Test flag size:", len(test_flag), "x", len(test_flag[0]))
        ##Type1_refered a BruteForceKNN: not recommend for current stage; didn't apply bootstrap
        model_results = KNN(train_data, test_data, train_flag, test_flag) #results included T/F/P/N

    elif model_type == 'LogisticRegression':
        threshold = data_jsonIn["threshold"]
        model_results = logisticRegressionDemo(threshold, segmentation_ratio)
        # model_results = [TP, FP, TN, FN, pred, pred_prob, test_data, test_flag]

    return model_results