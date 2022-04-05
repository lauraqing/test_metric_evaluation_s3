import json
from stage3.NumpyEncoder import NumpyEncoder
from stage3.calculate_accuracy import calculate_accuracy
from stage3.calculate_sensitivity import calculate_sensitivity
from stage3.calculate_specificity import calculate_specificity
from stage3.calculate_AUC import calculate_AUC
from stage3.calculate_SSEP import calculate_SSEP

def metric_calculation(model_results, input_json_file):
    '''
    :param model_results: [TP, FP, TN, FN, pred, pred_prob, test_data, test_flag]
    :param input_json_file: metric_input.json
    :return:
    '''
    f = open(input_json_file) #metric_input.json
    data_jsonIn = json.load(f)

    #using metric_input.json configurations to launch metric calculation
    for i in data_jsonIn['metrics']:
        if i == "Accuracy":
            accuracy = calculate_accuracy(model_results)
        if i == "Sensitivity":
            sensitivity = calculate_sensitivity(model_results)
        if i == "Specificity": #negative class situation
            specificity = calculate_specificity(model_results)
        if i == "AUC":
            AUC = calculate_AUC(model_results)
        if i == "SSEP": #SensitivitySpecificityEquivalencePoint
            SSEP = calculate_SSEP(model_results)

    metric_results = [accuracy, sensitivity, specificity, AUC, SSEP] #, lower_bound, upper_bound]

    # save model_outputs & gt_labels into "middleStage_MetricInput.json" for later usages
    metrics = []
    for i in data_jsonIn['metrics']:
        metrics.append(i)
    print(metrics)

    for i in data_jsonIn:
        if i == 'threshold': #training & test data segmentation ratio for accuracy
            threshold = data_jsonIn["threshold"]
        if i == 'ci': # if True include confidence interval in output
            ci = data_jsonIn["ci"] #could be true/false inside the Json file just read it from default settings
        if i == 'alpha':  # float, alpha parameter for confidence level of the interval
            alpha = data_jsonIn["alpha"]
        if i == 'num_bootstraps': #number of iterations of bootstrapping to compute confidence interval
            num_bootstraps = data_jsonIn["num_bootstraps"]

    jsonString = json.dumps({"metrics": metrics,
                             "model_outputs": model_results[5],
                             "gt_labels": model_results[7],
                             "threshold": threshold,
                             "ci": ci,
                             "num_bootstraps": num_bootstraps,
                             "alpha": alpha
                             },
                             cls=NumpyEncoder, indent=4)
    # print(jsonString)
    data_jsonOut = r'middleStage_MetricInput.json' #data_jsonIn
    with open(data_jsonOut, 'w', encoding='utf-8') as json_out:
        json_out.write(jsonString)
    # print(">>> Finished loading input metric json files. >>>")

    return metric_results

# # unit-test#1 3/28/2022 QW:
# #merge test-1 for model_training
# # unit-test-2 3/30/2022 QW: compiled. 3/30/2022 5:45AM
# json_tarFile = r'systemConfiguration_2022-03-29_15_52_25.240380.json' #ratio=0.7 test
# model_type = 'LogisticRegression'
# from stage3.model_training import model_training
# model_results = model_training(json_tarFile, model_type) #[TP, FP, TN, FN, pred, pred_prob, test_data, test_flag]
# # print("Model Results are as:", model_results) #[99, 0, 0, 6]
# # unit-test 3/28/2022 QW: ratio=0.7 passed
# json_tarFile = r'metric_input.json'
# metric_calculation(model_results, json_tarFile)