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

    lower_bound, upper_bound = 0.0, 0.0 # Due to regular testing: not activated the bootstraps
    metric_results = [accuracy, sensitivity, specificity, AUC, SSEP, lower_bound, upper_bound]

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

    print(">>> Print middle Stage input metric json files. >>>")
    data_jsonOut = r'middleStage_MetricInput.json' #data_jsonIn
    with open(data_jsonOut, 'w', encoding='utf-8') as json_out:
        json_out.write(jsonString)

    return metric_results
