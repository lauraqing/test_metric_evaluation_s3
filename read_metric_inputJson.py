import json

def read_metric_inputJson(input_json_file, alpha_confidence_interval):
    print(">>> Prepare metric_input.json file to be updated with current variables:")
    f = open(input_json_file)
    data_jsonIn = json.load(f)

    # initial system variables from inputJson file:
    metrics = []  # string list of metrics for evaluation platform
    confidence_interval = alpha_confidence_interval
    # model_outputs, gt_labels = [[]], [] #Not using for current stage, transferred to next stages

    print(">>> Read input Json file content list: ")  # Thus allowed parameter outside settings:
    for i in data_jsonIn['metrics']:
        metrics.append(i)
    print("Metrics are listed as:", metrics)  # ['Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'SSEP']

    # write inside train_data, test_data, train_flag, test_flag
    for i in data_jsonIn:
        if i == 'threshold':  # training & test data segmentation ratio for accuracy
            threshold = data_jsonIn["threshold"]
            print("Threshold is: ", threshold)
        if i == 'ci':  # if True include confidence interval in output
            confidence_interval_flag = data_jsonIn["ci"]  # could be true/false inside the Json file just read it from default settings
            print("Confidence Interval is: ", confidence_interval_flag)
        if i == 'alpha':  # float, alpha parameter for confidence level of the interval
            alpha_confidenceInterval = data_jsonIn["alpha"]
            print("Alpha parameter for confidence level of the interval is: ", alpha_confidenceInterval)
        if i == 'num_bootstraps':  # number of iterations of bootstrapping to compute confidence interval
            num_bootstraps = data_jsonIn["num_bootstraps"]
            print("The number of bootstraps is: ", num_bootstraps)

    metric_input_reader_results = [metrics, threshold, confidence_interval, alpha_confidenceInterval, num_bootstraps]

    f.close

    return metric_input_reader_results