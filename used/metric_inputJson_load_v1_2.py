import json
import csv
import numpy as np
import datetime
from stage3.NumpyEncoder import NumpyEncoder

def metric_inputJson_load(input_json_file, bootstrap_parameters=[]):
    '''
    # Generation of System Configuration files for parameter saving purposes;
    :param input_json_file: metric_input.json
    :param bootstrap_parameters: [num_bootstraps, threshold, alpha_confidence_interval, confidence_interval_flag]
    :return: System Configuration files with time stamps; #2nd layers of file system input
    '''

    # 1.1 Using main program settings to overwriting metric_input.json file parameters
    if bootstrap_parameters[3]: #confidence_interval_flag == True
        print(">>> Confidence_interval activated for bootstrap functioning:")
        num_bootstraps, threshold = bootstrap_parameters[0], bootstrap_parameters[1]
        alpha_confidence_interval, confidence_interval_flag = bootstrap_parameters[2], bootstrap_parameters[3]
        metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'SSEP']
        alpha_confidenceInterval = alpha_confidence_interval

    # Or -> 1.2. If main program settings are none, just use metric_input.json file parameters
    if not bootstrap_parameters or not bootstrap_parameters[3]: #[] or confidence_interval_flag == False
        print(">>> Warning: Variable bootstrap_parameters hasn't been defined or confidence_interval de-activated:")
        print(">>> Prepare metric_input.json file to be updated with current variables:")
        f = open(input_json_file)
        data_jsonIn = json.load(f)

        # initial system variables from inputJson file:
        metrics = [] #string list of metrics for evaluation platform
        confidence_interval = alpha_confidence_interval
        # model_outputs, gt_labels = [[]], [] #Not using for current stage, transferred to next stages

        print(">>> Read input Json file content list: ") #Thus allowed parameter outside settings:
        for i in data_jsonIn['metrics']:
            metrics.append(i)
        print("Metrics are listed as:", metrics) #['Accuracy', 'Sensitivity', 'Specificity', 'AUC', 'SSEP']

        # write inside train_data, test_data, train_flag, test_flag
        for i in data_jsonIn:
            if i == 'threshold': #training & test data segmentation ratio for accuracy
                threshold = data_jsonIn["threshold"]
                print("Threshold is: ", threshold)
            if i == 'ci': # if True include confidence interval in output
                confidence_interval = data_jsonIn["ci"] #could be true/false inside the Json file just read it from default settings
                print("Confidence Interval is: ", confidence_interval)
            if i == 'alpha':  # float, alpha parameter for confidence level of the interval
                alpha_confidenceInterval = data_jsonIn["alpha"]
                print("Alpha parameter for confidence level of the interval is: ", alpha_confidenceInterval)
            if i == 'num_bootstraps': #number of iterations of bootstrapping to compute confidence interval
                num_bootstraps = data_jsonIn["num_bootstraps"]
                print("The number of bootstraps is: ", num_bootstraps)

    # 2. prepare jsonString to write into system configuration json file as temporary saving purposes
    jsonString = json.dumps({"metrics_init": metrics,
                             "confidence_interval_flag": confidence_interval_flag,
                             "ratio": threshold,
                             "alpha_confidenceInterval": alpha_confidenceInterval,
                             "num_bootstraps": num_bootstraps
                             },
                             cls=NumpyEncoder, indent=4)

    json_tarFile = r'systemConfiguration_%s.json' %(str(datetime.datetime.now()).replace(' ', '_').replace(':', '_'))
    print(json_tarFile)
    with open(json_tarFile, 'w', encoding='utf-8') as json_out:
        json_out.write(jsonString)
        print(">>> Finished loading input metric json files. >>>")
    # print(jsonString) #tested passed.

    f.close

    return json_tarFile