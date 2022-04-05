import json
import csv
import numpy as np
import datetime
from stage3.NumpyEncoder import NumpyEncoder

def metric_inputJson_load(input_json_file):
    '''
    :param input_json_file:
    --> Note:  train_data, test_data, train_flag, test_flag are read from local base
    :return: big Json file
    '''

    print(">>> Load train/test matrix from upper random generator :")
    # Load training & testing data and flags
    train_data_srcFile, train_flag_srcFile = r'train_data.csv', r'train_flag.csv'
    test_data_srcFile, test_flag_srcFile = r'test_data.csv', r'test_flag.csv'

    def read_csv(csv_srcFile):
        csv_obj = []
        with open(csv_srcFile, encoding='utf-8') as csv_in:
            #csv.DictReader brings problems you previously noticed; below is working
            csv_lines = csv.reader(csv_in, delimiter=' ') # this part got error if use csr_src
            #csv data transfer to matrix
            for row in csv_lines:
                row = np.array(row)
                csv_obj.append(row)
        return csv_obj

    train_data = read_csv(train_data_srcFile)
    train_flag = read_csv(train_flag_srcFile)
    test_data = read_csv(test_data_srcFile)
    test_flag = read_csv(test_flag_srcFile)

    print("Train data size:", len(train_data), "x", len(train_data[0]))
    print("Train flag size:", len(train_flag), "x", len(train_flag[0]))
    print("Test data size:", len(test_data), "x", len(test_data[0]))
    print("Test flag size:", len(test_flag), "x", len(test_flag[0]))

    print(">>> Prepare metric_input.json file to be updated with current variables:")
    f = open(input_json_file)
    data_jsonIn = json.load(f)

    # initial system variables from inputJson file:
    metrics = [] #string list of metrics for evaluation platform
    threshold, confidence_interval, num_bootstraps, alpha_confidenceInterval = 0, False, 0, 0.0
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

    # prepare jsonString to write into system configuration json file as temporary saving purposes
    jsonString = json.dumps({"metrics_init": metrics,
                             "confidence_interval": confidence_interval,
                             "ratio": threshold,
                             "alpha_confidenceInterval": alpha_confidenceInterval,
                             "num_bootstraps": num_bootstraps,
                             "train_data": train_data,
                             "train_flag": train_flag,
                             "test_data": test_data,
                             "test_flag": test_flag
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

# # v03_unit test: passed. QW 3/38/2022; timestamp adding succeed; hist bug fixed.
# print('>>> Testing Json Reader:')
# input_json_file = r'metric_input.json'
# json_tarFile = metric_inputJson_load(input_json_file)
# print("End of program unitest. >>>")