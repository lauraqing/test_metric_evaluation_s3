import json
import csv
import numpy as np
import datetime
from stage3.NumpyEncoder import NumpyEncoder
from stage3.read_metric_inputJson import read_metric_inputJson

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
        alpha_confidence_interval = False #default set
        metric_input_reader_results = read_metric_inputJson(input_json_file, alpha_confidence_interval)
        metrics, threshold, confidence_interval_flag = metric_input_reader_results[0], metric_input_reader_results[1], metric_input_reader_results[2]
        alpha_confidenceInterval, num_bootstraps = metric_input_reader_results[3], metric_input_reader_results[4]

    # 2. prepare jsonString to write into system configuration json file as temporary saving purposes
    jsonString = json.dumps({"metrics_init": metrics,
                             "confidence_interval_flag": confidence_interval_flag,
                             "threshold": threshold,
                             "alpha_confidenceInterval": alpha_confidenceInterval,
                             "num_bootstraps": num_bootstraps
                             },
                             cls=NumpyEncoder, indent=4)

    # write systemConfiguration_json
    json_tarFile = r'systemConfiguration_%s.json' %(str(datetime.datetime.now()).replace(' ', '_').replace(':', '_'))
    print(json_tarFile)
    with open(json_tarFile, 'w', encoding='utf-8') as json_out:
        json_out.write(jsonString)
        print(">>> Finished loading input metric json files. >>>")
    # print(jsonString)

    return json_tarFile