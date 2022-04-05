import json

def define_bootstrap_parameters(input_json_file, num_bootstraps, threshold, alpha_confidence_interval, confidence_interval_flag):
    '''
    :param input_json_file: metric_input.json
    :param num_bootstraps: int
    :param threshold: float
    :param alpha_confidence_interval: float
    :param confidence_interval_flag: boolean
    :return: bootstrap_parameters = [num_bootstraps, threshold, alpha_confidence_interval, confidence_interval_flag]
    '''

    # Init to bypass corner cases:
    bootstrap_parameters = [0, 0.0, 0.0, False]

    # Priority Setting Logic: Only if main.py script didn't define bootStrap parameters, we read from metric_input.json
    # followed by Familiar Testing Rules
    if not num_bootstraps or not threshold or not alpha_confidence_interval or not confidence_interval_flag:
        print(">>> Not found full variable definitions for bootstrap_parameters, read metric_input.json as default:")
        # load metric_input.json & check the variable settings
        f = open(input_json_file)
        data_jsonIn = json.load(f)

        for i in data_jsonIn:
            if i == 'threshold':
                threshold = data_jsonIn["threshold"]
            if i == 'ci':
                confidence_interval_flag = data_jsonIn["ci"]
            if i == 'alpha':
                alpha_confidence_interval = data_jsonIn["alpha"]
            if i == 'num_bootstraps':
                num_bootstraps = data_jsonIn["num_bootstraps"]

    print(">>> Loading bootstrap variables for parameter settings:")
    bootstrap_parameters = [num_bootstraps, threshold, alpha_confidence_interval, confidence_interval_flag]

    return bootstrap_parameters