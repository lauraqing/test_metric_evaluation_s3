from stage3.txt_srcData_load import txt_srcData_load
from stage3.metric_inputJson_load import metric_inputJson_load
from stage3.metric_inputJsonLoad_srcTextReader import metric_inputJsonLoad_srcTextReader

def model_selection(input_json_file, segmentation_ratio, model_type, bootstrap_parameters):
    '''
    # To apply different machine learning models
    :param input_json_file: metric_input.json
    :param segmentation_ratio: train/test data/flag segmentation ratio
    :param model_type: different machine learning models
    :param bootstrap_parameters: [num_bootstraps, threshold, alpha_confidence_interval, confidence_interval_flag]
    :return: System Configuration files with time stamps; #2nd layers of file system input
    '''

    # main testing entrance for current stage:
    if model_type == 'LogisticRegression':
        print(">>> Applying Logistic Regression:")
        json_tarFile = metric_inputJson_load(input_json_file, bootstrap_parameters)
        print(">>> Generated system configuration file. >>>")

        return json_tarFile

    # Examples: Not recommend for current stage: 'KNN' is referred a brutalForceSample
    if model_type == 'KNN':
        print(">>> Warning: You are testing a referred a brutal Force KNN Sample:")
        # open source data referred link -- https://archive.ics.uci.edu/ml/datasets/iris
        input_txt_file = r'irisdata.txt'
        [train_data, test_data, train_flag, test_flag] = txt_srcData_load(input_txt_file, segmentation_ratio)
        json_tarFile = metric_inputJsonLoad_srcTextReader(input_json_file)
        json_tarFile_KNN = [train_data, test_data, train_flag, test_flag, json_tarFile]
        print(">>> Finished data preparation & generated system configuration file. >>>")

        return json_tarFile_KNN