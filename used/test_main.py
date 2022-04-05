# v1 => renamed as main.py : test_main for main structure
from stage3.txt_srcData_load import txt_srcData_load
from stage3.metric_inputJson_load import metric_inputJson_load
from stage3.model_training import model_training
from stage3.metric_calculation import metric_calculation
from stage3.json_output_writer import json_output_writer

if __name__ == '__main__':
    #merge Test Aim 1-> use ratio 0.7; Aim 2 -> 0.6
    print('>>> Testing metric reader & writer using IrIs dataset:')
    # source open data reference link -- https://archive.ics.uci.edu/ml/datasets/iris
    input_txt_file = r'/stage3/irisdata.txt'
    input_json_file = r'/stage3/metric_input.json'
    output_json_file = r'/stage3/metric_output.json'
    ratio = 0.7 #train vs. test ratio -> update this value from metric_input.json threshold
    model_type = 'LogisticRegression' #'KNN' is a bruteForceType referred not suggest to use for now

    # note required input file for json file as well
    [train_data, test_data, train_flag, test_flag] = txt_srcData_load(input_txt_file, ratio)
    json_tarFile = metric_inputJson_load(input_json_file)
    model_results = model_training(json_tarFile, model_type) # [TP, FP, TN, FN, pred, pred_prob, test_data, test_flag]
    # print("MergeTest_#1", model_results) # passed QW 3/29/2022 3:30pm
    #merge_test2 failed: not sure why. will check later. 3:55pm
    metric_results = metric_calculation(model_results, input_json_file) # #metric calculations: SSEP for SensitivitySpecificityEquivalencePoint
    json_output_writer(metric_results, output_json_file) #results writer into json file
    print('Program end! >>>')