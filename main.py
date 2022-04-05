from stage3.launch_model_training import launch_model_training
from stage3.metric_calculation import metric_calculation
from stage3.model_selection import model_selection
from stage3.json_output_writer import json_output_writer
from stage3.define_bootstrap_parameters import define_bootstrap_parameters

def main():
    '''
    Launch Metric Calculations: SSEP for SensitivitySpecificityEquivalencePoint
    :return: metric_results =
    '''
    print('>>> Launching Metric Calculation Platform:')
    # 1. Parameter preSetting:
    segmentation_ratio = 0.6  # data segmentation ratio: train vs. test
    confidence_interval_flag = False  # ci for confidence interval in output
    num_bootstraps = 1000  # number of iterations of bootstrapping to compute confidence interval
    threshold = 0.7  # threshold for metrics with operating points (eg. accuracy)
    alpha_confidence_interval = 0.1  # alpha parameter for confidence level of the interval
    model_type = 'LogisticRegression'
    input_json_file, output_json_file = r'metric_input.json', r'metric_output.json'
    #load inputJson Value is not predefined with this script
    bootstrap_parameters = define_bootstrap_parameters(input_json_file, num_bootstraps, threshold,
                                                       alpha_confidence_interval, confidence_interval_flag)
    print("Bootstrap Parameters are: ", bootstrap_parameters)
    # 2. Model selection: generate a systemConfiguration file
    json_tarFile = model_selection(input_json_file, segmentation_ratio, model_type, bootstrap_parameters)

    # 3. Model training: [TP, FP, TN, FN, pred, pred_prob, test_data, test_flag]
    # model_results = launch_model_training(json_tarFile, model_type, confidence_interval_flag, bootstrap_parameters)
    model_results = launch_model_training(json_tarFile, model_type, segmentation_ratio, bootstrap_parameters)

    # 4. Metric evaluating stage
    metric_results = metric_calculation(model_results, input_json_file)
    print(">>> Returned metric results with length as : ", len(metric_results))

    # 5. Write the results into the metric_output.json
    json_output_writer(metric_results, output_json_file) #results writer into json file
    print('Program end! >>>')

if __name__ == '__main__':
    main()