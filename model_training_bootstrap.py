from stage3.txt_srcData_load import txt_srcData_load
from stage3.flag_StringToInt import flag_StringToInt
from sklearn.utils import resample
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from stage3.results_counter import results_counter
from stage3.data_StringToInt import data_StringToInt

def model_training_bootstrap(model_type, segmentation_ratio, bootstrap_parameters):
    '''
    :param model_type: machine learning model
    :param segmentation_ratio: data segmentation ratio
    :param bootstrap_parameters: [num_bootstraps, threshold, alpha_confidence_interval, confidence_interval_flag]
    :return: training results to the next level and save important results inside metric_input.json
    '''

    if not bootstrap_parameters[3]:
        print(">>> Detected de-activation of bootstrap function", bootstrap_parameters[3], " & exit:")
        return []
    else:
        print(">>> Detected activation of bootstrap function", bootstrap_parameters[3], " & continue:")

    print(">>> Apply Bootstrapping method for model training process:")
    if model_type is not 'LogisticRegression':
        # Display the warning since not offering KNN for current version
        print(">>> Warning: current metric evaluation platform is offered default testing with Logistic Regression.")

    # input guaranteed from main.py define_bootstrap_parameters()
    print(">>> Loading bootstrap_parameters: ")
    num_bootstraps, threshold = bootstrap_parameters[0], bootstrap_parameters[1]
    alpha_confidence_interval, confidence_interval_flag = bootstrap_parameters[2], bootstrap_parameters[3]

    ## logisticRegressionDemo with bootstrap activation
    print(">>> Applied sklearn LogisticRegression to process Iris data samples (segsegmentation_ration=", segmentation_ratio, "):")
    ##Type2_LogisticRegressionFromSklearnLib
    data = datasets.load_iris()
    X = data['data']
    Y = data['target']

    all = [] #2x150
    all[0:4] = X
    all[:,5] = Y

    print("All merged as ", len(all), "x", len(all[0]))
    print(all)
    print(">>> Applied Bootstrap in the number iterations as:", num_bootstraps)
    for i in 100:
        bootstrap_all = resample(all, n_samples=num_bootstraps, replace=True, random_state=100)
    # bootstrap_X = resample(X, n_samples=num_bootstraps, replace=True, random_state=100)
    # bootstrap_Y = resample(Y, n_samples=num_bootstraps, replace=True, random_state=100)
    # out-of-bag observations:
    # oob_X = [x for x in X if x not in bootstrap_X] #no out of bag
    # oob_Y = [y for y in Y if y not in bootstrap_Y] #no out of bag
    # print("Check if any out-of-bag:")
    # print(oob_X)
    # print(oob_Y)
    # print(bootstrap_X)
    # print("Before Bootstrap X Data Length:", len(X), "; After Bootstrap X Data Length:", len(bootstrap_X)) # 150x1000
    # print(bootstrap_Y)
    # print("Before Bootstrap Y Data Length:", len(Y), "; After Bootstrap Y Data Length:", len(bootstrap_Y)) # 150x1000


    train_data, test_data, train_flag, test_flag = train_test_split(X, Y, test_size=segmentation_ratio, random_state=42)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(train_data, train_flag)

    pred = clf.predict(test_data)
    pred_prob = clf.predict_proba(test_data)
    # print("Prediction data:", pred) #[1 0 2 1 1 0 1 2 1 1 2 0 0 0 0
    # print("Prediciton probability matrix:", pred_prob) # [8.39395247e-01 1.60440790e-01 1.63963183e-04]
    print(">>> Saved model results & Exited.")
    #starting test point #1:
    model_results_tmp = [pred, pred_prob, test_data, test_flag]

    #model_results_TF_counter:
    pred, pred_prob, test_data, test_flag = model_results_tmp[0], model_results_tmp[1], model_results_tmp[2], model_results_tmp[3]
    print(">>> Loaded prediction results with size 1x", len(pred), "; Sample Data:", pred[0]) # 1x105: [1 0 2 1 1 0 1 ...
    print(">>> Loaded test_flag results with size 1x", len(test_flag), "; Sample Data:", test_flag[0]) # 105:[1 0 2 1 1 0 1 ..
    print(">>> Loaded pred_prob results with size 3x", len(pred_prob), "; Sample Data:", pred_prob[0]) # 3x105: [[0.01535705 0.65225435 0.3323886 ]
    print(">>> Loaded test_data results with size 4x", len(test_data), "; Sample Data:", test_data[0]) # 4x105: [[6.1 2.8 4.7 1.2]

    [TP_counter, FP_counter, TN_counter, FN_counter] = results_counter(pred, test_flag, threshold)
    model_results = [TP_counter, FP_counter, TN_counter, FN_counter, pred, pred_prob, test_data, test_flag] #TP, FP, TN, FN
    # print(model_results)

    return model_results


#Merge Test: v1.7
from stage3.metric_calculation import metric_calculation

segmentation_ratio = 0.6  # data segmentation ratio: train vs. test
confidence_interval_flag = True  # ci for confidence interval in output
num_bootstraps = 1000  # number of iterations of bootstrapping to compute confidence interval
threshold = 0.7  # threshold for metrics with operating points (eg. accuracy)
alpha_confidence_interval = 0.1  # alpha parameter for confidence level of the interval
model_type = 'LogisticRegression'
input_json_file, output_json_file = r'metric_input.json', r'metric_output.json'
#load inputJson Value is not predefined with this script
bootstrap_parameters = [num_bootstraps, threshold, alpha_confidence_interval, confidence_interval_flag]
model_results = model_training_bootstrap(model_type, segmentation_ratio, bootstrap_parameters)
metric_results = metric_calculation(model_results, input_json_file)
print("===============================================================================================")
print(metric_results)