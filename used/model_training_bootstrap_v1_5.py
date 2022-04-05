from stage3.txt_srcData_load import txt_srcData_load
from stage3.flag_StringToInt import flag_StringToInt
from sklearn.utils import resample
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
    #Type1
    input_txt_file = r'../irisdata.txt'
    [train_data, test_data, train_flag, test_flag] = txt_srcData_load(input_txt_file, segmentation_ratio)
    # print("---AfterRandomGenerator--->")
    # print("Train Data Information: ", (len(train_data)), "x", len(train_data[0])) #87 x 4
    # print("Train Flag Information: ", (len(train_flag)), "x", len(train_flag[0])) #63 x 11 <--StringType

    # flag switch function
    # print("---AfterDataFormatTransfer--->")
    train_data = data_StringToInt(r'train_data.csv')
    # print("Train Data Information: ", (len(train_data)), "x")#, len(train_data[0])) #
    train_flag = flag_StringToInt(r'train_flag.csv') #StringFlag to Int
    # print(train_flag)
    # print("Train Flag Information: ", (len(train_flag)), "x 1") #63 x 1 <---IntType

    print(train_data)
    print(">>> Applied Bootstrap in the number iterations as:", num_bootstraps)
    bootstrap_X = resample(train_data, n_samples=num_bootstraps, replace=True)
    bootstrap_Y = resample(train_flag, n_samples=num_bootstraps, replace=True)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(train_data, train_flag)

    pred = clf.predict(test_data)
    pred_prob = clf.predict_proba(test_data)
    model_results_tmp = [pred, pred_prob, test_data, test_flag]

    #model_results_TF_counter:
    pred, pred_prob, test_data, test_flag = model_results_tmp[0], model_results_tmp[1], model_results_tmp[2], model_results_tmp[3]
    print(">>> Loaded prediction results with size 1x", len(pred), "; Sample Data:", pred[0]) # 1x105: [1 0 2 1 1 0 1 ...
    print(">>> Loaded test_flag results with size 1x", len(test_flag), "; Sample Data:", test_flag[0]) # 105:[1 0 2 1 1 0 1 ..
    print(">>> Loaded pred_prob results with size 3x", len(pred_prob), "; Sample Data:", pred_prob[0]) # 3x105: [[0.01535705 0.65225435 0.3323886 ]
    print(">>> Loaded test_data results with size 4x", len(test_data), "; Sample Data:", test_data[0]) # 4x105: [[6.1 2.8 4.7 1.2]

    [TP_counter, FP_counter, TN_counter, FN_counter] = results_counter(pred, test_flag, threshold)
    model_results = [TP_counter, FP_counter, TN_counter, FN_counter, pred, pred_prob, test_data, test_flag] #TP, FP, TN, FN
    print(model_results[:4])
    model_results = model_training_bootstrap(model_type, segmentation_ratio, bootstrap_parameters)

