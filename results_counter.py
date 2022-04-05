def results_counter(pred, test_flag, threshold):
    '''
    # threshold not applied with results_couonter QW 4/1/22
    # PrincipleDescriptions => we want to find flower #1 & #2; not flower #0:
    # calculate TP, FP, TN, FN using pred, pred_prob, test_data, test_flag
    :param pred: prediction results
    :param test_flag: ground truth
    :return:  TP_counter, FP_counter, TN_counter, FN_counter
    '''

    print(">>> Launch the metric result counter for truth/false cases:")
    # initial result metric counters and res_metric list
    TP_counter, FP_counter, TN_counter, FN_counter = 0, 0, 0, 0

    # Please note we are treating 3 different labels inputs with combined cases for binoinal cases
    for i in range(len(pred)):
         if test_flag[i] is not 0: #target
            if pred[i] == test_flag[i]:
                TP_counter += 1
                continue
            else:
                FP_counter += 1
                continue

    for i in range(len(pred)):
        if test_flag[i] == 0: #non-target
            if pred[i] == test_flag[i]:
                TN_counter += 1
                continue
            else:
                FN_counter += 1
                continue

    print("True Positive cases are ", TP_counter)
    print("True Negative cases are ", TN_counter)
    print("False Positive cases are ", FP_counter)
    print("False Negative cases are ", FN_counter)

    return TP_counter, FP_counter, TN_counter, FN_counter