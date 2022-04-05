def calculate_accuracy(model_results):
    '''
    :param model_results: [TP, FP, TN, FN]
    :return:
    '''
    true_positive, false_positive, true_negative, false_negative = model_results[0], model_results[1], model_results[2], model_results[3]

    # accuracy = 0.0 #init
    total = (true_positive + false_positive + true_negative + false_negative)

    if true_positive + true_negative == 0:
        accuracy = 0
    elif total is not 0:
        accuracy = (true_positive + true_negative) / total * 100
    elif total == 0:
        print("Warning: Dividend total amount is 0. >>>")
        accuracy = 100

    print("Accucracy =", accuracy, "%")

    return accuracy