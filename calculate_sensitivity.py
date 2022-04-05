def calculate_sensitivity(model_results):
    print(">>> Calculating sensitivity:")
    true_positive, false_positive, true_negative, false_negative = model_results[0], model_results[1], model_results[2], model_results[3]

    # accuracy = 0.0 #init
    total = (true_positive + false_positive + true_negative + false_negative)
    total_positive = true_positive + false_negative
    total_negative = true_negative + false_positive

    sensitivity = true_positive / total_positive * 100

    if true_positive == 0:
        sensitivity = 0
    elif total_positive is not 0:
        sensitivity = true_positive / total_positive * 100
    elif total_positive == 0:
        print("Warning: Dividend total positive amount is 0. >>>")
        sensitivity = 100

    print("Sensitivity =", sensitivity, "%")

    return sensitivity