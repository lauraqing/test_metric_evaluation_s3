def calculate_specificity(model_results):
    print(">>> Calculating specificity: ")
    true_positive, false_positive, true_negative, false_negative = model_results[0], model_results[1], model_results[2], model_results[3]

    # accuracy = 0.0 #init
    total = (true_positive + false_positive + true_negative + false_negative)
    total_positive = true_positive + false_negative
    total_negative = true_negative + false_positive

    if true_negative == 0:
        specificity = 0
    elif total_negative is not 0:
        specificity = true_negative / total_negative * 100
    elif total_negative == 0:
        print("Warning: Dividend total amount is 0. >>>")
        specificity = 100

    print("Specificity =", specificity, "%")

    return specificity