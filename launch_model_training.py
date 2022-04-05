from stage3.model_training_bootstrap import model_training_bootstrap
from stage3.model_training import model_training

def launch_model_training(json_tarFile, model_type, segmentation_ratio, bootstrap_parameters=[]):
    '''
    :param json_tarFile: auto-load most recent systemConfiguration file
    :param model_type: machine learnring type models to specify
    :param confidence_interval_flag: False/True to activate bootstraps
    :param bootstrap_parameters = [num_bootstraps, threshold, alpha_confidence_interval, confidence_interval_flag]
    :return:
    '''

    # Read in confidence_interval_flag from bootstrap_parameters;
    ## check if variable has been defined properly or use default value
    if not bootstrap_parameters and len(bootstrap_parameters) == 4:
        confidence_interval_flag = bootstrap_parameters[3]
    else:
        confidence_interval_flag = False  #default setting

    # BootStrap Activation/De-activation
    if confidence_interval_flag:
        print(">>> Apply Boot Strapping for model training:")
        model_results = model_training_bootstrap(model_type, segmentation_ratio, bootstrap_parameters)
    else:
        print(">>> Launching model training:")
        model_results = model_training(json_tarFile, model_type, segmentation_ratio) #without bootstrap activation
        # model_results = [TP, FP, TN, FN, pred, pred_prob, test_data, test_flag]

    return model_results