import json
from stage3.NumpyEncoder import NumpyEncoder

def json_output_writer(metric_results, output_json_file):

    #detach metric_results
    accuracy, sensitivity, specificity, AUC, SSEP, lower_bound, upper_bound = metric_results[0],\
                                                                              metric_results[1],\
                                                                              metric_results[2],\
                                                                              metric_results[3],\
                                                                              metric_results[4],\
                                                                              metric_results[5],\
                                                                              metric_results[6]

    # save metric_results into "output_json_file.json"
    jsonString = json.dumps({"values": {
                                        "Accuracy": accuracy,
                                        "Sensitivity": sensitivity,
                                        "Specificity": specificity,
                                        "AUC": AUC,
                                        "SSEP": "Please check local image output: Multiclass SSEP.png"
                                        },
                             "lower_bound": lower_bound,
                             "upper_bound": upper_bound,
                             },
                             cls=NumpyEncoder, indent=4)

    with open(output_json_file, 'w', encoding='utf-8') as json_out:
        json_out.write(jsonString)
    print(">>> Finished saving output metric json files. >>>")

