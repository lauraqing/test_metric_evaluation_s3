import numpy as np
import random

def txt_srcData_load(file_in, ratio):
    print(">>> Reading input file:")
    train_set = []
    test_set = []

    with open(file_in, 'r') as f:
        lines = f.readlines()
        idx = 1
        print(lines) # load_in data checker
        for line in lines:
            # print(line)
            if line[0] == "/n":
                # print(line)
                print("End of file, exit： >>>")
                break
            idx += 1
            datasets = line.split(',') #sample output: ['5.1', '3.5', '1.4', '0.2', 'Iris-setosa\n']
            # print(datasets)
            dataset = np.array(datasets)

            if random.random() < ratio:
                train_set.append(dataset)
            else:
                test_set.append(dataset)
    print("Input matrix size as ", len(datasets), " x ", idx) #1x152

    # segment flags from src datasets
    print("train data matrix size: ", len(train_set), "x", len(train_set[0])) #114 x 5
    print("train data matrix size: ", len(test_set), "x", len(test_set[0]))   #37 x 5

    print(">>>")
    idx_flag = 0
    train_data, test_data, train_flag, test_flag, tmp = [[]], [[]], [], [[]], []
    for i in range(len(train_set)):
        train_data_tmp = []
        for j in range(len(train_set[0])): # data src file has 2 empty lines cause issues
            if j is not 4 and train_set[i][j] is not ['\n']:
                # print(train_set[i][:4])
                train_data_tmp.append(train_set[i][j]) #[0->4)
        train_flag.append(train_set[i][4])  # 5th element
        # print(i, train_data_tmp)
        # print(len(train_data_tmp))
        if len(train_data_tmp):
            train_data.append(train_data_tmp)
            # print(train_data_tmp)

    for i in range(len(test_set)):
        test_set_tmp = []
        for j in range(len(test_set[0])): # data src file has 2 empty lines cause issues
            # print(train_set[:][j])
            if j is not 4 and test_set[i][j] is not ['\n']:
                # print(train_set[i][:4])
                test_set_tmp.append(test_set[i][j]) #[0->4)
        test_flag.append(test_set[i][4])  # 5th element
        # print(i, train_data_tmp)
        # print(len(test_set_tmp))
        if len(test_set_tmp):
            test_data.append(test_set_tmp)

    # delete the head empty element
    train_data = [x for x in train_data if x]
    test_data = [x for x in test_data if x]
    test_flag = [x for x in test_flag if x]

    # delete flag '\n'
    print('----------------------------------------------------------')
    test_flag_clean = []
    for string in test_flag:
        newString = string.replace("\n", "")
        # print(newString)
        test_flag_clean.append(newString)

    train_flag_clean = []
    for string in test_flag:
        newString = string.replace("\n", "")
        # print(newString)
        train_flag_clean.append(newString)

    print(">>> Check random generated & segmented training & testing datasets:")
    print("training data as:", train_data) #looks fine, except the first is an empty []
    print("training flag as:", train_flag_clean) #looks fine, except the first is an empty []
    print("testing data as:", test_data)   #looks fine
    print("testing flag as:", test_flag_clean)   #looks fine

    # # check train_set
    print(">>> Return train_set & test_set, exit. >>>")
    print(">>>")
    print("Train data size:", len(train_data), "x", len(train_data[0]))
    print("Train flag size:", len(train_flag), "x", len(train_flag[0]))
    print("Test data size:", len(test_data), "x", len(test_data[0]))
    print("Test flag size:", len(test_flag), "x", len(test_flag[0]))

    # save datasets into csv files
    np.savetxt("train_data.csv", np.array(train_data), delimiter=" ", newline="\n", fmt="%s")  # working
    np.savetxt("test_data.csv", np.array(test_data), delimiter=" ", newline="\n", fmt="%s")  # working
    np.savetxt("train_flag.csv", np.array(train_flag_clean), delimiter=" ", newline="\n", fmt="%s")  # working
    np.savetxt("test_flag.csv", np.array(test_flag_clean), delimiter=" ", fmt="%s")  # working

    return train_data, test_data, train_flag_clean, test_flag_clean

# #unit test passed: QW 3/18/22; upgrading test passed: 3/23/2022
# print('>>> Testing metric reader & writer using Iris training datasets:')
# input_txt_file = r'E:\Projects\Practice\pythonProject\test\codeBase\stage2\irisdata.txt'
# ratio = 0.7  # train vs. test ratio
# # note required input file for json file as well
# [train_data, test_data, train_flag, test_flag] = txt_srcData_load(input_txt_file, ratio)
#
# print("End of program. >>>")