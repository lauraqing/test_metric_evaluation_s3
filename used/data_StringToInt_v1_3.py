import numpy as np

def data_StringToInt(file_in = []):
    '''
    # file_in = r'train_data.csv'
    # train_flag_boolen = flag_boolen_switch(train_flag)
    :param train_flag:
    :return:
    '''

    file_in = r'train_data.csv'
    train_data = []
    with open(file_in, 'r') as f:
        lines = f.readlines()
        train_data.append(lines)
    print(len(train_data), "x", len(train_data[0])) #given string type as 1 x 83
    len_i, len_j = 4, len(train_data[0])
    train_data_tmp = [[0]*len_i for _ in range(len_j)] #target is 4x83
    for items in train_data:
        print("--------")
        print(items) #['4.7 3.2 1.3 0.2\n', '4.6 3.1 1.5 0.2\n',
        datasets = str(items).split(',') #  #'4.7 3.2 1.3 0.2\n': 1x20 items

        for data in datasets:
            # print(len(data)) #20...21

            for i in range(len(data)):
                data_tmp = []
                data_tmp_str = data[2:17]
                # print(type(data_tmp_str)) # str
                data_tmp = np.array(data_tmp_str)
                print(type(data_tmp)) # numpy.ndarray
                print(data_tmp) #4.7 3.2 1.3 0.2

                for res_i in range(len_i):
                    for res_j in range(len_j):
                        train_data_tmp[res_i][res_j] = data_tmp

        print(train_data_tmp)

    # print(">>> Numpy Array Datatype Output:", train_data_tmp)
    #
    # return train_data_tmp

data_StringToInt()