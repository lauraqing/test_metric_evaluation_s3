import numpy as np

def data_StringToInt(train_data):
    '''
    # train_flag_boolen = flag_boolen_switch(train_flag)
    :param train_flag:
    :return:
    '''
    train_data_tmp = []
    for items in train_data:
        print("--------")
        print(items) #['4.7 3.2 1.3 0.2\n', '4.6 3.1 1.5 0.2\n',
        datasets = str(items).split(',') #  #'4.7 3.2 1.3 0.2\n': 1x20 items
        # for i in len(datasets):
        # datasets_np = np.array(datasets)

        for data in datasets:
            # print(len(data)) #20...21

            for i in range(len(data)):
                data_tmp = []
                data_tmp_str = data[2:17]
                # print(type(data_tmp_str)) # str
                data_tmp = np.array(data_tmp_str)
                # print(type(data_tmp)) # numpy.ndarray
                # print(data_tmp) #4.7 3.2 1.3 0.2
                train_data_tmp.append(np.array(data_tmp))
        print(train_data_tmp)

    print(">>> Numpy Array Datatype Output:", train_data_tmp)

    return train_data_tmp

#unitest: passed QW 4/4/2022 4:28PM
file_in = r'train_data.csv'
train_data = []
with open(file_in, 'r') as f:
    lines = f.readlines()
    train_data.append(lines)
# print(len(train_flag), "x", len(train_flag[0])) #1 x 58

train_data_array = data_StringToInt(train_data)
