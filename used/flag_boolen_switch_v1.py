def flag_StringToInt(train_flag):
    '''
    # train_flag_boolen = flag_boolen_switch(train_flag)
    :param train_flag:
    :return:
    '''
    train_flag_boolen = []
    for items in train_flag:
        print("--------")
        print(items)
        datasets = str(items).split(',')
        for data in datasets:
            if "setosa" in str(data):
                train_flag_boolen.append(0)
            elif 'versicolor' in str(data):
                train_flag_boolen.append(1)
            elif 'virginica' in str(data):
                train_flag_boolen.append(2)

    print(">>> Digitalized Datatype:", train_flag_boolen)

    return train_flag_boolen

#unitest: passed QW 4/4/2022 3:47pm =>< change name into flag_StringToInt
file_in = r'train_flag.csv'
train_flag = []
with open(file_in, 'r') as f:
    lines = f.readlines()
    train_flag.append(lines)
# print(len(train_flag), "x", len(train_flag[0])) #1 x 58

train_flag_boolen = flag_boolen_switch(train_flag)
