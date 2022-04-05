def flag_StringToInt(file_in):
    '''
    # file_in = r'train_flag.csv'
    :param train_flag:
    :return:
    '''

    train_flag = []
    with open(file_in, 'r') as f:
        lines = f.readlines()
        train_flag.append(lines)
    # print(len(train_flag), "x", len(train_flag[0])) #1 x 58

    train_flag_boolen = []
    for items in train_flag:
        # print("--------")
        # print(items)
        datasets = str(items).split(',')
        for data in datasets:
            if "setosa" in str(data):
                train_flag_boolen.append(0)
            elif 'versicolor' in str(data):
                train_flag_boolen.append(1)
            elif 'virginica' in str(data):
                train_flag_boolen.append(2)

    print(">>> Finished Digitalized Datatype:", train_flag_boolen)

    return train_flag_boolen
