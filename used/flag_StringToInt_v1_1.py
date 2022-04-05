def flag_StringToInt(train_flag):
    '''
    # train_flag_boolen = flag_boolen_switch(train_flag)
    :param train_flag:
    :return:
    '''
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
