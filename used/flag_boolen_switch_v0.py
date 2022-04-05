def flag_boolen_switch(train_flag):
    '''
    # train_flag_boolen = flag_boolen_switch(train_flag)
    :param train_flag:
    :return:
    '''
    train_flag_boolen = []
    for i in train_flag:
        print("--------")
        print(i)
        if "setosa" in i:
            print("1")
        elif 'versicolor' in i:
            print("2")
        elif 'virginica' in i:
            print("3")

    return train_flag_boolen

#unitest:
file_in = r'train_flag.csv'
train_flag = []
with open(file_in, 'r') as f:
    lines = f.readlines()
    idx = 1
    # print(lines)  #1 x 58
    for line in lines:
        # print(line)
        if line[0] == "/n":
            # print(line)
            print("End of file, exit： >>>")
            break
        idx += 1
        data = line.split(',')  # sample output: ['5.1', '3.5', '1.4', '0.2', 'Iris-setosa\n']
        # print(data) #['Iris-setosa\n']
        if "setosa" in str(data):
            print("0")

        # train_flag.append(data_tmp)

# print(len(train_flag), "x", len(train_flag[0]))

# train_flag_boolen = flag_boolen_switch(train_flag)