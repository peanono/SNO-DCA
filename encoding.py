import numpy as np


# 说明： one of K编码
# 输入： data
# 输出： data_X, data_Y
def one_hot(data, windows=41):
    # define input string
    data = data
    length = len(data)
    # define empty array
    data_X = np.zeros((length, windows, 21))
    data_Y = []
    for i in range(length):
        x = data[i].split()
        # get label
        data_Y.append(int(x[1]))
        # define universe of possible input values
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX-BJOUZ'
        # define a mapping of chars to integers
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        # integer encode input data
        integer_encoded = [char_to_int[char] for char in x[2]]
        # one hot encode
        j = 0
        for value in integer_encoded:
            if value in [21, 22, 23, 24, 25, 26]:
                for k in range(21):
                    data_X[i][j][k] = 0.05
            else:
                data_X[i][j][value] = 1.0
            j = j + 1
    data_Y = np.array(data_Y)

    return data_X, data_Y