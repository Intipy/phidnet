import numpy as np



def encode(num, length=0):   # One-hot encoding for scalar value
    arr = np.zeros(length)
    arr[num] = 1
    return arr



def encode_array(arr, length=0):   # One-hot encoding for matrix and vector
    array = np.zeros((arr.size, length), dtype=np.int)
    for i in range(arr.size):
        array[i, arr[i]] = 1
    return array



def get_number(one_hot):   # One-hot encoding to number
    return np.argwhere(one_hot == 1)