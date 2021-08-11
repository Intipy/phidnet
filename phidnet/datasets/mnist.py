import numpy as np
import csv
import os


def csv2list(filename):
    file = open(os.path.realpath(__file__)[:-8] + filename, 'r', encoding='utf-8')
    csvfile = csv.reader(file)
    lists = []
    for item in csvfile:
        lists.append(item)
    return lists


def load():
    dir = os.path.realpath(__file__)[:-8]

    data = np.load(dir + 'mnist_train.npy')

    X = data[0:, 1:]
    T = data[0:, :1]
    data_test = np.load(dir + 'mnist_test.npy')
    x_test = data_test[0:, 1:]
    t_test = data_test[0:, :1]
    return X, T, x_test, t_test


def load_2d():
    dir = os.path.realpath(__file__)[:-8]

    data = np.load(dir + 'mnist_train.npy')

    X = data[0:, 1:]
    T = data[0:, :1]
    data_test = np.load(dir + 'mnist_test.npy')
    X_test = data_test[0:, 1:]
    T_test = data_test[0:, :1]

    X = X.reshape(42000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)

    return X, T, X_test, T_test

