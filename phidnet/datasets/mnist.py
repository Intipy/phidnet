import numpy as np
import csv
import os


def csv2list(filename):
    csv_file = open(os.path.realpath(__file__)[:-8] + filename, 'r', encoding='utf-8')
    csv_read = csv.reader(csv_file)
    lists = [item for item in csv_read]
    return lists



def load():
    csv_file = open(os.path.realpath(__file__)[:-8] + "mnist_train.csv", 'r', encoding='utf-8')
    csv_read = csv.reader(csv_file)
    train = np.array([item for item in csv_read], dtype='int16')

    X = train[0:, 1:]
    T = train[0:, :1]


    csv_file = open(os.path.realpath(__file__)[:-8] + "mnist_test.csv", 'r', encoding='utf-8')
    csv_read = csv.reader(csv_file)
    test = np.array([item for item in csv_read], dtype='int16')

    x_test = test[0:, 1:]
    t_test = test[0:, :1]

    return X, T, x_test, t_test


def load_2d():
    csv_file = open(os.path.realpath(__file__)[:-8] + "mnist_train.csv", 'r', encoding='utf-8')
    csv_read = csv.reader(csv_file)
    train = np.array([item for item in csv_read], dtype='int16')

    X = train[0:, 1:]
    T = train[0:, :1]


    csv_file = open(os.path.realpath(__file__)[:-8] + "mnist_test.csv", 'r', encoding='utf-8')
    csv_read = csv.reader(csv_file)
    test = np.array([item for item in csv_read], dtype='int16')

    x_test = test[0:, 1:]
    t_test = test[0:, :1]


    X = X.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)

    return X, T, x_test, t_test

