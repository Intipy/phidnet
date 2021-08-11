import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

from phidnet.error import mean_squared_error, cross_entropy_error
from phidnet.one_hot_encode import encode, encode_array, get_number
from phidnet import network_data
from phidnet import feedforward
from phidnet import loss
from phidnet import gradient





def fit(epoch=1, optimizer=None, batch=100, print_rate=1, save=False):   # Fit model that we`ve built
    len_t = len(network_data.target)

    for e in range(0, epoch + 1):   # Repeat for epochs

        for iterate in range(0, len_t - batch + 1, batch):
            T = network_data.target[iterate:iterate+batch-1]
            network_data.z[0] = network_data.X[iterate:iterate+batch-1]
            Y = feedforward.feedforward(network_data.X[iterate:iterate+batch-1])   # Get last 'z' value in Y every epochs

            loss.loss(Y, T)
            gradient.gradient()
            optimizer.update()


        error = mean_squared_error(Y, T)
        acc = accuracy(Y, T)

        network_data.Loss_list.append(error)   # Append values to list that we`ve made
        network_data.Epoch_list.append(e)
        network_data.Acc_list.append(acc)


        if (e % print_rate == 0):   # Print loss
            print("|____________________________")
            print("|epoch: ", e)
            print("|loss: ", error)
            print("|acc: ", acc, '%')
            print("|____________________________")
            print('\n')


    if save == True:
        print("|____________________________")
        print("|Model saved in current directory.")
        print("|____________________________")
        with open("saved_weight.pickle", "wb") as fw:  # Save weight and bias in pickle
            pickle.dump(network_data.weight, fw)
        with open("saved_bias.pickle", "wb") as fw:
            pickle.dump(network_data.bias, fw)
    else:
        print("|____________________________")
        print("|Model not saved.")
        print("|____________________________")
    return 0



def predict(inp, exponential=True, precision=6):   # Predict
    if exponential == True:
        X = np.array(inp)
        np.set_printoptions(precision=precision, suppress=False)
        predict_output = feedforward.feedforward(X)
        return predict_output
    else:
        X = np.array(inp)
        np.set_printoptions(precision=precision, suppress=True)
        predict_output = feedforward.feedforward(X)
        return predict_output



def show_fit():   # Show change of epoch, and loss
    plt.plot(network_data.Epoch_list, network_data.Loss_list, color='red')
    plt.plot(network_data.Epoch_list, network_data.Acc_list, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss & Accuracy')
    plt.legend(['Loss', 'Accuracy'])
    plt.show()
    return 0



def accuracy(Y, T):   # Get accuracy
    sum = 0
    for i in range(len(T)):
        if np.argmax(Y[i]) == np.argmax(T[i]):
            sum = sum + 1
    return (sum / len(T)) * 100




