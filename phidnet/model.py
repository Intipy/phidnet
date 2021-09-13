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





def fit(epoch=1, optimizer=None, batch=100, val_loss=False, print_rate=1):   # Fit model that we`ve built
    iteration = 0
    len_t = len(network_data.target)
    len_test = len(network_data.T_test)

    T = network_data.target   # initial Y, T, error, accuracy for "0 epoch"
    Y = feedforward.feedforward(network_data.X)
    error = mean_squared_error(Y, T) / batch
    acc = accuracy(Y, T)


    for e in range(0, epoch + 1):   # Repeat for epochs

        if (e % print_rate == 0):   # Print loss
            print("|============================")
            print("|epoch: ", e, "/",epoch, sep="")
            print("|loss: ", error)
            print("|acc: ", acc, '%')
            print("|============================")
            print('\n')

        for iterate in range(0, len_t - batch + 1, batch):
            T = network_data.target[iterate:iterate+batch-1]
            network_data.z[0] = network_data.X[iterate:iterate+batch-1]
            Y = feedforward.feedforward(network_data.X[iterate:iterate+batch-1])   # Get last 'z' value in Y every epochs

            loss.loss(Y, T)
            gradient.gradient()
            optimizer.update()

            iteration += 1
            error = mean_squared_error(Y, T) / batch
            acc = accuracy(Y, T)

            network_data.Epoch_list.append(iteration)   # Append values to list that we`ve made
            network_data.Loss_list.append(error)
            network_data.Acc_list.append(acc)

            if val_loss == True:
                T_test = network_data.T_test
                Y_test = feedforward.feedforward(network_data.X_test)
                val_error = mean_squared_error(Y_test, T_test) / len_test
                network_data.Validation_loss_list.append(val_error)

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



def show_loss():   # Show change of epoch, and loss
    plt.plot(network_data.Epoch_list, network_data.Loss_list, color='red')
    if len(network_data.Validation_loss_list) == 0:
        pass
    else:
        plt.plot(network_data.Epoch_list, network_data.Validation_loss_list, color='orange')
    plt.xlabel('Epoch (iteration * epoch)')
    plt.ylabel('Loss')
    plt.legend(['Loss', 'Validation loss'])
    plt.show()
    return 0



def show_accuracy():   # Show change of epoch, and accuracy
    plt.plot(network_data.Epoch_list, network_data.Acc_list, color='green')
    plt.xlabel('Epoch (iteration * epoch)')
    plt.ylabel('Accuracy')
    plt.legend(['Accuracy'])
    plt.ylim([0, 100])
    plt.show()
    return 0



def accuracy(Y, T):   # Get accuracy
    sum = 0
    for i in range(len(T)):
        if np.argmax(Y[i]) == np.argmax(T[i]):
            sum = sum + 1
    return (sum / len(T)) * 100




