import numpy as np


'''
def fit(epoch=1, optimizer=None, batch=100, val_loss=False, print_rate=1):   # Fit model that we`ve built
    iteration = 0
    len_t = len(network_data.target)

    for e in range(0, epoch + 1):   # Repeat for epochs

        for iterate in range(0, len_t - batch + 1, batch):
            T = network_data.target[iterate:iterate+batch-1]
            network_data.z[0] = network_data.X[iterate:iterate+batch-1]
            Y = feedforward.feedforward(network_data.X[iterate:iterate+batch-1])   # Get last 'z' value in Y every epochs

            loss.loss(Y, T)
            gradient.gradient()
            optimizer.update()

            iteration += 1
            error = mean_squared_error(Y, T)
            acc = accuracy(Y, T)

            network_data.Epoch_list.append(iteration)   # Append values to list that we`ve made
            network_data.Loss_list.append(error)
            network_data.Acc_list.append(acc)

            if val_loss == True:
                T_test = network_data.T_test
                Y_test = feedforward.feedforward(network_data.X_test)
                val_error = mean_squared_error(Y_test, T_test)
                network_data.Validation_loss_list.append(val_error)


        if (e % print_rate == 0):   # Print loss
            print("|============================")
            print("|epoch: ", e, "/",epoch, sep="")
            print("|loss: ", error)
            print("|acc: ", acc, '%')
            print("|============================")
            print('\n')

    return 0
'''
