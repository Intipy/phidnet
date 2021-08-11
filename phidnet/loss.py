import numpy as np
from phidnet import network_data



def loss(Y, T):
    for i in range(network_data.layerNumber, 0, -1):  # i: 3, 2, 1...  Get loss of layer
        if i == network_data.layerNumber:  # If last layer
            network_data.loss[i] = (Y - T) * network_data.active[i - 1].backward(network_data.a[i])
        else:
            network_data.loss[i] = np.dot(network_data.loss[i + 1], network_data.weight[i + 1].T) * network_data.active[i - 1].backward(network_data.a[i])
    return 0
