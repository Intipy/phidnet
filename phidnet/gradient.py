import numpy as np
from phidnet import network_data



def gradient():
    for i in range(network_data.layerNumber, 0, -1):  # i: 3, 2, 1...  Get gradients of each weights and biases
        network_data.deltaWeight[i] = np.dot(network_data.z[i - 1].T, network_data.loss[i])
        network_data.deltaBias[i] = np.sum(network_data.loss[i], axis=0)  # Sum or mean of bias (need fixing)
    return 0