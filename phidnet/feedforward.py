import numpy as np
from phidnet import network_data



def affine_to_z(X, z_num, active):   # Multiply and add weights and biases, use activation function
    if z_num == 1:
        network_data.a[z_num] = np.dot(X, network_data.weight[z_num]) + network_data.bias[z_num]
        network_data.z[z_num] = active.forward(network_data.a[z_num])
    else:
        network_data.a[z_num] = np.dot(network_data.z[z_num-1], network_data.weight[z_num]) + network_data.bias[z_num]
        network_data.z[z_num] = active.forward(network_data.a[z_num])
    return 0



def feedforward(X):   # Feedforward for fitting
    for i in range(1, network_data.layerNumber + 1):
        affine_to_z(X, i, network_data.active[i-1])
    return network_data.z[network_data.layerNumber]