import numpy as np
import phidnet.network_data



def layer(l, activation=None):   # Make shape of layer
    phidnet.network_data.layer_shape.append(l)
    if activation == None:
        pass
    else:
        phidnet.network_data.active.append(activation)



def compile(input=None, target=None):   # Compile model
    phidnet.network_data.layerNumber = len(phidnet.network_data.layer_shape) - 1
    for i in range(1, phidnet.network_data.layerNumber + 1):
        weight(phidnet.network_data.layer_shape[i-1], phidnet.network_data.layer_shape[i], layer=i)
        bias(phidnet.network_data.layer_shape[i], layer=i)

    input_data(input)
    target_data(target)



def weight(axis0, axis1, layer=1):   # Make weights in neural network dictionary
    mat = np.random.randn(axis0, axis1).astype(np.float32)
    phidnet.network_data.weight[layer] = mat
    phidnet.network_data.deltaWeight[layer] = None
    return 0



def bias(axis1, layer=1):   # Make biases in neural network dictionary
    mat = np.random.randn(1, axis1).astype(np.float32)
    phidnet.network_data.bias[layer] = mat
    phidnet.network_data.deltaBias[layer] = None
    return 0



def build_network(layer_num):   # Build neural network, make nodes and set number of layers
    for i in range(1, layer_num+1):
        phidnet.network_data.a[i] = None

    for i in range(1, layer_num+1):
        phidnet.network_data.z[i] = None

    phidnet.network_data.layerNumber = layer_num
    return 0





def input_data(inp):   # Set input data X
    data = np.array(inp)
    phidnet.network_data.X = data
    phidnet.network_data.z[0] = data
    return 0



def target_data(inp):   # Set output data T (Target)
    data = np.array(inp)
    phidnet.network_data.target = data
    return 0
