import numpy as np
from phidnet.CNN import convolution, pool, affine
from phidnet.CNN import network_data



def filter(filter_number=10, input_channel=1, filter_size=5, stride=1, pad=0):
    W = np.random.randn(filter_number, input_channel, filter_size, filter_size)
    b = np.zeros(filter_number)

    network_data.layer.append(convolution.Convolution(W, b, stride=stride, pad=pad))

    return 0



def pooling(height=2, width=2, stride=2):
    network_data.layer.append(pool.Max(pool_h=height, pool_w=width, stride=stride))

    return 0



def activation(func):
    network_data.layer.append(func)

    return 0



def layer(l, activation=None):
    network_data.affine_shape.append(l)
    if activation == None:
        pass
    else:
        network_data.active.append(activation)

    return 0



def compile(input=None, target=None):
    network_data.X = input
    network_data.target = target

    length = len(network_data.affine_shape)
    for i in range(length-1):
        W = np.random.randn(network_data.affine_shape[i], network_data.affine_shape[i+1])
        b = np.random.randn(network_data.affine_shape[i+1])
        network_data.layer.append(affine.Affine(W, b))
        network_data.layer.append(network_data.active[i])

    idx = 0
    for i in network_data.layer:
        if (str(type(i)) == "<class 'phidnet.CNN.convolution.Convolution'>") | (str(type(i)) == "<class 'phidnet.CNN.affine.Affine'>"):
            network_data.layer_weight_index.append(idx)
        idx += 1


    return 0



def test(input=None, target=None):
    network_data.X_test = input
    network_data.T_test = target

    return 0




