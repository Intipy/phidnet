import numpy as np
from phidnet.CNN import convolution, pool
from phidnet.CNN import network_data



def filter(number=10, channel=1, size=5, stride=1, pad=0):
    W = np.random.randn(number, channel, size, size)
    b = np.zeros(number)

    network_data.layer.append(convolution.Convolution(W, b, stride=stride, pad=pad))

    return 0



def pooling(height=2, width=2, stride=2):
    network_data.layer.append(pool.Max(pool_h=height, pool_w=width, stride=stride))

    return 0



def activation(func):
    network_data.layer.append(func)

    return 0



