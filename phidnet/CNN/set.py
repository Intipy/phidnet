import numpy as np
from phidnet.CNN import convolution, pool, flat
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



def flatten(input_dim=(1, 12, 12)):
    network_data.layer.append(flat.Flatten(input_dim=input_dim))

    return 0



