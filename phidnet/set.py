import numpy as np
from phidnet.CNN import *
from phidnet import RNN
from phidnet.RNN import *
from phidnet import network_data, activation, affine



def kernel(kernel_number=10, input_channel=1, kernel_size=(3, 3), stride=1, pad=0):
    W = np.random.randn(kernel_number, input_channel, kernel_size[0], kernel_size[1])
    b = np.zeros(kernel_number)

    network_data.layer.append(convolution.Convolution(W, b, stride=stride, pad=pad))

    return 0



def pooling(size=(2, 2), stride=2):
    network_data.layer.append(pool.Max(pool_h=size[0], pool_w=size[1], stride=stride))

    return 0



def embed(v, d):
    w = np.random.randn(v, d)
    network_data.layer.append(embedding.TimeEmbedding(w))

    return 0



def rnn(d, h):
    rnn_Wx = (np.random.randn(d, h) / np.sqrt(d))
    rnn_Wh = (np.random.randn(h, h) / np.sqrt(h))
    rnn_b = np.zeros(h)
    network_data.layer.append(recurrent.TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True))

    return 0



def nn(v, h):
    affine_W = (np.random.randn(h, v) / np.sqrt(h))
    affine_b = np.zeros(v)
    network_data.layer.append(RNN.affine.TimeAffine(affine_W, affine_b))




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
        if (str(type(i)) == "<class 'phidnet.CNN.convolution.Convolution'>") | (str(type(i)) == "<class 'phidnet.affine.Affine'>") | (str(type(i)) == "<class 'phidnet.RNN.recurrent.TimeRNN'>") | (str(type(i)) == "<class 'phidnet.RNN.affine.TimeAffine'>"):
            network_data.layer_weight_index.append(idx)
        idx += 1


    return 0



def test(input=None, target=None):
    network_data.X_test = input
    network_data.T_test = target

    return 0