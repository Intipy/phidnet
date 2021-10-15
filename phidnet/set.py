import numpy as np
from phidnet.CNN import *
from phidnet import RNN
from phidnet.RNN import *
import phidnet
from phidnet import network_data, activation, affine
from phidnet import string_to_class



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
    func = string_to_class.convert(func)
    network_data.layer.append(func)

    return 0



def layer(l, activation=None):
    network_data.layer.append(l)

    if activation == None:
        pass
    else:
        activation = string_to_class.convert(activation)
        network_data.layer.append(activation)

    return 0



def compile(input=None, target=None):
    network_data.X = input
    network_data.target = target

    shape = []   # Weight matrix shape (784-200-r-10-s) > (784, 200) (200, 10)
    copied_layer = network_data.layer.copy()  # Copy the layer. Because we change the element, and it makes confusion in loop

    for i in range(len(copied_layer)):
        if str(type(copied_layer[i])) == "<class 'int'>":
            shape.append(copied_layer[i])
        if len(shape) >= 2:
            W = np.random.randn(shape[0], shape[1])
            b = np.random.randn(shape[1])
            network_data.layer[i] = affine.Affine(W, b)

            shape_last = shape[1]   # Reset weight shape, make first shape's column to second shape's row
            shape.clear()
            shape.append(shape_last)


    copied_layer = network_data.layer.copy()   # Copy the layer. Because we remove the element, and it makes confusion in loop
    for i in copied_layer:
        if str(type(i)) == "<class 'int'>":
            network_data.layer.remove(i)


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


