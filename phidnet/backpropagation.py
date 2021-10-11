import numpy as np
from phidnet import network_data



def gradient(error):
    dout = error
    for i in reversed(network_data.layer):
        #print("============")
        #print("backpropagation:", type(i).__name__)
        #print(type(i).__name__, "input", dout.shape)
        dout = i.backward(dout)
        #print(type(i).__name__, "output:", dout.shape)


    return 0