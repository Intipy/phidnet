import numpy as np
from phidnet.CNN import network_data



def gradient(error):
    dout = error
    for i in reversed(network_data.layer):

        if str(type(i)) == "<class 'phidnet.activation.Softmax'>":
            pass
        else:
            #print("============")
            #print("backpropagation:", type(i).__name__)
            #print(type(i).__name__, "input", dout.shape)
            dout = i.backward(dout)
            #print(type(i).__name__, "output:", dout.shape)


    return 0