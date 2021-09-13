import numpy as np
from phidnet.CNN import network_data



def gradient(error):
    dout = error
    for i in reversed(network_data.layer):

        if str(type(i)) == "<class 'phidnet.activation.Softmax'>":
            pass
        else:
            dout = i.backward(dout)

    return 0