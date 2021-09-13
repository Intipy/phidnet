import numpy as np
from phidnet.CNN import network_data



def feedforward(X):
    out = X
    for i in network_data.layer:
        if str(type(i)) == "<class 'phidnet.activation.Softmax'>":
            pass
        else:
            out = i.forward(out)
            #print("feedforward:", type(i).__name__)

    return out

