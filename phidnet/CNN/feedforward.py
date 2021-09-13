import numpy as np
from phidnet.CNN import network_data



def feedforward(X):
    length = len(network_data.layer)
    out = X
    for i in range(0, length):
        out = network_data.layer[i].forward(out)

    return out

