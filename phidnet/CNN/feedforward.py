import numpy as np
from phidnet.CNN import network_data



def feedforward(X):
    out = X
    for i in network_data.layer:
        #print("============")
        #print("feedforward:", type(i).__name__)
        #print(type(i).__name__, "input", out.shape)
        out = i.forward(out)
        #print(type(i).__name__, "output:", out.shape)

    return out

