import pickle
from phidnet import network_data



def model(name, dir=None):

    if dir == None:
        with open(name + ".pickle", "wb") as fw:
            pickle.dump(network_data.layer, fw)
    else:
        with open(dir + "/" + name + ".pickle", "wb") as fw:
            pickle.dump(network_data.layer, fw)

    return 0