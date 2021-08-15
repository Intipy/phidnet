import pickle
from phidnet import network_data



def model(name, dir=None):
    network_data.params['weight'] = network_data.weight
    network_data.params['bias'] = network_data.bias
    if dir == None:
        with open(name + ".pickle", "wb") as fw:  # Save weight and bias in pickle
            pickle.dump(network_data.params, fw)
    else:
        with open(dir + "/" + name + ".pickle", "wb") as fw:  # Save weight and bias in pickle
            pickle.dump(network_data.params, fw)
    return 0