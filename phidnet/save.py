import pickle
from phidnet import network_data



def model(name, dir=None):
    network_data.params['weight'] = network_data.weight
    network_data.params['bias'] = network_data.bias
    network_data.params['layerNumber'] = network_data.layerNumber
    network_data.params['layer_shape'] = network_data.layer_shape
    network_data.params['active'] = network_data.active

    if dir == None:
        with open(name + ".pickle", "wb") as fw:
            pickle.dump(network_data.params, fw)
    else:
        with open(dir + "/" + name + ".pickle", "wb") as fw:
            pickle.dump(network_data.params, fw)

    return 0