import pickle
from phidnet import network_data



def model(direct):
    with open(direct, "rb") as fr:   # Load saved weight
        network_data.params = pickle.load(fr)

    network_data.weight = network_data.params['weight']
    network_data.bias = network_data.params['bias']
    network_data.layerNumber = network_data.params['layerNumber']
    network_data.layer_shape = network_data.params['layer_shape']
    network_data.active = network_data.params['active']

    network_data.params = None
    return 0
