import pickle

from phidnet import network_data



def model(direct):
    with open(direct, "rb") as fr:   # Load saved weight
        network_data.params = pickle.load(fr)
    network_data.weight = network_data.params['weight']
    network_data.bias = network_data.params['bias']
    network_data.params = None
    return 0
