import pickle
from phidnet import network_data



def model(direct):
    with open(direct, "rb") as fr:   # Load saved weight
        network_data.layer = pickle.load(fr)
    return 0
