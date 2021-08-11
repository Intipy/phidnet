import pickle
import os

from phidnet import network_data



def model(dir):
    with open(dir+"\saved_weight.pickle", "rb") as fr:   # Load saved weight
        network_data.weight = pickle.load(fr)
    with open(dir+"\saved_bias.pickle", "rb") as fr:   # Load saved bias
        network_data.bias = pickle.load(fr)