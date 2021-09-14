import numpy as np
import random

from phidnet import network_data



class SGD:   # SGD optimizer
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self):
        for i in network_data.layer_weight_index:
            network_data.layer[i].W -= self.lr * network_data.layer[i].dW
            network_data.layer[i].b -= self.lr * network_data.layer[i].db



class Momentum:   # Momentum optimizer
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.weight_v = None
        self.bias_v = None

    def update(self):
        if self.weight_v is None:
            self.weight_v = {}
            for i in network_data.layer_weight_index:
                self.weight_v[i] = np.zeros_like(network_data.layer[i].W)

        if self.bias_v is None:
            self.bias_v = {}
            for i in network_data.layer_weight_index:
                self.bias_v[i] = np.zeros_like(network_data.layer[i].b)

        for i in network_data.layer_weight_index:
            self.weight_v[i] = self.momentum * self.weight_v[i] - self.lr * network_data.layer[i].dW
            self.bias_v[i] = self.momentum * self.bias_v[i] - self.lr * network_data.layer[i].db
            network_data.layer[i].W += self.weight_v[i]
            network_data.layer[i].b += self.bias_v[i]



class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.weight_h = None
        self.bias_h = None

    def update(self):
        if self.weight_h is None:
            self.weight_h = {}
            for i in network_data.layer_weight_index:
                self.weight_h[i] = np.zeros_like(network_data.layer[i].W)

        if self.bias_h is None:
            self.bias_h = {}
            for i in network_data.layer_weight_index:
                self.bias_h[i] = np.zeros_like(network_data.layer[i].b)

        for i in network_data.layer_weight_index:
            self.weight_h[i] += network_data.layer[i].dW * network_data.layer[i].dW
            self.bias_h[i] += network_data.layer[i].db * network_data.layer[i].db
            network_data.layer[i].W -= self.lr * network_data.layer[i].dW / np.sqrt(self.weight_h[i] + 1e-7)
            network_data.layer[i].b -= self.lr * network_data.layer[i].db / np.sqrt(self.bias_h[i] + 1e-7)