import numpy as np


class Flatten:
    def __init__(self, input_dim=(1, 12, 12)):
        self.dim = input_dim

    def forward(self, x):
        flat = np.array([i.flatten() for i in x])
        return flat

    def backward(self, dout):
        dx = np.array([i.reshape(self.dim) for i in dout])
        return dx
