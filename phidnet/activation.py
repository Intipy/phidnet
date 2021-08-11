import numpy as np



class Sigmoid:  # Sigmoid class
    def __init__(self):
        self.mask = None
        self.mask1 = None
        self.out = None

    def forward(self, x):    # Sigmoid function
        self.mask = (x > 100)
        self.mask1 = (x < -100)
        x[self.mask] = 100
        x[self.mask1] = -100
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, x):    # derivative of sigmoid function
        self.out = self.forward(x) * (1 - self.forward(x))
        return self.out



class Relu:   # ReLU class
    def __init__(self):
        self.mask = None
        self.mask1 = None
        self.out = None

    def forward(self, x):   # ReLU function
        self.mask = (x <= 0)
        self.out = x.copy()
        self.out[self.mask] = 0
        return self.out

    def backward(self, x):   # derivative of ReLU function
        self.mask = (x <= 0)
        self.mask1 = (x > 0)
        self.out = x.copy()
        self.out[self.mask] = 0
        self.out[self.mask1] = 1
        return self.out



class Softmax:   # Softmax class
    def __init__(self):
        self.mask = None
        self.mask1 = None
        self.out = None

    def forward(self, x):   # Softmax function
        self.mask = (x > 100)
        self.mask1 = (x < -100)
        x[self.mask] = 100
        x[self.mask1] = -100
        self.out = np.exp(x) / np.sum(np.exp(x))
        return self.out

    def backward(self, x):   # derivative of softmax function
        self.out = self.forward(x) * (1 - self.forward(x))
        return self.out



class Linear:   # Linear class
    def __init__(self):
        self.out = None

    def forward(self, x):   # Linear function
        self.out = x
        return self.out

    def backward(self, x):   # derivative of linear function
        return 1