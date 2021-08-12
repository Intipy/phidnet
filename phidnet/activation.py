import numpy as np



class Sigmoid:  # Sigmoid class
    def __init__(self):
        self.out = None

    def forward(self, x):    # Sigmoid function
        x[x > 100] = 100
        x[x < -100] = -100
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, x):    # derivative of sigmoid function
        self.out = self.forward(x) * (1 - self.forward(x))
        return self.out



class Relu:   # ReLU class
    def __init__(self):
        self.out = None

    def forward(self, x):   # ReLU function
        self.out = x.copy()
        self.out[x < 0] = 0
        return self.out

    def backward(self, x):   # derivative of ReLU function
        self.out = x.copy()
        self.out[x < 0] = 0
        self.out[x > 0] = 1
        return self.out



class Softmax:   # Softmax class
    def __init__(self):
        self.out = None

    def forward(self, x):   # Softmax function
        x[x > 100] = 100
        x[x < -100] = -100
        self.out = np.exp(x) / np.sum(np.exp(x))
        return self.out

    def backward(self, x):   # derivative of softmax function
        pass



class Linear:   # Linear class
    def __init__(self):
        self.out = None

    def forward(self, x):   # Linear function
        self.out = x.copy()
        return self.out

    def backward(self, x):   # derivative of linear function
        return 1
