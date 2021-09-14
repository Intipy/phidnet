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



class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


'''
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
'''



class Softmax:   # Softmax class
    def __init__(self):
        self.out = None

    def forward(self, x):  # Softmax function
        x[x > 100] = 100
        x[x < -100] = -100
        if x.ndim == 2:
            self.out = [np.exp(x[i]) / np.sum(np.exp(x[i])) for i in range(x.shape[0])]
            self.out = np.array(self.out)
        elif x.ndim == 1:
            self.out = np.exp(x) / np.sum(np.exp(x))
        else:
            raise Exception("The input of the softmax function cannot be an array of more than three dimensions")
        return self.out

    def backward(self, x):   # derivative of softmax function
        return 1



class Linear:   # Linear class
    def __init__(self):
        self.out = None

    def forward(self, x):   # Linear function
        self.out = x.copy()
        return self.out

    def backward(self, x):   # derivative of linear function
        return 1
