import numpy as np; np.random.seed(121);
from matplotlib import pyplot as plt
import phidnet   # Import phidnet



############################################### Data
X, T, X_test, T_test = phidnet.datasets.mnist.load_2d()
X, X_test = X / 255, X_test / 255
T = phidnet.one_hot_encode.encode_array(T, length=10)
print("input shape:", X.shape)
print("output shape:", T.shape)
print("test input shape:", X_test.shape)
print("test output shape:", T_test.shape)
###############################################



############################################### Optimizer & Activation function setting
Relu = phidnet.activation.Relu()
Softmax = phidnet.activation.Softmax()

SGD = phidnet.optimizer.SGD(lr=0.01)
Momentum = phidnet.optimizer.Momentum(lr=0.01, momentum=0.9)
AdaGrad = phidnet.optimizer.AdaGrad(lr=0.01)
###############################################



###############################################
phidnet.CNN.set.filter(number=1, channel=1, size=5, stride=1, pad=0)
phidnet.CNN.set.activation(Relu)
phidnet.CNN.set.pooling(height=2, width=2, stride=2)
###############################################



ret = phidnet.CNN.feedforward.feedforward(X_test)
print(ret.shape)


