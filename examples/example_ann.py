import numpy as np; np.random.seed(121);
from matplotlib import pyplot as plt

import phidnet   # Import phidnet



############################################### Data
X, T, X_test, T_test = phidnet.datasets.mnist.load()
X, X_test = X / 255, X_test / 255
T = phidnet.one_hot_encode.encode_array(T, length=10)
print("input shape:", X.shape)
print("output shape:", T.shape)
print("test input shape:", X_test.shape)
print("test output shape:", T_test.shape)
###############################################



############################################### Optimizer & Activation function setting
Relu = phidnet.activation.Relu()
Sigmoid = phidnet.activation.Sigmoid()
Softmax = phidnet.activation.Softmax()
SGD = phidnet.optimizer.SGD(lr=0.01)
Momentum = phidnet.optimizer.Momentum(lr=0.01, momentum=0.9)
AdaGrad = phidnet.optimizer.AdaGrad(lr=0.01)
###############################################



############################################### Build neural network
phidnet.set.layer(784)
phidnet.set.layer(200, activation=Relu)
phidnet.set.layer(10, activation=Sigmoid)
phidnet.set.compile(input=X, target=T)
###############################################



############################################### Fit model
# phidnet.load.model('E:\Programming\Project\phidnet\examples')
phidnet.model.fit(epoch=20, optimizer=AdaGrad, batch=500, print_rate=1, save=False)
phidnet.model.show_fit()
###############################################



############################################### Predict
number = 110
predicted = phidnet.model.predict(X_test[number], exponential=False, precision=2)   # Default: exponential=True, precision=6
print('predict:', predicted)
print('predict:', np.argmax(predicted))
print("right answer:", T_test[number])
img = X_test[number].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()
###############################################s