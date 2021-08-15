import numpy as np; np.random.seed(121);
from matplotlib import pyplot as plt
import phidnet   # Import phidnet



############################################### Data
X, T, X_test, T_test = phidnet.datasets.mnist.load()
X, X_test = X / 255, X_test / 255
T = phidnet.one_hot_encode.encode_array(T, length=10)
T_test = phidnet.one_hot_encode.encode_array(T_test, length=10)
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



############################################### Build neural network
phidnet.set.layer(784)
phidnet.set.layer(200, activation=Relu)
phidnet.set.layer(10, activation=Softmax)
phidnet.set.compile(input=X, target=T)
phidnet.set.test(input=X_test, target=T_test)   # If you want to get loss of test data, set the test data and fit the model with "val_loss=True"
###############################################



############################################### Fit model
# phidnet.load.model('E:\Programming\Project\phidnet\examples')
phidnet.model.fit(epoch=20, optimizer=AdaGrad, batch=5000, val_loss=True, print_rate=1, save=True)   # Showing validation loss make fitting slow
phidnet.model.show_loss()
phidnet.model.show_accuracy()
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